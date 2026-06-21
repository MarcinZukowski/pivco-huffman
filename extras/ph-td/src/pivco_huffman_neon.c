#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif
#include <stdlib.h>
#include <string.h>

/* Dispatch threshold: only attempt FSE if the partition's
 * frequent-bit probability >= this value (fractional, [0.5, 1.0]).
 * Default 0.625 is between table 2's 0.5796 and table 3's 0.6464 --
 * i.e. don't bother on partitions that are only mildly skewed.
 * pivco_fse_select_table() will pick the appropriate table. */
#ifndef PIVCO_FSE_MIN_THRESHOLD
#define PIVCO_FSE_MIN_THRESHOLD 0.625
#endif

/* Per-codeword fallback threshold.  Commit FSE iff the compressed
 * AVERAGE codeword length is at most this fraction of the raw
 * codeword length.  At a node at depth D with n codes routed through
 * its bitmap, every codeword passing through pays D + 1 bits raw
 * (D bits in ancestors + 1 partition bit here) vs D + fse_frac bits
 * after FSE, where fse_frac = (fse_len + 2 wire-prefix) * 8 / n.
 * So we commit iff (D + fse_frac) / (D + 1) <= MIN_RATIO.
 *
 * Effect: at the root (D=0), we just need FSE to clear MIN_RATIO of
 * the raw bitmap.  At D=5, we need a much sharper bitmap-level
 * compression to be worth committing -- the 5 ancestor bits paid by
 * every codeword dilute the local saving.  Matches the intuition
 * "1 bit -> 0.9 bits at the root is worth it; 6 bits -> 5.9 bits is
 * mostly noise." */
#ifndef PIVCO_FSE_MIN_RATIO
#define PIVCO_FSE_MIN_RATIO 0.95
#endif

/* Don't even attempt FSE on bitmaps smaller than this many bytes.
 * Below ~32 bytes (256 codes routed through the node), the FSE
 * per-stream flush overhead (~3 bytes) plus our 2-byte length
 * prefix is a meaningful fraction of the raw size, so commits are
 * unlikely.  Also: deep nodes that have few codes routed through
 * them already use long Huffman codes -- relative FSE benefit
 * there is small. */
#ifndef PIVCO_FSE_MIN_BITMAP_BYTES
#define PIVCO_FSE_MIN_BITMAP_BYTES 32
#endif

/* FSE per-table-id stats storage moved to src/pivco_huffman.c (backend-
 * neutral TU) so the accessor symbols resolve on every platform and
 * codec.c can write the counters directly.  This file used to define
 * the storage as static and the public accessors; the legacy encode
 * path here updates them via extern. */
extern uint64_t g_pivco_fse_commit  [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_attempt [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_bytes_in [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_bytes_out[PIVCO_FSE_STATS_SLOTS];

#define PIVCO_FSE_ROOT_LOG_MAX 65536
extern pivco_huffman_fse_root_event_t g_pivco_fse_root_log[PIVCO_FSE_ROOT_LOG_MAX];
extern int g_pivco_fse_root_n;

#ifdef PIVCO_HAS_NEON
#include <arm_neon.h>
#include "pivco_huffman_neon_flat.h"

/* ---------- uarch gate: D=5/D=6 flat-subtree TBL paths ----------
 *
 * On Apple silicon (M-series), vqtbl2q_u8 / vqtbl4q_u8 retire fast
 * enough that processing 8/16 codes per iteration via a single multi-
 * register TBL is a clean win over scalar 8-byte chunked lookups.
 *
 * On AWS Graviton 4 (Neoverse-V2), the same paths measure markedly
 * slower than the scalar (NEON_FLAT_UNPACK_SWITCH) fallback — and a
 * 2x vqtbl1 + or emulation is slower still.  Disable on non-Apple ARM
 * and fall through to the scalar switch.
 *
 * Override with -DPIVCO_NEON_FAST_MULTI_TBL=0/1 to force one path. */
#ifndef PIVCO_NEON_FAST_MULTI_TBL
#  if defined(__APPLE__)
#    define PIVCO_NEON_FAST_MULTI_TBL 1
#  else
#    define PIVCO_NEON_FAST_MULTI_TBL 0
#  endif
#endif

/* SIMD compress shuffle table + init function: storage and constructor
 * live in pivco_huffman_neon_tables.{c,h} so codec.c (compiled per-
 * backend) can share the same runtime tables.  See the header for the
 * layout and the (mask -> shuf, popcount) semantics. */
#include "pivco_huffman_neon_tables.h"

/* Partition 8 uint16_t by an 8-bit mask.
   bit=1 → right_out, bit=0 → left_out.
   Source is loaded into register first, so left_out may overlap src
   as long as left_out <= src (which holds when n_left <= j).
   Returns count of right (bit=1) elements. */
static inline int partition_8(const uint16_t *src,
                               uint8_t mask,
                               uint16_t *left_out,
                               uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);

    /* Load both shuffle patterns with one ldp (32 bytes, contiguous) */
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);       /* bytes 0-15: right */
    uint8x16_t shuf_l = vld1q_u8(tab + 16);  /* bytes 16-31: left */

    uint8x16_t right = vqtbl1q_u8(data, shuf_r);
    uint8x16_t left  = vqtbl1q_u8(data, shuf_l);

    int n_right = compress_popcnt[mask];

    vst1q_u8((uint8_t *)right_out, right);
    vst1q_u8((uint8_t *)left_out, left);

    return n_right;
}

/* ---------- NEON Encode (Tree-Walk) ---------- */

/* Pack `n` values of `D` bits each into `out`, LSB-first within each byte.
   `vals[i]` supplies the low D bits for element i.  Writes ceil(n*D/8)
   bytes.  Used for the flat-subtree fast path. */
static inline void pack_D_bits(uint8_t *out, int n, int D,
                                const uint16_t *indices,
                                const uint16_t *codes)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0; /* clear tail partial byte */
    uint32_t mask = (1u << D) - 1;
    uint64_t buf = 0;
    int bits_in_buf = 0;
    int byte_idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t local = (uint32_t)codes[indices[i]] & mask;
        buf |= ((uint64_t)local) << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) {
        out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
    }
}

/* Extract D bits at bit position `bit_pos` from `in`.  D <= 16. */
static inline uint32_t extract_D_bits(const uint8_t *in, int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3;
    int bit_off  = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8)  val |= ((uint32_t)in[byte_idx + 1]) << 8;
    if (bit_off + D > 16) val |= ((uint32_t)in[byte_idx + 2]) << 16;
    return (val >> bit_off) & ((1u << D) - 1);
}

/* Flat-subtree decode body — shared switch over D in {2..8} with a
 * scalar tail for anything else.  Instantiated twice below (scatter via
 * `indices[]`, and direct-write where indices are identity).
 * The `DST(k)` macro selects the destination byte for element k. */
#define NEON_FLAT_UNPACK_SWITCH(DST)                                        \
    int i = 0;                                                                \
    switch (D) {                                                              \
    case 2:                                                                   \
        for (; i + 4 <= n; i += 4) {                                          \
            uint8_t b = bm[i >> 2];                                           \
            DST(i    ) = c2s[(b     ) & 3];                                   \
            DST(i + 1) = c2s[(b >> 2) & 3];                                   \
            DST(i + 2) = c2s[(b >> 4) & 3];                                   \
            DST(i + 3) = c2s[(b >> 6) & 3];                                   \
        } break;                                                              \
    case 3:                                                                   \
        for (; i + 8 <= n; i += 8) {                                          \
            const uint8_t *p = bm + ((i * 3) >> 3);                           \
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16); \
            DST(i    ) = c2s[(w      ) & 7];                                  \
            DST(i + 1) = c2s[(w >>  3) & 7];                                  \
            DST(i + 2) = c2s[(w >>  6) & 7];                                  \
            DST(i + 3) = c2s[(w >>  9) & 7];                                  \
            DST(i + 4) = c2s[(w >> 12) & 7];                                  \
            DST(i + 5) = c2s[(w >> 15) & 7];                                  \
            DST(i + 6) = c2s[(w >> 18) & 7];                                  \
            DST(i + 7) = c2s[(w >> 21) & 7];                                  \
        } break;                                                              \
    case 4:                                                                   \
        for (; i + 2 <= n; i += 2) {                                          \
            uint8_t b = bm[i >> 1];                                           \
            DST(i    ) = c2s[b & 0x0F];                                       \
            DST(i + 1) = c2s[b >> 4];                                         \
        } break;                                                              \
    case 5:                                                                   \
        for (; i + 8 <= n; i += 8) {                                          \
            const uint8_t *p = bm + ((i * 5) >> 3);                           \
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)               \
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)      \
                       | ((uint64_t)p[4] << 32);                              \
            DST(i    ) = c2s[(w      ) & 0x1F];                               \
            DST(i + 1) = c2s[(w >>  5) & 0x1F];                               \
            DST(i + 2) = c2s[(w >> 10) & 0x1F];                               \
            DST(i + 3) = c2s[(w >> 15) & 0x1F];                               \
            DST(i + 4) = c2s[(w >> 20) & 0x1F];                               \
            DST(i + 5) = c2s[(w >> 25) & 0x1F];                               \
            DST(i + 6) = c2s[(w >> 30) & 0x1F];                               \
            DST(i + 7) = c2s[(w >> 35) & 0x1F];                               \
        } break;                                                              \
    case 6:                                                                   \
        for (; i + 4 <= n; i += 4) {                                          \
            const uint8_t *p = bm + ((i * 6) >> 3);                           \
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16); \
            DST(i    ) = c2s[(w      ) & 0x3F];                               \
            DST(i + 1) = c2s[(w >>  6) & 0x3F];                               \
            DST(i + 2) = c2s[(w >> 12) & 0x3F];                               \
            DST(i + 3) = c2s[(w >> 18) & 0x3F];                               \
        } break;                                                              \
    case 7:                                                                   \
        for (; i + 8 <= n; i += 8) {                                          \
            const uint8_t *p = bm + ((i * 7) >> 3);                           \
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)               \
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)      \
                       | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)      \
                       | ((uint64_t)p[6] << 48);                              \
            DST(i    ) = c2s[(w      ) & 0x7F];                               \
            DST(i + 1) = c2s[(w >>  7) & 0x7F];                               \
            DST(i + 2) = c2s[(w >> 14) & 0x7F];                               \
            DST(i + 3) = c2s[(w >> 21) & 0x7F];                               \
            DST(i + 4) = c2s[(w >> 28) & 0x7F];                               \
            DST(i + 5) = c2s[(w >> 35) & 0x7F];                               \
            DST(i + 6) = c2s[(w >> 42) & 0x7F];                               \
            DST(i + 7) = c2s[(w >> 49) & 0x7F];                               \
        } break;                                                              \
    case 8:                                                                   \
        for (; i < n; i++) DST(i) = c2s[bm[i]];                               \
        break;                                                                \
    }                                                                          \
    for (; i < n; i++) {                                                       \
        uint32_t code = extract_D_bits(bm, i * D, D);                          \
        DST(i) = c2s[code];                                                    \
    }

/* flat_d{2,3,4,5,6}_unpack() and their tables live in
 * pivco_huffman_neon_flat.h (shared with bench/bench_micro.c). */

/* Unpack n D-bit codes from bm, look up in c2s, scatter to
 * symbols[indices[i]].  Used by decode_node_neon. */
static inline void flat_decode_scatter_neon(uint8_t *symbols,
                                             const uint16_t *indices, int n,
                                             const uint8_t *bm, int D,
                                             const uint8_t *c2s)
{
    if (D == 2) {
        /* Keep c2s (4 entries) in a NEON register; look up 16 codes per
         * iteration with one vqtbl1q_u8 instead of 16 scalar c2s LDRs. */
        uint8x16_t c2s_vec = vld1q_u8(c2s);  /* upper 12 bytes are unused */
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d2_unpack(bm + (i >> 2));
            uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
            symbols[indices[i     ]] = vgetq_lane_u8(syms, 0);
            symbols[indices[i +  1]] = vgetq_lane_u8(syms, 1);
            symbols[indices[i +  2]] = vgetq_lane_u8(syms, 2);
            symbols[indices[i +  3]] = vgetq_lane_u8(syms, 3);
            symbols[indices[i +  4]] = vgetq_lane_u8(syms, 4);
            symbols[indices[i +  5]] = vgetq_lane_u8(syms, 5);
            symbols[indices[i +  6]] = vgetq_lane_u8(syms, 6);
            symbols[indices[i +  7]] = vgetq_lane_u8(syms, 7);
            symbols[indices[i +  8]] = vgetq_lane_u8(syms, 8);
            symbols[indices[i +  9]] = vgetq_lane_u8(syms, 9);
            symbols[indices[i + 10]] = vgetq_lane_u8(syms, 10);
            symbols[indices[i + 11]] = vgetq_lane_u8(syms, 11);
            symbols[indices[i + 12]] = vgetq_lane_u8(syms, 12);
            symbols[indices[i + 13]] = vgetq_lane_u8(syms, 13);
            symbols[indices[i + 14]] = vgetq_lane_u8(syms, 14);
            symbols[indices[i + 15]] = vgetq_lane_u8(syms, 15);
        }
        /* Tail: scalar 4-wide, then 1-wide (same as generic D=2 case) */
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            symbols[indices[i    ]] = c2s[(b     ) & 3];
            symbols[indices[i + 1]] = c2s[(b >> 2) & 3];
            symbols[indices[i + 2]] = c2s[(b >> 4) & 3];
            symbols[indices[i + 3]] = c2s[(b >> 6) & 3];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
    if (D == 3) {
        /* c2s has 8 entries — fits in low half of a 16-byte TBL register.
         * Process 8 codes per iteration. */
        uint8x16_t c2s_vec = vld1q_u8(c2s);  /* upper 8 bytes unused */
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d3_unpack_safe(bm + ((i * 3) >> 3));
            uint8x8_t syms  = vqtbl1_u8(c2s_vec, codes);
            symbols[indices[i    ]] = vget_lane_u8(syms, 0);
            symbols[indices[i + 1]] = vget_lane_u8(syms, 1);
            symbols[indices[i + 2]] = vget_lane_u8(syms, 2);
            symbols[indices[i + 3]] = vget_lane_u8(syms, 3);
            symbols[indices[i + 4]] = vget_lane_u8(syms, 4);
            symbols[indices[i + 5]] = vget_lane_u8(syms, 5);
            symbols[indices[i + 6]] = vget_lane_u8(syms, 6);
            symbols[indices[i + 7]] = vget_lane_u8(syms, 7);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
#if PIVCO_NEON_FAST_MULTI_TBL
    if (D == 5) {
        /* c2s has 32 entries — needs a 2-register TBL (vqtbl2_u8).
         * Process 8 codes per iteration. */
        uint8x16x2_t c2s_vec;
        c2s_vec.val[0] = vld1q_u8(c2s);
        c2s_vec.val[1] = vld1q_u8(c2s + 16);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d5_unpack_safe(bm + ((i * 5) >> 3));
            uint8x8_t syms  = vqtbl2_u8(c2s_vec, codes);
            symbols[indices[i    ]] = vget_lane_u8(syms, 0);
            symbols[indices[i + 1]] = vget_lane_u8(syms, 1);
            symbols[indices[i + 2]] = vget_lane_u8(syms, 2);
            symbols[indices[i + 3]] = vget_lane_u8(syms, 3);
            symbols[indices[i + 4]] = vget_lane_u8(syms, 4);
            symbols[indices[i + 5]] = vget_lane_u8(syms, 5);
            symbols[indices[i + 6]] = vget_lane_u8(syms, 6);
            symbols[indices[i + 7]] = vget_lane_u8(syms, 7);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
    if (D == 6) {
        /* c2s has 64 entries — needs a 4-register TBL (vqtbl4_u8).
         * Process 8 codes per iteration from 6 bytes. */
        uint8x16x4_t c2s_vec;
        c2s_vec.val[0] = vld1q_u8(c2s);
        c2s_vec.val[1] = vld1q_u8(c2s + 16);
        c2s_vec.val[2] = vld1q_u8(c2s + 32);
        c2s_vec.val[3] = vld1q_u8(c2s + 48);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d6_unpack_safe(bm + ((i * 6) >> 3));
            uint8x8_t syms  = vqtbl4_u8(c2s_vec, codes);
            symbols[indices[i    ]] = vget_lane_u8(syms, 0);
            symbols[indices[i + 1]] = vget_lane_u8(syms, 1);
            symbols[indices[i + 2]] = vget_lane_u8(syms, 2);
            symbols[indices[i + 3]] = vget_lane_u8(syms, 3);
            symbols[indices[i + 4]] = vget_lane_u8(syms, 4);
            symbols[indices[i + 5]] = vget_lane_u8(syms, 5);
            symbols[indices[i + 6]] = vget_lane_u8(syms, 6);
            symbols[indices[i + 7]] = vget_lane_u8(syms, 7);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
#endif /* PIVCO_NEON_FAST_MULTI_TBL */
    if (D == 4) {
        /* c2s has 16 entries — exactly fills a 16-byte TBL register.
         * Process 16 codes per iteration. */
        uint8x16_t c2s_vec = vld1q_u8(c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d4_unpack(bm + (i >> 1));
            uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
            symbols[indices[i     ]] = vgetq_lane_u8(syms, 0);
            symbols[indices[i +  1]] = vgetq_lane_u8(syms, 1);
            symbols[indices[i +  2]] = vgetq_lane_u8(syms, 2);
            symbols[indices[i +  3]] = vgetq_lane_u8(syms, 3);
            symbols[indices[i +  4]] = vgetq_lane_u8(syms, 4);
            symbols[indices[i +  5]] = vgetq_lane_u8(syms, 5);
            symbols[indices[i +  6]] = vgetq_lane_u8(syms, 6);
            symbols[indices[i +  7]] = vgetq_lane_u8(syms, 7);
            symbols[indices[i +  8]] = vgetq_lane_u8(syms, 8);
            symbols[indices[i +  9]] = vgetq_lane_u8(syms, 9);
            symbols[indices[i + 10]] = vgetq_lane_u8(syms, 10);
            symbols[indices[i + 11]] = vgetq_lane_u8(syms, 11);
            symbols[indices[i + 12]] = vgetq_lane_u8(syms, 12);
            symbols[indices[i + 13]] = vgetq_lane_u8(syms, 13);
            symbols[indices[i + 14]] = vgetq_lane_u8(syms, 14);
            symbols[indices[i + 15]] = vgetq_lane_u8(syms, 15);
        }
        /* 2-wide and 1-wide tail (same as generic D=4 case). */
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[indices[i    ]] = c2s[b & 0x0F];
            symbols[indices[i + 1]] = c2s[b >> 4];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
#define DST_SCATTER(k) symbols[indices[k]]
    NEON_FLAT_UNPACK_SWITCH(DST_SCATTER)
#undef DST_SCATTER
}

/* Same, but write directly to symbols[i] (indices are identity — used
 * for root-flat in pivco_huffman_decode_neon). */
static inline void flat_decode_direct_neon(uint8_t *symbols, int n,
                                            const uint8_t *bm, int D,
                                            const uint8_t *c2s)
{
    if (D == 2) {
        /* Same unpack as scatter, but store 16 symbols per iter as a
         * single contiguous vst1q_u8 — indices are identity. */
        uint8x16_t c2s_vec = vld1q_u8(c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d2_unpack(bm + (i >> 2));
            uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
            vst1q_u8(symbols + i, syms);
        }
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            symbols[i    ] = c2s[(b     ) & 3];
            symbols[i + 1] = c2s[(b >> 2) & 3];
            symbols[i + 2] = c2s[(b >> 4) & 3];
            symbols[i + 3] = c2s[(b >> 6) & 3];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 3) {
        /* 16 codes per iter via two 8-code unpacks combined into uint8x16
         * for a single vst1q_u8. */
        uint8x16_t c2s_vec = vld1q_u8(c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d3_unpack_safe(bm + ((i      * 3) >> 3));
            uint8x8_t codes_hi = flat_d3_unpack_safe(bm + (((i + 8) * 3) >> 3));
            uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
            uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
            vst1q_u8(symbols + i, syms);
        }
        /* 8-code tail (also NEON-fast) then 1-wide scalar. */
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d3_unpack_safe(bm + ((i * 3) >> 3));
            uint8x8_t syms  = vqtbl1_u8(c2s_vec, codes);
            vst1_u8(symbols + i, syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    /* D=5 / D=6 in the *direct* (root-flat) path always use the SIMD
     * unpack regardless of `PIVCO_NEON_FAST_MULTI_TBL` — n is the full
     * block size (8192) here, so the per-call TBL setup is amortised
     * over enough work that even Neoverse-V2's slower vqtbl{2,4}q_u8
     * still beats the scalar switch.  The gate stays on the scatter
     * path below where n is much smaller and the overhead dominates. */
    if (D == 5) {
        /* 16 codes per iter via two 8-code unpacks + vqtbl2q_u8 on the
         * 32-byte c2s table, single vst1q_u8. */
        uint8x16x2_t c2s_vec;
        c2s_vec.val[0] = vld1q_u8(c2s);
        c2s_vec.val[1] = vld1q_u8(c2s + 16);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d5_unpack_safe(bm + ((i      * 5) >> 3));
            uint8x8_t codes_hi = flat_d5_unpack_safe(bm + (((i + 8) * 5) >> 3));
            uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
            uint8x16_t syms  = vqtbl2q_u8(c2s_vec, codes);
            vst1q_u8(symbols + i, syms);
        }
        /* 8-code tail. */
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d5_unpack_safe(bm + ((i * 5) >> 3));
            uint8x8_t syms  = vqtbl2_u8(c2s_vec, codes);
            vst1_u8(symbols + i, syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 6) {
        /* 16 codes per iter via two 8-code unpacks + vqtbl4q_u8 on the
         * 64-byte c2s table, single vst1q_u8. */
        uint8x16x4_t c2s_vec;
        c2s_vec.val[0] = vld1q_u8(c2s);
        c2s_vec.val[1] = vld1q_u8(c2s + 16);
        c2s_vec.val[2] = vld1q_u8(c2s + 32);
        c2s_vec.val[3] = vld1q_u8(c2s + 48);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d6_unpack_safe(bm + ((i      * 6) >> 3));
            uint8x8_t codes_hi = flat_d6_unpack_safe(bm + (((i + 8) * 6) >> 3));
            uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
            uint8x16_t syms  = vqtbl4q_u8(c2s_vec, codes);
            vst1q_u8(symbols + i, syms);
        }
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d6_unpack_safe(bm + ((i * 6) >> 3));
            uint8x8_t syms  = vqtbl4_u8(c2s_vec, codes);
            vst1_u8(symbols + i, syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 4) {
        /* 16 codes per iter, single vst1q_u8 — indices are identity. */
        uint8x16_t c2s_vec = vld1q_u8(c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d4_unpack(bm + (i >> 1));
            uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
            vst1q_u8(symbols + i, syms);
        }
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[i    ] = c2s[b & 0x0F];
            symbols[i + 1] = c2s[b >> 4];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
#define DST_DIRECT(k) symbols[k]
    NEON_FLAT_UNPACK_SWITCH(DST_DIRECT)
#undef DST_DIRECT
}

/* Decode/merge primitives come from the SHARED main-repo header (TD tracks the
 * current decode primitives).  The u16 (code_la) ENCODE kernels — u16init_neon,
 * enc_mask8_codes_la_neon, the bitmap+partition kernel, pack_d2..8_neon +
 * u16pack_dN_neon — were retired from the production codec (it encodes on u8 ranks
 * now), so TD owns them in pivco_huffman_u16enc.h, included after the shared
 * header (which still provides compress_tab / the *_pack.h helpers / the macros). */
#include "pivco_huffman_primitives_neon.h"
#include "pivco_huffman_u16enc.h"
static inline void pack_D_bits_dense(uint8_t *out, int n, int D, int depth,
                                      const uint16_t *codes_la)
{ u16pack_dN_neon(out, codes_la, n, D, depth); }

static void encode_node_neon(const pivco_huffman_table_t *table,
                              int16_t node_id,
                              uint16_t *codes_la, int n,
                              int depth,
                              uint8_t **out_ptr,
                              uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return; /* leaf */

    PROF_COUNT_ONLY(PROF_ENC_NODE_VISIT, n);

    /* Flat-subtree fast path: emit N*D packed bits instead of D levels of
       bitmaps.  Detected at build_table time. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        uint8_t *out = *out_ptr;
        *out_ptr += total_bytes;
        PROF_TIC();
        pack_D_bits_dense(out, n, D, depth, codes_la);
        PROF_TOC(PROF_ENC_FLAT, n);
        return;
    }

    /* K_right header (2026-05-12 wire format). */
    int need_kr = kr_header_needed(table, node_id);
    uint8_t *kr_hdr = NULL;
    if (need_kr) {
        kr_hdr = *out_ptr;
        *out_ptr += KR_HEADER_BYTES;
    }

    /* FSE marker byte (v0.2 wire format): defaults to 0 = raw bitmap.
     * Replaced after partition if FSE compression wins.  See FSE-V0.md. */
    uint8_t *fse_marker = *out_ptr;
    *out_ptr += 1;
    *fse_marker = 0;

    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    *out_ptr += nbytes;

    PROF_TIC();
    int n_right = prim_enc_partition_full(codes_la, n, depth, bm, tmp);
    int n_left  = n - n_right;
    PROF_TOC(PROF_ENC_NODE_FULL, n);

    if (need_kr) {
        kr_hdr[0] = (uint8_t)(n_right & 0xFF);
        kr_hdr[1] = (uint8_t)((n_right >> 8) & 0xFF);
    }

    /* FSE dispatch (v0): if the partition is skewed enough, FSE-encode
     * the bitmap and replace [marker=0][raw bitmap] with
     * [marker=table|xor][fse_len:u16][fse payload]. */
#ifdef PIVCO_HAS_FSE
    /* For root-event log: capture final t_id / committed / fse_len. */
    int root_t_id_log = 0;
    int root_committed_log = 0;
    size_t root_fse_len_log = (size_t)nbytes;
    double root_p_major_log = 0.0;
    if (n > 0) {
        int rn_major = (n_left >= n_right) ? n_left : n_right;
        root_p_major_log = (double)rn_major / (double)n;
    }
    if (pivco_huffman_get_fse_enabled() &&
        nbytes >= PIVCO_FSE_MIN_BITMAP_BYTES) {
        int n_major = (n_left >= n_right) ? n_left : n_right;
        double p_major = (n > 0) ? (double)n_major / (double)n : 0.0;
        int xor_flag = (n_right > n_left);
        int emitted_fse = 0;
        if (p_major >= PIVCO_FSE_MIN_THRESHOLD) {
            int t_id = pivco_fse_select_table(p_major);
            if (t_id >= 1) {
                uint8_t scratch[PIVCO_BLOCK_SIZE / 8 + 16];
                uint8_t fse_out[PIVCO_BLOCK_SIZE];
                if (xor_flag) {
                    for (int i = 0; i < nbytes; i++) scratch[i] = (uint8_t)~bm[i];
                } else {
                    memcpy(scratch, bm, (size_t)nbytes);
                }
                PROF_TIC();
                size_t fse_len = 0;
                pivco_fse_status_t rc = pivco_fse_compress(
                    t_id, scratch, (size_t)nbytes,
                    fse_out, sizeof(fse_out), &fse_len);
                PROF_TOC(PROF_FSE_ENC, (uint64_t)nbytes);
                g_pivco_fse_attempt[t_id]++;
                root_t_id_log = t_id;
                /* Per-codeword commit gate.  Each codeword passing through
                 * this node costs (depth + 1) bits raw or
                 * (depth + fse_frac) bits with FSE, where
                 *   fse_frac = (fse_len + 2 wire prefix) * 8 / n.
                 * Commit iff (depth + fse_frac) <= ratio * (depth + 1). */
                double fse_frac = (double)(fse_len + 2) * 8.0 / (double)n;
                double codeword_ratio =
                    ((double)depth + fse_frac) / ((double)depth + 1.0);
                if (rc == PIVCO_FSE_OK &&
                    codeword_ratio <= (double)PIVCO_FSE_MIN_RATIO) {
                    *fse_marker = (uint8_t)((xor_flag ? 0x80 : 0) | t_id);
                    uint8_t *p = bm;
                    *p++ = (uint8_t)(fse_len & 0xFF);
                    *p++ = (uint8_t)((fse_len >> 8) & 0xFF);
                    memcpy(p, fse_out, fse_len);
                    *out_ptr = p + fse_len;
                    emitted_fse = 1;
                    g_pivco_fse_commit[t_id]++;
                    g_pivco_fse_bytes_in[t_id]  += (uint64_t)nbytes;
                    g_pivco_fse_bytes_out[t_id] += (uint64_t)(fse_len + 3); /* +marker +2-byte len */
                    root_committed_log = 1;
                    root_fse_len_log = fse_len + 3;
                    PROF_COUNT_ONLY(PROF_FSE_HIT_COUNT, 1);
                } else {
                    g_pivco_fse_commit[0]++;   /* slot 0: attempted, rejected */
                    PROF_COUNT_ONLY(PROF_FSE_FALLBACK_COUNT, 1);
                }
            }
        }
        if (!emitted_fse) PROF_COUNT_ONLY(PROF_FSE_RAW_COUNT, 1);
    } else {
        PROF_COUNT_ONLY(PROF_FSE_RAW_COUNT, 1);
    }
    /* Root-event log: one entry per block's root non-flat node. */
    if (depth == 0 && g_pivco_fse_root_n < PIVCO_FSE_ROOT_LOG_MAX) {
        pivco_huffman_fse_root_event_t *e = &g_pivco_fse_root_log[g_pivco_fse_root_n++];
        e->table_id   = root_t_id_log;
        e->p_major    = root_p_major_log;
        e->committed  = root_committed_log;
        e->nbytes_in  = nbytes;
        e->nbytes_out = (int)root_fse_len_log;
    }
#endif

    /* Recurse.  Left child reads codes_la[0..n_left); right child reads
     * tmp[0..n_right).  Each grandchild gets a `tmp` cursor advanced
     * past the right half (so siblings don't trample each other). */
    encode_node_neon(table, node->left, codes_la, n_left,
                     depth + 1, out_ptr, tmp + n_right);
    encode_node_neon(table, node->right, tmp,      n_right,
                     depth + 1, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_neon(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    init_compress_table();
    PROF_COUNT_ONLY(PROF_ENC_ENTRY, PIVCO_BLOCK_SIZE);

    const int N = PIVCO_BLOCK_SIZE;

    /* Dense left-aligned codes — the input encode_node_neon walks
     * down the tree.  +16 slack: flat-pack overpack zero-pads past n;
     * partition_8's 16-byte TBL store can write at n_left + 8. */
    uint16_t codes_la[PIVCO_BLOCK_SIZE + 16];

    PROF_TIC();
    u16init_neon(codes_la, N, symbols, table->code_la);
    PROF_TOC(PROF_ENC_INIT, N);

    /* `tmp` scratch sizing: each RIGHT-going recursion advances the tmp
     * cursor by the parent's n_right.  In the worst case (highly skewed
     * partitions where most elements keep going the same way), the
     * accumulated offset can reach (max_tree_depth) * N elements --
     * the sum over all levels of "n_right at that level" can be as
     * large as max_depth * N when each level passes ~all elements
     * through to the same child.  Bound by PIVCO_MAX_CODE_LEN+2 so we
     * have room for the SIMD 16-byte overrun and a depth that includes
     * the K_right-header step at the bottom.
     *
     * 16K stack-alloc would only suffice for balanced trees; we hit a
     * real OOB on cat-image.jpg block 34 in 2026-05-13, where a tree
     * with three consecutive ~7K n_right partitions overran a 16K tmp
     * into the table struct.  Heap-alloc with the worst-case bound. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)malloc(tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;

    uint8_t *ptr = out;

    encode_node_neon(table, table->tree_root, codes_la, N,
                     0, &ptr, tmp);

    free(tmp);
    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- NEON Decode (Tree-Walk with SIMD Partition) ---------- */

/* Half-partition: extract only the right (bit=1) elements.
   One TBL + one store instead of two. Returns count of right elements. */
static inline int partition_8_right(const uint16_t *src,
                                     uint8_t mask,
                                     uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
    vst1q_u8((uint8_t *)right_out, vqtbl1q_u8(data, shuf_r));
    return compress_popcnt[mask];
}

/* Half-partition: extract only the left (bit=0) elements. */
static inline int partition_8_left(const uint16_t *src,
                                    uint8_t mask,
                                    uint16_t *left_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
    vst1q_u8((uint8_t *)left_out, vqtbl1q_u8(data, shuf_l));
    return 8 - compress_popcnt[mask];
}

/* Scatter a single symbol to indices[0..n-1] positions in symbols[].
   NEON-assisted: bulk-load 8 indices per vector + lane extracts. */
static inline void scatter_sym(uint8_t *symbols,
                                const uint16_t *indices, int n,
                                uint8_t sym)
{
    /* Direct indices[] access (not vld1q+vgetq_lane): on Neoverse V2
     * the UMOV vector->GPR extract is ~1.45x slower than a plain LDRH
     * here; on Apple M4 the two are within noise.  Direct is
     * strictly better-or-equal on both ARM targets.  See the A/B in
     * the scatter-index-style investigation. */
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        symbols[indices[j +  0]] = sym; symbols[indices[j +  1]] = sym;
        symbols[indices[j +  2]] = sym; symbols[indices[j +  3]] = sym;
        symbols[indices[j +  4]] = sym; symbols[indices[j +  5]] = sym;
        symbols[indices[j +  6]] = sym; symbols[indices[j +  7]] = sym;
        symbols[indices[j +  8]] = sym; symbols[indices[j +  9]] = sym;
        symbols[indices[j + 10]] = sym; symbols[indices[j + 11]] = sym;
        symbols[indices[j + 12]] = sym; symbols[indices[j + 13]] = sym;
        symbols[indices[j + 14]] = sym; symbols[indices[j + 15]] = sym;
    }
    for (; j + 8 <= n; j += 8) {
        symbols[indices[j +  0]] = sym; symbols[indices[j +  1]] = sym;
        symbols[indices[j +  2]] = sym; symbols[indices[j +  3]] = sym;
        symbols[indices[j +  4]] = sym; symbols[indices[j +  5]] = sym;
        symbols[indices[j +  6]] = sym; symbols[indices[j +  7]] = sym;
    }
    for (; j < n; j++) {
        symbols[indices[j]] = sym;
    }
}

/* Both children are leaves: scatter sym0 (bit=0) or sym1 (bit=1) to each
   index position, selecting via SIMD vtst/veor from the bitmap. */
static inline void scatter_both_leaves(uint8_t *symbols,
                                        const uint16_t *indices, int n,
                                        const uint8_t *bm,
                                        uint8_t sym0, uint8_t sym1)
{
    uint8x16_t vsym0q  = vdupq_n_u8(sym0);
    uint8x16_t vdeltaq = vdupq_n_u8(sym0 ^ sym1);
    static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);

    int j = 0;
    for (; j + 16 <= n; j += 16) {
        /* Build 16 lanes of 0/0xFF from two bitmap bytes, combine
         * into one 128-bit vector, then a single veorq/vandq for
         * the value computation. */
        uint8x8_t  bits0 = vtst_u8(vdup_n_u8(bm[j >> 3]),        vbit_pos);
        uint8x8_t  bits1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]),  vbit_pos);
        uint8x16_t bits  = vcombine_u8(bits0, bits1);
        uint8x16_t vals  = veorq_u8(vsym0q, vandq_u8(vdeltaq, bits));
        /* Indexed (scatter-style) stores -- lane-by-lane on NEON.
         * Direct indices[] access beats vld1q+vgetq_lane on Neoverse
         * V2 (~1.45x); a wash on M4.  Only the value (vals) stays in
         * a vector. */
        symbols[indices[j +  0]] = vgetq_lane_u8(vals,  0);
        symbols[indices[j +  1]] = vgetq_lane_u8(vals,  1);
        symbols[indices[j +  2]] = vgetq_lane_u8(vals,  2);
        symbols[indices[j +  3]] = vgetq_lane_u8(vals,  3);
        symbols[indices[j +  4]] = vgetq_lane_u8(vals,  4);
        symbols[indices[j +  5]] = vgetq_lane_u8(vals,  5);
        symbols[indices[j +  6]] = vgetq_lane_u8(vals,  6);
        symbols[indices[j +  7]] = vgetq_lane_u8(vals,  7);
        symbols[indices[j +  8]] = vgetq_lane_u8(vals,  8);
        symbols[indices[j +  9]] = vgetq_lane_u8(vals,  9);
        symbols[indices[j + 10]] = vgetq_lane_u8(vals, 10);
        symbols[indices[j + 11]] = vgetq_lane_u8(vals, 11);
        symbols[indices[j + 12]] = vgetq_lane_u8(vals, 12);
        symbols[indices[j + 13]] = vgetq_lane_u8(vals, 13);
        symbols[indices[j + 14]] = vgetq_lane_u8(vals, 14);
        symbols[indices[j + 15]] = vgetq_lane_u8(vals, 15);
    }
    for (; j + 8 <= n; j += 8) {
        uint8x8_t bits = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
        uint8x8_t vals = veor_u8(vget_low_u8(vsym0q),
                                  vand_u8(vget_low_u8(vdeltaq), bits));
        symbols[indices[j +  0]] = vget_lane_u8(vals, 0);
        symbols[indices[j +  1]] = vget_lane_u8(vals, 1);
        symbols[indices[j +  2]] = vget_lane_u8(vals, 2);
        symbols[indices[j +  3]] = vget_lane_u8(vals, 3);
        symbols[indices[j +  4]] = vget_lane_u8(vals, 4);
        symbols[indices[j +  5]] = vget_lane_u8(vals, 5);
        symbols[indices[j +  6]] = vget_lane_u8(vals, 6);
        symbols[indices[j +  7]] = vget_lane_u8(vals, 7);
    }
    for (; j < n; j++) {
        uint8_t bit = (bm[j >> 3] >> (j & 7)) & 1;
        symbols[indices[j]] = sym0 ^ ((sym0 ^ sym1) & (uint8_t)(-(int8_t)bit));
    }
}

/* ---------- Per-call-site partition loops (interior recursion) ----------
 *
 * Each kind of partition loop in decode_node_neon is extracted as its
 * own named static function so the profiler can attribute time to the
 * exact call site.  These are not recursive; they only run their
 * loops over n elements and update n_left / n_right counters.
 */

/* Full partition loop: both children non-leaf.  stride-16 + stride-8
 * + scalar tail.  Mirrors the loop body in decode_node_neon's else
 * branch. */
static inline void node_full(uint16_t *indices, int n,
                              const uint8_t *bm,
                              uint16_t *tmp,
                              int *n_left_out, int *n_right_out)
{
    PROF_TIC();
    int n_left = 0, n_right = 0;
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint8_t m0 = bm[j >> 3];
        int nr0 = partition_8(indices + j, m0,
                              indices + n_left, tmp + n_right);
        n_right += nr0;
        n_left  += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        int nr1 = partition_8(indices + j + 8, m1,
                              indices + n_left, tmp + n_right);
        n_right += nr1;
        n_left  += (8 - nr1);
    }
    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8(indices + j, mask,
                             indices + n_left, tmp + n_right);
        n_right += nr;
        n_left  += (8 - nr);
    }
    /* Vector tail (1..7 elements) using two separate half-partition
     * calls.  Earlier attempt (e9a668f) used the same mask_r/mask_l
     * trick but with the combined partition_8; revisited here as
     * partition_8_right + partition_8_left after node_half_{right,left}
     * proved the trick safe.  Each call only writes to ONE buffer:
     * partition_8_right -> tmp (separate), partition_8_left -> indices
     * in-place at n_left <= j.  Verified by the new test_roundtrip
     * coverage (4f2fd5c) which exercises every backend directly. */
    /* Vector tail (1..7 elements).  partition_8_left's 16-byte store
     * includes 8-nl filler zeros past indices+n_left+nl.  The previous
     * bug (e9a668f / 1399cee): in RIGHT recursion where tmp = indices
     * + parent_n with no padding, the filler corrupted the right child's
     * just-written tmp data.
     *
     * Fix: caller (decode_node_neon at FULL case) passes the right
     * child's tmp at indices + parent_n + 8 instead of + parent_n,
     * leaving 8 elements of padding between right's indices range and
     * its tmp.  Since n_left + 8 <= parent_n + 8, the filler now
     * harmlessly lands in the padding zone.  See decode_node_neon
     * comment at the FULL case + the IDEAS.md entry. */
    if (j < n) {
        int rem = n - j;
        uint8_t valid = (uint8_t)((1u << rem) - 1);
        uint8_t mask_r = bm[j >> 3] & valid;
        uint8_t mask_l = bm[j >> 3] | (uint8_t)~valid;
        n_right += partition_8_right(indices + j, mask_r, tmp + n_right);
        n_left  += partition_8_left (indices + j, mask_l, indices + n_left);
    }
    *n_left_out  = n_left;
    *n_right_out = n_right;
    PROF_TOC(PROF_NODE_FULL, n);
}

/* Half-right loop: skip_node = left, partition only the right side. */
static inline int node_half_right(uint16_t *indices, int n,
                                   const uint8_t *bm,
                                   uint16_t *tmp_right_out)
{
    PROF_TIC();
    int n_right = 0;
    int j = 0;
    /* 2x unroll (stride-16): two partition_8_right calls per iteration.
     * Mirrors node_full's stride-16 path — adjacent 8-elem groups have
     * independent loads and TBLs so OOO overlaps the second's load with
     * the first's store; only the destination address (n_right + nr0)
     * has a real dep, which is short-latency integer arithmetic. */
    for (; j + 16 <= n; j += 16) {
        int nr0 = partition_8_right(indices + j,     bm[j >> 3],
                                     tmp_right_out + n_right);
        n_right += nr0;
        int nr1 = partition_8_right(indices + j + 8, bm[(j >> 3) + 1],
                                     tmp_right_out + n_right);
        n_right += nr1;
    }
    for (; j + 8 <= n; j += 8) {
        n_right += partition_8_right(indices + j, bm[j >> 3],
                                      tmp_right_out + n_right);
    }
    /* Vector tail (1..7 elements): mask out invalid bits so partition_8
     * sees them as "left" and ignores them.  Safe because tmp_right_out
     * is a separate buffer — partition_8_right only writes there, never
     * to indices, so there's no in-place aliasing risk like node_full
     * has.  The 16-byte vector load past indices[n-1] is bounded by
     * the BLK-sized scratch, and the trailing filler bytes of the
     * vector store land in tmp_right_out beyond n_right and are not
     * referenced by the recursive caller. */
    if (j < n) {
        int rem = n - j;
        uint8_t mask = bm[j >> 3] & (uint8_t)((1u << rem) - 1);
        n_right += partition_8_right(indices + j, mask,
                                      tmp_right_out + n_right);
    }
    PROF_TOC(PROF_NODE_HALF_RIGHT, n);
    return n_right;
}

/* Half-left loop: skip_node = right, partition only the left side. */
static inline int node_half_left(uint16_t *indices, int n,
                                  const uint8_t *bm)
{
    PROF_TIC();
    int n_left = 0;
    int j = 0;
    /* 2x unroll (stride-16) — same rationale as node_half_right. */
    for (; j + 16 <= n; j += 16) {
        int nl0 = partition_8_left(indices + j,     bm[j >> 3],
                                    indices + n_left);
        n_left += nl0;
        int nl1 = partition_8_left(indices + j + 8, bm[(j >> 3) + 1],
                                    indices + n_left);
        n_left += nl1;
    }
    for (; j + 8 <= n; j += 8) {
        n_left += partition_8_left(indices + j, bm[j >> 3],
                                    indices + n_left);
    }
    /* Vector tail (1..7 elements): set invalid bits to 1 so partition_8
     * sees them as "right" and ignores them on the left side.  In-place
     * write to indices+n_left, but n_left <= j (loop invariant), and
     * partition_8_left loaded indices[j..j+7] into a register before
     * issuing its store, so no read-after-write hazard.  Trailing
     * filler bytes from the 16B store land in indices[n_left+nl..
     * n_left+8) which is past the valid left-side range and never
     * referenced by the caller (which sees n_left elements only). */
    if (j < n) {
        int rem = n - j;
        uint8_t mask = bm[j >> 3] | (uint8_t)~((1u << rem) - 1);
        n_left += partition_8_left(indices + j, mask,
                                    indices + n_left);
    }
    PROF_TOC(PROF_NODE_HALF_LEFT, n);
    return n_left;
}

/* Read [marker, then (raw bitmap) OR (fse_len + fse payload)] from
 * *in_ptr.  Same wire-format helper as the BU decoder; see
 * pivco_huffman_bu_neon.c read_bitmap_bu() for the contract. */
static inline const uint8_t *read_bitmap_td(const uint8_t **in_ptr,
                                             int n,
                                             uint8_t *scratch)
{
    int nbytes = bitmap_bytes(n);
    uint8_t marker = **in_ptr;
    *in_ptr += 1;
    if (marker == 0) {
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        return bm;
    }
#ifdef PIVCO_HAS_FSE
    int t_id = marker & 0x7F;
    int xor_flag = (marker >> 7) & 1;
    uint16_t fse_len;
    memcpy(&fse_len, *in_ptr, 2);
    *in_ptr += 2;
    size_t out_len = 0;
    PROF_TIC();
    (void)pivco_fse_decompress(t_id, *in_ptr, fse_len,
                                scratch, (size_t)nbytes,
                                (size_t)nbytes, &out_len);
    PROF_TOC(PROF_FSE_DEC, (uint64_t)nbytes);
    *in_ptr += fse_len;
    if (xor_flag) pivco_fse_flip_bits(scratch, (size_t)nbytes);
    return scratch;
#else
    (void)scratch;
    *in_ptr += nbytes;
    return *in_ptr - nbytes;
#endif
}

static void decode_node_neon(const pivco_huffman_table_t *table,
                              int16_t node_id,
                              uint16_t *indices, int n,
                              uint8_t *symbols,
                              const uint8_t **in_ptr,
                              uint16_t *tmp,
                              int16_t skip_node)
{
    if (n == 0) return;
    PROF_COUNT_ONLY(PROF_DECODE_NODE, n);

    const pivco_tree_node_t *node = &table->tree[node_id];

    /* Single dispatch on pre-classified node type — replaces the chain
     * of skip_node/leaf/flat/both-leaves/half-prefilled checks that
     * were re-evaluated per call.  The compiler emits a jump table.
     * skip_node parameter is kept for API compatibility with recursive
     * children but no longer compared against here (PIVCO_NODE_SKIP
     * already encodes that). */
    (void)skip_node;
    switch ((pivco_node_type_t)table->node_type[node_id]) {
    case PIVCO_NODE_SKIP:
        return;

    case PIVCO_NODE_LEAF: {
        PROF_TIC();
        scatter_sym(symbols, indices, n, (uint8_t)node->symbol);
        PROF_TOC(PROF_SCATTER_SYM, n);
        return;
    }

    case PIVCO_NODE_INTERNAL_FLAT: {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[node_id]];
        PROF_TIC();
        flat_decode_scatter_neon(symbols, indices, n, bm, D, c2s);
        PROF_TOC(PROF_FLAT_DECODE_SCATTER, n);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        /* No K_right header for BOTH_LEAVES (encoder didn't write one). */
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_td(in_ptr, n, bm_scratch);
        const pivco_tree_node_t *left_child  = &table->tree[node->left];
        const pivco_tree_node_t *right_child = &table->tree[node->right];
        PROF_TIC();
        scatter_both_leaves(symbols, indices, n, bm,
                            (uint8_t)left_child->symbol,
                            (uint8_t)right_child->symbol);
        PROF_TOC(PROF_SCATTER_BOTH_LEAVES, n);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_td(in_ptr, n, bm_scratch);
        int n_right = node_half_right(indices, n, bm, tmp);
        decode_node_neon(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right, skip_node);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_td(in_ptr, n, bm_scratch);
        int n_left = node_half_left(indices, n, bm);
        decode_node_neon(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp, skip_node);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_td(in_ptr, n, bm_scratch);
        int n_left, n_right;
        node_full(indices, n, bm, tmp, &n_left, &n_right);
        /* The right child's indices alias parent's tmp[0..n_right).
         * Pass its scratch at tmp + n_right + 8 (not + n_right) so
         * the right child's node_full tail can use a masked vector
         * partition_8_left whose 8-byte filler harmlessly lands in
         * the 8-element padding gap between its indices and tmp.
         * Buf2 (tmp) is sized 2*BLK so the cumulative padding offset
         * across deep right recursion has plenty of room. */
        decode_node_neon(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp + n_right + 8, skip_node);
        decode_node_neon(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right + 8, skip_node);
        return;
    }
    }
}

/* Partition 8 identity indices starting at base.
   Generates [base, base+1, ..., base+7] in-register (no memory read)
   then partitions via TBL shuffle like partition_8. */
static inline int partition_root_8(int base, uint8_t mask,
                                    uint16_t *left_out,
                                    uint16_t *right_out)
{
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    uint8x16_t data = vreinterpretq_u8_u16(
        vaddq_u16(vdupq_n_u16((uint16_t)base), vld1q_u16(off)));

    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);

    vst1q_u8((uint8_t *)right_out, vqtbl1q_u8(data, shuf_r));
    vst1q_u8((uint8_t *)left_out, vqtbl1q_u8(data, shuf_l));

    return compress_popcnt[mask];
}

/* ---------- Per-call-site partition loops at the root (entry) ----------
 *
 * Each loop generates identity indices in-register (partition_root_8)
 * — N elements per BLK, no indices array read.  Extracted as named
 * static functions so the profiler can attribute time per call site. */

static inline void root_full(int N, const uint8_t *bm,
                              uint16_t *indices, uint16_t *tmp,
                              int *n_left_out, int *n_right_out)
{
    PROF_TIC();
    int n_left = 0, n_right = 0;
    int j = 0;
    for (; j + 16 <= N; j += 16) {
        uint8_t m0 = bm[j >> 3];
        int nr0 = partition_root_8(j, m0,
                                    indices + n_left, tmp + n_right);
        n_right += nr0; n_left += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        int nr1 = partition_root_8(j + 8, m1,
                                    indices + n_left, tmp + n_right);
        n_right += nr1; n_left += (8 - nr1);
    }
    for (; j + 8 <= N; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_root_8(j, mask,
                                   indices + n_left, tmp + n_right);
        n_right += nr; n_left += (8 - nr);
    }
    for (; j < N; j++) {
        if (bitmap_get(bm, j)) tmp[n_right++] = (uint16_t)j;
        else                   indices[n_left++] = (uint16_t)j;
    }
    *n_left_out  = n_left;
    *n_right_out = n_right;
    PROF_TOC(PROF_ROOT_FULL, N);
}

static inline int root_half_right(int N, const uint8_t *bm,
                                   uint16_t *tmp_right_out)
{
    PROF_TIC();
    int n_right = 0;
    int j = 0;
    for (; j + 8 <= N; j += 8) {
        uint8_t mask = bm[j >> 3];
        static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
        uint8x16_t data = vreinterpretq_u8_u16(
            vaddq_u16(vdupq_n_u16((uint16_t)j), vld1q_u16(off)));
        uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
        vst1q_u8((uint8_t *)(tmp_right_out + n_right),
                 vqtbl1q_u8(data, shuf_r));
        n_right += compress_popcnt[mask];
    }
    for (; j < N; j++) {
        if (bitmap_get(bm, j))
            tmp_right_out[n_right++] = (uint16_t)j;
    }
    PROF_TOC(PROF_ROOT_HALF_RIGHT, N);
    return n_right;
}

static inline int root_half_left(int N, const uint8_t *bm,
                                  uint16_t *indices_left_out)
{
    PROF_TIC();
    int n_left = 0;
    int j = 0;
    for (; j + 8 <= N; j += 8) {
        uint8_t mask = bm[j >> 3];
        static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
        uint8x16_t data = vreinterpretq_u8_u16(
            vaddq_u16(vdupq_n_u16((uint16_t)j), vld1q_u16(off)));
        uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
        vst1q_u8((uint8_t *)(indices_left_out + n_left),
                 vqtbl1q_u8(data, shuf_l));
        n_left += 8 - compress_popcnt[mask];
    }
    for (; j < N; j++) {
        if (!bitmap_get(bm, j))
            indices_left_out[n_left++] = (uint16_t)j;
    }
    PROF_TOC(PROF_ROOT_HALF_LEFT, N);
    return n_left;
}

int pivco_huffman_decode_neon(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    PROF_COUNT_ONLY(PROF_DECODE_ENTRY, PIVCO_BLOCK_SIZE);

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root is leaf — fill everything */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Root is a flat subtree (whole tree is flat, D >= 2).  Stream is
       N*D packed bits; indices are identity so we write symbols[i]
       directly.  No prefill memset needed (every byte is written). */
    if (table->flat_depth[table->tree_root] >= 2) {
        int D = table->flat_depth[table->tree_root];
        int total_bytes = (N * D + 7) >> 3;
        const uint8_t *bm = ptr;
        ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[table->tree_root]];
        PROF_TIC();
        flat_decode_direct_neon(symbols, N, bm, D, c2s);
        PROF_TOC(PROF_FLAT_DECODE_DIRECT, N);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* K_right header for root (skipped by TD; encoder wrote it iff
     * root has any non-leaf child). */
    if (kr_header_needed(table, table->tree_root)) ptr += KR_HEADER_BYTES;
    /* Read root bitmap (handles per-node FSE marker; bm may point into
     * the input stream or into bm_scratch). */
    uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
    const uint8_t *bm = read_bitmap_td(&ptr, N, bm_scratch);

    const pivco_tree_node_t *left_child  = &table->tree[root->left];
    const pivco_tree_node_t *right_child = &table->tree[root->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf) {
        /* Both-leaves at root — sequential vst1 stores, no scatter.
           indices[j] == j so symbols[indices[j]] = symbols[j].
           Overwrites the memset, but vst1 is equally fast. */
        uint8_t sym0 = (uint8_t)left_child->symbol;
        uint8_t sym1 = (uint8_t)right_child->symbol;
        uint8x8_t vsym0  = vdup_n_u8(sym0);
        uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);
        static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
        uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);

        int j = 0;
        for (; j + 16 <= N; j += 16) {
            uint8x8_t bits0 = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
            uint8x8_t vals0 = veor_u8(vsym0, vand_u8(vdelta, bits0));
            uint8x8_t bits1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]), vbit_pos);
            uint8x8_t vals1 = veor_u8(vsym0, vand_u8(vdelta, bits1));
            vst1_u8(symbols + j, vals0);
            vst1_u8(symbols + j + 8, vals1);
        }
        for (; j + 8 <= N; j += 8) {
            uint8x8_t bits = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
            uint8x8_t vals = veor_u8(vsym0, vand_u8(vdelta, bits));
            vst1_u8(symbols + j, vals);
        }
        for (; j < N; j++) {
            uint8_t bit = (bm[j >> 3] >> (j & 7)) & 1;
            symbols[j] = sym0 ^ ((sym0 ^ sym1) & (uint8_t)(-(int8_t)bit));
        }
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Prefill output with the most frequent symbol (precomputed in table).
       The tree walk skips scattering this symbol — it's already in place. */
    uint8_t prefill_sym = table->prefill_sym;
    memset(symbols, prefill_sym, (size_t)N);

    int16_t skip_node = table->prefill_node;
    /* +8 padding on indices: top-level partition_8_left's 16-byte
     * filler may land at indices[n_left..n_left+8), and n_left can
     * reach BLK-1 in pathological partitions.  Sizing to BLK+8
     * keeps the filler in-bounds; its bytes are never read.
     * 64B alignment keeps the layout deterministic across runs and
     * avoids cache-set-conflict outliers on synthetic distributions
     * (uniform / two_sym_*) where the +8 offset alone shifted into
     * unfortunate associativity. */
    uint16_t indices[PIVCO_BLOCK_SIZE + 8] __attribute__((aligned(64)));
    /* `tmp` scratch sizing: same OOB hazard as the encoder side, see
     * pivco_huffman_encode_neon comment.  Each RIGHT-going recursion
     * advances the tmp cursor by parent's n_right; in the worst case
     * accumulated offset reaches max_tree_depth × N.  Heap-alloc with
     * (PIVCO_MAX_CODE_LEN+2)*BLOCK_SIZE capacity. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)aligned_alloc(64, tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;

    if (left_leaf && root->left == skip_node) {
        int n_right = root_half_right(N, bm, tmp);
        decode_node_neon(table, root->right, tmp, n_right,
                         symbols, &ptr, tmp + n_right, skip_node);
    } else if (right_leaf && root->right == skip_node) {
        int n_left = root_half_left(N, bm, indices);
        decode_node_neon(table, root->left, indices, n_left,
                         symbols, &ptr, tmp, skip_node);
    } else {
        int n_left, n_right;
        root_full(N, bm, indices, tmp, &n_left, &n_right);
        /* +8 padding gap before right child's tmp - see decode_node_neon
         * FULL case for rationale. */
        decode_node_neon(table, root->left, indices, n_left,
                         symbols, &ptr, tmp + n_right + 8, skip_node);
        decode_node_neon(table, root->right, tmp, n_right,
                         symbols, &ptr, tmp + n_right + 8, skip_node);
    }

    free(tmp);
    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

/* ============================================================
 * "Naive-tree / SIMD-primitives" decoder for grid completeness.
 *
 * Decodes the slim wire format produced by pivco_huffman_encode_naive
 * (raw bitmap per internal node in DFS preorder, no FSE marker, no
 * K_right header, no flat-subtree path).  Uses the NEON SIMD
 * partition + scatter primitives.  Pair with
 * pivco_huffman_build_table_naive (every internal -> INTERNAL_FULL,
 * every leaf -> LEAF, no prefill, no FLAT, no BOTH_LEAVES, no
 * HALF_*).  This fills the (tree shape) x (primitives) grid cell
 * that the paper otherwise lacks.
 * ============================================================ */

static void decode_node_naive_simd_neon(
        const pivco_huffman_table_t *table, int16_t node_id,
        uint16_t *indices, int n, uint8_t *symbols,
        const uint8_t **in_ptr, uint16_t *tmp)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        PROF_TIC();
        scatter_sym(symbols, indices, n, (uint8_t)node->symbol);
        PROF_TOC(PROF_SCATTER_SYM, n);
        return;
    }
    /* Slim wire: just N bits of raw bitmap.  No marker, no K_right. */
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;
    int n_left, n_right;
    node_full(indices, n, bm, tmp, &n_left, &n_right);
    decode_node_naive_simd_neon(table, node->left,  indices, n_left,
                                  symbols, in_ptr, tmp + n_right + 8);
    decode_node_naive_simd_neon(table, node->right, tmp, n_right,
                                  symbols, in_ptr, tmp + n_right + 8);
}

int pivco_huffman_decode_naive_simd_neon(
        const uint8_t *in, size_t in_len,
        const pivco_huffman_table_t *table,
        uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;
    init_compress_table();
    const int N = PIVCO_BLOCK_SIZE;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    uint16_t indices[PIVCO_BLOCK_SIZE + 8] __attribute__((aligned(64)));
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)aligned_alloc(64, tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;

    /* Naive tree has no prefill -- every byte gets scattered. */
    for (int k = 0; k < N; k++) indices[k] = (uint16_t)k;

    decode_node_naive_simd_neon(table, table->tree_root, indices, N,
                                  symbols, &ptr, tmp);

    free(tmp);
    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

/* Non-static wrapper exposed for the bottom-up decoder
 * (src/pivco_huffman_bu_neon.c) so it can reuse the vectorised
 * D=2..8 flat-decode without duplicating the per-D unpackers. */
void pivco_huffman_flat_decode_direct_neon_(uint8_t *symbols, int n,
                                             const uint8_t *bm, int D,
                                             const uint8_t *c2s) {
    flat_decode_direct_neon(symbols, n, bm, D, c2s);
}

#endif /* PIVCO_HAS_NEON */
