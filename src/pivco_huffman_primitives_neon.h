/* pivco_huffman_primitives_neon.h — NEON implementations of the codec
 * primitive interface (see pivco_huffman_primitives.h).
 *
 * Specialized names end in `_neon`; the codec calls the aliases
 * `prim_*` defined at the bottom as always-inline wrappers.
 *
 * Internal header.  Included by pivco_huffman_primitives.h when
 * PIVCO_BACKEND_NEON is defined.  Also #included by the legacy
 * src/pivco_huffman_bu_neon.c during the Phase 3 transition (the
 * legacy file calls these primitives directly until step 3.8 retires
 * it).  Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_PRIMITIVES_NEON_H
#define PIVCO_HUFFMAN_PRIMITIVES_NEON_H

#if !defined(__aarch64__)
#error "pivco_huffman_primitives_neon.h requires aarch64 NEON"
#endif

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_neon_tables.h"   /* expand_tab*, compress_tab* */
#include "pivco_huffman_neon_flat.h"     /* flat_d{2,3,4,5,6}_unpack */
#include "pivco_prof.h"

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

/* Backend lifecycle.  Lazily build the compress_tab + expand_tab pre-
 * bake tables that the NEON partition / tree_merge primitives index
 * into.  Idempotent and cheap after the first call. */
static inline void codec_init_neon(void)
{
    init_compress_table();
    init_expand_table();
}

/* ---------- Decode primitives (bottom-up) ---------- */

/* popcount_K_right_neon — count "1" bits in the first K bits of bm.
 * Vectorised: 64-byte main path with 4-wide ILP, then 16-byte mop-up,
 * scalar tail for the trailing 0..15 full bytes + the optional partial
 * byte (K & 7).  `nbytes` is derivable from K; kept for signature
 * stability with the BU x86 backend. */
static inline int popcount_K_right_neon(const uint8_t *bm, int nbytes, int K)
{
    (void)nbytes;
    PROF_TIC();
    int full_bytes = K >> 3;
    int partial_bits = K & 7;

    uint16x8_t acc_v = vdupq_n_u16(0);
    int b = 0;
    for (; b + 64 <= full_bytes; b += 64) {
        uint8x16_t v0 = vld1q_u8(bm + b);
        uint8x16_t v1 = vld1q_u8(bm + b + 16);
        uint8x16_t v2 = vld1q_u8(bm + b + 32);
        uint8x16_t v3 = vld1q_u8(bm + b + 48);
        uint8x16_t c0 = vcntq_u8(v0);
        uint8x16_t c1 = vcntq_u8(v1);
        uint8x16_t c2 = vcntq_u8(v2);
        uint8x16_t c3 = vcntq_u8(v3);
        /* 3-level lane-wise add tree, all in u8 (max 32 at root). */
        uint8x16_t s01 = vaddq_u8(c0, c1);
        uint8x16_t s23 = vaddq_u8(c2, c3);
        uint8x16_t s   = vaddq_u8(s01, s23);
        acc_v = vaddq_u16(acc_v, vpaddlq_u8(s));
    }
    for (; b + 16 <= full_bytes; b += 16) {
        uint8x16_t v = vld1q_u8(bm + b);
        acc_v = vaddq_u16(acc_v, vpaddlq_u8(vcntq_u8(v)));
    }
    int K_right = (int)vaddvq_u16(acc_v);
    for (; b < full_bytes; b++) K_right += __builtin_popcount(bm[b]);
    if (partial_bits) {
        uint8_t valid_mask = (uint8_t)((1u << partial_bits) - 1);
        K_right += __builtin_popcount(bm[full_bytes] & valid_mask);
    }
    PROF_TOC(PROF_BU_POPCOUNT_K, K);
    return K_right;
}

/* tree_merge_neon — V4 stride-16 main path.
 *
 * Each iter loads 16-byte L_full / R_full once, produces two 8-byte
 * output halves.  Iter 0 uses vqtbl1 over vcombine(low(L_full),
 * low(R_full)) with the baseline expand_tab[m0].  Iter 1 uses vqtbl2
 * over the full 32-byte (L_full, R_full) with a PRECOMPUTED shuf
 * indexed by (nr0, m1) — collapses what would be ~4 runtime ALU ops
 * (the iter-0-induced shift adjustment) into one indexed load.  See
 * pivco_huffman_neon_tables.c for the (nr0, m1) table algebra.
 *
 * +13-15% bench gain on M4, +7-10% on Graviton 4 vs the older stride-
 * 16 path with 4 × vld_8 per iter.  See IDEAS.md "NEON bu_tree_merge:
 * 16-byte loads + precomputed (nr0, m1) shuf — SHIPPED (2026-05-11)". */
static inline void tree_merge_neon(const uint8_t *bm, int K,
                                     const uint8_t *left,
                                     const uint8_t *right,
                                     uint8_t *out)
{
    PROF_TIC();
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x16_t L_full = vld1q_u8(left  + lc);
        uint8x16_t R_full = vld1q_u8(right + rc);

        /* Iter 0: vqtbl1 on the low halves with baseline shuf. */
        uint8_t m0 = bm[j >> 3];
        uint8x16_t both0 = vcombine_u8(vget_low_u8(L_full),
                                        vget_low_u8(R_full));
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];

        /* Iter 1: vqtbl2 over (L_full, R_full) with pre-shifted shuf. */
        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x16x2_t src = {{ L_full, R_full }};
        uint8x8_t shuf1  = vld1_u8(expand_tab_pre[nr0][m1]);
        uint8x8_t o1     = vqtbl2_u8(src, shuf1);
        vst1_u8(out + j + 8, o1);
        int nr1 = expand_popcnt[m1];

        rc += nr0 + nr1;
        lc += (16 - nr0 - nr1);
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m  = bm[j >> 3];
        uint8x8_t  L    = vld1_u8(left + lc);
        uint8x8_t  R    = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(L, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        int nr = expand_popcnt[m];
        rc += nr;
        lc += (8 - nr);
    }
    /* Scalar tail (1..7 leftover). */
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
    PROF_TOC(PROF_BU_TREE_MERGE, K);
}

/* tree_merge_bcast_left_neon — left input is a broadcast constant.
 * Same V4 strategy as tree_merge_neon; the L lane of every vqtbl2
 * iteration reads from a duplicated 16-byte register holding left_sym. */
static inline void tree_merge_bcast_left_neon(const uint8_t *bm, int K,
                                                uint8_t left_sym,
                                                const uint8_t *right,
                                                uint8_t *out)
{
    PROF_TIC();
    int rc = 0;
    int j = 0;
    uint8x8_t  Lbcast   = vdup_n_u8(left_sym);
    uint8x16_t Lbcast_q = vdupq_n_u8(left_sym);
    for (; j + 16 <= K; j += 16) {
        uint8x16_t R_full = vld1q_u8(right + rc);
        uint8_t m0 = bm[j >> 3];
        uint8x16_t both0 = vcombine_u8(Lbcast, vget_low_u8(R_full));
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x16x2_t src = {{ Lbcast_q, R_full }};
        uint8x8_t shuf1  = vld1_u8(expand_tab_pre[nr0][m1]);
        uint8x8_t o1     = vqtbl2_u8(src, shuf1);
        vst1_u8(out + j + 8, o1);

        rc += nr0 + expand_popcnt[m1];
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x8_t  R    = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(Lbcast, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        rc += expand_popcnt[m];
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left_sym;
    }
    PROF_TOC(PROF_BU_TREE_MERGE_BCAST_LEFT, K);
}

/* tree_merge_bcast_right_neon — mirror of tree_merge_bcast_left_neon. */
static inline void tree_merge_bcast_right_neon(const uint8_t *bm, int K,
                                                 const uint8_t *left,
                                                 uint8_t right_sym,
                                                 uint8_t *out)
{
    PROF_TIC();
    int lc = 0;
    int j = 0;
    uint8x8_t  Rbcast   = vdup_n_u8(right_sym);
    uint8x16_t Rbcast_q = vdupq_n_u8(right_sym);
    for (; j + 16 <= K; j += 16) {
        uint8x16_t L_full = vld1q_u8(left + lc);
        uint8_t m0 = bm[j >> 3];
        uint8x16_t both0 = vcombine_u8(vget_low_u8(L_full), Rbcast);
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x16x2_t src = {{ L_full, Rbcast_q }};
        uint8x8_t shuf1  = vld1_u8(expand_tab_pre[nr0][m1]);
        uint8x8_t o1     = vqtbl2_u8(src, shuf1);
        vst1_u8(out + j + 8, o1);

        lc += (16 - nr0 - expand_popcnt[m1]);
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x8_t  L    = vld1_u8(left + lc);
        uint8x16_t both = vcombine_u8(L, Rbcast);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        lc += (8 - expand_popcnt[m]);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left[lc++];
    }
    PROF_TOC(PROF_BU_TREE_MERGE_BCAST_RIGHT, K);
}

/* merge_both_const_neon — both inputs are constants.  vtst+veor blend
 * (no TBL needed).  16 outputs per iter via 2× 8-lane blends. */
static inline void merge_both_const_neon(const uint8_t *bm, int K,
                                           uint8_t left_sym, uint8_t right_sym,
                                           uint8_t *out)
{
    PROF_TIC();
    uint8x8_t vleft  = vdup_n_u8(left_sym);
    uint8x8_t vdelta = vdup_n_u8(left_sym ^ right_sym);
    static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbits = vld1_u8(bit_pos_tab);

    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x8_t bits0 = vtst_u8(vdup_n_u8(bm[j >> 3]), vbits);
        uint8x8_t bits1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]), vbits);
        vst1_u8(out + j,     veor_u8(vleft, vand_u8(vdelta, bits0)));
        vst1_u8(out + j + 8, veor_u8(vleft, vand_u8(vdelta, bits1)));
    }
    for (; j + 8 <= K; j += 8) {
        uint8x8_t bits = vtst_u8(vdup_n_u8(bm[j >> 3]), vbits);
        vst1_u8(out + j, veor_u8(vleft, vand_u8(vdelta, bits)));
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left_sym;
    }
    PROF_TOC(PROF_BU_MERGE_BOTH_CONST, K);
}

/* ---------- Flat-subtree decode (contiguous output) ----------
 *
 * Reads n*D packed bits, looks up each D-bit code in c2s, writes the
 * resulting bytes to out[0..n).  Output is dense / sequential -- the
 * BU codec calls this when it hits a PIVCO_NODE_INTERNAL_FLAT.
 *
 * Per-D SIMD specialisations (D=2..6) live below; D=7..8 and any
 * D > 6 fall through to the NEON_FLAT_UNPACK_SWITCH scalar tail.
 * The per-D unpack helpers (flat_d{2,3,4,5,6}_unpack) come from
 * pivco_huffman_neon_flat.h.
 */

/* Extract D bits at bit position `bit_pos` from `in`.  D <= 16.  Used
 * by NEON_FLAT_UNPACK_SWITCH's scalar tail. */
static inline uint32_t extract_D_bits_neon(const uint8_t *in,
                                             int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3;
    int bit_off  = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8)  val |= ((uint32_t)in[byte_idx + 1]) << 8;
    if (bit_off + D > 16) val |= ((uint32_t)in[byte_idx + 2]) << 16;
    return (val >> bit_off) & ((1u << D) - 1);
}

/* Per-D switch shared by anyone unpacking D-bit codes into a byte
 * stream.  `DST(k)` is the destination expression for output element k. */
#define NEON_FLAT_UNPACK_SWITCH(DST)                                          \
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
        uint32_t code = extract_D_bits_neon(bm, i * D, D);                     \
        DST(i) = c2s[code];                                                    \
    }

/* uarch gate: D=5/D=6 flat-subtree TBL paths -- historical relic from
 * the TD scatter path (since retired).  The legacy comment:
 *
 *   "vqtbl2/vqtbl4 over a 32/64-byte source register pair are fast on
 *    Apple M-series (single-cycle per 16 lanes), but on Neoverse-V2
 *    (Graviton 4) they are slow enough that the per-element scalar
 *    lookup in NEON_FLAT_UNPACK_SWITCH wins."
 *
 * The BU direct-path below IGNORES the gate per the legacy
 * pivco_huffman_neon.c comment: n in the BU path is a contiguous
 * flat-subtree count (typically the full block size 8192 for root-flat
 * distributions like flat_M5), large enough to amortise the
 * vqtbl{2,4} setup even on Neoverse-V2 -- on Graviton 4 the SIMD path
 * beats scalar 3x for flat_M5 (9121 vs 2845 M/s, measured 2026-05-14
 * after this code was briefly gated and regressed in the unify-
 * framework refactor).  Override the gate value at build time with
 * -DPIVCO_NEON_FAST_MULTI_TBL=0/1 only if a future scatter-path
 * variant resurfaces; today it has no callers. */
#ifndef PIVCO_NEON_FAST_MULTI_TBL
#  define PIVCO_NEON_FAST_MULTI_TBL 1
#endif

/* flat_decode_direct_neon — write n D-bit symbols contiguously to
 * out[].  Internal dispatcher: SIMD per-D specialisation for D=2..6,
 * scalar tail for D=7..8 and any unhandled D. */
static inline void flat_decode_direct_neon_inner(uint8_t *symbols, int n,
                                                   const uint8_t *bm, int D,
                                                   const uint8_t *c2s)
{
    if (D == 2) {
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
            uint32_t code = extract_D_bits_neon(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 3) {
        uint8x16_t c2s_vec = vld1q_u8(c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d3_unpack(bm + ((i      * 3) >> 3));
            uint8x8_t codes_hi = flat_d3_unpack(bm + (((i + 8) * 3) >> 3));
            uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
            uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
            vst1q_u8(symbols + i, syms);
        }
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d3_unpack(bm + ((i * 3) >> 3));
            uint8x8_t syms  = vqtbl1_u8(c2s_vec, codes);
            vst1_u8(symbols + i, syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_neon(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
#if PIVCO_NEON_FAST_MULTI_TBL
    if (D == 5) {
        uint8x16x2_t c2s_vec;
        c2s_vec.val[0] = vld1q_u8(c2s);
        c2s_vec.val[1] = vld1q_u8(c2s + 16);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d5_unpack(bm + ((i      * 5) >> 3));
            uint8x8_t codes_hi = flat_d5_unpack(bm + (((i + 8) * 5) >> 3));
            uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
            uint8x16_t syms  = vqtbl2q_u8(c2s_vec, codes);
            vst1q_u8(symbols + i, syms);
        }
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d5_unpack(bm + ((i * 5) >> 3));
            uint8x8_t syms  = vqtbl2_u8(c2s_vec, codes);
            vst1_u8(symbols + i, syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_neon(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 6) {
        uint8x16x4_t c2s_vec;
        c2s_vec.val[0] = vld1q_u8(c2s);
        c2s_vec.val[1] = vld1q_u8(c2s + 16);
        c2s_vec.val[2] = vld1q_u8(c2s + 32);
        c2s_vec.val[3] = vld1q_u8(c2s + 48);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d6_unpack(bm + ((i      * 6) >> 3));
            uint8x8_t codes_hi = flat_d6_unpack(bm + (((i + 8) * 6) >> 3));
            uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
            uint8x16_t syms  = vqtbl4q_u8(c2s_vec, codes);
            vst1q_u8(symbols + i, syms);
        }
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d6_unpack(bm + ((i * 6) >> 3));
            uint8x8_t syms  = vqtbl4_u8(c2s_vec, codes);
            vst1_u8(symbols + i, syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_neon(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 7) {
        /* 128-entry c2s exceeds a single TBL (vqtbl4 = 64), so use two
         * vqtbl4 lookups: the low table covers codes 0..63, the high table
         * codes 64..127 (indexed by code-64, which wraps <64 out of range).
         * Each out-of-range lookup yields 0, so OR-ing the two selects the
         * right half. */
        uint8x16x4_t lo, hi;
        lo.val[0]=vld1q_u8(c2s);     lo.val[1]=vld1q_u8(c2s+16);
        lo.val[2]=vld1q_u8(c2s+32);  lo.val[3]=vld1q_u8(c2s+48);
        hi.val[0]=vld1q_u8(c2s+64);  hi.val[1]=vld1q_u8(c2s+80);
        hi.val[2]=vld1q_u8(c2s+96);  hi.val[3]=vld1q_u8(c2s+112);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            uint8x8_t cl = flat_d7_unpack(bm + ((i      * 7) >> 3));
            uint8x8_t ch = flat_d7_unpack(bm + (((i + 8) * 7) >> 3));
            uint8x16_t codes = vcombine_u8(cl, ch);
            uint8x16_t s0 = vqtbl4q_u8(lo, codes);
            uint8x16_t s1 = vqtbl4q_u8(hi, vsubq_u8(codes, vdupq_n_u8(64)));
            vst1q_u8(symbols + i, vorrq_u8(s0, s1));
        }
        for (; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d7_unpack(bm + ((i * 7) >> 3));
            uint8x8_t s0 = vqtbl4_u8(lo, codes);
            uint8x8_t s1 = vqtbl4_u8(hi, vsub_u8(codes, vdup_n_u8(64)));
            vst1_u8(symbols + i, vorr_u8(s0, s1));
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_neon(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
#endif /* PIVCO_NEON_FAST_MULTI_TBL */
    if (D == 4) {
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
            uint32_t code = extract_D_bits_neon(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
#define DST_DIRECT(k) symbols[k]
    NEON_FLAT_UNPACK_SWITCH(DST_DIRECT)
#undef DST_DIRECT
}

/* flat_decode_to_buffer_neon — D-bit flat-subtree decode into a
 * contiguous output buffer.  Calls the dispatcher above. */
static inline void flat_decode_to_buffer_neon(uint8_t *out, int n,
                                                const uint8_t *bm, int D,
                                                const uint8_t *c2s)
{
    PROF_TIC();
    flat_decode_direct_neon_inner(out, n, bm, D, c2s);
    PROF_TOC(PROF_BU_FLAT_DECODE, n);
}

/* ---------- Encode primitives (bitmap + partition) ----------
 *
 * The non-flat-internal-node hot path.  Builds the n-bit partition
 * bitmap from codes_la[0..n) (each codes_la[i] is the per-symbol left-
 * aligned Huffman code; bit (15 - depth) is the current depth's
 * partition decision) and partitions codes_la in place: left (bit==0)
 * stays in codes_la[0..n_left), right (bit==1) moves to right_out[0..n_right).
 * codes_la lanes are written through to next-level recursion unchanged
 * -- the codes_la representation is depth-threaded, NOT shifted across
 * levels.
 *
 * See pivco_huffman_primitives.h for the codec.c boundary convention.
 */

/* Dense movmask helper: given 8 left-aligned codes and a negative shift
 * amount = -(15 - depth), produce the 8-bit partition mask for this
 * batch.  Right-shifts each lane by (15-depth) so the partition bit
 * lands in the LSB, then horizontal-add weighted by 2^k.
 * Cost: 4 NEON ops (shl, and, shl, addv) per 8 codes. */
static inline uint8_t enc_mask8_codes_la_neon(uint16x8_t code_vec,
                                                int neg_shift_d)
{
    int16x8_t shr_vec = vdupq_n_s16((int16_t)neg_shift_d);
    uint16x8_t bit_lsb = vandq_u16(vshlq_u16(code_vec, shr_vec),
                                    vdupq_n_u16(1));
    static const int16_t weights[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint16x8_t weighted = vshlq_u16(bit_lsb, vld1q_s16(weights));
    return (uint8_t)vaddvq_u16(weighted);
}

/* part_core_neon — the single shared partition loop.  Parameterized by
 * compile-time-constant flags so each always-inline wrapper folds to a
 * specialized branch-free loop with no dead stores:
 *   BUILD=1  build the bitmap from codes_la's depth bit (fused, encode)
 *   BUILD=0  read the mask from bm_in (from-bitmap; future TD-decode share)
 *   EMIT_RIGHT/EMIT_LEFT  scatter that half (full=1,1 right=1,0 left=0,1 none=0,0)
 * The stride-8 scatter (the "partition8" step) is inlined here rather than
 * factored into a helper — a helper-call boundary cost the FULL path ~9% on
 * M4 because the compiler stopped folding the cursor math.  LEFT is written in
 * place over codes_la (n_left <= j keeps the 16-byte store safe); RIGHT goes to
 * right_out.  Returns n_right. */
/* build_bitmap_partition_full_neon — the FULL (both-sides) fused partition,
 * kept hand-written.  The generic part_core_neon below (with EMIT_RIGHT=
 * EMIT_LEFT=1) is logically identical but schedules ~8% slower on M4 for this
 * specific 1,1,1 case — and FULL is the hot common path, so it stays
 * specialized.  part_core handles every other variant (right/left/none and
 * the from-bitmap share) where the generic form matches hand-written speed. */
static inline int build_bitmap_partition_full_neon(uint16_t *codes_la, int n,
                                                    int depth, uint8_t *bm,
                                                    uint16_t *right_out)
{
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);
    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask = enc_mask8_codes_la_neon(code_vec, neg_shift_d);
        bm[j >> 3] = mask;
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t shuf_r = vld1q_u8(tab);
        uint8x16_t shuf_l = vld1q_u8(tab + 16);
        uint8x16_t data   = vreinterpretq_u8_u16(code_vec);
        uint8x16_t right  = vqtbl1q_u8(data, shuf_r);
        uint8x16_t left   = vqtbl1q_u8(data, shuf_l);
        int nr = compress_popcnt[mask];
        vst1q_u8((uint8_t *)(right_out + n_right), right);
        vst1q_u8((uint8_t *)(codes_la  + n_left ), left);
        n_right += nr;
        n_left  += (8 - nr);
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask = 0;
        for (int k = 0; k < tail; k++)
            mask |= (uint8_t)(((tail_buf[k] >> shift_d) & 1) << k);
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k)) right_out[n_right++] = tail_buf[k];
            else                 codes_la[n_left++]   = tail_buf[k];
        }
    }
    return n_right;
}

__attribute__((always_inline)) static inline
int part_core_neon(uint16_t *codes_la, int n, int depth,
                                  uint8_t *bm, const uint8_t *bm_in,
                                  uint16_t *right_out,
                                  int BUILD, int EMIT_RIGHT, int EMIT_LEFT)
{
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);
    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask;
        if (BUILD) { mask = enc_mask8_codes_la_neon(code_vec, neg_shift_d); bm[j >> 3] = mask; }
        else         mask = bm_in[j >> 3];
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t data = vreinterpretq_u8_u16(code_vec);
        if (EMIT_RIGHT) vst1q_u8((uint8_t *)(right_out + n_right),
                                 vqtbl1q_u8(data, vld1q_u8(tab)));
        if (EMIT_LEFT)  vst1q_u8((uint8_t *)(codes_la + n_left),
                                 vqtbl1q_u8(data, vld1q_u8(tab + 16)));
        int nr = compress_popcnt[mask];
        n_right += nr;
        n_left  += 8 - nr;
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask;
        if (BUILD) {
            mask = 0;
            for (int k = 0; k < tail; k++)
                mask |= (uint8_t)(((tail_buf[k] >> shift_d) & 1) << k);
            bm[j >> 3] = mask;
        } else mask = bm_in[j >> 3];
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k)) { if (EMIT_RIGHT) right_out[n_right] = tail_buf[k]; n_right++; }
            else                 { if (EMIT_LEFT)  codes_la[n_left]   = tail_buf[k]; n_left++;  }
        }
    }
    return n_right;
}

/* ---------- Encode primitives (init) ----------
 *
 * enc_init_neon — gather per-symbol left-aligned codes into codes_la.
 * `code_la_lut` is table->code_la (256 uint16 entries).
 *
 * Today this is a straight scalar loop; the compiler often auto-
 * vectorises it via vqtbl1q_u8 over a 256-entry LUT, but the codegen
 * is fragile and the LSU is the bottleneck in either form (microbench
 * at extras/bench_enc_init.c established the NEON TBL pattern buys
 * only ~11% over the scalar loop on M4 -- not worth the source
 * complexity).  Kept here as a primitive so AVX-512's actual SIMD
 * win via vpermi2w (commit 7c08c19) has a contract slot to fill. */
static inline void enc_init_neon(uint16_t *codes_la, int n,
                                   const uint8_t *symbols,
                                   const uint16_t *code_la_lut)
{
    for (int i = 0; i < n; i++) codes_la[i] = code_la_lut[symbols[i]];
}

/* ---------- Encode primitives (flat-subtree pack) ----------
 *
 * Per-D SIMD bit-pack helpers.  Each reads D-bit codes from codes_la
 * (each lane holds the left-aligned Huffman code -- bit-d of the
 * original code is at position 15-d) and packs them LSB-first into
 * the output byte stream.
 *
 * The legacy ergonomic: each `pack_dN_neon` helper takes `right_shift
 * = 16 - depth - D`, applies it via vshlq_u16 with a runtime negative
 * shift vector, ANDs to (1<<D)-1, then packs.  Each helper OVERPACKS
 * (processes ceil(n / stride) * stride elements) so callers can drop
 * the scalar tail entirely if they pre-zero `codes_la[n .. n+15]`.
 * The dispatcher pack_dN_dispatch_neon below handles the residual
 * scalar tail when overpacking isn't possible.
 *
 * For the codec.c contract, prim_enc_pack_dN(codes_la, n, D, depth, out_packed)
 * forwards to pack_dN_neon(out_packed, codes_la, n, D, depth). */

/* D=2: 16 codes -> 4 bytes (4 codes per byte, no byte crossings). */
static inline int pack_d2_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int8_t shifts_d2[16] = {
        0, 2, 4, 6,  0, 2, 4, 6,  0, 2, 4, 6,  0, 2, 4, 6
    };
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t v0 = vshlq_u16(vld1q_u16(codes_la + i    ),
                                   vdupq_n_s16((int16_t)-right_shift));
        uint16x8_t v1 = vshlq_u16(vld1q_u16(codes_la + i + 8),
                                   vdupq_n_s16((int16_t)-right_shift));
        v0 = vandq_u16(v0, vdupq_n_u16(0x3));
        v1 = vandq_u16(v1, vdupq_n_u16(0x3));
        uint8x16_t bytes = vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
        bytes = vshlq_u8(bytes, vld1q_s8(shifts_d2));
        /* Sum groups of 4 via two paired-adds; low 4 lanes = 4 output bytes. */
        uint8x16_t s1 = vpaddq_u8(bytes, bytes);
        uint8x16_t s2 = vpaddq_u8(s1, s1);
        uint32_t packed4 = vgetq_lane_u32(vreinterpretq_u32_u8(s2), 0);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

/* D=3: 8 codes -> 24 bits.  Cross byte boundaries; uint32 accumulator
 * (max shift 7*3 = 21).  Overpacks to ceil(n/8)*8. */
static inline int pack_d3_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int32_t shifts_lo[4] = {0,   3,  6,  9};
    static const int32_t shifts_hi[4] = {12, 15, 18, 21};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vshlq_u16(vld1q_u16(codes_la + i),
                                  vdupq_n_s16((int16_t)-right_shift));
        v = vandq_u16(v, vdupq_n_u16(0x7));
        uint32x4_t lo = vshlq_u32(vmovl_u16(vget_low_u16(v)),
                                   vld1q_s32(shifts_lo));
        uint32x4_t hi = vshlq_u32(vmovl_u16(vget_high_u16(v)),
                                   vld1q_s32(shifts_hi));
        uint32x4_t sum = vaddq_u32(lo, hi);
        uint32_t packed = vaddvq_u32(sum);
        int bi = i * 3 / 8;
        out[bi    ] = (uint8_t)(packed       & 0xff);
        out[bi + 1] = (uint8_t)((packed >> 8 ) & 0xff);
        out[bi + 2] = (uint8_t)((packed >> 16) & 0xff);
    }
    return i;
}

/* D=4: 16 codes -> 8 bytes.  Pair (c[2k], c[2k+1]) into one byte each. */
static inline int pack_d4_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int8_t shifts_d4[16] = {
        0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4
    };
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t v0 = vshlq_u16(vld1q_u16(codes_la + i    ),
                                   vdupq_n_s16((int16_t)-right_shift));
        uint16x8_t v1 = vshlq_u16(vld1q_u16(codes_la + i + 8),
                                   vdupq_n_s16((int16_t)-right_shift));
        v0 = vandq_u16(v0, vdupq_n_u16(0xF));
        v1 = vandq_u16(v1, vdupq_n_u16(0xF));
        uint8x16_t bytes = vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
        bytes = vshlq_u8(bytes, vld1q_s8(shifts_d4));
        uint8x16_t paired = vpaddq_u8(bytes, bytes);
        vst1_u8(out + (i * 4 / 8), vget_low_u8(paired));
    }
    return i;
}

/* D=5: 8 codes -> 5 bytes.  uint64 (max shift 7*5=35). */
static inline int pack_d5_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int32_t sh4[4] = {0, 5, 10, 15};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vandq_u16(vshlq_u16(vld1q_u16(codes_la + i),
                                            vdupq_n_s16((int16_t)-right_shift)),
                                  vdupq_n_u16(0x1F));
        uint32x4_t lo = vshlq_u32(vmovl_u16(vget_low_u16 (v)), vld1q_s32(sh4));
        uint32x4_t hi = vshlq_u32(vmovl_u16(vget_high_u16(v)), vld1q_s32(sh4));
        uint64_t packed = (uint64_t)vaddvq_u32(lo)
                        | ((uint64_t)vaddvq_u32(hi) << 20);
        memcpy(out + i * 5 / 8, &packed, 5);   /* exact 5 bytes */
    }
    return i;
}

/* D=6: 8 codes -> 6 bytes.  uint64 (max shift 7*6=42). */
static inline int pack_d6_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int32_t sh4[4] = {0, 6, 12, 18};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vandq_u16(vshlq_u16(vld1q_u16(codes_la + i),
                                            vdupq_n_s16((int16_t)-right_shift)),
                                  vdupq_n_u16(0x3F));
        uint32x4_t lo = vshlq_u32(vmovl_u16(vget_low_u16 (v)), vld1q_s32(sh4));
        uint32x4_t hi = vshlq_u32(vmovl_u16(vget_high_u16(v)), vld1q_s32(sh4));
        uint64_t packed = (uint64_t)vaddvq_u32(lo)
                        | ((uint64_t)vaddvq_u32(hi) << 24);
        memcpy(out + i * 6 / 8, &packed, 6);   /* exact 6 bytes */
    }
    return i;
}

/* D=7: 8 codes -> 7 bytes.  Pack each half of 4 codes in uint32 lanes
 * (4*7=28 bits < 32), horizontal-add to a scalar, then combine the two
 * 28-bit halves in a uint64.  Avoids the uint16->uint64 widen chain. */
static inline int pack_d7_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int32_t sh4[4] = {0, 7, 14, 21};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vandq_u16(vshlq_u16(vld1q_u16(codes_la + i),
                                            vdupq_n_s16((int16_t)-right_shift)),
                                  vdupq_n_u16(0x7F));
        uint32x4_t lo = vshlq_u32(vmovl_u16(vget_low_u16 (v)), vld1q_s32(sh4));
        uint32x4_t hi = vshlq_u32(vmovl_u16(vget_high_u16(v)), vld1q_s32(sh4));
        uint64_t packed = (uint64_t)vaddvq_u32(lo)
                        | ((uint64_t)vaddvq_u32(hi) << 28);
        memcpy(out + i * 7 / 8, &packed, 7);   /* exact 7 bytes */
    }
    return i;
}

/* D=8: 16 codes -> 16 bytes.  Byte-aligned; one shift+AND pass. */
static inline int pack_d8_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t v0 = vshlq_u16(vld1q_u16(codes_la + i    ),
                                   vdupq_n_s16((int16_t)-right_shift));
        uint16x8_t v1 = vshlq_u16(vld1q_u16(codes_la + i + 8),
                                   vdupq_n_s16((int16_t)-right_shift));
        uint8x16_t bytes = vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
        vst1q_u8(out + i, bytes);
    }
    return i;
}

/* Dispatcher: pack n D-bit codes from codes_la into out[].  Selects the
 * SIMD per-D pack helper, then handles any residual scalar tail (the
 * per-D helpers stride at 8 or 16; callers that haven't pre-zero-padded
 * codes_la beyond n need the tail).
 *
 * codec.c contract: codes_la is the per-block left-aligned-codes array
 * (NOT mutated across recursion levels), `depth` is the current tree
 * depth.  The D-bit local code at a flat-subtree node lives at
 * positions [15-depth .. 15-depth-D+1] of each codes_la[i] = bits
 * shifted right by (16 - depth - D). */
static inline void pack_dN_neon(uint8_t *out, const uint16_t *codes_la,
                                  int n, int D, int depth)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int right_shift = 16 - depth - D;

    int i = 0;
    switch (D) {
    case 2: i = pack_d2_neon(out, codes_la, n, right_shift); break;
    case 3: i = pack_d3_neon(out, codes_la, n, right_shift); break;
    case 4: i = pack_d4_neon(out, codes_la, n, right_shift); break;
    case 5: i = pack_d5_neon(out, codes_la, n, right_shift); break;
    case 6: i = pack_d6_neon(out, codes_la, n, right_shift); break;
    case 7: i = pack_d7_neon(out, codes_la, n, right_shift); break;
    case 8: i = pack_d8_neon(out, codes_la, n, right_shift); break;
    default: break;  /* D >= 9: scalar tail below handles it
                      * (shouldn't happen with PIVCO_MAX_CODE_LEN = 11) */
    }

    /* With overpacking, i = ceil(n / stride) * stride >= n.  Clamp the
     * counter to avoid uint64 underflow in PROF_COUNT_ONLY. */
    int simd_n = i > n ? n : i;
    PROF_COUNT_ONLY(PROF_ENC_FLAT_SIMD_ELEMS, simd_n);
    PROF_COUNT_ONLY(PROF_ENC_FLAT_TAIL_ELEMS, n - simd_n);

    if (i >= n) return;

    /* Scalar tail: pick up where the SIMD path left off. */
    uint32_t mask = (1u << D) - 1;
    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = ((uint32_t)codes_la[i] >> right_shift) & mask;
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

/* ---------- Aliases consumed by codec.c ---------- */

#define PIVCO_PRIM_ALWAYS_INLINE __attribute__((always_inline)) static inline

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ codec_init_neon(); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ enc_init_neon(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_full(uint16_t *codes_la,
                                                      int n, int depth,
                                                      uint8_t *bm,
                                                      uint16_t *right_out)
{ return build_bitmap_partition_full_neon(codes_la, n, depth, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_right(uint16_t *codes_la,
                                                      int n, int depth,
                                                      uint8_t *bm,
                                                      uint16_t *right_out)
{ return part_core_neon(codes_la, n, depth, bm, NULL, right_out, 1, 1, 0); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_left(uint16_t *codes_la,
                                                     int n, int depth,
                                                     uint8_t *bm)
{ return part_core_neon(codes_la, n, depth, bm, NULL, NULL, 1, 0, 1); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_none(uint16_t *codes_la,
                                                     int n, int depth,
                                                     uint8_t *bm)
{ return part_core_neon(codes_la, n, depth, bm, NULL, NULL, 1, 0, 0); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_pack_dN(const uint16_t *codes_la,
                                             int n, int D, int depth,
                                             uint8_t *out_packed)
{ pack_dN_neon(out_packed, codes_la, n, D, depth); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat(uint8_t *out, int n,
                                                          const uint8_t *bm, int D,
                                                          const uint8_t *c2s)
{ flat_decode_to_buffer_neon(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_two(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_both_const_neon(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_constant_left(const uint8_t *bm, int K,
                                                          uint8_t left_sym,
                                                          const uint8_t *right_buf,
                                                          uint8_t *out)
{ tree_merge_bcast_left_neon(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_constant_right(const uint8_t *bm, int K,
                                                           const uint8_t *left_buf,
                                                           uint8_t right_sym,
                                                           uint8_t *out)
{ tree_merge_bcast_right_neon(bm, K, left_buf, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge(const uint8_t *bm, int K,
                                               const uint8_t *left_buf,
                                               const uint8_t *right_buf,
                                               uint8_t *out)
{ tree_merge_neon(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_NEON_H */
