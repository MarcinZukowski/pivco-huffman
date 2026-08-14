/* pivco_huffman_primitives_neon.h — NEON implementations of the codec
 * primitive interface (see pivco_huffman_primitives.h).
 *
 * Specialized names end in `_neon`; the codec calls the aliases
 * `prim_*` defined at the bottom as always-inline wrappers.
 *
 * Internal header.  Included by pivco_huffman_primitives.h when
 * PIVCO_BACKEND_NEON is defined.  Also #included by the legacy
 * src/pivco_bu_neon.c during the Phase 3 transition (the
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
#include "pivco_huffman_neon_pack.h"     /* pack_d{5,6,7}_neon (variable-shift pack) */
#include "pivco_prof.h"

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

/* Backend lifecycle.  Lazily build the compress_tab pre-bake the NEON
 * partition primitives index into, plus the merge shuffle pair.
 * Of the expand tables only expand_popcnt (256 B) is still read by the
 * shipped kernels (merge-tail cursor advance); expand_tab/expand_tab_pre
 * stay built for the bench prim_variants that index them.  Idempotent
 * and cheap after the first call. */
static void init_merge_tables(void);   /* two-table merge_vec_vec shuffles (below) */
static inline void codec_init_neon(void)
{
    init_compress_table();
    init_expand_table();
    init_merge_tables();
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

/* ---- merge_vec_vec_neon: two-table SABD merge, 64 bytes/iter ----
 *
 * One 2-source vqtbl2q over {R16, L16} per 16-byte chunk; the cross-half
 * cursor offset is folded into the shuffle index by SABD (|shuf0 - shuf1|),
 * so no explicit add.  Four chunks per 64-byte iter share one vcnt + 64-bit
 * multiply prefix-sum for the per-chunk cursor splits and the L/R advance.  The
 * two 256x16 index tables (g_merge_shuf0/1, 8 KiB) are built once in
 * codec_init_neon.  Tail (K mod 64) runs the same SABD merge at 16- and
 * 8-wide on the same tables (the 8-wide form stores the low half only),
 * so the whole kernel touches only g_merge_shuf0/1 plus the 256-byte
 * expand_popcnt (tail cursor advance; aarch64 has no GPR popcount) --
 * the old expand_tab/expand_tab_pre ladder dragged up to 20 KiB of
 * cold table lines into L1 for at most two tail iterations per node. */
static int8_t g_merge_shuf0[256 * 16] __attribute__((aligned(16)));
static int8_t g_merge_shuf1[256 * 16] __attribute__((aligned(16)));
static void init_merge_tables(void)
{
    static int built = 0;
    if (built) return;
    for (int i = 0; i < 256; i++) {
        int8_t pop = 0;
        int8_t *o0 = &g_merge_shuf0[i * 16];
        int8_t *o1 = &g_merge_shuf1[i * 16];
        for (int j = 0; j < 8; j++) {
            if ((i >> j) & 1) {
                o0[j] = pop; o1[j + 8] = (int8_t)(-pop); pop++;
            } else {
                int8_t v = (int8_t)(-16 - j + pop);
                o0[j] = v; o1[j + 8] = (int8_t)(8 - v);
            }
        }
        for (int j = 0; j < 8; j++) { o0[j + 8] = pop; o1[j] = 0; }
    }
    built = 1;
}
/* one 16-byte merge: 2-source TBL over {R,L}, SABD-fused index. */
static inline void merge_neon_16B(uint8_t *dest, const uint8_t *l_list,
                                  const uint8_t *r_list, intptr_t mask,
                                  const int8_t *tab0, const int8_t *tab1)
{
    int8x16_t shuf0 = vld1q_s8(&tab0[(mask << 4) & 0xff0]);
    int8x16_t shuf1 = vld1q_s8(&tab1[(mask >> 4) & 0xff0]);
    uint8x16_t shuf = vreinterpretq_u8_s8(vabdq_s8(shuf0, shuf1));
    uint8x16x2_t src;
    src.val[0] = vld1q_u8(r_list);
    src.val[1] = vld1q_u8(l_list);
    vst1q_u8(dest, vqtbl2q_u8(src, shuf));
}
/* 8-byte residue on the same tables: the 16-bit-mask path with the high
 * mask byte zero (tab1 row 0), storing only the low 8 output lanes.
 * Both sides consume <= 8 bytes and every lane index stays in its
 * half's low 8 lanes, so 8-byte D-register loads suffice (they
 * zero-extend for free) -- no over-read past cursor+8. */
static inline void merge_neon_8B_lo(uint8_t *dest, const uint8_t *l_list,
                                    const uint8_t *r_list, intptr_t m8,
                                    const int8_t *tab0, const int8_t *tab1)
{
    int8x16_t shuf0 = vld1q_s8(&tab0[(m8 << 4) & 0xff0]);
    int8x16_t shuf1 = vld1q_s8(&tab1[0]);
    uint8x16_t shuf = vreinterpretq_u8_s8(vabdq_s8(shuf0, shuf1));
    uint8x16x2_t src;
    src.val[0] = vcombine_u8(vld1_u8(r_list), vdup_n_u8(0));
    src.val[1] = vcombine_u8(vld1_u8(l_list), vdup_n_u8(0));
    vst1_u8(dest, vget_low_u8(vqtbl2q_u8(src, shuf)));
}
static inline void merge_vec_vec_neon(const uint8_t *bm, int K,
                                     const uint8_t *left,
                                     const uint8_t *right,
                                     uint8_t *out)
{
    PROF_TIC();
    const uint8_t *l_list = left, *r_list = right;
    intptr_t i = 0;
    /* Software-pipelined one iteration deep: the next iteration's carried
     * chain (bitmap load -> vcnt -> 64-bit multiply -> cursor advance,
     * ~12cy) is started up front so it resolves under the current four
     * merges (~16cy).  The popcount loads the bitmap straight into SIMD
     * (vld1_u8): a GPR->SIMD fmov costs a load-port uop on Apple and
     * would sit mid-chain; mask stays in the GPR for merge_neon_16B's
     * SABD index.  pfx byte k = sum(bytepopcount[0..k]); bytes 1/3/5/7
     * are the 16-lane chunk boundaries c0, c0+c1, c0+c1+c2, total. */
#define MERGE_VV_64(msk, pf) do {                                                     \
        intptr_t p0 = ((pf) >> 8) & 0xff, p1 = ((pf) >> 24) & 0xff,                    \
                 p2 = ((pf) >> 40) & 0xff, p3 = (pf) >> 56;                            \
        merge_neon_16B(out + i,      l_list,           r_list,      (msk),             g_merge_shuf0, g_merge_shuf1); \
        merge_neon_16B(out + i + 16, l_list + 16 - p0, r_list + p0, (msk) >> 16,       g_merge_shuf0, g_merge_shuf1); \
        merge_neon_16B(out + i + 32, l_list + 32 - p1, r_list + p1, (msk) >> 32,       g_merge_shuf0, g_merge_shuf1); \
        merge_neon_16B(out + i + 48, l_list + 48 - p2, r_list + p2, (msk) >> 48,       g_merge_shuf0, g_merge_shuf1); \
        r_list += p3; l_list += 64 - p3;                                              \
    } while (0)
    if (i + 64 <= K) {
        uint64_t mask; memcpy(&mask, bm, 8);
        uint64_t pfx = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(vld1_u8(bm))), 0)
                       * 0x0101010101010101ull;
        for (; i + 128 <= K; i += 64) {
            const uint8_t *nbm = bm + ((i + 64) >> 3);
            uint64_t nmask; memcpy(&nmask, nbm, 8);
            uint64_t npfx = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(vld1_u8(nbm))), 0)
                            * 0x0101010101010101ull;
            MERGE_VV_64(mask, pfx);
            mask = nmask; pfx = npfx;
        }
        MERGE_VV_64(mask, pfx);
        i += 64;
    }
#undef MERGE_VV_64
    int j = (int)i;

    /* Residue on the main tables: 16-wide, then 8-wide (low half). */
    for (; j + 16 <= K; j += 16) {
        uint16_t m16; memcpy(&m16, bm + (j >> 3), 2);
        merge_neon_16B(out + j, l_list, r_list, (intptr_t)m16,
                       g_merge_shuf0, g_merge_shuf1);
        /* expand_popcnt, not __builtin_popcount: aarch64 has no GPR
         * popcount (fmov+cnt+addv+fmov, ~7cy) and this sits on the
         * serial cursor chain between tail iterations. */
        int pop = expand_popcnt[m16 & 0xff] + expand_popcnt[m16 >> 8];
        r_list += pop; l_list += 16 - pop;
    }
    if (j + 8 <= K) {
        intptr_t m8 = bm[j >> 3];
        merge_neon_8B_lo(out + j, l_list, r_list, m8,
                         g_merge_shuf0, g_merge_shuf1);
        int pop = expand_popcnt[m8];
        r_list += pop; l_list += 8 - pop;
        j += 8;
    }
    /* Scalar tail (1..7 leftover). */
    int lc = (int)(l_list - left), rc = (int)(r_list - right);
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
    PROF_TOC(PROF_BU_MERGE_VEC_VEC, K);
}

/* merge_cst_vec_neon — left input is a broadcast constant.
 * Same V5 strategy as merge_vec_vec_neon; the L lane of every chunk's
 * vqtbl2 reads from a duplicated 16-byte register holding left_sym, so
 * no L loads are issued in the V5 main loop. */
/* merge_cst_vec_neon — two-table SABD merge, L = broadcast const (no L load
 * or cursor); only the R cursor advances.  See merge_vec_vec_neon. */
static inline void merge_cst_vec_neon(const uint8_t *bm, int K,
                                      uint8_t left_sym,
                                      const uint8_t *right,
                                      uint8_t *out)
{
    PROF_TIC();
    uint8x16_t Lb = vdupq_n_u8(left_sym);
    const uint8_t *r_list = right;
    intptr_t i = 0;
    for (; i + 64 <= K; i += 64) {
        uint64_t mask; memcpy(&mask, bm + (i >> 3), 8);
        uint8x8_t pop8 = vcnt_u8(vcreate_u8(mask));
        uint64_t pfx = vget_lane_u64(vreinterpret_u64_u8(pop8), 0) * 0x0101010101010101ull;
        intptr_t p0 = (pfx >> 8) & 0xff, p1 = (pfx >> 24) & 0xff, p2 = (pfx >> 40) & 0xff, p3 = pfx >> 56;
#define _MCV(off, rd, mk) do {                                                   \
        int8x16_t s0 = vld1q_s8(&g_merge_shuf0[(((intptr_t)(mk)) << 4) & 0xff0]); \
        int8x16_t s1 = vld1q_s8(&g_merge_shuf1[(((intptr_t)(mk)) >> 4) & 0xff0]); \
        uint8x16_t sh = vreinterpretq_u8_s8(vabdq_s8(s0, s1));                    \
        uint8x16x2_t src; src.val[0] = vld1q_u8(rd); src.val[1] = Lb;            \
        vst1q_u8(out + i + (off), vqtbl2q_u8(src, sh));                          \
    } while (0)
        _MCV(0,  r_list,      mask);
        _MCV(16, r_list + p0, mask >> 16);
        _MCV(32, r_list + p1, mask >> 32);
        _MCV(48, r_list + p2, mask >> 48);
#undef _MCV
        r_list += p3;
    }
    int j = (int)i;
    for (; j + 16 <= K; j += 16) {   /* 16-byte ryg tail before the scalar mop-up */
        uint16_t m16; memcpy(&m16, bm + (j >> 3), 2);
        int8x16_t s0 = vld1q_s8(&g_merge_shuf0[((intptr_t)m16 << 4) & 0xff0]);
        int8x16_t s1 = vld1q_s8(&g_merge_shuf1[((intptr_t)m16 >> 4) & 0xff0]);
        uint8x16_t sh = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
        uint8x16x2_t src; src.val[0] = vld1q_u8(r_list); src.val[1] = Lb;
        vst1q_u8(out + j, vqtbl2q_u8(src, sh));
        r_list += expand_popcnt[m16 & 0xff] + expand_popcnt[m16 >> 8];
    }
    if (j + 8 <= K) {   /* 8-wide residue: high mask byte 0, store low half.
                         * 8B D-load on R -- consumes <= 8, no wider
                         * over-read than the scalar loop it replaces. */
        intptr_t m8 = bm[j >> 3];
        int8x16_t s0 = vld1q_s8(&g_merge_shuf0[(m8 << 4) & 0xff0]);
        int8x16_t s1 = vld1q_s8(&g_merge_shuf1[0]);
        uint8x16_t sh = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
        uint8x16x2_t src;
        src.val[0] = vcombine_u8(vld1_u8(r_list), vdup_n_u8(0));
        src.val[1] = Lb;
        vst1_u8(out + j, vget_low_u8(vqtbl2q_u8(src, sh)));
        r_list += expand_popcnt[m8];
        j += 8;
    }
    int rc = (int)(r_list - right);
    for (; j < K; j++) { int mb = (bm[j >> 3] >> (j & 7)) & 1; out[j] = mb ? right[rc++] : left_sym; }
    PROF_TOC(PROF_BU_MERGE_CST_VEC, K);
}

/* merge_cst_cst_neon — both inputs are constants.  Treated as a
 * D=1 flat decode: a 2-byte (left, right) "c2s" table replicated across
 * 16 lanes via vdupq_n_u16, indexed by the bm bit (0 or 1).  Bit-spread
 * uses the same vqtbl(dup_tab) + vshlq(shift_tab) + vandq pattern as
 * merge_flat_d2_neon, scaled down for D=1 (8 codes / bm byte
 * instead of 4).  Faster than vtst+vand+veor by ~1.6× on M4 NEON and
 * ~1.4× on Neoverse V2. */
static const uint8_t merge_two_dup_tab[16]   = {0,0,0,0,0,0,0,0,
                                                1,1,1,1,1,1,1,1};
static const int8_t  merge_two_shift_tab[16] = {0,-1,-2,-3,-4,-5,-6,-7,
                                                0,-1,-2,-3,-4,-5,-6,-7};
static inline void merge_cst_cst_neon(const uint8_t *bm, int K,
                                           uint8_t left_sym, uint8_t right_sym,
                                           uint8_t *out)
{
    PROF_TIC();
    uint16_t lr_word = (uint16_t)left_sym | ((uint16_t)right_sym << 8);
    uint8x16_t c2s_vec = vreinterpretq_u8_u16(vdupq_n_u16(lr_word));
    uint8x16_t dup_v   = vld1q_u8(merge_two_dup_tab);
    int8x16_t  shift_v = vld1q_s8(merge_two_shift_tab);
    uint8x16_t one_v   = vdupq_n_u8(1);

    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint16_t bm_word; memcpy(&bm_word, bm + (j >> 3), 2);
        uint8x16_t bm_lo = vreinterpretq_u8_u16(
            vsetq_lane_u16(bm_word, vdupq_n_u16(0), 0));
        uint8x16_t dup     = vqtbl1q_u8(bm_lo, dup_v);
        uint8x16_t shifted = vshlq_u8(dup, shift_v);
        uint8x16_t idx     = vandq_u8(shifted, one_v);
        vst1q_u8(out + j, vqtbl1q_u8(c2s_vec, idx));
    }
    for (; j + 8 <= K; j += 8) {
        uint8x8_t bm_v = vdup_n_u8(bm[j >> 3]);
        uint8x8_t dup     = vtbl1_u8(bm_v, vget_low_u8(dup_v));
        uint8x8_t shifted = vshl_u8(dup, vget_low_s8(shift_v));
        uint8x8_t idx     = vand_u8(shifted, vget_low_u8(one_v));
        vst1_u8(out + j, vtbl1_u8(vget_low_u8(c2s_vec), idx));
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left_sym;
    }
    PROF_TOC(PROF_BU_MERGE_CST_CST, K);
}

/* ---------- Flat-subtree decode (contiguous output) ----------
 *
 * Reads n*D packed bits, looks up each D-bit code in c2s, writes the
 * resulting bytes to out[0..n).  Output is dense / sequential -- the
 * BU codec calls this when it hits a PIVCO_NODE_INTERNAL_FLAT.
 *
 * One static-inline per supported D (2..8); merge_flat_neon
 * is a switch dispatcher.  The per-D unpack helpers
 * (flat_d{2,3,4,5,6,7}_unpack) come from pivco_huffman_neon_flat.h.
 */

/* Extract D bits at bit position `bit_pos` from `in`.  D <= 16.  Used
 * by each per-D function's non-aligned scalar tail. */
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

/* D=2 (4 codes/byte) */
static inline void merge_flat_d2_neon(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    int i = 0;
    if (n >= 64) {
        /* fast path maps each input nibble straight to a symbol pair via two
         * prepped tables (TL[n]=c2s[n&3], TH[n]=c2s[(n>>2)&3]) */
        static const uint8_t th_idx[16] = {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
        uint32_t w; memcpy(&w, c2s, 4);
        const uint8x16_t TL = vreinterpretq_u8_u32(vdupq_n_u32(w));   /* c2s[n&3] */
        const uint8x16_t TH = vqtbl1q_u8(TL, vld1q_u8(th_idx));       /* c2s[(n>>2)&3] */
        const uint8x16_t m  = vdupq_n_u8(0x0F);
        for (; i + 64 <= n; i += 64) {
            uint8x16_t v  = vld1q_u8(bm + (i >> 2));
            uint8x16_t lo = vandq_u8(v, m), hi = vshrq_n_u8(v, 4);
            /* four planar 16-symbol vectors, one per 2-bit code position */
            uint8x16x4_t o = {{ vqtbl1q_u8(TL, lo), vqtbl1q_u8(TH, lo),
                                vqtbl1q_u8(TL, hi), vqtbl1q_u8(TH, hi) }};
            /* store interleaved, restoring the original code order */
            vst4q_u8(symbols + i, o);
        }
    }
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    /* smaller inputs/tail => simpler 16-wide path with no extra prep. */
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
        uint32_t code = extract_D_bits_neon(bm, i * 2, 2);
        symbols[i] = c2s[code];
    }
}

/* D=3 (byte-crossing): 32 codes/iter.  Use the D=6 6-bit unpack to grab TWO
 * D=3 codes per byte (pair6 = c[2k] | c[2k+1]<<3) -- one gather+shift pass does
 * 32 codes -- then split lo=&7 (vqtbl1 over c2s16) / hi=>>3 (vqtbl2 over the
 * 32-byte repeated table, which ignores the high junk) and interleave with
 * vst2q.  A single 16-wide pair-gather block mops up the <32 remainder; the
 * trailing <=16 codes use the no-overread safe path (bounded by fast_end). */
static inline void merge_flat_d3_neon(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    const uint8x8_t  c2s8  = vld1_u8(c2s);
    const uint8x16_t c2s16 = vcombine_u8(c2s8, c2s8);
    const uint8x16_t m7    = vdupq_n_u8(7);
    int i = 0;
    int fast_end = n >= 16 ? n - 16 : 0;
    const uint8_t *bp = bm;
    if (n >= 48) {
        uint8x16x2_t c2s32; c2s32.val[0] = c2s16; c2s32.val[1] = c2s16;
        static const uint8_t pair6_shuf_t[16] = { 0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11 };
        static const int16_t hshift6_t[8]     = { 2,-2, 2,-2, 2,-2, 2,-2 };
        static const int8_t  bshr6_t[16]      = { -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0 };
        const uint8x16_t pair6_shuf = vld1q_u8(pair6_shuf_t);
        const int16x8_t  hshift6    = vld1q_s16(hshift6_t);
        const int8x16_t  bshr6      = vld1q_s8(bshr6_t);
        for (; i + 32 <= fast_end; i += 32, bp += 12) {
            uint8x16_t packed = vld1q_u8(bp);
            uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(packed, pair6_shuf));
            x = vshlq_u16(x, hshift6);
            uint8x16_t pair6 = vshlq_u8(vreinterpretq_u8_u16(x), bshr6);
            uint8x16x2_t out;
            out.val[0] = vqtbl1q_u8(c2s16, vandq_u8(pair6, m7));
            out.val[1] = vqtbl2q_u8(c2s32, vshrq_n_u8(pair6, 3));
            vst2q_u8(symbols + i, out);
        }
    }
    if (i + 16 <= fast_end) {   /* one 16-wide pair-gather block for the <32 remainder */
        static const uint8_t pair_shuf_t[16] = { 0,1, 0,1, 1,2, 2,3, 3,4, 3,4, 4,5, 5,6 };
        static const int16_t hshift_t[8]     = { 5,-1, 1, 3, 5,-1, 1, 3 };
        static const int8_t  bshr_t[16]      = { -5,0, -5,0, -5,0, -5,0, -5,0, -5,0, -5,0, -5,0 };
        uint8x16_t packed = vld1q_u8(bp);
        uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(packed, vld1q_u8(pair_shuf_t)));
        x = vshlq_u16(x, vld1q_s16(hshift_t));
        uint8x16_t y = vshlq_u8(vreinterpretq_u8_u16(x), vld1q_s8(bshr_t));
        vst1q_u8(symbols + i, vqtbl1q_u8(c2s16, vandq_u8(y, m7)));
        i += 16; bp += 6;
    }
    for (; i + 8 <= n; i += 8) {
        uint8x8_t codes = flat_d3_unpack_safe(bm + ((i * 3) >> 3));
        vst1_u8(symbols + i, vqtbl1_u8(c2s16, codes));
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 3, 3);
        symbols[i] = c2s[code];
    }
}

/* D=4: codes are nibbles (2/byte), so &0xF / >>4 index the plain c2s directly
 * (no dup-shuffle TBL); 32/iter via vzip + plain vst1q.  Stock 16-wide tail. */
static inline void merge_flat_d4_neon(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    const uint8x16_t m = vdupq_n_u8(0x0F);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        uint8x16_t v  = vld1q_u8(bm + (i >> 1));
        uint8x16_t lo = vandq_u8(v, m), hi = vshrq_n_u8(v, 4);
        uint8x16_t a = vqtbl1q_u8(c2s_vec, lo), b = vqtbl1q_u8(c2s_vec, hi);
        vst1q_u8(symbols + i,      vzip1q_u8(a, b));
        vst1q_u8(symbols + i + 16, vzip2q_u8(a, b));
    }
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
        uint32_t code = extract_D_bits_neon(bm, i * 4, 4);
        symbols[i] = c2s[code];
    }
}

/* D=5 (byte-crossing): pair-gather puts two adjacent codes in one u16 lane,
 * positioned so a byte reinterpret interleaves even/odd for free (no vtrn1);
 * vshr.u8(even lanes) + vand clean to 0..31; vqtbl2 scatter.  Setup is gated on the
 * block condition; the stock safe path handles the remainder. */
static inline void merge_flat_d5_neon(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16x2_t c2s_vec;
    c2s_vec.val[0] = vld1q_u8(c2s);
    c2s_vec.val[1] = vld1q_u8(c2s + 16);
    int i = 0;
    if (n >= 25) {
        static const uint8_t pair_shuf_t[16] = { 0,1, 1,2, 2,3, 3,4, 5,6, 6,7, 7,8, 8,9 };
        static const int16_t hshift_t[8]     = { 3, 1, -1, -3, 3, 1, -1, -3 };
        static const int8_t  bshr_t[16]      = { -3,0, -3,0, -3,0, -3,0, -3,0, -3,0, -3,0, -3,0 };
        const uint8x16_t pair_shuf = vld1q_u8(pair_shuf_t);
        const int16x8_t  hshift    = vld1q_s16(hshift_t);
        const int8x16_t  bshr      = vld1q_s8(bshr_t);
        const uint8x16_t m31       = vdupq_n_u8(0x1f);
        int blocks = (n - 9) >> 4;
        for (int b = 0; b < blocks; ++b) {
            uint8x16_t packed = vld1q_u8(bm + b * 10);
            uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(packed, pair_shuf));
            x = vshlq_u16(x, hshift);
            uint8x16_t y = vshlq_u8(vreinterpretq_u8_u16(x), bshr);
            uint8x16_t idx = vandq_u8(y, m31);
            vst1q_u8(symbols + (b << 4), vqtbl2q_u8(c2s_vec, idx));
        }
        i = blocks << 4;
    }
    for (; i + 8 <= n; i += 8) {
        uint8x8_t codes = flat_d5_unpack_safe(bm + ((i * 5) >> 3));
        uint8x8_t syms  = vqtbl2_u8(c2s_vec, codes);
        vst1_u8(symbols + i, syms);
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 5, 5);
        symbols[i] = c2s[code];
    }
}

/* D=6: same pair-gather as D=5 (12-bit pairs, even/odd in one u16 lane), but
 * the c2s is 64 bytes so the scatter is vqtbl4q.  Setup gated; stock safe tail. */
static inline void merge_flat_d6_neon(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16x4_t c2s_vec;
    c2s_vec.val[0] = vld1q_u8(c2s);
    c2s_vec.val[1] = vld1q_u8(c2s + 16);
    c2s_vec.val[2] = vld1q_u8(c2s + 32);
    c2s_vec.val[3] = vld1q_u8(c2s + 48);
    int i = 0;
    if (n >= 24) {
        static const uint8_t pair_shuf_t[16] = { 0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11 };
        static const int16_t hshift_t[8]     = { 2,-2, 2,-2, 2,-2, 2,-2 };
        static const int8_t  bshr_t[16]      = { -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0 };
        const uint8x16_t pair_shuf = vld1q_u8(pair_shuf_t);
        const int16x8_t  hshift    = vld1q_s16(hshift_t);
        const int8x16_t  bshr      = vld1q_s8(bshr_t);
        const uint8x16_t m63       = vdupq_n_u8(0x3f);
        int blocks = (n - 8) >> 4;
        for (int b = 0; b < blocks; ++b) {
            uint8x16_t packed = vld1q_u8(bm + b * 12);
            uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(packed, pair_shuf));
            x = vshlq_u16(x, hshift);
            uint8x16_t y = vshlq_u8(vreinterpretq_u8_u16(x), bshr);
            uint8x16_t idx = vandq_u8(y, m63);
            vst1q_u8(symbols + (b << 4), vqtbl4q_u8(c2s_vec, idx));
        }
        i = blocks << 4;
    }
    for (; i + 8 <= n; i += 8) {
        uint8x8_t codes = flat_d6_unpack_safe(bm + ((i * 6) >> 3));
        uint8x8_t syms  = vqtbl4_u8(c2s_vec, codes);
        vst1_u8(symbols + i, syms);
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 6, 6);
        symbols[i] = c2s[code];
    }
}

/* D=7: 128-entry c2s = 2 * vqtbl4 (= 64).  vqtbl4 on the low half +
 * vqtbx4 on the high half (with code-64 indexing) — vqtbx keeps the
 * first result for out-of-range lanes, so no OR-merge needed. */
static inline void merge_flat_d7_neon(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16x4_t lo, hi;
    lo.val[0] = vld1q_u8(c2s);       lo.val[1] = vld1q_u8(c2s + 16);
    lo.val[2] = vld1q_u8(c2s + 32);  lo.val[3] = vld1q_u8(c2s + 48);
    hi.val[0] = vld1q_u8(c2s + 64);  hi.val[1] = vld1q_u8(c2s + 80);
    hi.val[2] = vld1q_u8(c2s + 96);  hi.val[3] = vld1q_u8(c2s + 112);
    uint8x16_t sub64q = vdupq_n_u8(64);
    uint8x8_t  sub64  = vdup_n_u8(64);
    int i = 0;
    int fast_end = n >= 24 ? n - 24 : 0;
    /* 16-wide ryg unpack (from the 2026-07-30 csimd study, +14% over the
     * two-per-8-helper form on Graviton 4): 16 codes span exactly 14 bytes
     * (16*7 bits), so one 16 B window feeds a TBL byte-pair gather; a
     * per-lane USHL right-shift ((pos&7) as negative counts) bottoms each
     * field, vuzp1 keeps the low bytes, AND 0x7F masks bit 7.  Groups are
     * byte-aligned every 16 codes (stride 14).  Map via independent
     * tbl/tbl/orr (no tbx dependency chain).  Instruction-for-instruction
     * the sequence clang emits for the csimd-ryg-map bench variant. */
    {
        static const uint8_t d7_gather_lo_t[16] =
            { 0,1, 0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7 };
        static const uint8_t d7_gather_hi_t[16] =
            { 7,8, 7,8, 8,9, 9,10, 10,11, 11,12, 12,13, 13,14 };
        static const int16_t d7_shift_t[8] =           /* -(pos & 7) */
            { 0, -7, -6, -5, -4, -3, -2, -1 };
        const uint8x16_t gather_lo = vld1q_u8(d7_gather_lo_t);
        const uint8x16_t gather_hi = vld1q_u8(d7_gather_hi_t);
        const int16x8_t  shift     = vld1q_s16(d7_shift_t);
        const uint8x16_t m7f       = vdupq_n_u8(0x7F);
        const uint8_t   *wp        = bm;
        for (; i + 16 <= fast_end; i += 16, wp += 14) {
            uint8x16_t win = vld1q_u8(wp);
            uint16x8_t vl = vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(win, gather_lo)), shift);
            uint16x8_t vh = vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(win, gather_hi)), shift);
            uint8x16_t codes = vandq_u8(vuzp1q_u8(vreinterpretq_u8_u16(vl),
                                                  vreinterpretq_u8_u16(vh)), m7f);
            uint8x16_t s = vorrq_u8(vqtbl4q_u8(lo, codes),
                                    vqtbl4q_u8(hi, vsubq_u8(codes, sub64q)));
            vst1q_u8(symbols + i, s);
        }
    }
    for (; i + 8 <= fast_end; i += 8) {
        uint8x8_t codes = flat_d7_unpack_fast(bm + ((i * 7) >> 3));
        uint8x8_t s = vqtbl4_u8(lo, codes);
        s = vqtbx4_u8(s, hi, vsub_u8(codes, sub64));
        vst1_u8(symbols + i, s);
    }
    for (; i + 8 <= n; i += 8) {
        uint8x8_t codes = flat_d7_unpack_safe(bm + ((i * 7) >> 3));
        uint8x8_t s = vqtbl4_u8(lo, codes);
        s = vqtbx4_u8(s, hi, vsub_u8(codes, sub64));
        vst1_u8(symbols + i, s);
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 7, 7);
        symbols[i] = c2s[code];
    }
}

/* D=8: a depth-8 flat region has 2^8 = 256 leaves = the WHOLE byte alphabet,
 * all at code length 8.  A full-alphabet equal-length canonical code is the
 * identity permutation (rank == symbol), so c2s[k] == k and the byte-aligned
 * 8-bit codes ARE the symbols: out[i] = c2s[bm[i]] = bm[i].  Hence a plain
 * memcpy -- no 256-entry vqtbl4/vqtbx4 needed.  The caller (a full-alphabet
 * flat root) guarantees c2s == identity for D=8; only reachable for
 * near-uniform / incompressible blocks (ratio ~1.0). */
static inline void merge_flat_d8_neon(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    (void)c2s;
    memcpy(symbols, bm, (size_t)n);
}

/* merge_flat_neon -- D-bit flat-subtree decode into a
 * contiguous output buffer.  Dispatches to the per-D specialisation. */
static inline void merge_flat_neon(uint8_t *out, int n,
                                                const uint8_t *bm, int D,
                                                const uint8_t *c2s)
{
    PROF_TIC();
    switch (D) {
    case 2: merge_flat_d2_neon(out, n, bm, c2s); break;
    case 3: merge_flat_d3_neon(out, n, bm, c2s); break;
    case 4: merge_flat_d4_neon(out, n, bm, c2s); break;
    case 5: merge_flat_d5_neon(out, n, bm, c2s); break;
    case 6: merge_flat_d6_neon(out, n, bm, c2s); break;
    case 7: merge_flat_d7_neon(out, n, bm, c2s); break;
    case 8: merge_flat_d8_neon(out, n, bm, c2s); break;
    default: {
        /* Generic fallback for any unhandled D <= 16. */
        for (int i = 0; i < n; i++) {
            uint32_t code = extract_D_bits_neon(bm, i * D, D);
            out[i] = c2s[code];
        }
        break;
    }
    }
    PROF_TOC(PROF_BU_MERGE_FLAT, n);
}


/* ---------- Encode primitives: rank-based encoding (8-bit in-order ranks) ----------
 * Partition 8-bit leaf ranks against a per-node threshold (split_rank).  A
 * u8 port of the code_la COM64 partition: masks64_neon builds the
 * 8 chunk masks (vcgtq > thr replacing the code bit-test), a vcnt + 0x0101..
 * prefix sum precomputes per-chunk cursors, and each 8-rank chunk is compacted
 * by a vtbl1_u8 over ctab8 (the 1-byte-per-rank analog of compress_tab).
 * Flat pack subtracts flat_base_rank then reuses the production pack_dN. */
#include <stdlib.h>

/* Per-mask LUTs (built once by build_tabs):
 *   pc8[m]          popcount of mask byte m
 *   ctab8[m][0:8]   right source lanes packed at [0,n_right), 0xff fill
 *   ctab8[m][8:16]  left  source lanes packed at [0,n_left), 0xff fill
 * vtbl1_u8 returns 0 for the 0xff (out-of-range) padding indices. */
static uint8_t pc8[256];
static uint8_t ctab8[256][16]  __attribute__((aligned(16)));

/* p16rev partition LUTs (part_full_neon).  One combined index per 16-lane group
 * packs {left, forward, front} | {right, reversed, back}; left+right tile the
 * 16 lanes so the OR of two disjoint-support tables is exact.
 *   p16rev_tabA[m0]       low-byte (positions 0..7): left -> front [0,8-pc0),
 *                       right -> back lanes 15,14,... (reversed)
 *   p16rev_tabB0[m1]     high-byte (positions 8..15): continues both runs after
 *                       the low byte, for pc0=0.  The pc0>0 layout is just this
 *                       one shifted left by pc0 lanes, so tabB[pc0][m1] is
 *                       recovered as a byte-offset load `tabB0[m1] + pc0` (no
 *                       separate per-pc0 table).  Padded to 32 B/entry so the
 *                       offset-16 load (pc0<=8) stays inside one cache line;
 *                       8 KB total vs the former 36 KB (fits L1 alongside tabA).
 * The right side is recovered with a single loop-invariant full-reverse
 * constant in part_full_neon. */
static uint8_t p16rev_tabA[256][16]  __attribute__((aligned(16)));
static uint8_t p16rev_tabB0[256][32] __attribute__((aligned(32)));
static int     tabs_ready = 0;

static void build_tabs(void)
{
    if (tabs_ready) return;
    for (int m = 0; m < 256; m++) {
        pc8[m] = (uint8_t)__builtin_popcount(m);
        memset(ctab8[m], 0xff, 16);
        int qr = 0, ql = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) ctab8[m][qr++]     = (uint8_t)k;     /* right -> [0:8]  */
            else              ctab8[m][8 + ql++] = (uint8_t)k;     /* left  -> [8:16] */
        }
    }
    for (int m0 = 0; m0 < 256; m0++) {
        memset(p16rev_tabA[m0], 0, 16);
        int lp = 0, rp = 15;
        for (int k = 0; k < 8; k++) {
            if ((m0 >> k) & 1) p16rev_tabA[m0][rp--] = (uint8_t)k;
            else               p16rev_tabA[m0][lp++] = (uint8_t)k;
        }
    }
    for (int m1 = 0; m1 < 256; m1++) {
        memset(p16rev_tabB0[m1], 0, 32);
        int lp = 8, rp = 15;   /* pc0 = 0 layout; pc0 > 0 handled by the load offset */
        for (int k = 0; k < 8; k++) {
            if ((m1 >> k) & 1) p16rev_tabB0[m1][rp--] = (uint8_t)(8 + k);
            else               p16rev_tabB0[m1][lp++] = (uint8_t)(8 + k);
        }
    }
    tabs_ready = 1;
}

static const uint8_t BW8[8] = {1, 2, 4, 8, 16, 32, 64, 128};

/* 8-bit mask of (ids > thr) over the 8 ranks in `ids`. */
static inline uint8_t nmask8(uint8x8_t ids, uint8x8_t thr)
{
    return vaddv_u8(vand_u8(vcgt_u8(ids, thr), vld1_u8(BW8)));
}

/* enc_init: ranks[i] = sym_to_rank[sym[i]], a 256-entry byte gather.
 * "simd20" version from #5 by dougallj.
 * The s2r table lives in 16 NEON regs (4x uint8x16x4_t).
 * Each 16-lane input does one vqtbl4 over the [0,63] half
 * + three vqtbx4 over the +64/+128/+192 halves (with offset adjusted).
 * The tbl/tbx are microcoded and leave scalar load slots idle, so
 * 4 extra symbols/iter are done with GPR gathers interleaved between them
 * (20 sym/iter total).
 */
static inline void init_neon(uint8_t *ranks, int n,
                                const uint8_t *sym, const uint8_t *s2r)
{
    int i = 0;
    if (n >= 20) {
        uint8x16x4_t t0, t1, t2, t3;
        t0.val[0]=vld1q_u8(s2r     ); t0.val[1]=vld1q_u8(s2r + 16);
        t0.val[2]=vld1q_u8(s2r + 32); t0.val[3]=vld1q_u8(s2r + 48);
        t1.val[0]=vld1q_u8(s2r + 64); t1.val[1]=vld1q_u8(s2r + 80);
        t1.val[2]=vld1q_u8(s2r + 96); t1.val[3]=vld1q_u8(s2r +112);
        t2.val[0]=vld1q_u8(s2r +128); t2.val[1]=vld1q_u8(s2r +144);
        t2.val[2]=vld1q_u8(s2r +160); t2.val[3]=vld1q_u8(s2r +176);
        t3.val[0]=vld1q_u8(s2r +192); t3.val[1]=vld1q_u8(s2r +208);
        t3.val[2]=vld1q_u8(s2r +224); t3.val[3]=vld1q_u8(s2r +240);
        const uint8x16_t s64  = vdupq_n_u8(64);
        const uint8x16_t s128 = vdupq_n_u8(128);
        const uint8x16_t s192 = vdupq_n_u8(192);
        for (; i + 20 <= n; i += 20) {
            uint8x16_t c = vld1q_u8(sym + i);
            uint32_t a; memcpy(&a, sym + i + 16, 4);
            uint8x16_t r = vqtbl4q_u8(t0, c);
            unsigned r0 = s2r[(uint8_t)a];
            r = vqtbx4q_u8(r, t1, vsubq_u8(c, s64));
            unsigned r1 = s2r[(uint8_t)(a >> 8)];
            r = vqtbx4q_u8(r, t2, vsubq_u8(c, s128));
            unsigned r2 = s2r[(uint8_t)(a >> 16)];
            r = vqtbx4q_u8(r, t3, vsubq_u8(c, s192));
            unsigned r3 = s2r[(uint8_t)(a >> 24)];
            vst1q_u8(ranks + i, r);
            uint32_t h = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
            memcpy(ranks + i + 16, &h, 4);
        }
    }
    for (; i < n; i++) ranks[i] = s2r[sym[i]];
}

/* Build 8 partition mask bytes for 64 ranks in one vpaddq_u8 reduction tree,
 * packed LE into a u64 (byte k = mask of chunk k = ranks[8k .. 8k+7]).  The
 * rank analog of enc_masks8x8_codes_la_neon: vcgtq replaces the code bit-test,
 * and since u8 packs two 8-rank chunks per 128-bit vector, FOUR inputs (not
 * eight) feed the pairwise-add tree.  Each lane already holds its bit-weight
 * (0 or 2^(lane&7)); the 4 vpaddq_u8 collapse all 8 lanes of every chunk into
 * one byte, so r's low 8 bytes are mask_0..mask_7 directly.  This replaces the
 * old 4x mred (12 vpaddq) + 8 vgetq_lane SIMD->GPR extracts with 4 vpaddq +
 * one vget_lane_u64 -- the chunk masks now arrive as a single word that also
 * feeds a vcnt popcount with no stack round-trip. */
/* masks64v returns the mask bytes in a D-register so the caller can vcnt
 * them before the GPR move, keeping the cursor chain off the GPR->SIMD
 * fmov (a load-port uop, ~6cy mid-chain). */
static inline uint8x8_t masks64v_neon(uint8x16_t v0, uint8x16_t v1,
                                      uint8x16_t v2, uint8x16_t v3,
                                      uint8x16_t vt, uint8x16_t bw)
{
    uint8x16_t w0 = vandq_u8(vcgtq_u8(v0, vt), bw);   /* chunks 0,1 */
    uint8x16_t w1 = vandq_u8(vcgtq_u8(v1, vt), bw);   /* chunks 2,3 */
    uint8x16_t w2 = vandq_u8(vcgtq_u8(v2, vt), bw);   /* chunks 4,5 */
    uint8x16_t w3 = vandq_u8(vcgtq_u8(v3, vt), bw);   /* chunks 6,7 */
    uint8x16_t t0 = vpaddq_u8(w0, w1);
    uint8x16_t t1 = vpaddq_u8(w2, w3);
    uint8x16_t u0 = vpaddq_u8(t0, t1);
    return vget_low_u8(vpaddq_u8(u0, u0));            /* low 8 bytes = mask_0..7 */
}
static inline uint64_t masks64_neon(uint8x16_t v0, uint8x16_t v1,
                                       uint8x16_t v2, uint8x16_t v3,
                                       uint8x16_t vt, uint8x16_t bw)
{
    return vget_lane_u64(vreinterpret_u64_u8(
               masks64v_neon(v0, v1, v2, v3, vt, bw)), 0);
}

/* full: both sides compacted (right -> tmp, left in place into ranks).
 * p16rev: per 16-lane group, ONE combined shuffle index packs {left, forward,
 * front} | {right, reversed, back}.  Left and right exactly tile the 16 lanes,
 * so the OR of two disjoint-support tables (p16rev_tabA over the low-byte mask m0,
 * p16rev_tabB over [pc0][m1]) is exact.  One vqtbl1q over that index yields BOTH
 * sides at once: the register IS the left output (store it, advance by the left
 * count — the right tail is overwritten by the next group / recursion level);
 * the right output is recovered with a second vqtbl1q over the SAME register
 * using a single loop-invariant full-reverse constant (full reverse lands the
 * top-pc reversed right lanes at output [0,pc); the tail is overwritten).
 * vs the prior per-8-chunk ctab8 COM64 path: one table-pair OR + one shuffle
 * per 16 lanes instead of two independent 8-lane shuffles — measured 4–22 %
 * faster across M4 / Graviton2..4 / Neoverse V3 (see bench_prim `com64` vs
 * `p16rev`).  The ~40 KB p16rev tables (tabA 4 KB + tabB 36 KB) make it NEON / big-
 * L1 only; the 16-byte tail overstore is absorbed by the ranks +64 / tmp +2N
 * scratch slack (codec.c). */
/* Scatter one 64-rank group-set (4 p16rev groups as above).  Prefix-summed
 * popcounts give each group's store offsets up front, so the cursors
 * advance once per 64 (issue #5).  Stores run 16 wide, up to +48/+16 past
 * the valid counts into the ranks+64 / tmp+2N scratch slack.  Returns the
 * group-set's right count. */
__attribute__((always_inline)) static inline
int part64_full_neon(uint8x16_t v0, uint8x16_t v1, uint8x16_t v2, uint8x16_t v3,
                     uint64_t mask_word, uint64_t pcw,
                     uint8_t *ldst, uint8_t *rdst)
{
    uint64_t pfx = pcw * 0x0101010101010101ULL;
    uint8x16_t vg[4] = { v0, v1, v2, v3 };
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
#define _PART(g) do {                                                       \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                     \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                 \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);            \
        uint32_t cr  = (g) == 0 ? 0u                                        \
                     : (uint32_t)((pfx >> (8*(2*(g) - 1))) & 0xFF);         \
        uint8x16_t ri = vorrq_u8(vld1q_u8(p16rev_tabA[m0]),                 \
                                 vld1q_u8(&p16rev_tabB0[m1][pc0]));         \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                           \
        vst1q_u8(ldst + (16*(g) - cr), comb);                              \
        vst1q_u8(rdst + cr, vqtbl1q_u8(comb, rev16));                       \
    } while (0)
    _PART(0); _PART(1); _PART(2); _PART(3);
#undef _PART
    return (int)(pfx >> 56);
}
static inline int part_full_neon(uint8_t *ranks, int n, uint8_t thr,
                                    uint8_t *bm, uint8_t *tmp)
{
    build_tabs();
    int n_left = 0, n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    /* Software-pipelined one iteration deep (like the decode merges): the
     * carried chain (loads -> cgt -> three serial vpaddq -> lane move ->
     * multiply -> cursors) is ~2x the loop's port work, so the next
     * group-set's mask + popcount start under the current scatter.  The
     * mask is vcnt'd SIMD-side (masks64v_neon), keeping the GPR->SIMD
     * fmov off the chain. */
    if (j + 64 <= n) {
        uint8x16_t c0 = vld1q_u8(ranks + j),      c1 = vld1q_u8(ranks + j + 16);
        uint8x16_t c2 = vld1q_u8(ranks + j + 32), c3 = vld1q_u8(ranks + j + 48);
        uint8x8_t mv = masks64v_neon(c0, c1, c2, c3, vt, bw);
        uint64_t w   = vget_lane_u64(vreinterpret_u64_u8(mv), 0);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(mv)), 0);
        for (; j + 128 <= n; j += 64) {
            uint8x16_t n0 = vld1q_u8(ranks + j + 64), n1 = vld1q_u8(ranks + j + 80);
            uint8x16_t n2 = vld1q_u8(ranks + j + 96), n3 = vld1q_u8(ranks + j + 112);
            uint8x8_t nmv = masks64v_neon(n0, n1, n2, n3, vt, bw);
            uint64_t nw   = vget_lane_u64(vreinterpret_u64_u8(nmv), 0);
            uint64_t npcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(nmv)), 0);
            memcpy(bm + (j >> 3), &w, 8);
            int tr = part64_full_neon(c0, c1, c2, c3, w, pcw,
                                      ranks + n_left, tmp + n_right);
            n_right += tr;
            n_left  += 64 - tr;
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
            w = nw; pcw = npcw;
        }
        memcpy(bm + (j >> 3), &w, 8);
        int tr = part64_full_neon(c0, c1, c2, c3, w, pcw,
                                  ranks + n_left, tmp + n_right);
        n_right += tr;
        n_left  += 64 - tr;
        j += 64;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}

/* part_core_neon — the one-sided (right/none) rank partition, a
 * u8 port of the code_la partition core: same 64/iter COM64 wide path (mask via
 * masks64_neon, vcnt + 0x0101.. prefix-sum cursors, per-8-chunk ctab8
 * shuffle), same 8/iter middle loop, same scalar tail.  EMIT_RIGHT is
 * compile-time, so the none form folds to a pure bitmap build.  Right ->
 * tmp; the left side is never scattered (a leaf child's ranks are dead). */
__attribute__((always_inline)) static inline
int part_core_neon(uint8_t *ranks, int n, uint8_t thr,
                      uint8_t *bm, uint8_t *tmp, int EMIT_RIGHT)
{
    build_tabs();
    int n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    uint8x8_t  vt8 = vdup_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = masks64_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint8x8_t pc_v = vcnt_u8(vcreate_u8(mask_word));
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_word * 0x0101010101010101ULL;
        uint8x8_t cv[8] = {
            vget_low_u8(v0), vget_high_u8(v0),
            vget_low_u8(v1), vget_high_u8(v1),
            vget_low_u8(v2), vget_high_u8(v2),
            vget_low_u8(v3), vget_high_u8(v3),
        };
#define _PART1(K_) do {                                                    \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF); \
        if (EMIT_RIGHT) {                                                    \
            const uint8_t *tab = ctab8[(uint8_t)(mask_word >> (8*(K_)))];   \
            vst1_u8(tmp + n_right + cr, vtbl1_u8(cv[K_], vld1_u8(tab)));      \
        }                                                                    \
    } while (0)
        _PART1(0); _PART1(1); _PART1(2); _PART1(3);
        _PART1(4); _PART1(5); _PART1(6); _PART1(7);
#undef _PART1
        n_right += (uint32_t)(pfx >> 56);
    }
    for (; j + 8 <= n; j += 8) {
        uint8x8_t v = vld1_u8(ranks + j);
        uint8_t mask = nmask8(v, vt8);
        bm[j >> 3] = mask;
        if (EMIT_RIGHT) vst1_u8(tmp + n_right, vtbl1_u8(v, vld1_u8(ctab8[mask])));
        n_right += pc8[mask];
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7));
                       if (EMIT_RIGHT) tmp[n_right] = r; n_right++; }
    }
    return n_right;
}

/* Flat pack, native u8: the local code (rank - base) is already a D-bit value
 * in the low bits of each byte, so we pack straight from u8 — no u16 widen, no
 * round-trip.  Per-D byte kernels mirror the code_la packers (D5/6/7 reuse the
 * byte-laid backend from pivco_huffman_neon_pack.h via pack_d{5,6,7}). */

/* D=2: 16 ranks -> 4 bytes (4 ranks per byte, no byte crossings). */
static inline int pack_d2_neon(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d2[16] = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    const int8x16_t sh = vld1q_s8(shifts_d2);
    /* Distribute the base subtract: shift raw ranks, fold four into a
     * byte, subtract 85*base once --
     * (r0-b)+4(r1-b)+16(r2-b)+64(r3-b) = r0+4r1+16r2+64r3 - 85b, exact
     * mod 256. */
    const uint8x16_t b85 = vdupq_n_u8((uint8_t)(85 * base));
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        uint8x16_t b0 = vshlq_u8(vld1q_u8(ranks + i),      sh);
        uint8x16_t b1 = vshlq_u8(vld1q_u8(ranks + i + 16), sh);
        uint8x16_t b2 = vshlq_u8(vld1q_u8(ranks + i + 32), sh);
        uint8x16_t b3 = vshlq_u8(vld1q_u8(ranks + i + 48), sh);
        uint8x16_t r  = vpaddq_u8(vpaddq_u8(b0, b1), vpaddq_u8(b2, b3));
        vst1q_u8(out + (i >> 2), vsubq_u8(r, b85));
    }
    for (; i + 16 <= n; i += 16) {   /* 16-wide cleanup */
        uint8x16_t b  = vshlq_u8(vld1q_u8(ranks + i), sh);
        uint8x16_t s1 = vpaddq_u8(b, b);
        uint8x16_t s2 = vsubq_u8(vpaddq_u8(s1, s1), b85);
        uint32_t packed4 = vgetq_lane_u32(vreinterpretq_u32_u8(s2), 0);
        memcpy(out + (i >> 2), &packed4, 4);
    }
    return i;
}

/* D=3: pair adjacent codes into 6-bit values the D=4 way (per-lane {0,3}
 * shifts + one vpaddq: pair = c_even + 8 c_odd, one per byte; base
 * subtract distributed as - 9*base, exact since the true pair < 64),
 * then run the D=6 variable-shift pyramid on the pairs -- a 3-bit
 * LSB-first stream is exactly the 6-bit LSB-first stream of its pairs.
 * 32 codes/iter plus a 16-code self-paired cleanup, store-bounded like
 * pack_d{5,6,7} with the scalar tail packing the rest. */
static inline int pack_d3_neon(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t  shifts_p[16] = { 0,3, 0,3, 0,3, 0,3, 0,3, 0,3, 0,3, 0,3 };
    static const int8_t  shifts_1[16] = { 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0 };
    static const int16_t shifts_2[8]  = { 2,-2, 2,-2, 2,-2, 2,-2 };
    static const int32_t shifts_4[4]  = { 4,-4, 4,-4 };
    const int8x16_t  shp = vld1q_s8(shifts_p);
    const uint8x16_t b9  = vdupq_n_u8((uint8_t)(9 * base));
    const int8x16_t  s1  = vld1q_s8(shifts_1);
    const int16x8_t  s2  = vld1q_s16(shifts_2);
    const int32x4_t  s3  = vld1q_s32(shifts_4);
    const uint8x16_t compact = vld1q_u8(pivco_pack_compact_d6_neon);
    const int total_bytes = (n * 3 + 7) >> 3;
    int i = 0;
    for (; i + 32 <= n && ((i * 3) >> 3) + 16 <= total_bytes; i += 32) {
        uint8x16_t b0   = vshlq_u8(vld1q_u8(ranks + i),      shp);
        uint8x16_t b1   = vshlq_u8(vld1q_u8(ranks + i + 16), shp);
        uint8x16_t pair = vsubq_u8(vpaddq_u8(b0, b1), b9);
        uint16x8_t w16  = vreinterpretq_u16_u8(vshlq_u8(pair, s1));
        uint32x4_t w32  = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
        uint64x2_t w64  = vreinterpretq_u64_u32(vshlq_u32(w32, s3));
        vst1q_u8(out + ((i * 3) >> 3),
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));
    }
    for (; i + 16 <= n && ((i * 3) >> 3) + 16 <= total_bytes; i += 16) {
        uint8x16_t b    = vshlq_u8(vld1q_u8(ranks + i), shp);
        uint8x16_t pair = vsubq_u8(vpaddq_u8(b, b), b9);
        uint16x8_t w16  = vreinterpretq_u16_u8(vshlq_u8(pair, s1));
        uint32x4_t w32  = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
        uint64x2_t w64  = vreinterpretq_u64_u32(vshlq_u32(w32, s3));
        vst1q_u8(out + ((i * 3) >> 3),
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));
    }
    return i;
}

/* D=4: 16 ranks -> 8 bytes.  Pair (r[2k], r[2k+1]) into one byte each. */
static inline int pack_d4_neon(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d4[16] = { 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4 };
    const int8x16_t sh = vld1q_s8(shifts_d4);
    /* Distributed base subtract, unrolled once so the vpaddq_u8 pairs two
     * full input vectors into one 16-byte store:
     * (r0-b)+16(r1-b) = r0+16r1 - 17b, exact mod 256. */
    const uint8x16_t b17 = vdupq_n_u8((uint8_t)(17 * base));
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        uint8x16_t b0 = vshlq_u8(vld1q_u8(ranks + i),      sh);
        uint8x16_t b1 = vshlq_u8(vld1q_u8(ranks + i + 16), sh);
        vst1q_u8(out + (i >> 1), vsubq_u8(vpaddq_u8(b0, b1), b17));
    }
    for (; i + 16 <= n; i += 16) {   /* 16-wide cleanup */
        uint8x16_t b = vshlq_u8(vld1q_u8(ranks + i), sh);
        vst1_u8(out + (i >> 1), vget_low_u8(vsubq_u8(vpaddq_u8(b, b), b17)));
    }
    return i;
}

/* D=8: 16 ranks -> 16 bytes.  Byte-aligned; one shift+AND pass. */
static inline int pack_d8_neon(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    uint8x16_t vb = vdupq_n_u8(base);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_u8(out + i, vsubq_u8(vld1q_u8(ranks + i), vb));
    }
    return i;
}

/* Dispatcher: SIMD per-D path + scalar tail (packs (rank - base) LSB-first). */
static inline void pack_dN_neon(uint8_t *out, const uint8_t *ranks,
                                   int n, int D, uint8_t base)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;

    int i = 0;
    switch (D) {
    case 2: i = pack_d2_neon(out, ranks, n, base); break;
    case 3: i = pack_d3_neon(out, ranks, n, base); break;
    case 4: i = pack_d4_neon(out, ranks, n, base); break;
    case 5: i = pack_d5_neon(out, ranks, n, base); break;
    case 6: i = pack_d6_neon(out, ranks, n, base); break;
    case 7: i = pack_d7_neon(out, ranks, n, base); break;
    case 8: i = pack_d8_neon(out, ranks, n, base); break;
    default: break;
    }
    if (i >= n) return;

    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = (uint32_t)(uint8_t)(ranks[i] - base);  /* code in [0,2^D); no mask */
        buf |= (uint64_t)local << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
}

/* ---------- Aliases consumed by codec.c ---------- */

#define PIVCO_PRIM_ALWAYS_INLINE __attribute__((always_inline)) static inline

#include "pivco_huffman_hist_scalar.h"

/* NEON has no histogram win over the shared scalar core (measured);
 * alias it explicitly. */
PIVCO_PRIM_ALWAYS_INLINE void prim_histogram_chunk(const uint8_t *in, size_t n,
                                                   uint32_t hist[256],
                                                   uint8_t *scratch)
{ histogram_chunk_scalar(in, n, hist, scratch); }


/* Widest load a merge kernel issues at a child-buffer cursor (16B vld1q at child cursors);
 * the cursor can rest AT `size` on the exhausted side, so buffers a
 * merge reads need this much trailing slack.  Consumed by the decode
 * placement logic (scratch_carve / place_tail). */
#define PIVCO_PRIM_MERGE_OVERREAD 16

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ codec_init_neon(); }


/* rank-based encode aliases (consumed by codec.c) */
PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint8_t *ranks, int n,
                                             const uint8_t *symbols, const uint8_t *sym_to_rank,
                                             const pivco_enc_init_aux_t *aux)
{ (void)aux; init_neon(ranks, n, symbols, sym_to_rank); }
PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_full(uint8_t *ranks, int n,
                                             uint8_t thr, uint8_t *bm, uint8_t *right_out)
{ return part_full_neon(ranks, n, thr, bm, right_out); }
PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_right(uint8_t *ranks, int n,
                                             uint8_t thr, uint8_t *bm, uint8_t *right_out)
{ return part_core_neon(ranks, n, thr, bm, right_out, 1); }
PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_none(uint8_t *ranks, int n,
                                             uint8_t thr, uint8_t *bm)
{ return part_core_neon(ranks, n, thr, bm, NULL, 0); }
PIVCO_PRIM_ALWAYS_INLINE void prim_enc_pack_dN(const uint8_t *ranks,
                                             int n, int D, uint8_t base, uint8_t *out_packed)
{ pack_dN_neon(out_packed, ranks, n, D, base); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat(uint8_t *out, int n,
                                                          const uint8_t *bm, int D,
                                                          const uint8_t *c2s)
{ merge_flat_neon(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_cst(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_cst_cst_neon(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_vec(const uint8_t *bm, int K,
                                                          uint8_t left_sym,
                                                          const uint8_t *right_buf,
                                                          uint8_t *out)
{ merge_cst_vec_neon(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_vec_vec(const uint8_t *bm, int K,
                                               const uint8_t *left_buf,
                                               const uint8_t *right_buf,
                                               uint8_t *out)
{ merge_vec_vec_neon(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_NEON_H */
