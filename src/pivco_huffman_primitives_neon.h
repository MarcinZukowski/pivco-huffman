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
#include "pivco_prof.h"

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

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

/* flat_decode_to_buffer_neon — D-bit flat-subtree decode into a
 * contiguous output buffer.  Calls into the legacy SIMD per-D
 * dispatcher in pivco_huffman_neon.c; once that file is retired in
 * Phase 3 step 3.8, this wrapper will pull the dispatch inline. */
extern void pivco_huffman_flat_decode_direct_neon_(uint8_t *symbols, int n,
                                                    const uint8_t *bm, int D,
                                                    const uint8_t *c2s);
static inline void flat_decode_to_buffer_neon(uint8_t *out, int n,
                                                const uint8_t *bm, int D,
                                                const uint8_t *c2s)
{
    PROF_TIC();
    pivco_huffman_flat_decode_direct_neon_(out, n, bm, D, c2s);
    PROF_TOC(PROF_BU_FLAT_DECODE, n);
}

/* ---------- Aliases consumed by codec.c ---------- */

#define PIVCO_PRIM_ALWAYS_INLINE __attribute__((always_inline)) static inline

PIVCO_PRIM_ALWAYS_INLINE void prim_flat_decode_to_buffer(uint8_t *out, int n,
                                                          const uint8_t *bm, int D,
                                                          const uint8_t *c2s)
{ flat_decode_to_buffer_neon(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_both_const(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_both_const_neon(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge_bcast_left(const uint8_t *bm, int K,
                                                          uint8_t left_sym,
                                                          const uint8_t *right_buf,
                                                          uint8_t *out)
{ tree_merge_bcast_left_neon(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge_bcast_right(const uint8_t *bm, int K,
                                                           const uint8_t *left_buf,
                                                           uint8_t right_sym,
                                                           uint8_t *out)
{ tree_merge_bcast_right_neon(bm, K, left_buf, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge(const uint8_t *bm, int K,
                                               const uint8_t *left_buf,
                                               const uint8_t *right_buf,
                                               uint8_t *out)
{ tree_merge_neon(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_NEON_H */
