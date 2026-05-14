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

/* ---------- Encode primitives (bitmap + partition) ----------
 *
 * The non-flat-internal-node hot path.  Builds the n-bit partition
 * bitmap from codes_la[0..n) (each codes_la[i] is the per-symbol left-
 * aligned Huffman code; bit (15 - depth) is the current depth's
 * partition decision) and partitions codes_la in place: left (bit==0)
 * stays in codes_la[0..n_left), right (bit==1) moves to tmp[0..n_right).
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

/* Stride-8 SIMD main path: load 8 left-aligned codes, build mask byte
 * via the dense movmask, partition the SAME register into left/right
 * halves using compress_tab[mask].  In-place write of the LEFT half
 * over codes_la (n_left <= j invariant keeps this safe even when the
 * 16-byte store extends past the cursor); RIGHT half goes to tmp.
 *
 * Per 8 elements: 1 vld, 4 NEON mask ops, 2 vld (shuf), 2 vqtbl,
 * 2 vst.  Scalar tail handles the residual 1..7 elements with the
 * same logic in plain C. */
static inline int build_bitmap_partition_neon(uint16_t *codes_la, int n,
                                                int depth,
                                                uint8_t *bm,
                                                uint16_t *tmp)
{
    int n_left = 0, n_right = 0;
    int j = 0;
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
        vst1q_u8((uint8_t *)(tmp      + n_right), right);
        vst1q_u8((uint8_t *)(codes_la + n_left ), left);
        n_right += nr;
        n_left  += (8 - nr);
    }

    /* Scalar tail.  Read all tail codes into a temporary before writing
     * back, since the in-place left write can overlap the read when
     * n_left + 8 > j (always true once we drop below a full group). */
    if (j < n) {
        int tail = n - j;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask = 0;
        int shift_d = 15 - depth;
        for (int k = 0; k < tail; k++) {
            int bit = (tail_buf[k] >> shift_d) & 1;
            mask |= (uint8_t)(bit << k);
        }
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k))
                tmp[n_right++] = tail_buf[k];
            else
                codes_la[n_left++] = tail_buf[k];
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
 * For the codec.c contract, prim_pack_dN(out, codes_la, n, D, depth)
 * forwards to pack_dN_dispatch_neon(out, codes_la, n, D, depth). */

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
    static const int64_t shifts_lo [2] = {0,   5};
    static const int64_t shifts_mid[2] = {10, 15};
    static const int64_t shifts_hi [2] = {20, 25};
    static const int64_t shifts_top[2] = {30, 35};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vshlq_u16(vld1q_u16(codes_la + i),
                                  vdupq_n_s16((int16_t)-right_shift));
        v = vandq_u16(v, vdupq_n_u16(0x1F));
        uint32x4_t v32_lo = vmovl_u16(vget_low_u16(v));
        uint32x4_t v32_hi = vmovl_u16(vget_high_u16(v));
        uint64x2_t a = vshlq_u64(vmovl_u32(vget_low_u32 (v32_lo)),
                                  vld1q_s64(shifts_lo));
        uint64x2_t b = vshlq_u64(vmovl_u32(vget_high_u32(v32_lo)),
                                  vld1q_s64(shifts_mid));
        uint64x2_t c = vshlq_u64(vmovl_u32(vget_low_u32 (v32_hi)),
                                  vld1q_s64(shifts_hi));
        uint64x2_t d = vshlq_u64(vmovl_u32(vget_high_u32(v32_hi)),
                                  vld1q_s64(shifts_top));
        uint64x2_t sum = vaddq_u64(vaddq_u64(a, b), vaddq_u64(c, d));
        uint64_t packed = vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1);
        int bi = i * 5 / 8;
        out[bi    ] = (uint8_t)(packed       );
        out[bi + 1] = (uint8_t)(packed >>  8 );
        out[bi + 2] = (uint8_t)(packed >> 16);
        out[bi + 3] = (uint8_t)(packed >> 24);
        out[bi + 4] = (uint8_t)(packed >> 32);
    }
    return i;
}

/* D=6: 8 codes -> 6 bytes.  uint64 (max shift 7*6=42). */
static inline int pack_d6_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int64_t shifts_lo [2] = {0,   6};
    static const int64_t shifts_mid[2] = {12, 18};
    static const int64_t shifts_hi [2] = {24, 30};
    static const int64_t shifts_top[2] = {36, 42};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vshlq_u16(vld1q_u16(codes_la + i),
                                  vdupq_n_s16((int16_t)-right_shift));
        v = vandq_u16(v, vdupq_n_u16(0x3F));
        uint32x4_t v32_lo = vmovl_u16(vget_low_u16(v));
        uint32x4_t v32_hi = vmovl_u16(vget_high_u16(v));
        uint64x2_t a = vshlq_u64(vmovl_u32(vget_low_u32 (v32_lo)),
                                  vld1q_s64(shifts_lo));
        uint64x2_t b = vshlq_u64(vmovl_u32(vget_high_u32(v32_lo)),
                                  vld1q_s64(shifts_mid));
        uint64x2_t c = vshlq_u64(vmovl_u32(vget_low_u32 (v32_hi)),
                                  vld1q_s64(shifts_hi));
        uint64x2_t d = vshlq_u64(vmovl_u32(vget_high_u32(v32_hi)),
                                  vld1q_s64(shifts_top));
        uint64x2_t sum = vaddq_u64(vaddq_u64(a, b), vaddq_u64(c, d));
        uint64_t packed = vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1);
        int bi = i * 6 / 8;
        out[bi    ] = (uint8_t)(packed       );
        out[bi + 1] = (uint8_t)(packed >>  8 );
        out[bi + 2] = (uint8_t)(packed >> 16);
        out[bi + 3] = (uint8_t)(packed >> 24);
        out[bi + 4] = (uint8_t)(packed >> 32);
        out[bi + 5] = (uint8_t)(packed >> 40);
    }
    return i;
}

/* D=7: 8 codes -> 7 bytes.  uint64 (max shift 7*7=49). */
static inline int pack_d7_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int64_t shifts_lo [2] = {0,   7};
    static const int64_t shifts_mid[2] = {14, 21};
    static const int64_t shifts_hi [2] = {28, 35};
    static const int64_t shifts_top[2] = {42, 49};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vshlq_u16(vld1q_u16(codes_la + i),
                                  vdupq_n_s16((int16_t)-right_shift));
        v = vandq_u16(v, vdupq_n_u16(0x7F));
        uint32x4_t v32_lo = vmovl_u16(vget_low_u16(v));
        uint32x4_t v32_hi = vmovl_u16(vget_high_u16(v));
        uint64x2_t a = vshlq_u64(vmovl_u32(vget_low_u32 (v32_lo)),
                                  vld1q_s64(shifts_lo));
        uint64x2_t b = vshlq_u64(vmovl_u32(vget_high_u32(v32_lo)),
                                  vld1q_s64(shifts_mid));
        uint64x2_t c = vshlq_u64(vmovl_u32(vget_low_u32 (v32_hi)),
                                  vld1q_s64(shifts_hi));
        uint64x2_t d = vshlq_u64(vmovl_u32(vget_high_u32(v32_hi)),
                                  vld1q_s64(shifts_top));
        uint64x2_t sum = vaddq_u64(vaddq_u64(a, b), vaddq_u64(c, d));
        uint64_t packed = vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1);
        int bi = i * 7 / 8;
        out[bi    ] = (uint8_t)(packed       );
        out[bi + 1] = (uint8_t)(packed >>  8 );
        out[bi + 2] = (uint8_t)(packed >> 16);
        out[bi + 3] = (uint8_t)(packed >> 24);
        out[bi + 4] = (uint8_t)(packed >> 32);
        out[bi + 5] = (uint8_t)(packed >> 40);
        out[bi + 6] = (uint8_t)(packed >> 48);
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

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ enc_init_neon(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int prim_build_bitmap_partition(uint16_t *codes_la,
                                                           int n, int depth,
                                                           uint8_t *bm,
                                                           uint16_t *tmp)
{ return build_bitmap_partition_neon(codes_la, n, depth, bm, tmp); }

PIVCO_PRIM_ALWAYS_INLINE void prim_pack_dN(uint8_t *out,
                                             const uint16_t *codes_la,
                                             int n, int D, int depth)
{ pack_dN_neon(out, codes_la, n, D, depth); }

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
