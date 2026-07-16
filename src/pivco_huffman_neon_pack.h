/* pivco_huffman_neon_pack.h — flat-subtree D-bit pack (NEON), D=5/6/7.
 * D=2/3/4 stay on their per-D special cases in
 * pivco_huffman_primitives_neon.h (byte-aligned paired adds for D=2/4,
 * a converging-shift pack for D=3).
 *
 * Variable-shift "converging pyramid" pack, 16 codes per q-vector iter.
 * Each pairing level shifts the two halves of a lane pair towards each
 * other with one USHL of {+s,-s} per-lane counts, so the fields meet at
 * the lane boundary and each level is a single instruction:
 *   L1 u8  {8-D, 0}:            u16 = pair  << (8-D)
 *   L2 u16 {8-D, -(8-D)}:       u32 = quad  << (16-2D)
 *   L3 u32 {16-2D, -(16-2D)}:   u64 = octet << (32-4D)
 * The compact shuffle absorbs the whole bytes of the final (32-4D)
 * re-basing shift (its tables start at byte 1 for D=5/6), leaving a
 * residual >>4 for D=5/7 and no final shift for D=6.
 *
 * Internal header.  Not part of the public API. */

#ifndef PIVCO_HUFFMAN_NEON_PACK_H
#define PIVCO_HUFFMAN_NEON_PACK_H

#ifndef __aarch64__
#error "pivco_huffman_neon_pack.h requires aarch64 NEON"
#endif

#include <arm_neon.h>
#include <stdint.h>

/* Per-D compact shuffles: gather the 2D valid packed bytes (D from each
 * 128-bit lane's low half, after the final re-basing shift is folded in)
 * into output lanes [0..2D); 0xff indices zero the trailing lanes. */
static const uint8_t pivco_pack_compact_d5_neon[16] = {
    1, 2, 3, 4, 5,   9, 10, 11, 12, 13,   0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivco_pack_compact_d6_neon[16] = {
    1, 2, 3, 4, 5, 6,   9, 10, 11, 12, 13, 14,   0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivco_pack_compact_d7_neon[16] = {
    0, 1, 2, 3, 4, 5, 6,   8, 9, 10, 11, 12, 13, 14,   0xff, 0xff
};

/* D and BITSHR as compile-time constants so the USHL count tables and
 * the vshrq_n_u64 fold.  Returns the number of codes packed by the SIMD
 * loop; the caller's scalar tail packs [i, n). */
#define PIVCO_PACK_NEON_DN(NAME, D_VAL, BITSHR, COMPACT_TAB)                    \
static inline int NAME(uint8_t *out, const uint8_t *ranks,                     \
                       int n, uint8_t base)                                    \
{                                                                              \
    static const int8_t  sh1[16] = { 8-(D_VAL),0, 8-(D_VAL),0, 8-(D_VAL),0,    \
                                     8-(D_VAL),0, 8-(D_VAL),0, 8-(D_VAL),0,    \
                                     8-(D_VAL),0, 8-(D_VAL),0 };               \
    static const int16_t sh2[8]  = { 8-(D_VAL), -(8-(D_VAL)),                  \
                                     8-(D_VAL), -(8-(D_VAL)),                  \
                                     8-(D_VAL), -(8-(D_VAL)),                  \
                                     8-(D_VAL), -(8-(D_VAL)) };                \
    static const int32_t sh3[4]  = { 16-2*(D_VAL), -(16-2*(D_VAL)),            \
                                     16-2*(D_VAL), -(16-2*(D_VAL)) };          \
    const int8x16_t  s1 = vld1q_s8(sh1);                                       \
    const int16x8_t  s2 = vld1q_s16(sh2);                                      \
    const int32x4_t  s3 = vld1q_s32(sh3);                                      \
    const uint8x16_t compact = vld1q_u8(COMPACT_TAB);                          \
    const int total_bytes = (n * (D_VAL) + 7) >> 3;                            \
    int i = 0;                                                                 \
    for (; i + 16 <= n && ((i * (D_VAL)) >> 3) + 16 <= total_bytes; i += 16) { \
        uint8x16_t cb  = vsubq_u8(vld1q_u8(ranks + i), vdupq_n_u8(base));      \
        uint16x8_t w16 = vreinterpretq_u16_u8(vshlq_u8(cb, s1));               \
        uint32x4_t w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));            \
        uint64x2_t w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));            \
        if (BITSHR) w64 = vshrq_n_u64(w64, (BITSHR) ? (BITSHR) : 1);           \
        vst1q_u8(out + ((i * (D_VAL)) >> 3),                                   \
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));              \
    }                                                                          \
    return i;                                                                  \
}
PIVCO_PACK_NEON_DN(pack_d5_neon, 5, 4, pivco_pack_compact_d5_neon)
PIVCO_PACK_NEON_DN(pack_d6_neon, 6, 0, pivco_pack_compact_d6_neon)
PIVCO_PACK_NEON_DN(pack_d7_neon, 7, 4, pivco_pack_compact_d7_neon)
#undef PIVCO_PACK_NEON_DN

#endif /* PIVCO_HUFFMAN_NEON_PACK_H */
