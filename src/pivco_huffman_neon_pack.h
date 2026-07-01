/* pivco_huffman_neon_pack.h — flat-subtree D-bit pack (NEON port of
 * ryg's multiply-as-shift pack), D=5/6/7 only.  D=2/3/4 stay on the
 * per-D special cases in pivco_huffman_primitives_neon.h (byte-aligned
 * paired adds for D=2/4 and u32-horadd for D=3 each beat this pipeline
 * by 2-3x on M4 + Graviton 4, since NEON-128 caps the throughput at
 * 16 codes/iter regardless of D).
 *
 * 16 codes per q-vector iter via byte-laid intermediate:
 *   - vmull_u8 + vpaddq_u16   word[i] = code[2i] + code[2i+1] * 2^D     (2D bits)
 *   - vmull_u16 + vpaddq_u32  dword[i] = word[2i] + word[2i+1] * 2^(2D) (4D bits)
 *   - vshrq_n_u64 + vbic/vand/vorr merge per-u64 dword pair             (8D bits)
 *   - vqtbl1q_u8 compact per-128-bit lane                                (2D bytes)
 *   - vst1q_u8                                                           (16 B, 2D valid)
 *
 * The trailing junk in each 16-byte store (16 - 2D bytes) gets
 * overwritten by the next iter's low bytes.  Caller's output buffer
 * needs at least 16 bytes of slack past the last valid byte of the
 * packed stream so the LAST iter's trailing junk lands somewhere safe;
 * PIVCO_MAX_ENCODED_SIZE = 2 * block_size gives plenty.
 *
 * Internal header.  Not part of the public API. */

#ifndef PIVCO_HUFFMAN_NEON_PACK_H
#define PIVCO_HUFFMAN_NEON_PACK_H

#ifndef __aarch64__
#error "pivco_huffman_neon_pack.h requires aarch64 NEON"
#endif

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

/* Load 16 ranks, subtract base (1 rank/byte) — the byte-laid intermediate the
 * multiply-as-shift pack expects, with no u16 narrow.  The local code is already
 * in [0,2^D) (rank - flat_base_rank over a depth-D flat subtree), so no mask to D
 * bits is needed. */
static inline uint8x16_t
pivco_pack_load_byte_neon(const uint8_t *ranks, uint8_t base)
{
    return vsubq_u8(vld1q_u8(ranks), vdupq_n_u8(base));
}

/* Per-D compact shuffles: bytes [0..D-1] from u64[0]'s low part go to
 * output lanes [0..D-1], bytes [0..D-1] from u64[1]'s low part go to
 * output lanes [D..2D-1].  Remaining lanes are zeroed via 0xff index
 * (vqtbl1q_u8 returns 0 for out-of-range index).
 *
 * D=2/3/4 use byte-aligned (D=2, D=4) or u32-horadd (D=3) special cases
 * in pivco_huffman_primitives_neon.h — those beat the generic pipeline
 * here by 2-3x.  The ryg pack only wins (modestly) for D=5/6/7. */
static const uint8_t pivco_pack_compact_d5_neon[16] = {
    0, 1, 2, 3, 4,   8, 9, 10, 11, 12,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivco_pack_compact_d6_neon[16] = {
    0, 1, 2, 3, 4, 5,   8, 9, 10, 11, 12, 13,
    0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivco_pack_compact_d7_neon[16] = {
    0, 1, 2, 3, 4, 5, 6,   8, 9, 10, 11, 12, 13, 14,
    0xff, 0xff
};

/* D as compile-time constant so vshrq_n_u64 / mask constants fold. */
#define PIVCO_PACK_NEON_DN(NAME, D_VAL, COMPACT_TAB)                            \
static inline int NAME(uint8_t *out, const uint8_t *ranks,                     \
                       int n, uint8_t base)                                     \
{                                                                                \
    const uint8x16_t c0 = vreinterpretq_u8_u16(                                  \
        vdupq_n_u16((uint16_t)(((1u << (D_VAL)) << 8) | 1u)));                   \
    const uint16x8_t c1 = vreinterpretq_u16_u32(                                 \
        vdupq_n_u32((uint32_t)(((1u << (2*(D_VAL))) << 16) | 1u)));              \
    const uint64x2_t c3 = vdupq_n_u64(((uint64_t)1 << (4*(D_VAL))) - 1);         \
    const uint8x16_t compact = vld1q_u8(COMPACT_TAB);                            \
    int i = 0;                                                                   \
    for (; i + 16 <= n; i += 16) {                                               \
        uint8x16_t cb = pivco_pack_load_byte_neon(ranks + i, base);              \
        /* Step 1: word[i] = cb[2i] + cb[2i+1] * 2^D  (8 u16 lanes)   */         \
        uint16x8_t prod_lo = vmull_u8(vget_low_u8(cb),  vget_low_u8(c0));        \
        uint16x8_t prod_hi = vmull_high_u8(cb, c0);                              \
        uint16x8_t w = vpaddq_u16(prod_lo, prod_hi);                             \
        /* Step 2: dword[i] = word[2i] + word[2i+1] * 2^(2D)  (4 u32 lanes) */   \
        uint32x4_t prod32_lo = vmull_u16(vget_low_u16(w),  vget_low_u16(c1));    \
        uint32x4_t prod32_hi = vmull_high_u16(w, c1);                            \
        uint32x4_t d  = vpaddq_u32(prod32_lo, prod32_hi);                        \
        /* Step 3: per-u64 lane, merge dword[2i+1] (right-shifted) with         \
         * dword[2i].  After srli by (32 - 4D): the high-32 dword sits at       \
         * bits [4D..4D+31].  Mask keeps low 4D bits of x, takes high 4D bits  \
         * from xs — together 8D bits per u64.                                  */ \
        uint64x2_t x  = vreinterpretq_u64_u32(d);                                \
        uint64x2_t xs = vshrq_n_u64(x, 32 - 4*(D_VAL));                          \
        uint64x2_t m  = vorrq_u64(vandq_u64(x, c3),                              \
                                   vbicq_u64(xs, c3));                           \
        /* Step 4: compact 2D consecutive bytes per 128-bit lane.   */           \
        uint8x16_t packed = vqtbl1q_u8(vreinterpretq_u8_u64(m), compact);        \
        vst1q_u8(out + ((i * (D_VAL)) >> 3), packed);                            \
    }                                                                            \
    return i;                                                                    \
}
PIVCO_PACK_NEON_DN(pack_d5_neon, 5, pivco_pack_compact_d5_neon)
PIVCO_PACK_NEON_DN(pack_d6_neon, 6, pivco_pack_compact_d6_neon)
PIVCO_PACK_NEON_DN(pack_d7_neon, 7, pivco_pack_compact_d7_neon)
#undef PIVCO_PACK_NEON_DN

#endif /* PIVCO_HUFFMAN_NEON_PACK_H */
