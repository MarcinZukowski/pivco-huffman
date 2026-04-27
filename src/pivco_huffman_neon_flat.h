/* pivco_huffman_neon_flat.h — flat-subtree D-bit code unpackers (NEON).
 *
 * Internal header.  Each `flat_dN_unpack()` reads N D-bit codes from a
 * packed bitstream and returns them in NEON vector lanes (one byte per
 * code, value < 2^D).  Used by the production decoder
 * (pivco_huffman_neon.c) and the per-D microbench (bench/bench_micro.c).
 *
 * All helpers + unpack tables live here so the two TUs share a single
 * source of truth without giving up inlining.  Tables are `static const`
 * (per-TU) and helpers are `static inline` — values fold into the
 * inlined function and no extern symbols are emitted.
 *
 * Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_NEON_FLAT_H
#define PIVCO_HUFFMAN_NEON_FLAT_H

#ifndef __aarch64__
#error "pivco_huffman_neon_flat.h requires aarch64 NEON"
#endif

#include <stdint.h>
#include <string.h>
#include <arm_neon.h>

/* D=2 unpack constants: each byte of input holds 4 codes; replicate each
 * input byte to 4 output lanes, then right-shift lane k by 2k to align
 * the desired 2-bit code at the low bits. */
static const uint8_t flat_d2_dup_tab[16] = {
    0,0,0,0,  1,1,1,1,  2,2,2,2,  3,3,3,3
};
static const int8_t flat_d2_shift_tab[16] = {
    0,-2,-4,-6,  0,-2,-4,-6,  0,-2,-4,-6,  0,-2,-4,-6
};

/* Unpack 16 consecutive D=2 codes from 4 bytes of bm into a 16-lane byte
 * vector (values 0..3). */
static inline uint8x16_t flat_d2_unpack(const uint8_t *bm_ptr)
{
    uint32_t packed;
    memcpy(&packed, bm_ptr, 4);
    uint8x16_t bm_lo = vreinterpretq_u8_u32(
        vsetq_lane_u32(packed, vdupq_n_u32(0), 0));
    uint8x16_t dup = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d2_dup_tab));
    uint8x16_t shifted = vshlq_u8(dup, vld1q_s8(flat_d2_shift_tab));
    return vandq_u8(shifted, vdupq_n_u8(0x03));
}

/* D=3 unpack: 3 bytes = 24 bits = 8 codes.  Two of the 8 codes cross a
 * byte boundary, so we work in uint16 lanes (each holding a 16-bit
 * window with enough bits to shift out any one 3-bit code). */
static const uint8_t flat_d3_shuf_tab[16] = {
    /* 5 lanes of (b0, b1): for codes 0..4 (shifts 0,3,6,9,12) */
    0, 1,  0, 1,  0, 1,  0, 1,  0, 1,
    /* 3 lanes of (b1, b2): for codes 5..7 (shifts 7,10,13) */
    1, 2,  1, 2,  1, 2
};
static const int16_t flat_d3_shift_tab[8] = {
    0, -3, -6, -9, -12, -7, -10, -13
};

/* Unpack 8 consecutive D=3 codes from 3 bytes starting at bm_ptr into
 * the low 8 lanes of a uint8x8 vector (values 0..7). */
static inline uint8x8_t flat_d3_unpack(const uint8_t *bm_ptr)
{
    /* Load 3 bytes byte-by-byte into a vector with top bytes zero.
     * Avoid a 4-byte read so we don't run past the end of the stream. */
    uint8x16_t bm_lo = vdupq_n_u8(0);
    bm_lo = vsetq_lane_u8(bm_ptr[0], bm_lo, 0);
    bm_lo = vsetq_lane_u8(bm_ptr[1], bm_lo, 1);
    bm_lo = vsetq_lane_u8(bm_ptr[2], bm_lo, 2);
    uint8x16_t shuffled = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d3_shuf_tab));
    uint16x8_t w = vreinterpretq_u16_u8(shuffled);
    uint16x8_t shifted = vshlq_u16(w, vld1q_s16(flat_d3_shift_tab));
    uint16x8_t masked = vandq_u16(shifted, vdupq_n_u16(0x07));
    return vmovn_u16(masked);
}

/* D=4 unpack: 8 bytes hold 16 codes (2 per byte, no byte-crossings).
 * Replicate each byte to 2 lanes and shift lane k by (k & 1) * 4. */
static const uint8_t flat_d4_dup_tab[16] = {
    0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7
};
static const int8_t flat_d4_shift_tab[16] = {
    0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4
};

/* Unpack 16 consecutive D=4 codes from 8 bytes of bm. */
static inline uint8x16_t flat_d4_unpack(const uint8_t *bm_ptr)
{
    uint64_t packed;
    memcpy(&packed, bm_ptr, 8);
    uint8x16_t bm_lo = vreinterpretq_u8_u64(
        vsetq_lane_u64(packed, vdupq_n_u64(0), 0));
    uint8x16_t dup = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d4_dup_tab));
    uint8x16_t shifted = vshlq_u8(dup, vld1q_s8(flat_d4_shift_tab));
    return vandq_u8(shifted, vdupq_n_u8(0x0F));
}

/* D=5 unpack: 5 bytes = 40 bits = 8 codes.  5 of 8 codes cross byte
 * boundaries, so we work in uint16 lanes.  Lane layout:
 *   lanes 0,1,2: (b0, b1)  — codes 0,1,2 (shifts  0, 5, 10)
 *   lane    3: (b1, b2)  — code  3        (shift    7)
 *   lanes 4,5:   (b2, b3)  — codes 4,5    (shifts  4, 9)
 *   lanes 6,7:   (b3, b4)  — codes 6,7    (shifts  6, 11)                 */
static const uint8_t flat_d5_shuf_tab[16] = {
    0,1,  0,1,  0,1,  1,2,  2,3,  2,3,  3,4,  3,4
};
static const int16_t flat_d5_shift_tab[8] = {
    0, -5, -10, -7, -4, -9, -6, -11
};

/* Unpack 8 consecutive D=5 codes from 5 bytes starting at bm_ptr. */
static inline uint8x8_t flat_d5_unpack(const uint8_t *bm_ptr)
{
    /* 5-byte load via memcpy into the low 40 bits of a uint64 — doesn't
     * overrun the stream. */
    uint64_t packed = 0;
    memcpy(&packed, bm_ptr, 5);
    uint8x16_t bm_lo = vreinterpretq_u8_u64(
        vsetq_lane_u64(packed, vdupq_n_u64(0), 0));
    uint8x16_t shuffled = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d5_shuf_tab));
    uint16x8_t w = vreinterpretq_u16_u8(shuffled);
    uint16x8_t shifted = vshlq_u16(w, vld1q_s16(flat_d5_shift_tab));
    uint16x8_t masked = vandq_u16(shifted, vdupq_n_u16(0x1F));
    return vmovn_u16(masked);
}

/* D=6 unpack: 3 bytes = 24 bits = 4 codes.  2 of 4 codes cross byte
 * boundaries (codes 1 and 2).  To produce 8 codes we process 6 bytes.
 * Lane layout:
 *   lanes 0,1: (b0, b1)  — codes 0, 1 (shifts  0, 6)
 *   lanes 2,3: (b1, b2)  — codes 2, 3 (shifts  4, 10)
 *   lanes 4,5: (b3, b4)  — codes 4, 5 (shifts  0, 6)
 *   lanes 6,7: (b4, b5)  — codes 6, 7 (shifts  4, 10)                    */
static const uint8_t flat_d6_shuf_tab[16] = {
    0,1,  0,1,  1,2,  1,2,    3,4,  3,4,  4,5,  4,5
};
static const int16_t flat_d6_shift_tab[8] = {
    0, -6, -4, -10,  0, -6, -4, -10
};

/* Unpack 8 consecutive D=6 codes from 6 bytes starting at bm_ptr. */
static inline uint8x8_t flat_d6_unpack(const uint8_t *bm_ptr)
{
    uint64_t packed = 0;
    memcpy(&packed, bm_ptr, 6);
    uint8x16_t bm_lo = vreinterpretq_u8_u64(
        vsetq_lane_u64(packed, vdupq_n_u64(0), 0));
    uint8x16_t shuffled = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d6_shuf_tab));
    uint16x8_t w = vreinterpretq_u16_u8(shuffled);
    uint16x8_t shifted = vshlq_u16(w, vld1q_s16(flat_d6_shift_tab));
    uint16x8_t masked = vandq_u16(shifted, vdupq_n_u16(0x3F));
    return vmovn_u16(masked);
}

#endif /* PIVCO_HUFFMAN_NEON_FLAT_H */
