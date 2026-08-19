/* pivco_huffman_vertical.h — hybrid vertical flat-region bit-packing:
 * shared layout definition + scalar reference kernels.
 *
 * WIRE FORMAT (the PIVCO_FLAT_VERTICAL layout, the default; replaces
 * the natural row-major layout inside flat regions only — region byte
 * size is unchanged, so no other wire structure moves):
 *
 *   A flat region of n D-bit codes with D in {2..7} stores its first
 *   n512 = pivco_vert_n512(n) codes in vertical blocks of 512 (64 lanes),
 *   the next n128 = pivco_vert_n(n - n512) codes in vertical blocks of
 *   128 (16 lanes), and the remainder in the legacy natural layout.
 *   D = 8 and n < 128 stay fully natural (D = 8 is the identity memcpy).
 *
 *   The 512 block is 4 interleaved 16-lane sub-blocks: narrow (128-bit)
 *   engines process quarter q with the 16-lane kernels at column stride
 *   64 and output stride 64; 512-bit engines get one uniform-shift row
 *   per step.
 *
 *   Block layout (16*D bytes, byte-column major):
 *     code v of a block (v = 0..127) lives in lane L = v & 15 at
 *     step s = v >> 4.  Lane L's sub-stream is D bytes, little-endian,
 *     holding its 8 codes at bit offsets {0, D, ..., 7D}.  Byte j of
 *     lane L is stored at block_base + 16*j + L — i.e. byte-column j
 *     is 16 contiguous bytes, one per lane.
 *
 *   Decode property: extracting bit-field s from every lane of a block
 *   yields codes 16s .. 16s+15 — sixteen consecutive outputs — from
 *   byte-column loads + uniform shifts, for ANY D (no per-code byte
 *   misalignment, the FastLanes idea at 128-value granularity).
 *
 * The layout itself is selected at table build (pivco_cfg_t.flat_layout,
 * baked into table->flat_layout; the pivcohuf container records it in
 * its FLAGS byte): PIVCO_FLAT_VERTICAL is the hybrid above, and
 * PIVCO_FLAT_VERTICAL_128 skips the 512 span (128-value blocks +
 * natural tail only — the fastest decode+encode on ARM servers).
 * Within a layout, both sides gate with the pure functions below, so
 * nothing per-region goes on the wire.
 *
 * SIMD implementations live in the backend primitive headers (NEON) and
 * pivco_huffman_x86_vertical.h (SSE/AVX2/AVX-512); the scalar kernels
 * below are the reference + scalar-backend forms.
 */
#ifndef PIVCO_HUFFMAN_VERTICAL_H
#define PIVCO_HUFFMAN_VERTICAL_H

#include <stdint.h>
#include <string.h>

/* Hybrid gates.  A flat region with D in {2..7} stores, in order:
 *   [n512 codes in 512-value/64-lane blocks]
 *   [n128 codes in 128-value/16-lane blocks]   (n128 = gate on the remainder)
 *   [natural tail]
 * Both sides evaluate the same pure functions, so no wire flag is needed. */
static inline int pivco_vert_n512(int n, int D)
{
    return (D >= 2 && D <= 7 && n >= 512) ? (n & ~511) : 0;
}

/* Length of the 128-value vertical span (applied to the post-512 remainder;
 * 0 = fully natural). */
static inline int pivco_vert_n(int n, int D)
{
    return (D >= 2 && D <= 7 && n >= 128) ? (n & ~127) : 0;
}

/* Scalar reference: pack n_v ranks vertically at lane count LN (n_v a
 * multiple of 8*LN). */
static inline void vert_pack_scalar_w(uint8_t *out, const uint8_t *ranks,
                                      int n_v, int D, uint8_t base, int LN)
{
    uint32_t mask = (1u << D) - 1u;
    int BV = LN * 8;
    for (int b = 0; b < n_v / BV; b++) {
        uint8_t *blk = out + (size_t)b * LN * D;
        memset(blk, 0, (size_t)LN * D);
        for (int v = 0; v < BV; v++) {
            uint32_t val = (uint32_t)(uint8_t)(ranks[b * BV + v] - base) & mask;
            int L = v % LN, bit = (v / LN) * D, j = bit >> 3, off = bit & 7;
            blk[LN * j + L] |= (uint8_t)(val << off);
            if (off + D > 8)
                blk[LN * (j + 1) + L] |= (uint8_t)(val >> (8 - off));
        }
    }
}

/* Scalar reference: fused decode of a vertical span at lane count LN. */
static inline void vert_merge_scalar_w(uint8_t *out, int n_v, const uint8_t *bm,
                                       int D, const uint8_t *c2s, int LN)
{
    uint32_t mask = (1u << D) - 1u;
    int BV = LN * 8;
    for (int b = 0; b < n_v / BV; b++) {
        const uint8_t *blk = bm + (size_t)b * LN * D;
        for (int v = 0; v < BV; v++) {
            int L = v % LN, bit = (v / LN) * D, j = bit >> 3, off = bit & 7;
            uint32_t w = blk[LN * j + L];
            if (off + D > 8)
                w |= (uint32_t)blk[LN * (j + 1) + L] << 8;
            out[b * BV + v] = c2s[(w >> off) & mask];
        }
    }
}

static inline void vert_pack_scalar(uint8_t *out, const uint8_t *ranks,
                                    int n_v, int D, uint8_t base)
{ vert_pack_scalar_w(out, ranks, n_v, D, base, 16); }
static inline void vert_merge_scalar(uint8_t *out, int n_v, const uint8_t *bm,
                                     int D, const uint8_t *c2s)
{ vert_merge_scalar_w(out, n_v, bm, D, c2s, 16); }
static inline void vert512_pack_scalar(uint8_t *out, const uint8_t *ranks,
                                       int n_v, int D, uint8_t base)
{ vert_pack_scalar_w(out, ranks, n_v, D, base, 64); }
static inline void vert512_merge_scalar(uint8_t *out, int n_v, const uint8_t *bm,
                                        int D, const uint8_t *c2s)
{ vert_merge_scalar_w(out, n_v, bm, D, c2s, 64); }

#endif /* PIVCO_HUFFMAN_VERTICAL_H */
