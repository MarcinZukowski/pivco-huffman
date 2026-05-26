/* pivco_huffman_avx512_flat.h — flat-subtree D-bit code unpackers (AVX-512 VBMI2).
 *
 * Internal header.  Mirrors src/pivco_huffman_neon_flat.h: each
 * `flat_dN_unpack_avx512()` (and the fast/safe pair for D ∈ {3,5,6})
 * reads N D-bit codes from a packed bitstream and returns them in a
 * 128-bit vector lane (one byte per code, value < 2^D).  Used by the
 * production decoder (pivco_huffman_avx512.c) and the per-D microbench
 * (bench/bench_micro.c).
 *
 * Tables are folded into the helpers as `_mm_setr_epi8` constants so
 * each helper is fully self-contained.  All helpers are `static
 * inline` — values fold into the inlined function and no extern
 * symbols are emitted.
 *
 * The "fast" variants for D=3, D=5, D=6 use power-of-2 byte loads
 * (8 / 16 bytes) which overread the valid bm region by a few bytes
 * but compile to a single load instruction.  The "safe" variants use
 * the exact byte count — caller picks the safe form for the final
 * chunk.  See the AVX-512 revisit ship note in IDEAS.md for context.
 *
 * Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_AVX512_FLAT_H
#define PIVCO_HUFFMAN_AVX512_FLAT_H

#if !defined(__AVX512BW__) || !defined(__AVX512VBMI__) || !defined(__AVX512VBMI2__)
#error "pivco_huffman_avx512_flat.h requires AVX-512 BW + VBMI + VBMI2"
#endif

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

/* D=2: 16 codes from 4 bytes of bm.  Replicate 4 bytes to 16 bytes, then
 * multishift with offsets {0,2,..,14, 16,18,..,30} across 2 uint64 lanes. */
static inline __m128i flat_d2_unpack_avx512(const uint8_t *bm_ptr)
{
    uint32_t packed;
    memcpy(&packed, bm_ptr, 4);
    __m128i data = _mm_set1_epi32((int32_t)packed);
    const __m128i ctrl = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        16, 18, 20, 22, 24, 26, 28, 30);
    __m128i raw = _mm_multishift_epi64_epi8(ctrl, data);
    return _mm_and_si128(raw, _mm_set1_epi8(0x03));
}

/* D=3 fast: loads 8 bytes (2 past the end of the 6-valid-byte region).
 * Caller must guarantee buffer slack. */
static inline __m128i flat_d3_unpack_avx512_fast(const uint8_t *bm_ptr)
{
    uint64_t packed;
    memcpy(&packed, bm_ptr, 8);
    __m128i data = _mm_set1_epi64x((int64_t)packed);
    const __m128i ctrl = _mm_setr_epi8(
        0, 3, 6, 9, 12, 15, 18, 21,
        24, 27, 30, 33, 36, 39, 42, 45);
    __m128i raw = _mm_multishift_epi64_epi8(ctrl, data);
    return _mm_and_si128(raw, _mm_set1_epi8(0x07));
}

/* D=3 safe: 6-byte memcpy for the last chunk. */
static inline __m128i flat_d3_unpack_avx512_safe(const uint8_t *bm_ptr)
{
    uint64_t packed = 0;
    memcpy(&packed, bm_ptr, 6);
    __m128i data = _mm_set1_epi64x((int64_t)packed);
    const __m128i ctrl = _mm_setr_epi8(
        0, 3, 6, 9, 12, 15, 18, 21,
        24, 27, 30, 33, 36, 39, 42, 45);
    __m128i raw = _mm_multishift_epi64_epi8(ctrl, data);
    return _mm_and_si128(raw, _mm_set1_epi8(0x07));
}

/* D=4: 16 codes from 8 bytes of bm.  2 codes per byte, no cross-byte
 * carries. */
static inline __m128i flat_d4_unpack_avx512(const uint8_t *bm_ptr)
{
    uint64_t packed;
    memcpy(&packed, bm_ptr, 8);
    __m128i data = _mm_set1_epi64x((int64_t)packed);
    const __m128i ctrl = _mm_setr_epi8(
        0, 4,  8, 12, 16, 20, 24, 28,
        32, 36, 40, 44, 48, 52, 56, 60);
    __m128i raw = _mm_multishift_epi64_epi8(ctrl, data);
    return _mm_and_si128(raw, _mm_set1_epi8(0x0F));
}

/* D=5 fast: 16 codes from 10 valid bytes, with a 16-byte load. */
static inline __m128i flat_d5_unpack_avx512_fast(const uint8_t *bm_ptr)
{
    __m128i raw = _mm_loadu_si128((const __m128i *)bm_ptr);
    const __m128i shuf = _mm_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,
        2, 3, 4, 5, 6, 7, 8, 9);
    __m128i data = _mm_shuffle_epi8(raw, shuf);
    const __m128i ctrl = _mm_setr_epi8(
        0,   5, 10, 15, 20, 25, 30, 35,
        24, 29, 34, 39, 44, 49, 54, 59);
    __m128i ms = _mm_multishift_epi64_epi8(ctrl, data);
    return _mm_and_si128(ms, _mm_set1_epi8(0x1F));
}

/* D=5 safe: 10-byte memcpy for the last chunk. */
static inline __m128i flat_d5_unpack_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[16] = {0};
    memcpy(buf, bm_ptr, 10);
    return flat_d5_unpack_avx512_fast(buf);
}

/* D=6 fast: 16 codes from 12 valid bytes, with a 16-byte load. */
static inline __m128i flat_d6_unpack_avx512_fast(const uint8_t *bm_ptr)
{
    __m128i raw = _mm_loadu_si128((const __m128i *)bm_ptr);
    const __m128i shuf = _mm_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,
        4, 5, 6, 7, 8, 9, 10, 11);
    __m128i data = _mm_shuffle_epi8(raw, shuf);
    const __m128i ctrl = _mm_setr_epi8(
        0,   6, 12, 18, 24, 30, 36, 42,
        16, 22, 28, 34, 40, 46, 52, 58);
    __m128i ms = _mm_multishift_epi64_epi8(ctrl, data);
    return _mm_and_si128(ms, _mm_set1_epi8(0x3F));
}

/* D=6 safe: 12-byte memcpy for the last chunk. */
static inline __m128i flat_d6_unpack_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[16] = {0};
    memcpy(buf, bm_ptr, 12);
    return flat_d6_unpack_avx512_fast(buf);
}

/* D=7: 16 codes = 112 bits = 14 bytes.  Code i is at bit 7i.  Two 64-bit
 * windows (input bytes 0..7 and 7..14) each hold 8 codes at offsets
 * {0,7,14,21,28,35,42,49}; vpmultishift extracts them.  Mask to 7 bits. */
static inline __m128i flat_d7_unpack_avx512_fast(const uint8_t *bm_ptr)
{
    __m128i raw = _mm_loadu_si128((const __m128i *)bm_ptr);
    const __m128i shuf = _mm_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,
        7, 8, 9, 10, 11, 12, 13, 14);
    __m128i data = _mm_shuffle_epi8(raw, shuf);
    const __m128i ctrl = _mm_setr_epi8(
        0, 7, 14, 21, 28, 35, 42, 49,
        0, 7, 14, 21, 28, 35, 42, 49);
    __m128i ms = _mm_multishift_epi64_epi8(ctrl, data);
    return _mm_and_si128(ms, _mm_set1_epi8(0x7F));
}

/* D=7 safe: 14-byte memcpy for the last chunk (avoid the 16-byte over-read). */
static inline __m128i flat_d7_unpack_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[16] = {0};
    memcpy(buf, bm_ptr, 14);
    return flat_d7_unpack_avx512_fast(buf);
}

#endif /* PIVCO_HUFFMAN_AVX512_FLAT_H */
