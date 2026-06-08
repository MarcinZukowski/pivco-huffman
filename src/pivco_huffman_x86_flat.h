/* pivco_huffman_x86_flat.h — flat-subtree D-bit code unpacker (SSE4.1).
 *
 * Internal header.  Only D=4 has a SIMD unpack under pure SSE4.1: there
 * is no per-byte variable shift (no `_mm_srlv_epi8` / no `vpsrlvw` until
 * AVX2) and no `vpmultishiftqb` (VBMI2), so D=2/3/5/6 require ~4-8
 * separate pshufb + immediate shifts + blends — slower than the
 * scalar `FLAT_UNPACK_SWITCH_IDX` fallback.  D=4 is the special case
 * (duplicate + mask + single-immediate-shift + blend).
 *
 * Used by the production decoder (pivco_huffman_x86.c) and the
 * per-D microbench (bench/bench_micro.c) so both share a single
 * source of truth.
 *
 * Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_X86_FLAT_H
#define PIVCO_HUFFMAN_X86_FLAT_H

#if !defined(__SSE4_1__)
#error "pivco_huffman_x86_flat.h requires SSE4.1"
#endif

#include <stdint.h>
#include <string.h>
#include <smmintrin.h>
#if defined(PIVCO_HAS_AVX2)
#include <immintrin.h>

/* D=2 AVX2 unpack: 16 codes from 4 bytes.  Broadcast the window to four 32-bit
 * lanes and vpsrlvd lane j by 2*j ({0,2,4,6}), so lane j byte b holds code
 * (4*b + j) in its low bits.  pshufb transposes that 4x4 byte matrix and the
 * 0x3 mask clears the upper 6 bits, leaving code i in byte i. */
static inline __m128i flat_d2_unpack_avx2(const uint8_t *bm_ptr)
{
    const __m128i s = _mm_setr_epi32(0, 2, 4, 6);
    const __m128i m = _mm_set1_epi8(0x3);
    const __m128i shuf = _mm_setr_epi8(
         0,  4,  8, 12,
         1,  5,  9, 13,
         2,  6, 10, 14,
         3,  7, 11, 15);

    uint32_t packed; memcpy(&packed, bm_ptr, 4);
    __m128i v = _mm_srlv_epi32(_mm_set1_epi32((int)packed), s);

    return _mm_and_si128(_mm_shuffle_epi8(v, shuf), m);
}

/* D=5 AVX2 unpack: 8 codes from 5 bytes.  A D=5 code at sub-offset s ≤ 7
 * spans at most 2 bytes (s+5 ≤ 12 < 16), so each lane is filled with a
 * 2-byte window starting at the code's start-byte (gathered via pshufb from
 * a broadcast 16-byte load), then vpsrlvd by the sub-offset isolates it. */
static inline __m128i flat_d5_unpack_avx2(const uint8_t *bm_ptr)
{
    __m128i raw128 = _mm_loadu_si128((const __m128i *)bm_ptr);
    __m256i src = _mm256_broadcastsi128_si256(raw128);
    /* lane j (32-bit) gets bytes {start_byte_j, start_byte_j+1}; rest zeroed. */
    const __m256i byteidx = _mm256_setr_epi8(
        0,1,-1,-1, 0,1,-1,-1, 1,2,-1,-1, 1,2,-1,-1,   /* lanes 0-3 */
        2,3,-1,-1, 3,4,-1,-1, 3,4,-1,-1, 4,5,-1,-1);  /* lanes 4-7 */
    __m256i bytes = _mm256_shuffle_epi8(src, byteidx);
    const __m256i sub = _mm256_setr_epi32(0, 5, 2, 7, 4, 1, 6, 3);
    __m256i v = _mm256_and_si256(_mm256_srlv_epi32(bytes, sub),
                                 _mm256_set1_epi32(0x1F));
    const __m256i bshuf = _mm256_setr_epi8(
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    __m256i p = _mm256_shuffle_epi8(v, bshuf);
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(p),
                              _mm256_extracti128_si256(p, 1));
}

/* D=6 AVX2 unpack: 8 codes from 6 bytes.  Same 2-byte-window scheme as D=5. */
static inline __m128i flat_d6_unpack_avx2(const uint8_t *bm_ptr)
{
    __m128i raw128 = _mm_loadu_si128((const __m128i *)bm_ptr);
    __m256i src = _mm256_broadcastsi128_si256(raw128);
    const __m256i byteidx = _mm256_setr_epi8(
        0,1,-1,-1, 0,1,-1,-1, 1,2,-1,-1, 2,3,-1,-1,   /* lanes 0-3 */
        3,4,-1,-1, 3,4,-1,-1, 4,5,-1,-1, 5,6,-1,-1);  /* lanes 4-7 */
    __m256i bytes = _mm256_shuffle_epi8(src, byteidx);
    const __m256i sub = _mm256_setr_epi32(0, 6, 4, 2, 0, 6, 4, 2);
    __m256i v = _mm256_and_si256(_mm256_srlv_epi32(bytes, sub),
                                 _mm256_set1_epi32(0x3F));
    const __m256i bshuf = _mm256_setr_epi8(
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    __m256i p = _mm256_shuffle_epi8(v, bshuf);
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(p),
                              _mm256_extracti128_si256(p, 1));
}

/* D=3 AVX2 unpack: 8 codes from 3 bytes.  Code i is at bit 3i of the packed
 * LSB-first stream, so broadcasting the 4-byte window and doing a per-lane
 * variable right-shift by {0,3,..,21} (vpsrlvd) drops each code into its
 * lane's low bits -- no shuffle-into-windows needed.  Returns the 8 codes in
 * the low 8 bytes of the result. */
static inline __m128i flat_d3_unpack_avx2(const uint8_t *bm_ptr)
{
    uint32_t packed; memcpy(&packed, bm_ptr, 4);   /* 3 used + 1 slop byte */
    __m256i v = _mm256_set1_epi32((int)packed);
    const __m256i sh = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    v = _mm256_and_si256(_mm256_srlv_epi32(v, sh), _mm256_set1_epi32(0x7));
    /* low byte of each 32-bit lane -> contiguous; lanes 0-3 then 4-7 */
    const __m256i bshuf = _mm256_setr_epi8(
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    __m256i s = _mm256_shuffle_epi8(v, bshuf);
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(s),
                              _mm256_extracti128_si256(s, 1));
}
#endif /* PIVCO_HAS_AVX2 */

/* D=4 SSE4.1 unpack: 16 codes from 8 bytes of bm.
 * 8 bytes loaded, duplicated to 16 bytes via pshufb: [b0,b0,b1,b1,..].
 * Treat each pair as uint16: mask & 0xF00F → (b&0x0F, b&0xF0); shift right 4
 * → (0, b>>4); blend with 0xFF00 mask picks (b&0x0F, b>>4) per pair. */
static inline __m128i flat_d4_unpack_x86(const uint8_t *bm_ptr)
{
    __m128i raw = _mm_loadl_epi64((const __m128i *)bm_ptr);
    const __m128i dup_idx = _mm_setr_epi8(
        0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7);
    __m128i dup = _mm_shuffle_epi8(raw, dup_idx);
    __m128i masked = _mm_and_si128(dup, _mm_set1_epi16(0xF00F));
    __m128i shifted = _mm_srli_epi16(masked, 4);
    __m128i blend_mask = _mm_set1_epi16((int16_t)0xFF00);
    return _mm_blendv_epi8(masked, shifted, blend_mask);
}

#endif /* PIVCO_HUFFMAN_X86_FLAT_H */
