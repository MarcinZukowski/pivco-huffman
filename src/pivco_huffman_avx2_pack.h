/* pivco_huffman_avx2_pack.h — flat-subtree D-bit pack (AVX2 port of
 * ryg's multiply-as-shift pack).  Mirrors pivco_huffman_avx512_pack.h
 * for the x86 backend hosts that have AVX2 but not AVX-512 VBMI2.
 *
 * 32 codes per ymm iter via byte-laid intermediate:
 *   - vpmaddubsw c0   word[i] = code[2i] + code[2i+1] * 2^D    (2D bits)
 *   - vpmaddwd   c1   dword[i] = word[2i] + word[2i+1] * 2^(2D) (4D bits)
 *   - vpsrlq    + (a&c)|(b&~c) via vpand/vpandn/vpor             (8D bits)
 *   - vpshufb compact per-128-bit lane                            (2D bytes)
 *   - 2 x vmovdqu storeu (low + high 128-bit halves)
 *
 * The trailing junk in each 128-bit store (16 - 2D bytes) gets
 * overwritten by the next iter's low store.  Caller's output buffer
 * needs at least 16 bytes of slack past the last valid byte of the
 * packed stream so the LAST iter's trailing junk lands somewhere safe;
 * PIVCO_MAX_ENCODED_SIZE = 2 * block_size gives plenty.
 *
 * Internal header.  Not part of the public API. */

#ifndef PIVCO_HUFFMAN_AVX2_PACK_H
#define PIVCO_HUFFMAN_AVX2_PACK_H

#if !defined(__AVX2__) || !defined(__SSE4_1__)
#error "pivco_huffman_avx2_pack.h requires AVX2 + SSE4.1"
#endif

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

/* Load 32 ranks, subtract base (1 rank/byte) — the byte-laid intermediate the
 * multiply-as-shift pack expects, with no u16 narrow.  The local code is already
 * in [0,2^D) (rank - flat_base_rank over a depth-D flat subtree), so no mask to D
 * bits is needed. */
static inline __m256i pivco_pack_load_byte_avx2(const uint8_t *ranks, uint8_t base)
{
    return _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)ranks),
                           _mm256_set1_epi8((char)base));
}

/* Per-D compact-shuf tables: pattern is bytes [0..D-1] from positions
 * 0..D-1, bytes [D..2D-1] from positions 8..8+D-1, junk at 2D..15.
 * Replicated identically in both 128-bit halves of the ymm shuf. */
#define PIVCO_PACK_AVX2_COMPACT_D2  _mm256_setr_epi8(                 \
    0, 1, 8, 9, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,                 \
    0, 1, 8, 9, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1)
#define PIVCO_PACK_AVX2_COMPACT_D3  _mm256_setr_epi8(                 \
    0, 1, 2, 8, 9,10, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1,                 \
    0, 1, 2, 8, 9,10, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1)
#define PIVCO_PACK_AVX2_COMPACT_D5  _mm256_setr_epi8(                 \
    0, 1, 2, 3, 4, 8, 9,10, 11,12, -1,-1, -1,-1,-1,-1,                 \
    0, 1, 2, 3, 4, 8, 9,10, 11,12, -1,-1, -1,-1,-1,-1)
#define PIVCO_PACK_AVX2_COMPACT_D6  _mm256_setr_epi8(                 \
    0, 1, 2, 3, 4, 5, 8, 9, 10,11,12,13, -1,-1,-1,-1,                  \
    0, 1, 2, 3, 4, 5, 8, 9, 10,11,12,13, -1,-1,-1,-1)
#define PIVCO_PACK_AVX2_COMPACT_D7  _mm256_setr_epi8(                 \
    0, 1, 2, 3, 4, 5, 6, 8,  9,10,11,12,13,14, -1,-1,                  \
    0, 1, 2, 3, 4, 5, 6, 8,  9,10,11,12,13,14, -1,-1)

#define PIVCO_PACK_AVX2_DN(NAME, D_VAL, COMPACT_SHUF)                           \
static inline int NAME(uint8_t *out, const uint8_t *ranks,                      \
                       int n, uint8_t base)                                      \
{                                                                                \
    const __m256i c0 = _mm256_set1_epi16(                                        \
        (int16_t)(((1 << (D_VAL)) << 8) | 1));                                   \
    const __m256i c1 = _mm256_set1_epi32(                                        \
        (int32_t)(((int32_t)1 << (2*(D_VAL))) << 16) | 1);                       \
    const __m256i c3 = _mm256_set1_epi64x(                                       \
        (int64_t)(((int64_t)1 << (4*(D_VAL))) - 1));                             \
    const __m256i compact = COMPACT_SHUF;                                        \
    int i = 0;                                                                   \
    for (; i + 32 <= n; i += 32) {                                               \
        __m256i cb = pivco_pack_load_byte_avx2(ranks + i, base);                 \
        __m256i x  = _mm256_maddubs_epi16(c0, cb);                               \
        x = _mm256_madd_epi16(x, c1);                                            \
        __m256i xs = _mm256_srli_epi64(x, 32 - 4*(D_VAL));                       \
        x = _mm256_or_si256(_mm256_and_si256(x, c3),                             \
                             _mm256_andnot_si256(c3, xs));                       \
        __m256i out_y = _mm256_shuffle_epi8(x, compact);                         \
        int outpos = (i * (D_VAL)) >> 3;                                         \
        _mm_storeu_si128((__m128i *)(out + outpos),                              \
                          _mm256_castsi256_si128(out_y));                        \
        _mm_storeu_si128((__m128i *)(out + outpos + 2*(D_VAL)),                  \
                          _mm256_extracti128_si256(out_y, 1));                   \
    }                                                                            \
    return i;                                                                    \
}
PIVCO_PACK_AVX2_DN(pack_d2_avx2_x86, 2, PIVCO_PACK_AVX2_COMPACT_D2)
PIVCO_PACK_AVX2_DN(pack_d3_avx2_x86, 3, PIVCO_PACK_AVX2_COMPACT_D3)
PIVCO_PACK_AVX2_DN(pack_d5_avx2_x86, 5, PIVCO_PACK_AVX2_COMPACT_D5)
PIVCO_PACK_AVX2_DN(pack_d6_avx2_x86, 6, PIVCO_PACK_AVX2_COMPACT_D6)
PIVCO_PACK_AVX2_DN(pack_d7_avx2_x86, 7, PIVCO_PACK_AVX2_COMPACT_D7)
#undef PIVCO_PACK_AVX2_DN

#endif  /* PIVCO_HUFFMAN_AVX2_PACK_H */
