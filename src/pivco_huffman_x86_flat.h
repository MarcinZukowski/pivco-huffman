/* pivco_huffman_x86_flat.h — flat-subtree D-bit code spreader (SSE4.1).
 *
 * Internal header.  Only D=4 has a SIMD spread under pure SSE4.1: there
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
#include <smmintrin.h>

/* D=4 SSE4.1 spread: 16 codes from 8 bytes of bm.
 * 8 bytes loaded, duplicated to 16 bytes via pshufb: [b0,b0,b1,b1,..].
 * Treat each pair as uint16: mask & 0xF00F → (b&0x0F, b&0xF0); shift right 4
 * → (0, b>>4); blend with 0xFF00 mask picks (b&0x0F, b>>4) per pair. */
static inline __m128i flat_d4_spread_x86(const uint8_t *bm_ptr)
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
