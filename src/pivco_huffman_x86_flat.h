/* pivco_huffman_x86_flat.h — flat-subtree D-bit code unpacker (SSE4.1).
 *
 * One unpack helper per D in {2,3,4,5,6}.  All use ryg's PSHUFB+PMULLO
 * "multiply-as-shift" trick: gather two adjacent bytes per uint16 lane,
 * multiply by a per-lane constant `1 << (16 - D - (pos & 7))` so the
 * field lands at the MSB of the lane, then PSRLI by (16 - D) and AND
 * the mask to LSB-align.  Works on SSE4.1 — no AVX2 vpsrlv, no VBMI2
 * vpmultishiftqb needed.
 *
 * D=4 has a SSE2 3-op specialisation (psrlw + punpcklbw + and) that
 * extracts all 16 codes in one shot from 8 bytes.
 *
 * Replaces the earlier vpsrlvd-based AVX2 paths: ryg's pattern was
 * measured uniformly faster on c3 (Ivy Bridge SSE), c4 (Haswell),
 * c5 (Cascade Lake), c5a (Zen 2, -43% to -53%), c6a (Zen 3, -25%).
 *
 * Internal header.  Used by the production decoder
 * (pivco_huffman_primitives_x86.h::merge_flat_x86_impl) and the
 * per-D microbench (bench/bench_micro.c).
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

/* D=4 SSE2 unpack: 16 codes from 8 bytes via 3 ops.
 *   raw     = 8-byte load (8 bytes, 16 nibbles)
 *   top_nib = srli_epi16(raw, 4)        — top nibble of each byte moves down
 *   merged  = unpacklo_epi8(raw, top_nib) — interleave: [b0_lo, b0_hi, b1_lo, b1_hi, ...]
 *   final   = and(merged, 0xF)
 * Returns __m128i with codes in all 16 lanes (low byte = code value). */
static inline __m128i flat_d4_unpack_x86(const uint8_t *bm_ptr)
{
    __m128i raw     = _mm_loadl_epi64((const __m128i *)bm_ptr);
    __m128i top_nib = _mm_srli_epi16(raw, 4);
    __m128i merged  = _mm_unpacklo_epi8(raw, top_nib);
    return _mm_and_si128(merged, _mm_set1_epi8(0xF));
}

/* D=2 SSE4.1 unpack: 8 codes from 2 bytes via ryg multiply-as-shift.
 * Reads up to 16 bytes (loadu); caller must ensure tail safety.
 * Returns __m128i with 8 codes in low 8 bytes. */
static inline __m128i flat_d2_unpack_x86(const uint8_t *bm_ptr)
{
    __m128i raw = _mm_loadu_si128((const __m128i *)bm_ptr);
    /* per-lane shuf: bytes (pos>>3, pos>>3 + 1) for pos = 0,2,4,...,14 */
    const __m128i shuf = _mm_setr_epi8(
        0,1, 0,1, 0,1, 0,1,
        1,2, 1,2, 1,2, 1,2);
    __m128i gathered = _mm_shuffle_epi8(raw, shuf);
    /* mult = 1 << (16 - D - (pos & 7)), pos&7 = 0,2,4,6 */
    const __m128i mult = _mm_setr_epi16(
        1<<14, 1<<12, 1<<10, 1<<8,
        1<<14, 1<<12, 1<<10, 1<<8);
    __m128i mh = _mm_mullo_epi16(gathered, mult);
    __m128i lsb = _mm_srli_epi16(mh, 14);              /* 16 - D = 14 */
    /* PSRLI by 14 already leaves only 2 bits, no AND needed.  Pack u16 -> u8. */
    return _mm_packus_epi16(lsb, _mm_setzero_si128());
}

/* D=3 SSE4.1 unpack: 8 codes from 3 bytes (+1 slop) via ryg multiply-as-shift.
 * Returns __m128i with 8 codes in low 8 bytes. */
static inline __m128i flat_d3_unpack_x86(const uint8_t *bm_ptr)
{
    __m128i raw = _mm_loadu_si128((const __m128i *)bm_ptr);
    /* pos = 0,3,6,9,12,15,18,21.  pos>>3 = 0,0,0,1,1,1,2,2 */
    const __m128i shuf = _mm_setr_epi8(
        0,1, 0,1, 0,1, 1,2, 1,2, 1,2, 2,3, 2,3);
    __m128i gathered = _mm_shuffle_epi8(raw, shuf);
    /* (pos & 7) = 0,3,6,1,4,7,2,5.  mult = 1 << (16 - 3 - (pos&7)) */
    const __m128i mult = _mm_setr_epi16(
        1<<13, 1<<10, 1<<7, 1<<12, 1<<9, 1<<6, 1<<11, 1<<8);
    __m128i mh = _mm_mullo_epi16(gathered, mult);
    __m128i lsb = _mm_srli_epi16(mh, 13);
    return _mm_packus_epi16(lsb, _mm_setzero_si128());
}

/* D=5 SSE4.1 unpack: 8 codes from 5 bytes via ryg multiply-as-shift.
 * 16-byte loadu over-reads up to 11 bytes — caller bounds the loop. */
static inline __m128i flat_d5_unpack_x86(const uint8_t *bm_ptr)
{
    __m128i raw = _mm_loadu_si128((const __m128i *)bm_ptr);
    /* pos = 0,5,10,15,20,25,30,35.  pos>>3 = 0,0,1,1,2,3,3,4 */
    const __m128i shuf = _mm_setr_epi8(
        0,1, 0,1, 1,2, 1,2, 2,3, 3,4, 3,4, 4,5);
    __m128i gathered = _mm_shuffle_epi8(raw, shuf);
    /* (pos & 7) = 0,5,2,7,4,1,6,3.  mult = 1 << (16 - 5 - (pos&7)) */
    const __m128i mult = _mm_setr_epi16(
        1<<11, 1<<6, 1<<9, 1<<4, 1<<7, 1<<10, 1<<5, 1<<8);
    __m128i mh = _mm_mullo_epi16(gathered, mult);
    __m128i lsb = _mm_srli_epi16(mh, 11);
    return _mm_packus_epi16(lsb, _mm_setzero_si128());
}

/* D=6 SSE4.1 unpack: 8 codes from 6 bytes via ryg multiply-as-shift.
 * 16-byte loadu over-reads up to 10 bytes — caller bounds the loop. */
static inline __m128i flat_d6_unpack_x86(const uint8_t *bm_ptr)
{
    __m128i raw = _mm_loadu_si128((const __m128i *)bm_ptr);
    /* pos = 0,6,12,18,24,30,36,42.  pos>>3 = 0,0,1,2,3,3,4,5 */
    const __m128i shuf = _mm_setr_epi8(
        0,1, 0,1, 1,2, 2,3, 3,4, 3,4, 4,5, 5,6);
    __m128i gathered = _mm_shuffle_epi8(raw, shuf);
    /* (pos & 7) = 0,6,4,2,0,6,4,2.  mult = 1 << (16 - 6 - (pos&7)) */
    const __m128i mult = _mm_setr_epi16(
        1<<10, 1<<4, 1<<6, 1<<8, 1<<10, 1<<4, 1<<6, 1<<8);
    __m128i mh = _mm_mullo_epi16(gathered, mult);
    __m128i lsb = _mm_srli_epi16(mh, 10);
    return _mm_packus_epi16(lsb, _mm_setzero_si128());
}

#if defined(PIVCO_HAS_AVX2)
#include <immintrin.h>
/* D=2 AVX2 unpack: 16 codes from 4 bytes.  Broadcast the 4-byte window to four
 * 32-bit lanes and vpsrlvd lane j by 2*j ({0,2,4,6}), so lane j byte b holds
 * code (4*b + j) in its low bits (D=2 packs exactly 4 codes per byte, which is
 * what makes this clean — no other D aligns this way).  A single pshufb
 * transposes that 4x4 byte matrix and the 0x3 mask clears the upper 6 bits,
 * leaving code i in byte i.  ~1.3-1.9x faster than two ryg flat_d2_unpack_x86
 * calls; from terrelln's PR #1.  Reads exactly 4 bytes (memcpy, no over-read). */
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
#endif /* PIVCO_HAS_AVX2 */

#endif /* PIVCO_HUFFMAN_X86_FLAT_H */
