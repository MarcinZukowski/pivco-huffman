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

/* ---- 64-at-a-time unpacks (codes/call = 64, one zmm output) ----------
 *
 * All variants follow the same shape: load enough bytes for lane 7 to be
 * fully sourced (`max_byte = D*7 + 7`), `vpermb` to gather the 8 per-lane
 * windows, `vpmultishiftqb` with per-lane ctrl {0, D, 2D, …, 7D}, then
 * AND with the D-bit mask.  The load width is 32 B (ymm zext) when
 * `max_byte ≤ 31`, 64 B (zmm) otherwise.  Strict end-of-stream slack is
 * `ceil(load_bytes * 8 / D)` codes — the caller leaves that much room at
 * the end and runs the existing 16-wide tail there. */

/* D=2 x4: 64 codes from 16 valid bm bytes via a 32-byte ymm load (lane 7
 * needs byte 21). */
static inline __m512i flat_d2_unpack64_avx512_fast(const uint8_t *bm_ptr)
{
    __m256i raw256 = _mm256_loadu_si256((const __m256i *)bm_ptr);
    __m512i raw    = _mm512_zextsi256_si512(raw256);
    const __m512i shuf = _mm512_set_epi8(
        21, 20, 19, 18, 17, 16, 15, 14,   /* lane 7 */
        19, 18, 17, 16, 15, 14, 13, 12,   /* lane 6 */
        17, 16, 15, 14, 13, 12, 11, 10,   /* lane 5 */
        15, 14, 13, 12, 11, 10,  9,  8,   /* lane 4 */
        13, 12, 11, 10,  9,  8,  7,  6,   /* lane 3 */
        11, 10,  9,  8,  7,  6,  5,  4,   /* lane 2 */
         9,  8,  7,  6,  5,  4,  3,  2,   /* lane 1 */
         7,  6,  5,  4,  3,  2,  1,  0);  /* lane 0 */
    __m512i data = _mm512_permutexvar_epi8(shuf, raw);
    /* {0,2,4,6,8,10,12,14} packed LE: 0x0E 0C 0A 08 06 04 02 00 */
    const __m512i ctrl = _mm512_set1_epi64(
        (int64_t)0x0E0C0A0806040200LL);
    __m512i ms = _mm512_multishift_epi64_epi8(ctrl, data);
    return _mm512_and_si512(ms, _mm512_set1_epi8(0x03));
}

/* D=2 safe x4: 16-byte memcpy. */
static inline __m512i flat_d2_unpack64_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[32] = {0};
    memcpy(buf, bm_ptr, 16);
    return flat_d2_unpack64_avx512_fast(buf);
}

/* D=3 x4: 64 codes from 24 valid bm bytes via a 32-byte ymm load (lane 7
 * needs byte 28). */
static inline __m512i flat_d3_unpack64_avx512_fast(const uint8_t *bm_ptr)
{
    __m256i raw256 = _mm256_loadu_si256((const __m256i *)bm_ptr);
    __m512i raw    = _mm512_zextsi256_si512(raw256);
    const __m512i shuf = _mm512_set_epi8(
        28, 27, 26, 25, 24, 23, 22, 21,   /* lane 7 */
        25, 24, 23, 22, 21, 20, 19, 18,   /* lane 6 */
        22, 21, 20, 19, 18, 17, 16, 15,   /* lane 5 */
        19, 18, 17, 16, 15, 14, 13, 12,   /* lane 4 */
        16, 15, 14, 13, 12, 11, 10,  9,   /* lane 3 */
        13, 12, 11, 10,  9,  8,  7,  6,   /* lane 2 */
        10,  9,  8,  7,  6,  5,  4,  3,   /* lane 1 */
         7,  6,  5,  4,  3,  2,  1,  0);  /* lane 0 */
    __m512i data = _mm512_permutexvar_epi8(shuf, raw);
    const __m512i ctrl = _mm512_set1_epi64(
        (int64_t)0x15120F0C09060300LL);   /* {0,3,6,9,12,15,18,21} */
    __m512i ms = _mm512_multishift_epi64_epi8(ctrl, data);
    return _mm512_and_si512(ms, _mm512_set1_epi8(0x07));
}

/* D=3 safe x4: 24-byte memcpy. */
static inline __m512i flat_d3_unpack64_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[32] = {0};
    memcpy(buf, bm_ptr, 24);
    return flat_d3_unpack64_avx512_fast(buf);
}

/* D=4 x4: 64 codes from 32 valid bm bytes via a 32-byte ymm load.  Lane 7
 * reads bytes 28..35; bytes 32..35 are the (zero) high half of the zext
 * but the output mask 0x0F drops their contribution. */
static inline __m512i flat_d4_unpack64_avx512_fast(const uint8_t *bm_ptr)
{
    __m256i raw256 = _mm256_loadu_si256((const __m256i *)bm_ptr);
    __m512i raw    = _mm512_zextsi256_si512(raw256);
    const __m512i shuf = _mm512_set_epi8(
        35, 34, 33, 32, 31, 30, 29, 28,   /* lane 7 (top 4 bytes zero) */
        31, 30, 29, 28, 27, 26, 25, 24,   /* lane 6 */
        27, 26, 25, 24, 23, 22, 21, 20,   /* lane 5 */
        23, 22, 21, 20, 19, 18, 17, 16,   /* lane 4 */
        19, 18, 17, 16, 15, 14, 13, 12,   /* lane 3 */
        15, 14, 13, 12, 11, 10,  9,  8,   /* lane 2 */
        11, 10,  9,  8,  7,  6,  5,  4,   /* lane 1 */
         7,  6,  5,  4,  3,  2,  1,  0);  /* lane 0 */
    __m512i data = _mm512_permutexvar_epi8(shuf, raw);
    /* {0,4,8,12,16,20,24,28} packed LE: 0x1C 18 14 10 0C 08 04 00 */
    const __m512i ctrl = _mm512_set1_epi64(
        (int64_t)0x1C1814100C080400LL);
    __m512i ms = _mm512_multishift_epi64_epi8(ctrl, data);
    return _mm512_and_si512(ms, _mm512_set1_epi8(0x0F));
}

/* D=4 safe x4: 32-byte memcpy. */
static inline __m512i flat_d4_unpack64_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[32] = {0};
    memcpy(buf, bm_ptr, 32);
    return flat_d4_unpack64_avx512_fast(buf);
}

/* D=5 fast x4: 64 codes from 40 valid bm bytes in a single zmm chain.
 * Lane k of the multishift input gets bytes [5k .. 5k+7] of bm (lane stride
 * 5 — code 8k starts at bit 40k = byte 5k bit 0), so each 64-bit lane
 * holds 8 codes at bit offsets {0,5,10,15,20,25,30,35}.  Highest source
 * byte is 42 → caller must guarantee ≥43 valid bm bytes (64-byte load
 * over-reads by 21). */
static inline __m512i flat_d5_unpack64_avx512_fast(const uint8_t *bm_ptr)
{
    __m512i raw = _mm512_loadu_si512((const __m512i *)bm_ptr);
    /* _mm512_set_epi8 takes bytes top-first; rows below are written
     * lane-7 ... lane-0 so the in-memory layout is
     *   lane 0: bm[0..7],  lane 1: bm[5..12], ..., lane 7: bm[35..42] */
    const __m512i shuf = _mm512_set_epi8(
        42, 41, 40, 39, 38, 37, 36, 35,   /* lane 7 */
        37, 36, 35, 34, 33, 32, 31, 30,   /* lane 6 */
        32, 31, 30, 29, 28, 27, 26, 25,   /* lane 5 */
        27, 26, 25, 24, 23, 22, 21, 20,   /* lane 4 */
        22, 21, 20, 19, 18, 17, 16, 15,   /* lane 3 */
        17, 16, 15, 14, 13, 12, 11, 10,   /* lane 2 */
        12, 11, 10,  9,  8,  7,  6,  5,   /* lane 1 */
         7,  6,  5,  4,  3,  2,  1,  0);  /* lane 0 */
    __m512i data = _mm512_permutexvar_epi8(shuf, raw);
    /* bit-offsets {0,5,10,15,20,25,30,35} packed LE into a u64. */
    const __m512i ctrl = _mm512_set1_epi64(
        (int64_t)0x231E19140F0A0500LL);
    __m512i ms = _mm512_multishift_epi64_epi8(ctrl, data);
    return _mm512_and_si512(ms, _mm512_set1_epi8(0x1F));
}

/* D=5 safe x4: 40-byte memcpy for the last chunk. */
static inline __m512i flat_d5_unpack64_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[64] = {0};
    memcpy(buf, bm_ptr, 40);
    return flat_d5_unpack64_avx512_fast(buf);
}

/* D=6 x4: 64 codes from 48 valid bm bytes via a 64-byte zmm load (lane 7
 * needs byte 49). */
static inline __m512i flat_d6_unpack64_avx512_fast(const uint8_t *bm_ptr)
{
    __m512i raw = _mm512_loadu_si512((const __m512i *)bm_ptr);
    const __m512i shuf = _mm512_set_epi8(
        49, 48, 47, 46, 45, 44, 43, 42,   /* lane 7 */
        43, 42, 41, 40, 39, 38, 37, 36,   /* lane 6 */
        37, 36, 35, 34, 33, 32, 31, 30,   /* lane 5 */
        31, 30, 29, 28, 27, 26, 25, 24,   /* lane 4 */
        25, 24, 23, 22, 21, 20, 19, 18,   /* lane 3 */
        19, 18, 17, 16, 15, 14, 13, 12,   /* lane 2 */
        13, 12, 11, 10,  9,  8,  7,  6,   /* lane 1 */
         7,  6,  5,  4,  3,  2,  1,  0);  /* lane 0 */
    __m512i data = _mm512_permutexvar_epi8(shuf, raw);
    /* {0,6,12,18,24,30,36,42} packed LE: 0x2A 24 1E 18 12 0C 06 00 */
    const __m512i ctrl = _mm512_set1_epi64(
        (int64_t)0x2A241E18120C0600LL);
    __m512i ms = _mm512_multishift_epi64_epi8(ctrl, data);
    return _mm512_and_si512(ms, _mm512_set1_epi8(0x3F));
}

/* D=6 safe x4: 48-byte memcpy. */
static inline __m512i flat_d6_unpack64_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[64] = {0};
    memcpy(buf, bm_ptr, 48);
    return flat_d6_unpack64_avx512_fast(buf);
}

/* D=7 x4: 64 codes from 56 valid bm bytes via a 64-byte zmm load (lane 7
 * needs byte 56). */
static inline __m512i flat_d7_unpack64_avx512_fast(const uint8_t *bm_ptr)
{
    __m512i raw = _mm512_loadu_si512((const __m512i *)bm_ptr);
    const __m512i shuf = _mm512_set_epi8(
        56, 55, 54, 53, 52, 51, 50, 49,   /* lane 7 */
        49, 48, 47, 46, 45, 44, 43, 42,   /* lane 6 */
        42, 41, 40, 39, 38, 37, 36, 35,   /* lane 5 */
        35, 34, 33, 32, 31, 30, 29, 28,   /* lane 4 */
        28, 27, 26, 25, 24, 23, 22, 21,   /* lane 3 */
        21, 20, 19, 18, 17, 16, 15, 14,   /* lane 2 */
        14, 13, 12, 11, 10,  9,  8,  7,   /* lane 1 */
         7,  6,  5,  4,  3,  2,  1,  0);  /* lane 0 */
    __m512i data = _mm512_permutexvar_epi8(shuf, raw);
    /* {0,7,14,21,28,35,42,49} packed LE: 0x31 2A 23 1C 15 0E 07 00 */
    const __m512i ctrl = _mm512_set1_epi64(
        (int64_t)0x312A231C150E0700LL);
    __m512i ms = _mm512_multishift_epi64_epi8(ctrl, data);
    return _mm512_and_si512(ms, _mm512_set1_epi8(0x7F));
}

/* D=7 safe x4: 56-byte memcpy. */
static inline __m512i flat_d7_unpack64_avx512_safe(const uint8_t *bm_ptr)
{
    uint8_t buf[64] = {0};
    memcpy(buf, bm_ptr, 56);
    return flat_d7_unpack64_avx512_fast(buf);
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
