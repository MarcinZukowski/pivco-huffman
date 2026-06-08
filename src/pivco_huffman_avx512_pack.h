/* pivco_huffman_avx512_pack.h — flat-subtree D-bit pack (AVX-512 VBMI2).
 *
 * 64 codes per zmm iter via byte-laid intermediate + vpmultishiftqb.
 *
 * The codes_la input lane has the D-bit code at [right_shift, right_shift+D)
 * (left-aligned encoder format).  load_codes_byte right-shifts, narrows
 * u16 -> u8 via vpmovwb, and assembles 64 codes into one zmm (one byte
 * per code, low D bits valid).
 *
 * Pack strategy per D:
 *   - D=2, D=4: codes don't cross byte boundaries, so 4 (D=2) or 2 (D=4)
 *     vpermb gathers at code-stride 4 / 2, plus a fixed left-shift per
 *     group, OR'd together.  Simpler than multishift.
 *   - D=3, D=5, D=6, D=7: codes cross byte boundaries.  Split codes into
 *     G groups (codes mod G, G = ceil(8/D)+1 for D=3, 3 for D=5, 2 for
 *     D=6/7) such that within each group no output byte gets contribution
 *     from two same-group codes.  For each group: mask the byte-laid
 *     input + vpmultishiftqb with broadcast ctrl that pulls each code's
 *     bits to its absolute bit position in the packed stream.  OR the G
 *     group results; vpermb compacts the 8 lanes' 0..D-1 bytes into a
 *     contiguous 8*D-byte stream; masked store writes the valid prefix.
 *
 * Op count per 64 codes (D=5 example):
 *   load_codes_byte (~5 ops) + 3*(mask + multishift) + 2 OR + vpermb
 *   + masked store ~= 12 ops, vs the prior 8-codes-per-iter sllv +
 *   reduce_add path's ~6 ops/chunk * 8 chunks = ~48 ops.
 *
 * Same-session microbench (ns/code on c8i / c8a) vs v1 vector and BMI2:
 *
 *       v1-vec      bmi2       this
 *   D=2  0.204/.088  0.068/.071  0.046/0.021
 *   D=3  0.206/.106  0.102/.115  0.050/0.022
 *   D=4  0.206/.114  0.089/.074  0.038/0.014
 *   D=5  0.207/.106  0.104/.099  0.046/0.019
 *   D=6  0.209/.108  0.104/.114  0.043/0.016
 *   D=7  0.211/.125  0.117/.116  0.043/0.017
 *
 * Internal header.  Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_AVX512_PACK_H
#define PIVCO_HUFFMAN_AVX512_PACK_H

#if !defined(__AVX512BW__) || !defined(__AVX512VBMI__) || !defined(__AVX512VBMI2__)
#error "pivco_huffman_avx512_pack.h requires AVX-512 BW + VBMI + VBMI2"
#endif

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

/* Compact-shuf tables for D=3,5,6,7: gather lane k bytes [0..D-1] into
 * output bytes [k*D .. k*D+D-1].  Bytes beyond 8*D are 0 (masked store). */
static const uint8_t pivco_pack_compact_d3[64] __attribute__((aligned(64))) = {
     0, 1, 2,  8, 9,10, 16,17,18, 24,25,26, 32,33,34, 40,41,42, 48,49,50, 56,57,58,
     0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0
};
static const uint8_t pivco_pack_compact_d5[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4,  8, 9,10,11,12, 16,17,18,19,20, 24,25,26,27,28,
    32,33,34,35,36, 40,41,42,43,44, 48,49,50,51,52, 56,57,58,59,60,
     0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0
};
static const uint8_t pivco_pack_compact_d6[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4, 5,  8, 9,10,11,12,13, 16,17,18,19,20,21, 24,25,26,27,28,29,
    32,33,34,35,36,37, 40,41,42,43,44,45, 48,49,50,51,52,53, 56,57,58,59,60,61,
     0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0
};
static const uint8_t pivco_pack_compact_d7[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4, 5, 6,  8, 9,10,11,12,13,14, 16,17,18,19,20,21,22, 24,25,26,27,28,29,30,
    32,33,34,35,36,37,38, 40,41,42,43,44,45,46, 48,49,50,51,52,53,54, 56,57,58,59,60,61,62,
     0,0,0,0,0,0,0,0
};

/* Load 64 left-aligned u16 codes, right-shift to align code into low D
 * bits, narrow u16 -> u8, assemble into one zmm (1 code per byte).
 *
 * NB: the returned bytes carry whatever sat in the low byte of each
 * codes_la lane after right-shift — high bits above D may be GARBAGE.
 * Callers must mask to D bits if their subsequent pipeline would leak
 * those bits.  The multishift-based D=3/5/6/7 helpers don't mask here
 * because their per-group `_mm512_and_si512(cb, mX)` already clips both
 * the unwanted bytes (zero outside the group) AND the high bits within
 * each group byte (0x07 / 0x1F / 0x3F / 0x7F).  The vpermb-stride D=2
 * and D=4 helpers DO mask because their shift-then-OR step would
 * otherwise leak high bits across byte boundaries within each u32. */
static inline __m512i pivco_pack_load_codes_byte(const uint16_t *codes_la,
                                                  int right_shift)
{
    __m512i lo16 = _mm512_loadu_si512((const __m512i *)(codes_la));
    __m512i hi16 = _mm512_loadu_si512((const __m512i *)(codes_la + 32));
    __m256i lo_b = _mm512_cvtepi16_epi8(_mm512_srli_epi16(lo16, right_shift));
    __m256i hi_b = _mm512_cvtepi16_epi8(_mm512_srli_epi16(hi16, right_shift));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo_b), hi_b, 1);
}

/* D=2 (4 codes per byte): 4 groups by code mod 4, gather + shift + OR. */
static inline int pack_d2_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    /* Group g (g in 0..3) gathers codes (g, g+4, g+8, ..., g+60) into the
     * low 16 output bytes; group g's bits land at position 2g within each
     * output byte. */
    const __m512i shuf0 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        60,56,52,48,44,40,36,32, 28,24,20,16,12,8,4,0);
    const __m512i shuf1 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        61,57,53,49,45,41,37,33, 29,25,21,17,13,9,5,1);
    const __m512i shuf2 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        62,58,54,50,46,42,38,34, 30,26,22,18,14,10,6,2);
    const __m512i shuf3 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        63,59,55,51,47,43,39,35, 31,27,23,19,15,11,7,3);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        /* D=2 needs the explicit mask: the slli-then-OR below would otherwise
         * leak high bits across byte boundaries within each u32 lane. */
        __m512i cb = _mm512_and_si512(
            pivco_pack_load_codes_byte(codes_la + i, right_shift),
            _mm512_set1_epi8(0x03));
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i g2 = _mm512_permutexvar_epi8(shuf2, cb);
        __m512i g3 = _mm512_permutexvar_epi8(shuf3, cb);
        __m512i packed = _mm512_or_si512(
            _mm512_or_si512(g0, _mm512_slli_epi32(g1, 2)),
            _mm512_or_si512(_mm512_slli_epi32(g2, 4), _mm512_slli_epi32(g3, 6)));
        _mm512_mask_storeu_epi8(out + ((i * 2) >> 3),
                                 (__mmask64)0xFFFFULL, packed);
    }
    return i;
}

/* D=3: 4 groups (codes mod 4).  Each chunk of 8 codes -> 3 output bytes. */
static inline int pack_d3_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    const __m512i mA = _mm512_set1_epi64((int64_t)0x0000000700000007ULL); /* bytes 0,4 */
    const __m512i mB = _mm512_set1_epi64((int64_t)0x0000070000000700ULL); /* bytes 1,5 */
    const __m512i mC = _mm512_set1_epi64((int64_t)0x0007000000070000ULL); /* bytes 2,6 */
    const __m512i mD = _mm512_set1_epi64((int64_t)0x0700000007000000ULL); /* bytes 3,7 */
    /* Per-byte multishift ctrls (lo->hi byte order).  Byte 2 of cA reads
     * a zero region of lane_A (Group A doesn't contribute to output byte
     * 2; pulling from a masked-zero byte avoids leaking code 0 in). */
    const __m512i cA = _mm512_set1_epi64((int64_t)0x0000000000081C00ULL); /* {0,28,8,...} */
    const __m512i cB = _mm512_set1_epi64((int64_t)0x0000000000292105ULL); /* {5,33,41,...} */
    const __m512i cC = _mm512_set1_epi64((int64_t)0x00000000002E120AULL); /* {10,18,46,...} */
    const __m512i cD = _mm512_set1_epi64((int64_t)0x0000000000331700ULL); /* {0,23,51,...} */
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pivco_pack_load_codes_byte(codes_la + i, right_shift);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i c = _mm512_multishift_epi64_epi8(cC, _mm512_and_si512(cb, mC));
        __m512i d = _mm512_multishift_epi64_epi8(cD, _mm512_and_si512(cb, mD));
        __m512i packed = _mm512_or_si512(_mm512_or_si512(a, b),
                                          _mm512_or_si512(c, d));
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pivco_pack_compact_d3), packed);
        _mm512_mask_storeu_epi8(out + ((i * 3) >> 3),
                                 (__mmask64)0xFFFFFFULL, compact);
    }
    return i;
}

/* D=4 (2 codes per byte): 2 groups (even/odd), gather + shift + OR. */
static inline int pack_d4_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    const __m512i shuf0 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        62,60,58,56,54,52,50,48, 46,44,42,40,38,36,34,32,
        30,28,26,24,22,20,18,16, 14,12,10, 8, 6, 4, 2, 0);
    const __m512i shuf1 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        63,61,59,57,55,53,51,49, 47,45,43,41,39,37,35,33,
        31,29,27,25,23,21,19,17, 15,13,11, 9, 7, 5, 3, 1);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        /* D=4 needs the explicit mask: see D=2 comment. */
        __m512i cb = _mm512_and_si512(
            pivco_pack_load_codes_byte(codes_la + i, right_shift),
            _mm512_set1_epi8(0x0F));
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i packed = _mm512_or_si512(g0, _mm512_slli_epi32(g1, 4));
        _mm512_mask_storeu_epi8(out + ((i * 4) >> 3),
                                 (__mmask64)0xFFFFFFFFULL, packed);
    }
    return i;
}

/* D=5: 3 groups (codes mod 3: A={0,3,6}, B={1,4,7}, C={2,5}). */
static inline int pack_d5_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    const __m512i mA = _mm512_set1_epi64((int64_t)0x001F00001F00001FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x1F00001F00001F00ULL);
    const __m512i mC = _mm512_set1_epi64((int64_t)0x00001F00001F0000ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x000000322A191100ULL); /* {0,17,25,42,50,...} */
    const __m512i cB = _mm512_set1_epi64((int64_t)0x00000035241C0B03ULL); /* {3,11,28,36,53,...} */
    const __m512i cC = _mm512_set1_epi64((int64_t)0x0000000027000E00ULL); /* {0,14,0,39,...} */
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pivco_pack_load_codes_byte(codes_la + i, right_shift);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i c = _mm512_multishift_epi64_epi8(cC, _mm512_and_si512(cb, mC));
        __m512i packed = _mm512_or_si512(_mm512_or_si512(a, b), c);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pivco_pack_compact_d5), packed);
        _mm512_mask_storeu_epi8(out + ((i * 5) >> 3),
                                 (__mmask64)0xFFFFFFFFFFULL, compact);
    }
    return i;
}

/* D=6: 2 groups (even/odd). */
static inline int pack_d6_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    const __m512i mA = _mm512_set1_epi64((int64_t)0x003F003F003F003FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x3F003F003F003F00ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x0000342C20140C00ULL); /* {0,12,20,32,44,52,...} */
    const __m512i cB = _mm512_set1_epi64((int64_t)0x0000362A22160A02ULL); /* {2,10,22,34,42,54,...} */
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pivco_pack_load_codes_byte(codes_la + i, right_shift);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i packed = _mm512_or_si512(a, b);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pivco_pack_compact_d6), packed);
        _mm512_mask_storeu_epi8(out + ((i * 6) >> 3),
                                 (__mmask64)0xFFFFFFFFFFFFULL, compact);
    }
    return i;
}

/* D=7: 2 groups (even/odd). */
static inline int pack_d7_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    const __m512i mA = _mm512_set1_epi64((int64_t)0x007F007F007F007FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x7F007F007F007F00ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x00362E241C120A00ULL); /* {0,10,18,28,36,46,54,...} */
    const __m512i cB = _mm512_set1_epi64((int64_t)0x00372D251B130901ULL); /* {1,9,19,27,37,45,55,...} */
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pivco_pack_load_codes_byte(codes_la + i, right_shift);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i packed = _mm512_or_si512(a, b);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pivco_pack_compact_d7), packed);
        _mm512_mask_storeu_epi8(out + ((i * 7) >> 3),
                                 (__mmask64)0x00FFFFFFFFFFFFFFULL, compact);
    }
    return i;
}

#endif /* PIVCO_HUFFMAN_AVX512_PACK_H */
