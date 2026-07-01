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


/* ---- rank-based (partbyrank) variants -------------------------------------
 * The flat local code is (rank - base), already a D-bit value in each byte —
 * so the byte-laid `cb` comes straight from a u8 load + subtract (no u16 load
 * + cvtepi16_epi8 narrow).  The pack BACKEND is byte-for-byte the same. */
static inline __m512i pivco_pack_load_byte(const uint8_t *ranks, uint8_t base)
{
    return _mm512_sub_epi8(_mm512_loadu_si512((const __m512i *)ranks),
                           _mm512_set1_epi8((char)base));
}

static inline int pack_d2_avx512(uint8_t *out, const uint8_t *ranks,
                                   int n, uint8_t base)
{
    /* Group g (g in 0..3) gathers ranks (g, g+4, g+8, ..., g+60) into the
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
        /* Local code is in [0,2^D) (rank - flat_base_rank over the flat subtree),
         * so the slli-then-OR below can't leak high bits across byte boundaries
         * within a u32 lane -- no mask needed. */
        __m512i cb = pivco_pack_load_byte(ranks + i, base);
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

/* D=3: 4 groups (ranks mod 4).  Each chunk of 8 ranks -> 3 output bytes. */

static inline int pack_d3_avx512(uint8_t *out, const uint8_t *ranks,
                                   int n, uint8_t base)
{
    const __m512i mA = _mm512_set1_epi64((int64_t)0x0000000700000007ULL); /* bytes 0,4 */
    const __m512i mB = _mm512_set1_epi64((int64_t)0x0000070000000700ULL); /* bytes 1,5 */
    const __m512i mC = _mm512_set1_epi64((int64_t)0x0007000000070000ULL); /* bytes 2,6 */
    const __m512i mD = _mm512_set1_epi64((int64_t)0x0700000007000000ULL); /* bytes 3,7 */
    /* Per-byte multishift ctrls (lo->hi byte order).  Byte 2 of cA reads
     * a zero region of lane_A (Group A doesn't contribute to output byte
     * 2; pulling from a masked-zero byte avoids leaking rank 0 in). */
    const __m512i cA = _mm512_set1_epi64((int64_t)0x0000000000081C00ULL); /* {0,28,8,...} */
    const __m512i cB = _mm512_set1_epi64((int64_t)0x0000000000292105ULL); /* {5,33,41,...} */
    const __m512i cC = _mm512_set1_epi64((int64_t)0x00000000002E120AULL); /* {10,18,46,...} */
    const __m512i cD = _mm512_set1_epi64((int64_t)0x0000000000331700ULL); /* {0,23,51,...} */
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pivco_pack_load_byte(ranks + i, base);
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

/* D=4 (2 ranks per byte): 2 groups (even/odd), gather + shift + OR. */

static inline int pack_d4_avx512(uint8_t *out, const uint8_t *ranks,
                                   int n, uint8_t base)
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
        /* Local code in [0,2^D); no mask needed (see D=2). */
        __m512i cb = pivco_pack_load_byte(ranks + i, base);
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i packed = _mm512_or_si512(g0, _mm512_slli_epi32(g1, 4));
        _mm512_mask_storeu_epi8(out + ((i * 4) >> 3),
                                 (__mmask64)0xFFFFFFFFULL, packed);
    }
    return i;
}

/* D=5/6/7 pack via ryg multiply-as-shift (port of AVX2 a1aa6b9):
 *   - mask byte-laid ranks to D bits
 *   - vpmaddubsw c0   word[i] = cb[2i]   + cb[2i+1]   * 2^D    (2D bits)
 *   - vpmaddwd   c1   dword[i] = word[2i] + word[2i+1] * 2^(2D) (4D bits)
 *   - vpsrlq + vpternlogq 0xE4 to merge dword[2i+1] into dword[2i]'s
 *     u64 lane: bits [0..4D-1] = dword[2i], bits [4D..8D-1] = dword[2i+1]
 *   - vpermb compact + masked store
 *
 * Beats the per-group multishift path on Intel (Granite Rapids -16 to
 * -23% cyc/elem, Sapphire Rapids -10 to -18%); ties or marginally loses
 * on AMD (Zen 4 D=5 -24% else tied, Zen 5 D=5 -10% / D=6,7 +5%).  See
 * scratch bench results in the commit message.
 *
 * For D=5 only, the two-byte mults of vpmaddubsw can produce u16 lanes
 * up to 31 + 31*32 = 1023 (fits u16), then vpmaddwd up to 1023 + 1023 *
 * 1024 ≈ 1.05M (fits u32) -- 4D = 20 bits is the maximum used.  Same
 * envelope analysis for D=6 (24 bits) and D=7 (28 bits). */

/* D=8: byte-aligned; 64 ranks -> 64 bytes (sub base + store). */
static inline int pack_d8_avx512(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    int i = 0;
    for (; i + 64 <= n; i += 64)
        _mm512_storeu_si512((__m512i *)(out + i), pivco_pack_load_byte(ranks + i, base));
    return i;
}

#define PIVCO_PACK_AVX512_RYG_DN(NAME, D_VAL, COMPACT_TAB, STORE_MASK)            \
static inline int NAME(uint8_t *out, const uint8_t *ranks,                  \
                       int n, uint8_t base)                                    \
{                                                                                 \
    const __m512i c0 = _mm512_set1_epi16(                                         \
        (int16_t)(((1 << (D_VAL)) << 8) | 1));                                    \
    const __m512i c1 = _mm512_set1_epi32(                                         \
        (int32_t)(((int32_t)1 << (2*(D_VAL))) << 16) | 1);                        \
    const __m512i c3 = _mm512_set1_epi64(                                         \
        (int64_t)(((int64_t)1 << (4*(D_VAL))) - 1));                              \
    int i = 0;                                                                    \
    for (; i + 64 <= n; i += 64) {                                                \
        /* local code already in [0,2^D) -- no per-byte mask needed */            \
        __m512i cb = pivco_pack_load_byte(ranks + i, base);                       \
        __m512i x  = _mm512_maddubs_epi16(c0, cb);                                \
        x = _mm512_madd_epi16(x, c1);                                             \
        __m512i xs = _mm512_srli_epi64(x, 32 - 4*(D_VAL));                        \
        /* (x & c3) | (xs & ~c3)  via vpternlogq 0xE4 */                          \
        x = _mm512_ternarylogic_epi64(x, xs, c3, 0xE4);                           \
        __m512i compact = _mm512_permutexvar_epi8(                                \
            _mm512_load_si512((const __m512i *)COMPACT_TAB), x);                  \
        _mm512_mask_storeu_epi8(out + ((i * (D_VAL)) >> 3),                       \
                                 (__mmask64)(STORE_MASK), compact);               \
    }                                                                             \
    return i;                                                                     \
}
PIVCO_PACK_AVX512_RYG_DN(pack_d5_avx512, 5, pivco_pack_compact_d5, 0xFFFFFFFFFFULL)
PIVCO_PACK_AVX512_RYG_DN(pack_d6_avx512, 6, pivco_pack_compact_d6, 0xFFFFFFFFFFFFULL)
PIVCO_PACK_AVX512_RYG_DN(pack_d7_avx512, 7, pivco_pack_compact_d7, 0x00FFFFFFFFFFFFFFULL)
#undef PIVCO_PACK_AVX512_RYG_DN

#endif /* PIVCO_HUFFMAN_AVX512_PACK_H */
