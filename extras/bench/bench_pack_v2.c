/* bench_pack_v2.c — multishift-based pack prototype + bench, D=2..7.
 *
 * Compares four implementations of "pack n D-bit codes (LSB-first) into a
 * contiguous bitstream":
 *   - scalar     : reference, one bit at a time
 *   - bmi2       : production path (_pext_u64, 8 codes per pext)
 *   - vec_v1     : production AVX-512 vector path (sllv + reduce_add, 8 codes/lane)
 *   - vec_v2     : new multishift-based path, 64 codes per zmm
 *
 * Input: codes_la[i] are uint16 values; the D-bit code lives at
 * [right_shift, right_shift+D) inside each lane.  After right-shift the
 * code value occupies low D bits of the u16.
 *
 * Build (Intel/AMD with AVX-512 VBMI2):
 *   cc -O3 -march=native -o bench_pack_v2 extras/bench/bench_pack_v2.c
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if !defined(__AVX512F__) || !defined(__AVX512BW__) || \
    !defined(__AVX512VBMI__) || !defined(__AVX512VBMI2__) || !defined(__BMI2__)
int main(void) { puts("bench_pack_v2: needs AVX-512 VBMI2 + BMI2"); return 0; }
#else

#include <immintrin.h>

#define N 8192          /* codes per call */
#define REPS 100000     /* iterations for timing */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============================================================
 *   Scalar reference
 * ============================================================ */
static void pack_scalar(uint8_t *out, const uint16_t *codes_la, int n,
                        int D, int right_shift)
{
    uint32_t mask = (1u << D) - 1;
    int bit_pos = 0;
    int total = (n * D + 7) >> 3;
    memset(out, 0, total);
    for (int i = 0; i < n; i++) {
        uint32_t code = ((uint32_t)codes_la[i] >> right_shift) & mask;
        int byte_idx = bit_pos >> 3;
        int bit_off  = bit_pos & 7;
        out[byte_idx] |= (uint8_t)(code << bit_off);
        if (bit_off + D > 8)
            out[byte_idx + 1] |= (uint8_t)(code >> (8 - bit_off));
        if (bit_off + D > 16)
            out[byte_idx + 2] |= (uint8_t)(code >> (16 - bit_off));
        bit_pos += D;
    }
}

/* ============================================================
 *   BMI2 pext path (production)
 * ============================================================ */
static int pack_bmi2(uint8_t *out, const uint16_t *codes_la, int n,
                     int D, int right_shift)
{
    uint64_t field = (((uint64_t)1 << D) - 1) << right_shift;
    uint64_t mask  = field | (field << 16) | (field << 32) | (field << 48);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w0, w1;
        memcpy(&w0, codes_la + i,     8);
        memcpy(&w1, codes_la + i + 4, 8);
        uint64_t packed = _pext_u64(w0, mask)
                        | (_pext_u64(w1, mask) << (4 * D));
        memcpy(out + ((i * D) >> 3), &packed, (size_t)D);
    }
    return i;
}

/* ============================================================
 *   v1: production AVX-512 vector path
 *   8 codes/iter via sllv + reduce_add across 8 u64 lanes.
 * ============================================================ */
#define PACK_DN_VEC_V1(NAME, D_VAL, BITS_OUT)                                   \
static int NAME(uint8_t *out, const uint16_t *codes_la,                         \
                int n, int right_shift)                                          \
{                                                                                \
    static const int64_t shifts[8] = {                                           \
        0, D_VAL, 2*D_VAL, 3*D_VAL, 4*D_VAL, 5*D_VAL, 6*D_VAL, 7*D_VAL          \
    };                                                                           \
    __m512i shift_vec = _mm512_loadu_si512((const __m512i *)shifts);             \
    __m512i mask_vec  = _mm512_set1_epi64((1ULL << D_VAL) - 1);                  \
    int i = 0;                                                                   \
    for (; i + 8 <= n; i += 8) {                                                 \
        __m128i v16 = _mm_loadu_si128((const __m128i *)(codes_la + i));          \
        __m512i v64 = _mm512_cvtepu16_epi64(v16);                                \
        v64 = _mm512_srli_epi64(v64, right_shift);                               \
        v64 = _mm512_and_si512(v64, mask_vec);                                   \
        v64 = _mm512_sllv_epi64(v64, shift_vec);                                 \
        uint64_t packed = _mm512_reduce_add_epi64(v64);                          \
        int bi = i * D_VAL / 8;                                                  \
        memcpy(out + bi, &packed, (BITS_OUT + 7) / 8);                           \
    }                                                                            \
    return i;                                                                    \
}
PACK_DN_VEC_V1(pack_v1_d2, 2, 16)
PACK_DN_VEC_V1(pack_v1_d3, 3, 24)
PACK_DN_VEC_V1(pack_v1_d4, 4, 32)
PACK_DN_VEC_V1(pack_v1_d5, 5, 40)
PACK_DN_VEC_V1(pack_v1_d6, 6, 48)
PACK_DN_VEC_V1(pack_v1_d7, 7, 56)
#undef PACK_DN_VEC_V1

/* ============================================================
 *   v2: multishift pack, 64 codes per zmm.
 *
 * Strategy (D ∈ {3,5,6,7}):
 *   - Load 64 codes (u16), right-shift, narrow to 64 bytes (1 code per byte).
 *   - Mask + multishift per group; OR groups; compact via vpermb; masked store.
 *   - Groups split codes-mod-G where G = ceil(8/D)+1 so within each group
 *     each output byte gets ≤ 1 code → no in-lane reduce.
 *
 * Strategy for D=2 and D=4 (powers of 2 ≤ 4):
 *   - 4 codes-per-byte (D=2) or 2 codes-per-byte (D=4) layout — codes
 *     don't cross byte boundaries, so a simple vpermb gather + shift + OR
 *     is shorter than multishift.
 * ============================================================ */

/* Helper: load 64 u16 codes, right-shift, narrow to 1 byte per code,
 * mask to low D bits.  Returns a zmm with codes 0..63 at byte positions
 * 0..63. */
static inline __m512i load_codes_byte(const uint16_t *codes_la, int right_shift, uint8_t code_mask) {
    __m512i lo16 = _mm512_loadu_si512((const __m512i *)(codes_la));
    __m512i hi16 = _mm512_loadu_si512((const __m512i *)(codes_la + 32));
    __m256i lo_b = _mm512_cvtepi16_epi8(_mm512_srli_epi16(lo16, right_shift));
    __m256i hi_b = _mm512_cvtepi16_epi8(_mm512_srli_epi16(hi16, right_shift));
    __m512i codes_byte = _mm512_inserti64x4(_mm512_castsi256_si512(lo_b), hi_b, 1);
    return _mm512_and_si512(codes_byte, _mm512_set1_epi8(code_mask));
}

/* D=2: 4 groups (codes mod 4), simple gather + shift + OR.  No multishift. */
static int pack_v2_d2(uint8_t *out, const uint16_t *codes_la, int n, int right_shift) {
    int i = 0;
    /* Per-group vpermb tables: pull codes at stride 4 starting at offset g into
     * output bytes 0..15.  High 48 bytes of result don't-care (masked store). */
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
    for (; i + 64 <= n; i += 64) {
        __m512i cb = load_codes_byte(codes_la + i, right_shift, 0x03);
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i g2 = _mm512_permutexvar_epi8(shuf2, cb);
        __m512i g3 = _mm512_permutexvar_epi8(shuf3, cb);
        __m512i packed = _mm512_or_si512(
            _mm512_or_si512(g0, _mm512_slli_epi32(g1, 2)),
            _mm512_or_si512(_mm512_slli_epi32(g2, 4), _mm512_slli_epi32(g3, 6)));
        _mm512_storeu_si512(out + ((i * 2) >> 3), packed);
    }
    return i;
}

/* D=4: 2 groups (even/odd codes), simple gather + shift + OR.  No multishift. */
static int pack_v2_d4(uint8_t *out, const uint16_t *codes_la, int n, int right_shift) {
    int i = 0;
    /* Even codes 0,2,4,...,62 → output low nibbles (bytes 0..31). */
    const __m512i shuf0 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        62,60,58,56,54,52,50,48, 46,44,42,40,38,36,34,32,
        30,28,26,24,22,20,18,16, 14,12,10, 8, 6, 4, 2, 0);
    /* Odd codes 1,3,...,63 → output high nibbles. */
    const __m512i shuf1 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        63,61,59,57,55,53,51,49, 47,45,43,41,39,37,35,33,
        31,29,27,25,23,21,19,17, 15,13,11, 9, 7, 5, 3, 1);
    for (; i + 64 <= n; i += 64) {
        __m512i cb = load_codes_byte(codes_la + i, right_shift, 0x0F);
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i packed = _mm512_or_si512(g0, _mm512_slli_epi32(g1, 4));
        _mm512_storeu_si512(out + ((i * 4) >> 3), packed);
    }
    return i;
}

/* Compact shuf tables: each u64 lane k contributes bytes [0..D-1] →
 * output bytes [k*D..k*D+D-1].  Other positions don't-care (masked store). */
static const uint8_t compact_d3_tab[64] __attribute__((aligned(64))) = {
     0, 1, 2,  8, 9,10, 16,17,18, 24,25,26, 32,33,34, 40,41,42, 48,49,50, 56,57,58,
     0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0
};
static const uint8_t compact_d5_tab[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4,  8, 9,10,11,12, 16,17,18,19,20, 24,25,26,27,28,
    32,33,34,35,36, 40,41,42,43,44, 48,49,50,51,52, 56,57,58,59,60,
     0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0
};
static const uint8_t compact_d6_tab[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4, 5,  8, 9,10,11,12,13, 16,17,18,19,20,21, 24,25,26,27,28,29,
    32,33,34,35,36,37, 40,41,42,43,44,45, 48,49,50,51,52,53, 56,57,58,59,60,61,
     0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0
};
static const uint8_t compact_d7_tab[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4, 5, 6,  8, 9,10,11,12,13,14, 16,17,18,19,20,21,22, 24,25,26,27,28,29,30,
    32,33,34,35,36,37,38, 40,41,42,43,44,45,46, 48,49,50,51,52,53,54, 56,57,58,59,60,61,62,
     0,0,0,0,0,0,0,0
};

/* D=3: 4 groups (codes mod 4). */
static int pack_v2_d3(uint8_t *out, const uint16_t *codes_la, int n, int right_shift) {
    const __m512i mA = _mm512_set1_epi64((int64_t)0x0000000700000007ULL); /* bytes 0,4 */
    const __m512i mB = _mm512_set1_epi64((int64_t)0x0000070000000700ULL); /* bytes 1,5 */
    const __m512i mC = _mm512_set1_epi64((int64_t)0x0007000000070000ULL); /* bytes 2,6 */
    const __m512i mD = _mm512_set1_epi64((int64_t)0x0700000007000000ULL); /* bytes 3,7 */
    const __m512i cA = _mm512_set1_epi64((int64_t)0x0000000000081C00ULL); /* {0,28,8,...} — byte 2 reads zero byte */
    const __m512i cB = _mm512_set1_epi64((int64_t)0x0000000000292105ULL); /* {5,33,41,...} */
    const __m512i cC = _mm512_set1_epi64((int64_t)0x00000000002E120AULL); /* {10,18,46,...} */
    const __m512i cD = _mm512_set1_epi64((int64_t)0x0000000000331700ULL); /* {0,23,51,...} */
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = load_codes_byte(codes_la + i, right_shift, 0x07);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i c = _mm512_multishift_epi64_epi8(cC, _mm512_and_si512(cb, mC));
        __m512i d = _mm512_multishift_epi64_epi8(cD, _mm512_and_si512(cb, mD));
        __m512i packed = _mm512_or_si512(_mm512_or_si512(a, b),
                                          _mm512_or_si512(c, d));
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)compact_d3_tab), packed);
        _mm512_storeu_si512(out + ((i * 3) >> 3), compact);
    }
    return i;
}

/* D=5: 3 groups (codes mod 3 within a chunk: A={0,3,6}, B={1,4,7}, C={2,5}). */
static int pack_v2_d5(uint8_t *out, const uint16_t *codes_la, int n, int right_shift) {
    const __m512i mA = _mm512_set1_epi64((int64_t)0x001F00001F00001FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x1F00001F00001F00ULL);
    const __m512i mC = _mm512_set1_epi64((int64_t)0x00001F00001F0000ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x000000322A191100ULL);
    const __m512i cB = _mm512_set1_epi64((int64_t)0x00000035241C0B03ULL);
    const __m512i cC = _mm512_set1_epi64((int64_t)0x0000000027000E00ULL); /* {0,14,0,39,...} */
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = load_codes_byte(codes_la + i, right_shift, 0x1F);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i c = _mm512_multishift_epi64_epi8(cC, _mm512_and_si512(cb, mC));
        __m512i packed = _mm512_or_si512(_mm512_or_si512(a, b), c);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)compact_d5_tab), packed);
        _mm512_storeu_si512(out + ((i * 5) >> 3), compact);
    }
    return i;
}

/* D=6: 2 groups (even/odd). */
static int pack_v2_d6(uint8_t *out, const uint16_t *codes_la, int n, int right_shift) {
    const __m512i mA = _mm512_set1_epi64((int64_t)0x003F003F003F003FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x3F003F003F003F00ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x0000342C20140C00ULL);
    const __m512i cB = _mm512_set1_epi64((int64_t)0x0000362A22160A02ULL);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = load_codes_byte(codes_la + i, right_shift, 0x3F);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i packed = _mm512_or_si512(a, b);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)compact_d6_tab), packed);
        _mm512_storeu_si512(out + ((i * 6) >> 3), compact);
    }
    return i;
}

/* D=7: 2 groups (even/odd). */
static int pack_v2_d7(uint8_t *out, const uint16_t *codes_la, int n, int right_shift) {
    const __m512i mA = _mm512_set1_epi64((int64_t)0x007F007F007F007FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x7F007F007F007F00ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x00362E241C120A00ULL);
    const __m512i cB = _mm512_set1_epi64((int64_t)0x00372D251B130901ULL);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = load_codes_byte(codes_la + i, right_shift, 0x7F);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i packed = _mm512_or_si512(a, b);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)compact_d7_tab), packed);
        _mm512_storeu_si512(out + ((i * 7) >> 3), compact);
    }
    return i;
}

/* ============================================================
 *   v3: ryg's multiply-as-shift pack (pmaddubsw + pmaddwd + ternlogd + vpcompressb).
 *
 * Step-by-step for D-bit codes (D in 1..7), starting from 64 codes laid
 * out 1-per-byte (low D bits valid, high bits zero):
 *
 *   1. vpmaddubsw  c0 = (1 | (1<<D) << 8) per word
 *        word[i] = code[2i] + code[2i+1] * 2^D    -> 2 codes packed in 2D bits
 *   2. vpmaddwd    c1 = (1 | (1<<2D) << 16) per dword
 *        dword[i] = word[2i] + word[2i+1] * 2^(2D) -> 4 codes packed in 4D bits
 *   3. ternlogd    c3 = (1 << 4D) - 1  (mask for low 4D bits)
 *        result = (x & c3) | ((x >> (32-4D)) & ~c3)  -> 8 codes in 8D bits per qword
 *   4. vpcompressb mask = (1<<D)-1 repeated per byte position
 *        compact 8 valid bytes per qword to contiguous output
 *
 * Per 64 codes: 1 load_codes + maddubs + madd + psrlq + ternlog + compressstore
 *   = ~6 ops vs v2's ~9-10.
 * ============================================================ */
#define PACK_V3_DN(NAME, D_VAL)                                                 \
static int NAME(uint8_t *out, const uint16_t *codes_la,                         \
                int n, int right_shift)                                          \
{                                                                                \
    const __m512i c0 = _mm512_set1_epi16(                                        \
        (int16_t)(((1 << (D_VAL)) << 8) | 1));                                   \
    const __m512i c1 = _mm512_set1_epi32(                                        \
        (int32_t)(((int32_t)1 << (2*(D_VAL))) << 16) | 1);                       \
    const __m512i c3 = _mm512_set1_epi64(                                        \
        (int64_t)(((int64_t)1 << (4*(D_VAL))) - 1));                             \
    const uint64_t _b = (1u << (D_VAL)) - 1;                                     \
    const __mmask64 cmask = (__mmask64)(                                         \
        _b | (_b << 8)  | (_b << 16) | (_b << 24) |                              \
        (_b << 32) | (_b << 40) | (_b << 48) | (_b << 56));                      \
    int i = 0;                                                                   \
    for (; i + 64 <= n; i += 64) {                                               \
        __m512i cb = load_codes_byte(codes_la + i, right_shift,                  \
                                      (uint8_t)((1 << (D_VAL)) - 1));            \
        __m512i x  = _mm512_maddubs_epi16(c0, cb);                               \
        x = _mm512_madd_epi16(x, c1);                                            \
        __m512i xs = _mm512_srli_epi64(x, 32 - 4*(D_VAL));                       \
        x = _mm512_ternarylogic_epi64(x, xs, c3, 0xE4);  /* (A&C)|(B&~C) */      \
        __m512i zc = _mm512_maskz_compress_epi8(cmask, x);                       \
        _mm512_storeu_si512(out + (i * (D_VAL) >> 3), zc);                       \
    }                                                                            \
    return i;                                                                    \
}
PACK_V3_DN(pack_v3_d2, 2)
PACK_V3_DN(pack_v3_d3, 3)
PACK_V3_DN(pack_v3_d4, 4)
PACK_V3_DN(pack_v3_d5, 5)
PACK_V3_DN(pack_v3_d6, 6)
PACK_V3_DN(pack_v3_d7, 7)
#undef PACK_V3_DN

/* ============================================================
 *   Driver
 * ============================================================ */

static void scalar_tail(uint8_t *out, const uint16_t *codes_la,
                        int start, int n, int D, int right_shift) {
    uint32_t mask = (1u << D) - 1;
    for (int i = start; i < n; i++) {
        uint32_t code = ((uint32_t)codes_la[i] >> right_shift) & mask;
        int bit_pos = i * D;
        int byte_idx = bit_pos >> 3;
        int bit_off  = bit_pos & 7;
        out[byte_idx] |= (uint8_t)(code << bit_off);
        if (bit_off + D > 8)
            out[byte_idx + 1] |= (uint8_t)(code >> (8 - bit_off));
        if (bit_off + D > 16)
            out[byte_idx + 2] |= (uint8_t)(code >> (16 - bit_off));
    }
}

static int verify(const uint8_t *a, const uint8_t *b, int total) {
    for (int i = 0; i < total; i++) if (a[i] != b[i]) return i;
    return -1;
}

typedef int (*pack_fn)(uint8_t *, const uint16_t *, int, int);
/* Per-D wrapper to give pack_bmi2 a 4-arg signature like the others. */
#define BMI2_WRAP(NAME, DVAL)                                                   \
    static int NAME(uint8_t *out, const uint16_t *codes_la, int n, int rs) {    \
        return pack_bmi2(out, codes_la, n, DVAL, rs);                            \
    }
BMI2_WRAP(pack_bmi2_d2, 2)
BMI2_WRAP(pack_bmi2_d3, 3)
BMI2_WRAP(pack_bmi2_d4, 4)
BMI2_WRAP(pack_bmi2_d5, 5)
BMI2_WRAP(pack_bmi2_d6, 6)
BMI2_WRAP(pack_bmi2_d7, 7)
#undef BMI2_WRAP

static double time_fn(pack_fn fn, uint8_t *out, const uint16_t *codes_la,
                      int n, int right_shift, int D, int reps)
{
    int total = (n * D + 7) >> 3;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        memset(out, 0, total);
        int i = fn(out, codes_la, n, right_shift);
        scalar_tail(out, codes_la, i, n, D, right_shift);
    }
    double t1 = now_sec();
    return (t1 - t0) / ((double)n * reps) * 1e9;   /* ns/code */
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : REPS;
    printf("bench_pack_v2: N=%d, REPS=%d\n\n", N, reps);

    uint16_t *codes_la = aligned_alloc(64, N * 2 + 128);
    uint8_t *out_scalar = aligned_alloc(64, N + 128);
    uint8_t *out_bmi2   = aligned_alloc(64, N + 128);
    uint8_t *out_v1     = aligned_alloc(64, N + 128);
    uint8_t *out_v2     = aligned_alloc(64, N + 128);
    uint8_t *out_v3     = aligned_alloc(64, N + 128);

    srand(42);
    for (int i = 0; i < N; i++) codes_la[i] = (uint16_t)rand();
    int right_shift = 0;

    struct { int D; pack_fn bmi2; pack_fn v1; pack_fn v2; pack_fn v3; } cfg[] = {
        {2, pack_bmi2_d2, pack_v1_d2, pack_v2_d2, pack_v3_d2},
        {3, pack_bmi2_d3, pack_v1_d3, pack_v2_d3, pack_v3_d3},
        {4, pack_bmi2_d4, pack_v1_d4, pack_v2_d4, pack_v3_d4},
        {5, pack_bmi2_d5, pack_v1_d5, pack_v2_d5, pack_v3_d5},
        {6, pack_bmi2_d6, pack_v1_d6, pack_v2_d6, pack_v3_d6},
        {7, pack_bmi2_d7, pack_v1_d7, pack_v2_d7, pack_v3_d7},
    };

    printf("%-3s %8s %8s %8s %8s %8s   %s\n",
           "D", "scalar", "bmi2", "vec_v1", "vec_v2", "vec_v3", "check");
    for (size_t c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
        int D = cfg[c].D;
        int total = (N * D + 7) >> 3;

        /* Correctness check. */
        pack_scalar(out_scalar, codes_la, N, D, right_shift);

        memset(out_bmi2, 0, total);
        int i_bmi = cfg[c].bmi2(out_bmi2, codes_la, N, right_shift);
        scalar_tail(out_bmi2, codes_la, i_bmi, N, D, right_shift);

        memset(out_v1, 0, total);
        int i_v1 = cfg[c].v1(out_v1, codes_la, N, right_shift);
        scalar_tail(out_v1, codes_la, i_v1, N, D, right_shift);

        memset(out_v2, 0, total);
        int i_v2 = cfg[c].v2(out_v2, codes_la, N, right_shift);
        scalar_tail(out_v2, codes_la, i_v2, N, D, right_shift);

        memset(out_v3, 0, total);
        int i_v3 = cfg[c].v3(out_v3, codes_la, N, right_shift);
        scalar_tail(out_v3, codes_la, i_v3, N, D, right_shift);

        int e_bmi = verify(out_scalar, out_bmi2, total);
        int e_v1  = verify(out_scalar, out_v1,   total);
        int e_v2  = verify(out_scalar, out_v2,   total);
        int e_v3  = verify(out_scalar, out_v3,   total);
        const char *check =
            (e_bmi < 0 && e_v1 < 0 && e_v2 < 0 && e_v3 < 0) ? "ok" : "FAIL";

        /* Time scalar (separate loop, no fn-pointer call overhead). */
        double t0 = now_sec();
        for (int r = 0; r < 1000; r++) pack_scalar(out_scalar, codes_la, N, D, right_shift);
        double t1 = now_sec();
        double t_scalar = (t1 - t0) / ((double)N * 1000) * 1e9;

        double t_bmi = time_fn(cfg[c].bmi2, out_bmi2, codes_la, N, right_shift, D, reps);
        double t_v1  = time_fn(cfg[c].v1,   out_v1,   codes_la, N, right_shift, D, reps);
        double t_v2  = time_fn(cfg[c].v2,   out_v2,   codes_la, N, right_shift, D, reps);
        double t_v3  = time_fn(cfg[c].v3,   out_v3,   codes_la, N, right_shift, D, reps);

        printf("%-3d %6.3fns %6.3fns %6.3fns %6.3fns %6.3fns   %s\n",
               D, t_scalar, t_bmi, t_v1, t_v2, t_v3, check);
        if (e_bmi >= 0) printf("    bmi2 mismatch at byte %d\n", e_bmi);
        if (e_v1  >= 0) printf("    v1   mismatch at byte %d\n", e_v1);
        if (e_v2  >= 0) printf("    v2 mismatch at byte %d: scalar=0x%02x v2=0x%02x\n",
                                e_v2, out_scalar[e_v2], out_v2[e_v2]);
        if (e_v3  >= 0) {
            printf("    v3 mismatch at byte %d: scalar=0x%02x v3=0x%02x\n",
                   e_v3, out_scalar[e_v3], out_v3[e_v3]);
            printf("    first 12 bytes (s=scalar, v=v3):\n");
            for (int k = 0; k < 12; k++) {
                printf("      byte %2d  s=0x%02x  v=0x%02x  %s\n",
                       k, out_scalar[k], out_v3[k],
                       out_scalar[k] == out_v3[k] ? "" : "DIFF");
            }
            printf("    codes_la[0..7]: ");
            for (int k = 0; k < 8; k++)
                printf("0x%04x ", codes_la[k] & ((1u << D) - 1));
            printf("\n");
        }
    }

    free(codes_la);
    free(out_scalar); free(out_bmi2); free(out_v1); free(out_v2); free(out_v3);
    return 0;
}

#endif /* AVX-512 VBMI2 + BMI2 */
