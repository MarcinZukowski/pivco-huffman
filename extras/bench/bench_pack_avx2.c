/* bench_pack_avx2.c — AVX2 port of ryg's multiply-as-shift pack, D=2..7.
 *
 * Compares four implementations on AVX2-only hosts (c4/c5/c6a):
 *   - scalar     : reference
 *   - bmi2       : _pext_u64 (Intel-fast / pre-Zen4 AMD-slow)
 *   - vec_v1     : current x86 backend path (sllv + reduce_add, 8 codes/iter,
 *                  256-bit u64 lanes; D=5..7 only — D=2/3/4/8 use SSE forms)
 *   - vec_v3     : new ryg AVX2 port — pmaddubsw + pmaddwd + and/andn/or
 *                  + per-lane vpshufb compact, 32 codes/iter
 *
 * Build:
 *   cc -O3 -march=native -o bench_pack_avx2 extras/bench/bench_pack_avx2.c
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if !defined(__AVX2__) || !defined(__BMI2__)
int main(void) { puts("bench_pack_avx2: needs AVX2 + BMI2"); return 0; }
#else

#include <immintrin.h>

#define N    8192
#define REPS 100000

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
 *   BMI2 pext path
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
 *   v1: current production AVX2 x86 path
 *   8 codes/iter via __m256i u64 lanes (4 lanes lo/hi halves).
 * ============================================================ */
#define PACK_DN_VEC_V1(NAME, D_VAL, BITS_OUT)                                   \
static int NAME(uint8_t *out, const uint16_t *codes_la,                         \
                int n, int right_shift)                                          \
{                                                                                \
    static const int64_t shifts_lo[4] = {0,         D_VAL,   2*D_VAL, 3*D_VAL}; \
    static const int64_t shifts_hi[4] = {4*D_VAL, 5*D_VAL, 6*D_VAL, 7*D_VAL};   \
    __m256i sl = _mm256_loadu_si256((const __m256i *)shifts_lo);                \
    __m256i sh = _mm256_loadu_si256((const __m256i *)shifts_hi);                \
    __m256i mv = _mm256_set1_epi64x((1LL << D_VAL) - 1);                        \
    int i = 0;                                                                   \
    for (; i + 8 <= n; i += 8) {                                                \
        __m128i v16 = _mm_loadu_si128((const __m128i *)(codes_la + i));         \
        __m256i lo = _mm256_cvtepu16_epi64(v16);                                \
        __m256i hi = _mm256_cvtepu16_epi64(_mm_unpackhi_epi64(v16, v16));       \
        lo = _mm256_and_si256(_mm256_srli_epi64(lo, right_shift), mv);          \
        hi = _mm256_and_si256(_mm256_srli_epi64(hi, right_shift), mv);          \
        lo = _mm256_sllv_epi64(lo, sl);                                         \
        hi = _mm256_sllv_epi64(hi, sh);                                         \
        __m256i sum = _mm256_add_epi64(lo, hi);                                 \
        __m128i s128 = _mm_add_epi64(_mm256_castsi256_si128(sum),               \
                                      _mm256_extracti128_si256(sum, 1));        \
        uint64_t packed = _mm_cvtsi128_si64(s128)                               \
                        + _mm_cvtsi128_si64(_mm_unpackhi_epi64(s128, s128));    \
        int bi = i * D_VAL / 8;                                                 \
        memcpy(out + bi, &packed, (BITS_OUT + 7) / 8);                          \
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
 *   SSE-specialized helpers (copied verbatim from
 *   src/pivco_huffman_primitives_x86.h) — used by the production
 *   x86 dispatcher for D=2/3/4 (D=8 too, but we skip it here).
 *   These often beat the AVX2 sllv+reduce path because the byte-
 *   aligned (D=2/4) or 3-byte-output (D=3) layouts are too small
 *   for the wider lane to amortize. */
static int pack_d2_sse(uint8_t *out, const uint16_t *codes_la,
                        int n, int right_shift)
{
    const __m128i weights = _mm_setr_epi8(1, 4, 16, 64, 1, 4, 16, 64,
                                           1, 4, 16, 64, 1, 4, 16, 64);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(codes_la + i    ));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(codes_la + i + 8));
        v0 = _mm_srli_epi16(v0, right_shift);
        v1 = _mm_srli_epi16(v1, right_shift);
        v0 = _mm_and_si128(v0, _mm_set1_epi16(0x3));
        v1 = _mm_and_si128(v1, _mm_set1_epi16(0x3));
        __m128i bytes = _mm_packus_epi16(v0, v1);
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        __m128i step2 = _mm_hadd_epi16(step1, _mm_setzero_si128());
        __m128i out_bytes = _mm_packus_epi16(step2, _mm_setzero_si128());
        uint32_t packed4 = (uint32_t)_mm_cvtsi128_si32(out_bytes);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

static int pack_d3_sse(uint8_t *out, const uint16_t *codes_la,
                        int n, int right_shift)
{
    const __m128i mlo = _mm_setr_epi32(1, 8, 64, 512);
    const __m128i mhi = _mm_setr_epi32(4096, 32768, 262144, 2097152);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128((const __m128i *)(codes_la + i));
        v = _mm_srli_epi16(v, right_shift);
        v = _mm_and_si128(v, _mm_set1_epi16(0x7));
        __m128i vlo = _mm_unpacklo_epi16(v, _mm_setzero_si128());
        __m128i vhi = _mm_unpackhi_epi16(v, _mm_setzero_si128());
        vlo = _mm_mullo_epi32(vlo, mlo);
        vhi = _mm_mullo_epi32(vhi, mhi);
        __m128i s = _mm_add_epi32(vlo, vhi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        uint32_t packed = (uint32_t)_mm_cvtsi128_si32(s);
        int bi = i * 3 / 8;
        out[bi    ] = (uint8_t)(packed       );
        out[bi + 1] = (uint8_t)(packed >>  8);
        out[bi + 2] = (uint8_t)(packed >> 16);
    }
    return i;
}

static int pack_d4_sse(uint8_t *out, const uint16_t *codes_la,
                        int n, int right_shift)
{
    const __m128i weights = _mm_setr_epi8(1, 16, 1, 16, 1, 16, 1, 16,
                                           1, 16, 1, 16, 1, 16, 1, 16);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(codes_la + i    ));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(codes_la + i + 8));
        v0 = _mm_srli_epi16(v0, right_shift);
        v1 = _mm_srli_epi16(v1, right_shift);
        v0 = _mm_and_si128(v0, _mm_set1_epi16(0xF));
        v1 = _mm_and_si128(v1, _mm_set1_epi16(0xF));
        __m128i bytes = _mm_packus_epi16(v0, v1);
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        __m128i out_bytes = _mm_packus_epi16(step1, _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)(out + (i * 4 / 8)), out_bytes);
    }
    return i;
}

/* ============================================================
 *   v3: ryg's multiply-as-shift pack, AVX2 port (32 codes/iter)
 *
 * Same algebra as the AVX-512 v3:
 *   pmaddubsw -> pmaddwd -> psrlq + (a&c)|(b&~c) ternlog-equiv
 *   -> vpshufb compact per 128-bit lane -> 2x movdqu stores.
 *
 * No AVX-512 instructions used.  ternlog -> 3 ops (vpand, vpandn, vpor).
 * vpcompressb -> per-lane vpshufb with constant table.
 * ============================================================ */
static inline __m256i load_codes_byte_avx2(const uint16_t *codes_la,
                                            int right_shift,
                                            uint8_t code_mask)
{
    /* Load 32 u16 codes, right-shift, narrow to 32 bytes. */
    __m256i a = _mm256_loadu_si256((const __m256i *)(codes_la));
    __m256i b = _mm256_loadu_si256((const __m256i *)(codes_la + 16));
    a = _mm256_srli_epi16(a, right_shift);
    b = _mm256_srli_epi16(b, right_shift);
    /* vpackuswb SATURATES (u16 >255 -> 255).  Clear high byte first so
     * the pack becomes equivalent to truncation. */
    const __m256i lo_mask = _mm256_set1_epi16(0x00FF);
    a = _mm256_and_si256(a, lo_mask);
    b = _mm256_and_si256(b, lo_mask);
    __m256i packed = _mm256_packus_epi16(a, b);
    __m256i bytes  = _mm256_permute4x64_epi64(packed, 0xD8);
    return _mm256_and_si256(bytes, _mm256_set1_epi8((char)code_mask));
}

/* Compact-shuf tables (per 128-bit lane) — bytes [0..D-1] from low qword,
 * bytes [D..2D-1] from high qword, rest don't-care. */
#define COMPACT_AVX2(D_VAL, ...)                                                \
    _mm256_setr_epi8(__VA_ARGS__)

#define PACK_V3_AVX2(NAME, D_VAL, COMPACT_SHUF)                                 \
static int NAME(uint8_t *out, const uint16_t *codes_la,                         \
                int n, int right_shift)                                          \
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
        __m256i cb = load_codes_byte_avx2(codes_la + i, right_shift,             \
                                           (uint8_t)((1 << (D_VAL)) - 1));       \
        __m256i x  = _mm256_maddubs_epi16(c0, cb);                               \
        x = _mm256_madd_epi16(x, c1);                                            \
        __m256i xs = _mm256_srli_epi64(x, 32 - 4*(D_VAL));                       \
        x = _mm256_or_si256(_mm256_and_si256(x, c3),                             \
                             _mm256_andnot_si256(c3, xs));                       \
        /* Compact: per 128-bit lane, byte 0..D-1 from qword 0, D..2D-1 from qword 1 */ \
        __m256i out_y = _mm256_shuffle_epi8(x, compact);                         \
        /* Store low + high 128-bit halves at consecutive offsets.               \
         * Each 128-bit store writes 16 bytes; only first 2D are valid.          \
         * Next chunk's lo store at (i*D/8) + 4D overwrites the trailing junk. */\
        int base = (i * (D_VAL)) >> 3;                                           \
        _mm_storeu_si128((__m128i *)(out + base),                                \
                          _mm256_castsi256_si128(out_y));                        \
        _mm_storeu_si128((__m128i *)(out + base + 2*(D_VAL)),                    \
                          _mm256_extracti128_si256(out_y, 1));                   \
    }                                                                            \
    return i;                                                                    \
}

/* Per-D compact tables.  Pattern per 128-bit lane: bytes 0..D-1 from
 * positions 0..D-1, bytes D..2D-1 from positions 8..8+D-1, junk at the rest.
 * Replicated identically in both 128-bit halves of the ymm shuf. */
#define CSH_D2  COMPACT_AVX2(2,                                                  \
    0, 1, 8, 9, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,                            \
    0, 1, 8, 9, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1)
#define CSH_D3  COMPACT_AVX2(3,                                                  \
    0, 1, 2, 8, 9,10, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1,                            \
    0, 1, 2, 8, 9,10, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1)
#define CSH_D4  COMPACT_AVX2(4,                                                  \
    0, 1, 2, 3, 8, 9,10,11, -1,-1,-1,-1, -1,-1,-1,-1,                             \
    0, 1, 2, 3, 8, 9,10,11, -1,-1,-1,-1, -1,-1,-1,-1)
#define CSH_D5  COMPACT_AVX2(5,                                                  \
    0, 1, 2, 3, 4, 8, 9,10, 11,12, -1,-1, -1,-1,-1,-1,                            \
    0, 1, 2, 3, 4, 8, 9,10, 11,12, -1,-1, -1,-1,-1,-1)
#define CSH_D6  COMPACT_AVX2(6,                                                  \
    0, 1, 2, 3, 4, 5, 8, 9, 10,11,12,13, -1,-1,-1,-1,                             \
    0, 1, 2, 3, 4, 5, 8, 9, 10,11,12,13, -1,-1,-1,-1)
#define CSH_D7  COMPACT_AVX2(7,                                                  \
    0, 1, 2, 3, 4, 5, 6, 8,  9,10,11,12,13,14, -1,-1,                             \
    0, 1, 2, 3, 4, 5, 6, 8,  9,10,11,12,13,14, -1,-1)

PACK_V3_AVX2(pack_v3_d2, 2, CSH_D2)
PACK_V3_AVX2(pack_v3_d3, 3, CSH_D3)
PACK_V3_AVX2(pack_v3_d4, 4, CSH_D4)
PACK_V3_AVX2(pack_v3_d5, 5, CSH_D5)
PACK_V3_AVX2(pack_v3_d6, 6, CSH_D6)
PACK_V3_AVX2(pack_v3_d7, 7, CSH_D7)

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
                      int n, int right_shift, int D, int reps) {
    int total = (n * D + 7) >> 3;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        memset(out, 0, total);
        int i = fn(out, codes_la, n, right_shift);
        scalar_tail(out, codes_la, i, n, D, right_shift);
    }
    double t1 = now_sec();
    return (t1 - t0) / ((double)n * reps) * 1e9;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : REPS;
    printf("bench_pack_avx2: N=%d, REPS=%d\n\n", N, reps);

    uint16_t *codes_la = aligned_alloc(64, N * 2 + 128);
    uint8_t *out_scalar = aligned_alloc(64, N + 128);
    uint8_t *out_bmi2   = aligned_alloc(64, N + 128);
    uint8_t *out_v1     = aligned_alloc(64, N + 128);
    uint8_t *out_v3     = aligned_alloc(64, N + 128);
    uint8_t *out_sse    = aligned_alloc(64, N + 128);

    srand(42);
    for (int i = 0; i < N; i++) codes_la[i] = (uint16_t)rand();
    int right_shift = 0;

    struct { int D; pack_fn bmi2; pack_fn v1; pack_fn v3; pack_fn sse; } cfg[] = {
        {2, pack_bmi2_d2, pack_v1_d2, pack_v3_d2, pack_d2_sse},
        {3, pack_bmi2_d3, pack_v1_d3, pack_v3_d3, pack_d3_sse},
        {4, pack_bmi2_d4, pack_v1_d4, pack_v3_d4, pack_d4_sse},
        {5, pack_bmi2_d5, pack_v1_d5, pack_v3_d5, NULL},
        {6, pack_bmi2_d6, pack_v1_d6, pack_v3_d6, NULL},
        {7, pack_bmi2_d7, pack_v1_d7, pack_v3_d7, NULL},
    };

    printf("%-3s %8s %8s %8s %8s %8s   %s\n",
           "D", "scalar", "bmi2", "vec_v1", "vec_v3", "sse_old", "check");
    for (size_t c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
        int D = cfg[c].D;
        int total = (N * D + 7) >> 3;

        pack_scalar(out_scalar, codes_la, N, D, right_shift);

        memset(out_bmi2, 0, total);
        int ib = cfg[c].bmi2(out_bmi2, codes_la, N, right_shift);
        scalar_tail(out_bmi2, codes_la, ib, N, D, right_shift);

        memset(out_v1, 0, total);
        int i1 = cfg[c].v1(out_v1, codes_la, N, right_shift);
        scalar_tail(out_v1, codes_la, i1, N, D, right_shift);

        memset(out_v3, 0, total);
        int i3 = cfg[c].v3(out_v3, codes_la, N, right_shift);
        scalar_tail(out_v3, codes_la, i3, N, D, right_shift);

        int es = 0;
        if (cfg[c].sse) {
            memset(out_sse, 0, total);
            int isse = cfg[c].sse(out_sse, codes_la, N, right_shift);
            scalar_tail(out_sse, codes_la, isse, N, D, right_shift);
            es = verify(out_scalar, out_sse, total);
        } else {
            es = -1;
        }

        int eb = verify(out_scalar, out_bmi2, total);
        int e1 = verify(out_scalar, out_v1,   total);
        int e3 = verify(out_scalar, out_v3,   total);
        const char *chk = (eb<0 && e1<0 && e3<0 && es<0) ? "ok" : "FAIL";

        double t0 = now_sec();
        for (int r = 0; r < 1000; r++) pack_scalar(out_scalar, codes_la, N, D, right_shift);
        double ts = (now_sec() - t0) / ((double)N * 1000) * 1e9;

        double tb = time_fn(cfg[c].bmi2, out_bmi2, codes_la, N, right_shift, D, reps);
        double t1 = time_fn(cfg[c].v1,   out_v1,   codes_la, N, right_shift, D, reps);
        double t3 = time_fn(cfg[c].v3,   out_v3,   codes_la, N, right_shift, D, reps);
        double tsse = cfg[c].sse
            ? time_fn(cfg[c].sse, out_sse, codes_la, N, right_shift, D, reps) : 0.0;

        if (cfg[c].sse)
            printf("%-3d %6.3fns %6.3fns %6.3fns %6.3fns %6.3fns   %s\n",
                   D, ts, tb, t1, t3, tsse, chk);
        else
            printf("%-3d %6.3fns %6.3fns %6.3fns %6.3fns %8s   %s\n",
                   D, ts, tb, t1, t3, "—", chk);
        if (eb >= 0) printf("    bmi2 mismatch at %d\n", eb);
        if (e1 >= 0) printf("    v1   mismatch at %d\n", e1);
        if (e3 >= 0) printf("    v3 mismatch at byte %d: scalar=0x%02x v3=0x%02x\n",
                            e3, out_scalar[e3], out_v3[e3]);
        if (cfg[c].sse && es >= 0)
            printf("    sse  mismatch at %d\n", es);
    }

    free(codes_la);
    free(out_scalar); free(out_bmi2); free(out_v1); free(out_v3); free(out_sse);
    return 0;
}

#endif /* AVX2 + BMI2 */
