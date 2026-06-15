/* bench_partition_unroll.c — partition primitive: current 1x 8-element
 * vs 2x-unrolled (16-element outer) variant.
 *
 *   - v1x : current production-style path.  8 codes per iter via SSE
 *           pshufb on compress_tab[mask].
 *   - v2x : same compress_tab + pshufb skeleton, but two 8-code chunks
 *           per outer iter with all per-chunk deps independent until
 *           the final cursor math.  More work in flight per cycle.
 *
 * Wire format unchanged: bm[] same byte-for-byte layout, right_out and
 * codes_la (in-place left) layouts identical to the 1x version.  The
 * 2x version just writes 2 of each per outer iter, with the second
 * store of each pair overlapping & overwriting the trailing junk from
 * the first.
 *
 * Build (c6a / c4 / c5):
 *   cc -O3 -march=native -o bench_partition_unroll extras/bench/bench_partition_unroll.c
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if !defined(__SSE4_1__)
int main(void) { puts("bench_partition_unroll: needs SSE4.1"); return 0; }
#else

#include <immintrin.h>
#include <smmintrin.h>

#define N    8192
#define REPS 100000

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* compress_tab: 256 * 32 bytes.  Low 16 = right-pack indices, high 16 =
 * left-pack indices.  Trailing junk lanes are 0xFF so pshufb scribbles
 * garbage past the popcount, which the next iter overwrites. */
static uint8_t compress_tab[256][32] __attribute__((aligned(32)));
static uint8_t compress_popcnt[256];

static void init_compress_tab(void) {
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0, out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                compress_tab[mask][out_r * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][out_r * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_r++;
            } else {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_r * 2; j < 16; j++)
            compress_tab[mask][j] = 0xFF;
        for (int j = out_l * 2; j < 16; j++)
            compress_tab[mask][16 + j] = 0xFF;
        compress_popcnt[mask] = (uint8_t)out_r;
    }
}

/* Scalar reference. */
static int partition_scalar(uint16_t *codes_la, int n, int depth,
                             uint8_t *bm, uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int shift_d = 15 - depth;
    uint16_t *buf = malloc((size_t)n * sizeof(uint16_t));
    memcpy(buf, codes_la, (size_t)n * sizeof(uint16_t));
    for (int j = 0; j < n; j += 8) {
        int end = j + 8 < n ? j + 8 : n;
        uint8_t mask = 0;
        for (int k = j; k < end; k++) {
            int bit = (buf[k] >> shift_d) & 1;
            mask |= (uint8_t)(bit << (k - j));
        }
        bm[j >> 3] = mask;
        for (int k = j; k < end; k++) {
            int bit = (buf[k] >> shift_d) & 1;
            if (bit) right_out[n_right++] = buf[k];
            else     codes_la[n_left++]  = buf[k];
        }
    }
    free(buf);
    return n_right;
}

/* v1x: production-style 8-element compress_tab partition. */
static int partition_1x(uint16_t *codes_la, int n, int depth,
                          uint8_t *bm, uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 8 <= n; j += 8) {
        __m128i code = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted = _mm_sll_epi16(code, shift_count);
        __m128i packed  = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask    = (uint8_t)_mm_movemask_epi8(packed);
        bm[j >> 3] = mask;
        const uint8_t *tab = compress_tab[mask];
        __m128i sr = _mm_load_si128((const __m128i *)tab);
        __m128i sl = _mm_load_si128((const __m128i *)(tab + 16));
        __m128i r  = _mm_shuffle_epi8(code, sr);
        __m128i l  = _mm_shuffle_epi8(code, sl);
        int nr = compress_popcnt[mask];
        _mm_storeu_si128((__m128i *)(right_out + n_right), r);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left ), l);
        n_right += nr;
        n_left  += (8 - nr);
    }
    return n_right;
}

/* v2x: 2x-unrolled compress_tab partition.  Two 8-code chunks per outer
 * iter with all per-chunk dependencies fully independent until the
 * cursor math at the end. */
static int partition_2x(uint16_t *codes_la, int n, int depth,
                          uint8_t *bm, uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 16 <= n; j += 16) {
        __m128i code0 = _mm_loadu_si128((const __m128i *)(codes_la + j    ));
        __m128i code1 = _mm_loadu_si128((const __m128i *)(codes_la + j + 8));

        __m128i shifted0 = _mm_sll_epi16(code0, shift_count);
        __m128i shifted1 = _mm_sll_epi16(code1, shift_count);
        __m128i packed0  = _mm_packs_epi16(shifted0, _mm_setzero_si128());
        __m128i packed1  = _mm_packs_epi16(shifted1, _mm_setzero_si128());
        uint8_t m0 = (uint8_t)_mm_movemask_epi8(packed0);
        uint8_t m1 = (uint8_t)_mm_movemask_epi8(packed1);
        bm[j >> 3]       = m0;
        bm[(j >> 3) + 1] = m1;

        const uint8_t *tab0 = compress_tab[m0];
        const uint8_t *tab1 = compress_tab[m1];
        __m128i sr0 = _mm_load_si128((const __m128i *)tab0);
        __m128i sl0 = _mm_load_si128((const __m128i *)(tab0 + 16));
        __m128i sr1 = _mm_load_si128((const __m128i *)tab1);
        __m128i sl1 = _mm_load_si128((const __m128i *)(tab1 + 16));

        __m128i r0 = _mm_shuffle_epi8(code0, sr0);
        __m128i l0 = _mm_shuffle_epi8(code0, sl0);
        __m128i r1 = _mm_shuffle_epi8(code1, sr1);
        __m128i l1 = _mm_shuffle_epi8(code1, sl1);

        int nr0 = compress_popcnt[m0];
        int nr1 = compress_popcnt[m1];

        /* The second store of each pair overlaps the first and overwrites
         * its trailing junk with chunk 1's valid prefix.  Safe because
         * trailing junk lies past the popcount of chunk 0's valid count. */
        _mm_storeu_si128((__m128i *)(right_out + n_right),       r0);
        _mm_storeu_si128((__m128i *)(right_out + n_right + nr0), r1);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left),        l0);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left + (8 - nr0)), l1);

        n_right += nr0 + nr1;
        n_left  += (8 - nr0) + (8 - nr1);
    }
    /* Residual 8-element step. */
    if (j + 8 <= n) {
        __m128i code = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted = _mm_sll_epi16(code, shift_count);
        __m128i packed  = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask    = (uint8_t)_mm_movemask_epi8(packed);
        bm[j >> 3] = mask;
        const uint8_t *tab = compress_tab[mask];
        __m128i sr = _mm_load_si128((const __m128i *)tab);
        __m128i sl = _mm_load_si128((const __m128i *)(tab + 16));
        __m128i r  = _mm_shuffle_epi8(code, sr);
        __m128i l  = _mm_shuffle_epi8(code, sl);
        int nr = compress_popcnt[mask];
        _mm_storeu_si128((__m128i *)(right_out + n_right), r);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left ), l);
        n_right += nr;
        n_left  += (8 - nr);
    }
    return n_right;
}

/* v2y: same 2x-unrolled shape as v2x but the mask gen + load is fused
 * into one ymm chain.  Requires AVX2. */
#ifdef __AVX2__
static int partition_2y(uint16_t *codes_la, int n, int depth,
                          uint8_t *bm, uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 16 <= n; j += 16) {
        __m256i code01 = _mm256_loadu_si256((const __m256i *)(codes_la + j));
        __m256i shifted = _mm256_sll_epi16(code01, shift_count);
        __m256i packed  = _mm256_packs_epi16(shifted, _mm256_setzero_si256());
        uint32_t mfull = (uint32_t)_mm256_movemask_epi8(packed);
        uint8_t m0 = (uint8_t)(mfull        & 0xFF);
        uint8_t m1 = (uint8_t)((mfull >> 16) & 0xFF);
        bm[j >> 3]       = m0;
        bm[(j >> 3) + 1] = m1;

        __m128i code0 = _mm256_castsi256_si128(code01);
        __m128i code1 = _mm256_extracti128_si256(code01, 1);

        const uint8_t *tab0 = compress_tab[m0];
        const uint8_t *tab1 = compress_tab[m1];
        __m128i sr0 = _mm_load_si128((const __m128i *)tab0);
        __m128i sl0 = _mm_load_si128((const __m128i *)(tab0 + 16));
        __m128i sr1 = _mm_load_si128((const __m128i *)tab1);
        __m128i sl1 = _mm_load_si128((const __m128i *)(tab1 + 16));

        __m128i r0 = _mm_shuffle_epi8(code0, sr0);
        __m128i l0 = _mm_shuffle_epi8(code0, sl0);
        __m128i r1 = _mm_shuffle_epi8(code1, sr1);
        __m128i l1 = _mm_shuffle_epi8(code1, sl1);

        int nr0 = compress_popcnt[m0];
        int nr1 = compress_popcnt[m1];

        _mm_storeu_si128((__m128i *)(right_out + n_right),       r0);
        _mm_storeu_si128((__m128i *)(right_out + n_right + nr0), r1);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left),        l0);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left + (8 - nr0)), l1);

        n_right += nr0 + nr1;
        n_left  += (8 - nr0) + (8 - nr1);
    }
    if (j + 8 <= n) {
        __m128i code = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted = _mm_sll_epi16(code, shift_count);
        __m128i packed  = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask    = (uint8_t)_mm_movemask_epi8(packed);
        bm[j >> 3] = mask;
        const uint8_t *tab = compress_tab[mask];
        __m128i sr = _mm_load_si128((const __m128i *)tab);
        __m128i sl = _mm_load_si128((const __m128i *)(tab + 16));
        __m128i r  = _mm_shuffle_epi8(code, sr);
        __m128i l  = _mm_shuffle_epi8(code, sl);
        int nr = compress_popcnt[mask];
        _mm_storeu_si128((__m128i *)(right_out + n_right), r);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left ), l);
        n_right += nr;
        n_left  += (8 - nr);
    }
    return n_right;
}
#endif

/* Driver. */
typedef int (*part_fn)(uint16_t *, int, int, uint8_t *, uint16_t *);

static double time_fn(part_fn fn, uint16_t *scratch, const uint16_t *orig,
                      int n, int depth, uint8_t *bm, uint16_t *right_out,
                      int reps)
{
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        memcpy(scratch, orig, (size_t)n * sizeof(uint16_t));
        (void)fn(scratch, n, depth, bm, right_out);
    }
    double t1 = now_sec();
    return (t1 - t0) / ((double)n * reps) * 1e9;
}

static int verify_partition(uint16_t *a_left, uint16_t *a_right, int a_nr,
                              uint16_t *b_left, uint16_t *b_right, int b_nr,
                              int n)
{
    if (a_nr != b_nr) return -1;
    int nl = n - a_nr;
    for (int i = 0; i < nl;   i++) if (a_left[i]  != b_left[i])  return i;
    for (int i = 0; i < a_nr; i++) if (a_right[i] != b_right[i]) return 1000 + i;
    return -1;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : REPS;
    init_compress_tab();
    printf("bench_partition_unroll: N=%d, REPS=%d\n\n", N, reps);

    size_t sz_codes = ((N + 64) * sizeof(uint16_t) + 63) / 64 * 64;
    size_t sz_bm    = (N / 8 + 32 + 63) / 64 * 64;
    uint16_t *orig      = aligned_alloc(64, sz_codes);
    uint16_t *scratch   = aligned_alloc(64, sz_codes);
    uint16_t *right_out = aligned_alloc(64, sz_codes);
    uint16_t *right_ref = aligned_alloc(64, sz_codes);
    uint16_t *left_ref  = aligned_alloc(64, sz_codes);
    uint8_t  *bm_ref    = aligned_alloc(64, sz_bm);
    uint8_t  *bm        = aligned_alloc(64, sz_bm);

    srand(42);
    for (int i = 0; i < N; i++) orig[i] = (uint16_t)rand();

    int depths[] = {0, 1, 2, 4, 8, 12};
    printf("%-6s %10s %10s %10s %10s   %s\n",
           "depth", "scalar", "v1x", "v2x", "v2y", "check");
    for (size_t d = 0; d < sizeof(depths)/sizeof(depths[0]); d++) {
        int depth = depths[d];

        memcpy(left_ref, orig, (size_t)N * sizeof(uint16_t));
        int nr_ref = partition_scalar(left_ref, N, depth, bm_ref, right_ref);

        memcpy(scratch, orig, (size_t)N * sizeof(uint16_t));
        int nr_1x = partition_1x(scratch, N, depth, bm, right_out);
        int e_1x = verify_partition(left_ref, right_ref, nr_ref,
                                     scratch, right_out, nr_1x, N);

        memcpy(scratch, orig, (size_t)N * sizeof(uint16_t));
        int nr_2x = partition_2x(scratch, N, depth, bm, right_out);
        int e_2x = verify_partition(left_ref, right_ref, nr_ref,
                                     scratch, right_out, nr_2x, N);

#ifdef __AVX2__
        memcpy(scratch, orig, (size_t)N * sizeof(uint16_t));
        int nr_2y = partition_2y(scratch, N, depth, bm, right_out);
        int e_2y = verify_partition(left_ref, right_ref, nr_ref,
                                     scratch, right_out, nr_2y, N);
#else
        int e_2y = -1;
#endif

        const char *chk = (e_1x < 0 && e_2x < 0 && e_2y < 0) ? "ok" : "FAIL";

        double t0 = now_sec();
        for (int r = 0; r < 500; r++) {
            memcpy(scratch, orig, (size_t)N * sizeof(uint16_t));
            (void)partition_scalar(scratch, N, depth, bm, right_out);
        }
        double ts = (now_sec() - t0) / ((double)N * 500) * 1e9;

        double t_1x = time_fn(partition_1x, scratch, orig, N, depth, bm, right_out, reps);
        double t_2x = time_fn(partition_2x, scratch, orig, N, depth, bm, right_out, reps);
#ifdef __AVX2__
        double t_2y = time_fn(partition_2y, scratch, orig, N, depth, bm, right_out, reps);
#else
        double t_2y = 0.0;
#endif

        printf("%-6d %8.3fns %8.3fns %8.3fns %8.3fns   %s\n",
               depth, ts, t_1x, t_2x, t_2y, chk);
        if (e_1x >= 0) printf("    1x mismatch at %d (nr_ref=%d, nr=%d)\n",
                                e_1x, nr_ref, nr_1x);
        if (e_2x >= 0) printf("    2x mismatch at %d (nr_ref=%d, nr=%d)\n",
                                e_2x, nr_ref, nr_2x);
        if (e_2y >= 0) printf("    2y mismatch at %d\n", e_2y);
    }

    free(orig); free(scratch); free(right_out); free(right_ref); free(left_ref);
    free(bm_ref); free(bm);
    return 0;
}

#endif /* SSE4.1 */
