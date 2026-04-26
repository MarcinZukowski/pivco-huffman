/* extras/bench_coalesce_avx512.c — AVX-512 store-coalescing experiment.
 *
 * Same idea as bench_coalesce.c but for the AVX-512 32-wide partition
 * (production: src/pivco_huffman_avx512.c).  Three variants:
 *
 *   baseline       : current production — vpcompressw → full 64-byte
 *                    vmovdqu64 store (most bytes overlap next iter's).
 *   compressstoreu : single-instruction _mm512_mask_compressstoreu_epi16
 *                    (writes only popcount*2 bytes, advances by the
 *                    same).  No coalescing in software — hardware
 *                    decides how many store-port cycles it costs.
 *   macro          : 2-iter macro-block coalesce.  Accumulate two
 *                    consecutive iters' compressed data into one
 *                    64-byte zmm via runtime byte-shift (vpermb),
 *                    flush always at the end.  Avg 0.5 stores/iter
 *                    vs 1 baseline.
 *
 * Build (Xeon AVX-512 VBMI2):
 *   cc -O3 -march=native -o bench_coalesce_avx512 \
 *     extras/bench_coalesce_avx512.c
 *
 * The AVX-512 production backend already exists in
 * src/pivco_huffman_avx512.c — this bench reproduces its store
 * pattern then tests two alternatives.
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define N    8192
#define REPS 50000   /* AVX-512 iters are 4× wider than NEON's, so half the reps */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* === Variant 1: baseline (production form) === */
__attribute__((noinline))
static void bench_baseline(const uint16_t *src, const uint32_t *masks,
                            uint16_t *left, uint16_t *right,
                            int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_right = 0, n_left = 0;
        for (int j = 0; j + 32 <= n; j += 32) {
            __m512i data = _mm512_loadu_si512((const __m512i *)(src + j));
            __mmask32 mask = masks[j >> 5];
            __m512i r_v = _mm512_maskz_compress_epi16(mask, data);
            _mm512_storeu_si512((__m512i *)(right + n_right), r_v);
            int nr = _mm_popcnt_u32(mask);
            n_right += nr;
            __m512i l_v = _mm512_maskz_compress_epi16(~mask, data);
            _mm512_storeu_si512((__m512i *)(left + n_left), l_v);
            n_left += 32 - nr;
        }
    }
}

/* === Variant 2: compressstoreu (single-instruction compress+store) === */
__attribute__((noinline))
static void bench_compressstoreu(const uint16_t *src, const uint32_t *masks,
                                  uint16_t *left, uint16_t *right,
                                  int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_right = 0, n_left = 0;
        for (int j = 0; j + 32 <= n; j += 32) {
            __m512i data = _mm512_loadu_si512((const __m512i *)(src + j));
            __mmask32 mask = masks[j >> 5];
            _mm512_mask_compressstoreu_epi16(right + n_right, mask, data);
            int nr = _mm_popcnt_u32(mask);
            n_right += nr;
            _mm512_mask_compressstoreu_epi16(left + n_left, ~mask, data);
            n_left += 32 - nr;
        }
    }
}

/* === Variant 3: 2-iter macro-block coalesce =================
 *
 * Process 2 consecutive 32-wide iters per macro-block.  Combined
 * popcount per side is in [0, 64], so we need a 64-byte (1 zmm)
 * accumulator per side.  Always emit 1 store per side per macro =
 * 0.5 stores/iter (vs 1/iter baseline).
 *
 * Place at byte offset `cum*2` via vpermb with a runtime control:
 *   shuf[i] = i - cum*2  (mod 256, vpermb maps >=64 → 0 with maskz)
 * On Sapphire Rapids vpermb is 3 cycles latency / 1 op throughput. */
__attribute__((noinline))
static void bench_macro(const uint16_t *src, const uint32_t *masks,
                         uint16_t *left, uint16_t *right,
                         int n, int reps)
{
    /* iota = 0..63 for vpermb shuffle base */
    static const uint8_t iota_init[64] __attribute__((aligned(64))) = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
       16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
       32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
       48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};
    const __m512i iota = _mm512_loadu_si512((const __m512i *)iota_init);
    const __m512i zero = _mm512_setzero_si512();

    for (int r = 0; r < reps; r++) {
        int n_right = 0, n_left = 0;
        for (int j = 0; j + 64 <= n; j += 64) {
            __m512i d0 = _mm512_loadu_si512((const __m512i *)(src + j));
            __m512i d1 = _mm512_loadu_si512((const __m512i *)(src + j + 32));
            __mmask32 m0 = masks[j >> 5];
            __mmask32 m1 = masks[(j >> 5) + 1];

            /* Right side */
            __m512i r0 = _mm512_maskz_compress_epi16(m0, d0);
            __m512i r1 = _mm512_maskz_compress_epi16(m1, d1);
            int pr0 = _mm_popcnt_u32(m0);
            int pr1 = _mm_popcnt_u32(m1);
            int total_r = pr0 + pr1;

            /* Place r1 at byte offset pr0*2 in a 64-byte register, OR with r0. */
            __m512i shuf_r = _mm512_sub_epi8(iota,
                _mm512_set1_epi8((char)(pr0 * 2)));
            __m512i r1_placed = _mm512_permutexvar_epi8(shuf_r, r1);
            __m512i r_acc = _mm512_or_si512(r0, r1_placed);
            _mm512_storeu_si512((__m512i *)(right + n_right), r_acc);
            n_right += total_r;

            /* Left side */
            __m512i l0 = _mm512_maskz_compress_epi16(~m0, d0);
            __m512i l1 = _mm512_maskz_compress_epi16(~m1, d1);
            int pl0 = 32 - pr0;
            int pl1 = 32 - pr1;
            int total_l = pl0 + pl1;
            __m512i shuf_l = _mm512_sub_epi8(iota,
                _mm512_set1_epi8((char)(pl0 * 2)));
            __m512i l1_placed = _mm512_permutexvar_epi8(shuf_l, l1);
            __m512i l_acc = _mm512_or_si512(l0, l1_placed);
            _mm512_storeu_si512((__m512i *)(left + n_left), l_acc);
            n_left += total_l;

            (void)zero;
        }
        /* Tail: small, ignore for the bench. */
    }
}

/* ---------- Bitmap (mask) generators ---------- */
static void fill_random_masks(uint32_t *masks, int n_masks, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < n_masks; i++) {
        masks[i] = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
    }
}

static void fill_skewed_masks(uint32_t *masks, int n_masks, unsigned seed)
{
    /* Each mask has popcount ~ 32*0.25 = 8 (heavily skewed) */
    srand(seed);
    for (int i = 0; i < n_masks; i++) {
        uint32_t v = 0;
        for (int b = 0; b < 32; b++) {
            if ((rand() & 3) == 0) v |= (1u << b);   /* 1/4 prob */
        }
        masks[i] = v;
    }
}

int main(void)
{
    /* AVX-512 needs 32-bit masks; we pack them into a uint32_t array
     * of size N/32. */
    uint16_t *src   = (uint16_t *)aligned_alloc(64, N * sizeof(uint16_t));
    uint16_t *left  = (uint16_t *)aligned_alloc(64, (size_t)(N + 64) * sizeof(uint16_t));
    uint16_t *right = (uint16_t *)aligned_alloc(64, (size_t)(N + 64) * sizeof(uint16_t));
    uint32_t *masks = (uint32_t *)aligned_alloc(64, (N / 32 + 1) * sizeof(uint32_t));
    if (!src || !left || !right || !masks) { perror("alloc"); return 1; }

    for (int i = 0; i < N; i++) src[i] = (uint16_t)i;

    printf("== bench_coalesce_avx512: store-coalescing experiments for partition_32 ==\n");
    printf("N = %d, REPS = %d, total = %lld elems per row\n\n",
           N, REPS, (long long)N * REPS);

    struct {
        const char *label;
        void (*fill)(uint32_t *, int, unsigned);
    } scenarios[] = {
        { "50% random",        fill_random_masks },
        { "skewed (popcount ~8 / 32)", fill_skewed_masks },
    };

    for (int s = 0; s < (int)(sizeof(scenarios) / sizeof(*scenarios)); s++) {
        scenarios[s].fill(masks, N / 32, 42);
        printf("-- %s --\n", scenarios[s].label);
        double t0, t1, ns;

        t0 = now_sec();
        bench_baseline(src, masks, left, right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  baseline       (vpcompressw + vmovdqu64): %5.3f ns/elem  (%5.2f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_compressstoreu(src, masks, left, right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  compressstoreu (1-instr compress+store):  %5.3f ns/elem  (%5.2f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_macro(src, masks, left, right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  macro          (2-iter coalesce):         %5.3f ns/elem  (%5.2f GB/s)\n",
               ns, 1.0 / ns);
        printf("\n");
    }

    free(src); free(left); free(right); free(masks);
    return 0;
}
