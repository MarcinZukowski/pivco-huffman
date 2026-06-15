/* bench_merge_avx2.c — merge_vec_vec primitive: current expand_tab path
 * (8-byte chunks, 2x unrolled = 16-byte outer) vs prefix-popcount path
 * (true 16-byte chunks with on-the-fly shuf control derivation).
 *
 *   - tab    : current production-style path.
 *              For each 8-byte chunk: load 8 bytes from L and R, unpacklo
 *              to concat, load expand_tab[mask] for shuf control, pshufb,
 *              storeu_si64.  Advance lc/rc by popcount(mask) / 8-popcount.
 *
 *   - prepop : 16 bytes per iter.  Read 16-bit mask, compute 16 prefix
 *              popcounts in parallel via the vpshufb-byte-popcnt +
 *              vpmaddubsw-pair-sum chain.  Pack to u8 to derive two
 *              pshufb controls (right: prefix-1, left: k-prefix).  Load
 *              16-byte windows from L and R, two pshufb gathers, vpblendvb
 *              based on mask-expanded-to-byte-mask, store 16 bytes.
 *
 * Build (c6a / c4 / c5):
 *   cc -O3 -march=native -o bench_merge_avx2 extras/bench/bench_merge_avx2.c
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if !defined(__AVX2__) || !defined(__SSE4_1__)
int main(void) { puts("bench_merge_avx2: needs SSE4.1 + AVX2"); return 0; }
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

/* ============================================================
 *   expand_tab: 256 * 8 bytes.  expand_tab[mask][k] = lane index
 *   (0..15) into the unpacklo64(L8,R8) register for output byte k.
 *   Values 0..7 select from L; 8..15 select from R. */
static uint8_t expand_tab[256][8] __attribute__((aligned(8)));
static uint8_t expand_popcnt[256];

static void init_expand_tab(void) {
    for (int m = 0; m < 256; m++) {
        int n_zeros = 0, n_ones = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) {
                expand_tab[m][k] = (uint8_t)(8 + n_ones);
                n_ones++;
            } else {
                expand_tab[m][k] = (uint8_t)n_zeros;
                n_zeros++;
            }
        }
        expand_popcnt[m] = (uint8_t)n_ones;
    }
}

/* ============================================================
 *   Scalar reference */
static void merge_scalar(uint8_t *out, int K, const uint8_t *left,
                          const uint8_t *right, const uint8_t *bm)
{
    int lc = 0, rc = 0;
    for (int j = 0; j < K; j++) {
        int bit = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = bit ? right[rc++] : left[lc++];
    }
}

/* ============================================================
 *   v_tab: current production-style x86 path.
 *   2x unrolled 8-byte chunks (= 16-byte outer iter). */
static void merge_tab(uint8_t *out, int K, const uint8_t *left,
                       const uint8_t *right, const uint8_t *bm)
{
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i L0 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(L0, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        int nr0 = expand_popcnt[m0];
        rc += nr0; lc += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i L1 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(L1, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        int nr1 = expand_popcnt[m1];
        rc += nr1; lc += (8 - nr1);
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(L, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        int nr = expand_popcnt[m];
        rc += nr; lc += (8 - nr);
    }
    for (; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left[lc++];
    }
}

/* ============================================================
 *   Per-u16-lane popcount via 4-bit nibble lookup + pmaddubsw pair-sum.
 *   16 u16 lanes (256-bit ymm). */
static inline __m256i popcnt_epi16_avx2(__m256i v)
{
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4,
        0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(v, nibble_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble_mask);
    __m256i byte_pop = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, lo),
        _mm256_shuffle_epi8(lookup, hi));
    return _mm256_maddubs_epi16(byte_pop, _mm256_set1_epi8(1));
}

/* ============================================================
 *   v_prepop: prefix-popcount merge, 16 bytes per iter.
 *   No expand_tab; pshufb controls derived on the fly. */
static void merge_prepop(uint8_t *out, int K, const uint8_t *left,
                          const uint8_t *right, const uint8_t *bm)
{
    int lc = 0, rc = 0;
    int j = 0;
    const __m256i prefix_masks_16 = _mm256_setr_epi16(
        0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF,
        0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, (short)0xFFFF);
    const __m128i ones        = _mm_set1_epi8(1);
    const __m128i indices_16  = _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    const __m128i bcast_shuf  = _mm_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1);
    const __m128i bit_pos     = _mm_setr_epi8(1,2,4,8,16,32,64,(char)128,
                                                1,2,4,8,16,32,64,(char)128);

    for (; j + 16 <= K; j += 16) {
        /* Read 16-bit mask spanning two bm bytes. */
        uint16_t m;
        memcpy(&m, bm + (j >> 3), 2);

        /* 16 parallel prefix popcounts. */
        __m256i mvec = _mm256_set1_epi16((short)m);
        __m256i masked = _mm256_and_si256(mvec, prefix_masks_16);
        __m256i prefix = popcnt_epi16_avx2(masked);

        /* Pack 16 u16 prefix values (range 0..16) to 16 u8.
         * vpackuswb on ymm splits lanes; vpermq fixes order. */
        __m256i packed = _mm256_packus_epi16(prefix, _mm256_setzero_si256());
        __m256i prefix_compact = _mm256_permute4x64_epi64(packed, 0xD8);
        __m128i prefix_u8 = _mm256_castsi256_si128(prefix_compact);

        /* Build the two pshufb controls.
         *   shuf_right[k] = prefix[k] - 1   (selects right byte at position prefix[k]-1)
         *   shuf_left[k]  = k - prefix[k]   (selects left byte at position k-prefix[k]) */
        __m128i shuf_right = _mm_sub_epi8(prefix_u8, ones);
        __m128i shuf_left  = _mm_sub_epi8(indices_16, prefix_u8);

        /* Load 16-byte source windows.  Over-reads by up to 16 bytes are
         * fine in this bench (buffers sized with slack). */
        __m128i left_data  = _mm_loadu_si128((const __m128i *)(left  + lc));
        __m128i right_data = _mm_loadu_si128((const __m128i *)(right + rc));
        __m128i g_left  = _mm_shuffle_epi8(left_data,  shuf_left);
        __m128i g_right = _mm_shuffle_epi8(right_data, shuf_right);

        /* Expand 16-bit mask to 16-byte mask (0xFF where bit set). */
        __m128i mvec_b = _mm_cvtsi32_si128((int32_t)m);
        __m128i mbcast = _mm_shuffle_epi8(mvec_b, bcast_shuf);
        __m128i mask_vec = _mm_cmpeq_epi8(_mm_and_si128(mbcast, bit_pos),
                                            bit_pos);

        __m128i out_vec = _mm_blendv_epi8(g_left, g_right, mask_vec);
        _mm_storeu_si128((__m128i *)(out + j), out_vec);

        int nr = __builtin_popcount((uint32_t)m);
        rc += nr;
        lc += 16 - nr;
    }
    /* Tail: scalar. */
    for (; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left[lc++];
    }
}

/* ============================================================
 *   Driver
 * ============================================================ */
typedef void (*merge_fn)(uint8_t *, int, const uint8_t *, const uint8_t *, const uint8_t *);

static double time_fn(merge_fn fn, uint8_t *out, int K,
                      const uint8_t *left, const uint8_t *right,
                      const uint8_t *bm, int reps)
{
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) fn(out, K, left, right, bm);
    double t1 = now_sec();
    return (t1 - t0) / ((double)K * reps) * 1e9;
}

static int verify(const uint8_t *a, const uint8_t *b, int K) {
    for (int i = 0; i < K; i++) if (a[i] != b[i]) return i;
    return -1;
}

/* Density-controlled mask generator (probability ≈ p/256 per bit). */
static void make_bm(uint8_t *bm, int K, int p) {
    int nb = (K + 7) >> 3;
    for (int i = 0; i < nb; i++) {
        uint8_t m = 0;
        for (int k = 0; k < 8; k++) {
            if ((rand() & 0xFF) < p) m |= (uint8_t)(1u << k);
        }
        bm[i] = m;
    }
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : REPS;
    init_expand_tab();
    printf("bench_merge_avx2: K=%d, REPS=%d\n\n", N, reps);

    /* Allocate with generous slack for the prefix-popcount path's 16-byte
     * over-reads (lc / rc can each advance by up to popcount in worst
     * cases; we use N + 64 bytes of slack on each source buffer). */
    uint8_t *left  = aligned_alloc(64, N + 128);
    uint8_t *right = aligned_alloc(64, N + 128);
    uint8_t *bm    = aligned_alloc(64, N / 8 + 32);
    uint8_t *out_ref  = aligned_alloc(64, N + 128);
    uint8_t *out_tab  = aligned_alloc(64, N + 128);
    uint8_t *out_pp   = aligned_alloc(64, N + 128);

    srand(42);
    for (int i = 0; i < N + 128; i++) left[i]  = (uint8_t)('a' + (rand() & 0x1F));
    for (int i = 0; i < N + 128; i++) right[i] = (uint8_t)('A' + (rand() & 0x1F));

    int densities[] = {32, 64, 128, 192, 224};   /* p/256 */
    const char *dlbl[] = {"12%", "25%", "50%", "75%", "87%"};
    printf("%-8s %10s %10s %10s   %s\n",
           "density", "scalar", "tab", "prepop", "check");
    for (size_t d = 0; d < sizeof(densities)/sizeof(densities[0]); d++) {
        make_bm(bm, N, densities[d]);

        merge_scalar(out_ref, N, left, right, bm);

        memset(out_tab, 0, N);
        merge_tab(out_tab, N, left, right, bm);
        int e_tab = verify(out_ref, out_tab, N);

        memset(out_pp, 0, N);
        merge_prepop(out_pp, N, left, right, bm);
        int e_pp = verify(out_ref, out_pp, N);

        const char *chk = (e_tab < 0 && e_pp < 0) ? "ok" : "FAIL";

        double t0 = now_sec();
        for (int r = 0; r < 1000; r++) merge_scalar(out_ref, N, left, right, bm);
        double t1 = now_sec();
        double ts = (t1 - t0) / ((double)N * 1000) * 1e9;

        double t_tab = time_fn(merge_tab,    out_tab, N, left, right, bm, reps);
        double t_pp  = time_fn(merge_prepop, out_pp,  N, left, right, bm, reps);

        printf("%-8s %8.3fns %8.3fns %8.3fns   %s\n",
               dlbl[d], ts, t_tab, t_pp, chk);
        if (e_tab >= 0) printf("    tab    mismatch at %d (s=0x%02x t=0x%02x)\n",
                                 e_tab, out_ref[e_tab], out_tab[e_tab]);
        if (e_pp  >= 0) printf("    prepop mismatch at %d (s=0x%02x p=0x%02x)\n",
                                 e_pp,  out_ref[e_pp],  out_pp[e_pp]);
    }

    free(left); free(right); free(bm);
    free(out_ref); free(out_tab); free(out_pp);
    return 0;
}

#endif /* AVX2 + SSE4.1 */
