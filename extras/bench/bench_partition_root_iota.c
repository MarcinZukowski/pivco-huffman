/* extras/bench/bench_partition_root_iota.c — A/B microbench for the root
 * partition's index-synthesis op (NEON).
 *
 * The non-root partition_8 reads its 8 source uint16 indices from
 * memory.  The *root* partition has identity indices [base..base+7],
 * which today are synthesised via vdupq_n_u16(base) + vaddq_u16(off).
 *
 * Variant tested here: replace synthesis with a vld1q_u8 from a
 * precomputed static iota table (1 SIMD op vs 2).
 *
 * Microbench result (M4 Max, 2026-04-26):
 *   partition_root        : 14.4 GB/s   (current vdup+vadd)
 *   partition_root_iota   : 15.5 GB/s   (+8%, this variant)
 *   partition_root_half        : 20.2 GB/s
 *   partition_root_half_iota   : 22.0 GB/s   (+9%)
 *
 * BUT productionising the iota variant in pivco_huffman_decode_neon
 * (commit was reverted) showed essentially **no end-to-end win**:
 * partition_root_8 fires once per block (1024×) but the decoder
 * spends most of its time in 7 deeper levels of partition_8, which
 * are unaffected.  Net change ±2% inside noise.
 *
 * Kept for posterity in case the M4-style microbench gap is ever
 * load-bearing on a different uarch (e.g. a future ARM with cheaper
 * vld1q_u8 vs vdup+vadd).  See IDEAS.md "iota-table for partition_root".
 *
 * Build:  cc -O3 -o bench_partition_root_iota \
 *           extras/bench/bench_partition_root_iota.c
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define N    8192
#define REPS 200000

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ----- compress_tab[256][32], compress_popcnt[256] (verbatim from
 *       src/pivco_huffman_neon.c::init_compress_table) ----- */

static uint8_t compress_tab[256][32] __attribute__((aligned(64)));
static uint8_t compress_popcnt[256]  __attribute__((aligned(64)));

static void init_compress_table(void)
{
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0;
        for (int i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                compress_tab[mask][out_r * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][out_r * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_r++;
            }
        }
        compress_popcnt[mask] = (uint8_t)out_r;
        for (int j = out_r * 2; j < 16; j++) compress_tab[mask][j] = 0xFF;

        int out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (!(mask & (1 << i))) {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_l * 2; j < 16; j++) compress_tab[mask][16 + j] = 0xFF;
    }
}

/* ----- Static iota table: [0,1,2,...,N-1] as uint16_t ----- */
static uint16_t static_iota_tab[N] __attribute__((aligned(64)));
static void init_static_iota(void)
{
    for (int i = 0; i < N; i++) static_iota_tab[i] = (uint16_t)i;
}

/* ============== current (vdup + vadd index synthesis) ============== */

__attribute__((noinline))
static void bench_partition_root(const uint8_t *bitmap,
                                  uint16_t *left, uint16_t *right,
                                  int n, int reps)
{
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vreinterpretq_u8_u16(
                vaddq_u16(vdupq_n_u16((uint16_t)j), vld1q_u16(off)));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            vst1q_u8((uint8_t *)(right + n_right), vqtbl1q_u8(data, shuf_r));
            vst1q_u8((uint8_t *)(left  + n_left ), vqtbl1q_u8(data, shuf_l));
            n_right += compress_popcnt[mask];
            n_left  += (8 - compress_popcnt[mask]);
        }
    }
}

__attribute__((noinline))
static void bench_partition_root_half(const uint8_t *bitmap,
                                       uint16_t *right,
                                       int n, int reps)
{
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    for (int r = 0; r < reps; r++) {
        int n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vreinterpretq_u8_u16(
                vaddq_u16(vdupq_n_u16((uint16_t)j), vld1q_u16(off)));
            uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
            vst1q_u8((uint8_t *)(right + n_right), vqtbl1q_u8(data, shuf_r));
            n_right += compress_popcnt[mask];
        }
    }
}

/* ============== iota_static (single vld1q from precomputed table) ===== */

__attribute__((noinline))
static void bench_partition_root_iota(const uint8_t *bitmap,
                                       uint16_t *left, uint16_t *right,
                                       int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(static_iota_tab + j));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            vst1q_u8((uint8_t *)(right + n_right), vqtbl1q_u8(data, shuf_r));
            vst1q_u8((uint8_t *)(left  + n_left ), vqtbl1q_u8(data, shuf_l));
            n_right += compress_popcnt[mask];
            n_left  += (8 - compress_popcnt[mask]);
        }
    }
}

__attribute__((noinline))
static void bench_partition_root_half_iota(const uint8_t *bitmap,
                                            uint16_t *right,
                                            int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(static_iota_tab + j));
            uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
            vst1q_u8((uint8_t *)(right + n_right), vqtbl1q_u8(data, shuf_r));
            n_right += compress_popcnt[mask];
        }
    }
}

/* =========================== driver ============================ */

int main(void)
{
    init_compress_table();
    init_static_iota();

    uint8_t  *bitmap = aligned_alloc(64, N / 8 + 64);
    uint16_t *left   = aligned_alloc(64, N * sizeof(uint16_t) + 64);
    uint16_t *right  = aligned_alloc(64, N * sizeof(uint16_t) + 64);
    if (!bitmap || !left || !right) { perror("alloc"); return 1; }

    srand(42);
    for (int i = 0; i < N / 8; i++) bitmap[i] = (uint8_t)rand();

    printf("== bench_partition_root_iota: index-synthesis A/B ==\n");
    printf("N = %d, REPS = %d, total = %lld elements per variant\n\n",
           N, REPS, (long long)N * REPS);

    double t0, t1, ns;

    for (int run = 0; run < 3; run++) {
        printf("-- run %d --\n", run + 1);

        t0 = now_sec();
        bench_partition_root(bitmap, left, right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  partition_root            (vdup+vadd) :  %5.2f ns/elem  (%5.1f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_partition_root_iota(bitmap, left, right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  partition_root_iota       (vld1q_u8)  :  %5.2f ns/elem  (%5.1f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_partition_root_half(bitmap, right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  partition_root_half       (vdup+vadd) :  %5.2f ns/elem  (%5.1f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_partition_root_half_iota(bitmap, right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  partition_root_half_iota  (vld1q_u8)  :  %5.2f ns/elem  (%5.1f GB/s)\n",
               ns, 1.0 / ns);
        printf("\n");
    }

    free(bitmap); free(left); free(right);
    return 0;
}
