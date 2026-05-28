/* extras/bench/bench_unpack_dN.c — pure D-bit unpack throughput bench.
 *
 * For each D in {2,3,4,5,6}, measure GB/s of OUTPUT bytes produced when
 * unpacking N*(D bits) of packed input into N*(8 bits) of unpacked
 * output (one byte per code, value < 2^D, no c2s lookup).
 *
 * Variants per D:
 *   flat_dX : current flat_dX_unpack + sequential vst1q (or vst1 for
 *             8-lane unpacks), as used in production.
 *   fl_dX   : FastLanes-style — only naturally available when D divides 8
 *             (D=2 and D=4).  Single load of 16 input bytes, K = 8/D
 *             groups produced via shr+and, output via vstKq_u8 interleave.
 *
 * Build:
 *   cc -O3 -o bench_unpack_dN extras/bench/bench_unpack_dN.c
 *
 * Reports GB/s of OUTPUT bytes (1 byte per code).
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "../../src/pivco_huffman_neon_flat.h"

#define N    8192
#define REPS 200000

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============================== D=2 ================================== */

__attribute__((noinline))
static void unpack_d2_flat(uint8_t *out, const uint8_t *bm, int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d2_unpack(bm + (i >> 2));
            vst1q_u8(out + i, codes);
        }
    }
}

__attribute__((noinline))
static void unpack_d2_fl(uint8_t *out, const uint8_t *bm, int n, int reps)
{
    uint8x16_t mask3 = vdupq_n_u8(0x03);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 64 <= n; i += 64) {
            uint8x16_t reg = vld1q_u8(bm + (i >> 2));
            uint8x16_t g0 = vandq_u8(reg, mask3);
            uint8x16_t g1 = vandq_u8(vshrq_n_u8(reg, 2), mask3);
            uint8x16_t g2 = vandq_u8(vshrq_n_u8(reg, 4), mask3);
            uint8x16_t g3 = vandq_u8(vshrq_n_u8(reg, 6), mask3);
            uint8x16x4_t v = {{g0, g1, g2, g3}};
            vst4q_u8(out + i, v);
        }
    }
}

/* ============================== D=3 ================================== */

__attribute__((noinline))
static void unpack_d3_flat(uint8_t *out, const uint8_t *bm, int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d3_unpack(bm + ((i      * 3) >> 3));
            uint8x8_t codes_hi = flat_d3_unpack(bm + (((i + 8) * 3) >> 3));
            vst1q_u8(out + i, vcombine_u8(codes_lo, codes_hi));
        }
    }
}

/* ============================== D=4 ================================== */

__attribute__((noinline))
static void unpack_d4_flat(uint8_t *out, const uint8_t *bm, int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d4_unpack(bm + (i >> 1));
            vst1q_u8(out + i, codes);
        }
    }
}

__attribute__((noinline))
static void unpack_d4_fl(uint8_t *out, const uint8_t *bm, int n, int reps)
{
    uint8x16_t maskF = vdupq_n_u8(0x0F);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 32 <= n; i += 32) {
            uint8x16_t reg = vld1q_u8(bm + (i >> 1));
            uint8x16_t g0 = vandq_u8(reg, maskF);
            uint8x16_t g1 = vandq_u8(vshrq_n_u8(reg, 4), maskF);
            uint8x16x2_t v = {{g0, g1}};
            vst2q_u8(out + i, v);
        }
    }
}

/* ============================== D=5 ================================== */

__attribute__((noinline))
static void unpack_d5_flat(uint8_t *out, const uint8_t *bm, int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d5_unpack(bm + ((i      * 5) >> 3));
            uint8x8_t codes_hi = flat_d5_unpack(bm + (((i + 8) * 5) >> 3));
            vst1q_u8(out + i, vcombine_u8(codes_lo, codes_hi));
        }
    }
}

/* ============================== D=6 ================================== */

__attribute__((noinline))
static void unpack_d6_flat(uint8_t *out, const uint8_t *bm, int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t codes_lo = flat_d6_unpack(bm + ((i      * 6) >> 3));
            uint8x8_t codes_hi = flat_d6_unpack(bm + (((i + 8) * 6) >> 3));
            vst1q_u8(out + i, vcombine_u8(codes_lo, codes_hi));
        }
    }
}

/* =========================== driver ================================ */

static double bench_one(void (*fn)(uint8_t *, const uint8_t *, int, int),
                        uint8_t *out, const uint8_t *bm)
{
    double best = 1e9;
    for (int run = 0; run < 3; run++) {
        double t0 = now_sec();
        fn(out, bm, N, REPS);
        double t1 = now_sec();
        double gbs = (double)N * REPS / (t1 - t0) / 1e9;
        if (gbs > 1.0 / best) best = 1.0 / gbs;
    }
    return 1.0 / best;  /* GB/s of best run */
}

static int verify(uint8_t *a, uint8_t *b, int n, int D, const char *tag)
{
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("  MISMATCH %s at i=%d: %02x vs %02x (D=%d)\n",
                   tag, i, a[i], b[i], D);
            return 0;
        }
        if (a[i] >= (1 << D)) {
            printf("  BAD OUTPUT %s at i=%d: %02x out of range for D=%d\n",
                   tag, i, a[i], D);
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    uint8_t *out_a = (uint8_t *)aligned_alloc(64, N + 64);
    uint8_t *out_b = (uint8_t *)aligned_alloc(64, N + 64);
    uint8_t *bm    = (uint8_t *)aligned_alloc(64, N + 64);
    if (!out_a || !out_b || !bm) { perror("alloc"); return 1; }

    srand(42);
    for (int i = 0; i < N; i++) bm[i] = (uint8_t)rand();

    /* Correctness: flat vs FL produce same output on D=2 and D=4 */
    memset(out_a, 0, N); memset(out_b, 0, N);
    unpack_d2_flat(out_a, bm, N, 1);
    unpack_d2_fl  (out_b, bm, N, 1);
    if (!verify(out_a, out_b, N, 2, "d2")) return 2;

    memset(out_a, 0, N); memset(out_b, 0, N);
    unpack_d4_flat(out_a, bm, N, 1);
    unpack_d4_fl  (out_b, bm, N, 1);
    if (!verify(out_a, out_b, N, 4, "d4")) return 2;

    printf("== bench_unpack_dN: pure D-bit unpack to bytes ==\n");
    printf("N = %d, REPS = %d, total = %lld output bytes per variant\n\n",
           N, REPS, (long long)N * REPS);
    printf("                   flat_dX        FL-equiv     FL/flat\n");
    printf("                   -------        --------     -------\n");

    double f, l;

    f = bench_one(unpack_d2_flat, out_a, bm);
    l = bench_one(unpack_d2_fl,   out_a, bm);
    printf("D=2 (4 codes/B):  %5.1f GB/s    %5.1f GB/s    %.2fx\n", f, l, l / f);

    f = bench_one(unpack_d3_flat, out_a, bm);
    printf("D=3 (no FL):      %5.1f GB/s        ---          ---\n", f);

    f = bench_one(unpack_d4_flat, out_a, bm);
    l = bench_one(unpack_d4_fl,   out_a, bm);
    printf("D=4 (2 codes/B):  %5.1f GB/s    %5.1f GB/s    %.2fx\n", f, l, l / f);

    f = bench_one(unpack_d5_flat, out_a, bm);
    printf("D=5 (no FL):      %5.1f GB/s        ---          ---\n", f);

    f = bench_one(unpack_d6_flat, out_a, bm);
    printf("D=6 (no FL):      %5.1f GB/s        ---          ---\n", f);

    printf("\nGB/s = output bytes per second (1 byte per code).\n");
    printf("FL-equiv only available when D divides 8 (D=2: vst4q; D=4: vst2q).\n");

    free(out_a); free(out_b); free(bm);
    return 0;
}
