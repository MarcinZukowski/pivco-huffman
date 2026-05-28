/* extras/bench/bench_unpack_fl_layout.c — D-bit unpack throughput, three layouts.
 *
 * Variants per D ∈ {2..7}:
 *   flat_dX          : current production layout (row-major bit-packed),
 *                      flat_dX_unpack + vst1q.
 *   fl_natural       : same row-major layout, "FL-style" shift+mask + vstKq
 *                      (only D=2: vst4q, D=4: vst2q).
 *   fl_layout        : FastLanes transposed layout (16-lane interleaved
 *                      bytes), shift+mask only — works for every D.
 *                      Code lifted verbatim from FastLanes
 *                      cwida/FastLanes publications/data_parallelized_*.
 *
 * Block = 1024 codes (FastLanes "vector"); we run 8 blocks per call so
 * N = 8192, matching pivco's block size.
 *
 * Build: cc -O3 -o bench_unpack_fl_layout extras/bench/bench_unpack_fl_layout.c
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "../../src/pivco_huffman_neon_flat.h"

#define BLOCK 1024
#define BLOCKS 8
#define N (BLOCK * BLOCKS)
#define REPS 50000

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* =================== flat_dX (current production layout) ============= */

__attribute__((noinline))
static void current_d2_unpack(uint8_t *out, const uint8_t *in, int n, int reps) {
    for (int r = 0; r < reps; r++)
        for (int i = 0; i + 16 <= n; i += 16)
            vst1q_u8(out + i, flat_d2_unpack(in + (i >> 2)));
}
__attribute__((noinline))
static void current_d3_unpack(uint8_t *out, const uint8_t *in, int n, int reps) {
    for (int r = 0; r < reps; r++)
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t lo = flat_d3_unpack(in + ((i      * 3) >> 3));
            uint8x8_t hi = flat_d3_unpack(in + (((i + 8) * 3) >> 3));
            vst1q_u8(out + i, vcombine_u8(lo, hi));
        }
}
__attribute__((noinline))
static void current_d4_unpack(uint8_t *out, const uint8_t *in, int n, int reps) {
    for (int r = 0; r < reps; r++)
        for (int i = 0; i + 16 <= n; i += 16)
            vst1q_u8(out + i, flat_d4_unpack(in + (i >> 1)));
}
__attribute__((noinline))
static void current_d5_unpack(uint8_t *out, const uint8_t *in, int n, int reps) {
    for (int r = 0; r < reps; r++)
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t lo = flat_d5_unpack(in + ((i      * 5) >> 3));
            uint8x8_t hi = flat_d5_unpack(in + (((i + 8) * 5) >> 3));
            vst1q_u8(out + i, vcombine_u8(lo, hi));
        }
}
__attribute__((noinline))
static void current_d6_unpack(uint8_t *out, const uint8_t *in, int n, int reps) {
    for (int r = 0; r < reps; r++)
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t lo = flat_d6_unpack(in + ((i      * 6) >> 3));
            uint8x8_t hi = flat_d6_unpack(in + (((i + 8) * 6) >> 3));
            vst1q_u8(out + i, vcombine_u8(lo, hi));
        }
}
/* No flat_d7 unpack in production; skip */

/* =================== fl_natural (D=2/D=4 only) ====================== */

__attribute__((noinline))
static void fl_natural_d2(uint8_t *out, const uint8_t *in, int n, int reps) {
    uint8x16_t mask3 = vdupq_n_u8(0x03);
    for (int r = 0; r < reps; r++)
        for (int i = 0; i + 64 <= n; i += 64) {
            uint8x16_t reg = vld1q_u8(in + (i >> 2));
            uint8x16_t g0 = vandq_u8(reg, mask3);
            uint8x16_t g1 = vandq_u8(vshrq_n_u8(reg, 2), mask3);
            uint8x16_t g2 = vandq_u8(vshrq_n_u8(reg, 4), mask3);
            uint8x16_t g3 = vandq_u8(vshrq_n_u8(reg, 6), mask3);
            uint8x16x4_t v = {{g0, g1, g2, g3}};
            vst4q_u8(out + i, v);
        }
}
__attribute__((noinline))
static void fl_natural_d4(uint8_t *out, const uint8_t *in, int n, int reps) {
    uint8x16_t maskF = vdupq_n_u8(0x0F);
    for (int r = 0; r < reps; r++)
        for (int i = 0; i + 32 <= n; i += 32) {
            uint8x16_t reg = vld1q_u8(in + (i >> 1));
            uint8x16_t g0 = vandq_u8(reg, maskF);
            uint8x16_t g1 = vandq_u8(vshrq_n_u8(reg, 4), maskF);
            uint8x16x2_t v = {{g0, g1}};
            vst2q_u8(out + i, v);
        }
}

/* =================== fl_layout (FastLanes transposed) ================ */
/* Each function consumes 128*D input bytes and produces 1024 output
 * bytes (= one FL "vector"). Code lifted from
 * FastLanes/.../arm64v8_neon_intrinsic_1024_uf1_unpack_src.cpp. */

#define FL_BLOCK_OUT 1024  /* one FL vector */

static void fl_layout_block_d2(const uint8_t *in, uint8_t *out) {
    uint8x16_t reg, t;
    for (int i = 0; i < 8; ++i) {
        reg = vld1q_u8(in + i*16 + 0);
        t = vandq_u8(reg, vdupq_n_u8(3));               vst1q_u8(out + i*16 + 128*0, t);
        t = vandq_u8(vshrq_n_u8(reg, 2), vdupq_n_u8(3)); vst1q_u8(out + i*16 + 128*1, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(3)); vst1q_u8(out + i*16 + 128*2, t);
        t = vandq_u8(vshrq_n_u8(reg, 6), vdupq_n_u8(3)); vst1q_u8(out + i*16 + 128*3, t);
        reg = vld1q_u8(in + i*16 + 128);
        t = vandq_u8(reg, vdupq_n_u8(3));               vst1q_u8(out + i*16 + 128*4, t);
        t = vandq_u8(vshrq_n_u8(reg, 2), vdupq_n_u8(3)); vst1q_u8(out + i*16 + 128*5, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(3)); vst1q_u8(out + i*16 + 128*6, t);
        t = vandq_u8(vshrq_n_u8(reg, 6), vdupq_n_u8(3)); vst1q_u8(out + i*16 + 128*7, t);
    }
}

static void fl_layout_block_d3(const uint8_t *in, uint8_t *out) {
    uint8x16_t reg, t;
    for (int i = 0; i < 8; ++i) {
        reg = vld1q_u8(in + i*16 + 0);
        t = vandq_u8(reg, vdupq_n_u8(7));               vst1q_u8(out + i*16 + 128*0, t);
        t = vandq_u8(vshrq_n_u8(reg, 3), vdupq_n_u8(7)); vst1q_u8(out + i*16 + 128*1, t);
        t = vandq_u8(vshrq_n_u8(reg, 6), vdupq_n_u8(3));
        reg = vld1q_u8(in + i*16 + 128);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(1)), 2), t);
        vst1q_u8(out + i*16 + 128*2, t);
        t = vandq_u8(vshrq_n_u8(reg, 1), vdupq_n_u8(7)); vst1q_u8(out + i*16 + 128*3, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(7)); vst1q_u8(out + i*16 + 128*4, t);
        t = vandq_u8(vshrq_n_u8(reg, 7), vdupq_n_u8(1));
        reg = vld1q_u8(in + i*16 + 256);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(3)), 1), t);
        vst1q_u8(out + i*16 + 128*5, t);
        t = vandq_u8(vshrq_n_u8(reg, 2), vdupq_n_u8(7)); vst1q_u8(out + i*16 + 128*6, t);
        t = vandq_u8(vshrq_n_u8(reg, 5), vdupq_n_u8(7)); vst1q_u8(out + i*16 + 128*7, t);
    }
}

static void fl_layout_block_d4(const uint8_t *in, uint8_t *out) {
    uint8x16_t reg, t;
    for (int i = 0; i < 8; ++i) {
        reg = vld1q_u8(in + i*16 + 0);
        t = vandq_u8(reg, vdupq_n_u8(15));               vst1q_u8(out + i*16 + 128*0, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(15)); vst1q_u8(out + i*16 + 128*1, t);
        reg = vld1q_u8(in + i*16 + 128);
        t = vandq_u8(reg, vdupq_n_u8(15));               vst1q_u8(out + i*16 + 128*2, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(15)); vst1q_u8(out + i*16 + 128*3, t);
        reg = vld1q_u8(in + i*16 + 256);
        t = vandq_u8(reg, vdupq_n_u8(15));               vst1q_u8(out + i*16 + 128*4, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(15)); vst1q_u8(out + i*16 + 128*5, t);
        reg = vld1q_u8(in + i*16 + 384);
        t = vandq_u8(reg, vdupq_n_u8(15));               vst1q_u8(out + i*16 + 128*6, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(15)); vst1q_u8(out + i*16 + 128*7, t);
    }
}

static void fl_layout_block_d5(const uint8_t *in, uint8_t *out) {
    uint8x16_t reg, t;
    for (int i = 0; i < 8; ++i) {
        reg = vld1q_u8(in + i*16 + 0);
        t = vandq_u8(reg, vdupq_n_u8(31));               vst1q_u8(out + i*16 + 128*0, t);
        t = vandq_u8(vshrq_n_u8(reg, 5), vdupq_n_u8(7));
        reg = vld1q_u8(in + i*16 + 128);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(3)), 3), t);
        vst1q_u8(out + i*16 + 128*1, t);
        t = vandq_u8(vshrq_n_u8(reg, 2), vdupq_n_u8(31)); vst1q_u8(out + i*16 + 128*2, t);
        t = vandq_u8(vshrq_n_u8(reg, 7), vdupq_n_u8(1));
        reg = vld1q_u8(in + i*16 + 256);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(15)), 1), t);
        vst1q_u8(out + i*16 + 128*3, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(15));
        reg = vld1q_u8(in + i*16 + 384);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(1)), 4), t);
        vst1q_u8(out + i*16 + 128*4, t);
        t = vandq_u8(vshrq_n_u8(reg, 1), vdupq_n_u8(31)); vst1q_u8(out + i*16 + 128*5, t);
        t = vandq_u8(vshrq_n_u8(reg, 6), vdupq_n_u8(3));
        reg = vld1q_u8(in + i*16 + 512);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(7)), 2), t);
        vst1q_u8(out + i*16 + 128*6, t);
        t = vandq_u8(vshrq_n_u8(reg, 3), vdupq_n_u8(31)); vst1q_u8(out + i*16 + 128*7, t);
    }
}

static void fl_layout_block_d6(const uint8_t *in, uint8_t *out) {
    uint8x16_t reg, t;
    for (int i = 0; i < 8; ++i) {
        reg = vld1q_u8(in + i*16 + 0);
        t = vandq_u8(reg, vdupq_n_u8(63));               vst1q_u8(out + i*16 + 128*0, t);
        t = vandq_u8(vshrq_n_u8(reg, 6), vdupq_n_u8(3));
        reg = vld1q_u8(in + i*16 + 128);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(15)), 2), t);
        vst1q_u8(out + i*16 + 128*1, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(15));
        reg = vld1q_u8(in + i*16 + 256);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(3)), 4), t);
        vst1q_u8(out + i*16 + 128*2, t);
        t = vandq_u8(vshrq_n_u8(reg, 2), vdupq_n_u8(63)); vst1q_u8(out + i*16 + 128*3, t);
        reg = vld1q_u8(in + i*16 + 384);
        t = vandq_u8(reg, vdupq_n_u8(63));               vst1q_u8(out + i*16 + 128*4, t);
        t = vandq_u8(vshrq_n_u8(reg, 6), vdupq_n_u8(3));
        reg = vld1q_u8(in + i*16 + 512);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(15)), 2), t);
        vst1q_u8(out + i*16 + 128*5, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(15));
        reg = vld1q_u8(in + i*16 + 640);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(3)), 4), t);
        vst1q_u8(out + i*16 + 128*6, t);
        t = vandq_u8(vshrq_n_u8(reg, 2), vdupq_n_u8(63)); vst1q_u8(out + i*16 + 128*7, t);
    }
}

static void fl_layout_block_d7(const uint8_t *in, uint8_t *out) {
    uint8x16_t reg, t;
    for (int i = 0; i < 8; ++i) {
        reg = vld1q_u8(in + i*16 + 0);
        t = vandq_u8(reg, vdupq_n_u8(127));              vst1q_u8(out + i*16 + 128*0, t);
        t = vandq_u8(vshrq_n_u8(reg, 7), vdupq_n_u8(1));
        reg = vld1q_u8(in + i*16 + 128);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(63)), 1), t);
        vst1q_u8(out + i*16 + 128*1, t);
        t = vandq_u8(vshrq_n_u8(reg, 6), vdupq_n_u8(3));
        reg = vld1q_u8(in + i*16 + 256);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(31)), 2), t);
        vst1q_u8(out + i*16 + 128*2, t);
        t = vandq_u8(vshrq_n_u8(reg, 5), vdupq_n_u8(7));
        reg = vld1q_u8(in + i*16 + 384);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(15)), 3), t);
        vst1q_u8(out + i*16 + 128*3, t);
        t = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(15));
        reg = vld1q_u8(in + i*16 + 512);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(7)), 4), t);
        vst1q_u8(out + i*16 + 128*4, t);
        t = vandq_u8(vshrq_n_u8(reg, 3), vdupq_n_u8(31));
        reg = vld1q_u8(in + i*16 + 640);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(3)), 5), t);
        vst1q_u8(out + i*16 + 128*5, t);
        t = vandq_u8(vshrq_n_u8(reg, 2), vdupq_n_u8(63));
        reg = vld1q_u8(in + i*16 + 768);
        t = vorrq_u8(vshlq_n_u8(vandq_u8(reg, vdupq_n_u8(1)), 6), t);
        vst1q_u8(out + i*16 + 128*6, t);
        t = vandq_u8(vshrq_n_u8(reg, 1), vdupq_n_u8(127)); vst1q_u8(out + i*16 + 128*7, t);
    }
}

/* Wrappers that loop over BLOCKS=8 vectors. */
#define DEFINE_FL_LAYOUT_WRAPPER(D, in_bytes_per_block) \
__attribute__((noinline))                              \
static void fl_layout_d##D(uint8_t *out, const uint8_t *in, int n, int reps) { \
    (void)n; \
    for (int r = 0; r < reps; r++)                                       \
        for (int b = 0; b < BLOCKS; b++)                                 \
            fl_layout_block_d##D(in + b*(in_bytes_per_block), out + b*1024); \
}
DEFINE_FL_LAYOUT_WRAPPER(2, 256)
DEFINE_FL_LAYOUT_WRAPPER(3, 384)
DEFINE_FL_LAYOUT_WRAPPER(4, 512)
DEFINE_FL_LAYOUT_WRAPPER(5, 640)
DEFINE_FL_LAYOUT_WRAPPER(6, 768)
DEFINE_FL_LAYOUT_WRAPPER(7, 896)

/* =========================== driver ================================ */

static double bench_one(void (*fn)(uint8_t *, const uint8_t *, int, int),
                        uint8_t *out, const uint8_t *in)
{
    double best_gbs = 0;
    for (int run = 0; run < 3; run++) {
        double t0 = now_sec();
        fn(out, in, N, REPS);
        double t1 = now_sec();
        double gbs = (double)N * REPS / (t1 - t0) / 1e9;
        if (gbs > best_gbs) best_gbs = gbs;
    }
    return best_gbs;
}

int main(void) {
    uint8_t *in  = (uint8_t *)aligned_alloc(64, N + 64);
    uint8_t *out = (uint8_t *)aligned_alloc(64, N + 64);
    if (!in || !out) { perror("alloc"); return 1; }

    srand(42);
    for (int i = 0; i < N; i++) in[i] = (uint8_t)rand();

    printf("== bench_unpack_fl_layout: D-bit unpack across 3 layouts ==\n");
    printf("N = %d codes, REPS = %d, output = %lld bytes total\n\n",
           N, REPS, (long long)N * REPS);
    printf("                flat_dX     fl_natural    fl_layout    layout vs flat\n");
    printf("                (current)   (natural+vstK)(transposed)  speedup\n");
    printf("                ---------   ------------- -----------   ------\n");

    double f, n_, l;

#define ROW_NATURAL(D)                                                       \
    do {                                                                     \
        f = bench_one(current_d##D##_unpack, out, in);                          \
        l = bench_one(fl_layout_d##D, out, in);                              \
        n_ = bench_one(fl_natural_d##D, out, in);                            \
        printf("D=%d:           %5.1f GB/s   %5.1f GB/s   %5.1f GB/s   %.2fx\n", \
               D, f, n_, l, l / f);                                          \
    } while (0)

#define ROW_NO_NATURAL(D)                                                    \
    do {                                                                     \
        f = bench_one(current_d##D##_unpack, out, in);                          \
        l = bench_one(fl_layout_d##D, out, in);                              \
        printf("D=%d:           %5.1f GB/s        ---     %5.1f GB/s   %.2fx\n", \
               D, f, l, l / f);                                              \
    } while (0)

    ROW_NATURAL(2);
    ROW_NO_NATURAL(3);
    ROW_NATURAL(4);
    ROW_NO_NATURAL(5);
    ROW_NO_NATURAL(6);
    /* No flat_d7_unpack in production — skip flat column for D=7. */
    {
        l = bench_one(fl_layout_d7, out, in);
        printf("D=7:                ---             ---     %5.1f GB/s    n/a\n", l);
    }

    printf("\nGB/s = output bytes/sec.  fl_natural defined only when D|8.\n");
    printf("fl_layout = FastLanes 1024-vector transposed layout (16 byte-lanes).\n");

    free(in); free(out);
    return 0;
}
