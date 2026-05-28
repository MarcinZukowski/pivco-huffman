/* bench_partition_micro.c — minimal microbench for the partition kernels
 * with controllable loop length (INNER).
 *
 * Sweeps INNER from short (decoder-realistic, ~32-256 iters) to long
 * (~1024 iters) so we can see how loop length affects per-call cost of:
 *
 *   - 1-cursor stride-8  (1 partition_8 per loop body)
 *   - 1-cursor stride-16 (2-way unroll, serial counter chain — same
 *                         shape as decode_node_neon's hot loop)
 *
 * On x86 with AVX-512, also benches partition_32 across the same inner
 * lengths.  Total work is held constant across INNER values.
 *
 * Each "outer" resets counters and destination cursors, mimicking the
 * decoder's per-call setup pattern.
 */
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifdef __aarch64__
#include "../pivco_huffman_neon_common.h"
#include <arm_neon.h>

static inline int micro_partition_8(const uint16_t *src, uint8_t mask,
                                     uint16_t *left_out, uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    uint8x16_t right = vqtbl1q_u8(data, shuf_r);
    uint8x16_t left  = vqtbl1q_u8(data, shuf_l);
    int n_right = compress_popcnt[mask];
    vst1q_u8((uint8_t *)right_out, right);
    vst1q_u8((uint8_t *)left_out, left);
    return n_right;
}
#define HAVE_PARTITION 1
#endif

#if defined(__x86_64__) && defined(PIVCO_HAS_AVX512)
#include <immintrin.h>
static inline int micro_partition_32(const uint16_t *src, uint32_t mask,
                                      uint16_t *left_out,
                                      uint16_t *right_out)
{
    __m512i v   = _mm512_loadu_si512((const __m512i *)src);
    __m512i lv  = _mm512_maskz_compress_epi16((__mmask32)~mask, v);
    __m512i rv  = _mm512_maskz_compress_epi16((__mmask32)mask, v);
    _mm512_storeu_si512((__m512i *)left_out,  lv);
    _mm512_storeu_si512((__m512i *)right_out, rv);
    return __builtin_popcount(mask);
}
#define HAVE_PARTITION_32 1
#endif

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#ifdef HAVE_PARTITION
static double bench_p8_stride8(const uint16_t *in, const uint8_t *masks,
                                uint16_t *out_L, uint16_t *out_R,
                                int inner, long long outer)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (long long o = 0; o < outer; o++) {
        int nl = 0, nr = 0;
        for (int i = 0; i < inner; i++) {
            int x = micro_partition_8(in + i * 8, masks[i],
                                       out_L + nl, out_R + nr);
            nr += x;
            nl += (8 - x);
        }
        sink += nl + nr;
    }
    (void)sink;
    return now_sec() - t0;
}

static double bench_p8_stride16(const uint16_t *in, const uint8_t *masks,
                                 uint16_t *out_L, uint16_t *out_R,
                                 int inner, long long outer)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (long long o = 0; o < outer; o++) {
        int nl = 0, nr = 0;
        int i = 0;
        for (; i + 2 <= inner; i += 2) {
            int x0 = micro_partition_8(in + (i+0) * 8, masks[i+0],
                                        out_L + nl, out_R + nr);
            nr += x0; nl += (8 - x0);
            int x1 = micro_partition_8(in + (i+1) * 8, masks[i+1],
                                        out_L + nl, out_R + nr);
            nr += x1; nl += (8 - x1);
        }
        for (; i < inner; i++) {
            int x = micro_partition_8(in + i * 8, masks[i],
                                       out_L + nl, out_R + nr);
            nr += x; nl += (8 - x);
        }
        sink += nl + nr;
    }
    (void)sink;
    return now_sec() - t0;
}
#endif  /* HAVE_PARTITION */

#define MAX_INNER 1024
#define TOTAL_WORK (200LL * 1000LL * 1024LL)  /* ~200M iters total per kernel */

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    /* Sweep INNER values to see how loop length affects throughput.
     * Decoder's avg paired loop is ~91 iters at BLK=16384/prose_pride. */
    static const int INNERS[] = { 8, 16, 32, 64, 91, 128, 256, 512, 1024 };
    static const int N_INNERS = sizeof(INNERS) / sizeof(INNERS[0]);

    static uint16_t out_L[MAX_INNER * 16] __attribute__((aligned(64)));
    static uint16_t out_R[MAX_INNER * 16] __attribute__((aligned(64)));

#ifdef HAVE_PARTITION
    init_compress_table();

    static uint16_t in_buf[MAX_INNER * 8 + 16] __attribute__((aligned(64)));
    static uint8_t  masks[MAX_INNER];

    srand(0xBEEFCAFE);
    for (int i = 0; i < MAX_INNER * 8; i++) in_buf[i] = (uint16_t)(rand() & 0xFFFF);
    for (int i = 0; i < MAX_INNER; i++)     masks[i] = (uint8_t)(rand() & 0xFF);

    /* Warmup */
    bench_p8_stride8(in_buf, masks, out_L, out_R, 1024, 1000);

    printf("\n=== partition_8 microbench (INNER sweep) ===\n");
    printf("Total work held constant at ~%lld inner iters per kernel.\n",
           TOTAL_WORK);
    printf("Each row reruns with a different INNER (loop length per outer).\n\n");

    printf("  %5s   %10s   %10s   %10s   %10s   %5s\n",
           "INNER", "outer", "stride-8", "stride-16", "(ns/call)", "ratio");
    printf("  -------------------------------------------------------------\n");

    for (int k = 0; k < N_INNERS; k++) {
        int inner = INNERS[k];
        long long outer = TOTAL_WORK / inner;
        if (outer < 1) outer = 1;

        double t_s8  = 1e9;
        double t_s16 = 1e9;

        for (int r = 0; r < 3; r++) {
            double t = bench_p8_stride8(in_buf, masks, out_L, out_R,
                                         inner, outer);
            if (t < t_s8) t_s8 = t;
        }
        for (int r = 0; r < 3; r++) {
            double t = bench_p8_stride16(in_buf, masks, out_L, out_R,
                                          inner, outer);
            if (t < t_s16) t_s16 = t;
        }

        long long calls = outer * inner;
        double ns_s8  = t_s8  * 1e9 / calls;
        double ns_s16 = t_s16 * 1e9 / calls;

        printf("  %5d   %10lld   %10.3f   %10.3f   %10s   %5.2f\n",
               inner, outer, ns_s8, ns_s16, "ns/call",
               ns_s16 / ns_s8);
    }
    printf("\n  ratio = stride-16 / stride-8 (lower = 2-way unroll wins)\n");
#endif

#ifdef HAVE_PARTITION_32
    static uint16_t in_buf32[MAX_INNER * 32 + 32] __attribute__((aligned(64)));
    static uint32_t masks32[MAX_INNER];
    srand(0xBEEFCAFE);
    for (int i = 0; i < MAX_INNER * 32; i++) in_buf32[i] = (uint16_t)(rand() & 0xFFFF);
    for (int i = 0; i < MAX_INNER; i++)      masks32[i] = (uint32_t)rand();

    printf("\n=== partition_32 (AVX-512) microbench (INNER sweep) ===\n");
    printf("  %5s   %10s   %10s\n", "INNER", "outer", "ns/call");
    printf("  -----------------------------------------\n");
    for (int k = 0; k < N_INNERS; k++) {
        int inner = INNERS[k];
        long long outer = TOTAL_WORK / inner;
        if (outer < 1) outer = 1;
        double t_min = 1e9;
        volatile int sink = 0;
        for (int r = 0; r < 3; r++) {
            double t0 = now_sec();
            for (long long o = 0; o < outer; o++) {
                int nl = 0, nr = 0;
                for (int i = 0; i < inner; i++) {
                    int x = micro_partition_32(in_buf32 + i * 32, masks32[i],
                                                out_L + nl, out_R + nr);
                    nr += x; nl += (32 - x);
                }
                sink += nl + nr;
            }
            double t = now_sec() - t0;
            if (t < t_min) t_min = t;
        }
        long long calls = outer * inner;
        printf("  %5d   %10lld   %10.3f\n",
               inner, outer, t_min * 1e9 / calls);
        (void)sink;
    }
#endif

    return 0;
}
