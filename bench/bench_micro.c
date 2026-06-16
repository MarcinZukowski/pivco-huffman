/* Microbenchmark: scatter vs partition vs flat-decode per-element cost.
 *
 * Build (NEON, M4 / Graviton 4):
 *   cc -O2 -o bench_micro bench/bench_micro.c -I include -I src
 * Build (AVX-512 VBMI2, Xeon):
 *   cc -O3 -march=native -o bench_micro bench/bench_micro.c -I include -I src
 * Build (SSE4.1 / AVX2, Zen 3):
 *   cc -O3 -march=native -o bench_micro bench/bench_micro.c -I include -I src
 *
 * Backend is auto-detected by the platform predefined macros below. */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef __aarch64__
#include <arm_neon.h>
#include "pivco_huffman_neon_flat.h"  /* flat_d{2..6}_unpack() + tables */
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

#if defined(__AVX512BW__) && defined(__AVX512VBMI__) && defined(__AVX512VBMI2__)
#include <immintrin.h>
#include "pivco_huffman_avx512_flat.h"  /* flat_d{2..6}_unpack_avx512* */
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif

#if defined(__SSE4_1__)
#include <smmintrin.h>
#include "pivco_huffman_x86_flat.h"     /* flat_d4_unpack_x86 */
#define HAS_SSE4 1
#else
#define HAS_SSE4 0
#endif

#ifndef PIVCO_BLOCK_SIZE
#define PIVCO_BLOCK_SIZE 8192
#endif

#define N PIVCO_BLOCK_SIZE
#define REPS 100000

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- Memset (universal reference floor) ---- */

__attribute__((noinline))
static void bench_memset(uint8_t *symbols, int n, uint8_t sym, int reps)
{
    for (int r = 0; r < reps; r++)
        memset(symbols, sym, (size_t)n);
}

/* ---- Scatter: write one constant byte to n random positions ---- */

__attribute__((noinline)) static void bench_scatter_scalar(uint8_t *symbols, const uint16_t *indices,
                                  int n, uint8_t sym, int reps)
{
    for (int r = 0; r < reps; r++) {
        for (int j = 0; j < n; j++)
            symbols[indices[j]] = sym;
    }
}

#if HAS_NEON
__attribute__((noinline)) static void bench_scatter_neon(uint8_t *symbols, const uint16_t *indices,
                                int n, uint8_t sym, int reps)
{
    for (int r = 0; r < reps; r++) {
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            uint16x8_t idx = vld1q_u16(indices + j);
            symbols[vgetq_lane_u16(idx, 0)] = sym;
            symbols[vgetq_lane_u16(idx, 1)] = sym;
            symbols[vgetq_lane_u16(idx, 2)] = sym;
            symbols[vgetq_lane_u16(idx, 3)] = sym;
            symbols[vgetq_lane_u16(idx, 4)] = sym;
            symbols[vgetq_lane_u16(idx, 5)] = sym;
            symbols[vgetq_lane_u16(idx, 6)] = sym;
            symbols[vgetq_lane_u16(idx, 7)] = sym;
        }
        for (; j < n; j++)
            symbols[indices[j]] = sym;
    }
}
#endif

/* ---- Partition: TBL shuffle 8 uint16 indices ---- */

#if HAS_NEON
static uint8_t compress_tab[256][32] __attribute__((aligned(32)));
static uint8_t compress_popcnt[256] __attribute__((aligned(64)));

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
        for (int j = out_r * 2; j < 16; j++)
            compress_tab[mask][j] = 0xFF;

        int out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (!(mask & (1 << i))) {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_l * 2; j < 16; j++)
            compress_tab[mask][16 + j] = 0xFF;
    }
}

__attribute__((noinline)) static void bench_partition_neon(const uint16_t *indices, const uint8_t *bitmap,
                                  uint16_t *left, uint16_t *right,
                                  int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(indices + j));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            vst1q_u8((uint8_t *)(right + n_right), vqtbl1q_u8(data, shuf_r));
            vst1q_u8((uint8_t *)(left + n_left), vqtbl1q_u8(data, shuf_l));
            n_right += compress_popcnt[mask];
            n_left += (8 - compress_popcnt[mask]);
        }
    }
}

/* ---- Partition from identity (no index load) ---- */

__attribute__((noinline)) static void bench_partition_root_neon(const uint8_t *bitmap,
                                       uint16_t *left, uint16_t *right,
                                       int n, int reps)
{
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    uint16x8_t voff = vld1q_u16(off);

    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vreinterpretq_u8_u16(
                vaddq_u16(vdupq_n_u16((uint16_t)j), voff));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            vst1q_u8((uint8_t *)(right + n_right), vqtbl1q_u8(data, shuf_r));
            vst1q_u8((uint8_t *)(left + n_left), vqtbl1q_u8(data, shuf_l));
            n_right += compress_popcnt[mask];
            n_left += (8 - compress_popcnt[mask]);
        }
    }
}

/* ---- Partition one side only (right) ---- */

__attribute__((noinline))
static void bench_partition_half_neon(const uint16_t *indices,
                                       const uint8_t *bitmap,
                                       uint16_t *right,
                                       int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(indices + j));
            uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
            vst1q_u8((uint8_t *)(right + n_right), vqtbl1q_u8(data, shuf_r));
            n_right += compress_popcnt[mask];
        }
    }
}

__attribute__((noinline))
static void bench_partition_root_half_neon(const uint8_t *bitmap,
                                            uint16_t *right,
                                            int n, int reps)
{
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    uint16x8_t voff = vld1q_u16(off);

    for (int r = 0; r < reps; r++) {
        int n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vreinterpretq_u8_u16(
                vaddq_u16(vdupq_n_u16((uint16_t)j), voff));
            uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
            vst1q_u8((uint8_t *)(right + n_right), vqtbl1q_u8(data, shuf_r));
            n_right += compress_popcnt[mask];
        }
    }
}

/* ---- Store-pressure isolation tests ---- */

/* Every iteration uses mask=0x00: all 8 src elements go to the left
 * side, n_left advances by 8 (= 16 bytes) each iter, so the left
 * vst1q_u8 produces a tight non-overlapping run of 16-byte stores.
 * Output is a raw byte buffer so the caller can deliberately
 * misalign by passing left_bytes + offset. */
__attribute__((noinline))
static void bench_partition_left_full(const uint16_t *src,
                                       uint8_t *left_bytes,
                                       int n, int reps)
{
    const uint8_t *tab = compress_tab[0x00];
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    for (int r = 0; r < reps; r++) {
        int n_left_bytes = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
            uint8x16_t left = vqtbl1q_u8(data, shuf_l);
            vst1q_u8(left_bytes + n_left_bytes, left);
            n_left_bytes += 16;
        }
    }
}

/* TBL throughput probes: 16 independent chains per iter, exceeds the
 * throughput × latency product of any plausible NEON TBL pipeline so
 * the port throughput shows up directly. */
__attribute__((noinline))
static void bench_tbl1_throughput(int reps)
{
    static const uint8_t init[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    uint8x16_t tab = vld1q_u8(init);
    uint8x16_t a0=tab, a1=tab, a2=tab, a3=tab, a4=tab, a5=tab, a6=tab, a7=tab;
    uint8x16_t a8=tab, a9=tab, aA=tab, aB=tab, aC=tab, aD=tab, aE=tab, aF=tab;
    for (int r = 0; r < reps; r++) {
        a0 = vqtbl1q_u8(a0, tab);  a1 = vqtbl1q_u8(a1, tab);
        a2 = vqtbl1q_u8(a2, tab);  a3 = vqtbl1q_u8(a3, tab);
        a4 = vqtbl1q_u8(a4, tab);  a5 = vqtbl1q_u8(a5, tab);
        a6 = vqtbl1q_u8(a6, tab);  a7 = vqtbl1q_u8(a7, tab);
        a8 = vqtbl1q_u8(a8, tab);  a9 = vqtbl1q_u8(a9, tab);
        aA = vqtbl1q_u8(aA, tab);  aB = vqtbl1q_u8(aB, tab);
        aC = vqtbl1q_u8(aC, tab);  aD = vqtbl1q_u8(aD, tab);
        aE = vqtbl1q_u8(aE, tab);  aF = vqtbl1q_u8(aF, tab);
    }
    volatile uint8_t sink8 =
        vgetq_lane_u8(a0,0) ^ vgetq_lane_u8(a1,0) ^ vgetq_lane_u8(a2,0) ^
        vgetq_lane_u8(a3,0) ^ vgetq_lane_u8(a4,0) ^ vgetq_lane_u8(a5,0) ^
        vgetq_lane_u8(a6,0) ^ vgetq_lane_u8(a7,0) ^ vgetq_lane_u8(a8,0) ^
        vgetq_lane_u8(a9,0) ^ vgetq_lane_u8(aA,0) ^ vgetq_lane_u8(aB,0) ^
        vgetq_lane_u8(aC,0) ^ vgetq_lane_u8(aD,0) ^ vgetq_lane_u8(aE,0) ^
        vgetq_lane_u8(aF,0);
    (void)sink8;
}

__attribute__((noinline))
static void bench_tbl2_throughput(int reps)
{
    static const uint8_t init[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    uint8x16_t v = vld1q_u8(init);
    uint8x16x2_t tab = {{ v, v }};
    uint8x16_t a0=v, a1=v, a2=v, a3=v, a4=v, a5=v, a6=v, a7=v;
    uint8x16_t a8=v, a9=v, aA=v, aB=v, aC=v, aD=v, aE=v, aF=v;
    for (int r = 0; r < reps; r++) {
        a0 = vqtbl2q_u8(tab, a0);  a1 = vqtbl2q_u8(tab, a1);
        a2 = vqtbl2q_u8(tab, a2);  a3 = vqtbl2q_u8(tab, a3);
        a4 = vqtbl2q_u8(tab, a4);  a5 = vqtbl2q_u8(tab, a5);
        a6 = vqtbl2q_u8(tab, a6);  a7 = vqtbl2q_u8(tab, a7);
        a8 = vqtbl2q_u8(tab, a8);  a9 = vqtbl2q_u8(tab, a9);
        aA = vqtbl2q_u8(tab, aA);  aB = vqtbl2q_u8(tab, aB);
        aC = vqtbl2q_u8(tab, aC);  aD = vqtbl2q_u8(tab, aD);
        aE = vqtbl2q_u8(tab, aE);  aF = vqtbl2q_u8(tab, aF);
    }
    volatile uint8_t sink8 =
        vgetq_lane_u8(a0,0) ^ vgetq_lane_u8(a1,0) ^ vgetq_lane_u8(a2,0) ^
        vgetq_lane_u8(a3,0) ^ vgetq_lane_u8(a4,0) ^ vgetq_lane_u8(a5,0) ^
        vgetq_lane_u8(a6,0) ^ vgetq_lane_u8(a7,0) ^ vgetq_lane_u8(a8,0) ^
        vgetq_lane_u8(a9,0) ^ vgetq_lane_u8(aA,0) ^ vgetq_lane_u8(aB,0) ^
        vgetq_lane_u8(aC,0) ^ vgetq_lane_u8(aD,0) ^ vgetq_lane_u8(aE,0) ^
        vgetq_lane_u8(aF,0);
    (void)sink8;
}

__attribute__((noinline))
static void bench_tbl4_throughput(int reps)
{
    static const uint8_t init[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    uint8x16_t v = vld1q_u8(init);
    uint8x16x4_t tab = {{ v, v, v, v }};
    uint8x16_t a0=v, a1=v, a2=v, a3=v, a4=v, a5=v, a6=v, a7=v;
    uint8x16_t a8=v, a9=v, aA=v, aB=v, aC=v, aD=v, aE=v, aF=v;
    for (int r = 0; r < reps; r++) {
        a0 = vqtbl4q_u8(tab, a0);  a1 = vqtbl4q_u8(tab, a1);
        a2 = vqtbl4q_u8(tab, a2);  a3 = vqtbl4q_u8(tab, a3);
        a4 = vqtbl4q_u8(tab, a4);  a5 = vqtbl4q_u8(tab, a5);
        a6 = vqtbl4q_u8(tab, a6);  a7 = vqtbl4q_u8(tab, a7);
        a8 = vqtbl4q_u8(tab, a8);  a9 = vqtbl4q_u8(tab, a9);
        aA = vqtbl4q_u8(tab, aA);  aB = vqtbl4q_u8(tab, aB);
        aC = vqtbl4q_u8(tab, aC);  aD = vqtbl4q_u8(tab, aD);
        aE = vqtbl4q_u8(tab, aE);  aF = vqtbl4q_u8(tab, aF);
    }
    volatile uint8_t sink8 =
        vgetq_lane_u8(a0,0) ^ vgetq_lane_u8(a1,0) ^ vgetq_lane_u8(a2,0) ^
        vgetq_lane_u8(a3,0) ^ vgetq_lane_u8(a4,0) ^ vgetq_lane_u8(a5,0) ^
        vgetq_lane_u8(a6,0) ^ vgetq_lane_u8(a7,0) ^ vgetq_lane_u8(a8,0) ^
        vgetq_lane_u8(a9,0) ^ vgetq_lane_u8(aA,0) ^ vgetq_lane_u8(aB,0) ^
        vgetq_lane_u8(aC,0) ^ vgetq_lane_u8(aD,0) ^ vgetq_lane_u8(aE,0) ^
        vgetq_lane_u8(aF,0);
    (void)sink8;
}

/* vextq_u8 throughput probe — relevant to the no-extra-TBL coalesce
 * variant.  16 independent chains, each doing one vextq_u8 per iter. */
__attribute__((noinline))
static void bench_vext_throughput(int reps)
{
    static const uint8_t init[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    uint8x16_t v = vld1q_u8(init);
    uint8x16_t a0=v, a1=v, a2=v, a3=v, a4=v, a5=v, a6=v, a7=v;
    uint8x16_t a8=v, a9=v, aA=v, aB=v, aC=v, aD=v, aE=v, aF=v;
    for (int r = 0; r < reps; r++) {
        a0 = vextq_u8(a0, v, 4);  a1 = vextq_u8(a1, v, 4);
        a2 = vextq_u8(a2, v, 4);  a3 = vextq_u8(a3, v, 4);
        a4 = vextq_u8(a4, v, 4);  a5 = vextq_u8(a5, v, 4);
        a6 = vextq_u8(a6, v, 4);  a7 = vextq_u8(a7, v, 4);
        a8 = vextq_u8(a8, v, 4);  a9 = vextq_u8(a9, v, 4);
        aA = vextq_u8(aA, v, 4);  aB = vextq_u8(aB, v, 4);
        aC = vextq_u8(aC, v, 4);  aD = vextq_u8(aD, v, 4);
        aE = vextq_u8(aE, v, 4);  aF = vextq_u8(aF, v, 4);
    }
    volatile uint8_t sink8 =
        vgetq_lane_u8(a0,0) ^ vgetq_lane_u8(a1,0) ^ vgetq_lane_u8(a2,0) ^
        vgetq_lane_u8(a3,0) ^ vgetq_lane_u8(a4,0) ^ vgetq_lane_u8(a5,0) ^
        vgetq_lane_u8(a6,0) ^ vgetq_lane_u8(a7,0) ^ vgetq_lane_u8(a8,0) ^
        vgetq_lane_u8(a9,0) ^ vgetq_lane_u8(aA,0) ^ vgetq_lane_u8(aB,0) ^
        vgetq_lane_u8(aC,0) ^ vgetq_lane_u8(aD,0) ^ vgetq_lane_u8(aE,0) ^
        vgetq_lane_u8(aF,0);
    (void)sink8;
}

/* Coalesce-store partition prototype (vextq variant).
 *
 * For each side, maintains (accum, so_far) where accum is a 16-byte
 * register holding `so_far` already-placed elements (= so_far*2 valid
 * bytes at low lanes, rest zero).  Each iter:
 *   1. Compute the compacted side's data via the existing TBL.
 *   2. Shift the compacted data left by `so_far * 2` bytes (vextq with
 *      compile-time immediate; switch on so_far in [0,7]).
 *   3. OR into accum.
 *   4. If so_far + cnt >= 8, flush accum (vst1q), set up new accum from
 *      the overflow bytes (right-shifted from compacted).
 *   5. Otherwise, store the merged result back into accum and bump so_far.
 *
 * Saves stores: ~1 per iter on average (vs 2 in baseline) for random
 * masks where popcount averages 4.  Adds 1 vextq + 1 vorr per side per
 * iter, plus 1 indirect branch per side (switch jump table).
 *
 * Each of the 8 cases per side has a runtime "is this iter going to
 * flush?" branch; that's 1 nested branch per case.  Each is keyed by
 * a different so_far value, so the predictor can track them
 * independently — and most cases are themselves predictable (e.g.,
 * so_far=0 only flushes when cnt=8, so_far=7 always flushes).
 */
/* Store-coalescing prototypes (per-iter switch / per-iter TBL / 4-iter
 * macro-block) lived here briefly; all three lost to baseline on M4
 * (38–78% slower).  They're moved to extras/bench/bench_coalesce.c with a
 * full discussion in docs/COALESCE.md.  Keeping only the throughput probes
 * and store-port topology probe here. */
#if 0
/* 4-iter macro-block coalesce with lookahead.
 *
 * Process 4 mask bytes per macro-block.  Compute popcounts and prefix
 * sums upfront (scalar), so each iter's destination offset within the
 * macro-block is precomputed and *independent* of the other iters'
 * accumulator state — breaking the cross-iter dep chain that killed
 * the per-iter version.
 *
 * Per macro-block, we maintain a 32-byte accumulator (lo + hi
 * registers) for each side.  Each iter contributes its compressed
 * data to lo and hi via two place-shift TBLs (one for each half).
 * After 4 iters, we always emit exactly 2 vst1q_u8 stores per side
 * (lo, then hi), and advance the side's byte count by total*2.
 *
 * Stores per macro-block: 4 (2 per side).  Stores per iter: 1.
 * Baseline is 2 stores/iter.  So this saves 50% of stores. */
__attribute__((noinline))
static void bench_partition_coalesce_macro(const uint16_t *src, const uint8_t *bitmap,
                                            uint8_t *left_bytes, uint8_t *right_bytes,
                                            int n, int reps)
{
    const uint8x16_t zero_v = vdupq_n_u8(0);
    static const uint8_t iota_init[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    const uint8x16_t iota = vld1q_u8(iota_init);

    for (int r = 0; r < reps; r++) {
        int n_left_bytes = 0, n_right_bytes = 0;
        int j = 0;
        for (; j + 32 <= n; j += 32) {
            /* Read 4 masks */
            uint8_t m0 = bitmap[(j >> 3) + 0];
            uint8_t m1 = bitmap[(j >> 3) + 1];
            uint8_t m2 = bitmap[(j >> 3) + 2];
            uint8_t m3 = bitmap[(j >> 3) + 3];
            int pr0 = compress_popcnt[m0], pl0 = 8 - pr0;
            int pr1 = compress_popcnt[m1], pl1 = 8 - pr1;
            int pr2 = compress_popcnt[m2], pl2 = 8 - pr2;
            int pr3 = compress_popcnt[m3], pl3 = 8 - pr3;

            /* Right-side prefix sums (in elements; *2 for byte offsets) */
            int cr1 = pr0, cr2 = cr1 + pr1, cr3 = cr2 + pr2;
            int total_r = cr3 + pr3;
            /* Left-side prefix sums */
            int cl1 = pl0, cl2 = cl1 + pl1, cl3 = cl2 + pl2;
            int total_l = cl3 + pl3;

            /* Load 4 data registers (32 input elements = 64 bytes total) */
            uint8x16_t d0 = vld1q_u8((const uint8_t *)(src + j + 0));
            uint8x16_t d1 = vld1q_u8((const uint8_t *)(src + j + 8));
            uint8x16_t d2 = vld1q_u8((const uint8_t *)(src + j + 16));
            uint8x16_t d3 = vld1q_u8((const uint8_t *)(src + j + 24));

            /* Compress each iter's right + left side */
            const uint8_t *t0 = compress_tab[m0], *t1 = compress_tab[m1];
            const uint8_t *t2 = compress_tab[m2], *t3 = compress_tab[m3];
            uint8x16_t r0 = vqtbl1q_u8(d0, vld1q_u8(t0));
            uint8x16_t r1 = vqtbl1q_u8(d1, vld1q_u8(t1));
            uint8x16_t r2 = vqtbl1q_u8(d2, vld1q_u8(t2));
            uint8x16_t r3 = vqtbl1q_u8(d3, vld1q_u8(t3));
            uint8x16_t l0 = vqtbl1q_u8(d0, vld1q_u8(t0 + 16));
            uint8x16_t l1 = vqtbl1q_u8(d1, vld1q_u8(t1 + 16));
            uint8x16_t l2 = vqtbl1q_u8(d2, vld1q_u8(t2 + 16));
            uint8x16_t l3 = vqtbl1q_u8(d3, vld1q_u8(t3 + 16));

            /* Place each iter's compressed data into the (lo, hi) 32-byte
             * accumulator per side.  Iter 0 has cum=0 so no shift; iters 1-3
             * shift left by `cum*2` bytes via runtime-computed TBL.
             * `shuf_lo[i] = i - cum*2`  (mod 256, vqtbl1q maps >=16 → 0)
             * `shuf_hi[i] = i + 16 - cum*2` */
            #define PLACE(side_v, cum, lo_acc, hi_acc) do {                       \
                uint8x16_t _shuf_lo = vsubq_u8(iota, vdupq_n_u8((uint8_t)((cum) * 2))); \
                uint8x16_t _shuf_hi = vaddq_u8(iota, vdupq_n_u8((uint8_t)(16 - (cum) * 2))); \
                (lo_acc) = vorrq_u8((lo_acc), vqtbl1q_u8((side_v), _shuf_lo));    \
                (hi_acc) = vorrq_u8((hi_acc), vqtbl1q_u8((side_v), _shuf_hi));    \
            } while (0)

            uint8x16_t lo_r = r0, hi_r = zero_v;          /* iter 0: cum=0 */
            PLACE(r1, cr1, lo_r, hi_r);
            PLACE(r2, cr2, lo_r, hi_r);
            PLACE(r3, cr3, lo_r, hi_r);

            uint8x16_t lo_l = l0, hi_l = zero_v;
            PLACE(l1, cl1, lo_l, hi_l);
            PLACE(l2, cl2, lo_l, hi_l);
            PLACE(l3, cl3, lo_l, hi_l);
            #undef PLACE

            /* Always store both halves (32 bytes per side); advance by
             * the actual element count so the next macro-block overwrites
             * the don't-care tail. */
            vst1q_u8(right_bytes + n_right_bytes,      lo_r);
            vst1q_u8(right_bytes + n_right_bytes + 16, hi_r);
            n_right_bytes += total_r * 2;
            vst1q_u8(left_bytes  + n_left_bytes,       lo_l);
            vst1q_u8(left_bytes  + n_left_bytes + 16,  hi_l);
            n_left_bytes  += total_l  * 2;
        }
        /* Tail: small, ignore for the bench (n=8192 → 256 macro-blocks, no tail). */
    }
}

/* Same coalescing strategy but without the switch.  The shift
 * amount is computed at runtime via `vsubq_u8(iota, broadcast(so_far*2))`,
 * applied via a single TBL.  Only 1 conditional branch per side
 * per iter (the flush check).  With M4 TBL throughput at ~4/cycle
 * this avoids the indirect-branch mispredict storm of the switch
 * variant. */
__attribute__((noinline))
static void bench_partition_coalesce_tbl(const uint16_t *src, const uint8_t *bitmap,
                                          uint8_t *left_bytes, uint8_t *right_bytes,
                                          int n, int reps)
{
    const uint8x16_t zero_v = vdupq_n_u8(0);
    static const uint8_t iota_init[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    const uint8x16_t iota = vld1q_u8(iota_init);

    for (int r = 0; r < reps; r++) {
        uint8x16_t accum_l = zero_v, accum_r = zero_v;
        int so_far_l = 0, so_far_r = 0;
        int n_left_bytes = 0, n_right_bytes = 0;

        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            uint8x16_t r_v = vqtbl1q_u8(data, shuf_r);
            uint8x16_t l_v = vqtbl1q_u8(data, shuf_l);
            int cnt_r = compress_popcnt[mask];
            int cnt_l = 8 - cnt_r;

            /* Right side */
            {
                /* shift compacted left by so_far_r elements (= 2 bytes each):
                 * shuf[i] = i - so_far_r*2  (mod 256, vqtbl1q maps >=16 → 0) */
                uint8x16_t shuf_left = vsubq_u8(iota, vdupq_n_u8((uint8_t)(so_far_r * 2)));
                uint8x16_t shifted   = vqtbl1q_u8(r_v, shuf_left);
                uint8x16_t merged    = vorrq_u8(accum_r, shifted);
                int new_sf = so_far_r + cnt_r;
                if (new_sf >= 8) {
                    vst1q_u8(right_bytes + n_right_bytes, merged);
                    n_right_bytes += 16;
                    /* New accum = compacted right-shifted by (8-so_far_r) elements */
                    uint8x16_t shuf_rt = vaddq_u8(iota, vdupq_n_u8((uint8_t)((8 - so_far_r) * 2)));
                    accum_r = vqtbl1q_u8(r_v, shuf_rt);
                    so_far_r = new_sf - 8;
                } else {
                    accum_r = merged;
                    so_far_r = new_sf;
                }
            }
            /* Left side, same shape */
            {
                uint8x16_t shuf_left = vsubq_u8(iota, vdupq_n_u8((uint8_t)(so_far_l * 2)));
                uint8x16_t shifted   = vqtbl1q_u8(l_v, shuf_left);
                uint8x16_t merged    = vorrq_u8(accum_l, shifted);
                int new_sf = so_far_l + cnt_l;
                if (new_sf >= 8) {
                    vst1q_u8(left_bytes + n_left_bytes, merged);
                    n_left_bytes += 16;
                    uint8x16_t shuf_rt = vaddq_u8(iota, vdupq_n_u8((uint8_t)((8 - so_far_l) * 2)));
                    accum_l = vqtbl1q_u8(l_v, shuf_rt);
                    so_far_l = new_sf - 8;
                } else {
                    accum_l = merged;
                    so_far_l = new_sf;
                }
            }
        }

        if (so_far_l > 0) vst1q_u8(left_bytes  + n_left_bytes,  accum_l);
        if (so_far_r > 0) vst1q_u8(right_bytes + n_right_bytes, accum_r);
    }
}

__attribute__((noinline))
static void bench_partition_coalesce(const uint16_t *src, const uint8_t *bitmap,
                                      uint8_t *left_bytes, uint8_t *right_bytes,
                                      int n, int reps)
{
    const uint8x16_t zero_v = vdupq_n_u8(0);

    for (int r = 0; r < reps; r++) {
        uint8x16_t accum_l = zero_v, accum_r = zero_v;
        int so_far_l = 0, so_far_r = 0;
        int n_left_bytes = 0, n_right_bytes = 0;

        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            uint8x16_t r_v = vqtbl1q_u8(data, shuf_r);
            uint8x16_t l_v = vqtbl1q_u8(data, shuf_l);
            int cnt_r = compress_popcnt[mask];
            int cnt_l = 8 - cnt_r;

            COALESCE_SWITCH(r_v, cnt_r, accum_r, so_far_r, right_bytes, n_right_bytes)
            COALESCE_SWITCH(l_v, cnt_l, accum_l, so_far_l, left_bytes,  n_left_bytes)
        }

        /* End-of-rep tail: flush partial accums if anything remains.
         * Negligible cost at REPS=100k, but ensures we touch the tail
         * each rep so the compiler doesn't optimise the bookkeeping. */
        if (so_far_l > 0) vst1q_u8(left_bytes  + n_left_bytes,  accum_l);
        if (so_far_r > 0) vst1q_u8(right_bytes + n_right_bytes, accum_r);
    }
}
#endif /* 0 — coalesce variants moved to extras/bench/bench_coalesce.c */

/* Helper: fill a bitmap with `pct_full` percent of bytes = 0xFF
 * (popcount 8 → advance 8, no overlap) and the rest = 0x1F
 * (popcount 5 → advance 5, 6-byte overlap on right + 10-byte
 * overlap on left).  Lets the partition bench measure how store
 * overlap affects throughput. */
static void fill_bitmap_density(uint8_t *bitmap, int n_bytes, int pct_full)
{
    srand(0x1234);
    for (int i = 0; i < n_bytes; i++) {
        bitmap[i] = ((rand() % 100) < pct_full) ? 0xFF : 0x1F;
    }
}

/* Store-port hypothesis test: Apple M-series chips reportedly have 2
 * scalar store AGUs but only 1 SIMD store port.  These three kernels
 * all do the same partition work but with different store mixes:
 *
 *   simd_only:    2 × vst1q_u8 (16 B each) per iter — full partition, baseline
 *   scalar_only:  4 × vst1_u8 (8 B each) per iter   — same bytes, all scalar
 *   mixed:        1 × vst1q + 2 × vst1_u8 per iter  — 32 B total, mixed issue
 *
 * If the hypothesis holds, `mixed` issues all 3 stores in 1 cycle (1
 * SIMD + 2 scalar in parallel) and runs ~2x faster than baseline. */

__attribute__((noinline))
static void bench_partition_simd_only(const uint16_t *src, const uint8_t *bitmap,
                                       uint16_t *left, uint16_t *right,
                                       int n, int reps)
{
    /* Identical to bench_partition_neon — re-defined here for clarity
     * of comparison. */
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            uint8x16_t r_v = vqtbl1q_u8(data, shuf_r);
            uint8x16_t l_v = vqtbl1q_u8(data, shuf_l);
            vst1q_u8((uint8_t *)(right + n_right), r_v);
            vst1q_u8((uint8_t *)(left + n_left), l_v);
            n_right += compress_popcnt[mask];
            n_left += (8 - compress_popcnt[mask]);
        }
    }
}

/* Force the compiler to emit literal `str d` (8-byte scalar) stores
 * rather than fusing two adjacent ones into a single `str q` or
 * `stp d, d`.  Without `volatile` + explicit asm the optimizer
 * coalesces them and the test becomes a no-op. */
#define FORCE_STR_D_PAIR(ptr, lo64, hi64)                              \
    __asm__ volatile(                                                  \
        "str %d[a], [%[p]]\n\t"                                        \
        "str %d[b], [%[p], #8]\n\t"                                    \
        : : [a] "w"(lo64), [b] "w"(hi64), [p] "r"(ptr) : "memory")

__attribute__((noinline))
static void bench_partition_scalar_only(const uint16_t *src, const uint8_t *bitmap,
                                         uint16_t *left, uint16_t *right,
                                         int n, int reps)
{
    /* 4 × str d (8-byte scalar) per iter, forced via inline asm so
     * the compiler can't fold them back into str q's.  Same total
     * 32 bytes per iter as simd_only. */
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            uint8x16_t r_v = vqtbl1q_u8(data, shuf_r);
            uint8x16_t l_v = vqtbl1q_u8(data, shuf_l);
            uint8_t *rp = (uint8_t *)(right + n_right);
            uint8_t *lp = (uint8_t *)(left + n_left);
            FORCE_STR_D_PAIR(rp, vget_low_u8(r_v), vget_high_u8(r_v));
            FORCE_STR_D_PAIR(lp, vget_low_u8(l_v), vget_high_u8(l_v));
            n_right += compress_popcnt[mask];
            n_left += (8 - compress_popcnt[mask]);
        }
    }
}

__attribute__((noinline))
static void bench_partition_mixed(const uint16_t *src, const uint8_t *bitmap,
                                   uint16_t *left, uint16_t *right,
                                   int n, int reps)
{
    /* 1 × str q + 2 × str d per iter.  If M4 has 1 SIMD port + 2
     * scalar AGUs running in parallel, the 3 stores issue in 1
     * cycle.  If only 1 store dispatch slot exists regardless of
     * width, this should be slower than simd_only (3 vs 2 dispatches). */
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
            const uint8_t *tab = compress_tab[mask];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            uint8x16_t r_v = vqtbl1q_u8(data, shuf_r);
            uint8x16_t l_v = vqtbl1q_u8(data, shuf_l);
            vst1q_u8((uint8_t *)(right + n_right), r_v);
            uint8_t *lp = (uint8_t *)(left + n_left);
            FORCE_STR_D_PAIR(lp, vget_low_u8(l_v), vget_high_u8(l_v));
            n_right += compress_popcnt[mask];
            n_left += (8 - compress_popcnt[mask]);
        }
    }
}

/* ---- Both-leaves sequential vst1 ---- */

__attribute__((noinline)) static void bench_both_leaves_vst1(uint8_t *symbols, const uint8_t *bitmap,
                                    uint8_t sym0, uint8_t sym1,
                                    int n, int reps)
{
    uint8x8_t vsym0 = vdup_n_u8(sym0);
    uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);
    static const uint8_t bpt[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbp = vld1_u8(bpt);

    for (int r = 0; r < reps; r++) {
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8x8_t bits = vtst_u8(vdup_n_u8(bitmap[j >> 3]), vbp);
            uint8x8_t vals = veor_u8(vsym0, vand_u8(vdelta, bits));
            vst1_u8(symbols + j, vals);
        }
    }
}

/* ---- Both-leaves scattered stores ---- */

__attribute__((noinline)) static void bench_both_leaves_scatter(uint8_t *symbols, const uint16_t *indices,
                                       const uint8_t *bitmap,
                                       uint8_t sym0, uint8_t sym1,
                                       int n, int reps)
{
    uint8x8_t vsym0 = vdup_n_u8(sym0);
    uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);
    static const uint8_t bpt[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbp = vld1_u8(bpt);

    for (int r = 0; r < reps; r++) {
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8x8_t bits = vtst_u8(vdup_n_u8(bitmap[j >> 3]), vbp);
            uint8x8_t vals = veor_u8(vsym0, vand_u8(vdelta, bits));
            uint16x8_t idx = vld1q_u16(indices + j);
            symbols[vgetq_lane_u16(idx, 0)] = vget_lane_u8(vals, 0);
            symbols[vgetq_lane_u16(idx, 1)] = vget_lane_u8(vals, 1);
            symbols[vgetq_lane_u16(idx, 2)] = vget_lane_u8(vals, 2);
            symbols[vgetq_lane_u16(idx, 3)] = vget_lane_u8(vals, 3);
            symbols[vgetq_lane_u16(idx, 4)] = vget_lane_u8(vals, 4);
            symbols[vgetq_lane_u16(idx, 5)] = vget_lane_u8(vals, 5);
            symbols[vgetq_lane_u16(idx, 6)] = vget_lane_u8(vals, 6);
            symbols[vgetq_lane_u16(idx, 7)] = vget_lane_u8(vals, 7);
        }
    }
}
/* ============================================================
 * Flat-subtree decode microbench (per D).
 *
 * The production flat-subtree decoder reads N consecutive D-bit codes
 * from a packed bitstream, looks up a per-leaf c2s table (2^D entries),
 * and writes the resulting symbols.  Two output flavours:
 *   _direct  — sequential vst1q (root-flat path: indices are identity).
 *   _scatter — write via indices[]: simulates a non-root flat subtree.
 *
 * These benchmarks isolate (unpack + TBL + store) per element at each
 * D, so we can compare their per-element cost against the
 * scatter/partition floors above.
 * ============================================================ */

/* unpack() helpers + their tables come from pivco_huffman_neon_flat.h
 * (shared with src/pivco_huffman_neon.c — single source of truth). */

/* ---- D=2: 4 packed bytes → 16 codes (uint8x16_t), 1 vqtbl1q_u8 ---- */
__attribute__((noinline))
static void bench_flat_direct_d2(uint8_t *out, const uint8_t *bm,
                                  const uint8_t *c2s, int n, int reps) {
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d2_unpack(bm + (i >> 2));
            vst1q_u8(out + i, vqtbl1q_u8(c2s_vec, codes));
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d2(uint8_t *out, const uint16_t *idx,
                                   const uint8_t *bm, const uint8_t *c2s,
                                   int n, int reps) {
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d2_unpack(bm + (i >> 2));
            uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
            out[idx[i +  0]] = vgetq_lane_u8(syms,  0);
            out[idx[i +  1]] = vgetq_lane_u8(syms,  1);
            out[idx[i +  2]] = vgetq_lane_u8(syms,  2);
            out[idx[i +  3]] = vgetq_lane_u8(syms,  3);
            out[idx[i +  4]] = vgetq_lane_u8(syms,  4);
            out[idx[i +  5]] = vgetq_lane_u8(syms,  5);
            out[idx[i +  6]] = vgetq_lane_u8(syms,  6);
            out[idx[i +  7]] = vgetq_lane_u8(syms,  7);
            out[idx[i +  8]] = vgetq_lane_u8(syms,  8);
            out[idx[i +  9]] = vgetq_lane_u8(syms,  9);
            out[idx[i + 10]] = vgetq_lane_u8(syms, 10);
            out[idx[i + 11]] = vgetq_lane_u8(syms, 11);
            out[idx[i + 12]] = vgetq_lane_u8(syms, 12);
            out[idx[i + 13]] = vgetq_lane_u8(syms, 13);
            out[idx[i + 14]] = vgetq_lane_u8(syms, 14);
            out[idx[i + 15]] = vgetq_lane_u8(syms, 15);
        }
    }
}

/* ---- D=3: 3 packed bytes → 8 codes (uint8x8_t), 1 vqtbl1_u8 ---- */
__attribute__((noinline))
static void bench_flat_direct_d3(uint8_t *out, const uint8_t *bm,
                                  const uint8_t *c2s, int n, int reps) {
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t lo = flat_d3_unpack_safe(bm + (((i)      * 3) >> 3));
            uint8x8_t hi = flat_d3_unpack_safe(bm + (((i + 8)  * 3) >> 3));
            uint8x16_t codes = vcombine_u8(lo, hi);
            vst1q_u8(out + i, vqtbl1q_u8(c2s_vec, codes));
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d3(uint8_t *out, const uint16_t *idx,
                                   const uint8_t *bm, const uint8_t *c2s,
                                   int n, int reps) {
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d3_unpack_safe(bm + ((i * 3) >> 3));
            uint8x8_t syms  = vqtbl1_u8(c2s_vec, codes);
            out[idx[i+0]] = vget_lane_u8(syms, 0);
            out[idx[i+1]] = vget_lane_u8(syms, 1);
            out[idx[i+2]] = vget_lane_u8(syms, 2);
            out[idx[i+3]] = vget_lane_u8(syms, 3);
            out[idx[i+4]] = vget_lane_u8(syms, 4);
            out[idx[i+5]] = vget_lane_u8(syms, 5);
            out[idx[i+6]] = vget_lane_u8(syms, 6);
            out[idx[i+7]] = vget_lane_u8(syms, 7);
        }
    }
}

/* ---- D=4: 8 packed bytes → 16 codes (uint8x16_t), 1 vqtbl1q_u8 ---- */
__attribute__((noinline))
static void bench_flat_direct_d4(uint8_t *out, const uint8_t *bm,
                                  const uint8_t *c2s, int n, int reps) {
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d4_unpack(bm + (i >> 1));
            vst1q_u8(out + i, vqtbl1q_u8(c2s_vec, codes));
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d4(uint8_t *out, const uint16_t *idx,
                                   const uint8_t *bm, const uint8_t *c2s,
                                   int n, int reps) {
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d4_unpack(bm + (i >> 1));
            uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
            out[idx[i +  0]] = vgetq_lane_u8(syms,  0);
            out[idx[i +  1]] = vgetq_lane_u8(syms,  1);
            out[idx[i +  2]] = vgetq_lane_u8(syms,  2);
            out[idx[i +  3]] = vgetq_lane_u8(syms,  3);
            out[idx[i +  4]] = vgetq_lane_u8(syms,  4);
            out[idx[i +  5]] = vgetq_lane_u8(syms,  5);
            out[idx[i +  6]] = vgetq_lane_u8(syms,  6);
            out[idx[i +  7]] = vgetq_lane_u8(syms,  7);
            out[idx[i +  8]] = vgetq_lane_u8(syms,  8);
            out[idx[i +  9]] = vgetq_lane_u8(syms,  9);
            out[idx[i + 10]] = vgetq_lane_u8(syms, 10);
            out[idx[i + 11]] = vgetq_lane_u8(syms, 11);
            out[idx[i + 12]] = vgetq_lane_u8(syms, 12);
            out[idx[i + 13]] = vgetq_lane_u8(syms, 13);
            out[idx[i + 14]] = vgetq_lane_u8(syms, 14);
            out[idx[i + 15]] = vgetq_lane_u8(syms, 15);
        }
    }
}

/* ---- D=5: 5 bytes → 8 codes (uint8x8_t), c2s 32B → vqtbl2_u8 ----
 *
 * c2s sits in two 16-byte regs.  On Apple silicon vqtbl2_u8 is fast;
 * on Neoverse-V2 it's measurably slower (production gates this off
 * for K=5/6).  This bench measures the SIMD path unconditionally so
 * we see the per-platform cost. */
__attribute__((noinline))
static void bench_flat_direct_d5(uint8_t *out, const uint8_t *bm,
                                  const uint8_t *c2s, int n, int reps) {
    uint8x16x2_t c2s_vec = {{vld1q_u8(c2s), vld1q_u8(c2s + 16)}};
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t lo = flat_d5_unpack_safe(bm + (((i)     * 5) >> 3));
            uint8x8_t hi = flat_d5_unpack_safe(bm + (((i + 8) * 5) >> 3));
            uint8x16_t codes = vcombine_u8(lo, hi);
            vst1q_u8(out + i, vqtbl2q_u8(c2s_vec, codes));
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d5(uint8_t *out, const uint16_t *idx,
                                   const uint8_t *bm, const uint8_t *c2s,
                                   int n, int reps) {
    uint8x16x2_t c2s_vec = {{vld1q_u8(c2s), vld1q_u8(c2s + 16)}};
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d5_unpack_safe(bm + ((i * 5) >> 3));
            uint8x8_t syms  = vqtbl2_u8(c2s_vec, codes);
            out[idx[i+0]] = vget_lane_u8(syms, 0);
            out[idx[i+1]] = vget_lane_u8(syms, 1);
            out[idx[i+2]] = vget_lane_u8(syms, 2);
            out[idx[i+3]] = vget_lane_u8(syms, 3);
            out[idx[i+4]] = vget_lane_u8(syms, 4);
            out[idx[i+5]] = vget_lane_u8(syms, 5);
            out[idx[i+6]] = vget_lane_u8(syms, 6);
            out[idx[i+7]] = vget_lane_u8(syms, 7);
        }
    }
}

/* ---- D=6: 6 bytes → 8 codes (uint8x8_t), c2s 64B → vqtbl4_u8 ---- */
__attribute__((noinline))
static void bench_flat_direct_d6(uint8_t *out, const uint8_t *bm,
                                  const uint8_t *c2s, int n, int reps) {
    uint8x16x4_t c2s_vec = {{vld1q_u8(c2s),     vld1q_u8(c2s+16),
                              vld1q_u8(c2s+32),  vld1q_u8(c2s+48)}};
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x8_t lo = flat_d6_unpack_safe(bm + (((i)     * 6) >> 3));
            uint8x8_t hi = flat_d6_unpack_safe(bm + (((i + 8) * 6) >> 3));
            uint8x16_t codes = vcombine_u8(lo, hi);
            vst1q_u8(out + i, vqtbl4q_u8(c2s_vec, codes));
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d6(uint8_t *out, const uint16_t *idx,
                                   const uint8_t *bm, const uint8_t *c2s,
                                   int n, int reps) {
    uint8x16x4_t c2s_vec = {{vld1q_u8(c2s),     vld1q_u8(c2s+16),
                              vld1q_u8(c2s+32),  vld1q_u8(c2s+48)}};
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 8 <= n; i += 8) {
            uint8x8_t codes = flat_d6_unpack_safe(bm + ((i * 6) >> 3));
            uint8x8_t syms  = vqtbl4_u8(c2s_vec, codes);
            out[idx[i+0]] = vget_lane_u8(syms, 0);
            out[idx[i+1]] = vget_lane_u8(syms, 1);
            out[idx[i+2]] = vget_lane_u8(syms, 2);
            out[idx[i+3]] = vget_lane_u8(syms, 3);
            out[idx[i+4]] = vget_lane_u8(syms, 4);
            out[idx[i+5]] = vget_lane_u8(syms, 5);
            out[idx[i+6]] = vget_lane_u8(syms, 6);
            out[idx[i+7]] = vget_lane_u8(syms, 7);
        }
    }
}

#endif /* HAS_NEON */

/* ============================================================
 * AVX-512 VBMI2 backend (Intel Xeon Sapphire/Granite Rapids).
 *
 * partition primitive: 32-wide vpcompressw (one mask → packed lane order).
 * flat-decode TBL: pshufb (D=2/3/4), vpermb-ymm (D=5), vpermb-zmm (D=6).
 *
 * Lane-extract macro: AVX-512 has no per-lane gather store, so leaf
 * scatter writes go through scalar lane extracts (matches production).
 * ============================================================ */
#if HAS_AVX512

#define X16(syms, idx, base)                                                  \
    out[(idx)[(base) +  0]] = (uint8_t)_mm_extract_epi8((syms),  0);          \
    out[(idx)[(base) +  1]] = (uint8_t)_mm_extract_epi8((syms),  1);          \
    out[(idx)[(base) +  2]] = (uint8_t)_mm_extract_epi8((syms),  2);          \
    out[(idx)[(base) +  3]] = (uint8_t)_mm_extract_epi8((syms),  3);          \
    out[(idx)[(base) +  4]] = (uint8_t)_mm_extract_epi8((syms),  4);          \
    out[(idx)[(base) +  5]] = (uint8_t)_mm_extract_epi8((syms),  5);          \
    out[(idx)[(base) +  6]] = (uint8_t)_mm_extract_epi8((syms),  6);          \
    out[(idx)[(base) +  7]] = (uint8_t)_mm_extract_epi8((syms),  7);          \
    out[(idx)[(base) +  8]] = (uint8_t)_mm_extract_epi8((syms),  8);          \
    out[(idx)[(base) +  9]] = (uint8_t)_mm_extract_epi8((syms),  9);          \
    out[(idx)[(base) + 10]] = (uint8_t)_mm_extract_epi8((syms), 10);          \
    out[(idx)[(base) + 11]] = (uint8_t)_mm_extract_epi8((syms), 11);          \
    out[(idx)[(base) + 12]] = (uint8_t)_mm_extract_epi8((syms), 12);          \
    out[(idx)[(base) + 13]] = (uint8_t)_mm_extract_epi8((syms), 13);          \
    out[(idx)[(base) + 14]] = (uint8_t)_mm_extract_epi8((syms), 14);          \
    out[(idx)[(base) + 15]] = (uint8_t)_mm_extract_epi8((syms), 15);

/* Note: there is no AVX-512 byte-scatter advantage worth measuring as a
 * standalone primitive — `_mm512_i32scatter_epi8` only exists on
 * AVX-512BW + spec-VL, and production uses scalar lane extracts
 * (matches the scalar scatter floor).  Use `scatter_scalar` as the
 * comparable scatter row on x86_64. */

/* Diagnostic: the AVX-512 "SIMD" scatter pattern used in extras/ph-td
 * (scatter_write_avx512): load 8 uint16_t indices via SIMD, then
 * _mm_extract_epi16 each + store sym.  We expect this to be slower
 * than plain scalar — pextrw goes through port 5 single-issue while
 * scalar movzwl can issue across multiple load ports. */
__attribute__((noinline))
static void bench_scatter_avx512_pextrw(uint8_t *symbols,
                                          const uint16_t *indices,
                                          int n, uint8_t sym, int reps)
{
    for (int r = 0; r < reps; r++) {
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            __m128i idx = _mm_loadu_si128((const __m128i *)(indices + j));
            symbols[_mm_extract_epi16(idx, 0)] = sym;
            symbols[_mm_extract_epi16(idx, 1)] = sym;
            symbols[_mm_extract_epi16(idx, 2)] = sym;
            symbols[_mm_extract_epi16(idx, 3)] = sym;
            symbols[_mm_extract_epi16(idx, 4)] = sym;
            symbols[_mm_extract_epi16(idx, 5)] = sym;
            symbols[_mm_extract_epi16(idx, 6)] = sym;
            symbols[_mm_extract_epi16(idx, 7)] = sym;
        }
        for (; j < n; j++) symbols[indices[j]] = sym;
    }
}

__attribute__((noinline))
static void bench_partition_avx512(const uint16_t *src,
                                    const uint32_t *masks32,
                                    uint16_t *left, uint16_t *right,
                                    int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 32 <= n; j += 32) {
            __m512i data = _mm512_loadu_si512((const __m512i *)(src + j));
            __mmask32 mask = (__mmask32)masks32[j >> 5];
            __m512i r_v = _mm512_maskz_compress_epi16(mask, data);
            __m512i l_v = _mm512_maskz_compress_epi16(~mask, data);
            int nr = _mm_popcnt_u32((uint32_t)mask);
            _mm512_storeu_si512((__m512i *)(right + n_right), r_v);
            _mm512_storeu_si512((__m512i *)(left  + n_left ), l_v);
            n_right += nr;
            n_left  += 32 - nr;
        }
    }
}

/* Identity-gen variant: no index load, generate base + lane offset on the fly. */
__attribute__((noinline))
static void bench_partition_root_avx512(const uint32_t *masks32,
                                         uint16_t *left, uint16_t *right,
                                         int n, int reps)
{
    static const int16_t lane_off_arr[32] __attribute__((aligned(64))) = {
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    const __m512i lane_off = _mm512_loadu_si512((const __m512i *)lane_off_arr);
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 32 <= n; j += 32) {
            __m512i data = _mm512_add_epi16(_mm512_set1_epi16((short)j), lane_off);
            __mmask32 mask = (__mmask32)masks32[j >> 5];
            __m512i r_v = _mm512_maskz_compress_epi16(mask, data);
            __m512i l_v = _mm512_maskz_compress_epi16(~mask, data);
            int nr = _mm_popcnt_u32((uint32_t)mask);
            _mm512_storeu_si512((__m512i *)(right + n_right), r_v);
            _mm512_storeu_si512((__m512i *)(left  + n_left ), l_v);
            n_right += nr;
            n_left  += 32 - nr;
        }
    }
}

/* Half-tree variant: only emit the right side. */
__attribute__((noinline))
static void bench_partition_half_avx512(const uint16_t *src,
                                         const uint32_t *masks32,
                                         uint16_t *right,
                                         int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_right = 0;
        for (int j = 0; j + 32 <= n; j += 32) {
            __m512i data = _mm512_loadu_si512((const __m512i *)(src + j));
            __mmask32 mask = (__mmask32)masks32[j >> 5];
            __m512i r_v = _mm512_maskz_compress_epi16(mask, data);
            int nr = _mm_popcnt_u32((uint32_t)mask);
            _mm512_storeu_si512((__m512i *)(right + n_right), r_v);
            n_right += nr;
        }
    }
}

__attribute__((noinline))
static void bench_partition_root_half_avx512(const uint32_t *masks32,
                                              uint16_t *right,
                                              int n, int reps)
{
    static const int16_t lane_off_arr[32] __attribute__((aligned(64))) = {
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    const __m512i lane_off = _mm512_loadu_si512((const __m512i *)lane_off_arr);
    for (int r = 0; r < reps; r++) {
        int n_right = 0;
        for (int j = 0; j + 32 <= n; j += 32) {
            __m512i data = _mm512_add_epi16(_mm512_set1_epi16((short)j), lane_off);
            __mmask32 mask = (__mmask32)masks32[j >> 5];
            __m512i r_v = _mm512_maskz_compress_epi16(mask, data);
            int nr = _mm_popcnt_u32((uint32_t)mask);
            _mm512_storeu_si512((__m512i *)(right + n_right), r_v);
            n_right += nr;
        }
    }
}

/* Both-leaves: 32 bits of bitmap → 32-byte mask-blend → sequential store. */
__attribute__((noinline))
static void bench_both_leaves_vst1_avx512(uint8_t *symbols,
                                           const uint32_t *masks32,
                                           uint8_t sym0, uint8_t sym1,
                                           int n, int reps)
{
    __m256i s0 = _mm256_set1_epi8((char)sym0);
    __m256i s1 = _mm256_set1_epi8((char)sym1);
    for (int r = 0; r < reps; r++) {
        for (int j = 0; j + 32 <= n; j += 32) {
            __mmask32 m = (__mmask32)masks32[j >> 5];
            __m256i v = _mm256_mask_blend_epi8(m, s0, s1);
            _mm256_storeu_si256((__m256i *)(symbols + j), v);
        }
    }
}

/* Both-leaves indexed: 32-byte blend then 32 indexed lane-byte stores. */
__attribute__((noinline))
static void bench_both_leaves_scatter_avx512(uint8_t *symbols,
                                              const uint16_t *indices,
                                              const uint32_t *masks32,
                                              uint8_t sym0, uint8_t sym1,
                                              int n, int reps)
{
    __m256i s0 = _mm256_set1_epi8((char)sym0);
    __m256i s1 = _mm256_set1_epi8((char)sym1);
    for (int r = 0; r < reps; r++) {
        for (int j = 0; j + 32 <= n; j += 32) {
            __mmask32 m = (__mmask32)masks32[j >> 5];
            __m256i v = _mm256_mask_blend_epi8(m, s0, s1);
            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);
            symbols[indices[j +  0]] = (uint8_t)_mm_extract_epi8(lo,  0);
            symbols[indices[j +  1]] = (uint8_t)_mm_extract_epi8(lo,  1);
            symbols[indices[j +  2]] = (uint8_t)_mm_extract_epi8(lo,  2);
            symbols[indices[j +  3]] = (uint8_t)_mm_extract_epi8(lo,  3);
            symbols[indices[j +  4]] = (uint8_t)_mm_extract_epi8(lo,  4);
            symbols[indices[j +  5]] = (uint8_t)_mm_extract_epi8(lo,  5);
            symbols[indices[j +  6]] = (uint8_t)_mm_extract_epi8(lo,  6);
            symbols[indices[j +  7]] = (uint8_t)_mm_extract_epi8(lo,  7);
            symbols[indices[j +  8]] = (uint8_t)_mm_extract_epi8(lo,  8);
            symbols[indices[j +  9]] = (uint8_t)_mm_extract_epi8(lo,  9);
            symbols[indices[j + 10]] = (uint8_t)_mm_extract_epi8(lo, 10);
            symbols[indices[j + 11]] = (uint8_t)_mm_extract_epi8(lo, 11);
            symbols[indices[j + 12]] = (uint8_t)_mm_extract_epi8(lo, 12);
            symbols[indices[j + 13]] = (uint8_t)_mm_extract_epi8(lo, 13);
            symbols[indices[j + 14]] = (uint8_t)_mm_extract_epi8(lo, 14);
            symbols[indices[j + 15]] = (uint8_t)_mm_extract_epi8(lo, 15);
            symbols[indices[j + 16]] = (uint8_t)_mm_extract_epi8(hi,  0);
            symbols[indices[j + 17]] = (uint8_t)_mm_extract_epi8(hi,  1);
            symbols[indices[j + 18]] = (uint8_t)_mm_extract_epi8(hi,  2);
            symbols[indices[j + 19]] = (uint8_t)_mm_extract_epi8(hi,  3);
            symbols[indices[j + 20]] = (uint8_t)_mm_extract_epi8(hi,  4);
            symbols[indices[j + 21]] = (uint8_t)_mm_extract_epi8(hi,  5);
            symbols[indices[j + 22]] = (uint8_t)_mm_extract_epi8(hi,  6);
            symbols[indices[j + 23]] = (uint8_t)_mm_extract_epi8(hi,  7);
            symbols[indices[j + 24]] = (uint8_t)_mm_extract_epi8(hi,  8);
            symbols[indices[j + 25]] = (uint8_t)_mm_extract_epi8(hi,  9);
            symbols[indices[j + 26]] = (uint8_t)_mm_extract_epi8(hi, 10);
            symbols[indices[j + 27]] = (uint8_t)_mm_extract_epi8(hi, 11);
            symbols[indices[j + 28]] = (uint8_t)_mm_extract_epi8(hi, 12);
            symbols[indices[j + 29]] = (uint8_t)_mm_extract_epi8(hi, 13);
            symbols[indices[j + 30]] = (uint8_t)_mm_extract_epi8(hi, 14);
            symbols[indices[j + 31]] = (uint8_t)_mm_extract_epi8(hi, 15);
        }
    }
}

/* ---- D=2 (pshufb on 4-byte c2s) ---- */
__attribute__((noinline))
static void bench_flat_direct_d2_avx512(uint8_t *out, const uint8_t *bm,
                                         const uint8_t *c2s, int n, int reps)
{
    uint32_t c2s_lo;
    memcpy(&c2s_lo, c2s, 4);
    __m128i c2s_vec = _mm_set1_epi32((int32_t)c2s_lo);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d2_unpack_avx512(bm + (i >> 2));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(out + i), syms);
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d2_avx512(uint8_t *out, const uint16_t *idx,
                                          const uint8_t *bm, const uint8_t *c2s,
                                          int n, int reps)
{
    uint32_t c2s_lo;
    memcpy(&c2s_lo, c2s, 4);
    __m128i c2s_vec = _mm_set1_epi32((int32_t)c2s_lo);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d2_unpack_avx512(bm + (i >> 2));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            X16(syms, idx, i)
        }
    }
}

/* ---- D=3 (pshufb on 8-byte c2s) ---- */
__attribute__((noinline))
static void bench_flat_direct_d3_avx512(uint8_t *out, const uint8_t *bm,
                                         const uint8_t *c2s, int n, int reps)
{
    uint64_t c2s_lo;
    memcpy(&c2s_lo, c2s, 8);
    __m128i c2s_vec = _mm_cvtsi64_si128((int64_t)c2s_lo);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(out + i), syms);
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d3_avx512(uint8_t *out, const uint16_t *idx,
                                          const uint8_t *bm, const uint8_t *c2s,
                                          int n, int reps)
{
    uint64_t c2s_lo;
    memcpy(&c2s_lo, c2s, 8);
    __m128i c2s_vec = _mm_cvtsi64_si128((int64_t)c2s_lo);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            X16(syms, idx, i)
        }
    }
}

/* ---- D=4 (pshufb on 16-byte c2s) ---- */
__attribute__((noinline))
static void bench_flat_direct_d4_avx512(uint8_t *out, const uint8_t *bm,
                                         const uint8_t *c2s, int n, int reps)
{
    __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_avx512(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(out + i), syms);
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d4_avx512(uint8_t *out, const uint16_t *idx,
                                          const uint8_t *bm, const uint8_t *c2s,
                                          int n, int reps)
{
    __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_avx512(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            X16(syms, idx, i)
        }
    }
}

/* ---- D=5 (vpermb on 32-byte ymm c2s) ---- */
__attribute__((noinline))
static void bench_flat_direct_d5_avx512(uint8_t *out, const uint8_t *bm,
                                         const uint8_t *c2s, int n, int reps)
{
    __m256i c2s_vec = _mm256_loadu_si256((const __m256i *)c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3));
            __m256i ext   = _mm256_zextsi128_si256(codes);
            __m256i full  = _mm256_permutexvar_epi8(ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(out + i), _mm256_castsi256_si128(full));
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d5_avx512(uint8_t *out, const uint16_t *idx,
                                          const uint8_t *bm, const uint8_t *c2s,
                                          int n, int reps)
{
    __m256i c2s_vec = _mm256_loadu_si256((const __m256i *)c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3));
            __m256i ext   = _mm256_zextsi128_si256(codes);
            __m256i full  = _mm256_permutexvar_epi8(ext, c2s_vec);
            __m128i syms  = _mm256_castsi256_si128(full);
            X16(syms, idx, i)
        }
    }
}

/* ---- D=6 (vpermb on 64-byte zmm c2s) ---- */
__attribute__((noinline))
static void bench_flat_direct_d6_avx512(uint8_t *out, const uint8_t *bm,
                                         const uint8_t *c2s, int n, int reps)
{
    __m512i c2s_vec = _mm512_loadu_si512((const __m512i *)c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3));
            __m512i ext   = _mm512_castsi128_si512(codes);
            __m512i full  = _mm512_permutexvar_epi8(ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(out + i), _mm512_castsi512_si128(full));
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d6_avx512(uint8_t *out, const uint16_t *idx,
                                          const uint8_t *bm, const uint8_t *c2s,
                                          int n, int reps)
{
    __m512i c2s_vec = _mm512_loadu_si512((const __m512i *)c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3));
            __m512i ext   = _mm512_castsi128_si512(codes);
            __m512i full  = _mm512_permutexvar_epi8(ext, c2s_vec);
            __m128i syms  = _mm512_castsi512_si128(full);
            X16(syms, idx, i)
        }
    }
}

#undef X16
#endif /* HAS_AVX512 */

/* ============================================================
 * SSE4.1 backend (AMD Zen 3, older Intel without AVX-512).
 *
 * partition primitive: 8-wide pshufb via compress_tab (same layout as
 * the NEON path).  Only D=4 has a SIMD flat-decode under pure SSE4.1
 * (no per-byte variable shift / no vpmultishiftqb); D=2/3/5/6 fall
 * through to scalar in production.
 * ============================================================ */
#if HAS_SSE4

/* Local compress shuffle table; same shape as the NEON one. */
static uint8_t compress_tab_sse[256][32] __attribute__((aligned(32)));
static uint8_t compress_popcnt_sse[256] __attribute__((aligned(64)));
static int     compress_table_sse_ready = 0;

static void init_compress_table_sse(void)
{
    if (compress_table_sse_ready) return;
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0;
        for (int i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                compress_tab_sse[mask][out_r * 2]     = (uint8_t)(i * 2);
                compress_tab_sse[mask][out_r * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_r++;
            }
        }
        compress_popcnt_sse[mask] = (uint8_t)out_r;
        for (int j = out_r * 2; j < 16; j++) compress_tab_sse[mask][j] = 0x80;
        int out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (!(mask & (1 << i))) {
                compress_tab_sse[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab_sse[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_l * 2; j < 16; j++) compress_tab_sse[mask][16 + j] = 0x80;
    }
    compress_table_sse_ready = 1;
}

#define X16_SSE(syms, idx, base)                                              \
    out[(idx)[(base) +  0]] = (uint8_t)_mm_extract_epi8((syms),  0);          \
    out[(idx)[(base) +  1]] = (uint8_t)_mm_extract_epi8((syms),  1);          \
    out[(idx)[(base) +  2]] = (uint8_t)_mm_extract_epi8((syms),  2);          \
    out[(idx)[(base) +  3]] = (uint8_t)_mm_extract_epi8((syms),  3);          \
    out[(idx)[(base) +  4]] = (uint8_t)_mm_extract_epi8((syms),  4);          \
    out[(idx)[(base) +  5]] = (uint8_t)_mm_extract_epi8((syms),  5);          \
    out[(idx)[(base) +  6]] = (uint8_t)_mm_extract_epi8((syms),  6);          \
    out[(idx)[(base) +  7]] = (uint8_t)_mm_extract_epi8((syms),  7);          \
    out[(idx)[(base) +  8]] = (uint8_t)_mm_extract_epi8((syms),  8);          \
    out[(idx)[(base) +  9]] = (uint8_t)_mm_extract_epi8((syms),  9);          \
    out[(idx)[(base) + 10]] = (uint8_t)_mm_extract_epi8((syms), 10);          \
    out[(idx)[(base) + 11]] = (uint8_t)_mm_extract_epi8((syms), 11);          \
    out[(idx)[(base) + 12]] = (uint8_t)_mm_extract_epi8((syms), 12);          \
    out[(idx)[(base) + 13]] = (uint8_t)_mm_extract_epi8((syms), 13);          \
    out[(idx)[(base) + 14]] = (uint8_t)_mm_extract_epi8((syms), 14);          \
    out[(idx)[(base) + 15]] = (uint8_t)_mm_extract_epi8((syms), 15);

__attribute__((noinline))
static void bench_scatter_sse(uint8_t *symbols, const uint16_t *indices,
                               int n, uint8_t sym, int reps)
{
    for (int r = 0; r < reps; r++) {
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            __m128i idx = _mm_loadu_si128((const __m128i *)(indices + j));
            symbols[(uint16_t)_mm_extract_epi16(idx, 0)] = sym;
            symbols[(uint16_t)_mm_extract_epi16(idx, 1)] = sym;
            symbols[(uint16_t)_mm_extract_epi16(idx, 2)] = sym;
            symbols[(uint16_t)_mm_extract_epi16(idx, 3)] = sym;
            symbols[(uint16_t)_mm_extract_epi16(idx, 4)] = sym;
            symbols[(uint16_t)_mm_extract_epi16(idx, 5)] = sym;
            symbols[(uint16_t)_mm_extract_epi16(idx, 6)] = sym;
            symbols[(uint16_t)_mm_extract_epi16(idx, 7)] = sym;
        }
        for (; j < n; j++) symbols[indices[j]] = sym;
    }
}

__attribute__((noinline))
static void bench_partition_sse(const uint16_t *src, const uint8_t *bitmap,
                                 uint16_t *left, uint16_t *right,
                                 int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            __m128i data = _mm_loadu_si128((const __m128i *)(src + j));
            uint8_t mask = bitmap[j >> 3];
            const uint8_t *tab = compress_tab_sse[mask];
            __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
            __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));
            __m128i r_v = _mm_shuffle_epi8(data, shuf_r);
            __m128i l_v = _mm_shuffle_epi8(data, shuf_l);
            int nr = compress_popcnt_sse[mask];
            _mm_storeu_si128((__m128i *)(right + n_right), r_v);
            _mm_storeu_si128((__m128i *)(left + n_left), l_v);
            n_right += nr;
            n_left  += 8 - nr;
        }
    }
}

__attribute__((noinline))
static void bench_partition_root_sse(const uint8_t *bitmap,
                                      uint16_t *left, uint16_t *right,
                                      int n, int reps)
{
    const __m128i lane_off = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            __m128i data = _mm_add_epi16(_mm_set1_epi16((short)j), lane_off);
            uint8_t mask = bitmap[j >> 3];
            const uint8_t *tab = compress_tab_sse[mask];
            __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
            __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));
            __m128i r_v = _mm_shuffle_epi8(data, shuf_r);
            __m128i l_v = _mm_shuffle_epi8(data, shuf_l);
            int nr = compress_popcnt_sse[mask];
            _mm_storeu_si128((__m128i *)(right + n_right), r_v);
            _mm_storeu_si128((__m128i *)(left + n_left), l_v);
            n_right += nr;
            n_left  += 8 - nr;
        }
    }
}

__attribute__((noinline))
static void bench_partition_half_sse(const uint16_t *src, const uint8_t *bitmap,
                                      uint16_t *right, int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            __m128i data = _mm_loadu_si128((const __m128i *)(src + j));
            uint8_t mask = bitmap[j >> 3];
            __m128i shuf_r = _mm_load_si128((const __m128i *)compress_tab_sse[mask]);
            __m128i r_v = _mm_shuffle_epi8(data, shuf_r);
            _mm_storeu_si128((__m128i *)(right + n_right), r_v);
            n_right += compress_popcnt_sse[mask];
        }
    }
}

__attribute__((noinline))
static void bench_partition_root_half_sse(const uint8_t *bitmap,
                                           uint16_t *right, int n, int reps)
{
    const __m128i lane_off = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
    for (int r = 0; r < reps; r++) {
        int n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            __m128i data = _mm_add_epi16(_mm_set1_epi16((short)j), lane_off);
            uint8_t mask = bitmap[j >> 3];
            __m128i shuf_r = _mm_load_si128((const __m128i *)compress_tab_sse[mask]);
            __m128i r_v = _mm_shuffle_epi8(data, shuf_r);
            _mm_storeu_si128((__m128i *)(right + n_right), r_v);
            n_right += compress_popcnt_sse[mask];
        }
    }
}

/* Both-leaves: 8 bits of mask → 8-byte mask-blend → 8-byte sequential store. */
__attribute__((noinline))
static void bench_both_leaves_vst1_sse(uint8_t *symbols, const uint8_t *bitmap,
                                        uint8_t sym0, uint8_t sym1,
                                        int n, int reps)
{
    __m128i s0 = _mm_set1_epi8((char)sym0);
    __m128i s1 = _mm_set1_epi8((char)sym1);
    const __m128i bpt = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)128,
                                       0, 0, 0, 0, 0, 0, 0, 0);
    for (int r = 0; r < reps; r++) {
        for (int j = 0; j + 8 <= n; j += 8) {
            __m128i bcast = _mm_set1_epi8((char)bitmap[j >> 3]);
            __m128i bits  = _mm_and_si128(bcast, bpt);
            __m128i sel   = _mm_cmpeq_epi8(bits, bpt);  /* 0xFF where set */
            __m128i v     = _mm_blendv_epi8(s0, s1, sel);
            _mm_storel_epi64((__m128i *)(symbols + j), v);
        }
    }
}

__attribute__((noinline))
static void bench_both_leaves_scatter_sse(uint8_t *symbols,
                                           const uint16_t *indices,
                                           const uint8_t *bitmap,
                                           uint8_t sym0, uint8_t sym1,
                                           int n, int reps)
{
    __m128i s0 = _mm_set1_epi8((char)sym0);
    __m128i s1 = _mm_set1_epi8((char)sym1);
    const __m128i bpt = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)128,
                                       0, 0, 0, 0, 0, 0, 0, 0);
    for (int r = 0; r < reps; r++) {
        for (int j = 0; j + 8 <= n; j += 8) {
            __m128i bcast = _mm_set1_epi8((char)bitmap[j >> 3]);
            __m128i bits  = _mm_and_si128(bcast, bpt);
            __m128i sel   = _mm_cmpeq_epi8(bits, bpt);
            __m128i v     = _mm_blendv_epi8(s0, s1, sel);
            symbols[indices[j + 0]] = (uint8_t)_mm_extract_epi8(v, 0);
            symbols[indices[j + 1]] = (uint8_t)_mm_extract_epi8(v, 1);
            symbols[indices[j + 2]] = (uint8_t)_mm_extract_epi8(v, 2);
            symbols[indices[j + 3]] = (uint8_t)_mm_extract_epi8(v, 3);
            symbols[indices[j + 4]] = (uint8_t)_mm_extract_epi8(v, 4);
            symbols[indices[j + 5]] = (uint8_t)_mm_extract_epi8(v, 5);
            symbols[indices[j + 6]] = (uint8_t)_mm_extract_epi8(v, 6);
            symbols[indices[j + 7]] = (uint8_t)_mm_extract_epi8(v, 7);
        }
    }
}

/* D=4 only (others fall through to scalar in production). */
__attribute__((noinline))
static void bench_flat_direct_d4_sse(uint8_t *out, const uint8_t *bm,
                                      const uint8_t *c2s, int n, int reps)
{
    __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_x86(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(out + i), syms);
        }
    }
}

__attribute__((noinline))
static void bench_flat_scatter_d4_sse(uint8_t *out, const uint16_t *idx,
                                       const uint8_t *bm, const uint8_t *c2s,
                                       int n, int reps)
{
    __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_x86(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            X16_SSE(syms, idx, i)
        }
    }
}

#undef X16_SSE
#endif /* HAS_SSE4 */

int main(void)
{
    uint8_t  *symbols = calloc(N, 1);
    uint16_t *indices = calloc(N, sizeof(uint16_t));
    uint16_t *left    = calloc(N, sizeof(uint16_t));
    uint16_t *right   = calloc(N, sizeof(uint16_t));
    uint8_t  *bitmap  = calloc((N + 7) / 8, 1);

    /* Identity indices */
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    /* Random-ish bitmap (~50% set) */
    srand(42);
    for (int i = 0; i < (N + 7) / 8; i++) bitmap[i] = (uint8_t)rand();

    /* Packed code stream big enough for D=2..6 (D=6 needs N*6/8 bytes;
     * +16 bytes pad for the 6/8-byte memcpy reads in flat_d{4,5,6}_unpack). */
    int flat_bm_bytes = (N * 6) / 8 + 16;
    uint8_t *flat_bm = calloc(flat_bm_bytes, 1);
    for (int i = 0; i < flat_bm_bytes; i++) flat_bm[i] = (uint8_t)rand();

    /* c2s table: 64 entries cover D=2..6 (D=6 = 2^6 = 64). */
    uint8_t c2s[64];
    for (int i = 0; i < 64; i++) c2s[i] = (uint8_t)(0x40 + i);

    /* Volatile sink to prevent dead-code elimination */
    volatile uint8_t sink = 0;

    /* Shuffled indices — simulates non-root after prior partition */
    uint16_t *shuffled = calloc(N, sizeof(uint16_t));
    for (int i = 0; i < N; i++) shuffled[i] = (uint16_t)i;
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        uint16_t t = shuffled[i]; shuffled[i] = shuffled[j]; shuffled[j] = t;
    }

    printf("N = %d, REPS = %d, total = %lld elements per test\n\n",
           N, REPS, (long long)N * REPS);

    double t0, t1, ns_per_elem;

#if HAS_NEON
    init_compress_table();

    /* Scatter NEON (constant sym to random positions) */
    t0 = now_sec();
    bench_scatter_neon(symbols, indices, N, 0x42, REPS);
    t1 = now_sec(); sink = symbols[0];
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("scatter_neon (const sym):     %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* Partition NEON (load indices + TBL shuffle) */
    t0 = now_sec();
    bench_partition_neon(indices, bitmap, left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("partition_neon (load+TBL):    %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* Partition root (generate identity + TBL shuffle) */
    t0 = now_sec();
    bench_partition_root_neon(bitmap, left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("partition_root (gen+TBL):     %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* Partition half (load indices, one TBL, one store) */
    t0 = now_sec();
    bench_partition_half_neon(indices, bitmap, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("partition_half (load+1 TBL):  %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* Partition root half (gen identity, one TBL, one store) */
    t0 = now_sec();
    bench_partition_root_half_neon(bitmap, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("partition_root_half (gen+1):  %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* ---- Store-pressure isolation tests ----
     *
     * partition_left_full: mask=0x00 every iter, all 8 elements go left,
     * n_left advances by 16 bytes per iter (no overlap).  Compare aligned
     * vs +1-byte-misaligned output buffer to isolate alignment cost.
     *
     * Then run partition (both sides) on bitmaps with X% of bytes = 0xFF
     * (popcount 8, advance 8 = no overlap on right; advance 0 = full
     * overlap on left) and (100-X)% = 0x1F (popcount 5, advance 5 = 6 B
     * overlap on right; advance 3 = 10 B overlap on left).  Varying X
     * shows how store-overlap pressure scales. */
    printf("\n-- store-pressure isolation --\n");

    /* Need a large-enough byte buffer to hold all left writes per rep
     * (n=8192 elements × 2 bytes = 16384, plus the +1 misalignment) */
    static uint8_t left_buf_aligned[8192 * 2 + 32] __attribute__((aligned(64)));
    static uint8_t * const left_buf_misaligned = left_buf_aligned + 1;

    t0 = now_sec();
    bench_partition_left_full(indices, left_buf_aligned, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("partition_left_full ALIGNED:  %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_left_full(indices, left_buf_misaligned, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("partition_left_full +1-byte:  %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    int densities[] = {100, 95, 90, 80, 75};
    for (int di = 0; di < (int)(sizeof(densities)/sizeof(*densities)); di++) {
        int pct = densities[di];
        fill_bitmap_density(bitmap, (N + 7) / 8, pct);
        t0 = now_sec();
        bench_partition_neon(indices, bitmap, left, right, N, REPS);
        t1 = now_sec();
        ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("partition mix %3d%%/0xFF + %2d%%/0x1F:  "
               "%5.2f ns/elem  (%5.1f GB/s)\n",
               pct, 100 - pct, ns_per_elem, 1.0 / ns_per_elem);
    }

    /* TBL & vext throughput probes — 16 independent chains per iter. */
    {
        const int probe_reps = (int)((double)REPS * 8);   /* ≈ equal total work */
        printf("\n-- TBL/vext throughput probes (16 indep chains) --\n");

        #define PROBE(label_, fn_)                                                  \
            do {                                                                    \
                t0 = now_sec();                                                     \
                fn_(probe_reps);                                                    \
                t1 = now_sec();                                                     \
                double ops = 16.0 * probe_reps;                                     \
                double ns_op = (t1 - t0) / ops * 1e9;                               \
                double cyc_op = ns_op * 4.4;                                        \
                printf("%-22s %5.3f ns/op  (%.2f cyc/op @4.4GHz, %.2f ops/cyc)\n",  \
                       label_, ns_op, cyc_op, 1.0 / cyc_op);                        \
            } while (0)

        PROBE("vqtbl1q_u8:",      bench_tbl1_throughput);
        PROBE("vqtbl2q_u8:",      bench_tbl2_throughput);
        PROBE("vqtbl4q_u8:",      bench_tbl4_throughput);
        PROBE("vextq_u8:",        bench_vext_throughput);
        #undef PROBE
    }

    /* Store-port hypothesis: 2× vst1q vs 4× vst1_u8 vs 1×vst1q+2×vst1_u8.
     * If M-series has 1 SIMD store port + 2 scalar AGUs running in
     * parallel, the mixed variant should run ~2x faster than simd_only. */
    printf("\n-- store-port topology probe --\n");
    /* Use the now-restored ~50% bitmap (above; will be reset before flat tests) */
    srand(42);
    for (int i = 0; i < (N + 7) / 8; i++) bitmap[i] = (uint8_t)rand();

    t0 = now_sec();
    bench_partition_simd_only(indices, bitmap, left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("simd_only   (2× vst1q):       %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_scalar_only(indices, bitmap, left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("scalar_only (4× vst1_u8):     %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_mixed(indices, bitmap, left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("mixed       (1×vst1q+2×vst1): %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* Coalesce-store partition prototypes were tested here; all three
     * lost to baseline on M4.  Code moved to extras/bench/bench_coalesce.c
     * with the full investigation in docs/COALESCE.md. */

    /* Memset */
    t0 = now_sec();
    bench_memset(symbols, N, 0x42, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("memset:                       %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* Both-leaves sequential vst1 (root identity) */
    t0 = now_sec();
    bench_both_leaves_vst1(symbols, bitmap, 0x41, 0x42, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("both_leaves_vst1 (seq):       %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* Both-leaves scattered stores (non-root) */
    t0 = now_sec();
    bench_both_leaves_scatter(symbols, indices, bitmap, 0x41, 0x42, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("both_leaves_scatter (idx):    %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    /* ---- Flat-subtree decode: per-D (unpack + TBL + store) ---- */
    printf("\n-- flat-subtree decode (unpack + TBL + store) --\n");

#define BENCH_FLAT(label_, fn_)                                            \
    do {                                                                   \
        t0 = now_sec();                                                    \
        fn_;                                                               \
        t1 = now_sec(); sink = symbols[0];                                 \
        ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;                \
        printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n", label_,              \
               ns_per_elem, 1.0 / ns_per_elem);                            \
    } while (0)

    BENCH_FLAT("flat_direct_d2 (vqtbl1q):",
               bench_flat_direct_d2(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_scatter_d2 (vqtbl1q):",
               bench_flat_scatter_d2(symbols, shuffled, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_direct_d3 (vqtbl1q):",
               bench_flat_direct_d3(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_scatter_d3 (vqtbl1):",
               bench_flat_scatter_d3(symbols, shuffled, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_direct_d4 (vqtbl1q):",
               bench_flat_direct_d4(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_scatter_d4 (vqtbl1q):",
               bench_flat_scatter_d4(symbols, shuffled, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_direct_d5 (vqtbl2q):",
               bench_flat_direct_d5(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_scatter_d5 (vqtbl2):",
               bench_flat_scatter_d5(symbols, shuffled, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_direct_d6 (vqtbl4q):",
               bench_flat_direct_d6(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT("flat_scatter_d6 (vqtbl4):",
               bench_flat_scatter_d6(symbols, shuffled, flat_bm, c2s, N, REPS));
#undef BENCH_FLAT
#endif /* HAS_NEON main dispatch */

#if HAS_AVX512
    printf("\n-- AVX-512 VBMI2 backend --\n");
    /* (scatter floor: see scatter_scalar row at end) */

    t0 = now_sec();
    bench_memset(symbols, N, 0x42, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "memset:", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_avx512(indices, (const uint32_t *)bitmap,
                            left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "partition_avx512 (vpcompressw):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_root_avx512((const uint32_t *)bitmap, left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "partition_root (gen+compress):", ns_per_elem, 1.0 / ns_per_elem);

    /* Diagnostic: SIMD-extract scatter (extras/ph-td's scatter_write_avx512). */
    t0 = now_sec();
    bench_scatter_avx512_pextrw(symbols, indices, N, 0x42, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "scatter_avx512 (load+pextrw):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_half_avx512(indices, (const uint32_t *)bitmap, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "partition_half (load+1):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_root_half_avx512((const uint32_t *)bitmap, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "partition_root_half (gen+1):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_both_leaves_vst1_avx512(symbols, (const uint32_t *)bitmap,
                                   0x41, 0x42, N, REPS);
    t1 = now_sec(); sink = symbols[0];
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "both_leaves_vst1 (seq):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_both_leaves_scatter_avx512(symbols, indices, (const uint32_t *)bitmap,
                                      0x41, 0x42, N, REPS);
    t1 = now_sec(); sink = symbols[0];
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "both_leaves_scatter (idx):", ns_per_elem, 1.0 / ns_per_elem);

    printf("\n-- flat-subtree decode (unpack + TBL + store), AVX-512 --\n");
#define BENCH_FLAT_X86(label_, fn_)                                          \
    do {                                                                     \
        t0 = now_sec();                                                      \
        fn_;                                                                 \
        t1 = now_sec(); sink = symbols[0];                                   \
        ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;                  \
        printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n", label_,                \
               ns_per_elem, 1.0 / ns_per_elem);                              \
    } while (0)

    BENCH_FLAT_X86("flat_direct_d2 (pshufb):",
                   bench_flat_direct_d2_avx512(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_scatter_d2 (pshufb):",
                   bench_flat_scatter_d2_avx512(symbols, shuffled, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_direct_d3 (pshufb):",
                   bench_flat_direct_d3_avx512(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_scatter_d3 (pshufb):",
                   bench_flat_scatter_d3_avx512(symbols, shuffled, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_direct_d4 (pshufb):",
                   bench_flat_direct_d4_avx512(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_scatter_d4 (pshufb):",
                   bench_flat_scatter_d4_avx512(symbols, shuffled, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_direct_d5 (vpermb-ymm):",
                   bench_flat_direct_d5_avx512(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_scatter_d5 (vpermb-ymm):",
                   bench_flat_scatter_d5_avx512(symbols, shuffled, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_direct_d6 (vpermb-zmm):",
                   bench_flat_direct_d6_avx512(symbols, flat_bm, c2s, N, REPS));
    BENCH_FLAT_X86("flat_scatter_d6 (vpermb-zmm):",
                   bench_flat_scatter_d6_avx512(symbols, shuffled, flat_bm, c2s, N, REPS));
#undef BENCH_FLAT_X86
#endif /* HAS_AVX512 main dispatch */

#if HAS_SSE4 && !HAS_AVX512
    printf("\n-- SSE4.1 backend --\n");
    init_compress_table_sse();

    t0 = now_sec();
    bench_memset(symbols, N, 0x42, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "memset:", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_scatter_sse(symbols, indices, N, 0x42, REPS);
    t1 = now_sec(); sink = symbols[0];
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "scatter_sse (8-wide):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_sse(indices, bitmap, left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "partition_sse (pshufb):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_root_sse(bitmap, left, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "partition_root (gen+pshufb):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_half_sse(indices, bitmap, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "partition_half (load+1):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_partition_root_half_sse(bitmap, right, N, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "partition_root_half (gen+1):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_both_leaves_vst1_sse(symbols, bitmap, 0x41, 0x42, N, REPS);
    t1 = now_sec(); sink = symbols[0];
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "both_leaves_vst1 (seq):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_both_leaves_scatter_sse(symbols, indices, bitmap, 0x41, 0x42, N, REPS);
    t1 = now_sec(); sink = symbols[0];
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "both_leaves_scatter (idx):", ns_per_elem, 1.0 / ns_per_elem);

    printf("\n-- flat-subtree decode (D=4 only on pure SSE4.1) --\n");
    t0 = now_sec();
    bench_flat_direct_d4_sse(symbols, flat_bm, c2s, N, REPS);
    t1 = now_sec(); sink = symbols[0];
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "flat_direct_d4 (pshufb):", ns_per_elem, 1.0 / ns_per_elem);

    t0 = now_sec();
    bench_flat_scatter_d4_sse(symbols, shuffled, flat_bm, c2s, N, REPS);
    t1 = now_sec(); sink = symbols[0];
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("%-30s %5.2f ns/elem  (%5.1f GB/s)\n",
           "flat_scatter_d4 (pshufb):", ns_per_elem, 1.0 / ns_per_elem);

    printf("(D=2/3/5/6: pure SSE4.1 falls through to scalar in production.)\n");
#endif /* HAS_SSE4 main dispatch */

    /* Scatter scalar */
    t0 = now_sec();
    bench_scatter_scalar(symbols, indices, N, 0x42, REPS);
    t1 = now_sec();
    ns_per_elem = (t1 - t0) / ((double)N * REPS) * 1e9;
    printf("scatter_scalar:               %5.2f ns/elem  (%5.1f GB/s)\n",
           ns_per_elem, 1.0 / ns_per_elem);

    free(symbols);
    free(indices);
    free(left);
    free(right);
    free(bitmap);
    free(shuffled);
    free(flat_bm);
    (void)sink;
    return 0;
}
