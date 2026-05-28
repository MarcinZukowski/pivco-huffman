/* extras/bench/bench_fl_unpack.c — FastLanes-style D-bit unpack vs our
 * flat_dN_unpack helpers.
 *
 * Investigation: FastLanes (cwida/FastLanes) uses a *transposed* layout
 * for its bit-unpacked columns.  Their inner unpack loop avoids TBL
 * entirely — just immediate shifts + AND masks, since in transposed
 * layout each shift amount maps to one full output lane group.
 *
 * The layout itself doesn't transfer to our stream-layout flat-subtree
 * regions, but the *operation pattern* might: produce the 4 (D=2) /
 * 2 (D=4) interleaved groups via shifts + masks, then use `vst4q_u8` /
 * `vst2q_u8` to interleave-store them in correct stream order.
 *
 * Variants benchmarked (D=2 only for now):
 *   flat_direct_d2_baseline : current flat_direct_d2 (vqtbl1q dup +
 *                             vshlq variable-shift + c2s TBL + vst1q,
 *                             16 codes/iter, 4 iters per 64 codes).
 *   flat_direct_d2_fl       : FL-style (1 load + 4 shift+mask groups +
 *                             4 c2s TBLs + 1 vst4q, 64 codes/iter).
 *
 * Build:  cc -O3 -o bench_fl_unpack extras/bench/bench_fl_unpack.c
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

/* Baseline D=2 unpack (verbatim copy from src/pivco_huffman_neon_flat.h). */
static const uint8_t flat_d2_dup_tab[16] = {
    0,0,0,0,  1,1,1,1,  2,2,2,2,  3,3,3,3
};
static const int8_t flat_d2_shift_tab[16] = {
    0,-2,-4,-6,  0,-2,-4,-6,  0,-2,-4,-6,  0,-2,-4,-6
};

static inline uint8x16_t flat_d2_unpack(const uint8_t *bm_ptr)
{
    uint32_t packed;
    memcpy(&packed, bm_ptr, 4);
    uint8x16_t bm_lo = vreinterpretq_u8_u32(
        vsetq_lane_u32(packed, vdupq_n_u32(0), 0));
    uint8x16_t dup = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d2_dup_tab));
    uint8x16_t shifted = vshlq_u8(dup, vld1q_s8(flat_d2_shift_tab));
    return vandq_u8(shifted, vdupq_n_u8(0x03));
}

__attribute__((noinline))
static void bench_flat_direct_d2_baseline(uint8_t *out, const uint8_t *bm,
                                           const uint8_t *c2s, int n, int reps)
{
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d2_unpack(bm + (i >> 2));
            vst1q_u8(out + i, vqtbl1q_u8(c2s_vec, codes));
        }
    }
}

/* FL-style D=2: 16 input bytes → 64 codes, written via vst4q_u8.
 * Each input byte contains 4 D=2 codes at bit positions 0,2,4,6.
 * Group g_k contains all codes at bit-position k across the 16 bytes,
 * which corresponds to codes c_{4i + k} for i in [0,15].  vst4q_u8
 * interleaves the 4 groups into [c0,c1,c2,c3, c4,c5,c6,c7, ...]. */
__attribute__((noinline))
static void bench_flat_direct_d2_fl(uint8_t *out, const uint8_t *bm,
                                     const uint8_t *c2s, int n, int reps)
{
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    uint8x16_t mask3   = vdupq_n_u8(0x03);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 64 <= n; i += 64) {
            uint8x16_t reg = vld1q_u8(bm + (i >> 2));
            uint8x16_t g0 = vandq_u8(reg, mask3);
            uint8x16_t g1 = vandq_u8(vshrq_n_u8(reg, 2), mask3);
            uint8x16_t g2 = vandq_u8(vshrq_n_u8(reg, 4), mask3);
            uint8x16_t g3 = vandq_u8(vshrq_n_u8(reg, 6), mask3);
            uint8x16x4_t syms = {{
                vqtbl1q_u8(c2s_vec, g0),
                vqtbl1q_u8(c2s_vec, g1),
                vqtbl1q_u8(c2s_vec, g2),
                vqtbl1q_u8(c2s_vec, g3)
            }};
            vst4q_u8(out + i, syms);
        }
    }
}

int main(void)
{
    uint8_t *out      = (uint8_t *)aligned_alloc(64, N + 64);
    uint8_t *bm       = (uint8_t *)aligned_alloc(64, N + 64);
    uint8_t  c2s[16];
    if (!out || !bm) { perror("alloc"); return 1; }

    srand(42);
    for (int i = 0; i < N; i++) bm[i] = (uint8_t)rand();
    for (int i = 0; i < 16; i++) c2s[i] = (uint8_t)(0x40 + i);

    /* Sanity: both variants should produce identical output. */
    memset(out, 0, N);
    bench_flat_direct_d2_baseline(out, bm, c2s, N, 1);
    uint8_t out_baseline[64];
    memcpy(out_baseline, out, 64);

    memset(out, 0, N);
    bench_flat_direct_d2_fl(out, bm, c2s, N, 1);
    uint8_t out_fl[64];
    memcpy(out_fl, out, 64);

    if (memcmp(out_baseline, out_fl, 64) != 0) {
        printf("CORRECTNESS MISMATCH between baseline and FL variants:\n");
        printf("baseline: ");
        for (int i = 0; i < 64; i++) printf("%02x ", out_baseline[i]);
        printf("\n     FL: ");
        for (int i = 0; i < 64; i++) printf("%02x ", out_fl[i]);
        printf("\n");
        return 2;
    }
    printf("(correctness OK — first 64 output bytes match)\n\n");

    printf("== bench_fl_unpack: D=2 unpack baseline vs FL-style ==\n");
    printf("N = %d, REPS = %d, total = %lld codes per row\n\n",
           N, REPS, (long long)N * REPS);

    double t0, t1, ns;

    /* Multiple runs to check stability */
    for (int run = 0; run < 3; run++) {
        printf("-- run %d --\n", run + 1);
        t0 = now_sec();
        bench_flat_direct_d2_baseline(out, bm, c2s, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  baseline (vqtbl1q dup + vshlq):   %5.3f ns/code  (%5.1f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_flat_direct_d2_fl(out, bm, c2s, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  FL-style (vshr + vand + vst4q):   %5.3f ns/code  (%5.1f GB/s)\n",
               ns, 1.0 / ns);
        printf("\n");
    }

    free(out);
    free(bm);
    return 0;
}
