/* extras/bench_coalesce.c — store-coalescing experiments for partition_8.
 *
 * Three variants tested, all losers on M4.  See COALESCE.md at the
 * project root for the full investigation log.  This file is the
 * working code for the experiments.
 *
 *   baseline      : current production shape — 2 × vst1q_u8 per iter,
 *                   one per side.  Store-port-saturated at 1/cycle.
 *   coalesce_vext : per-iter coalesce, switch on so_far in [0,7]
 *                   dispatching to vextq_u8 with constant immediates.
 *   coalesce_tbl  : per-iter coalesce, runtime-computed shuffle vector
 *                   applied via vqtbl1q_u8.  No switch, single conditional
 *                   branch per side per iter (the flush check).
 *   coalesce_macro: 4-iter macro-block with lookahead.  Prefix-sum the
 *                   popcounts of 4 masks upfront (scalar), then place 4
 *                   compressed registers into a 32-byte (lo, hi)
 *                   accumulator with no cross-iter dep on accumulator
 *                   state.  Stores 2 × 16 bytes per side per macro-block.
 *
 * Build (standalone):
 *   cc -O2 -o bench_coalesce extras/bench_coalesce.c
 * Or via cmake target `pivco_bench_coalesce`.
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define N    8192
#define REPS 100000

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------- Compress shuffle table (same shape as production) ---------- */
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

/* ---------- Baseline: 2 × vst1q per iter ---------- */
__attribute__((noinline))
static void bench_baseline(const uint16_t *src, const uint8_t *bitmap,
                            uint16_t *left, uint16_t *right,
                            int n, int reps)
{
    for (int r = 0; r < reps; r++) {
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8_t mask = bitmap[j >> 3];
            uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
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

/* ---------- Variant 1: coalesce-vext (switch on so_far) ---------- */
#define COALESCE_CASE_0(V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
    case 0: {                                                              \
        uint8x16_t _merged = vorrq_u8((accum_var), (V_v));                 \
        if ((cnt_var) < 8) {                                               \
            (accum_var)  = _merged;                                        \
            (so_far_var) = (cnt_var);                                      \
        } else {                                                           \
            vst1q_u8((out_p) + (n_var), _merged);                          \
            (n_var) += 16;                                                 \
            (accum_var)  = zero_v;                                         \
            (so_far_var) = (cnt_var) - 8;                                  \
        }                                                                  \
    } break;

#define COALESCE_CASE_K(K, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
    case K: {                                                                 \
        uint8x16_t _shifted = vextq_u8(zero_v, (V_v), 16 - (K) * 2);          \
        uint8x16_t _merged  = vorrq_u8((accum_var), _shifted);                \
        if ((K) + (cnt_var) < 8) {                                            \
            (accum_var)   = _merged;                                          \
            (so_far_var)  = (K) + (cnt_var);                                  \
        } else {                                                              \
            vst1q_u8((out_p) + (n_var), _merged);                             \
            (n_var) += 16;                                                    \
            (accum_var)  = vextq_u8((V_v), zero_v, (8 - (K)) * 2);            \
            (so_far_var) = (K) + (cnt_var) - 8;                               \
        }                                                                     \
    } break;

#define COALESCE_SWITCH(V_v, cnt_var, accum_var, so_far_var, out_p, n_var)    \
    switch (so_far_var) {                                                     \
        COALESCE_CASE_0(   V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(1, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(2, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(3, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(4, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(5, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(6, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(7, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
    }

__attribute__((noinline))
static void bench_coalesce_vext(const uint16_t *src, const uint8_t *bitmap,
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
            uint8x16_t r_v = vqtbl1q_u8(data, vld1q_u8(tab));
            uint8x16_t l_v = vqtbl1q_u8(data, vld1q_u8(tab + 16));
            int cnt_r = compress_popcnt[mask];
            int cnt_l = 8 - cnt_r;
            COALESCE_SWITCH(r_v, cnt_r, accum_r, so_far_r, right_bytes, n_right_bytes)
            COALESCE_SWITCH(l_v, cnt_l, accum_l, so_far_l, left_bytes,  n_left_bytes)
        }
        if (so_far_l > 0) vst1q_u8(left_bytes  + n_left_bytes,  accum_l);
        if (so_far_r > 0) vst1q_u8(right_bytes + n_right_bytes, accum_r);
    }
}

/* ---------- Variant 2: coalesce-tbl (runtime-computed shuf) ---------- */
__attribute__((noinline))
static void bench_coalesce_tbl(const uint16_t *src, const uint8_t *bitmap,
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
            uint8x16_t r_v = vqtbl1q_u8(data, vld1q_u8(tab));
            uint8x16_t l_v = vqtbl1q_u8(data, vld1q_u8(tab + 16));
            int cnt_r = compress_popcnt[mask];
            int cnt_l = 8 - cnt_r;

            {   /* Right */
                uint8x16_t shuf_left = vsubq_u8(iota, vdupq_n_u8((uint8_t)(so_far_r * 2)));
                uint8x16_t shifted   = vqtbl1q_u8(r_v, shuf_left);
                uint8x16_t merged    = vorrq_u8(accum_r, shifted);
                int new_sf = so_far_r + cnt_r;
                if (new_sf >= 8) {
                    vst1q_u8(right_bytes + n_right_bytes, merged);
                    n_right_bytes += 16;
                    uint8x16_t shuf_rt = vaddq_u8(iota, vdupq_n_u8((uint8_t)((8 - so_far_r) * 2)));
                    accum_r = vqtbl1q_u8(r_v, shuf_rt);
                    so_far_r = new_sf - 8;
                } else {
                    accum_r = merged;
                    so_far_r = new_sf;
                }
            }
            {   /* Left */
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

/* ---------- Variant 3: coalesce-macro (4-iter lookahead) ---------- */
__attribute__((noinline))
static void bench_coalesce_macro(const uint16_t *src, const uint8_t *bitmap,
                                  uint8_t *left_bytes, uint8_t *right_bytes,
                                  int n, int reps)
{
    const uint8x16_t zero_v = vdupq_n_u8(0);
    static const uint8_t iota_init[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    const uint8x16_t iota = vld1q_u8(iota_init);

    for (int r = 0; r < reps; r++) {
        int n_left_bytes = 0, n_right_bytes = 0;
        for (int j = 0; j + 32 <= n; j += 32) {
            uint8_t m0 = bitmap[(j >> 3) + 0];
            uint8_t m1 = bitmap[(j >> 3) + 1];
            uint8_t m2 = bitmap[(j >> 3) + 2];
            uint8_t m3 = bitmap[(j >> 3) + 3];
            int pr0 = compress_popcnt[m0], pl0 = 8 - pr0;
            int pr1 = compress_popcnt[m1], pl1 = 8 - pr1;
            int pr2 = compress_popcnt[m2], pl2 = 8 - pr2;
            int pr3 = compress_popcnt[m3], pl3 = 8 - pr3;
            int cr1 = pr0, cr2 = cr1 + pr1, cr3 = cr2 + pr2;
            int total_r = cr3 + pr3;
            int cl1 = pl0, cl2 = cl1 + pl1, cl3 = cl2 + pl2;
            int total_l = cl3 + pl3;

            uint8x16_t d0 = vld1q_u8((const uint8_t *)(src + j +  0));
            uint8x16_t d1 = vld1q_u8((const uint8_t *)(src + j +  8));
            uint8x16_t d2 = vld1q_u8((const uint8_t *)(src + j + 16));
            uint8x16_t d3 = vld1q_u8((const uint8_t *)(src + j + 24));

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

            #define PLACE(side_v, cum, lo_acc, hi_acc) do {                                 \
                uint8x16_t _shuf_lo = vsubq_u8(iota, vdupq_n_u8((uint8_t)((cum) * 2)));      \
                uint8x16_t _shuf_hi = vaddq_u8(iota, vdupq_n_u8((uint8_t)(16 - (cum) * 2))); \
                (lo_acc) = vorrq_u8((lo_acc), vqtbl1q_u8((side_v), _shuf_lo));              \
                (hi_acc) = vorrq_u8((hi_acc), vqtbl1q_u8((side_v), _shuf_hi));              \
            } while (0)

            uint8x16_t lo_r = r0, hi_r = zero_v;
            PLACE(r1, cr1, lo_r, hi_r);
            PLACE(r2, cr2, lo_r, hi_r);
            PLACE(r3, cr3, lo_r, hi_r);

            uint8x16_t lo_l = l0, hi_l = zero_v;
            PLACE(l1, cl1, lo_l, hi_l);
            PLACE(l2, cl2, lo_l, hi_l);
            PLACE(l3, cl3, lo_l, hi_l);
            #undef PLACE

            vst1q_u8(right_bytes + n_right_bytes,      lo_r);
            vst1q_u8(right_bytes + n_right_bytes + 16, hi_r);
            n_right_bytes += total_r * 2;
            vst1q_u8(left_bytes  + n_left_bytes,       lo_l);
            vst1q_u8(left_bytes  + n_left_bytes + 16,  hi_l);
            n_left_bytes  += total_l  * 2;
        }
        /* No tail handling — n=8192 is a multiple of 32. */
    }
}

/* ---------- Bitmap generators ---------- */
static void fill_random(uint8_t *bitmap, int n_bytes, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < n_bytes; i++) bitmap[i] = (uint8_t)rand();
}

static void fill_skewed_2_6(uint8_t *bitmap, int n_bytes, unsigned seed)
{
    /* Half the bytes have popcount 2, half have popcount 6.  Each side's
     * cnt alternates between 2 and 6 → so_far walks more predictably. */
    srand(seed);
    for (int i = 0; i < n_bytes; i++) {
        int v = rand() % 256, pc = __builtin_popcount(v);
        for (int t = 0; t < 20 && pc != 2 && pc != 6; t++) {
            v = rand() % 256; pc = __builtin_popcount(v);
        }
        bitmap[i] = (uint8_t)v;
    }
}

/* ---------- Main ---------- */
int main(void)
{
    init_compress_table();

    uint16_t *src     = (uint16_t *)calloc(N, sizeof(uint16_t));
    uint16_t *left    = (uint16_t *)calloc(N, sizeof(uint16_t));
    uint16_t *right   = (uint16_t *)calloc(N, sizeof(uint16_t));
    uint8_t  *bitmap  = (uint8_t  *)calloc((N + 7) / 8, 1);
    if (!src || !left || !right || !bitmap) { perror("calloc"); return 1; }

    /* Identity src */
    for (int i = 0; i < N; i++) src[i] = (uint16_t)i;

    printf("== bench_coalesce: store-coalescing experiments for partition_8 ==\n");
    printf("N = %d, REPS = %d, total = %lld elems per row\n\n",
           N, REPS, (long long)N * REPS);

    struct {
        const char *label;
        void (*fill)(uint8_t *, int, unsigned);
    } scenarios[] = {
        { "50% random",  fill_random },
        { "skew popcount 2/6", fill_skewed_2_6 },
    };

    for (int s = 0; s < (int)(sizeof(scenarios) / sizeof(*scenarios)); s++) {
        scenarios[s].fill(bitmap, (N + 7) / 8, 42);
        printf("-- %s --\n", scenarios[s].label);
        double t0, t1, ns;

        t0 = now_sec();
        bench_baseline(src, bitmap, left, right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  baseline       (2 vst1q):  %5.2f ns/elem  (%5.2f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_coalesce_vext(src, bitmap, (uint8_t *)left, (uint8_t *)right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  coalesce_vext  (switch):   %5.2f ns/elem  (%5.2f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_coalesce_tbl(src, bitmap, (uint8_t *)left, (uint8_t *)right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  coalesce_tbl   (no switch):%5.2f ns/elem  (%5.2f GB/s)\n",
               ns, 1.0 / ns);

        t0 = now_sec();
        bench_coalesce_macro(src, bitmap, (uint8_t *)left, (uint8_t *)right, N, REPS);
        t1 = now_sec();
        ns = (t1 - t0) / ((double)N * REPS) * 1e9;
        printf("  coalesce_macro (4-iter):   %5.2f ns/elem  (%5.2f GB/s)\n",
               ns, 1.0 / ns);
        printf("\n");
    }

    free(src); free(left); free(right); free(bitmap);
    return 0;
}
