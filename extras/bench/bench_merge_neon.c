/* bench_merge_neon.c — merge_vec_vec on NEON: expand_tab lookup vs
 * prefix-popcount derivation.
 *
 *   - tab    : 2 KB expand_tab path (the simple version of NEON's prod
 *              merge_vec_vec, before the 18 KB expand_tab_pre serial-dep
 *              workaround).  Used as the baseline here because (a) it's
 *              the closest apples-to-apples for "table-based 8-byte
 *              merge", and (b) the prefix-popcount path also has the
 *              serial dep on lc/rc, so comparing to the simpler tab path
 *              is fair.
 *
 *   - prepop : 8-byte chunks via parallel prefix popcount on 8 u16 lanes.
 *              No table.  NEON's vcntq_u8 + vpaddlq_u8 is a 2-op u16
 *              popcount (vs AVX2's 7-op vpshufb chain).
 *
 * Build (M4 / Graviton 4):
 *   cc -O3 -march=native -o bench_merge_neon extras/bench/bench_merge_neon.c
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if !defined(__aarch64__)
int main(void) { puts("bench_merge_neon: needs aarch64 NEON"); return 0; }
#else

#include <arm_neon.h>

#define N    8192
#define REPS 100000

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* expand_tab[mask][k] = lane index (0..15) for output byte k into the
 * vcombine(L8, R8) source pair.  Values 0..7 → L; 8..15 → R. */
static uint8_t expand_tab[256][8] __attribute__((aligned(16)));
static uint8_t expand_popcnt[256];

static void init_expand_tab(void) {
    for (int m = 0; m < 256; m++) {
        int nz = 0, no = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) { expand_tab[m][k] = (uint8_t)(8 + no); no++; }
            else              { expand_tab[m][k] = (uint8_t)nz; nz++; }
        }
        expand_popcnt[m] = (uint8_t)no;
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
 *   v_tab: simple 2 KB expand_tab path, 8-byte chunks. */
static void merge_tab(uint8_t *out, int K, const uint8_t *left,
                       const uint8_t *right, const uint8_t *bm)
{
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x8_t L = vld1_u8(left  + lc);
        uint8x8_t R = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(L, R);
        uint8x8_t shuf = vld1_u8(expand_tab[m]);
        uint8x8_t o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        int nr = expand_popcnt[m];
        rc += nr;
        lc += 8 - nr;
    }
    for (; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left[lc++];
    }
}

/* ============================================================
 *   v_prepop: 8-byte chunks, prefix popcount derivation, no table.
 *
 *   Steps per iter:
 *     1. m = bm[j>>3].
 *     2. mvec = vdupq_n_u16(m), mask with cumulative bit masks.
 *     3. Per-byte popcount via vcntq_u8, then pair-sum via vpaddlq_u8.
 *        That gives 8 u16 prefix popcounts in 2 ops.
 *     4. Pack to u8 via vmovn_u16.
 *     5. shuf_right = prefix - 1, shuf_left = indices - prefix.
 *     6. Combine via vbsl: for byte positions with mask=1, use
 *        shuf_right offset by 8 (selects from R half of combined reg);
 *        else use shuf_left (selects from L half).
 *     7. vqtbl1_u8 on vcombine(L, R) with the combined shuf, store. */
static void merge_prepop(uint8_t *out, int K, const uint8_t *left,
                          const uint8_t *right, const uint8_t *bm)
{
    int lc = 0, rc = 0;
    int j = 0;
    const uint16x8_t prefix_masks = {
        0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF };
    const uint8x8_t indices = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint8x8_t bit_pos = {1, 2, 4, 8, 16, 32, 64, 128};

    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];

        /* Parallel prefix popcounts: 2 ops for 8 u16 lanes. */
        uint16x8_t mvec = vdupq_n_u16((uint16_t)m);
        uint16x8_t masked = vandq_u16(mvec, prefix_masks);
        uint8x16_t bp = vcntq_u8(vreinterpretq_u8_u16(masked));
        uint16x8_t prefix_u16 = vpaddlq_u8(bp);     /* prefix[k] = popcount of bits 0..k */
        uint8x8_t  prefix_u8  = vmovn_u16(prefix_u16);

        /* Two shuf candidates: right needs prefix-1, left needs k-prefix. */
        uint8x8_t shuf_right = vsub_u8(prefix_u8, vdup_n_u8(1));
        uint8x8_t shuf_left  = vsub_u8(indices, prefix_u8);
        /* Right side goes to the high half of vcombine(L, R) → offset by 8. */
        uint8x8_t shuf_right_combined = vadd_u8(shuf_right, vdup_n_u8(8));

        /* Expand 8-bit mask to 8-byte mask (0xFF where bit set). */
        uint8x8_t mbcast = vdup_n_u8(m);
        uint8x8_t mask_bytes = vceq_u8(vand_u8(mbcast, bit_pos), bit_pos);

        /* Combined shuf: pick right for set bits, left for clear bits. */
        uint8x8_t shuf = vbsl_u8(mask_bytes, shuf_right_combined, shuf_left);

        /* Gather. */
        uint8x8_t L = vld1_u8(left  + lc);
        uint8x8_t R = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(L, R);
        uint8x8_t output = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, output);

        int nr = __builtin_popcount((unsigned)m);
        rc += nr;
        lc += 8 - nr;
    }
    for (; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left[lc++];
    }
}

/* ============================================================
 *   Driver */
typedef void (*merge_fn)(uint8_t *, int, const uint8_t *, const uint8_t *, const uint8_t *);

static double time_fn(merge_fn fn, uint8_t *out, int K,
                      const uint8_t *L, const uint8_t *R, const uint8_t *bm,
                      int reps)
{
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) fn(out, K, L, R, bm);
    double t1 = now_sec();
    return (t1 - t0) / ((double)K * reps) * 1e9;
}

static int verify(const uint8_t *a, const uint8_t *b, int K) {
    for (int i = 0; i < K; i++) if (a[i] != b[i]) return i;
    return -1;
}

static void make_bm(uint8_t *bm, int K, int p) {
    int nb = (K + 7) >> 3;
    for (int i = 0; i < nb; i++) {
        uint8_t m = 0;
        for (int k = 0; k < 8; k++)
            if ((rand() & 0xFF) < p) m |= (uint8_t)(1u << k);
        bm[i] = m;
    }
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : REPS;
    init_expand_tab();
    printf("bench_merge_neon: K=%d, REPS=%d\n\n", N, reps);

    /* macOS aligned_alloc requires size % alignment == 0; round up. */
    size_t sz_big = ((N + 128 + 63) / 64) * 64;
    size_t sz_bm  = ((N/8 + 32 + 63) / 64) * 64;
    uint8_t *L     = aligned_alloc(64, sz_big);
    uint8_t *R     = aligned_alloc(64, sz_big);
    uint8_t *bm    = aligned_alloc(64, sz_bm);
    uint8_t *o_ref = aligned_alloc(64, sz_big);
    uint8_t *o_tab = aligned_alloc(64, sz_big);
    uint8_t *o_pp  = aligned_alloc(64, sz_big);

    srand(42);
    for (int i = 0; i < N + 128; i++) L[i] = (uint8_t)('a' + (rand() & 0x1F));
    for (int i = 0; i < N + 128; i++) R[i] = (uint8_t)('A' + (rand() & 0x1F));

    int densities[] = {32, 64, 128, 192, 224};
    const char *dlbl[] = {"12%", "25%", "50%", "75%", "87%"};
    printf("%-8s %10s %10s %10s   %s\n",
           "density", "scalar", "tab", "prepop", "check");
    for (size_t d = 0; d < sizeof(densities)/sizeof(densities[0]); d++) {
        make_bm(bm, N, densities[d]);

        merge_scalar(o_ref, N, L, R, bm);
        memset(o_tab, 0, N); merge_tab(o_tab, N, L, R, bm);
        memset(o_pp,  0, N); merge_prepop(o_pp,  N, L, R, bm);

        int e_tab = verify(o_ref, o_tab, N);
        int e_pp  = verify(o_ref, o_pp,  N);
        const char *chk = (e_tab < 0 && e_pp < 0) ? "ok" : "FAIL";

        double t0 = now_sec();
        for (int r = 0; r < 1000; r++) merge_scalar(o_ref, N, L, R, bm);
        double ts = (now_sec() - t0) / ((double)N * 1000) * 1e9;

        double t_tab = time_fn(merge_tab,    o_tab, N, L, R, bm, reps);
        double t_pp  = time_fn(merge_prepop, o_pp,  N, L, R, bm, reps);

        printf("%-8s %8.3fns %8.3fns %8.3fns   %s\n",
               dlbl[d], ts, t_tab, t_pp, chk);
        if (e_tab >= 0) printf("    tab    mismatch at %d (s=0x%02x t=0x%02x)\n",
                                 e_tab, o_ref[e_tab], o_tab[e_tab]);
        if (e_pp  >= 0) printf("    prepop mismatch at %d (s=0x%02x p=0x%02x)\n",
                                 e_pp,  o_ref[e_pp],  o_pp[e_pp]);
    }

    free(L); free(R); free(bm); free(o_ref); free(o_tab); free(o_pp);
    return 0;
}

#endif /* aarch64 */
