/* sve_probe/partition_probe.c -- compare ph's production NEON partition
 * primitive vs an SVE-256 partition that issues svcompact_u32 at 8 lanes
 * (full V1 vector width).
 *
 * Why this matters: partition is ph's hottest internal-node primitive
 * (encode + decode both call it for every non-flat internal node).  The
 * NEON version uses a 4 KB compress_tab lookup + vqtbl1q_u8.  SVE could
 * in principle do the same work with one svcompact_u32 and no table.
 *
 * On Neoverse V1 with VL=256:
 *   - NEON: 8 codes per iter, 5-7 NEON ops + 1 L1 load (compress_tab)
 *   - SVE-256: 8 codes per iter, 1 svcompact_u32 + widening overhead
 *
 * The svcompact_u32 path widens uint16 -> uint32 (needed because
 * svcompact only supports 32/64-bit elements), which halves effective
 * lane density compared to a hypothetical svcompact_u16.
 *
 * Build: gcc -O3 -march=armv8.4-a+sve partition_probe.c -o partition_probe
 *
 * Runs only on V1 (VL=256).  V2 has VL=128 — would need a different
 * comparison (test against bdep_u8 from SVE2).
 */

#include <arm_neon.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- compress_tab (32 bytes per 8-bit mask): byte indices for vqtbl --- */
static uint8_t compress_tab[256][32];
static uint8_t compress_popcnt[256];

static void init_compress_tab(void) {
    for (int m = 0; m < 256; m++) {
        uint8_t *t = compress_tab[m];
        int nr = 0, nl = 0;
        for (int k = 0; k < 8; k++) {
            int bit = (m >> k) & 1;
            if (bit) {
                t[2 * nr    ] = (uint8_t)(2 * k);
                t[2 * nr + 1] = (uint8_t)(2 * k + 1);
                nr++;
            } else {
                t[16 + 2 * nl    ] = (uint8_t)(2 * k);
                t[16 + 2 * nl + 1] = (uint8_t)(2 * k + 1);
                nl++;
            }
        }
        /* Fill remaining bytes with 0xFF (NEON TBL maps to 0 on out-of-range). */
        for (int k = 2 * nr;     k < 16; k++) t[k]      = 0xFF;
        for (int k = 16 + 2 * nl;k < 32; k++) t[k]      = 0xFF;
        compress_popcnt[m] = (uint8_t)nr;
    }
}

/* --- NEON variant (lifted from production build_bitmap_partition_full_neon
 *     core; assumes mask is pre-computed since we want to isolate the
 *     partition step, not the mask-compute) --- */
static inline int partition_8_neon(uint16_t *codes_la, int n_off,
                                    uint8_t mask,
                                    uint16_t *left_out, int n_left,
                                    uint16_t *right_out, int n_right)
{
    uint16x8_t code_vec = vld1q_u16(codes_la + n_off);
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    uint8x16_t data   = vreinterpretq_u8_u16(code_vec);
    uint8x16_t right  = vqtbl1q_u8(data, shuf_r);
    uint8x16_t left   = vqtbl1q_u8(data, shuf_l);
    int nr = compress_popcnt[mask];
    vst1q_u8((uint8_t *)(right_out + n_right), right);
    vst1q_u8((uint8_t *)(left_out  + n_left ), left);
    return nr;
}

/* --- SVE-256 variant: load 8x u16, widen to u32, svcompact, narrow back.
 *     VL=256 means 8 x u32 in one SVE register — perfect fit. --- */
static inline int partition_8_sve256(uint16_t *codes_la, int n_off,
                                       uint8_t mask,
                                       uint16_t *left_out, int n_left,
                                       uint16_t *right_out, int n_right)
{
    svbool_t pg8 = svwhilelt_b32(0, 8);

    /* Build right-predicate from the 8-bit mask: lane k is "right" if
     * bit k of mask is set.  Use svindex + svdup + svtbl on the mask
     * bits... cleanest portable form: shift+and per lane. */
    svuint32_t lane_id = svindex_u32(0, 1);                  /* {0,1,...,7} */
    svuint32_t one     = svdup_u32(1);
    svuint32_t mask_v  = svdup_u32((uint32_t)mask);
    svuint32_t shifted = svlsl_u32_x(pg8, one, lane_id);     /* {1,2,4,...,128} */
    svuint32_t bits    = svand_u32_x(pg8, mask_v, shifted);
    svbool_t  right_p  = svcmpne_n_u32(pg8, bits, 0);
    svbool_t  left_p   = svnot_b_z(pg8, right_p);

    /* Load 8 x u16 and widen to u32 in one SVE register. */
    uint32_t tmp32[8];
    for (int k = 0; k < 8; k++) tmp32[k] = codes_la[n_off + k];
    svuint32_t data32 = svld1_u32(pg8, tmp32);

    /* Compact right and left. */
    svuint32_t right32 = svcompact_u32(right_p, data32);
    svuint32_t left32  = svcompact_u32(left_p,  data32);

    /* Narrow back and store. */
    uint32_t outr[8], outl[8];
    svst1_u32(pg8, outr, right32);
    svst1_u32(pg8, outl, left32);

    int nr = (int)svcntp_b32(pg8, right_p);
    for (int k = 0; k < nr;     k++) right_out[n_right + k] = (uint16_t)outr[k];
    for (int k = 0; k < 8 - nr; k++) left_out [n_left  + k] = (uint16_t)outl[k];
    return nr;
}

/* --- bench driver: partition N codes (random 8-bit masks per block of 8) --- */

static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s; x ^= x<<13; x ^= x>>7; x ^= x<<17; *s = x; return x;
}

typedef int (*partition_fn)(uint16_t *, int, uint8_t,
                            uint16_t *, int, uint16_t *, int);

static double measure(partition_fn fn,
                       uint16_t *src, int n,
                       const uint8_t *masks,
                       uint16_t *left, uint16_t *right,
                       int reps)
{
    /* Best of N runs.  Each run partitions n codes (n/8 mask blocks). */
    double best = 0;
    uint16_t *codes_scratch = aligned_alloc(64, n * sizeof(uint16_t));
    for (int r = 0; r < 5; r++) {
        double t0 = now_sec();
        for (int k = 0; k < reps; k++) {
            memcpy(codes_scratch, src, n * sizeof(uint16_t));
            int nL = 0, nR = 0;
            for (int j = 0; j + 8 <= n; j += 8) {
                int nr = fn(codes_scratch, j, masks[j >> 3], left, nL, right, nR);
                nR += nr;
                nL += (8 - nr);
            }
        }
        double dt = now_sec() - t0;
        double mbs = (double)reps * n * sizeof(uint16_t) / dt / 1e6;
        if (mbs > best) best = mbs;
    }
    free(codes_scratch);
    return best;
}

int main(int argc, char **argv) {
    init_compress_tab();

    const int N = 1 << 14;            /* 16 K codes per partition pass */
    const int REPS = argc > 1 ? atoi(argv[1]) : 5000;

    printf("partition_probe: N=%d codes, reps=%d  VL=%llu bits\n", N, REPS,
           (unsigned long long)(svcntb() * 8));
    if (svcntb() != 32) {
        printf("WARNING: VL != 256 bits; SVE-256 path will not be representative.\n");
    }

    uint16_t *src   = aligned_alloc(64, N * sizeof(uint16_t));
    uint16_t *left  = aligned_alloc(64, N * sizeof(uint16_t));
    uint16_t *right = aligned_alloc(64, N * sizeof(uint16_t));
    uint8_t  *masks = aligned_alloc(64, N / 8);
    if (!src || !left || !right || !masks) { fprintf(stderr,"OOM\n"); return 1; }

    uint64_t s = 0xdecaf12345ull;
    for (int i = 0; i < N; i++)    src[i]   = (uint16_t)xorshift64(&s);
    for (int i = 0; i < N/8; i++)  masks[i] = (uint8_t)xorshift64(&s);

    /* Correctness: run both, check left/right results match. */
    uint16_t *l_ref = aligned_alloc(64, N * sizeof(uint16_t));
    uint16_t *r_ref = aligned_alloc(64, N * sizeof(uint16_t));
    uint16_t *l_sve = aligned_alloc(64, N * sizeof(uint16_t));
    uint16_t *r_sve = aligned_alloc(64, N * sizeof(uint16_t));
    uint16_t *codes_scratch = aligned_alloc(64, N * sizeof(uint16_t));

    int nL_ref = 0, nR_ref = 0;
    memcpy(codes_scratch, src, N * sizeof(uint16_t));
    for (int j = 0; j + 8 <= N; j += 8) {
        int nr = partition_8_neon(codes_scratch, j, masks[j >> 3],
                                    l_ref, nL_ref, r_ref, nR_ref);
        nR_ref += nr; nL_ref += (8 - nr);
    }
    int nL_sve = 0, nR_sve = 0;
    memcpy(codes_scratch, src, N * sizeof(uint16_t));
    for (int j = 0; j + 8 <= N; j += 8) {
        int nr = partition_8_sve256(codes_scratch, j, masks[j >> 3],
                                      l_sve, nL_sve, r_sve, nR_sve);
        nR_sve += nr; nL_sve += (8 - nr);
    }
    int ok = (nL_ref == nL_sve) && (nR_ref == nR_sve)
          && memcmp(l_ref, l_sve, nL_ref * sizeof(uint16_t)) == 0
          && memcmp(r_ref, r_sve, nR_ref * sizeof(uint16_t)) == 0;
    if (!ok) {
        fprintf(stderr, "MISMATCH: nL ref=%d sve=%d  nR ref=%d sve=%d\n",
                nL_ref, nL_sve, nR_ref, nR_sve);
        return 2;
    }
    printf("correctness: OK (nL=%d nR=%d, NEON == SVE-256)\n\n", nL_ref, nR_ref);

    double neon_mbs = measure(partition_8_neon,
                               src, N, masks, left, right, REPS);
    double sve_mbs  = measure(partition_8_sve256,
                               src, N, masks, left, right, REPS);

    printf("%-12s  in-MB/s\n", "variant");
    printf("%-12s  %7.0f\n", "NEON-128", neon_mbs);
    printf("%-12s  %7.0f\n", "SVE-256",  sve_mbs);
    printf("%-12s  %7.2fx\n", "sve/neon", sve_mbs / neon_mbs);

    free(src); free(left); free(right); free(masks);
    free(l_ref); free(r_ref); free(l_sve); free(r_sve); free(codes_scratch);
    return 0;
}
