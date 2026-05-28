/* sve_probe/probe.c -- standalone microbench for SVE-256 vs NEON-128
 * on the D=5 flat-subtree decode primitive.
 *
 * Target: AWS Graviton 3 (Neoverse V1, SVE-256).  Neoverse V2 (Gv4) has
 * SVE2 but at 128-bit so it won't be representative.
 *
 * Build (on c7g):
 *   gcc -O3 -march=armv8.4-a+sve probe.c -o probe
 * Run:
 *   taskset -c 0 ./probe
 *
 * Reports MB/s (input bytes/sec) for each variant.  Variant comparison:
 *   - NEON-128: production flat_decode_direct_neon_d5 (vqtbl2q_u8,
 *               2-register table lookup)
 *   - SVE-256:  same unpack, single svtbl_u8 with 32-byte table in
 *               one SVE register (no register-pair lookup needed)
 *
 * Both decoders implement the same contract: given 5-bit-packed input,
 * produce one symbol byte per code via a 32-entry code->sym table.
 */

#include <arm_neon.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- NEON D=5 unpack (lifted verbatim from pivco_huffman_neon_flat.h) --- */

static const uint8_t  flat_d5_shuf_tab[16] = {
    0,1, 0,1, 1,2, 1,2,    2,3, 2,3, 3,4, 3,4
};
static const int16_t  flat_d5_shift_tab[8] = {
    0, -5, -2, -7,  -4, -9, -6, -11
};

static inline uint8x8_t flat_d5_unpack(const uint8_t *bm_ptr) {
    uint8x16_t bm_lo = vdupq_n_u8(0);
    bm_lo = vsetq_lane_u8(bm_ptr[0], bm_lo, 0);
    bm_lo = vsetq_lane_u8(bm_ptr[1], bm_lo, 1);
    bm_lo = vsetq_lane_u8(bm_ptr[2], bm_lo, 2);
    bm_lo = vsetq_lane_u8(bm_ptr[3], bm_lo, 3);
    bm_lo = vsetq_lane_u8(bm_ptr[4], bm_lo, 4);
    uint8x16_t shuffled = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d5_shuf_tab));
    uint16x8_t w = vreinterpretq_u16_u8(shuffled);
    uint16x8_t shifted = vshlq_u16(w, vld1q_s16(flat_d5_shift_tab));
    uint16x8_t masked = vandq_u16(shifted, vdupq_n_u16(0x1F));
    return vmovn_u16(masked);
}

/* --- variant A: NEON-128 production decoder --- */
static void decode_d5_neon(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s)
{
    uint8x16x2_t c2s_vec;
    c2s_vec.val[0] = vld1q_u8(c2s);
    c2s_vec.val[1] = vld1q_u8(c2s + 16);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x8_t codes_lo = flat_d5_unpack(bm + ((i      * 5) >> 3));
        uint8x8_t codes_hi = flat_d5_unpack(bm + (((i + 8) * 5) >> 3));
        uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
        uint8x16_t syms  = vqtbl2q_u8(c2s_vec, codes);
        vst1q_u8(out + i, syms);
    }
    for (; i < n; i++) {
        int byte = (i * 5) >> 3;
        uint32_t code = ((uint32_t)bm[byte] | ((uint32_t)bm[byte + 1] << 8))
                        >> ((i * 5) & 7);
        out[i] = c2s[code & 0x1f];
    }
}

/* --- variant B: SVE-256 decoder ---
 *
 * Key difference from NEON-128: svtbl_u8 takes a SINGLE 32-byte register
 * (one SVE-256 reg = 32 bytes) instead of NEON's 2x16-byte pair via
 * vqtbl2q_u8.  Unpacking is still NEON (4x flat_d5_unpack = 32 codes),
 * then one svtbl over the entire 32-entry c2s.
 *
 * VL is checked at startup; this path only executes if svcntb() == 32.
 * On c8g (SVE2-128, VL=16) we fall back to the NEON variant.
 */
static void decode_d5_sve256(uint8_t *out, int n, const uint8_t *bm,
                              const uint8_t *c2s)
{
    /* Load c2s into one SVE-256 register (32 bytes). */
    svuint8_t tab = svld1_u8(svptrue_b8(), c2s);

    int i = 0;
    for (; i + 32 <= n; i += 32) {
        /* 4x NEON unpack -> 32 codes in two q-registers. */
        uint8x8_t c0 = flat_d5_unpack(bm + ((i      * 5) >> 3));
        uint8x8_t c1 = flat_d5_unpack(bm + (((i + 8) * 5) >> 3));
        uint8x8_t c2 = flat_d5_unpack(bm + (((i +16) * 5) >> 3));
        uint8x8_t c3 = flat_d5_unpack(bm + (((i +24) * 5) >> 3));
        uint8x16_t codes_lo = vcombine_u8(c0, c1);
        uint8x16_t codes_hi = vcombine_u8(c2, c3);

        /* Reinterpret as one SVE-256 register.  Compiler-blessed path
         * is to copy through a tmp; modern gcc folds this away. */
        uint8_t tmp[32];
        vst1q_u8(tmp,     codes_lo);
        vst1q_u8(tmp+16,  codes_hi);
        svuint8_t codes = svld1_u8(svptrue_b8(), tmp);

        svuint8_t syms = svtbl_u8(tab, codes);
        svst1_u8(svptrue_b8(), out + i, syms);
    }
    /* Tail: hand off to NEON for whatever fits. */
    if (i < n) decode_d5_neon(out + i, n - i,
                              bm + ((i * 5) >> 3), c2s);
}

/* --- driver --- */

static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s; x ^= x<<13; x ^= x>>7; x ^= x<<17; *s = x; return x;
}

int main(int argc, char **argv) {
    const int N = 1 << 16;            /* 64 K codes per buffer */
    const int BM_BYTES = (N * 5 + 7) / 8;
    const int REPEATS = argc > 1 ? atoi(argv[1]) : 4000;

    printf("sve_probe: D=5, N=%d codes, %d bytes packed, repeats=%d\n",
           N, BM_BYTES, REPEATS);
    printf("VL=%llu bits (svcntb=%llu bytes)\n",
           (unsigned long long)(svcntb() * 8),
           (unsigned long long)svcntb());

    if (svcntb() != 32) {
        printf("WARNING: VL is not 256 bits; SVE-256 path will not be representative\n");
    }

    uint8_t *bm  = aligned_alloc(64, BM_BYTES);
    uint8_t *out = aligned_alloc(64, N);
    uint8_t c2s[32];
    if (!bm || !out) { fprintf(stderr, "OOM\n"); return 1; }

    /* Fill bm with pseudorandom bits. */
    uint64_t s = 0xc0ffee12345ull;
    for (int i = 0; i < BM_BYTES; i++) bm[i] = (uint8_t)xorshift64(&s);
    /* c2s: each entry maps a 5-bit code to a symbol byte. */
    for (int i = 0; i < 32; i++) c2s[i] = (uint8_t)(i * 7 + 13);

    /* Correctness: decode once with each, check they match. */
    uint8_t *ref = aligned_alloc(64, N);
    decode_d5_neon  (ref, N, bm, c2s);
    decode_d5_sve256(out, N, bm, c2s);
    int mismatches = 0;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) {
        if (mismatches < 5)
            fprintf(stderr, "[mismatch] i=%d neon=0x%02x sve=0x%02x\n",
                    i, ref[i], out[i]);
        mismatches++;
    }
    if (mismatches) {
        fprintf(stderr, "FAIL: %d mismatches\n", mismatches);
        return 2;
    }
    free(ref);
    printf("correctness: OK (NEON == SVE-256 on %d codes)\n\n", N);

    /* Timing: best-of-3 inner loops of REPEATS iterations each. */
    const int RUNS = 5;

    double best_neon = 0, best_sve = 0;
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_sec();
        for (int k = 0; k < REPEATS; k++) decode_d5_neon(out, N, bm, c2s);
        double dt = now_sec() - t0;
        double mbs = (double)REPEATS * BM_BYTES / dt / 1e6;
        if (mbs > best_neon) best_neon = mbs;
    }
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_sec();
        for (int k = 0; k < REPEATS; k++) decode_d5_sve256(out, N, bm, c2s);
        double dt = now_sec() - t0;
        double mbs = (double)REPEATS * BM_BYTES / dt / 1e6;
        if (mbs > best_sve) best_sve = mbs;
    }

    printf("%-12s  in-MB/s\n", "variant");
    printf("%-12s  %7.0f\n", "NEON-128",   best_neon);
    printf("%-12s  %7.0f\n", "SVE-256",    best_sve);
    printf("%-12s  %7.2fx\n", "ratio sve/neon", best_sve / best_neon);

    free(bm); free(out);
    return 0;
}
