/* bench_enc_init.c — microbench for the encoder's per-block init pass.
 *
 *   codes_la[i] = code_la_table[symbols[i]]   for i in 0..BLK
 *
 * The table is 256 × uint16 = 512 bytes (fits L1d easily).  Goal: hit
 * M4's LSU throughput ceiling (3 loads + 2 stores per cycle).
 *
 * Variants:
 *   SCALAR    one-element-at-a-time loop (current encoder)
 *   UNROLL8   manual 8-way unroll, scalar gathers and one wide write
 *   UNROLL16  16-way unroll
 *   NEON_TBL  hierarchical vqtbl4q_u8 over 4 chunks of 64 entries
 *             (= 64 bytes for one byte-half of the table) + or-blend
 */
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifndef __aarch64__
int main(void) { puts("bench_enc_init: NEON only"); return 0; }
#else
#include <arm_neon.h>

#define BLK 8192

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static __attribute__((noinline))
void init_scalar(uint16_t *dst, const uint8_t *symbols,
                  const uint16_t *table, int n)
{
    for (int i = 0; i < n; i++) dst[i] = table[symbols[i]];
}

static __attribute__((noinline))
void init_unroll8(uint16_t *dst, const uint8_t *symbols,
                   const uint16_t *table, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16_t c0 = table[symbols[i+0]];
        uint16_t c1 = table[symbols[i+1]];
        uint16_t c2 = table[symbols[i+2]];
        uint16_t c3 = table[symbols[i+3]];
        uint16_t c4 = table[symbols[i+4]];
        uint16_t c5 = table[symbols[i+5]];
        uint16_t c6 = table[symbols[i+6]];
        uint16_t c7 = table[symbols[i+7]];
        dst[i+0] = c0; dst[i+1] = c1; dst[i+2] = c2; dst[i+3] = c3;
        dst[i+4] = c4; dst[i+5] = c5; dst[i+6] = c6; dst[i+7] = c7;
    }
    for (; i < n; i++) dst[i] = table[symbols[i]];
}

static __attribute__((noinline))
void init_unroll16(uint16_t *dst, const uint8_t *symbols,
                    const uint16_t *table, int n)
{
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16_t c0 = table[symbols[i+0]],  c1 = table[symbols[i+1]];
        uint16_t c2 = table[symbols[i+2]],  c3 = table[symbols[i+3]];
        uint16_t c4 = table[symbols[i+4]],  c5 = table[symbols[i+5]];
        uint16_t c6 = table[symbols[i+6]],  c7 = table[symbols[i+7]];
        uint16_t c8 = table[symbols[i+8]],  c9 = table[symbols[i+9]];
        uint16_t cA = table[symbols[i+10]], cB = table[symbols[i+11]];
        uint16_t cC = table[symbols[i+12]], cD = table[symbols[i+13]];
        uint16_t cE = table[symbols[i+14]], cF = table[symbols[i+15]];
        dst[i+0]=c0;  dst[i+1]=c1;  dst[i+2]=c2;  dst[i+3]=c3;
        dst[i+4]=c4;  dst[i+5]=c5;  dst[i+6]=c6;  dst[i+7]=c7;
        dst[i+8]=c8;  dst[i+9]=c9;  dst[i+10]=cA; dst[i+11]=cB;
        dst[i+12]=cC; dst[i+13]=cD; dst[i+14]=cE; dst[i+15]=cF;
    }
    for (; i < n; i++) dst[i] = table[symbols[i]];
}

/* Hierarchical NEON TBL.  Split 256-byte halves (low/high byte of each
 * 16-bit code) into 4 chunks of 64 entries each.  Per 16-char input:
 *   - 4× vqtbl4q_u8(chunk_k, chars - k*64) for the low byte half
 *   - 4× vqtbl4q_u8(...) for the high byte half
 *   - 3× vorrq for each half (out-of-range chunks return 0)
 *   - vzip1q_u8 + vzip2q_u8 to interleave lo/hi into uint16
 *   - 2× vst1q_u8
 * Total: 8 TBL + 6 OR + 2 ZIP + 2 ST per 16 chars = ~18 NEON ops.
 *
 * We pre-build the chunk tables at setup time.  Each chunk = 64 bytes
 * = 4× uint8x16_t (held in a uint8x16x4_t struct). */
typedef struct {
    uint8x16x4_t lo[4];   /* 4 chunks × 64 bytes of low-byte values */
    uint8x16x4_t hi[4];   /* 4 chunks × 64 bytes of high-byte values */
} tbl4_split_t;

static void build_tbl4_split(tbl4_split_t *ts, const uint16_t *table)
{
    uint8_t lo[256], hi[256];
    for (int s = 0; s < 256; s++) {
        lo[s] = (uint8_t)(table[s] & 0xFF);
        hi[s] = (uint8_t)(table[s] >> 8);
    }
    for (int k = 0; k < 4; k++) {
        ts->lo[k].val[0] = vld1q_u8(lo + k * 64 +  0);
        ts->lo[k].val[1] = vld1q_u8(lo + k * 64 + 16);
        ts->lo[k].val[2] = vld1q_u8(lo + k * 64 + 32);
        ts->lo[k].val[3] = vld1q_u8(lo + k * 64 + 48);
        ts->hi[k].val[0] = vld1q_u8(hi + k * 64 +  0);
        ts->hi[k].val[1] = vld1q_u8(hi + k * 64 + 16);
        ts->hi[k].val[2] = vld1q_u8(hi + k * 64 + 32);
        ts->hi[k].val[3] = vld1q_u8(hi + k * 64 + 48);
    }
}

static __attribute__((noinline))
void init_neon_tbl(uint16_t *dst, const uint8_t *symbols,
                    const tbl4_split_t *ts, int n)
{
    const uint8x16_t off1 = vdupq_n_u8(64);
    const uint8x16_t off2 = vdupq_n_u8(128);
    const uint8x16_t off3 = vdupq_n_u8(192);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t chars = vld1q_u8(symbols + i);
        /* For each chunk k=0..3, indices for that chunk are
         * (chars - k*64).  vqtbl4q_u8 returns 0 for indices >= 64. */
        uint8x16_t i0 = chars;
        uint8x16_t i1 = vsubq_u8(chars, off1);
        uint8x16_t i2 = vsubq_u8(chars, off2);
        uint8x16_t i3 = vsubq_u8(chars, off3);
        /* LO byte */
        uint8x16_t r0 = vqtbl4q_u8(ts->lo[0], i0);
        uint8x16_t r1 = vqtbl4q_u8(ts->lo[1], i1);
        uint8x16_t r2 = vqtbl4q_u8(ts->lo[2], i2);
        uint8x16_t r3 = vqtbl4q_u8(ts->lo[3], i3);
        uint8x16_t lo_v = vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3));
        /* HI byte */
        uint8x16_t s0 = vqtbl4q_u8(ts->hi[0], i0);
        uint8x16_t s1 = vqtbl4q_u8(ts->hi[1], i1);
        uint8x16_t s2 = vqtbl4q_u8(ts->hi[2], i2);
        uint8x16_t s3 = vqtbl4q_u8(ts->hi[3], i3);
        uint8x16_t hi_v = vorrq_u8(vorrq_u8(s0, s1), vorrq_u8(s2, s3));
        /* Interleave lo+hi into uint16: dst[0]=(lo0,hi0), dst[1]=(lo1,hi1),... */
        uint8x16_t out_lo = vzip1q_u8(lo_v, hi_v);
        uint8x16_t out_hi = vzip2q_u8(lo_v, hi_v);
        vst1q_u8((uint8_t *)(dst + i    ), out_lo);
        vst1q_u8((uint8_t *)(dst + i + 8), out_hi);
    }
    /* Scalar tail through the original table reconstructed from the split. */
    if (i < n) {
        uint8_t lo[256], hi[256];
        vst1q_u8(lo +  0, ts->lo[0].val[0]); vst1q_u8(lo + 16, ts->lo[0].val[1]);
        vst1q_u8(lo + 32, ts->lo[0].val[2]); vst1q_u8(lo + 48, ts->lo[0].val[3]);
        vst1q_u8(lo + 64, ts->lo[1].val[0]); vst1q_u8(lo + 80, ts->lo[1].val[1]);
        vst1q_u8(lo + 96, ts->lo[1].val[2]); vst1q_u8(lo +112, ts->lo[1].val[3]);
        vst1q_u8(lo +128, ts->lo[2].val[0]); vst1q_u8(lo +144, ts->lo[2].val[1]);
        vst1q_u8(lo +160, ts->lo[2].val[2]); vst1q_u8(lo +176, ts->lo[2].val[3]);
        vst1q_u8(lo +192, ts->lo[3].val[0]); vst1q_u8(lo +208, ts->lo[3].val[1]);
        vst1q_u8(lo +224, ts->lo[3].val[2]); vst1q_u8(lo +240, ts->lo[3].val[3]);
        vst1q_u8(hi +  0, ts->hi[0].val[0]); vst1q_u8(hi + 16, ts->hi[0].val[1]);
        vst1q_u8(hi + 32, ts->hi[0].val[2]); vst1q_u8(hi + 48, ts->hi[0].val[3]);
        vst1q_u8(hi + 64, ts->hi[1].val[0]); vst1q_u8(hi + 80, ts->hi[1].val[1]);
        vst1q_u8(hi + 96, ts->hi[1].val[2]); vst1q_u8(hi +112, ts->hi[1].val[3]);
        vst1q_u8(hi +128, ts->hi[2].val[0]); vst1q_u8(hi +144, ts->hi[2].val[1]);
        vst1q_u8(hi +160, ts->hi[2].val[2]); vst1q_u8(hi +176, ts->hi[2].val[3]);
        vst1q_u8(hi +192, ts->hi[3].val[0]); vst1q_u8(hi +208, ts->hi[3].val[1]);
        vst1q_u8(hi +224, ts->hi[3].val[2]); vst1q_u8(hi +240, ts->hi[3].val[3]);
        for (; i < n; i++) {
            uint8_t s = symbols[i];
            dst[i] = (uint16_t)lo[s] | ((uint16_t)hi[s] << 8);
        }
    }
}

int main(int argc, char **argv)
{
    int repeats = (argc > 1) ? atoi(argv[1]) : 500000;
    if (repeats < 1) repeats = 1;

    static uint16_t table[256];
    static uint8_t  symbols[BLK];
    static uint16_t dst_ref[BLK];
    static uint16_t dst[BLK];
    srand(0xBEEF);
    for (int s = 0; s < 256; s++) table[s] = (uint16_t)rand();
    for (int i = 0; i < BLK; i++) symbols[i] = (uint8_t)(rand() & 0xFF);

    init_scalar(dst_ref, symbols, table, BLK);

    tbl4_split_t ts;
    build_tbl4_split(&ts, table);

    struct { const char *name; double ns_elem; int ok; } rows[4];
    int ri = 0;

#define RUN(name_str, fn) do { \
    memset(dst, 0xAA, sizeof(dst)); \
    double t0 = now_sec(); \
    for (int r = 0; r < repeats; r++) (fn); \
    double t1 = now_sec(); \
    rows[ri].name = (name_str); \
    rows[ri].ns_elem = (t1 - t0) * 1e9 / ((double)repeats * BLK); \
    rows[ri].ok = (memcmp(dst, dst_ref, sizeof(dst)) == 0); \
    ri++; \
} while (0)

    RUN("SCALAR    (loop)",
        init_scalar(dst, symbols, table, BLK));
    RUN("UNROLL8   (8-way scalar)",
        init_unroll8(dst, symbols, table, BLK));
    RUN("UNROLL16  (16-way scalar)",
        init_unroll16(dst, symbols, table, BLK));
    RUN("NEON_TBL  (4-chunk vqtbl4q_u8)",
        init_neon_tbl(dst, symbols, &ts, BLK));

    printf("\n=== enc_init microbench (N=%d, repeats=%d) ===\n\n", BLK, repeats);
    printf("  %-36s %10s %10s  %s\n",
           "variant", "ns/elem", "GB/s", "correct");
    printf("  -----------------------------------------------------------------\n");
    for (int i = 0; i < ri; i++)
        printf("  %-36s %10.3f %10.2f  %s\n",
               rows[i].name, rows[i].ns_elem,
               2.0 / rows[i].ns_elem, rows[i].ok ? "yes" : "NO");
    printf("\n  Speedup vs SCALAR:\n");
    for (int i = 1; i < ri; i++)
        printf("    %-36s %.2fx\n", rows[i].name,
               rows[0].ns_elem / rows[i].ns_elem);
    return 0;
}

#endif
