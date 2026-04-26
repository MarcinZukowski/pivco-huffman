/* Microbenchmark: scatter vs partition vs flat-decode per-element cost.
   Build: cc -O2 -o bench_micro bench/bench_micro.c -I include -I src
   (NEON version auto-detected on aarch64) */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef __aarch64__
#include <arm_neon.h>
#include "pivco_huffman_neon_flat.h"  /* flat_d{2..6}_spread() + tables */
#define HAS_NEON 1
#else
#define HAS_NEON 0
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

/* ---- Memset ---- */

__attribute__((noinline)) static void bench_memset(uint8_t *symbols, int n, uint8_t sym, int reps)
{
    for (int r = 0; r < reps; r++)
        memset(symbols, sym, (size_t)n);
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
 * These benchmarks isolate (spread + TBL + store) per element at each
 * D, so we can compare their per-element cost against the
 * scatter/partition floors above.
 * ============================================================ */

/* spread() helpers + their tables come from pivco_huffman_neon_flat.h
 * (shared with src/pivco_huffman_neon.c — single source of truth). */

/* ---- D=2: 4 packed bytes → 16 codes (uint8x16_t), 1 vqtbl1q_u8 ---- */
__attribute__((noinline))
static void bench_flat_direct_d2(uint8_t *out, const uint8_t *bm,
                                  const uint8_t *c2s, int n, int reps) {
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i + 16 <= n; i += 16) {
            uint8x16_t codes = flat_d2_spread(bm + (i >> 2));
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
            uint8x16_t codes = flat_d2_spread(bm + (i >> 2));
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
            uint8x8_t lo = flat_d3_spread(bm + (((i)      * 3) >> 3));
            uint8x8_t hi = flat_d3_spread(bm + (((i + 8)  * 3) >> 3));
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
            uint8x8_t codes = flat_d3_spread(bm + ((i * 3) >> 3));
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
            uint8x16_t codes = flat_d4_spread(bm + (i >> 1));
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
            uint8x16_t codes = flat_d4_spread(bm + (i >> 1));
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
            uint8x8_t lo = flat_d5_spread(bm + (((i)     * 5) >> 3));
            uint8x8_t hi = flat_d5_spread(bm + (((i + 8) * 5) >> 3));
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
            uint8x8_t codes = flat_d5_spread(bm + ((i * 5) >> 3));
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
            uint8x8_t lo = flat_d6_spread(bm + (((i)     * 6) >> 3));
            uint8x8_t hi = flat_d6_spread(bm + (((i + 8) * 6) >> 3));
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
            uint8x8_t codes = flat_d6_spread(bm + ((i * 6) >> 3));
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
     * +16 bytes pad for the 6/8-byte memcpy reads in flat_d{4,5,6}_spread). */
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

    /* ---- Flat-subtree decode: per-D (spread + TBL + store) ---- */
    printf("\n-- flat-subtree decode (spread + TBL + store) --\n");

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
#endif

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
