/* bench_partition_u8_vs_u16.c — microbench for the uint8-subtree
 * proposal.
 *
 * Today's encoder partitions left-aligned uint16 codes through
 * compress_tab[256][32] (NEON vqtbl1q / SSE pshufb) or vpcompressw
 * (AVX-512 VBMI2).  If we know a subtree's remaining bits fit in 1
 * byte we could repack to uint8 and partition at twice the lane
 * density.  This bench isolates the per-element partition cost for
 * the candidate primitives across our ISAs.
 *
 * Variants:
 *   NEON u16 / stride-8    (current encoder shape: 8 uint16 / iter)
 *   NEON u8  / stride-8    (8 uint8 / iter via byte-granular compress_tab)
 *   NEON u8  / stride-16   (16 uint8 / iter via mask split + 2x pshufb)
 *   SSE  u16 / stride-8    (current x86 encoder shape)
 *   SSE  u8  / stride-8    (8 byte partition)
 *   SSE  u8  / stride-16   (split mask, 2x SSE)
 *   AVX-512 u16 / stride-32  (vpcompressw, current)
 *   AVX-512 u8  / stride-64  (vpcompressb -- the candidate win)
 *
 * Each iter mimics the encoder's cursor pattern (n_left/n_right
 * advance with popcount), so we measure realistic throughput
 * including the carry-by-popcount chain.
 *
 * Cycle / ns reported per element processed (8192-element block,
 * many outer iters for stable timing).
 */
#include "pivco_huffman.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLK 8192

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============ Byte-granular compress table (built locally). ============
 * compress_tab_byte[mask][0..7]  = right shuf (bytes where bit=1)
 * compress_tab_byte[mask][8..15] = left shuf  (bytes where bit=0)
 * Out-of-range slots use index 0x80 so TBL returns 0. */
static uint8_t compress_tab_byte[256][16] __attribute__((aligned(16)));
/* compress_popcnt[256] from the shared NEON header on aarch64;
 * inline a local copy for portability of this microbench. */
static uint8_t local_popcnt[256];

static void init_tables_local(void) {
    for (int m = 0; m < 256; m++) {
        int nr = 0, nl = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) compress_tab_byte[m][     nr++] = (uint8_t)k;
            else              compress_tab_byte[m][8 +  nl++] = (uint8_t)k;
        }
        for (; nr < 8; nr++) compress_tab_byte[m][    nr] = 0x80;
        for (; nl < 8; nl++) compress_tab_byte[m][8 + nl] = 0x80;
        local_popcnt[m] = (uint8_t)__builtin_popcount(m);
    }
}

#ifdef __aarch64__
#include "../pivco_huffman_neon_common.h"   /* compress_tab[256][32] + compress_popcnt[256] */
#include <arm_neon.h>

static __attribute__((noinline))
double bench_neon_u16_s8(const uint16_t *src, const uint8_t *masks,
                          uint16_t *left, uint16_t *right,
                          int n, int repeats)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (int r = 0; r < repeats; r++) {
        int nl = 0, nr = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
            const uint8_t *tab = compress_tab[masks[j >> 3]];
            uint8x16_t shuf_r = vld1q_u8(tab);
            uint8x16_t shuf_l = vld1q_u8(tab + 16);
            uint8x16_t rv = vqtbl1q_u8(data, shuf_r);
            uint8x16_t lv = vqtbl1q_u8(data, shuf_l);
            int x = compress_popcnt[masks[j >> 3]];
            vst1q_u8((uint8_t *)(right + nr), rv);
            vst1q_u8((uint8_t *)(left  + nl), lv);
            nr += x;
            nl += (8 - x);
        }
        sink ^= nl ^ nr;
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)repeats * n);
}

static __attribute__((noinline))
double bench_neon_u8_s8(const uint8_t *src, const uint8_t *masks,
                        uint8_t *left, uint8_t *right,
                        int n, int repeats)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (int r = 0; r < repeats; r++) {
        int nl = 0, nr = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            /* Load 8 bytes into low half of __m128 (high half = 0). */
            uint8x8_t data8 = vld1_u8(src + j);
            uint8x16_t data  = vcombine_u8(data8, vdup_n_u8(0));
            uint8x16_t shuf  = vld1q_u8(compress_tab_byte[masks[j >> 3]]);
            /* shuf low half = right indices, high half = left indices. */
            uint8x16_t out   = vqtbl1q_u8(data, shuf);
            int x = local_popcnt[masks[j >> 3]];
            vst1_u8(right + nr, vget_low_u8(out));
            vst1_u8(left  + nl, vget_high_u8(out));
            nr += x;
            nl += (8 - x);
        }
        sink ^= nl ^ nr;
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)repeats * n);
}

static __attribute__((noinline))
double bench_neon_u8_s16(const uint8_t *src, const uint8_t *masks,
                         uint8_t *left, uint8_t *right,
                         int n, int repeats)
{
    /* Stride-16: load 16 bytes, split mask into 2x 8-bit halves,
     * run two byte-granular partition_8 sequentially. */
    double t0 = now_sec();
    volatile int sink = 0;
    for (int r = 0; r < repeats; r++) {
        int nl = 0, nr = 0;
        for (int j = 0; j + 16 <= n; j += 16) {
            uint8x16_t data = vld1q_u8(src + j);
            uint8_t mask_lo = masks[j >> 3];
            uint8_t mask_hi = masks[(j >> 3) + 1];
            uint8x16_t shuf_lo = vld1q_u8(compress_tab_byte[mask_lo]);
            uint8x16_t shuf_hi = vld1q_u8(compress_tab_byte[mask_hi]);
            uint8x8_t data_lo8 = vget_low_u8(data);
            uint8x8_t data_hi8 = vget_high_u8(data);
            uint8x16_t out_lo = vqtbl1q_u8(vcombine_u8(data_lo8, vdup_n_u8(0)), shuf_lo);
            uint8x16_t out_hi = vqtbl1q_u8(vcombine_u8(data_hi8, vdup_n_u8(0)), shuf_hi);
            int nr_lo = local_popcnt[mask_lo];
            int nr_hi = local_popcnt[mask_hi];
            vst1_u8(right + nr,                 vget_low_u8(out_lo));
            vst1_u8(left  + nl,                 vget_high_u8(out_lo));
            nr += nr_lo;
            nl += (8 - nr_lo);
            vst1_u8(right + nr,                 vget_low_u8(out_hi));
            vst1_u8(left  + nl,                 vget_high_u8(out_hi));
            nr += nr_hi;
            nl += (8 - nr_hi);
        }
        sink ^= nl ^ nr;
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)repeats * n);
}

#endif  /* __aarch64__ */

#if defined(__x86_64__) && defined(__SSE4_1__)
#include <smmintrin.h>
/* Build our own local copy of compress_tab[256][32] for uint16
 * partitioning -- the encoder's symbol is file-static. */
static uint8_t local_compress_tab_u16[256][32] __attribute__((aligned(32)));

static void init_local_compress_tab_u16(void) {
    for (int m = 0; m < 256; m++) {
        int nr = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) {
                local_compress_tab_u16[m][2*nr]     = (uint8_t)(2*k);
                local_compress_tab_u16[m][2*nr + 1] = (uint8_t)(2*k + 1);
                nr++;
            }
        }
        for (int s = 2*nr; s < 16; s++) local_compress_tab_u16[m][s] = 0x80;
        int nl = 0;
        for (int k = 0; k < 8; k++) {
            if (!(m & (1 << k))) {
                local_compress_tab_u16[m][16 + 2*nl]     = (uint8_t)(2*k);
                local_compress_tab_u16[m][16 + 2*nl + 1] = (uint8_t)(2*k + 1);
                nl++;
            }
        }
        for (int s = 16 + 2*nl; s < 32; s++) local_compress_tab_u16[m][s] = 0x80;
    }
}

static __attribute__((noinline))
double bench_sse_u16_s8(const uint16_t *src, const uint8_t *masks,
                         uint16_t *left, uint16_t *right,
                         int n, int repeats)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (int r = 0; r < repeats; r++) {
        int nl = 0, nr = 0;
        for (int j = 0; j + 8 <= n; j += 8) {
            __m128i data   = _mm_loadu_si128((const __m128i *)(src + j));
            uint8_t mask   = masks[j >> 3];
            __m128i shuf_r = _mm_load_si128((const __m128i *) local_compress_tab_u16[mask]);
            __m128i shuf_l = _mm_load_si128((const __m128i *)(local_compress_tab_u16[mask] + 16));
            __m128i rv     = _mm_shuffle_epi8(data, shuf_r);
            __m128i lv     = _mm_shuffle_epi8(data, shuf_l);
            int x = local_popcnt[mask];
            _mm_storeu_si128((__m128i *)(right + nr), rv);
            _mm_storeu_si128((__m128i *)(left  + nl), lv);
            nr += x;
            nl += (8 - x);
        }
        sink ^= nl ^ nr;
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)repeats * n);
}

static __attribute__((noinline))
double bench_sse_u8_s16(const uint8_t *src, const uint8_t *masks,
                        uint8_t *left, uint8_t *right,
                        int n, int repeats)
{
    /* Same shape as NEON stride-16, using SSE pshufb on byte indices. */
    double t0 = now_sec();
    volatile int sink = 0;
    for (int r = 0; r < repeats; r++) {
        int nl = 0, nr = 0;
        for (int j = 0; j + 16 <= n; j += 16) {
            uint8_t mask_lo = masks[j >> 3];
            uint8_t mask_hi = masks[(j >> 3) + 1];
            __m128i shuf_lo = _mm_load_si128((const __m128i *)compress_tab_byte[mask_lo]);
            __m128i shuf_hi = _mm_load_si128((const __m128i *)compress_tab_byte[mask_hi]);
            __m128i data_lo = _mm_loadl_epi64((const __m128i *)(src + j     ));
            __m128i data_hi = _mm_loadl_epi64((const __m128i *)(src + j + 8 ));
            __m128i out_lo  = _mm_shuffle_epi8(data_lo, shuf_lo);
            __m128i out_hi  = _mm_shuffle_epi8(data_hi, shuf_hi);
            int nr_lo = local_popcnt[mask_lo];
            int nr_hi = local_popcnt[mask_hi];
            /* Store lo half (right), then hi half (left), per partition. */
            _mm_storel_epi64((__m128i *)(right + nr), out_lo);
            _mm_storel_epi64((__m128i *)(left  + nl), _mm_unpackhi_epi64(out_lo, out_lo));
            nr += nr_lo;  nl += (8 - nr_lo);
            _mm_storel_epi64((__m128i *)(right + nr), out_hi);
            _mm_storel_epi64((__m128i *)(left  + nl), _mm_unpackhi_epi64(out_hi, out_hi));
            nr += nr_hi;  nl += (8 - nr_hi);
        }
        sink ^= nl ^ nr;
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)repeats * n);
}
#endif  /* SSE4.1 */

#if defined(__x86_64__) && defined(__AVX512VBMI2__)
#include <immintrin.h>

/* ============ "Realistic" AVX-512 variants ============
 * Match the actual encoder body:
 *   - mask is COMPUTED from data (sll + movepi*_mask), not loaded — creates
 *     the load→shift→mask→compress serial dep
 *   - bitmap byte/word is written every stride
 *   - the function is invoked many times with small n through noinline
 *     wrappers, so function-call overhead per partition is included
 * Returned popcount sums prevent the compiler from eliding the work.   */

static __attribute__((noinline))
int realistic_avx512_u16_call(const uint16_t *src, int n, int depth,
                               uint16_t *left, uint16_t *right, uint8_t *bm)
{
    __m128i shift_cnt = _mm_cvtsi32_si128(depth);
    int nl = 0, nr = 0;
    int j = 0;
    for (; j + 32 <= n; j += 32) {
        __m512i data    = _mm512_loadu_si512((const __m512i *)(src + j));
        __m512i shifted = _mm512_sll_epi16(data, shift_cnt);
        uint32_t m = _mm512_movepi16_mask(shifted);
        memcpy(bm + (j >> 3), &m, 4);
        __m512i rv = _mm512_maskz_compress_epi16((__mmask32)m,  data);
        __m512i lv = _mm512_maskz_compress_epi16((__mmask32)~m, data);
        _mm512_storeu_si512((__m512i *)(right + nr), rv);
        _mm512_storeu_si512((__m512i *)(left  + nl), lv);
        int x = __builtin_popcount(m);
        nr += x;
        nl += 32 - x;
    }
    /* Stride-8 tail (mirrors encoder). */
    __m128i sh = _mm_cvtsi32_si128(depth);
    for (; j + 8 <= n; j += 8) {
        __m128i d  = _mm_loadu_si128((const __m128i *)(src + j));
        __m128i sd = _mm_sll_epi16(d, sh);
        __m128i bs = _mm_packs_epi16(sd, _mm_setzero_si128());
        uint8_t m  = (uint8_t)_mm_movemask_epi8(bs);
        bm[j >> 3] = m;
        __m128i rv = _mm_maskz_compress_epi16((__mmask8)m,  d);
        __m128i lv = _mm_maskz_compress_epi16((__mmask8)~m, d);
        _mm_storeu_si128((__m128i *)(right + nr), rv);
        _mm_storeu_si128((__m128i *)(left  + nl), lv);
        int x = __builtin_popcount(m);
        nr += x;
        nl += 8 - x;
    }
    return nl ^ nr;
}

static __attribute__((noinline))
int realistic_avx512_u8_call(const uint8_t *src, int n, int d_rel,
                              uint8_t *left, uint8_t *right, uint8_t *bm)
{
    __m128i shift_cnt = _mm_cvtsi32_si128(d_rel);
    int nl = 0, nr = 0;
    int j = 0;
    for (; j + 64 <= n; j += 64) {
        __m512i data    = _mm512_loadu_si512((const __m512i *)(src + j));
        __m512i shifted = _mm512_sll_epi16(data, shift_cnt);
        __mmask64 m = _mm512_movepi8_mask(shifted);
        memcpy(bm + (j >> 3), &m, 8);
        __m512i rv = _mm512_maskz_compress_epi8(m,  data);
        __m512i lv = _mm512_maskz_compress_epi8(~m, data);
        _mm512_storeu_si512((__m512i *)(right + nr), rv);
        _mm512_storeu_si512((__m512i *)(left  + nl), lv);
        int x = (int)__builtin_popcountll((unsigned long long)m);
        nr += x;
        nl += 64 - x;
    }
    /* Stride-32 + stride-16 cleanup (mirror encoder). */
    for (; j + 32 <= n; j += 32) {
        __m256i data    = _mm256_loadu_si256((const __m256i *)(src + j));
        __m256i shifted = _mm256_sll_epi16(data, shift_cnt);
        __mmask32 m = _mm256_movepi8_mask(shifted);
        memcpy(bm + (j >> 3), &m, 4);
        __m256i rv = _mm256_maskz_compress_epi8(m,  data);
        __m256i lv = _mm256_maskz_compress_epi8(~m, data);
        _mm256_storeu_si256((__m256i *)(right + nr), rv);
        _mm256_storeu_si256((__m256i *)(left  + nl), lv);
        int x = __builtin_popcount((unsigned int)m);
        nr += x;
        nl += 32 - x;
    }
    for (; j + 16 <= n; j += 16) {
        __m128i data    = _mm_loadu_si128((const __m128i *)(src + j));
        __m128i shifted = _mm_sll_epi16(data, shift_cnt);
        __mmask16 m = _mm_movepi8_mask(shifted);
        memcpy(bm + (j >> 3), &m, 2);
        __m128i rv = _mm_maskz_compress_epi8(m,  data);
        __m128i lv = _mm_maskz_compress_epi8(~m, data);
        _mm_storeu_si128((__m128i *)(right + nr), rv);
        _mm_storeu_si128((__m128i *)(left  + nl), lv);
        int x = __builtin_popcount((unsigned int)m);
        nr += x;
        nl += 16 - x;
    }
    return nl ^ nr;
}

/* ============ Recursive microbench ============
 *
 * Mirror of the encoder's call tree.  Each call: partition body, then
 * recurse left (uses codes_la portion) and recurse right (uses tmp
 * portion) — exactly the buffer-aliasing pattern of encode_node_avx512
 * / encode_node_avx512_u8.
 *
 * Halves n at each recursion level (balanced tree).  More uniform
 * than real Huffman but exercises the recursion shape the flat-loop
 * microbench skips. */

static __attribute__((noinline))
int recursive_avx512_u16_node(uint16_t *codes_la, uint16_t *tmp, uint8_t *bm,
                               int n, int depth, int max_depth)
{
    if (n < 16 || depth >= max_depth) return n ^ depth;
    int sink = realistic_avx512_u16_call(codes_la, n, depth, codes_la, tmp, bm);
    int n_left = n / 2, n_right = n - n_left;
    sink ^= recursive_avx512_u16_node(codes_la, tmp + n_right, bm + (n >> 3),
                                      n_left,  depth + 1, max_depth);
    sink ^= recursive_avx512_u16_node(tmp,      tmp + n_right, bm + (n >> 3),
                                      n_right, depth + 1, max_depth);
    return sink;
}

static __attribute__((noinline))
int recursive_avx512_u8_node(uint8_t *codes_la, uint8_t *tmp, uint8_t *bm,
                              int n, int d_rel, int max_depth)
{
    if (n < 16 || d_rel >= max_depth) return n ^ d_rel;
    int sink = realistic_avx512_u8_call(codes_la, n, d_rel, codes_la, tmp, bm);
    int n_left = n / 2, n_right = n - n_left;
    sink ^= recursive_avx512_u8_node(codes_la, tmp + n_right, bm + (n >> 3),
                                     n_left,  d_rel + 1, max_depth);
    sink ^= recursive_avx512_u8_node(tmp,      tmp + n_right, bm + (n >> 3),
                                     n_right, d_rel + 1, max_depth);
    return sink;
}

static double bench_avx512_u8_recursive(uint8_t *codes_la, uint8_t *tmp,
                                         uint8_t *bm,
                                         int n_root, int max_depth, int iters)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (int it = 0; it < iters; it++) {
        sink ^= recursive_avx512_u8_node(codes_la, tmp, bm,
                                         n_root, 0, max_depth);
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)iters * n_root * max_depth);
}

static double bench_avx512_u16_recursive(uint16_t *codes_la, uint16_t *tmp,
                                          uint8_t *bm,
                                          int n_root, int max_depth, int iters)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (int it = 0; it < iters; it++) {
        sink ^= recursive_avx512_u16_node(codes_la, tmp, bm,
                                          n_root, 0, max_depth);
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)iters * n_root * max_depth);
}

/* Outer drivers: many small partitions to mimic the encoder's recursive
 * shape.  call_n is the average elements per partition (= encoder's
 * profiled avg n per node body, ~909 for u8 and ~1559 for u16 in the
 * prose_pride profile). */
static double bench_avx512_u16_realistic(const uint16_t *src,
                                          uint16_t *left, uint16_t *right,
                                          uint8_t *bm,
                                          int call_n, int total_calls)
{
    double t0 = now_sec();
    volatile int sink = 0;
    int depth = 4;  /* arbitrary; matches "few levels of u16 then dispatch" */
    for (int r = 0; r < total_calls; r++) {
        sink ^= realistic_avx512_u16_call(src, call_n, depth, left, right, bm);
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)total_calls * call_n);
}

static double bench_avx512_u8_realistic(const uint8_t *src,
                                         uint8_t *left, uint8_t *right,
                                         uint8_t *bm,
                                         int call_n, int total_calls)
{
    double t0 = now_sec();
    volatile int sink = 0;
    int d_rel = 0;
    for (int r = 0; r < total_calls; r++) {
        sink ^= realistic_avx512_u8_call(src, call_n, d_rel, left, right, bm);
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)total_calls * call_n);
}

static __attribute__((noinline))
double bench_avx512_u16_s32(const uint16_t *src, const uint32_t *masks32,
                             uint16_t *left, uint16_t *right,
                             int n, int repeats)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (int r = 0; r < repeats; r++) {
        int nl = 0, nr = 0;
        for (int j = 0; j + 32 <= n; j += 32) {
            __m512i data = _mm512_loadu_si512((const __m512i *)(src + j));
            uint32_t m   = masks32[j >> 5];
            __m512i rv   = _mm512_maskz_compress_epi16((__mmask32)m,  data);
            __m512i lv   = _mm512_maskz_compress_epi16((__mmask32)~m, data);
            int x = __builtin_popcount(m);
            _mm512_storeu_si512((__m512i *)(right + nr), rv);
            _mm512_storeu_si512((__m512i *)(left  + nl), lv);
            nr += x;
            nl += (32 - x);
        }
        sink ^= nl ^ nr;
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)repeats * n);
}

static __attribute__((noinline))
double bench_avx512_u8_s64(const uint8_t *src, const uint64_t *masks64,
                            uint8_t *left, uint8_t *right,
                            int n, int repeats)
{
    double t0 = now_sec();
    volatile int sink = 0;
    for (int r = 0; r < repeats; r++) {
        int nl = 0, nr = 0;
        for (int j = 0; j + 64 <= n; j += 64) {
            __m512i data = _mm512_loadu_si512((const __m512i *)(src + j));
            uint64_t m   = masks64[j >> 6];
            __m512i rv   = _mm512_maskz_compress_epi8((__mmask64)m,  data);
            __m512i lv   = _mm512_maskz_compress_epi8((__mmask64)~m, data);
            int x = __builtin_popcountll(m);
            _mm512_storeu_si512((__m512i *)(right + nr), rv);
            _mm512_storeu_si512((__m512i *)(left  + nl), lv);
            nr += x;
            nl += (64 - x);
        }
        sink ^= nl ^ nr;
    }
    double t1 = now_sec();
    (void)sink;
    return (t1 - t0) * 1e9 / ((double)repeats * n);
}
#endif  /* AVX-512 VBMI2 */

int main(int argc, char **argv)
{
    int repeats = (argc > 1) ? atoi(argv[1]) : 300000;
    if (repeats < 1) repeats = 1;

    init_tables_local();
#if defined(__x86_64__) && defined(__SSE4_1__)
    init_local_compress_tab_u16();
#endif
#ifdef __aarch64__
    init_compress_table();
#endif

    /* Inputs.  All variants share the same logical content so the
     * mask histogram and popcount distribution are identical across
     * runs (otherwise comparing ns/elem would be unfair). */
    static uint16_t src_u16[BLK + 64] __attribute__((aligned(64)));
    static uint8_t  src_u8 [BLK + 64] __attribute__((aligned(64)));
    static uint8_t  masks  [BLK / 8 + 8];
    static uint32_t masks32[BLK / 32 + 1];
    static uint64_t masks64[BLK / 64 + 1];
    static uint16_t left_u16 [BLK + 64], right_u16[BLK + 64];
    static uint8_t  left_u8  [BLK + 64], right_u8 [BLK + 64];

    srand(0xBEEF);
    for (int i = 0; i < BLK; i++) {
        src_u16[i] = (uint16_t)rand();
        src_u8[i]  = (uint8_t)rand();
    }
    for (int i = 0; i < BLK / 8 + 1; i++) masks[i] = (uint8_t)rand();
    for (int i = 0; i < BLK / 32 + 1; i++)
        masks32[i] = ((uint32_t)masks[i*4 + 0])         |
                     ((uint32_t)masks[i*4 + 1] << 8)    |
                     ((uint32_t)masks[i*4 + 2] << 16)   |
                     ((uint32_t)masks[i*4 + 3] << 24);
    for (int i = 0; i < BLK / 64 + 1; i++) {
        uint64_t m = 0;
        for (int k = 0; k < 8; k++) m |= ((uint64_t)masks[i*8 + k]) << (k*8);
        masks64[i] = m;
    }

    printf("=== partition u8 vs u16 microbench (BLK=%d, repeats=%d) ===\n\n",
           BLK, repeats);
    printf("  %-36s %12s %12s\n", "variant", "ns/elem", "GB/s in");
    printf("  ---------------------------------------------------------------\n");

    struct { const char *name; double ns_elem; double bytes_per_elem; } rows[16];
    int ri = 0;

#define ROW(NAME, FN, BYTES) do { \
    rows[ri].name = NAME; \
    rows[ri].ns_elem = FN; \
    rows[ri].bytes_per_elem = BYTES; \
    ri++; \
} while (0)

#ifdef __aarch64__
    ROW("NEON u16 stride-8  (current encoder)",
        bench_neon_u16_s8(src_u16, masks, left_u16, right_u16, BLK, repeats), 2);
    ROW("NEON u8  stride-8",
        bench_neon_u8_s8(src_u8,  masks, left_u8,  right_u8,  BLK, repeats), 1);
    ROW("NEON u8  stride-16",
        bench_neon_u8_s16(src_u8, masks, left_u8,  right_u8,  BLK, repeats), 1);
#endif
#if defined(__x86_64__) && defined(__SSE4_1__)
    ROW("SSE  u16 stride-8  (current encoder)",
        bench_sse_u16_s8(src_u16, masks, left_u16, right_u16, BLK, repeats), 2);
    ROW("SSE  u8  stride-16",
        bench_sse_u8_s16(src_u8, masks, left_u8, right_u8, BLK, repeats), 1);
#endif
#if defined(__x86_64__) && defined(__AVX512VBMI2__)
    ROW("AVX-512 u16 stride-32 (vpcompressw)",
        bench_avx512_u16_s32(src_u16, masks32, left_u16, right_u16, BLK, repeats), 2);
    ROW("AVX-512 u8  stride-64 (vpcompressb)",
        bench_avx512_u8_s64(src_u8, masks64, left_u8, right_u8, BLK, repeats), 1);

    /* Realistic variants: small n + data-derived mask + bitmap store +
     * per-call function-call overhead, matched to the encoder's
     * profiled per-call element counts. */
    static uint8_t bm_buf[BLK / 8 + 64];
    int u16_call_n = 1559;   /* prose_pride avg n for enc_node_full (OFF) */
    int u8_call_n  = 909;    /* prose_pride avg n for enc_node_full_u8   */
    /* Match total work so per-element ns/elem is fairly comparable. */
    int u16_total_calls = (int)((double)BLK * repeats / u16_call_n);
    int u8_total_calls  = (int)((double)BLK * repeats / u8_call_n);

    char nameu16[80], nameu8[80];
    snprintf(nameu16, sizeof nameu16,
             "AVX-512 u16 REALISTIC (n=%d, mask-from-data, bm)", u16_call_n);
    snprintf(nameu8, sizeof nameu8,
             "AVX-512 u8  REALISTIC (n=%d, mask-from-data, bm)", u8_call_n);
    ROW(strdup(nameu16),
        bench_avx512_u16_realistic(src_u16, left_u16, right_u16, bm_buf,
                                    u16_call_n, u16_total_calls), 2);
    ROW(strdup(nameu8),
        bench_avx512_u8_realistic(src_u8, left_u8, right_u8, bm_buf,
                                   u8_call_n, u8_total_calls), 1);

    /* Recursive variants: mirror encoder's call tree (each node partitions
     * then recurses left/right with halving n).  Same partition body as
     * REALISTIC; the difference is the recursion + buffer-aliasing shape. */
    int rec_n_u16 = 8192;   /* matches encoder root */
    int rec_d_u16 = 11;     /* full tree depth */
    int rec_n_u8  = 4096;   /* matches u8 dispatch entry n */
    int rec_d_u8  = 8;      /* u8 path goes through ~8 levels */
    int rec_iters_u16 = (int)((double)BLK * repeats / (rec_n_u16 * rec_d_u16));
    int rec_iters_u8  = (int)((double)BLK * repeats / (rec_n_u8  * rec_d_u8));
    if (rec_iters_u16 < 1) rec_iters_u16 = 1;
    if (rec_iters_u8  < 1) rec_iters_u8  = 1;

    /* Need uint16 tmp = 2x n_root elements; reuse our static bufs (they
     * are sized [BLK+64], BLK=8192, so left_u16+right_u16 packed is 2x). */
    static uint16_t rec_codes_la_u16[BLK + 64];
    static uint16_t rec_tmp_u16     [BLK * 2 + 64];
    static uint8_t  rec_codes_la_u8 [BLK + 64];
    static uint8_t  rec_tmp_u8      [BLK * 2 + 64];
    memcpy(rec_codes_la_u16, src_u16, sizeof rec_codes_la_u16);
    memcpy(rec_codes_la_u8,  src_u8,  sizeof rec_codes_la_u8);

    char rname16[80], rname8[80];
    snprintf(rname16, sizeof rname16,
             "AVX-512 u16 RECURSIVE (n_root=%d, depth=%d)", rec_n_u16, rec_d_u16);
    snprintf(rname8, sizeof rname8,
             "AVX-512 u8  RECURSIVE (n_root=%d, depth=%d)", rec_n_u8,  rec_d_u8);
    ROW(strdup(rname16),
        bench_avx512_u16_recursive(rec_codes_la_u16, rec_tmp_u16, bm_buf,
                                    rec_n_u16, rec_d_u16, rec_iters_u16), 2);
    ROW(strdup(rname8),
        bench_avx512_u8_recursive(rec_codes_la_u8, rec_tmp_u8, bm_buf,
                                   rec_n_u8, rec_d_u8, rec_iters_u8), 1);

    /* Apples-to-apples: u16 with the SAME recursion shape as u8
     * (n_root=4096, depth=8) so the per-call n distribution is identical. */
    int rec_iters_u16_a2a =
        (int)((double)BLK * repeats / (rec_n_u8 * rec_d_u8));
    if (rec_iters_u16_a2a < 1) rec_iters_u16_a2a = 1;
    char rname16_a2a[80];
    snprintf(rname16_a2a, sizeof rname16_a2a,
             "AVX-512 u16 RECURSIVE same-shape (n=%d, depth=%d)",
             rec_n_u8, rec_d_u8);
    ROW(strdup(rname16_a2a),
        bench_avx512_u16_recursive(rec_codes_la_u16, rec_tmp_u16, bm_buf,
                                    rec_n_u8, rec_d_u8, rec_iters_u16_a2a), 2);

    /* Also: shallow recursion that matches what the encoder actually
     * does (flat-path bails out fast; profile shows avg n=1559 for u16,
     * suggesting effective depth ~3-4). */
    int rec_d_shallow = 4;
    int rec_iters_shallow =
        (int)((double)BLK * repeats / (8192 * rec_d_shallow));
    if (rec_iters_shallow < 1) rec_iters_shallow = 1;
    char rname_shallow_u16[80], rname_shallow_u8[80];
    snprintf(rname_shallow_u16, sizeof rname_shallow_u16,
             "AVX-512 u16 RECURSIVE shallow (n=8192, depth=%d)", rec_d_shallow);
    snprintf(rname_shallow_u8, sizeof rname_shallow_u8,
             "AVX-512 u8  RECURSIVE shallow (n=8192, depth=%d)", rec_d_shallow);
    ROW(strdup(rname_shallow_u16),
        bench_avx512_u16_recursive(rec_codes_la_u16, rec_tmp_u16, bm_buf,
                                    8192, rec_d_shallow, rec_iters_shallow), 2);
    /* For u8: src is uint8, n_root must fit codes_la_u8 (BLK+64). */
    ROW(strdup(rname_shallow_u8),
        bench_avx512_u8_recursive(rec_codes_la_u8, rec_tmp_u8, bm_buf,
                                   8192, rec_d_shallow, rec_iters_shallow), 1);
#endif

    for (int i = 0; i < ri; i++) {
        double gbs = rows[i].bytes_per_elem / rows[i].ns_elem;
        printf("  %-36s %12.3f %12.2f\n",
               rows[i].name, rows[i].ns_elem, gbs);
    }
    printf("\n  Speedup u8 vs u16 (same ISA tier):\n");
    /* Pair adjacent u16 with u8 variants — emit ratios. */
    for (int i = 0; i + 1 < ri; i++) {
        if (strstr(rows[i].name, "u16") && strstr(rows[i+1].name, "u8")) {
            printf("    %-36s  %.2fx (u8 stride next to u16)\n",
                   rows[i+1].name, rows[i].ns_elem / rows[i+1].ns_elem);
        }
    }
    return 0;
}
