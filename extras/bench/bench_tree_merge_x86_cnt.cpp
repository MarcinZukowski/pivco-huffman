// Microbench for tree_merge on x86: SSE4.1 (8-byte chunks via pshufb)
// and AVX-512 VBMI2 (64-byte chunks via vpexpandb).  See
// bench_tree_merge_neon_cnt.cpp for the primitive spec.
//
// Variants:
//   scalar          : reference impl
//   sse             : 8-byte chunks, pshufb on combined left+right
//   sse_x2          : 16-byte chunks (two 8-byte unrolled)
//   avx512          : 64-byte chunks via 2x vpexpandb + OR (VBMI2 only)
//   sse_broadcast_L : constant-left-leaf optimisation, no left buffer

#include "counters/bench.h"
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

alignas(32) static uint8_t expand_tab[256][8];
alignas(64) static uint8_t expand_popcnt[256];

static void init_expand_table(void) {
    for (int m = 0; m < 256; m++) {
        int n_zeros = 0, n_ones = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) {
                expand_tab[m][k] = (uint8_t)(8 + n_ones);
                n_ones++;
            } else {
                expand_tab[m][k] = (uint8_t)n_zeros;
                n_zeros++;
            }
        }
        expand_popcnt[m] = (uint8_t)n_ones;
    }
}

/* ---------- scalar reference ---------- */
static inline void merge_scalar(const uint8_t *bm, int n,
                                 const uint8_t *left,
                                 const uint8_t *right,
                                 uint8_t *out) {
    int lc = 0, rc = 0;
    for (int j = 0; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}

/* ---------- SSE: 8-byte chunks via pshufb ----------
 * Per chunk: load 8 left + 8 right, unpacklo_epi64 into 16-byte reg,
 * pshufb with the 8-byte expand_tab[mask] pattern. */
static inline void merge_sse(const uint8_t *bm, int n,
                              const uint8_t *left,
                              const uint8_t *right,
                              uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(L, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        int nr = expand_popcnt[m];
        rc += nr;
        lc += (8 - nr);
    }
    for (; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}

/* ---------- SSE 16-byte unrolled ---------- */
static inline void merge_sse_x2(const uint8_t *bm, int n,
                                 const uint8_t *left,
                                 const uint8_t *right,
                                 uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint8_t m0 = bm[j >> 3];
        uint8_t m1 = bm[(j >> 3) + 1];

        __m128i L0 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(L0, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        int nr0 = expand_popcnt[m0];
        rc += nr0; lc += (8 - nr0);

        __m128i L1 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(L1, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        int nr1 = expand_popcnt[m1];
        rc += nr1; lc += (8 - nr1);
    }
    for (; j + 8 <= n; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(L, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        int nr = expand_popcnt[m];
        rc += nr; lc += (8 - nr);
    }
    for (; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}

#ifdef __AVX512VBMI2__
/* ---------- AVX-512 VBMI2: 64-byte chunks via vpexpandb ----------
 * Per chunk:
 *   left_part  = vpexpandb_epi8(~mask_64, left + lc)
 *     -> for each 0-bit in mask, place next left byte at that position
 *   right_part = vpexpandb_epi8( mask_64, right + rc)
 *     -> for each 1-bit, place next right byte at that position
 *   output = left_part | right_part
 * Throughput: 64 output bytes per ~4 cycles (vpexpandb is latency 3-5,
 * throughput 0.5/cycle on Ice Lake+). */
static inline void merge_avx512(const uint8_t *bm, int n,
                                 const uint8_t *left,
                                 const uint8_t *right,
                                 uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 64 <= n; j += 64) {
        uint64_t mask;
        std::memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __mmask64 nm = ~m;

        __m512i L = _mm512_maskz_expandloadu_epi8(nm, left + lc);
        __m512i R = _mm512_maskz_expandloadu_epi8(m,  right + rc);
        __m512i o = _mm512_or_si512(L, R);
        _mm512_storeu_si512((__m512i *)(out + j), o);

        int nr = __builtin_popcountll(mask);
        rc += nr;
        lc += (64 - nr);
    }
    /* Tail: SSE-fashion 8-byte chunks */
    for (; j + 8 <= n; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(L, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        int nr = expand_popcnt[m];
        rc += nr; lc += (8 - nr);
    }
    for (; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}
#endif

/* ---------- SSE broadcast-left ---------- */
static inline void merge_sse_broadcast_left(const uint8_t *bm, int n,
                                             uint8_t left_sym,
                                             const uint8_t *right,
                                             uint8_t *out) {
    int rc = 0;
    int j = 0;
    /* Broadcast left_sym into 8 low bytes; we'll combine with right. */
    __m128i Lbcast8 = _mm_set1_epi8((char)left_sym);
    for (; j + 8 <= n; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(Lbcast8, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        rc += expand_popcnt[m];
    }
    for (; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left_sym;
    }
}

/* ---------- harness ---------- */
static void row(const char *name, const counters::event_aggregate &agg,
                bool have, long elems_per_iter) {
    double ns  = agg.fastest_elapsed_ns();
    double cyc = agg.fastest_cycles();
    double ins = agg.fastest_instructions();
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    double ns_per_elem = ns / elems_per_iter;
    if (have)
        std::printf("  %-26s %10.1f %9.1f %9.1f %6.2f %6.2f %12.4f\n",
                    name, ns, cyc, ins, ipc, ghz, ns_per_elem);
    else
        std::printf("  %-26s %10.1f                                       %12.4f\n",
                    name, ns, ns_per_elem);
}

int main(int argc, char **argv) {
    init_expand_table();
    int N = (argc > 1) ? std::atoi(argv[1]) : 4096;

    std::vector<uint8_t> bm((N + 7) / 8 + 64);
    std::vector<uint8_t> left(N + 64);
    std::vector<uint8_t> right(N + 64);
    std::vector<uint8_t> output(N + 64);
    std::vector<uint8_t> ref(N + 64);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &b : bm)    b = (uint8_t)rng();
    for (auto &l : left)  l = (uint8_t)rng();
    for (auto &r : right) r = (uint8_t)rng();

    merge_scalar(bm.data(), N, left.data(), right.data(), ref.data());

    merge_sse(bm.data(), N, left.data(), right.data(), output.data());
    if (std::memcmp(output.data(), ref.data(), N) != 0) {
        std::fprintf(stderr, "merge_sse: MISMATCH\n"); return 1;
    }
    merge_sse_x2(bm.data(), N, left.data(), right.data(), output.data());
    if (std::memcmp(output.data(), ref.data(), N) != 0) {
        std::fprintf(stderr, "merge_sse_x2: MISMATCH\n"); return 2;
    }
#ifdef __AVX512VBMI2__
    merge_avx512(bm.data(), N, left.data(), right.data(), output.data());
    if (std::memcmp(output.data(), ref.data(), N) != 0) {
        std::fprintf(stderr, "merge_avx512: MISMATCH\n"); return 3;
    }
#endif
    {
        std::vector<uint8_t> virt_left(N + 64, 0xAA);
        std::vector<uint8_t> ref_bcast(N + 64);
        merge_scalar(bm.data(), N, virt_left.data(), right.data(), ref_bcast.data());
        merge_sse_broadcast_left(bm.data(), N, 0xAA, right.data(), output.data());
        if (std::memcmp(output.data(), ref_bcast.data(), N) != 0) {
            std::fprintf(stderr, "merge_sse_broadcast_left: MISMATCH\n"); return 4;
        }
    }

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");
    counters::bench_parameter params;
    params.min_repeat = 30;
    params.min_time_ns = 200000000;

    const uint8_t *bmp = bm.data();
    const uint8_t *Lp  = left.data();
    const uint8_t *Rp  = right.data();
    uint8_t       *Op  = output.data();

    std::printf("\ntree_merge x86 microbench (N=%d output bytes per iter)\n", N);
    if (have) {
        std::printf("\n  %-26s %10s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "ns/elem");
    } else {
        std::printf("\n  %-26s %10s %12s\n", "variant", "ns/iter", "ns/elem");
    }

    {
        auto agg = counters::bench(
            [bmp, N, Lp, Rp, Op]() {
            merge_scalar(bmp, N, Lp, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_scalar", agg, have, N);
    }
    {
        auto agg = counters::bench(
            [bmp, N, Lp, Rp, Op]() {
            merge_sse(bmp, N, Lp, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_sse (8B chunk)", agg, have, N);
    }
    {
        auto agg = counters::bench(
            [bmp, N, Lp, Rp, Op]() {
            merge_sse_x2(bmp, N, Lp, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_sse_x2 (16B chunk)", agg, have, N);
    }
#ifdef __AVX512VBMI2__
    {
        auto agg = counters::bench(
            [bmp, N, Lp, Rp, Op]() {
            merge_avx512(bmp, N, Lp, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_avx512 (64B vpexpandb)", agg, have, N);
    }
#endif
    {
        auto agg = counters::bench(
            [bmp, N, Rp, Op]() {
            merge_sse_broadcast_left(bmp, N, 0xAA, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_sse_broadcast_left", agg, have, N);
    }

    return 0;
}
