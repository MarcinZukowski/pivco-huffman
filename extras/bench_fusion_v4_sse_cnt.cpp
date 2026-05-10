// Block-realistic fusion microbench (SSE4.1).  See bench_fusion_v4_cnt.cpp
// for design rationale.  Same shape, SSE intrinsics for the kernels
// (matches src/pivco_huffman_x86.c).

#include "counters/bench.h"
#include <smmintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

alignas(32) static uint8_t compress_tab[256][32];
alignas(64) static uint8_t compress_popcnt[256];
static void init_compress_table(void) {
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0;
        for (int i = 0; i < 8; i++) if (mask & (1 << i)) {
            compress_tab[mask][out_r * 2]     = (uint8_t)(i * 2);
            compress_tab[mask][out_r * 2 + 1] = (uint8_t)(i * 2 + 1);
            out_r++;
        }
        compress_popcnt[mask] = (uint8_t)out_r;
        for (int j = out_r * 2; j < 16; j++) compress_tab[mask][j] = 0xFF;
        int out_l = 0;
        for (int i = 0; i < 8; i++) if (!(mask & (1 << i))) {
            compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
            compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
            out_l++;
        }
        for (int j = 16 + out_l * 2; j < 32; j++) compress_tab[mask][j] = 0xFF;
    }
}

#ifndef BENCH_K
#define BENCH_K 4
#endif

/* partition_root_8 body (SSE).  src points at 8 packed uint16 base
 * indices, mask is bm[j>>3].  Writes 8 elements split L/R. */
static inline void p_chunk(int j, const uint16_t *psrc, const uint8_t *bm,
                            uint16_t *L, int &nL,
                            uint16_t *R, int &nR) {
    __m128i data = _mm_loadu_si128((const __m128i *)(psrc + j));
    uint8_t mask = bm[j >> 3];
    __m128i shuf_r = _mm_load_si128((const __m128i *)compress_tab[mask]);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(compress_tab[mask] + 16));
    _mm_storeu_si128((__m128i *)(R + nR), _mm_shuffle_epi8(data, shuf_r));
    _mm_storeu_si128((__m128i *)(L + nL), _mm_shuffle_epi8(data, shuf_l));
    int nr = compress_popcnt[mask];
    nR += nr;
    nL += 8 - nr;
}

static inline void s_chunk(uint8_t *symbols, const uint16_t *indices,
                            uint8_t sym) {
    __m128i i0 = _mm_loadu_si128((const __m128i *)indices);
    __m128i i1 = _mm_loadu_si128((const __m128i *)(indices + 8));
#define X(V, K) symbols[_mm_extract_epi16(V, K)] = sym
    X(i0,0); X(i0,1); X(i0,2); X(i0,3); X(i0,4); X(i0,5); X(i0,6); X(i0,7);
    X(i1,0); X(i1,1); X(i1,2); X(i1,3); X(i1,4); X(i1,5); X(i1,6); X(i1,7);
#undef X
}

template<int GAP_OPS>
static inline int filler(int seed) {
    int x = seed;
    #pragma GCC unroll 8
    for (int i = 0; i < GAP_OPS; i++) x = x * 7 + 3;
    return x;
}

static void row(const char *name, const counters::event_aggregate &agg,
                bool have, long total_elems_per_iter) {
    double ns  = agg.fastest_elapsed_ns();
    double cyc = agg.fastest_cycles();
    double ins = agg.fastest_instructions();
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    double ns_per_elem = ns / total_elems_per_iter;
    if (have)
        std::printf("  %-26s %10.1f %9.1f %9.1f %6.2f %6.2f %12.4f\n",
                    name, ns, cyc, ins, ipc, ghz, ns_per_elem);
    else
        std::printf("  %-26s %10.1f                                       %12.4f\n",
                    name, ns, ns_per_elem);
}

int main(int argc, char **argv) {
    init_compress_table();
    const int P_ELEM = 8192;
    int S_ELEM = (argc > 1) ? std::atoi(argv[1]) : 4096;
    if (S_ELEM < 16) S_ELEM = 16;
    S_ELEM &= ~15;

    const int N_P = P_ELEM / 8;
    const int N_S = S_ELEM / 16;
    const int K   = BENCH_K;

    std::vector<uint16_t> psrc(P_ELEM + 32);
    std::vector<uint8_t>  bm(P_ELEM / 8 + 16);
    std::vector<uint16_t> L(P_ELEM + 64);
    std::vector<uint16_t> R(P_ELEM + 64);
    std::vector<uint8_t>  symbols(P_ELEM + 64);
    std::vector<uint16_t> sindices(S_ELEM + 64);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc) v = (uint16_t)rng();
    for (auto &b : bm)   b = (uint8_t)rng();

    int step = P_ELEM / S_ELEM;
    if (step < 1) step = 1;
    for (int i = 0; i < S_ELEM; i++)
        sindices[i] = (uint16_t)(i * step);

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");

    counters::bench_parameter params;
    params.min_repeat = 30;
    params.min_time_ns = 200000000;

    const uint16_t *psrcp = psrc.data();
    const uint8_t  *bmp = bm.data();
    uint16_t *Lp = L.data();
    uint16_t *Rp = R.data();
    uint8_t  *symp = symbols.data();
    const uint16_t *sip = sindices.data();

    std::printf("\nfusion_v4 SSE (block-realistic)\n");
    std::printf("  P_ELEM=%d   S_ELEM=%d   K=%d   total=%d elem/iter\n",
                P_ELEM, S_ELEM, K, P_ELEM + S_ELEM);
    if (have) {
        std::printf("  %-26s %10s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "ns/elem");
    } else {
        std::printf("  %-26s %10s %12s\n", "variant", "ns/iter", "ns/elem");
    }

    {
        auto agg = counters::bench(
            [N_P, N_S, psrcp, bmp, Lp, Rp, symp, sip]() {
            int nL = 0, nR = 0;
            for (int j = 0; j < N_P * 8; j += 8)
                p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
            for (int i = 0; i < N_S; i++)
                s_chunk(symp, sip + i * 16, 0x42);
            ((volatile int *)&nL)[0] = nL + nR + symp[0];
        }, params);
        row("serial_tight (P then S)", agg, have, P_ELEM + S_ELEM);
    }

    auto run_gap = [&](const char *name, auto fn) {
        auto agg = counters::bench(fn, params);
        row(name, agg, have, P_ELEM + S_ELEM);
    };

    run_gap("serial_gap (gap=64 ALU)", [N_P, N_S, psrcp, bmp, Lp, Rp, symp, sip]() {
        int nL = 0, nR = 0;
        for (int j = 0; j < N_P * 8; j += 8)
            p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
        int g = filler<64>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 16, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });
    run_gap("serial_gap (gap=256 ALU)", [N_P, N_S, psrcp, bmp, Lp, Rp, symp, sip]() {
        int nL = 0, nR = 0;
        for (int j = 0; j < N_P * 8; j += 8)
            p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
        int g = filler<256>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 16, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });
    run_gap("serial_gap (gap=1024 ALU)", [N_P, N_S, psrcp, bmp, Lp, Rp, symp, sip]() {
        int nL = 0, nR = 0;
        for (int j = 0; j < N_P * 8; j += 8)
            p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
        int g = filler<1024>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 16, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });

    {
        auto agg = counters::bench(
            [N_P, N_S, K, psrcp, bmp, Lp, Rp, symp, sip]() {
            int nL = 0, nR = 0;
            int j  = 0;
            int N_FUSED_ITERS = N_S < (N_P / K) ? N_S : (N_P / K);
            for (int i = 0; i < N_FUSED_ITERS; i++) {
                s_chunk(symp, sip + i * 16, 0x42);
                #pragma GCC unroll 8
                for (int k = 0; k < K; k++) {
                    p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
                    j += 8;
                }
            }
            for (; j < N_P * 8; j += 8)
                p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
            for (int i = N_FUSED_ITERS; i < N_S; i++)
                s_chunk(symp, sip + i * 16, 0x42);
            ((volatile int *)&nL)[0] = nL + nR + symp[0];
        }, params);
        char nm[40];
        std::snprintf(nm, sizeof(nm), "fused (K=%d)", K);
        row(nm, agg, have, P_ELEM + S_ELEM);
    }

    return 0;
}
