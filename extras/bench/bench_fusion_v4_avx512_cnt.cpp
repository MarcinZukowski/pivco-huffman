// Block-realistic fusion microbench (AVX-512 VBMI2).  See bench_fusion_v4_cnt.cpp
// for design rationale.  Uses vpcompressw for 32-element partition chunks
// and 32-element scatter chunks (matches src/pivco_huffman_avx512.c).

#include "counters/bench.h"
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#ifndef BENCH_K
#define BENCH_K 4
#endif

/* partition_root_32 body: read 32 packed u16 base indices, compress
 * via mask into right (1-bits) and left (0-bits).  Note partition
 * chunk size = 32 here vs 8 on NEON/SSE; mask is 32-bit. */
static inline void p_chunk(int j, const uint16_t *psrc, const uint32_t *bm,
                            uint16_t *L, int &nL,
                            uint16_t *R, int &nR) {
    int word = j >> 5;
    __m512i data = _mm512_loadu_si512((const __m512i *)(psrc + j));
    __mmask32 m = (__mmask32)bm[word];
    __m512i right = _mm512_maskz_compress_epi16(m, data);
    int n_right = _mm_popcnt_u32(m);
    _mm512_storeu_si512((__m512i *)(R + nR), right);
    __m512i left = _mm512_maskz_compress_epi16((__mmask32)~m, data);
    _mm512_storeu_si512((__m512i *)(L + nL), left);
    nR += n_right;
    nL += 32 - n_right;
}

/* scatter32 chunk: 32 byte stores via lane-extracted indices. */
static inline void s_chunk(uint8_t *symbols, const uint16_t *indices,
                            uint8_t sym) {
    __m128i i0 = _mm_loadu_si128((const __m128i *)indices);
    __m128i i1 = _mm_loadu_si128((const __m128i *)(indices + 8));
    __m128i i2 = _mm_loadu_si128((const __m128i *)(indices + 16));
    __m128i i3 = _mm_loadu_si128((const __m128i *)(indices + 24));
#define X(V, K) symbols[_mm_extract_epi16(V, K)] = sym
    X(i0,0); X(i0,1); X(i0,2); X(i0,3); X(i0,4); X(i0,5); X(i0,6); X(i0,7);
    X(i1,0); X(i1,1); X(i1,2); X(i1,3); X(i1,4); X(i1,5); X(i1,6); X(i1,7);
    X(i2,0); X(i2,1); X(i2,2); X(i2,3); X(i2,4); X(i2,5); X(i2,6); X(i2,7);
    X(i3,0); X(i3,1); X(i3,2); X(i3,3); X(i3,4); X(i3,5); X(i3,6); X(i3,7);
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
    /* For AVX-512: scatter chunk = 32 elements, partition chunk = 32
     * elements.  Adjust S_ELEM to multiple of 32 and partition stride
     * accordingly. */
    const int P_ELEM = 8192;
    int S_ELEM = (argc > 1) ? std::atoi(argv[1]) : 4096;
    if (S_ELEM < 32) S_ELEM = 32;
    S_ELEM &= ~31;

    const int N_P = P_ELEM / 32;
    const int N_S = S_ELEM / 32;
    const int K   = BENCH_K;

    std::vector<uint16_t> psrc(P_ELEM + 64);
    std::vector<uint32_t> bm(P_ELEM / 32 + 16);
    std::vector<uint16_t> L(P_ELEM + 128);
    std::vector<uint16_t> R(P_ELEM + 128);
    std::vector<uint8_t>  symbols(P_ELEM + 128);
    std::vector<uint16_t> sindices(S_ELEM + 64);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc) v = (uint16_t)rng();
    for (auto &m : bm)   m = (uint32_t)rng();

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
    const uint32_t *bmp = bm.data();
    uint16_t *Lp = L.data();
    uint16_t *Rp = R.data();
    uint8_t  *symp = symbols.data();
    const uint16_t *sip = sindices.data();

    std::printf("\nfusion_v4 AVX-512 (block-realistic)\n");
    std::printf("  P_ELEM=%d (32-wide)   S_ELEM=%d (32-wide)   K=%d   total=%d\n",
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
            for (int j = 0; j < N_P * 32; j += 32)
                p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
            for (int i = 0; i < N_S; i++)
                s_chunk(symp, sip + i * 32, 0x42);
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
        for (int j = 0; j < N_P * 32; j += 32)
            p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
        int g = filler<64>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 32, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });
    run_gap("serial_gap (gap=256 ALU)", [N_P, N_S, psrcp, bmp, Lp, Rp, symp, sip]() {
        int nL = 0, nR = 0;
        for (int j = 0; j < N_P * 32; j += 32)
            p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
        int g = filler<256>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 32, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });
    run_gap("serial_gap (gap=1024 ALU)", [N_P, N_S, psrcp, bmp, Lp, Rp, symp, sip]() {
        int nL = 0, nR = 0;
        for (int j = 0; j < N_P * 32; j += 32)
            p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
        int g = filler<1024>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 32, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });

    {
        auto agg = counters::bench(
            [N_P, N_S, K, psrcp, bmp, Lp, Rp, symp, sip]() {
            int nL = 0, nR = 0;
            int j  = 0;
            int N_FUSED_ITERS = N_S < (N_P / K) ? N_S : (N_P / K);
            for (int i = 0; i < N_FUSED_ITERS; i++) {
                s_chunk(symp, sip + i * 32, 0x42);
                #pragma GCC unroll 8
                for (int k = 0; k < K; k++) {
                    p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
                    j += 32;
                }
            }
            for (; j < N_P * 32; j += 32)
                p_chunk(j, psrcp, bmp, Lp, nL, Rp, nR);
            for (int i = N_FUSED_ITERS; i < N_S; i++)
                s_chunk(symp, sip + i * 32, 0x42);
            ((volatile int *)&nL)[0] = nL + nR + symp[0];
        }, params);
        char nm[40];
        std::snprintf(nm, sizeof(nm), "fused (K=%d)", K);
        row(nm, agg, have, P_ELEM + S_ELEM);
    }

    return 0;
}
