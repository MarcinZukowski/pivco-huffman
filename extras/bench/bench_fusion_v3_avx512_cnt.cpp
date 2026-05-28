// AVX-512 version of bench_fusion_v3_cnt.  Partition operates on 32
// elements at a time (vpcompressw); scatter chunks size 32 to match.

#include "counters/bench.h"
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

static inline int p32(const uint16_t *src, uint32_t mask,
                      uint16_t *L, uint16_t *R) {
    __m512i data = _mm512_loadu_si512((const __m512i *)src);
    __m512i right = _mm512_maskz_compress_epi16((__mmask32)mask, data);
    int n_right = _mm_popcnt_u32(mask);
    _mm512_storeu_si512((__m512i *)R, right);
    __m512i left = _mm512_maskz_compress_epi16((__mmask32)~mask, data);
    _mm512_storeu_si512((__m512i *)L, left);
    return n_right;
}

static inline void scatter32(uint8_t *symbols, const uint16_t *indices,
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

template<int PpS>
static counters::event_aggregate run_PP_SS(
    const counters::bench_parameter &p,
    int N, const uint16_t *psrcp, const uint32_t *pmp,
    uint16_t *pLp, uint16_t *pRp,
    uint8_t *symp, const uint16_t *sip)
{
    volatile int sink = 0;
    auto agg = counters::bench(
        [N, psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
        int acc = 0;
        for (int i = 0; i < N * PpS; i++)
            acc += p32(psrcp + (i % 256)*32, pmp[i % 256],
                        pLp + i*32, pRp + i*32);
        for (int i = 0; i < N; i++)
            scatter32(symp, sip + i*32, 0x42);
        sink = acc + symp[0];
    }, p);
    (void)sink;
    return agg;
}

template<int PpS>
static counters::event_aggregate run_PSPS(
    const counters::bench_parameter &p,
    int N, const uint16_t *psrcp, const uint32_t *pmp,
    uint16_t *pLp, uint16_t *pRp,
    uint8_t *symp, const uint16_t *sip)
{
    volatile int sink = 0;
    auto agg = counters::bench(
        [N, psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
        int acc = 0;
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < PpS; k++) {
                int j = i*PpS + k;
                acc += p32(psrcp + (j % 256)*32, pmp[j % 256],
                            pLp + j*32, pRp + j*32);
            }
            scatter32(symp, sip + i*32, 0x42);
        }
        sink = acc + symp[0];
    }, p);
    (void)sink;
    return agg;
}

static void row(const char *name, const counters::event_aggregate &agg,
                bool have, int total_stores) {
    double ns  = agg.fastest_elapsed_ns();
    double cyc = agg.fastest_cycles();
    double ins = agg.fastest_instructions();
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    double sps = cyc > 0 ? total_stores / cyc : 0;
    if (have)
        std::printf("  %-22s %9.1f %9.1f %9.1f %6.2f %6.2f %12.3f\n",
                    name, ns, cyc, ins, ipc, ghz, sps);
    else
        std::printf("  %-22s %9.1f\n", name, ns);
}

int main() {
    /* Partition_32 takes 32 elems; scatter32 stores 32 bytes.  Element
     * ratio interpretation: PpS=1 -> 32 P-elems vs 32 S-elems (1:1).
     * To get the 4:1 real-decoder ratio we need PpS=4. */
    const int N = 32;          // S chunks per fn-call
    const int N_BUFS = 256;

    std::vector<uint16_t> psrc(N_BUFS * 32 + 32);
    std::vector<uint32_t> pmask(N_BUFS + 16);
    std::vector<uint16_t> pL(N * 32 * 8 + 32);
    std::vector<uint16_t> pR(N * 32 * 8 + 32);
    std::vector<uint8_t>  symbols(4096);
    std::vector<uint16_t> sindices(N * 32 + 32);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc) v = (uint16_t)rng();
    for (auto &m : pmask) m = rng();
    int step = (int)symbols.size() / (N * 32);
    if (step < 1) step = 1;
    for (int i = 0; i < N * 32; i++) sindices[i] = (uint16_t)(i * step);

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo or set perf_event_paranoid=0]\n");

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    const uint16_t *psrcp = psrc.data();
    const uint32_t *pmp   = pmask.data();
    uint16_t       *pLp   = pL.data();
    uint16_t       *pRp   = pR.data();
    uint8_t        *symp  = symbols.data();
    const uint16_t *sip   = sindices.data();

    std::printf("\nfusion_v3 AVX-512 (sorted scatter indices, varying P:S)\n");
    std::printf("N=%d outer iters; sindices step=%d\n", N, step);
    if (have) std::printf("  %-22s %9s %9s %9s %6s %6s %12s\n",
                          "variant", "ns/iter", "cyc/iter", "ins/iter",
                          "IPC", "GHz", "stores/cyc");

    {
        volatile int sink = 0;
        auto agg = counters::bench([N, psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p32(psrcp + (i % 256)*32, pmp[i % 256],
                            pLp + i*32, pRp + i*32);
            sink = acc;
        }, params);
        (void)sink;
        std::printf("\n--- baselines ---\n");
        row("P_only (32-elem)", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([N, symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter32(symp, sip + i*32, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("S_only (32 stores)", agg, have, 32 * N);
    }

    auto run_pair = [&](int PpS, auto pp_ss, auto psps) {
        std::printf("\n--- PpS=%d (elem %d:%d) ---\n", PpS, PpS*32, 32);
        int stores = (2 * PpS + 32) * N;
        char nm[32];
        std::snprintf(nm, sizeof(nm), "PP_SS PpS=%d", PpS);
        row(nm, pp_ss(params, N, psrcp, pmp, pLp, pRp, symp, sip), have, stores);
        std::snprintf(nm, sizeof(nm), "PSPS PpS=%d", PpS);
        row(nm, psps(params, N, psrcp, pmp, pLp, pRp, symp, sip), have, stores);
    };

    run_pair(1, run_PP_SS<1>, run_PSPS<1>);
    run_pair(2, run_PP_SS<2>, run_PSPS<2>);
    run_pair(4, run_PP_SS<4>, run_PSPS<4>);
    run_pair(8, run_PP_SS<8>, run_PSPS<8>);
    return 0;
}
