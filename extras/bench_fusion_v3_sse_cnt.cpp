// SSE4.1 version of bench_fusion_v3_cnt: realistic fusion microbench
// with sorted-ascending scatter indices + variable P:S element ratio.
//
// All buffers L1-resident. Uses pshufb-based partition_8 (matches
// pivco_huffman_x86.c).

#include "counters/bench.h"
#include <smmintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

alignas(32) static uint8_t compress_tab[256][32];
alignas(64) static uint8_t compress_popcnt[256];
static void init_compress_table(void) {
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
        for (int j = out_r * 2; j < 16; j++) compress_tab[mask][j] = 0xFF;
        int out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (!(mask & (1 << i))) {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = 16 + out_l * 2; j < 32; j++) compress_tab[mask][j] = 0xFF;
    }
}

static inline int p8(const uint16_t *src, uint8_t mask,
                      uint16_t *L, uint16_t *R) {
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_r = _mm_load_si128((const __m128i *)compress_tab[mask]);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(compress_tab[mask] + 16));
    _mm_storeu_si128((__m128i *)R, _mm_shuffle_epi8(data, shuf_r));
    _mm_storeu_si128((__m128i *)L, _mm_shuffle_epi8(data, shuf_l));
    return compress_popcnt[mask];
}

static inline void scatter16(uint8_t *symbols, const uint16_t *indices,
                              uint8_t sym) {
    __m128i i0 = _mm_loadu_si128((const __m128i *)indices);
    __m128i i1 = _mm_loadu_si128((const __m128i *)(indices + 8));
#define X(V, K) symbols[_mm_extract_epi16(V, K)] = sym
    X(i0,0); X(i0,1); X(i0,2); X(i0,3); X(i0,4); X(i0,5); X(i0,6); X(i0,7);
    X(i1,0); X(i1,1); X(i1,2); X(i1,3); X(i1,4); X(i1,5); X(i1,6); X(i1,7);
#undef X
}

template<int PpS>
static counters::event_aggregate run_PP_SS(
    const counters::bench_parameter &p,
    int N, const uint16_t *psrcp, const uint8_t *pmp,
    uint16_t *pLp, uint16_t *pRp,
    uint8_t *symp, const uint16_t *sip)
{
    volatile int sink = 0;
    auto agg = counters::bench(
        [N, psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
        int acc = 0;
        for (int i = 0; i < N * PpS; i++)
            acc += p8(psrcp + (i % 1024)*8, pmp[i % 1024],
                       pLp + i*8, pRp + i*8);
        for (int i = 0; i < N; i++)
            scatter16(symp, sip + i*16, 0x42);
        sink = acc + symp[0];
    }, p);
    (void)sink;
    return agg;
}

template<int PpS>
static counters::event_aggregate run_PSPS(
    const counters::bench_parameter &p,
    int N, const uint16_t *psrcp, const uint8_t *pmp,
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
                acc += p8(psrcp + (j % 1024)*8, pmp[j % 1024],
                           pLp + j*8, pRp + j*8);
            }
            scatter16(symp, sip + i*16, 0x42);
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
    init_compress_table();
    const int N = 64;
    const int N_BUFS = 1024;

    std::vector<uint16_t> psrc(N_BUFS * 8 + 16);
    std::vector<uint8_t>  pmask(N_BUFS + 16);
    std::vector<uint16_t> pL(N * 8 * 8 + 16);
    std::vector<uint16_t> pR(N * 8 * 8 + 16);
    std::vector<uint8_t>  symbols(4096);
    std::vector<uint16_t> sindices(N * 16 + 16);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc) v = (uint16_t)rng();
    for (auto &m : pmask) m = (uint8_t)rng();
    int step = (int)symbols.size() / (N * 16);
    if (step < 1) step = 1;
    for (int i = 0; i < N * 16; i++) sindices[i] = (uint16_t)(i * step);

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo or set perf_event_paranoid=0]\n");

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    const uint16_t *psrcp = psrc.data();
    const uint8_t  *pmp   = pmask.data();
    uint16_t       *pLp   = pL.data();
    uint16_t       *pRp   = pR.data();
    uint8_t        *symp  = symbols.data();
    const uint16_t *sip   = sindices.data();

    std::printf("\nfusion_v3 SSE (sorted scatter indices, varying P:S)\n");
    std::printf("N=%d outer iters; sindices step=%d\n", N, step);
    if (have) std::printf("  %-22s %9s %9s %9s %6s %6s %12s\n",
                          "variant", "ns/iter", "cyc/iter", "ins/iter",
                          "IPC", "GHz", "stores/cyc");

    {
        volatile int sink = 0;
        auto agg = counters::bench([N, psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(psrcp + (i % 1024)*8, pmp[i % 1024],
                           pLp + i*8, pRp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        std::printf("\n--- baselines ---\n");
        row("P_only", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([N, symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter16(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("S_only", agg, have, 16 * N);
    }

    auto run_pair = [&](int PpS, auto pp_ss, auto psps) {
        std::printf("\n--- PpS=%d (%d P calls per S; elem %d:%d) ---\n",
                    PpS, PpS, PpS*8, 16);
        int stores = (2 * PpS + 16) * N;
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
