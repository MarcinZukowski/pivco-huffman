// Realistic fusion microbench: sorted-ascending scatter indices +
// variable P:S element ratio.  Designed to match the real decoder's
// access pattern.
//
// Real decoder:
//   - Scatter indices are sorted ascending (partition preserves order).
//     For symbol with probability p, indices spread over [0, BLK) with
//     avg gap 1/p, giving cache-line locality + store-buffer coalescing.
//   - P:S element ratio is ~4:1 in real text (each input elem traverses
//     ~4 partition levels before reaching its leaf scatter).
//
// This bench:
//   - Uses sorted indices uniformly distributed in [0, range)
//   - Sweeps PpS (partition calls per scatter chunk) in {1, 2, 4, 8}
//     giving elem ratios {1:2, 1:1, 2:1, 4:1}
//   - For each, measures PP_SS (sequential) vs PSPS (interleaved)
//
// Tiny working set, all L1-resident.

#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#include "counters/bench.h"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

extern "C" {
extern uint8_t compress_tab[256][32];
extern uint8_t compress_popcnt[256];
void init_compress_table(void);
}

static inline int p8(const uint16_t *src, uint8_t mask,
                      uint16_t *L, uint16_t *R) {
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    vst1q_u8((uint8_t *)R, vqtbl1q_u8(data, shuf_r));
    vst1q_u8((uint8_t *)L, vqtbl1q_u8(data, shuf_l));
    return compress_popcnt[mask];
}

static inline void scatter16(uint8_t *symbols, const uint16_t *indices,
                              uint8_t sym) {
    uint16x8_t i0 = vld1q_u16(indices);
    uint16x8_t i1 = vld1q_u16(indices + 8);
    symbols[vgetq_lane_u16(i0, 0)] = sym; symbols[vgetq_lane_u16(i0, 1)] = sym;
    symbols[vgetq_lane_u16(i0, 2)] = sym; symbols[vgetq_lane_u16(i0, 3)] = sym;
    symbols[vgetq_lane_u16(i0, 4)] = sym; symbols[vgetq_lane_u16(i0, 5)] = sym;
    symbols[vgetq_lane_u16(i0, 6)] = sym; symbols[vgetq_lane_u16(i0, 7)] = sym;
    symbols[vgetq_lane_u16(i1, 0)] = sym; symbols[vgetq_lane_u16(i1, 1)] = sym;
    symbols[vgetq_lane_u16(i1, 2)] = sym; symbols[vgetq_lane_u16(i1, 3)] = sym;
    symbols[vgetq_lane_u16(i1, 4)] = sym; symbols[vgetq_lane_u16(i1, 5)] = sym;
    symbols[vgetq_lane_u16(i1, 6)] = sym; symbols[vgetq_lane_u16(i1, 7)] = sym;
}

/* 4-store scatter chunk: 1 d-reg load + 4 lane extracts + 4 byte stores. */
static inline void scatter4(uint8_t *symbols, const uint16_t *indices,
                             uint8_t sym) {
    uint16x4_t idx = vld1_u16(indices);
    symbols[vget_lane_u16(idx, 0)] = sym;
    symbols[vget_lane_u16(idx, 1)] = sym;
    symbols[vget_lane_u16(idx, 2)] = sym;
    symbols[vget_lane_u16(idx, 3)] = sym;
}

/* 8-store scatter chunk: 1 q-reg load + 8 lane extracts + 8 byte stores. */
static inline void scatter8(uint8_t *symbols, const uint16_t *indices,
                             uint8_t sym) {
    uint16x8_t idx = vld1q_u16(indices);
    symbols[vgetq_lane_u16(idx, 0)] = sym;
    symbols[vgetq_lane_u16(idx, 1)] = sym;
    symbols[vgetq_lane_u16(idx, 2)] = sym;
    symbols[vgetq_lane_u16(idx, 3)] = sym;
    symbols[vgetq_lane_u16(idx, 4)] = sym;
    symbols[vgetq_lane_u16(idx, 5)] = sym;
    symbols[vgetq_lane_u16(idx, 6)] = sym;
    symbols[vgetq_lane_u16(idx, 7)] = sym;
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
        // All P first
        for (int i = 0; i < N * PpS; i++)
            acc += p8(psrcp + (i % 1024)*8, pmp[i % 1024],
                       pLp + i*8, pRp + i*8);
        // Then all S
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
    const int N = 64;          // outer iters per fn-call (S chunks)
    const int N_BUFS = 1024;   // # of P buffer slots (cycle through)

    // P buffers - reused across PpS partition calls per S chunk
    std::vector<uint16_t> psrc(N_BUFS * 8 + 16);
    std::vector<uint8_t>  pmask(N_BUFS + 16);
    // Output buffers must be sized for max PpS = 8
    std::vector<uint16_t> pL(N * 8 * 8 + 16);
    std::vector<uint16_t> pR(N * 8 * 8 + 16);
    std::vector<uint8_t>  symbols(4096);
    std::vector<uint16_t> sindices(N * 16 + 16);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc) v = (uint16_t)rng();
    for (auto &m : pmask) m = (uint8_t)rng();

    // Sorted-ascending sparse indices: spread N*16 values uniformly in
    // [0, symbols.size()).  step = symbols.size()/(N*16) = 4 for N=64.
    // Mimics a leaf scatter where every position in [0, BLK) is a hit
    // with probability 1.0 — densest case.
    int step = (int)symbols.size() / (N * 16);
    if (step < 1) step = 1;
    for (int i = 0; i < N * 16; i++)
        sindices[i] = (uint16_t)(i * step);

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    const uint16_t *psrcp = psrc.data();
    const uint8_t  *pmp   = pmask.data();
    uint16_t       *pLp   = pL.data();
    uint16_t       *pRp   = pR.data();
    uint8_t        *symp  = symbols.data();
    const uint16_t *sip   = sindices.data();

    std::printf("\nfusion_v3 (sorted scatter indices, varying P:S)\n");
    std::printf("N=%d outer iters; sindices step=%d (range [0..%d))\n",
                N, step, (int)symbols.size());
    if (have) {
        std::printf("  %-22s %9s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "stores/cyc");
    }

    // P_only baseline (per-iter cost of P at PpS=1)
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [N, psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(psrcp + (i % 1024)*8, pmp[i % 1024],
                           pLp + i*8, pRp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        std::printf("\n--- baselines ---\n");
        row("P_only (1 P/iter)", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([N, symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter16(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("S_only (1 S/iter)", agg, have, 16 * N);
    }

    auto run_pair = [&](int PpS, auto pp_ss, auto psps) {
        std::printf("\n--- PpS=%d (%d P calls per 1 S chunk; elem ratio %d:%d) ---\n",
                    PpS, PpS, PpS*8, 16);
        int stores = (2 * PpS + 16) * N;
        char nm[32];
        std::snprintf(nm, sizeof(nm), "PP_SS PpS=%d", PpS);
        row(nm, pp_ss(params, N, psrcp, pmp, pLp, pRp, symp, sip),
            have, stores);
        std::snprintf(nm, sizeof(nm), "PSPS PpS=%d", PpS);
        row(nm, psps(params, N, psrcp, pmp, pLp, pRp, symp, sip),
            have, stores);
    };

    run_pair(1, run_PP_SS<1>, run_PSPS<1>);
    run_pair(2, run_PP_SS<2>, run_PSPS<2>);
    run_pair(4, run_PP_SS<4>, run_PSPS<4>);
    run_pair(8, run_PP_SS<8>, run_PSPS<8>);

    /* Fine-grained interleaving: split scatter16 into chunks of 4 (or 8)
     * and intersperse a P call between chunks.  Same total work as
     * PSPS PpS=4 but smaller batches per "phase". */
    std::printf("\n--- fine-grained interleaving (PpS=4, scatter split) ---\n");

    // P, scatter4, P, scatter4, P, scatter4, P, scatter4   (4P + 4*4 = 4P + 16 stores)
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [N, psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                int j = i*4;
                acc += p8(psrcp + (j%1024)*8, pmp[j%1024], pLp + j*8, pRp + j*8);
                scatter4(symp, sip + i*16 + 0, 0x42);
                acc += p8(psrcp + ((j+1)%1024)*8, pmp[(j+1)%1024], pLp + (j+1)*8, pRp + (j+1)*8);
                scatter4(symp, sip + i*16 + 4, 0x42);
                acc += p8(psrcp + ((j+2)%1024)*8, pmp[(j+2)%1024], pLp + (j+2)*8, pRp + (j+2)*8);
                scatter4(symp, sip + i*16 + 8, 0x42);
                acc += p8(psrcp + ((j+3)%1024)*8, pmp[(j+3)%1024], pLp + (j+3)*8, pRp + (j+3)*8);
                scatter4(symp, sip + i*16 + 12, 0x42);
            }
            sink = acc + symp[0];
        }, params);
        (void)sink;
        row("P,s4 x4 (4P + 4*scatter4)", agg, have, (8 + 16) * N);
    }

    // P, P, scatter8, P, P, scatter8   (4P + 2*8 = 4P + 16 stores)
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [N, psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                int j = i*4;
                acc += p8(psrcp + (j%1024)*8, pmp[j%1024], pLp + j*8, pRp + j*8);
                acc += p8(psrcp + ((j+1)%1024)*8, pmp[(j+1)%1024], pLp + (j+1)*8, pRp + (j+1)*8);
                scatter8(symp, sip + i*16 + 0, 0x42);
                acc += p8(psrcp + ((j+2)%1024)*8, pmp[(j+2)%1024], pLp + (j+2)*8, pRp + (j+2)*8);
                acc += p8(psrcp + ((j+3)%1024)*8, pmp[(j+3)%1024], pLp + (j+3)*8, pRp + (j+3)*8);
                scatter8(symp, sip + i*16 + 8, 0x42);
            }
            sink = acc + symp[0];
        }, params);
        (void)sink;
        row("PP,s8 x2 (4P + 2*scatter8)", agg, have, (8 + 16) * N);
    }

    return 0;
}
