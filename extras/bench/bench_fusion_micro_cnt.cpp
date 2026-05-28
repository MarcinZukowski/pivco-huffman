// Microbench: can we get partition and scatter to run in parallel?
//
// partition_8 is critical-path bound (0.27 stores/cyc, lots of slack).
// scatter_sym is store-port bound (1.86 stores/cyc, near peak).
// Different bottlenecks -> if we run them interleaved in the same inner
// loop, OOO should overlap them and the fused cost approaches max(P, S)
// rather than P + S.
//
// Variants per fused-iter (fused-iter = 1 partition_8 call + 16 scatter
// stores, the atomic batch sizes of each kernel):
//
//   P_only       just partition_8 (baseline)
//   S_only       just 16 scatter stores (baseline)
//   serial       inline P; inline S (same inner loop, P first)
//   interleaved  manually interleaved so writes/reads from each are
//                woven instead of all-P-then-all-S
//
// Reports per "fused-iter" (1 P + 1 S worth of work).  If serial cost
// is close to max(P_only, S_only) the OOO is already overlapping for
// free, and explicit fusion buys nothing.  If serial cost > max then
// fusion has headroom.

#include "counters/bench.h"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

extern "C" {
extern uint8_t compress_tab[256][32];
extern uint8_t compress_popcnt[256];
void init_compress_table(void);
}

// partition_8 — same as in pivco_huffman_neon.c
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

// One 16-store scatter chunk (the inner body of scatter_sym).
static inline void scatter16(uint8_t *symbols, const uint16_t *indices,
                              uint8_t sym) {
    uint16x8_t i0 = vld1q_u16(indices);
    uint16x8_t i1 = vld1q_u16(indices + 8);
    symbols[vgetq_lane_u16(i0, 0)] = sym;
    symbols[vgetq_lane_u16(i0, 1)] = sym;
    symbols[vgetq_lane_u16(i0, 2)] = sym;
    symbols[vgetq_lane_u16(i0, 3)] = sym;
    symbols[vgetq_lane_u16(i0, 4)] = sym;
    symbols[vgetq_lane_u16(i0, 5)] = sym;
    symbols[vgetq_lane_u16(i0, 6)] = sym;
    symbols[vgetq_lane_u16(i0, 7)] = sym;
    symbols[vgetq_lane_u16(i1, 0)] = sym;
    symbols[vgetq_lane_u16(i1, 1)] = sym;
    symbols[vgetq_lane_u16(i1, 2)] = sym;
    symbols[vgetq_lane_u16(i1, 3)] = sym;
    symbols[vgetq_lane_u16(i1, 4)] = sym;
    symbols[vgetq_lane_u16(i1, 5)] = sym;
    symbols[vgetq_lane_u16(i1, 6)] = sym;
    symbols[vgetq_lane_u16(i1, 7)] = sym;
}

static void print_header(bool have, int n_iters) {
    std::printf("\nfusion microbench  (n_iters=%d fused-iters per call)\n", n_iters);
    if (have) {
        std::printf("%-22s %9s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "stores/cyc");
    } else {
        std::printf("%-22s %9s\n", "variant", "ns/iter");
    }
}

static void print_row(const char *name,
                      const counters::event_aggregate &agg,
                      bool have, int n_iters, int stores_per_iter) {
    double ns  = agg.fastest_elapsed_ns() / (double)n_iters;
    double cyc = agg.fastest_cycles() / (double)n_iters;
    double ins = agg.fastest_instructions() / (double)n_iters;
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    double sps = cyc > 0 ? stores_per_iter / cyc : 0;
    if (have) {
        std::printf("%-22s %9.3f %9.3f %9.3f %6.2f %6.2f %12.3f\n",
                    name, ns, cyc, ins, ipc, ghz, sps);
    } else {
        std::printf("%-22s %9.3f\n", name, ns);
    }
}

int main() {
    init_compress_table();
    const int N = 1024; // fused-iters per bench fn-call

    // partition state (shared by all variants; output buffers may not
    // line up across iterations but that's fine for timing)
    std::vector<uint16_t> psrc(N * 8 + 16);
    std::vector<uint8_t>  pmask(N + 16);
    std::vector<uint16_t> pL(N * 8 + 16);
    std::vector<uint16_t> pR(N * 8 + 16);

    // scatter state — symbol buffer + scattered indices
    std::vector<uint8_t>  symbols(64 * 1024);
    std::vector<uint16_t> sindices(N * 16 + 16);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc) v = (uint16_t)rng();
    for (auto &m : pmask) m = (uint8_t)rng();
    // scattered shuffled indices for scatter
    for (int i = 0; i < N * 16; i++) sindices[i] = (uint16_t)(rng() % symbols.size());

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");
    print_header(have, N);

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    const uint16_t *psrcp = psrc.data();
    const uint8_t  *pmp   = pmask.data();
    uint16_t       *pLp   = pL.data();
    uint16_t       *pRp   = pR.data();
    uint8_t        *symp  = symbols.data();
    const uint16_t *sip   = sindices.data();

    // Baseline 1: just partition (2 stores per iter)
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        print_row("P_only (2 st/iter)", agg, have, N, 2);
    }

    // Baseline 2: just scatter (16 stores per iter)
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++)
                scatter16(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("S_only (16 st/iter)", agg, have, N, 16);
    }

    // Serial: P first, then S, in same inner loop body (18 stores/iter)
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
                scatter16(symp, sip + i*16, 0x42);
            }
            sink = acc + symp[0];
        }, params);
        (void)sink;
        print_row("serial P;S (18 st/iter)", agg, have, N, 18);
    }

    // Interleaved: spread the 16 scatter stores between/around the 2
    // partition stores by manually splitting scatter16 into halves.
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                // First half of scatter, then partition, then second half.
                // Goal: more cycles between the two scatter stores so
                // partition's slack overlaps better.
                uint16x8_t si0 = vld1q_u16(sip + i*16);
                uint16x8_t si1 = vld1q_u16(sip + i*16 + 8);
                symp[vgetq_lane_u16(si0, 0)] = 0x42;
                symp[vgetq_lane_u16(si0, 1)] = 0x42;
                symp[vgetq_lane_u16(si0, 2)] = 0x42;
                symp[vgetq_lane_u16(si0, 3)] = 0x42;
                symp[vgetq_lane_u16(si0, 4)] = 0x42;
                symp[vgetq_lane_u16(si0, 5)] = 0x42;
                symp[vgetq_lane_u16(si0, 6)] = 0x42;
                symp[vgetq_lane_u16(si0, 7)] = 0x42;
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
                symp[vgetq_lane_u16(si1, 0)] = 0x42;
                symp[vgetq_lane_u16(si1, 1)] = 0x42;
                symp[vgetq_lane_u16(si1, 2)] = 0x42;
                symp[vgetq_lane_u16(si1, 3)] = 0x42;
                symp[vgetq_lane_u16(si1, 4)] = 0x42;
                symp[vgetq_lane_u16(si1, 5)] = 0x42;
                symp[vgetq_lane_u16(si1, 6)] = 0x42;
                symp[vgetq_lane_u16(si1, 7)] = 0x42;
            }
            sink = acc + symp[0];
        }, params);
        (void)sink;
        print_row("interleaved (18 st/iter)", agg, have, N, 18);
    }
    return 0;
}
