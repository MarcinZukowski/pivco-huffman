// Better fusion microbench: tests whether P and S can pipeline their
// orthogonal bottlenecks by comparing layouts that differ ONLY in
// the order of work.
//
// Variants (all do the same total work: N partitions + N scatter chunks):
//
//   PP_SS    two separate loops: for(i) P(); for(i) S();
//            -> bottlenecks measured INDEPENDENTLY (no pipelining)
//
//   PSPS     interleaved: for(i) { P(); S(); }
//            -> if OOO can pipeline orthogonal bottlenecks, faster than PP_SS
//
//   PPSS_NF  sequential, no fusion: for(i) P(); barrier; for(i) S();
//            -> control: same as PP_SS but with explicit fence
//
// If PSPS < PP_SS, fusion has measurable value — the microbench
// captures it.  If PSPS >= PP_SS, OOO cannot extract the bottleneck
// orthogonality (some shared resource we haven't identified, or call
// boundaries serialize too aggressively).

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

static void hdr(bool have, int N) {
    std::printf("\nfusion_v2  (N=%d outer iters; total work = N*P + N*S in each variant)\n", N);
    if (have)
        std::printf("%-30s %9s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/N", "cyc/N", "ins/N", "IPC", "GHz", "stores/cyc");
    else
        std::printf("%-30s %9s\n", "variant", "ns/N");
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
        std::printf("%-30s %9.3f %9.3f %9.3f %6.2f %6.2f %12.3f\n",
                    name, ns, cyc, ins, ipc, ghz, sps);
    else
        std::printf("%-30s %9.3f\n", name, ns);
}

int main() {
    init_compress_table();
    /* Tiny working set: keep all buffers in L1d (M4: 128 KiB).
     *   P: psrc 8N*2 + pmask N + pL 8N*2 + pR 8N*2 = ~49N bytes
     *   S: sindices 16N*2 + symbols (separate)
     * For N=64 + symbols=4096:
     *   P: ~3.1 KiB,  sindices: 2 KiB,  symbols: 4 KiB  -> ~9 KiB total.
     * The lemire bench framework warms up before measurement; with this
     * working set everything stays L1-resident across the full run. */
    const int N = 64;

    std::vector<uint16_t> psrc(N * 8 + 16);
    std::vector<uint8_t>  pmask(N + 16);
    std::vector<uint16_t> pL(N * 8 + 16);
    std::vector<uint16_t> pR(N * 8 + 16);
    std::vector<uint8_t>  symbols(4096);
    std::vector<uint16_t> sindices(N * 16 + 16);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc) v = (uint16_t)rng();
    for (auto &m : pmask) m = (uint8_t)rng();
    for (int i = 0; i < N * 16; i++)
        sindices[i] = (uint16_t)(rng() % (symbols.size() - 32));

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");
    hdr(have, N);

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    const uint16_t *psrcp = psrc.data();
    const uint8_t  *pmp   = pmask.data();
    uint16_t       *pLp   = pL.data();
    uint16_t       *pRp   = pR.data();
    uint8_t        *symp  = symbols.data();
    const uint16_t *sip   = sindices.data();

    /* total work per fn-call: N partition_8 calls (2 stores each)
     * + N scatter16 calls (16 stores each)
     * = 2N + 16N = 18N stores */
    const int total_stores = 18 * N;

    // Baseline: just P
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        row("P_only (N P calls)", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter16(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("S_only (N S calls)", agg, have, 16 * N);
    }

    // PP_SS: two separate loops back-to-back
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
            for (int i = 0; i < N; i++)
                scatter16(symp, sip + i*16, 0x42);
            sink = acc + symp[0];
        }, params);
        (void)sink;
        row("PP_SS (loop P; loop S)", agg, have, total_stores);
    }

    // PSPS: interleaved (P then S in one iteration)
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
        row("PSPS (1 P; 1 S per iter)", agg, have, total_stores);
    }

    // SPSP: interleaved with S first
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                scatter16(symp, sip + i*16, 0x42);
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
            }
            sink = acc + symp[0];
        }, params);
        (void)sink;
        row("SPSP (1 S; 1 P per iter)", agg, have, total_stores);
    }

    // PSPS but with explicit asm("":::"memory") barrier between P and S
    // — to see if a memory barrier in the middle changes anything
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
                asm volatile("" : : : "memory");
                scatter16(symp, sip + i*16, 0x42);
            }
            sink = acc + symp[0];
        }, params);
        (void)sink;
        row("PSPS + memory barrier", agg, have, total_stores);
    }

    return 0;
}
