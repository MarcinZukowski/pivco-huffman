// Test: are 2 partition_8 calls in the same loop body cheaper than the
// sum of 1 + 1?  If partition_8 is truly critical-path bound, two
// independent invocations should overlap freely and 2x cost ~= 1x.
//
// Variants:
//   single        1 p8 per iter
//   dual_indep    2 p8 per iter, INDEPENDENT input/output buffers
//   dual_same_src 2 p8 per iter, SAME source array (but indep out)
//   dual_back2back  same as dual_indep but textually concatenated
//                  (compiler-scheduled vs manually back-to-back)

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

static void hdr(bool have, int n) {
    std::printf("\np8 single vs dual  (n_iters=%d)\n", n);
    if (have) {
        std::printf("%-26s %9s %9s %9s %6s %6s %12s %10s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "stores/cyc", "p8/iter");
    } else {
        std::printf("%-26s %9s\n", "variant", "ns/iter");
    }
}

static void row(const char *name, const counters::event_aggregate &agg,
                bool have, int N, int p8_per_iter) {
    double ns  = agg.fastest_elapsed_ns() / (double)N;
    double cyc = agg.fastest_cycles() / (double)N;
    double ins = agg.fastest_instructions() / (double)N;
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    int stores = 2 * p8_per_iter;
    double sps = cyc > 0 ? stores / cyc : 0;
    if (have) {
        std::printf("%-26s %9.3f %9.3f %9.3f %6.2f %6.2f %12.3f %10d\n",
                    name, ns, cyc, ins, ipc, ghz, sps, p8_per_iter);
    } else {
        std::printf("%-26s %9.3f\n", name, ns);
    }
}

int main() {
    init_compress_table();
    const int N = 1024;

    // Two independent partition contexts (A and B)
    std::vector<uint16_t> srcA(N * 8 + 16), srcB(N * 8 + 16);
    std::vector<uint8_t>  maskA(N + 16),     maskB(N + 16);
    std::vector<uint16_t> LA(N * 8 + 16),    LB(N * 8 + 16);
    std::vector<uint16_t> RA(N * 8 + 16),    RB(N * 8 + 16);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : srcA)  v = (uint16_t)rng();
    for (auto &v : srcB)  v = (uint16_t)rng();
    for (auto &m : maskA) m = (uint8_t)rng();
    for (auto &m : maskB) m = (uint8_t)rng();

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");
    hdr(have, N);

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    const uint16_t *sA = srcA.data();
    const uint16_t *sB = srcB.data();
    const uint8_t  *mA = maskA.data();
    const uint8_t  *mB = maskB.data();
    uint16_t *laA = LA.data(); uint16_t *raA = RA.data();
    uint16_t *laB = LB.data(); uint16_t *raB = RB.data();

    // 1 p8 per iter
    {
        volatile int sink = 0;
        auto agg = counters::bench([sA, mA, laA, raA, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(sA + i*8, mA[i], laA + i*8, raA + i*8);
            sink = acc;
        }, params);
        (void)sink;
        row("single", agg, have, N, 1);
    }

    // 2 p8 per iter, INDEPENDENT
    {
        volatile int sink = 0;
        auto agg = counters::bench([sA, mA, laA, raA, sB, mB, laB, raB, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                acc += p8(sA + i*8, mA[i], laA + i*8, raA + i*8);
                acc += p8(sB + i*8, mB[i], laB + i*8, raB + i*8);
            }
            sink = acc;
        }, params);
        (void)sink;
        row("dual_indep", agg, have, N, 2);
    }

    // 2 p8 per iter, SAME source (forces some load reuse) but indep outputs
    {
        volatile int sink = 0;
        auto agg = counters::bench([sA, mA, laA, raA, mB, laB, raB, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                acc += p8(sA + i*8, mA[i], laA + i*8, raA + i*8);
                acc += p8(sA + i*8, mB[i], laB + i*8, raB + i*8);
            }
            sink = acc;
        }, params);
        (void)sink;
        row("dual_same_src", agg, have, N, 2);
    }

    // 4 p8 per iter (test even more parallelism)
    {
        std::vector<uint16_t> srcC(N * 8 + 16), srcD(N * 8 + 16);
        std::vector<uint8_t>  maskC(N + 16),     maskD(N + 16);
        std::vector<uint16_t> LC(N * 8 + 16),    LD(N * 8 + 16);
        std::vector<uint16_t> RC(N * 8 + 16),    RD(N * 8 + 16);
        for (auto &v : srcC) v = (uint16_t)rng();
        for (auto &v : srcD) v = (uint16_t)rng();
        for (auto &m : maskC) m = (uint8_t)rng();
        for (auto &m : maskD) m = (uint8_t)rng();

        const uint16_t *sC = srcC.data();
        const uint16_t *sD = srcD.data();
        const uint8_t  *mC = maskC.data();
        const uint8_t  *mD = maskD.data();
        uint16_t *laC = LC.data(); uint16_t *raC = RC.data();
        uint16_t *laD = LD.data(); uint16_t *raD = RD.data();

        volatile int sink = 0;
        auto agg = counters::bench(
            [sA, mA, laA, raA, sB, mB, laB, raB,
             sC, mC, laC, raC, sD, mD, laD, raD, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                acc += p8(sA + i*8, mA[i], laA + i*8, raA + i*8);
                acc += p8(sB + i*8, mB[i], laB + i*8, raB + i*8);
                acc += p8(sC + i*8, mC[i], laC + i*8, raC + i*8);
                acc += p8(sD + i*8, mD[i], laD + i*8, raD + i*8);
            }
            sink = acc;
        }, params);
        (void)sink;
        row("quad_indep", agg, have, N, 4);
    }
    return 0;
}
