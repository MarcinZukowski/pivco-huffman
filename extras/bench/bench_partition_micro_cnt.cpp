// Microbenchmark for partition_8 / partition_8_right / partition_8_left
// using lemire/counters for cycles/instructions/branch-miss/cache-miss.
//
// Mirrors the canasort bench_canasearch_cnt.cpp pattern.  The kernels are
// copied verbatim from src/pivco_huffman_neon.c since they are static
// inline (and we want the same single-function inlined codegen the real
// decoder gets).
//
// Usage:
//   sudo build-prof/bench_partition_micro_cnt          # default N=8192
//   sudo build-prof/bench_partition_micro_cnt --n=4096
//
// hardware counters need elevated privileges:
//   macOS:  sudo
//   Linux:  sudo, or sysctl kernel.perf_event_paranoid=0
//
// Reports per partition_8 *call* (8 elems processed):
//   ns/call cyc/call ins/call IPC cmiss/call bmiss/call

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

static inline int partition_8(const uint16_t *src, uint8_t mask,
                              uint16_t *left_out, uint16_t *right_out) {
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    uint8x16_t right = vqtbl1q_u8(data, shuf_r);
    uint8x16_t left  = vqtbl1q_u8(data, shuf_l);
    int n_right = compress_popcnt[mask];
    vst1q_u8((uint8_t *)right_out, right);
    vst1q_u8((uint8_t *)left_out, left);
    return n_right;
}

static inline int partition_8_right(const uint16_t *src, uint8_t mask,
                                    uint16_t *right_out) {
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
    vst1q_u8((uint8_t *)right_out, vqtbl1q_u8(data, shuf_r));
    return compress_popcnt[mask];
}

static inline int partition_8_left(const uint16_t *src, uint8_t mask,
                                   uint16_t *left_out) {
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
    vst1q_u8((uint8_t *)left_out, vqtbl1q_u8(data, shuf_l));
    return 8 - compress_popcnt[mask];
}

static void print_header(bool have_counters, int n_groups) {
    std::printf("\npartition_8 microbench  (n_groups=%d, ~%zu KiB working set)\n",
                n_groups, (size_t)n_groups * 8 * 3 * sizeof(uint16_t) / 1024);
    if (have_counters) {
        std::printf("%-22s %9s %9s %9s %6s %6s %10s %10s\n",
                    "kernel", "ns/call", "cyc/call", "ins/call",
                    "IPC", "GHz", "cmiss/call", "bmiss/call");
    } else {
        std::printf("%-22s %9s\n", "kernel", "ns/call");
    }
}

static void print_row(const char *name,
                      const counters::event_aggregate &agg,
                      bool have_counters,
                      size_t calls_per_iter) {
    double ns_per_call = agg.fastest_elapsed_ns() / (double)calls_per_iter;
    if (have_counters) {
        double cyc_per = agg.fastest_cycles() / (double)calls_per_iter;
        double ins_per = agg.fastest_instructions() / (double)calls_per_iter;
        double cm_per  = agg.fastest_cache_misses() / (double)calls_per_iter;
        double bm_per  = agg.fastest_branch_misses() / (double)calls_per_iter;
        double ipc     = cyc_per > 0 ? ins_per / cyc_per : 0.0;
        double ghz     = agg.cycles() / agg.elapsed_ns();
        std::printf("%-22s %9.3f %9.3f %9.3f %6.2f %6.2f %10.4f %10.4f\n",
                    name, ns_per_call, cyc_per, ins_per, ipc, ghz, cm_per, bm_per);
    } else {
        std::printf("%-22s %9.3f\n", name, ns_per_call);
    }
}

int main(int argc, char **argv) {
    // Default: 1024 groups (8192 uint16 = 16 KiB per buffer × 3 = 48 KiB
    // total — fits in M4's 128/192 KiB L1d).  Override with --n=N to test
    // L2/L3 behaviour.
    int n_groups = 1024;
    unsigned seed = 0xCAFEBABE;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!std::strncmp(a, "--n=", 4)) n_groups = std::atoi(a + 4);
        else if (!std::strncmp(a, "--seed=", 7)) seed = (unsigned)std::atoi(a + 7);
    }

    init_compress_table();

    std::vector<uint16_t> src(n_groups * 8 + 16);
    std::vector<uint8_t>  masks(n_groups);
    std::vector<uint16_t> left(n_groups * 8 + 16);
    std::vector<uint16_t> right(n_groups * 8 + 16);

    std::mt19937 rng(seed);
    for (auto &v : src)   v = (uint16_t)rng();
    for (auto &m : masks) m = (uint8_t)rng();   // ~4 bits set on avg

    bool have_counters = counters::has_performance_counters();
    if (!have_counters) {
        std::printf("[hw counters unavailable: rerun under sudo for cycles/IPC/misses]\n");
    }
    print_header(have_counters, n_groups);

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;        // 0.2s

    // partition_8 (full: both sides)
    {
        volatile int sink = 0;
        auto agg = counters::bench([&]() {
            int acc = 0;
            for (int i = 0; i < n_groups; i++) {
                acc += partition_8(src.data() + i*8, masks[i],
                                    left.data() + i*8, right.data() + i*8);
            }
            sink = acc;
        }, params);
        (void)sink;
        print_row("partition_8", agg, have_counters, n_groups);
    }

    // partition_8_right
    {
        volatile int sink = 0;
        auto agg = counters::bench([&]() {
            int acc = 0;
            for (int i = 0; i < n_groups; i++) {
                acc += partition_8_right(src.data() + i*8, masks[i],
                                          right.data() + i*8);
            }
            sink = acc;
        }, params);
        (void)sink;
        print_row("partition_8_right", agg, have_counters, n_groups);
    }

    // partition_8_left
    {
        volatile int sink = 0;
        auto agg = counters::bench([&]() {
            int acc = 0;
            for (int i = 0; i < n_groups; i++) {
                acc += partition_8_left(src.data() + i*8, masks[i],
                                         left.data() + i*8);
            }
            sink = acc;
        }, params);
        (void)sink;
        print_row("partition_8_left", agg, have_counters, n_groups);
    }

    // Hypothetical: only the loads (no stores) — to test "store-port limited"
    {
        volatile int sink = 0;
        auto agg = counters::bench([&]() {
            int acc = 0;
            for (int i = 0; i < n_groups; i++) {
                uint8x16_t data = vld1q_u8((const uint8_t *)(src.data() + i*8));
                uint8_t mask = masks[i];
                uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
                uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
                uint8x16_t right = vqtbl1q_u8(data, shuf_r);
                uint8x16_t left  = vqtbl1q_u8(data, shuf_l);
                acc += vgetq_lane_u8(right, 0) + vgetq_lane_u8(left, 0);
            }
            sink = acc;
        }, params);
        (void)sink;
        print_row("loads+TBL only (no st)", agg, have_counters, n_groups);
    }

    // Stores only (no TBL) — see how fast a 2x16B-store loop runs
    {
        volatile int sink = 0;
        auto agg = counters::bench([&]() {
            for (int i = 0; i < n_groups; i++) {
                uint8x16_t data = vld1q_u8((const uint8_t *)(src.data() + i*8));
                vst1q_u8((uint8_t *)(right.data() + i*8), data);
                vst1q_u8((uint8_t *)(left.data() + i*8), data);
            }
            sink = 0;
        }, params);
        (void)sink;
        print_row("1ld + 2st (no TBL)", agg, have_counters, n_groups);
    }

    return 0;
}
