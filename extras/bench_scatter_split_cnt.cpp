// Decompose scatter_sym's per-iter cost into lane-extract vs store.
//
// scatter_sym does, per iter (16 elements):
//   - 1 vld (load 8 indices)  ×2 = 2 vector loads
//   - 16 vgetq_lane_u16 (lane extracts of indices)
//   - 16 strb (byte stores at scattered addresses)
//
// Question: is the bottleneck the lane extracts, the stores, or both?
//
// Test isolates:
//   A baseline:       16 lane-extracts + 16 stores       (full scatter_sym)
//   B stores only:    1 lane-extract + 16 stores         (all stores to indices[0])
//   C extracts only:  16 lane-extracts + 0 stores        (XOR all into accumulator)
//   D fewer lanes:    8 lane-extracts + 16 stores         (use lane[i&7] twice — half the extracts)
//   E full vec store: 0 lane-extracts + 1 vec store       (sequential store of 16B vector)
//
// Reports per CALL (which processes N stores).
//
// Run with sudo for cycle counters.

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

// ---------- Kernel variants (per 16-element chunk) ----------

// A: full scatter_sym — 16 lane-extracts + 16 byte stores
static inline void chunk_full(uint8_t *symbols,
                                const uint16_t *indices,
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

// B: stores only — 1 lane-extract, 16 stores to the SAME address.
// Removes lane-extract pressure but keeps store-port pressure.
static inline void chunk_stores_only(uint8_t *symbols,
                                       const uint16_t *indices,
                                       uint8_t sym) {
    uint16x8_t i0 = vld1q_u16(indices);
    uint16_t addr = vgetq_lane_u16(i0, 0);
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
    symbols[addr] = sym;
}

// C: lane extracts only — 16 lane extracts, no store (XOR into accumulator).
// Removes store-port pressure but keeps lane-extract pressure.
static inline uint16_t chunk_extracts_only(const uint16_t *indices) {
    uint16x8_t i0 = vld1q_u16(indices);
    uint16x8_t i1 = vld1q_u16(indices + 8);
    uint16_t acc = 0;
    acc ^= vgetq_lane_u16(i0, 0);
    acc ^= vgetq_lane_u16(i0, 1);
    acc ^= vgetq_lane_u16(i0, 2);
    acc ^= vgetq_lane_u16(i0, 3);
    acc ^= vgetq_lane_u16(i0, 4);
    acc ^= vgetq_lane_u16(i0, 5);
    acc ^= vgetq_lane_u16(i0, 6);
    acc ^= vgetq_lane_u16(i0, 7);
    acc ^= vgetq_lane_u16(i1, 0);
    acc ^= vgetq_lane_u16(i1, 1);
    acc ^= vgetq_lane_u16(i1, 2);
    acc ^= vgetq_lane_u16(i1, 3);
    acc ^= vgetq_lane_u16(i1, 4);
    acc ^= vgetq_lane_u16(i1, 5);
    acc ^= vgetq_lane_u16(i1, 6);
    acc ^= vgetq_lane_u16(i1, 7);
    return acc;
}

// D: half the lane extracts — extract 8 indices, scatter 16 stores
// (re-using each lane twice into adjacent positions).  This isn't
// semantically identical to full scatter (writes 16 stores to only 8
// distinct addresses), but it has the same store count and half the
// lane-extract count, so cyc/iter difference vs A pinpoints lane cost.
static inline void chunk_half_lanes(uint8_t *symbols,
                                      const uint16_t *indices,
                                      uint8_t sym) {
    uint16x8_t i0 = vld1q_u16(indices);
    uint16_t a0 = vgetq_lane_u16(i0, 0);
    uint16_t a1 = vgetq_lane_u16(i0, 1);
    uint16_t a2 = vgetq_lane_u16(i0, 2);
    uint16_t a3 = vgetq_lane_u16(i0, 3);
    uint16_t a4 = vgetq_lane_u16(i0, 4);
    uint16_t a5 = vgetq_lane_u16(i0, 5);
    uint16_t a6 = vgetq_lane_u16(i0, 6);
    uint16_t a7 = vgetq_lane_u16(i0, 7);
    symbols[a0] = sym; symbols[a1] = sym;
    symbols[a2] = sym; symbols[a3] = sym;
    symbols[a4] = sym; symbols[a5] = sym;
    symbols[a6] = sym; symbols[a7] = sym;
    symbols[a0+1] = sym; symbols[a1+1] = sym;
    symbols[a2+1] = sym; symbols[a3+1] = sym;
    symbols[a4+1] = sym; symbols[a5+1] = sym;
    symbols[a6+1] = sym; symbols[a7+1] = sym;
}

// E: full vector store — 0 lane extracts, 1 vec store of 16 bytes.
// Best case for store throughput; impossible without dense indices.
static inline void chunk_vec_store(uint8_t *symbols,
                                     const uint16_t *indices,
                                     uint8_t sym) {
    uint16x8_t i0 = vld1q_u16(indices);
    uint16_t addr = vgetq_lane_u16(i0, 0);
    uint8x16_t v = vdupq_n_u8(sym);
    vst1q_u8(symbols + addr, v);
}

static void print_header(bool have, int n) {
    std::printf("\nscatter decomposition  (n=%d stores per call, working over scattered indices)\n", n);
    if (have) {
        std::printf("%-26s %9s %9s %9s %6s %6s %12s %14s\n",
                    "variant", "ns/call", "cyc/call", "ins/call",
                    "IPC", "GHz", "stores/cyc", "extracts/cyc");
    } else {
        std::printf("%-26s %9s\n", "variant", "ns/call");
    }
}

static void print_row(const char *name,
                      const counters::event_aggregate &agg,
                      bool have, int n_stores, int n_extracts) {
    double ns  = agg.fastest_elapsed_ns();
    double cyc = agg.fastest_cycles();
    double ins = agg.fastest_instructions();
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    double sps = cyc > 0 ? n_stores / cyc : 0;
    double xps = cyc > 0 ? n_extracts / cyc : 0;
    if (have) {
        std::printf("%-26s %9.3f %9.3f %9.3f %6.2f %6.2f %12.3f %14.3f\n",
                    name, ns, cyc, ins, ipc, ghz, sps, xps);
    } else {
        std::printf("%-26s %9.3f\n", name, ns);
    }
}

int main() {
    init_compress_table();

    const int N = 1024; // chunks per fn-call (16 elements per chunk)
    const int n_per = 16;

    std::vector<uint8_t>  symbols(64 * 1024);
    std::vector<uint16_t> sindices(N * 16 + 16);

    std::mt19937 rng(0xCAFEBABE);
    for (int i = 0; i < N * 16; i++)
        sindices[i] = (uint16_t)(rng() % (symbols.size() - 32));

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");
    print_header(have, N * n_per);

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    uint8_t        *symp = symbols.data();
    const uint16_t *sip  = sindices.data();

    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) chunk_full(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("A full (16ext + 16st)", agg, have, N*16, N*16);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) chunk_stores_only(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("B stores only (1ext+16st)", agg, have, N*16, N);
    }
    {
        volatile uint16_t sink = 0;
        auto agg = counters::bench([sip, &sink]() {
            uint16_t acc = 0;
            for (int i = 0; i < N; i++) acc ^= chunk_extracts_only(sip + i*16);
            sink = acc;
        }, params);
        (void)sink;
        print_row("C extracts only (16ext+0st)", agg, have, 0, N*16);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) chunk_half_lanes(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("D half-ext (8ext + 16st)", agg, have, N*16, N*8);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) chunk_vec_store(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("E vec-store (0ext + 1vst)", agg, have, N*16, 0);
    }

    return 0;
}
