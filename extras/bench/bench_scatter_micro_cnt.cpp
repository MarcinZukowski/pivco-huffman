// Microbenchmark for scatter kernels (scatter_sym, scatter_both_leaves,
// flat_decode_scatter D=2) under lemire/counters.
//
// Question: are these store-saturated?  Each call scatters N bytes to
// scattered positions in symbols[], with various amounts of compute per
// store (none / TBL+EOR / unpack+TBL).  Compare cycles, IPC, and
// stores/cycle against partition_8 (which we know is critical-path
// limited, not store-port limited at 0.27 stores/cyc).
//
// All variants use heap (std::vector) buffers + capture-by-value of
// pointers, dodging both gotchas in docs/LEMIRE-NOTES.md.
//
// Reports per CALL (which processes N indices, i.e. N stores).
//
// Usage:
//   sudo build-cnt/pivco_scatter_micro_cnt
//   sudo build-cnt/pivco_scatter_micro_cnt --n=4096

#include "counters/bench.h"
#include <arm_neon.h>
#include <algorithm>
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

// ---------- Kernels (verbatim copies from pivco_huffman_neon.c) ----------

static inline void scatter_sym(uint8_t *symbols,
                                const uint16_t *indices, int n,
                                uint8_t sym) {
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint16x8_t i0 = vld1q_u16(indices + j);
        uint16x8_t i1 = vld1q_u16(indices + j + 8);
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
    for (; j + 8 <= n; j += 8) {
        uint16x8_t idx = vld1q_u16(indices + j);
        symbols[vgetq_lane_u16(idx, 0)] = sym;
        symbols[vgetq_lane_u16(idx, 1)] = sym;
        symbols[vgetq_lane_u16(idx, 2)] = sym;
        symbols[vgetq_lane_u16(idx, 3)] = sym;
        symbols[vgetq_lane_u16(idx, 4)] = sym;
        symbols[vgetq_lane_u16(idx, 5)] = sym;
        symbols[vgetq_lane_u16(idx, 6)] = sym;
        symbols[vgetq_lane_u16(idx, 7)] = sym;
    }
    for (; j < n; j++) symbols[indices[j]] = sym;
}

static inline void scatter_both_leaves(uint8_t *symbols,
                                        const uint16_t *indices, int n,
                                        const uint8_t *bm,
                                        uint8_t sym0, uint8_t sym1) {
    uint8x8_t vsym0  = vdup_n_u8(sym0);
    uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);
    static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);

    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint8x8_t bits0 = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
        uint8x8_t vals0 = veor_u8(vsym0, vand_u8(vdelta, bits0));
        uint8x8_t bits1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]), vbit_pos);
        uint8x8_t vals1 = veor_u8(vsym0, vand_u8(vdelta, bits1));
        uint16x8_t i0 = vld1q_u16(indices + j);
        uint16x8_t i1 = vld1q_u16(indices + j + 8);
        symbols[vgetq_lane_u16(i0, 0)] = vget_lane_u8(vals0, 0);
        symbols[vgetq_lane_u16(i0, 1)] = vget_lane_u8(vals0, 1);
        symbols[vgetq_lane_u16(i0, 2)] = vget_lane_u8(vals0, 2);
        symbols[vgetq_lane_u16(i0, 3)] = vget_lane_u8(vals0, 3);
        symbols[vgetq_lane_u16(i0, 4)] = vget_lane_u8(vals0, 4);
        symbols[vgetq_lane_u16(i0, 5)] = vget_lane_u8(vals0, 5);
        symbols[vgetq_lane_u16(i0, 6)] = vget_lane_u8(vals0, 6);
        symbols[vgetq_lane_u16(i0, 7)] = vget_lane_u8(vals0, 7);
        symbols[vgetq_lane_u16(i1, 0)] = vget_lane_u8(vals1, 0);
        symbols[vgetq_lane_u16(i1, 1)] = vget_lane_u8(vals1, 1);
        symbols[vgetq_lane_u16(i1, 2)] = vget_lane_u8(vals1, 2);
        symbols[vgetq_lane_u16(i1, 3)] = vget_lane_u8(vals1, 3);
        symbols[vgetq_lane_u16(i1, 4)] = vget_lane_u8(vals1, 4);
        symbols[vgetq_lane_u16(i1, 5)] = vget_lane_u8(vals1, 5);
        symbols[vgetq_lane_u16(i1, 6)] = vget_lane_u8(vals1, 6);
        symbols[vgetq_lane_u16(i1, 7)] = vget_lane_u8(vals1, 7);
    }
    /* tail omitted for microbench (n is always a multiple of 16) */
}

// Simplified flat D=2 scatter: pretend bm-unpack produces codes; just
// look up c2s and scatter.  Inlines the inner loop body of D=2 path.
static inline void flat_d2_scatter(uint8_t *symbols,
                                    const uint16_t *indices, int n,
                                    const uint8_t *codes_bytes,  // pre-unpacked codes 0..3
                                    const uint8_t *c2s) {
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t codes = vld1q_u8(codes_bytes + i);
        uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
        symbols[indices[i     ]] = vgetq_lane_u8(syms, 0);
        symbols[indices[i +  1]] = vgetq_lane_u8(syms, 1);
        symbols[indices[i +  2]] = vgetq_lane_u8(syms, 2);
        symbols[indices[i +  3]] = vgetq_lane_u8(syms, 3);
        symbols[indices[i +  4]] = vgetq_lane_u8(syms, 4);
        symbols[indices[i +  5]] = vgetq_lane_u8(syms, 5);
        symbols[indices[i +  6]] = vgetq_lane_u8(syms, 6);
        symbols[indices[i +  7]] = vgetq_lane_u8(syms, 7);
        symbols[indices[i +  8]] = vgetq_lane_u8(syms, 8);
        symbols[indices[i +  9]] = vgetq_lane_u8(syms, 9);
        symbols[indices[i + 10]] = vgetq_lane_u8(syms, 10);
        symbols[indices[i + 11]] = vgetq_lane_u8(syms, 11);
        symbols[indices[i + 12]] = vgetq_lane_u8(syms, 12);
        symbols[indices[i + 13]] = vgetq_lane_u8(syms, 13);
        symbols[indices[i + 14]] = vgetq_lane_u8(syms, 14);
        symbols[indices[i + 15]] = vgetq_lane_u8(syms, 15);
    }
}

// Reference: bare scatter (no value compute) — same as scatter_sym but
// stripped to one-byte-store-per-elem with no SIMD lane extracts on the
// value side.  Tests the absolute scatter-store throughput floor.
static inline void scatter_bare(uint8_t *symbols, const uint16_t *indices,
                                 int n, uint8_t sym) {
    for (int j = 0; j < n; j++) symbols[indices[j]] = sym;
}

// Reference: sequential write — the easy case for the store buffer.
__attribute__((unused))
static inline void scatter_seq(uint8_t *symbols, int n, uint8_t sym) {
    std::memset(symbols, sym, (size_t)n);
}

// ---------- Bench harness ----------

static void print_header(bool have, int n) {
    std::printf("\nscatter microbench  (n=%d stores per call, ~%d KiB symbols[])\n", n, n / 1024);
    if (have) {
        std::printf("%-26s %9s %9s %9s %6s %6s %10s %10s %12s\n",
                    "kernel", "ns/call", "cyc/call", "ins/call",
                    "IPC", "GHz", "cmiss/call", "bmiss/call", "stores/cyc");
    } else {
        std::printf("%-26s %9s\n", "kernel", "ns/call");
    }
}

static void print_row(const char *name,
                      const counters::event_aggregate &agg,
                      bool have, int n_stores) {
    double ns  = agg.fastest_elapsed_ns();
    double cyc = agg.fastest_cycles();
    double ins = agg.fastest_instructions();
    double cm  = agg.fastest_cache_misses();
    double bm  = agg.fastest_branch_misses();
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    double sps = cyc > 0 ? n_stores / cyc : 0;
    if (have) {
        std::printf("%-26s %9.3f %9.3f %9.3f %6.2f %6.2f %10.4f %10.4f %12.3f\n",
                    name, ns, cyc, ins, ipc, ghz, cm, bm, sps);
    } else {
        std::printf("%-26s %9.3f\n", name, ns);
    }
}

int main(int argc, char **argv) {
    int n = 1024;
    for (int i = 1; i < argc; i++) {
        if (!std::strncmp(argv[i], "--n=", 4)) n = std::atoi(argv[i] + 4);
    }

    init_compress_table();

    // symbols[] sized for max index value (0..n-1), heap allocated.
    std::vector<uint8_t>  symbols(n + 64);
    std::vector<uint16_t> indices(n + 16);
    std::vector<uint8_t>  bm((size_t)((n + 7) / 8) + 16);
    std::vector<uint8_t>  codes_d2(n + 16);    // pre-unpacked D=2 codes
    std::vector<uint8_t>  c2s(16, 0);

    std::mt19937 rng(0xCAFEBABE);
    // indices: shuffled 0..n-1 → realistic scattered access pattern
    for (int i = 0; i < n; i++) indices[i] = (uint16_t)i;
    std::shuffle(indices.begin(), indices.begin() + n, rng);
    for (auto &b : bm) b = (uint8_t)rng();
    for (auto &c : codes_d2) c = (uint8_t)(rng() & 3);
    for (int i = 0; i < 4; i++) c2s[i] = (uint8_t)('A' + i);

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");
    print_header(have, n);

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    // Capture pointers by value (see docs/LEMIRE-NOTES.md gotcha #1).
    uint8_t        *symp = symbols.data();
    const uint16_t *idxp = indices.data();
    const uint8_t  *bmp  = bm.data();
    const uint8_t  *cdp  = codes_d2.data();
    const uint8_t  *c2sp = c2s.data();
    int N = n;

    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, N, &sink]() {
            std::memset(symp, 0xAA, (size_t)N);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("memset (seq stores)", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, idxp, N, &sink]() {
            scatter_bare(symp, idxp, N, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("scatter_bare", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, idxp, N, &sink]() {
            scatter_sym(symp, idxp, N, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("scatter_sym", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, idxp, bmp, N, &sink]() {
            scatter_both_leaves(symp, idxp, N, bmp, 0x41, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("scatter_both_leaves", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, idxp, cdp, c2sp, N, &sink]() {
            flat_d2_scatter(symp, idxp, N, cdp, c2sp);
            sink = symp[0];
        }, params);
        (void)sink;
        print_row("flat_d2_scatter", agg, have, N);
    }
    return 0;
}
