// Block-realistic fusion microbench (NEON).  Replaces v3's tiny N=64
// loops with sizes that match a real decode of one PIVCO_BLOCK_SIZE=8192
// block, so we can predict end-to-end gains more accurately.
//
// Why v3 over-predicts:
//   - v3's PP_SS runs 256 P calls then 64 S calls in one lambda.  At
//     that scale the OOO window can almost span both loops, so the
//     observed "drain stall" is exaggerated relative to real code where
//     hundreds of cycles of recursion/dispatch sit between root_full's
//     last partition store and the first scatter's first store, naturally
//     draining the store buffer.
//
// What v4 does:
//   - serial_tight  : block-sized P loop (8192 partition elem) followed
//                     IMMEDIATELY by block-sized S loop (configurable elem).
//                     This is v3's PP_SS at realistic scale.
//   - serial_gap    : same but with a configurable filler ALU/load gap
//                     between phases (no stores) to model the recursion
//                     drain window in real code.
//   - fused         : one tight loop, 16 scatter + K * partition_root_8
//                     interleaved per chunk.  Tail loop for whichever
//                     side runs out first.
//
// Compares the EXACT kernel body shapes we use in the decoder
// (partition_root_8 fully inlined, scatter chunk fully inlined).  No
// function-call overhead.  Results should match end-to-end behavior much
// better than v3.

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

#ifndef BENCH_K
#define BENCH_K 4
#endif

/* partition_root_8 body (matches src/pivco_huffman_neon.c).  Reads the
 * current j into a vec, splits into left/right via TBL using mask. */
static inline void p_chunk(int j, const uint8_t *bm,
                            uint16_t *L, int &nL,
                            uint16_t *R, int &nR) {
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    uint16x8_t voff = vld1q_u16(off);
    uint16x8_t base = vdupq_n_u16((uint16_t)j);
    uint8x16_t data = vreinterpretq_u8_u16(vaddq_u16(base, voff));
    uint8_t mask = bm[j >> 3];
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    vst1q_u8((uint8_t *)(R + nR), vqtbl1q_u8(data, shuf_r));
    vst1q_u8((uint8_t *)(L + nL), vqtbl1q_u8(data, shuf_l));
    int nr = compress_popcnt[mask];
    nR += nr;
    nL += 8 - nr;
}

/* scatter_sym chunk: 16 byte-stores indexed via sorted indices. */
static inline void s_chunk(uint8_t *symbols, const uint16_t *indices,
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

/* Filler: GAP_OPS dependent ALU ops, no memory traffic.  Used to model
 * the recursion-dispatch window between root_full and scatter in real
 * code, which naturally drains pending stores. */
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
    init_compress_table();

    /* Block-realistic sizes.  P_ELEM = full BLK partition (8192 elements
     * → 1024 chunks of 8).  S_ELEM defaults to ~half a block, the rough
     * scatter coverage for prose_pride; override via argv[1]. */
    const int P_ELEM = 8192;
    int S_ELEM = (argc > 1) ? std::atoi(argv[1]) : 4096;
    if (S_ELEM < 16) S_ELEM = 16;
    /* Round S_ELEM down to multiple of 16. */
    S_ELEM &= ~15;

    const int N_P = P_ELEM / 8;       // partition chunks (each does 8 elem)
    const int N_S = S_ELEM / 16;      // scatter chunks (each does 16 elem)
    const int K   = BENCH_K;

    /* Buffers — partition outputs and scatter symbol buffer.
     * partition L/R sized for the worst case (all bits 1 / all 0). */
    std::vector<uint8_t>  bm(P_ELEM / 8 + 16);
    std::vector<uint16_t> L(P_ELEM + 64);
    std::vector<uint16_t> R(P_ELEM + 64);
    std::vector<uint8_t>  symbols(P_ELEM + 64);
    std::vector<uint16_t> sindices(S_ELEM + 64);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &b : bm) b = (uint8_t)rng();

    /* Sorted-ascending sparse scatter indices spanning [0, P_ELEM). */
    int step = P_ELEM / S_ELEM;
    if (step < 1) step = 1;
    for (int i = 0; i < S_ELEM; i++)
        sindices[i] = (uint16_t)(i * step);

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");

    counters::bench_parameter params;
    params.min_repeat = 30;
    params.min_time_ns = 200000000;

    const uint8_t *bmp = bm.data();
    uint16_t *Lp = L.data();
    uint16_t *Rp = R.data();
    uint8_t  *symp = symbols.data();
    const uint16_t *sip = sindices.data();

    std::printf("\nfusion_v4 (block-realistic; sweeps S_ELEM)\n");
    std::printf("  P_ELEM=%d (N_P=%d chunks)   S_ELEM=%d (N_S=%d chunks)   K=%d\n",
                P_ELEM, N_P, S_ELEM, N_S, K);
    std::printf("  total elems/iter (P+S) = %d\n", P_ELEM + S_ELEM);
    if (have) {
        std::printf("  %-26s %10s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "ns/elem");
    } else {
        std::printf("  %-26s %10s %12s\n", "variant", "ns/iter", "ns/elem");
    }

    /* serial_tight: full P loop, then full S loop, no gap. */
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [N_P, N_S, bmp, Lp, Rp, symp, sip, &sink]() {
            int nL = 0, nR = 0;
            for (int j = 0; j < N_P * 8; j += 8)
                p_chunk(j, bmp, Lp, nL, Rp, nR);
            for (int i = 0; i < N_S; i++)
                s_chunk(symp, sip + i * 16, 0x42);
            sink = nL + nR + symp[0];
        }, params);
        (void)sink;
        row("serial_tight (P then S)", agg, have, P_ELEM + S_ELEM);
    }

    /* serial_gap: full P loop, FILLER work, full S loop.  Models the
     * post-root_full / pre-scatter recursion in real code. */
    auto run_gap = [&](const char *name, auto fn) {
        volatile int sink = 0;
        auto agg = counters::bench(fn, params);
        (void)sink;
        row(name, agg, have, P_ELEM + S_ELEM);
    };

    run_gap("serial_gap (gap=64 ALU)", [N_P, N_S, bmp, Lp, Rp, symp, sip]() {
        int nL = 0, nR = 0;
        for (int j = 0; j < N_P * 8; j += 8)
            p_chunk(j, bmp, Lp, nL, Rp, nR);
        int g = filler<64>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 16, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });

    run_gap("serial_gap (gap=256 ALU)", [N_P, N_S, bmp, Lp, Rp, symp, sip]() {
        int nL = 0, nR = 0;
        for (int j = 0; j < N_P * 8; j += 8)
            p_chunk(j, bmp, Lp, nL, Rp, nR);
        int g = filler<256>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 16, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });

    run_gap("serial_gap (gap=1024 ALU)", [N_P, N_S, bmp, Lp, Rp, symp, sip]() {
        int nL = 0, nR = 0;
        for (int j = 0; j < N_P * 8; j += 8)
            p_chunk(j, bmp, Lp, nL, Rp, nR);
        int g = filler<1024>(nL + nR);
        for (int i = 0; i < N_S; i++)
            s_chunk(symp, sip + i * 16, (uint8_t)(0x42 ^ (g & 1)));
        ((volatile int *)&nL)[0] = g + symp[0];
    });

    /* fused: one loop, 16 scatter + K partition per iter, until either
     * side runs out.  Tail handles whichever side has remaining work. */
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [N_P, N_S, K, bmp, Lp, Rp, symp, sip, &sink]() {
            int nL = 0, nR = 0;
            int j  = 0;
            int N_FUSED_ITERS = N_S < (N_P / K) ? N_S : (N_P / K);
            for (int i = 0; i < N_FUSED_ITERS; i++) {
                s_chunk(symp, sip + i * 16, 0x42);
                #pragma GCC unroll 8
                for (int k = 0; k < K; k++) {
                    p_chunk(j, bmp, Lp, nL, Rp, nR);
                    j += 8;
                }
            }
            /* P tail: if partition has more chunks left */
            for (; j < N_P * 8; j += 8)
                p_chunk(j, bmp, Lp, nL, Rp, nR);
            /* S tail: if scatter has more chunks left */
            for (int i = N_FUSED_ITERS; i < N_S; i++)
                s_chunk(symp, sip + i * 16, 0x42);
            sink = nL + nR + symp[0];
        }, params);
        (void)sink;
        char nm[40];
        std::snprintf(nm, sizeof(nm), "fused (K=%d)", K);
        row(nm, agg, have, P_ELEM + S_ELEM);
    }

    return 0;
}
