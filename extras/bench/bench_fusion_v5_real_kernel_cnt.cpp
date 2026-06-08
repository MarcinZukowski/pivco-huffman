// v5: like v4, but the fused inner-loop body is COPIED VERBATIM from
// the production scatter_sym_fused_root_full in src/pivco_huffman_neon.c.
//
// Goal: isolate "kernel body" from "integration overhead".
//   - If v5 ≈ v4 → the v4 reimplementation is faithful, kernel is fine,
//     and the real-world erosion is purely integration overhead
//     (recursion, register pressure, etc.).
//   - If v5 << v4 → the real kernel body has hidden costs (function-
//     call boundary, write-back of nxt fields, fused_calls++, etc.)
//     that the v4 reimplementation hides.
//
// The serial baseline uses the same partition_root_8 body (matches
// real root_full) and the same scatter_sym body (matches real
// scatter_sym).  All three are pulled directly from the production
// source — no reimplementations.

#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#include "counters/bench.h"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

/* Mirror of pivco_la_neon_t (private to pivco_huffman_neon.c).  Same
 * memory layout — only used to feed the real fused kernel below. */
typedef struct {
    int j;
    int n_left;
    int n_right;
    const uint8_t *bm;
    uint16_t *indices;
    uint16_t *tmp;
} pivco_la_neon_t;

/* Counter mirror — the production fused kernel does
 * `g_pivco_fused_calls++` at entry.  We provide it here so the body
 * compiles unchanged.  This is one global store per call, identical
 * to production. */
static unsigned long g_pivco_fused_calls = 0;

/* ============== PRODUCTION KERNEL — VERBATIM COPY ==============
 * Lifted byte-for-byte from src/pivco_huffman_neon.c.  Do not modify
 * here without updating the production source. */

#define PIVCO_LA_K BENCH_K
#ifndef PIVCO_BLOCK_SIZE
#define PIVCO_BLOCK_SIZE 8192
#endif

static inline void scatter_sym(uint8_t *symbols, const uint16_t *indices,
                                int n, uint8_t sym) {
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

static inline void scatter_sym_fused_root_full(
    uint8_t *symbols, const uint16_t *indices, int n,
    uint8_t sym, pivco_la_neon_t *nxt)
{
    g_pivco_fused_calls++;

    int max_p_iters    = (PIVCO_BLOCK_SIZE - nxt->j) / (8 * PIVCO_LA_K);
    int max_n_fused    = max_p_iters * 16;
    int n_fused        = (n / 16) * 16;
    if (n_fused > max_n_fused) n_fused = max_n_fused;

    if (n_fused > 0) {
        static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
        uint16x8_t voff = vld1q_u16(off);

        int nxt_j        = nxt->j;
        int nxt_n_left   = nxt->n_left;
        int nxt_n_right  = nxt->n_right;
        const uint8_t *nxt_bm  = nxt->bm;
        uint16_t *nxt_indices  = nxt->indices;
        uint16_t *nxt_tmp      = nxt->tmp;

        for (int j = 0; j < n_fused; j += 16) {
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

            #pragma GCC unroll 8
            for (int k = 0; k < PIVCO_LA_K; k++) {
                uint16x8_t base = vdupq_n_u16((uint16_t)nxt_j);
                uint8x16_t data = vreinterpretq_u8_u16(vaddq_u16(base, voff));
                uint8_t mask = nxt_bm[nxt_j >> 3];
                const uint8_t *tab = compress_tab[mask];
                uint8x16_t shuf_r = vld1q_u8(tab);
                uint8x16_t shuf_l = vld1q_u8(tab + 16);
                vst1q_u8((uint8_t *)(nxt_tmp     + nxt_n_right),
                          vqtbl1q_u8(data, shuf_r));
                vst1q_u8((uint8_t *)(nxt_indices + nxt_n_left),
                          vqtbl1q_u8(data, shuf_l));
                int nr = compress_popcnt[mask];
                nxt_n_right += nr;
                nxt_n_left  += (8 - nr);
                nxt_j += 8;
            }
        }

        nxt->j        = nxt_j;
        nxt->n_left   = nxt_n_left;
        nxt->n_right  = nxt_n_right;
    }

    int n_rest = n - n_fused;
    if (n_rest > 0) {
        scatter_sym(symbols, indices + n_fused, n_rest, sym);
    }
}

/* The serial baseline: the production root_full body (full-block
 * partition) followed by production scatter_sym.  Mirrors what real
 * code does when fusion is off. */
static inline void root_full_body(int N, const uint8_t *bm,
                                   uint16_t *indices_out,
                                   uint16_t *tmp_out,
                                   int *n_left_out, int *n_right_out) {
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    uint16x8_t voff = vld1q_u16(off);
    int n_left = 0, n_right = 0;
    for (int j = 0; j + 8 <= N; j += 8) {
        uint16x8_t base = vdupq_n_u16((uint16_t)j);
        uint8x16_t data = vreinterpretq_u8_u16(vaddq_u16(base, voff));
        uint8_t mask = bm[j >> 3];
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t shuf_r = vld1q_u8(tab);
        uint8x16_t shuf_l = vld1q_u8(tab + 16);
        vst1q_u8((uint8_t *)(tmp_out + n_right), vqtbl1q_u8(data, shuf_r));
        vst1q_u8((uint8_t *)(indices_out + n_left), vqtbl1q_u8(data, shuf_l));
        int nr = compress_popcnt[mask];
        n_right += nr;
        n_left  += (8 - nr);
    }
    *n_left_out = n_left;
    *n_right_out = n_right;
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
        std::printf("  %-30s %10.1f %9.1f %9.1f %6.2f %6.2f %12.4f\n",
                    name, ns, cyc, ins, ipc, ghz, ns_per_elem);
    else
        std::printf("  %-30s %10.1f                                       %12.4f\n",
                    name, ns, ns_per_elem);
}

int main(int argc, char **argv) {
    init_compress_table();

    const int P_ELEM = PIVCO_BLOCK_SIZE;
    int S_ELEM = (argc > 1) ? std::atoi(argv[1]) : (P_ELEM / 2);
    if (S_ELEM < 16) S_ELEM = 16;
    S_ELEM &= ~15;

    /* Buffers — same layout as production (cur + nxt slots). */
    std::vector<uint8_t>  bm(P_ELEM / 8 + 16);
    std::vector<uint16_t> cur_indices(P_ELEM + 64);   /* current-block scatter source */
    std::vector<uint16_t> nxt_indices(P_ELEM + 64);
    std::vector<uint16_t> nxt_tmp(P_ELEM * 2 + 64);
    std::vector<uint16_t> ser_indices(P_ELEM + 64);   /* serial baseline output */
    std::vector<uint16_t> ser_tmp(P_ELEM * 2 + 64);
    std::vector<uint8_t>  symbols(P_ELEM + 64);
    std::vector<uint16_t> sindices(S_ELEM + 64);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &b : bm) b = (uint8_t)rng();
    /* Sorted-ascending scatter targets — same as v4. */
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
    uint16_t *cur_idx_p = cur_indices.data();
    uint8_t  *symp = symbols.data();
    const uint16_t *sip = sindices.data();

    std::printf("\nfusion_v5 (REAL kernels — body verbatim from production)\n");
    std::printf("  P_ELEM=%d   S_ELEM=%d   K=%d   total=%d\n",
                P_ELEM, S_ELEM, BENCH_K, P_ELEM + S_ELEM);
    if (have) {
        std::printf("  %-30s %10s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "ns/elem");
    } else {
        std::printf("  %-30s %10s %12s\n", "variant", "ns/iter", "ns/elem");
    }

    /* Serial: full block partition (root_full body) then plain
     * scatter_sym over the scatter indices. */
    {
        auto agg = counters::bench(
            [P_ELEM, S_ELEM, bmp, cur_idx_p, symp, sip,
             ser_idx = ser_indices.data(), ser_tmp_p = ser_tmp.data()]() {
            int n_left, n_right;
            root_full_body(P_ELEM, bmp, ser_idx, ser_tmp_p, &n_left, &n_right);
            scatter_sym(symp, sip, S_ELEM, 0x42);
            ((volatile int *)&n_left)[0] = n_left + n_right + symp[0];
        }, params);
        row("serial_tight (real kernels)", agg, have, P_ELEM + S_ELEM);
    }

    /* Fused (single big call): the production kernel called once with
     * the full S_ELEM.  Partition tail loop afterwards completes any
     * partition work the fused kernel didn't reach (only matters when
     * S_ELEM*2 < P_ELEM at K=4). */
    {
        auto agg = counters::bench(
            [P_ELEM, S_ELEM, bmp, cur_idx_p, symp, sip,
             nxt_i = nxt_indices.data(), nxt_t = nxt_tmp.data()]() {
            pivco_la_neon_t nxt;
            nxt.j = 0;
            nxt.n_left = 0;
            nxt.n_right = 0;
            nxt.bm = bmp;
            nxt.indices = nxt_i;
            nxt.tmp = nxt_t;

            scatter_sym_fused_root_full(symp, sip, S_ELEM, 0x42, &nxt);

            /* Finish whatever partition the fused kernel didn't cover. */
            for (int j = nxt.j; j + 8 <= P_ELEM; j += 8) {
                static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
                uint16x8_t voff = vld1q_u16(off);
                uint16x8_t base = vdupq_n_u16((uint16_t)j);
                uint8x16_t data = vreinterpretq_u8_u16(vaddq_u16(base, voff));
                uint8_t mask = bmp[j >> 3];
                const uint8_t *tab = compress_tab[mask];
                uint8x16_t shuf_r = vld1q_u8(tab);
                uint8x16_t shuf_l = vld1q_u8(tab + 16);
                vst1q_u8((uint8_t *)(nxt_t + nxt.n_right), vqtbl1q_u8(data, shuf_r));
                vst1q_u8((uint8_t *)(nxt_i + nxt.n_left), vqtbl1q_u8(data, shuf_l));
                int nr = compress_popcnt[mask];
                nxt.n_right += nr;
                nxt.n_left  += (8 - nr);
            }
            ((volatile int *)&nxt)[0] = nxt.n_left + nxt.n_right + symp[0];
        }, params);
        char nm[40];
        std::snprintf(nm, sizeof(nm), "fused 1-call (REAL, K=%d)", BENCH_K);
        row(nm, agg, have, P_ELEM + S_ELEM);
    }

    /* Fused (many small calls): split S_ELEM into N_CALLS equal-sized
     * pieces and call scatter_sym_fused_root_full once per piece, all
     * writing into the same nxt state.  This mimics what the real
     * decoder does (~18 leaf scatters per block on prose_pride),
     * exposing per-call function-entry overhead, register reload of
     * nxt locals, the g_pivco_fused_calls global increment, etc. */
    auto run_many_calls = [&](const char *name, int n_calls) {
        if (n_calls < 1 || S_ELEM % n_calls != 0) return;
        int call_size = S_ELEM / n_calls;
        if (call_size % 16 != 0) return;
        auto agg = counters::bench(
            [P_ELEM, n_calls, call_size, bmp, symp, sip,
             nxt_i = nxt_indices.data(), nxt_t = nxt_tmp.data()]() {
            pivco_la_neon_t nxt;
            nxt.j = 0;
            nxt.n_left = 0;
            nxt.n_right = 0;
            nxt.bm = bmp;
            nxt.indices = nxt_i;
            nxt.tmp = nxt_t;

            for (int c = 0; c < n_calls; c++) {
                scatter_sym_fused_root_full(symp, sip + c * call_size,
                                             call_size, 0x42, &nxt);
            }
            /* Tail any leftover partition. */
            for (int j = nxt.j; j + 8 <= P_ELEM; j += 8) {
                static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
                uint16x8_t voff = vld1q_u16(off);
                uint16x8_t base = vdupq_n_u16((uint16_t)j);
                uint8x16_t data = vreinterpretq_u8_u16(vaddq_u16(base, voff));
                uint8_t mask = bmp[j >> 3];
                const uint8_t *tab = compress_tab[mask];
                uint8x16_t shuf_r = vld1q_u8(tab);
                uint8x16_t shuf_l = vld1q_u8(tab + 16);
                vst1q_u8((uint8_t *)(nxt_t + nxt.n_right), vqtbl1q_u8(data, shuf_r));
                vst1q_u8((uint8_t *)(nxt_i + nxt.n_left), vqtbl1q_u8(data, shuf_l));
                int nr = compress_popcnt[mask];
                nxt.n_right += nr;
                nxt.n_left  += (8 - nr);
            }
            ((volatile int *)&nxt)[0] = nxt.n_left + nxt.n_right + symp[0];
        }, params);
        row(name, agg, have, P_ELEM + S_ELEM);
    };

    char nm[40];
    std::snprintf(nm, sizeof(nm), "fused 2-calls  (S/2 each)");
    run_many_calls(nm, 2);
    std::snprintf(nm, sizeof(nm), "fused 4-calls  (S/4 each)");
    run_many_calls(nm, 4);
    std::snprintf(nm, sizeof(nm), "fused 8-calls  (S/8 each)");
    run_many_calls(nm, 8);
    std::snprintf(nm, sizeof(nm), "fused 16-calls (S/16 each)");
    run_many_calls(nm, 16);
    std::snprintf(nm, sizeof(nm), "fused 32-calls (S/32 each)");
    run_many_calls(nm, 32);

    return 0;
}
