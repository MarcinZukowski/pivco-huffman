// SSE4.1 microbenches: partition + scatter + dual + fusion.  Mirrors
// the NEON suite (bench_{partition,p8_dual,fusion,scatter_split}_micro
// _cnt.cpp) for Zen 3 / Skylake / etc. without AVX-512.
//
// Kernels copied from src/pivco_huffman_x86.c (partition_8_sse,
// partition_8_sse_right/left, scatter_write_sse).
//
// All variants use heap buffers + capture-by-value of pointers
// (see docs/LEMIRE-NOTES.md for why).
//
// Run with sudo or sysctl kernel.perf_event_paranoid=0.

#include "counters/bench.h"
#include <smmintrin.h>      // SSE4.1
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

// Local copies of compress_tab/compress_popcnt — the originals live in
// pivco_huffman_neon.c and are only compiled on aarch64.  For an x86
// microbench we just rebuild them here.
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

// ---------- Kernels (copied from pivco_huffman_x86.c) ----------

static inline int p8(const uint16_t *src, uint8_t mask,
                      uint16_t *L, uint16_t *R) {
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    const uint8_t *tab = compress_tab[mask];
    __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));
    __m128i right = _mm_shuffle_epi8(data, shuf_r);
    __m128i left  = _mm_shuffle_epi8(data, shuf_l);
    int n_right = compress_popcnt[mask];
    _mm_storeu_si128((__m128i *)R, right);
    _mm_storeu_si128((__m128i *)L, left);
    return n_right;
}

static inline int p8r(const uint16_t *src, uint8_t mask, uint16_t *R) {
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_r = _mm_load_si128((const __m128i *)compress_tab[mask]);
    _mm_storeu_si128((__m128i *)R, _mm_shuffle_epi8(data, shuf_r));
    return compress_popcnt[mask];
}

static inline int p8l(const uint16_t *src, uint8_t mask, uint16_t *L) {
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(compress_tab[mask] + 16));
    _mm_storeu_si128((__m128i *)L, _mm_shuffle_epi8(data, shuf_l));
    return 8 - compress_popcnt[mask];
}

// scatter_sym (16 elems per call): two 8-wide chunks
static inline void scatter16(uint8_t *symbols, const uint16_t *indices,
                              uint8_t sym) {
    __m128i i0 = _mm_loadu_si128((const __m128i *)indices);
    __m128i i1 = _mm_loadu_si128((const __m128i *)(indices + 8));
    symbols[_mm_extract_epi16(i0, 0)] = sym;
    symbols[_mm_extract_epi16(i0, 1)] = sym;
    symbols[_mm_extract_epi16(i0, 2)] = sym;
    symbols[_mm_extract_epi16(i0, 3)] = sym;
    symbols[_mm_extract_epi16(i0, 4)] = sym;
    symbols[_mm_extract_epi16(i0, 5)] = sym;
    symbols[_mm_extract_epi16(i0, 6)] = sym;
    symbols[_mm_extract_epi16(i0, 7)] = sym;
    symbols[_mm_extract_epi16(i1, 0)] = sym;
    symbols[_mm_extract_epi16(i1, 1)] = sym;
    symbols[_mm_extract_epi16(i1, 2)] = sym;
    symbols[_mm_extract_epi16(i1, 3)] = sym;
    symbols[_mm_extract_epi16(i1, 4)] = sym;
    symbols[_mm_extract_epi16(i1, 5)] = sym;
    symbols[_mm_extract_epi16(i1, 6)] = sym;
    symbols[_mm_extract_epi16(i1, 7)] = sym;
}

// ---------- Bench harness ----------

static void hdr_partition(bool have, int n) {
    std::printf("\n=== partition_8_sse  (n_groups=%d) ===\n", n);
    if (have) {
        std::printf("%-26s %9s %9s %9s %6s %6s %12s\n",
                    "kernel", "ns/call", "cyc/call", "ins/call",
                    "IPC", "GHz", "stores/cyc");
    } else {
        std::printf("%-26s %9s\n", "kernel", "ns/call");
    }
}

static void hdr_dual(bool have, int n) {
    std::printf("\n=== p8_sse single vs dual  (n_iters=%d) ===\n", n);
    if (have) {
        std::printf("%-26s %9s %9s %9s %6s %6s %12s %10s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "stores/cyc", "p8/iter");
    } else {
        std::printf("%-26s %9s\n", "variant", "ns/iter");
    }
}

static void hdr_fusion(bool have, int n) {
    std::printf("\n=== fusion_sse  (n_iters=%d) ===\n", n);
    if (have) {
        std::printf("%-26s %9s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "stores/cyc");
    } else {
        std::printf("%-26s %9s\n", "variant", "ns/iter");
    }
}

static void hdr_scatter_split(bool have, int n) {
    std::printf("\n=== scatter_sse split  (n=%d stores/call) ===\n", n);
    if (have) {
        std::printf("%-26s %9s %9s %9s %6s %6s %12s %14s\n",
                    "variant", "ns/call", "cyc/call", "ins/call",
                    "IPC", "GHz", "stores/cyc", "extracts/cyc");
    } else {
        std::printf("%-26s %9s\n", "variant", "ns/call");
    }
}

static void row(const char *name, const counters::event_aggregate &agg,
                bool have, int stores, int extracts = -1, int p8s = -1,
                int calls_per_fn = 1024) {
    double ns  = agg.fastest_elapsed_ns()      / (double)calls_per_fn;
    double cyc = agg.fastest_cycles()          / (double)calls_per_fn;
    double ins = agg.fastest_instructions()    / (double)calls_per_fn;
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    int    stores_per_call   = stores   / calls_per_fn;
    int    extracts_per_call = extracts / calls_per_fn;
    double sps = cyc > 0 ? stores_per_call / cyc : 0;
    if (!have) { std::printf("%-26s %9.3f\n", name, ns); return; }
    if (extracts >= 0) {
        double xps = cyc > 0 ? extracts_per_call / cyc : 0;
        std::printf("%-26s %9.3f %9.3f %9.3f %6.2f %6.2f %12.3f %14.3f\n",
                    name, ns, cyc, ins, ipc, ghz, sps, xps);
    } else if (p8s >= 0) {
        std::printf("%-26s %9.3f %9.3f %9.3f %6.2f %6.2f %12.3f %10d\n",
                    name, ns, cyc, ins, ipc, ghz, sps, p8s);
    } else {
        std::printf("%-26s %9.3f %9.3f %9.3f %6.2f %6.2f %12.3f\n",
                    name, ns, cyc, ins, ipc, ghz, sps);
    }
}

int main() {
    init_compress_table();
    const int N = 1024;

    std::vector<uint16_t> psrc(N * 8 + 16);
    std::vector<uint8_t>  pmask(N + 16);
    std::vector<uint16_t> pL(N * 8 + 16);
    std::vector<uint16_t> pR(N * 8 + 16);
    std::vector<uint16_t> psrc2(N * 8 + 16), pL2(N * 8 + 16), pR2(N * 8 + 16);
    std::vector<uint8_t>  pmask2(N + 16);
    std::vector<uint8_t>  symbols(64 * 1024);
    std::vector<uint16_t> sindices(N * 16 + 16);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc)   v = (uint16_t)rng();
    for (auto &v : psrc2)  v = (uint16_t)rng();
    for (auto &m : pmask)  m = (uint8_t)rng();
    for (auto &m : pmask2) m = (uint8_t)rng();
    for (int i = 0; i < N * 16; i++)
        sindices[i] = (uint16_t)(rng() % (symbols.size() - 32));

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo or set perf_event_paranoid=0]\n");

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    const uint16_t *psrcp = psrc.data();
    const uint8_t  *pmp   = pmask.data();
    uint16_t       *pLp   = pL.data();
    uint16_t       *pRp   = pR.data();
    const uint16_t *psrcp2 = psrc2.data();
    const uint8_t  *pmp2   = pmask2.data();
    uint16_t       *pLp2   = pL2.data();
    uint16_t       *pRp2   = pR2.data();
    uint8_t        *symp   = symbols.data();
    const uint16_t *sip    = sindices.data();

    // --- partition_8 family ---
    hdr_partition(have, N);
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        row("partition_8_sse", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8r(psrcp + i*8, pmp[i], pRp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        row("partition_8_sse_right", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8l(psrcp + i*8, pmp[i], pLp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        row("partition_8_sse_left", agg, have, N);
    }

    // --- scatter ---
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter16(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("scatter16_sse", agg, have, 16 * N);
    }

    // --- dual cursor ---
    hdr_dual(have, N);
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        row("single", agg, have, 2 * N, -1, 1);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [psrcp, pmp, pLp, pRp, psrcp2, pmp2, pLp2, pRp2, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                acc += p8(psrcp  + i*8, pmp[i],  pLp  + i*8, pRp  + i*8);
                acc += p8(psrcp2 + i*8, pmp2[i], pLp2 + i*8, pRp2 + i*8);
            }
            sink = acc;
        }, params);
        (void)sink;
        row("dual_indep", agg, have, 4 * N, -1, 2);
    }

    // --- fusion P + S ---
    hdr_fusion(have, N);
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p8(psrcp + i*8, pmp[i], pLp + i*8, pRp + i*8);
            sink = acc;
        }, params);
        (void)sink;
        row("P_only (2 st/iter)", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter16(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("S_only (16 st/iter)", agg, have, 16 * N);
    }
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
        row("serial P;S (18 st/iter)", agg, have, 18 * N);
    }

    // --- scatter split: extracts vs stores ---
    hdr_scatter_split(have, 16 * N);
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter16(symp, sip + i*16, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("A full (16ext + 16st)", agg, have, 16 * N, 16 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) {
                __m128i i0 = _mm_loadu_si128((const __m128i *)(sip + i*16));
                int addr = _mm_extract_epi16(i0, 0);
                for (int k = 0; k < 16; k++) symp[addr] = 0x42;
            }
            sink = symp[0];
        }, params);
        (void)sink;
        row("B stores only (1ext+16st)", agg, have, 16 * N, N);
    }
    {
        volatile uint16_t sink = 0;
        auto agg = counters::bench([sip, &sink]() {
            uint16_t acc = 0;
            for (int i = 0; i < N; i++) {
                __m128i i0 = _mm_loadu_si128((const __m128i *)(sip + i*16));
                __m128i i1 = _mm_loadu_si128((const __m128i *)(sip + i*16 + 8));
                acc ^= _mm_extract_epi16(i0, 0); acc ^= _mm_extract_epi16(i0, 1);
                acc ^= _mm_extract_epi16(i0, 2); acc ^= _mm_extract_epi16(i0, 3);
                acc ^= _mm_extract_epi16(i0, 4); acc ^= _mm_extract_epi16(i0, 5);
                acc ^= _mm_extract_epi16(i0, 6); acc ^= _mm_extract_epi16(i0, 7);
                acc ^= _mm_extract_epi16(i1, 0); acc ^= _mm_extract_epi16(i1, 1);
                acc ^= _mm_extract_epi16(i1, 2); acc ^= _mm_extract_epi16(i1, 3);
                acc ^= _mm_extract_epi16(i1, 4); acc ^= _mm_extract_epi16(i1, 5);
                acc ^= _mm_extract_epi16(i1, 6); acc ^= _mm_extract_epi16(i1, 7);
            }
            sink = acc;
        }, params);
        (void)sink;
        row("C extracts only (16ext+0st)", agg, have, 0, 16 * N);
    }
    return 0;
}
