// AVX-512 microbenches: partition + scatter + dual + fusion.  For
// Xeon Granite Rapids and Zen 5 EPYC.
//
// Kernels copied from src/pivco_huffman_avx512.c (partition_32_full,
// partition_32_right/left, scatter_write_avx512).
//
// All variants use heap buffers + capture-by-value of pointers
// (see docs/LEMIRE-NOTES.md for why).
//
// Run with sudo or sysctl kernel.perf_event_paranoid=0.

#include "counters/bench.h"
#include <immintrin.h>      // AVX-512
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

static inline int p32(const uint16_t *src, uint32_t mask,
                      uint16_t *L, uint16_t *R) {
    __m512i data = _mm512_loadu_si512((const __m512i *)src);
    __m512i right = _mm512_maskz_compress_epi16((__mmask32)mask, data);
    int n_right = _mm_popcnt_u32(mask);
    _mm512_storeu_si512((__m512i *)R, right);
    __m512i left = _mm512_maskz_compress_epi16((__mmask32)~mask, data);
    _mm512_storeu_si512((__m512i *)L, left);
    return n_right;
}

static inline int p32r(const uint16_t *src, uint32_t mask, uint16_t *R) {
    __m512i data = _mm512_loadu_si512((const __m512i *)src);
    __m512i right = _mm512_maskz_compress_epi16((__mmask32)mask, data);
    _mm512_storeu_si512((__m512i *)R, right);
    return _mm_popcnt_u32(mask);
}

static inline int p32l(const uint16_t *src, uint32_t mask, uint16_t *L) {
    __m512i data = _mm512_loadu_si512((const __m512i *)src);
    __m512i left = _mm512_maskz_compress_epi16((__mmask32)~mask, data);
    _mm512_storeu_si512((__m512i *)L, left);
    return 32 - _mm_popcnt_u32(mask);
}

// scatter_sym style: 32 byte-stores per call via SSE-extract chunks.
// Mirrors scatter_write_avx512 in src/pivco_huffman_avx512.c.
static inline void scatter32(uint8_t *symbols, const uint16_t *indices,
                              uint8_t sym) {
    __m128i i0 = _mm_loadu_si128((const __m128i *)indices);
    __m128i i1 = _mm_loadu_si128((const __m128i *)(indices + 8));
    __m128i i2 = _mm_loadu_si128((const __m128i *)(indices + 16));
    __m128i i3 = _mm_loadu_si128((const __m128i *)(indices + 24));
#define X(V, K) symbols[_mm_extract_epi16(V, K)] = sym
    X(i0,0); X(i0,1); X(i0,2); X(i0,3); X(i0,4); X(i0,5); X(i0,6); X(i0,7);
    X(i1,0); X(i1,1); X(i1,2); X(i1,3); X(i1,4); X(i1,5); X(i1,6); X(i1,7);
    X(i2,0); X(i2,1); X(i2,2); X(i2,3); X(i2,4); X(i2,5); X(i2,6); X(i2,7);
    X(i3,0); X(i3,1); X(i3,2); X(i3,3); X(i3,4); X(i3,5); X(i3,6); X(i3,7);
#undef X
}

static void hdr(const char *title, bool have, int n) {
    std::printf("\n=== %s  (n_iters=%d) ===\n", title, n);
    if (have) {
        std::printf("%-26s %9s %9s %9s %6s %6s %12s\n",
                    "kernel", "ns/call", "cyc/call", "ins/call",
                    "IPC", "GHz", "stores/cyc");
    } else {
        std::printf("%-26s %9s\n", "kernel", "ns/call");
    }
}

static void row(const char *name, const counters::event_aggregate &agg,
                bool have, int stores, int calls_per_fn = 256) {
    double ns  = agg.fastest_elapsed_ns()      / (double)calls_per_fn;
    double cyc = agg.fastest_cycles()          / (double)calls_per_fn;
    double ins = agg.fastest_instructions()    / (double)calls_per_fn;
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    int stores_per_call = stores / calls_per_fn;
    double sps = cyc > 0 ? stores_per_call / cyc : 0;
    if (!have) { std::printf("%-26s %9.3f\n", name, ns); return; }
    std::printf("%-26s %9.3f %9.3f %9.3f %6.2f %6.2f %12.3f\n",
                name, ns, cyc, ins, ipc, ghz, sps);
}

int main() {
    const int N = 256;   // 32-element groups, working set ~ 32KB/buf

    std::vector<uint16_t> psrc(N * 32 + 32);
    std::vector<uint32_t> pmask(N + 16);
    std::vector<uint16_t> pL(N * 32 + 32);
    std::vector<uint16_t> pR(N * 32 + 32);
    std::vector<uint16_t> psrc2(N * 32 + 32), pL2(N * 32 + 32), pR2(N * 32 + 32);
    std::vector<uint32_t> pmask2(N + 16);
    std::vector<uint8_t>  symbols(64 * 1024);
    std::vector<uint16_t> sindices(N * 32 + 32);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : psrc)   v = (uint16_t)rng();
    for (auto &v : psrc2)  v = (uint16_t)rng();
    for (auto &m : pmask)  m = rng();
    for (auto &m : pmask2) m = rng();
    for (int i = 0; i < N * 32; i++)
        sindices[i] = (uint16_t)(rng() % (symbols.size() - 64));

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo or set perf_event_paranoid=0]\n");

    counters::bench_parameter params;
    params.min_repeat = 50;
    params.min_time_ns = 200000000;

    const uint16_t *psrcp  = psrc.data();
    const uint32_t *pmp    = pmask.data();
    uint16_t       *pLp    = pL.data();
    uint16_t       *pRp    = pR.data();
    const uint16_t *psrcp2 = psrc2.data();
    const uint32_t *pmp2   = pmask2.data();
    uint16_t       *pLp2   = pL2.data();
    uint16_t       *pRp2   = pR2.data();
    uint8_t        *symp   = symbols.data();
    const uint16_t *sip    = sindices.data();

    // --- partition_32 family ---
    hdr("partition_32 (vpcompressw)", have, N);
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p32(psrcp + i*32, pmp[i], pLp + i*32, pRp + i*32);
            sink = acc;
        }, params);
        (void)sink;
        row("partition_32 (full, 2 st)", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p32r(psrcp + i*32, pmp[i], pRp + i*32);
            sink = acc;
        }, params);
        (void)sink;
        row("partition_32_right (1 st)", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p32l(psrcp + i*32, pmp[i], pLp + i*32);
            sink = acc;
        }, params);
        (void)sink;
        row("partition_32_left  (1 st)", agg, have, N);
    }

    // --- scatter (32 byte-stores per call) ---
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter32(symp, sip + i*32, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("scatter32_avx512 (32 st)", agg, have, 32 * N);
    }

    // --- dual cursor partition_32 ---
    hdr("p32 single vs dual", have, N);
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p32(psrcp + i*32, pmp[i], pLp + i*32, pRp + i*32);
            sink = acc;
        }, params);
        (void)sink;
        row("single (2 st/iter)", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [psrcp, pmp, pLp, pRp, psrcp2, pmp2, pLp2, pRp2, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                acc += p32(psrcp  + i*32, pmp[i],  pLp  + i*32, pRp  + i*32);
                acc += p32(psrcp2 + i*32, pmp2[i], pLp2 + i*32, pRp2 + i*32);
            }
            sink = acc;
        }, params);
        (void)sink;
        row("dual_indep (4 st/iter)", agg, have, 4 * N);
    }

    // --- fusion: 1 partition_32 + 1 scatter32 (matched element counts) ---
    hdr("fusion p32 + scatter32  (32 elem/iter each)", have, N);
    {
        volatile int sink = 0;
        auto agg = counters::bench([psrcp, pmp, pLp, pRp, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++)
                acc += p32(psrcp + i*32, pmp[i], pLp + i*32, pRp + i*32);
            sink = acc;
        }, params);
        (void)sink;
        row("P_only (2 st/iter)", agg, have, 2 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench([symp, sip, &sink]() {
            for (int i = 0; i < N; i++) scatter32(symp, sip + i*32, 0x42);
            sink = symp[0];
        }, params);
        (void)sink;
        row("S_only (32 st/iter)", agg, have, 32 * N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [psrcp, pmp, pLp, pRp, symp, sip, &sink]() {
            int acc = 0;
            for (int i = 0; i < N; i++) {
                acc += p32(psrcp + i*32, pmp[i], pLp + i*32, pRp + i*32);
                scatter32(symp, sip + i*32, 0x42);
            }
            sink = acc + symp[0];
        }, params);
        (void)sink;
        row("serial P;S (34 st/iter)", agg, have, 34 * N);
    }
    return 0;
}
