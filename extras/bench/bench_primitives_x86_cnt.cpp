// Microbench for P1/P2/P3/P4 primitives, x86 variants.
// See bench_primitives_neon_cnt.cpp for the spec of each.
//
// Key advantage on x86 (Zen3+, Intel Haswell+): BMI2 PDEP is a single
// instruction (~3 cycle latency) that does exactly what P2 needs at
// 64-bit granularity.
//
// On AMD pre-Zen3, PDEP is microcoded and very slow (~250 cyc) — this
// bench will expose that.  Test hosts: c6a (Zen 3) and c8i (Xeon).

#include "counters/bench.h"
#include <immintrin.h>
#ifdef __BMI2__
#include <x86intrin.h>
#endif
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

/* ---------- P1: scalar baseline ---------- */
static inline int p1_scalar(const uint8_t *V, int n, uint8_t C, uint16_t *out) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (V[i] == C) out[k++] = (uint16_t)i;
    }
    return k;
}

/* ---------- P1: SSE 16-byte chunk via _mm_movemask_epi8 ----------
 * For each 16 input bytes:
 *   1. _mm_cmpeq_epi8(data, vC) → 0xFF/0x00 per lane
 *   2. _mm_movemask_epi8 → 16-bit mask
 *   3. Split low 8 / high 8 bits → 2 compress_tab lookups
 *   4. Pack identity [j, j+8) and [j+8, j+16) via pshufb
 *   5. Store n_match*2 bytes per half */
static inline int p1_sse(const uint8_t *V, int n, uint8_t C, uint16_t *out) {
    __m128i vC = _mm_set1_epi8((char)C);
    /* Identity offsets [0..7] and [8..15] as uint16x8 packed in __m128i */
    int k = 0;
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        __m128i data = _mm_loadu_si128((const __m128i *)(V + j));
        __m128i eq   = _mm_cmpeq_epi8(data, vC);
        uint32_t mm  = (uint32_t)_mm_movemask_epi8(eq);
        uint8_t lo   = (uint8_t)(mm & 0xFF);
        uint8_t hi   = (uint8_t)((mm >> 8) & 0xFF);

        /* Identity uint16 indices [j..j+7], [j+8..j+15] */
        __m128i id_lo = _mm_setr_epi16((short)(j),     (short)(j + 1),
                                       (short)(j + 2), (short)(j + 3),
                                       (short)(j + 4), (short)(j + 5),
                                       (short)(j + 6), (short)(j + 7));
        __m128i id_hi = _mm_setr_epi16((short)(j + 8),  (short)(j + 9),
                                       (short)(j + 10), (short)(j + 11),
                                       (short)(j + 12), (short)(j + 13),
                                       (short)(j + 14), (short)(j + 15));

        __m128i shuf_lo = _mm_load_si128((const __m128i *)compress_tab[lo]);
        __m128i shuf_hi = _mm_load_si128((const __m128i *)compress_tab[hi]);
        __m128i out_lo  = _mm_shuffle_epi8(id_lo, shuf_lo);
        __m128i out_hi  = _mm_shuffle_epi8(id_hi, shuf_hi);
        _mm_storeu_si128((__m128i *)(out + k), out_lo);
        k += compress_popcnt[lo];
        _mm_storeu_si128((__m128i *)(out + k), out_hi);
        k += compress_popcnt[hi];
    }
    for (; j < n; j++) {
        if (V[j] == C) out[k++] = (uint16_t)j;
    }
    return k;
}

#ifdef __AVX512BW__
/* ---------- P1: AVX-512 32-byte chunk via mask compress ----------
 * 32 input bytes per iter, vpcmpeqb → __mmask32, then
 * _mm512_maskz_compress_epi16(mask, identity) → packed indices. */
static inline int p1_avx512(const uint8_t *V, int n, uint8_t C, uint16_t *out) {
    __m256i vC = _mm256_set1_epi8((char)C);
    int k = 0;
    int j = 0;
    /* Identity vector [0..31] as 32 u16 values in a zmm register. */
    __m512i id_base = _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24,
                                       23, 22, 21, 20, 19, 18, 17, 16,
                                       15, 14, 13, 12, 11, 10,  9,  8,
                                        7,  6,  5,  4,  3,  2,  1,  0);
    for (; j + 32 <= n; j += 32) {
        __m256i data = _mm256_loadu_si256((const __m256i *)(V + j));
        __mmask32 m  = _mm256_cmpeq_epi8_mask(data, vC);

        __m512i id = _mm512_add_epi16(id_base, _mm512_set1_epi16((short)j));
        __m512i compressed = _mm512_maskz_compress_epi16(m, id);

        int n_match = __builtin_popcount(m);
        _mm512_storeu_si512((__m512i *)(out + k), compressed);
        k += n_match;
    }
    for (; j < n; j++) {
        if (V[j] == C) out[k++] = (uint16_t)j;
    }
    return k;
}
#endif

/* ---------- P2: scalar baseline ---------- */
static inline void p2_scalar(const uint8_t *A, int n_bytes,
                              const uint8_t *B, uint8_t *C) {
    int b_pos = 0;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t a = A[i];
        uint8_t out = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (a & (1u << bit)) {
                int byte_idx = b_pos >> 3;
                int bit_idx  = b_pos & 7;
                uint8_t bbit = (B[byte_idx] >> bit_idx) & 1;
                out |= (uint8_t)(bbit << bit);
                b_pos++;
            }
        }
        C[i] = out;
    }
}

#ifdef __BMI2__
/* ---------- P2: BMI2 PDEP, 64-bit chunks ----------
 * Process 64 A-bits per iter via single _pdep_u64 instruction.
 * Need to slide B by popcount(A_chunk) each iter. */
static inline void p2_bmi2(const uint8_t *A, int n_bytes,
                            const uint8_t *B, uint8_t *C) {
    int b_pos = 0;
    int i = 0;
    /* Process 8 A bytes (64 bits) at a time. */
    for (; i + 8 <= n_bytes; i += 8) {
        uint64_t a_chunk;
        std::memcpy(&a_chunk, A + i, 8);
        int n_b = __builtin_popcountll(a_chunk);

        /* Load up to 64+8 bits of B starting at b_pos. */
        int byte_idx = b_pos >> 3;
        int bit_idx  = b_pos & 7;
        uint64_t b_chunk;
        std::memcpy(&b_chunk, B + byte_idx, 8);
        uint64_t b_chunk_hi = (bit_idx + n_b > 64)
            ? ((uint64_t)B[byte_idx + 8]) : 0;
        uint64_t b_bits;
        if (bit_idx == 0) {
            b_bits = b_chunk;
        } else {
            b_bits = (b_chunk >> bit_idx) | (b_chunk_hi << (64 - bit_idx));
        }

        uint64_t out = _pdep_u64(b_bits, a_chunk);
        std::memcpy(C + i, &out, 8);
        b_pos += n_b;
    }
    /* Scalar tail */
    for (; i < n_bytes; i++) {
        uint8_t a = A[i];
        uint8_t out = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (a & (1u << bit)) {
                int byte_idx = b_pos >> 3;
                int bit_idx  = b_pos & 7;
                uint8_t bbit = (B[byte_idx] >> bit_idx) & 1;
                out |= (uint8_t)(bbit << bit);
                b_pos++;
            }
        }
        C[i] = out;
    }
}
#endif

/* ---------- P3: scalar baseline (same as NEON version) ---------- */
static inline int p3_scalar(const uint8_t *A, int n_bytes,
                             const uint8_t *B, uint16_t *out) {
    int b_pos = 0;
    int k = 0;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t a = A[i];
        for (int bit = 0; bit < 8; bit++) {
            if (a & (1u << bit)) {
                int byte_idx = b_pos >> 3;
                int bit_idx  = b_pos & 7;
                uint8_t bbit = (B[byte_idx] >> bit_idx) & 1;
                if (bbit) out[k++] = (uint16_t)(i * 8 + bit);
                b_pos++;
            }
        }
    }
    return k;
}

#ifdef __BMI2__
/* ---------- P3: BMI2 PDEP + tzcnt-loop to extract bits ----------
 * 1. Compute C_chunk = PDEP(B, A_chunk).
 * 2. Iterate set bits of C_chunk via tzcnt+blsr, emitting indices. */
static inline int p3_bmi2(const uint8_t *A, int n_bytes,
                           const uint8_t *B, uint16_t *out) {
    int b_pos = 0;
    int k = 0;
    int i = 0;
    for (; i + 8 <= n_bytes; i += 8) {
        uint64_t a_chunk;
        std::memcpy(&a_chunk, A + i, 8);
        int n_b = __builtin_popcountll(a_chunk);

        int byte_idx = b_pos >> 3;
        int bit_idx  = b_pos & 7;
        uint64_t b_chunk;
        std::memcpy(&b_chunk, B + byte_idx, 8);
        uint64_t b_chunk_hi = (bit_idx + n_b > 64)
            ? ((uint64_t)B[byte_idx + 8]) : 0;
        uint64_t b_bits = (bit_idx == 0) ? b_chunk
            : ((b_chunk >> bit_idx) | (b_chunk_hi << (64 - bit_idx)));

        uint64_t c_chunk = _pdep_u64(b_bits, a_chunk);
        int base = i * 8;
        while (c_chunk) {
            int bit = __builtin_ctzll(c_chunk);
            out[k++] = (uint16_t)(base + bit);
            c_chunk &= c_chunk - 1;  /* clear lowest set bit */
        }
        b_pos += n_b;
    }
    for (; i < n_bytes; i++) {
        uint8_t a = A[i];
        for (int bit = 0; bit < 8; bit++) {
            if (a & (1u << bit)) {
                int byte_idx = b_pos >> 3;
                int bit_idx  = b_pos & 7;
                uint8_t bbit = (B[byte_idx] >> bit_idx) & 1;
                if (bbit) out[k++] = (uint16_t)(i * 8 + bit);
                b_pos++;
            }
        }
    }
    return k;
}
#endif

/* ---------- P4: scalar baseline ---------- */
static inline void p4_scalar(const uint8_t *A, int n_bytes,
                              const uint8_t *B, uint8_t *C) {
    int b_pos = 0;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t a = A[i];
        uint16_t out16 = 0;
        for (int bit = 0; bit < 8; bit++) {
            int abit = (a >> bit) & 1;
            out16 |= (uint16_t)(abit << (2 * bit));
            if (abit) {
                int byte_idx = b_pos >> 3;
                int bit_idx  = b_pos & 7;
                int bbit = (B[byte_idx] >> bit_idx) & 1;
                out16 |= (uint16_t)(bbit << (2 * bit + 1));
                b_pos++;
            }
        }
        C[2 * i    ] = (uint8_t)(out16 & 0xFF);
        C[2 * i + 1] = (uint8_t)(out16 >> 8);
    }
}

#ifdef __BMI2__
/* ---------- P4: BMI2 PDEP, 32-bit A chunks → 64-bit output chunks ----
 * Per chunk:
 *   a_doubled_even = pdep(a, 0x5555...)  // A at even positions in 64-bit
 *   a_at_odd       = a_doubled_even << 1
 *   b_deposited    = pdep(b, a_at_odd)
 *   out64          = a_doubled_even | b_deposited */
static inline void p4_bmi2(const uint8_t *A, int n_bytes,
                            const uint8_t *B, uint8_t *C) {
    int b_pos = 0;
    int i = 0;
    /* Process 4 A bytes (32 bits) at a time → 8 bytes (64 bits) output. */
    for (; i + 4 <= n_bytes; i += 4) {
        uint32_t a_chunk;
        std::memcpy(&a_chunk, A + i, 4);
        int n_b = __builtin_popcount(a_chunk);

        int byte_idx = b_pos >> 3;
        int bit_idx  = b_pos & 7;
        uint64_t b_chunk;
        std::memcpy(&b_chunk, B + byte_idx, 8);
        uint64_t b_bits = (b_chunk >> bit_idx) & ((n_b == 64) ? ~0ULL
                                                              : (1ULL << n_b) - 1);

        uint64_t a_even = _pdep_u64((uint64_t)a_chunk, 0x5555555555555555ULL);
        uint64_t a_odd  = a_even << 1;
        uint64_t b_dep  = _pdep_u64(b_bits, a_odd);
        uint64_t out64  = a_even | b_dep;
        std::memcpy(C + 2 * i, &out64, 8);
        b_pos += n_b;
    }
    /* Scalar tail */
    for (; i < n_bytes; i++) {
        uint8_t a = A[i];
        uint16_t out16 = 0;
        for (int bit = 0; bit < 8; bit++) {
            int abit = (a >> bit) & 1;
            out16 |= (uint16_t)(abit << (2 * bit));
            if (abit) {
                int byte_idx = b_pos >> 3;
                int bit_idx  = b_pos & 7;
                int bbit = (B[byte_idx] >> bit_idx) & 1;
                out16 |= (uint16_t)(bbit << (2 * bit + 1));
                b_pos++;
            }
        }
        C[2 * i    ] = (uint8_t)(out16 & 0xFF);
        C[2 * i + 1] = (uint8_t)(out16 >> 8);
    }
}
#endif

/* ---------- harness ---------- */

static void row(const char *name, const counters::event_aggregate &agg,
                bool have, long elems_per_iter) {
    double ns = agg.fastest_elapsed_ns();
    double cyc = agg.fastest_cycles();
    double ins = agg.fastest_instructions();
    double ipc = cyc > 0 ? ins / cyc : 0;
    double ghz = agg.cycles() / agg.elapsed_ns();
    double ns_per_elem = ns / elems_per_iter;
    if (have)
        std::printf("  %-26s %10.1f %9.1f %9.1f %6.2f %6.2f %12.4f\n",
                    name, ns, cyc, ins, ipc, ghz, ns_per_elem);
    else
        std::printf("  %-26s %10.1f                                       %12.4f\n",
                    name, ns, ns_per_elem);
}

/* compress_tab needed for SSE P1 — local copy to avoid linker entanglement. */
alignas(32) uint8_t compress_tab[256][32];
alignas(64) uint8_t compress_popcnt[256];
void init_compress_table(void) {
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0;
        for (int i = 0; i < 8; i++) if (mask & (1 << i)) {
            compress_tab[mask][out_r * 2]     = (uint8_t)(i * 2);
            compress_tab[mask][out_r * 2 + 1] = (uint8_t)(i * 2 + 1);
            out_r++;
        }
        compress_popcnt[mask] = (uint8_t)out_r;
        for (int j = out_r * 2; j < 16; j++) compress_tab[mask][j] = 0x80;
    }
}

int main(int argc, char **argv) {
    init_compress_table();

    int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int N_BYTES_A = N / 8;

    std::vector<uint8_t>  V(N + 64);
    std::vector<uint16_t> out_indices(N + 64);
    std::vector<uint8_t>  A(N_BYTES_A + 64);
    std::vector<uint8_t>  B(N_BYTES_A + 64);
    std::vector<uint8_t>  C(N_BYTES_A + 64);
    std::vector<uint8_t>  C4(N_BYTES_A * 2 + 64);

    std::mt19937 rng(0xCAFEBABE);
    for (auto &v : V) v = (uint8_t)rng();
    for (auto &a : A) a = (uint8_t)rng();
    for (auto &b : B) b = (uint8_t)rng();
    uint8_t C_target = 0x42;

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");

    counters::bench_parameter params;
    params.min_repeat = 30;
    params.min_time_ns = 200000000;

    const uint8_t *Vp = V.data();
    uint16_t *outp = out_indices.data();
    const uint8_t *Ap = A.data();
    const uint8_t *Bp = B.data();
    uint8_t *Cp = C.data();
    uint8_t *Cp4 = C4.data();

    /* Correctness checks */
    int k_sc = p1_scalar(Vp, N, C_target, outp);
    std::vector<uint16_t> tmp(N + 64);
    int k_sse = p1_sse(Vp, N, C_target, tmp.data());
    if (k_sc != k_sse || std::memcmp(outp, tmp.data(),
                                      (size_t)k_sc * 2) != 0) {
        std::fprintf(stderr, "P1 SSE FAIL: scalar=%d sse=%d\n", k_sc, k_sse);
        return 1;
    }
#ifdef __AVX512BW__
    int k_avx = p1_avx512(Vp, N, C_target, tmp.data());
    if (k_sc != k_avx || std::memcmp(outp, tmp.data(),
                                      (size_t)k_sc * 2) != 0) {
        std::fprintf(stderr, "P1 AVX-512 FAIL: scalar=%d avx=%d\n",
                     k_sc, k_avx);
        return 1;
    }
#endif

    std::vector<uint8_t> C_sc(N_BYTES_A + 64);
    std::vector<uint8_t> C_bm(N_BYTES_A + 64);
    p2_scalar(Ap, N_BYTES_A, Bp, C_sc.data());
#ifdef __BMI2__
    p2_bmi2(Ap, N_BYTES_A, Bp, C_bm.data());
    if (std::memcmp(C_sc.data(), C_bm.data(), N_BYTES_A) != 0) {
        std::fprintf(stderr, "P2 BMI2 FAIL\n");
        return 2;
    }
#endif

    std::vector<uint16_t> p3_sc(N + 64), p3_bm(N + 64);
    int p3k_sc = p3_scalar(Ap, N_BYTES_A, Bp, p3_sc.data());
#ifdef __BMI2__
    int p3k_bm = p3_bmi2  (Ap, N_BYTES_A, Bp, p3_bm.data());
    if (p3k_sc != p3k_bm || std::memcmp(p3_sc.data(), p3_bm.data(),
                                          (size_t)p3k_sc * 2) != 0) {
        std::fprintf(stderr, "P3 BMI2 FAIL: scalar=%d bmi2=%d\n",
                     p3k_sc, p3k_bm);
        return 3;
    }
#endif

    std::vector<uint8_t> C4_sc(N_BYTES_A * 2 + 64);
    std::vector<uint8_t> C4_bm(N_BYTES_A * 2 + 64);
    p4_scalar(Ap, N_BYTES_A, Bp, C4_sc.data());
#ifdef __BMI2__
    p4_bmi2  (Ap, N_BYTES_A, Bp, C4_bm.data());
    if (std::memcmp(C4_sc.data(), C4_bm.data(), N_BYTES_A * 2) != 0) {
        std::fprintf(stderr, "P4 BMI2 FAIL\n");
        return 4;
    }
#endif

    std::printf("\nN = %d input elements (P1) / %d bytes (P2/P4)\n",
                N, N_BYTES_A);
    std::printf("P1: %d matches (rate %.2f%%) ; P3: %d set bits\n",
                k_sc, 100.0 * k_sc / N, p3k_sc);
    if (have) {
        std::printf("\n  %-26s %10s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "ns/elem");
    } else {
        std::printf("\n  %-26s %10s %12s\n",
                    "variant", "ns/iter", "ns/elem");
    }

    std::printf("\n--- P1: select_vec ---\n");
    {
        auto agg = counters::bench(
            [Vp, N, C_target, outp]() {
            int k = p1_scalar(Vp, N, C_target, outp);
            ((volatile int *)outp)[0] = k + outp[0];
        }, params);
        row("p1_scalar", agg, have, N);
    }
    {
        auto agg = counters::bench(
            [Vp, N, C_target, outp]() {
            int k = p1_sse(Vp, N, C_target, outp);
            ((volatile int *)outp)[0] = k + outp[0];
        }, params);
        row("p1_sse (16-byte chunk)", agg, have, N);
    }
#ifdef __AVX512BW__
    {
        auto agg = counters::bench(
            [Vp, N, C_target, outp]() {
            int k = p1_avx512(Vp, N, C_target, outp);
            ((volatile int *)outp)[0] = k + outp[0];
        }, params);
        row("p1_avx512 (32-byte chunk)", agg, have, N);
    }
#endif

    std::printf("\n--- P2: pdep_bm ---\n");
    {
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, Cp]() {
            p2_scalar(Ap, N_BYTES_A, Bp, Cp);
            ((volatile uint8_t *)Cp)[0] = Cp[0];
        }, params);
        row("p2_scalar (per-bit)", agg, have, N);
    }
#ifdef __BMI2__
    {
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, Cp]() {
            p2_bmi2(Ap, N_BYTES_A, Bp, Cp);
            ((volatile uint8_t *)Cp)[0] = Cp[0];
        }, params);
        row("p2_bmi2 (PDEP 64-bit)", agg, have, N);
    }
#endif

    std::printf("\n--- P3: pdep_idx ---\n");
    {
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, outp]() {
            int k = p3_scalar(Ap, N_BYTES_A, Bp, outp);
            ((volatile int *)outp)[0] = k + outp[0];
        }, params);
        row("p3_scalar", agg, have, N);
    }
#ifdef __BMI2__
    {
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, outp]() {
            int k = p3_bmi2(Ap, N_BYTES_A, Bp, outp);
            ((volatile int *)outp)[0] = k + outp[0];
        }, params);
        row("p3_bmi2 (PDEP + tzcnt)", agg, have, N);
    }
#endif

    std::printf("\n--- P4: interleave (2 bits per A bit) ---\n");
    {
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, Cp4]() {
            p4_scalar(Ap, N_BYTES_A, Bp, Cp4);
            ((volatile uint8_t *)Cp4)[0] = Cp4[0];
        }, params);
        row("p4_scalar", agg, have, N);
    }
#ifdef __BMI2__
    {
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, Cp4]() {
            p4_bmi2(Ap, N_BYTES_A, Bp, Cp4);
            ((volatile uint8_t *)Cp4)[0] = Cp4[0];
        }, params);
        row("p4_bmi2 (2x PDEP)", agg, have, N);
    }
#endif

    return 0;
}
