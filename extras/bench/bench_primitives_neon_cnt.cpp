// Microbench for two new primitives (NEON):
//
//   P1 (select_vec): given V[N], C → produce uint16_t indices[K] of all
//                    positions where V[i] == C.  K = popcount of the
//                    match mask.
//
//   P2 (pdep_bm):    given bitmap A (N bits) and bitmap B (popcount(A)
//                    bits) → produce bitmap C (N bits) where C has B's
//                    bits deposited into the 1-positions of A.
//                    LSB-first inside each byte.
//
//   P3 (pdep_idx):   same inputs as P2, but the output is a uint16_t
//                    list of indices where the (notional) C[i] == 1.
//                    Equivalent to P1(C, 1) at bit-level, fused.
//
//   P4 (interleave): A is N bits, B is popcount(A) bits.  Output C is
//                    2N bits, encoded by replacing each A[i]=0 with "00"
//                    and each A[i]=1 with "1<B[k]>" where k is A's
//                    cumulative set-bit count at i.  LSB-first.
//                      i.e. C[2i]   = A[i]
//                           C[2i+1] = A[i] ? B[rank(A,i)] : 0
//
// Measures ns/iter and ns/INPUT-elem for several variants:
//   P1:
//     - scalar         (loop, branch on match)
//     - neon_v1        (vceqq_u8 + 8-elem chunk, reusing compress_tab[256])
//   P2:
//     - scalar_byte    (loop, scalar pdep_byte per A byte)
//     - neon_table     (per-A-byte spreader table + AND masking)
//
// Sizes: N=4096 input elements (block-ish).  All buffers L1-resident.

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

/* ---------- P1: scalar baseline ---------- */
static inline int p1_scalar(const uint8_t *V, int n, uint8_t C, uint16_t *out) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (V[i] == C) out[k++] = (uint16_t)i;
    }
    return k;
}

/* ---------- P1: NEON, 8-elem chunk reusing compress_tab[256] ----------
 * For each 8 input bytes:
 *   1. Compare equal to C → 8 lane mask (0xFF per match)
 *   2. Pack 8 lane mask to 8-bit value (one bit per lane)
 *   3. Look up compress_tab[mask] — pre-computed shuffle that packs
 *      matching uint16 indices to the front
 *   4. Generate identity vector [j, j+1, ..., j+7] and apply shuffle
 *   5. Store n_match*2 bytes, advance by n_match */
static inline int p1_neon_v1(const uint8_t *V, int n, uint8_t C, uint16_t *out) {
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    uint16x8_t voff = vld1q_u16(off);
    uint8x8_t  vC   = vdup_n_u8(C);
    /* Bit-position weights for packing 8-lane mask → 8-bit value */
    static const uint8_t bitpos[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbits = vld1_u8(bitpos);

    int k = 0;
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        uint8x8_t data = vld1_u8(V + j);
        uint8x8_t eq   = vceq_u8(data, vC);            /* 0xFF per match */
        /* Pack lane mask to byte via AND with bit weights + horizontal add */
        uint8x8_t bitmasked = vand_u8(eq, vbits);
        uint8_t   bm = vaddv_u8(bitmasked);            /* horizontal sum = mask byte */

        /* Compress identity indices via shuffle table */
        uint16x8_t base = vdupq_n_u16((uint16_t)j);
        uint8x16_t data16 = vreinterpretq_u8_u16(vaddq_u16(base, voff));
        uint8x16_t shuf = vld1q_u8(compress_tab[bm]);
        uint8x16_t compressed = vqtbl1q_u8(data16, shuf);
        vst1q_u8((uint8_t *)(out + k), compressed);

        k += compress_popcnt[bm];
    }
    /* Tail: scalar */
    for (; j < n; j++) {
        if (V[j] == C) out[k++] = (uint16_t)j;
    }
    return k;
}

/* ---------- P2: scalar baseline ---------- */
static inline void p2_scalar(const uint8_t *A, int n_bytes,
                              const uint8_t *B, uint8_t *C) {
    int b_pos = 0;   /* bit position within B */
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

/* ---------- P2: NEON, per-A-byte spreader table approach ----------
 *
 * Idea: for each 8-bit value `a`, precompute a 64-bit `spreader[a]` such
 * that if you take `b` (low popcnt(a) bits valid) and compute
 *   ((uint64_t)b * spreader[a]) & magic_mask[a]
 * ...you get a 64-bit value whose bits, when re-packed, equal pdep(b, a).
 *
 * That's the classic "magic multiplication PDEP emulation" but it's
 * tricky.  Simpler approach: scalar pdep_byte via 256-entry lookup
 * tables.
 *
 * We use TWO tables of size 256×256 = 64KB each:
 *   - We instead use a small (256×8) "bit-position map" table:
 *     bit_pos[a][k] = bit index in A where the k-th set bit lives.
 *   - Reading B's bits in order and OR-ing 1 << bit_pos[a][k] gives C.
 *
 * The straightforward scalar version with that table is what we'll
 * test as "neon" (despite being scalar — there's no bit-PDEP on NEON).
 *
 * Per-byte cost:
 *   - 1 load of A
 *   - popcnt(a) lookups + ORs
 *   - 1 store of C
 *   - 1 advance of b_pos */

static uint8_t pdep_bit_pos[256][8];
static uint8_t pdep_popcnt[256];

static void init_pdep_tables(void) {
    for (int a = 0; a < 256; a++) {
        int k = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (a & (1 << bit)) {
                pdep_bit_pos[a][k++] = (uint8_t)bit;
            }
        }
        pdep_popcnt[a] = (uint8_t)k;
        /* Pad unused slots with bit 0 (won't be touched because k < 8) */
        for (; k < 8; k++) pdep_bit_pos[a][k] = 0;
    }
}

static inline void p2_neon_table(const uint8_t *A, int n_bytes,
                                  const uint8_t *B, uint8_t *C) {
    int b_pos = 0;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t a = A[i];
        int n = pdep_popcnt[a];
        uint8_t out = 0;
        /* Read up to 8 bits from B starting at b_pos. */
        int byte_idx = b_pos >> 3;
        int bit_idx  = b_pos & 7;
        uint32_t b_window = (uint32_t)B[byte_idx];
        if (bit_idx + n > 8) b_window |= ((uint32_t)B[byte_idx + 1]) << 8;
        uint32_t b_bits = (b_window >> bit_idx) & ((1u << n) - 1);
        /* Spread b_bits into output via bit_pos table. */
        for (int k = 0; k < n; k++) {
            uint8_t bb = (uint8_t)((b_bits >> k) & 1);
            out |= (uint8_t)(bb << pdep_bit_pos[a][k]);
        }
        C[i] = out;
        b_pos += n;
    }
}

/* ---------- P3: scalar baseline ---------- */
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

/* ---------- P3: NEON, per-A-byte AND + compress ----------
 *
 * For each byte of A:
 *   1. Compute output_byte = pdep(B_window, a)
 *   2. compress_tab[output_byte] gives shuffle for the positions where
 *      output_byte is 1
 *   3. Apply to identity [base, base+1, ..., base+7] uint16 indices,
 *      where base = i * 8
 *   4. Store n = popcnt(output_byte) uint16 indices */
static inline int p3_neon(const uint8_t *A, int n_bytes,
                           const uint8_t *B, uint16_t *out) {
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    uint16x8_t voff = vld1q_u16(off);

    int b_pos = 0;
    int k = 0;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t a = A[i];
        int n_b = pdep_popcnt[a];

        /* Read n_b bits from B starting at b_pos. */
        int byte_idx = b_pos >> 3;
        int bit_idx  = b_pos & 7;
        uint32_t b_window = (uint32_t)B[byte_idx];
        if (bit_idx + n_b > 8) b_window |= ((uint32_t)B[byte_idx + 1]) << 8;
        uint32_t b_bits = (b_window >> bit_idx) & ((1u << n_b) - 1);
        b_pos += n_b;

        /* Spread b_bits → output byte */
        uint8_t out_byte = 0;
        for (int j = 0; j < n_b; j++) {
            uint8_t bb = (uint8_t)((b_bits >> j) & 1);
            out_byte |= (uint8_t)(bb << pdep_bit_pos[a][j]);
        }

        /* Compress identity [i*8, i*8+1, ..., i*8+7] via shuffle table */
        uint16_t base = (uint16_t)(i * 8);
        uint16x8_t base_v = vdupq_n_u16(base);
        uint8x16_t data16 = vreinterpretq_u8_u16(vaddq_u16(base_v, voff));
        uint8x16_t shuf = vld1q_u8(compress_tab[out_byte]);
        uint8x16_t compressed = vqtbl1q_u8(data16, shuf);
        vst1q_u8((uint8_t *)(out + k), compressed);

        k += compress_popcnt[out_byte];
    }
    return k;
}

/* ---------- P4: scalar baseline ----------
 * Per A bit: emit A_bit at output position 2*i, emit (A_bit ? B[bpos] : 0)
 * at output position 2*i+1.  Output has 2N bits total. */
static inline void p4_scalar(const uint8_t *A, int n_bytes,
                              const uint8_t *B, uint8_t *C) {
    int b_pos = 0;
    /* Each A byte (8 bits) expands to 2 output bytes (16 bits).
     * Output bit 2i = A[i]; output bit 2i+1 = A[i] ? B[k++] : 0. */
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

/* ---------- P4: NEON, per-A-byte table ----------
 * Per byte of A:
 *   - doubled_A[a] = uint16 with A's bits at positions 0,2,4,...,14 (the even bits)
 *   - For the (popcnt(a)) B-bits, deposit them at positions 2k+1 where
 *     A's k-th set bit lives.  Reuse pdep_bit_pos[a][k] which gives bit
 *     index in A (0..7), so the output position is 2*pdep_bit_pos[a][k]+1. */
static uint16_t doubled_A_table[256];

static void init_p4_tables(void) {
    for (int a = 0; a < 256; a++) {
        uint16_t out = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (a & (1 << bit)) out |= (uint16_t)(1u << (2 * bit));
        }
        doubled_A_table[a] = out;
    }
}

static inline void p4_neon_table(const uint8_t *A, int n_bytes,
                                  const uint8_t *B, uint8_t *C) {
    int b_pos = 0;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t a = A[i];
        int n_b = pdep_popcnt[a];

        int byte_idx = b_pos >> 3;
        int bit_idx  = b_pos & 7;
        uint32_t b_window = (uint32_t)B[byte_idx];
        if (bit_idx + n_b > 8) b_window |= ((uint32_t)B[byte_idx + 1]) << 8;
        uint32_t b_bits = (b_window >> bit_idx) & ((1u << n_b) - 1);
        b_pos += n_b;

        uint16_t out16 = doubled_A_table[a];
        /* Deposit B bits at odd positions (2k+1 in output). */
        for (int j = 0; j < n_b; j++) {
            uint16_t bb = (uint16_t)((b_bits >> j) & 1);
            out16 |= (uint16_t)(bb << (2 * pdep_bit_pos[a][j] + 1));
        }
        C[2 * i    ] = (uint8_t)(out16 & 0xFF);
        C[2 * i + 1] = (uint8_t)(out16 >> 8);
    }
}

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

int main(int argc, char **argv) {
    init_compress_table();
    init_pdep_tables();
    init_p4_tables();

    /* Input size — chars for P1, bits for P2.  Use the same N as our
     * Huffman block size so cache footprint matches real-decoder
     * scenarios. */
    int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int N_BYTES_A = N / 8;            /* P2 A is bitmap of N bits */

    /* P1 buffers */
    std::vector<uint8_t>  V(N + 64);
    std::vector<uint16_t> out_indices(N + 64);

    /* P2 buffers */
    std::vector<uint8_t>  A(N_BYTES_A + 64);
    std::vector<uint8_t>  B(N_BYTES_A + 64);       /* up to N bits, ≥ popcount */
    std::vector<uint8_t>  C(N_BYTES_A + 64);

    /* P4 buffers - output is 2N bits = N_BYTES_A * 2 bytes */
    std::vector<uint8_t>  C4(N_BYTES_A * 2 + 64);

    std::mt19937 rng(0xCAFEBABE);
    /* P1 V: random bytes; we'll search for byte=0x42 (~1/256 hit rate
     * → ~16 matches in N=4096) */
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

    /* Correctness check for P1 */
    int k_scalar = p1_scalar(Vp, N, C_target, outp);
    std::vector<uint16_t> tmp(N + 64);
    int k_neon = p1_neon_v1(Vp, N, C_target, tmp.data());
    if (k_scalar != k_neon || std::memcmp(outp, tmp.data(),
                                          (size_t)k_scalar * 2) != 0) {
        std::fprintf(stderr, "P1 correctness FAIL: scalar=%d neon=%d\n",
                     k_scalar, k_neon);
        return 1;
    }

    /* Correctness check for P2 */
    std::vector<uint8_t> C_scalar(N_BYTES_A + 64);
    std::vector<uint8_t> C_neon(N_BYTES_A + 64);
    p2_scalar(Ap, N_BYTES_A, Bp, C_scalar.data());
    p2_neon_table(Ap, N_BYTES_A, Bp, C_neon.data());
    if (std::memcmp(C_scalar.data(), C_neon.data(), N_BYTES_A) != 0) {
        std::fprintf(stderr, "P2 correctness FAIL\n");
        return 2;
    }

    /* Correctness check for P3 */
    std::vector<uint16_t> p3_scalar_out(N + 64);
    std::vector<uint16_t> p3_neon_out(N + 64);
    int p3_k_scalar = p3_scalar(Ap, N_BYTES_A, Bp, p3_scalar_out.data());
    int p3_k_neon   = p3_neon  (Ap, N_BYTES_A, Bp, p3_neon_out.data());
    if (p3_k_scalar != p3_k_neon
        || std::memcmp(p3_scalar_out.data(), p3_neon_out.data(),
                       (size_t)p3_k_scalar * 2) != 0) {
        std::fprintf(stderr, "P3 correctness FAIL: scalar=%d neon=%d\n",
                     p3_k_scalar, p3_k_neon);
        return 3;
    }

    /* Correctness check for P4 */
    std::vector<uint8_t> C4_scalar(N_BYTES_A * 2 + 64);
    std::vector<uint8_t> C4_neon  (N_BYTES_A * 2 + 64);
    p4_scalar(Ap, N_BYTES_A, Bp, C4_scalar.data());
    p4_neon_table(Ap, N_BYTES_A, Bp, C4_neon.data());
    if (std::memcmp(C4_scalar.data(), C4_neon.data(), N_BYTES_A * 2) != 0) {
        std::fprintf(stderr, "P4 correctness FAIL\n");
        return 4;
    }

    std::printf("\nN = %d input elements (P1) / %d bytes (P2)\n",
                N, N_BYTES_A);
    std::printf("P1: %d matches found (rate %.2f%%)\n",
                k_scalar, 100.0 * k_scalar / N);
    if (have) {
        std::printf("\n  %-26s %10s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "ns/elem");
    } else {
        std::printf("\n  %-26s %10s %12s\n",
                    "variant", "ns/iter", "ns/elem");
    }

    std::printf("\n--- P1: select_vec (find positions where V[i]==C) ---\n");
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [Vp, N, C_target, outp, &sink]() {
            int k = p1_scalar(Vp, N, C_target, outp);
            sink = k + outp[0];
        }, params);
        row("p1_scalar", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [Vp, N, C_target, outp, &sink]() {
            int k = p1_neon_v1(Vp, N, C_target, outp);
            sink = k + outp[0];
        }, params);
        row("p1_neon_v1", agg, have, N);
    }

    std::printf("\n--- P3: pdep_idx (indices where C[i]==1) ---  %d set bits\n",
                p3_k_scalar);
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, outp, &sink]() {
            int k = p3_scalar(Ap, N_BYTES_A, Bp, outp);
            sink = k + outp[0];
        }, params);
        row("p3_scalar", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, outp, &sink]() {
            int k = p3_neon(Ap, N_BYTES_A, Bp, outp);
            sink = k + outp[0];
        }, params);
        row("p3_neon", agg, have, N);
    }

    std::printf("\n--- P4: interleave (each A bit → 2 output bits) ---\n");
    {
        volatile int sink = 0;
        uint8_t *Cp4 = C4.data();
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, Cp4, &sink]() {
            p4_scalar(Ap, N_BYTES_A, Bp, Cp4);
            sink = Cp4[0];
        }, params);
        row("p4_scalar", agg, have, N);
    }
    {
        volatile int sink = 0;
        uint8_t *Cp4 = C4.data();
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, Cp4, &sink]() {
            p4_neon_table(Ap, N_BYTES_A, Bp, Cp4);
            sink = Cp4[0];
        }, params);
        row("p4_neon_table", agg, have, N);
    }

    std::printf("\n--- P2: pdep_bm (deposit B's bits into 1-positions of A) ---\n");
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, Cp, &sink]() {
            p2_scalar(Ap, N_BYTES_A, Bp, Cp);
            sink = Cp[0];
        }, params);
        row("p2_scalar (per-bit)", agg, have, N);
    }
    {
        volatile int sink = 0;
        auto agg = counters::bench(
            [Ap, N_BYTES_A, Bp, Cp, &sink]() {
            p2_neon_table(Ap, N_BYTES_A, Bp, Cp);
            sink = Cp[0];
        }, params);
        row("p2_neon_table (per-byte)", agg, have, N);
    }

    return 0;
}
