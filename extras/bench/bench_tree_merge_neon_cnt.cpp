// Microbench for tree_merge(bmap, left_chars, right_chars) — the
// fundamental primitive for a bottom-up PIVCO decoder.
//
// Spec:
//   Input:  bitmap A of N bits, two byte arrays left (n_left bytes) and
//           right (n_right bytes), where n_left = popcount(~A) and
//           n_right = popcount(A).
//   Output: byte array C of N bytes where C[k] = (A[k] ? right[rank1(A,k)]
//                                                       : left[rank0(A,k)]).
//
// Per 8-bit mask chunk:
//   - load 8 left bytes and 8 right bytes (over-read tolerated)
//   - vcombine into 16-byte vector [L0..L7, R0..R7]
//   - look up shuffle pattern in expand_tab[mask]  (256x8 byte table = 2KB)
//   - one vqtbl1_u8 produces 8 output bytes
//   - advance left_cur by (8-popcount(mask)), right_cur by popcount(mask)
//
// Variants benched:
//   scalar          : reference impl, per-bit loop
//   neon            : 8-byte chunks, single TBL per chunk
//   neon_x2         : 2-byte chunks unrolled (better ILP)
//   broadcast_left  : left side is a single sym broadcast register
//                     (models the constant-left-leaf case — no left buffer)

#include "counters/bench.h"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

/* 256-entry shuffle patterns for tree_merge on 8-bit mask chunks.
 *
 * expand_tab[mask][k] = 0..15:
 *   - 0..7   means "take from left[count_zeros_before_k]"
 *   - 8..15  means "take from right[8 + count_ones_before_k - 8]"
 *
 * Used as the TBL pattern over a 16-byte vector built from
 * vcombine(left_8bytes, right_8bytes). */
alignas(32) static uint8_t expand_tab[256][8];
alignas(64) static uint8_t expand_popcnt[256];  /* count of 1 bits (right side) */

static void init_expand_table(void) {
    for (int m = 0; m < 256; m++) {
        int n_zeros = 0, n_ones = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) {
                expand_tab[m][k] = (uint8_t)(8 + n_ones);
                n_ones++;
            } else {
                expand_tab[m][k] = (uint8_t)n_zeros;
                n_zeros++;
            }
        }
        expand_popcnt[m] = (uint8_t)n_ones;
    }
}

/* ---------- scalar reference ---------- */
static inline void merge_scalar(const uint8_t *bm, int n,
                                 const uint8_t *left,
                                 const uint8_t *right,
                                 uint8_t *out) {
    int lc = 0, rc = 0;
    for (int j = 0; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}

/* ---------- neon: 8-byte chunk per iter ---------- */
static inline void merge_neon(const uint8_t *bm, int n,
                               const uint8_t *left,
                               const uint8_t *right,
                               uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        uint8_t m  = bm[j >> 3];
        uint8x8_t  L   = vld1_u8(left + lc);
        uint8x8_t  R   = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(L, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        int nr = expand_popcnt[m];
        rc += nr;
        lc += (8 - nr);
    }
    /* scalar tail */
    for (; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}

/* ---------- neon_x2: two 8-byte chunks unrolled per iter ---------- */
static inline void merge_neon_x2(const uint8_t *bm, int n,
                                  const uint8_t *left,
                                  const uint8_t *right,
                                  uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint8_t m0 = bm[j >> 3];
        uint8_t m1 = bm[(j >> 3) + 1];

        uint8x8_t  L0   = vld1_u8(left + lc);
        uint8x8_t  R0   = vld1_u8(right + rc);
        uint8x16_t both0 = vcombine_u8(L0, R0);
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];
        rc += nr0;
        lc += (8 - nr0);

        uint8x8_t  L1   = vld1_u8(left + lc);
        uint8x8_t  R1   = vld1_u8(right + rc);
        uint8x16_t both1 = vcombine_u8(L1, R1);
        uint8x8_t  shuf1 = vld1_u8(expand_tab[m1]);
        uint8x8_t  o1    = vqtbl1_u8(both1, shuf1);
        vst1_u8(out + j + 8, o1);
        int nr1 = expand_popcnt[m1];
        rc += nr1;
        lc += (8 - nr1);
    }
    for (; j + 8 <= n; j += 8) {
        uint8_t m  = bm[j >> 3];
        uint8x8_t  L   = vld1_u8(left + lc);
        uint8x8_t  R   = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(L, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        int nr = expand_popcnt[m];
        rc += nr;
        lc += (8 - nr);
    }
    for (; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}

/* ---------- neon broadcast-left: left side is a constant symbol ----
 * Models the merge-with-leaf-child case where one child is the
 * left-leaf symbol and we don't materialize a left buffer at all. */
static inline void merge_neon_broadcast_left(const uint8_t *bm, int n,
                                              uint8_t left_sym,
                                              const uint8_t *right,
                                              uint8_t *out) {
    int rc = 0;
    int j = 0;
    uint8x8_t Lbcast = vdup_n_u8(left_sym);
    for (; j + 8 <= n; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x8_t  R    = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(Lbcast, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        rc += expand_popcnt[m];
    }
    for (; j < n; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left_sym;
    }
}

/* ---------- harness ---------- */
static void row(const char *name, const counters::event_aggregate &agg,
                bool have, long elems_per_iter) {
    double ns  = agg.fastest_elapsed_ns();
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
    init_expand_table();

    /* N output bytes per iter.  Default matches typical PIVCO_BLOCK_SIZE. */
    int N = (argc > 1) ? std::atoi(argv[1]) : 4096;

    std::vector<uint8_t> bm((N + 7) / 8 + 16);
    std::vector<uint8_t> left(N + 64);
    std::vector<uint8_t> right(N + 64);
    std::vector<uint8_t> output(N + 64);
    std::vector<uint8_t> ref(N + 64);

    std::mt19937 rng(0xCAFEBABE);
    /* Random bitmap: ~50% ones, ~N/2 left and ~N/2 right per merge.
     * That's the worst-case bandwidth scenario where neither side is
     * heavily skewed. */
    for (auto &b : bm)    b = (uint8_t)rng();
    for (auto &l : left)  l = (uint8_t)rng();
    for (auto &r : right) r = (uint8_t)rng();

    /* Correctness checks */
    merge_scalar(bm.data(), N, left.data(), right.data(), ref.data());

    merge_neon(bm.data(), N, left.data(), right.data(), output.data());
    if (std::memcmp(output.data(), ref.data(), N) != 0) {
        std::fprintf(stderr, "merge_neon: MISMATCH\n");
        return 1;
    }
    merge_neon_x2(bm.data(), N, left.data(), right.data(), output.data());
    if (std::memcmp(output.data(), ref.data(), N) != 0) {
        std::fprintf(stderr, "merge_neon_x2: MISMATCH\n");
        return 2;
    }
    /* Broadcast variant: use left_sym=0xAA, right buffer as before.
     * Reference for it: scalar with left being a virtual all-0xAA stream. */
    {
        std::vector<uint8_t> virt_left(N + 64, 0xAA);
        std::vector<uint8_t> ref_bcast(N + 64);
        merge_scalar(bm.data(), N, virt_left.data(), right.data(), ref_bcast.data());
        merge_neon_broadcast_left(bm.data(), N, 0xAA, right.data(), output.data());
        if (std::memcmp(output.data(), ref_bcast.data(), N) != 0) {
            std::fprintf(stderr, "merge_neon_broadcast_left: MISMATCH\n");
            return 3;
        }
    }

    bool have = counters::has_performance_counters();
    if (!have) std::printf("[hw counters unavailable: rerun under sudo]\n");

    counters::bench_parameter params;
    params.min_repeat = 30;
    params.min_time_ns = 200000000;

    const uint8_t *bmp = bm.data();
    const uint8_t *Lp  = left.data();
    const uint8_t *Rp  = right.data();
    uint8_t       *Op  = output.data();

    std::printf("\ntree_merge microbench (N=%d output bytes per iter)\n", N);
    std::printf("Random bitmap (~50%% ones).  Left+right buffers are dense input arrays.\n");
    if (have) {
        std::printf("\n  %-26s %10s %9s %9s %6s %6s %12s\n",
                    "variant", "ns/iter", "cyc/iter", "ins/iter",
                    "IPC", "GHz", "ns/elem");
    } else {
        std::printf("\n  %-26s %10s %12s\n",
                    "variant", "ns/iter", "ns/elem");
    }

    {
        auto agg = counters::bench(
            [bmp, N, Lp, Rp, Op]() {
            merge_scalar(bmp, N, Lp, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_scalar", agg, have, N);
    }
    {
        auto agg = counters::bench(
            [bmp, N, Lp, Rp, Op]() {
            merge_neon(bmp, N, Lp, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_neon (8B chunk)", agg, have, N);
    }
    {
        auto agg = counters::bench(
            [bmp, N, Lp, Rp, Op]() {
            merge_neon_x2(bmp, N, Lp, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_neon_x2 (16B chunk)", agg, have, N);
    }
    {
        auto agg = counters::bench(
            [bmp, N, Rp, Op]() {
            merge_neon_broadcast_left(bmp, N, 0xAA, Rp, Op);
            ((volatile uint8_t *)Op)[0] = Op[0];
        }, params);
        row("merge_neon_broadcast_left", agg, have, N);
    }

    return 0;
}
