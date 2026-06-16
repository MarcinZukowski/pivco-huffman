/* bench_merge_neon.c -- NEON merge_vec_vec variants, isolating the
 * prefix-sum cursor-decoupling idea.
 *
 * Background: the production V4 merge (`old`) does 16 codes/iter as
 * vqtbl1 + vqtbl2, where iter-1's shuffle is a load from
 * expand_tab_pre[popcnt(m0)][m1] -- a TABLE LOAD whose address depends
 * on the previous chunk's popcount.  That cursor-dependent load is the
 * loop-carried bottleneck.  Jeff Plaisance (@jeffplaisance) pointed out
 * it can be broken: popcount all the bitmap bytes up front (vcnt), turn
 * them into a byte-wise prefix sum with the classic `* 0x0101010101010101`
 * multiply, and precompute every chunk's left/right cursor -- so the
 * chunks become independent and the OoO core overlaps them.
 *
 * Variants benchmarked here (all verified against a scalar reference):
 *   old     - V4 production: stride-16, vqtbl1 + vqtbl2(expand_tab_pre)
 *   new     - 128 codes/iter, 16 independent 8-code vqtbl1 chunks
 *             (Jeff's original pseudocode shape: separate L/R vld per chunk)
 *   com     - 32 codes/iter, 2 chunks of vqtbl1+vqtbl2
 *   com64   - 64 codes/iter, 4 chunks of vqtbl1+vqtbl2  *** the shipped one
 *   com128  - 128 codes/iter, 8 chunks of vqtbl1+vqtbl2
 *   jeff    - Jeff's PR #4 prefix128 verbatim (vpaddl fold + u16 SWAR
 *             prefix, half-level cursors).  Credit: Jeff Plaisance.
 *   jeff64  - Jeff's block style narrowed to 64-code stride
 *
 * Finding: com64 dominates the cross-uarch / cross-compiler matrix.  The
 * 128-wide forms (com128, jeff) extract more parallelism but stress NEON
 * register allocation and add a cross-half cursor recombine that costs
 * 30-57% on Graviton V1/V2/V3; com64 stays within the register file and
 * wins everywhere.  Per-elem cost (cyc/elem via perf_event_open on
 * Graviton, ns/elem via CLOCK_MONOTONIC on Apple), old -> com64:
 *   M4 (Apple)          0.043 -> 0.032 ns   (-24%)
 *   c7g (Neoverse V1)   0.266 -> 0.233 cyc  (-13%)
 *   c8g (Neoverse V2)   0.278 -> 0.215 cyc  (-23%)
 *   m9g (Neoverse V3)   0.235 -> 0.178 cyc  (-24%)
 *
 * Linux: CPU_CYCLES via perf_event_open.  macOS: CLOCK_MONOTONIC ns
 * (with QoS boost + warmup spin to land on a P-core at full clock).
 *
 * Build:
 *   M4: clang -O3 -arch arm64 -o bm extras/bench/bench_merge_neon.c
 *   Graviton: clang-20 -O3 -march=native -o bm extras/bench/bench_merge_neon.c
 */
#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

#if defined(__linux__)
#include <errno.h>
#include <unistd.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sched.h>
#endif
#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
/* Newer SDKs hide this prototype behind a Darwin-internal header. */
extern int pthread_set_qos_class_self_np(qos_class_t, int);
#endif

/* ============ thread pinning / qos: get on the P-core, max frequency.
 * Without this the macOS scheduler will run a short-lived process on
 * an E-core and at a derated frequency; first ~100ms of timing show
 * ~2x worse numbers than the steady-state P-core measurement. */
static void boost_thread(void) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#elif defined(__linux__)
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    (void)sched_setaffinity(0, sizeof set, &set);
#endif
}

/* Spin for ~200ms doing pointless vector work to ramp the clock and
 * land on a P-core BEFORE any of the variants get measured. */
static volatile uint64_t g_warmup_sink;
static void warmup_spin(void) {
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    uint8x16_t v = vdupq_n_u8(0x5A);
    uint64_t total = 0;
    do {
        for (int k = 0; k < 100000; k++) {
            v = vaddq_u8(v, vdupq_n_u8(1));
            v = veorq_u8(v, vdupq_n_u8(0x35));
        }
        total += vgetq_lane_u64(vreinterpretq_u64_u8(v), 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
    } while ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec) < 200e6);
    g_warmup_sink ^= total;
}

/* ============ expand_tab / expand_popcnt / expand_tab_pre */
/* Convention: bm bit==1 -> take right.
 * vcombine(L_low, R_low) lanes: [0..7] L, [8..15] R.
 * expand_tab[mask] gives uint8x8 shuffle: lane k = 8 + popcnt(mask[0..k-1])
 *   if bit k is 1, else popcnt of zeros in mask[0..k-1]. */
static uint8_t expand_tab[256][8];
static uint8_t expand_popcnt[256];
/* expand_tab_pre[nr0][m1]: shuf for vqtbl2 over (L_full, R_full) (32 bytes)
 * for the SECOND 8-lane chunk after the first chunk consumed nr0 right /
 * 8-nr0 left.  Lane k in (L_full, R_full) = 0..15 for L_full, 16..31 for R_full. */
static uint8_t expand_tab_pre[8 + 1][256][8];

static void build_tables(void) {
    for (int m = 0; m < 256; m++) {
        int r = 0, l = 0;
        for (int k = 0; k < 8; k++) {
            if ((m >> k) & 1) { expand_tab[m][k] = (uint8_t)(8 + r); r++; }
            else              { expand_tab[m][k] = (uint8_t)l;       l++; }
        }
        expand_popcnt[m] = (uint8_t)__builtin_popcount(m);
    }
    for (int nr0 = 0; nr0 <= 8; nr0++) {
        for (int m = 0; m < 256; m++) {
            int r = 0, l = 0;
            for (int k = 0; k < 8; k++) {
                if ((m >> k) & 1) {
                    /* take right: vqtbl2 over (L_full, R_full).  R_full starts
                     * at lane 16.  Right index after iter 0 consumed nr0 = nr0+r. */
                    expand_tab_pre[nr0][m][k] = (uint8_t)(16 + nr0 + r);
                    r++;
                } else {
                    /* take left: L_full lanes 0..15.  Left index after iter 0
                     * consumed (8-nr0) = (8-nr0) + l. */
                    expand_tab_pre[nr0][m][k] = (uint8_t)((8 - nr0) + l);
                    l++;
                }
            }
        }
    }
}

/* ============ OLD: production merge_vec_vec stride-16 (extracted) */
__attribute__((always_inline)) static inline void old_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right,
                      uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x16_t L_full = vld1q_u8(left  + lc);
        uint8x16_t R_full = vld1q_u8(right + rc);
        uint8_t m0 = bm[j >> 3];
        uint8x16_t both0 = vcombine_u8(vget_low_u8(L_full), vget_low_u8(R_full));
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x16x2_t src = {{ L_full, R_full }};
        uint8x8_t shuf1  = vld1_u8(expand_tab_pre[nr0][m1]);
        uint8x8_t o1     = vqtbl2_u8(src, shuf1);
        vst1_u8(out + j + 8, o1);
        int nr1 = expand_popcnt[m1];

        rc += nr0 + nr1;
        lc += (16 - nr0 - nr1);
    }
}

/* ============ NEW: 128-codes/iter, matches the user's pseudocode.
 *   - vld1q_u8 to load 16 bm bytes (= 128 codes)
 *   - vcntq_u8 on the 16-byte vector
 *   - extract low + high u64 of mask and popcnt via vgetq_lane_u64
 *   - prefix_sum_lo = popcnt_lo * 0x0101...
 *   - 8 inner iters over low half (codes 0..63)
 *   - prefix_sum_hi = popcnt_hi * 0x0101..., bias by lo_total
 *   - 8 inner iters over high half (codes 64..127)
 *   - 16 inner iters total, all fully independent of each other. */
__attribute__((always_inline)) static inline void new_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right,
                      uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 128 <= K; j += 128) {
        uint8x16_t bm_v = vld1q_u8(bm + (j >> 3));
        uint8x16_t pc_v = vcntq_u8(bm_v);
        uint64_t mask_lo = vgetq_lane_u64(vreinterpretq_u64_u8(bm_v), 0);
        uint64_t mask_hi = vgetq_lane_u64(vreinterpretq_u64_u8(bm_v), 1);
        uint64_t pc_lo   = vgetq_lane_u64(vreinterpretq_u64_u8(pc_v), 0);
        uint64_t pc_hi   = vgetq_lane_u64(vreinterpretq_u64_u8(pc_v), 1);
        uint64_t pfx_lo = pc_lo * 0x0101010101010101ULL;
        uint64_t pfx_hi = pc_hi * 0x0101010101010101ULL;
        uint64_t lo_total_bcst = (pfx_lo >> 56) * 0x0101010101010101ULL;
        pfx_hi += lo_total_bcst;  /* every byte = popcnt[0..idx_in_full] */

#define DO_ITER(k, mask_u64, pfx) do {                                   \
            uint8_t m_k = (uint8_t)(mask_u64 >> (8*((k)&7)));             \
            uint8_t cr  = ((k)&7) == 0 ? 0 : (uint8_t)((pfx) >> (8*(((k)&7)-1))); \
            uint8_t cl  = (uint8_t)(8*(k) - cr);                          \
            uint8x8_t  L = vld1_u8(left  + lc + cl);                      \
            uint8x8_t  R = vld1_u8(right + rc + cr);                      \
            uint8x16_t both = vcombine_u8(L, R);                          \
            uint8x8_t  shuf = vld1_u8(expand_tab[m_k]);                   \
            uint8x8_t  o    = vqtbl1_u8(both, shuf);                      \
            vst1_u8(out + j + 8*(k), o);                                  \
        } while (0)
        /* Low half (codes 0..63): pfx_lo's byte 0..6 give cnt for iters 1..7. */
        DO_ITER(0, mask_lo, pfx_lo); DO_ITER(1, mask_lo, pfx_lo);
        DO_ITER(2, mask_lo, pfx_lo); DO_ITER(3, mask_lo, pfx_lo);
        DO_ITER(4, mask_lo, pfx_lo); DO_ITER(5, mask_lo, pfx_lo);
        DO_ITER(6, mask_lo, pfx_lo); DO_ITER(7, mask_lo, pfx_lo);
        /* High half (codes 64..127): pfx_hi already includes lo_total bias. */
#define DO_ITER_HI(k_full) do {                                          \
            int kk = (k_full) - 8;                                       \
            uint8_t m_k = (uint8_t)(mask_hi >> (8*kk));                  \
            uint8_t cr  = kk == 0 ? (uint8_t)(pfx_lo >> 56)               \
                                  : (uint8_t)(pfx_hi >> (8*(kk-1)));     \
            uint8_t cl  = (uint8_t)(8*(k_full) - cr);                    \
            uint8x8_t  L = vld1_u8(left  + lc + cl);                     \
            uint8x8_t  R = vld1_u8(right + rc + cr);                     \
            uint8x16_t both = vcombine_u8(L, R);                         \
            uint8x8_t  shuf = vld1_u8(expand_tab[m_k]);                  \
            uint8x8_t  o    = vqtbl1_u8(both, shuf);                     \
            vst1_u8(out + j + 8*(k_full), o);                            \
        } while (0)
        DO_ITER_HI( 8); DO_ITER_HI( 9); DO_ITER_HI(10); DO_ITER_HI(11);
        DO_ITER_HI(12); DO_ITER_HI(13); DO_ITER_HI(14); DO_ITER_HI(15);
#undef DO_ITER
#undef DO_ITER_HI

        uint8_t total_right = (uint8_t)(pfx_hi >> 56);
        rc += total_right;
        lc += 128 - total_right;
    }
    (void)j;
}

/* ============ COMBINED: 32 codes/iter, vqtbl1+vqtbl2 + prefix sum.
 *
 * Each outer iter handles 32 codes via 2 chunks of 16 codes each.
 * Each chunk loads L_full/R_full once and does iter0 (vqtbl1) +
 * iter1 (vqtbl2 with expand_tab_pre[cnt_in_chunk][m]).
 *
 * The two chunks are INDEPENDENT because all cursors are precomputed
 * from the prefix sum over the 4 bm bytes (32 codes = 4 bytes). */
__attribute__((always_inline)) static inline void com_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right,
                      uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 32 <= K; j += 32) {
        uint32_t mask_u32;
        memcpy(&mask_u32, bm + (j >> 3), 4);

        /* popcnt the 4 bm bytes via NEON, then prefix sum. */
        uint8x8_t bm_v = vcreate_u8((uint64_t)mask_u32);
        uint8x8_t pc_v = vcnt_u8(bm_v);
        uint64_t pc_u64 = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;

        /* Chunk 0: bm bytes 0,1.  Cursor offset: 0 right, 0 left.
         * Chunk 1: bm bytes 2,3.  Cursor offset: byte 1 of pfx = cnt[0]+cnt[1].
         * Within each chunk, iter 1's nr0 = popcnt(byte k) = byte k of pc_u64. */
        uint8_t cnt_in_chunk0 = (uint8_t)(pc_u64);          /* = popcnt(m0) */
        uint8_t cnt_chunk1_r  = (uint8_t)(pfx >> 8);        /* total right at start of chunk 1 */
        uint8_t cnt_in_chunk1 = (uint8_t)(pc_u64 >> 16);    /* = popcnt(m2) */

        uint8_t m0 = (uint8_t)mask_u32;
        uint8_t m1 = (uint8_t)(mask_u32 >> 8);
        uint8_t m2 = (uint8_t)(mask_u32 >> 16);
        uint8_t m3 = (uint8_t)(mask_u32 >> 24);

        /* Chunk 0: load L_full / R_full at base cursors. */
        uint8x16_t L0 = vld1q_u8(left  + lc);
        uint8x16_t R0 = vld1q_u8(right + rc);
        uint8x16_t both00 = vcombine_u8(vget_low_u8(L0), vget_low_u8(R0));
        uint8x8_t  s00    = vld1_u8(expand_tab[m0]);
        vst1_u8(out + j,     vqtbl1_u8(both00, s00));
        uint8x16x2_t src0  = {{ L0, R0 }};
        uint8x8_t  s01     = vld1_u8(expand_tab_pre[cnt_in_chunk0][m1]);
        vst1_u8(out + j + 8, vqtbl2_u8(src0, s01));

        /* Chunk 1: load L_full / R_full at chunk-1 cursors (precomputed). */
        uint8_t cl1 = (uint8_t)(16 - cnt_chunk1_r);
        uint8x16_t L1 = vld1q_u8(left  + lc + cl1);
        uint8x16_t R1 = vld1q_u8(right + rc + cnt_chunk1_r);
        uint8x16_t both10 = vcombine_u8(vget_low_u8(L1), vget_low_u8(R1));
        uint8x8_t  s10    = vld1_u8(expand_tab[m2]);
        vst1_u8(out + j + 16, vqtbl1_u8(both10, s10));
        uint8x16x2_t src1  = {{ L1, R1 }};
        uint8x8_t  s11     = vld1_u8(expand_tab_pre[cnt_in_chunk1][m3]);
        vst1_u8(out + j + 24, vqtbl2_u8(src1, s11));

        uint8_t total_r = (uint8_t)(pfx >> 24);
        rc += total_r;
        lc += 32 - total_r;
    }
    (void)j;
}

/* ============ COM64: 64 codes/iter, 4 chunks of vqtbl1+vqtbl2.
 * Prefix sum over 8 bm bytes precomputes all 4 chunk-start cursors;
 * all 4 chunks are independent. */
__attribute__((always_inline)) static inline void com64_merge(const uint8_t *bm, int K,
                        const uint8_t *left, const uint8_t *right,
                        uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64;
        memcpy(&mask_u64, bm + (j >> 3), 8);
        uint8x8_t bm_v = vcreate_u8(mask_u64);
        uint8x8_t pc_v = vcnt_u8(bm_v);
        uint64_t pc_u64 = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;

        /* Per-chunk cnt offsets (right). */
        uint8_t cr0 = 0;
        uint8_t cr1 = (uint8_t)(pfx >>  8);   /* after 2 bytes */
        uint8_t cr2 = (uint8_t)(pfx >> 24);   /* after 4 bytes */
        uint8_t cr3 = (uint8_t)(pfx >> 40);   /* after 6 bytes */
        /* Per-chunk iter-1 nr0 = popcnt(chunk's first bm byte). */
        uint8_t in0 = (uint8_t)pc_u64;
        uint8_t in1 = (uint8_t)(pc_u64 >> 16);
        uint8_t in2 = (uint8_t)(pc_u64 >> 32);
        uint8_t in3 = (uint8_t)(pc_u64 >> 48);

        uint8_t m0 = (uint8_t)mask_u64;
        uint8_t m1 = (uint8_t)(mask_u64 >>  8);
        uint8_t m2 = (uint8_t)(mask_u64 >> 16);
        uint8_t m3 = (uint8_t)(mask_u64 >> 24);
        uint8_t m4 = (uint8_t)(mask_u64 >> 32);
        uint8_t m5 = (uint8_t)(mask_u64 >> 40);
        uint8_t m6 = (uint8_t)(mask_u64 >> 48);
        uint8_t m7 = (uint8_t)(mask_u64 >> 56);

#define CHUNK(idx, cr, in, ma, mb) do {                                       \
            uint8_t cl = (uint8_t)((idx)*16 - (cr));                          \
            uint8x16_t L = vld1q_u8(left  + lc + cl);                         \
            uint8x16_t R = vld1q_u8(right + rc + (cr));                       \
            uint8x16_t both = vcombine_u8(vget_low_u8(L), vget_low_u8(R));    \
            uint8x8_t  s0   = vld1_u8(expand_tab[ma]);                        \
            vst1_u8(out + j + (idx)*16,     vqtbl1_u8(both, s0));             \
            uint8x16x2_t src = {{ L, R }};                                    \
            uint8x8_t s1 = vld1_u8(expand_tab_pre[in][mb]);                   \
            vst1_u8(out + j + (idx)*16 + 8, vqtbl2_u8(src, s1));              \
        } while (0)
        CHUNK(0, cr0, in0, m0, m1);
        CHUNK(1, cr1, in1, m2, m3);
        CHUNK(2, cr2, in2, m4, m5);
        CHUNK(3, cr3, in3, m6, m7);
#undef CHUNK

        uint8_t total_r = (uint8_t)(pfx >> 56);
        rc += total_r;
        lc += 64 - total_r;
    }
    (void)j;
}

/* ============ COM128: 128 codes/iter, 8 chunks of vqtbl1+vqtbl2.
 * Prefix sum spans 16 bm bytes: two u64 multiplies + one fix-up to add
 * the low-half total to every byte of the high prefix sum. */
__attribute__((always_inline)) static inline void com128_merge(const uint8_t *bm, int K,
                         const uint8_t *left, const uint8_t *right,
                         uint8_t *out) {
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 128 <= K; j += 128) {
        uint8x16_t bm_v = vld1q_u8(bm + (j >> 3));
        uint8x16_t pc_v = vcntq_u8(bm_v);
        uint64_t mask_lo = vgetq_lane_u64(vreinterpretq_u64_u8(bm_v), 0);
        uint64_t mask_hi = vgetq_lane_u64(vreinterpretq_u64_u8(bm_v), 1);
        uint64_t pc_lo   = vgetq_lane_u64(vreinterpretq_u64_u8(pc_v), 0);
        uint64_t pc_hi   = vgetq_lane_u64(vreinterpretq_u64_u8(pc_v), 1);
        uint64_t pfx_lo = pc_lo * 0x0101010101010101ULL;
        uint64_t pfx_hi = pc_hi * 0x0101010101010101ULL;
        uint64_t lo_total = (pfx_lo >> 56) * 0x0101010101010101ULL;
        pfx_hi += lo_total;  /* every byte of pfx_hi now = popcnt[0..byte_idx_in_full] */

        /* Per-chunk start cursor for right = byte (2k-1) of full prefix sum. */
        uint8_t cr0 = 0;
        uint8_t cr1 = (uint8_t)(pfx_lo >>  8);
        uint8_t cr2 = (uint8_t)(pfx_lo >> 24);
        uint8_t cr3 = (uint8_t)(pfx_lo >> 40);
        uint8_t cr4 = (uint8_t)(pfx_lo >> 56);    /* lo total */
        uint8_t cr5 = (uint8_t)(pfx_hi >>  8);
        uint8_t cr6 = (uint8_t)(pfx_hi >> 24);
        uint8_t cr7 = (uint8_t)(pfx_hi >> 40);
        /* Per-chunk iter-1 nr0 = popcnt of chunk's first bm byte (= even bytes). */
        uint8_t in0 = (uint8_t)pc_lo;
        uint8_t in1 = (uint8_t)(pc_lo >> 16);
        uint8_t in2 = (uint8_t)(pc_lo >> 32);
        uint8_t in3 = (uint8_t)(pc_lo >> 48);
        uint8_t in4 = (uint8_t)pc_hi;
        uint8_t in5 = (uint8_t)(pc_hi >> 16);
        uint8_t in6 = (uint8_t)(pc_hi >> 32);
        uint8_t in7 = (uint8_t)(pc_hi >> 48);

        uint8_t m0  = (uint8_t) mask_lo;
        uint8_t m1  = (uint8_t)(mask_lo >>  8);
        uint8_t m2  = (uint8_t)(mask_lo >> 16);
        uint8_t m3  = (uint8_t)(mask_lo >> 24);
        uint8_t m4  = (uint8_t)(mask_lo >> 32);
        uint8_t m5  = (uint8_t)(mask_lo >> 40);
        uint8_t m6  = (uint8_t)(mask_lo >> 48);
        uint8_t m7  = (uint8_t)(mask_lo >> 56);
        uint8_t m8  = (uint8_t) mask_hi;
        uint8_t m9  = (uint8_t)(mask_hi >>  8);
        uint8_t m10 = (uint8_t)(mask_hi >> 16);
        uint8_t m11 = (uint8_t)(mask_hi >> 24);
        uint8_t m12 = (uint8_t)(mask_hi >> 32);
        uint8_t m13 = (uint8_t)(mask_hi >> 40);
        uint8_t m14 = (uint8_t)(mask_hi >> 48);
        uint8_t m15 = (uint8_t)(mask_hi >> 56);
        (void)mask_lo; (void)mask_hi;  /* registers now held in m0..m15 */

#define CHUNK(idx, cr, in, ma, mb) do {                                       \
            uint8_t cl = (uint8_t)((idx)*16 - (cr));                          \
            uint8x16_t L = vld1q_u8(left  + lc + cl);                         \
            uint8x16_t R = vld1q_u8(right + rc + (cr));                       \
            uint8x16_t both = vcombine_u8(vget_low_u8(L), vget_low_u8(R));    \
            uint8x8_t  s0   = vld1_u8(expand_tab[ma]);                        \
            vst1_u8(out + j + (idx)*16,     vqtbl1_u8(both, s0));             \
            uint8x16x2_t src = {{ L, R }};                                    \
            uint8x8_t s1 = vld1_u8(expand_tab_pre[in][mb]);                   \
            vst1_u8(out + j + (idx)*16 + 8, vqtbl2_u8(src, s1));              \
        } while (0)
        CHUNK(0, cr0, in0, m0,  m1);
        CHUNK(1, cr1, in1, m2,  m3);
        CHUNK(2, cr2, in2, m4,  m5);
        CHUNK(3, cr3, in3, m6,  m7);
        CHUNK(4, cr4, in4, m8,  m9);
        CHUNK(5, cr5, in5, m10, m11);
        CHUNK(6, cr6, in6, m12, m13);
        CHUNK(7, cr7, in7, m14, m15);
#undef CHUNK

        uint8_t total_r = (uint8_t)(pfx_hi >> 56);
        rc += total_r;
        lc += 128 - total_r;
    }
    (void)j;
}

/* ============ JEFF: PR#4 prefix128 -- COM128 with vpaddlq fold + u16 SWAR
 * prefix sum and half-level cursors.  Verbatim from Jeff Plaisance's
 * PR #4 (https://github.com/MarcinZukowski/pivco-huffman/pull/4). */
__attribute__((always_inline)) static inline void jeff_merge(const uint8_t *bm, int K,
                                        const uint8_t *left, const uint8_t *right,
                                        uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 128 <= K; j += 128) {
        uint8x16_t bmv = vld1q_u8(bm + (j >> 3));
        uint8x16_t pcv = vcntq_u8(bmv);

        uint64x2_t bm64 = vreinterpretq_u64_u8(bmv);
        uint64_t bm_lo = vgetq_lane_u64(bm64, 0);
        uint64_t bm_hi = vgetq_lane_u64(bm64, 1);
        uint64x2_t pc64 = vreinterpretq_u64_u8(pcv);
        uint64_t pc_lo = vgetq_lane_u64(pc64, 0);
        uint64_t pc_hi = vgetq_lane_u64(pc64, 1);

        uint16x8_t pair = vpaddlq_u8(pcv);          /* 8 block right-counts */
        uint64x2_t pr64 = vreinterpretq_u64_u16(pair);
        uint64_t pref_lo = vgetq_lane_u64(pr64, 0) * 0x0001000100010001ULL;
        uint64_t pref_hi = vgetq_lane_u64(pr64, 1) * 0x0001000100010001ULL;

        #define PFX_BLOCK(BMW, PCW, K_, EXCL, EBASE)                            \
        do {                                                                    \
            uint32_t excl_ = (EXCL);                                            \
            uint8_t  m0  = (uint8_t)((BMW) >> (16 * (K_)));                      \
            uint8_t  m1  = (uint8_t)((BMW) >> (16 * (K_) + 8));                  \
            uint8_t  nr0 = (uint8_t)((PCW) >> (16 * (K_)));                      \
            uint8x16_t L = vld1q_u8(left  + lc + (16 * (K_) - excl_));           \
            uint8x16_t R = vld1q_u8(right + rc + excl_);                         \
            uint8x16_t both0 = vcombine_u8(vget_low_u8(L), vget_low_u8(R));      \
            uint8x8_t  o0 = vqtbl1_u8(both0, vld1_u8(expand_tab[m0]));           \
            vst1_u8(out + j + (EBASE) + 16 * (K_), o0);                          \
            uint8x16x2_t src = {{ L, R }};                                       \
            uint8x8_t  o1 = vqtbl2_u8(src, vld1_u8(expand_tab_pre[nr0][m1]));    \
            vst1_u8(out + j + (EBASE) + 16 * (K_) + 8, o1);                      \
        } while (0)

        PFX_BLOCK(bm_lo, pc_lo, 0, 0,                        0);
        PFX_BLOCK(bm_lo, pc_lo, 1, (pref_lo)       & 0xFFFF, 0);
        PFX_BLOCK(bm_lo, pc_lo, 2, (pref_lo >> 16) & 0xFFFF, 0);
        PFX_BLOCK(bm_lo, pc_lo, 3, (pref_lo >> 32) & 0xFFFF, 0);
        uint32_t r_lo = (uint32_t)(pref_lo >> 48);
        rc += r_lo; lc += 64 - r_lo;

        PFX_BLOCK(bm_hi, pc_hi, 0, 0,                        64);
        PFX_BLOCK(bm_hi, pc_hi, 1, (pref_hi)       & 0xFFFF, 64);
        PFX_BLOCK(bm_hi, pc_hi, 2, (pref_hi >> 16) & 0xFFFF, 64);
        PFX_BLOCK(bm_hi, pc_hi, 3, (pref_hi >> 32) & 0xFFFF, 64);
        uint32_t r_hi = (uint32_t)(pref_hi >> 48);
        rc += r_hi; lc += 64 - r_hi;
        #undef PFX_BLOCK
    }
    (void)j;
}

/* ============ JEFF64: Jeff's block style (vpaddl fold + u16 SWAR prefix +
 * inline-shift PFX_BLOCK) but at 64-code stride -- single u64, no lo/hi
 * split, no cross-half cursor recombine.  Isolates block-style vs width. */
__attribute__((always_inline)) static inline void jeff64_merge(const uint8_t *bm, int K,
                                        const uint8_t *left, const uint8_t *right,
                                        uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t bm_word;
        memcpy(&bm_word, bm + (j >> 3), 8);
        uint8x8_t bm_v = vcreate_u8(bm_word);
        uint8x8_t pc_v = vcnt_u8(bm_v);
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);

        uint16x4_t pair = vpaddl_u8(pc_v);          /* 4 block right-counts */
        uint64_t pref = vget_lane_u64(vreinterpret_u64_u16(pair), 0)
                      * 0x0001000100010001ULL;

        #define PFX_BLOCK64(K_, EXCL)                                           \
        do {                                                                    \
            uint32_t excl_ = (EXCL);                                            \
            uint8_t  m0  = (uint8_t)(bm_word >> (16 * (K_)));                    \
            uint8_t  m1  = (uint8_t)(bm_word >> (16 * (K_) + 8));                \
            uint8_t  nr0 = (uint8_t)(pc_word >> (16 * (K_)));                    \
            uint8x16_t L = vld1q_u8(left  + lc + (16 * (K_) - excl_));           \
            uint8x16_t R = vld1q_u8(right + rc + excl_);                         \
            uint8x16_t both0 = vcombine_u8(vget_low_u8(L), vget_low_u8(R));      \
            uint8x8_t  o0 = vqtbl1_u8(both0, vld1_u8(expand_tab[m0]));           \
            vst1_u8(out + j + 16 * (K_), o0);                                    \
            uint8x16x2_t src = {{ L, R }};                                       \
            uint8x8_t  o1 = vqtbl2_u8(src, vld1_u8(expand_tab_pre[nr0][m1]));    \
            vst1_u8(out + j + 16 * (K_) + 8, o1);                                \
        } while (0)

        PFX_BLOCK64(0, 0);
        PFX_BLOCK64(1, (pref)       & 0xFFFF);
        PFX_BLOCK64(2, (pref >> 16) & 0xFFFF);
        PFX_BLOCK64(3, (pref >> 32) & 0xFFFF);
        uint32_t r = (uint32_t)(pref >> 48);
        rc += r; lc += 64 - r;
        #undef PFX_BLOCK64
    }
    (void)j;
}

/* ============ scalar reference */
static void sca_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right,
                      uint8_t *out) {
    int lc = 0, rc = 0;
    for (int j = 0; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left[lc++];
    }
}

/* ============ perf */
#if defined(__linux__)
static int g_perf_fd = -1, g_use_cyc = 0;
static void perf_init(void) {
    struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_HARDWARE; pe.size = sizeof pe;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
    g_perf_fd = (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    if (g_perf_fd < 0) { g_use_cyc = 0; return; }
    g_use_cyc = 1;
}
static inline void perf_start(void) { if (g_use_cyc) { ioctl(g_perf_fd, PERF_EVENT_IOC_RESET, 0); ioctl(g_perf_fd, PERF_EVENT_IOC_ENABLE, 0); } }
static inline uint64_t perf_stop(void) {
    if (!g_use_cyc) return 0;
    ioctl(g_perf_fd, PERF_EVENT_IOC_DISABLE, 0);
    uint64_t c = 0; if (read(g_perf_fd, &c, sizeof c) != sizeof c) c = 0; return c;
}
#else
static int g_use_cyc = 0;
static void perf_init(void) {}
static inline void perf_start(void) {}
static inline uint64_t perf_stop(void) { return 0; }
#endif
static double ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e9 + t.tv_nsec; }

#define K       (8 * 1024)
#define N_REPS  5000
#define N_ROUNDS 20
static uint8_t g_bm[K/8];
static uint8_t g_left[K + 64];
static uint8_t g_right[K + 64];
static uint8_t g_out[K + 64];
static uint8_t g_ref[K + 64];
static volatile uint64_t g_sink;

/* Direct-call bench: the variant body is referenced by name so the compiler
 * can inline it across the inner-loop.  Function-pointer dispatch through
 * `f` (the prior shape) was preventing inlining and adding ~2x apparent
 * cost vs the production bench_prim measurement. */
#define BENCH_ONE(VARIANT_NAME) do {                                          \
    VARIANT_NAME##_merge(g_bm, K, g_left, g_right, g_out);  /* warmup */       \
    uint64_t best_cyc = UINT64_MAX;                                            \
    double   best_ns  = 1e18;                                                  \
    for (int _r = 0; _r < N_ROUNDS; _r++) {                                    \
        perf_start();                                                          \
        double _t0 = ns();                                                     \
        for (int _rep = 0; _rep < N_REPS; _rep++)                              \
            VARIANT_NAME##_merge(g_bm, K, g_left, g_right, g_out);             \
        double _t1 = ns();                                                     \
        uint64_t _cyc = perf_stop();                                           \
        uint64_t _s = 0;                                                       \
        for (int _b = 0; _b < K; _b++) _s += g_out[_b];                        \
        g_sink ^= _s;                                                          \
        if (_cyc < best_cyc) best_cyc = _cyc;                                  \
        if (_t1 - _t0 < best_ns) best_ns = _t1 - _t0;                          \
    }                                                                          \
    uint64_t _elems = (uint64_t)N_REPS * K;                                    \
    if (g_use_cyc)                                                             \
        printf("  %-7s %7.4f cyc/elem   %7.4f ns/elem\n",                      \
               #VARIANT_NAME,                                                  \
               (double)best_cyc / (double)_elems,                              \
               best_ns / (double)_elems);                                      \
    else                                                                       \
        printf("  %-7s %7.4f ns/elem\n",                                       \
               #VARIANT_NAME, best_ns / (double)_elems);                       \
} while (0)

int main(void) {
    boost_thread();
    build_tables();
    uint32_t x = 0xC0FFEE13;
    for (size_t i = 0; i < sizeof g_bm; i++)   { x ^= x<<13; x ^= x>>17; x ^= x<<5; g_bm[i]    = (uint8_t)x; }
    for (size_t i = 0; i < sizeof g_left; i++) { x ^= x<<13; x ^= x>>17; x ^= x<<5; g_left[i]  = (uint8_t)x; g_right[i] = (uint8_t)(x ^ 0xA5); }
    perf_init();
    warmup_spin();
    printf("# NEON merge K=%d reps=%d best-of-%d\n", K, N_REPS, N_ROUNDS);
    printf("# Counter: %s\n", g_use_cyc ? "CPU_CYCLES (perf_event_open)" : "CLOCK_MONOTONIC ns");

    sca_merge(g_bm, K, g_left, g_right, g_ref);
    memset(g_out, 0, sizeof g_out);
    old_merge(g_bm, K, g_left, g_right, g_out);
    if (memcmp(g_ref, g_out, K) != 0) { printf("OLD MISMATCH\n"); return 1; }
    memset(g_out, 0, sizeof g_out);
    new_merge(g_bm, K, g_left, g_right, g_out);
    if (memcmp(g_ref, g_out, K) != 0) {
        for (int i = 0; i < 32; i++) if (g_ref[i] != g_out[i]) {
            printf("NEW MISMATCH at %d: ref=%02x got=%02x\n", i, g_ref[i], g_out[i]);
            break;
        }
        return 1;
    }
    memset(g_out, 0, sizeof g_out);
    com_merge(g_bm, K, g_left, g_right, g_out);
    if (memcmp(g_ref, g_out, K) != 0) {
        for (int i = 0; i < 64; i++) if (g_ref[i] != g_out[i]) {
            printf("COM MISMATCH at %d: ref=%02x got=%02x\n", i, g_ref[i], g_out[i]);
            break;
        }
        return 1;
    }
    memset(g_out, 0, sizeof g_out);
    com64_merge(g_bm, K, g_left, g_right, g_out);
    if (memcmp(g_ref, g_out, K) != 0) {
        for (int i = 0; i < 80; i++) if (g_ref[i] != g_out[i]) {
            printf("COM64 MISMATCH at %d: ref=%02x got=%02x\n", i, g_ref[i], g_out[i]);
            break;
        }
        return 1;
    }
    memset(g_out, 0, sizeof g_out);
    com128_merge(g_bm, K, g_left, g_right, g_out);
    if (memcmp(g_ref, g_out, K) != 0) {
        for (int i = 0; i < 144; i++) if (g_ref[i] != g_out[i]) {
            printf("COM128 MISMATCH at %d: ref=%02x got=%02x\n", i, g_ref[i], g_out[i]);
            break;
        }
        return 1;
    }
    memset(g_out, 0, sizeof g_out);
    jeff_merge(g_bm, K, g_left, g_right, g_out);
    if (memcmp(g_ref, g_out, K) != 0) {
        for (int i = 0; i < 144; i++) if (g_ref[i] != g_out[i]) {
            printf("JEFF MISMATCH at %d: ref=%02x got=%02x\n", i, g_ref[i], g_out[i]);
            break;
        }
        return 1;
    }
    memset(g_out, 0, sizeof g_out);
    jeff64_merge(g_bm, K, g_left, g_right, g_out);
    if (memcmp(g_ref, g_out, K) != 0) {
        for (int i = 0; i < 80; i++) if (g_ref[i] != g_out[i]) {
            printf("JEFF64 MISMATCH at %d: ref=%02x got=%02x\n", i, g_ref[i], g_out[i]);
            break;
        }
        return 1;
    }
    printf("# verify OK\n");
    BENCH_ONE(old);
    BENCH_ONE(com64);
    BENCH_ONE(com128);
    BENCH_ONE(jeff);
    BENCH_ONE(jeff64);
    return 0;
}
