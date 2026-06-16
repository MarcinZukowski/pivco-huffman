/* bench_partition_neon.c -- NEON encode partition: production 8-code
 * serial-cursor loop vs wide-window prefix-sum cursor-decoupled variants.
 *
 *   old    : production build_bitmap_partition_full (8 codes/iter, serial
 *            n_left/n_right adds, compress_popcnt[mask] load per chunk)
 *   com    : 64 codes/iter, 8 chunks.  Load all 8 code_vecs first (so the
 *            in-place left writes have no RAW hazard), build 8 masks + one
 *            8-byte bm store, popcount via vcnt + 0x0101.. prefix sum to
 *            precompute every chunk's left/right cursor, then 8 independent
 *            compact+scatter.  No compress_popcnt[] load; no serial cursor
 *            add chain.  Masks built with the per-chunk vaddvq (as old).
 *   com_v2 : com, but the 8 masks built via an 8x8 transpose + vsli.  LOSES
 *            (transpose costs more than the 8 vaddvq it removes).  Kept as a
 *            negative result.
 *   com_v3 : com, but the 8 masks built via vtst + powers + a vpaddq merge
 *            tree (ARM "movemask to NEON" pairwise reduction, batched 8-way
 *            so the tree's output lanes ARE the 8 masks -- one fmov).  WINS:
 *            replacing 8 lane-crossing vaddvq + 8 fmov with 7 vpaddq + 1 fmov
 *            roughly doubles the partition speedup on Graviton.  *** shipped
 *
 * Per-element partition cost, old -> com_v3 (cyc/elem perf_event_open on
 * Graviton, ns/elem CLOCK_MONOTONIC on Apple, 3-round median):
 *   M4 (Apple)          0.083 -> 0.070 ns   (-16%)
 *   c7g (Neoverse V1)   0.681 -> 0.516 cyc  (-24%)
 *   c8g (Neoverse V2)   0.671 -> 0.465 cyc  (-30%)
 *   m9g (Neoverse V3)   0.625 -> 0.437 cyc  (-30%)
 *
 * Partition is destructive (left compacts in place over codes_la), so each
 * timed call first memcpy-restores a pristine input; the memcpy cost is
 * measured separately and subtracted.
 *
 * Linux: CPU_CYCLES via perf_event_open.  macOS: CLOCK_MONOTONIC ns
 * (QoS boost + warmup spin to land on a P-core at full clock).
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
extern int pthread_set_qos_class_self_np(qos_class_t, int);
#endif

static void boost_thread(void) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#elif defined(__linux__)
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    (void)sched_setaffinity(0, sizeof set, &set);
#endif
}
static volatile uint64_t g_warmup_sink;
static void warmup_spin(void) {
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    uint8x16_t v = vdupq_n_u8(0x5A); uint64_t total = 0;
    do {
        for (int k = 0; k < 100000; k++) { v = vaddq_u8(v, vdupq_n_u8(1)); v = veorq_u8(v, vdupq_n_u8(0x35)); }
        total += vgetq_lane_u64(vreinterpretq_u64_u8(v), 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
    } while ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec) < 200e6);
    g_warmup_sink ^= total;
}

/* ============ compress_tab[256][32] + compress_popcnt[256] (prod layout) */
static uint8_t compress_tab[256][32] __attribute__((aligned(32)));
static uint8_t compress_popcnt[256];
static void build_tables(void) {
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0;
        for (int i = 0; i < 8; i++) if (mask & (1<<i)) {
            compress_tab[mask][out_r*2]   = (uint8_t)(i*2);
            compress_tab[mask][out_r*2+1] = (uint8_t)(i*2+1);
            out_r++;
        }
        for (; out_r < 8; out_r++) { compress_tab[mask][out_r*2]=0xFF; compress_tab[mask][out_r*2+1]=0xFF; }
        int out_l = 0;
        for (int i = 0; i < 8; i++) if (!(mask & (1<<i))) {
            compress_tab[mask][16+out_l*2]   = (uint8_t)(i*2);
            compress_tab[mask][16+out_l*2+1] = (uint8_t)(i*2+1);
            out_l++;
        }
        for (; out_l < 8; out_l++) { compress_tab[mask][16+out_l*2]=0xFF; compress_tab[mask][16+out_l*2+1]=0xFF; }
        compress_popcnt[mask] = (uint8_t)__builtin_popcount(mask);
    }
}

static inline uint8_t enc_mask8(uint16x8_t code_vec, int neg_shift_d) {
    int16x8_t shr_vec = vdupq_n_s16((int16_t)neg_shift_d);
    uint16x8_t bit_lsb = vandq_u16(vshlq_u16(code_vec, shr_vec), vdupq_n_u16(1));
    static const int16_t weights[8] = {0,1,2,3,4,5,6,7};
    uint16x8_t weighted = vshlq_u16(bit_lsb, vld1q_s16(weights));
    return (uint8_t)vaddvq_u16(weighted);
}

/* ============ OLD: production full partition (8 codes/iter, serial). */
__attribute__((always_inline)) static inline int old_partition(uint16_t *codes_la, int n,
                       int depth, uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);
    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask = enc_mask8(code_vec, neg_shift_d);
        bm[j >> 3] = mask;
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t data  = vreinterpretq_u8_u16(code_vec);
        uint8x16_t right = vqtbl1q_u8(data, vld1q_u8(tab));
        uint8x16_t left  = vqtbl1q_u8(data, vld1q_u8(tab + 16));
        int nr = compress_popcnt[mask];
        vst1q_u8((uint8_t *)(right_out + n_right), right);
        vst1q_u8((uint8_t *)(codes_la  + n_left ), left);
        n_right += nr; n_left += (8 - nr);
    }
    /* tail */
    int shift_d = 15 - depth;
    for (; j < n; j++) {
        uint16_t c = codes_la[j];
        if ((c >> shift_d) & 1) right_out[n_right++] = c;
        else                    codes_la[n_left++]   = c;
    }
    return n_right;
}

/* ============ COM: 64 codes/iter, 8 chunks, prefix-sum cursors. */
__attribute__((always_inline)) static inline int com_partition(uint16_t *codes_la, int n,
                       int depth, uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);
    for (; j + 64 <= n; j += 64) {
        /* Load all 8 code_vecs first (reads precede in-place writes). */
        uint16x8_t cv0=vld1q_u16(codes_la+j),    cv1=vld1q_u16(codes_la+j+8),
                   cv2=vld1q_u16(codes_la+j+16), cv3=vld1q_u16(codes_la+j+24),
                   cv4=vld1q_u16(codes_la+j+32), cv5=vld1q_u16(codes_la+j+40),
                   cv6=vld1q_u16(codes_la+j+48), cv7=vld1q_u16(codes_la+j+56);
        uint8_t m0=enc_mask8(cv0,neg_shift_d), m1=enc_mask8(cv1,neg_shift_d),
                m2=enc_mask8(cv2,neg_shift_d), m3=enc_mask8(cv3,neg_shift_d),
                m4=enc_mask8(cv4,neg_shift_d), m5=enc_mask8(cv5,neg_shift_d),
                m6=enc_mask8(cv6,neg_shift_d), m7=enc_mask8(cv7,neg_shift_d);
        uint64_t mask_word = (uint64_t)m0 | ((uint64_t)m1<<8) | ((uint64_t)m2<<16)
                           | ((uint64_t)m3<<24) | ((uint64_t)m4<<32) | ((uint64_t)m5<<40)
                           | ((uint64_t)m6<<48) | ((uint64_t)m7<<56);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint8x8_t pc_v = vcnt_u8(vcreate_u8(mask_word));
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_word * 0x0101010101010101ULL;  /* inclusive prefix */

        /* per-chunk right cursor = exclusive prefix; left = 8k - that */
#define PCHUNK(K_, CV, M) do {                                              \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
        uint32_t cl = 8u*(K_) - cr;                                         \
        const uint8_t *tab = compress_tab[(M)];                            \
        uint8x16_t data  = vreinterpretq_u8_u16(CV);                       \
        uint8x16_t right = vqtbl1q_u8(data, vld1q_u8(tab));                \
        uint8x16_t left  = vqtbl1q_u8(data, vld1q_u8(tab + 16));           \
        vst1q_u8((uint8_t *)(right_out + n_right + cr), right);            \
        vst1q_u8((uint8_t *)(codes_la  + n_left  + cl), left);            \
    } while (0)
        PCHUNK(0,cv0,m0); PCHUNK(1,cv1,m1); PCHUNK(2,cv2,m2); PCHUNK(3,cv3,m3);
        PCHUNK(4,cv4,m4); PCHUNK(5,cv5,m5); PCHUNK(6,cv6,m6); PCHUNK(7,cv7,m7);
#undef PCHUNK
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += total_r; n_left += 64 - total_r;
    }
    int shift_d = 15 - depth;
    for (; j < n; j++) {
        uint16_t c = codes_la[j];
        if ((c >> shift_d) & 1) right_out[n_right++] = c;
        else                    codes_la[n_left++]   = c;
    }
    return n_right;
}

/* ============ build all 8 partition-mask bytes at once via an 8x8
 * transpose + vsli accumulate (no per-chunk addv, no per-chunk fmov).
 * Returns the 8 masks packed LE into a u64 (byte k = mask_k). */
static inline uint64_t build_8_masks_transpose(
        uint16x8_t cv0, uint16x8_t cv1, uint16x8_t cv2, uint16x8_t cv3,
        uint16x8_t cv4, uint16x8_t cv5, uint16x8_t cv6, uint16x8_t cv7,
        int neg_shift_d) {
    int16x8_t s = vdupq_n_s16((int16_t)neg_shift_d);
    uint16x8_t one = vdupq_n_u16(1);
    /* isolate the depth bit into the LSB of each u16 lane */
    uint16x8_t b0=vandq_u16(vshlq_u16(cv0,s),one), b1=vandq_u16(vshlq_u16(cv1,s),one),
               b2=vandq_u16(vshlq_u16(cv2,s),one), b3=vandq_u16(vshlq_u16(cv3,s),one),
               b4=vandq_u16(vshlq_u16(cv4,s),one), b5=vandq_u16(vshlq_u16(cv5,s),one),
               b6=vandq_u16(vshlq_u16(cv6,s),one), b7=vandq_u16(vshlq_u16(cv7,s),one);
    /* 8x8 transpose (u16): level 1 (16b), level 2 (32b), level 3 (64b) */
    uint16x8_t a0=vtrn1q_u16(b0,b1), a1=vtrn2q_u16(b0,b1),
               a2=vtrn1q_u16(b2,b3), a3=vtrn2q_u16(b2,b3),
               a4=vtrn1q_u16(b4,b5), a5=vtrn2q_u16(b4,b5),
               a6=vtrn1q_u16(b6,b7), a7=vtrn2q_u16(b6,b7);
    uint32x4_t c0=vtrn1q_u32(vreinterpretq_u32_u16(a0),vreinterpretq_u32_u16(a2)),
               c2=vtrn2q_u32(vreinterpretq_u32_u16(a0),vreinterpretq_u32_u16(a2)),
               c1=vtrn1q_u32(vreinterpretq_u32_u16(a1),vreinterpretq_u32_u16(a3)),
               c3=vtrn2q_u32(vreinterpretq_u32_u16(a1),vreinterpretq_u32_u16(a3)),
               c4=vtrn1q_u32(vreinterpretq_u32_u16(a4),vreinterpretq_u32_u16(a6)),
               c6=vtrn2q_u32(vreinterpretq_u32_u16(a4),vreinterpretq_u32_u16(a6)),
               c5=vtrn1q_u32(vreinterpretq_u32_u16(a5),vreinterpretq_u32_u16(a7)),
               c7=vtrn2q_u32(vreinterpretq_u32_u16(a5),vreinterpretq_u32_u16(a7));
    uint64x2_t t0=vtrn1q_u64(vreinterpretq_u64_u32(c0),vreinterpretq_u64_u32(c4)),
               t4=vtrn2q_u64(vreinterpretq_u64_u32(c0),vreinterpretq_u64_u32(c4)),
               t1=vtrn1q_u64(vreinterpretq_u64_u32(c1),vreinterpretq_u64_u32(c5)),
               t5=vtrn2q_u64(vreinterpretq_u64_u32(c1),vreinterpretq_u64_u32(c5)),
               t2=vtrn1q_u64(vreinterpretq_u64_u32(c2),vreinterpretq_u64_u32(c6)),
               t6=vtrn2q_u64(vreinterpretq_u64_u32(c2),vreinterpretq_u64_u32(c6)),
               t3=vtrn1q_u64(vreinterpretq_u64_u32(c3),vreinterpretq_u64_u32(c7)),
               t7=vtrn2q_u64(vreinterpretq_u64_u32(c3),vreinterpretq_u64_u32(c7));
    /* t_i lane k = bit i of chunk k.  narrow to u8x8, vsli into bit i. */
    uint8x8_t acc = vmovn_u16(vreinterpretq_u16_u64(t0));         /* bit 0 */
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t1)), 1);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t2)), 2);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t3)), 3);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t4)), 4);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t5)), 5);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t6)), 6);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t7)), 7);
    return vget_lane_u64(vreinterpret_u64_u8(acc), 0);
}

/* ============ COM_V2: com_partition but masks via transpose+vsli. */
__attribute__((always_inline)) static inline int com_v2_partition(uint16_t *codes_la, int n,
                       int depth, uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);
    for (; j + 64 <= n; j += 64) {
        uint16x8_t cv0=vld1q_u16(codes_la+j),    cv1=vld1q_u16(codes_la+j+8),
                   cv2=vld1q_u16(codes_la+j+16), cv3=vld1q_u16(codes_la+j+24),
                   cv4=vld1q_u16(codes_la+j+32), cv5=vld1q_u16(codes_la+j+40),
                   cv6=vld1q_u16(codes_la+j+48), cv7=vld1q_u16(codes_la+j+56);
        uint64_t mask_word = build_8_masks_transpose(cv0,cv1,cv2,cv3,cv4,cv5,cv6,cv7,neg_shift_d);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint8x8_t pc_v = vcnt_u8(vcreate_u8(mask_word));
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_word * 0x0101010101010101ULL;
#define PCHUNK2(K_, CV) do {                                                \
        uint8_t M = (uint8_t)(mask_word >> (8*(K_)));                        \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
        uint32_t cl = 8u*(K_) - cr;                                         \
        const uint8_t *tab = compress_tab[M];                              \
        uint8x16_t data  = vreinterpretq_u8_u16(CV);                       \
        uint8x16_t right = vqtbl1q_u8(data, vld1q_u8(tab));                \
        uint8x16_t left  = vqtbl1q_u8(data, vld1q_u8(tab + 16));           \
        vst1q_u8((uint8_t *)(right_out + n_right + cr), right);            \
        vst1q_u8((uint8_t *)(codes_la  + n_left  + cl), left);            \
    } while (0)
        PCHUNK2(0,cv0); PCHUNK2(1,cv1); PCHUNK2(2,cv2); PCHUNK2(3,cv3);
        PCHUNK2(4,cv4); PCHUNK2(5,cv5); PCHUNK2(6,cv6); PCHUNK2(7,cv7);
#undef PCHUNK2
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += total_r; n_left += 64 - total_r;
    }
    int shift_d = 15 - depth;
    for (; j < n; j++) {
        uint16_t c = codes_la[j];
        if ((c >> shift_d) & 1) right_out[n_right++] = c;
        else                    codes_la[n_left++]   = c;
    }
    return n_right;
}

/* ============ build all 8 mask bytes via vtst + powers + vpaddq tree
 * (ARM blog "tertiary" movemask, adapted to u16 x 8 chunks).  One fmov. */
static inline uint64_t build_8_masks_vpaddq(
        uint16x8_t cv0, uint16x8_t cv1, uint16x8_t cv2, uint16x8_t cv3,
        uint16x8_t cv4, uint16x8_t cv5, uint16x8_t cv6, uint16x8_t cv7,
        int shift_d) {
    uint16x8_t bitsel = vdupq_n_u16((uint16_t)(1u << shift_d));
    static const uint16_t powers_arr[8] = {1,2,4,8,16,32,64,128};
    uint16x8_t powers = vld1q_u16(powers_arr);
    /* per lane: (cv & bit) ? power : 0 */
    uint16x8_t w0=vandq_u16(vtstq_u16(cv0,bitsel),powers), w1=vandq_u16(vtstq_u16(cv1,bitsel),powers),
               w2=vandq_u16(vtstq_u16(cv2,bitsel),powers), w3=vandq_u16(vtstq_u16(cv3,bitsel),powers),
               w4=vandq_u16(vtstq_u16(cv4,bitsel),powers), w5=vandq_u16(vtstq_u16(cv5,bitsel),powers),
               w6=vandq_u16(vtstq_u16(cv6,bitsel),powers), w7=vandq_u16(vtstq_u16(cv7,bitsel),powers);
    /* vpaddq tree: 8 vectors -> one uint16x8 with lane k = mask_k */
    uint16x8_t p01=vpaddq_u16(w0,w1), p23=vpaddq_u16(w2,w3),
               p45=vpaddq_u16(w4,w5), p67=vpaddq_u16(w6,w7);
    uint16x8_t q0=vpaddq_u16(p01,p23), q1=vpaddq_u16(p45,p67);
    uint16x8_t r=vpaddq_u16(q0,q1);   /* lane k = mask_k */
    return vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(r)), 0);
}

/* ============ COM_V3: com_partition, masks via vtst+powers+vpaddq. */
__attribute__((always_inline)) static inline int com_v3_partition(uint16_t *codes_la, int n,
                       int depth, uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    int shift_d = 15 - depth;
    for (; j + 64 <= n; j += 64) {
        uint16x8_t cv0=vld1q_u16(codes_la+j),    cv1=vld1q_u16(codes_la+j+8),
                   cv2=vld1q_u16(codes_la+j+16), cv3=vld1q_u16(codes_la+j+24),
                   cv4=vld1q_u16(codes_la+j+32), cv5=vld1q_u16(codes_la+j+40),
                   cv6=vld1q_u16(codes_la+j+48), cv7=vld1q_u16(codes_la+j+56);
        uint64_t mask_word = build_8_masks_vpaddq(cv0,cv1,cv2,cv3,cv4,cv5,cv6,cv7,shift_d);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint8x8_t pc_v = vcnt_u8(vcreate_u8(mask_word));
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_word * 0x0101010101010101ULL;
#define PCHUNK3(K_, CV) do {                                                \
        uint8_t M = (uint8_t)(mask_word >> (8*(K_)));                        \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
        uint32_t cl = 8u*(K_) - cr;                                         \
        const uint8_t *tab = compress_tab[M];                              \
        uint8x16_t data  = vreinterpretq_u8_u16(CV);                       \
        uint8x16_t right = vqtbl1q_u8(data, vld1q_u8(tab));                \
        uint8x16_t left  = vqtbl1q_u8(data, vld1q_u8(tab + 16));           \
        vst1q_u8((uint8_t *)(right_out + n_right + cr), right);            \
        vst1q_u8((uint8_t *)(codes_la  + n_left  + cl), left);            \
    } while (0)
        PCHUNK3(0,cv0); PCHUNK3(1,cv1); PCHUNK3(2,cv2); PCHUNK3(3,cv3);
        PCHUNK3(4,cv4); PCHUNK3(5,cv5); PCHUNK3(6,cv6); PCHUNK3(7,cv7);
#undef PCHUNK3
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += total_r; n_left += 64 - total_r;
    }
    for (; j < n; j++) {
        uint16_t c = codes_la[j];
        if ((c >> shift_d) & 1) right_out[n_right++] = c;
        else                    codes_la[n_left++]   = c;
    }
    return n_right;
}

/* ============ scalar reference */
static int sca_partition(uint16_t *codes_la, int n, int depth, uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, shift_d = 15 - depth;
    for (int j = 0; j < n; j++) {
        if ((j & 7) == 0) bm[j>>3] = 0;
        uint16_t c = codes_la[j];
        if ((c >> shift_d) & 1) { bm[j>>3] |= (uint8_t)(1 << (j&7)); right_out[n_right++] = c; }
        else                    { codes_la[n_left++] = c; }
    }
    return n_right;
}

/* ============ perf */
#if defined(__linux__)
static int g_perf_fd=-1, g_use_cyc=0;
static void perf_init(void){ struct perf_event_attr pe; memset(&pe,0,sizeof pe);
    pe.type=PERF_TYPE_HARDWARE; pe.size=sizeof pe; pe.config=PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled=1; pe.exclude_kernel=1; pe.exclude_hv=1;
    g_perf_fd=(int)syscall(SYS_perf_event_open,&pe,0,-1,-1,0);
    if(g_perf_fd<0){fprintf(stderr,"perf: %s\n",strerror(errno)); g_use_cyc=0; return;} g_use_cyc=1; }
static inline void perf_start(void){ if(g_use_cyc){ioctl(g_perf_fd,PERF_EVENT_IOC_RESET,0);ioctl(g_perf_fd,PERF_EVENT_IOC_ENABLE,0);} }
static inline uint64_t perf_stop(void){ if(!g_use_cyc)return 0; ioctl(g_perf_fd,PERF_EVENT_IOC_DISABLE,0); uint64_t c=0; if(read(g_perf_fd,&c,sizeof c)!=sizeof c)c=0; return c; }
#else
static int g_use_cyc=0; static void perf_init(void){} static inline void perf_start(void){} static inline uint64_t perf_stop(void){return 0;}
#endif
static double ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }

#define N       8192
#define N_REPS  4000
#define N_ROUNDS 12
#define DEPTH   3
static uint16_t g_pristine[N];
static uint16_t g_work[N + 16];
static uint16_t g_right[N + 16];
static uint8_t  g_bm[N/8 + 8];
static volatile uint64_t g_sink;

/* Measure: (partition + memcpy-restore) - (memcpy-restore only), best of rounds. */
#define BENCH_ONE(NAME) do {                                                  \
    memcpy(g_work, g_pristine, sizeof g_pristine);                            \
    NAME##_partition(g_work, N, DEPTH, g_bm, g_right);                         \
    uint64_t best_cyc=UINT64_MAX; double best_ns=1e18;                        \
    for (int _r=0;_r<N_ROUNDS;_r++){                                          \
        perf_start(); double _t0=ns();                                        \
        for(int _i=0;_i<N_REPS;_i++){ memcpy(g_work,g_pristine,sizeof g_pristine); \
            int nr=NAME##_partition(g_work,N,DEPTH,g_bm,g_right); g_sink^=(uint64_t)nr; } \
        double _t1=ns(); uint64_t _c=perf_stop();                             \
        if(_c<best_cyc)best_cyc=_c; if(_t1-_t0<best_ns)best_ns=_t1-_t0;        \
    }                                                                         \
    /* memcpy-only baseline */                                                \
    uint64_t mc_cyc=UINT64_MAX; double mc_ns=1e18;                            \
    for (int _r=0;_r<N_ROUNDS;_r++){                                          \
        perf_start(); double _t0=ns();                                        \
        for(int _i=0;_i<N_REPS;_i++){ memcpy(g_work,g_pristine,sizeof g_pristine); g_sink^=g_work[0]; } \
        double _t1=ns(); uint64_t _c=perf_stop();                             \
        if(_c<mc_cyc)mc_cyc=_c; if(_t1-_t0<mc_ns)mc_ns=_t1-_t0;                \
    }                                                                         \
    uint64_t _e=(uint64_t)N_REPS*N;                                          \
    double cyc=(double)(best_cyc-mc_cyc)/_e, nsec=(best_ns-mc_ns)/_e;          \
    if(g_use_cyc) printf("  %-7s %7.4f cyc/elem  %7.4f ns/elem\n",#NAME,cyc,nsec); \
    else          printf("  %-7s %7.4f ns/elem\n",#NAME,nsec);                \
} while(0)

static int verify(const char *nm, int (*f)(uint16_t*,int,int,uint8_t*,uint16_t*)) {
    uint16_t ref_work[N], ref_right[N]; uint8_t ref_bm[N/8+8];
    memcpy(ref_work, g_pristine, sizeof g_pristine);
    int rnr = sca_partition(ref_work, N, DEPTH, ref_bm, ref_right);
    memcpy(g_work, g_pristine, sizeof g_pristine);
    int nr = f(g_work, N, DEPTH, g_bm, g_right);
    if (nr != rnr) { printf("%s n_right %d != ref %d\n", nm, nr, rnr); return 1; }
    if (memcmp(g_bm, ref_bm, N/8) != 0) { printf("%s bm mismatch\n", nm); return 1; }
    if (memcmp(g_work, ref_work, (size_t)(N-rnr)*2) != 0) { printf("%s left mismatch\n", nm); return 1; }
    if (memcmp(g_right, ref_right, (size_t)rnr*2) != 0) { printf("%s right mismatch\n", nm); return 1; }
    return 0;
}

int main(void) {
    boost_thread(); build_tables();
    uint32_t x=0xC0FFEE13;
    for (int i=0;i<N;i++){ x^=x<<13;x^=x>>17;x^=x<<5; g_pristine[i]=(uint16_t)(x & 0xFFFF); }
    perf_init(); warmup_spin();
    printf("# NEON partition N=%d depth=%d reps=%d best-of-%d\n", N, DEPTH, N_REPS, N_ROUNDS);
    printf("# Counter: %s\n", g_use_cyc?"CPU_CYCLES (perf_event_open)":"CLOCK_MONOTONIC ns");
    int fail=0;
    fail+=verify("old", old_partition);
    fail+=verify("com", com_partition);
    fail+=verify("com_v2", com_v2_partition);
    fail+=verify("com_v3", com_v3_partition);
    if(fail) return 1;
    printf("# verify OK\n");
    BENCH_ONE(old);
    BENCH_ONE(com);
    BENCH_ONE(com_v3);
    return 0;
}
