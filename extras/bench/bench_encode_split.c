/* bench_encode_split.c — microbench for the encode-side tree-walk split
 * primitive.  Compares variants by input representation and SIMD shape.
 *
 *   OLD       — today's encoder body.  uint16 indices[] indirection;
 *               scalar mask build by reading codes[idx] and lens[idx]
 *               per element; SIMD partition_8 on indices.  Stride 8.
 *
 *   INDICES   — uint16 indices[] indirection; SIMD mask build (scalar
 *               gather of codes[indices[k]] into a vector, then
 *               vshlq+vaddvq movmask).  Same partition shape as OLD.
 *               Stride 8.  Isolates the win of vectorising mask build
 *               *without* removing the indirection.
 *
 *   CODES16   — dense-codes path.  Codes left-aligned at build-time so
 *               bit-d is at fixed position 15-d.  No lens[] lookup, no
 *               indirection.  SIMD mask, SIMD partition_8 on codes.
 *               Stride 8.
 *
 *   CODES16U  — same as CODES16 but stride-16 (two partition_8 per iter)
 *               — measures the ILP win of unrolling without changing
 *               the store pattern.
 *
 *   CHARS8    — dense 8-bit chars.  bit_at_depth[] LUT precomputed once
 *               per depth.  Byte-granular partition (compress_tab_byte
 *               built at init).  Stride 8 chars per iter, single 8-byte
 *               write per side.  Two-data-bytes-per-element reduction.
 *
 * The microbench fixes N=8192 and a representative random symbol mix.
 * Real distributions (prose_pride etc.) are bursty (skewed mask
 * histograms); the relative shape between variants is what matters.
 */
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifndef __aarch64__
int main(void) { puts("bench_encode_split: NEON only"); return 0; }
#else

#include "../pivco_huffman_neon_common.h"
#include <arm_neon.h>

#define BLK 8192

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============ partition_8 — shared by OLD and NEW ============
 * Takes a 16-byte (= 8 × uint16) source in a register, splits into
 * left/right via compress_tab[mask], writes both halves to output. */
static inline int partition_8_reg(uint8x16_t data, uint8_t mask,
                                   uint16_t *left_out, uint16_t *right_out)
{
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    uint8x16_t right = vqtbl1q_u8(data, shuf_r);
    uint8x16_t left  = vqtbl1q_u8(data, shuf_l);
    int n_right = compress_popcnt[mask];
    vst1q_u8((uint8_t *)right_out, right);
    vst1q_u8((uint8_t *)left_out, left);
    return n_right;
}

/* ============ OLD: today's encoder body ============
 * Iterates over an indices[] array, scalar-builds the mask per group of
 * 8 by reading codes_dense[idx] and lens_dense[idx] per element.  The
 * "dense" suffix means codes_dense[i] is the code for the i-th element
 * (i.e., codes_dense[i] == code_table[symbols[i]]); this matches what
 * the real encoder precomputes in its `enc_init` setup. */
static int split_old(const uint16_t *indices, int n,
                      const uint16_t *codes_dense, const uint8_t *lens_dense,
                      int depth,
                      uint8_t *bm,
                      uint16_t *left_out, uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        uint8_t mask = 0;
        for (int k = 0; k < 8; k++) {
            int idx = indices[j + k];
            int bit = (codes_dense[idx] >> (lens_dense[idx] - 1 - depth)) & 1;
            mask |= (uint8_t)(bit << k);
        }
        bm[j >> 3] = mask;
        uint8x16_t data = vld1q_u8((const uint8_t *)(indices + j));
        int nr = partition_8_reg(data, mask, left_out + n_left, right_out + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    return n_right;
}

/* ============ NEW16: dense-codes, SIMD mask build, stride-8 ============
 * Codes are left-aligned (bit-d at position 15-d).  No lens[] lookup,
 * no indirection.  Mask byte built via vshlq_u16 + vaddvq_u16 — the
 * standard NEON "movmask" recipe.  `vshrq_n_u16` requires an immediate
 * shift, so we use `vshlq_u16` with a negative runtime shift vector. */
static inline uint8_t simd_mask8(uint16x8_t code_vec, int neg_shift_d)
{
    /* Place bit-d into lane bit 0.  Then weight each lane k by 2^k via
     * a per-lane left-shift, and horizontally add.
     * neg_shift_d = -(15 - depth) — we use vshlq_u16 with a negative
     * shift to right-shift by (15-depth). */
    int16x8_t shr_vec = vdupq_n_s16((int16_t)neg_shift_d);
    uint16x8_t bit_lsb = vandq_u16(vshlq_u16(code_vec, shr_vec),
                                    vdupq_n_u16(1));
    /* weight lane k by 2^k */
    static const int16_t weights_shift[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int16x8_t weights = vld1q_s16(weights_shift);
    uint16x8_t weighted = vshlq_u16(bit_lsb, weights);
    return (uint8_t)vaddvq_u16(weighted);
}

static int split_new16(const uint16_t *codes_la, int n, int depth,
                       uint8_t *bm,
                       uint16_t *left_out, uint16_t *right_out)
{
    int neg_shift_d = -(15 - depth);
    int n_left = 0, n_right = 0;
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask = simd_mask8(code_vec, neg_shift_d);
        bm[j >> 3] = mask;
        uint8x16_t data = vreinterpretq_u8_u16(code_vec);
        int nr = partition_8_reg(data, mask, left_out + n_left, right_out + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    return n_right;
}

/* ============ INDICES: indirection + SIMD mask ============
 * Today's data layout (indices indirection) but with the scalar 8-bit
 * mask build replaced by a SIMD movmask.  Codes are LEFT-ALIGNED so the
 * bit at depth d is at position 15-d, removing the per-element shift
 * variance (this assumes all-same-length tables; for variable lens we'd
 * need a per-element shift, which NEON doesn't do for scalar gather).
 * Isolates the gain of SIMD-mask alone vs. SIMD-mask + dense layout. */
static int split_indices_simd(const uint16_t *indices, int n,
                               const uint16_t *codes_la_table_by_sym_in_idx,
                               int depth,
                               uint8_t *bm,
                               uint16_t *left_out, uint16_t *right_out)
{
    int neg_shift_d = -(15 - depth);
    int n_left = 0, n_right = 0;
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        /* Scalar gather of 8 left-aligned codes through the indirection.
         * NEON has no uint16 gather; this is the unavoidable cost of
         * keeping the indices layout. */
        uint16_t buf[8] __attribute__((aligned(16)));
        for (int k = 0; k < 8; k++)
            buf[k] = codes_la_table_by_sym_in_idx[indices[j + k]];
        uint16x8_t code_vec = vld1q_u16(buf);
        uint8_t mask = simd_mask8(code_vec, neg_shift_d);
        bm[j >> 3] = mask;
        uint8x16_t data = vld1q_u8((const uint8_t *)(indices + j));
        int nr = partition_8_reg(data, mask, left_out + n_left, right_out + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    return n_right;
}

/* ============ CHARS8: dense chars + byte-granular partition ============
 * Input is the 8-bit symbol stream directly.  `bit_at_depth[c]` is a
 * 256-byte LUT, precomputed once per tree depth (8 ops, called O(tree
 * depth) ≈ 20 times per block).  Partition is byte-granular so the
 * compress_tab_byte[256][16] table is built here (8-byte source → two
 * 8-byte halves, packed into one 16-byte vector for one TBL).
 *
 * Per iter: load 8 chars, gather 8 bits via TBL through bit_at_depth,
 * collect mask byte, byte-partition.  Single 8-byte store per side
 * (vst1_u8 vs vst1q_u8 for 16-byte). */

/* compress_tab_byte[mask][0..7]  = right shuffle indices (chars where bit=1)
 * compress_tab_byte[mask][8..15] = left  shuffle indices (chars where bit=0)
 * Slots past the valid count use index 0x80 → TBL returns 0 (don't-care). */
static uint8_t compress_tab_byte[256][16];

static void init_compress_tab_byte(void) {
    for (int m = 0; m < 256; m++) {
        int nr = 0, nl = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) compress_tab_byte[m][nr++]     = (uint8_t)k;
            else              compress_tab_byte[m][8 + nl++] = (uint8_t)k;
        }
        for (; nr < 8;  nr++) compress_tab_byte[m][nr]     = 0x80;
        for (; nl < 8;  nl++) compress_tab_byte[m][8 + nl] = 0x80;
    }
}

static int split_chars8(const uint8_t *chars, int n, int depth,
                         const uint16_t *code_table, const uint8_t *len_table,
                         uint8_t *bm,
                         uint8_t *left_out, uint8_t *right_out)
{
    /* Build bit_at_depth LUT for this depth.  Real encoder would build
     * this once per depth at recursion entry; for the microbench we
     * rebuild per call to make timing honest. */
    uint8_t bit_at_depth[256];
    for (int s = 0; s < 256; s++) {
        bit_at_depth[s] = (uint8_t)(
            (code_table[s] >> (len_table[s] - 1 - depth)) & 1);
    }

    int n_left = 0, n_right = 0;
    int j = 0;
    /* Standard movmask weights: shift lane k by k bits, then horiz-add. */
    static const int8_t weights8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int8x8_t wv = vld1_s8(weights8);
    for (; j + 8 <= n; j += 8) {
        /* 8 chars in low half of a 16-byte vector (pad upper with 0). */
        uint8x8_t chars_v = vld1_u8(chars + j);
        /* Gather 8 bits via TBL through bit_at_depth — but the table
         * is 256 entries, larger than vqtbl{1,2,3,4}q_u8 can cover (max
         * 64 entries).  Scalar gather of 8 bits, build the bit vector. */
        uint8_t bits[8];
        for (int k = 0; k < 8; k++) bits[k] = bit_at_depth[chars[j + k]];
        uint8x8_t bit_v = vld1_u8(bits);
        /* movmask: shift each lane k left by k, horiz-add. */
        uint8x8_t weighted = vshl_u8(bit_v, wv);
        uint8_t mask = (uint8_t)vaddv_u8(weighted);
        bm[j >> 3] = mask;

        /* Byte-granular partition.  Load 16-byte shuf (right[0..7] in
         * low half, left[0..7] in high half), one TBL, then two 8-byte
         * stores at running cursors. */
        uint8x16_t shuf = vld1q_u8(compress_tab_byte[mask]);
        uint8x16_t data = vcombine_u8(chars_v, vdup_n_u8(0));
        uint8x16_t out  = vqtbl1q_u8(data, shuf);
        vst1_u8(right_out + n_right, vget_low_u8(out));
        vst1_u8(left_out  + n_left,  vget_high_u8(out));
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += (8 - nr);
    }
    return n_right;
}

/* ============ NEW16U: dense-codes, stride-16 with double-buffered store ============
 * Loads 16 codes per iter (two 16-byte vectors), splits each via
 * partition_8_reg, but writes through running cursors as usual.  The
 * "double-buffer then write 16" optimisation requires accumulating
 * partial halves across iters with a fragment-aware shuf — V4 style.
 * That's a separate experiment; this variant just measures whether
 * stride-16 alone helps via ILP between the two halves. */
static int split_new16_unrolled(const uint16_t *codes_la, int n, int depth,
                                 uint8_t *bm,
                                 uint16_t *left_out, uint16_t *right_out)
{
    int neg_shift_d = -(15 - depth);
    int n_left = 0, n_right = 0;
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint16x8_t code_vec0 = vld1q_u16(codes_la + j);
        uint16x8_t code_vec1 = vld1q_u16(codes_la + j + 8);
        uint8_t mask0 = simd_mask8(code_vec0, neg_shift_d);
        uint8_t mask1 = simd_mask8(code_vec1, neg_shift_d);
        bm[j >> 3]       = mask0;
        bm[(j >> 3) + 1] = mask1;

        uint8x16_t data0 = vreinterpretq_u8_u16(code_vec0);
        uint8x16_t data1 = vreinterpretq_u8_u16(code_vec1);
        int nr0 = partition_8_reg(data0, mask0,
                                   left_out + n_left,    right_out + n_right);
        n_right += nr0; n_left += (8 - nr0);
        int nr1 = partition_8_reg(data1, mask1,
                                   left_out + n_left,    right_out + n_right);
        n_right += nr1; n_left += (8 - nr1);
    }
    return n_right;
}

/* ============ Driver ============ */
int main(int argc, char **argv)
{
    int repeats = (argc > 1) ? atoi(argv[1]) : 200000;
    if (repeats < 1) repeats = 1;

    init_compress_table();
    init_compress_tab_byte();

    /* Synthetic input.  Every symbol has len=11 (length-limited
     * canonical, matching PIVCO_MAX_CODE_LEN=11).  Code random in low
     * 11 bits.  This gives every iter a fresh mask drawn from a
     * roughly-uniform 8-bit distribution, which is representative
     * because the encoder's mask space is essentially defined by the
     * spread of bit-d across the elements within the group. */
    static uint16_t code_table[256];
    static uint8_t  len_table[256];
    static uint16_t code_la_table[256];   /* code << (16 - len) */
    srand(0xBEEF);
    for (int s = 0; s < 256; s++) {
        len_table[s] = 11;
        code_table[s] = (uint16_t)(rand() & ((1 << 11) - 1));
        code_la_table[s] = (uint16_t)(code_table[s] << (16 - 11));
    }

    /* The dense per-element arrays. */
    static uint8_t  symbols[BLK];                              /* CHARS8 input */
    static uint16_t indices[BLK];                              /* OLD/INDICES indirection */
    static uint16_t codes_dense[BLK];                          /* OLD reads through indices */
    static uint8_t  lens_dense[BLK];
    static uint16_t codes_la_dense[BLK];                       /* CODES16(U) input */
    for (int i = 0; i < BLK; i++) {
        symbols[i] = (uint8_t)(rand() & 0xFF);
        indices[i] = (uint16_t)i;
        codes_dense[i]    = code_table[symbols[i]];
        lens_dense[i]     = len_table[symbols[i]];
        codes_la_dense[i] = code_la_table[symbols[i]];
    }

    /* Buffers — sized 2× to be safe. */
    static uint16_t left_out[BLK * 2];
    static uint16_t right_out[BLK * 2];
    static uint8_t  left_out_b[BLK * 2];
    static uint8_t  right_out_b[BLK * 2];
    static uint8_t  bm[BLK];

    /* Fixed depth.  depth=3 is the typical interior-node depth on real
     * text (prose_pride: ~18 internal nodes / block, distribution
     * peaked near depths 2-5 from per-block profile). */
    const int depth = 3;

    /* Reference values from OLD. */
    int ref_n_right = split_old(indices, BLK, codes_dense, lens_dense, depth,
                                bm, left_out, right_out);
    uint64_t ref_bm_cksum = 0;
    for (int i = 0; i < BLK/8; i++) ref_bm_cksum ^= (uint64_t)bm[i] << (i & 63);

    struct row { const char *name; double ns_elem; int n_right; uint64_t bm_cksum; };
    enum { MAX_ROWS = 8 };
    struct row rows[MAX_ROWS];
    int ri = 0;

#define RUN(name_str, body) do { \
    double t0 = now_sec(); \
    int last_nr = 0; \
    for (int r = 0; r < repeats; r++) { last_nr = (body); } \
    double t1 = now_sec(); \
    uint64_t bms = 0; \
    for (int i = 0; i < BLK/8; i++) bms ^= (uint64_t)bm[i] << (i & 63); \
    rows[ri].name = (name_str); \
    rows[ri].ns_elem = (t1 - t0) * 1e9 / ((double)repeats * BLK); \
    rows[ri].n_right = last_nr; \
    rows[ri].bm_cksum = bms; \
    ri++; \
} while (0)

    RUN("OLD       (indices, scalar mask)",
        split_old(indices, BLK, codes_dense, lens_dense, depth,
                  bm, left_out, right_out));
    RUN("INDICES   (indices, SIMD mask)",
        split_indices_simd(indices, BLK, codes_la_dense, depth,
                           bm, left_out, right_out));
    RUN("CODES16   (dense codes, SIMD mask)",
        split_new16(codes_la_dense, BLK, depth, bm, left_out, right_out));
    RUN("CODES16U  (dense codes, stride-16)",
        split_new16_unrolled(codes_la_dense, BLK, depth, bm, left_out, right_out));
    RUN("CHARS8    (dense chars, byte partition)",
        split_chars8(symbols, BLK, depth, code_table, len_table,
                     bm, left_out_b, right_out_b));

    printf("\n=== encode_split microbench (N=%d, depth=%d, repeats=%d) ===\n",
           BLK, depth, repeats);
    printf("Ref n_right = %d, ref bm_cksum = 0x%016llx\n\n",
           ref_n_right, (unsigned long long)ref_bm_cksum);
    printf("  %-42s %10s %10s %10s  %s\n",
           "variant", "ns/elem", "GB/s in", "n_right", "bm_cksum");
    printf("  ----------------------------------------------------------------------------------\n");
    for (int i = 0; i < ri; i++) {
        /* GB/s on the *input* — denominator differs between variants:
         * 2B/elem for codes/indices, 1B/elem for chars. */
        double bytes_per_elem =
            (strstr(rows[i].name, "CHARS8") != NULL) ? 1.0 : 2.0;
        double gbs = bytes_per_elem / rows[i].ns_elem;
        int ok = (rows[i].n_right == ref_n_right &&
                  rows[i].bm_cksum == ref_bm_cksum);
        printf("  %-42s %10.3f %10.2f %10d  %s\n",
               rows[i].name, rows[i].ns_elem, gbs, rows[i].n_right,
               ok ? "match" : "MISMATCH");
    }
    printf("\n  Speedup vs OLD:\n");
    for (int i = 1; i < ri; i++)
        printf("    %-42s %.2fx\n", rows[i].name,
               rows[0].ns_elem / rows[i].ns_elem);
    return 0;
}

#endif /* __aarch64__ */
