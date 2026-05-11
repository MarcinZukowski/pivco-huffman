/* Bottom-up PIVCO-Huffman decoder (NEON).
 *
 * Inverts the conventional top-down "partition + scatter" decode into
 * a bottom-up tree_merge sequence.  At each leaf we materialise (or
 * broadcast) the leaf symbol.  At each internal node we merge the two
 * child output buffers per the node's bitmap.  At the root, the merge
 * output IS the final symbols[N] — no scatter, no indices array, all
 * u8 throughout.
 *
 * Core primitive: tree_merge(bitmap, left, right) -> output where
 *   output[k] = bitmap[k] ? right[rank1(bitmap, k)]
 *                         : left [rank0(bitmap, k)]
 *
 * For each 8-bit mask chunk, a single vqtbl1_u8 over a 16-byte
 * vcombine(left[0..8), right[0..8)) with a 256-entry shuffle pattern
 * indexed by the mask byte (expand_tab below).
 *
 * Variants for the common cases that avoid materialising a buffer:
 *   - tree_merge_bcast_left:  one input is a broadcast constant
 *                             (HALF_RIGHT: left child is the prefill leaf)
 *   - tree_merge_bcast_right: mirror (HALF_LEFT)
 *   - merge_both_const:       both inputs constants (BOTH_LEAVES)
 *
 * Flat-subtree fast path stays as a "leaf-style" terminal node — it
 * produces K bytes directly into the parent's reserved slot via the
 * existing flat_decode_direct logic.
 *
 * See README.md / IDEAS.md for the design rationale.  This file is
 * NEW; it does NOT share decode logic with src/pivco_huffman_neon.c. */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
#include <string.h>

#ifdef PIVCO_HAS_NEON
#include <arm_neon.h>

/* Popcount K_right bits from the first nbytes of bm, where K total bits
 * are valid (the last byte may have fewer than 8 valid bits).
 *
 * Vectorised: 64-byte main loop processes four 16-byte vectors per iter
 * with 4-wide ILP — 4 independent vcntq_u8, then a 3-level lane-wise
 * add tree (vaddq_u8 × 3) keeping the intermediate sums in u8 (max 32
 * at the root), then one vpaddlq_u8 widen-and-pair-add to u16 (max 64),
 * then one vaddq_u16 into the u16x8 accumulator.  vaddq_u8 (~4/cycle on
 * M-series) beats vpaddq_u8 (~2/cycle, shuffle-mux cost) for the same
 * data-reduction shape.  16-byte mop-up handles 1..3 leftover vectors,
 * scalar tail handles 0..15 full bytes plus the optional partial last
 * byte (K & 7 bits valid). */
static inline int popcount_K_right(const uint8_t *bm, int nbytes, int K) {
    (void)nbytes;   /* derivable from K; kept for API stability */
    PROF_TIC();
    int full_bytes = K >> 3;
    int partial_bits = K & 7;

    uint16x8_t acc_v = vdupq_n_u16(0);
    int b = 0;
    for (; b + 64 <= full_bytes; b += 64) {
        uint8x16_t v0 = vld1q_u8(bm + b);
        uint8x16_t v1 = vld1q_u8(bm + b + 16);
        uint8x16_t v2 = vld1q_u8(bm + b + 32);
        uint8x16_t v3 = vld1q_u8(bm + b + 48);
        uint8x16_t c0 = vcntq_u8(v0);
        uint8x16_t c1 = vcntq_u8(v1);
        uint8x16_t c2 = vcntq_u8(v2);
        uint8x16_t c3 = vcntq_u8(v3);
        /* 3-level lane-wise add tree, all in u8 (max 32 at the root). */
        uint8x16_t s01 = vaddq_u8(c0, c1);
        uint8x16_t s23 = vaddq_u8(c2, c3);
        uint8x16_t s   = vaddq_u8(s01, s23);
        acc_v = vaddq_u16(acc_v, vpaddlq_u8(s));
    }
    for (; b + 16 <= full_bytes; b += 16) {
        uint8x16_t v = vld1q_u8(bm + b);
        acc_v = vaddq_u16(acc_v, vpaddlq_u8(vcntq_u8(v)));
    }
    int K_right = (int)vaddvq_u16(acc_v);

    /* Scalar tail: remaining 0..15 full bytes. */
    for (; b < full_bytes; b++) {
        K_right += __builtin_popcount(bm[b]);
    }

    /* Optional partial byte holding the final (K & 7) bits. */
    if (partial_bits) {
        uint8_t valid_mask = (uint8_t)((1u << partial_bits) - 1);
        K_right += __builtin_popcount(bm[full_bytes] & valid_mask);
    }
    PROF_TOC(PROF_BU_POPCOUNT_K, K);
    return K_right;
}

/* ---------- expand_tab: per-mask-byte shuffle for tree_merge ----------
 *
 * expand_tab[m][k] for k in 0..7:
 *   - 0..7   means "take from left[count_zeros_in_m[0..k-1]]"
 *   - 8..15  means "take from right[count_ones_in_m[0..k-1]] + 8"
 *
 * Used as vqtbl1_u8 indices over a 16-byte vector built by
 *   vcombine(left[0..8), right[0..8)).
 *
 * expand_popcnt[m] = popcount(m) (= number of right bytes consumed). */
static uint8_t expand_tab[256][8] __attribute__((aligned(32)));
static uint8_t expand_popcnt[256]  __attribute__((aligned(64)));
static int expand_table_ready = 0;

static void init_expand_table(void) {
    if (expand_table_ready) return;
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
    expand_table_ready = 1;
}

/* ---------- tree_merge primitives ---------- */

/* Full merge: both inputs are dense byte buffers.
 *   left:  n_left  bytes  (consumed in order)
 *   right: n_right bytes  (consumed in order)
 *   bm:    K bits = bitmap selecting per output position
 *   out:   K bytes (written sequentially)
 * Requires n_left + n_right == K. */
static inline void tree_merge(const uint8_t *bm, int K,
                               const uint8_t *left,
                               const uint8_t *right,
                               uint8_t *out) {
    PROF_TIC();
    int lc = 0, rc = 0;
    int j = 0;
    /* 2x unroll (stride-16): two independent merges per iteration.
     * Adjacent 8-byte groups have independent loads / TBLs / stores so
     * OOO overlaps the second's load with the first's TBL.  Only
     * lc/rc carry a real dep, and that's short-latency integer add.
     * Mirrors node_full's stride-16 unroll in pivco_huffman_neon.c. */
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        uint8x8_t  L0    = vld1_u8(left + lc);
        uint8x8_t  R0    = vld1_u8(right + rc);
        uint8x16_t both0 = vcombine_u8(L0, R0);
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];
        rc += nr0;
        lc += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x8_t  L1    = vld1_u8(left + lc);
        uint8x8_t  R1    = vld1_u8(right + rc);
        uint8x16_t both1 = vcombine_u8(L1, R1);
        uint8x8_t  shuf1 = vld1_u8(expand_tab[m1]);
        uint8x8_t  o1    = vqtbl1_u8(both1, shuf1);
        vst1_u8(out + j + 8, o1);
        int nr1 = expand_popcnt[m1];
        rc += nr1;
        lc += (8 - nr1);
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m  = bm[j >> 3];
        uint8x8_t  L    = vld1_u8(left + lc);
        uint8x8_t  R    = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(L, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        int nr = expand_popcnt[m];
        rc += nr;
        lc += (8 - nr);
    }
    /* scalar tail (1..7 leftover).  out_buf has no SIMD-tail padding
     * guarantee (root call writes user buffer; child recursions pack
     * scratch buffers contiguously), so we keep this scalar. */
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
    PROF_TOC(PROF_BU_TREE_MERGE, K);
}

/* Broadcast-left merge: left is a constant symbol (the prefill leaf).
 * Used when this node's left child is a SKIP/prefill leaf. */
static inline void tree_merge_bcast_left(const uint8_t *bm, int K,
                                          uint8_t left_sym,
                                          const uint8_t *right,
                                          uint8_t *out) {
    PROF_TIC();
    int rc = 0;
    int j = 0;
    uint8x8_t Lbcast = vdup_n_u8(left_sym);
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        uint8x8_t  R0    = vld1_u8(right + rc);
        uint8x16_t both0 = vcombine_u8(Lbcast, R0);
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        rc += expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x8_t  R1    = vld1_u8(right + rc);
        uint8x16_t both1 = vcombine_u8(Lbcast, R1);
        uint8x8_t  shuf1 = vld1_u8(expand_tab[m1]);
        uint8x8_t  o1    = vqtbl1_u8(both1, shuf1);
        vst1_u8(out + j + 8, o1);
        rc += expand_popcnt[m1];
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x8_t  R    = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(Lbcast, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        rc += expand_popcnt[m];
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left_sym;
    }
    PROF_TOC(PROF_BU_TREE_MERGE_BCAST_LEFT, K);
}

/* Broadcast-right merge: right is a constant symbol (HALF_LEFT case). */
static inline void tree_merge_bcast_right(const uint8_t *bm, int K,
                                           const uint8_t *left,
                                           uint8_t right_sym,
                                           uint8_t *out) {
    PROF_TIC();
    int lc = 0;
    int j = 0;
    uint8x8_t Rbcast = vdup_n_u8(right_sym);
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        uint8x8_t  L0    = vld1_u8(left + lc);
        uint8x16_t both0 = vcombine_u8(L0, Rbcast);
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        lc += (8 - expand_popcnt[m0]);

        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x8_t  L1    = vld1_u8(left + lc);
        uint8x16_t both1 = vcombine_u8(L1, Rbcast);
        uint8x8_t  shuf1 = vld1_u8(expand_tab[m1]);
        uint8x8_t  o1    = vqtbl1_u8(both1, shuf1);
        vst1_u8(out + j + 8, o1);
        lc += (8 - expand_popcnt[m1]);
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x8_t  L    = vld1_u8(left + lc);
        uint8x16_t both = vcombine_u8(L, Rbcast);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        lc += (8 - expand_popcnt[m]);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left[lc++];
    }
    PROF_TOC(PROF_BU_TREE_MERGE_BCAST_RIGHT, K);
}

/* Both-leaves merge: BOTH inputs are constants.  Output is just
 * left_sym at bm=0 positions and right_sym at bm=1 positions.
 * SIMD: vtst+veor blend, no TBL needed. */
static inline void merge_both_const(const uint8_t *bm, int K,
                                     uint8_t left_sym, uint8_t right_sym,
                                     uint8_t *out) {
    PROF_TIC();
    uint8x8_t vleft  = vdup_n_u8(left_sym);
    uint8x8_t vdelta = vdup_n_u8(left_sym ^ right_sym);
    static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbits = vld1_u8(bit_pos_tab);

    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x8_t bits0 = vtst_u8(vdup_n_u8(bm[j >> 3]), vbits);
        uint8x8_t bits1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]), vbits);
        vst1_u8(out + j,     veor_u8(vleft, vand_u8(vdelta, bits0)));
        vst1_u8(out + j + 8, veor_u8(vleft, vand_u8(vdelta, bits1)));
    }
    for (; j + 8 <= K; j += 8) {
        uint8x8_t bits = vtst_u8(vdup_n_u8(bm[j >> 3]), vbits);
        vst1_u8(out + j, veor_u8(vleft, vand_u8(vdelta, bits)));
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left_sym;
    }
    PROF_TOC(PROF_BU_MERGE_BOTH_CONST, K);
}

/* ---------- D-bit flat decode (in-place sequential output) ----------
 *
 * Reads n*D packed bits, looks up each D-bit code in c2s, writes a byte
 * to out[i].  Output is dense / sequential (perfect for bottom-up).
 *
 * Reuses the vectorised D=2..8 flat-decode from pivco_huffman_neon.c
 * via a non-static wrapper, so the existing per-D unpackers don't get
 * duplicated. */
extern void pivco_huffman_flat_decode_direct_neon_(uint8_t *symbols, int n,
                                                    const uint8_t *bm, int D,
                                                    const uint8_t *c2s);
static inline void flat_decode_to_buffer(uint8_t *out, int n,
                                          const uint8_t *bm, int D,
                                          const uint8_t *c2s) {
    PROF_TIC();
    pivco_huffman_flat_decode_direct_neon_(out, n, bm, D, c2s);
    PROF_TOC(PROF_BU_FLAT_DECODE, n);
}

/* ---------- Recursive bottom-up decode ----------
 *
 * Each call writes K output bytes into out_buf.  scratch_top is the
 * arena pointer used for child output buffers; the parent reserves
 * left_buf+right_buf (K bytes) starting at scratch_top, and the
 * child recursions use scratch beyond that.
 *
 * Bitmap stream is consumed via *in_ptr in DFS order, matching the
 * top-down decoder's reading order. */
static void decode_subtree_bu(const pivco_huffman_table_t *table,
                               int16_t node_id, int K,
                               uint8_t *out_buf,
                               const uint8_t **in_ptr,
                               uint8_t *scratch_top)
{
    if (K == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];

    switch ((pivco_node_type_t)table->node_type[node_id]) {

    case PIVCO_NODE_SKIP:
        /* Prefilled leaf reached as a child of something other than
         * HALF_*.  Just write the prefill symbol K times. */
        { PROF_TIC();
          memset(out_buf, table->prefill_sym, (size_t)K);
          PROF_TOC(PROF_BU_LEAF_MEMSET, K); }
        return;

    case PIVCO_NODE_LEAF:
        /* Non-prefill leaf: K copies of node->symbol. */
        { PROF_TIC();
          memset(out_buf, (uint8_t)node->symbol, (size_t)K);
          PROF_TOC(PROF_BU_LEAF_MEMSET, K); }
        return;

    case PIVCO_NODE_INTERNAL_FLAT: {
        int D = table->flat_depth[node_id];
        int total_bytes = (K * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[node_id]];
        flat_decode_to_buffer(out_buf, K, bm, D, c2s);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        const pivco_tree_node_t *left_child  = &table->tree[node->left];
        const pivco_tree_node_t *right_child = &table->tree[node->right];
        merge_both_const(bm, K,
                          (uint8_t)left_child->symbol,
                          (uint8_t)right_child->symbol,
                          out_buf);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        /* Left child is the prefill leaf (SKIP), right child is internal
         * OR a non-prefill leaf.  Fast path: if right is itself a LEAF
         * (a single non-prefill symbol), the whole node degenerates to
         * merge_both_const — no buffer materialisation needed. */
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;

        if (table->node_type[node->right] == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const(bm, K,
                              table->prefill_sym,
                              (uint8_t)table->tree[node->right].symbol,
                              out_buf);
            return;
        }

        /* General case: recurse into right (only K_right bytes needed). */
        int K_right = popcount_K_right(bm, nbytes, K);
        uint8_t *right_buf = scratch_top;
        decode_subtree_bu(table, node->right, K_right,
                          right_buf, in_ptr, scratch_top + K_right);
        tree_merge_bcast_left(bm, K, table->prefill_sym, right_buf, out_buf);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        /* Right child is the prefill leaf (SKIP), left child is internal
         * OR a non-prefill leaf. */
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;

        if (table->node_type[node->left] == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const(bm, K,
                              (uint8_t)table->tree[node->left].symbol,
                              table->prefill_sym,
                              out_buf);
            return;
        }

        int K_right = popcount_K_right(bm, nbytes, K);
        int K_left = K - K_right;
        uint8_t *left_buf = scratch_top;
        decode_subtree_bu(table, node->left, K_left,
                          left_buf, in_ptr, scratch_top + K_left);
        tree_merge_bcast_right(bm, K, left_buf, table->prefill_sym, out_buf);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;

        /* If one child is a (non-prefill) LEAF, skip materialising
         * its buffer and use the broadcast variant of tree_merge.
         * BOTH children leaf would have been BOTH_LEAVES upstream
         * but is checked here defensively. */
        int left_kind  = table->node_type[node->left];
        int right_kind = table->node_type[node->right];
        if (left_kind == (uint8_t)PIVCO_NODE_LEAF
            && right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const(bm, K,
                              (uint8_t)table->tree[node->left].symbol,
                              (uint8_t)table->tree[node->right].symbol,
                              out_buf);
            return;
        }
        if (left_kind == (uint8_t)PIVCO_NODE_LEAF) {
            int K_right = popcount_K_right(bm, nbytes, K);
            uint8_t *right_buf = scratch_top;
            decode_subtree_bu(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top + K_right);
            tree_merge_bcast_left(bm, K,
                                   (uint8_t)table->tree[node->left].symbol,
                                   right_buf, out_buf);
            return;
        }
        if (right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            int K_right = popcount_K_right(bm, nbytes, K);
            int K_left = K - K_right;
            uint8_t *left_buf = scratch_top;
            decode_subtree_bu(table, node->left, K_left,
                              left_buf, in_ptr, scratch_top + K_left);
            tree_merge_bcast_right(bm, K, left_buf,
                                    (uint8_t)table->tree[node->right].symbol,
                                    out_buf);
            return;
        }

        /* General case: both children are internal — recurse into both. */
        int K_right = popcount_K_right(bm, nbytes, K);
        int K_left = K - K_right;

        uint8_t *left_buf  = scratch_top;
        uint8_t *right_buf = scratch_top + K_left;
        uint8_t *new_scratch_top = scratch_top + K;  /* K = K_left + K_right */

        decode_subtree_bu(table, node->left,  K_left,
                          left_buf,  in_ptr, new_scratch_top);
        decode_subtree_bu(table, node->right, K_right,
                          right_buf, in_ptr, new_scratch_top);
        tree_merge(bm, K, left_buf, right_buf, out_buf);
        return;
    }
    }
}

/* ---------- Entry point ---------- */

int pivco_huffman_decode_bu_neon(const uint8_t *in, size_t in_len,
                                  const pivco_huffman_table_t *table,
                                  uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    init_expand_table();

    (void)in_len;
    const uint8_t *ptr = in;
    const int N = PIVCO_BLOCK_SIZE;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root-is-leaf: fill everything with the single symbol. */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Fast path: root is BOTH_LEAVES — no recursion needed, just the
     * merge with two constants over the entire block. */
    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_BOTH_LEAVES) {
        int nbytes = bitmap_bytes(N);
        const uint8_t *bm = ptr;
        ptr += nbytes;
        const pivco_tree_node_t *left_child  = &table->tree[root->left];
        const pivco_tree_node_t *right_child = &table->tree[root->right];
        merge_both_const(bm, N,
                          (uint8_t)left_child->symbol,
                          (uint8_t)right_child->symbol,
                          symbols);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Root flat subtree: handled by the recursive case (INTERNAL_FLAT). */

    /* Scratch arena: bottom-up max usage is ~2N bytes for balanced trees,
     * ~3N for adversarial unbalanced.  Allocate 3N to be safe.  64B
     * aligned for SIMD-friendly cache lines. */
    static uint8_t scratch[3 * PIVCO_BLOCK_SIZE + 64]
        __attribute__((aligned(64)));

    decode_subtree_bu(table, table->tree_root, N,
                      symbols, &ptr, scratch);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
