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
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif
#include <string.h>

#ifdef PIVCO_HAS_NEON
#include <arm_neon.h>

/* Decode primitives (popcount_K_right, tree_merge, tree_merge_bcast_*,
 * merge_both_const, flat_decode_to_buffer) live in
 * pivco_huffman_primitives_neon.h.  This file calls them via the
 * unsuffixed-alias static-inline wrappers below — the actual SIMD
 * code is shared with the codec.c-compiled-as-NEON object library. */
#include "pivco_huffman_primitives_neon.h"
static inline int popcount_K_right(const uint8_t *bm, int nbytes, int K)
{ return popcount_K_right_neon(bm, nbytes, K); }

/* Unsuffixed-name aliases for the legacy call sites in this file. */
static inline void tree_merge(const uint8_t *bm, int K,
                               const uint8_t *left, const uint8_t *right,
                               uint8_t *out)
{ tree_merge_neon(bm, K, left, right, out); }

static inline void tree_merge_bcast_left(const uint8_t *bm, int K,
                                          uint8_t left_sym,
                                          const uint8_t *right,
                                          uint8_t *out)
{ tree_merge_bcast_left_neon(bm, K, left_sym, right, out); }

static inline void tree_merge_bcast_right(const uint8_t *bm, int K,
                                           const uint8_t *left,
                                           uint8_t right_sym,
                                           uint8_t *out)
{ tree_merge_bcast_right_neon(bm, K, left, right_sym, out); }

static inline void merge_both_const(const uint8_t *bm, int K,
                                     uint8_t left_sym, uint8_t right_sym,
                                     uint8_t *out)
{ merge_both_const_neon(bm, K, left_sym, right_sym, out); }

static inline void flat_decode_to_buffer(uint8_t *out, int n,
                                          const uint8_t *bm, int D,
                                          const uint8_t *c2s)
{ flat_decode_to_buffer_neon(out, n, bm, D, c2s); }

/* ---------- Recursive bottom-up decode ----------
 *
 * Each call writes K output bytes into out_buf.  scratch_top is the
 * arena pointer used for child output buffers; the parent reserves
 * left_buf+right_buf (K bytes) starting at scratch_top, and the
 * child recursions use scratch beyond that.
 *
 * Bitmap stream is consumed via *in_ptr in DFS order, matching the
 * top-down decoder's reading order. */
/* Read the per-non-flat-internal-node wire bytes that come AFTER any
 * optional K_right header: a 1-byte marker followed by either the raw
 * bitmap (marker == 0) or the FSE-compressed payload (marker > 0).
 *
 * Returns a pointer to the usable bitmap (either pointing into the
 * input stream for the raw path, or into the caller-provided scratch
 * for the FSE path).  Advances *in_ptr past the whole record.
 *
 * scratch must hold at least bitmap_bytes(K) bytes and stay live for
 * the entire span where the returned pointer is dereferenced (i.e.,
 * across any recursive decode_subtree_bu calls that follow). */
static inline const uint8_t *read_bitmap_bu(const uint8_t **in_ptr,
                                             int K,
                                             uint8_t *scratch)
{
    int nbytes = bitmap_bytes(K);
    uint8_t marker = **in_ptr;
    *in_ptr += 1;
    if (marker == 0) {
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        return bm;
    }
#ifdef PIVCO_HAS_FSE
    int t_id = marker & 0x7F;
    int xor_flag = (marker >> 7) & 1;
    uint16_t fse_len;
    memcpy(&fse_len, *in_ptr, 2);
    *in_ptr += 2;
    size_t out_len = 0;
    PROF_TIC();
    (void)pivco_fse_decompress(t_id, *in_ptr, fse_len,
                                scratch, (size_t)nbytes,
                                (size_t)nbytes, &out_len);
    PROF_TOC(PROF_FSE_DEC, (uint64_t)nbytes);
    *in_ptr += fse_len;
    if (xor_flag) pivco_fse_flip_bits(scratch, (size_t)nbytes);
    return scratch;
#else
    /* FSE not built in -- corrupt stream produced by an FSE build.
     * Best we can do without an error channel is consume `nbytes` and
     * return zeros; the caller will produce wrong output, the file
     * codec will catch the mismatch.  Don't fault. */
    (void)scratch;
    *in_ptr += nbytes;
    return *in_ptr - nbytes;
#endif
}

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
        /* No K_right header (kr_header_needed returns false). */
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_bu(in_ptr, K, bm_scratch);
        const pivco_tree_node_t *left_child  = &table->tree[node->left];
        const pivco_tree_node_t *right_child = &table->tree[node->right];
        merge_both_const(bm, K,
                          (uint8_t)left_child->symbol,
                          (uint8_t)right_child->symbol,
                          out_buf);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        int K_right = 0;
        int has_kr = kr_header_needed(table, node_id);
        if (has_kr) {
            uint16_t v; memcpy(&v, *in_ptr, 2); *in_ptr += 2;
            K_right = (int)v;
        }
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_bu(in_ptr, K, bm_scratch);

        if (table->node_type[node->right] == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const(bm, K,
                              table->prefill_sym,
                              (uint8_t)table->tree[node->right].symbol,
                              out_buf);
            return;
        }
        uint8_t *right_buf = scratch_top;
        decode_subtree_bu(table, node->right, K_right,
                          right_buf, in_ptr, scratch_top + K_right);
        tree_merge_bcast_left(bm, K, table->prefill_sym, right_buf, out_buf);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        int K_right = 0;
        int has_kr = kr_header_needed(table, node_id);
        if (has_kr) {
            uint16_t v; memcpy(&v, *in_ptr, 2); *in_ptr += 2;
            K_right = (int)v;
        }
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_bu(in_ptr, K, bm_scratch);

        if (table->node_type[node->left] == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const(bm, K,
                              (uint8_t)table->tree[node->left].symbol,
                              table->prefill_sym,
                              out_buf);
            return;
        }

        int K_left = K - K_right;
        uint8_t *left_buf = scratch_top;
        decode_subtree_bu(table, node->left, K_left,
                          left_buf, in_ptr, scratch_top + K_left);
        tree_merge_bcast_right(bm, K, left_buf, table->prefill_sym, out_buf);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        int K_right = 0;
        int has_kr = kr_header_needed(table, node_id);
        if (has_kr) {
            uint16_t v; memcpy(&v, *in_ptr, 2); *in_ptr += 2;
            K_right = (int)v;
        }
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_bu(in_ptr, K, bm_scratch);

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
            uint8_t *right_buf = scratch_top;
            decode_subtree_bu(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top + K_right);
            tree_merge_bcast_left(bm, K,
                                   (uint8_t)table->tree[node->left].symbol,
                                   right_buf, out_buf);
            return;
        }
        if (right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            int K_left = K - K_right;
            uint8_t *left_buf = scratch_top;
            decode_subtree_bu(table, node->left, K_left,
                              left_buf, in_ptr, scratch_top + K_left);
            tree_merge_bcast_right(bm, K, left_buf,
                                    (uint8_t)table->tree[node->right].symbol,
                                    out_buf);
            return;
        }

        int K_left = K - K_right;
        uint8_t *left_buf  = scratch_top;
        uint8_t *right_buf = scratch_top + K_left;
        uint8_t *new_scratch_top = scratch_top + K;

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
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = read_bitmap_bu(&ptr, N, bm_scratch);
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

    /* Scratch arena: each recursion level advances scratch_top by the
     * parent's K_right (or K_left); in the worst case (highly skewed
     * partitions, e.g. cat-image.jpg block 34 in 2026-05-13) the
     * accumulated offset can reach max_tree_depth × N bytes -- the sum
     * over all levels of K_right at that level can be as large as
     * MAX_CODE_LEN × N when each level passes nearly all bytes through
     * to the same child.  Size for the worst case.  64B aligned. */
    static uint8_t scratch[(PIVCO_MAX_CODE_LEN + 2) * PIVCO_BLOCK_SIZE + 64]
        __attribute__((aligned(64)));

    decode_subtree_bu(table, table->tree_root, N,
                      symbols, &ptr, scratch);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
