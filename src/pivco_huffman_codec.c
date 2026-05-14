/* pivco_huffman_codec.c — unified encode + bottom-up decode.
 *
 * One source file, compiled once per backend tier (CMake passes
 * -DPIVCO_BACKEND_{SCALAR,NEON,X86,AVX512}).  The tree walk + dispatch
 * + wire format are identical across backends; the per-node SIMD work
 * lives in pivco_huffman_primitives_<backend>.h, selected via the
 * router pivco_huffman_primitives.h.
 *
 * Two responsibilities only:
 *
 *   1. Walk the Huffman tree (encode recursion + BU decode recursion).
 *   2. Read/write per-node wire records via pivco_huffman_wire.h.
 *
 * Everything backend-shaped (bitmap build, partition, flat-decode,
 * tree_merge, etc.) is a `prim_*` call.  No vector types here.
 *
 * The bottom-up decoder is the production path (top-down has been
 * parked).  Encode is shared.
 */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_wire.h"
#include "pivco_huffman_primitives.h"

#include <stdlib.h>
#include <string.h>

/* ---------- Backend → entry-point name ---------- */

#if defined(PIVCO_BACKEND_SCALAR)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_scalar
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_scalar
#elif defined(PIVCO_BACKEND_NEON)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_neon
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_neon
#elif defined(PIVCO_BACKEND_X86)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_x86
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_x86
#elif defined(PIVCO_BACKEND_AVX512)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_avx512
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_avx512
#else
#  error "pivco_huffman_codec.c needs PIVCO_BACKEND_{SCALAR,NEON,X86,AVX512}"
#endif

/* ---------- Encode tree walk ---------- *
 *
 * DFS, in-order: emit the partition bitmap at this node, then recurse
 * left, then right.  At each non-flat internal node, `codes_la[0..n)`
 * holds the surviving codes (left-aligned: top bit = current depth's
 * partition).  After partition, left stays in codes_la[0..n_left) and
 * right goes to tmp[0..n_right); the shift-by-1 in the primitive lets
 * the next recursion read bit 15 again. */

static void codec_encode_node(const pivco_huffman_table_t *table,
                               int16_t node_id,
                               uint16_t *codes_la, int n,
                               uint8_t **out_ptr,
                               uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;  /* leaf — nothing to emit */

    /* Flat-subtree fast path. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        prim_pack_dN(*out_ptr, codes_la, n, D);
        *out_ptr += total_bytes;
        return;
    }

    /* Non-flat internal node: reserve wire-format slots, partition,
     * commit K_right, recurse. */
    uint8_t *kr_slot = wire_reserve_kr_header(table, node_id, out_ptr);
    (void)wire_reserve_fse_marker(out_ptr);

    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    *out_ptr += nbytes;

    int n_right = prim_encode_partition(codes_la, n, bm, tmp);
    int n_left  = n - n_right;
    wire_commit_kr_header(kr_slot, n_right);

    codec_encode_node(table, node->left,  codes_la,         n_left,
                       out_ptr, tmp + n_right);
    codec_encode_node(table, node->right, tmp,              n_right,
                       out_ptr, tmp + n_right);
}

int CODEC_ENCODE_ENTRY(const uint8_t *symbols,
                       const pivco_huffman_table_t *table,
                       uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    const int N = PIVCO_BLOCK_SIZE;

    /* Per-block left-aligned codes.  Built once per block via
     * prim_enc_init (a gather from table->code_la[symbols[i]]). */
    uint16_t codes_la[PIVCO_BLOCK_SIZE];
    prim_enc_init(codes_la, N, symbols, table->code_la);

    /* Scratch arena for the right-half partitions at each recursion
     * level.  Worst-case usage on a skewed tree: every recursion
     * pushes ~n_right elements which can be near n if the partition
     * is one-sided.  Size for (MAX_CODE_LEN+2) * BLOCK_SIZE — matches
     * the bound used elsewhere in the codec. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)malloc(tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;

    uint8_t *ptr = out;
    codec_encode_node(table, table->tree_root, codes_la, N, &ptr, tmp);

    free(tmp);
    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- Bottom-up decode tree walk ---------- *
 *
 * Bottom-up: each call decodes a subtree into a contiguous K-byte
 * output buffer.  Internal nodes recurse into their children (which
 * write to scratch arenas), then merge per the bitmap.  The flat-
 * subtree fast path bypasses recursion entirely.
 *
 * Dispatch on node_type, computed at build-table time:
 *
 *   SKIP            — prefilled leaf, memset prefill_sym
 *   LEAF            — non-prefill leaf, memset node->symbol
 *   INTERNAL_FLAT   — packed-bits flat decode into out_buf
 *   BOTH_LEAVES     — both children leaves, merge_both_const directly
 *   HALF_RIGHT      — left child is the prefilled leaf, recurse right
 *   HALF_LEFT       — right child is the prefilled leaf, recurse left
 *   INTERNAL_FULL   — general merge: recurse both, tree_merge
 *
 * `scratch_top` is the arena pointer for child output buffers; each
 * caller bumps it past its own K bytes when calling further down. */

static void codec_decode_subtree(const pivco_huffman_table_t *table,
                                   int16_t node_id, int K,
                                   uint8_t *out_buf,
                                   const uint8_t **in_ptr,
                                   uint8_t *scratch_top)
{
    if (K == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];

    switch ((pivco_node_type_t)table->node_type[node_id]) {

    case PIVCO_NODE_SKIP:
        memset(out_buf, table->prefill_sym, (size_t)K);
        return;

    case PIVCO_NODE_LEAF:
        memset(out_buf, (uint8_t)node->symbol, (size_t)K);
        return;

    case PIVCO_NODE_INTERNAL_FLAT: {
        int D = table->flat_depth[node_id];
        int total_bytes = (K * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s =
            &table->flat_code_to_sym[table->flat_offset[node_id]];
        prim_flat_decode_to_buffer(out_buf, K, bm, D, c2s);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        /* No K_right header (kr_header_needed returns false). */
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);
        prim_merge_both_const(bm, K,
                               (uint8_t)table->tree[node->left].symbol,
                               (uint8_t)table->tree[node->right].symbol,
                               out_buf);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        /* If the right child is also a leaf, no recursion needed:
         * merge_both_const with prefill_sym on the left and the
         * right child's symbol on the right. */
        if (table->node_type[node->right] == (uint8_t)PIVCO_NODE_LEAF) {
            prim_merge_both_const(bm, K, table->prefill_sym,
                                   (uint8_t)table->tree[node->right].symbol,
                                   out_buf);
            return;
        }
        uint8_t *right_buf = scratch_top;
        codec_decode_subtree(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top + K_right);
        prim_tree_merge_bcast_left(bm, K, table->prefill_sym,
                                    right_buf, out_buf);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        if (table->node_type[node->left] == (uint8_t)PIVCO_NODE_LEAF) {
            prim_merge_both_const(bm, K,
                                   (uint8_t)table->tree[node->left].symbol,
                                   table->prefill_sym, out_buf);
            return;
        }
        int K_left = K - K_right;
        uint8_t *left_buf = scratch_top;
        codec_decode_subtree(table, node->left, K_left,
                              left_buf, in_ptr, scratch_top + K_left);
        prim_tree_merge_bcast_right(bm, K, left_buf,
                                     table->prefill_sym, out_buf);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        int left_kind  = table->node_type[node->left];
        int right_kind = table->node_type[node->right];

        if (left_kind == (uint8_t)PIVCO_NODE_LEAF
            && right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            prim_merge_both_const(bm, K,
                                   (uint8_t)table->tree[node->left].symbol,
                                   (uint8_t)table->tree[node->right].symbol,
                                   out_buf);
            return;
        }
        if (left_kind == (uint8_t)PIVCO_NODE_LEAF) {
            uint8_t *right_buf = scratch_top;
            codec_decode_subtree(table, node->right, K_right,
                                  right_buf, in_ptr, scratch_top + K_right);
            prim_tree_merge_bcast_left(bm, K,
                                        (uint8_t)table->tree[node->left].symbol,
                                        right_buf, out_buf);
            return;
        }
        if (right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            int K_left = K - K_right;
            uint8_t *left_buf = scratch_top;
            codec_decode_subtree(table, node->left, K_left,
                                  left_buf, in_ptr, scratch_top + K_left);
            prim_tree_merge_bcast_right(bm, K, left_buf,
                                         (uint8_t)table->tree[node->right].symbol,
                                         out_buf);
            return;
        }

        /* General case: both children non-leaf.  Recurse into both
         * with disjoint scratch slices, then tree_merge. */
        int K_left = K - K_right;
        uint8_t *left_buf  = scratch_top;
        uint8_t *right_buf = scratch_top + K_left;
        uint8_t *new_scratch_top = scratch_top + K;

        codec_decode_subtree(table, node->left,  K_left,
                              left_buf,  in_ptr, new_scratch_top);
        codec_decode_subtree(table, node->right, K_right,
                              right_buf, in_ptr, new_scratch_top);
        prim_tree_merge(bm, K, left_buf, right_buf, out_buf);
        return;
    }
    }
}

int CODEC_DECODE_ENTRY(const uint8_t *in, size_t in_len,
                       const pivco_huffman_table_t *table,
                       uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;

    const int N = PIVCO_BLOCK_SIZE;
    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root-is-leaf: fill everything with the single symbol. */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Scratch arena.  Worst case at a heavily-skewed node, the
     * partition is one-sided so a single recursion can consume up to
     * N bytes.  Bounded by (MAX_CODE_LEN+2) * N. */
    static __thread uint8_t scratch[(size_t)PIVCO_BLOCK_SIZE *
                                     (PIVCO_MAX_CODE_LEN + 2)];

    const uint8_t *ptr = in;
    codec_decode_subtree(table, table->tree_root, N,
                          symbols, &ptr, scratch);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}
