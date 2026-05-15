/* Bottom-up PIVCO-Huffman decoder (x86: SSE4.1 + optional AVX-512 VBMI2).
 *
 * See src/pivco_huffman_bu_neon.c for the algorithm description.  This
 * file is the x86 port.  Same recursive DFS structure, same expand_tab
 * layout — only the SIMD primitives differ.
 *
 * SSE 8-byte chunk per iter:
 *   pshufb on _mm_unpacklo_epi64(L8, R8) with expand_tab[mask] -> 8 bytes.
 *
 * AVX-512 VBMI2 (when available): 64-byte chunks via two
 *   _mm512_maskz_expandloadu_epi8 calls (one with mask, one with ~mask)
 *   ORed together.  ~0.023 ns/byte on Xeon Ice Lake+ per microbench.
 *
 * For now we use AVX-512 only for the LARGE merges (K >= 64) and fall
 * back to SSE for smaller merges; the per-call overhead of vpexpandb's
 * mask setup is amortised only when K is reasonably large. */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
#include <string.h>

#ifdef PIVCO_HAS_SSE4
/* Encode + BU decode primitives (SSE4.1 + AVX2 + AVX-512 fast paths),
 * including the lookup tables they index into.  This file's only role
 * after the Phase 4.2 refactor is the recursive BU tree walk in
 * decode_subtree_bu and the entry-point bookkeeping; all SIMD kernels
 * come from primitives_x86.h.  Phase 4.3 cutover will delete this
 * file entirely once codec.c-x86 takes over. */
#include "pivco_huffman_primitives_x86.h"

/* ---------- Recursive bottom-up decode ---------- */
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
        { PROF_TIC();
          memset(out_buf, table->prefill_sym, (size_t)K);
          PROF_TOC(PROF_BU_LEAF_MEMSET, K); }
        return;

    case PIVCO_NODE_LEAF:
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
        flat_decode_to_buffer_x86(out_buf, K, bm, D, c2s);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        const pivco_tree_node_t *left_child  = &table->tree[node->left];
        const pivco_tree_node_t *right_child = &table->tree[node->right];
        merge_both_const_x86(bm, K,
                          (uint8_t)left_child->symbol,
                          (uint8_t)right_child->symbol,
                          out_buf);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        int K_right = 0;
        if (kr_header_needed(table, node_id)) {
            uint16_t v; memcpy(&v, *in_ptr, 2); *in_ptr += 2;
            K_right = (int)v;
        }
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        if (table->node_type[node->right] == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const_x86(bm, K, table->prefill_sym,
                              (uint8_t)table->tree[node->right].symbol,
                              out_buf);
            return;
        }
        uint8_t *right_buf = scratch_top;
        decode_subtree_bu(table, node->right, K_right,
                          right_buf, in_ptr, scratch_top + K_right);
        tree_merge_bcast_left_x86(bm, K, table->prefill_sym, right_buf, out_buf);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        int K_right = 0;
        if (kr_header_needed(table, node_id)) {
            uint16_t v; memcpy(&v, *in_ptr, 2); *in_ptr += 2;
            K_right = (int)v;
        }
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        if (table->node_type[node->left] == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const_x86(bm, K,
                              (uint8_t)table->tree[node->left].symbol,
                              table->prefill_sym, out_buf);
            return;
        }
        int K_left = K - K_right;
        uint8_t *left_buf = scratch_top;
        decode_subtree_bu(table, node->left, K_left,
                          left_buf, in_ptr, scratch_top + K_left);
        tree_merge_bcast_right_x86(bm, K, left_buf, table->prefill_sym, out_buf);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        int K_right = 0;
        if (kr_header_needed(table, node_id)) {
            uint16_t v; memcpy(&v, *in_ptr, 2); *in_ptr += 2;
            K_right = (int)v;
        }
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        int left_kind  = table->node_type[node->left];
        int right_kind = table->node_type[node->right];

        if (left_kind == (uint8_t)PIVCO_NODE_LEAF
            && right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const_x86(bm, K,
                              (uint8_t)table->tree[node->left].symbol,
                              (uint8_t)table->tree[node->right].symbol,
                              out_buf);
            return;
        }
        if (left_kind == (uint8_t)PIVCO_NODE_LEAF) {
            uint8_t *right_buf = scratch_top;
            decode_subtree_bu(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top + K_right);
            tree_merge_bcast_left_x86(bm, K,
                                   (uint8_t)table->tree[node->left].symbol,
                                   right_buf, out_buf);
            return;
        }
        if (right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            int K_left = K - K_right;
            uint8_t *left_buf = scratch_top;
            decode_subtree_bu(table, node->left, K_left,
                              left_buf, in_ptr, scratch_top + K_left);
            tree_merge_bcast_right_x86(bm, K, left_buf,
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
        tree_merge_x86(bm, K, left_buf, right_buf, out_buf);
        return;
    }
    }
}

/* ---------- Entry point ---------- */
int pivco_huffman_decode_bu_x86(const uint8_t *in, size_t in_len,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    init_expand_table_x86();

    (void)in_len;
    const uint8_t *ptr = in;
    const int N = PIVCO_BLOCK_SIZE;
    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Fast path: root is BOTH_LEAVES. */
    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_BOTH_LEAVES) {
        int nbytes = bitmap_bytes(N);
        const uint8_t *bm = ptr;
        ptr += nbytes;
        const pivco_tree_node_t *left_child  = &table->tree[root->left];
        const pivco_tree_node_t *right_child = &table->tree[root->right];
        merge_both_const_x86(bm, N,
                          (uint8_t)left_child->symbol,
                          (uint8_t)right_child->symbol,
                          symbols);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* See pivco_huffman_bu_neon.c -- skewed Huffman trees can need up
     * to max_tree_depth × N bytes of scratch (3N is insufficient on
     * cat-image.jpg in 2026-05-13). */
    static uint8_t scratch[(PIVCO_MAX_CODE_LEN + 2) * PIVCO_BLOCK_SIZE + 64]
        __attribute__((aligned(64)));

    decode_subtree_bu(table, table->tree_root, N,
                      symbols, &ptr, scratch);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_SSE4 */
