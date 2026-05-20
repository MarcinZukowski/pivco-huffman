/* pivco_huffman_naive.c -- scalar P + S1 only.
 *
 * Implements pivco_huffman_decode_naive: a TD decoder built from
 * exactly two primitives:
 *
 *   P  -- scalar partition: split an N-element index array into left
 *         and right halves based on N bits read from the wire bitmap.
 *   S1 -- scalar scatter-symbol: write the leaf symbol to every
 *         output position named by the index array.
 *
 * No SIMD, no flat-subtree path (PIVCO_NODE_INTERNAL_FLAT), no
 * half-partition variant (PIVCO_NODE_HALF_*), no fused both-leaves
 * scatter (PIVCO_NODE_BOTH_LEAVES), no constant-prefill (no
 * PIVCO_NODE_SKIP).  Use pivco_huffman_build_table_naive to produce
 * a table whose node_type[] reflects this -- it forces every
 * internal node to PIVCO_NODE_INTERNAL_FULL and every leaf to
 * PIVCO_NODE_LEAF, and disables prefill + flat-subtree marking.
 *
 * Reads the same wire format the existing ph-td encoder produces
 * when fed a naively-classified table.  Maximally unoptimised --
 * exists for "what does the codec look like with all the
 * shape-specific specialisation turned off?" baselines.
 */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 *  The two primitives
 * ============================================================ */

/* P (partition).  src[0..n) -> left[0..*lc) and right[0..*rc), with
 * bit b at position k routing src[k] to right (b=1) or left (b=0).
 * Bitmap is LSB-first within bytes.  *lc + *rc == n on return. */
static inline void p_partition(const uint16_t *src, int n,
                                const uint8_t *bm,
                                uint16_t *left, uint16_t *right,
                                int *lc, int *rc)
{
    int li = 0, ri = 0;
    for (int k = 0; k < n; k++) {
        int b = (bm[k >> 3] >> (k & 7)) & 1;
        if (b) right[ri++] = src[k];
        else   left [li++] = src[k];
    }
    *lc = li;
    *rc = ri;
}

/* S1 (scatter-symbol).  symbols[indices[k]] = sym for k in [0, n). */
static inline void s1_scatter(uint8_t *symbols,
                                const uint16_t *indices, int n,
                                uint8_t sym)
{
    for (int k = 0; k < n; k++) symbols[indices[k]] = sym;
}

/* ============================================================
 *  Naive wire format
 *
 *  No FSE marker byte, no K_right header.  Just concatenated raw
 *  bitmaps in DFS-preorder of internal nodes (one bitmap per
 *  internal, ceil(n/8) bytes each).  Leaves contribute nothing.
 *  TD decode never needs K_right (it tracks indices directly) and
 *  this slice compiles without FSE, so both headers are pure
 *  overhead for the naive baseline.
 * ============================================================ */

static inline const uint8_t *read_bm(const uint8_t **in_ptr, int n)
{
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;
    return bm;
}

/* ============================================================
 *  Recursive descent
 * ============================================================ */

static void decode_node_naive(const pivco_huffman_table_t *table,
                                int16_t node_id,
                                uint16_t *indices, int n,
                                uint8_t *symbols,
                                const uint8_t **in_ptr,
                                uint16_t *workspace)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &table->tree[node_id];

    if (node->symbol >= 0) {
        s1_scatter(symbols, indices, n, (uint8_t)node->symbol);
        return;
    }

    const uint8_t *bm = read_bm(in_ptr, n);
    uint16_t *left  = workspace;
    uint16_t *right = workspace + n;
    int lc, rc;
    p_partition(indices, n, bm, left, right, &lc, &rc);

    /* Children claim fresh workspace past parent's left/right slot.
     * The left child finishes before the right starts, so they may
     * reuse the same start address. */
    decode_node_naive(table, node->left,  left,  lc, symbols,
                        in_ptr, workspace + 2 * n);
    decode_node_naive(table, node->right, right, rc, symbols,
                        in_ptr, workspace + 2 * n);
}

/* ============================================================
 *  Public entries
 * ============================================================ */

int pivco_huffman_decode_naive(const uint8_t *in, size_t in_len,
                                const pivco_huffman_table_t *table,
                                uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;

    const int N = PIVCO_BLOCK_SIZE;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Single-symbol data: root is a leaf, no wire bytes consumed. */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Naive wire format: no K_right header at the root either. */
    /* Identity index set for the root partition. */
    uint16_t *indices = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    if (!indices) return PIVCO_ERR_NULL;
    for (int k = 0; k < N; k++) indices[k] = (uint16_t)k;

    /* Workspace: ~2 * N * depth shorts upper bound.  Conservative
     * allocation (depth bounded by PIVCO_MAX_CODE_LEN). */
    size_t ws_n = (size_t)N * 2 * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *workspace = (uint16_t *)malloc(ws_n * sizeof(uint16_t));
    if (!workspace) { free(indices); return PIVCO_ERR_NULL; }

    decode_node_naive(table, table->tree_root, indices, N,
                        symbols, &ptr, workspace);

    if (consumed) *consumed = (size_t)(ptr - in);
    free(indices);
    free(workspace);
    return PIVCO_OK;
}

/* ============================================================
 *  Naive encoder -- mirrors the decoder.  Walks TD, emits a raw
 *  bitmap for each internal in DFS-preorder, partitions the
 *  current index set into left/right, recurses.  No FSE marker,
 *  no K_right header.
 * ============================================================ */

static void encode_node_naive(const pivco_huffman_table_t *table,
                                int16_t node_id, int depth,
                                const uint8_t *symbols,
                                const uint16_t *indices, int n,
                                uint8_t **out_ptr,
                                uint16_t *workspace)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;     /* leaf: nothing on the wire */

    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    memset(bm, 0, (size_t)nbytes);
    *out_ptr += nbytes;

    uint16_t *left  = workspace;
    uint16_t *right = workspace + n;
    int li = 0, ri = 0;
    for (int k = 0; k < n; k++) {
        uint16_t idx = indices[k];
        /* code_la is left-aligned; depth-d bit lives at position 15-d. */
        int b = (table->code_la[symbols[idx]] >> (15 - depth)) & 1;
        if (b) {
            bm[k >> 3] |= (uint8_t)(1u << (k & 7));
            right[ri++] = idx;
        } else {
            left[li++] = idx;
        }
    }

    encode_node_naive(table, node->left,  depth + 1, symbols,
                        left,  li, out_ptr, workspace + 2 * n);
    encode_node_naive(table, node->right, depth + 1, symbols,
                        right, ri, out_ptr, workspace + 2 * n);
}

int pivco_huffman_encode_naive(const uint8_t *symbols,
                                const pivco_huffman_table_t *table,
                                uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    const int N = PIVCO_BLOCK_SIZE;
    const pivco_tree_node_t *root = &table->tree[table->tree_root];
    if (root->symbol >= 0) { *out_len = 0; return PIVCO_OK; }

    uint16_t *indices = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    if (!indices) return PIVCO_ERR_NULL;
    for (int k = 0; k < N; k++) indices[k] = (uint16_t)k;

    size_t ws_n = (size_t)N * 2 * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *workspace = (uint16_t *)malloc(ws_n * sizeof(uint16_t));
    if (!workspace) { free(indices); return PIVCO_ERR_NULL; }

    uint8_t *out_ptr = out;
    encode_node_naive(table, table->tree_root, 0, symbols,
                        indices, N, &out_ptr, workspace);

    *out_len = (size_t)(out_ptr - out);
    free(indices);
    free(workspace);
    return PIVCO_OK;
}

/* ----- Naive table classifier -----
 *
 * Calls the standard build_table, then overrides node_type[] and
 * disables prefill + flat-subtree marking so the encoder emits a
 * uniform "internal-full / leaf" wire format that the naive decoder
 * can read.  Resulting trees use only P and S1 at decode time. */
int pivco_huffman_build_table_naive(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                     pivco_huffman_table_t *table)
{
    int rc = pivco_huffman_build_table(freq, table);
    if (rc != PIVCO_OK) return rc;

    table->prefill_node = -1;
    for (int16_t i = 0; i < table->tree_node_count; i++) {
        table->flat_depth[i]  = 0;
        table->flat_offset[i] = 0;
        const pivco_tree_node_t *n = &table->tree[i];
        table->node_type[i] = (uint8_t)((n->symbol >= 0)
                                            ? PIVCO_NODE_LEAF
                                            : PIVCO_NODE_INTERNAL_FULL);
    }
    return PIVCO_OK;
}
