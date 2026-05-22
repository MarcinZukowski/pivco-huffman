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
#include "pivco_prof.h"

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
    PROF_TIC();
    int li = 0, ri = 0;
    for (int k = 0; k < n; k++) {
        int b = (bm[k >> 3] >> (k & 7)) & 1;
        if (b) right[ri++] = src[k];
        else   left [li++] = src[k];
    }
    *lc = li;
    *rc = ri;
    PROF_TOC(PROF_P_PARTITION, n);
}

/* S1 (scatter-symbol).  symbols[indices[k]] = sym for k in [0, n). */
static inline void s1_scatter(uint8_t *symbols,
                                const uint16_t *indices, int n,
                                uint8_t sym)
{
    PROF_TIC();
    for (int k = 0; k < n; k++) symbols[indices[k]] = sym;
    PROF_TOC(PROF_S1_SCATTER, n);
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

/* ============================================================
 *  Scalar-opt decoder: all tree-shape optimisations on, every
 *  primitive in scalar C.
 *
 *  Reads the FULL ph wire format produced by pivco_huffman_encode_neon
 *  (FSE marker byte + bitmap, plus K_right header before INTERNAL_FULL
 *  and HALF_* nodes, plus N*D packed bits at INTERNAL_FLAT nodes).
 *  Handles every pivco_node_type_t case the NEON decoder handles --
 *  prefill memset, scatter, both-leaves fused write, half-partition,
 *  flat-subtree D-bit scatter, full partition + recurse.  The
 *  difference from the production NEON decoder is purely in the
 *  primitive implementations: scalar loops, no NEON intrinsics, no
 *  TBL shuffle tables.
 * ============================================================ */

/* ----- additional scalar primitives ----- */

/* Half-partition right: emit only bit=1 indices.  Branchless --
 * always store src[k] to right[ri] and conditionally advance ri.
 * The conditional-store form (write only when bit=1) measured 1.65x
 * slower than p_partition on M4 because the compiler emits a branchy
 * loop with hard-to-predict skips at 50/50 splits. */
static inline int p_half_right(const uint16_t *src, int n,
                                 const uint8_t *bm, uint16_t *right)
{
    PROF_TIC();
    int ri = 0;
    for (int k = 0; k < n; k++) {
        int b = (bm[k >> 3] >> (k & 7)) & 1;
        right[ri] = src[k];
        ri += b;
    }
    PROF_TOC(PROF_P_HALF_RIGHT, n);
    return ri;
}

/* Half-partition left: emit only bit=0 indices.  Branchless and
 * in-place (the discarded src[k] for bit=1 is harmlessly overwritten
 * by the next kept element). */
static inline int p_half_left(uint16_t *src, int n, const uint8_t *bm)
{
    PROF_TIC();
    int li = 0;
    for (int k = 0; k < n; k++) {
        int b = (bm[k >> 3] >> (k & 7)) & 1;
        src[li] = src[k];
        li += !b;
    }
    PROF_TOC(PROF_P_HALF_LEFT, n);
    return li;
}

/* Both-leaves fused scatter (S2): per-bit choice between two
 * symbol broadcasts.  Saves the partition step entirely. */
static inline void s2_scatter_both(uint8_t *symbols,
                                    const uint16_t *indices, int n,
                                    const uint8_t *bm,
                                    uint8_t sym0, uint8_t sym1)
{
    PROF_TIC();
    for (int k = 0; k < n; k++) {
        int bit = (bm[k >> 3] >> (k & 7)) & 1;
        symbols[indices[k]] = bit ? sym1 : sym0;
    }
    PROF_TOC(PROF_S2_SCATTER_BOTH, n);
}

/* Flat scatter (SFx): read N D-bit codes packed in bm, look up
 * each via c2s, scatter to symbols[indices[k]].  Folds D levels
 * of internal-node partition + scatter into one wire record. */
static inline void sfx_scatter_flat(uint8_t *symbols,
                                     const uint16_t *indices, int n,
                                     const uint8_t *bm, int D,
                                     const uint8_t *c2s)
{
    PROF_TIC();
    uint32_t mask = (1u << D) - 1u;
    for (int k = 0; k < n; k++) {
        uint64_t off  = (uint64_t)k * (uint64_t)D;
        uint64_t byte = off >> 3;
        uint32_t shft = off & 7u;
        uint32_t w    = (uint32_t)bm[byte]
                      | ((uint32_t)bm[byte + 1] <<  8)
                      | ((uint32_t)bm[byte + 2] << 16);
        if (D > 16) w |= (uint32_t)bm[byte + 3] << 24;
        symbols[indices[k]] = c2s[(w >> shft) & mask];
    }
    PROF_TOC(PROF_SFX_SCATTER_FLAT, n);
}

/* Read bitmap from the standard ph wire (skip 1-byte FSE marker;
 * no FSE compiled in this slice, marker is always 0). */
static inline const uint8_t *read_bm_full(const uint8_t **in_ptr, int n)
{
    int nbytes = bitmap_bytes(n);
    (*in_ptr)++;                          /* FSE marker */
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;
    return bm;
}

/* ----- recursive dispatch ----- */

static void decode_node_scalar_opt(const pivco_huffman_table_t *table,
                                     int16_t node_id,
                                     uint16_t *indices, int n,
                                     uint8_t *symbols,
                                     const uint8_t **in_ptr,
                                     uint16_t *ws)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &table->tree[node_id];

    switch ((pivco_node_type_t)table->node_type[node_id]) {
    case PIVCO_NODE_SKIP:
        return;     /* prefilled -- memset already covered these */

    case PIVCO_NODE_LEAF:
        s1_scatter(symbols, indices, n, (uint8_t)node->symbol);
        return;

    case PIVCO_NODE_INTERNAL_FLAT: {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s =
            &table->flat_code_to_sym[table->flat_offset[node_id]];
        sfx_scatter_flat(symbols, indices, n, bm, D, c2s);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        /* No K_right header at BOTH_LEAVES. */
        const uint8_t *bm = read_bm_full(in_ptr, n);
        const pivco_tree_node_t *lc = &table->tree[node->left];
        const pivco_tree_node_t *rc = &table->tree[node->right];
        s2_scatter_both(symbols, indices, n, bm,
                          (uint8_t)lc->symbol, (uint8_t)rc->symbol);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        const uint8_t *bm = read_bm_full(in_ptr, n);
        int n_right = p_half_right(indices, n, bm, ws);
        decode_node_scalar_opt(table, node->right, ws, n_right,
                                 symbols, in_ptr, ws + n_right);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        const uint8_t *bm = read_bm_full(in_ptr, n);
        int n_left = p_half_left(indices, n, bm);
        decode_node_scalar_opt(table, node->left, indices, n_left,
                                 symbols, in_ptr, ws);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        const uint8_t *bm = read_bm_full(in_ptr, n);
        uint16_t *left  = ws;
        uint16_t *right = ws + n;
        int lc, rc;
        p_partition(indices, n, bm, left, right, &lc, &rc);
        decode_node_scalar_opt(table, node->left,  left,  lc,
                                 symbols, in_ptr, ws + 2 * n);
        decode_node_scalar_opt(table, node->right, right, rc,
                                 symbols, in_ptr, ws + 2 * n);
        return;
    }
    }
}

int pivco_huffman_decode_scalar_opt(const uint8_t *in, size_t in_len,
                                     const pivco_huffman_table_t *table,
                                     uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;

    const int N = PIVCO_BLOCK_SIZE;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root is a leaf (single-symbol data). */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Root is a flat subtree: N*D packed bits, scatter directly. */
    if (table->flat_depth[table->tree_root] >= 2) {
        int D = table->flat_depth[table->tree_root];
        int total_bytes = (N * D + 7) >> 3;
        const uint8_t *bm = ptr;
        ptr += total_bytes;
        const uint8_t *c2s =
            &table->flat_code_to_sym[table->flat_offset[table->tree_root]];
        /* Identity scatter == direct write: symbols[k] = c2s[code_k]. */
        uint32_t mask = (1u << D) - 1u;
        for (int k = 0; k < N; k++) {
            uint64_t off  = (uint64_t)k * (uint64_t)D;
            uint64_t byte = off >> 3;
            uint32_t shft = off & 7u;
            uint32_t w    = (uint32_t)bm[byte]
                          | ((uint32_t)bm[byte + 1] <<  8)
                          | ((uint32_t)bm[byte + 2] << 16);
            if (D > 16) w |= (uint32_t)bm[byte + 3] << 24;
            symbols[k] = c2s[(w >> shft) & mask];
        }
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Prefill output with the most-frequent symbol; PIVCO_NODE_SKIP
     * leaves rely on the memset already having covered their slots. */
    memset(symbols, table->prefill_sym, (size_t)N);

    /* Identity indices for the root partition. */
    uint16_t *indices = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    if (!indices) return PIVCO_ERR_NULL;
    for (int k = 0; k < N; k++) indices[k] = (uint16_t)k;

    size_t ws_n = (size_t)N * 2 * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *ws = (uint16_t *)malloc(ws_n * sizeof(uint16_t));
    if (!ws) { free(indices); return PIVCO_ERR_NULL; }

    decode_node_scalar_opt(table, table->tree_root, indices, N,
                             symbols, &ptr, ws);

    *consumed = (size_t)(ptr - in);
    free(indices);
    free(ws);
    return PIVCO_OK;
}

/* ============================================================
 *  Scalar-opt encoder: mirror of encode_node_neon but pure C.
 *  Produces the FULL ph wire format (FSE marker + K_right header
 *  + bitmap, or N*D packed bits for flat subtrees).  Required
 *  on hosts without NEON so the scalar-opt decoder has data to
 *  read; equivalent output to the NEON encoder bit-for-bit.
 * ============================================================ */

static void encode_node_scalar_opt(const pivco_huffman_table_t *table,
                                     int16_t node_id,
                                     uint16_t *codes_la, int n,
                                     int depth,
                                     uint8_t **out_ptr,
                                     uint16_t *tmp)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;     /* leaf -- no wire bytes */

    pivco_node_type_t nt = (pivco_node_type_t)table->node_type[node_id];

    /* INTERNAL_FLAT: pack N D-bit local codes (no marker, no K_right). */
    if (nt == PIVCO_NODE_INTERNAL_FLAT) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        uint8_t *out = *out_ptr;
        memset(out, 0, (size_t)total_bytes);
        *out_ptr += total_bytes;
        uint32_t mask = (1u << D) - 1u;
        uint64_t buf = 0; int bits_in_buf = 0; int byte_idx = 0;
        for (int k = 0; k < n; k++) {
            uint32_t local = (uint32_t)(codes_la[k] >> (16 - depth - D)) & mask;
            buf |= ((uint64_t)local) << bits_in_buf;
            bits_in_buf += D;
            while (bits_in_buf >= 8) {
                out[byte_idx++] = (uint8_t)(buf & 0xff);
                buf >>= 8;
                bits_in_buf -= 8;
            }
        }
        if (bits_in_buf > 0) {
            out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
        }
        return;
    }

    /* K_right header (2-byte uint16 LE) -- only for the wire-format
     * cases that need it.  See kr_header_needed(). */
    int need_kr = kr_header_needed(table, node_id);
    uint8_t *kr_hdr = NULL;
    if (need_kr) { kr_hdr = *out_ptr; *out_ptr += KR_HEADER_BYTES; }

    /* FSE marker byte (always 0 -- this slice has no FSE). */
    *(*out_ptr)++ = 0;

    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    memset(bm, 0, (size_t)nbytes);
    *out_ptr += nbytes;

    /* Build bitmap + partition codes_la by the depth-bit.  bit=1 ->
     * right (moved to tmp); bit=0 -> left (in-place compaction). */
    int li = 0, ri = 0;
    const int shift = 15 - depth;
    for (int k = 0; k < n; k++) {
        int b = (codes_la[k] >> shift) & 1;
        if (b) {
            bm[k >> 3] |= (uint8_t)(1u << (k & 7));
            tmp[ri++] = codes_la[k];
        } else {
            codes_la[li++] = codes_la[k];
        }
    }
    int n_left = li, n_right = ri;

    if (need_kr) {
        kr_hdr[0] = (uint8_t)(n_right & 0xFF);
        kr_hdr[1] = (uint8_t)((n_right >> 8) & 0xFF);
    }

    switch (nt) {
    case PIVCO_NODE_BOTH_LEAVES:
        /* Both children are leaves -- bitmap alone tells the decoder
         * which symbol goes where; no recursion. */
        return;
    case PIVCO_NODE_HALF_RIGHT:
        /* Left child is the prefilled leaf (skip).  Recurse right. */
        encode_node_scalar_opt(table, node->right, tmp, n_right, depth + 1,
                                 out_ptr, tmp + n_right);
        return;
    case PIVCO_NODE_HALF_LEFT:
        /* Right child is the prefilled leaf.  Recurse left
         * (already compacted in-place at codes_la[0..n_left)). */
        encode_node_scalar_opt(table, node->left, codes_la, n_left, depth + 1,
                                 out_ptr, tmp);
        return;
    case PIVCO_NODE_INTERNAL_FULL:
    default:
        encode_node_scalar_opt(table, node->left,  codes_la, n_left,  depth + 1,
                                 out_ptr, tmp + n_right);
        encode_node_scalar_opt(table, node->right, tmp,      n_right, depth + 1,
                                 out_ptr, tmp + n_right);
        return;
    }
}

int pivco_huffman_encode_scalar_opt(const uint8_t *symbols,
                                      const pivco_huffman_table_t *table,
                                      uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    const int N = PIVCO_BLOCK_SIZE;
    const pivco_tree_node_t *root = &table->tree[table->tree_root];
    if (root->symbol >= 0) { *out_len = 0; return PIVCO_OK; }

    uint16_t *codes_la = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    if (!codes_la) return PIVCO_ERR_NULL;
    for (int k = 0; k < N; k++) codes_la[k] = table->code_la[symbols[k]];

    /* tmp workspace: enough for the recursion's worst case. */
    size_t tmp_n = (size_t)N * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)malloc(tmp_n * sizeof(uint16_t));
    if (!tmp) { free(codes_la); return PIVCO_ERR_NULL; }

    uint8_t *out_ptr = out;
    encode_node_scalar_opt(table, table->tree_root, codes_la, N, 0,
                             &out_ptr, tmp);

    *out_len = (size_t)(out_ptr - out);
    free(codes_la);
    free(tmp);
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
