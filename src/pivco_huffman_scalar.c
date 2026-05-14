#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_wire.h"
#include <stdlib.h>
#include <string.h>

/* Pack n*D bits, LSB-first within each byte: D bits per element, local
   code = codes[indices[i]] & ((1<<D)-1).  Used for the flat-subtree
   fast path. */
static inline void pack_D_bits_scalar(uint8_t *out, int n, int D,
                                       const uint16_t *indices,
                                       const uint16_t *codes)
{
    uint32_t mask = (1u << D) - 1;
    uint64_t buf = 0;
    int bits_in_buf = 0;
    int byte_idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t local = (uint32_t)codes[indices[i]] & mask;
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
}

/* Extract D bits at bit position `bit_pos`. */
static inline uint32_t extract_D_bits_scalar(const uint8_t *in,
                                              int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3;
    int bit_off  = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8)  val |= ((uint32_t)in[byte_idx + 1]) << 8;
    if (bit_off + D > 16) val |= ((uint32_t)in[byte_idx + 2]) << 16;
    return (val >> bit_off) & ((1u << D) - 1);
}

/* Unpack n D-bit codes from bm, look up in c2s, scatter to
 * symbols[indices[i]].  Used by decode_node for flat-subtree dispatch. */
static inline void flat_decode_scatter_scalar(uint8_t *symbols,
                                               const uint16_t *indices, int n,
                                               const uint8_t *bm, int D,
                                               const uint8_t *c2s)
{
    for (int i = 0; i < n; i++) {
        uint32_t code = extract_D_bits_scalar(bm, i * D, D);
        symbols[indices[i]] = c2s[code];
    }
}

/* ---------- PIVCO Huffman Encode (Scalar, Tree-Walk) ----------
 *
 * DFS tree walk. At each internal node with n indices:
 *   - Write n code bits (one per index) as ceil(n/8) bytes
 *   - Partition indices into left (bit=0) and right (bit=1)
 *   - Recurse left, then right
 * At each leaf: nothing to write (symbol is known from tree position)
 *
 * The encoder knows each symbol's code, so it extracts the correct
 * bit at each depth from the precomputed code.
 */

static void encode_node(const pivco_huffman_table_t *table,
                         int16_t node_id,
                         uint16_t *indices, int n,
                         int depth,
                         const uint16_t *codes, const uint8_t *lens,
                         uint8_t **out_ptr,
                         uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        /* Leaf — nothing to write, symbol known from tree */
        return;
    }

    /* Flat-subtree fast path: emit n*D packed bits instead of D levels
       of bitmaps.  Detected at build_table time. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        uint8_t *out = *out_ptr;
        if (total_bytes > 0) out[total_bytes - 1] = 0;
        pack_D_bits_scalar(out, n, D, indices, codes);
        *out_ptr += total_bytes;
        return;
    }

    /* Per-node wire record: optional K_right header + FSE marker + bitmap.
     * The wire-format helpers in pivco_huffman_wire.h are the single
     * source of truth across all backends.  v0: scalar reference always
     * emits marker=0 (FSE attempt lives only in SIMD encoders). */
    uint8_t *kr_slot  = wire_reserve_kr_header(table, node_id, out_ptr);
    (void)wire_reserve_fse_marker(out_ptr);

    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    memset(bm, 0, (size_t)nbytes);

    for (int j = 0; j < n; j++) {
        int idx = indices[j];
        int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
        if (bit) bitmap_set(bm, j);
    }
    *out_ptr += nbytes;

    /* Partition indices into left (bit=0) and right (bit=1) */
    int n_right = 0;
    int n_left = 0;

    /* Left (bit=0) stays in place conceptually; right goes to tmp */
    for (int j = 0; j < n; j++) {
        if (bitmap_get(bm, j)) {
            tmp[n_right++] = indices[j];
        } else {
            /* In-place: safe because n_left <= j always */
            indices[n_left++] = indices[j];
        }
    }

    wire_commit_kr_header(kr_slot, n_right);

    /* Recurse left, then right.
       Left scratch starts at tmp + n_right to avoid clobbering
       the right partition stored in tmp[0..n_right-1].
       After left returns, right reuses tmp + n_right as scratch. */
    encode_node(table, node->left, indices, n_left,
                depth + 1, codes, lens, out_ptr, tmp + n_right);
    encode_node(table, node->right, tmp, n_right,
                depth + 1, codes, lens, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_scalar(const uint8_t *symbols,
                                const pivco_huffman_table_t *table,
                                uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    const int N = PIVCO_BLOCK_SIZE;

    /* Precompute codes and lengths for this block */
    uint16_t codes[PIVCO_BLOCK_SIZE];
    uint8_t  lens[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) {
        codes[i] = table->code[symbols[i]];
        lens[i]  = table->code_len[symbols[i]];
    }

    /* Initial index array: 0, 1, 2, ..., N-1 */
    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    /* See pivco_huffman_neon.c for tmp sizing rationale. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)malloc(tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;
    uint8_t *ptr = out;

    encode_node(table, table->tree_root, indices, N,
                0, codes, lens, &ptr, tmp);

    free(tmp);
    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- PIVCO Huffman Decode (Scalar, Tree-Walk) ----------
 *
 * DFS tree walk. At each internal node with n indices:
 *   - Read n code bits from stream
 *   - Partition indices into left (bit=0) and right (bit=1)
 *   - Recurse left, then right
 * At each leaf: scatter-write the leaf symbol to all n indices
 */

static void decode_node(const pivco_huffman_table_t *table,
                         int16_t node_id,
                         uint16_t *indices, int n,
                         uint8_t *symbols,
                         const uint8_t **in_ptr,
                         uint16_t *tmp,
                         int16_t skip_node)
{
    if (n == 0) return;
    if (node_id == skip_node) return;  /* prefilled by memset */

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        uint8_t sym = (uint8_t)node->symbol;
        for (int j = 0; j < n; j++)
            symbols[indices[j]] = sym;
        return;
    }

    /* Flat-subtree fast path. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[node_id]];
        flat_decode_scatter_scalar(symbols, indices, n, bm, D, c2s);
        return;
    }

    /* Per-node wire record.  TD doesn't use the K_right value (it
     * recomputes the partition inline), so the read result is discarded.
     * wire_read_bitmap returns either the raw stream pointer (marker==0)
     * or bm_scratch (marker!=0, FSE path). */
    (void)wire_read_kr_header(table, node_id, in_ptr);

    uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
    const uint8_t *bm = wire_read_bitmap(in_ptr, n, bm_scratch);

    /* Check if children are leaves for stage fusion */
    const pivco_tree_node_t *left_child  = &table->tree[node->left];
    const pivco_tree_node_t *right_child = &table->tree[node->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf
        && node->left != skip_node && node->right != skip_node) {
        uint8_t syms[2] = {(uint8_t)left_child->symbol,
                           (uint8_t)right_child->symbol};
        for (int j = 0; j < n; j++)
            symbols[indices[j]] = syms[bitmap_get(bm, j)];
        return;
    }

    if (left_leaf && node->left == skip_node) {
        /* Left is prefilled — half-partition right only */
        int n_right = 0;
        for (int j = 0; j < n; j++)
            if (bitmap_get(bm, j)) tmp[n_right++] = indices[j];
        decode_node(table, node->right, tmp, n_right,
                    symbols, in_ptr, tmp + n_right, skip_node);
    } else if (right_leaf && node->right == skip_node) {
        /* Right is prefilled — half-partition left only */
        int n_left = 0;
        for (int j = 0; j < n; j++)
            if (!bitmap_get(bm, j)) indices[n_left++] = indices[j];
        decode_node(table, node->left, indices, n_left,
                    symbols, in_ptr, tmp, skip_node);
    } else {
        int n_right = 0, n_left = 0;
        for (int j = 0; j < n; j++) {
            if (bitmap_get(bm, j))
                tmp[n_right++] = indices[j];
            else
                indices[n_left++] = indices[j];
        }

        /* Recurse into both; child's entry handles leaf/skip_node. */
        decode_node(table, node->left, indices, n_left,
                    symbols, in_ptr, tmp + n_right, skip_node);
        decode_node(table, node->right, tmp, n_right,
                    symbols, in_ptr, tmp + n_right, skip_node);
    }
}

int pivco_huffman_decode_scalar(const uint8_t *in, size_t in_len,
                                const pivco_huffman_table_t *table,
                                uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    int16_t skip_node = table->prefill_node;
    memset(symbols, table->prefill_sym, (size_t)N);

    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    /* See pivco_huffman_neon.c comment. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)malloc(tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;
    const uint8_t *ptr = in;

    decode_node(table, table->tree_root, indices, N,
                symbols, &ptr, tmp, skip_node);

    free(tmp);
    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}
