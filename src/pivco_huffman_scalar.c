#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
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

    /* Write n code bits: bit = (code >> (len - 1 - depth)) & 1 */
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

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    uint8_t *ptr = out;

    encode_node(table, table->tree_root, indices, N,
                0, codes, lens, &ptr, tmp);

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

    /* Read n code bits */
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

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

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    const uint8_t *ptr = in;

    decode_node(table, table->tree_root, indices, N,
                symbols, &ptr, tmp, skip_node);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

/* ---------- PIVCO Huffman X2 (Scalar, two-cursor) ----------
 *
 * Two cursors walk the tree in lockstep on disjoint halves of the batch:
 *   cursor A owns output positions [0 .. N/2)
 *   cursor B owns output positions [N/2 .. N)
 * Each cursor has its own bitstream; both bitstreams are emitted by the
 * encoder and concatenated with a 16-bit little-endian length prefix
 * for stream A.
 *
 * Rebalance rule (must match exactly between encode and decode):
 *   After partition at each internal node, run rebalance_x2 on the left
 *   pair (indices_A, indices_B) and the right pair (tmp_A, tmp_B) so
 *   that n_A == ceil((n_A + n_B) / 2) and n_B == floor(...).  Excess
 *   indices move from the larger cursor's tail to the smaller's tail,
 *   without touching the bitstream.  Indices that move at level L will
 *   have their level-(L+1) bit emitted by the encoder into the
 *   destination cursor's stream — and read by the decoder from there
 *   too — so rebalance is purely an indices[] reorganization.
 *
 * Scalar X2 won't show ILP wins (no parallelism to expose), but locks
 * the wire format and gives test_roundtrip_x2 a reference.  The win
 * comes when SIMD backends issue paired primitive calls per node.
 */

#define PIVCO_X2_HALF (PIVCO_BLOCK_SIZE / 2)

/* Move excess from arr_A↔arr_B so that n_A == ceil((n_A + n_B) / 2).
 * Caller must ensure both buffers have room for incoming entries. */
static inline void rebalance_x2(uint16_t *arr_A, int *n_A,
                                 uint16_t *arr_B, int *n_B)
{
    int total = *n_A + *n_B;
    int target_A = (total + 1) >> 1;
    if (*n_A > target_A) {
        int excess = *n_A - target_A;
        memcpy(arr_B + *n_B, arr_A + target_A,
               (size_t)excess * sizeof(uint16_t));
        *n_A = target_A;
        *n_B += excess;
    } else if (*n_A < target_A) {
        int excess = target_A - *n_A;
        memcpy(arr_A + *n_A, arr_B + (*n_B - excess),
               (size_t)excess * sizeof(uint16_t));
        *n_A += excess;
        *n_B -= excess;
    }
}

static void encode_node_x2(const pivco_huffman_table_t *table,
                            int16_t node_id,
                            uint16_t *indices_A, int n_A,
                            uint16_t *indices_B, int n_B,
                            int depth,
                            const uint16_t *codes, const uint8_t *lens,
                            uint8_t **out_ptr_A, uint8_t **out_ptr_B,
                            uint16_t *tmp_A, uint16_t *tmp_B)
{
    if (n_A == 0 && n_B == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;

    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        if (n_A > 0) {
            int bytes_A = (n_A * D + 7) >> 3;
            if (bytes_A > 0) (*out_ptr_A)[bytes_A - 1] = 0;
            pack_D_bits_scalar(*out_ptr_A, n_A, D, indices_A, codes);
            *out_ptr_A += bytes_A;
        }
        if (n_B > 0) {
            int bytes_B = (n_B * D + 7) >> 3;
            if (bytes_B > 0) (*out_ptr_B)[bytes_B - 1] = 0;
            pack_D_bits_scalar(*out_ptr_B, n_B, D, indices_B, codes);
            *out_ptr_B += bytes_B;
        }
        return;
    }

    int nbytes_A = bitmap_bytes(n_A);
    int nbytes_B = bitmap_bytes(n_B);
    uint8_t *bm_A = *out_ptr_A;
    uint8_t *bm_B = *out_ptr_B;
    if (nbytes_A) memset(bm_A, 0, (size_t)nbytes_A);
    if (nbytes_B) memset(bm_B, 0, (size_t)nbytes_B);

    for (int j = 0; j < n_A; j++) {
        int idx = indices_A[j];
        int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
        if (bit) bitmap_set(bm_A, j);
    }
    for (int j = 0; j < n_B; j++) {
        int idx = indices_B[j];
        int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
        if (bit) bitmap_set(bm_B, j);
    }
    *out_ptr_A += nbytes_A;
    *out_ptr_B += nbytes_B;

    int n_left_A = 0, n_right_A = 0;
    for (int j = 0; j < n_A; j++) {
        if (bitmap_get(bm_A, j)) tmp_A[n_right_A++] = indices_A[j];
        else                     indices_A[n_left_A++] = indices_A[j];
    }
    int n_left_B = 0, n_right_B = 0;
    for (int j = 0; j < n_B; j++) {
        if (bitmap_get(bm_B, j)) tmp_B[n_right_B++] = indices_B[j];
        else                     indices_B[n_left_B++] = indices_B[j];
    }

    rebalance_x2(indices_A, &n_left_A, indices_B, &n_left_B);
    rebalance_x2(tmp_A,     &n_right_A, tmp_B,     &n_right_B);

    encode_node_x2(table, node->left,
                    indices_A, n_left_A, indices_B, n_left_B,
                    depth + 1, codes, lens,
                    out_ptr_A, out_ptr_B,
                    tmp_A + n_right_A, tmp_B + n_right_B);
    encode_node_x2(table, node->right,
                    tmp_A, n_right_A, tmp_B, n_right_B,
                    depth + 1, codes, lens,
                    out_ptr_A, out_ptr_B,
                    tmp_A + n_right_A, tmp_B + n_right_B);
}

int pivco_huffman_encode_scalar_x2(const uint8_t *symbols,
                                    const pivco_huffman_table_t *table,
                                    uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    const int N = PIVCO_BLOCK_SIZE;

    uint16_t codes[PIVCO_BLOCK_SIZE];
    uint8_t  lens[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) {
        codes[i] = table->code[symbols[i]];
        lens[i]  = table->code_len[symbols[i]];
    }

    /* Per-cursor index buffers sized to N: after rebalance, one cursor
     * can hold up to ceil((n_A+n_B)/2) <= N/2 entries — but since we
     * also need scratch room for the incoming memcpy during rebalance
     * (which appends to the smaller cursor's tail before the count is
     * updated), oversize to N to be safe. */
    uint16_t indices_A[PIVCO_BLOCK_SIZE];
    uint16_t indices_B[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < PIVCO_X2_HALF; i++) {
        indices_A[i] = (uint16_t)i;
        indices_B[i] = (uint16_t)(PIVCO_X2_HALF + i);
    }

    /* tmp_X mirrors the existing scalar encoder's 2N convention. */
    uint16_t tmp_A[PIVCO_BLOCK_SIZE * 2];
    uint16_t tmp_B[PIVCO_BLOCK_SIZE * 2];

    /* Two scratch streams, concatenated after encoding. */
    uint8_t scratch_A[PIVCO_MAX_ENCODED_SIZE];
    uint8_t scratch_B[PIVCO_MAX_ENCODED_SIZE];
    uint8_t *ptr_A = scratch_A;
    uint8_t *ptr_B = scratch_B;

    encode_node_x2(table, table->tree_root,
                    indices_A, PIVCO_X2_HALF,
                    indices_B, PIVCO_X2_HALF,
                    0, codes, lens,
                    &ptr_A, &ptr_B,
                    tmp_A, tmp_B);

    size_t len_A = (size_t)(ptr_A - scratch_A);
    size_t len_B = (size_t)(ptr_B - scratch_B);

    out[0] = (uint8_t)(len_A & 0xff);
    out[1] = (uint8_t)((len_A >> 8) & 0xff);
    memcpy(out + 2,         scratch_A, len_A);
    memcpy(out + 2 + len_A, scratch_B, len_B);

    *out_len = 2 + len_A + len_B;
    return PIVCO_OK;
}

static void decode_node_x2(const pivco_huffman_table_t *table,
                            int16_t node_id,
                            uint16_t *indices_A, int n_A,
                            uint16_t *indices_B, int n_B,
                            uint8_t *symbols,
                            const uint8_t **in_ptr_A,
                            const uint8_t **in_ptr_B,
                            uint16_t *tmp_A, uint16_t *tmp_B,
                            int16_t skip_node)
{
    if (n_A == 0 && n_B == 0) return;
    if (node_id == skip_node) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        uint8_t sym = (uint8_t)node->symbol;
        for (int j = 0; j < n_A; j++) symbols[indices_A[j]] = sym;
        for (int j = 0; j < n_B; j++) symbols[indices_B[j]] = sym;
        return;
    }

    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[node_id]];
        if (n_A > 0) {
            int bytes_A = (n_A * D + 7) >> 3;
            flat_decode_scatter_scalar(symbols, indices_A, n_A,
                                        *in_ptr_A, D, c2s);
            *in_ptr_A += bytes_A;
        }
        if (n_B > 0) {
            int bytes_B = (n_B * D + 7) >> 3;
            flat_decode_scatter_scalar(symbols, indices_B, n_B,
                                        *in_ptr_B, D, c2s);
            *in_ptr_B += bytes_B;
        }
        return;
    }

    int nbytes_A = bitmap_bytes(n_A);
    int nbytes_B = bitmap_bytes(n_B);
    const uint8_t *bm_A = *in_ptr_A;
    const uint8_t *bm_B = *in_ptr_B;
    *in_ptr_A += nbytes_A;
    *in_ptr_B += nbytes_B;

    /* Both-leaves and prefilled-leaf fusion paths are decoder-side
     * optimizations that don't change the number of bits emitted by
     * the encoder; we keep them here for parity with single-cursor
     * scalar (and to match its prefill_node semantics).
     */
    const pivco_tree_node_t *left_child  = &table->tree[node->left];
    const pivco_tree_node_t *right_child = &table->tree[node->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf
        && node->left != skip_node && node->right != skip_node) {
        uint8_t syms[2] = {(uint8_t)left_child->symbol,
                           (uint8_t)right_child->symbol};
        for (int j = 0; j < n_A; j++)
            symbols[indices_A[j]] = syms[bitmap_get(bm_A, j)];
        for (int j = 0; j < n_B; j++)
            symbols[indices_B[j]] = syms[bitmap_get(bm_B, j)];
        return;
    }

    if (left_leaf && node->left == skip_node) {
        /* Half-partition right only, per cursor.  No rebalance needed:
         * we don't recurse with paired (n_A, n_B) here. */
        int n_right_A = 0;
        for (int j = 0; j < n_A; j++)
            if (bitmap_get(bm_A, j)) tmp_A[n_right_A++] = indices_A[j];
        int n_right_B = 0;
        for (int j = 0; j < n_B; j++)
            if (bitmap_get(bm_B, j)) tmp_B[n_right_B++] = indices_B[j];
        rebalance_x2(tmp_A, &n_right_A, tmp_B, &n_right_B);
        decode_node_x2(table, node->right,
                        tmp_A, n_right_A, tmp_B, n_right_B,
                        symbols, in_ptr_A, in_ptr_B,
                        tmp_A + n_right_A, tmp_B + n_right_B,
                        skip_node);
        return;
    }
    if (right_leaf && node->right == skip_node) {
        int n_left_A = 0;
        for (int j = 0; j < n_A; j++)
            if (!bitmap_get(bm_A, j)) indices_A[n_left_A++] = indices_A[j];
        int n_left_B = 0;
        for (int j = 0; j < n_B; j++)
            if (!bitmap_get(bm_B, j)) indices_B[n_left_B++] = indices_B[j];
        rebalance_x2(indices_A, &n_left_A, indices_B, &n_left_B);
        decode_node_x2(table, node->left,
                        indices_A, n_left_A, indices_B, n_left_B,
                        symbols, in_ptr_A, in_ptr_B,
                        tmp_A, tmp_B, skip_node);
        return;
    }

    /* General case: partition both cursors, rebalance both children, recurse. */
    int n_left_A = 0, n_right_A = 0;
    for (int j = 0; j < n_A; j++) {
        if (bitmap_get(bm_A, j)) tmp_A[n_right_A++] = indices_A[j];
        else                     indices_A[n_left_A++] = indices_A[j];
    }
    int n_left_B = 0, n_right_B = 0;
    for (int j = 0; j < n_B; j++) {
        if (bitmap_get(bm_B, j)) tmp_B[n_right_B++] = indices_B[j];
        else                     indices_B[n_left_B++] = indices_B[j];
    }

    rebalance_x2(indices_A, &n_left_A, indices_B, &n_left_B);
    rebalance_x2(tmp_A,     &n_right_A, tmp_B,     &n_right_B);

    decode_node_x2(table, node->left,
                    indices_A, n_left_A, indices_B, n_left_B,
                    symbols, in_ptr_A, in_ptr_B,
                    tmp_A + n_right_A, tmp_B + n_right_B,
                    skip_node);
    decode_node_x2(table, node->right,
                    tmp_A, n_right_A, tmp_B, n_right_B,
                    symbols, in_ptr_A, in_ptr_B,
                    tmp_A + n_right_A, tmp_B + n_right_B,
                    skip_node);
}

int pivco_huffman_decode_scalar_x2(const uint8_t *in, size_t in_len,
                                    const pivco_huffman_table_t *table,
                                    uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    if (in_len < 2) return PIVCO_ERR_CORRUPT;

    const int N = PIVCO_BLOCK_SIZE;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    int16_t skip_node = table->prefill_node;
    memset(symbols, table->prefill_sym, (size_t)N);

    size_t len_A = (size_t)in[0] | ((size_t)in[1] << 8);
    if (2 + len_A > in_len) return PIVCO_ERR_CORRUPT;

    const uint8_t *ptr_A = in + 2;
    const uint8_t *ptr_B = in + 2 + len_A;

    uint16_t indices_A[PIVCO_BLOCK_SIZE];
    uint16_t indices_B[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < PIVCO_X2_HALF; i++) {
        indices_A[i] = (uint16_t)i;
        indices_B[i] = (uint16_t)(PIVCO_X2_HALF + i);
    }

    uint16_t tmp_A[PIVCO_BLOCK_SIZE * 2];
    uint16_t tmp_B[PIVCO_BLOCK_SIZE * 2];

    decode_node_x2(table, table->tree_root,
                    indices_A, PIVCO_X2_HALF,
                    indices_B, PIVCO_X2_HALF,
                    symbols, &ptr_A, &ptr_B,
                    tmp_A, tmp_B, skip_node);

    *consumed = (size_t)(ptr_B - in);
    return PIVCO_OK;
}
