#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>

#ifdef PIVCO_HAS_NEON
#include <arm_neon.h>

/* ---------- SIMD Compress Shuffle Table ----------
 *
 * For each 8-bit mask, a 16-byte TBL shuffle that packs selected
 * uint16_t elements (2 bytes each) to the front of the register.
 */
/* Combined shuffle table: [256][32] where bytes 0-15 are the shuffle
   for mask (right partition) and bytes 16-31 are for ~mask (left partition).
   Both loaded with a single ldp q0, q1 — one cache line access instead
   of two separate lookups at unrelated addresses. */
uint8_t compress_tab[256][32] __attribute__((aligned(32)));
uint8_t compress_popcnt[256] __attribute__((aligned(64)));
int compress_table_ready = 0;

void init_compress_table(void)
{
    if (compress_table_ready) return;
    for (int mask = 0; mask < 256; mask++) {
        /* Right (bit=1): pack selected to front */
        int out_r = 0;
        for (int i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                compress_tab[mask][out_r * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][out_r * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_r++;
            }
        }
        compress_popcnt[mask] = (uint8_t)out_r;
        for (int j = out_r * 2; j < 16; j++)
            compress_tab[mask][j] = 0xFF;

        /* Left (bit=0): pack complement to front */
        int out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (!(mask & (1 << i))) {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_l * 2; j < 16; j++)
            compress_tab[mask][16 + j] = 0xFF;
    }
    compress_table_ready = 1;
}

/* Partition 8 uint16_t by an 8-bit mask.
   bit=1 → right_out, bit=0 → left_out.
   Source is loaded into register first, so left_out may overlap src
   as long as left_out <= src (which holds when n_left <= j).
   Returns count of right (bit=1) elements. */
static inline int partition_8(const uint16_t *src,
                               uint8_t mask,
                               uint16_t *left_out,
                               uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);

    /* Load both shuffle patterns with one ldp (32 bytes, contiguous) */
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);       /* bytes 0-15: right */
    uint8x16_t shuf_l = vld1q_u8(tab + 16);  /* bytes 16-31: left */

    uint8x16_t right = vqtbl1q_u8(data, shuf_r);
    uint8x16_t left  = vqtbl1q_u8(data, shuf_l);

    int n_right = compress_popcnt[mask];

    vst1q_u8((uint8_t *)right_out, right);
    vst1q_u8((uint8_t *)left_out, left);

    return n_right;
}

/* ---------- NEON Encode (Tree-Walk) ---------- */

static void encode_node_neon(const pivco_huffman_table_t *table,
                              int16_t node_id,
                              uint16_t *indices, int n,
                              int depth,
                              const uint16_t *codes, const uint8_t *lens,
                              uint8_t **out_ptr,
                              uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return; /* leaf */

    /* Build bitmap and partition in one fused pass.
       For each group of 8 indices, construct the mask byte directly
       and feed it to partition_8 + write to the output bitmap. */
    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    *out_ptr += nbytes;

    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 8 <= n; j += 8) {
        /* Build mask byte from 8 code bits */
        uint8_t mask = 0;
        for (int k = 0; k < 8; k++) {
            int idx = indices[j + k];
            int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
            mask |= (uint8_t)(bit << k);
        }
        bm[j >> 3] = mask;

        int nr = partition_8(indices + j, mask,
                             indices + n_left, tmp + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    /* Scalar remainder */
    if (j < n) {
        uint8_t mask = 0;
        for (int k = 0; j + k < n; k++) {
            int idx = indices[j + k];
            int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
            mask |= (uint8_t)(bit << k);
        }
        bm[j >> 3] = mask;
        for (int k = 0; j < n; j++, k++) {
            if (mask & (1 << k))
                tmp[n_right++] = indices[j];
            else
                indices[n_left++] = indices[j];
        }
    }

    encode_node_neon(table, node->left, indices, n_left,
                     depth + 1, codes, lens, out_ptr, tmp + n_right);
    encode_node_neon(table, node->right, tmp, n_right,
                     depth + 1, codes, lens, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_neon(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;

    uint16_t codes[PIVCO_BLOCK_SIZE];
    uint8_t  lens[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) {
        codes[i] = table->code[symbols[i]];
        lens[i]  = table->code_len[symbols[i]];
    }

    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    uint8_t *ptr = out;

    encode_node_neon(table, table->tree_root, indices, N,
                     0, codes, lens, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- NEON Decode (Tree-Walk with SIMD Partition) ---------- */

/* Scatter a single symbol to indices[0..n-1] positions in symbols[].
   NEON-assisted: bulk-load 8 indices per vector + lane extracts. */
static inline void scatter_sym(uint8_t *symbols,
                                const uint16_t *indices, int n,
                                uint8_t sym)
{
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint16x8_t i0 = vld1q_u16(indices + j);
        uint16x8_t i1 = vld1q_u16(indices + j + 8);
        symbols[vgetq_lane_u16(i0, 0)] = sym;
        symbols[vgetq_lane_u16(i0, 1)] = sym;
        symbols[vgetq_lane_u16(i0, 2)] = sym;
        symbols[vgetq_lane_u16(i0, 3)] = sym;
        symbols[vgetq_lane_u16(i0, 4)] = sym;
        symbols[vgetq_lane_u16(i0, 5)] = sym;
        symbols[vgetq_lane_u16(i0, 6)] = sym;
        symbols[vgetq_lane_u16(i0, 7)] = sym;
        symbols[vgetq_lane_u16(i1, 0)] = sym;
        symbols[vgetq_lane_u16(i1, 1)] = sym;
        symbols[vgetq_lane_u16(i1, 2)] = sym;
        symbols[vgetq_lane_u16(i1, 3)] = sym;
        symbols[vgetq_lane_u16(i1, 4)] = sym;
        symbols[vgetq_lane_u16(i1, 5)] = sym;
        symbols[vgetq_lane_u16(i1, 6)] = sym;
        symbols[vgetq_lane_u16(i1, 7)] = sym;
    }
    for (; j + 8 <= n; j += 8) {
        uint16x8_t idx = vld1q_u16(indices + j);
        symbols[vgetq_lane_u16(idx, 0)] = sym;
        symbols[vgetq_lane_u16(idx, 1)] = sym;
        symbols[vgetq_lane_u16(idx, 2)] = sym;
        symbols[vgetq_lane_u16(idx, 3)] = sym;
        symbols[vgetq_lane_u16(idx, 4)] = sym;
        symbols[vgetq_lane_u16(idx, 5)] = sym;
        symbols[vgetq_lane_u16(idx, 6)] = sym;
        symbols[vgetq_lane_u16(idx, 7)] = sym;
    }
    for (; j < n; j++) {
        symbols[indices[j]] = sym;
    }
}

static void decode_node_neon(const pivco_huffman_table_t *table,
                              int16_t node_id,
                              uint16_t *indices, int n,
                              uint8_t *symbols,
                              const uint8_t **in_ptr,
                              uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        /* Leaf — scatter symbol to all indices */
        scatter_sym(symbols, indices, n, (uint8_t)node->symbol);
        return;
    }

    /* Read n code bits */
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

    /* Check children for stage fusion */
    const pivco_tree_node_t *left_child  = &table->tree[node->left];
    const pivco_tree_node_t *right_child = &table->tree[node->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf) {
        /* Both children are leaves — scatter directly from bitmap,
           no partition needed.  Branchless: index into syms[]. */
        uint8_t syms[2] = {(uint8_t)left_child->symbol,
                           (uint8_t)right_child->symbol};
        int j = 0;
        for (; j + 16 <= n; j += 16) {
            uint8_t m0 = bm[j >> 3];
            uint8_t m1 = bm[(j >> 3) + 1];
            uint16x8_t i0 = vld1q_u16(indices + j);
            uint16x8_t i1 = vld1q_u16(indices + j + 8);
            symbols[vgetq_lane_u16(i0, 0)] = syms[(m0 >> 0) & 1];
            symbols[vgetq_lane_u16(i0, 1)] = syms[(m0 >> 1) & 1];
            symbols[vgetq_lane_u16(i0, 2)] = syms[(m0 >> 2) & 1];
            symbols[vgetq_lane_u16(i0, 3)] = syms[(m0 >> 3) & 1];
            symbols[vgetq_lane_u16(i0, 4)] = syms[(m0 >> 4) & 1];
            symbols[vgetq_lane_u16(i0, 5)] = syms[(m0 >> 5) & 1];
            symbols[vgetq_lane_u16(i0, 6)] = syms[(m0 >> 6) & 1];
            symbols[vgetq_lane_u16(i0, 7)] = syms[(m0 >> 7) & 1];
            symbols[vgetq_lane_u16(i1, 0)] = syms[(m1 >> 0) & 1];
            symbols[vgetq_lane_u16(i1, 1)] = syms[(m1 >> 1) & 1];
            symbols[vgetq_lane_u16(i1, 2)] = syms[(m1 >> 2) & 1];
            symbols[vgetq_lane_u16(i1, 3)] = syms[(m1 >> 3) & 1];
            symbols[vgetq_lane_u16(i1, 4)] = syms[(m1 >> 4) & 1];
            symbols[vgetq_lane_u16(i1, 5)] = syms[(m1 >> 5) & 1];
            symbols[vgetq_lane_u16(i1, 6)] = syms[(m1 >> 6) & 1];
            symbols[vgetq_lane_u16(i1, 7)] = syms[(m1 >> 7) & 1];
        }
        for (; j + 8 <= n; j += 8) {
            uint8_t mask = bm[j >> 3];
            uint16x8_t idx = vld1q_u16(indices + j);
            symbols[vgetq_lane_u16(idx, 0)] = syms[(mask >> 0) & 1];
            symbols[vgetq_lane_u16(idx, 1)] = syms[(mask >> 1) & 1];
            symbols[vgetq_lane_u16(idx, 2)] = syms[(mask >> 2) & 1];
            symbols[vgetq_lane_u16(idx, 3)] = syms[(mask >> 3) & 1];
            symbols[vgetq_lane_u16(idx, 4)] = syms[(mask >> 4) & 1];
            symbols[vgetq_lane_u16(idx, 5)] = syms[(mask >> 5) & 1];
            symbols[vgetq_lane_u16(idx, 6)] = syms[(mask >> 6) & 1];
            symbols[vgetq_lane_u16(idx, 7)] = syms[(mask >> 7) & 1];
        }
        for (; j < n; j++) {
            symbols[indices[j]] = syms[bitmap_get(bm, j)];
        }
        return;
    }

    /* SIMD partition in-place, unrolled 2x */
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 16 <= n; j += 16) {
        uint8_t m0 = bm[j >> 3];
        int nr0 = partition_8(indices + j, m0,
                              indices + n_left, tmp + n_right);
        n_right += nr0;
        n_left += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        int nr1 = partition_8(indices + j + 8, m1,
                              indices + n_left, tmp + n_right);
        n_right += nr1;
        n_left += (8 - nr1);
    }
    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8(indices + j, mask,
                             indices + n_left, tmp + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    /* Scalar remainder */
    for (; j < n; j++) {
        if (bitmap_get(bm, j))
            tmp[n_right++] = indices[j];
        else
            indices[n_left++] = indices[j];
    }

    if (left_leaf) {
        /* Left child is leaf — scatter inline, recurse right only */
        scatter_sym(symbols, indices, n_left,
                    (uint8_t)left_child->symbol);
        decode_node_neon(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right);
    } else if (right_leaf) {
        /* Right child is leaf — scatter inline, recurse left only */
        scatter_sym(symbols, tmp, n_right,
                    (uint8_t)right_child->symbol);
        decode_node_neon(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp + n_right);
    } else {
        decode_node_neon(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp + n_right);
        decode_node_neon(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right);
    }
}

int pivco_huffman_decode_neon(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;

    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    const uint8_t *ptr = in;

    decode_node_neon(table, table->tree_root, indices, N,
                     symbols, &ptr, tmp);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
