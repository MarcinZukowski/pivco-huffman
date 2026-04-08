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
static uint8_t compress_shuf[256][16];
static uint8_t compress_popcnt[256];
static int     compress_table_ready = 0;

static void init_compress_table(void)
{
    if (compress_table_ready) return;
    for (int mask = 0; mask < 256; mask++) {
        int out = 0;
        for (int i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                compress_shuf[mask][out * 2]     = (uint8_t)(i * 2);
                compress_shuf[mask][out * 2 + 1] = (uint8_t)(i * 2 + 1);
                out++;
            }
        }
        compress_popcnt[mask] = (uint8_t)out;
        for (int j = out * 2; j < 16; j++) {
            compress_shuf[mask][j] = 0xFF;
        }
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

    /* Right (bit=1) */
    uint8x16_t shuf_r = vld1q_u8(compress_shuf[mask]);
    uint8x16_t right = vqtbl1q_u8(data, shuf_r);
    int n_right = compress_popcnt[mask];
    vst1q_u8((uint8_t *)right_out, right);

    /* Left (bit=0) */
    uint8_t inv = (uint8_t)~mask;
    uint8x16_t shuf_l = vld1q_u8(compress_shuf[inv]);
    uint8x16_t left = vqtbl1q_u8(data, shuf_l);
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

    /* Write n code bits */
    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    memset(bm, 0, (size_t)nbytes);

    for (int j = 0; j < n; j++) {
        int idx = indices[j];
        int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
        if (bit) bitmap_set(bm, j);
    }
    *out_ptr += nbytes;

    /* SIMD partition in-place: left stays in indices, right goes to tmp */
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8(indices + j, mask,
                             indices + n_left, tmp + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    /* Scalar remainder */
    for (; j < n; j++) {
        if (bitmap_get(bm, j)) {
            tmp[n_right++] = indices[j];
        } else {
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

    uint16_t tmp[PIVCO_BLOCK_SIZE];
    uint8_t *ptr = out;

    encode_node_neon(table, table->tree_root, indices, N,
                     0, codes, lens, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- NEON Decode (Tree-Walk with SIMD Partition) ---------- */

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
        /* Leaf — scatter-write symbol to all indices */
        uint8_t sym = (uint8_t)node->symbol;
        for (int j = 0; j < n; j++) {
            symbols[indices[j]] = sym;
        }
        return;
    }

    /* Read n code bits */
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

    /* SIMD partition in-place */
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8(indices + j, mask,
                             indices + n_left, tmp + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    /* Scalar remainder */
    for (; j < n; j++) {
        if (bitmap_get(bm, j)) {
            tmp[n_right++] = indices[j];
        } else {
            indices[n_left++] = indices[j];
        }
    }

    decode_node_neon(table, node->left, indices, n_left,
                     symbols, in_ptr, tmp + n_right);
    decode_node_neon(table, node->right, tmp, n_right,
                     symbols, in_ptr, tmp + n_right);
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

    uint16_t tmp[PIVCO_BLOCK_SIZE];
    const uint8_t *ptr = in;

    decode_node_neon(table, table->tree_root, indices, N,
                     symbols, &ptr, tmp);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
