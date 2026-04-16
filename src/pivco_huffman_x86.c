#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>

#ifdef PIVCO_HAS_SSE4
#include <smmintrin.h>  /* SSE4.1 */
#ifdef PIVCO_HAS_AVX2
#include <immintrin.h>  /* AVX2 */
#endif

/* ---------- SSE4.1 Compress Shuffle Table ----------
 *
 * Identical to the NEON version: for each 8-bit mask, a 16-byte
 * shuffle that packs selected uint16_t elements to the front.
 * pshufb (_mm_shuffle_epi8) is the x86 equivalent of NEON TBL.
 */
/* Combined shuffle table: [256][32] where bytes 0-15 are the shuffle
   for mask (right) and bytes 16-31 are for ~mask (left).
   Loaded as two aligned 16-byte loads from contiguous memory. */
static uint8_t compress_tab[256][32] __attribute__((aligned(32)));
static uint8_t compress_popcnt[256] __attribute__((aligned(64)));
static int     compress_table_ready = 0;

static void init_compress_table(void)
{
    if (compress_table_ready) return;
    for (int mask = 0; mask < 256; mask++) {
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
            compress_tab[mask][j] = 0x80;

        int out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (!(mask & (1 << i))) {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_l * 2; j < 16; j++)
            compress_tab[mask][16 + j] = 0x80;
    }
    compress_table_ready = 1;
}

/* Partition 8 uint16_t by an 8-bit mask using SSE4.1 pshufb.
   bit=1 → right_out, bit=0 → left_out.
   Source is loaded first, so left_out may overlap src (n_left <= j).
   Returns count of right (bit=1) elements. */
static inline int partition_8_sse(const uint16_t *src,
                                   uint8_t mask,
                                   uint16_t *left_out,
                                   uint16_t *right_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);

    /* Load both shuffle patterns from combined table (contiguous) */
    const uint8_t *tab = compress_tab[mask];
    __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));

    __m128i right = _mm_shuffle_epi8(data, shuf_r);
    __m128i left  = _mm_shuffle_epi8(data, shuf_l);

    int n_right = compress_popcnt[mask];

    _mm_storeu_si128((__m128i *)right_out, right);
    _mm_storeu_si128((__m128i *)left_out, left);

    return n_right;
}

/* ---------- Leaf scatter-write (SSE4.1) ---------- */

static inline void scatter_write_sse(uint8_t *symbols,
                                      const uint16_t *indices, int n,
                                      uint8_t sym)
{
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m128i idx = _mm_loadu_si128((const __m128i *)(indices + j));
        symbols[_mm_extract_epi16(idx, 0)] = sym;
        symbols[_mm_extract_epi16(idx, 1)] = sym;
        symbols[_mm_extract_epi16(idx, 2)] = sym;
        symbols[_mm_extract_epi16(idx, 3)] = sym;
        symbols[_mm_extract_epi16(idx, 4)] = sym;
        symbols[_mm_extract_epi16(idx, 5)] = sym;
        symbols[_mm_extract_epi16(idx, 6)] = sym;
        symbols[_mm_extract_epi16(idx, 7)] = sym;
    }
    for (; j < n; j++) {
        symbols[indices[j]] = sym;
    }
}

/* ---------- x86 Encode (Tree-Walk) ---------- */

static void encode_node_x86(const pivco_huffman_table_t *table,
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

    /* SSE partition in-place */
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8_sse(indices + j, mask,
                                  indices + n_left, tmp + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    for (; j < n; j++) {
        if (bitmap_get(bm, j)) {
            tmp[n_right++] = indices[j];
        } else {
            indices[n_left++] = indices[j];
        }
    }

    encode_node_x86(table, node->left, indices, n_left,
                     depth + 1, codes, lens, out_ptr, tmp + n_right);
    encode_node_x86(table, node->right, tmp, n_right,
                     depth + 1, codes, lens, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_x86(const uint8_t *symbols,
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

    encode_node_x86(table, table->tree_root, indices, N,
                     0, codes, lens, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- x86 Decode (Tree-Walk with SSE Partition) ---------- */

/* Half-partition helpers: extract only one side */
static inline int partition_8_sse_right(const uint16_t *src,
                                         uint8_t mask,
                                         uint16_t *right_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_r = _mm_load_si128((const __m128i *)compress_tab[mask]);
    _mm_storeu_si128((__m128i *)right_out, _mm_shuffle_epi8(data, shuf_r));
    return compress_popcnt[mask];
}

static inline int partition_8_sse_left(const uint16_t *src,
                                        uint8_t mask,
                                        uint16_t *left_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(compress_tab[mask] + 16));
    _mm_storeu_si128((__m128i *)left_out, _mm_shuffle_epi8(data, shuf_l));
    return 8 - compress_popcnt[mask];
}

static void decode_node_x86(const pivco_huffman_table_t *table,
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
        scatter_write_sse(symbols, indices, n, (uint8_t)node->symbol);
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

    if (left_leaf && right_leaf
        && node->left != skip_node && node->right != skip_node) {
        /* Both children are leaves (neither prefilled) — scatter directly */
        uint8_t syms[2] = {(uint8_t)left_child->symbol,
                           (uint8_t)right_child->symbol};
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            uint8_t mask = bm[j >> 3];
            __m128i idx = _mm_loadu_si128((const __m128i *)(indices + j));
            symbols[_mm_extract_epi16(idx, 0)] = syms[(mask >> 0) & 1];
            symbols[_mm_extract_epi16(idx, 1)] = syms[(mask >> 1) & 1];
            symbols[_mm_extract_epi16(idx, 2)] = syms[(mask >> 2) & 1];
            symbols[_mm_extract_epi16(idx, 3)] = syms[(mask >> 3) & 1];
            symbols[_mm_extract_epi16(idx, 4)] = syms[(mask >> 4) & 1];
            symbols[_mm_extract_epi16(idx, 5)] = syms[(mask >> 5) & 1];
            symbols[_mm_extract_epi16(idx, 6)] = syms[(mask >> 6) & 1];
            symbols[_mm_extract_epi16(idx, 7)] = syms[(mask >> 7) & 1];
        }
        for (; j < n; j++) {
            symbols[indices[j]] = syms[bitmap_get(bm, j)];
        }
        return;
    }

    if (left_leaf && node->left == skip_node) {
        /* Left is prefilled leaf — half-partition right only */
        int n_right = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8)
            n_right += partition_8_sse_right(indices + j, bm[j >> 3],
                                              tmp + n_right);
        for (; j < n; j++)
            if (bitmap_get(bm, j)) tmp[n_right++] = indices[j];
        decode_node_x86(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right, skip_node);
    } else if (right_leaf && node->right == skip_node) {
        /* Right is prefilled leaf — half-partition left only */
        int n_left = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8)
            n_left += partition_8_sse_left(indices + j, bm[j >> 3],
                                            indices + n_left);
        for (; j < n; j++)
            if (!bitmap_get(bm, j)) indices[n_left++] = indices[j];
        decode_node_x86(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp, skip_node);
    } else {
        /* Full partition */
        int n_left = 0, n_right = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            uint8_t mask = bm[j >> 3];
            int nr = partition_8_sse(indices + j, mask,
                                      indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (8 - nr);
        }
        for (; j < n; j++) {
            if (bitmap_get(bm, j))
                tmp[n_right++] = indices[j];
            else
                indices[n_left++] = indices[j];
        }

        if (left_leaf) {
            if (node->left != skip_node)
                scatter_write_sse(symbols, indices, n_left,
                                  (uint8_t)left_child->symbol);
            decode_node_x86(table, node->right, tmp, n_right,
                             symbols, in_ptr, tmp + n_right, skip_node);
        } else if (right_leaf) {
            if (node->right != skip_node)
                scatter_write_sse(symbols, tmp, n_right,
                                  (uint8_t)right_child->symbol);
            decode_node_x86(table, node->left, indices, n_left,
                             symbols, in_ptr, tmp + n_right, skip_node);
        } else {
            decode_node_x86(table, node->left, indices, n_left,
                             symbols, in_ptr, tmp + n_right, skip_node);
            decode_node_x86(table, node->right, tmp, n_right,
                             symbols, in_ptr, tmp + n_right, skip_node);
        }
    }
}

int pivco_huffman_decode_x86(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root is leaf — fill everything */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Read root bitmap */
    int nbytes = bitmap_bytes(N);
    const uint8_t *bm = ptr;
    ptr += nbytes;

    const pivco_tree_node_t *left_child  = &table->tree[root->left];
    const pivco_tree_node_t *right_child = &table->tree[root->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf) {
        /* Both-leaves at root — sequential stores, no scatter */
        uint8_t syms[2] = {(uint8_t)left_child->symbol,
                           (uint8_t)right_child->symbol};
        for (int j = 0; j < N; j++)
            symbols[j] = syms[(bm[j >> 3] >> (j & 7)) & 1];
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Prefill output with most frequent symbol */
    uint8_t prefill_sym = table->prefill_sym;
    int16_t skip_node = table->prefill_node;
    memset(symbols, prefill_sym, (size_t)N);

    /* Partition at root — generate identity indices in-place */
    uint16_t indices[PIVCO_BLOCK_SIZE];
    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];

    if (left_leaf && root->left == skip_node) {
        /* Left is prefilled — half-partition right only at root */
        int n_right = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            /* Generate identity indices [j..j+7] and partition right */
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            n_right += partition_8_sse_right(id, bm[j >> 3], tmp + n_right);
        }
        decode_node_x86(table, root->right, tmp, n_right,
                         symbols, &ptr, tmp + n_right, skip_node);
    } else if (right_leaf && root->right == skip_node) {
        /* Right is prefilled — half-partition left only at root */
        int n_left = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            n_left += partition_8_sse_left(id, bm[j >> 3], indices + n_left);
        }
        decode_node_x86(table, root->left, indices, n_left,
                         symbols, &ptr, tmp, skip_node);
    } else {
        /* Full partition at root */
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            uint8_t mask = bm[j >> 3];
            int nr = partition_8_sse(id, mask,
                                      indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (8 - nr);
        }

        if (left_leaf) {
            if (root->left != skip_node)
                scatter_write_sse(symbols, indices, n_left,
                                  (uint8_t)left_child->symbol);
            decode_node_x86(table, root->right, tmp, n_right,
                             symbols, &ptr, tmp + n_right, skip_node);
        } else if (right_leaf) {
            if (root->right != skip_node)
                scatter_write_sse(symbols, tmp, n_right,
                                  (uint8_t)right_child->symbol);
            decode_node_x86(table, root->left, indices, n_left,
                             symbols, &ptr, tmp + n_right, skip_node);
        } else {
            decode_node_x86(table, root->left, indices, n_left,
                             symbols, &ptr, tmp + n_right, skip_node);
            decode_node_x86(table, root->right, tmp, n_right,
                             symbols, &ptr, tmp + n_right, skip_node);
        }
    }

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_SSE4 */
