#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>

#ifdef PIVCO_HAS_AVX512
#include <immintrin.h>

/* ---------- AVX-512 VBMI2 Partition ----------
 *
 * vpcompressw: compress selected uint16_t elements to the front
 * of a 512-bit register in ONE instruction. No shuffle table needed.
 * Processes 32 × uint16_t per iteration (4x the SSE/NEON path).
 */

/* Partition up to 32 uint16_t by a 32-bit mask.
   bit=1 → right_out, bit=0 → left_out.
   Returns count of right (bit=1) elements. */
static inline int partition_32(const uint16_t *src, int n,
                                __mmask32 mask,
                                uint16_t *left_out,
                                uint16_t *right_out)
{
    __m512i data = _mm512_loadu_si512((const __m512i *)src);

    /* Right (bit=1): compress selected elements to front */
    __m512i right = _mm512_maskz_compress_epi16(mask, data);
    int n_right = _mm_popcnt_u32((uint32_t)mask & ((1u << n) - 1));
    _mm512_storeu_si512((__m512i *)right_out, right);

    /* Left (bit=0): compress complement */
    __mmask32 inv = ~mask & (((__mmask32)1 << n) - 1);
    __m512i left = _mm512_maskz_compress_epi16(inv, data);
    _mm512_storeu_si512((__m512i *)left_out, left);

    return n_right;
}

/* Partition exactly 32 elements (fast path, no n masking needed) */
static inline int partition_32_full(const uint16_t *src,
                                     uint32_t mask,
                                     uint16_t *left_out,
                                     uint16_t *right_out)
{
    __m512i data = _mm512_loadu_si512((const __m512i *)src);

    __m512i right = _mm512_maskz_compress_epi16((__mmask32)mask, data);
    int n_right = _mm_popcnt_u32(mask);
    _mm512_storeu_si512((__m512i *)right_out, right);

    __m512i left = _mm512_maskz_compress_epi16((__mmask32)~mask, data);
    _mm512_storeu_si512((__m512i *)left_out, left);

    return n_right;
}

/* ---------- Leaf scatter-write (AVX-512) ---------- */

static inline void scatter_write_avx512(uint8_t *symbols,
                                         const uint16_t *indices, int n,
                                         uint8_t sym)
{
    /* AVX-512 doesn't have byte scatter, but we can use vpscatterd
       with 32-bit indices for dword scatter. For byte writes, we
       fall back to SSE-style extract or scalar.
       Use SSE extract for now (same as x86 backend). */
    int j = 0;
    /* Process 8 at a time using SSE extract */
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

/* ---------- AVX-512 Encode (Tree-Walk) ---------- */

static void encode_node_avx512(const pivco_huffman_table_t *table,
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

    /* AVX-512 partition: 32 indices at a time */
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 32 <= n; j += 32) {
        /* Load 32 bits of mask from 4 consecutive bitmap bytes */
        uint32_t mask;
        memcpy(&mask, bm + (j >> 3), 4);
        int nr = partition_32_full(indices + j, mask,
                                    indices + n_left, tmp + n_right);
        n_right += nr;
        n_left += (32 - nr);
    }
    /* SSE remainder: 8 at a time */
    for (; j + 8 <= n; j += 8) {
        uint8_t mask8 = bm[j >> 3];
        __m128i data = _mm_loadu_si128((const __m128i *)(indices + j));
        /* Use pshufb-based partition for the 8-wide remainder */
        /* Inline simple scalar for now — the 32-wide loop handles most */
        for (int k = 0; k < 8; k++) {
            if (mask8 & (1 << k))
                tmp[n_right++] = indices[j + k];
            else
                indices[n_left++] = indices[j + k];
        }
    }
    /* Scalar remainder */
    for (; j < n; j++) {
        if (bitmap_get(bm, j)) {
            tmp[n_right++] = indices[j];
        } else {
            indices[n_left++] = indices[j];
        }
    }

    encode_node_avx512(table, node->left, indices, n_left,
                        depth + 1, codes, lens, out_ptr, tmp + n_right);
    encode_node_avx512(table, node->right, tmp, n_right,
                        depth + 1, codes, lens, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_avx512(const uint8_t *symbols,
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

    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    uint8_t *ptr = out;

    encode_node_avx512(table, table->tree_root, indices, N,
                        0, codes, lens, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- AVX-512 Decode (Tree-Walk) ---------- */

static void decode_node_avx512(const pivco_huffman_table_t *table,
                                int16_t node_id,
                                uint16_t *indices, int n,
                                uint8_t *symbols,
                                const uint8_t **in_ptr,
                                uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        scatter_write_avx512(symbols, indices, n, (uint8_t)node->symbol);
        return;
    }

    /* Read n code bits */
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

    /* AVX-512 partition: 32 indices at a time */
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 32 <= n; j += 32) {
        uint32_t mask;
        memcpy(&mask, bm + (j >> 3), 4);
        int nr = partition_32_full(indices + j, mask,
                                    indices + n_left, tmp + n_right);
        n_right += nr;
        n_left += (32 - nr);
    }
    /* Scalar remainder for < 32 elements */
    for (; j < n; j++) {
        if (bitmap_get(bm, j)) {
            tmp[n_right++] = indices[j];
        } else {
            indices[n_left++] = indices[j];
        }
    }

    decode_node_avx512(table, node->left, indices, n_left,
                        symbols, in_ptr, tmp + n_right);
    decode_node_avx512(table, node->right, tmp, n_right,
                        symbols, in_ptr, tmp + n_right);
}

int pivco_huffman_decode_avx512(const uint8_t *in, size_t in_len,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;

    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    const uint8_t *ptr = in;

    decode_node_avx512(table, table->tree_root, indices, N,
                        symbols, &ptr, tmp);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_AVX512 */
