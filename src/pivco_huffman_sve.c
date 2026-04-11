#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>

#ifdef PIVCO_HAS_SVE
#include <arm_sve.h>

/* ---------- SVE Partition ----------
 *
 * svcompact: compress selected elements to front using predication.
 * No shuffle table needed — the predicate mask directly drives the
 * compress. Same width as NEON on Graviton 4 (128-bit = 8 × uint16),
 * but eliminates the 4KB shuffle table from L1D.
 *
 * svcompact only works on 32-bit and 64-bit elements (not 16-bit).
 * So we widen uint16→uint32, compact, then narrow back.
 * Alternative: use svtbl with a computed index vector.
 */

/* Partition 8 uint16_t by an 8-bit mask using SVE.
   bit=1 → right_out, bit=0 → left_out.
   Returns count of right (bit=1) elements. */
static inline int partition_8_sve(const uint16_t *src,
                                   uint8_t mask,
                                   uint16_t *left_out,
                                   uint16_t *right_out)
{
    /* svcompact works on 32-bit elements. With 128-bit SVE, that's
       4 elements per vector. Process our 8 uint16 as two halves of 4,
       widened to uint32 for svcompact, then narrowed back.
       No shuffle table needed. */
    svbool_t pg4 = svwhilelt_b32(0, 4);
    int n_right = 0, n_left = 0;

    /* Process two halves: low 4 (mask bits 0-3), high 4 (mask bits 4-7) */
    for (int half = 0; half < 2; half++) {
        const uint16_t *s = src + half * 4;
        uint8_t hmask = (mask >> (half * 4)) & 0xF;

        /* Load 4 × uint16 and widen to uint32 */
        uint32_t tmp32[4];
        for (int k = 0; k < 4; k++) tmp32[k] = s[k];
        svuint32_t data32 = svld1_u32(pg4, tmp32);

        /* Build 4-element predicate from 4-bit mask */
        svuint32_t mask_vec = svdup_u32((uint32_t)hmask);
        svuint32_t bits = svindex_u32(0, 1);
        svuint32_t shifted = svlsl_u32_x(pg4, svdup_u32(1), bits);
        svuint32_t masked = svand_u32_x(pg4, mask_vec, shifted);
        svbool_t right_pred = svcmpne_u32(pg4, masked, svdup_u32(0));
        svbool_t left_pred = svnot_b_z(pg4, right_pred);

        /* Compact right */
        svuint32_t right32 = svcompact_u32(right_pred, data32);
        int nr = (int)svcntp_b32(pg4, right_pred);
        uint32_t out32[4];
        svst1_u32(pg4, out32, right32);
        for (int k = 0; k < nr; k++)
            right_out[n_right++] = (uint16_t)out32[k];

        /* Compact left */
        svuint32_t left32 = svcompact_u32(left_pred, data32);
        int nl = 4 - nr;
        svst1_u32(pg4, out32, left32);
        for (int k = 0; k < nl; k++)
            left_out[n_left++] = (uint16_t)out32[k];
    }
    return n_right;
}

/* ---------- Leaf scatter-write (SVE) ---------- */

static inline void scatter_write_sve(uint8_t *symbols,
                                      const uint16_t *indices, int n,
                                      uint8_t sym)
{
    /* SVE doesn't have byte-granularity scatter stores either.
       Use the same lane-extract approach as NEON/SSE. */
    int j = 0;
    svbool_t pg8 = svwhilelt_b16(0, 8);
    for (; j + 8 <= n; j += 8) {
        svuint16_t idx = svld1_u16(pg8, indices + j);
        /* Extract lanes — SVE doesn't have direct lane extract like NEON.
           Use svlastb with rotating predicates, or just read from memory
           (the load already brought it to cache). */
        symbols[indices[j + 0]] = sym;
        symbols[indices[j + 1]] = sym;
        symbols[indices[j + 2]] = sym;
        symbols[indices[j + 3]] = sym;
        symbols[indices[j + 4]] = sym;
        symbols[indices[j + 5]] = sym;
        symbols[indices[j + 6]] = sym;
        symbols[indices[j + 7]] = sym;
        (void)idx;
    }
    for (; j < n; j++) {
        symbols[indices[j]] = sym;
    }
}

/* ---------- SVE Encode (Tree-Walk) ---------- */

static void encode_node_sve(const pivco_huffman_table_t *table,
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

    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    memset(bm, 0, (size_t)nbytes);

    for (int j = 0; j < n; j++) {
        int idx = indices[j];
        int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
        if (bit) bitmap_set(bm, j);
    }
    *out_ptr += nbytes;

    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8_sve(indices + j, mask,
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

    encode_node_sve(table, node->left, indices, n_left,
                     depth + 1, codes, lens, out_ptr, tmp + n_right);
    encode_node_sve(table, node->right, tmp, n_right,
                     depth + 1, codes, lens, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_sve(const uint8_t *symbols,
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

    encode_node_sve(table, table->tree_root, indices, N,
                     0, codes, lens, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- SVE Decode (Tree-Walk) ---------- */

static void decode_node_sve(const pivco_huffman_table_t *table,
                             int16_t node_id,
                             uint16_t *indices, int n,
                             uint8_t *symbols,
                             const uint8_t **in_ptr,
                             uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        scatter_write_sve(symbols, indices, n, (uint8_t)node->symbol);
        return;
    }

    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8_sve(indices + j, mask,
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

    decode_node_sve(table, node->left, indices, n_left,
                     symbols, in_ptr, tmp + n_right);
    decode_node_sve(table, node->right, tmp, n_right,
                     symbols, in_ptr, tmp + n_right);
}

int pivco_huffman_decode_sve(const uint8_t *in, size_t in_len,
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

    decode_node_sve(table, table->tree_root, indices, N,
                     symbols, &ptr, tmp);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_SVE */
