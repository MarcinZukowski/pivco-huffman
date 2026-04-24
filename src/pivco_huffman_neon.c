#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_neon_common.h"
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

/* Half-partition: extract only the right (bit=1) elements.
   One TBL + one store instead of two. Returns count of right elements. */
static inline int partition_8_right(const uint16_t *src,
                                     uint8_t mask,
                                     uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
    vst1q_u8((uint8_t *)right_out, vqtbl1q_u8(data, shuf_r));
    return compress_popcnt[mask];
}

/* Half-partition: extract only the left (bit=0) elements. */
static inline int partition_8_left(const uint16_t *src,
                                    uint8_t mask,
                                    uint16_t *left_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
    vst1q_u8((uint8_t *)left_out, vqtbl1q_u8(data, shuf_l));
    return 8 - compress_popcnt[mask];
}

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

/* Both children are leaves: scatter sym0 (bit=0) or sym1 (bit=1) to each
   index position, selecting via SIMD vtst/veor from the bitmap. */
static inline void scatter_both_leaves(uint8_t *symbols,
                                        const uint16_t *indices, int n,
                                        const uint8_t *bm,
                                        uint8_t sym0, uint8_t sym1)
{
    uint8x8_t vsym0  = vdup_n_u8(sym0);
    uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);
    static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);

    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint8x8_t bits0 = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
        uint8x8_t vals0 = veor_u8(vsym0, vand_u8(vdelta, bits0));
        uint8x8_t bits1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]), vbit_pos);
        uint8x8_t vals1 = veor_u8(vsym0, vand_u8(vdelta, bits1));
        uint16x8_t i0 = vld1q_u16(indices + j);
        uint16x8_t i1 = vld1q_u16(indices + j + 8);
        symbols[vgetq_lane_u16(i0, 0)] = vget_lane_u8(vals0, 0);
        symbols[vgetq_lane_u16(i0, 1)] = vget_lane_u8(vals0, 1);
        symbols[vgetq_lane_u16(i0, 2)] = vget_lane_u8(vals0, 2);
        symbols[vgetq_lane_u16(i0, 3)] = vget_lane_u8(vals0, 3);
        symbols[vgetq_lane_u16(i0, 4)] = vget_lane_u8(vals0, 4);
        symbols[vgetq_lane_u16(i0, 5)] = vget_lane_u8(vals0, 5);
        symbols[vgetq_lane_u16(i0, 6)] = vget_lane_u8(vals0, 6);
        symbols[vgetq_lane_u16(i0, 7)] = vget_lane_u8(vals0, 7);
        symbols[vgetq_lane_u16(i1, 0)] = vget_lane_u8(vals1, 0);
        symbols[vgetq_lane_u16(i1, 1)] = vget_lane_u8(vals1, 1);
        symbols[vgetq_lane_u16(i1, 2)] = vget_lane_u8(vals1, 2);
        symbols[vgetq_lane_u16(i1, 3)] = vget_lane_u8(vals1, 3);
        symbols[vgetq_lane_u16(i1, 4)] = vget_lane_u8(vals1, 4);
        symbols[vgetq_lane_u16(i1, 5)] = vget_lane_u8(vals1, 5);
        symbols[vgetq_lane_u16(i1, 6)] = vget_lane_u8(vals1, 6);
        symbols[vgetq_lane_u16(i1, 7)] = vget_lane_u8(vals1, 7);
    }
    for (; j + 8 <= n; j += 8) {
        uint8x8_t bits = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
        uint8x8_t vals = veor_u8(vsym0, vand_u8(vdelta, bits));
        uint16x8_t idx = vld1q_u16(indices + j);
        symbols[vgetq_lane_u16(idx, 0)] = vget_lane_u8(vals, 0);
        symbols[vgetq_lane_u16(idx, 1)] = vget_lane_u8(vals, 1);
        symbols[vgetq_lane_u16(idx, 2)] = vget_lane_u8(vals, 2);
        symbols[vgetq_lane_u16(idx, 3)] = vget_lane_u8(vals, 3);
        symbols[vgetq_lane_u16(idx, 4)] = vget_lane_u8(vals, 4);
        symbols[vgetq_lane_u16(idx, 5)] = vget_lane_u8(vals, 5);
        symbols[vgetq_lane_u16(idx, 6)] = vget_lane_u8(vals, 6);
        symbols[vgetq_lane_u16(idx, 7)] = vget_lane_u8(vals, 7);
    }
    for (; j < n; j++) {
        uint8_t bit = (bm[j >> 3] >> (j & 7)) & 1;
        symbols[indices[j]] = sym0 ^ ((sym0 ^ sym1) & (uint8_t)(-(int8_t)bit));
    }
}

static void decode_node_neon(const pivco_huffman_table_t *table,
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

    if (left_leaf && right_leaf
        && node->left != skip_node && node->right != skip_node) {
        /* Both children are leaves (neither prefilled) — scatter directly
           from bitmap.  If one IS prefilled, fall through to the
           half-partition path below which is faster (1 TBL vs 8 stores). */
        scatter_both_leaves(symbols, indices, n, bm,
                            (uint8_t)left_child->symbol,
                            (uint8_t)right_child->symbol);
        return;
    }

    if (left_leaf && node->left == skip_node) {
        /* Left child is the prefilled leaf — half-partition right only.
           Left side already covered by memset, no scatter needed. */
        int n_right = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            n_right += partition_8_right(indices + j, bm[j >> 3],
                                          tmp + n_right);
        }
        for (; j < n; j++) {
            if (bitmap_get(bm, j))
                tmp[n_right++] = indices[j];
        }
        decode_node_neon(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right, skip_node);
    } else if (right_leaf && node->right == skip_node) {
        /* Right child is the prefilled leaf — half-partition left only. */
        int n_left = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            n_left += partition_8_left(indices + j, bm[j >> 3],
                                        indices + n_left);
        }
        for (; j < n; j++) {
            if (!bitmap_get(bm, j))
                indices[n_left++] = indices[j];
        }
        decode_node_neon(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp, skip_node);
    } else {
        /* Standard full partition */
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
        for (; j < n; j++) {
            if (bitmap_get(bm, j))
                tmp[n_right++] = indices[j];
            else
                indices[n_left++] = indices[j];
        }

        if (left_leaf) {
            if (node->left != skip_node)
                scatter_sym(symbols, indices, n_left,
                            (uint8_t)left_child->symbol);
            decode_node_neon(table, node->right, tmp, n_right,
                             symbols, in_ptr, tmp + n_right, skip_node);
        } else if (right_leaf) {
            if (node->right != skip_node)
                scatter_sym(symbols, tmp, n_right,
                            (uint8_t)right_child->symbol);
            decode_node_neon(table, node->left, indices, n_left,
                             symbols, in_ptr, tmp + n_right, skip_node);
        } else {
            decode_node_neon(table, node->left, indices, n_left,
                             symbols, in_ptr, tmp + n_right, skip_node);
            decode_node_neon(table, node->right, tmp, n_right,
                             symbols, in_ptr, tmp + n_right, skip_node);
        }
    }
}

/* Partition 8 identity indices starting at base.
   Generates [base, base+1, ..., base+7] in-register (no memory read)
   then partitions via TBL shuffle like partition_8. */
static inline int partition_root_8(int base, uint8_t mask,
                                    uint16_t *left_out,
                                    uint16_t *right_out)
{
    static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
    uint8x16_t data = vreinterpretq_u8_u16(
        vaddq_u16(vdupq_n_u16((uint16_t)base), vld1q_u16(off)));

    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);

    vst1q_u8((uint8_t *)right_out, vqtbl1q_u8(data, shuf_r));
    vst1q_u8((uint8_t *)left_out, vqtbl1q_u8(data, shuf_l));

    return compress_popcnt[mask];
}

int pivco_huffman_decode_neon(const uint8_t *in, size_t in_len,
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
        /* Both-leaves at root — sequential vst1 stores, no scatter.
           indices[j] == j so symbols[indices[j]] = symbols[j].
           Overwrites the memset, but vst1 is equally fast. */
        uint8_t sym0 = (uint8_t)left_child->symbol;
        uint8_t sym1 = (uint8_t)right_child->symbol;
        uint8x8_t vsym0  = vdup_n_u8(sym0);
        uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);
        static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
        uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);

        int j = 0;
        for (; j + 16 <= N; j += 16) {
            uint8x8_t bits0 = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
            uint8x8_t vals0 = veor_u8(vsym0, vand_u8(vdelta, bits0));
            uint8x8_t bits1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]), vbit_pos);
            uint8x8_t vals1 = veor_u8(vsym0, vand_u8(vdelta, bits1));
            vst1_u8(symbols + j, vals0);
            vst1_u8(symbols + j + 8, vals1);
        }
        for (; j + 8 <= N; j += 8) {
            uint8x8_t bits = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
            uint8x8_t vals = veor_u8(vsym0, vand_u8(vdelta, bits));
            vst1_u8(symbols + j, vals);
        }
        for (; j < N; j++) {
            uint8_t bit = (bm[j >> 3] >> (j & 7)) & 1;
            symbols[j] = sym0 ^ ((sym0 ^ sym1) & (uint8_t)(-(int8_t)bit));
        }
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Prefill output with the most frequent symbol (precomputed in table).
       The tree walk skips scattering this symbol — it's already in place. */
    uint8_t prefill_sym = table->prefill_sym;
    memset(symbols, prefill_sym, (size_t)N);

    int16_t skip_node = table->prefill_node;
    uint16_t indices[PIVCO_BLOCK_SIZE];
    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];

    if (left_leaf && root->left == skip_node) {
        /* Left child is the prefilled leaf at root — half-partition
           right only, generate identity indices in-register. */
        int n_right = 0;
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            uint8_t mask = bm[j >> 3];
            static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
            uint8x16_t data = vreinterpretq_u8_u16(
                vaddq_u16(vdupq_n_u16((uint16_t)j), vld1q_u16(off)));
            uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
            vst1q_u8((uint8_t *)(tmp + n_right),
                     vqtbl1q_u8(data, shuf_r));
            n_right += compress_popcnt[mask];
        }
        for (; j < N; j++) {
            if (bitmap_get(bm, j))
                tmp[n_right++] = (uint16_t)j;
        }
        decode_node_neon(table, root->right, tmp, n_right,
                         symbols, &ptr, tmp + n_right, skip_node);
    } else if (right_leaf && root->right == skip_node) {
        /* Right child is the prefilled leaf at root — half-partition
           left only, generate identity indices in-register. */
        int n_left = 0;
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            uint8_t mask = bm[j >> 3];
            static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
            uint8x16_t data = vreinterpretq_u8_u16(
                vaddq_u16(vdupq_n_u16((uint16_t)j), vld1q_u16(off)));
            uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
            vst1q_u8((uint8_t *)(indices + n_left),
                     vqtbl1q_u8(data, shuf_l));
            n_left += 8 - compress_popcnt[mask];
        }
        for (; j < N; j++) {
            if (!bitmap_get(bm, j))
                indices[n_left++] = (uint16_t)j;
        }
        decode_node_neon(table, root->left, indices, n_left,
                         symbols, &ptr, tmp, skip_node);
    } else {
        /* Full partition at root with identity indices */
        int n_left = 0, n_right = 0;
        int j = 0;

        for (; j + 16 <= N; j += 16) {
            uint8_t m0 = bm[j >> 3];
            int nr0 = partition_root_8(j, m0,
                                        indices + n_left, tmp + n_right);
            n_right += nr0;
            n_left += (8 - nr0);

            uint8_t m1 = bm[(j >> 3) + 1];
            int nr1 = partition_root_8(j + 8, m1,
                                        indices + n_left, tmp + n_right);
            n_right += nr1;
            n_left += (8 - nr1);
        }
        for (; j + 8 <= N; j += 8) {
            uint8_t mask = bm[j >> 3];
            int nr = partition_root_8(j, mask,
                                       indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (8 - nr);
        }
        for (; j < N; j++) {
            if (bitmap_get(bm, j))
                tmp[n_right++] = (uint16_t)j;
            else
                indices[n_left++] = (uint16_t)j;
        }

        if (left_leaf) {
            scatter_sym(symbols, indices, n_left,
                        (uint8_t)left_child->symbol);
            decode_node_neon(table, root->right, tmp, n_right,
                             symbols, &ptr, tmp + n_right, skip_node);
        } else if (right_leaf) {
            scatter_sym(symbols, tmp, n_right,
                        (uint8_t)right_child->symbol);
            decode_node_neon(table, root->left, indices, n_left,
                             symbols, &ptr, tmp + n_right, skip_node);
        } else {
            decode_node_neon(table, root->left, indices, n_left,
                             symbols, &ptr, tmp + n_right, skip_node);
            decode_node_neon(table, root->right, tmp, n_right,
                             symbols, &ptr, tmp + n_right, skip_node);
        }
    }

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

/* ---------- Internal wrappers exposed for pivco_huffman_neon_prefix.c ----------
 *
 * The prefix-radix backend performs a top-level K-way partition and then
 * needs to delegate each non-leaf bin's subtree (at depth M below the
 * root) to the standard 2-way neon encoder/decoder.  These thin wrappers
 * expose exactly the internal recursive entry points, preserving the
 * same contract. */

void pivco_neon_decode_subtree_(const pivco_huffman_table_t *table,
                                 int16_t node_id,
                                 uint16_t *indices, int n,
                                 uint8_t *symbols,
                                 const uint8_t **in_ptr,
                                 uint16_t *tmp,
                                 int16_t skip_node)
{
    decode_node_neon(table, node_id, indices, n,
                     symbols, in_ptr, tmp, skip_node);
}

void pivco_neon_encode_subtree_(const pivco_huffman_table_t *table,
                                 int16_t node_id,
                                 uint16_t *indices, int n,
                                 int depth,
                                 const uint16_t *codes, const uint8_t *lens,
                                 uint8_t **out_ptr,
                                 uint16_t *tmp)
{
    encode_node_neon(table, node_id, indices, n, depth,
                     codes, lens, out_ptr, tmp);
}

#endif /* PIVCO_HAS_NEON */
