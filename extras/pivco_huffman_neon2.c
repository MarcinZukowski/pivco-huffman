#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>
#include <stdlib.h>

#ifdef PIVCO_HAS_NEON
#include <arm_neon.h>

/* Reuse the combined shuffle table from pivco_neon.c.
   These are defined there and shared via linkage. */
extern uint8_t compress_tab[256][32];
extern uint8_t compress_popcnt[256];
extern void init_compress_table(void);

/* ---------- 4-way partition ----------
 *
 * Given 8 uint16_t indices and two mask bytes (bit0, bit1),
 * partition into 4 groups based on the 2-bit code per element:
 *   00 → out0 (left-left)
 *   01 → out1 (left-right)
 *   10 → out2 (right-left)
 *   11 → out3 (right-right)
 *
 * Returns counts via pointers. */
static inline void partition_8_4way(const uint16_t *src,
                                     uint8_t b0, uint8_t b1,
                                     uint16_t *out0, int *cnt0,
                                     uint16_t *out1, int *cnt1,
                                     uint16_t *out2, int *cnt2,
                                     uint16_t *out3, int *cnt3)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);

    uint8_t m00 = (uint8_t)(~b0 & ~b1);
    uint8_t m01 = (uint8_t)(~b0 &  b1);
    uint8_t m10 = (uint8_t)( b0 & ~b1);
    uint8_t m11 = (uint8_t)( b0 &  b1);

    /* Compress each group using the right half of compress_tab
       (we only need the "selected" direction, not the complement) */
    vst1q_u8((uint8_t *)out0, vqtbl1q_u8(data, vld1q_u8(compress_tab[m00])));
    *cnt0 += compress_popcnt[m00];

    vst1q_u8((uint8_t *)out1, vqtbl1q_u8(data, vld1q_u8(compress_tab[m01])));
    *cnt1 += compress_popcnt[m01];

    vst1q_u8((uint8_t *)out2, vqtbl1q_u8(data, vld1q_u8(compress_tab[m10])));
    *cnt2 += compress_popcnt[m10];

    vst1q_u8((uint8_t *)out3, vqtbl1q_u8(data, vld1q_u8(compress_tab[m11])));
    *cnt3 += compress_popcnt[m11];
}

/* ---------- Leaf scatter (same as neon backend) ---------- */

static inline void scatter_write(uint8_t *symbols,
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
    for (; j < n; j++)
        symbols[indices[j]] = sym;
}

/* Reuse partition_8 from neon backend */
static inline int partition_8(const uint16_t *src, uint8_t mask,
                               uint16_t *left_out, uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    uint8x16_t right = vqtbl1q_u8(data, shuf_r);
    uint8x16_t left  = vqtbl1q_u8(data, shuf_l);
    int n_right = compress_popcnt[mask];
    vst1q_u8((uint8_t *)right_out, right);
    vst1q_u8((uint8_t *)left_out, left);
    return n_right;
}

/* ---------- Encode (Tree-Walk with 4-way fusion) ---------- */

static void encode_node_neon2(const pivco_table_t *table,
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

    /* Check if both children are internal (candidates for 4-way fusion) */
    const pivco_tree_node_t *lc = &table->tree[node->left];
    const pivco_tree_node_t *rc = &table->tree[node->right];

    if (lc->symbol < 0 && rc->symbol < 0) {
        /* 4-way: write 2 bits per symbol, partition into 4 groups */
        int nbytes = bitmap_bytes(n);
        uint8_t *bm0 = *out_ptr;
        uint8_t *bm1 = *out_ptr + nbytes;
        *out_ptr += 2 * nbytes;

        /* Build both bitmaps and partition in one fused pass */
        int n_ll = 0, n_lr = 0, n_rl = 0, n_rr = 0;
        /* Layout in tmp: [LR data] [RL data] [RR data]
           LL goes in-place into indices */
        uint16_t *out_lr = tmp;
        uint16_t *out_rl = tmp;  /* offset adjusted after loop */
        uint16_t *out_rr = tmp;  /* offset adjusted after loop */
        int j = 0;

        for (; j + 8 <= n; j += 8) {
            /* Build mask bytes from codes */
            uint8_t b0 = 0, b1 = 0;
            for (int k = 0; k < 8; k++) {
                int idx = indices[j + k];
                int shift = lens[idx] - 1 - depth;
                b0 |= (uint8_t)(((codes[idx] >> shift) & 1) << k);
                b1 |= (uint8_t)(((codes[idx] >> (shift - 1)) & 1) << k);
            }
            bm0[j >> 3] = b0;
            bm1[j >> 3] = b1;

            partition_8_4way(indices + j, b0, b1,
                             indices + n_ll, &n_ll,
                             tmp + n_lr, &n_lr,
                             tmp + (PIVCO_BLOCK_SIZE) + n_rl, &n_rl,
                             tmp + (PIVCO_BLOCK_SIZE) + (PIVCO_BLOCK_SIZE / 2) + n_rr, &n_rr);
        }
        /* Scalar remainder */
        if (j < n) {
            uint8_t b0 = 0, b1 = 0;
            for (int k = 0; j + k < n; k++) {
                int idx = indices[j + k];
                int shift = lens[idx] - 1 - depth;
                b0 |= (uint8_t)(((codes[idx] >> shift) & 1) << k);
                b1 |= (uint8_t)(((codes[idx] >> (shift - 1)) & 1) << k);
            }
            bm0[j >> 3] = b0;
            bm1[j >> 3] = b1;
            for (int k = 0; j < n; j++, k++) {
                int code2 = (((b0 >> k) & 1) << 1) | ((b1 >> k) & 1);
                switch (code2) {
                case 0: indices[n_ll++] = indices[j]; break;
                case 1: tmp[n_lr++] = indices[j]; break;
                case 2: tmp[PIVCO_BLOCK_SIZE + n_rl++] = indices[j]; break;
                case 3: tmp[PIVCO_BLOCK_SIZE * 2 + n_rr++] = indices[j]; break;
                }
            }
        }

        uint16_t *lr_base = tmp;
        uint16_t *rl_base = tmp + PIVCO_BLOCK_SIZE;
        uint16_t *rr_base = tmp + PIVCO_BLOCK_SIZE * 2;

        encode_node_neon2(table, lc->left, indices, n_ll,
                          depth + 2, codes, lens, out_ptr,
                          rr_base + n_rr);
        encode_node_neon2(table, lc->right, lr_base, n_lr,
                          depth + 2, codes, lens, out_ptr,
                          rr_base + n_rr);
        encode_node_neon2(table, rc->left, rl_base, n_rl,
                          depth + 2, codes, lens, out_ptr,
                          rr_base + n_rr);
        encode_node_neon2(table, rc->right, rr_base, n_rr,
                          depth + 2, codes, lens, out_ptr,
                          rr_base + n_rr);
    } else {
        /* Normal 1-bit: at least one child is a leaf */
        int nbytes = bitmap_bytes(n);
        uint8_t *bm = *out_ptr;
        *out_ptr += nbytes;

        int n_left = 0, n_right = 0;
        int j = 0;

        for (; j + 8 <= n; j += 8) {
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

        encode_node_neon2(table, node->left, indices, n_left,
                          depth + 1, codes, lens, out_ptr, tmp + n_right);
        encode_node_neon2(table, node->right, tmp, n_right,
                          depth + 1, codes, lens, out_ptr, tmp + n_right);
    }
}

int pivco_encode_neon2(const uint8_t *symbols,
                                const pivco_table_t *table,
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

    /* Heap-allocate scratch: 4-way partition needs ~3N per level,
       with up to MAX_CODE_LEN levels. Allocate generously. */
    uint16_t *tmp = (uint16_t *)malloc((size_t)PIVCO_BLOCK_SIZE * 8 * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_OVERFLOW;
    uint8_t *ptr = out;

    encode_node_neon2(table, table->tree_root, indices, N,
                       0, codes, lens, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    free(tmp);
    return PIVCO_OK;
}

/* ---------- Decode (Tree-Walk with 4-way fusion) ---------- */

static void decode_node_neon2(const pivco_table_t *table,
                               int16_t node_id,
                               uint16_t *indices, int n,
                               uint8_t *symbols,
                               const uint8_t **in_ptr,
                               uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        scatter_write(symbols, indices, n, (uint8_t)node->symbol);
        return;
    }

    const pivco_tree_node_t *lc = &table->tree[node->left];
    const pivco_tree_node_t *rc = &table->tree[node->right];

    if (lc->symbol < 0 && rc->symbol < 0) {
        /* 4-way: read 2 bitmaps, partition into 4 groups */
        int nbytes = bitmap_bytes(n);
        const uint8_t *bm0 = *in_ptr;
        const uint8_t *bm1 = *in_ptr + nbytes;
        *in_ptr += 2 * nbytes;

        int n_ll = 0, n_lr = 0, n_rl = 0, n_rr = 0;
        int j = 0;

        for (; j + 8 <= n; j += 8) {
            partition_8_4way(indices + j, bm0[j >> 3], bm1[j >> 3],
                             indices + n_ll, &n_ll,
                             tmp + n_lr, &n_lr,
                             tmp + PIVCO_BLOCK_SIZE + n_rl, &n_rl,
                             tmp + PIVCO_BLOCK_SIZE * 2 + n_rr, &n_rr);
        }
        for (; j < n; j++) {
            int bit0 = bitmap_get(bm0, j);
            int bit1 = bitmap_get(bm1, j);
            int code2 = (bit0 << 1) | bit1;
            switch (code2) {
            case 0: indices[n_ll++] = indices[j]; break;
            case 1: tmp[n_lr++] = indices[j]; break;
            case 2: tmp[PIVCO_BLOCK_SIZE + n_rl++] = indices[j]; break;
            case 3: tmp[PIVCO_BLOCK_SIZE * 2 + n_rr++] = indices[j]; break;
            }
        }

        /* Layout: LL in indices, LR in tmp[0..], RL in tmp[BS..], RR in tmp[BS+BS/2..]
           Each recursive call gets scratch AFTER all preserved group data.
           DFS order: process LL first (uses scratch past all groups),
           then LR, then RL, then RR. Each call's scratch can overlap
           with previously-completed calls' data since it's dead. */
        uint16_t *lr_base = tmp;
        uint16_t *rl_base = tmp + PIVCO_BLOCK_SIZE;
        uint16_t *rr_base = tmp + PIVCO_BLOCK_SIZE * 2;

        /* LL: scratch must be past ALL group data */
        decode_node_neon2(table, lc->left, indices, n_ll,
                          symbols, in_ptr, rr_base + n_rr);
        /* LR: LL is done, its data (in indices) no longer needs tmp.
           scratch past RL+RR data */
        decode_node_neon2(table, lc->right, lr_base, n_lr,
                          symbols, in_ptr, rr_base + n_rr);
        /* RL: LR is done, tmp[0..n_lr-1] is dead.
           scratch past RR data */
        decode_node_neon2(table, rc->left, rl_base, n_rl,
                          symbols, in_ptr, rr_base + n_rr);
        /* RR: RL is done. scratch can reuse earlier space */
        decode_node_neon2(table, rc->right, rr_base, n_rr,
                          symbols, in_ptr, rr_base + n_rr);
    } else {
        /* Normal 1-bit partition */
        int nbytes = bitmap_bytes(n);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;

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

        decode_node_neon2(table, node->left, indices, n_left,
                          symbols, in_ptr, tmp + n_right);
        decode_node_neon2(table, node->right, tmp, n_right,
                          symbols, in_ptr, tmp + n_right);
    }
}

int pivco_decode_neon2(const uint8_t *in, size_t in_len,
                                const pivco_table_t *table,
                                uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;

    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    uint16_t *tmp = (uint16_t *)malloc((size_t)PIVCO_BLOCK_SIZE * 8 * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_OVERFLOW;
    const uint8_t *ptr = in;

    decode_node_neon2(table, table->tree_root, indices, N,
                       symbols, &ptr, tmp);

    *consumed = (size_t)(ptr - in);
    free(tmp);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
