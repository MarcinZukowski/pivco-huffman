/* pivco_neon2b.c — 4-way fused partition, ping-pong/packed scratch.
 *
 * Idea: when both children of an internal node are also internal, read TWO
 * code bits per element (parent bit + grandparent bit) and partition into
 * 4 groups (LL, LR, RL, RR) in one pass.  This skips one full partition
 * level's worth of loads / stores / function calls.
 *
 * Scratch layout (differs from neon2):
 *   LL stays in-place in `indices`         (n_ll elements)
 *   LR / RL / RR packed in `tmp` with 8-uint16 SAFETY GAPS between them to
 *   absorb the trailing-zero overflow from `vst1q_u8` (which always writes
 *   16 bytes, regardless of the group's popcount for the chunk):
 *     tmp[0 ..                       n_lr)           = LR
 *     tmp[n_lr + 8 ..                n_lr+8+n_rl)    = RL
 *     tmp[n_lr+n_rl+16 .. n_lr+n_rl+16+n_rr)         = RR
 *   Children recurse with scratch = tmp + (n - n_ll) + 24 — past all groups
 *   and the trailing RR-overflow gap.
 *
 * Correctness: two-pass partition.
 *   Pass 1 scans both bitmaps and computes n_ll, n_lr, n_rl, n_rr via
 *   popcount; Pass 2 emits compacted writes with each group's pre-computed
 *   offset.  No inter-group overflow because destinations are known up front.
 */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>
#include <stdlib.h>

#ifdef PIVCO_HAS_NEON
#include <arm_neon.h>

/* Shared with pivco_neon.c */
extern uint8_t compress_tab[256][32];
extern uint8_t compress_popcnt[256];
extern void    init_compress_table(void);

/* When n falls below this, 4-way overhead exceeds savings — use 2-way. */
#ifndef PIVCO_NEON2B_MIN_N
#define PIVCO_NEON2B_MIN_N 32
#endif

/* ---------- Basic partition helpers (same as neon backend) ---------- */

static inline int partition_8(const uint16_t *src, uint8_t mask,
                               uint16_t *left_out, uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    vst1q_u8((uint8_t *)right_out, vqtbl1q_u8(data, shuf_r));
    vst1q_u8((uint8_t *)left_out,  vqtbl1q_u8(data, shuf_l));
    return compress_popcnt[mask];
}

static inline int partition_8_right(const uint16_t *src, uint8_t mask,
                                     uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
    vst1q_u8((uint8_t *)right_out, vqtbl1q_u8(data, shuf_r));
    return compress_popcnt[mask];
}

static inline int partition_8_left(const uint16_t *src, uint8_t mask,
                                    uint16_t *left_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
    vst1q_u8((uint8_t *)left_out, vqtbl1q_u8(data, shuf_l));
    return 8 - compress_popcnt[mask];
}

/* ---------- Scatter helpers (copied from neon backend) ---------- */

static inline void scatter_sym(uint8_t *symbols,
                                const uint16_t *indices, int n, uint8_t sym)
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

/* ---------- 4-way counting (Pass 1) ----------
 *
 * For each pair of bitmap bytes (bm0, bm1), count elements landing in each
 * of the 4 groups via popcount identities:
 *   p11 = popcount(bm0 & bm1)            -> RR
 *   p10 = popcount(bm0) - p11            -> RL
 *   p01 = popcount(bm1) - p11            -> LR
 *   p00 = 8 - p0 - p1 + p11              -> LL   (per byte)
 *
 * Last byte is masked to only valid bits.
 */
static inline void count_4way(const uint8_t *bm0, const uint8_t *bm1, int n,
                               int *n_ll, int *n_lr, int *n_rl, int *n_rr)
{
    int c_ll = 0, c_lr = 0, c_rl = 0, c_rr = 0;
    int full = n >> 3;
    for (int i = 0; i < full; i++) {
        uint8_t b0 = bm0[i], b1 = bm1[i];
        int p11 = __builtin_popcount(b0 & b1);
        int p0  = __builtin_popcount(b0);
        int p1  = __builtin_popcount(b1);
        c_rr += p11;
        c_rl += p0 - p11;
        c_lr += p1 - p11;
        c_ll += 8 - p0 - p1 + p11;
    }
    int tail = n & 7;
    if (tail) {
        uint8_t mask = (uint8_t)((1u << tail) - 1u);
        uint8_t b0 = bm0[full] & mask;
        uint8_t b1 = bm1[full] & mask;
        int p11 = __builtin_popcount(b0 & b1);
        int p0  = __builtin_popcount(b0);
        int p1  = __builtin_popcount(b1);
        c_rr += p11;
        c_rl += p0 - p11;
        c_lr += p1 - p11;
        c_ll += tail - p0 - p1 + p11;
    }
    *n_ll = c_ll; *n_lr = c_lr; *n_rl = c_rl; *n_rr = c_rr;
}

/* ---------- 4-way partition of a single 8-element chunk ----------
 *
 * Given 8 indices + 2 mask bytes (b0, b1), emit each element to its group's
 * current write position.  Each group's offset is tracked by the caller.
 */
static inline void partition_8_4way(const uint16_t *src, uint8_t b0, uint8_t b1,
                                     uint16_t *out_ll, int *o_ll,
                                     uint16_t *out_lr, int *o_lr,
                                     uint16_t *out_rl, int *o_rl,
                                     uint16_t *out_rr, int *o_rr)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);

    uint8_t m_ll = (uint8_t)(~b0 & ~b1);
    uint8_t m_lr = (uint8_t)(~b0 &  b1);
    uint8_t m_rl = (uint8_t)( b0 & ~b1);
    uint8_t m_rr = (uint8_t)( b0 &  b1);

    vst1q_u8((uint8_t *)(out_ll + *o_ll),
             vqtbl1q_u8(data, vld1q_u8(compress_tab[m_ll])));
    *o_ll += compress_popcnt[m_ll];

    vst1q_u8((uint8_t *)(out_lr + *o_lr),
             vqtbl1q_u8(data, vld1q_u8(compress_tab[m_lr])));
    *o_lr += compress_popcnt[m_lr];

    vst1q_u8((uint8_t *)(out_rl + *o_rl),
             vqtbl1q_u8(data, vld1q_u8(compress_tab[m_rl])));
    *o_rl += compress_popcnt[m_rl];

    vst1q_u8((uint8_t *)(out_rr + *o_rr),
             vqtbl1q_u8(data, vld1q_u8(compress_tab[m_rr])));
    *o_rr += compress_popcnt[m_rr];
}

/* ---------- Encode ---------- */

static void encode_node_neon2b(const pivco_table_t *table,
                                int16_t node_id,
                                uint16_t *indices, int n,
                                int depth,
                                const uint16_t *codes, const uint8_t *lens,
                                uint8_t **out_ptr,
                                uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;

    const pivco_tree_node_t *lc = &table->tree[node->left];
    const pivco_tree_node_t *rc = &table->tree[node->right];
    int left_leaf  = (lc->symbol >= 0);
    int right_leaf = (rc->symbol >= 0);

    /* 4-way path: both children internal AND big enough to amortize */
    if (!left_leaf && !right_leaf && n >= PIVCO_NEON2B_MIN_N) {
        int nbytes = bitmap_bytes(n);
        uint8_t *bm0 = *out_ptr;
        uint8_t *bm1 = *out_ptr + nbytes;
        memset(bm0, 0, (size_t)(2 * nbytes));
        *out_ptr += 2 * nbytes;

        /* Build bm0 (parent bit = depth) and bm1 (grandparent bit = depth+1). */
        for (int j = 0; j < n; j++) {
            int idx = indices[j];
            int len = lens[idx];
            uint16_t c = codes[idx];
            int b0 = (c >> (len - 1 - depth)) & 1;
            int b1 = (c >> (len - 2 - depth)) & 1;
            if (b0) bm0[j >> 3] |= (uint8_t)(1u << (j & 7));
            if (b1) bm1[j >> 3] |= (uint8_t)(1u << (j & 7));
        }

        int n_ll, n_lr, n_rl, n_rr;
        count_4way(bm0, bm1, n, &n_ll, &n_lr, &n_rl, &n_rr);

        /* Pass 2: partition into groups. LL in-place (indices),
           LR/RL/RR packed into tmp with 8-uint16 gaps absorbing TBL overflow. */
        int o_ll = 0;
        int o_lr = 0;
        int o_rl = n_lr + 8;
        int o_rr = n_lr + n_rl + 16;

        int j = 0;
        for (; j + 8 <= n; j += 8) {
            partition_8_4way(indices + j, bm0[j >> 3], bm1[j >> 3],
                             indices, &o_ll,
                             tmp, &o_lr,
                             tmp, &o_rl,
                             tmp, &o_rr);
        }
        if (j < n) {
            for (int k = 0; j + k < n; k++) {
                int b0 = (bm0[j >> 3] >> k) & 1;
                int b1 = (bm1[j >> 3] >> k) & 1;
                uint16_t idx = indices[j + k];
                int code = (b0 << 1) | b1;
                switch (code) {
                case 0: indices[o_ll++] = idx; break;
                case 1: tmp[o_lr++] = idx; break;
                case 2: tmp[o_rl++] = idx; break;
                case 3: tmp[o_rr++] = idx; break;
                }
            }
        }

        /* Recurse on each grandchild in DFS order. Children get scratch
           past all 3 live groups + their safety gaps (24 uint16 total). */
        uint16_t *child_tmp = tmp + (n - n_ll) + 24;
        encode_node_neon2b(table, lc->left,  indices,                   n_ll,
                           depth + 2, codes, lens, out_ptr, child_tmp);
        encode_node_neon2b(table, lc->right, tmp,                       n_lr,
                           depth + 2, codes, lens, out_ptr, child_tmp);
        encode_node_neon2b(table, rc->left,  tmp + n_lr + 8,            n_rl,
                           depth + 2, codes, lens, out_ptr, child_tmp);
        encode_node_neon2b(table, rc->right, tmp + n_lr + n_rl + 16,    n_rr,
                           depth + 2, codes, lens, out_ptr, child_tmp);
        return;
    }

    /* 2-way fallback (matches neon.c encode format). */
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
        n_left  += (8 - nr);
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
            if (mask & (1u << k))
                tmp[n_right++] = indices[j];
            else
                indices[n_left++] = indices[j];
        }
    }

    encode_node_neon2b(table, node->left,  indices, n_left,
                       depth + 1, codes, lens, out_ptr, tmp + n_right);
    encode_node_neon2b(table, node->right, tmp,     n_right,
                       depth + 1, codes, lens, out_ptr, tmp + n_right);
}

int pivco_encode_neon2b(const uint8_t *symbols,
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

    /* Scratch: worst-case sum of group sizes along DFS path in 4-way mode.
       Generous 4N upper bound is enough for max_code_len = 15 in practice
       (4-way depth is bounded, groups shrink geometrically). */
    /* Stack-allocated scratch — avoids per-call malloc/free overhead. */
    uint16_t tmp_stack[PIVCO_BLOCK_SIZE * 8];
    uint16_t *tmp = tmp_stack;

    uint8_t *ptr = out;
    encode_node_neon2b(table, table->tree_root, indices, N,
                        0, codes, lens, &ptr, tmp);
    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- Decode ---------- */

static void decode_node_neon2b(const pivco_table_t *table,
                                int16_t node_id,
                                uint16_t *indices, int n,
                                uint8_t *symbols,
                                const uint8_t **in_ptr,
                                uint16_t *tmp,
                                int16_t skip_node)
{
    if (n == 0) return;
    if (node_id == skip_node) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        scatter_sym(symbols, indices, n, (uint8_t)node->symbol);
        return;
    }

    const pivco_tree_node_t *lc = &table->tree[node->left];
    const pivco_tree_node_t *rc = &table->tree[node->right];
    int left_leaf  = (lc->symbol >= 0);
    int right_leaf = (rc->symbol >= 0);

    /* 4-way path: both children internal AND large enough. */
    if (!left_leaf && !right_leaf && n >= PIVCO_NEON2B_MIN_N) {
        int nbytes = bitmap_bytes(n);
        const uint8_t *bm0 = *in_ptr;
        const uint8_t *bm1 = *in_ptr + nbytes;
        *in_ptr += 2 * nbytes;

        int n_ll, n_lr, n_rl, n_rr;
        count_4way(bm0, bm1, n, &n_ll, &n_lr, &n_rl, &n_rr);

        int o_ll = 0;
        int o_lr = 0;
        int o_rl = n_lr + 8;
        int o_rr = n_lr + n_rl + 16;

        int j = 0;
        for (; j + 8 <= n; j += 8) {
            partition_8_4way(indices + j, bm0[j >> 3], bm1[j >> 3],
                             indices, &o_ll,
                             tmp, &o_lr,
                             tmp, &o_rl,
                             tmp, &o_rr);
        }
        if (j < n) {
            for (int k = 0; j + k < n; k++) {
                int b0 = (bm0[j >> 3] >> k) & 1;
                int b1 = (bm1[j >> 3] >> k) & 1;
                uint16_t idx = indices[j + k];
                int code = (b0 << 1) | b1;
                switch (code) {
                case 0: indices[o_ll++] = idx; break;
                case 1: tmp[o_lr++] = idx; break;
                case 2: tmp[o_rl++] = idx; break;
                case 3: tmp[o_rr++] = idx; break;
                }
            }
        }

        uint16_t *child_tmp = tmp + (n - n_ll) + 24;
        decode_node_neon2b(table, lc->left,  indices,                    n_ll,
                           symbols, in_ptr, child_tmp, skip_node);
        decode_node_neon2b(table, lc->right, tmp,                        n_lr,
                           symbols, in_ptr, child_tmp, skip_node);
        decode_node_neon2b(table, rc->left,  tmp + n_lr + 8,             n_rl,
                           symbols, in_ptr, child_tmp, skip_node);
        decode_node_neon2b(table, rc->right, tmp + n_lr + n_rl + 16,     n_rr,
                           symbols, in_ptr, child_tmp, skip_node);
        return;
    }

    /* 2-way fallback with the same stage-fusion optimizations as neon.c. */
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

    if (left_leaf && right_leaf
        && node->left != skip_node && node->right != skip_node) {
        scatter_both_leaves(symbols, indices, n, bm,
                            (uint8_t)lc->symbol, (uint8_t)rc->symbol);
        return;
    }

    if (left_leaf && node->left == skip_node) {
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
        decode_node_neon2b(table, node->right, tmp, n_right,
                           symbols, in_ptr, tmp + n_right, skip_node);
        return;
    }
    if (right_leaf && node->right == skip_node) {
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
        decode_node_neon2b(table, node->left, indices, n_left,
                           symbols, in_ptr, tmp, skip_node);
        return;
    }

    int n_left = 0, n_right = 0;
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8(indices + j, mask,
                             indices + n_left, tmp + n_right);
        n_right += nr;
        n_left  += (8 - nr);
    }
    for (; j < n; j++) {
        if (bitmap_get(bm, j))
            tmp[n_right++] = indices[j];
        else
            indices[n_left++] = indices[j];
    }

    if (left_leaf) {
        if (node->left != skip_node)
            scatter_sym(symbols, indices, n_left, (uint8_t)lc->symbol);
        decode_node_neon2b(table, node->right, tmp, n_right,
                           symbols, in_ptr, tmp + n_right, skip_node);
    } else if (right_leaf) {
        if (node->right != skip_node)
            scatter_sym(symbols, tmp, n_right, (uint8_t)rc->symbol);
        decode_node_neon2b(table, node->left, indices, n_left,
                           symbols, in_ptr, tmp + n_right, skip_node);
    } else {
        decode_node_neon2b(table, node->left, indices, n_left,
                           symbols, in_ptr, tmp + n_right, skip_node);
        decode_node_neon2b(table, node->right, tmp, n_right,
                           symbols, in_ptr, tmp + n_right, skip_node);
    }
}

int pivco_decode_neon2b(const uint8_t *in, size_t in_len,
                                 const pivco_table_t *table,
                                 uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Prefill with most frequent symbol; skip_node check elides its scatter. */
    memset(symbols, table->prefill_sym, (size_t)N);
    int16_t skip_node = table->prefill_node;

    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    /* Stack-allocated scratch — avoids per-call malloc/free overhead. */
    uint16_t tmp_stack[PIVCO_BLOCK_SIZE * 8];
    uint16_t *tmp = tmp_stack;

    decode_node_neon2b(table, table->tree_root, indices, N,
                        symbols, &ptr, tmp, skip_node);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
