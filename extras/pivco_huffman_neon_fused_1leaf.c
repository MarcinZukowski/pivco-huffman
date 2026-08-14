/* pivco_neon_fused_1leaf.c — FAILED EXPERIMENT (kept for reference).
 *
 * Tried to fuse the partition + leaf-side scatter at non-prefill one-leaf
 * nodes: keep the TBL-compacted leaf indices IN REGISTER and drain them
 * via lane-extract + strb, instead of writing them to memory and reading
 * them back during a separate scatter pass.
 *
 * Three variants tried; all slower than the baseline neon backend on M4:
 *   1. switch/fallthrough dispatch   — compiler emitted binary-search
 *                                      tree of compares; mispredicts
 *                                      dominated.  proba80 −50%.
 *   2. computed-goto (real jump tbl) — indirect branch instead of tree.
 *                                      Same or worse.  proba80 −60%.
 *   3. Trick 2: bucket chunks by     — recovered most of the regression
 *      n_left, then straight-line      but still −36% proba80 due to the
 *      per-bucket scatter (no           two-pass re-read of indices and
 *      dispatch at all).                extra vld+TBL per chunk.
 *
 * Core lesson: on M4 at 9+ GB/s, the baseline's bulk `scatter_sym` that
 * amortizes its vld over 8 *leaf* elements is already efficient enough
 * that ANY per-chunk rework of the same elements pays more in extra
 * loads/TBLs than it saves in avoided stores.
 *
 * See extras/README_FUSED_1LEAF.md for the full writeup.
 *
 * The code below is the Trick 2 (bucketed) version — the best of the
 * three variants.  It compiles standalone against the exported NEON
 * tables from pivco_neon.c but is not built or wired into the
 * library.  Encoded format matches the baseline neon encoder.
 */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>

#ifdef PIVCO_HAS_NEON
#include <arm_neon.h>

/* Shared with pivco_neon.c */
extern uint8_t compress_tab[256][32];
extern uint8_t compress_popcnt[256];
extern void    init_compress_table(void);

/* ---------- Shared 2-way primitives (copied inline for visibility) ---------- */

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

/* ---------- Jump-table fused partition + scatter ---------- */

/* ---------- Trick 2: bucketed scatter ----------
 *
 * At a one-leaf non-prefill node, do two passes over the chunks:
 *
 *   Phase 1: stream through chunks, partition RIGHT (non-leaf) side to
 *            tmp, and record each chunk's n_left (leaf-side count).
 *
 *   Phase 2: bucket-sort chunk IDs by n_left into 8 buckets.
 *
 *   Phase 3: process buckets 1..8 each with its OWN straight-line loop;
 *            inside, v is a compile-time constant so the N strbs are
 *            fully unrolled with no dispatch / no mispredicting branch.
 *
 * The v-as-const inline helper expands every `if (v >= k)` at compile
 * time, leaving only the `k` stores that are actually needed. */
static inline __attribute__((always_inline))
void scatter_v_lanes(uint8_t *symbols, uint16x8_t idx, uint8_t sym, int v) {
    if (v >= 1) symbols[vgetq_lane_u16(idx, 0)] = sym;
    if (v >= 2) symbols[vgetq_lane_u16(idx, 1)] = sym;
    if (v >= 3) symbols[vgetq_lane_u16(idx, 2)] = sym;
    if (v >= 4) symbols[vgetq_lane_u16(idx, 3)] = sym;
    if (v >= 5) symbols[vgetq_lane_u16(idx, 4)] = sym;
    if (v >= 6) symbols[vgetq_lane_u16(idx, 5)] = sym;
    if (v >= 7) symbols[vgetq_lane_u16(idx, 6)] = sym;
    if (v >= 8) symbols[vgetq_lane_u16(idx, 7)] = sym;
}

/* Per-bucket inner loop with compile-time-constant V.
 * LEFT_OR_RIGHT selects which side of the TBL table to read (0 = left
 * shuffle at tab+16, 1 = right shuffle at tab+0). */
#define BUCKETED_SCATTER_LOOP(V, LEFT_OR_RIGHT, SYM)                     \
    do {                                                                 \
        int __end = bucket_offset[(V) + 1];                              \
        for (int __k = bucket_offset[V]; __k < __end; __k++) {           \
            int __c = bucket_chunks[__k];                                \
            uint8_t __m = bm[__c];                                       \
            uint8x16_t __d = vld1q_u8((const uint8_t *)(indices + __c*8));\
            const uint8_t *__shuf = (LEFT_OR_RIGHT) ? compress_tab[__m]  \
                                                    : compress_tab[__m] + 16;\
            uint16x8_t __li = vreinterpretq_u16_u8(                      \
                vqtbl1q_u8(__d, vld1q_u8(__shuf)));                      \
            scatter_v_lanes(symbols, __li, (SYM), (V));                  \
        }                                                                \
    } while (0)

/* ---------- Decode ---------- */

static void decode_node_neon_jt(const pivco_table_t *table,
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

    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

    const pivco_tree_node_t *left_child  = &table->tree[node->left];
    const pivco_tree_node_t *right_child = &table->tree[node->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    /* Both-leaves stage fusion (neither prefilled). */
    if (left_leaf && right_leaf
        && node->left != skip_node && node->right != skip_node) {
        scatter_both_leaves(symbols, indices, n, bm,
                            (uint8_t)left_child->symbol,
                            (uint8_t)right_child->symbol);
        return;
    }

    /* Prefill half-partition paths — left leaf is skip_node. */
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
        decode_node_neon_jt(table, node->right, tmp, n_right,
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
        decode_node_neon_jt(table, node->left, indices, n_left,
                            symbols, in_ptr, tmp, skip_node);
        return;
    }

    /* Trick 2 bucketed scatter: left-leaf non-prefill. */
    if (left_leaf) {
        uint8_t sym_L = (uint8_t)left_child->symbol;
        int full_chunks = n >> 3;

        /* Phase 1: right-side partition to tmp; record n_left per chunk. */
        uint8_t  n_left_arr[PIVCO_BLOCK_SIZE / 8];
        int n_right = 0;
        for (int c = 0; c < full_chunks; c++) {
            uint8_t mask = bm[c];
            uint8x16_t data = vld1q_u8((const uint8_t *)(indices + c*8));
            uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
            vst1q_u8((uint8_t *)(tmp + n_right),
                     vqtbl1q_u8(data, shuf_r));
            uint8_t nr = compress_popcnt[mask];
            n_right += nr;
            n_left_arr[c] = (uint8_t)(8 - nr);
        }

        /* Phase 2: bucket-sort chunk IDs by n_left ∈ [0..8]. */
        int bucket_count[9] = {0};
        for (int c = 0; c < full_chunks; c++) bucket_count[n_left_arr[c]]++;
        int bucket_offset[10];
        bucket_offset[0] = 0;
        for (int v = 0; v < 9; v++)
            bucket_offset[v+1] = bucket_offset[v] + bucket_count[v];
        uint16_t bucket_chunks[PIVCO_BLOCK_SIZE / 8];
        int place[9];
        for (int v = 0; v < 9; v++) place[v] = bucket_offset[v];
        for (int c = 0; c < full_chunks; c++) {
            int v = n_left_arr[c];
            bucket_chunks[place[v]++] = (uint16_t)c;
        }

        /* Phase 3: per-bucket straight-line scatter (V is compile-time const,
         * scatter_v_lanes unrolls to exactly V strbs, no branches). */
        /* Bucket 0 scatters nothing — skip. */
        BUCKETED_SCATTER_LOOP(1, 0, sym_L);
        BUCKETED_SCATTER_LOOP(2, 0, sym_L);
        BUCKETED_SCATTER_LOOP(3, 0, sym_L);
        BUCKETED_SCATTER_LOOP(4, 0, sym_L);
        BUCKETED_SCATTER_LOOP(5, 0, sym_L);
        BUCKETED_SCATTER_LOOP(6, 0, sym_L);
        BUCKETED_SCATTER_LOOP(7, 0, sym_L);
        BUCKETED_SCATTER_LOOP(8, 0, sym_L);

        /* Scalar tail — handle any elements past the last full chunk. */
        for (int j = full_chunks * 8; j < n; j++) {
            if (bitmap_get(bm, j)) tmp[n_right++] = indices[j];
            else                   symbols[indices[j]] = sym_L;
        }

        decode_node_neon_jt(table, node->right, tmp, n_right,
                            symbols, in_ptr, tmp + n_right, skip_node);
        return;
    }
    if (right_leaf) {
        uint8_t sym_R = (uint8_t)right_child->symbol;
        int full_chunks = n >> 3;

        /* Phase 1: left-side partition (in-place into indices) + record n_right. */
        uint8_t  n_right_arr[PIVCO_BLOCK_SIZE / 8];
        uint8_t  mask_arr[PIVCO_BLOCK_SIZE / 8];
        /* Input indices must be preserved for Phase 3's re-read of per-chunk
         * original data.  In this symmetric case we can't do in-place left
         * compaction first — that would clobber the originals.  Instead
         * compact left into tmp (swap roles), then pass tmp as the recursive
         * indices. */
        int n_left = 0;
        for (int c = 0; c < full_chunks; c++) {
            uint8_t mask = bm[c];
            mask_arr[c] = mask;
            uint8x16_t data = vld1q_u8((const uint8_t *)(indices + c*8));
            uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
            vst1q_u8((uint8_t *)(tmp + n_left),
                     vqtbl1q_u8(data, shuf_l));
            uint8_t nr = compress_popcnt[mask];
            n_left += (8 - nr);
            n_right_arr[c] = nr;
        }

        /* Phase 2: bucket-sort chunk IDs by n_right ∈ [0..8]. */
        int bucket_count[9] = {0};
        for (int c = 0; c < full_chunks; c++) bucket_count[n_right_arr[c]]++;
        int bucket_offset[10];
        bucket_offset[0] = 0;
        for (int v = 0; v < 9; v++)
            bucket_offset[v+1] = bucket_offset[v] + bucket_count[v];
        uint16_t bucket_chunks[PIVCO_BLOCK_SIZE / 8];
        int place[9];
        for (int v = 0; v < 9; v++) place[v] = bucket_offset[v];
        for (int c = 0; c < full_chunks; c++) {
            int v = n_right_arr[c];
            bucket_chunks[place[v]++] = (uint16_t)c;
        }

        /* Phase 3: per-bucket straight-line scatter of right (bit=1) positions. */
        BUCKETED_SCATTER_LOOP(1, 1, sym_R);
        BUCKETED_SCATTER_LOOP(2, 1, sym_R);
        BUCKETED_SCATTER_LOOP(3, 1, sym_R);
        BUCKETED_SCATTER_LOOP(4, 1, sym_R);
        BUCKETED_SCATTER_LOOP(5, 1, sym_R);
        BUCKETED_SCATTER_LOOP(6, 1, sym_R);
        BUCKETED_SCATTER_LOOP(7, 1, sym_R);
        BUCKETED_SCATTER_LOOP(8, 1, sym_R);
        (void)mask_arr; /* currently unused — kept for symmetry */

        for (int j = full_chunks * 8; j < n; j++) {
            if (bitmap_get(bm, j)) symbols[indices[j]] = sym_R;
            else                   tmp[n_left++] = indices[j];
        }

        /* Recurse with tmp as the left-side indices (compacted during phase 1). */
        decode_node_neon_jt(table, node->left, tmp, n_left,
                            symbols, in_ptr, tmp + n_left, skip_node);
        return;
    }

    /* Both-internal: standard full partition + recurse both. */
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

    decode_node_neon_jt(table, node->left, indices, n_left,
                        symbols, in_ptr, tmp + n_right, skip_node);
    decode_node_neon_jt(table, node->right, tmp, n_right,
                        symbols, in_ptr, tmp + n_right, skip_node);
}

/* Root-level wrapper — mirrors pivco_decode_neon's root dispatch,
 * only difference is it calls decode_node_neon_jt for subtree work. */
int pivco_decode_neon_jt(const uint8_t *in, size_t in_len,
                                  const pivco_table_t *table,
                                  uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    int nbytes = bitmap_bytes(N);
    const uint8_t *bm = ptr;
    ptr += nbytes;

    const pivco_tree_node_t *left_child  = &table->tree[root->left];
    const pivco_tree_node_t *right_child = &table->tree[root->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf) {
        uint8_t sym0 = (uint8_t)left_child->symbol;
        uint8_t sym1 = (uint8_t)right_child->symbol;
        uint8x8_t vsym0  = vdup_n_u8(sym0);
        uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);
        static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
        uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);
        int j = 0;
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

    uint8_t prefill_sym = table->prefill_sym;
    memset(symbols, prefill_sym, (size_t)N);
    int16_t skip_node = table->prefill_node;

    uint16_t indices[PIVCO_BLOCK_SIZE];
    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];

    if (left_leaf && root->left == skip_node) {
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
        decode_node_neon_jt(table, root->right, tmp, n_right,
                            symbols, &ptr, tmp + n_right, skip_node);
    } else if (right_leaf && root->right == skip_node) {
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
        decode_node_neon_jt(table, root->left, indices, n_left,
                            symbols, &ptr, tmp, skip_node);
    } else {
        /* Root full partition with identity indices. */
        static const uint16_t off[8] = {0,1,2,3,4,5,6,7};
        int n_left = 0, n_right = 0;
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            uint8_t mask = bm[j >> 3];
            uint8x16_t data = vreinterpretq_u8_u16(
                vaddq_u16(vdupq_n_u16((uint16_t)j), vld1q_u16(off)));
            uint8x16_t shuf_r = vld1q_u8(compress_tab[mask]);
            uint8x16_t shuf_l = vld1q_u8(compress_tab[mask] + 16);
            vst1q_u8((uint8_t *)(indices + n_left),
                     vqtbl1q_u8(data, shuf_l));
            vst1q_u8((uint8_t *)(tmp + n_right),
                     vqtbl1q_u8(data, shuf_r));
            n_right += compress_popcnt[mask];
            n_left += 8 - compress_popcnt[mask];
        }
        for (; j < N; j++) {
            if (bitmap_get(bm, j))
                tmp[n_right++] = (uint16_t)j;
            else
                indices[n_left++] = (uint16_t)j;
        }

        if (left_leaf) {
            /* Root left-leaf, not skip_node.  Scatter then recurse right. */
            scatter_sym(symbols, indices, n_left,
                        (uint8_t)left_child->symbol);
            decode_node_neon_jt(table, root->right, tmp, n_right,
                                symbols, &ptr, tmp + n_right, skip_node);
        } else if (right_leaf) {
            scatter_sym(symbols, tmp, n_right,
                        (uint8_t)right_child->symbol);
            decode_node_neon_jt(table, root->left, indices, n_left,
                                symbols, &ptr, tmp + n_right, skip_node);
        } else {
            decode_node_neon_jt(table, root->left, indices, n_left,
                                symbols, &ptr, tmp + n_right, skip_node);
            decode_node_neon_jt(table, root->right, tmp, n_right,
                                symbols, &ptr, tmp + n_right, skip_node);
        }
    }

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
