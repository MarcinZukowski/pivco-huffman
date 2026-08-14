/* bench_kr_storage.c -- compare wire-format storage cost of K_right
 * (per non-flat internal-with-non-leaf-child) vs K_leaf (per leaf /
 * skip / flat-subtree terminal) for each distribution.
 *
 * Both candidate formats trade ~1-3% encoded-size bloat for the
 * elimination of bu_popcount_K.  This analyzer computes the actual
 * per-block bytes needed under each scheme, using:
 *   - tree topology (which determines the number of storage slots)
 *   - per-block symbol histogram (which determines each slot's value)
 *
 * No popcount required: K_right at a node = sum of histogram values
 * for symbols rooted in its RIGHT subtree.
 *
 * Reports: fixed-2-byte cost and varint cost (1B if K<128 else 2B)
 * per distribution, plus % bloat vs encoded block size.
 */
#include "pivco_huffman.h"
#include "bench_ctx.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void           bench_init(void);
extern int            bench_num_distributions(void);
extern const char    *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int            bench_dist_is_main(int idx);
extern void           bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                              int n_symbols, uint64_t seed);

#define BLK PIVCO_BLOCK_SIZE
#define TOTAL_SYMBOLS (4 * 1024 * 1024)
#define ENC_SLOT (2 * PIVCO_BLOCK_SIZE)

/* Sum histogram values for all leaves in the subtree rooted at node_id. */
static int subtree_hist_sum(const pivco_table_t *t, int16_t node_id,
                             const int *hist)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return hist[n->symbol];
    return subtree_hist_sum(t, n->left, hist)
         + subtree_hist_sum(t, n->right, hist);
}

typedef struct {
    int kr_slots;        /* # K_right entries needed per block (max possible) */
    int kleaf_slots;     /* # K_leaf entries needed per block */

    /* Per-block actual values across all sampled blocks. */
    int blocks_seen;
    long long kr_bytes_fixed2;
    long long kr_bytes_varint;
    long long kleaf_bytes_fixed2;
    long long kleaf_bytes_varint;
    long long encoded_bytes;
} stats_t;

/* Walk the dispatch tree, accumulate K_right sites and their K_right
 * values for this block.  Mirrors the popcount call-sites in
 * pivco_bu_x86.c. */
static void walk_kr(const pivco_table_t *t, int16_t node_id, int K,
                     const int *hist, stats_t *s)
{
    if (K == 0) return;
    const pivco_tree_node_t *n = &t->tree[node_id];
    pivco_node_type_t type = (pivco_node_type_t)t->node_type[node_id];
    switch (type) {
    case PIVCO_NODE_LEAF:
    case PIVCO_NODE_INTERNAL_FLAT:
    case PIVCO_NODE_BOTH_LEAVES:
        return;  /* no K_right write site */
    case PIVCO_NODE_LEAF_LEFT: {
        int K_right = subtree_hist_sum(t, n->right, hist);
        s->kr_bytes_fixed2 += 2;
        s->kr_bytes_varint += (K_right < 128) ? 1 : 2;
        walk_kr(t, n->right, K_right, hist, s);
        return;
    }
    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        int K_right = subtree_hist_sum(t, n->right, hist);
        int K_left  = subtree_hist_sum(t, n->left,  hist);
        s->kr_bytes_fixed2 += 2;
        s->kr_bytes_varint += (K_right < 128) ? 1 : 2;
        walk_kr(t, n->left,  K_left,  hist, s);
        walk_kr(t, n->right, K_right, hist, s);
        return;
    }
    }
}

/* Walk the dispatch tree, accumulate K_leaf sites (= leaves / skips /
 * flat-subtree terminals) and their K_leaf values for this block. */
static void walk_kleaf(const pivco_table_t *t, int16_t node_id,
                        const int *hist, stats_t *s)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    pivco_node_type_t type = (pivco_node_type_t)t->node_type[node_id];
    switch (type) {
    case PIVCO_NODE_LEAF: {
        int K = hist[n->symbol];
        s->kleaf_bytes_fixed2 += 2;
        s->kleaf_bytes_varint += (K < 128) ? 1 : 2;
        return;
    }
    case PIVCO_NODE_INTERNAL_FLAT: {
        int K = subtree_hist_sum(t, node_id, hist);
        s->kleaf_bytes_fixed2 += 2;
        s->kleaf_bytes_varint += (K < 128) ? 1 : 2;
        return;
    }
    case PIVCO_NODE_BOTH_LEAVES:
    case PIVCO_NODE_LEAF_LEFT:
    case PIVCO_NODE_INTERNAL_FULL:
    default:
        walk_kleaf(t, n->left,  hist, s);
        walk_kleaf(t, n->right, hist, s);
        return;
    }
}

/* Count topology slots once (don't need hist). */
static int count_kr_slots(const pivco_table_t *t, int16_t node_id)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    pivco_node_type_t type = (pivco_node_type_t)t->node_type[node_id];
    switch (type) {
    case PIVCO_NODE_LEAF:
    case PIVCO_NODE_INTERNAL_FLAT:
    case PIVCO_NODE_BOTH_LEAVES: return 0;
    case PIVCO_NODE_LEAF_LEFT:
        return 1 + count_kr_slots(t, n->right);
    case PIVCO_NODE_INTERNAL_FULL:
    default:
        return 1 + count_kr_slots(t, n->left) + count_kr_slots(t, n->right);
    }
}

static int count_kleaf_slots(const pivco_table_t *t, int16_t node_id)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    pivco_node_type_t type = (pivco_node_type_t)t->node_type[node_id];
    switch (type) {
    case PIVCO_NODE_LEAF:
    case PIVCO_NODE_INTERNAL_FLAT:
        return 1;
    default:
        return count_kleaf_slots(t, n->left) + count_kleaf_slots(t, n->right);
    }
}

static void analyze_dist(int dist_idx, int blocks)
{
    const char *name = bench_dist_name(dist_idx);
    const uint64_t *freq = bench_dist_freq(dist_idx);
    pivco_table_t table;
    if (pivco_build_table(bench_cfg(), freq, &table) != PIVCO_OK) {
        printf("%-13s | build_table failed\n", name); return;
    }

    stats_t s = {0};
    s.kr_slots    = count_kr_slots(&table, table.tree_root);
    s.kleaf_slots = count_kleaf_slots(&table, table.tree_root);

    uint8_t *symbols = (uint8_t *)malloc(TOTAL_SYMBOLS);
    bench_generate_symbols(dist_idx, symbols, TOTAL_SYMBOLS, 0xFEEDC0DE);
    uint8_t *enc = (uint8_t *)malloc(ENC_SLOT);

    int total_blocks_in_stream = TOTAL_SYMBOLS / BLK;
    if (blocks > total_blocks_in_stream) blocks = total_blocks_in_stream;
    s.blocks_seen = blocks;

    for (int b = 0; b < blocks; b++) {
        int hist[256] = {0};
        for (int i = 0; i < BLK; i++) hist[symbols[b * BLK + i]]++;

        size_t enc_len;
        if (pivco_encode(bench_enc_ctx(), &table, symbols + (size_t)b * BLK, BLK, enc, &enc_len) == PIVCO_OK) {
            s.encoded_bytes += enc_len;
        }
        walk_kr(&table, table.tree_root, BLK, hist, &s);
        walk_kleaf(&table, table.tree_root, hist, &s);
    }

    double avg_enc = (double)s.encoded_bytes / s.blocks_seen;
    double kr_f2_per_blk    = (double)s.kr_bytes_fixed2 / s.blocks_seen;
    double kr_var_per_blk   = (double)s.kr_bytes_varint / s.blocks_seen;
    double kleaf_f2_per_blk = (double)s.kleaf_bytes_fixed2 / s.blocks_seen;
    double kleaf_var_per_blk= (double)s.kleaf_bytes_varint / s.blocks_seen;

    printf("%-13s | %4d / %4d | %5.0f / %5.0f / %5.0f / %5.0f | "
           "%5.2f%% / %5.2f%% / %5.2f%% / %5.2f%%\n",
           name, s.kr_slots, s.kleaf_slots,
           kr_f2_per_blk, kr_var_per_blk, kleaf_f2_per_blk, kleaf_var_per_blk,
           100.0 * kr_f2_per_blk    / avg_enc,
           100.0 * kr_var_per_blk   / avg_enc,
           100.0 * kleaf_f2_per_blk / avg_enc,
           100.0 * kleaf_var_per_blk/ avg_enc);

    free(symbols); free(enc);
}

int main(int argc, char **argv)
{
    int blocks = (argc > 1) ? atoi(argv[1]) : 100;
    bench_init();

    printf("=== K_right vs K_leaf wire-format storage cost ===\n");
    printf("Block size: %d.  Sampled %d blocks per dist.\n", BLK, blocks);
    printf("Slots: # storage entries per block (topology-only).\n");
    printf("Bytes: avg per-block storage cost (fixed-2-byte / varint 1B<128).\n");
    printf("Bloat: cost relative to encoded block size.\n\n");

    printf("%-13s | slots K_r/K_l | bytes/block: kr_f2 / kr_var / kl_f2 / kl_var | "
           "bloat: kr_f2 / kr_var / kl_f2 / kl_var\n", "dist");
    printf("--------------+---------------+-----------------------------------------+"
           "----------------------------------------\n");
    int n = bench_num_distributions();
    for (int i = 0; i < n; i++) {
        if (!bench_dist_is_main(i)) continue;
        analyze_dist(i, blocks);
    }
    return 0;
}
