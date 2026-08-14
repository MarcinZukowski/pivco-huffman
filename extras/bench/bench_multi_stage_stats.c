/* One-off analyzer: measure where multi-stage prefix-radix would fire.
 *
 * Model: a single-stage prefix-radix at the root uses M_top = table->min_len,
 * producing K = 2^M_top bins.  Each bin is either:
 *   - a LEAF bin (the M_top-bit prefix is a complete code), or
 *   - a SUBTREE bin (the prefix is a proper prefix of longer codes).
 *
 * For each SUBTREE bin, compute local_min — the shallowest leaf depth
 * within that subtree relative to its root.  Multi-stage radix would fire
 * on a subtree bin iff local_min >= 2.
 *
 * This reduces to asking whether the Huffman code-length histogram has a
 * gap right after min_len.  We report that explicitly per distribution,
 * plus the element-weighted fraction where multi-stage would engage.
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

extern void         bench_init(void);
extern int          bench_num_distributions(void);
extern const char  *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

/* Shortest leaf depth relative to this node, i.e. local_min. */
static int compute_local_min(const pivco_table_t *t, int16_t node_id)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return 0;
    int l = compute_local_min(t, n->left);
    int r = compute_local_min(t, n->right);
    return 1 + (l < r ? l : r);
}

/* Deepest leaf depth relative to this node. */
__attribute__((unused))
static int compute_local_max(const pivco_table_t *t, int16_t node_id)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return 0;
    int l = compute_local_max(t, n->left);
    int r = compute_local_max(t, n->right);
    return 1 + (l > r ? l : r);
}

/* Walk M_top bits from the root following `prefix` (MSB-first, matching the
 * canonical code layout), return the resulting tree node id. */
static int16_t walk_prefix(const pivco_table_t *t, uint32_t prefix, int M)
{
    int16_t node_id = t->tree_root;
    for (int b = M - 1; b >= 0; b--) {
        const pivco_tree_node_t *n = &t->tree[node_id];
        if (n->symbol >= 0) return node_id;
        int bit = (prefix >> b) & 1;
        node_id = bit ? n->right : n->left;
    }
    return node_id;
}

static void analyze_distribution(int d)
{
    const char *name = bench_dist_name(d);
    const uint64_t *freq = bench_dist_freq(d);

    pivco_table_t *t =
        (pivco_table_t *)malloc(sizeof(pivco_table_t));
    if (pivco_build_table(bench_cfg(), freq, t) != PIVCO_OK) {
        printf("%-14s | build_table failed\n", name);
        free(t);
        return;
    }

    int M_top = t->min_len;
    int max_len = t->max_len;
    int K = 1 << M_top;

    /* Total symbol frequency (for weighting). */
    uint64_t total_freq = 0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) total_freq += freq[s];

    /* Length histogram of Huffman codes. */
    int len_count[PIVCO_MAX_CODE_LEN + 2] = {0};
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (t->code_len[s] > 0) len_count[t->code_len[s]]++;
    }

    /* Weight per M_top-bit bin — sum of frequencies of symbols whose code
     * starts with that prefix. */
    uint64_t *bin_weight = (uint64_t *)calloc((size_t)K, sizeof(uint64_t));
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (t->code_len[s] == 0) continue;
        int L = t->code_len[s];
        /* M_top <= L always since M_top == min_len. */
        int bin = (int)(t->code[s] >> (L - M_top));
        bin_weight[bin] += freq[s];
    }

    /* Classify each bin + compute local_min for subtree bins. */
    int leaf_bins = 0;
    int subtree_bins = 0;
    uint64_t w_leaf = 0, w_subtree = 0;
    /* Per-local_min weight among subtree bins. */
    uint64_t w_by_lmin[16] = {0};
    int count_by_lmin[16] = {0};

    for (int v = 0; v < K; v++) {
        int16_t node = walk_prefix(t, (uint32_t)v, M_top);
        const pivco_tree_node_t *n = &t->tree[node];
        if (n->symbol >= 0) {
            leaf_bins++;
            w_leaf += bin_weight[v];
        } else {
            subtree_bins++;
            int lmin = compute_local_min(t, node);
            if (lmin > 15) lmin = 15;
            w_by_lmin[lmin] += bin_weight[v];
            count_by_lmin[lmin]++;
            w_subtree += bin_weight[v];
        }
    }

    /* Code-length gap right after min_len — the direct predictor of
     * whether multi-stage fires. */
    int has_len_mplus1 = (len_count[M_top + 1] > 0);

    printf("\n=== %s ===\n", name);
    printf("  min_len=%d  max_len=%d  K=%d (bins)\n", M_top, max_len, K);
    printf("  code-length histogram:");
    int first = 1;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        if (len_count[L] > 0) {
            printf("%s %d:%d", first ? "" : ",", L, len_count[L]);
            first = 0;
        }
    }
    printf("\n");
    printf("  gap after min_len? %s  (codes of length %d %s)\n",
           has_len_mplus1 ? "NO" : "YES",
           M_top + 1, has_len_mplus1 ? "exist" : "absent");

    printf("  bins: %d leaf, %d subtree (total %d)\n",
           leaf_bins, subtree_bins, K);

    if (total_freq > 0) {
        double p_leaf = 100.0 * (double)w_leaf / (double)total_freq;
        double p_subtree = 100.0 * (double)w_subtree / (double)total_freq;
        printf("  element-weighted: leaf bins %.2f%%, subtree bins %.2f%%\n",
               p_leaf, p_subtree);
    }

    if (subtree_bins > 0) {
        printf("  subtree bins by local_min (element-weighted %% of all elems):\n");
        uint64_t w_multistage_ge2 = 0;
        uint64_t w_multistage_ge3 = 0;
        for (int lmin = 1; lmin < 16; lmin++) {
            if (count_by_lmin[lmin] == 0) continue;
            double pct_elems = total_freq ?
                100.0 * (double)w_by_lmin[lmin] / (double)total_freq : 0.0;
            printf("    local_min=%d: %d subtree bins, %.2f%% of elements\n",
                   lmin, count_by_lmin[lmin], pct_elems);
            if (lmin >= 2) w_multistage_ge2 += w_by_lmin[lmin];
            if (lmin >= 3) w_multistage_ge3 += w_by_lmin[lmin];
        }
        printf("  *** multi-stage radix addressable (local_min >= 2): "
               "%.2f%% of elements\n",
               total_freq ? 100.0 * (double)w_multistage_ge2 / (double)total_freq : 0.0);
        printf("  *** multi-stage savings would exceed 1 level (local_min >= 3): "
               "%.2f%% of elements\n",
               total_freq ? 100.0 * (double)w_multistage_ge3 / (double)total_freq : 0.0);
    }

    free(bin_weight);
    free(t);
}

int main(void)
{
    bench_init();
    int n = bench_num_distributions();
    printf("Multi-stage prefix-radix applicability analysis\n");
    printf("===============================================\n");
    for (int d = 0; d < n; d++) analyze_distribution(d);
    return 0;
}
