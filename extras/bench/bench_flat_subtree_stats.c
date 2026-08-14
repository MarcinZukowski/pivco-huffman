/* Flat-subtree applicability analyzer.
 *
 * For each benchmark distribution, walk the Huffman tree and find every
 * MAXIMAL flat subtree — an internal node whose subtree has
 * local_min == local_max == D for some D >= 2, and whose parent's subtree
 * is NOT flat (i.e., the subtree is maximal: we cannot extend it upward
 * while preserving flatness).
 *
 * Equivalently: DFS from root; at any internal node whose subtree is flat
 * with depth D >= 2, record (D, weight) and prune recursion.  Elsewhere,
 * recurse into both children.
 *
 * Weight of a flat subtree = sum of frequencies of its 2^D leaves.  Report
 * per-depth counts + element-weighted coverage, bucketed at D=2 (4 leaves),
 * D=3 (8), D=4 (16), D=5 (32), D=6 (64), and a tail for D >= 7.
 *
 * Separately flag the "entire tree is flat" case (root has local_min ==
 * local_max) — that case is already handled by the existing full-tree
 * flat fast path, so it shouldn't be counted as new addressable weight
 * for a flat-subtree fast path.
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

/* Walk: at each internal node, if it is a maximal flat-subtree root, count
 * it and stop.  Otherwise recurse into children.  Reads the table's flat
 * metadata (flat_depth / flat_offset / flat_code_to_sym) rather than
 * recomputing flatness by descending the subtree -- since the flat-leaf-skip
 * build, flat roots have no materialized children to descend into. */
static void collect_flat(const pivco_table_t *t,
                          int16_t node_id,
                          const uint64_t *freq,
                          uint64_t *w_by_depth,  /* depth 0..15 */
                          int *count_by_depth)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return;   /* single leaf: no subtree */

    int D = t->flat_depth[node_id];
    if (D >= 2) {
        int d = D > 15 ? 15 : D;
        count_by_depth[d] += 1;
        uint64_t w = 0;
        int off = t->flat_offset[node_id];
        for (int k = 0; k < (1 << D); k++) w += freq[t->flat_code_to_sym[off + k]];
        w_by_depth[d] += w;
        return;  /* maximal flat subtree — don't descend further */
    }

    collect_flat(t, n->left,  freq, w_by_depth, count_by_depth);
    collect_flat(t, n->right, freq, w_by_depth, count_by_depth);
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

    int min_len = t->min_len;
    int max_len = t->max_len;
    int root_flat = (min_len == max_len);

    uint64_t total_freq = 0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) total_freq += freq[s];

    uint64_t w_by_depth[16] = {0};
    int      c_by_depth[16] = {0};

    /* Special-case the root: if the whole tree is flat, we report it as
     * "root flat" and do not descend into the flat-subtree tally (the
     * existing full-tree flat path already handles this). */
    if (!root_flat) {
        collect_flat(t, t->tree_root, freq, w_by_depth, c_by_depth);
    }

    uint64_t w_total_flat = 0;
    for (int i = 0; i < 16; i++) {
        w_total_flat += w_by_depth[i];
    }

    printf("%-14s | min=%d max=%d ", name, min_len, max_len);
    if (root_flat) {
        printf("| ROOT FLAT (handled by full-tree fast path)\n");
        free(t);
        return;
    }
    printf("| %5.1f%% elems in maximal flat subtrees (D>=2): ",
           total_freq ? 100.0 * (double)w_total_flat / (double)total_freq : 0.0);

    /* Bucketed columns: D=2..8 (a 256-symbol alphabet caps flat depth at 8) */
    int d_list[] = {2, 3, 4, 5, 6, 7, 8};
    for (size_t i = 0; i < sizeof(d_list) / sizeof(d_list[0]); i++) {
        int D = d_list[i];
        double pct = total_freq ?
            100.0 * (double)w_by_depth[D] / (double)total_freq : 0.0;
        printf("D=%d(%d,%5.2f%%) ", D, c_by_depth[D], pct);
    }
    printf("\n");
    free(t);
}

int main(void)
{
    bench_init();
    int n = bench_num_distributions();
    printf("Flat-subtree applicability analysis\n");
    printf("===================================\n");
    printf("Maximal flat subtrees: subtree is flat (local_min == local_max >= 2)\n");
    printf("AND its parent is not flat.  Depth D means 2^D leaves.\n");
    printf("Per-column values: (count of such subtrees, %% of elements).\n\n");
    printf("%-14s | %s | %s\n",
           "distribution",
           "len-span     ",
           "coverage: total, D=2(4 lvs), D=3(8), D=4(16), D=5(32), D=6(64), D=7(128), D=8(256)");
    for (int d = 0; d < n; d++) analyze_distribution(d);
    return 0;
}
