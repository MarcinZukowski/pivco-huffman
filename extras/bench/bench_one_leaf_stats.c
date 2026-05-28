/* One-off analyzer: at each internal tree node visited during decode,
 * check if one child is a leaf (the OTHER child being internal) AND that
 * leaf is NOT the prefill leaf.  For those nodes, record n_left/n — the
 * fraction of elements at the parent that go to the leaf side.
 *
 * We weight every sample by n (the number of elements at that node),
 * since large nodes contribute proportionally more partition work.
 *
 * Output: per-distribution histogram of leaf-side share in 10% bins,
 * plus the total fraction of decode "work" (element-visits at internal
 * nodes) that lands in the non-prefill one-leaf case at all.
 *
 * We run this in pure scalar — we're measuring tree structure + input
 * symbols, not decode perf.  Uses a 4M symbol sequence sliced into
 * blocks of PIVCO_BLOCK_SIZE so block-level skew is included.
 */

#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

extern void         bench_init(void);
extern int          bench_num_distributions(void);
extern const char  *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern void         bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                           int n_symbols, uint64_t seed);

#define TOTAL_SYMBOLS (4 * 1024 * 1024)
#define BLK PIVCO_BLOCK_SIZE
#define NBLOCKS (TOTAL_SYMBOLS / BLK)
#define SEED 0xBEEFCAFE12345678ULL

#define NBINS 10  /* 10% buckets: [0,10), [10,20), ..., [90,100] */

typedef struct {
    /* Elements-weighted counters */
    uint64_t elems_total;                 /* sum of n across ALL internal nodes visited */
    uint64_t elems_one_leaf_nonprefill;   /* sum of n at non-prefill one-leaf nodes */
    uint64_t elems_one_leaf_prefill;      /* sum of n at prefill one-leaf nodes (skip_node path) */
    uint64_t elems_both_leaves;           /* sum of n at both-leaves nodes */
    uint64_t elems_both_internal;         /* sum of n at both-internal nodes */

    /* Histogram of leaf-side share at non-prefill one-leaf nodes,
       weighted by n_elements (so a 60% bin entry of size 1000 counts
       as 1000 sample-elements in that bin). */
    uint64_t hist_elems[NBINS];
    /* Same histogram weighted by node count (each node = 1 sample,
       regardless of size). */
    uint64_t hist_nodes[NBINS];
} stats_t;

/* Partition `indices` by the code bit at `depth` (MSB-first, matches
 * encoder).  Returns leaf-side count; rearranges indices in-place:
 * left (bit=0) at [0..n_left), right (bit=1) at [n_left..n). */
static int partition_inplace(uint16_t *indices, int n, int depth,
                              const uint16_t *codes, const uint8_t *lens,
                              uint16_t *scratch)
{
    int n_left = 0, n_right = 0;
    for (int j = 0; j < n; j++) {
        int idx = indices[j];
        int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
        if (bit) scratch[n_right++] = indices[j];
        else     indices[n_left++] = indices[j];
    }
    for (int j = 0; j < n_right; j++) indices[n_left + j] = scratch[j];
    return n_left;
}

static void walk(const pivco_huffman_table_t *table,
                 int16_t node_id, int depth,
                 uint16_t *indices, int n,
                 const uint16_t *codes, const uint8_t *lens,
                 uint16_t *scratch, stats_t *st)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;  /* leaf: no partition work here */

    /* Don't count root — question is about non-root nodes. */
    int is_root = (node_id == table->tree_root);

    const pivco_tree_node_t *lc = &table->tree[node->left];
    const pivco_tree_node_t *rc = &table->tree[node->right];
    int left_leaf  = (lc->symbol >= 0);
    int right_leaf = (rc->symbol >= 0);

    int n_left = partition_inplace(indices, n, depth, codes, lens, scratch);
    int n_right = n - n_left;

    if (!is_root) {
        st->elems_total += (uint64_t)n;
        if (left_leaf && right_leaf) {
            st->elems_both_leaves += (uint64_t)n;
        } else if (!left_leaf && !right_leaf) {
            st->elems_both_internal += (uint64_t)n;
        } else {
            /* One-leaf case.  Determine the leaf side + whether it's prefill. */
            int leaf_is_left = left_leaf;
            int16_t leaf_node = leaf_is_left ? node->left : node->right;
            int leaf_count = leaf_is_left ? n_left : n_right;
            int is_prefill = (leaf_node == table->prefill_node);
            if (is_prefill) {
                st->elems_one_leaf_prefill += (uint64_t)n;
            } else {
                st->elems_one_leaf_nonprefill += (uint64_t)n;
                /* Bin leaf-side share of total n. */
                int bin = (int)((uint64_t)leaf_count * NBINS / (uint64_t)n);
                if (bin >= NBINS) bin = NBINS - 1;
                st->hist_elems[bin] += (uint64_t)n;
                st->hist_nodes[bin] += 1;
            }
        }
    }

    /* Recurse into both children (only internal ones contribute). */
    if (!left_leaf) {
        walk(table, node->left, depth + 1, indices, n_left,
             codes, lens, scratch, st);
    }
    if (!right_leaf) {
        walk(table, node->right, depth + 1, indices + n_left, n_right,
             codes, lens, scratch, st);
    }
}

static void run_distribution(int d, const uint8_t *symbols_all, int nblocks)
{
    const char *name = bench_dist_name(d);
    const uint64_t *freq = bench_dist_freq(d);
    pivco_huffman_table_t *table = (pivco_huffman_table_t *)
        malloc(sizeof(pivco_huffman_table_t));
    pivco_huffman_build_table(freq, table);

    /* Precompute codes/lens per symbol. */
    uint16_t codes_s[PIVCO_MAX_SYMBOLS];
    uint8_t  lens_s[PIVCO_MAX_SYMBOLS];
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        codes_s[i] = table->code[i];
        lens_s[i]  = table->code_len[i];
    }

    stats_t st; memset(&st, 0, sizeof(st));

    uint16_t *indices = (uint16_t *)malloc(BLK * sizeof(uint16_t));
    uint16_t *scratch = (uint16_t *)malloc(BLK * sizeof(uint16_t));

    /* Re-key indices to code/len per position so walk can read directly. */
    uint16_t *codes_pos = (uint16_t *)malloc(BLK * sizeof(uint16_t));
    uint8_t  *lens_pos  = (uint8_t  *)malloc(BLK * sizeof(uint8_t));

    for (int b = 0; b < nblocks; b++) {
        const uint8_t *blk = symbols_all + (size_t)b * BLK;
        for (int i = 0; i < BLK; i++) {
            codes_pos[i] = codes_s[blk[i]];
            lens_pos[i]  = lens_s[blk[i]];
            indices[i] = (uint16_t)i;
        }
        walk(table, table->tree_root, 0, indices, BLK,
             codes_pos, lens_pos, scratch, &st);
    }

    /* Denominator for percentages: total element-visits at non-root
       internal nodes (all four categories combined). */
    uint64_t total = st.elems_total;
    if (total == 0) {
        printf("%-14s | no non-root internals\n", name);
        goto cleanup;
    }

    uint64_t one_leaf_np = st.elems_one_leaf_nonprefill;
    printf("\n=== %s ===\n", name);
    printf("  non-root internal element-visits: %" PRIu64 "\n", total);
    printf("    both-leaves:         %6.2f%%\n",
           100.0 * (double)st.elems_both_leaves / (double)total);
    printf("    both-internal:       %6.2f%%\n",
           100.0 * (double)st.elems_both_internal / (double)total);
    printf("    one-leaf PREFILL:    %6.2f%%  (fast path, no scatter)\n",
           100.0 * (double)st.elems_one_leaf_prefill / (double)total);
    printf("    one-leaf non-prefill:%6.2f%%  (candidate for speculative)\n",
           100.0 * (double)one_leaf_np / (double)total);

    if (one_leaf_np == 0) goto cleanup;

    printf("  leaf-side share within non-prefill one-leaf nodes:\n");
    printf("    bin            | %% of elems | %% of nodes\n");
    for (int b = 0; b < NBINS; b++) {
        int lo = b * (100 / NBINS);
        int hi = (b + 1) * (100 / NBINS);
        uint64_t elems = st.hist_elems[b];
        uint64_t nodes = st.hist_nodes[b];
        double pct_e = 100.0 * (double)elems / (double)one_leaf_np;
        uint64_t node_total = 0;
        for (int k = 0; k < NBINS; k++) node_total += st.hist_nodes[k];
        double pct_n = node_total ? 100.0 * (double)nodes / (double)node_total : 0;
        printf("    [%3d-%3d%%)    | %8.2f%% | %8.2f%%\n", lo, hi, pct_e, pct_n);
    }

    /* Summary: cumulative above 60%, 70%, 80%. */
    uint64_t ge60 = 0, ge70 = 0, ge80 = 0;
    for (int b = 0; b < NBINS; b++) {
        int lo = b * (100 / NBINS);
        if (lo >= 60) ge60 += st.hist_elems[b];
        if (lo >= 70) ge70 += st.hist_elems[b];
        if (lo >= 80) ge80 += st.hist_elems[b];
    }
    printf("  cumulative (within non-prefill one-leaf, elem-weighted):\n");
    printf("    >= 60%% leaf share: %6.2f%%\n", 100.0 * (double)ge60 / (double)one_leaf_np);
    printf("    >= 70%% leaf share: %6.2f%%\n", 100.0 * (double)ge70 / (double)one_leaf_np);
    printf("    >= 80%% leaf share: %6.2f%%  <-- speculative break-even on M4\n",
           100.0 * (double)ge80 / (double)one_leaf_np);

    printf("  *** of ALL partition work, share where speculative would win:\n");
    printf("      %6.3f%% (>=80%% leaf) of all non-root element-visits\n",
           100.0 * (double)ge80 / (double)total);

cleanup:
    free(indices); free(scratch); free(codes_pos); free(lens_pos);
    free(table);
}

int main(void)
{
    bench_init();
    int n_dist = bench_num_distributions();

    uint8_t *symbols = (uint8_t *)malloc(TOTAL_SYMBOLS);

    for (int d = 0; d < n_dist; d++) {
        bench_generate_symbols(d, symbols, TOTAL_SYMBOLS, SEED);
        run_distribution(d, symbols, NBLOCKS);
    }

    free(symbols);
    return 0;
}
