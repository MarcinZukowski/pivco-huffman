/* bench_partition_skew — histogram of partition-node skewness per
 * distribution, weighted by node population.
 *
 * For each internal Huffman node, "skewness" = smaller-side fraction =
 * min(left_freq, right_freq) / total_freq_at_node, in percent.  0%
 * means the partition is totally one-sided (1 element on the small
 * side, all the rest on the big side); 50% means perfectly balanced.
 *
 * Histogram is weighted by element count flowing through each node
 * (= total subtree freq), so a row of "30 / 25 / 20 / 15 / 10" reads
 * as "30% of decoded elements pass through nodes that are nearly
 * one-sided, …, 10% pass through nearly-balanced nodes".
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

#define NBINS 11   /* 0-5, 5-10, 10-15, ..., 45-50 */

/* Returns subtree freq, accumulates weight × bin into hist. */
static double walk(const pivco_table_t *t, int16_t node,
                    const uint64_t *freq, double hist[NBINS])
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) return (double)freq[n->symbol];

    double left  = walk(t, n->left, freq, hist);
    double right = walk(t, n->right, freq, hist);
    double total = left + right;
    if (total > 0) {
        double small_side = (left < right) ? left : right;
        double skew_pct = 100.0 * small_side / total;   /* 0..50 */
        int bin = (int)(skew_pct / 5.0);
        if (bin >= NBINS) bin = NBINS - 1;
        hist[bin] += total;   /* weight by node population */
    }
    return total;
}

static void print_one(const char *name, const uint64_t *freq)
{
    pivco_table_t t;
    if (pivco_build_table(bench_cfg(), freq, &t) != PIVCO_OK) {
        printf("%-15s  build_table failed\n", name);
        return;
    }
    double hist[NBINS] = {0};
    walk(&t, t.tree_root, freq, hist);

    double total = 0;
    for (int i = 0; i < NBINS; i++) total += hist[i];
    if (total <= 0) return;

    /* Element-weighted average of smaller-side fraction (0..50). */
    double mean_skew = 0;
    for (int i = 0; i < NBINS; i++) {
        double mid_pct = i * 5.0 + 2.5;
        mean_skew += (hist[i] / total) * mid_pct;
    }

    printf("%-15s |", name);
    for (int i = 0; i < NBINS; i++) {
        double pct = 100.0 * hist[i] / total;
        if (pct < 0.05)      printf("    . ");
        else if (pct < 9.95) printf(" %4.1f ", pct);
        else                 printf(" %4.0f ", pct);
    }
    printf("| %4.1f\n", mean_skew);
}

int main(void)
{
    bench_init();
    int n = bench_num_distributions();

    printf("Partition-node skewness histogram, weighted by element count.\n");
    printf("Each cell = %% of decoded elements that pass through an internal\n");
    printf("node whose smaller-side fraction (min(L,R) / total) falls in that bin.\n\n");

    printf("%-15s |  0–5  5–10 10–15 15–20 20–25 25–30 30–35 35–40 40–45 45–50    .   | mean\n", "distribution");
    printf("                |  (one-sided)  ←──── more balanced ────→  (50/50)        | skew\n");
    printf("----------------+-----------------------------------------------------------+------\n");

    for (int i = 0; i < n; i++) {
        print_one(bench_dist_name(i), bench_dist_freq(i));
    }

    printf("\nLegend:\n");
    printf("  bins are 5%% wide, covering smaller-side fractions 0–50%%\n");
    printf("  '.' = <0.05%%   mean skew = element-weighted average min(L,R)/total\n");
    printf("  high mean (closer to 50%%) → balanced partitions, branch-on-popcount unpredictable\n");
    printf("  low mean (closer to 0%%)  → one-sided partitions, branch-on-popcount predictable\n");
    return 0;
}
