/* bench_tree_table: per-dataset tree-shape ablation stats for the paper.
 *
 * For each main dataset, prints H, weighted Huffman code length, and per-mode
 * (NAIVE / CANONICAL_FLAT / OPTIMIZED) effective tree node count + freq-weighted
 * ops/byte.  CSV output is fed to a small awk/python wrapper that emits a .typ
 * table.
 *
 * Definitions:
 *   - effective node count = nodes that emit an operation in the BU traversal.
 *     INTERNAL_FULL / INTERNAL_FLAT / HALF_* / BOTH_LEAVES / LEAF count;
 *     SKIP and the leaves *inside* a flat subtree do not (the flat-root absorbs
 *     them).
 *   - ops/byte = freq-weighted: Σ freq[s] × ops(s) / Σ freq[s], where ops(s) is
 *     the number of primitive invocations required to emit one byte of symbol s
 *     (depth-from-root for a regular leaf, flat_root_depth+1 for a leaf inside
 *     a flat-D subtree).
 *   - weighted code length = Σ freq[s] × depth(s) / Σ freq[s] — mode-independent
 *     for canonical Huffman; OPTIMIZED reshuffles preserve average depth.
 */
#include "pivco_huffman.h"
#include "bench_ctx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void        bench_init(void);
extern int         bench_num_distributions(void);
extern const char *bench_dist_name(int idx);
extern int         bench_dist_is_main(int idx);
extern void        bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                          int n_symbols, unsigned seed);

#define N_SYMBOLS  (1 << 20)
#define SEED       0xC0FFEEu

static double shannon_H(const uint64_t *freq) {
    uint64_t total = 0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) total += freq[s];
    if (!total) return 0.0;
    double H = 0.0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (!freq[s]) continue;
        double p = (double)freq[s] / (double)total;
        H -= p * log2(p);
    }
    return H;
}

/* Recursive walk to compute weighted ops/byte and weighted code length.
 * Flat-D>=2 subtrees are terminal: their 2^D leaves are NOT materialized
 * in tree[] (build_table absorbs them into flat_code_to_sym), so we
 * enumerate them via flat_offset / flat_code_to_sym.  All such leaves
 * share ops = depth+1 and Huffman depth = depth+D. */
static void leaf_stats_walk(const pivco_table_t *t,
                             int16_t node, int depth,
                             const uint64_t *freq,
                             uint64_t *out_total_freq,
                             double *out_freq_ops,
                             double *out_freq_depth) {
    if (t->tree[node].symbol >= 0) {
        int ops = depth;
        int sym = t->tree[node].symbol;
        uint64_t fw = freq[sym];
        *out_total_freq += fw;
        *out_freq_ops   += (double)fw * (double)ops;
        *out_freq_depth += (double)fw * (double)depth;
        return;
    }
    if (t->flat_depth[node] >= 2) {
        int D = t->flat_depth[node];
        int off = t->flat_offset[node];
        int n = 1 << D;
        int ops = depth + 1;      /* one flat_decode op for whole subtree */
        int leaf_depth = depth + D;
        for (int i = 0; i < n; i++) {
            int sym = t->flat_code_to_sym[off + i];
            uint64_t fw = freq[sym];
            *out_total_freq += fw;
            *out_freq_ops   += (double)fw * (double)ops;
            *out_freq_depth += (double)fw * (double)leaf_depth;
        }
        return;
    }
    leaf_stats_walk(t, t->tree[node].left,  depth+1, freq,
                    out_total_freq, out_freq_ops, out_freq_depth);
    leaf_stats_walk(t, t->tree[node].right, depth+1, freq,
                    out_total_freq, out_freq_ops, out_freq_depth);
}

/* Count internal nodes that emit a merge primitive in the BU traversal.
 * Leaves produce constants — no op. A flat-D>=2 subtree root counts once
 * (its children are absorbed and not materialized in tree[]). */
static int count_op_nodes(const pivco_table_t *t, int16_t node) {
    if (t->tree[node].symbol >= 0) return 0;        /* leaf */
    if (t->flat_depth[node] >= 2)  return 1;        /* flat root: 1 merge */
    int c = 1;
    c += count_op_nodes(t, t->tree[node].left);
    c += count_op_nodes(t, t->tree[node].right);
    return c;
}

typedef struct {
    int    node_count;
    double ops_per_byte;
} mode_stats_t;

static mode_stats_t build_and_measure(pivco_tree_mode_t mode,
                                       const uint64_t *freq,
                                       double *out_weighted_code_len) {
    mode_stats_t st = {0};
    bench_cfg()->tree_mode = (mode);

    pivco_table_t *t = calloc(1, sizeof(*t));
    if (pivco_build_table(bench_cfg(), freq, t) != PIVCO_OK) {
        fprintf(stderr, "build_table failed for mode %d\n", (int)mode);
        free(t);
        return st;
    }

    uint64_t tot_f = 0;
    double f_ops = 0.0, f_depth = 0.0;
    leaf_stats_walk(t, t->tree_root, 0, freq, &tot_f, &f_ops, &f_depth);
    st.ops_per_byte = tot_f ? f_ops / (double)tot_f : 0.0;
    if (out_weighted_code_len)
        *out_weighted_code_len = tot_f ? f_depth / (double)tot_f : 0.0;
    st.node_count = count_op_nodes(t, t->tree_root);
    free(t);
    return st;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    bench_init();

    uint8_t *sym = malloc(N_SYMBOLS);
    if (!sym) return 1;

    printf("dataset,H,Lbar,naive_nodes,naive_ops,flat_nodes,flat_ops,full_nodes,full_ops\n");

    int n = bench_num_distributions();
    for (int d = 0; d < n; d++) {
        if (!bench_dist_is_main(d)) continue;
        const char *name = bench_dist_name(d);
        bench_generate_symbols(d, sym, N_SYMBOLS, SEED);

        uint64_t freq[PIVCO_MAX_SYMBOLS] = {0};
        for (int i = 0; i < N_SYMBOLS; i++) freq[sym[i]]++;

        double H = shannon_H(freq);
        double Lbar = 0.0;

        mode_stats_t a = build_and_measure(PIVCO_TREE_MODE_NAIVE,          freq, &Lbar);
        mode_stats_t b = build_and_measure(PIVCO_TREE_MODE_CANONICAL_FLAT, freq, NULL);
        mode_stats_t c = build_and_measure(PIVCO_TREE_MODE_OPTIMIZED,      freq, NULL);

        printf("%s,%.3f,%.3f,%d,%.3f,%d,%.3f,%d,%.3f\n",
               name, H, Lbar,
               a.node_count, a.ops_per_byte,
               b.node_count, b.ops_per_byte,
               c.node_count, c.ops_per_byte);
        fflush(stdout);
    }
    free(sym);
    return 0;
}
