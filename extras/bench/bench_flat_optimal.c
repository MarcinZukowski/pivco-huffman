/* Flat-subtree coverage gap analyzer.
 *
 * For each benchmark distribution:
 *   1. Build the canonical Huffman tree (same as production build_table).
 *   2. Compute "canonical" flat-subtree coverage — leaves in maximal
 *      flat subtrees of D >= 2, and the freq-weighted equivalent.
 *   3. Compute "optimal" flat-subtree coverage — the best tree shape
 *      achievable for the same code-length multiset, plus the best
 *      assignment of symbols to leaf positions within each length
 *      class (top-freq-first into flat slots).
 *   4. Print the gap.
 *
 * The leaf-count optimal coverage is found by a per-length greedy:
 *
 *   for L from max_d down to min_d:
 *     while c_L >= 4:
 *       D = largest s.t. 2^D <= c_L  (so 2^D >= 4, i.e. D >= 2)
 *       record flat subtree of depth D rooted at depth L-D
 *       c_L -= 2^D
 *
 * The greedy is optimal: per length L, the only way to contribute to
 * flat-D>=2 coverage is to group 2^D same-length leaves; max packing
 * = c_L - (c_L mod 4) = c_L AND ~3.  The union of all per-length
 * greedy flats is realisable because each flat at slot-depth L-D
 * contributes Kraft 2^-(L-D), same as one internal node at depth L-D;
 * the leftover leaves (c_L mod 4 per length, 0..3 each) fill the rest.
 *
 *   (Old recursive split DP — kept commented for reference)
 *   cov(M):          # M = depth multiset, Kraft(M)=1, "this subtree"
 *     if M = {(0,1)}:                return 0          # leaf
 *     if M = {(D, 2^D)} for D >= 2:  return 2^D        # flat — take it
 *     best = 0
 *     for each split (M_L, M_R) with Kraft(M_L)=Kraft(M_R)=1/2:
 *       best = max(best, cov(shift(M_L)) + cov(shift(M_R)))
 *     return best
 *
 * Memoize on the sorted multiset.  In addition to total flat leaves,
 * we track per-depth flat counts so the freq-weighted upper bound can
 * be computed: for each depth L, top n_L_flat freqs of length-L symbols
 * go to flat slots, rest go to non-flat.
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_DEPTH 16   /* PIVCO_MAX_CODE_LEN is 15; 16 covers depth 0..15 */

extern void         bench_init(void);
extern int          bench_num_distributions(void);
extern const char  *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

/* ---------- Multiset & memoization ---------- */

typedef struct {
    uint16_t c[MAX_DEPTH];
} multiset_t;

typedef struct {
    int       total_flat;             /* total leaves in flat subtrees */
    uint16_t  per_depth[MAX_DEPTH];   /* flat leaves at each (relative) depth */
} cov_t;

/* ---------- Optimal-shape (per-length greedy) ----------
   Greedy: per length L, repeatedly carve out the largest 2^D chunk.
   Each chunk is a flat-D subtree contributing 2^D leaves at depth L.
   With min_D=2: only flat-subtree fast path eligible (>=4 leaves).
   With min_D=1: also includes sibling-pair stage-fusion eligible.
*/
static cov_t opt_cov(const multiset_t *m, int min_D)
{
    cov_t out;
    memset(&out, 0, sizeof(out));
    int min_chunk = 1 << min_D;
    for (int L = 0; L < MAX_DEPTH; L++) {
        int c = m->c[L];
        while (c >= min_chunk) {
            int D = min_D;
            while ((1 << (D + 1)) <= c) D++;
            int n = 1 << D;
            out.total_flat += n;
            out.per_depth[L] = (uint16_t)(out.per_depth[L] + n);
            c -= n;
        }
    }
    return out;
}

/* ---------- Canonical-tree coverage (current production) ---------- */

static int local_min(const pivco_table_t *t, int16_t node_id)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return 0;
    int l = local_min(t, n->left);
    int r = local_min(t, n->right);
    return 1 + (l < r ? l : r);
}

static int local_max(const pivco_table_t *t, int16_t node_id)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return 0;
    int l = local_max(t, n->left);
    int r = local_max(t, n->right);
    return 1 + (l > r ? l : r);
}

static uint64_t subtree_freq_sum(const pivco_table_t *t,
                                  int16_t node_id,
                                  const uint64_t *freq)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return freq[n->symbol];
    return subtree_freq_sum(t, n->left, freq) +
           subtree_freq_sum(t, n->right, freq);
}

/* Walk: classify each internal node as one of:
     - maximal flat-D>=2 subtree (counted in flat_d2)        -> 0 partition
     - D=1 sibling pair, both children leaves (counted in d1)-> 0 partition
     - non-fast                                              -> 1 partition
   Also accumulates freq-weighted partition cost (sum of subtree freqs
   over partitioning internals = total weight passing through partition
   ops = proxy for runtime cost). */
static void canonical_walk(const pivco_table_t *t,
                            int16_t node_id, int cur_depth,
                            const uint64_t *freq,
                            int *flat_d2_leaves,    uint64_t *flat_d2_freq,
                            uint16_t *flat_d2_per_depth,
                            int *d1_pair_leaves,    uint64_t *d1_pair_freq,
                            uint16_t *d1_pair_per_depth,
                            int *partition_count,   uint64_t *partition_freq)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return;

    int lmin = local_min(t, node_id);
    int lmax = local_max(t, node_id);
    if (lmin == lmax && lmin >= 2) {
        int D = lmin;
        int leaf_depth = cur_depth + D;
        int n_leaves = 1 << D;
        *flat_d2_leaves += n_leaves;
        if (leaf_depth < MAX_DEPTH) flat_d2_per_depth[leaf_depth] += n_leaves;
        *flat_d2_freq += subtree_freq_sum(t, node_id, freq);
        return;
    }

    const pivco_tree_node_t *lc = &t->tree[n->left];
    const pivco_tree_node_t *rc = &t->tree[n->right];
    if (lc->symbol >= 0 && rc->symbol >= 0) {
        int leaf_depth = cur_depth + 1;
        *d1_pair_leaves += 2;
        if (leaf_depth < MAX_DEPTH) d1_pair_per_depth[leaf_depth] += 2;
        *d1_pair_freq += freq[lc->symbol] + freq[rc->symbol];
        return;
    }

    /* Partitioning internal node. */
    *partition_count += 1;
    *partition_freq  += subtree_freq_sum(t, node_id, freq);

    canonical_walk(t, n->left,  cur_depth + 1, freq,
                   flat_d2_leaves, flat_d2_freq, flat_d2_per_depth,
                   d1_pair_leaves, d1_pair_freq, d1_pair_per_depth,
                   partition_count, partition_freq);
    canonical_walk(t, n->right, cur_depth + 1, freq,
                   flat_d2_leaves, flat_d2_freq, flat_d2_per_depth,
                   d1_pair_leaves, d1_pair_freq, d1_pair_per_depth,
                   partition_count, partition_freq);
}

/* ---------- Driver ---------- */

static int cmp_u64_desc(const void *a, const void *b)
{
    uint64_t ua = *(const uint64_t *)a;
    uint64_t ub = *(const uint64_t *)b;
    if (ua > ub) return -1;
    if (ua < ub) return  1;
    return 0;
}

/* Given per-depth flat slot counts and the symbol freqs grouped by code-
   length, return the maximum freq sum we can place into flat slots
   (top-K freqs of length L go to the K flat slots at depth L). */
static uint64_t freq_optimal_for_shape(const uint16_t *per_depth_flat,
                                        const pivco_table_t *t,
                                        const uint64_t *freq)
{
    uint64_t total = 0;
    for (int L = 1; L < MAX_DEPTH; L++) {
        int n_flat = per_depth_flat[L];
        if (n_flat == 0) continue;
        uint64_t buf[PIVCO_MAX_SYMBOLS];
        int nb = 0;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            if (t->code_len[s] == L) buf[nb++] = freq[s];
        }
        if (n_flat > nb) n_flat = nb;
        qsort(buf, nb, sizeof(buf[0]), cmp_u64_desc);
        for (int i = 0; i < n_flat; i++) total += buf[i];
    }
    return total;
}

/* Optimal-tree partition counts.

   For a leaf at depth L_full assigned to a chunk with depth D:
     #partitioning internals on its path = L_full - D (D=0 = singleton).
   We minimise total cost by assigning highest-freq leaves to largest-D
   chunks within each length class.  Returns:
     *count  = total partitioning internal-node count (= residual_internals
               minus D=1 sibling pairs).
     *cost   = freq-weighted partition cost (sum over partitioning internals
               of subtree-freq-sum); normalised by /total_freq it is the
               mean number of partitions traversed per decoded element. */
static void opt_partition(const multiset_t *m,
                           const pivco_table_t *t,
                           const uint64_t *freq,
                           int *count, uint64_t *cost)
{
    /* Counts. */
    int n_flats = 0;       /* # of D>=2 flat-subtree roots */
    int n_leftover = 0;    /* # of leftover leaves (c_L mod 4 each L) */
    int n_d1_pairs = 0;    /* # of D=1 sibling pairs among leftovers */

    /* Cost. */
    uint64_t total_cost = 0;

    for (int L = 1; L < MAX_DEPTH; L++) {
        int c = m->c[L];
        if (c == 0) continue;

        n_leftover += (c & 3);
        n_d1_pairs += ((c & 3) >> 1);     /* (c mod 4)/2 */

        /* Collect length-L freqs sorted desc. */
        uint64_t buf[PIVCO_MAX_SYMBOLS];
        int nb = 0;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            if (t->code_len[s] == L) buf[nb++] = freq[s];
        }
        qsort(buf, nb, sizeof(buf[0]), cmp_u64_desc);

        /* Assign in chunk order: largest D first (highest-freq -> deepest
           savings).  Decompose c by bits, descending. */
        int idx = 0;
        for (int bit = 15; bit >= 2; bit--) {
            if (c & (1 << bit)) {
                int n = 1 << bit;          /* chunk size 2^bit */
                int f = L - bit;           /* path partitions = L - D */
                if (f < 0) f = 0;
                for (int i = 0; i < n && idx < nb; i++) {
                    total_cost += buf[idx++] * (uint64_t)f;
                }
                n_flats++;
            }
        }
        if (c & 2) {
            int f = L - 1;                 /* D=1 pair: path partitions = L-1 */
            for (int i = 0; i < 2 && idx < nb; i++) {
                total_cost += buf[idx++] * (uint64_t)f;
            }
        }
        if (c & 1) {
            int f = L;                     /* singleton: full path partitions */
            if (idx < nb) total_cost += buf[idx++] * (uint64_t)f;
        }
    }

    int residual_internals = n_flats + n_leftover - 1;
    if (residual_internals < 0) residual_internals = 0;
    *count = residual_internals - n_d1_pairs;
    *cost  = total_cost;
}

static void analyze(int d)
{
    const char *name = bench_dist_name(d);
    const uint64_t *freq = bench_dist_freq(d);

    pivco_table_t *t =
        (pivco_table_t *)malloc(sizeof(*t));
    if (pivco_build_table(bench_cfg(), freq, t) != PIVCO_OK) {
        printf("%-14s | build_table failed\n", name);
        free(t);
        return;
    }

    uint64_t total_freq = 0;
    int total_leaves = 0;
    multiset_t m;
    memset(&m, 0, sizeof(m));
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        total_freq += freq[s];
        int len = t->code_len[s];
        if (len > 0) {
            m.c[len]++;
            total_leaves++;
        }
    }

    if (t->min_len == t->max_len) {
        printf("%-14s | %3d leaves | ROOT FLAT D=%d\n",
               name, total_leaves, t->max_len);
        free(t);
        return;
    }

    /* Canonical coverage + partition counts. */
    int      cn_d2_leaves = 0,  cn_d1_leaves = 0;
    uint64_t cn_d2_freq   = 0,  cn_d1_freq   = 0;
    uint16_t cn_d2_per_depth[MAX_DEPTH] = {0};
    uint16_t cn_d1_per_depth[MAX_DEPTH] = {0};
    int      cn_part_count = 0;
    uint64_t cn_part_cost  = 0;
    canonical_walk(t, t->tree_root, 0, freq,
                   &cn_d2_leaves, &cn_d2_freq, cn_d2_per_depth,
                   &cn_d1_leaves, &cn_d1_freq, cn_d1_per_depth,
                   &cn_part_count, &cn_part_cost);

    /* Optimal-shape coverage. */
    cov_t opt_d2 = opt_cov(&m, 2);
    cov_t opt_d1 = opt_cov(&m, 1);
    uint64_t opt_d2_freq = freq_optimal_for_shape(opt_d2.per_depth, t, freq);
    uint64_t opt_d1_freq = freq_optimal_for_shape(opt_d1.per_depth, t, freq);

    /* Optimal partition counts. */
    int      op_part_count = 0;
    uint64_t op_part_cost  = 0;
    opt_partition(&m, t, freq, &op_part_count, &op_part_cost);

    int      cn_total_leaves = cn_d2_leaves + cn_d1_leaves;
    uint64_t cn_total_freq   = cn_d2_freq   + cn_d1_freq;

    double cn_d2_lpct    = 100.0 * (double)cn_d2_leaves     / (double)total_leaves;
    double cn_total_lpct = 100.0 * (double)cn_total_leaves  / (double)total_leaves;
    double cn_d2_fpct    = total_freq ? 100.0 * (double)cn_d2_freq    / (double)total_freq : 0.0;
    double cn_total_fpct = total_freq ? 100.0 * (double)cn_total_freq / (double)total_freq : 0.0;

    double op_d2_lpct = 100.0 * (double)opt_d2.total_flat / (double)total_leaves;
    double op_d1_lpct = 100.0 * (double)opt_d1.total_flat / (double)total_leaves;
    double op_d2_fpct = total_freq ? 100.0 * (double)opt_d2_freq / (double)total_freq : 0.0;
    double op_d1_fpct = total_freq ? 100.0 * (double)opt_d1_freq / (double)total_freq : 0.0;

    /* Mean partitions per element = freq-weighted partition count / total_freq. */
    double cn_part_per_elem = total_freq ? (double)cn_part_cost / (double)total_freq : 0.0;
    double op_part_per_elem = total_freq ? (double)op_part_cost / (double)total_freq : 0.0;
    double part_savings_pct = cn_part_per_elem ?
        100.0 * (cn_part_per_elem - op_part_per_elem) / cn_part_per_elem : 0.0;

    printf("%-14s leaves=%3d max_len=%2d\n", name, total_leaves, t->max_len);
    printf("    canon D>=2 : %3d (%5.1f%%) leaf, %5.1f%% freq\n",
           cn_d2_leaves, cn_d2_lpct, cn_d2_fpct);
    printf("    canon D>=1 : %3d (%5.1f%%) leaf, %5.1f%% freq  (= D>=2 + %d D=1 sib pairs)\n",
           cn_total_leaves, cn_total_lpct, cn_total_fpct, cn_d1_leaves / 2);
    printf("    opt   D>=2 : %3d (%5.1f%%) leaf, %5.1f%% freq  | gap +%2d leaves, +%5.1f%% freq\n",
           opt_d2.total_flat, op_d2_lpct, op_d2_fpct,
           opt_d2.total_flat - cn_d2_leaves,
           op_d2_fpct - cn_d2_fpct);
    printf("    opt   D>=1 : %3d (%5.1f%%) leaf, %5.1f%% freq  | gap +%2d leaves, +%5.1f%% freq\n",
           opt_d1.total_flat, op_d1_lpct, op_d1_fpct,
           opt_d1.total_flat - cn_total_leaves,
           op_d1_fpct - cn_total_fpct);
    printf("    partitions : canon %3d nodes, %5.2f/elem  | opt %3d nodes, %5.2f/elem"
           "  -> %5.1f%% fewer partition ops\n",
           cn_part_count, cn_part_per_elem,
           op_part_count, op_part_per_elem,
           part_savings_pct);

    free(t);
}

int main(void)
{
    bench_init();
    int n = bench_num_distributions();
    printf("Flat-subtree coverage gap: canonical Huffman vs optimal tree\n");
    printf("=============================================================\n");
    printf("Same code-length multiset (= identical compression).\n");
    printf("\"leaf\" = %% of distinct symbols in flat subtrees.\n");
    printf("\"freq\" = %% of decoded occurrences in flat subtrees.\n\n");
    for (int d = 0; d < n; d++) analyze(d);
    return 0;
}
