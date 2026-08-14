/* Offline TANS-routing ratio bound.
 *
 * For a given file, compute the bit cost under four entropy models:
 *
 *   raw       : 8 * N
 *   huffman   : current pivco-huffman per-block bits (Σ freq[s] · code_len[s])
 *   tans_flat : per-internal-node H₂(p_node)-coded partition bits ONLY for
 *               non-flat nodes; flat-subtree roots keep their N·D packed
 *               bits (matches the "ship as extension, keep flat fast path
 *               intact" variant)
 *   shannon   : full TANS over the symbol stream (= Σ over ALL internal
 *               nodes of H₂(p_node) · subtree_freq[node]; provably equals
 *               -Σ freq[s] · log₂(freq[s]/N), the order-0 entropy bound)
 *
 * Analytical only — no encoder runs.  Per-block (PIVCO_BLOCK_SIZE)
 * histograms + Huffman tables, then walk each block's tree.
 *
 * Overheads excluded (table description bytes ~128 for Huffman, plus
 * ~K per-node TANS metadata for the TANS variant).  We're asking
 * "what's the entropy-bound upside?" not "is the full encoder smaller?"
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* MAIN distribution registry (linked from bench/bench_distributions.c). */
extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int             bench_dist_is_main(int idx);
extern void            bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                              int n_symbols, uint64_t seed);

#define MAX_DEPTH_BUCKETS 16

static double h2(double p)
{
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p);
}

static uint64_t fill_subtree_freq(const pivco_table_t *t,
                                  const uint64_t freq[256],
                                  int16_t node,
                                  uint64_t out[])
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) { out[node] = freq[n->symbol]; return out[node]; }
    uint64_t l = fill_subtree_freq(t, freq, n->left,  out);
    uint64_t r = fill_subtree_freq(t, freq, n->right, out);
    out[node] = l + r;
    return out[node];
}

/* Full TANS: every internal node contributes H₂(p) · subtree_freq. */
static void walk_full_tans(const pivco_table_t *t,
                           const uint64_t subtree_freq[],
                           int16_t node, double *bits)
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) return;
    uint64_t sf = subtree_freq[node];
    if (sf == 0) return;
    double p = (double)subtree_freq[n->left] / (double)sf;
    *bits += (double)sf * h2(p);
    walk_full_tans(t, subtree_freq, n->left,  bits);
    walk_full_tans(t, subtree_freq, n->right, bits);
}

/* TANS with flat carve-out: flat-subtree roots stay at N·D; other
   internal nodes contribute H₂(p) · subtree_freq. */
static void walk_flat_carve(const pivco_table_t *t,
                            const uint64_t subtree_freq[],
                            int16_t node, double *bits, double *flat_bits)
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) return;
    uint64_t sf = subtree_freq[node];
    if (sf == 0) return;
    int D = t->flat_depth[node];
    if (D >= 2) {
        *bits      += (double)sf * (double)D;
        *flat_bits += (double)sf * (double)D;
        return;
    }
    double p = (double)subtree_freq[n->left] / (double)sf;
    *bits += (double)sf * h2(p);
    walk_flat_carve(t, subtree_freq, n->left,  bits, flat_bits);
    walk_flat_carve(t, subtree_freq, n->right, bits, flat_bits);
}

/* Analyze a single histogram (treats freq[] as one logical block).  All
   four rates are in bits-per-source-byte (i.e. bits divided by total
   symbol count).  Reported tans_flat_bps / shannon_bps depend only on
   the relative shape of freq[]. */
static void analyze_freq(const uint64_t freq[256],
                         pivco_table_t *tbl,
                         uint64_t subtree_freq[],
                         double *huf_bps, double *tans_flat_bps,
                         double *shannon_bps, double *flat_share)
{
    uint64_t N = 0;
    int n_syms = 0;
    for (int s = 0; s < 256; s++) {
        N += freq[s];
        if (freq[s]) n_syms++;
    }
    if (n_syms < 2 || N == 0) {
        *huf_bps = *tans_flat_bps = *shannon_bps = 0.0;
        *flat_share = 0.0;
        return;
    }

    if (pivco_build_table(bench_cfg(), freq, tbl) != PIVCO_OK) {
        *huf_bps = *tans_flat_bps = *shannon_bps = 0.0/0.0;
        *flat_share = 0.0;
        return;
    }

    double huf_bits = 0.0;
    for (int s = 0; s < 256; s++)
        if (freq[s]) huf_bits += (double)freq[s] * tbl->code_len[s];

    memset(subtree_freq, 0, PIVCO_MAX_TREE_NODES * sizeof(uint64_t));
    fill_subtree_freq(tbl, freq, tbl->tree_root, subtree_freq);

    double full_tans = 0.0, flat_carve = 0.0, flat_only = 0.0;
    walk_full_tans (tbl, subtree_freq, tbl->tree_root, &full_tans);
    walk_flat_carve(tbl, subtree_freq, tbl->tree_root, &flat_carve, &flat_only);

    *huf_bps       = huf_bits   / (double)N;
    *tans_flat_bps = flat_carve / (double)N;
    *shannon_bps   = full_tans  / (double)N;
    *flat_share    = flat_carve > 0.0 ? flat_only / flat_carve : 0.0;
}

static int run_dist_mode(int main_only)
{
    bench_init();
    int n = bench_num_distributions();
    pivco_table_t *tbl = malloc(sizeof(*tbl));
    uint64_t *subtree_freq = calloc(PIVCO_MAX_TREE_NODES, sizeof(uint64_t));
    if (!tbl || !subtree_freq) { fprintf(stderr, "OOM\n"); return 1; }

    printf("%-18s %8s %10s %8s   %9s %9s   %5s\n",
           "distribution",
           "huff_bps", "huff_flat", "shannon",
           "hf-vs-h", "sh-vs-h", "flat%");
    printf("%-18s %8s %10s %8s   %9s %9s   %5s\n",
           "------------",
           "--------", "---------", "-------",
           "-------", "-------", "-----");

    for (int i = 0; i < n; i++) {
        if (main_only && !bench_dist_is_main(i)) continue;
        const char *name = bench_dist_name(i);
        const uint64_t *freq = bench_dist_freq(i);
        double huf, tans_flat, shan, flat_share;
        analyze_freq(freq, tbl, subtree_freq, &huf, &tans_flat, &shan, &flat_share);
        if (huf <= 0.0) {
            printf("%-18s  (singleton or build failed)\n", name);
            continue;
        }
        printf("%-18s %8.4f %10.4f %8.4f   %8.2f%% %8.2f%%   %4.0f%%\n",
               name, huf, tans_flat, shan,
               100.0 * (1.0 - tans_flat / huf),
               100.0 * (1.0 - shan      / huf),
               100.0 * flat_share);
    }
    free(tbl); free(subtree_freq);
    return 0;
}

/* ---------- byte-level vs bit-level verification ---------- */
/*
 * Simulates the actual tree-walk partition (same algorithm as
 * encode_node_neon but scalar) and, at each non-flat internal node,
 * accumulates:
 *
 *   bit_entropy   = n_bits · H₂(p)
 *   byte_entropy  = n_bytes · H_emp(byte_hist_of_this_bitmap)
 *   n_bytes       = ceil(n_bits / 8)
 *
 * Buckets by tree depth so we can see where the bit/byte agreement
 * breaks down (deep nodes have tiny bitmaps → empirical byte
 * histogram is sample-noise dominated).
 */
typedef struct {
    int64_t n_nodes;
    int64_t n_bits;
    int64_t n_bytes;
    double  bit_entropy;
    double  byte_entropy;
} depth_bucket_t;

static double byte_hist_entropy(const int *hist, int total_bytes)
{
    if (total_bytes <= 0) return 0.0;
    double H = 0.0;
    double inv = 1.0 / (double)total_bytes;
    for (int b = 0; b < 256; b++) {
        if (hist[b] > 0) {
            double pb = (double)hist[b] * inv;
            H += -pb * log2(pb);
        }
    }
    return H;
}

static void simulate_partition(const pivco_table_t *t,
                               uint16_t *codes_la, int n,
                               int16_t node_id, int depth,
                               double *flat_bits,
                               depth_bucket_t depth_stats[MAX_DEPTH_BUCKETS],
                               uint16_t *tmp)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &t->tree[node_id];
    if (node->symbol >= 0) return; /* leaf */

    int D = t->flat_depth[node_id];
    if (D >= 2) {
        *flat_bits += (double)n * (double)D;
        return;
    }

    int shift = 15 - depth;
    int byte_hist[256] = {0};
    int n_left = 0, n_right = 0;
    int j = 0;

    /* 8-wide groups: in-place LEFT, RIGHT to tmp.  Match encode_node_neon. */
    for (; j + 8 <= n; j += 8) {
        uint8_t mask = 0;
        uint16_t saved[8];
        for (int k = 0; k < 8; k++) {
            saved[k] = codes_la[j + k];
            int bit = (saved[k] >> shift) & 1;
            mask |= (uint8_t)(bit << k);
        }
        byte_hist[mask]++;
        for (int k = 0; k < 8; k++) {
            if (mask & (1 << k)) tmp[n_right++]      = saved[k];
            else                 codes_la[n_left++]  = saved[k];
        }
    }
    /* Tail. */
    int tail = n - j;
    if (tail > 0) {
        uint8_t mask = 0;
        uint16_t saved[8];
        for (int k = 0; k < tail; k++) saved[k] = codes_la[j + k];
        for (int k = 0; k < tail; k++) {
            int bit = (saved[k] >> shift) & 1;
            mask |= (uint8_t)(bit << k);
        }
        byte_hist[mask]++;
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k)) tmp[n_right++]      = saved[k];
            else                 codes_la[n_left++]  = saved[k];
        }
    }

    int total_bytes = (n + 7) >> 3;
    double p = (double)n_left / (double)n;
    double H_bit  = (double)n * h2(p);
    double H_byte = (double)total_bytes * byte_hist_entropy(byte_hist, total_bytes);

    int dbucket = depth < MAX_DEPTH_BUCKETS ? depth : MAX_DEPTH_BUCKETS - 1;
    depth_stats[dbucket].n_nodes++;
    depth_stats[dbucket].n_bits      += n;
    depth_stats[dbucket].n_bytes     += total_bytes;
    depth_stats[dbucket].bit_entropy += H_bit;
    depth_stats[dbucket].byte_entropy+= H_byte;

    simulate_partition(t, codes_la, n_left,  node->left,  depth + 1,
                       flat_bits, depth_stats, tmp + n_right);
    simulate_partition(t, tmp,      n_right, node->right, depth + 1,
                       flat_bits, depth_stats, tmp + n_right);
}

static void print_depth_table(const depth_bucket_t depth_stats[], double flat_bits,
                              const char *header)
{
    double total_bit_H = 0.0, total_byte_H = 0.0;
    int64_t total_bits_nf = 0, total_bytes_nf = 0;
    for (int d = 0; d < MAX_DEPTH_BUCKETS; d++) {
        total_bit_H    += depth_stats[d].bit_entropy;
        total_byte_H   += depth_stats[d].byte_entropy;
        total_bits_nf  += depth_stats[d].n_bits;
        total_bytes_nf += depth_stats[d].n_bytes;
    }
    printf("\n=== %s ===\n", header);
    printf("  %-5s %10s %12s %10s %12s %12s %10s\n",
           "depth", "n_nodes", "n_bits", "n_bytes", "bit_H2", "byte_H8", "delta_pct");
    printf("  %-5s %10s %12s %10s %12s %12s %10s\n",
           "-----", "-------", "------", "-------", "------", "-------", "---------");
    for (int d = 0; d < MAX_DEPTH_BUCKETS; d++) {
        if (depth_stats[d].n_nodes == 0) continue;
        double bit_H  = depth_stats[d].bit_entropy;
        double byte_H = depth_stats[d].byte_entropy;
        double delta_pct = bit_H > 0 ? 100.0 * (byte_H - bit_H) / bit_H : 0.0;
        printf("  %-5d %10lld %12lld %10lld %12.0f %12.0f   %+7.2f%%\n",
               d,
               (long long)depth_stats[d].n_nodes,
               (long long)depth_stats[d].n_bits,
               (long long)depth_stats[d].n_bytes,
               bit_H, byte_H, delta_pct);
    }
    double agg_delta = total_bit_H > 0 ? 100.0 * (total_byte_H - total_bit_H) / total_bit_H : 0.0;
    printf("  %-5s %10s %12lld %10lld %12.0f %12.0f   %+7.2f%%\n",
           "TOTAL", "",
           (long long)total_bits_nf, (long long)total_bytes_nf,
           total_bit_H, total_byte_H, agg_delta);
    printf("  flat: %.0f bits (already-packed N·D)\n", flat_bits);
}

/* ---------- exact (no-sampling) per-depth + D≤K cumulative ----------
 *
 * For each distribution we know the exact freq[256], so subtree_freq[node]
 * is exact (after scaling to one 8K block), p_node is exact, n_bits at
 * each node is exact.
 *
 * Bit-level entropy at non-flat node:  n_bits · H₂(p)
 * Byte-level entropy at non-flat node (IID model): same — see header.
 *
 * Reported per depth, plus the cumulative bit-saving if we TANS-code only
 * partition bitmaps at depths ≤ K (for K = 0, 1, 2, 3, 4, ∞).
 */
typedef struct {
    int64_t n_nodes;
    double  n_bits;       /* expected per-8K-block */
    double  bit_H;        /* = byte_H under IID */
} exact_depth_t;

static void walk_exact(const pivco_table_t *t,
                       const uint64_t subtree_freq[],
                       int16_t node_id, int depth,
                       double scale,
                       exact_depth_t depth_stats[MAX_DEPTH_BUCKETS],
                       double *flat_bits)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return;
    uint64_t sf = subtree_freq[node_id];
    if (sf == 0) return;

    int D = t->flat_depth[node_id];
    if (D >= 2) {
        *flat_bits += (double)sf * scale * (double)D;
        return;
    }

    double p = (double)subtree_freq[n->left] / (double)sf;
    double n_bits = (double)sf * scale;
    double H = n_bits * h2(p);

    int d = depth < MAX_DEPTH_BUCKETS ? depth : MAX_DEPTH_BUCKETS - 1;
    depth_stats[d].n_nodes++;
    depth_stats[d].n_bits += n_bits;
    depth_stats[d].bit_H  += H;

    walk_exact(t, subtree_freq, n->left,  depth + 1, scale, depth_stats, flat_bits);
    walk_exact(t, subtree_freq, n->right, depth + 1, scale, depth_stats, flat_bits);
}

static int run_exact_tier_mode(int main_only)
{
    bench_init();
    int n_dists = bench_num_distributions();

    pivco_table_t *tbl = malloc(sizeof(*tbl));
    uint64_t *subtree_freq = calloc(PIVCO_MAX_TREE_NODES, sizeof(uint64_t));
    if (!tbl || !subtree_freq) { fprintf(stderr, "OOM\n"); return 1; }

    /* Cumulative-savings summary across distributions. */
    struct {
        const char *name;
        double huff_bits;       /* total huff bits / 8K block */
        double bit_H_total;     /* sum bit_H over all non-flat */
        double save_at[5];      /* savings if D≤K for K=0..4 */
        double save_inf;        /* savings if D≤∞ (== bit_H total achievable) */
    } sm[64];
    int n_sm = 0;

    for (int i = 0; i < n_dists; i++) {
        if (main_only && !bench_dist_is_main(i)) continue;
        const char *name = bench_dist_name(i);
        const uint64_t *freq = bench_dist_freq(i);

        int n_syms = 0;
        uint64_t total_freq = 0;
        for (int s = 0; s < 256; s++) { total_freq += freq[s]; if (freq[s]) n_syms++; }
        if (n_syms < 2 || total_freq == 0) continue;

        if (pivco_build_table(bench_cfg(), freq, tbl) != PIVCO_OK) continue;

        /* Scale so one logical "block" is 8192 symbols. */
        double scale = (double)PIVCO_BLOCK_SIZE / (double)total_freq;

        memset(subtree_freq, 0, PIVCO_MAX_TREE_NODES * sizeof(uint64_t));
        fill_subtree_freq(tbl, freq, tbl->tree_root, subtree_freq);

        exact_depth_t depth_stats[MAX_DEPTH_BUCKETS] = {{0}};
        double flat_bits = 0.0;
        walk_exact(tbl, subtree_freq, tbl->tree_root, 0, scale, depth_stats, &flat_bits);

        /* Total Huffman bits in one 8K block. */
        double huff_bits = 0.0;
        for (int s = 0; s < 256; s++)
            if (freq[s]) huff_bits += (double)freq[s] * scale * tbl->code_len[s];

        /* Per-depth print. */
        printf("\n=== %s (huff = %.0f bits/block) ===\n", name, huff_bits);
        printf("  %-5s %8s %12s %12s %12s %10s\n",
               "depth", "n_nodes", "n_bits", "bit_H", "save", "save_%");
        printf("  %-5s %8s %12s %12s %12s %10s\n",
               "-----", "-------", "------", "-----", "----", "------");

        double tot_bits = 0.0, tot_H = 0.0;
        for (int d = 0; d < MAX_DEPTH_BUCKETS; d++) {
            if (depth_stats[d].n_nodes == 0) continue;
            double bits = depth_stats[d].n_bits;
            double H    = depth_stats[d].bit_H;
            double save = bits - H;
            tot_bits += bits;
            tot_H    += H;
            printf("  %-5d %8lld %12.2f %12.2f %12.2f   %6.2f%%\n",
                   d, (long long)depth_stats[d].n_nodes,
                   bits, H, save,
                   bits > 0 ? 100.0 * save / bits : 0.0);
        }
        printf("  %-5s %8s %12.2f %12.2f %12.2f   %6.2f%%\n",
               "TOTAL", "", tot_bits, tot_H, tot_bits - tot_H,
               tot_bits > 0 ? 100.0 * (tot_bits - tot_H) / tot_bits : 0.0);
        printf("  flat-subtree: %.2f bits/block (untouched)\n", flat_bits);

        /* D≤K cumulative savings. */
        double cum_save = 0.0;
        printf("  cumulative savings vs current pivco-huff (tier-K TANS):\n");
        for (int K = 0; K <= 4; K++) {
            if (K < MAX_DEPTH_BUCKETS) {
                double bits_K = depth_stats[K].n_bits;
                double H_K    = depth_stats[K].bit_H;
                cum_save += (bits_K - H_K);
            }
            double pct = huff_bits > 0 ? 100.0 * cum_save / huff_bits : 0.0;
            printf("    D≤%d: %8.2f bits  (%.3f%% of huff)\n", K, cum_save, pct);
            if (K < 4 && K + 1 < MAX_DEPTH_BUCKETS) {
                /* keep accumulating */
            }
        }
        /* D≤∞ — sum all non-flat */
        double inf_pct = huff_bits > 0 ? 100.0 * (tot_bits - tot_H) / huff_bits : 0.0;
        printf("    D≤∞: %8.2f bits  (%.3f%% of huff)\n",
               tot_bits - tot_H, inf_pct);

        /* Save for summary. */
        if (n_sm < (int)(sizeof(sm)/sizeof(sm[0]))) {
            sm[n_sm].name = name;
            sm[n_sm].huff_bits = huff_bits;
            sm[n_sm].bit_H_total = tot_H;
            double s = 0;
            for (int K = 0; K <= 4; K++) {
                if (K < MAX_DEPTH_BUCKETS) {
                    s += depth_stats[K].n_bits - depth_stats[K].bit_H;
                }
                sm[n_sm].save_at[K] = s;
            }
            sm[n_sm].save_inf = tot_bits - tot_H;
            n_sm++;
        }
    }

    /* Summary across distributions. */
    printf("\n=== SUMMARY: tier-K TANS savings as %% of pivco-huff bits ===\n");
    printf("  %-18s %8s %8s %8s %8s %8s %8s\n",
           "distribution", "D≤0", "D≤1", "D≤2", "D≤3", "D≤4", "D≤∞");
    printf("  %-18s %8s %8s %8s %8s %8s %8s\n",
           "------------", "----", "----", "----", "----", "----", "----");
    for (int k = 0; k < n_sm; k++) {
        double h = sm[k].huff_bits;
        if (h <= 0) continue;
        printf("  %-18s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n",
               sm[k].name,
               100.0 * sm[k].save_at[0] / h,
               100.0 * sm[k].save_at[1] / h,
               100.0 * sm[k].save_at[2] / h,
               100.0 * sm[k].save_at[3] / h,
               100.0 * sm[k].save_at[4] / h,
               100.0 * sm[k].save_inf  / h);
    }
    free(tbl); free(subtree_freq);
    return 0;
}

static int run_verify_dist_mode(int main_only)
{
    bench_init();
    int n_dists = bench_num_distributions();

    pivco_table_t *tbl = malloc(sizeof(*tbl));
    uint16_t *codes_la = malloc(PIVCO_BLOCK_SIZE * sizeof(uint16_t));
    uint16_t *tmp_buf  = malloc(PIVCO_BLOCK_SIZE * sizeof(uint16_t));
    uint8_t  *symbols  = malloc(PIVCO_BLOCK_SIZE);
    if (!tbl || !codes_la || !tmp_buf || !symbols) {
        fprintf(stderr, "OOM\n"); return 1;
    }

    const int N_BLOCKS = 100;

    /* Compact summary collected as we go. */
    typedef struct {
        const char *name;
        double bit_H_total;
        double byte_H_total;
        double flat_bits;
        int64_t n_bits_total;
        int64_t n_bytes_total;
    } summary_t;
    summary_t summary[64];
    int n_summary = 0;

    for (int i = 0; i < n_dists; i++) {
        if (main_only && !bench_dist_is_main(i)) continue;
        const char *name = bench_dist_name(i);
        const uint64_t *freq_true = bench_dist_freq(i);

        int n_syms = 0;
        for (int s = 0; s < 256; s++) if (freq_true[s]) n_syms++;
        if (n_syms < 2) continue;

        if (pivco_build_table(bench_cfg(), freq_true, tbl) != PIVCO_OK) continue;

        depth_bucket_t depth_stats[MAX_DEPTH_BUCKETS] = {{0}};
        double flat_bits = 0.0;

        for (int b = 0; b < N_BLOCKS; b++) {
            uint64_t seed = (uint64_t)(i + 1) * 1000003ULL + b;
            bench_generate_symbols(i, symbols, PIVCO_BLOCK_SIZE, seed);
            for (int j = 0; j < PIVCO_BLOCK_SIZE; j++) {
                /* code_la is no longer a table field (production encodes on
                 * ranks); compute the left-aligned code locally for the sim. */
                uint8_t s = symbols[j], len = tbl->code_len[s];
                codes_la[j] = len > 0
                    ? (uint16_t)(tbl->code[s] << (16 - len)) : 0;
            }
            simulate_partition(tbl, codes_la, PIVCO_BLOCK_SIZE, tbl->tree_root, 0,
                               &flat_bits, depth_stats, tmp_buf);
        }

        char title[128];
        snprintf(title, sizeof(title), "%s  (%d blocks × %d symbols)",
                 name, N_BLOCKS, PIVCO_BLOCK_SIZE);
        print_depth_table(depth_stats, flat_bits, title);

        if (n_summary < (int)(sizeof(summary)/sizeof(summary[0]))) {
            summary_t *s = &summary[n_summary++];
            s->name = name;
            s->flat_bits = flat_bits;
            s->bit_H_total = 0; s->byte_H_total = 0;
            s->n_bits_total = 0; s->n_bytes_total = 0;
            for (int d = 0; d < MAX_DEPTH_BUCKETS; d++) {
                s->bit_H_total  += depth_stats[d].bit_entropy;
                s->byte_H_total += depth_stats[d].byte_entropy;
                s->n_bits_total += depth_stats[d].n_bits;
                s->n_bytes_total+= depth_stats[d].n_bytes;
            }
        }
    }

    /* Cross-distribution summary. */
    printf("\n=== SUMMARY (across %d distributions × %d blocks each) ===\n",
           n_summary, N_BLOCKS);
    printf("  %-18s %12s %12s %12s %10s\n",
           "distribution", "bit_H2", "byte_H8", "delta", "delta_pct");
    printf("  %-18s %12s %12s %12s %10s\n",
           "------------", "------", "-------", "-----", "---------");
    for (int k = 0; k < n_summary; k++) {
        summary_t *s = &summary[k];
        double delta = s->byte_H_total - s->bit_H_total;
        double pct = s->bit_H_total > 0 ? 100.0 * delta / s->bit_H_total : 0.0;
        printf("  %-18s %12.0f %12.0f %12.0f   %+7.2f%%\n",
               s->name, s->bit_H_total, s->byte_H_total, delta, pct);
    }

    free(tbl); free(codes_la); free(tmp_buf); free(symbols);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--verify-bytes") == 0) {
        return run_verify_dist_mode(/*main_only=*/1);
    }
    if (argc == 2 && strcmp(argv[1], "--verify-bytes-all") == 0) {
        return run_verify_dist_mode(/*main_only=*/0);
    }
    if (argc == 2 && strcmp(argv[1], "--exact-tier") == 0) {
        return run_exact_tier_mode(/*main_only=*/1);
    }
    if (argc == 2 && strcmp(argv[1], "--exact-tier-all") == 0) {
        return run_exact_tier_mode(/*main_only=*/0);
    }
    if (argc == 1) {
        return run_dist_mode(/*main_only=*/1);
    }
    if (argc == 2 && strcmp(argv[1], "--dist") == 0) {
        return run_dist_mode(/*main_only=*/1);
    }
    if (argc == 2 && strcmp(argv[1], "--dist-all") == 0) {
        return run_dist_mode(/*main_only=*/0);
    }
    if (argc >= 2 && strcmp(argv[1], "--help") == 0) {
        fprintf(stderr,
                "Usage:\n"
                "  %s                        # MAIN bench distributions (default)\n"
                "  %s --dist                 # same as above\n"
                "  %s --dist-all             # ALL bench distributions\n"
                "  %s <file> [<file>...]     # per-block ratio bound on real files\n"
                "  %s --verify-bytes         # per-depth bit-vs-byte (MAIN dists)\n"
                "  %s --verify-bytes-all     # per-depth bit-vs-byte (ALL dists)\n",
                argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }

    pivco_table_t *tbl = malloc(sizeof(*tbl));
    uint64_t *subtree_freq = calloc(PIVCO_MAX_TREE_NODES, sizeof(uint64_t));
    if (!tbl || !subtree_freq) { fprintf(stderr, "OOM\n"); return 1; }

    printf("%-30s %12s %12s %12s %12s   %6s %6s   %6s\n",
           "file", "raw_B", "huff_B", "huff_flat_B", "shannon_B",
           "huff-%", "hflat-%", "shan-%");
    printf("%-30s %12s %12s %12s %12s   %6s %6s   %6s\n",
           "----", "-----", "------", "-----------", "---------",
           "------", "-------", "------");

    for (int ai = 1; ai < argc; ai++) {
        const char *path = argv[ai];
        FILE *f = fopen(path, "rb");
        if (!f) { perror(path); continue; }
        struct stat st;
        if (stat(path, &st) != 0) { perror("stat"); fclose(f); continue; }
        size_t total_size = (size_t)st.st_size;
        uint8_t *buf = malloc(total_size ? total_size : 1);
        if (!buf) { fprintf(stderr, "OOM\n"); fclose(f); continue; }
        if (total_size && fread(buf, 1, total_size, f) != total_size) {
            fprintf(stderr, "short read on %s\n", path);
            free(buf); fclose(f); continue;
        }
        fclose(f);

        double total_huf_bits = 0, total_tans_flat_bits = 0;
        double total_shannon_bits = 0, total_flat_only_bits = 0;
        double total_raw_bits = (double)total_size * 8.0;
        int n_blocks = 0, n_skipped_singleton = 0;

        size_t off = 0;
        while (off < total_size) {
            size_t blk = PIVCO_BLOCK_SIZE;
            if (off + blk > total_size) blk = total_size - off;

            uint64_t freq[256] = {0};
            for (size_t i = 0; i < blk; i++) freq[buf[off + i]]++;

            /* Singleton blocks: build_table emits a degenerate
               single-leaf tree.  Bits are 0 in any model.  Skip. */
            int n_syms = 0;
            for (int s = 0; s < 256; s++) if (freq[s]) n_syms++;
            if (n_syms < 2) { n_skipped_singleton++; off += blk; n_blocks++; continue; }

            if (pivco_build_table(bench_cfg(), freq, tbl) != PIVCO_OK) {
                fprintf(stderr, "build_table failed at block %d of %s\n",
                        n_blocks, path);
                break;
            }

            for (int s = 0; s < 256; s++)
                if (freq[s]) total_huf_bits += (double)freq[s] * tbl->code_len[s];

            memset(subtree_freq, 0, PIVCO_MAX_TREE_NODES * sizeof(uint64_t));
            fill_subtree_freq(tbl, freq, tbl->tree_root, subtree_freq);

            double full_tans = 0, flat_carve = 0, flat_only = 0;
            walk_full_tans (tbl, subtree_freq, tbl->tree_root, &full_tans);
            walk_flat_carve(tbl, subtree_freq, tbl->tree_root, &flat_carve, &flat_only);

            total_shannon_bits   += full_tans;
            total_tans_flat_bits += flat_carve;
            total_flat_only_bits += flat_only;

            off += blk;
            n_blocks++;
        }

        double raw_B       = total_raw_bits       / 8.0;
        double huf_B       = total_huf_bits       / 8.0;
        double tans_flat_B = total_tans_flat_bits / 8.0;
        double shannon_B   = total_shannon_bits   / 8.0;
        double flat_only_B = total_flat_only_bits / 8.0;

        printf("%-30s %12.0f %12.0f %12.0f %12.0f   %5.1f%% %5.1f%%   %5.1f%%",
               path, raw_B, huf_B, tans_flat_B, shannon_B,
               100.0 * (huf_B       / raw_B),
               100.0 * (tans_flat_B / raw_B),
               100.0 * (shannon_B   / raw_B));
        printf("   [flat: %.1f%% of huff_flat_B]\n",
               flat_only_B > 0 ? 100.0 * flat_only_B / tans_flat_B : 0.0);

        free(buf);
        (void)n_skipped_singleton;
    }

    free(tbl);
    free(subtree_freq);
    return 0;
}
