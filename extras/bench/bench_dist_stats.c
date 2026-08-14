/* bench_dist_stats — print distinct / total / entropy / avg Huffman code
 * length / min_len / max_len / max_node_benefit for distributions
 * registered in bench_distributions.c.
 *
 * Used to populate the "Test Datasets" table in README.md.
 *
 *   max_node_benefit = max over the OPTIMIZED tree's bitmap-emitting
 *   internal nodes of (n_node/N) * (1 - H_binary(p_split)) -- the largest
 *   single-node bits-saved-per-source-byte from entropy-coding one
 *   partition bitmap.  Tree comes from pivco_build_table (the
 *   library); flat subtrees (packed bits, no bitmap) are skipped.  Same
 *   metric as tree_viz "Top internal nodes by bits saved" (/byte).
 *
 * Flags:
 *   --main   only the MAIN (dev-iteration) distributions
 *   --csv    comma-separated output (no padding / rule line) */

#include <math.h>
#include "bench_ctx.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "pivco_huffman.h"

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int             bench_dist_is_main(int idx);

/* Curated one-line provenance per distribution.  Hardcoded here so the
 * emitted CSV carries the `source` column directly (no separate curated
 * file / JOIN).  Unknown names return "". */
static const char *dist_source(const char *name)
{
    struct { const char *n, *s; } tab[] = {
        { "proba80",      "Synthetic FSE benchmark: geometric, top symbol p~0.80 (very skewed)" },
        { "english",      "Synthetic: English letter relative frequencies" },
        { "html_wiki",    "English Wikipedia \"Cat\" article, served HTML (cat-wiki.html)" },
        { "prose_pride",  "Project Gutenberg plain-text Pride and Prejudice (pride.txt)" },
        { "image_jpeg",   "Wikimedia Commons JPEG photo Cat03.jpg (near-uniform 256 bytes)" },
        { "json_api",     "GitHub API commits feed JSON (json_api.json)" },
        { "source_c",     "zstd lib/compress/zstd_compress.c source file (source_c.c)" },
        { "log_apache",   "Apache HTTP access log sample (log_apache.log)" },
        { "dna_fasta",    "NCBI E. coli K-12 genome FASTA, truncated ~500 KB (dna_fasta.fa)" },
        { "csv_numeric",  "OWID CO2 dataset CSV, truncated (csv_numeric.csv)" },
        { "chinese_text", "Project Gutenberg Hong Lou Meng, Chinese UTF-8, truncated (chinese_text.txt)" },
        { "gzip_random",  "gzip(cat-wiki.html) -- near-uniform 256-byte corner case" },
        { "calgary_pic",  "Calgary Corpus 1bpp CCITT scanned page, proba80-like real data (calgary_pic)" },
    };
    for (size_t i = 0; i < sizeof(tab) / sizeof(tab[0]); i++)
        if (!strcmp(name, tab[i].n)) return tab[i].s;
    return "";
}

static double h_binary(double p)
{
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p);
}

/* Flat depth of the subtree rooted at `node`: D if every leaf sits at the
 * same depth D from here, else -1 (leaf = 0).  Pure tree-structure read of
 * the library-built tree -- no rebuild logic. */
static int subtree_flat_depth(const pivco_table_t *t, int16_t node)
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) return 0;
    int ld = subtree_flat_depth(t, n->left);
    if (ld < 0) return -1;
    int rd = subtree_flat_depth(t, n->right);
    if (rd < 0 || rd != ld) return -1;
    return ld + 1;
}

static uint64_t subtree_freq_sum(const pivco_table_t *t,
                                  const uint64_t *freq, int16_t node)
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) return freq[n->symbol];
    return subtree_freq_sum(t, freq, n->left)
         + subtree_freq_sum(t, freq, n->right);
}

/* Walk the optimized tree; for every internal node that emits a per-node
 * partition bitmap (i.e. NOT inside a maximal D>=2 flat subtree, which is
 * packed-bits instead), track max[ n_node * (1 - H_binary(p_split)) ].
 * Returns the subtree's total frequency count.  This is the per-node
 * "bits saved by entropy-coding the partition bitmap" maximum -- the same
 * metric tree_viz shows as "Top internal nodes by bits saved". */
static uint64_t walk_top1(const pivco_table_t *t,
                          const uint64_t *freq, int16_t node,
                          double *max_saved)
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) return freq[n->symbol];
    if (subtree_flat_depth(t, node) >= 2)
        return subtree_freq_sum(t, freq, node);   /* flat: no bitmap node */

    uint64_t left  = walk_top1(t, freq, n->left,  max_saved);
    uint64_t right = walk_top1(t, freq, n->right, max_saved);
    uint64_t total = left + right;
    if (total > 0) {
        double p = (double)left / (double)total;
        double saved = (double)total * (1.0 - h_binary(p));
        if (saved > *max_saved) *max_saved = saved;
    }
    return total;
}

int main(int argc, char **argv)
{
    int main_only = 0, csv = 0;
    const char *csv_out = NULL;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--main")) main_only = 1;
        else if (!strcmp(argv[a], "--csv")) csv = 1;
        else if (!strncmp(argv[a], "--csv-out=", 10)) {
            csv_out = argv[a] + 10; csv = 1;
        }
    }

    FILE *out = stdout;
    if (csv_out) {
        out = fopen(csv_out, "w");
        if (!out) { perror(csv_out); return 1; }
    }

    bench_init();
    int n = bench_num_distributions();

    if (csv) {
        fprintf(out, "name,distinct,total,entropy,avg_huff_len,min_len,"
                     "max_len,max_node_benefit,source\n");
    } else {
        printf("%-15s | %8s | %12s | %10s | %12s | %7s | %7s | %16s\n",
               "name", "distinct", "total", "entropy",
               "avg_huff_len", "min_len", "max_len", "max_node_benefit");
        printf("----------------+----------+--------------+------------"
               "+--------------+---------+---------+------------------\n");
    }

    for (int i = 0; i < n; i++) {
        if (main_only && !bench_dist_is_main(i)) continue;

        const char *nm     = bench_dist_name(i);
        const uint64_t *f  = bench_dist_freq(i);

        uint64_t total = 0;
        int distinct  = 0;
        for (int s = 0; s < 256; s++) {
            total += f[s];
            if (f[s]) distinct++;
        }

        double H = 0.0;
        if (total > 0) {
            for (int s = 0; s < 256; s++) {
                if (f[s]) {
                    double p = (double)f[s] / (double)total;
                    H -= p * log2(p);
                }
            }
        }

        pivco_table_t t;
        int min_len = 0, max_len = 0;
        double avg_len = 0.0, max_node_benefit = 0.0;
        if (pivco_build_table(bench_cfg(), f, &t) == PIVCO_OK) {
            int mn = 255, mx = 0;
            double wsum = 0.0;
            for (int s = 0; s < 256; s++) {
                if (f[s] && t.code_len[s]) {
                    if (t.code_len[s] < mn) mn = t.code_len[s];
                    if (t.code_len[s] > mx) mx = t.code_len[s];
                    wsum += (double)f[s] * (double)t.code_len[s];
                }
            }
            min_len = mn;
            max_len = mx;
            avg_len = (total > 0) ? wsum / (double)total : 0.0;

            double max_saved = 0.0;
            walk_top1(&t, f, t.tree_root, &max_saved);
            max_node_benefit = (total > 0) ? max_saved / (double)total : 0.0;
        }

        if (csv) {
            /* CSV-quote the source field (may contain commas / quotes). */
            fprintf(out, "%s,%d,%llu,%.4f,%.4f,%d,%d,%.4f,\"",
                    nm, distinct, (unsigned long long)total, H, avg_len,
                    min_len, max_len, max_node_benefit);
            for (const char *p = dist_source(nm); *p; p++) {
                if (*p == '"') fputc('"', out);
                fputc(*p, out);
            }
            fprintf(out, "\"\n");
        } else {
            printf("%-15s | %8d | %12llu | %10.3f | %12.3f | %7d | %7d | %16.4f\n",
                   nm, distinct, (unsigned long long)total, H, avg_len,
                   min_len, max_len, max_node_benefit);
        }
    }

    if (out != stdout) fclose(out);
    return 0;
}
