/* bench_dist_stats — print distinct / total / entropy / avg Huffman code
 * length / min_len / max_len for distributions registered in
 * bench_distributions.c.
 *
 * Used to populate the "Test Datasets" table in README.md.
 *
 * Flags:
 *   --main   only the MAIN (dev-iteration) distributions
 *   --csv    comma-separated output (no padding / rule line) */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "pivco_huffman.h"

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int             bench_dist_is_main(int idx);

int main(int argc, char **argv)
{
    int main_only = 0, csv = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--main")) main_only = 1;
        else if (!strcmp(argv[a], "--csv")) csv = 1;
    }

    bench_init();
    int n = bench_num_distributions();

    if (csv) {
        printf("name,distinct,total,entropy,avg_huff_len,min_len,max_len\n");
    } else {
        printf("%-15s | %8s | %12s | %10s | %12s | %7s | %7s\n",
               "name", "distinct", "total", "entropy",
               "avg_huff_len", "min_len", "max_len");
        printf("----------------+----------+--------------+------------"
               "+--------------+---------+--------\n");
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

        pivco_huffman_table_t t;
        int min_len = 0, max_len = 0;
        double avg_len = 0.0;
        if (pivco_huffman_build_table(f, &t) == PIVCO_OK) {
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
        }

        if (csv) {
            printf("%s,%d,%llu,%.4f,%.4f,%d,%d\n",
                   nm, distinct, (unsigned long long)total, H, avg_len,
                   min_len, max_len);
        } else {
            printf("%-15s | %8d | %12llu | %10.3f | %12.3f | %7d | %7d\n",
                   nm, distinct, (unsigned long long)total, H, avg_len,
                   min_len, max_len);
        }
    }
    return 0;
}
