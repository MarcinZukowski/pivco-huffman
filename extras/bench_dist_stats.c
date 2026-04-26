/* bench_dist_stats — print distinct / total / entropy / min_len / max_len
 * for every distribution registered in bench_distributions.c.
 *
 * Used to populate the "Test Datasets" table in README.md. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "pivco_huffman.h"

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

int main(void)
{
    bench_init();
    int n = bench_num_distributions();

    printf("%-15s | %8s | %12s | %10s | %7s | %7s\n",
           "name", "distinct", "total", "entropy", "min_len", "max_len");
    printf("----------------+----------+--------------+------------+---------+--------\n");
    for (int i = 0; i < n; i++) {
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
        if (pivco_huffman_build_table(f, &t) == PIVCO_OK) {
            int mn = 255, mx = 0;
            for (int s = 0; s < 256; s++) {
                if (f[s] && t.code_len[s]) {
                    if (t.code_len[s] < mn) mn = t.code_len[s];
                    if (t.code_len[s] > mx) mx = t.code_len[s];
                }
            }
            min_len = mn;
            max_len = mx;
        }

        printf("%-15s | %8d | %12llu | %10.3f | %7d | %7d\n",
               nm, distinct, (unsigned long long)total, H, min_len, max_len);
    }
    return 0;
}
