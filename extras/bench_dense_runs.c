/* bench_dense_runs - estimate fraction of length-K windows in the
 * original symbol stream that hit a single leaf, for each distribution.
 *
 * If a length-K window is all-same-symbol, the leaf scatter for that
 * window can be replaced by a single SIMD vector-store of K bytes
 * (the indices arrive at the leaf in original-stream order, sorted,
 * so all-same-symbol windows imply consecutive index runs at the
 * leaf level).
 *
 * For an i.i.d. source with leaf probability p_l, fraction of length-K
 * windows that hit leaf l = p_l^K.  Summing across leaves gives the
 * total fraction of stores that could be coalesced.
 *
 * Real distributions have autocorrelation (text, code, etc.) so actual
 * dense-run frequency is probably higher than this i.i.d. lower bound.
 *
 * Prints, per distribution:
 *   - cum prob of K=2, 4, 8, 16-cont windows
 *   - top-3 contributing symbols
 *   - "savings if coalesced": K=8 hit_rate × 7/8  (skip 7 of 8 stores)
 *
 * Compile: linked into pivco_dense_runs binary by CMakeLists.txt.
 */

#include "pivco_huffman.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

static double pow_int(double p, int k) {
    double r = 1.0;
    for (int i = 0; i < k; i++) r *= p;
    return r;
}

static int cmp_desc(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) - (da > db);
}

int main(void) {
    bench_init();
    int n_dist = bench_num_distributions();

    printf("Expected fraction of length-K original-stream windows that hit\n");
    printf("a single leaf, assuming i.i.d. symbols (lower bound for real\n");
    printf("text with autocorrelation).  Sum_leaf p_leaf^K.\n\n");
    printf("%-14s | %8s %8s %8s %8s | %s\n",
            "distribution", "K=2", "K=4", "K=8", "K=16",
            "top-3 leaves (p, p^8)");
    printf("%-14s-+-%8s-%8s-%8s-%8s-+-%s\n",
            "--------------", "--------", "--------", "--------",
            "--------", "--------");

    for (int d = 0; d < n_dist; d++) {
        const char *name = bench_dist_name(d);
        const uint64_t *freq = bench_dist_freq(d);

        uint64_t total = 0;
        for (int s = 0; s < 256; s++) total += freq[s];
        if (total == 0) continue;

        double p[256];
        for (int s = 0; s < 256; s++) p[s] = (double)freq[s] / (double)total;

        double cum[5] = {0,0,0,0,0};  /* K=1,2,4,8,16 */
        for (int s = 0; s < 256; s++) {
            cum[0] += p[s];
            cum[1] += pow_int(p[s], 2);
            cum[2] += pow_int(p[s], 4);
            cum[3] += pow_int(p[s], 8);
            cum[4] += pow_int(p[s], 16);
        }

        /* top-3 contributing leaves to p^8 */
        double sorted_p[256];
        memcpy(sorted_p, p, sizeof(p));
        qsort(sorted_p, 256, sizeof(double), cmp_desc);

        printf("%-14s | %7.3f%% %7.3f%% %7.3f%% %7.3f%% | ",
                name, cum[1]*100, cum[2]*100, cum[3]*100, cum[4]*100);
        for (int k = 0; k < 3 && sorted_p[k] > 0.001; k++) {
            printf("(%.2f→%.2f%%) ", sorted_p[k], pow_int(sorted_p[k],8)*100);
        }
        printf("\n");
    }

    /* "Headline number": for each distribution, fraction of stores
     * that could be saved if we coalesced K=8 dense windows.  A dense
     * window saves 7 of 8 byte stores → 87.5% × hit_rate. */
    printf("\nUpper bound on byte-store reduction if we did K=8 coalescing:\n");
    printf("  saving = K8_hit_rate × 7/8  (saves 7 of 8 stores per dense window)\n\n");
    printf("%-14s | %8s   %8s\n", "distribution", "K=8 hit", "store saving");
    for (int d = 0; d < n_dist; d++) {
        const char *name = bench_dist_name(d);
        const uint64_t *freq = bench_dist_freq(d);
        uint64_t total = 0;
        for (int s = 0; s < 256; s++) total += freq[s];
        if (total == 0) continue;
        double cum8 = 0;
        for (int s = 0; s < 256; s++) {
            double pp = (double)freq[s] / (double)total;
            cum8 += pow_int(pp, 8);
        }
        double saving = cum8 * 7.0 / 8.0;
        printf("%-14s | %7.3f%%    %7.3f%%\n", name, cum8*100, saving*100);
    }

    return 0;
}
