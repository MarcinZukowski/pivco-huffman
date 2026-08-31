/* nibbench: id-51 tableLog sweep bench.  Compress (PHA, dynamic on) +
 * decompress one file in-process; report size, enc/dec speed (best-of),
 * and slot-51 commit stats.  Link against a lib built with the desired
 * -DPIVCO_FSE_NIB_TABLELOG. */
#include "pivcohuf_file.h"
#include "pivco_huffman.h"
#include "pivco_fse.h"
#include "pivco_fse_tables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: nibbench FILE\n"); return 1; }
    struct stat st;
    if (stat(argv[1], &st)) { perror("stat"); return 1; }
    size_t n = (size_t)st.st_size;
    uint8_t *src = malloc(n ? n : 1);
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(src, 1, n, f) != n) { perror("read"); return 1; }
    fclose(f);

    size_t cap = pivcohuf_compress_bound(n);
    uint8_t *c = malloc(cap);
    uint8_t *d = malloc(n + 64);

    /* encode: best of 3 (stats from the last run) */
    double enc = 1e30; size_t cl = 0;
    for (int r = 0; r < 3; r++) {
        pivco_fse_stats_reset();
        size_t l = cap;
        double t = now();
        if (pivcohuf_compress_ex(src, n, c, &l, 1) != PIVCOHUF_OK) {
            fprintf(stderr, "compress fail\n"); return 1;
        }
        double dt = now() - t;
        if (dt < enc) enc = dt;
        cl = l;
    }
    uint64_t commit[PIVCO_FSE_STATS_SLOTS], attempt[PIVCO_FSE_STATS_SLOTS];
    uint64_t bin[PIVCO_FSE_STATS_SLOTS], bout[PIVCO_FSE_STATS_SLOTS];
    pivco_fse_stats_get(commit, attempt, bin, bout);
    uint64_t dyn_c = commit[PIVCO_FSE_DYNAMIC_ID];
    uint64_t dyn_in = bin[PIVCO_FSE_DYNAMIC_ID], dyn_out = bout[PIVCO_FSE_DYNAMIC_ID];
    uint64_t sta_c = 0;
    for (int i = 1; i <= PIVCO_FSE_NUM_TABLES; i++) sta_c += commit[i];

    /* decode: best of 15, verify once */
    double dec = 1e30;
    for (int r = 0; r < 15; r++) {
        size_t ol = n + 64;
        double t = now();
        if (pivcohuf_decompress(c, cl, d, &ol) != PIVCOHUF_OK || ol != n) {
            fprintf(stderr, "decompress fail\n"); return 1;
        }
        double dt = now() - t;
        if (dt < dec) dec = dt;
    }
    if (n && memcmp(src, d, n)) { fprintf(stderr, "VERIFY FAIL\n"); return 1; }

    printf("NB %s n %zu out %zu enc_MBps %.1f dec_MBps %.1f dyn51 %llu dynin %llu dynout %llu static %llu\n",
           argv[1], n, cl, n / enc / 1e6, n / dec / 1e6,
           (unsigned long long)dyn_c, (unsigned long long)dyn_in,
           (unsigned long long)dyn_out, (unsigned long long)sta_c);
    return 0;
}
