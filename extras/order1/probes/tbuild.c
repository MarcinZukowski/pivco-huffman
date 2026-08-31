/* Price pivco_build_table_from_code_lens: the per-new-table decode cost
 * of the block-table ring.  Lengths come from a real stream's table. */
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }
int main(int argc, char **argv){
    struct stat st; stat(argv[1], &st);
    size_t n = st.st_size; uint8_t *b = malloc(n);
    FILE *f = fopen(argv[1], "rb");
    if (fread(b,1,n,f) != n) return 1;
    uint64_t freq[256] = {0};
    for (size_t i = 0; i < n; i++) freq[b[i]]++;
    pivco_cfg_t cfg = pivco_cfg_default;
    cfg.fse_enabled = 1;
    pivco_table_t t0, t1;
    if (pivco_build_table(&cfg, freq, &t0) != PIVCO_OK) return 1;
    /* warm */
    for (int i = 0; i < 100; i++)
        pivco_build_table_from_code_lens(&cfg, t0.code_len, &t1);
    int N = 20000;
    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        double s = now();
        for (int i = 0; i < N; i++)
            pivco_build_table_from_code_lens(&cfg, t0.code_len, &t1);
        double dt = (now() - s) / N;
        if (dt < best) best = dt;
    }
    printf("build_from_code_lens: %.2f us  (num_symbols=%u max_len=%u)\n",
           best*1e6, (unsigned)t0.num_symbols, (unsigned)t0.max_len);
    return 0;
}
