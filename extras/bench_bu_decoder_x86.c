/* Bottom-up vs top-down x86 decoder A/B bench.  Mirror of
 * bench_bu_decoder.c but uses *_x86 functions. */

#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void bench_init(void);
extern int  bench_num_distributions(void);
extern const char *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern void bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                   int n_symbols, uint64_t seed);

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    bench_init();
    int reps = (argc > 1) ? atoi(argv[1]) : 5000;
    int nblocks = (argc > 2) ? atoi(argv[2]) : 256;

    printf("Bottom-up vs top-down x86 decoder (reps=%d, nblocks=%d/run)\n", reps, nblocks);
    printf("%-15s | %7s %7s | %6s\n", "DIST", "td_M/s", "bu_M/s", "Δ%");
    printf("----------------|-----------------|-------\n");

    for (int d = 0; d < bench_num_distributions(); d++) {
        const char *name = bench_dist_name(d);
        const uint64_t *freq = bench_dist_freq(d);

        pivco_huffman_table_t table;
        if (pivco_huffman_build_table(freq, &table) != PIVCO_OK) continue;

        uint8_t *symbols = malloc((size_t)nblocks * PIVCO_BLOCK_SIZE);
        bench_generate_symbols(d, symbols, nblocks * PIVCO_BLOCK_SIZE, 0xBEEFCAFE);

        uint8_t *enc_buf = malloc((size_t)nblocks * PIVCO_MAX_ENCODED_SIZE);
        size_t *enc_off = malloc(((size_t)nblocks + 1) * sizeof(size_t));
        enc_off[0] = 0;
        for (int b = 0; b < nblocks; b++) {
            size_t len;
            pivco_huffman_encode(symbols + (size_t)b * PIVCO_BLOCK_SIZE,
                                  &table, enc_buf + enc_off[b], &len);
            enc_off[b + 1] = enc_off[b] + len;
        }

        uint8_t *dec_buf = malloc((size_t)nblocks * PIVCO_BLOCK_SIZE);

        size_t consumed;
        pivco_huffman_decode_x86(enc_buf, enc_off[1], &table,
                                  dec_buf, &consumed);
        uint8_t bu_dec[PIVCO_BLOCK_SIZE];
        pivco_huffman_decode_bu_x86(enc_buf, enc_off[1], &table,
                                     bu_dec, &consumed);
        if (memcmp(dec_buf, bu_dec, PIVCO_BLOCK_SIZE) != 0) {
            printf("%-15s | CORRECTNESS FAIL\n", name);
            free(symbols); free(enc_buf); free(enc_off); free(dec_buf);
            continue;
        }

        for (int b = 0; b < nblocks; b++) {
            pivco_huffman_decode_x86(enc_buf + enc_off[b],
                                      enc_off[b + 1] - enc_off[b],
                                      &table, dec_buf + (size_t)b * PIVCO_BLOCK_SIZE,
                                      &consumed);
        }

        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (int b = 0; b < nblocks; b++) {
                pivco_huffman_decode_x86(enc_buf + enc_off[b],
                                          enc_off[b + 1] - enc_off[b],
                                          &table, dec_buf + (size_t)b * PIVCO_BLOCK_SIZE,
                                          &consumed);
            }
        }
        double t_td = now_ns() - t0;

        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (int b = 0; b < nblocks; b++) {
                pivco_huffman_decode_bu_x86(enc_buf + enc_off[b],
                                             enc_off[b + 1] - enc_off[b],
                                             &table, dec_buf + (size_t)b * PIVCO_BLOCK_SIZE,
                                             &consumed);
            }
        }
        double t_bu = now_ns() - t0;

        double td_mps = (double)reps * nblocks * PIVCO_BLOCK_SIZE / t_td * 1000.0;
        double bu_mps = (double)reps * nblocks * PIVCO_BLOCK_SIZE / t_bu * 1000.0;
        double delta = 100.0 * (bu_mps - td_mps) / td_mps;
        printf("%-15s | %7.0f %7.0f | %+5.1f%%\n", name, td_mps, bu_mps, delta);

        free(symbols); free(enc_buf); free(enc_off); free(dec_buf);
    }
    return 0;
}
