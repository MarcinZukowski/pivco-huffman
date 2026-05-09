/* Verify correctness of pivco_huffman_decode_dual_neon vs serial
 * decode, then measure throughput on a chosen distribution. */

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
    const char *dist = (argc > 1) ? argv[1] : "prose_pride";
    int reps = (argc > 2) ? atoi(argv[2]) : 100000;

    int dist_idx = -1;
    for (int i = 0; i < bench_num_distributions(); i++) {
        if (strcmp(bench_dist_name(i), dist) == 0) { dist_idx = i; break; }
    }
    if (dist_idx < 0) { fprintf(stderr, "no dist %s\n", dist); return 1; }

    pivco_huffman_table_t table;
    pivco_huffman_build_table(bench_dist_freq(dist_idx), &table);

    /* Generate two blocks of source symbols */
    uint8_t sym_A[PIVCO_BLOCK_SIZE];
    uint8_t sym_B[PIVCO_BLOCK_SIZE];
    bench_generate_symbols(dist_idx, sym_A, PIVCO_BLOCK_SIZE, 0xAAAA1234ULL);
    bench_generate_symbols(dist_idx, sym_B, PIVCO_BLOCK_SIZE, 0xBBBB5678ULL);

    /* Encode each block */
    uint8_t enc_A[PIVCO_MAX_ENCODED_SIZE];
    uint8_t enc_B[PIVCO_MAX_ENCODED_SIZE];
    size_t  elen_A, elen_B;
    pivco_huffman_encode_neon(sym_A, &table, enc_A, &elen_A);
    pivco_huffman_encode_neon(sym_B, &table, enc_B, &elen_B);
    printf("encoded A: %zu bytes; B: %zu bytes (dist=%s)\n", elen_A, elen_B, dist);

    /* Decode serial */
    uint8_t dec_A_serial[PIVCO_BLOCK_SIZE];
    uint8_t dec_B_serial[PIVCO_BLOCK_SIZE];
    size_t cA, cB;
    pivco_huffman_decode_neon(enc_A, elen_A, &table, dec_A_serial, &cA);
    pivco_huffman_decode_neon(enc_B, elen_B, &table, dec_B_serial, &cB);
    if (memcmp(sym_A, dec_A_serial, PIVCO_BLOCK_SIZE) != 0
        || memcmp(sym_B, dec_B_serial, PIVCO_BLOCK_SIZE) != 0) {
        fprintf(stderr, "SERIAL DECODE: roundtrip mismatch (this is a baseline bug, not dual)\n");
        return 1;
    }

    /* Decode dual */
    uint8_t dec_A_dual[PIVCO_BLOCK_SIZE];
    uint8_t dec_B_dual[PIVCO_BLOCK_SIZE];
    size_t cAd, cBd;
    pivco_huffman_decode_dual_neon(enc_A, elen_A, enc_B, elen_B, &table,
                                    dec_A_dual, dec_B_dual, &cAd, &cBd);
    if (memcmp(sym_A, dec_A_dual, PIVCO_BLOCK_SIZE) != 0) {
        fprintf(stderr, "DUAL DECODE: A mismatch\n");
        for (int i = 0; i < 32; i++)
            fprintf(stderr, "  pos %d: got %d, expected %d\n",
                    i, dec_A_dual[i], sym_A[i]);
        return 2;
    }
    if (memcmp(sym_B, dec_B_dual, PIVCO_BLOCK_SIZE) != 0) {
        fprintf(stderr, "DUAL DECODE: B mismatch\n");
        for (int i = 0; i < 32; i++) {
            if (dec_B_dual[i] != sym_B[i])
                fprintf(stderr, "  pos %d: got %d, expected %d\n",
                        i, dec_B_dual[i], sym_B[i]);
        }
        return 3;
    }
    printf("correctness PASS (both A and B match serial decode)\n");

    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        pivco_huffman_decode_neon(enc_A, elen_A, &table, dec_A_serial, &cA);
        pivco_huffman_decode_neon(enc_B, elen_B, &table, dec_B_serial, &cB);
    }

    /* Time serial */
    double t0 = now_ns();
    for (int i = 0; i < reps; i++) {
        pivco_huffman_decode_neon(enc_A, elen_A, &table, dec_A_serial, &cA);
        pivco_huffman_decode_neon(enc_B, elen_B, &table, dec_B_serial, &cB);
    }
    double t_serial = now_ns() - t0;

    /* Time dual */
    t0 = now_ns();
    for (int i = 0; i < reps; i++) {
        pivco_huffman_decode_dual_neon(enc_A, elen_A, enc_B, elen_B, &table,
                                        dec_A_dual, dec_B_dual, &cAd, &cBd);
    }
    double t_dual = now_ns() - t0;

    double ns_per_pair_serial = t_serial / reps;
    double ns_per_pair_dual   = t_dual / reps;
    double pct = 100.0 * (ns_per_pair_serial - ns_per_pair_dual) / ns_per_pair_serial;

    printf("dist=%s reps=%d  PIVCO_LA_K not visible from here\n", dist, reps);
    printf("  serial:  %.1f ns / 2-block-pair\n", ns_per_pair_serial);
    printf("  dual  :  %.1f ns / 2-block-pair\n", ns_per_pair_dual);
    printf("  delta :  %+.2f%% (positive = dual faster)\n", pct);
    printf("  M/s serial = %.1f, dual = %.1f\n",
           2.0 * PIVCO_BLOCK_SIZE / ns_per_pair_serial * 1000.0,
           2.0 * PIVCO_BLOCK_SIZE / ns_per_pair_dual * 1000.0);

    return 0;
}
