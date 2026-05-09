/* Verify correctness of cross-block fusion (g_pivco_fusion_enabled +
 * pivco_huffman_set_next_neon) vs plain serial decode, then measure
 * throughput on a chosen distribution. */

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

/* Decode a 2-block stream serially (plain decode, no fusion). */
static void decode_serial(const uint8_t *in_A, size_t len_A,
                           const uint8_t *in_B, size_t len_B,
                           const pivco_huffman_table_t *t,
                           uint8_t *sA, uint8_t *sB,
                           size_t *cA, size_t *cB)
{
    g_pivco_fusion_enabled = 0;
    pivco_huffman_set_next_neon(NULL);
    pivco_huffman_decode_neon(in_A, len_A, t, sA, cA);
    pivco_huffman_decode_neon(in_B, len_B, t, sB, cB);
}

/* Decode a 2-block stream with fusion enabled. */
static void decode_fused(const uint8_t *in_A, size_t len_A,
                          const uint8_t *in_B, size_t len_B,
                          const pivco_huffman_table_t *t,
                          uint8_t *sA, uint8_t *sB,
                          size_t *cA, size_t *cB)
{
    g_pivco_fusion_enabled = 1;
    pivco_huffman_set_next_neon(in_B);   /* A's scatters fuse into B's root partition */
    pivco_huffman_decode_neon(in_A, len_A, t, sA, cA);
    pivco_huffman_set_next_neon(NULL);   /* B is the last block, no fusion target */
    pivco_huffman_decode_neon(in_B, len_B, t, sB, cB);
    g_pivco_fusion_enabled = 0;
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

    uint8_t sym_A[PIVCO_BLOCK_SIZE], sym_B[PIVCO_BLOCK_SIZE];
    bench_generate_symbols(dist_idx, sym_A, PIVCO_BLOCK_SIZE, 0xAAAA1234ULL);
    bench_generate_symbols(dist_idx, sym_B, PIVCO_BLOCK_SIZE, 0xBBBB5678ULL);

    uint8_t enc_A[PIVCO_MAX_ENCODED_SIZE];
    uint8_t enc_B[PIVCO_MAX_ENCODED_SIZE];
    size_t  elen_A, elen_B;
    pivco_huffman_encode_neon(sym_A, &table, enc_A, &elen_A);
    pivco_huffman_encode_neon(sym_B, &table, enc_B, &elen_B);
    printf("encoded A: %zu bytes; B: %zu bytes (dist=%s)\n", elen_A, elen_B, dist);

    /* Correctness check: serial first */
    uint8_t dec_A_serial[PIVCO_BLOCK_SIZE], dec_B_serial[PIVCO_BLOCK_SIZE];
    size_t cA, cB;
    decode_serial(enc_A, elen_A, enc_B, elen_B, &table,
                  dec_A_serial, dec_B_serial, &cA, &cB);
    if (memcmp(sym_A, dec_A_serial, PIVCO_BLOCK_SIZE) != 0
        || memcmp(sym_B, dec_B_serial, PIVCO_BLOCK_SIZE) != 0) {
        fprintf(stderr, "SERIAL DECODE: roundtrip mismatch (baseline bug)\n");
        return 1;
    }

    /* Now fused */
    uint8_t dec_A_fused[PIVCO_BLOCK_SIZE], dec_B_fused[PIVCO_BLOCK_SIZE];
    size_t cAf, cBf;
    decode_fused(enc_A, elen_A, enc_B, elen_B, &table,
                 dec_A_fused, dec_B_fused, &cAf, &cBf);
    if (memcmp(sym_A, dec_A_fused, PIVCO_BLOCK_SIZE) != 0) {
        fprintf(stderr, "FUSED DECODE: A mismatch\n");
        for (int i = 0; i < 32; i++) {
            if (dec_A_fused[i] != sym_A[i])
                fprintf(stderr, "  pos %d: got %d, expected %d\n",
                        i, dec_A_fused[i], sym_A[i]);
        }
        return 2;
    }
    if (memcmp(sym_B, dec_B_fused, PIVCO_BLOCK_SIZE) != 0) {
        fprintf(stderr, "FUSED DECODE: B mismatch\n");
        for (int i = 0; i < 32; i++) {
            if (dec_B_fused[i] != sym_B[i])
                fprintf(stderr, "  pos %d: got %d, expected %d\n",
                        i, dec_B_fused[i], sym_B[i]);
        }
        return 3;
    }
    extern unsigned long g_pivco_fused_calls;
    extern unsigned long g_pivco_fused_chunks;
    extern unsigned long g_pivco_fused_partition_iters;
    g_pivco_fused_calls = g_pivco_fused_chunks = g_pivco_fused_partition_iters = 0;
    decode_fused(enc_A, elen_A, enc_B, elen_B, &table,
                 dec_A_fused, dec_B_fused, &cAf, &cBf);
    printf("correctness PASS; fused diagnostic for one decode_fused call:\n");
    printf("  fused_calls=%lu  fused_chunks=%lu  fused_partition_iters=%lu\n",
           g_pivco_fused_calls, g_pivco_fused_chunks, g_pivco_fused_partition_iters);

    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        decode_serial(enc_A, elen_A, enc_B, elen_B, &table,
                      dec_A_serial, dec_B_serial, &cA, &cB);
    }

    /* Time serial */
    double t0 = now_ns();
    for (int i = 0; i < reps; i++) {
        decode_serial(enc_A, elen_A, enc_B, elen_B, &table,
                      dec_A_serial, dec_B_serial, &cA, &cB);
    }
    double t_serial = now_ns() - t0;

    /* Time fused */
    t0 = now_ns();
    for (int i = 0; i < reps; i++) {
        decode_fused(enc_A, elen_A, enc_B, elen_B, &table,
                     dec_A_fused, dec_B_fused, &cAf, &cBf);
    }
    double t_fused = now_ns() - t0;

    double ns_serial = t_serial / reps;
    double ns_fused  = t_fused  / reps;
    double pct = 100.0 * (ns_serial - ns_fused) / ns_serial;

    printf("dist=%s reps=%d\n", dist, reps);
    printf("  serial:  %.1f ns / 2-block-pair    (%.1f M/s)\n",
           ns_serial, 2.0 * PIVCO_BLOCK_SIZE / ns_serial * 1000.0);
    printf("  fused :  %.1f ns / 2-block-pair    (%.1f M/s)\n",
           ns_fused, 2.0 * PIVCO_BLOCK_SIZE / ns_fused * 1000.0);
    printf("  delta :  %+.2f%% (positive = fused faster)\n", pct);
    return 0;
}
