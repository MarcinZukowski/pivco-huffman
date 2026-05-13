/* bench_kr.c -- A/B test of "store K_right per internal node" idea.
 *
 * Sidecar (no wire-format change): for each encoded block we
 * precompute K_right values once into a separate buffer.  Then we
 * time the BU decoder reading the precomputed values vs popcounting
 * them on the fly.  Difference = upper bound on the win from
 * inlining K_right into the stream (IDEAS.md 2026-05-12).
 *
 * Dispatches to x86 or NEON BU backend at compile time.
 */
#if defined(PIVCO_HAS_AVX512) || defined(PIVCO_HAS_SSE4)
#define COMPUTE_KR  pivco_huffman_compute_kr_x86
#define DECODE_KR   pivco_huffman_decode_bu_x86_kr
#define DECODE_POP  pivco_huffman_decode_bu_x86
#elif defined(PIVCO_HAS_NEON)
#define COMPUTE_KR  pivco_huffman_compute_kr_neon
#define DECODE_KR   pivco_huffman_decode_bu_neon_kr
#define DECODE_POP  pivco_huffman_decode_bu_neon
#else
#error "bench_kr requires x86 (SSE4+) or NEON backend"
#endif
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include "pivco_huffman.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void           bench_init(void);
extern int            bench_num_distributions(void);
extern const char    *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int            bench_dist_is_main(int idx);
extern void           bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                              int n_symbols, uint64_t seed);

#define BLK PIVCO_BLOCK_SIZE
#define TOTAL_SYMBOLS (4 * 1024 * 1024)
#define ENC_SLOT (2 * PIVCO_BLOCK_SIZE)   /* per-block worst-case encoded size */
#define KR_PER_BLOCK PIVCO_BLOCK_SIZE     /* worst-case # of internal nodes */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    const char *name;
    double bu_baseline_mps;       /* BU decoder w/ popcount */
    double bu_kr_mps;             /* BU decoder w/ precomputed kr */
    double speedup;
    size_t kr_avg_per_block;
    int    ok;
} result_t;

static void bench_dist(int dist_idx, int repeats, result_t *r)
{
    const char *name = bench_dist_name(dist_idx);
    const uint64_t *freq = bench_dist_freq(dist_idx);
    r->name = name; r->ok = 1;

    pivco_huffman_table_t table;
    if (pivco_huffman_build_table(freq, &table) != PIVCO_OK) {
        r->ok = 0; return;
    }

    int blocks = TOTAL_SYMBOLS / BLK;
    uint8_t *symbols = (uint8_t *)malloc(TOTAL_SYMBOLS);
    bench_generate_symbols(dist_idx, symbols, TOTAL_SYMBOLS, 0xFEEDC0DE);

    uint8_t *enc = (uint8_t *)malloc((size_t)blocks * ENC_SLOT);
    size_t *enc_lens = (size_t *)malloc((size_t)blocks * sizeof(size_t));
    uint16_t *kr = (uint16_t *)malloc((size_t)blocks * KR_PER_BLOCK * sizeof(uint16_t));
    size_t *kr_lens = (size_t *)malloc((size_t)blocks * sizeof(size_t));
    uint8_t *dec_buf = (uint8_t *)malloc((size_t)blocks * BLK);

    /* Encode + precompute kr (one-time, untimed). */
    size_t total_kr = 0;
    for (int b = 0; b < blocks; b++) {
        size_t enc_len;
        if (pivco_huffman_encode(symbols + (size_t)b * BLK, &table,
                                 enc + (size_t)b * ENC_SLOT, &enc_len) != PIVCO_OK)
        { r->ok = 0; goto cleanup; }
        enc_lens[b] = enc_len;
        size_t kr_count;
        if (COMPUTE_KR(enc + (size_t)b * ENC_SLOT, enc_len,
                                         &table,
                                         kr + (size_t)b * KR_PER_BLOCK,
                                         KR_PER_BLOCK, &kr_count) != PIVCO_OK)
        { r->ok = 0; goto cleanup; }
        kr_lens[b] = kr_count;
        total_kr += kr_count;
    }
    r->kr_avg_per_block = total_kr / blocks;

    /* Roundtrip sanity check via the kr variant on first block. */
    {
        size_t consumed;
        if (DECODE_KR(enc, enc_lens[0], &table,
                                            dec_buf, &consumed,
                                            kr) != PIVCO_OK ||
            memcmp(symbols, dec_buf, BLK) != 0)
        {
            fprintf(stderr, "  %s: kr decoder roundtrip FAILED\n", name);
            r->ok = 0; goto cleanup;
        }
    }

    double t0, t1;

    /* Baseline: BU decoder with popcount. */
    t0 = now_sec();
    for (int rep = 0; rep < repeats; rep++) {
        for (int b = 0; b < blocks; b++) {
            size_t consumed;
            DECODE_POP(enc + (size_t)b * ENC_SLOT,
                                         enc_lens[b], &table,
                                         dec_buf + (size_t)b * BLK, &consumed);
        }
    }
    t1 = now_sec();
    r->bu_baseline_mps = (double)blocks * BLK * repeats / (t1 - t0) / 1e6;

    /* KR variant. */
    t0 = now_sec();
    for (int rep = 0; rep < repeats; rep++) {
        for (int b = 0; b < blocks; b++) {
            size_t consumed;
            DECODE_KR(enc + (size_t)b * ENC_SLOT,
                                            enc_lens[b], &table,
                                            dec_buf + (size_t)b * BLK, &consumed,
                                            kr + (size_t)b * KR_PER_BLOCK);
        }
    }
    t1 = now_sec();
    r->bu_kr_mps = (double)blocks * BLK * repeats / (t1 - t0) / 1e6;
    r->speedup = r->bu_kr_mps / r->bu_baseline_mps;

cleanup:
    free(symbols); free(enc); free(enc_lens);
    free(kr); free(kr_lens); free(dec_buf);
}

int main(int argc, char **argv)
{
    int repeats = (argc > 1) ? atoi(argv[1]) : 10;
    bench_init();
    int n = bench_num_distributions();

    printf("=== K_right side-buffer experiment ===\n");
    printf("Sequence: %d × %d-byte blocks, repeats=%d\n",
           TOTAL_SYMBOLS / BLK, BLK, repeats);
    printf("Measures BU decoder speedup with K_right precomputed vs popcount\n\n");
    printf("%-13s | bu M/s pop  / kr  | speedup | kr/block\n", "dist");
    printf("--------------+------------------+---------+---------\n");
    for (int i = 0; i < n; i++) {
        if (!bench_dist_is_main(i)) continue;
        result_t r;
        bench_dist(i, repeats, &r);
        if (!r.ok) { printf("%-13s | (skipped)\n", r.name); continue; }
        printf("%-13s | %6.0f / %6.0f | %5.2fx  | %4zu\n",
               r.name, r.bu_baseline_mps, r.bu_kr_mps, r.speedup,
               r.kr_avg_per_block);
    }
    return 0;
}
