#include "pivco_huffman.h"
#include "mem.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* From bench_distributions.c */
extern void         bench_init(void);
extern int          bench_num_distributions(void);
extern const char  *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern void         bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                           int n_symbols, uint64_t seed);

/* From bench_rans.cpp */
extern void  *rans_alias_create(const uint64_t *freq256);
extern void   rans_alias_destroy(void *ctx);
extern size_t rans_alias_encode(void *ctx, const uint8_t *symbols, size_t n,
                                uint8_t *out, size_t out_cap);
extern size_t rans_alias_decode(void *ctx, const uint8_t *in, size_t in_len,
                                uint8_t *symbols, size_t n);
extern size_t rans_alias_encode_x2(void *ctx, const uint8_t *symbols, size_t n,
                                   uint8_t *out, size_t out_cap);
extern size_t rans_alias_decode_x2(void *ctx, const uint8_t *in, size_t in_len,
                                   uint8_t *symbols, size_t n);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- Configuration ---- */
#define TOTAL_SYMBOLS (4 * 1024 * 1024)  /* 4M symbol sequence */
#define DEFAULT_REPEATS 25               /* passes over 4M per timed run */
#define BLK           PIVCO_BLOCK_SIZE   /* our block size */
#define NBLOCKS       (TOTAL_SYMBOLS / BLK)
#define RUNS          5
#define DROP_WORST    2
#define MAX_SPREAD    0.05
#define SEED          0xBEEFCAFE12345678ULL

static int dbl_cmp_desc(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) - (da > db);
}

static double stable_median(double *results, int runs, int drop_worst,
                             const char *label)
{
    qsort(results, runs, sizeof(double), dbl_cmp_desc);
    int kept = runs - drop_worst;
    double best = results[0], worst_kept = results[kept - 1];
    double spread = best > 0 ? (best - worst_kept) / best : 0;
    if (spread > MAX_SPREAD && label)
        fprintf(stderr, "  WARNING: %s spread %.1f%% (%.0f..%.0f)\n",
                label, spread * 100, worst_kept, best);
    return results[kept / 2];
}

/* Simple FNV-1a checksum over buffer */
static uint64_t fnv1a(const uint8_t *data, size_t len)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++)
        h = (h ^ data[i]) * 0x100000001b3ULL;
    return h;
}

static double cpu_freq_check(void)
{
    volatile uint64_t x = 0;
    double t0 = now_sec();
    for (int i = 0; i < 100000000; i++) x += (uint64_t)i;
    double t1 = now_sec();
    return 100.0 / (t1 - t0);
}

int main(int argc, char **argv)
{
    int repeats = DEFAULT_REPEATS;
    if (argc > 1) repeats = atoi(argv[1]);
    if (repeats < 1) repeats = 1;

    /* PIVCO_BENCH_QUICK: skip every comparator (run only pivco_n), reduce
       runs to 2 (no drop).  ~5-10x faster wall, used for iteration; the
       documented sweeps still use the full 5-runs-drop-2 methodology by
       leaving the env var unset. */
    int quick = getenv("PIVCO_BENCH_QUICK") != NULL;
    int runs        = quick ? 2 : RUNS;
    int drop_worst  = quick ? 0 : DROP_WORST;

    bench_init();
    int n_dist = bench_num_distributions();
    double freq_before = cpu_freq_check();
    double wall_start = now_sec();
    int g_cksum_errors = 0;

    printf("=== PIVCO-Huffman Benchmarks%s ===\n", quick ? " (QUICK)" : "");
    printf("Sequence: %dM, Repeats: %d (%dM/run), Block: %d, Runs: %d (drop %d)\n\n",
           TOTAL_SYMBOLS / (1024*1024), repeats,
           (int)((size_t)TOTAL_SYMBOLS * repeats / (1024*1024)),
           BLK, runs, drop_worst);

    if (quick) {
        printf("%-13s | %7s\n", "DECODE M/s", "pivco_n");
        printf("--------------|--------\n");
    } else {
        printf("%-13s | %7s %7s %7s | %7s %7s | %7s %7s %7s | %7s | %7s\n",
               "DECODE M/s", "pivco_s", "pivco_n", "pivco_p",
               "trad_1s", "trad_4s",
               "huf0_1s", "huf0_x1", "huf0_x2",
               "rans_x2", "ratio");
        printf("--------------|-------------------------|-----------------|------"
               "----------------------|---------|--------\n");
    }

    for (int d = 0; d < n_dist; d++) {
        const char *name = bench_dist_name(d);
        const uint64_t *freq = bench_dist_freq(d);

        pivco_huffman_table_t *table = (pivco_huffman_table_t *)malloc(sizeof(pivco_huffman_table_t));
        int rc = pivco_huffman_build_table(freq, table);
        if (rc != PIVCO_OK) {
            printf("%-13s ERROR: build_table returned %d\n", name, rc);
            continue;
        }

        /* Generate full 4M symbol sequence */
        uint8_t *symbols = (uint8_t *)malloc(TOTAL_SYMBOLS);
        bench_generate_symbols(d, symbols, TOTAL_SYMBOLS, SEED);

        /* ---- Pre-encode: PIVCO (NBLOCKS × BLK) ---- */
        /* Each block's encoded data is variable-size; store offsets */
        uint8_t *pivco_enc_buf = (uint8_t *)malloc((size_t)NBLOCKS * PIVCO_MAX_ENCODED_SIZE);
        size_t  *pivco_enc_off = (size_t *)malloc((size_t)(NBLOCKS + 1) * sizeof(size_t));
        pivco_enc_off[0] = 0;
        for (int b = 0; b < NBLOCKS; b++) {
            size_t len;
            pivco_huffman_encode_scalar(symbols + (size_t)b * BLK, table,
                                        pivco_enc_buf + pivco_enc_off[b], &len);
            pivco_enc_off[b + 1] = pivco_enc_off[b] + len;
        }

#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_SSE4) || defined(PIVCO_HAS_AVX512) || defined(PIVCO_HAS_SVE)
        uint8_t *neon_enc_buf = (uint8_t *)malloc((size_t)NBLOCKS * PIVCO_MAX_ENCODED_SIZE);
        size_t  *neon_enc_off = (size_t *)malloc((size_t)(NBLOCKS + 1) * sizeof(size_t));
        neon_enc_off[0] = 0;
        for (int b = 0; b < NBLOCKS; b++) {
            size_t len;
            pivco_huffman_encode(symbols + (size_t)b * BLK, table,
                                      neon_enc_buf + neon_enc_off[b], &len);
            neon_enc_off[b + 1] = neon_enc_off[b] + len;
        }
#endif

#ifdef PIVCO_HAS_NEON
        /* Prefix backend: runs on any table with min_len in [1, 8]. */
        int pfx_applicable = (table->min_len >= 1 && table->min_len <= 8);
        uint8_t *pfx_enc_buf = NULL;
        size_t  *pfx_enc_off = NULL;
        if (pfx_applicable) {
            pfx_enc_buf = (uint8_t *)malloc((size_t)NBLOCKS * PIVCO_MAX_ENCODED_SIZE);
            pfx_enc_off = (size_t *)malloc((size_t)(NBLOCKS + 1) * sizeof(size_t));
            pfx_enc_off[0] = 0;
            for (int b = 0; b < NBLOCKS; b++) {
                size_t len;
                int rc = pivco_huffman_encode_neon_prefix(
                    symbols + (size_t)b * BLK, table,
                    pfx_enc_buf + pfx_enc_off[b], &len);
                if (rc != PIVCO_OK) { pfx_applicable = 0; break; }
                pfx_enc_off[b + 1] = pfx_enc_off[b] + len;
            }
        }
#endif


        /* ---- Pre-encode: trad 1-stream (chunked at BLK) ---- */
        #define TRAD_BLK BLK  /* trad uses same block size as PIVCO */
        int trad_nblocks = TOTAL_SYMBOLS / TRAD_BLK;
        uint8_t *trad_enc = (uint8_t *)malloc((size_t)trad_nblocks * TRAD_BLK * 2 + 8);
        size_t  *trad_enc_off  = (size_t *)calloc((size_t)(trad_nblocks + 1), sizeof(size_t));
        size_t  *trad_enc_bits_arr = (size_t *)calloc((size_t)trad_nblocks, sizeof(size_t));
        for (int b = 0; b < trad_nblocks; b++) {
            size_t len, bits;
            trad_huffman_encode(symbols + (size_t)b * TRAD_BLK, TRAD_BLK, table,
                                trad_enc + trad_enc_off[b], &len, &bits);
            trad_enc_bits_arr[b] = bits;
            trad_enc_off[b + 1] = trad_enc_off[b] + len;
        }
        memset(trad_enc + trad_enc_off[trad_nblocks], 0, 8);

        /* ---- Pre-encode: trad 4-stream (chunked at BLK) ---- */
        uint8_t *trad_4s_enc = (uint8_t *)malloc((size_t)trad_nblocks * TRAD_BLK * 2 + 16);
        size_t  *trad_4s_off = (size_t *)calloc((size_t)(trad_nblocks + 1), sizeof(size_t));
        for (int b = 0; b < trad_nblocks; b++) {
            size_t len;
            trad_huffman_encode_4s(symbols + (size_t)b * TRAD_BLK, TRAD_BLK, table,
                                   trad_4s_enc + trad_4s_off[b], &len);
            trad_4s_off[b + 1] = trad_4s_off[b] + len;
        }

        /* ---- Pre-encode: huff0 (full 4M, huf0 picks its own blocking) ---- */
        /* HUF_BLOCKSIZE_MAX is 128KB, so huf0 handles one 4M block fine
           since HUF_compress will process it. But actually HUF_compress
           may not accept > 128KB. Let's chunk at 128KB. */
        #define HUF0_CHUNK (128 * 1024)
        int huf0_nchunks = (TOTAL_SYMBOLS + HUF0_CHUNK - 1) / HUF0_CHUNK;
        uint8_t *huf0_enc = (uint8_t *)malloc((size_t)huf0_nchunks * (HUF0_CHUNK + 1024));
        size_t  *huf0_enc_off = (size_t *)calloc((size_t)(huf0_nchunks + 1), sizeof(size_t));
        int huf0_ok = 1;
        for (int c = 0; c < huf0_nchunks && huf0_ok; c++) {
            size_t chunk_sz = (c < huf0_nchunks - 1) ? HUF0_CHUNK
                             : TOTAL_SYMBOLS - (size_t)c * HUF0_CHUNK;
            size_t r = HUF_compress(huf0_enc + huf0_enc_off[c],
                                    chunk_sz + 1024,
                                    symbols + (size_t)c * HUF0_CHUNK,
                                    chunk_sz);
            if (HUF_isError(r) || r == 0) { huf0_ok = 0; break; }
            huf0_enc_off[c + 1] = huf0_enc_off[c] + r;
        }

        /* huf0 1-stream (same chunking) */
        uint8_t *huf0_1s_enc = (uint8_t *)malloc((size_t)huf0_nchunks * (HUF0_CHUNK + 1024));
        size_t  *huf0_1s_off = (size_t *)calloc((size_t)(huf0_nchunks + 1), sizeof(size_t));
        int huf0_1s_ok = 1;
        for (int c = 0; c < huf0_nchunks && huf0_1s_ok; c++) {
            size_t chunk_sz = (c < huf0_nchunks - 1) ? HUF0_CHUNK
                             : TOTAL_SYMBOLS - (size_t)c * HUF0_CHUNK;
            size_t r = HUF_compress1X(huf0_1s_enc + huf0_1s_off[c],
                                       chunk_sz + 1024,
                                       symbols + (size_t)c * HUF0_CHUNK,
                                       chunk_sz, 255, 11);
            if (HUF_isError(r) || r == 0) { huf0_1s_ok = 0; break; }
            huf0_1s_off[c + 1] = huf0_1s_off[c] + r;
        }

        /* ---- Pre-encode: rANS alias x2 (full 4M) ---- */
        void *rans_ctx = rans_alias_create(freq);
        uint8_t *rans_x2_enc = (uint8_t *)malloc(TOTAL_SYMBOLS * 2);
        size_t rans_x2_enc_len = rans_alias_encode_x2(rans_ctx, symbols, TOTAL_SYMBOLS,
                                                       rans_x2_enc, TOTAL_SYMBOLS * 2);

        /* ---- Verify correctness (first block / chunk only) ---- */
        {
            uint8_t *dec = (uint8_t *)malloc(TOTAL_SYMBOLS);
            size_t consumed;

            /* PIVCO scalar — first block */
            rc = pivco_huffman_decode_scalar(pivco_enc_buf, pivco_enc_off[1],
                                             table, dec, &consumed);
            if (rc != PIVCO_OK || memcmp(symbols, dec, BLK) != 0) {
                printf("%-13s ERROR: pivco roundtrip failed\n", name);
                free(dec); goto cleanup;
            }

            /* huf0 4-stream — first chunk */
            if (huf0_ok) {
                size_t dr = HUF_decompress(dec, HUF0_CHUNK,
                                           huf0_enc, huf0_enc_off[1]);
                if (HUF_isError(dr) || memcmp(symbols, dec, HUF0_CHUNK) != 0) {
                    printf("%-13s ERROR: huf0 roundtrip failed\n", name);
                    huf0_ok = 0;
                }
            }

            /* rANS x2 — full sequence */
            rans_alias_decode_x2(rans_ctx, rans_x2_enc, rans_x2_enc_len, dec, TOTAL_SYMBOLS);
            if (memcmp(symbols, dec, TOTAL_SYMBOLS) != 0) {
                printf("%-13s ERROR: rANS x2 roundtrip failed\n", name);
                free(dec); goto cleanup;
            }
            free(dec);
        }

        /* ---- Benchmark ---- */
        uint8_t *dec_buf = (uint8_t *)malloc(TOTAL_SYMBOLS);
        double runs_arr[RUNS];   /* RUNS is the max; we use `runs` of them */
        double t0, t1;
        char label[64];

/* Macro: time repeats passes over the 4M decode.
   Each run = repeats × 4M = 400M symbols, giving ~100ms per run.
   Checksums first and last run to verify consistency, and compares
   against `expected_cksum` (set BEFORE this macro by an untimed scalar
   decode — see the block immediately preceding the first BENCH call).
   Mismatches bump g_cksum_errors so main can exit non-zero. */
#define BENCH(var, block, lbl) do { \
    snprintf(label, sizeof(label), "%s/%s", name, lbl); \
    uint64_t cksum_first = 0, cksum_last = 0; \
    for (int r = 0; r < runs; r++) { \
        t0 = now_sec(); \
        for (int rep = 0; rep < repeats; rep++) { block; } \
        t1 = now_sec(); \
        runs_arr[r] = (double)TOTAL_SYMBOLS * repeats / (t1 - t0) / 1e6; \
        if (r == 0) cksum_first = fnv1a(dec_buf, TOTAL_SYMBOLS); \
        if (r == runs - 1) cksum_last = fnv1a(dec_buf, TOTAL_SYMBOLS); \
    } \
    if (cksum_first != cksum_last) { \
        fprintf(stderr, "  ERROR: %s checksum mismatch between runs!\n", label); \
        g_cksum_errors++; \
    } \
    if (cksum_first != expected_cksum) { \
        fprintf(stderr, "  ERROR: %s checksum differs from reference (scalar)!\n", label); \
        g_cksum_errors++; \
    } \
    var = stable_median(runs_arr, runs, drop_worst, label); \
} while(0)

        double p_dec_s = 0, p_dec_n = 0, p_dec_pfx = 0;
        double t_dec_1s = 0, t_dec_4s = 0;
        double h_dec_1s = 0, h_dec_4s = 0, h_dec_x2 = 0;
        double r_dec_2 = 0;

        /* Establish reference checksum from the scalar decoder (untimed).
         * Computing it here — outside any BENCH call — keeps scalar
         * timing out of the perf table while still cross-validating
         * every backend against scalar instead of against the first
         * BENCH'd decoder.  Pre-fix, pivco_n was the reference, so a
         * bug in pivco_n only flagged its peers as "wrong" — and we
         * had been ignoring those errors as harness flakiness.  See
         * commit 1399cee for the regression that motivated this. */
        for (int b = 0; b < NBLOCKS; b++) {
            size_t consumed;
            pivco_huffman_decode_scalar(
                pivco_enc_buf + pivco_enc_off[b],
                pivco_enc_off[b+1] - pivco_enc_off[b],
                table, dec_buf + (size_t)b * BLK, &consumed);
        }
        uint64_t expected_cksum = fnv1a(dec_buf, TOTAL_SYMBOLS);

#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_SSE4) || defined(PIVCO_HAS_AVX512) || defined(PIVCO_HAS_SVE)
        BENCH(p_dec_n, {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t consumed;
                pivco_huffman_decode(
                    neon_enc_buf + neon_enc_off[b],
                    neon_enc_off[b+1] - neon_enc_off[b],
                    table, dec_buf + (size_t)b * BLK, &consumed);
            }
        }, "pivco_n");
#endif

      if (!quick) {
        /* PIVCO scalar: decode NBLOCKS blocks */
        BENCH(p_dec_s, {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t consumed;
                pivco_huffman_decode_scalar(
                    pivco_enc_buf + pivco_enc_off[b],
                    pivco_enc_off[b+1] - pivco_enc_off[b],
                    table, dec_buf + (size_t)b * BLK, &consumed);
            }
        }, "pivco_s");

#ifdef PIVCO_HAS_NEON
        if (pfx_applicable) {
            BENCH(p_dec_pfx, {
                for (int b = 0; b < NBLOCKS; b++) {
                    size_t consumed;
                    pivco_huffman_decode_neon_prefix(
                        pfx_enc_buf + pfx_enc_off[b],
                        pfx_enc_off[b+1] - pfx_enc_off[b],
                        table, dec_buf + (size_t)b * BLK, &consumed);
                }
            }, "pivco_p");
        }
#endif


        /* Trad 1-stream: decode blocks */
        BENCH(t_dec_1s, {
            for (int b = 0; b < trad_nblocks; b++) {
                trad_huffman_decode(trad_enc + trad_enc_off[b],
                                    trad_enc_bits_arr[b], table,
                                    dec_buf + (size_t)b * TRAD_BLK, TRAD_BLK);
            }
        }, "trad_1s");

        /* Trad 4-stream: decode blocks */
        BENCH(t_dec_4s, {
            for (int b = 0; b < trad_nblocks; b++) {
                trad_huffman_decode_4s(trad_4s_enc + trad_4s_off[b],
                                       trad_4s_off[b+1] - trad_4s_off[b], table,
                                       dec_buf + (size_t)b * TRAD_BLK, TRAD_BLK);
            }
        }, "trad_4s");

        /* huf0 1-stream: decode chunks */
        if (huf0_1s_ok) {
            BENCH(h_dec_1s, {
                for (int c = 0; c < huf0_nchunks; c++) {
                    size_t chunk_sz = (c < huf0_nchunks - 1) ? HUF0_CHUNK
                                     : TOTAL_SYMBOLS - (size_t)c * HUF0_CHUNK;
                    HUF_decompress1X1(dec_buf + (size_t)c * HUF0_CHUNK, chunk_sz,
                                      huf0_1s_enc + huf0_1s_off[c],
                                      huf0_1s_off[c+1] - huf0_1s_off[c]);
                }
            }, "huf0_1s");
        }

        /* huf0 4-stream X1: decode chunks (single-symbol per lookup) */
        if (huf0_ok) {
            BENCH(h_dec_4s, {
                for (int c = 0; c < huf0_nchunks; c++) {
                    size_t chunk_sz = (c < huf0_nchunks - 1) ? HUF0_CHUNK
                                     : TOTAL_SYMBOLS - (size_t)c * HUF0_CHUNK;
                    HUF_decompress4X1(dec_buf + (size_t)c * HUF0_CHUNK, chunk_sz,
                                      huf0_enc + huf0_enc_off[c],
                                      huf0_enc_off[c+1] - huf0_enc_off[c]);
                }
            }, "huf0_4s");
        }

        /* huf0 4-stream X2: decode chunks (double-symbol per lookup) */
        if (huf0_ok) {
            BENCH(h_dec_x2, {
                for (int c = 0; c < huf0_nchunks; c++) {
                    size_t chunk_sz = (c < huf0_nchunks - 1) ? HUF0_CHUNK
                                     : TOTAL_SYMBOLS - (size_t)c * HUF0_CHUNK;
                    HUF_decompress4X2(dec_buf + (size_t)c * HUF0_CHUNK, chunk_sz,
                                      huf0_enc + huf0_enc_off[c],
                                      huf0_enc_off[c+1] - huf0_enc_off[c]);
                }
            }, "huf0_x2");
        }

        /* rANS 2-stream: full 4M at once */
        BENCH(r_dec_2, {
            rans_alias_decode_x2(rans_ctx, rans_x2_enc, rans_x2_enc_len,
                                  dec_buf, TOTAL_SYMBOLS);
        }, "rans_2");
      } /* !quick */
#undef BENCH

        double p_best = p_dec_n > p_dec_s ? p_dec_n : p_dec_s;
        if (p_dec_pfx > p_best) p_best = p_dec_pfx;
        double t_best = h_dec_4s;
        if (h_dec_x2 > t_best)  t_best = h_dec_x2;
        if (t_dec_4s > t_best) t_best = t_dec_4s;
        if (t_dec_1s > t_best) t_best = t_dec_1s;
        if (h_dec_1s > t_best) t_best = h_dec_1s;
        if (r_dec_2 > t_best)  t_best = r_dec_2;
        double ratio = t_best > 0 ? p_best / t_best : 0;

        if (quick) {
            (void)ratio;
            printf("%-13s | %7.0f\n", name, p_dec_n);
        } else {
            printf("%-13s | %7.0f %7.0f %7.0f | %7.0f %7.0f | %7.0f %7.0f %7.0f | %7.0f | %5.2fx\n",
                   name, p_dec_s, p_dec_n, p_dec_pfx, t_dec_1s, t_dec_4s,
                   h_dec_1s, h_dec_4s, h_dec_x2, r_dec_2, ratio);
        }

cleanup:
        free(dec_buf);
        rans_alias_destroy(rans_ctx);
        free(table);
        free(symbols); free(pivco_enc_buf); free(pivco_enc_off);
        free(trad_enc); free(trad_enc_off); free(trad_enc_bits_arr);
        free(trad_4s_enc); free(trad_4s_off);
        free(huf0_enc); free(huf0_enc_off);
        free(huf0_1s_enc); free(huf0_1s_off);
        free(rans_x2_enc);
#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_SSE4) || defined(PIVCO_HAS_AVX512) || defined(PIVCO_HAS_SVE)
        free(neon_enc_buf); free(neon_enc_off);
#endif
#ifdef PIVCO_HAS_NEON
        if (pfx_applicable) { free(pfx_enc_buf); free(pfx_enc_off); }
#endif
    }

    double freq_after = cpu_freq_check();
    double drift = (freq_after - freq_before) / freq_before;

    printf("\n  %d runs of %dM symbols each (%dx %dM), drop %d slowest, warn if spread > %.0f%%\n",
           runs, (int)((size_t)TOTAL_SYMBOLS * repeats / (1024*1024)),
           repeats, TOTAL_SYMBOLS / (1024*1024),
           drop_worst, MAX_SPREAD * 100);
    printf("  PIVCO/trad decode in %d-symbol blocks\n", BLK);
    printf("  huf0 uses 128KB chunks (its max block size)\n");
    printf("  rANS decodes full 4M at once\n");
    double wall_elapsed = now_sec() - wall_start;
    if (drift < -0.05)
        printf("  WARNING: CPU freq dropped %.1f%% (throttling?)\n", drift * -100);
    else
        printf("  CPU freq drift: %+.1f%% (OK)\n", drift * 100);
    printf("  Total wall time: %.1f seconds\n", wall_elapsed);

    if (g_cksum_errors > 0) {
        fprintf(stderr,
                "\nFAIL: %d decoder output(s) disagreed with the scalar "
                "reference.  Bench numbers above are unreliable.\n",
                g_cksum_errors);
        return 1;
    }
    return 0;
}
