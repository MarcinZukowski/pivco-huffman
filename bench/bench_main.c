#include "pivco_huffman.h"
#include "bench_ctx.h"
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
extern int          bench_dist_is_main(int idx);
extern void         bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                           int n_symbols, uint64_t seed);

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
    /* Defaults: MAIN-only distribution set, DEFAULT_REPEATS repeats.
     * argv parsing accepts a `--all` flag in any position and treats the
     * first non-flag argument as the repeat count. */
    int repeats = DEFAULT_REPEATS;
    int run_all = 0;
    int tdbu_only = 0;   /* --tdbu: pivco_n + pivco_bu only, skip comparators */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0) {
            run_all = 1;
        } else if (strcmp(argv[i], "--tdbu") == 0) {
            tdbu_only = 1;
        } else if (strcmp(argv[i], "--no-fse") == 0) {
            bench_cfg()->fse_enabled = (0);
        } else if (strncmp(argv[i], "--effort=", 9) == 0) {
            bench_cfg()->effort = ((pivco_effort_t)atoi(argv[i] + 9));
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [repeats] [--all] [--tdbu] [--no-fse] [--effort=N]\n"
                   "  repeats   passes over 4M symbols per timed run (default %d)\n"
                   "  --all     run every distribution AND every comparator\n"
                   "            (default MAIN: 9 distributions; pivco_s/n/bu,\n"
                   "             trad_4s, huf0_x1/x2 — no trad_1s,\n"
                   "             no huf0_1s)\n"
                   "  --tdbu    skip every comparator (run only pivco_n + pivco_bu);\n"
                   "            keeps the full 5-run methodology.  Use for prof-on/off\n"
                   "            A/B without paying for trad / huf0 timing.\n"
                   "  --no-fse  disable the encoder's FSE dispatch at runtime\n"
                   "            (still v0.2+ wire format; marker stays 0).\n"
                   "  --effort=N  table-build shaping effort (pivco_effort_t):\n"
                   "            0 simplest (plain Huffman), 1 balanced (default),\n"
                   "            2 faster-decompress, 3 fastest-decompress.\n",
                   argv[0], DEFAULT_REPEATS);
            return 0;
        } else {
            int r = atoi(argv[i]);
            if (r > 0) repeats = r;
        }
    }
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

    printf("=== PIVCO-Huffman Benchmarks%s (PIVCO_MAX_CODE_LEN=%d) ===\n",
           quick ? " (QUICK)" : "", PIVCO_MAX_CODE_LEN);
    printf("Sequence: %dM, Repeats: %d (%dM/run), Block: %d, Runs: %d (drop %d)\n",
           TOTAL_SYMBOLS / (1024*1024), repeats,
           (int)((size_t)TOTAL_SYMBOLS * repeats / (1024*1024)),
           BLK, runs, drop_worst);
    printf("Distribution set: %s%s\n\n",
           run_all ? "ALL (29 distributions)"
                   : "MAIN (9 distributions; pass --all for full sweep)",
           tdbu_only ? "  [--tdbu: pivco_n + pivco_bu only]" : "");

    /* Per-distribution compression-size + tree-shape stats.  Filled in
     * inside the per-distribution loop, printed at the end. */
    struct comp_stats_t {
        const char *name;
        double max_code_len;       /* max Huffman code length in this table */
        int    n_leaves;            /* unique symbols */
        int    n_internal_full;     /* PIVCO_NODE_INTERNAL_FULL */
        int    n_internal_flat;     /* PIVCO_NODE_INTERNAL_FLAT */
        int    n_half;              /* PIVCO_NODE_LEAF_LEFT (one-leaf nodes) */
        int    n_both_leaves;       /* PIVCO_NODE_BOTH_LEAVES */
        /* Flat-aware counts: subtract internals/leaves buried inside
         * maximal flat subtrees, since they don't need separate
         * communication in the header (flat root + 2^D leaves does it). */
        int    n_internal_visible;  /* total internals - internals-inside-flats */
        int    n_leaves_visible;    /* total leaves - leaves-inside-flats */
        size_t pivco_bytes;
        size_t trad_bytes;
        size_t trad_4s_bytes;
        size_t huf0_bytes;
        size_t huf0_1s_bytes;
    };
    struct comp_stats_t comp_stats[n_dist];
    memset(comp_stats, 0, sizeof(comp_stats));

    if (quick) {
        printf("%-13s | %7s\n", "DECODE M/s", "pivco_n");
        printf("--------------|--------\n");
    } else if (tdbu_only) {
        printf("%-13s | %7s %7s\n", "DECODE M/s", "pivco_n", "pivco_bu");
        printf("--------------|-----------------\n");
    } else if (run_all) {
        printf("%-13s | %7s %7s %7s | %7s %7s | %7s %7s %7s | %7s\n",
               "DECODE M/s", "pivco_s", "pivco_n", "pivco_bu",
               "trad_1s", "trad_4s",
               "huf0_1s", "huf0_x1", "huf0_x2",
               "ratio");
        printf("--------------|-------------------------|-----------------|------"
               "----------------------|--------\n");
    } else {
        /* MAIN comparator set: drop trad_1s, huf0_1s. */
        printf("%-13s | %7s %7s %7s | %7s | %7s %7s | %7s\n",
               "DECODE M/s", "pivco_s", "pivco_n", "pivco_bu",
               "trad_4s",
               "huf0_x1", "huf0_x2", "ratio");
        printf("--------------|-------------------------|---------|"
               "------------------|--------\n");
    }

    for (int d = 0; d < n_dist; d++) {
        if (!run_all && !bench_dist_is_main(d)) continue;
        const char *name = bench_dist_name(d);
        const uint64_t *freq = bench_dist_freq(d);

        pivco_table_t *table = (pivco_table_t *)malloc(sizeof(pivco_table_t));
        int rc = pivco_build_table(bench_cfg(), freq, table);
        if (rc != PIVCO_OK) {
            printf("%-13s ERROR: build_table returned %d\n", name, rc);
            continue;
        }
        /* trad_huffman_decode* read the 2^L flat table, no longer auto-built */
        pivco_build_traditional_table(table);

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
            pivco_encode_scalar(bench_enc_ctx(), table, symbols + (size_t)b * BLK, BLK, pivco_enc_buf + pivco_enc_off[b], &len);
            pivco_enc_off[b + 1] = pivco_enc_off[b] + len;
        }

#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_SSE4) || defined(PIVCO_HAS_AVX512) || defined(PIVCO_HAS_SVE)
        uint8_t *neon_enc_buf = (uint8_t *)malloc((size_t)NBLOCKS * PIVCO_MAX_ENCODED_SIZE);
        size_t  *neon_enc_off = (size_t *)malloc((size_t)(NBLOCKS + 1) * sizeof(size_t));
        neon_enc_off[0] = 0;
        for (int b = 0; b < NBLOCKS; b++) {
            size_t len;
            pivco_encode(bench_enc_ctx(), table, symbols + (size_t)b * BLK, BLK, neon_enc_buf + neon_enc_off[b], &len);
            neon_enc_off[b + 1] = neon_enc_off[b] + len;
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

        /* ---- Verify correctness (first block / chunk only) ---- */
        {
            uint8_t *dec = (uint8_t *)malloc(TOTAL_SYMBOLS);
            size_t consumed;

            /* PIVCO scalar — first block */
            rc = pivco_decode_scalar(bench_dec_ctx(), table, pivco_enc_buf, pivco_enc_off[1], dec, &consumed);
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

        double p_dec_s = 0, p_dec_n = 0, p_dec_pfx = 0, p_dec_bu = 0;
        double t_dec_1s = 0, t_dec_4s = 0;
        double h_dec_1s = 0, h_dec_4s = 0, h_dec_x2 = 0;

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
            pivco_decode_scalar(bench_dec_ctx(), table, pivco_enc_buf + pivco_enc_off[b], pivco_enc_off[b+1] - pivco_enc_off[b], dec_buf + (size_t)b * BLK, &consumed);
        }
        uint64_t expected_cksum = fnv1a(dec_buf, TOTAL_SYMBOLS);

#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_SSE4) || defined(PIVCO_HAS_AVX512) || defined(PIVCO_HAS_SVE)
        BENCH(p_dec_n, {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t consumed;
                pivco_decode(bench_dec_ctx(), table, neon_enc_buf + neon_enc_off[b], neon_enc_off[b+1] - neon_enc_off[b], dec_buf + (size_t)b * BLK, &consumed);
            }
        }, "pivco_n");

        /* Bottom-up merge decoder.  Same encoded stream as the
         * top-down decoder; routed per-arch.  See pivco_bu_*.c. */
        BENCH(p_dec_bu, {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t consumed;
#if defined(PIVCO_HAS_NEON)
                pivco_decode_bu_neon(bench_dec_ctx(), table, neon_enc_buf + neon_enc_off[b], neon_enc_off[b+1] - neon_enc_off[b], dec_buf + (size_t)b * BLK, &consumed);
#elif defined(PIVCO_HAS_AVX512)
                /* AVX-512 hosts: codec_avx512 (Phase 5 backend, 2026-05-14).
                 * Distinct symbol from pivco_decode_bu_x86 (= codec_x86,
                 * SSE/AVX2 only); the runtime dispatcher in pivco_huffman.c
                 * picks this entry on AVX-512 hosts too. */
                pivco_decode_bu_avx512(bench_dec_ctx(), table, neon_enc_buf + neon_enc_off[b], neon_enc_off[b+1] - neon_enc_off[b], dec_buf + (size_t)b * BLK, &consumed);
#elif defined(PIVCO_HAS_SSE4)
                pivco_decode_bu_x86(bench_dec_ctx(), table, neon_enc_buf + neon_enc_off[b], neon_enc_off[b+1] - neon_enc_off[b], dec_buf + (size_t)b * BLK, &consumed);
#endif
            }
        }, "pivco_bu");
#endif

      if (!quick && !tdbu_only) {
        /* PIVCO scalar: decode NBLOCKS blocks */
        BENCH(p_dec_s, {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t consumed;
                pivco_decode_scalar(bench_dec_ctx(), table, pivco_enc_buf + pivco_enc_off[b], pivco_enc_off[b+1] - pivco_enc_off[b], dec_buf + (size_t)b * BLK, &consumed);
            }
        }, "pivco_s");

        /* Trad 1-stream: decode blocks (ALL-only) */
        if (run_all) {
            BENCH(t_dec_1s, {
                for (int b = 0; b < trad_nblocks; b++) {
                    trad_huffman_decode(trad_enc + trad_enc_off[b],
                                        trad_enc_bits_arr[b], table,
                                        dec_buf + (size_t)b * TRAD_BLK, TRAD_BLK);
                }
            }, "trad_1s");
        }

        /* Trad 4-stream: decode blocks */
        BENCH(t_dec_4s, {
            for (int b = 0; b < trad_nblocks; b++) {
                trad_huffman_decode_4s(trad_4s_enc + trad_4s_off[b],
                                       trad_4s_off[b+1] - trad_4s_off[b], table,
                                       dec_buf + (size_t)b * TRAD_BLK, TRAD_BLK);
            }
        }, "trad_4s");

        /* huf0 1-stream: decode chunks (ALL-only) */
        if (huf0_1s_ok && run_all) {
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

      } /* !quick */
#undef BENCH

        double p_best = p_dec_n > p_dec_s ? p_dec_n : p_dec_s;
        if (p_dec_pfx > p_best) p_best = p_dec_pfx;
        if (p_dec_bu  > p_best) p_best = p_dec_bu;
        double t_best = h_dec_4s;
        if (h_dec_x2 > t_best)  t_best = h_dec_x2;
        if (t_dec_4s > t_best) t_best = t_dec_4s;
        if (t_dec_1s > t_best) t_best = t_dec_1s;
        if (h_dec_1s > t_best) t_best = h_dec_1s;
        double ratio = t_best > 0 ? p_best / t_best : 0;

        if (quick) {
            (void)ratio;
            printf("%-13s | %7.0f\n", name, p_dec_n);
        } else if (tdbu_only) {
            (void)ratio;
            printf("%-13s | %7.0f %7.0f\n", name, p_dec_n, p_dec_bu);
        } else if (run_all) {
            printf("%-13s | %7.0f %7.0f %7.0f %7.0f | %7.0f %7.0f | %7.0f %7.0f %7.0f | %5.2fx\n",
                   name, p_dec_s, p_dec_n, p_dec_bu, p_dec_pfx, t_dec_1s, t_dec_4s,
                   h_dec_1s, h_dec_4s, h_dec_x2, ratio);
        } else {
            /* MAIN: pivco_s, pivco_n, pivco_bu | trad_4s | huf0_x1, huf0_x2 | ratio */
            printf("%-13s | %7.0f %7.0f %7.0f | %7.0f | %7.0f %7.0f | %5.2fx\n",
                   name, p_dec_s, p_dec_n, p_dec_bu,
                   t_dec_4s,
                   h_dec_4s, h_dec_x2, ratio);
        }

        /* Record compression-size + tree-shape stats for the post-table. */
        comp_stats[d].name = name;
        comp_stats[d].max_code_len = (double)table->max_len;
        comp_stats[d].n_leaves = (int)table->num_symbols;
        for (int16_t i = 0; i < table->tree_node_count; i++) {
            switch ((pivco_node_type_t)table->node_type[i]) {
                case PIVCO_NODE_INTERNAL_FULL: comp_stats[d].n_internal_full++; break;
                case PIVCO_NODE_INTERNAL_FLAT: comp_stats[d].n_internal_flat++; break;
                case PIVCO_NODE_LEAF_LEFT:     comp_stats[d].n_half++; break;
                case PIVCO_NODE_BOTH_LEAVES:   comp_stats[d].n_both_leaves++; break;
                default: break; /* LEAF / SKIP */
            }
        }
        /* Flat-aware visible counts: subtract internals & leaves that
         * live INSIDE flat subtrees (the flat root + 2^D leaves
         * suffices to communicate them in a header). */
        {
            int internals_inside_flat = 0;
            int leaves_inside_flat    = 0;
            for (int16_t i = 0; i < table->tree_node_count; i++) {
                if (table->node_type[i] == (uint8_t)PIVCO_NODE_INTERNAL_FLAT) {
                    int D = table->flat_depth[i];
                    /* A flat subtree of depth D has 2^D - 1 internal
                     * nodes below the flat root and 2^D leaves below it. */
                    int n_subtree_internals = (1 << D) - 1;
                    int n_subtree_leaves    = 1 << D;
                    internals_inside_flat += n_subtree_internals;
                    leaves_inside_flat    += n_subtree_leaves;
                }
            }
            int total_internals = comp_stats[d].n_internal_full
                                + comp_stats[d].n_internal_flat
                                + comp_stats[d].n_half
                                + comp_stats[d].n_both_leaves;
            comp_stats[d].n_internal_visible = total_internals - internals_inside_flat;
            comp_stats[d].n_leaves_visible   = comp_stats[d].n_leaves - leaves_inside_flat;
        }
        comp_stats[d].pivco_bytes   = pivco_enc_off[NBLOCKS];
        comp_stats[d].trad_bytes    = trad_enc_off[trad_nblocks];
        comp_stats[d].trad_4s_bytes = trad_4s_off[trad_nblocks];
        comp_stats[d].huf0_bytes    = huf0_ok ? huf0_enc_off[huf0_nchunks] : 0;
        comp_stats[d].huf0_1s_bytes = huf0_1s_ok ? huf0_1s_off[huf0_nchunks] : 0;

cleanup:
        free(dec_buf);
        free(table);
        free(symbols); free(pivco_enc_buf); free(pivco_enc_off);
        free(trad_enc); free(trad_enc_off); free(trad_enc_bits_arr);
        free(trad_4s_enc); free(trad_4s_off);
        free(huf0_enc); free(huf0_enc_off);
        free(huf0_1s_enc); free(huf0_1s_off);
#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_SSE4) || defined(PIVCO_HAS_AVX512) || defined(PIVCO_HAS_SVE)
        free(neon_enc_buf); free(neon_enc_off);
#endif
    }

    /* ---- Compression-size + tree-shape table ---------
     * pivco_raw    = raw partition-bitmap stream (no table header).
     * pivco_hdr    = estimated overhead at ~7 bits per internal tree
     *                node + ~9 bits per leaf (sym + type tag), per block.
     *                Total = (n_internal*7 + n_leaves*9) * NBLOCKS / 8.
     * pivco_total  = pivco_raw + pivco_hdr.
     * "ratio_*"    = compressed / original bytes (lower = better).
     * trad/huf0 are codec-native (include their own headers). */
    if (!quick) {
        const size_t orig = (size_t)TOTAL_SYMBOLS;
        printf("\n=== Compression sizes (bytes for 4M input) ===\n");
        printf("%-13s | %5s %4s %4s %4s %4s %4s | %4s %4s | %10s %10s | %10s %10s %10s\n",
               "DIST",
               "Dmax", "Lvs", "Ful", "Flt", "Hal", "B2L",
               "vIN", "vLv",
               "pivco_raw", "+hdr_est",
               "trad_4s", "huf0_1s", "huf0_x2");
        printf("--------------|-----------------------------|"
               "-----------|"
               "----------------------|"
               "------------------------------------\n");
        for (int d = 0; d < n_dist; d++) {
            struct comp_stats_t *s = &comp_stats[d];
            if (!s->name) continue;
            /* Flat-aware: count only internals/leaves NOT buried inside
             * a flat subtree (the flat root + 2^D leaves communicates
             * the rest implicitly).  ~7 bits per visible internal node
             * (shape + flat-tag) + ~9 bits per visible leaf (sym byte
             * + type tag). */
            size_t hdr_bits_per_block =
                (size_t)(s->n_internal_visible * 7 + s->n_leaves_visible * 9);
            size_t hdr_total = hdr_bits_per_block * NBLOCKS / 8;
            size_t pivco_total = s->pivco_bytes + hdr_total;
            printf("%-13s | %5.0f %4d %4d %4d %4d %4d | %4d %4d | %10zu %10zu | %10zu %10zu %10zu\n",
                   s->name,
                   s->max_code_len,
                   s->n_leaves,
                   s->n_internal_full,
                   s->n_internal_flat,
                   s->n_half,
                   s->n_both_leaves,
                   s->n_internal_visible,
                   s->n_leaves_visible,
                   s->pivco_bytes,
                   pivco_total,
                   s->trad_4s_bytes,
                   s->huf0_1s_bytes,
                   s->huf0_bytes);
            (void)orig;
        }
        printf("  Dmax=max Huffman code length; Lvs=# leaves; "
               "Ful/Flt/Hal/B2L=internal-node-type counts\n");
        printf("  vIN/vLv = flat-aware VISIBLE internals/leaves "
               "(excludes nodes buried inside flat subtrees)\n");
        printf("  +hdr_est = pivco_raw + (vIN*7 + vLv*9) * NBLOCKS/8 "
               "bits per-block table-encoding overhead\n");
        printf("  Lower = better.  Original size = %zu bytes (4MB).\n", orig);
    }

    double freq_after = cpu_freq_check();
    double drift = (freq_after - freq_before) / freq_before;

    printf("\n  %d runs of %dM symbols each (%dx %dM), drop %d slowest, warn if spread > %.0f%%\n",
           runs, (int)((size_t)TOTAL_SYMBOLS * repeats / (1024*1024)),
           repeats, TOTAL_SYMBOLS / (1024*1024),
           drop_worst, MAX_SPREAD * 100);
    printf("  PIVCO/trad decode in %d-symbol blocks\n", BLK);
    printf("  huf0 uses 128KB chunks (its max block size)\n");
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
