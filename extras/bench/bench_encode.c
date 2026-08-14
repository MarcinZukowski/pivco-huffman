/* Encode-throughput benchmark.  Mirrors bench_main.c's distribution
 * loop and 5-runs-drop-2 methodology but times the encoders only.
 *
 * Backends timed:
 *   pivco_s_e   pivco_encode_scalar
 *   pivco_e     pivco_encode           (backend-specific)
 *   trad_4s_e   trad_huffman_encode_4s         (huf0-shape comparator)
 *   huf0_x2_e   HUF_compress                   (zstd huff0 4-stream)
 *   huf0_x1_e   HUF_compress1X                 (zstd huff0 1-stream)
 *
 * Output:  ENCODE M/s | pivco_s_e pivco_e | trad_4s_e | huf0_x1_e huf0_x2_e | ratio
 *          where ratio = pivco_e / huf0_x2_e.
 */
#include "pivco_huffman.h"
#include "bench_ctx.h"
#include "mem.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

#define TOTAL_SYMBOLS (4 * 1024 * 1024)
#define DEFAULT_REPEATS 25
#define BLK           PIVCO_BLOCK_SIZE
#define NBLOCKS       (TOTAL_SYMBOLS / BLK)
#define RUNS          5
#define DROP_WORST    2
#define MAX_SPREAD    0.05
#define SEED          0xBEEFCAFE12345678ULL

#define HUF0_CHUNK    (128 * 1024)

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
    int run_all = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0) {
            run_all = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [repeats] [--all]\n"
                   "  repeats   passes over 4M symbols per timed run (default %d)\n"
                   "  --all     run every distribution (default MAIN only)\n",
                   argv[0], DEFAULT_REPEATS);
            return 0;
        } else {
            int r = atoi(argv[i]);
            if (r > 0) repeats = r;
        }
    }
    if (repeats < 1) repeats = 1;

    bench_init();
    int n_dist = bench_num_distributions();
    double freq_before = cpu_freq_check();
    double wall_start = now_sec();

    printf("=== PIVCO-Huffman Encode Benchmarks (PIVCO_MAX_CODE_LEN=%d) ===\n",
           PIVCO_MAX_CODE_LEN);
    printf("Sequence: %dM, Repeats: %d (%dM/run), Block: %d, Runs: %d (drop %d)\n",
           TOTAL_SYMBOLS / (1024*1024), repeats,
           (int)((size_t)TOTAL_SYMBOLS * repeats / (1024*1024)),
           BLK, RUNS, DROP_WORST);
    printf("Distribution set: %s\n\n",
           run_all ? "ALL (29 distributions)"
                   : "MAIN (9 distributions; pass --all for full sweep)");

    printf("%-13s | %8s %8s | %8s | %8s %8s | %7s\n",
           "ENCODE M/s", "pivco_s", "pivco", "trad_4s", "huf0_x1", "huf0_x2", "ratio");
    printf("--------------|-------------------|----------|"
           "-------------------|--------\n");

    /* Per-backend output buffers, sized generously and reused across runs.
     * We don't bother compacting the output between iterations — each
     * block writes into its own pre-assigned slot, so a re-run just
     * overwrites in place.  Lengths are recorded once, before the timed
     * loop, for the sanity-check decode. */
    const size_t pivco_buf_bytes = (size_t)NBLOCKS * PIVCO_MAX_ENCODED_SIZE;
    const size_t trad_buf_bytes  = (size_t)NBLOCKS * (size_t)BLK * 2 + 16;
    const int    huf0_nchunks    = (TOTAL_SYMBOLS + HUF0_CHUNK - 1) / HUF0_CHUNK;
    const size_t huf0_slot       = HUF0_CHUNK + 1024;

    uint8_t *pivco_s_buf = (uint8_t *)malloc(pivco_buf_bytes);
    uint8_t *pivco_buf   = (uint8_t *)malloc(pivco_buf_bytes);
    uint8_t *trad_buf    = (uint8_t *)malloc(trad_buf_bytes);
    uint8_t *huf0_x2_buf = (uint8_t *)malloc((size_t)huf0_nchunks * huf0_slot);
    uint8_t *huf0_x1_buf = (uint8_t *)malloc((size_t)huf0_nchunks * huf0_slot);

    for (int d = 0; d < n_dist; d++) {
        if (!run_all && !bench_dist_is_main(d)) continue;
        const char *name = bench_dist_name(d);
        const uint64_t *freq = bench_dist_freq(d);

        pivco_table_t *table =
            (pivco_table_t *)malloc(sizeof(pivco_table_t));
        if (pivco_build_table(bench_cfg(), freq, table) != PIVCO_OK) {
            printf("%-13s ERROR: build_table failed\n", name);
            free(table);
            continue;
        }

        uint8_t *symbols = (uint8_t *)malloc(TOTAL_SYMBOLS);
        bench_generate_symbols(d, symbols, TOTAL_SYMBOLS, SEED);

        /* Sanity: one untimed encode-decode roundtrip per backend so we
         * don't quote throughput for a broken encoder. */
        {
            uint8_t *enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE);
            uint8_t *dec = (uint8_t *)malloc(BLK);
            size_t len, consumed;
            pivco_encode_scalar(bench_enc_ctx(), table, symbols, BLK, enc, &len);
            pivco_decode_scalar(bench_dec_ctx(), table, enc, len, dec, &consumed);
            if (memcmp(symbols, dec, BLK) != 0) {
                fprintf(stderr, "  %s: pivco_s encode roundtrip FAILED\n", name);
                free(enc); free(dec); goto skip;
            }
            pivco_encode(bench_enc_ctx(), table, symbols, BLK, enc, &len);
            /* Cross-compare against scalar encoder's output. */
            uint8_t *enc_scalar = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE);
            size_t len_scalar;
            pivco_encode_scalar(bench_enc_ctx(), table, symbols, BLK, enc_scalar, &len_scalar);
            if (len != len_scalar) {
                fprintf(stderr, "  %s: enc len mismatch neon=%zu scalar=%zu\n",
                        name, len, len_scalar);
            } else {
                for (size_t b = 0; b < len; b++) {
                    if (enc[b] != enc_scalar[b]) {
                        fprintf(stderr, "  %s: enc byte mismatch at %zu: "
                                "neon=0x%02x scalar=0x%02x\n",
                                name, b, enc[b], enc_scalar[b]);
                        break;
                    }
                }
            }
            free(enc_scalar);
            pivco_decode_scalar(bench_dec_ctx(), table, enc, len, dec, &consumed);
            if (memcmp(symbols, dec, BLK) != 0) {
                fprintf(stderr, "  %s: pivco encode roundtrip FAILED\n", name);
                free(enc); free(dec); goto skip;
            }
            free(enc); free(dec);
        }

        double runs_arr[RUNS];
        char label[64];

        /* Macro: time `repeats` passes over the encoder.  Each run = repeats
         * × 4M = 100M+ symbols, ~100-1000 ms per run depending on encoder. */
        #define BENCH_ENC(var, block, lbl) do { \
            snprintf(label, sizeof(label), "%s/%s", name, lbl); \
            for (int r = 0; r < RUNS; r++) { \
                double t0 = now_sec(); \
                for (int rep = 0; rep < repeats; rep++) { block; } \
                double t1 = now_sec(); \
                runs_arr[r] = (double)TOTAL_SYMBOLS * repeats / (t1 - t0) / 1e6; \
            } \
            var = stable_median(runs_arr, RUNS, DROP_WORST, label); \
        } while(0)

        double e_pivco_s = 0, e_pivco = 0, e_trad_4s = 0;
        double e_huf0_x1 = 0, e_huf0_x2 = 0;

        BENCH_ENC(e_pivco_s, {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t len;
                pivco_encode_scalar(bench_enc_ctx(), table, symbols + (size_t)b * BLK, BLK, pivco_s_buf + (size_t)b * PIVCO_MAX_ENCODED_SIZE, &len);
            }
        }, "pivco_s");

        BENCH_ENC(e_pivco, {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t len;
                pivco_encode(bench_enc_ctx(), table, symbols + (size_t)b * BLK, BLK, pivco_buf + (size_t)b * PIVCO_MAX_ENCODED_SIZE, &len);
            }
        }, "pivco");

        BENCH_ENC(e_trad_4s, {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t len;
                trad_huffman_encode_4s(
                    symbols + (size_t)b * BLK, BLK, table,
                    trad_buf + (size_t)b * BLK * 2, &len);
            }
        }, "trad_4s");

        BENCH_ENC(e_huf0_x2, {
            for (int c = 0; c < huf0_nchunks; c++) {
                size_t chunk_sz = (c < huf0_nchunks - 1) ? HUF0_CHUNK
                                 : TOTAL_SYMBOLS - (size_t)c * HUF0_CHUNK;
                HUF_compress(huf0_x2_buf + (size_t)c * huf0_slot, huf0_slot,
                             symbols + (size_t)c * HUF0_CHUNK, chunk_sz);
            }
        }, "huf0_x2");

        BENCH_ENC(e_huf0_x1, {
            for (int c = 0; c < huf0_nchunks; c++) {
                size_t chunk_sz = (c < huf0_nchunks - 1) ? HUF0_CHUNK
                                 : TOTAL_SYMBOLS - (size_t)c * HUF0_CHUNK;
                HUF_compress1X(huf0_x1_buf + (size_t)c * huf0_slot, huf0_slot,
                               symbols + (size_t)c * HUF0_CHUNK, chunk_sz, 255, 11);
            }
        }, "huf0_x1");

        double best_huf0 = e_huf0_x2 > e_huf0_x1 ? e_huf0_x2 : e_huf0_x1;
        double ratio = best_huf0 > 0 ? e_pivco / best_huf0 : 0;

        printf("%-13s | %8.0f %8.0f | %8.0f | %8.0f %8.0f | %5.2fx\n",
               name, e_pivco_s, e_pivco, e_trad_4s, e_huf0_x1, e_huf0_x2, ratio);

skip:
        free(symbols);
        free(table);
    }

    free(pivco_s_buf); free(pivco_buf); free(trad_buf);
    free(huf0_x1_buf); free(huf0_x2_buf);

    double freq_after = cpu_freq_check();
    double drift = (freq_after - freq_before) / freq_before;

    printf("\n  %d runs of %dM symbols each (%dx %dM), drop %d slowest, warn if spread > %.0f%%\n",
           RUNS, (int)((size_t)TOTAL_SYMBOLS * repeats / (1024*1024)),
           repeats, TOTAL_SYMBOLS / (1024*1024),
           DROP_WORST, MAX_SPREAD * 100);
    printf("  PIVCO/trad encode in %d-symbol blocks; huf0 in 128KB chunks\n", BLK);
    printf("  ratio = pivco / max(huf0_x1, huf0_x2)\n");
    double wall_elapsed = now_sec() - wall_start;
    if (drift < -0.05)
        printf("  WARNING: CPU freq dropped %.1f%% (throttling?)\n", drift * -100);
    else
        printf("  CPU freq drift: %+.1f%% (OK)\n", drift * 100);
    printf("  Total wall time: %.1f seconds\n", wall_elapsed);
    return 0;
}
