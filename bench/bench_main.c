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

#define ITERS       20000
#define RUNS        5
#define DROP_WORST  2
#define MAX_SPREAD  0.05
#define N           PIVCO_BLOCK_SIZE
#define SEED        0xBEEFCAFE12345678ULL

#define THROUGHPUT(iters, elapsed) \
    ((double)((size_t)N * (iters)) / (elapsed) / 1e6)

static int dbl_cmp_desc(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) - (da > db);
}

/* Run a timed block RUNS times, drop DROP_WORST slowest, return median of kept.
   Warn on stderr if kept runs spread > MAX_SPREAD. */
static double stable_median(double *results, const char *label)
{
    qsort(results, RUNS, sizeof(double), dbl_cmp_desc);
    int kept = RUNS - DROP_WORST;
    double best = results[0], worst_kept = results[kept - 1];
    double spread = best > 0 ? (best - worst_kept) / best : 0;
    if (spread > MAX_SPREAD && label)
        fprintf(stderr, "  WARNING: %s spread %.1f%% (%.0f..%.0f)\n",
                label, spread * 100, worst_kept, best);
    return results[kept / 2];
}

/* CPU freq baseline */
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
    (void)argc; (void)argv;

    bench_init();
    int n_dist = bench_num_distributions();
    double freq_before = cpu_freq_check();

    printf("=== PIVCO-Huffman Benchmarks ===\n");
    printf("Block size: %d, Iters: %d, Runs: %d (drop %d slowest)\n\n",
           N, ITERS, RUNS, DROP_WORST);

    printf("%-13s | %7s %7s | %7s %7s | %7s %7s | %7s %7s | %7s\n",
           "DECODE M/s", "pivco_s", "pivco_n",
           "trad_1s", "trad_4s",
           "huf0_1s", "huf0_4s",
           "rans_1", "rans_2", "ratio");
    printf("--------------|-----------------|-----------------|------"
           "-----------|-----------------|--------\n");

    for (int d = 0; d < n_dist; d++) {
        const char *name = bench_dist_name(d);
        const uint64_t *freq = bench_dist_freq(d);

        pivco_huffman_table_t table;
        int rc = pivco_huffman_build_table(freq, &table);
        if (rc != PIVCO_OK) {
            printf("%-13s ERROR: build_table returned %d\n", name, rc);
            continue;
        }

        uint8_t *symbols = (uint8_t *)malloc(N);
        bench_generate_symbols(d, symbols, N, SEED);

        /* ---- Pre-encode all formats ---- */
        uint8_t *pivco_enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE);
        size_t pivco_enc_len;
        pivco_huffman_encode_scalar(symbols, &table, pivco_enc, &pivco_enc_len);

#ifdef PIVCO_HAS_NEON
        uint8_t *neon_enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE);
        size_t neon_enc_len;
        pivco_huffman_encode_neon(symbols, &table, neon_enc, &neon_enc_len);
#endif

        uint8_t *trad_enc = (uint8_t *)malloc(N * 4 + 8);
        size_t trad_enc_len, trad_enc_bits;
        trad_huffman_encode(symbols, N, &table, trad_enc, &trad_enc_len, &trad_enc_bits);
        memset(trad_enc + trad_enc_len, 0, 8);

        uint8_t *trad_4s_enc = (uint8_t *)malloc(N * 4 + 16);
        size_t trad_4s_enc_len;
        trad_huffman_encode_4s(symbols, N, &table, trad_4s_enc, &trad_4s_enc_len);

        uint8_t *huf0_enc = (uint8_t *)malloc(N * 2 + 1024);
        size_t huf0_enc_len = HUF_compress(huf0_enc, N * 2 + 1024, symbols, N);
        int huf0_ok = !HUF_isError(huf0_enc_len) && huf0_enc_len > 0;

        uint8_t *huf0_1s_enc = (uint8_t *)malloc(N * 2 + 1024);
        size_t huf0_1s_enc_len = HUF_compress1X(huf0_1s_enc, N * 2 + 1024,
                                                  symbols, N, 255, 11);
        int huf0_1s_ok = !HUF_isError(huf0_1s_enc_len) && huf0_1s_enc_len > 0;

        void *rans_ctx = rans_alias_create(freq);
        uint8_t *rans_enc = (uint8_t *)malloc(N * 4);
        size_t rans_enc_len = rans_alias_encode(rans_ctx, symbols, N,
                                                 rans_enc, N * 4);
        uint8_t *rans_x2_enc = (uint8_t *)malloc(N * 4);
        size_t rans_x2_enc_len = rans_alias_encode_x2(rans_ctx, symbols, N,
                                                       rans_x2_enc, N * 4);

        /* ---- Verify correctness ---- */
        {
            uint8_t *dec = (uint8_t *)malloc(N);
            size_t consumed;
            rc = pivco_huffman_decode_scalar(pivco_enc, pivco_enc_len, &table, dec, &consumed);
            if (rc != PIVCO_OK || memcmp(symbols, dec, N) != 0) {
                printf("%-13s ERROR: pivco roundtrip failed\n", name);
                free(dec); goto cleanup;
            }
            rc = trad_huffman_decode_4s(trad_4s_enc, trad_4s_enc_len, &table, dec, N);
            if (rc != PIVCO_OK || memcmp(symbols, dec, N) != 0) {
                printf("%-13s ERROR: trad 4s roundtrip failed\n", name);
                free(dec); goto cleanup;
            }
            if (huf0_ok) {
                size_t dr = HUF_decompress(dec, N, huf0_enc, huf0_enc_len);
                if (HUF_isError(dr) || memcmp(symbols, dec, N) != 0) {
                    printf("%-13s ERROR: huf0 roundtrip failed\n", name);
                    huf0_ok = 0;
                }
            }
            rans_alias_decode(rans_ctx, rans_enc, rans_enc_len, dec, N);
            if (memcmp(symbols, dec, N) != 0) {
                printf("%-13s ERROR: rANS roundtrip failed\n", name);
                free(dec); goto cleanup;
            }
            free(dec);
        }

        /* ---- Benchmark with stability ---- */
        uint8_t *dec_buf = (uint8_t *)malloc(N);
        double runs[RUNS];
        double t0, t1;
        size_t consumed;
        char label[64];

        /* Build huf0 DTables once (outside timing) */
        HUF_DTable *dtable_1s = NULL, *dtable_4s = NULL;
        size_t huf0_1s_body_off = 0, huf0_4s_body_off = 0;
        if (huf0_1s_ok) {
            dtable_1s = (HUF_DTable *)malloc(HUF_DTABLE_SIZE(HUF_TABLELOG_MAX) * sizeof(HUF_DTable));
            dtable_1s[0] = (HUF_DTable)((U32)HUF_TABLELOG_MAX * 0x01000001);
            size_t hs = HUF_readDTableX1(dtable_1s, huf0_1s_enc, huf0_1s_enc_len);
            if (HUF_isError(hs)) { huf0_1s_ok = 0; free(dtable_1s); dtable_1s = NULL; }
            else huf0_1s_body_off = hs;
        }
        if (huf0_ok) {
            dtable_4s = (HUF_DTable *)malloc(HUF_DTABLE_SIZE(HUF_TABLELOG_MAX) * sizeof(HUF_DTable));
            dtable_4s[0] = (HUF_DTable)((U32)HUF_TABLELOG_MAX * 0x01000001);
            size_t hs = HUF_readDTableX1(dtable_4s, huf0_enc, huf0_enc_len);
            if (HUF_isError(hs)) { huf0_ok = 0; free(dtable_4s); dtable_4s = NULL; }
            else huf0_4s_body_off = hs;
        }

#define BENCH(var, code, lbl) do { \
    snprintf(label, sizeof(label), "%s/%s", name, lbl); \
    for (int r = 0; r < RUNS; r++) { \
        t0 = now_sec(); \
        for (int iter = 0; iter < ITERS; iter++) { code; } \
        t1 = now_sec(); \
        runs[r] = THROUGHPUT(ITERS, t1 - t0); \
    } \
    var = stable_median(runs, label); \
} while(0)

        double p_dec_s, p_dec_n = 0, t_dec_1s, t_dec_4s;
        double h_dec_1s = 0, h_dec_4s = 0, r_dec_1, r_dec_2;

        BENCH(p_dec_s,
              pivco_huffman_decode_scalar(pivco_enc, pivco_enc_len, &table, dec_buf, &consumed),
              "pivco_s");
#ifdef PIVCO_HAS_NEON
        BENCH(p_dec_n,
              pivco_huffman_decode_neon(neon_enc, neon_enc_len, &table, dec_buf, &consumed),
              "pivco_n");
#endif
        BENCH(t_dec_1s,
              trad_huffman_decode(trad_enc, trad_enc_bits, &table, dec_buf, N),
              "trad_1s");
        BENCH(t_dec_4s,
              trad_huffman_decode_4s(trad_4s_enc, trad_4s_enc_len, &table, dec_buf, N),
              "trad_4s");
        if (huf0_1s_ok) {
            const uint8_t *body = huf0_1s_enc + huf0_1s_body_off;
            size_t blen = huf0_1s_enc_len - huf0_1s_body_off;
            BENCH(h_dec_1s,
                  HUF_decompress1X_usingDTable(dec_buf, N, body, blen, dtable_1s),
                  "huf0_1s");
        }
        if (huf0_ok) {
            const uint8_t *body = huf0_enc + huf0_4s_body_off;
            size_t blen = huf0_enc_len - huf0_4s_body_off;
            BENCH(h_dec_4s,
                  HUF_decompress4X_usingDTable(dec_buf, N, body, blen, dtable_4s),
                  "huf0_4s");
        }
        BENCH(r_dec_1,
              rans_alias_decode(rans_ctx, rans_enc, rans_enc_len, dec_buf, N),
              "rans_1");
        BENCH(r_dec_2,
              rans_alias_decode_x2(rans_ctx, rans_x2_enc, rans_x2_enc_len, dec_buf, N),
              "rans_2");
#undef BENCH

        double p_best = p_dec_n > p_dec_s ? p_dec_n : p_dec_s;
        double t_best = h_dec_4s;
        if (t_dec_4s > t_best) t_best = t_dec_4s;
        if (t_dec_1s > t_best) t_best = t_dec_1s;
        if (h_dec_1s > t_best) t_best = h_dec_1s;
        if (r_dec_1 > t_best)  t_best = r_dec_1;
        if (r_dec_2 > t_best)  t_best = r_dec_2;
        double ratio = t_best > 0 ? p_best / t_best : 0;

        printf("%-13s | %7.0f %7.0f | %7.0f %7.0f | %7.0f %7.0f | %7.0f %7.0f | %5.2fx\n",
               name, p_dec_s, p_dec_n, t_dec_1s, t_dec_4s,
               h_dec_1s, h_dec_4s, r_dec_1, r_dec_2, ratio);

cleanup:
        free(dec_buf);
        free(dtable_1s); free(dtable_4s);
        rans_alias_destroy(rans_ctx);
        free(symbols); free(pivco_enc); free(trad_enc); free(trad_4s_enc);
        free(huf0_enc); free(huf0_1s_enc); free(rans_enc); free(rans_x2_enc);
#ifdef PIVCO_HAS_NEON
        free(neon_enc);
#endif
    }

    double freq_after = cpu_freq_check();
    double drift = (freq_after - freq_before) / freq_before;

    printf("\n  %d runs/measurement, drop %d slowest, warn if spread > %.0f%%\n",
           RUNS, DROP_WORST, MAX_SPREAD * 100);
    printf("  pivco_s/n = PIVCO scalar/NEON, trad_1s/4s = our trad impl\n");
    printf("  huf0_1s/4s = actual huff0, rans_1/2 = ryg_rans alias\n");
    if (drift < -0.05)
        printf("  WARNING: CPU freq dropped %.1f%% (throttling?)\n", drift * -100);
    else
        printf("  CPU freq drift: %+.1f%% (OK)\n", drift * 100);

    return 0;
}
