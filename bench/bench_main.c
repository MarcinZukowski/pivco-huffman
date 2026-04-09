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

/* From bench_rans.cpp — ryg's alias rANS */
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

#define ITERATIONS  100000
#define N           PIVCO_BLOCK_SIZE
#define SEED        0xBEEFCAFE12345678ULL

/* Throughput helper */
#define THROUGHPUT(iters, elapsed) \
    ((double)((size_t)N * (iters)) / (elapsed) / 1e6)

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    bench_init();
    int n_dist = bench_num_distributions();

    printf("=== PIVCO-Huffman Benchmarks ===\n");
    printf("Block size: %d symbols, Iterations: %d\n\n", N, ITERATIONS);

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

        uint8_t symbols[N];
        bench_generate_symbols(d, symbols, N, SEED);

        /* ---- Pre-encode: PIVCO ---- */
        uint8_t pivco_enc[PIVCO_MAX_ENCODED_SIZE];
        size_t pivco_enc_len;
        pivco_huffman_encode_scalar(symbols, &table, pivco_enc, &pivco_enc_len);

#ifdef PIVCO_HAS_NEON
        uint8_t neon_enc[PIVCO_MAX_ENCODED_SIZE];
        size_t neon_enc_len;
        pivco_huffman_encode_neon(symbols, &table, neon_enc, &neon_enc_len);
#endif

        /* ---- Pre-encode: trad single-stream ---- */
        uint8_t trad_enc[N * 4 + 8];
        size_t trad_enc_len, trad_enc_bits;
        trad_huffman_encode(symbols, N, &table, trad_enc, &trad_enc_len, &trad_enc_bits);
        memset(trad_enc + trad_enc_len, 0, 8);

        /* ---- Pre-encode: trad 4-stream ---- */
        uint8_t trad_4s_enc[N * 4 + 16];
        size_t trad_4s_enc_len;
        trad_huffman_encode_4s(symbols, N, &table, trad_4s_enc, &trad_4s_enc_len);

        /* ---- Pre-encode: huff0 ---- */
        uint8_t huf0_enc[N * 2 + 1024]; /* huff0 output includes table header */
        size_t huf0_enc_len = HUF_compress(huf0_enc, sizeof(huf0_enc), symbols, N);
        int huf0_ok = !HUF_isError(huf0_enc_len) && huf0_enc_len > 0;

        /* huff0 1-stream: use compress1X which includes table header */
        uint8_t huf0_1s_enc[N * 2 + 1024];
        size_t huf0_1s_enc_len = HUF_compress1X(huf0_1s_enc, sizeof(huf0_1s_enc),
                                                 symbols, N, 255, 11);
        int huf0_1s_ok = !HUF_isError(huf0_1s_enc_len) && huf0_1s_enc_len > 0;

        /* ---- Pre-encode: rANS alias ---- */
        void *rans_ctx = rans_alias_create(freq);
        uint8_t rans_enc[N * 4];
        size_t rans_enc_len = rans_alias_encode(rans_ctx, symbols, N,
                                                 rans_enc, sizeof(rans_enc));
        uint8_t rans_x2_enc[N * 4];
        size_t rans_x2_enc_len = rans_alias_encode_x2(rans_ctx, symbols, N,
                                                       rans_x2_enc, sizeof(rans_x2_enc));

        /* ---- Verify correctness ---- */
        {
            uint8_t dec[N];
            size_t consumed;

            /* PIVCO scalar */
            rc = pivco_huffman_decode_scalar(pivco_enc, pivco_enc_len, &table, dec, &consumed);
            if (rc != PIVCO_OK || memcmp(symbols, dec, N) != 0) {
                printf("%-13s ERROR: pivco scalar roundtrip failed\n", name);
                rans_alias_destroy(rans_ctx);
                continue;
            }
            /* Trad 4-stream */
            rc = trad_huffman_decode_4s(trad_4s_enc, trad_4s_enc_len, &table, dec, N);
            if (rc != PIVCO_OK || memcmp(symbols, dec, N) != 0) {
                printf("%-13s ERROR: trad 4-stream roundtrip failed\n", name);
                rans_alias_destroy(rans_ctx);
                continue;
            }
            /* huff0 */
            if (huf0_ok) {
                size_t dr = HUF_decompress(dec, N, huf0_enc, huf0_enc_len);
                if (HUF_isError(dr) || memcmp(symbols, dec, N) != 0) {
                    printf("%-13s ERROR: huf0 roundtrip failed\n", name);
                    huf0_ok = 0;
                }
            }
            /* rANS alias */
            rans_alias_decode(rans_ctx, rans_enc, rans_enc_len, dec, N);
            if (memcmp(symbols, dec, N) != 0) {
                printf("%-13s ERROR: rANS alias roundtrip failed\n", name);
                rans_alias_destroy(rans_ctx);
                continue;
            }
            rans_alias_decode_x2(rans_ctx, rans_x2_enc, rans_x2_enc_len, dec, N);
            if (memcmp(symbols, dec, N) != 0) {
                printf("%-13s ERROR: rANS alias x2 roundtrip failed\n", name);
                rans_alias_destroy(rans_ctx);
                continue;
            }
        }

        double t0, t1;
        uint8_t dec_buf[N];
        size_t consumed;

        /* ---- PIVCO decode scalar ---- */
        t0 = now_sec();
        for (int iter = 0; iter < ITERATIONS; iter++)
            pivco_huffman_decode_scalar(pivco_enc, pivco_enc_len, &table, dec_buf, &consumed);
        t1 = now_sec();
        double p_dec_s = THROUGHPUT(ITERATIONS, t1 - t0);

        /* ---- PIVCO decode neon ---- */
        double p_dec_n = 0;
#ifdef PIVCO_HAS_NEON
        t0 = now_sec();
        for (int iter = 0; iter < ITERATIONS; iter++)
            pivco_huffman_decode_neon(neon_enc, neon_enc_len, &table, dec_buf, &consumed);
        t1 = now_sec();
        p_dec_n = THROUGHPUT(ITERATIONS, t1 - t0);
#endif

        /* ---- Trad decode 1-stream (15-bit table) ---- */
        t0 = now_sec();
        for (int iter = 0; iter < ITERATIONS; iter++)
            trad_huffman_decode(trad_enc, trad_enc_bits, &table, dec_buf, N);
        t1 = now_sec();
        double t_dec_1s = THROUGHPUT(ITERATIONS, t1 - t0);

        /* ---- Trad decode 4-stream ---- */
        t0 = now_sec();
        for (int iter = 0; iter < ITERATIONS; iter++)
            trad_huffman_decode_4s(trad_4s_enc, trad_4s_enc_len, &table, dec_buf, N);
        t1 = now_sec();
        double t_dec_4s = THROUGHPUT(ITERATIONS, t1 - t0);

        /* ---- huff0 1-stream decode ---- */
        double h_dec_1s = 0;
        if (huf0_1s_ok) {
            HUF_DTable dtable[HUF_DTABLE_SIZE(HUF_TABLELOG_MAX)];
            /* Read table from encoded stream */
            size_t hdr_size = HUF_readDTableX1(dtable, huf0_1s_enc, huf0_1s_enc_len);
            if (!HUF_isError(hdr_size)) {
                const uint8_t *body = huf0_1s_enc + hdr_size;
                size_t body_len = huf0_1s_enc_len - hdr_size;

                t0 = now_sec();
                for (int iter = 0; iter < ITERATIONS; iter++)
                    HUF_decompress1X_usingDTable(dec_buf, N, body, body_len, dtable);
                t1 = now_sec();
                h_dec_1s = THROUGHPUT(ITERATIONS, t1 - t0);
            }
        }

        /* ---- huff0 4-stream decode ---- */
        double h_dec_4s = 0;
        if (huf0_ok) {
            /* HUF_decompress includes table reading, but for benchmarking
               we want to separate table build from decode.
               Use the DTable-based API. */
            HUF_DTable dtable[HUF_DTABLE_SIZE(HUF_TABLELOG_MAX)];
            size_t hdr_size = HUF_readDTableX1(dtable, huf0_enc, huf0_enc_len);
            if (!HUF_isError(hdr_size)) {
                const uint8_t *body = huf0_enc + hdr_size;
                size_t body_len = huf0_enc_len - hdr_size;

                t0 = now_sec();
                for (int iter = 0; iter < ITERATIONS; iter++)
                    HUF_decompress4X_usingDTable(dec_buf, N, body, body_len, dtable);
                t1 = now_sec();
                h_dec_4s = THROUGHPUT(ITERATIONS, t1 - t0);
            }
        }

        /* ---- rANS alias 1-stream decode ---- */
        double r_dec_1 = 0;
        t0 = now_sec();
        for (int iter = 0; iter < ITERATIONS; iter++)
            rans_alias_decode(rans_ctx, rans_enc, rans_enc_len, dec_buf, N);
        t1 = now_sec();
        r_dec_1 = THROUGHPUT(ITERATIONS, t1 - t0);

        /* ---- rANS alias 2-stream decode ---- */
        double r_dec_2 = 0;
        t0 = now_sec();
        for (int iter = 0; iter < ITERATIONS; iter++)
            rans_alias_decode_x2(rans_ctx, rans_x2_enc, rans_x2_enc_len, dec_buf, N);
        t1 = now_sec();
        r_dec_2 = THROUGHPUT(ITERATIONS, t1 - t0);

        rans_alias_destroy(rans_ctx);

        /* Best PIVCO decode */
        double p_best = p_dec_n > p_dec_s ? p_dec_n : p_dec_s;
        /* Best traditional/other */
        double t_best = h_dec_4s;
        if (t_dec_4s > t_best) t_best = t_dec_4s;
        if (t_dec_1s > t_best) t_best = t_dec_1s;
        if (h_dec_1s > t_best) t_best = h_dec_1s;
        if (r_dec_1 > t_best) t_best = r_dec_1;
        if (r_dec_2 > t_best) t_best = r_dec_2;

        double ratio = t_best > 0 ? p_best / t_best : 0;

        printf("%-13s | %7.0f %7.0f | %7.0f %7.0f | %7.0f %7.0f | %7.0f %7.0f | %5.2fx\n",
               name,
               p_dec_s, p_dec_n,
               t_dec_1s, t_dec_4s,
               h_dec_1s, h_dec_4s,
               r_dec_1, r_dec_2,
               ratio);
    }

    printf("\n  All values: decode throughput in M/s (millions of symbols/sec)\n");
    printf("  pivco_s/n = PIVCO scalar/NEON, trad_1s/4s = our trad impl\n");
    printf("  huf0_1s/4s = actual huff0 (cyan4973/FiniteStateEntropy)\n");
    printf("  rans_1/2 = ryg_rans alias method (rygorous/ryg_rans), 1/2-stream\n");
    printf("  ratio = best_pivco / best_other\n");

    return 0;
}
