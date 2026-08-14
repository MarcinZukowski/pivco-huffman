/* 4-way comparison bench: ph (pivco-huffman, FSE off) vs phe
 * (pivco-huffman + FSE on) vs huff0 (zstd's) vs fse (Yann's TANS).
 *
 * Runs on the MAIN bench distributions (same data source as
 * pivco_bench), measures compression ratio + comp/decomp
 * throughput for each codec.  Outputs a single clean table.
 *
 * Methodology mirrors pivco_bench: 4M symbols per
 * distribution, multi-run timing, drop slowest/fastest, report
 * median.  All codecs see the same input bytes.
 *
 * FSE settings are whatever pivco-huffman is currently built with
 * (MIN_THRESHOLD, MIN_RATIO, MIN_BITMAP_BYTES). */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#include "fse.h"
#include "huf.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

extern void  bench_init(void);
extern int   bench_num_distributions(void);
extern const char *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int   bench_dist_is_main(int idx);
extern void  bench_generate_symbols(int dist_idx, uint8_t *symbols, int n, uint64_t seed);

#define TOTAL_SYMBOLS  (4 * 1024 * 1024)
#define BLK            8192
#define CHUNK_HUF      (128 * 1024)
#define CHUNK_FSE      (128 * 1024)
#define N_RUNS         5

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Median across the inner three of N_RUNS = 5 measurements. */
static double med3of5(double v[N_RUNS])
{
    double s[N_RUNS]; memcpy(s, v, sizeof(s));
    qsort(s, N_RUNS, sizeof(double), cmp_double);
    return s[N_RUNS / 2];
}

typedef struct {
    size_t enc_bytes;
    double enc_mbs;
    double dec_mbs;
    int    ok;
} codec_result_t;

/* Encode/decode using pivco-huffman.  fse_enable = 0 or 1.  Output
 * format matches pivco_bench (per-block records). */
static codec_result_t bench_pivco(const uint8_t *symbols, int total,
                                   int fse_enable,
                                   pivco_table_t *table_in)
{
    codec_result_t r = {0};
    bench_cfg()->fse_enabled = (fse_enable);

    int nblocks = total / BLK;
    uint8_t *enc = (uint8_t *)malloc((size_t)nblocks * (BLK * 2 + 64));
    size_t  *off = (size_t *)calloc((size_t)(nblocks + 1), sizeof(size_t));
    if (!enc || !off) { free(enc); free(off); return r; }

    /* Encode all blocks (warm + measure) */
    double enc_times[N_RUNS];
    for (int run = 0; run < N_RUNS; run++) {
        size_t cur = 0;
        double t0 = now_sec();
        for (int b = 0; b < nblocks; b++) {
            size_t out_len = 0;
            int rc = pivco_encode(bench_enc_ctx(), table_in, symbols + (size_t)b * BLK, BLK, enc + cur, &out_len);
            if (rc != PIVCO_OK) { free(enc); free(off); return r; }
            off[b + 1] = cur + out_len;
            cur += out_len;
        }
        double t1 = now_sec();
        enc_times[run] = t1 - t0;
        for (int b = 1; b <= nblocks; b++) off[b] = off[b];  /* preserved */
        r.enc_bytes = cur;
    }
    r.enc_mbs = (double)total / 1e6 / med3of5(enc_times);

    /* Decode all blocks */
    uint8_t *dec = (uint8_t *)malloc((size_t)total);
    double dec_times[N_RUNS];
    for (int run = 0; run < N_RUNS; run++) {
        double t0 = now_sec();
        for (int b = 0; b < nblocks; b++) {
            size_t consumed = 0;
            size_t enc_len = off[b + 1] - off[b];
            int rc = pivco_decode(bench_dec_ctx(), table_in, enc + off[b], enc_len, dec + (size_t)b * BLK,
                                           &consumed);
            if (rc != PIVCO_OK) { free(enc); free(off); free(dec); return r; }
        }
        double t1 = now_sec();
        dec_times[run] = t1 - t0;
    }
    r.dec_mbs = (double)total / 1e6 / med3of5(dec_times);

    if (memcmp(symbols, dec, total) == 0) r.ok = 1;

    free(enc); free(off); free(dec);
    return r;
}

/* huf0 (zstd's huffman), chunked at 128KB. */
static codec_result_t bench_huf0(const uint8_t *symbols, int total)
{
    codec_result_t r = {0};
    int nchunks = (total + CHUNK_HUF - 1) / CHUNK_HUF;
    uint8_t *enc = (uint8_t *)malloc((size_t)nchunks * (CHUNK_HUF + 1024));
    size_t  *off = (size_t *)calloc((size_t)(nchunks + 1), sizeof(size_t));
    if (!enc || !off) { free(enc); free(off); return r; }

    double enc_times[N_RUNS];
    for (int run = 0; run < N_RUNS; run++) {
        size_t cur = 0;
        double t0 = now_sec();
        int ok = 1;
        for (int c = 0; c < nchunks && ok; c++) {
            size_t chunk_sz = (c < nchunks - 1) ? CHUNK_HUF
                             : (size_t)total - (size_t)c * CHUNK_HUF;
            size_t cr = HUF_compress(enc + cur, chunk_sz + 1024,
                                      symbols + (size_t)c * CHUNK_HUF, chunk_sz);
            if (HUF_isError(cr) || cr == 0) { ok = 0; break; }
            off[c + 1] = cur + cr; cur += cr;
        }
        if (!ok) { free(enc); free(off); return r; }
        double t1 = now_sec();
        enc_times[run] = t1 - t0;
        r.enc_bytes = cur;
    }
    r.enc_mbs = (double)total / 1e6 / med3of5(enc_times);

    uint8_t *dec = (uint8_t *)malloc((size_t)total);
    double dec_times[N_RUNS];
    for (int run = 0; run < N_RUNS; run++) {
        double t0 = now_sec();
        for (int c = 0; c < nchunks; c++) {
            size_t chunk_sz = (c < nchunks - 1) ? CHUNK_HUF
                             : (size_t)total - (size_t)c * CHUNK_HUF;
            (void)HUF_decompress(dec + (size_t)c * CHUNK_HUF, chunk_sz,
                                  enc + off[c], off[c + 1] - off[c]);
        }
        double t1 = now_sec();
        dec_times[run] = t1 - t0;
    }
    r.dec_mbs = (double)total / 1e6 / med3of5(dec_times);

    if (memcmp(symbols, dec, total) == 0) r.ok = 1;
    free(enc); free(off); free(dec);
    return r;
}

/* FSE (Yann's TANS), chunked at 128KB. */
static codec_result_t bench_fse(const uint8_t *symbols, int total)
{
    codec_result_t r = {0};
    int nchunks = (total + CHUNK_FSE - 1) / CHUNK_FSE;
    size_t bound_per_chunk = FSE_compressBound(CHUNK_FSE);
    uint8_t *enc = (uint8_t *)malloc((size_t)nchunks * bound_per_chunk);
    size_t  *off = (size_t *)calloc((size_t)(nchunks + 1), sizeof(size_t));
    /* Some chunks may end up incompressible -- store flag + raw copy. */
    uint8_t *raw_flag = (uint8_t *)calloc((size_t)nchunks, 1);
    if (!enc || !off || !raw_flag) { free(enc); free(off); free(raw_flag); return r; }

    double enc_times[N_RUNS];
    for (int run = 0; run < N_RUNS; run++) {
        size_t cur = 0;
        double t0 = now_sec();
        int ok = 1;
        for (int c = 0; c < nchunks && ok; c++) {
            size_t chunk_sz = (c < nchunks - 1) ? CHUNK_FSE
                             : (size_t)total - (size_t)c * CHUNK_FSE;
            size_t cr = FSE_compress(enc + cur, bound_per_chunk,
                                      symbols + (size_t)c * CHUNK_FSE, chunk_sz);
            if (FSE_isError(cr)) { ok = 0; break; }
            if (cr == 0 || cr == 1) {
                /* Incompressible / RLE.  Fall back to raw copy. */
                memcpy(enc + cur, symbols + (size_t)c * CHUNK_FSE, chunk_sz);
                raw_flag[c] = 1;
                cr = chunk_sz;
            } else {
                raw_flag[c] = 0;
            }
            off[c + 1] = cur + cr; cur += cr;
        }
        if (!ok) { free(enc); free(off); free(raw_flag); return r; }
        double t1 = now_sec();
        enc_times[run] = t1 - t0;
        r.enc_bytes = cur;
    }
    r.enc_mbs = (double)total / 1e6 / med3of5(enc_times);

    uint8_t *dec = (uint8_t *)malloc((size_t)total);
    double dec_times[N_RUNS];
    for (int run = 0; run < N_RUNS; run++) {
        double t0 = now_sec();
        for (int c = 0; c < nchunks; c++) {
            size_t chunk_sz = (c < nchunks - 1) ? CHUNK_FSE
                             : (size_t)total - (size_t)c * CHUNK_FSE;
            size_t enc_len = off[c + 1] - off[c];
            if (raw_flag[c]) {
                memcpy(dec + (size_t)c * CHUNK_FSE, enc + off[c], chunk_sz);
            } else {
                (void)FSE_decompress(dec + (size_t)c * CHUNK_FSE, chunk_sz,
                                      enc + off[c], enc_len);
            }
        }
        double t1 = now_sec();
        dec_times[run] = t1 - t0;
    }
    r.dec_mbs = (double)total / 1e6 / med3of5(dec_times);

    if (memcmp(symbols, dec, total) == 0) r.ok = 1;
    free(enc); free(off); free(raw_flag); free(dec);
    return r;
}

static int run_file_mode(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); fprintf(stderr, "empty/bad file\n"); return 1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 1; }

    uint8_t *bytes = (uint8_t *)malloc((size_t)sz);
    if (!bytes) { fclose(f); fprintf(stderr, "OOM\n"); return 1; }
    if (fread(bytes, 1, (size_t)sz, f) != (size_t)sz) {
        free(bytes); fclose(f); fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);

    /* Truncate to BLK boundary so pivco's per-block bench compares apples-to-apples
     * with huf0/fse (pivco only processes nblocks*BLK; the tail would otherwise be
     * uninitialized in dec[] and the memcmp would fail). */
    long sz_used = (sz / BLK) * BLK;
    if (sz_used == 0) {
        fprintf(stderr, "file %s too small (<%d bytes)\n", path, BLK);
        free(bytes); return 1;
    }
    if (sz_used != sz) {
        fprintf(stderr, "note: truncating %s from %ld to %ld bytes (BLK boundary)\n",
                path, sz, sz_used);
    }
    sz = sz_used;

    uint64_t freq[256] = {0};
    for (long i = 0; i < sz; i++) freq[bytes[i]]++;

    pivco_table_t table;
    if (pivco_build_table(bench_cfg(), freq, &table) != PIVCO_OK) {
        fprintf(stderr, "build_table failed\n");
        free(bytes); return 1;
    }

    /* For small files, replicate the bytes so each measurement sees >= ~4 MB
     * of work — keeps timings out of the per-pass noise floor and matches the
     * 4M-symbol scale of the synthetic bench. */
    const long TARGET_BYTES = 4L * 1024 * 1024;
    long reps = (TARGET_BYTES + sz - 1) / sz;
    if (reps < 1) reps = 1;
    long rep_total = sz * reps;
    uint8_t *rep_bytes = (uint8_t *)malloc((size_t)rep_total);
    if (!rep_bytes) { free(bytes); return 1; }
    for (long r = 0; r < reps; r++) memcpy(rep_bytes + r * sz, bytes, (size_t)sz);

    codec_result_t ph  = bench_pivco(rep_bytes, (int)rep_total, 0, &table);
    codec_result_t phe = bench_pivco(rep_bytes, (int)rep_total, 1, &table);
    codec_result_t hu  = bench_huf0(rep_bytes, (int)rep_total);
    codec_result_t fs  = bench_fse(rep_bytes, (int)rep_total);

    /* Report sizes per original file (divide by reps). */
    ph.enc_bytes  /= (size_t)reps;
    phe.enc_bytes /= (size_t)reps;
    hu.enc_bytes  /= (size_t)reps;
    fs.enc_bytes  /= (size_t)reps;
    free(rep_bytes);

    char fse_sz[16];
    if (fs.ok) snprintf(fse_sz, sizeof(fse_sz), "%9zu", fs.enc_bytes);
    else       snprintf(fse_sz, sizeof(fse_sz), "    FAIL ");

    /* Show ratio (compressed/raw %) so it's easy to compare to MAIN sweep. */
    double ratio_ph  = 100.0 * (double)ph.enc_bytes  / (double)sz;
    double ratio_phe = 100.0 * (double)phe.enc_bytes / (double)sz;
    double ratio_hu  = 100.0 * (double)hu.enc_bytes  / (double)sz;
    double ratio_fs  = fs.ok ? 100.0 * (double)fs.enc_bytes / (double)sz : 0.0;

    printf("# file: %s  (%ld bytes)\n", path, sz);
    printf("%-15s | %9s %9s %9s %9s | %7s %7s %7s %7s | %7s %7s %7s %7s\n",
           "", "ph", "phe", "huf0", "fse",
           "ph", "phe", "huf0", "fse",
           "ph", "phe", "huf0", "fse");
    printf("size (bytes)    | %9zu %9zu %9zu %s | %7.1f%% %6.1f%% %6.1f%% %6.1f%% | %s\n",
           ph.enc_bytes, phe.enc_bytes, hu.enc_bytes, fse_sz,
           ratio_ph, ratio_phe, ratio_hu, ratio_fs,
           "(ratios above; M/s below)");
    printf("encode M/s      |%32s| %7.0f %7.0f %7.0f %7.0f | %7.0f %7.0f %7.0f %7.0f\n",
           "",
           ph.enc_mbs, phe.enc_mbs, hu.enc_mbs, fs.enc_mbs,
           ph.dec_mbs, phe.dec_mbs, hu.dec_mbs, fs.dec_mbs);

    if (!ph.ok || !phe.ok || !hu.ok || (!fs.ok && fs.enc_bytes > 0)) {
        fprintf(stderr, "WARN: roundtrip mismatch (ph=%d phe=%d huf0=%d fse=%d)\n",
                ph.ok, phe.ok, hu.ok, fs.ok);
    }
    free(bytes);
    return 0;
}

int main(int argc, char **argv)
{
    bench_init();

    /* --file PATH [PATH ...]: run on real file bytes instead of synthetic. */
    if (argc >= 3 && strcmp(argv[1], "--file") == 0) {
        int rc = 0;
        for (int a = 2; a < argc; a++) {
            rc |= run_file_mode(argv[a]);
            printf("\n");
        }
        return rc;
    }

    int n = bench_num_distributions();
    int total = TOTAL_SYMBOLS;

    uint8_t *symbols = (uint8_t *)malloc((size_t)total);
    if (!symbols) { fprintf(stderr, "OOM\n"); return 1; }

    pivco_table_t table;

    printf("# 4-way codec comparison (4M symbols/distribution, %d runs each)\n", N_RUNS);
    printf("# ph  = pivco-huffman, FSE OFF        phe = pivco-huffman, FSE ON (current settings)\n");
    printf("# huf0 = zstd Huffman (HUF_compress)  fse = Yann's TANS (FSE_compress)\n");
    printf("\n");
    printf("%-15s | %-25s | %-31s | %-31s\n",
           "", "compressed size (bytes)", "ENCODE M/s", "DECODE M/s");
    printf("%-15s | %7s %7s %7s %4s | %7s %7s %7s %7s | %7s %7s %7s %7s\n",
           "distribution", "ph", "phe", "huf0", "fse",
           "ph", "phe", "huf0", "fse",
           "ph", "phe", "huf0", "fse");
    printf("%-15s-+-%-25s-+-%-31s-+-%-31s\n",
           "---------------",
           "-------------------------",
           "-------------------------------",
           "-------------------------------");

    for (int i = 0; i < n; i++) {
        if (!bench_dist_is_main(i)) continue;
        const char *name = bench_dist_name(i);
        const uint64_t *freq = bench_dist_freq(i);

        /* Generate the same 4M-symbol sequence as pivco_bench. */
        bench_generate_symbols(i, symbols, total, 0xdeadbeef0bULL);

        /* Build the pivco table from the real distribution. */
        if (pivco_build_table(bench_cfg(), freq, &table) != PIVCO_OK) {
            fprintf(stderr, "build_table failed on %s\n", name);
            continue;
        }

        codec_result_t ph  = bench_pivco(symbols, total, 0, &table);
        codec_result_t phe = bench_pivco(symbols, total, 1, &table);
        codec_result_t hu  = bench_huf0(symbols, total);
        codec_result_t fs  = bench_fse(symbols, total);

        char fse_sz[16];
        if (fs.ok) snprintf(fse_sz, sizeof(fse_sz), "%7zu", fs.enc_bytes);
        else       snprintf(fse_sz, sizeof(fse_sz), "  FAIL ");

        printf("%-15s | %7zu %7zu %7zu %s | %7.0f %7.0f %7.0f %7.0f | %7.0f %7.0f %7.0f %7.0f\n",
               name,
               ph.enc_bytes, phe.enc_bytes, hu.enc_bytes, fse_sz,
               ph.enc_mbs, phe.enc_mbs, hu.enc_mbs, fs.enc_mbs,
               ph.dec_mbs, phe.dec_mbs, hu.dec_mbs, fs.dec_mbs);

        if (!ph.ok || !phe.ok || !hu.ok || (!fs.ok && fs.enc_bytes > 0)) {
            fprintf(stderr, "WARN %s: roundtrip mismatch (ph=%d phe=%d huf0=%d fse=%d)\n",
                    name, ph.ok, phe.ok, hu.ok, fs.ok);
        }
    }

    free(symbols);
    return 0;
}
