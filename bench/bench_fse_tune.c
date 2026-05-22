/* bench_fse_tune.c -- sweep the wide-cursor (x) x unroll (y) FSE shapes
 * on real corpus byte distributions (proba80, prose) at ~1 MB, to pick
 * one "tuned FSE" shape to report alongside stock FSE_decompress in the
 * fair-bench.  Reuses the exact x*y codec from fse_xy_codec.h; the only
 * difference vs bench_fse_xy_micro is the input (corpus bytes, table
 * built from the data) instead of synthetic bitmaps + static tables.
 *
 * Buffer size 983040 = 2^16 * 15, divisible by every x in {2,4,6,8,10,
 * 12,16} so all shapes encode without a remainder tail.  Note: only
 * x in {2,4,8,16} divide a 128 KB fair-bench chunk, so a deployable
 * tuned shape should prefer those.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#include "bitstream.h"
#include "fse_xy_codec.h"

extern void         bench_init(void);
extern int          bench_num_distributions(void);
extern const char  *bench_dist_name(int idx);
extern void         bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                           int n_symbols, uint64_t seed);

#define N      983040            /* 0.94 MB, divisible by all x */
#define RUNS   5
#define REPEATS 10
#define SEED   0xBEEFCAFE12345678ULL
#define MAXLOG 12

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static int find_dist(const char *name) {
    for (int i = 0; i < bench_num_distributions(); i++)
        if (!strcmp(bench_dist_name(i), name)) return i;
    return -1;
}

static void sweep_dist(const char *name, uint8_t *src, uint8_t *enc, uint8_t *dec) {
    int d = find_dist(name);
    if (d < 0) { printf("dist %s not found\n", name); return; }
    bench_generate_symbols(d, src, N, SEED);

    unsigned cnt[256]; memset(cnt, 0, sizeof cnt);
    for (size_t i = 0; i < N; i++) cnt[src[i]]++;
    unsigned maxSym = 255; while (maxSym && !cnt[maxSym]) maxSym--;
    unsigned tlog = FSE_optimalTableLog(MAXLOG, N, maxSym);
    short norm[256];
    if (FSE_isError(FSE_normalizeCount(norm, tlog, cnt, N, maxSym))) { printf("%s: normalize fail\n", name); return; }
    /* FSE_decodeSymbolFast (used by the wide-cursor shapes) is unsafe
     * when any symbol has probability > 50% (normalized count past half
     * the table).  Such data (e.g. a single byte at 80%) can only use
     * stock FSE_decompress here. */
    int maxNorm = 0; for (int i = 0; i <= (int)maxSym; i++) if (norm[i] > maxNorm) maxNorm = norm[i];
    int fast_unsafe = (maxNorm > (1 << (tlog - 1)));
    FSE_CTable *ct = FSE_createCTable(maxSym, tlog);
    FSE_DTable *dt = FSE_createDTable(tlog);
    FSE_buildCTable(ct, norm, maxSym, tlog);
    FSE_buildDTable(dt, norm, maxSym, tlog);

    printf("== %-12s ==  N=%d  tableLog=%u  maxSym=%u\n", name, N, tlog, maxSym);
    printf("  shape   enc MB/s   dec MB/s   ratio\n");

    /* stock single-stream FSE for reference */
    size_t slen = FSE_compress(enc, N + N/2, src, N);
    if (!FSE_isError(slen) && slen > 1) {
        double best;
        best = 0; for (int r=0;r<RUNS;r++){double t0=now_ns();
            for(int i=0;i<REPEATS;i++) FSE_compress(enc, N+N/2, src, N);
            double mb=1000.0*(double)N*REPEATS/(now_ns()-t0); if(mb>best)best=mb;} double senc=best;
        best = 0; for (int r=0;r<RUNS;r++){double t0=now_ns();
            for(int i=0;i<REPEATS;i++) FSE_decompress(dec, N, enc, slen);
            double mb=1000.0*(double)N*REPEATS/(now_ns()-t0); if(mb>best)best=mb;} double sdec=best;
        printf("  %-7s %8.0f   %8.0f   %5.2f\n", "stock", senc, sdec, (double)N/slen);
    }

    if (fast_unsafe) {
        printf("  (wide-cursor shapes skipped: P(max symbol) > 50%%, FSE fast-decode unsafe)\n\n");
        FSE_freeCTable(ct); FSE_freeDTable(dt);
        return;
    }

    double best_dec = 0; const char *best_name = "?";
    for (size_t c = 0; c < N_CFGS; c++) {
        size_t elen = encode_x(cfgs[c].x, src, N, enc, N + N/2, ct);
        if (elen == 0) { printf("  %-7s   (enc fail)\n", cfgs[c].name); continue; }
        memset(dec, 0xCC, N);
        size_t dl = cfgs[c].fn(enc, elen, dec, N, dt);
        if (dl != N || memcmp(src, dec, N) != 0) { printf("  %-7s   (roundtrip FAIL)\n", cfgs[c].name); continue; }

        double best;
        best = 0; for (int r=0;r<RUNS;r++){double t0=now_ns();
            for(int i=0;i<REPEATS;i++) encode_x(cfgs[c].x, src, N, enc, N+N/2, ct);
            double mb=1000.0*(double)N*REPEATS/(now_ns()-t0); if(mb>best)best=mb;} double eenc=best;
        best = 0; for (int r=0;r<RUNS;r++){double t0=now_ns();
            for(int i=0;i<REPEATS;i++) cfgs[c].fn(enc, elen, dec, N, dt);
            double mb=1000.0*(double)N*REPEATS/(now_ns()-t0); if(mb>best)best=mb;} double edec=best;
        printf("  %-7s %8.0f   %8.0f   %5.2f\n", cfgs[c].name, eenc, edec, (double)N/elen);
        if (edec > best_dec) { best_dec = edec; best_name = cfgs[c].name; }
    }
    printf("  -> best dec shape: %s (%.0f MB/s)\n\n", best_name, best_dec);
    FSE_freeCTable(ct); FSE_freeDTable(dt);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    bench_init();
    printf("FSE shape tuning sweep: best of %dx%d over %d bytes\n\n", RUNS, REPEATS, N);
    uint8_t *src = malloc(N), *enc = malloc(N + N/2), *dec = malloc(N);
    sweep_dist("proba80", src, enc, dec);
    sweep_dist("english", src, enc, dec);
    sweep_dist("prose_pride", src, enc, dec);
    free(src); free(enc); free(dec);
    return 0;
}
