/* Dump the per-FSE-table-id usage histogram for a file's encode pass.
 *
 * Builds a Huffman table from the file's byte frequencies, encodes the
 * file block-by-block (FSE on), and reports for each of the 25 pre-built
 * pivco FSE tables how many internal-node bitmaps committed to it,
 * total input/output bytes, and effective per-table compression ratio.
 *
 * Usage: pivco_bench_fse_table_use PATH [PATH ...] */

#include "pivco_huffman.h"
#include "bench_ctx.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* From pivco_fse_tables.h (kept in sync by hand; only 25 entries). */
static const float pivco_fse_freq_nom[26] = {
    0.000f, 0.500f, 0.580f, 0.646f, 0.703f, 0.750f, 0.790f, 0.823f,
    0.851f, 0.875f, 0.895f, 0.912f, 0.926f, 0.938f, 0.947f, 0.956f,
    0.963f, 0.969f, 0.974f, 0.978f, 0.981f, 0.984f, 0.987f, 0.989f,
    0.991f, 0.992f,
};

static int run_one(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return 1; }
    long sz_used = (sz / PIVCO_BLOCK_SIZE) * PIVCO_BLOCK_SIZE;
    if (sz_used == 0) { fclose(f); fprintf(stderr, "file too small\n"); return 1; }
    uint8_t *buf = (uint8_t *)malloc((size_t)sz_used);
    if (!buf) { fclose(f); return 1; }
    if (fread(buf, 1, (size_t)sz_used, f) != (size_t)sz_used) {
        free(buf); fclose(f); return 1;
    }
    fclose(f);

    uint64_t freq[256] = {0};
    for (long i = 0; i < sz_used; i++) freq[buf[i]]++;

    pivco_table_t table;
    if (pivco_build_table(bench_cfg(), freq, &table) != PIVCO_OK) {
        fprintf(stderr, "build_table failed\n"); free(buf); return 1;
    }

    bench_cfg()->fse_enabled = (1);
    pivco_fse_stats_reset();

    int nblocks = (int)(sz_used / PIVCO_BLOCK_SIZE);
    uint8_t *enc = (uint8_t *)malloc((size_t)nblocks * (PIVCO_BLOCK_SIZE * 2 + 64));
    if (!enc) { free(buf); return 1; }
    size_t cur = 0, total_enc = 0;
    for (int b = 0; b < nblocks; b++) {
        size_t out_len = 0;
        int rc = pivco_encode(bench_enc_ctx(), &table, buf + (size_t)b * PIVCO_BLOCK_SIZE, PIVCO_BLOCK_SIZE, enc + cur, &out_len);
        if (rc != PIVCO_OK) { fprintf(stderr, "encode %d failed: %d\n", b, rc); break; }
        cur += out_len;
    }
    total_enc = cur;

    uint64_t commit[PIVCO_FSE_STATS_SLOTS];
    uint64_t attempt[PIVCO_FSE_STATS_SLOTS];
    uint64_t bin[PIVCO_FSE_STATS_SLOTS];
    uint64_t bout[PIVCO_FSE_STATS_SLOTS];
    pivco_fse_stats_get(commit, attempt, bin, bout);

    uint64_t commit_total = 0, attempt_total = 0;
    uint64_t bin_total = 0, bout_total = 0;
    for (int i = 0; i < PIVCO_FSE_STATS_SLOTS; i++) {
        commit_total  += commit[i];
        attempt_total += attempt[i];
        bin_total     += bin[i];
        bout_total    += bout[i];
    }

    printf("=== %s (%ld bytes, %d blocks of %d, encoded %zu bytes => %.2f%%) ===\n",
           path, sz_used, nblocks, PIVCO_BLOCK_SIZE,
           total_enc, 100.0 * total_enc / (double)sz_used);
    printf("FSE commit nodes: %llu  rejected: %llu\n",
           (unsigned long long)(commit_total - commit[0]),
           (unsigned long long)commit[0]);
    printf("FSE attempt nodes (table chosen): %llu\n",
           (unsigned long long)attempt_total);
    printf("Total bytes_in via FSE: %llu  bytes_out (incl marker+len): %llu  (%.2f%%)\n",
           (unsigned long long)bin_total, (unsigned long long)bout_total,
           bin_total ? 100.0 * (double)bout_total / (double)bin_total : 0.0);
    printf("\n");
    printf("tid  p_nom  attempt   commit  rej%%   bytes_in  bytes_out   ratio  avg_in\n");
    printf("---  -----  -------  -------  -----  ---------  ---------  ------  ------\n");
    /* Slot 0: aggregate rejected count. */
    if (commit[0] > 0) {
        printf("  0   -      %7llu  %7llu  -      %9s  %9s  %5s   %5s\n",
               (unsigned long long)0, (unsigned long long)commit[0], "-", "-", "-", "-");
    }
    for (int i = 1; i <= 25; i++) {
        if (attempt[i] == 0 && commit[i] == 0) continue;
        double rej_pct = attempt[i] ? 100.0 * (double)(attempt[i] - commit[i]) / (double)attempt[i] : 0.0;
        double ratio = bin[i] ? 100.0 * (double)bout[i] / (double)bin[i] : 0.0;
        double avg_in = commit[i] ? (double)bin[i] / (double)commit[i] : 0.0;
        printf("%3d  %.3f  %7llu  %7llu  %4.1f%%  %9llu  %9llu  %5.1f%%  %6.0f\n",
               i, pivco_fse_freq_nom[i],
               (unsigned long long)attempt[i],
               (unsigned long long)commit[i],
               rej_pct,
               (unsigned long long)bin[i],
               (unsigned long long)bout[i],
               ratio, avg_in);
    }
    printf("\n");

    /* ---------- Root-node-per-block view ---------- */
    int rn = pivco_fse_root_count();
    if (rn > 0) {
        printf("=== Root-node events (one per block, %d total) ===\n", rn);
        int hist[26]      = {0};   /* by committed table_id (slot 0 = no commit) */
        int hist_attempt[26] = {0};
        double pmax = 0, pmin = 1.0, psum = 0;
        int root_in = 0, root_out = 0;
        for (int i = 0; i < rn; i++) {
            pivco_fse_root_event_t e;
            pivco_fse_root_get(i, &e);
            if (e.committed) hist[e.table_id]++;
            else             hist[0]++;
            if (e.table_id) hist_attempt[e.table_id]++;
            if (e.p_major > pmax) pmax = e.p_major;
            if (e.p_major < pmin) pmin = e.p_major;
            psum += e.p_major;
            root_in  += e.nbytes_in;
            root_out += e.nbytes_out;
        }
        printf("p_major across blocks: min=%.3f  max=%.3f  mean=%.3f\n",
               pmin, pmax, psum / rn);
        printf("root bytes_in: %d   root bytes_out: %d   ratio: %.1f%%\n",
               root_in, root_out, 100.0 * root_out / (double)root_in);
        printf("\n");
        printf("Root-node committed-table histogram (blocks per table_id):\n");
        printf("tid  p_nom    blocks  attempt  bar\n");
        printf("---  -----    ------  -------  ----------------------------------------\n");
        /* Show row for slot 0 if any blocks didn't commit. */
        if (hist[0]) {
            printf("  0   no       %3d   (n/a)   ", hist[0]);
            for (int j = 0; j < hist[0] && j < 60; j++) putchar('#');
            printf("\n");
        }
        for (int t = 1; t <= 25; t++) {
            if (hist[t] == 0 && hist_attempt[t] == 0) continue;
            printf("%3d  %.3f    %4d    %4d    ", t, pivco_fse_freq_nom[t],
                   hist[t], hist_attempt[t]);
            for (int j = 0; j < hist[t] && j < 60; j++) putchar('#');
            printf("\n");
        }
        printf("\n");

        /* Per-block timeline: which table did the root pick, block-by-block. */
        printf("Block-by-block root table_id timeline (. = no commit):\n");
        for (int i = 0; i < rn; i++) {
            pivco_fse_root_event_t e;
            pivco_fse_root_get(i, &e);
            if ((i % 32) == 0) printf("[%4d] ", i);
            if (!e.committed) putchar('.');
            else if (e.table_id < 10) putchar('0' + e.table_id);
            else putchar('a' + (e.table_id - 10));   /* 10->a, 25->p */
            if ((i % 32) == 31 || i == rn - 1) putchar('\n');
        }
        printf("\n  (legend: 1-9 = table 1-9, a-p = table 10-25, . = below MIN_THRESHOLD or rejected)\n");
        printf("\n");
    }

    free(buf); free(enc);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s PATH [PATH ...]\n", argv[0]);
        return 1;
    }
    int rc = 0;
    for (int a = 1; a < argc; a++) rc |= run_one(argv[a]);
    return rc;
}
