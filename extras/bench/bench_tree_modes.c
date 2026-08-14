/* bench_tree_modes.c -- A/B/C/D ablation across tree-shape modes.
 *
 * Runs ph decode at the 4 tree modes (OPTIMIZED, CANONICAL_FLAT, FUSED, NAIVE)
 * on a fixed dataset, reporting decode MB/s + per-engine ratio + flat-subtree
 * stats.  Both encoder and decoder use the same mode -- this is a build-time
 * knob, not a wire-format change.
 *
 * FSE is force-disabled across all runs: the comparison is pure
 * tree-shape vs tree-shape.  Compression ratio differs slightly between
 * NAIVE / FUSED (fewer flat-region byte-padding savings) and the
 * CANONICAL_FLAT / OPTIMIZED modes (which can pack 2^D symbols into a
 * single N*D-bit packed region).
 *
 * Usage:
 *   pivco_bench_tree_modes [REPEATS [DIST_NAME]]
 *
 * Compiled as part of extras/bench/CMakeLists.txt.
 */
#include "pivco_huffman.h"
#include "bench_ctx.h"
#include "huf.h"
#ifdef PIVCO_HAS_OODLE
#  include "bench_oodle_wrapper.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

extern void        bench_init(void);
extern int         bench_num_distributions(void);
extern const char *bench_dist_name(int idx);
extern int         bench_dist_is_main(int idx);
extern void        bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                          int n_symbols, unsigned seed);

#define BUF_BYTES   (1 << 20)        /* 1 MB per measurement */
#define BLK         8192
#define RUNS        10               /* outer best-of-N, matches bench_fair */
#define REPEATS     10               /* inner pass count per run, matches bench_fair */
#define DEFAULT_REPEATS 20
#define SEED        0xC0FFEEu

static double now_ns_local(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static const char *MODE_NAMES[] = {
    "OPTIMIZED", "NAIVE", "FUSED", "CANONICAL_FLAT",
};

static void measure_mode(pivco_tree_mode_t mode, const uint8_t *sym, size_t n,
                          double *dec_mbs_out, double *ratio_out,
                          int *flat_internal_nodes_out)
{
    bench_cfg()->tree_mode = (mode);

    pivco_table_t *table = malloc(sizeof(*table));
    assert(table);

    /* Build table from per-byte frequencies. */
    uint64_t freq[PIVCO_MAX_SYMBOLS];
    memset(freq, 0, sizeof(freq));
    for (size_t i = 0; i < n; i++) freq[sym[i]]++;
    int rc = pivco_build_table(bench_cfg(), freq, table);
    if (rc != PIVCO_OK) { fprintf(stderr, "build_table failed: %d\n", rc); free(table); return; }

    /* Encode the whole buffer in 8K blocks, capture compressed bytes. */
    size_t cap = n + 65536;
    uint8_t *enc = malloc(cap);
    uint8_t *dec = malloc(n);
    assert(enc && dec);

    size_t total_enc = 0;
    size_t blocks = n / BLK;
    for (size_t b = 0; b < blocks; b++) {
        size_t got;
        int r = pivco_encode(bench_enc_ctx(), table, sym + b * BLK, BLK, enc + total_enc, &got);
        if (r != PIVCO_OK) { fprintf(stderr, "encode err blk %zu: %d\n", b, r); goto out; }
        total_enc += got;
    }
    *ratio_out = (double)(blocks * BLK) / (double)total_enc;

    /* Total symbols absorbed by flat-D>=2 subtrees -- a better proxy for
       "how much of the decode work is fast-pathed" than the bare node count. */
    int flat_syms_total = 0;
    for (int i = 0; i < table->tree_node_count; i++) {
        if (table->flat_depth[i] >= 2)
            flat_syms_total += (1 << table->flat_depth[i]);
    }
    *flat_internal_nodes_out = flat_syms_total;

    /* Correctness check ONCE before the timing loop. */
    {
        const uint8_t *p = enc;
        for (size_t b = 0; b < blocks; b++) {
            size_t consumed;
            int rc2 = pivco_decode(bench_dec_ctx(), table, p, total_enc - (size_t)(p - enc), dec + b * BLK, &consumed);
            if (rc2 != PIVCO_OK) { fprintf(stderr, "decode err blk %zu: %d\n", b, rc2); goto out; }
            p += consumed;
        }
        if (memcmp(sym, dec, blocks * BLK) != 0) {
            fprintf(stderr, "DECODE MISMATCH for mode %s\n", MODE_NAMES[mode]);
            goto out;
        }
    }

    /* Timing: best-of-RUNS, each run = REPEATS passes over the buffer.
       Matches bench_fair.c's BEST_MBPS macro. */
    double best = 0;
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_ns_local();
        for (int rep = 0; rep < REPEATS; rep++) {
            const uint8_t *p = enc;
            for (size_t b = 0; b < blocks; b++) {
                size_t consumed;
                pivco_decode(bench_dec_ctx(), table, p, total_enc - (size_t)(p - enc), dec + b * BLK, &consumed);
                p += consumed;
            }
        }
        double el = now_ns_local() - t0;
        double mbs = 1000.0 * (double)(blocks * BLK) * REPEATS / el;
        if (mbs > best) best = mbs;
    }
    *dec_mbs_out = best;

out:
    free(enc); free(dec); free(table);
}

/* ------- huf0 (stock HUF_decompress) + Oodle-Huffman -- same chunked
 * methodology as bench_fair: 128 KB HUF chunks, best-of-RUNS over the
 * whole buffer.  These give us the apples-to-apples baseline / SoTA
 * columns alongside the 4 ph tree modes. */

#define HUF_CHUNK_TM (128 * 1024)

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static double measure_huf0_stk(const uint8_t *sym, size_t n) {
    size_t nch = (n + HUF_CHUNK_TM - 1) / HUF_CHUNK_TM;
    uint8_t *enc = malloc(n + n / 2 + 4096);
    uint8_t *dec = malloc(n);
    size_t  *off = malloc((nch + 1) * sizeof(size_t));
    double best = 0;
    if (!enc || !dec || !off) goto out;
    off[0] = 0;
    for (size_t c = 0; c < nch; c++) {
        size_t sz = (c < nch - 1) ? HUF_CHUNK_TM : n - c * HUF_CHUNK_TM;
        size_t r = HUF_compress(enc + off[c], sz + 1024, sym + c * HUF_CHUNK_TM, sz);
        if (HUF_isError(r) || r == 0) goto out;
        off[c + 1] = off[c] + r;
    }
    for (int run = 0; run < RUNS; run++) {
        double t0 = now_ns();
        for (int rep = 0; rep < REPEATS; rep++) {
            for (size_t c = 0; c < nch; c++) {
                size_t sz = (c < nch - 1) ? HUF_CHUNK_TM : n - c * HUF_CHUNK_TM;
                HUF_decompress(dec, sz, enc + off[c], off[c + 1] - off[c]);
            }
        }
        double dt = now_ns() - t0;
        double mbs = (double)n * REPEATS * 1e3 / dt;
        if (mbs > best) best = mbs;
    }
out:
    free(enc); free(dec); free(off);
    return best;
}

#ifdef PIVCO_HAS_OODLE
static double measure_oodle_huff(const uint8_t *sym, size_t n) {
    size_t nch = (n + HUF_CHUNK_TM - 1) / HUF_CHUNK_TM;
    uint8_t *enc = malloc(n + n / 2 + 4096);
    uint8_t *dec = malloc(n);
    size_t  *off = malloc((nch + 1) * sizeof(size_t));
    int     *ht  = malloc(nch * sizeof(int));
    double best = 0;
    if (!enc || !dec || !off || !ht) goto out;
    off[0] = 0;
    for (size_t c = 0; c < nch; c++) {
        size_t sz = (c < nch - 1) ? HUF_CHUNK_TM : n - c * HUF_CHUNK_TM;
        size_t r = oodle_huff_encode(sym + c * HUF_CHUNK_TM, sz,
                                       enc + off[c], sz + 1024, &ht[c]);
        if (r == 0) goto out;
        off[c + 1] = off[c] + r;
    }
    for (int run = 0; run < RUNS; run++) {
        double t0 = now_ns();
        for (int rep = 0; rep < REPEATS; rep++) {
            for (size_t c = 0; c < nch; c++) {
                size_t sz = (c < nch - 1) ? HUF_CHUNK_TM : n - c * HUF_CHUNK_TM;
                oodle_huff_decode(enc + off[c], off[c + 1] - off[c],
                                  dec, sz, ht[c]);
            }
        }
        double dt = now_ns() - t0;
        double mbs = (double)n * REPEATS * 1e3 / dt;
        if (mbs > best) best = mbs;
    }
out:
    free(enc); free(dec); free(off); free(ht);
    return best;
}
#endif

int main(int argc, char **argv) {
    bench_init();

    int repeats = (argc > 1) ? atoi(argv[1]) : DEFAULT_REPEATS;
    (void)repeats;
    const char *dist_filter = (argc > 2) ? argv[2] : NULL;

    /* FSE disabled across all measurements -- pure tree-shape ablation. */
    bench_cfg()->fse_enabled = (0);

    uint8_t *sym = malloc(BUF_BYTES);
    assert(sym);
    size_t n = BUF_BYTES;
    int blocks = n / BLK;

    printf("bench_tree_modes: pure tree-shape ablation (FSE off)\n");
    printf("BUF=%zu bytes, BLK=%d, blocks=%d, RUNS=%d\n\n", n, BLK, blocks, RUNS);

    printf("%-16s %-16s %8s %8s %10s %10s\n",
           "dist", "mode", "MB/s", "ratio", "flat_syms", "v_opt");
    printf("%-16s %-16s %8s %8s %10s %10s\n",
           "----", "----", "----", "-----", "---------", "-----");

    int nd = bench_num_distributions();
    for (int d = 0; d < nd; d++) {
        if (dist_filter) {
            if (strcmp(dist_filter, bench_dist_name(d)) != 0) continue;
        } else if (!bench_dist_is_main(d)) {
            continue;
        }
        /* image_jpeg in NAIVE/FUSED tree-modes triggers a codec stack-canary
           abort with the deepest possible 256-leaf canonical Huffman tree.
           Issue isn't specific to the tree-mode flag -- the production codec
           has the same behavior at the structural limit -- but skip it from
           the default run so the rest of the table is readable. */
        if (!dist_filter && strcmp(bench_dist_name(d), "image_jpeg") == 0) continue;
        bench_generate_symbols(d, sym, (int)n, SEED);

        /* Warmup pass: build + decode each mode once so caches / branch
           predictors / code-cache are equally hot for the measured pass. */
        double mbs[4]; double rat[4]; int fn[4];
        for (int m = 0; m < 4; m++) {
            mbs[m] = 0; rat[m] = 0; fn[m] = 0;
            measure_mode((pivco_tree_mode_t)m, sym, n, &mbs[m], &rat[m], &fn[m]);
        }
        for (int m = 0; m < 4; m++) {
            mbs[m] = 0; rat[m] = 0; fn[m] = 0;
            measure_mode((pivco_tree_mode_t)m, sym, n, &mbs[m], &rat[m], &fn[m]);
        }
        /* Measure huf0 (stock) and oo-huff in the same session so noise
           floor matches across all engines. */
        double huf0_mbs = measure_huf0_stk(sym, n);
#ifdef PIVCO_HAS_OODLE
        double ooh_mbs  = measure_oodle_huff(sym, n);
#else
        double ooh_mbs  = -1.0;
#endif
        for (int m = 0; m < 4; m++) {
            printf("%-16s %-16s %8.0f %8.3f %10d %9.2fx\n",
                   bench_dist_name(d), MODE_NAMES[m],
                   mbs[m], rat[m], fn[m],
                   mbs[0] > 0 ? mbs[m] / mbs[0] : 0.0);
        }
        printf("%-16s %-16s %8.0f %8s %10s %9s\n",
               bench_dist_name(d), "huf0", huf0_mbs, "-", "-", "-");
        if (ooh_mbs > 0) {
            printf("%-16s %-16s %8.0f %8s %10s %9s\n",
                   bench_dist_name(d), "oo_huff", ooh_mbs, "-", "-", "-");
        } else {
            printf("%-16s %-16s %8s %8s %10s %9s\n",
                   bench_dist_name(d), "oo_huff", "n/a", "-", "-", "-");
        }
        printf("\n");
    }

    free(sym);
    return 0;
}
