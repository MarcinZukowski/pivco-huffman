/* Profile harness: decode a chosen distribution and either run it under
 * an external sampler (sample/perf/xctrace) OR use the built-in
 * pivco_prof per-call-site instrumentation (PIVCO_PROF=1 build).
 *
 * Usage:   ./build/pivco_profile_english [dist_name]
 *   dist_name defaults to "english" for back-compat; pass "prose_pride"
 *   etc. to profile a different distribution.
 *
 * External sampling:  sample <pid> 10 -f profile.txt
 *
 * Built-in profiling: build with -DPIVCO_PROF=1 and the program will
 * print the per-call-site breakdown (calls, elements, ticks, ns/call,
 * ns/elem) at the end of the run — see include/pivco_prof.h. */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#include "pivco_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

extern void bench_init(void);
extern int bench_num_distributions(void);
extern const char *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern void bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                   int n_symbols, uint64_t seed);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    bench_init();

    const char *dist_name = (argc > 1) ? argv[1] : "english";

    int dist_idx = -1;
    for (int i = 0; i < bench_num_distributions(); i++) {
        if (strcmp(bench_dist_name(i), dist_name) == 0) {
            dist_idx = i;
            break;
        }
    }
    if (dist_idx < 0) {
        fprintf(stderr, "distribution '%s' not found\n", dist_name);
        return 1;
    }

    const int N = PIVCO_BLOCK_SIZE;
    const int NBLOCKS = 4 * 1024 * 1024 / N;
    const int REPS = 20000;

    /* Generate symbols and encode */
    uint8_t *symbols = malloc(NBLOCKS * N);
    bench_generate_symbols(dist_idx, symbols, NBLOCKS * N,
                           0xBEEFCAFE12345678ULL);

    /* FSE-on-bitmaps OFF by default so we profile the pure ph primitives
     * (raw bitmaps), matching the top-down primitive benchmark.  Set
     * PIVCO_PROFILE_FSE=1 to measure the ph+ANS path instead. */
    int fse_on = (getenv("PIVCO_PROFILE_FSE") &&
                  strcmp(getenv("PIVCO_PROFILE_FSE"), "1") == 0);
    bench_cfg()->fse_enabled = (fse_on);
    printf("FSE-on-bitmaps: %s\n", fse_on ? "ON (ph+ANS)" : "OFF (raw bitmaps)");

    pivco_table_t table;
    pivco_build_table(bench_cfg(), bench_dist_freq(dist_idx), &table);

    /* Encode all blocks */
    /* English can exceed PIVCO_MAX_ENCODED_SIZE (deep tree, many nodes
       with byte-alignment rounding). Use 4x block size to be safe. */
    /* Allocate generous encode buffer — some distributions exceed
       PIVCO_MAX_ENCODED_SIZE per block due to byte-alignment rounding. */
    size_t total_enc_cap = (size_t)NBLOCKS * N * 8;
    uint8_t *enc_buf = malloc(total_enc_cap);
    if (!enc_buf) { fprintf(stderr, "malloc enc_buf failed\n"); return 1; }
    size_t *enc_off = malloc((NBLOCKS + 1) * sizeof(size_t));
    enc_off[0] = 0;
    for (int b = 0; b < NBLOCKS; b++) {
        size_t elen;
        pivco_encode(bench_enc_ctx(), &table, symbols + b * N, N,
                             enc_buf + enc_off[b], &elen);
        enc_off[b + 1] = enc_off[b] + elen;
    }

    printf("Total encoded: %zu bytes (%.1f per block, max_enc=%zu)\n",
           enc_off[NBLOCKS], (double)enc_off[NBLOCKS] / NBLOCKS, total_enc_cap);
    fflush(stdout);

    /* Verify first block decodes correctly */
    {
        uint8_t *test = malloc(N);
        size_t consumed;
        int rc = pivco_decode(bench_dec_ctx(), &table, enc_buf + enc_off[0], enc_off[1] - enc_off[0], test, &consumed);
        printf("Verify: rc=%d consumed=%zu match=%d\n",
               rc, consumed, memcmp(symbols, test, N) == 0);
        fflush(stdout);
        free(test);
    }

    uint8_t *out = malloc((size_t)NBLOCKS * N);
    if (!out) { fprintf(stderr, "malloc out failed\n"); return 1; }
    printf("out=%p enc_buf=%p\n", (void*)out, (void*)enc_buf);

    printf("Decoding %d blocks x %d reps = %lld symbols (%s)\n",
           NBLOCKS, REPS, (long long)NBLOCKS * N * REPS, dist_name);
    fflush(stdout);

    /* Pin to a P-core / first CPU to reduce noise from migration. */
    int pin_rc = pivco_prof_pin_cpu(0);
    printf("CPU pin: %s\n", pin_rc == 0 ? "ok" : "best-effort");

    /* Probe cycle-counter frequency (ns/tick conversion).  Negligible
     * overhead — runs once before the timed loop. */
    double tick_freq = pivco_prof_probe_tick_freq();
    if (tick_freq > 0)
        printf("Cycle counter freq: %.2f MHz\n", tick_freq / 1e6);

    pivco_prof_reset();

    /* Mode selection.  PIVCO_PROFILE_MODE = "td" (default, top-down decode),
     * "bu" (bottom-up decode), or "encode" (encode loop instead). */
    const char *mode_env = getenv("PIVCO_PROFILE_MODE");
    int use_bu     = (mode_env && strcmp(mode_env, "bu") == 0);
    int use_encode = (mode_env && strcmp(mode_env, "encode") == 0);
    const char *mode_label =
        use_encode ? "encode" : (use_bu ? "bottom-up" : "top-down");
    printf("Profile mode: %s\n", mode_label);

    /* Per-block encode output buffer: each block writes into its own
     * pre-assigned slot of size enc_off[b+1]-enc_off[b] (from the
     * untimed setup pass above).  We overwrite in place each rep. */
    double t0 = now_sec();
    if (use_encode) {
        /* Use the same NEON encoder we want to profile.  The buffer was
         * already sized large enough by the setup pass; we just
         * overwrite the same slots, REPS times. */
        printf("Starting encode loop...\n"); fflush(stdout);
        for (int r = 0; r < REPS; r++) {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t elen;
                pivco_encode(bench_enc_ctx(), &table, symbols + b * N, N,
                                     enc_buf + enc_off[b], &elen);
            }
        }
    } else {
        printf("Starting decode loop...\n"); fflush(stdout);
        for (int r = 0; r < REPS; r++) {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t consumed;
                if (r == 0 && b < 5) printf("  block %d: off=%zu len=%zu\n", b, enc_off[b], enc_off[b+1]-enc_off[b]);
                int rc;
                if (use_bu) {
#if defined(PIVCO_HAS_NEON)
                    rc = pivco_decode_bu_neon(bench_dec_ctx(), &table, enc_buf + enc_off[b], enc_off[b + 1] - enc_off[b], out + b * N, &consumed);
#elif defined(PIVCO_HAS_SSE4)
                    rc = pivco_decode_bu_x86(bench_dec_ctx(), &table, enc_buf + enc_off[b], enc_off[b + 1] - enc_off[b], out + b * N, &consumed);
#else
                    rc = pivco_decode(bench_dec_ctx(), &table, enc_buf + enc_off[b], enc_off[b + 1] - enc_off[b], out + b * N, &consumed);
#endif
                } else {
                    rc = pivco_decode(bench_dec_ctx(), &table, enc_buf + enc_off[b], enc_off[b + 1] - enc_off[b], out + b * N, &consumed);
                }
                if (r == 0 && b < 5) printf("    rc=%d consumed=%zu\n", rc, consumed);
                if (consumed != enc_off[b + 1] - enc_off[b]) {
                    printf("MISMATCH block %d: encoded=%zu consumed=%zu rc=%d\n",
                           b, enc_off[b + 1] - enc_off[b], consumed, rc);
                    return 1;
                }
            }
        }
    }
    double t1 = now_sec();

    double total = (double)NBLOCKS * N * REPS;
    printf("%.1f M symbols in %.2f s = %.0f M/s\n",
           total / 1e6, t1 - t0, total / (t1 - t0) / 1e6);

    char label[128];
    snprintf(label, sizeof(label), "%s%s / BLK=%d",
             use_encode ? "ENCODE " : "", dist_name, N);
    pivco_prof_dump(label, t1 - t0, tick_freq,
                    (uint64_t)NBLOCKS * (uint64_t)REPS);

    free(symbols);
    free(enc_buf);
    free(enc_off);
    free(out);
    return 0;
}
