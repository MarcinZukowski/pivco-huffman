/* bench_multicore — multi-threaded decode throughput, PIVCO vs huf0_x2.
 *
 * Each thread decodes the same encoded data (read-only, cache-shareable)
 * to its own per-thread output buffer (cache-line aligned, no false
 * sharing).  Reports aggregate symbols/sec across all threads, plus the
 * scaling factor vs single-thread.
 *
 * Goal: confirm that the per-core 66 GB/s store-port saturation in
 * partition_8 doesn't prevent linear scaling — each P-core has its own
 * private L1d, so independent decode loops shouldn't bottleneck each
 * other until total working set exceeds shared L2 / DRAM bandwidth.
 *
 * Usage:
 *   ./build/pivco_bench_multicore [dist_name] [reps] [max_threads]
 *     dist_name    default "prose_pride"
 *     reps         default 200
 *     max_threads  default 8
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern void            bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                              int n_symbols, uint64_t seed);

#define BLOCKSZ        8192     /* PIVCO block size */
#define TOTAL_SYMBOLS  (4 * 1024 * 1024)
#define NBLOCKS        (TOTAL_SYMBOLS / BLOCKSZ)
#define HUF0_CHUNK     (128 * 1024)
#define NCHUNKS        (TOTAL_SYMBOLS / HUF0_CHUNK)

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    int                  tid;
    int                  reps;

    /* PIVCO encoded data (shared, read-only) */
    const uint8_t       *pivco_enc;
    const size_t        *pivco_off;
    const pivco_table_t *pivco_table;

    /* huf0 encoded data (shared, read-only) */
    const uint8_t       *huf0_enc;
    const size_t        *huf0_off;

    /* per-thread output buffer (cache-line aligned, exclusive) */
    uint8_t             *out_buf;

    /* codec selection */
    int                  codec;          /* 0 = PIVCO, 1 = huf0_x2 */

    /* coordination */
    _Atomic int         *start_flag;
    double               t_finish;
} worker_arg_t;

static void *worker(void *arg)
{
    worker_arg_t *a = (worker_arg_t *)arg;

    /* Wait for the master to release everyone simultaneously. */
    while (atomic_load_explicit(a->start_flag, memory_order_acquire) == 0) {
        /* spin */
    }

    if (a->codec == 0) {
        /* PIVCO */
        for (int r = 0; r < a->reps; r++) {
            for (int b = 0; b < NBLOCKS; b++) {
                size_t consumed;
                pivco_decode(bench_dec_ctx(), a->pivco_table, a->pivco_enc + a->pivco_off[b], a->pivco_off[b + 1] - a->pivco_off[b], a->out_buf + (size_t)b * BLOCKSZ, &consumed);
            }
        }
    } else {
        /* huf0_x2 */
        for (int r = 0; r < a->reps; r++) {
            for (int c = 0; c < NCHUNKS; c++) {
                HUF_decompress4X2(
                    a->out_buf + (size_t)c * HUF0_CHUNK, HUF0_CHUNK,
                    a->huf0_enc + a->huf0_off[c],
                    a->huf0_off[c + 1] - a->huf0_off[c]);
            }
        }
    }
    a->t_finish = now_sec();
    return NULL;
}

static double run_codec(int codec, int n_threads, int reps,
                         const uint8_t *pivco_enc, const size_t *pivco_off,
                         const pivco_table_t *pivco_table,
                         const uint8_t *huf0_enc, const size_t *huf0_off,
                         uint8_t **out_bufs)
{
    pthread_t          threads[64];
    worker_arg_t       args[64];
    _Atomic int        start_flag = 0;

    for (int t = 0; t < n_threads; t++) {
        args[t] = (worker_arg_t){
            .tid         = t,
            .reps        = reps,
            .pivco_enc   = pivco_enc,
            .pivco_off   = pivco_off,
            .pivco_table = pivco_table,
            .huf0_enc    = huf0_enc,
            .huf0_off    = huf0_off,
            .out_buf     = out_bufs[t],
            .codec       = codec,
            .start_flag  = &start_flag,
            .t_finish    = 0,
        };
        pthread_create(&threads[t], NULL, worker, &args[t]);
    }

    /* Brief settle: let all threads reach the spin-wait. */
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 50 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    double t_start = now_sec();
    atomic_store_explicit(&start_flag, 1, memory_order_release);

    double t_max_finish = t_start;
    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
        if (args[t].t_finish > t_max_finish) t_max_finish = args[t].t_finish;
    }
    return t_max_finish - t_start;
}

int main(int argc, char **argv)
{
    bench_init();

    const char *dist_name   = (argc > 1) ? argv[1] : "prose_pride";
    int         reps        = (argc > 2) ? atoi(argv[2]) : 200;
    int         max_threads = (argc > 3) ? atoi(argv[3]) : 8;
    if (max_threads > 64) max_threads = 64;

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

    /* Generate the source 4M-symbol stream. */
    uint8_t *symbols = (uint8_t *)malloc(TOTAL_SYMBOLS);
    bench_generate_symbols(dist_idx, symbols, TOTAL_SYMBOLS, 0xBEEFCAFE12345678ULL);

    /* PIVCO encode (per 8 KB block). */
    pivco_table_t pivco_table;
    pivco_build_table(bench_cfg(), bench_dist_freq(dist_idx), &pivco_table);
    size_t  pivco_cap = (size_t)NBLOCKS * BLOCKSZ * 8;
    uint8_t *pivco_enc = (uint8_t *)malloc(pivco_cap);
    size_t  *pivco_off = (size_t *)malloc((NBLOCKS + 1) * sizeof(size_t));
    pivco_off[0] = 0;
    for (int b = 0; b < NBLOCKS; b++) {
        size_t elen;
        pivco_encode(bench_enc_ctx(), &pivco_table, symbols + (size_t)b * BLOCKSZ, BLOCKSZ, pivco_enc + pivco_off[b], &elen);
        pivco_off[b + 1] = pivco_off[b] + elen;
    }
    size_t pivco_total = pivco_off[NBLOCKS];

    /* huf0 encode (per 128 KB chunk). */
    size_t  huf0_cap = (size_t)NCHUNKS * (HUF0_CHUNK + 1024);
    uint8_t *huf0_enc = (uint8_t *)malloc(huf0_cap);
    size_t  *huf0_off = (size_t *)malloc((NCHUNKS + 1) * sizeof(size_t));
    huf0_off[0] = 0;
    int huf0_ok = 1;
    for (int c = 0; c < NCHUNKS; c++) {
        size_t r = HUF_compress(huf0_enc + huf0_off[c],
                                HUF0_CHUNK + 1024,
                                symbols + (size_t)c * HUF0_CHUNK, HUF0_CHUNK);
        if (HUF_isError(r) || r == 0) { huf0_ok = 0; break; }
        huf0_off[c + 1] = huf0_off[c] + r;
    }
    if (!huf0_ok) {
        fprintf(stderr, "huf0 encode failed (distribution may be too uniform)\n");
    }

    /* Per-thread output buffers, each cache-line-aligned and not adjacent. */
    uint8_t *out_bufs[64] = {0};
    for (int t = 0; t < max_threads; t++) {
        if (posix_memalign((void **)&out_bufs[t], 4096, TOTAL_SYMBOLS) != 0) {
            fprintf(stderr, "posix_memalign failed for thread %d\n", t);
            return 1;
        }
    }

    printf("== bench_multicore: %s, %d reps × %d-symbol stream per thread, codec scaling ==\n",
           dist_name, reps, TOTAL_SYMBOLS);
    printf("PIVCO encoded: %zu B (%.1f bits/symbol)\n",
           pivco_total, 8.0 * pivco_total / TOTAL_SYMBOLS);
    if (huf0_ok)
        printf("huf0  encoded: %zu B (%.1f bits/symbol)\n\n",
               huf0_off[NCHUNKS], 8.0 * huf0_off[NCHUNKS] / TOTAL_SYMBOLS);
    printf("threads | PIVCO M/s | scale | huf0_x2 M/s | scale | PIVCO/huf0\n");
    printf("--------+-----------+-------+-------------+-------+-----------\n");

    double pivco_t1 = 0, huf0_t1 = 0;
    int thread_counts[] = {1, 2, 4, 6, 8, 12, 16};
    for (int ti = 0; ti < (int)(sizeof(thread_counts) / sizeof(*thread_counts)); ti++) {
        int t = thread_counts[ti];
        if (t > max_threads) break;

        double dt_p = run_codec(0, t, reps,
                                pivco_enc, pivco_off, &pivco_table,
                                huf0_enc, huf0_off, out_bufs);
        double total_syms = (double)t * reps * TOTAL_SYMBOLS;
        double pivco_ms = total_syms / dt_p / 1e6;
        if (t == 1) pivco_t1 = pivco_ms;

        double huf0_ms = 0;
        if (huf0_ok) {
            double dt_h = run_codec(1, t, reps,
                                    pivco_enc, pivco_off, &pivco_table,
                                    huf0_enc, huf0_off, out_bufs);
            huf0_ms = total_syms / dt_h / 1e6;
            if (t == 1) huf0_t1 = huf0_ms;
        }

        printf("%7d | %9.0f | %4.2fx | %11.0f | %4.2fx | %8.2fx\n",
               t, pivco_ms, pivco_t1 ? pivco_ms / pivco_t1 : 1.0,
               huf0_ms, huf0_t1 ? huf0_ms / huf0_t1 : 1.0,
               huf0_ms ? pivco_ms / huf0_ms : 0);
    }

    free(symbols);
    free(pivco_enc); free(pivco_off);
    free(huf0_enc);  free(huf0_off);
    for (int t = 0; t < max_threads; t++) free(out_bufs[t]);
    return 0;
}
