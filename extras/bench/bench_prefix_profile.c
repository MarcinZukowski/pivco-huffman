/* Per-phase profiler for the non-flat prefix decoder.
 *
 * Reproduces the decode_neon_prefix non-flat path in-line, times each
 * of its phases separately across many iterations, and reports ns/block
 * and % of total.  Used to figure out where to optimize.
 *
 * Build: registered as an executable by CMakeLists.
 * Run:  ./build/pivco_prefix_profile [distribution]
 *         Defaults to "english" (M=3). */

#include "pivco_huffman.h"
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

extern void         bench_init(void);
extern int          bench_num_distributions(void);
extern const char  *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern void         bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                           int n_symbols, uint64_t seed);

/* External subtree hook (from pivco_huffman_neon.c). */
extern void pivco_neon_decode_subtree_(const pivco_huffman_table_t *table,
                                        int16_t node_id,
                                        uint16_t *indices, int n,
                                        uint8_t *symbols,
                                        const uint8_t **in_ptr,
                                        uint16_t *tmp,
                                        int16_t skip_node);

#define N PIVCO_BLOCK_SIZE
#define REPS 50000      /* inner-loop iterations per phase measurement */

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* --- Phase implementations (mirroring non-flat decoder) ---------------- */

/* Phase 1: M-bit prefix extract (M=3 fast path, unrolled by 8). */
static void phase1_extract_m3(const uint8_t *in, uint8_t *out, int n)
{
    for (int k = 0; k < n; k += 8) {
        uint32_t w = (uint32_t)in[0]
                   | ((uint32_t)in[1] << 8)
                   | ((uint32_t)in[2] << 16);
        in += 3;
        out[k    ] = (uint8_t)((w >>  0) & 7);
        out[k + 1] = (uint8_t)((w >>  3) & 7);
        out[k + 2] = (uint8_t)((w >>  6) & 7);
        out[k + 3] = (uint8_t)((w >>  9) & 7);
        out[k + 4] = (uint8_t)((w >> 12) & 7);
        out[k + 5] = (uint8_t)((w >> 15) & 7);
        out[k + 6] = (uint8_t)((w >> 18) & 7);
        out[k + 7] = (uint8_t)((w >> 21) & 7);
    }
}

/* Phase 2: histogram — 8-way parallel counter arrays (matches decoder). */
static void phase2_histogram(const uint8_t *prefix, int *bin_count, int K, int n)
{
    int bc[8][256] = {{0}};
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        bc[0][prefix[k    ]]++;
        bc[1][prefix[k + 1]]++;
        bc[2][prefix[k + 2]]++;
        bc[3][prefix[k + 3]]++;
        bc[4][prefix[k + 4]]++;
        bc[5][prefix[k + 5]]++;
        bc[6][prefix[k + 6]]++;
        bc[7][prefix[k + 7]]++;
    }
    for (; k < n; k++) bc[0][prefix[k]]++;
    for (int v = 0; v < K; v++)
        bin_count[v] = bc[0][v] + bc[1][v] + bc[2][v] + bc[3][v]
                     + bc[4][v] + bc[5][v] + bc[6][v] + bc[7][v];
}

/* Phase 3: prefix-sum for offsets. */
static void phase3_prefix_sum(const int *bin_count, int *bin_offset, int K)
{
    bin_offset[0] = 0;
    for (int v = 0; v < K; v++) bin_offset[v+1] = bin_offset[v] + bin_count[v];
}

/* Phase 4: bucket — 8-way parallel placement (matches decoder). */
static void phase4_bucket(const uint8_t *prefix, const int *bin_offset,
                           uint16_t *bin_elements, int K, int n)
{
    /* Recompute per-stream counts so we can derive per-stream starting
     * offsets (in the decoder these come for free from phase 2). */
    int bc[8][256] = {{0}};
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        bc[0][prefix[k    ]]++;
        bc[1][prefix[k + 1]]++;
        bc[2][prefix[k + 2]]++;
        bc[3][prefix[k + 3]]++;
        bc[4][prefix[k + 4]]++;
        bc[5][prefix[k + 5]]++;
        bc[6][prefix[k + 6]]++;
        bc[7][prefix[k + 7]]++;
    }
    int place[8][256];
    for (int v = 0; v < K; v++) {
        place[0][v] = bin_offset[v];
        place[1][v] = place[0][v] + bc[0][v];
        place[2][v] = place[1][v] + bc[1][v];
        place[3][v] = place[2][v] + bc[2][v];
        place[4][v] = place[3][v] + bc[3][v];
        place[5][v] = place[4][v] + bc[4][v];
        place[6][v] = place[5][v] + bc[5][v];
        place[7][v] = place[6][v] + bc[6][v];
    }
    k = 0;
    for (; k + 8 <= n; k += 8) {
        bin_elements[place[0][prefix[k    ]]++] = (uint16_t)(k    );
        bin_elements[place[1][prefix[k + 1]]++] = (uint16_t)(k + 1);
        bin_elements[place[2][prefix[k + 2]]++] = (uint16_t)(k + 2);
        bin_elements[place[3][prefix[k + 3]]++] = (uint16_t)(k + 3);
        bin_elements[place[4][prefix[k + 4]]++] = (uint16_t)(k + 4);
        bin_elements[place[5][prefix[k + 5]]++] = (uint16_t)(k + 5);
        bin_elements[place[6][prefix[k + 6]]++] = (uint16_t)(k + 6);
        bin_elements[place[7][prefix[k + 7]]++] = (uint16_t)(k + 7);
    }
    for (; k < n; k++)
        bin_elements[place[0][prefix[k]]++] = (uint16_t)k;
}

/* Phase 0 (for reference): the prefill memset. */
static void phase0_memset(uint8_t *symbols, uint8_t prefill_sym, int n)
{
    memset(symbols, prefill_sym, (size_t)n);
}

/* --- Driver ------------------------------------------------------------ */

static int find_dist(const char *name)
{
    int n = bench_num_distributions();
    for (int i = 0; i < n; i++)
        if (strcmp(bench_dist_name(i), name) == 0) return i;
    return -1;
}

int main(int argc, char **argv)
{
    bench_init();

    const char *dist_name = argc > 1 ? argv[1] : "english";
    int d = find_dist(dist_name);
    if (d < 0) {
        fprintf(stderr, "unknown distribution: %s\n", dist_name);
        return 1;
    }

    const uint64_t *freq = bench_dist_freq(d);
    pivco_huffman_table_t *table =
        (pivco_huffman_table_t *)malloc(sizeof(pivco_huffman_table_t));
    pivco_huffman_build_table(freq, table);

    int M = table->min_len;
    int K = 1 << M;
    printf("Distribution: %s  (M=%d, max_len=%d, K=%d, N=%d)\n\n",
           dist_name, M, table->max_len, K, N);

    /* Generate a block of symbols. */
    uint8_t *symbols = (uint8_t *)malloc(N);
    bench_generate_symbols(d, symbols, N, 0xBEEFCAFE12345678ULL);

    /* Encode it with the prefix backend so we have a real prefix stream.
     * Generous over-allocation — prefix format adds subtree bitmaps after
     * the M-bit prefix stream. */
    uint8_t *enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE * 8);
    size_t enc_len;
    int rc = pivco_huffman_encode_neon_prefix(symbols, table, enc, &enc_len);
    if (rc != PIVCO_OK) {
        fprintf(stderr, "encode failed: %d\n", rc);
        return 1;
    }
    printf("Encoded block: %zu bytes (%.2f%% of raw)\n\n",
           enc_len, 100.0 * (double)enc_len / (double)N);

    /* Allocate phase working buffers. */
    uint8_t  *prefix       = (uint8_t *)malloc(N);
    int      *bin_count    = (int *)malloc(K * sizeof(int));
    int      *bin_offset   = (int *)malloc((K + 1) * sizeof(int));
    uint16_t *bin_elements = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint8_t  *out_syms     = (uint8_t *)malloc(N);
    uint16_t *tmp          = (uint16_t *)malloc(N * 2 * sizeof(uint16_t));

    /* Pre-populate prefix[] and bin_count[] for phases that need them as
       inputs (so we measure only the phase under test, not phase_prev). */
    phase1_extract_m3(enc, prefix, N);
    phase2_histogram(prefix, bin_count, K, N);
    phase3_prefix_sum(bin_count, bin_offset, K);
    phase4_bucket(prefix, bin_offset, bin_elements, K, N);

    /* Time each phase independently over REPS iterations. */
    volatile uint64_t sink = 0;   /* prevent DCE */
    double phase_ns[8];
    const char *names[] = {
        "0: memset(prefill)",
        "1: extract M-bit prefix",
        "2: histogram",
        "3: prefix-sum",
        "4: bucket",
        "5: per-bin dispatch (scatter + subtree decodes)",
        "FULL decode (end-to-end)"
    };

    /* Phase 0 */
    {
        uint64_t t0 = now_ns();
        for (int r = 0; r < REPS; r++) {
            phase0_memset(out_syms, table->prefill_sym, N);
            sink ^= out_syms[r & (N - 1)];
        }
        phase_ns[0] = (double)(now_ns() - t0) / (double)REPS;
    }

    /* Phase 1 */
    {
        uint64_t t0 = now_ns();
        for (int r = 0; r < REPS; r++) {
            phase1_extract_m3(enc, prefix, N);
            sink ^= prefix[r & (N - 1)];
        }
        phase_ns[1] = (double)(now_ns() - t0) / (double)REPS;
    }

    /* Phase 2 */
    {
        uint64_t t0 = now_ns();
        for (int r = 0; r < REPS; r++) {
            phase2_histogram(prefix, bin_count, K, N);
            sink ^= bin_count[r & (K - 1)];
        }
        phase_ns[2] = (double)(now_ns() - t0) / (double)REPS;
    }

    /* Phase 3 */
    {
        uint64_t t0 = now_ns();
        for (int r = 0; r < REPS; r++) {
            phase3_prefix_sum(bin_count, bin_offset, K);
            sink ^= bin_offset[r & (K)];
        }
        phase_ns[3] = (double)(now_ns() - t0) / (double)REPS;
    }

    /* Phase 4 */
    {
        uint64_t t0 = now_ns();
        for (int r = 0; r < REPS; r++) {
            phase4_bucket(prefix, bin_offset, bin_elements, K, N);
            sink ^= bin_elements[r & (N - 1)];
        }
        phase_ns[4] = (double)(now_ns() - t0) / (double)REPS;
    }

    /* Phase 5 (per-bin dispatch): invoke the real decoder from phase 4 onward
       by calling pivco_huffman_decode_neon_prefix repeatedly, subtracting
       phases 0-4 from the total.  Simpler: run the full decode and subtract. */

    /* Full end-to-end decode (for total + phase 5 back-out). */
    {
        uint8_t dec[PIVCO_BLOCK_SIZE];
        size_t consumed;
        uint64_t t0 = now_ns();
        for (int r = 0; r < REPS; r++) {
            pivco_huffman_decode_neon_prefix(enc, enc_len, table, dec, &consumed);
            sink ^= dec[r & (N - 1)];
        }
        phase_ns[6] = (double)(now_ns() - t0) / (double)REPS;
    }
    phase_ns[5] = phase_ns[6] - (phase_ns[0] + phase_ns[1] + phase_ns[2]
                                 + phase_ns[3] + phase_ns[4]);
    if (phase_ns[5] < 0) phase_ns[5] = 0;  /* clamp if jitter made it negative */

    /* Report. */
    double total = phase_ns[6];
    printf("Per-block phase cost (averaged over %d iterations):\n\n", REPS);
    printf("  %-52s %10s %8s\n", "Phase", "ns/block", "%total");
    printf("  %.*s\n", 72, "------------------------------------------------------------------------");
    for (int i = 0; i < 7; i++) {
        double pct = total > 0 ? 100.0 * phase_ns[i] / total : 0.0;
        printf("  %-52s %10.1f %7.1f%%\n",
               names[i], phase_ns[i], pct);
    }

    /* Per-element cost — helps compare to pivco_n which does ~1.4 c/elem
       on english.  Convert ns/block → c/elem assuming 3.5 GHz. */
    printf("\nPer-element (approx, assuming 3.5 GHz):\n");
    for (int i = 0; i < 7; i++) {
        double c_per_elem = (phase_ns[i] * 3.5) / (double)N;
        printf("  %-52s %7.3f c/elem\n", names[i], c_per_elem);
    }

    /* Also print baseline pivco_n cost for comparison. */
    {
        uint8_t *enc_n = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE);
        size_t enc_n_len;
        pivco_huffman_encode(symbols, table, enc_n, &enc_n_len);

        uint8_t dec[PIVCO_BLOCK_SIZE];
        size_t consumed;
        /* Warm up */
        for (int w = 0; w < 1000; w++)
            pivco_huffman_decode(enc_n, enc_n_len, table, dec, &consumed);
        uint64_t t0 = now_ns();
        for (int r = 0; r < REPS; r++) {
            pivco_huffman_decode(enc_n, enc_n_len, table, dec, &consumed);
            sink ^= dec[r & (N - 1)];
        }
        double ns = (double)(now_ns() - t0) / (double)REPS;
        double ratio = ns > 0 ? total / ns : 0;
        printf("\nReference:\n");
        printf("  %-52s %10.1f ns/block  (prefix is %.2fx slower)\n",
               "pivco_n (baseline NEON decoder)", ns, ratio);
        free(enc_n);
    }

    (void)sink;
    free(symbols); free(enc);
    free(prefix); free(bin_count); free(bin_offset); free(bin_elements);
    free(out_syms); free(tmp);
    free(table);
    return 0;
}
