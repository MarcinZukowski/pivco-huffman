/* ph_td_bench_naive -- M4 throughput bench for ph-td-naive vs huf0.
 *
 * Generates synthetic data matching two distributions (proba80,
 * prose_pride), then for each:
 *
 *   1. Builds a Huffman table via pivco_huffman_build_table_naive.
 *      Forces uniform INTERNAL_FULL/LEAF classification -- no
 *      flat-subtree path, no half-partition, no fused both-leaves,
 *      no constant-prefill.
 *   2. Encodes the data using the existing ph-td encoder (which
 *      emits a per-node bitmap for every internal node since the
 *      naive table marks them all INTERNAL_FULL).
 *   3. Times pivco_huffman_decode_naive (P + S1 scalar primitives).
 *   4. Encodes + times huf0 for comparison (HUF_compress /
 *      HUF_decompress, default decoder).
 *   5. Prints decode throughput in GB/s for each.
 *
 *   Default: 256 blocks of PIVCO_BLOCK_SIZE = 2 MB total per dist;
 *   500 decode iterations, take best-of-3.
 */

#include "pivco_huffman.h"
#include "pivco_prof.h"

#define HUF_STATIC_LINKING_ONLY
#include "huf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* Pull the canonical distribution generator + table from the main
 * bench so we don't have a divergent copy here.  Header-style
 * include: bench_distributions.c declares `static distribution_t
 * distributions[]` and `static void init_distributions()` -- one
 * translation unit. */
#include "../../../bench/bench_distributions.c"

/* ---------- timer ---------- */
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* ---------- xorshift PRNG ---------- */
static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    uint64_t x = rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng = x;
}

static uint8_t draw_byte(const uint64_t cum[256]) {
    uint64_t r = rng_next() % cum[255];
    /* Binary search keeps it tight on prose_pride's wide support. */
    int lo = 0, hi = 255;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (r < cum[mid]) hi = mid;
        else              lo = mid + 1;
    }
    return (uint8_t)lo;
}

static void make_cum(const uint64_t freq[256], uint64_t cum[256]) {
    cum[0] = freq[0];
    for (int i = 1; i < 256; i++) cum[i] = cum[i - 1] + freq[i];
}

/* ---------- distributions ---------- */

static const uint64_t *find_dist_freq(const char *name) {
    for (size_t i = 0; i < NUM_DISTRIBUTIONS; i++) {
        if (strcmp(distributions[i].name, name) == 0)
            return distributions[i].freq;
    }
    fprintf(stderr, "unknown distribution: %s\n", name);
    return NULL;
}

/* ---------- bench harness ---------- */

#define N_BLOCKS    256
#define N_ITERS     500

/* Global CSV sinks (NULL = none).  g_csv: throughput rows
 * (distribution, decoder, throughput_gbs, bpc).  g_profile_csv:
 * per-primitive tic/toc profile rows (distribution, decoder,
 * primitive, calls, elements, ticks, ns_per_elem, ns_per_call). */
static FILE *g_csv         = NULL;
static FILE *g_profile_csv = NULL;
static double g_tick_freq  = 0.0;     /* probed once at startup */

static void csv_row(const char *dist, const char *decoder,
                     double gbs, double bpc)
{
    if (!g_csv) return;
    fprintf(g_csv, "%s,%s,%.4f,%.4f\n", dist, decoder, gbs, bpc);
}

static void profile_csv_dump(const char *dist, const char *decoder)
{
    if (!g_profile_csv || g_tick_freq <= 0.0) return;
    double ns_per_tick = 1e9 / g_tick_freq;
    for (int slot = 0; slot < PROF_NR_SLOTS; slot++) {
        pivco_prof_counter_t *c = &pivco_prof_counters[slot];
        if (c->calls == 0 || c->ticks == 0) continue;   /* count-only */
        const char *pname = pivco_prof_name((pivco_prof_slot_t)slot);
        if (!pname || pname[0] == '?') continue;
        double ns_total = (double)c->ticks * ns_per_tick;
        double ns_elem  = c->elements ? ns_total / (double)c->elements : 0.0;
        double ns_call  = (double)c->calls    ? ns_total / (double)c->calls    : 0.0;
        fprintf(g_profile_csv,
                "%s,%s,%s,%llu,%llu,%llu,%.6f,%.6f\n",
                dist, decoder, pname,
                (unsigned long long)c->calls,
                (unsigned long long)c->elements,
                (unsigned long long)c->ticks,
                ns_elem, ns_call);
    }
}

static void run_one(const char *name) {
    /* ---- frequencies (pulled from the shared bench distributions) ---- */
    const uint64_t *src_freq = find_dist_freq(name);
    if (!src_freq) return;
    uint64_t freq[256];
    memcpy(freq, src_freq, sizeof(freq));

    uint64_t cum[256];
    make_cum(freq, cum);
    if (cum[255] == 0) { fprintf(stderr, "%s: empty freq\n", name); return; }

    const size_t blksz = PIVCO_BLOCK_SIZE;
    const size_t total = blksz * N_BLOCKS;

    /* ---- generate data ---- */
    uint8_t *src = (uint8_t *)malloc(total);
    for (size_t i = 0; i < total; i++) src[i] = draw_byte(cum);

    /* ---- ph-td-naive: build naive table, encode/decode (slim wire) ---- */
    pivco_huffman_table_t table_naive;
    int rc = pivco_huffman_build_table_naive(freq, &table_naive);
    if (rc != PIVCO_OK) { fprintf(stderr, "%s: build_table_naive rc=%d\n", name, rc); free(src); return; }

    uint8_t *ph_enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE * (size_t)N_BLOCKS);
    size_t  *ph_off = (size_t *)calloc((size_t)N_BLOCKS + 1, sizeof(size_t));
    for (int b = 0; b < N_BLOCKS; b++) {
        size_t enc_len = PIVCO_MAX_ENCODED_SIZE;
        rc = pivco_huffman_encode_naive(src + (size_t)b * blksz,
                                         &table_naive,
                                         ph_enc + ph_off[b],
                                         &enc_len);
        if (rc != PIVCO_OK) {
            fprintf(stderr, "%s: ph naive encode b=%d rc=%d\n", name, b, rc);
            free(src); free(ph_enc); free(ph_off); return;
        }
        ph_off[b + 1] = ph_off[b] + enc_len;
    }
    size_t ph_total_enc = ph_off[N_BLOCKS];

    uint8_t *dec = (uint8_t *)malloc(blksz);
    size_t consumed = 0;
    rc = pivco_huffman_decode_naive(ph_enc, ph_off[1], &table_naive, dec, &consumed);
    if (rc != PIVCO_OK || memcmp(src, dec, blksz) != 0) {
        fprintf(stderr, "%s: naive roundtrip FAILED rc=%d\n", name, rc);
        free(src); free(ph_enc); free(ph_off); free(dec); return;
    }

    double ph_best_ns_per_byte = 1e18;
    for (int trial = 0; trial < 3; trial++) {
        if (trial == 2) pivco_prof_reset();   /* count last trial only */
        uint64_t t0 = now_ns();
        for (int it = 0; it < N_ITERS; it++) {
            for (int b = 0; b < N_BLOCKS; b++) {
                pivco_huffman_decode_naive(
                    ph_enc + ph_off[b],
                    ph_off[b + 1] - ph_off[b],
                    &table_naive, dec, &consumed);
            }
        }
        uint64_t t1 = now_ns();
        double ns = (double)(t1 - t0) / ((double)N_ITERS * (double)total);
        if (ns < ph_best_ns_per_byte) ph_best_ns_per_byte = ns;
    }
    /* Dump naive's per-primitive profile. */
    {
        char lbl[64]; snprintf(lbl, sizeof(lbl), "%s -- naive", name);
        pivco_prof_dump(lbl, 0.0, g_tick_freq);
        profile_csv_dump(name, "naive");
    }

    /* ---- ph-td-naive-simd: same SLIM wire (from encode_naive),
     * decoded with the SIMD primitives.  Fills the (naive-tree x
     * SIMD-primitives) grid cell. */
#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_AVX512)
    {
#if defined(PIVCO_HAS_AVX512)
        int (*naive_simd_decode)(const uint8_t *, size_t,
                                  const pivco_huffman_table_t *,
                                  uint8_t *, size_t *) =
            pivco_huffman_decode_naive_simd_avx512;
#else
        int (*naive_simd_decode)(const uint8_t *, size_t,
                                  const pivco_huffman_table_t *,
                                  uint8_t *, size_t *) =
            pivco_huffman_decode_naive_simd_neon;
#endif
        rc = naive_simd_decode(ph_enc, ph_off[1], &table_naive,
                                 dec, &consumed);
        if (rc == PIVCO_OK && memcmp(src, dec, blksz) == 0) {
            double naive_simd_ns = 1e18;
            for (int trial = 0; trial < 3; trial++) {
                if (trial == 2) pivco_prof_reset();
                uint64_t t0 = now_ns();
                for (int it = 0; it < N_ITERS; it++) {
                    for (int b = 0; b < N_BLOCKS; b++) {
                        naive_simd_decode(ph_enc + ph_off[b],
                                            ph_off[b + 1] - ph_off[b],
                                            &table_naive, dec, &consumed);
                    }
                }
                uint64_t t1 = now_ns();
                double ns = (double)(t1 - t0) /
                              ((double)N_ITERS * (double)total);
                if (ns < naive_simd_ns) naive_simd_ns = ns;
            }
            double gbs = 1.0 / naive_simd_ns;
            double bpc = (double)ph_total_enc * 8.0 / (double)total;
            csv_row(name, "naive_simd", gbs, bpc);
            char lbl[64];
            snprintf(lbl, sizeof(lbl), "%s -- naive_simd", name);
            pivco_prof_dump(lbl, 0.0, g_tick_freq);
            profile_csv_dump(name, "naive_simd");
            printf("%-14s | naive_simd %6.3f GB/s (%4.2f bpc)\n",
                   name, gbs, bpc);
        } else {
            fprintf(stderr,
                    "%s: naive_simd roundtrip FAILED rc=%d\n", name, rc);
        }
    }
#endif

    /* ---- ph-td-scalar-opt: build OPTIMISED table, encode via neon
     * (same wire format), decode via the scalar-opt dispatcher ---- */
    pivco_huffman_table_t table_opt;
    rc = pivco_huffman_build_table(freq, &table_opt);
    if (rc != PIVCO_OK) { fprintf(stderr, "%s: build_table rc=%d\n", name, rc); free(src); free(ph_enc); free(ph_off); free(dec); return; }

    /* ---- Diagnose: count node types in the optimised table ---- */
    int nt_full=0, nt_flat=0, nt_both=0, nt_hr=0, nt_hl=0, nt_leaf=0, nt_skip=0;
    int flat_d_sum = 0, flat_count = 0;
    for (int16_t i = 0; i < table_opt.tree_node_count; i++) {
        switch ((pivco_node_type_t)table_opt.node_type[i]) {
            case PIVCO_NODE_INTERNAL_FULL: nt_full++; break;
            case PIVCO_NODE_INTERNAL_FLAT: nt_flat++;
                flat_d_sum += table_opt.flat_depth[i]; flat_count++; break;
            case PIVCO_NODE_BOTH_LEAVES:   nt_both++; break;
            case PIVCO_NODE_HALF_RIGHT:    nt_hr++;   break;
            case PIVCO_NODE_HALF_LEFT:     nt_hl++;   break;
            case PIVCO_NODE_LEAF:          nt_leaf++; break;
            case PIVCO_NODE_SKIP:          nt_skip++; break;
        }
    }
    fprintf(stderr, "%s tree: total=%d  FULL=%d FLAT=%d (avg D=%.1f)  "
            "BOTH=%d HALF_R=%d HALF_L=%d  LEAF=%d SKIP=%d  prefill_sym=%d\n",
            name, (int)table_opt.tree_node_count,
            nt_full, nt_flat,
            flat_count ? (double)flat_d_sum / flat_count : 0.0,
            nt_both, nt_hr, nt_hl, nt_leaf, nt_skip,
            (int)table_opt.prefill_sym);

    uint8_t *opt_enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE * (size_t)N_BLOCKS);
    size_t  *opt_off = (size_t *)calloc((size_t)N_BLOCKS + 1, sizeof(size_t));
    for (int b = 0; b < N_BLOCKS; b++) {
        size_t enc_len = PIVCO_MAX_ENCODED_SIZE;
        rc = pivco_huffman_encode_scalar_opt(src + (size_t)b * blksz,
                                              &table_opt,
                                              opt_enc + opt_off[b],
                                              &enc_len);
        if (rc != PIVCO_OK) {
            fprintf(stderr, "%s: ph opt encode b=%d rc=%d\n", name, b, rc);
            free(src); free(ph_enc); free(ph_off); free(dec); free(opt_enc); free(opt_off); return;
        }
        opt_off[b + 1] = opt_off[b] + enc_len;
    }
    size_t opt_total_enc = opt_off[N_BLOCKS];

    rc = pivco_huffman_decode_scalar_opt(opt_enc, opt_off[1], &table_opt, dec, &consumed);
    if (rc != PIVCO_OK || memcmp(src, dec, blksz) != 0) {
        fprintf(stderr, "%s: scalar_opt roundtrip FAILED rc=%d\n", name, rc);
        free(src); free(ph_enc); free(ph_off); free(dec); free(opt_enc); free(opt_off); return;
    }

    double opt_best_ns = 1e18;
    for (int trial = 0; trial < 3; trial++) {
        if (trial == 2) pivco_prof_reset();
        uint64_t t0 = now_ns();
        for (int it = 0; it < N_ITERS; it++) {
            for (int b = 0; b < N_BLOCKS; b++) {
                pivco_huffman_decode_scalar_opt(
                    opt_enc + opt_off[b],
                    opt_off[b + 1] - opt_off[b],
                    &table_opt, dec, &consumed);
            }
        }
        uint64_t t1 = now_ns();
        double ns = (double)(t1 - t0) / ((double)N_ITERS * (double)total);
        if (ns < opt_best_ns) opt_best_ns = ns;
    }
    /* Dump scalar-opt's per-primitive profile. */
    {
        char lbl[64]; snprintf(lbl, sizeof(lbl), "%s -- scalar_opt", name);
        pivco_prof_dump(lbl, 0.0, g_tick_freq);
        profile_csv_dump(name, "scalar_opt");
    }

    /* ---- ph-td-simd-opt: same OPTIMISED table; encode + decode
     * via the platform-native SIMD TD path.  On M4 that's the
     * production NEON decoder from extras/ph-td/src/pivco_huffman_neon.c;
     * on c8i it's the resurrected AVX-512 TD from .../pivco_huffman_avx512.c.
     * Output emitted under decoder="simd_opt". */
#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_AVX512)
    {
#if defined(PIVCO_HAS_AVX512)
        int (*simd_encode)(const uint8_t *, const pivco_huffman_table_t *,
                            uint8_t *, size_t *) = pivco_huffman_encode_avx512;
        int (*simd_decode)(const uint8_t *, size_t,
                            const pivco_huffman_table_t *,
                            uint8_t *, size_t *) = pivco_huffman_decode_avx512;
        const char *simd_label = "avx512";
#else
        int (*simd_encode)(const uint8_t *, const pivco_huffman_table_t *,
                            uint8_t *, size_t *) = pivco_huffman_encode_neon;
        int (*simd_decode)(const uint8_t *, size_t,
                            const pivco_huffman_table_t *,
                            uint8_t *, size_t *) = pivco_huffman_decode_neon;
        const char *simd_label = "neon";
#endif
        uint8_t *simd_enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE * (size_t)N_BLOCKS);
        size_t  *simd_off = (size_t *)calloc((size_t)N_BLOCKS + 1, sizeof(size_t));
        int ok = 1;
        for (int b = 0; b < N_BLOCKS && ok; b++) {
            size_t enc_len = PIVCO_MAX_ENCODED_SIZE;
            rc = simd_encode(src + (size_t)b * blksz, &table_opt,
                              simd_enc + simd_off[b], &enc_len);
            if (rc != PIVCO_OK) { ok = 0; break; }
            simd_off[b + 1] = simd_off[b] + enc_len;
        }
        if (ok) {
            size_t simd_total_enc = simd_off[N_BLOCKS];
            /* Roundtrip check on first block. */
            rc = simd_decode(simd_enc, simd_off[1], &table_opt,
                              dec, &consumed);
            if (rc != PIVCO_OK || memcmp(src, dec, blksz) != 0) {
                fprintf(stderr, "%s: simd_opt (%s) roundtrip FAILED rc=%d\n",
                        name, simd_label, rc);
                ok = 0;
            }
            if (ok) {
                double simd_best_ns = 1e18;
                for (int trial = 0; trial < 3; trial++) {
                    if (trial == 2) pivco_prof_reset();
                    uint64_t t0 = now_ns();
                    for (int it = 0; it < N_ITERS; it++) {
                        for (int b = 0; b < N_BLOCKS; b++) {
                            simd_decode(simd_enc + simd_off[b],
                                          simd_off[b + 1] - simd_off[b],
                                          &table_opt, dec, &consumed);
                        }
                    }
                    uint64_t t1 = now_ns();
                    double ns = (double)(t1 - t0) / ((double)N_ITERS * (double)total);
                    if (ns < simd_best_ns) simd_best_ns = ns;
                }
                double simd_gbs = 1.0 / simd_best_ns;
                double simd_bpc = (double)simd_total_enc * 8.0 / (double)total;
                csv_row(name, "simd_opt", simd_gbs, simd_bpc);
                char lbl[64];
                snprintf(lbl, sizeof(lbl), "%s -- simd_opt (%s)", name, simd_label);
                pivco_prof_dump(lbl, 0.0, g_tick_freq);
                profile_csv_dump(name, "simd_opt");
                printf("%-14s | simd_opt (%s) %6.3f GB/s (%4.2f bpc)\n",
                       name, simd_label, simd_gbs, simd_bpc);
            }
        }
        free(simd_enc); free(simd_off);
    }
#endif

    /* ---- huf0: encode + decode ----
     * Use 128 KB chunks (HUF0_CHUNK) -- huf0's natural block size and
     * the chunking used by the main bench (bench/bench_main.c).  At
     * smaller chunks per-call overhead (codebook setup) dominates
     * and underreports huf0's throughput. */
    #define HUF0_CHUNK_SIZE (128 * 1024)
    int hu_nchunks = (int)((total + HUF0_CHUNK_SIZE - 1) / HUF0_CHUNK_SIZE);
    uint8_t *hu_enc = (uint8_t *)malloc((size_t)hu_nchunks * (HUF0_CHUNK_SIZE + 1024));
    size_t  *hu_off = (size_t *)calloc((size_t)hu_nchunks + 1, sizeof(size_t));
    int huf0_ok = 1;
    for (int c = 0; c < hu_nchunks && huf0_ok; c++) {
        size_t chunk_sz = (c < hu_nchunks - 1) ? (size_t)HUF0_CHUNK_SIZE
                          : total - (size_t)c * HUF0_CHUNK_SIZE;
        size_t enc_len = HUF_compress(hu_enc + hu_off[c],
                                       HUF0_CHUNK_SIZE + 1024,
                                       src + (size_t)c * HUF0_CHUNK_SIZE,
                                       chunk_sz);
        if (HUF_isError(enc_len) || enc_len == 0) { huf0_ok = 0; break; }
        hu_off[c + 1] = hu_off[c] + enc_len;
    }

    /* huf0 baseline = HUF_decompress4X2 (the 4-stream double-symbol
     * decoder).  This is the canonical comparison baseline -- huf0's
     * strongest decoder on the skewed distributions ph cares about. */
    double hu_x2_ns = -1.0;
    size_t hu_total_enc = 0;
    uint8_t *hu_dec_chunk = (uint8_t *)malloc(HUF0_CHUNK_SIZE);
    if (huf0_ok) {
        hu_total_enc = hu_off[hu_nchunks];

        /* Verify the first chunk roundtrips correctly. */
        size_t r = HUF_decompress4X2(hu_dec_chunk, HUF0_CHUNK_SIZE,
                                       hu_enc, hu_off[1]);
        if (HUF_isError(r) || r != HUF0_CHUNK_SIZE ||
            memcmp(src, hu_dec_chunk, HUF0_CHUNK_SIZE) != 0) {
            fprintf(stderr, "%s: huf0_x2 roundtrip FAILED\n", name);
            huf0_ok = 0;
        }

        if (huf0_ok) {
            hu_x2_ns = 1e18;
            for (int trial = 0; trial < 3; trial++) {
                uint64_t t0 = now_ns();
                for (int it = 0; it < N_ITERS; it++) {
                    for (int c = 0; c < hu_nchunks; c++) {
                        size_t chunk_sz = (c < hu_nchunks - 1)
                            ? (size_t)HUF0_CHUNK_SIZE
                            : total - (size_t)c * HUF0_CHUNK_SIZE;
                        HUF_decompress4X2(hu_dec_chunk, chunk_sz,
                                            hu_enc + hu_off[c],
                                            hu_off[c + 1] - hu_off[c]);
                    }
                }
                uint64_t t1 = now_ns();
                double ns = (double)(t1 - t0) /
                              ((double)N_ITERS * (double)total);
                if (ns < hu_x2_ns) hu_x2_ns = ns;
            }
        }
    }
    free(hu_dec_chunk);

    /* ---- output ---- */
    double ph_gbs   = 1.0 / ph_best_ns_per_byte;
    double opt_gbs  = 1.0 / opt_best_ns;
    double hu_gbs   = hu_x2_ns > 0 ? 1.0 / hu_x2_ns : 0.0;
    double ph_bpc   = (double)ph_total_enc  * 8.0 / (double)total;
    double opt_bpc  = (double)opt_total_enc * 8.0 / (double)total;
    double hu_bpc   = (double)hu_total_enc  * 8.0 / (double)total;
    /* Identifiers in CSV columns are SQL-safe (no dashes) per
     * paper/benches.yaml contract. */
    csv_row(name, "naive",      ph_gbs,  ph_bpc);
    csv_row(name, "scalar_opt", opt_gbs, opt_bpc);
    if (hu_gbs > 0) csv_row(name, "huf0_x2", hu_gbs, hu_bpc);
    printf("%-14s | naive %6.3f GB/s (%4.2f bpc) | "
           "scalar-opt %6.3f GB/s (%4.2f bpc) | "
           "huf0_x2 %6.3f GB/s (%4.2f bpc) | "
           "naive/x2 %.2fx  opt/x2 %.2fx\n",
           name, ph_gbs, ph_bpc, opt_gbs, opt_bpc, hu_gbs, hu_bpc,
           hu_gbs > 0 ? ph_gbs  / hu_gbs : 0.0,
           hu_gbs > 0 ? opt_gbs / hu_gbs : 0.0);

    free(src); free(ph_enc); free(ph_off);
    free(opt_enc); free(opt_off);
    free(hu_enc); free(hu_off); free(dec);
}

/* ===== Microbench: time the partition primitives in isolation =====
 * Same loop bodies as p_partition / p_half_right in pivco_huffman_naive.c.
 * Hypothesis: p_partition compiles to a branchless select-store (always
 * writes one of left/right), while p_half_right has a conditional store
 * the compiler may not turn branchless. */

__attribute__((noinline))
static void mb_partition(const uint16_t *src, int n, const uint8_t *bm,
                          uint16_t *left, uint16_t *right,
                          int *lc, int *rc)
{
    int li = 0, ri = 0;
    for (int k = 0; k < n; k++) {
        int b = (bm[k >> 3] >> (k & 7)) & 1;
        if (b) right[ri++] = src[k];
        else   left [li++] = src[k];
    }
    *lc = li; *rc = ri;
}

__attribute__((noinline))
static int mb_half_right(const uint16_t *src, int n, const uint8_t *bm,
                          uint16_t *right)
{
    int ri = 0;
    for (int k = 0; k < n; k++) {
        int b = (bm[k >> 3] >> (k & 7)) & 1;
        right[ri] = src[k];
        ri += b;
    }
    return ri;
}

static void microbench_primitives(void) {
    const int N = 8192;
    const int ITERS = 200000;
    uint16_t *src  = aligned_alloc(64, (size_t)N * sizeof(uint16_t));
    uint16_t *L    = aligned_alloc(64, (size_t)N * sizeof(uint16_t));
    uint16_t *R    = aligned_alloc(64, (size_t)N * sizeof(uint16_t));
    uint8_t  *bm   = aligned_alloc(64, (size_t)(N + 7) / 8);
    for (int i = 0; i < N; i++) src[i] = (uint16_t)i;

    printf("primitive microbench (N=%d, %d iters/trial, best-of-3):\n", N, ITERS);
    for (int p_pct = 80; p_pct >= 50; p_pct -= 30) {
        /* Build a bitmap with p_pct%% bits set to 1. */
        uint64_t r = 0xC0FFEEABCDEF0123ULL;
        int set_count = 0;
        for (int i = 0; i < (N + 7) / 8; i++) {
            uint8_t b = 0;
            for (int j = 0; j < 8; j++) {
                r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                if ((int)(r % 100) < p_pct) { b |= (uint8_t)(1u << j); set_count++; }
            }
            bm[i] = b;
        }

        double best_partition_ns = 1e18, best_half_ns = 1e18;
        for (int t = 0; t < 3; t++) {
            int lc = 0, rc = 0;
            uint64_t t0 = now_ns();
            for (int i = 0; i < ITERS; i++) mb_partition(src, N, bm, L, R, &lc, &rc);
            uint64_t t1 = now_ns();
            double ns = (double)(t1 - t0) / ((double)ITERS * N);
            if (ns < best_partition_ns) best_partition_ns = ns;
        }
        for (int t = 0; t < 3; t++) {
            int rc = 0;
            uint64_t t0 = now_ns();
            for (int i = 0; i < ITERS; i++) rc = mb_half_right(src, N, bm, R);
            uint64_t t1 = now_ns();
            (void)rc;
            double ns = (double)(t1 - t0) / ((double)ITERS * N);
            if (ns < best_half_ns) best_half_ns = ns;
        }
        printf("  p1=%d%% (bits=1): p_partition %.3f ns/byte | "
               "p_half_right %.3f ns/byte | half/full %.2fx\n",
               p_pct, best_partition_ns, best_half_ns,
               best_half_ns / best_partition_ns);
    }
    free(src); free(L); free(R); free(bm);
}

int main(int argc, char **argv) {
    init_distributions();

    int skip_micro = 0;
    const char *csv_path = NULL;
    /* Tiny ad-hoc CLI: consumes --no-micro and --csv-out=PATH from
     * the front of argv, then treats any remaining arg as a
     * distribution name. */
    while (argc > 1) {
        if (strcmp(argv[1], "--no-micro") == 0) {
            skip_micro = 1;
            argc--; argv++;
        } else if (strncmp(argv[1], "--csv-out=", 10) == 0) {
            csv_path = argv[1] + 10;
            argc--; argv++;
        } else {
            break;
        }
    }
    if (csv_path) {
        g_csv = fopen(csv_path, "w");
        if (!g_csv) {
            fprintf(stderr, "failed to open %s for CSV output\n", csv_path);
            return 1;
        }
        fprintf(g_csv, "distribution,decoder,throughput_gbs,bpc\n");

        /* Auto-derive profile CSV path: foo.csv -> foo.profile.csv */
        size_t n = strlen(csv_path);
        char *prof = (char *)malloc(n + 16);
        if (n >= 4 && strcmp(csv_path + n - 4, ".csv") == 0) {
            memcpy(prof, csv_path, n - 4);
            strcpy(prof + n - 4, ".profile.csv");
        } else {
            snprintf(prof, n + 16, "%s.profile.csv", csv_path);
        }
        g_profile_csv = fopen(prof, "w");
        if (!g_profile_csv) {
            fprintf(stderr, "failed to open %s\n", prof);
            free(prof); return 1;
        }
        fprintf(g_profile_csv,
                "distribution,decoder,primitive,"
                "calls,elements,ticks,ns_per_elem,ns_per_call\n");
        free(prof);
    }

    g_tick_freq = pivco_prof_probe_tick_freq();

    if (!skip_micro) {
        microbench_primitives();
        printf("\n");
    }
    static const char *all_names[] = { "proba80", "prose_pride" };
    const char *single = (argc > 1) ? argv[1] : NULL;
    const char *const *names = single ? &single : all_names;
    size_t n_names = single ? 1 : sizeof(all_names)/sizeof(all_names[0]);
    printf("ph-td-naive vs huf0 (decode throughput)\n");
    printf("    %d blocks of %d symbols = %d KB each pass; %d iters; best-of-3\n\n",
           N_BLOCKS, PIVCO_BLOCK_SIZE,
           (N_BLOCKS * PIVCO_BLOCK_SIZE) / 1024, N_ITERS);
    for (size_t i = 0; i < n_names; i++) {
        run_one(names[i]);
    }
    if (g_csv)         fclose(g_csv);
    if (g_profile_csv) fclose(g_profile_csv);
    return 0;
}
