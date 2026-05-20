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

#define HUF_STATIC_LINKING_ONLY
#include "huf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

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

/* proba80: symbol 0 at 80%, rest split evenly across 1..255. */
static void freq_proba80(uint64_t freq[256]) {
    memset(freq, 0, 256 * sizeof(uint64_t));
    freq[0] = 80000;
    for (int i = 1; i < 256; i++) freq[i] = 20000 / 255;
}

/* prose_pride: Project Gutenberg "Pride and Prejudice" byte frequency
 * (copy of freq_prose_pride from bench/dist_real_freqs.h). */
static const uint64_t freq_prose_pride_local[256] = {
    0,0,0,0,0,0,0,0,0,124,12651,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    121541,1,2502,0,0,0,0,4197,21,21,0,0,12017,1903,4923,0,
    32,18,12,15,5,5,5,2,3,9,1196,2107,0,0,0,80,
    0,1577,1010,591,706,373,544,386,799,5,219,191,517,1024,536,503,
    1110,30,798,1394,1402,225,238,1242,11,1117,46,0,0,0,0,0,
    0,42537,8413,10881,21924,68257,9962,9938,32855,38275,505,4317,21795,13191,38008,40915,
    9132,492,29874,33889,46930,15028,5340,11671,776,11141,278,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

/* ---------- bench harness ---------- */

#define N_BLOCKS    256
#define N_ITERS     500

typedef struct {
    const char *name;
    void      (*freq_fn)(uint64_t [256]);
    const uint64_t *freq_static;
} dist_t;

static void run_one(const dist_t *d) {
    /* ---- frequencies ---- */
    uint64_t freq[256];
    if (d->freq_static) memcpy(freq, d->freq_static, sizeof(freq));
    else                d->freq_fn(freq);

    uint64_t cum[256];
    make_cum(freq, cum);
    if (cum[255] == 0) { fprintf(stderr, "%s: empty freq\n", d->name); return; }

    const size_t blksz = PIVCO_BLOCK_SIZE;
    const size_t total = blksz * N_BLOCKS;

    /* ---- generate data ---- */
    uint8_t *src = (uint8_t *)malloc(total);
    for (size_t i = 0; i < total; i++) src[i] = draw_byte(cum);

    /* ---- ph-td-naive: build table, encode, decode ---- */
    pivco_huffman_table_t table;
    int rc = pivco_huffman_build_table_naive(freq, &table);
    if (rc != PIVCO_OK) { fprintf(stderr, "%s: build_table_naive rc=%d\n", d->name, rc); free(src); return; }

    uint8_t *ph_enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE * (size_t)N_BLOCKS);
    size_t  *ph_off = (size_t *)calloc((size_t)N_BLOCKS + 1, sizeof(size_t));
    for (int b = 0; b < N_BLOCKS; b++) {
        size_t enc_len = PIVCO_MAX_ENCODED_SIZE;
        rc = pivco_huffman_encode_naive(src + (size_t)b * blksz,
                                         &table,
                                         ph_enc + ph_off[b],
                                         &enc_len);
        if (rc != PIVCO_OK) {
            fprintf(stderr, "%s: ph encode b=%d rc=%d\n", d->name, b, rc);
            free(src); free(ph_enc); free(ph_off); return;
        }
        ph_off[b + 1] = ph_off[b] + enc_len;
    }
    size_t ph_total_enc = ph_off[N_BLOCKS];

    /* Verify roundtrip on first block. */
    uint8_t *dec = (uint8_t *)malloc(blksz);
    size_t consumed = 0;
    rc = pivco_huffman_decode_naive(ph_enc, ph_off[1], &table, dec, &consumed);
    if (rc != PIVCO_OK || memcmp(src, dec, blksz) != 0) {
        fprintf(stderr, "%s: naive roundtrip FAILED rc=%d\n", d->name, rc);
        free(src); free(ph_enc); free(ph_off); free(dec); return;
    }

    /* Time ph-td-naive decode: full set of N_BLOCKS, best-of-3. */
    double ph_best_ns_per_byte = 1e18;
    for (int trial = 0; trial < 3; trial++) {
        uint64_t t0 = now_ns();
        for (int it = 0; it < N_ITERS; it++) {
            for (int b = 0; b < N_BLOCKS; b++) {
                pivco_huffman_decode_naive(
                    ph_enc + ph_off[b],
                    ph_off[b + 1] - ph_off[b],
                    &table, dec, &consumed);
            }
        }
        uint64_t t1 = now_ns();
        double ns_per_byte = (double)(t1 - t0) /
                              ((double)N_ITERS * (double)total);
        if (ns_per_byte < ph_best_ns_per_byte)
            ph_best_ns_per_byte = ns_per_byte;
    }

    /* ---- huf0: encode + decode ---- */
    /* Per-block encode (HUF_compress runs internal blocking for inputs
     * over 128KB; we feed blksz=8KB, well under that). */
    uint8_t *hu_enc = (uint8_t *)malloc(HUF_compressBound(blksz) * (size_t)N_BLOCKS);
    size_t  *hu_off = (size_t *)calloc((size_t)N_BLOCKS + 1, sizeof(size_t));
    int huf0_ok = 1;
    for (int b = 0; b < N_BLOCKS && huf0_ok; b++) {
        size_t enc_len = HUF_compress(hu_enc + hu_off[b],
                                       HUF_compressBound(blksz),
                                       src + (size_t)b * blksz,
                                       blksz);
        if (HUF_isError(enc_len) || enc_len == 0) {
            /* enc_len == 0 means HUF declined (e.g. uncompressible).
             * Treat as failure for this bench. */
            huf0_ok = 0;
            break;
        }
        hu_off[b + 1] = hu_off[b] + enc_len;
    }

    /* huf0 baseline = HUF_decompress4X2 (the 4-stream double-symbol
     * decoder).  This is the canonical comparison baseline -- huf0's
     * strongest decoder on the skewed distributions ph cares about. */
    double hu_x2_ns = -1.0;
    size_t hu_total_enc = 0;
    if (huf0_ok) {
        hu_total_enc = hu_off[N_BLOCKS];

        memset(dec, 0, blksz);
        size_t r = HUF_decompress4X2(dec, blksz, hu_enc, hu_off[1]);
        if (HUF_isError(r) || r != blksz || memcmp(src, dec, blksz) != 0) {
            fprintf(stderr, "%s: huf0_x2 roundtrip FAILED\n", d->name);
            huf0_ok = 0;
        }

        if (huf0_ok) {
            hu_x2_ns = 1e18;
            for (int trial = 0; trial < 3; trial++) {
                uint64_t t0 = now_ns();
                for (int it = 0; it < N_ITERS; it++) {
                    for (int b = 0; b < N_BLOCKS; b++) {
                        HUF_decompress4X2(dec, blksz,
                                            hu_enc + hu_off[b],
                                            hu_off[b + 1] - hu_off[b]);
                    }
                }
                uint64_t t1 = now_ns();
                double ns = (double)(t1 - t0) /
                              ((double)N_ITERS * (double)total);
                if (ns < hu_x2_ns) hu_x2_ns = ns;
            }
        }
    }

    /* ---- output ---- */
    double ph_gbs   = 1.0 / ph_best_ns_per_byte;
    double hu_gbs   = hu_x2_ns > 0 ? 1.0 / hu_x2_ns : 0.0;
    printf("%-14s | ph-naive %6.3f GB/s (%5.2f bpc) | "
           "huf0_x2 %6.3f GB/s (%5.2f bpc) | ph/huf0_x2 %.2fx\n",
           d->name,
           ph_gbs, (double)ph_total_enc * 8.0 / (double)total,
           hu_gbs, (double)hu_total_enc * 8.0 / (double)total,
           hu_gbs > 0 ? ph_gbs / hu_gbs : 0.0);

    free(src); free(ph_enc); free(ph_off);
    free(hu_enc); free(hu_off); free(dec);
}

int main(void) {
    static const dist_t dists[] = {
        { "proba80",     freq_proba80, NULL                       },
        { "prose_pride", NULL,         freq_prose_pride_local     },
    };
    printf("ph-td-naive vs huf0 (decode throughput)\n");
    printf("    %d blocks of %d symbols = %d KB each pass; %d iters; best-of-3\n\n",
           N_BLOCKS, PIVCO_BLOCK_SIZE,
           (N_BLOCKS * PIVCO_BLOCK_SIZE) / 1024, N_ITERS);
    for (size_t i = 0; i < sizeof(dists)/sizeof(dists[0]); i++) {
        run_one(&dists[i]);
    }
    return 0;
}
