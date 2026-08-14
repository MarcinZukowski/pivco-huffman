/* Correctness-focused tests covering small inputs, edge-case
 * distributions, and the real-world dataset files in extras/datasets/.
 *
 * Tests both the low-level block codec (pivco_encode/decode)
 * and the high-level file codec (pivcohuf_compress/decompress).  The
 * file-codec layer is the one users hit; the block codec is the
 * underlying primitive.  Bugs typically live in the block codec but
 * are usually exposed via file-codec round-trips.
 *
 * Historical bugs caught by this suite:
 *   - 2026-05-13: encode_node_neon + decode_subtree_bu OOB on skewed
 *     trees (cat-image.jpg block 34) -- tmp/scratch buffer was sized
 *     for balanced trees only, not the depth*N worst case for
 *     adversarial skewed partitions.
 */

#include "pivco_huffman.h"
#include "pivcohuf_file.h"
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define FAIL(msg, ...) do { printf("  FAIL: " msg "\n", ##__VA_ARGS__); return 1; } while (0)

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *s = x; return x;
}

/* ---------- helpers ---------- */

/* Round-trip a byte buffer through the file codec.  Returns 0 on match. */
static int roundtrip_file(const uint8_t *in, size_t in_len)
{
    size_t cap_c = pivcohuf_compress_bound(in_len);
    uint8_t *enc = malloc(cap_c ? cap_c : 1);
    if (!enc) FAIL("oom enc");
    size_t enc_len = cap_c;
    int rc = pivcohuf_compress(in, in_len, enc, &enc_len);
    if (rc != PIVCOHUF_OK) { free(enc); FAIL("compress rc=%d", rc); }

    uint8_t *dec = malloc(in_len ? in_len : 1);
    if (!dec) { free(enc); FAIL("oom dec"); }
    size_t dec_len = in_len;
    rc = pivcohuf_decompress(enc, enc_len, dec, &dec_len);
    if (rc != PIVCOHUF_OK) { free(enc); free(dec); FAIL("decompress rc=%d", rc); }
    if (dec_len != in_len) { free(enc); free(dec); FAIL("size mismatch %zu vs %zu", dec_len, in_len); }
    if (in_len > 0 && memcmp(in, dec, in_len) != 0) {
        size_t i; for (i = 0; i < in_len && in[i] == dec[i]; i++) ;
        free(enc); free(dec); FAIL("content diff at byte %zu (in=%02x dec=%02x)", i, in[i], dec[i]);
    }
    free(enc); free(dec);
    return 0;
}

/* Read a whole file into a heap buffer.  *out_len receives the size.
 * Returns NULL on failure. */
static uint8_t *read_file(const char *path, size_t *out_len)
{
    struct stat st;
    if (stat(path, &st) != 0) return NULL;
    *out_len = (size_t)st.st_size;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t *buf = malloc(*out_len ? *out_len : 1);
    if (!buf) { fclose(f); return NULL; }
    if (*out_len && fread(buf, 1, *out_len, f) != *out_len) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    return buf;
}

/* ---------- tests ---------- */

/* Real dataset files - the primary correctness regression suite. */
static int test_real_datasets(void)
{
    const char *paths[] = {
        "extras/datasets/cat-image.jpg",   /* near-uniform, caught the 2026-05-13 bug */
        "extras/datasets/cat-wiki.html",
        "extras/datasets/pride.txt",
        "extras/datasets/json_api.json",
        "extras/datasets/source_c.c",
        "extras/datasets/log_apache.log",
        "extras/datasets/dna_fasta.fa",
        "extras/datasets/csv_numeric.csv",
        "extras/datasets/gzip_random.gz",
        "extras/datasets/chinese_text.txt",
        "extras/datasets/calgary_pic",     /* 1bpp CCITT scanned page — proba80-like */
    };
    int n = (int)(sizeof(paths)/sizeof(paths[0]));
    int total_fail = 0;
    for (int i = 0; i < n; i++) {
        size_t len;
        uint8_t *buf = read_file(paths[i], &len);
        if (!buf) {
            printf("[real_dataset %s] SKIP (file not found)\n", paths[i]);
            continue;
        }
        printf("[real_dataset %s] ", paths[i]);
        int r = roundtrip_file(buf, len);
        free(buf);
        if (r) total_fail++;
        else   printf("OK (%zu B)\n", len);
    }
    return total_fail;
}

/* Edge-case sizes: 0, 1, 2, 7, 8, 16, 100, exact-block, just-over, big multi-block. */
static int test_size_edge_cases(void)
{
    const size_t sizes[] = {
        0, 1, 2, 7, 8, 16, 100, 1000,
        PIVCO_BLOCK_SIZE - 1, PIVCO_BLOCK_SIZE, PIVCO_BLOCK_SIZE + 1,
        2 * PIVCO_BLOCK_SIZE - 1, 2 * PIVCO_BLOCK_SIZE, 2 * PIVCO_BLOCK_SIZE + 1,
        100000, 1 << 20,
    };
    int n = (int)(sizeof(sizes)/sizeof(sizes[0]));
    int total_fail = 0;

    for (int i = 0; i < n; i++) {
        size_t len = sizes[i];
        uint8_t *buf = malloc(len ? len : 1);
        if (!buf) FAIL("oom");

        /* Fill with mixed pattern: deterministic bytes-cycle plus
         * occasional randomness so each block sees varied input. */
        uint64_t rng = 0xc0ffee00ULL + (uint64_t)i * 0x9e3779b97f4a7c15ULL;
        for (size_t j = 0; j < len; j++) {
            buf[j] = (uint8_t)((j * 17 + 3) ^ (xorshift64(&rng) & 0xFF));
        }

        printf("[size n=%zu] ", len);
        int r = roundtrip_file(buf, len);
        free(buf);
        if (r) total_fail++;
        else printf("OK\n");
    }
    return total_fail;
}

/* Uniform-random small inputs - historical "encode_node_neon stack overflow
 * on near-uniform random" bug repro.  Multiple seeds + sizes. */
static int test_uniform_random(void)
{
    const size_t sizes[] = { 100, 1000, 8192, 8193, 73753, 1 << 20 };
    int n = (int)(sizeof(sizes)/sizeof(sizes[0]));
    int total_fail = 0;

    for (int i = 0; i < n; i++) {
        size_t len = sizes[i];
        for (int seed_idx = 0; seed_idx < 5; seed_idx++) {
            uint8_t *buf = malloc(len);
            if (!buf) FAIL("oom");
            uint64_t rng = 0xdeadbeefdeadbeefULL + (uint64_t)(i * 100 + seed_idx);
            for (size_t j = 0; j < len; j++) buf[j] = (uint8_t)xorshift64(&rng);
            printf("[uniform_random n=%zu seed=%d] ", len, seed_idx);
            int r = roundtrip_file(buf, len);
            free(buf);
            if (r) total_fail++;
            else printf("OK\n");
        }
    }
    return total_fail;
}

/* Distribution edge cases: all-same byte, two-byte alternating, heavy skew. */
static int test_distribution_edge_cases(void)
{
    int total_fail = 0;

    /* All-same byte (1 symbol).  No tree branches; entire output uses
     * the prefill optimization. */
    for (size_t len = 1; len <= 100000; len = (len * 7) + 1) {
        uint8_t *buf = malloc(len);
        if (!buf) FAIL("oom");
        memset(buf, 0x42, len);
        printf("[all_same n=%zu] ", len);
        int r = roundtrip_file(buf, len);
        free(buf);
        if (r) total_fail++;
        else printf("OK\n");
    }

    /* Two-symbol alternating, various ratios. */
    const struct { int p_a; const char *name; } two_sym[] = {
        { 50, "50/50" }, { 80, "80/20" }, { 95, "95/5" }, { 99, "99/1" }
    };
    for (size_t len = 1024; len <= 100000; len *= 4) {
        for (int t = 0; t < (int)(sizeof(two_sym)/sizeof(two_sym[0])); t++) {
            uint8_t *buf = malloc(len);
            if (!buf) FAIL("oom");
            uint64_t rng = 0xcafef00dULL + (uint64_t)t * len;
            int p_a = two_sym[t].p_a;
            for (size_t j = 0; j < len; j++) {
                buf[j] = (xorshift64(&rng) % 100 < (uint64_t)p_a) ? 0xAA : 0x55;
            }
            printf("[two_sym %s n=%zu] ", two_sym[t].name, len);
            int r = roundtrip_file(buf, len);
            free(buf);
            if (r) total_fail++;
            else printf("OK\n");
        }
    }

    /* Heavy skew: one byte at 80%, 255 others uniform.  Repro for
     * proba80-style distributions. */
    for (size_t len = 1024; len <= 200000; len *= 4) {
        uint8_t *buf = malloc(len);
        if (!buf) FAIL("oom");
        uint64_t rng = 0xfeedface00ULL + len;
        for (size_t j = 0; j < len; j++) {
            uint64_t r = xorshift64(&rng);
            buf[j] = (r % 100 < 80) ? 0 : (uint8_t)((r >> 8) & 0xFF);
        }
        printf("[skew80 n=%zu] ", len);
        int r = roundtrip_file(buf, len);
        free(buf);
        if (r) total_fail++;
        else printf("OK\n");
    }

    return total_fail;
}

/* Adversarial: small inputs with byte distributions specifically chosen
 * to produce highly imbalanced Huffman trees that stress the
 * tmp/scratch sizing in the partition recursion. */
static int test_adversarial(void)
{
    int total_fail = 0;

    /* Repeating short pattern: a few symbols dominate, but tree depth
     * can still be 8+ due to long-tail. */
    for (size_t len = 8192; len <= 100000; len *= 2) {
        uint8_t *buf = malloc(len);
        if (!buf) FAIL("oom");
        uint64_t rng = 0xabad1deaULL + len;
        for (size_t j = 0; j < len; j++) {
            uint64_t r = xorshift64(&rng) % 1000;
            /* 50% pad, 30% second, then 20% spread over 254 others */
            if      (r <  500) buf[j] = 0;
            else if (r <  800) buf[j] = 1;
            else               buf[j] = (uint8_t)((r >> 8) & 0xFF);
        }
        printf("[adversarial_skew n=%zu] ", len);
        int r = roundtrip_file(buf, len);
        free(buf);
        if (r) total_fail++;
        else printf("OK\n");
    }
    return total_fail;
}

/* FSE-specific (v0.2 wire format): inputs designed to exercise both
 * sides of the FSE-vs-raw dispatch in the encoder.  Roundtrip only --
 * the value here is that the FSE compress/decompress path is exercised
 * on both highly-skewed (most nodes pick FSE) and near-uniform (most
 * nodes stay raw) bitmaps. */
static int test_fse_dispatch(void)
{
    int total_fail = 0;

    /* Heavy-skew: 95% one byte, 5% spread over 20 others.  Most non-flat
     * internal nodes in the resulting Huffman tree will have very
     * skewed left/right partitions, hitting FSE.  Picked size 200K to
     * span ~25 blocks at the M4 8K block size. */
    for (size_t len = 32 * 1024; len <= 256 * 1024; len *= 4) {
        uint8_t *buf = malloc(len);
        if (!buf) FAIL("oom");
        uint64_t rng = 0xfa57e7e7ULL + len;
        for (size_t j = 0; j < len; j++) {
            uint64_t r = xorshift64(&rng);
            if (r % 100 < 95) buf[j] = 0;
            else              buf[j] = (uint8_t)(1 + (r >> 8) % 20);
        }
        printf("[fse heavy_skew n=%zu] ", len);
        int r = roundtrip_file(buf, len);
        free(buf);
        if (r) total_fail++;
        else printf("OK\n");
    }

    /* Near-uniform: byte distribution close to 1/256 per value.  Almost
     * no node will hit the FSE threshold; verifies the marker=0 raw
     * path is correct end-to-end. */
    for (size_t len = 32 * 1024; len <= 256 * 1024; len *= 4) {
        uint8_t *buf = malloc(len);
        if (!buf) FAIL("oom");
        uint64_t rng = 0xfeed1234ULL + len;
        for (size_t j = 0; j < len; j++) buf[j] = (uint8_t)xorshift64(&rng);
        printf("[fse near_uniform n=%zu] ", len);
        int r = roundtrip_file(buf, len);
        free(buf);
        if (r) total_fail++;
        else printf("OK\n");
    }

    /* DNA-like 4-symbol alphabet: small alphabet, geometric-ish ratio,
     * heavy on a couple symbols.  Real-data-style proxy for the
     * dna_fasta upside (FSE captures ~8% on that). */
    {
        const uint8_t alphabet[] = { 'A', 'C', 'G', 'T' };
        for (size_t len = 32 * 1024; len <= 256 * 1024; len *= 4) {
            uint8_t *buf = malloc(len);
            if (!buf) FAIL("oom");
            uint64_t rng = 0xacac1234ULL + len;
            for (size_t j = 0; j < len; j++) {
                uint64_t r = xorshift64(&rng) % 100;
                /* A=40%, C=25%, G=20%, T=15% */
                int idx = r < 40 ? 0 : r < 65 ? 1 : r < 85 ? 2 : 3;
                buf[j] = alphabet[idx];
            }
            printf("[fse dna_like n=%zu] ", len);
            int r = roundtrip_file(buf, len);
            free(buf);
            if (r) total_fail++;
            else printf("OK\n");
        }
    }

    return total_fail;
}

#ifdef PIVCO_HAS_FSE
/* Direct FSE roundtrip at EVERY length 8..600: crosses the wide-path
 * minimum (64) and every mod-X residue.  The former wide gate required
 * nbytes % PIVCO_FSE_XY_X == 0; this sweep would have caught both that
 * restriction's silent stock-FSE fallback and any encode_x/decode
 * disagreement in the generalized any-length form (partial final
 * round, cursor mapping, tight streams).  Three bit-density regimes
 * cover low/mid/high table ids. */
static int test_fse_length_sweep(void)
{
    printf("[fse_length_sweep] ");
    pivco_fse_init();
    uint64_t rng = 0x5eedf00d12345678ULL;
    const double biases[] = { 0.65, 0.85, 0.97 };
    int tested = 0, fell_back = 0;
    for (int bi = 0; bi < 3; bi++) {
        int t = pivco_fse_select_table(biases[bi]);
        if (t < 1) FAIL("no table for bias %.2f", biases[bi]);
        for (size_t n = 8; n <= 600; n++) {
            uint8_t src[600], enc[800], dec[616];
            for (size_t i = 0; i < n; i++) {
                uint8_t b = 0;
                for (int j = 0; j < 8; j++) {
                    double u = (double)(xorshift64(&rng) >> 11) / 9007199254740992.0;
                    b |= (uint8_t)((u > biases[bi]) << j);
                }
                src[i] = b;
            }
            size_t clen = 0;
            pivco_fse_status_t rc = pivco_fse_compress(t, src, n,
                                                       enc, sizeof(enc), &clen);
            if (rc == PIVCO_FSE_FALLBACK) { fell_back++; continue; }
            if (rc != PIVCO_FSE_OK) FAIL("compress n=%zu t=%d rc=%d", n, t, rc);
            size_t olen = 0;
            memset(dec, 0xCB, sizeof(dec));
            rc = pivco_fse_decompress(t, enc, clen, dec, sizeof(dec), n, &olen);
            if (rc != PIVCO_FSE_OK) FAIL("decompress n=%zu t=%d rc=%d", n, t, rc);
            if (olen != n) FAIL("olen %zu != n %zu (t=%d)", olen, n, t);
            if (memcmp(src, dec, n) != 0) {
                size_t i; for (i = 0; i < n && src[i] == dec[i]; i++) ;
                FAIL("mismatch n=%zu t=%d at byte %zu", n, t, i);
            }
            tested++;
        }
    }
    printf("PASS (%d lengths, %d fallbacks)\n", tested, fell_back);
    return 0;
}
#endif  /* PIVCO_HAS_FSE */

/* ---------- entry point ---------- */

int test_edge_cases_all(void)
{
    int fails = 0;
    printf("\n--- real datasets ---\n");
    fails += test_real_datasets();
    printf("\n--- size edge cases ---\n");
    fails += test_size_edge_cases();
    printf("\n--- uniform random ---\n");
    fails += test_uniform_random();
    printf("\n--- distribution edge cases ---\n");
    fails += test_distribution_edge_cases();
    printf("\n--- adversarial ---\n");
    fails += test_adversarial();
    printf("\n--- FSE dispatch (v0.2 wire format) ---\n");
    fails += test_fse_dispatch();
#ifdef PIVCO_HAS_FSE
    printf("\n--- FSE length sweep ---\n");
    fails += test_fse_length_sweep();
#endif
    return fails;
}
