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

/* Nibble table (PIVCO_FSE_NIBBLE_ID): the payload carries its
 * own FSE table description, so unlike the static ids the decoder is
 * handed nothing but the expected byte count.  Sweep every length
 * across bit densities the static schedule models badly (near-50/50)
 * as well as ones it models well, and go through both the direct
 * *_nibble entry points and the pivco_fse_compress/decompress
 * table-id dispatch -- the wire decoder only ever reaches the nibble
 * path via the latter. */
static int test_fse_nibble(void)
{
    printf("[fse_nibble] ");
    pivco_fse_init();
    uint64_t rng = 0x0dd1e5c0ffeeb0b1ULL;
    /* 0.50 is the case the static tables refuse outright; the nibble
     * histogram there is still non-flat for structured bitmaps. */
    const double biases[] = { 0.50, 0.60, 0.80, 0.95 };
    int tested = 0, fell_back = 0;
    for (int bi = 0; bi < 4; bi++) {
        for (size_t n = 8; n <= 600; n++) {
            uint8_t src[600], enc[1400], dec[616];
            for (size_t i = 0; i < n; i++) {
                uint8_t b = 0;
                for (int j = 0; j < 8; j++) {
                    double u = (double)(xorshift64(&rng) >> 11) / 9007199254740992.0;
                    b |= (uint8_t)((u > biases[bi]) << j);
                }
                src[i] = b;
            }
            size_t clen = 0;
            pivco_fse_status_t rc = pivco_fse_compress_nibble(src, n, enc,
                                                               sizeof(enc), &clen);
            if (rc == PIVCO_FSE_FALLBACK) { fell_back++; continue; }
            if (rc != PIVCO_FSE_OK) FAIL("compress n=%zu rc=%d", n, rc);
            if (clen >= n) FAIL("nibble table committed a payload >= raw (n=%zu clen=%zu)", n, clen);

            /* Route the decode through the table-id dispatch, exactly as
             * wire_read_bitmap does. */
            size_t olen = 0;
            memset(dec, 0xCB, sizeof(dec));
            rc = pivco_fse_decompress(PIVCO_FSE_NIBBLE_ID, enc, clen,
                                      dec, sizeof(dec), n, &olen);
            if (rc != PIVCO_FSE_OK) FAIL("decompress n=%zu rc=%d", n, rc);
            if (olen != n) FAIL("olen %zu != n %zu", olen, n);
            if (memcmp(src, dec, n) != 0) {
                size_t i; for (i = 0; i < n && src[i] == dec[i]; i++) ;
                FAIL("mismatch n=%zu at byte %zu", n, i);
            }

            /* Same payload via the compress-side dispatch must be
             * byte-identical to the direct call. */
            uint8_t enc2[1400];
            size_t clen2 = 0;
            rc = pivco_fse_compress(PIVCO_FSE_NIBBLE_ID, src, n,
                                    enc2, sizeof(enc2), &clen2);
            if (rc != PIVCO_FSE_OK || clen2 != clen || memcmp(enc, enc2, clen) != 0)
                FAIL("compress dispatch differs from direct call (n=%zu)", n);

            /* NB: no truncation test here.  An FSE bitstream carries no
             * integrity check, so a payload cut short can still decode
             * to dst_expected bytes of garbage.  The wire format doesn't
             * lean on detecting that -- fse_len is stored explicitly
             * ahead of the payload. */
            tested++;
        }
    }
    if (tested == 0) FAIL("nibble path never committed -- test is vacuous");
    printf("PASS (%d lengths, %d fallbacks)\n", tested, fell_back);
    return 0;
}

/* k=1 bit-context table (PIVCO_FSE_K1_ID): one recipe byte on the wire,
 * tables from the catalog.  Sweep every length from 16 to 615 (both
 * sides of the PIVCO_K1_SPLIT_MIN segment split) plus a set of large
 * sizes, over i.i.d. densities and Markov chains with strong
 * run persistence either way; go through the direct entry points and
 * the table-id dispatch, which is the only path the wire decoder
 * takes. */
static int test_fse_k1(void)
{
    printf("[fse_k1] ");
    uint64_t rng = 0x5eed0f0dd15c0b1eULL;
    /* {P(1 | prev 0), P(1 | prev 1)}: i.i.d. densities, then chains. */
    const double chains[][2] = {
        { 0.50, 0.50 }, { 0.20, 0.20 }, { 0.05, 0.05 },
        { 0.90, 0.10 }, { 0.05, 0.95 }, { 0.30, 0.70 }, { 0.98, 0.98 },
    };
    const size_t big[] = { 1000, 1023, 1024, 1025, 1040, 2048, 4095, 4096, 16384, 32768 };
    static uint8_t src[32768], enc[32768 + 64], enc2[32768 + 64], dec[32768 + 16];
    int tested = 0, fell_back = 0;
    for (size_t ci = 0; ci < sizeof(chains) / sizeof(chains[0]); ci++) {
        for (size_t li = 0; li < 600 + sizeof(big) / sizeof(big[0]); li++) {
            size_t n = li < 600 ? 16 + li : big[li - 600];
            int prev = 0;
            for (size_t i = 0; i < n; i++) {
                uint8_t b = 0;
                for (int j = 0; j < 8; j++) {
                    double u = (double)(xorshift64(&rng) >> 11) / 9007199254740992.0;
                    int bit = u < chains[ci][prev];
                    b |= (uint8_t)(bit << j);
                    prev = bit;
                }
                src[i] = b;
            }
            size_t clen = 0;
            pivco_fse_status_t rc = pivco_k1_compress(src, n, enc, sizeof(enc), 0, &clen);
            if (rc == PIVCO_FSE_FALLBACK) { fell_back++; continue; }
            if (rc != PIVCO_FSE_OK) FAIL("compress n=%zu rc=%d", n, rc);
            if (clen >= n) FAIL("k1 committed a payload >= raw (n=%zu clen=%zu)", n, clen);

            size_t olen = 0;
            memset(dec, 0xCB, sizeof(dec));
            rc = pivco_fse_decompress(PIVCO_FSE_K1_ID, enc, clen, dec, sizeof(dec), n, &olen);
            if (rc != PIVCO_FSE_OK) FAIL("decompress n=%zu rc=%d", n, rc);
            if (olen != n) FAIL("olen %zu != n %zu", olen, n);
            if (memcmp(src, dec, n) != 0) {
                size_t i; for (i = 0; i < n && src[i] == dec[i]; i++) ;
                FAIL("mismatch n=%zu chain=%zu at byte %zu", n, ci, i);
            }
            size_t clen2 = 0;
            rc = pivco_fse_compress(PIVCO_FSE_K1_ID, src, n, enc2, sizeof(enc2), &clen2);
            if (rc != PIVCO_FSE_OK || clen2 != clen || memcmp(enc, enc2, clen) != 0)
                FAIL("compress dispatch differs from direct call (n=%zu)", n);
            /* Damaged payloads: a tANS bitstream carries no integrity
             * check, so a payload cut short can still decode to n bytes
             * of garbage (fse_len on the wire is what bounds the read);
             * what must hold is that every read stays inside the payload
             * (ASan) and a success always fills n bytes. */
            if (li % 37 == 0) {
                const size_t cuts[] = { 1, 2, clen / 2, clen - 1 };
                for (size_t c = 0; c < sizeof(cuts) / sizeof(cuts[0]); c++) {
                    if (cuts[c] >= clen) continue;
                    olen = 0;
                    rc = pivco_fse_decompress(PIVCO_FSE_K1_ID, enc, cuts[c], dec, sizeof(dec), n, &olen);
                    if (rc == PIVCO_FSE_OK && olen != n) FAIL("truncated payload: OK with olen %zu != n %zu", olen, n);
                }
                memcpy(enc2, enc, clen);
                for (size_t i = 0; i < clen; i++) enc2[i] ^= (uint8_t)(xorshift64(&rng) >> 56);
                olen = 0;
                rc = pivco_fse_decompress(PIVCO_FSE_K1_ID, enc2, clen, dec, sizeof(dec), n, &olen);
                if (rc == PIVCO_FSE_OK && olen != n) FAIL("garbage payload: OK with olen %zu != n %zu", olen, n);
            }
            tested++;
        }
    }
    if (tested == 0) FAIL("k1 path never committed -- test is vacuous");
    printf("PASS (%d lengths, %d fallbacks)\n", tested, fell_back);
    return 0;
}

/* The k=2 coder on order-2 chains: round trip through the direct call
 * and the dispatch, damaged payloads stay in bounds. */
static int test_fse_k2(void)
{
    printf("[fse_k2] ");
    uint64_t rng = 0x2b1a5eed0f0dd15cULL;
    /* P(1 | prev two bits) indexed (newest << 1) | older. */
    const double chains[][4] = {
        { 0.50, 0.50, 0.50, 0.50 }, { 0.20, 0.20, 0.20, 0.20 },
        { 0.05, 0.05, 0.05, 0.05 }, { 0.90, 0.10, 0.90, 0.10 },
        { 0.05, 0.30, 0.70, 0.95 }, { 0.02, 0.98, 0.02, 0.98 },
        { 0.98, 0.50, 0.50, 0.02 },
    };
    const size_t big[] = { 1000, 1023, 1024, 1025, 1040, 2048, 4095, 4096, 16384, 32768 };
    static uint8_t src[32768], enc[32768 + 64], enc2[32768 + 64], dec[32768 + 16];
    int tested = 0, fell_back = 0;
    for (size_t ci = 0; ci < sizeof(chains) / sizeof(chains[0]); ci++) {
        for (size_t li = 0; li < 600 + sizeof(big) / sizeof(big[0]); li++) {
            size_t n = li < 600 ? 16 + li : big[li - 600];
            int ctx = 0;
            for (size_t i = 0; i < n; i++) {
                uint8_t b = 0;
                for (int j = 0; j < 8; j++) {
                    double u = (double)(xorshift64(&rng) >> 11) / 9007199254740992.0;
                    int bit = u < chains[ci][ctx];
                    b |= (uint8_t)(bit << j);
                    ctx = (bit << 1) | (ctx >> 1);
                }
                src[i] = b;
            }
            size_t clen = 0;
            pivco_fse_status_t rc = pivco_k2_compress(src, n, enc, sizeof(enc), 0, &clen);
            if (rc == PIVCO_FSE_FALLBACK) { fell_back++; continue; }
            if (rc != PIVCO_FSE_OK) FAIL("compress n=%zu rc=%d", n, rc);
            if (clen >= n) FAIL("k2 committed a payload >= raw (n=%zu clen=%zu)", n, clen);

            size_t olen = 0;
            memset(dec, 0xCB, sizeof(dec));
            rc = pivco_fse_decompress(PIVCO_FSE_K2_ID, enc, clen, dec, sizeof(dec), n, &olen);
            if (rc != PIVCO_FSE_OK) FAIL("decompress n=%zu rc=%d", n, rc);
            if (olen != n) FAIL("olen %zu != n %zu", olen, n);
            if (memcmp(src, dec, n) != 0) {
                size_t i; for (i = 0; i < n && src[i] == dec[i]; i++) ;
                FAIL("mismatch n=%zu chain=%zu at byte %zu", n, ci, i);
            }
            size_t clen2 = 0;
            rc = pivco_fse_compress(PIVCO_FSE_K2_ID, src, n, enc2, sizeof(enc2), &clen2);
            if (rc != PIVCO_FSE_OK || clen2 != clen || memcmp(enc, enc2, clen) != 0)
                FAIL("compress dispatch differs from direct call (n=%zu)", n);
            if (li % 37 == 0) {
                const size_t cuts[] = { 1, 2, clen / 2, clen - 1 };
                for (size_t c = 0; c < sizeof(cuts) / sizeof(cuts[0]); c++) {
                    if (cuts[c] >= clen) continue;
                    olen = 0;
                    rc = pivco_fse_decompress(PIVCO_FSE_K2_ID, enc, cuts[c], dec, sizeof(dec), n, &olen);
                    if (rc == PIVCO_FSE_OK && olen != n) FAIL("truncated payload: OK with olen %zu != n %zu", olen, n);
                }
                memcpy(enc2, enc, clen);
                for (size_t i = 0; i < clen; i++) enc2[i] ^= (uint8_t)(xorshift64(&rng) >> 56);
                olen = 0;
                rc = pivco_fse_decompress(PIVCO_FSE_K2_ID, enc2, clen, dec, sizeof(dec), n, &olen);
                if (rc == PIVCO_FSE_OK && olen != n) FAIL("garbage payload: OK with olen %zu != n %zu", olen, n);
            }
            tested++;
        }
    }
    if (tested == 0) FAIL("k2 path never committed -- test is vacuous");
    printf("PASS (%d lengths, %d fallbacks)\n", tested, fell_back);
    return 0;
}

/* k=1 on the wire, through the file API: the codec's static -> nibble ->
 * k1 chain, codec_fse_commit with id 52, and the decoder's dispatch of
 * marker 52 on both region kinds.  With one table for the whole input
 * (seg_blocks 0), a locally skewed block under a globally balanced
 * histogram gives regions the bit-pair model wins and the static
 * schedule cannot touch (its skew gate needs a lopsided bitmap):
 *   nodes -- a 2-state chain over {a, b} with rare extras: the root
 *            bitmap is balanced but runs
 *   flat  -- 16 equally frequent symbols (flat root, D = 4), each 32K
 *            block dominated by one of them: 16 KB bodies of one
 *            repeated code, which also drives codec_fse_try's staging
 *            past its stack budget onto the heap. */
static int test_fse_kn_wire(int k)
{
    const int id = k == 1 ? PIVCO_FSE_K1_ID : PIVCO_FSE_K2_ID;
    printf("[fse_k%d_wire] ", k);
    uint64_t rng = 0xC0FFEE0DDBA11ULL;
    uint64_t commit[PIVCO_FSE_STATS_SLOTS], attempt[PIVCO_FSE_STATS_SLOTS];
    uint64_t bin[PIVCO_FSE_STATS_SLOTS], bout[PIVCO_FSE_STATS_SLOTS];
    pivco_cfg_t cfg = pivco_cfg_default;
    cfg.fse_enabled = 1;
    cfg.fse_nibble_enabled = 0;
    cfg.fse_k1_enabled = k == 1;
    cfg.fse_k2_enabled = k == 2;

    for (int part = 0; part < 2; part++) {
        const size_t B = part == 0 ? (size_t)PIVCO_BLOCK_SIZE : 32768;
        const size_t N = part == 0 ? 200000 : 16 * 32768;
        uint8_t *in = malloc(N);
        if (!in) FAIL("oom in");
        if (part == 0) {
            int state = 0;
            for (size_t i = 0; i < N; i++) {
                uint64_t x = xorshift64(&rng);
                if ((x & 127) == 0) state ^= 1;                 /* runs of ~128 */
                uint8_t r = (uint8_t)((x >> 8) % 20);
                in[i] = r < 18 ? (uint8_t)('a' + state)
                               : (uint8_t)('c' + 2 * state + (r & 1));
            }
        } else {
            for (size_t i = 0; i < N; i++) {
                uint64_t x = xorshift64(&rng);
                uint8_t dominant = (uint8_t)((i / 32768) & 15);
                in[i] = (x & 15) ? dominant : (uint8_t)((x >> 8) & 15);
            }
        }
        size_t cap = pivcohuf_compress_bound_blk(N, B);
        uint8_t *enc = malloc(cap), *dec = malloc(N);
        if (!enc || !dec) { free(in); free(enc); free(dec); FAIL("oom bufs"); }

        pivco_fse_stats_reset();
        size_t enc_len = cap;
        int rc = pivcohuf_compress_seg(in, N, enc, &enc_len, &cfg, B, 0, NULL);
        if (rc != PIVCOHUF_OK) { free(in); free(enc); free(dec); FAIL("part %d: compress rc=%d", part, rc); }
        pivco_fse_stats_get(commit, attempt, bin, bout);
        if (commit[id] == 0) {
            free(in); free(enc); free(dec);
            FAIL("part %d: k%d never committed (attempts %llu) -- test is vacuous",
                 part, k, (unsigned long long)attempt[id]);
        }
        size_t dec_len = N;
        rc = pivcohuf_decompress(enc, enc_len, dec, &dec_len);
        if (rc != PIVCOHUF_OK || dec_len != N || memcmp(in, dec, N) != 0) {
            free(in); free(enc); free(dec);
            FAIL("part %d: roundtrip failed (rc=%d len=%zu)", part, rc, dec_len);
        }
        printf("%s%llu k%d regions", part ? ", flat: " : "nodes: ",
               (unsigned long long)commit[id], k);
        free(in); free(enc); free(dec);
    }
    printf(" PASS\n");
    return 0;
}
#endif  /* PIVCO_HAS_FSE */

/* Blocks of a pivcohuf stream that carry their own code-length table
 * (BLOCK_FLAGS NEW_TABLE, v0.10). */
static size_t count_table_blocks(const uint8_t *stream, size_t len)
{
    size_t p = PIVCOHUF_HEADER_SIZE + 11 + 128, n = 0;
    while (p + 4 <= len) {
        uint32_t f = (uint32_t)stream[p] | ((uint32_t)stream[p + 1] << 8)
                   | ((uint32_t)stream[p + 2] << 16) | ((uint32_t)stream[p + 3] << 24);
        p += 4;
        if ((f >> PIVCOHUF_BLOCK_FLAGS_SHIFT) & PIVCOHUF_BLOCK_FLAG_NEW_TABLE) { n++; p += 128; }
        p += f & PIVCOHUF_BLOCK_LEN_MASK;
    }
    return n;
}

/* ---------- flat-layout FLAGS byte in the pivcohuf container ---------- */

/* The container records cfg.flat_layout in the body FLAGS byte, so
 * decompress needs no matching cfg; unknown FLAGS bits are refused;
 * and a v0.8 stream (no FLAGS byte, natural flat regions) still
 * decodes -- synthesized here from a v0.9 natural stream by dropping
 * the byte, since natural v0.9 bodies are otherwise byte-identical. */
static int test_flat_layout_file(void)
{
    printf("[flat_layout_file] ");
    enum { N = 200000, FLAGS_OFF = PIVCOHUF_HEADER_SIZE + 10 };
    uint8_t *in = malloc(N);
    if (!in) FAIL("oom in");
    uint64_t rng = 0xF1A6BEEF00D5EEDULL;
    for (size_t i = 0; i < N; i++) {
        uint64_t x = xorshift64(&rng);
        in[i] = (uint8_t)(x % 3 ? x % 16 : x % 256);   /* flat-heavy mix */
    }

    size_t cap = pivcohuf_compress_bound(N);
    uint8_t *enc = malloc(cap), *dec = malloc(N);
    if (!enc || !dec) { free(in); free(enc); free(dec); FAIL("oom bufs"); }

    static const struct { pivco_flat_layout_t layout; uint8_t flags; } arms[] = {
        { PIVCO_FLAT_NATURAL,      0x00 },
        { PIVCO_FLAT_VERTICAL,     0x01 },
        { PIVCO_FLAT_VERTICAL_128, 0x02 },
    };
    size_t arm_len[3] = {0, 0, 0};
    for (size_t a = 0; a < sizeof(arms) / sizeof(arms[0]); a++) {
        pivco_cfg_t cfg = pivco_cfg_default;
        cfg.fse_nibble_enabled = 1;   /* opt in: this test exercises the FSE-flat path */
        cfg.fse_k1_enabled = 1;
        cfg.flat_layout = arms[a].layout;
        size_t enc_len = cap;
        int rc = pivcohuf_compress_cfg(in, N, enc, &enc_len, &cfg,
                                       PIVCO_BLOCK_SIZE, NULL);
        if (rc != PIVCOHUF_OK) FAIL("layout %d: compress rc=%d",
                                    (int)arms[a].layout, rc);
        if (enc[FLAGS_OFF] != arms[a].flags)
            FAIL("layout %d: FLAGS byte %02x, want %02x",
                 (int)arms[a].layout, enc[FLAGS_OFF], arms[a].flags);
        size_t dec_len = N;
        rc = pivcohuf_decompress(enc, enc_len, dec, &dec_len);
        if (rc != PIVCOHUF_OK) FAIL("layout %d: decompress rc=%d",
                                    (int)arms[a].layout, rc);
        if (dec_len != N || memcmp(in, dec, N) != 0)
            FAIL("layout %d: roundtrip diff", (int)arms[a].layout);

        /* Strictness: any unknown set FLAGS bit is a refusal.  Checksums
         * are currently disabled, so the byte can be patched in place. */
        enc[FLAGS_OFF] |= PIVCOHUF_FLAG_QUAD_NODES;
        dec_len = N;
        rc = pivcohuf_decompress(enc, enc_len, dec, &dec_len);
        if (rc != PIVCOHUF_ERR_BAD_VERSION)
            FAIL("layout %d: unknown FLAGS bit gave rc=%d, want BAD_VERSION",
                 (int)arms[a].layout, rc);
        enc[FLAGS_OFF] = arms[a].flags;

        /* The reserved layout value (3) is a refusal too. */
        enc[FLAGS_OFF] = 0x03;
        dec_len = N;
        rc = pivcohuf_decompress(enc, enc_len, dec, &dec_len);
        if (rc != PIVCOHUF_ERR_BAD_VERSION)
            FAIL("layout %d: reserved layout 3 gave rc=%d, want BAD_VERSION",
                 (int)arms[a].layout, rc);
        enc[FLAGS_OFF] = arms[a].flags;
        arm_len[a] = enc_len;
    }

    /* With the nibble table on (set explicitly above), the three
     * layouts must produce byte-IDENTICAL lengths: a raw flat region is
     * the same size in every layout, and an FSE-coded one is always
     * natural-packed regardless of the table's setting.  If a change
     * ever makes FSE'd regions honour flat_layout again, the vertical
     * arms grow (lane-stride-16 gather breaks the code adjacency the
     * nibble table feeds on) and this fires. */
    if (arm_len[0] != arm_len[1] || arm_len[0] != arm_len[2])
        FAIL("layout-dependent size with the nibble table on: "
             "natural %zu, vertical %zu, vertical128 %zu",
             arm_len[0], arm_len[1], arm_len[2]);

    /* Synthetic v0.8: a natural stream with one table for the whole
     * input (v0.8 has no segments), drop the FLAGS byte, relabel the
     * minor, shrink BODY_LENGTH. */
    {
        pivco_cfg_t cfg = pivco_cfg_default;
        cfg.fse_nibble_enabled = 1;
        cfg.fse_k1_enabled = 1;
        cfg.flat_layout = PIVCO_FLAT_NATURAL;
        size_t enc_len = cap;
        int rc = pivcohuf_compress_seg(in, N, enc, &enc_len, &cfg,
                                       PIVCO_BLOCK_SIZE, 0, NULL);
        if (rc != PIVCOHUF_OK || enc[FLAGS_OFF] != 0 || count_table_blocks(enc, enc_len) != 0)
            FAIL("v0.8 synth: re-compress rc=%d flags %02x", rc, enc[FLAGS_OFF]);
        memmove(enc + FLAGS_OFF, enc + FLAGS_OFF + 1,
                enc_len - FLAGS_OFF - 1);
        enc_len -= 1;
        enc[9] = 8;                            /* MINOR_VERSION */
        uint64_t bl = 0;
        for (int b = 7; b >= 0; b--) bl = (bl << 8) | enc[10 + b];
        bl -= 1;
        for (int b = 0; b < 8; b++) enc[10 + b] = (uint8_t)(bl >> (8 * b));
        size_t dec_len = N;
        rc = pivcohuf_decompress(enc, enc_len, dec, &dec_len);
        if (rc != PIVCOHUF_OK) FAIL("v0.8 synth: decompress rc=%d", rc);
        if (dec_len != N || memcmp(in, dec, N) != 0)
            FAIL("v0.8 synth: roundtrip diff");
    }

    free(in); free(enc); free(dec);
    printf("PASS\n");
    return 0;
}

/* ---------- entry point ---------- */

/* Per-segment Huffman tables in the pivcohuf container (v0.10): a
 * 3-part input whose byte distribution changes every 64 KiB, at a table
 * every 2 blocks, so the stream carries fresh tables at the changes
 * (BLOCK_FLAGS NEW_TABLE) and none inside the stationary stretches.
 * Roundtrip, smaller than the whole-file stream, the whole-file stream
 * (v0.9 layout) still roundtrips, the default entry point's bound covers
 * its output, and a truncated stream is refused rather than read past
 * its end. */

static int test_table_segments(void)
{
    printf("[table_segments] ");
    enum { PART = 64 * 1024, N = 3 * PART };
    uint8_t *in = malloc(N);
    if (!in) FAIL("oom in");
    uint64_t rng = 0x5E6E7A11E5ULL;
    for (size_t i = 0; i < N; i++) {
        uint64_t x = xorshift64(&rng);
        int part = (int)(i / PART);
        /* part 0: skewed on 'a'..'d'; part 1: uniform bytes; part 2: skewed on 'w'..'z' */
        in[i] = part == 1 ? (uint8_t)x
              : (uint8_t)((part ? 'w' : 'a') + ((x & 7) ? 0 : (x >> 8) & 3));
    }
    const size_t B = 16384;
    size_t cap = pivcohuf_compress_bound_seg(N, B, 2);
    uint8_t *seg = malloc(cap), *whole = malloc(cap), *dec = malloc(N);
    if (!seg || !whole || !dec) { free(in); free(seg); free(whole); free(dec); FAIL("oom bufs"); }
    pivco_cfg_t cfg = pivco_cfg_default;
    cfg.fse_enabled = 0;
    size_t seg_len = cap, whole_len = cap;
    int rc = pivcohuf_compress_seg(in, N, seg, &seg_len, &cfg, B, 2, NULL);
    if (rc != PIVCOHUF_OK) FAIL("segmented compress rc=%d", rc);
    rc = pivcohuf_compress_seg(in, N, whole, &whole_len, &cfg, B, 0, NULL);
    if (rc != PIVCOHUF_OK) FAIL("whole-file compress rc=%d", rc);
    if (seg_len >= whole_len) FAIL("segments did not shrink the stream (%zu vs %zu)", seg_len, whole_len);
    /* Two distribution changes, 6 segments: 2..5 blocks carry a table on
     * the segmented stream (fresh where the data changed, reused where
     * it did not), none on the whole-file one, and neither sets a FLAGS
     * bit beyond the layout. */
    size_t nt = count_table_blocks(seg, seg_len);
    if (nt < 2 || nt > 5) FAIL("segmented stream carries %zu block tables, want 2..5", nt);
    if (count_table_blocks(whole, whole_len) != 0) FAIL("whole-file stream carries block tables");
    if ((seg[PIVCOHUF_HEADER_SIZE + 10] | whole[PIVCOHUF_HEADER_SIZE + 10]) & ~PIVCOHUF_FLAGS_LAYOUT_MASK)
        FAIL("FLAGS byte has bits beyond the layout");
    for (int which = 0; which < 2; which++) {
        const uint8_t *s = which ? whole : seg; size_t sl = which ? whole_len : seg_len;
        size_t dl = N;
        memset(dec, 0, N);
        rc = pivcohuf_decompress(s, sl, dec, &dl);
        if (rc != PIVCOHUF_OK || dl != N || memcmp(in, dec, N) != 0)
            FAIL("%s stream roundtrip failed (rc=%d len=%zu)", which ? "whole-file" : "segmented", rc, dl);
    }
    /* The default entry point segments too; its bound must hold. */
    size_t def_cap = pivcohuf_compress_bound(N), def_len = def_cap;
    uint8_t *def = malloc(def_cap);
    if (!def) FAIL("oom def");
    rc = pivcohuf_compress(in, N, def, &def_len);
    if (rc != PIVCOHUF_OK || count_table_blocks(def, def_len) == 0)
        FAIL("default compress: rc=%d, no block table at 128 KiB on a 192 KiB input", rc);
    size_t dl = N;
    rc = pivcohuf_decompress(def, def_len, dec, &dl);
    if (rc != PIVCOHUF_OK || dl != N || memcmp(in, dec, N) != 0) FAIL("default stream roundtrip failed");
    free(def);
    /* Truncation inside a later segment's table must be refused. */
    {
        uint8_t *cut = malloc(seg_len);
        if (!cut) FAIL("oom cut");
        memcpy(cut, seg, seg_len);
        /* Lie about the body length so the reader runs off the buffer end. */
        size_t short_len = seg_len / 2;
        dl = N;
        rc = pivcohuf_decompress(cut, short_len, dec, &dl);
        if (rc == PIVCOHUF_OK) { free(cut); FAIL("truncated segmented stream decoded OK"); }
        free(cut);
    }
    free(in); free(seg); free(whole); free(dec);
    printf("PASS (segmented %zu B vs whole-file %zu B)\n", seg_len, whole_len);
    return 0;
}

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
    printf("\n--- flat-layout FLAGS byte ---\n");
    fails += test_flat_layout_file();
    fails += test_table_segments();
#ifdef PIVCO_HAS_FSE
    printf("\n--- FSE length sweep ---\n");
    fails += test_fse_length_sweep();
    printf("\n--- FSE nibble table ---\n");
    fails += test_fse_nibble();
    printf("\n--- FSE k=1 bit-context table ---\n");
    fails += test_fse_k1();
    fails += test_fse_k2();
    fails += test_fse_kn_wire(1);
    fails += test_fse_kn_wire(2);
#endif
    return fails;
}
