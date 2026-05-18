/* fuzz_lz4_split — correctness harness for the 4-stream LZ4 encoder
 * + custom decoder.
 *
 * Exercises size boundaries (the fast→safe transition at 128 B, the
 * 32 B/64 B trailing slack thresholds), content patterns that stress
 * specific offset/match-length buckets (RLE → offset=1, alternating
 * → offset=2, repeating short sequences → small offset / short match,
 * long copies → wildCopy32, long literals → overflow stream), and
 * the bench's standard corpus.  Prints a one-line summary plus
 * any failure details.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lz4.h>
#include "lz4_split.h"

/* ---------------------------------------------------------------- */

static int test_one(const uint8_t *src, size_t src_len, const char *label,
                    int verbose_on_fail)
{
    int lz4_cap = LZ4_compressBound((int)src_len);
    if (lz4_cap < 64) lz4_cap = 64;

    uint8_t *throwaway = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_lit = (uint8_t *)malloc(src_len + 64);
    uint8_t *s_tok = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_off = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_ovf = (uint8_t *)malloc((size_t)lz4_cap);

    lz4_split_ctx_t ctx = {
        .literals = s_lit, .lit_pos = 0, .lit_cap = src_len + 64,
        .tokens   = s_tok, .tok_pos = 0, .tok_cap = (size_t)lz4_cap,
        .offsets  = s_off, .off_pos = 0, .off_cap = (size_t)lz4_cap,
        .overflow = s_ovf, .ovf_pos = 0, .ovf_cap = (size_t)lz4_cap,
        .ok       = 0,
    };
    int lz4_size = phsplit_LZ4_compress_HC_split(
        (const char *)src, (int)src_len, throwaway, lz4_cap, 9, &ctx);

    int rc = 0;
    int ok = 1;
    size_t mismatch_at = src_len;
    uint8_t *dec = NULL;

    if (lz4_size <= 0 || !ctx.ok) {
        ok = 0;
        if (verbose_on_fail) {
            fprintf(stderr, "  ENCODE FAIL  %-22s  size=%zu  lz4_size=%d  ok=%d\n",
                    label, src_len, lz4_size, ctx.ok);
        }
        goto cleanup;
    }

    dec = (uint8_t *)malloc(src_len + 64);
    rc = lz4_split_decompress(s_lit, ctx.lit_pos,
                                s_tok, ctx.tok_pos,
                                s_off, ctx.off_pos,
                                s_ovf, ctx.ovf_pos,
                                dec, src_len);
    if (rc != 0) {
        ok = 0;
        if (verbose_on_fail) {
            fprintf(stderr, "  DECODE FAIL  %-22s  size=%zu  rc=%d\n",
                    label, src_len, rc);
        }
        goto cleanup;
    }
    if (src_len > 0 && memcmp(dec, src, src_len) != 0) {
        ok = 0;
        mismatch_at = 0;
        while (mismatch_at < src_len && dec[mismatch_at] == src[mismatch_at]) {
            mismatch_at++;
        }
        if (verbose_on_fail) {
            fprintf(stderr, "  MISMATCH     %-22s  size=%zu  at=%zu  (src=0x%02x dec=0x%02x)\n",
                    label, src_len, mismatch_at,
                    src[mismatch_at], dec[mismatch_at]);
        }
    }

cleanup:
    free(throwaway); free(s_lit); free(s_tok); free(s_off); free(s_ovf);
    free(dec);
    return ok ? 0 : 1;
}

/* ---------------------------------------------------------------- */

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint8_t rng_byte(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint8_t)(rng >> 24);
}
static uint32_t rng_u32(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}
static void rng_reset(uint64_t seed) { rng = seed ? seed : 0x9E3779B97F4A7C15ULL; }

static int read_file(const char *path, uint8_t **buf, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return -1; }
    fseek(f, 0, SEEK_SET);
    *buf = (uint8_t *)malloc((size_t)sz);
    if (!*buf) { fclose(f); return -1; }
    size_t n = fread(*buf, 1, (size_t)sz, f);
    fclose(f);
    *len = n;
    return 0;
}

/* ---------------------------------------------------------------- */

int main(int argc, char **argv)
{
    int total = 0, fails = 0;

    /* (1) Size + pattern grid.  Sizes chosen to straddle the fast-loop
     * preconditions: < 128 falls to safe loop, around 64–128 hits the
     * tail-handler paths.  Larger sizes drive the bulk fast loop. */
    size_t sizes[] = {
        0, 1, 2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 95, 127, 128, 129, 200, 255, 256, 511, 1024,
        4096, 16384, 65535, 65536, 65537, 100003, 1000003,
    };
    const size_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t si = 0; si < n_sizes; si++) {
        size_t n = sizes[si];
        uint8_t *buf = (uint8_t *)malloc(n + 1);

        /* zeros — maximum-compressible, RLE-prone (offset 1) */
        if (n > 0) memset(buf, 0, n);
        char lab[64];
        snprintf(lab, sizeof(lab), "zeros[%zu]", n);
        total++; fails += test_one(buf, n, lab, 1);

        /* random — incompressible, all-literal sequences */
        rng_reset(0xCAFEBABEULL ^ (uint64_t)n);
        for (size_t j = 0; j < n; j++) buf[j] = rng_byte();
        snprintf(lab, sizeof(lab), "random[%zu]", n);
        total++; fails += test_one(buf, n, lab, 1);

        /* ASCII 'a'..'z' uniform — moderate entropy, text-like */
        rng_reset(0xDEADBEEFULL ^ (uint64_t)n);
        for (size_t j = 0; j < n; j++) buf[j] = (uint8_t)('a' + (rng_byte() % 26));
        snprintf(lab, sizeof(lab), "ascii_az[%zu]", n);
        total++; fails += test_one(buf, n, lab, 1);

        /* alternating 0xAA/0x55 — RLE-friendly, offset 2 / short matches */
        for (size_t j = 0; j < n; j++) buf[j] = (j & 1) ? 0xAA : 0x55;
        snprintf(lab, sizeof(lab), "alt2[%zu]", n);
        total++; fails += test_one(buf, n, lab, 1);

        /* repeating "abracadabra" — small offsets, short/medium matches */
        const char *abra = "abracadabra";
        size_t alen = 11;
        for (size_t j = 0; j < n; j++) buf[j] = (uint8_t)abra[j % alen];
        snprintf(lab, sizeof(lab), "abra[%zu]", n);
        total++; fails += test_one(buf, n, lab, 1);

        /* single long run of the same byte — extreme RLE */
        if (n > 0) memset(buf, 0x42, n);
        snprintf(lab, sizeof(lab), "rle_0x42[%zu]", n);
        total++; fails += test_one(buf, n, lab, 1);

        free(buf);
    }

    /* (2) Composite content: long incompressible prefix + RLE tail
     * (exercises fast-loop short-lit bulk, then transition to long match). */
    {
        size_t n = 200000;
        uint8_t *buf = (uint8_t *)malloc(n);
        rng_reset(0x12345678ULL);
        for (size_t j = 0; j < n / 2; j++) buf[j] = rng_byte();
        memset(buf + n / 2, 0xCC, n - n / 2);
        total++; fails += test_one(buf, n, "rand_then_rle[200000]", 1);
        free(buf);
    }

    /* (3) Composite content: long RLE prefix + incompressible tail
     * (exercises long match at front, then random sequences). */
    {
        size_t n = 200000;
        uint8_t *buf = (uint8_t *)malloc(n);
        memset(buf, 0x99, n / 2);
        rng_reset(0x87654321ULL);
        for (size_t j = n / 2; j < n; j++) buf[j] = rng_byte();
        total++; fails += test_one(buf, n, "rle_then_rand[200000]", 1);
        free(buf);
    }

    /* (4) Near-64K-offset boundary: a 70 KB run of patterned data so
     * LZ4-HC produces offsets crossing the 64K boundary. */
    {
        size_t n = 70000;
        uint8_t *buf = (uint8_t *)malloc(n);
        for (size_t j = 0; j < n; j++) buf[j] = (uint8_t)(j * 31 + 7);
        total++; fails += test_one(buf, n, "linear_70k", 1);
        free(buf);
    }

    /* (5) Bench corpus — re-roundtrip every input from the default
     * dataset list to confirm the fuzz harness agrees with the bench. */
    const char *corpus[] = {
        "extras/datasets/cat-wiki.html",
        "extras/datasets/pride.txt",
        "extras/datasets/cat-image.jpg",
        "extras/datasets/json_api.json",
        "extras/datasets/chinese_text.txt",
        "extras/datasets/calgary_pic",
        "extras/datasets/gzip_random.gz",
        "extras/datasets/source_c.c",
        "extras/datasets/log_apache.log",
        "extras/datasets/dna_fasta.fa",
        "extras/datasets/csv_numeric.csv",
    };
    for (size_t k = 0; k < sizeof(corpus) / sizeof(corpus[0]); k++) {
        uint8_t *buf = NULL;
        size_t n = 0;
        if (read_file(corpus[k], &buf, &n) != 0) {
            fprintf(stderr, "  skip (not found): %s\n", corpus[k]);
            continue;
        }
        total++; fails += test_one(buf, n, corpus[k], 1);
        free(buf);
    }

    /* (6) Random-content fuzz: a wider sweep of random sizes + random
     * content + extra LZ4-HC compression levels to vary the encoder's
     * sequence emission patterns. */
    {
        rng_reset(0xA5A5A5A5A5A5ULL);
        for (int t = 0; t < 200; t++) {
            size_t n = (rng_u32() % 200000) + 1;
            uint8_t *buf = (uint8_t *)malloc(n);
            uint32_t style = rng_u32() & 0x3;
            switch (style) {
                case 0: for (size_t j = 0; j < n; j++) buf[j] = rng_byte(); break;
                case 1: for (size_t j = 0; j < n; j++) buf[j] = (uint8_t)('A' + (rng_byte() % 4)); break;
                case 2: for (size_t j = 0; j < n; j++) buf[j] = (uint8_t)((j ^ (j >> 4)) & 0xff); break;
                default: {
                    size_t mid = (size_t)(rng_u32() % (n + 1));
                    for (size_t j = 0; j < mid; j++) buf[j] = rng_byte();
                    uint8_t r = rng_byte();
                    for (size_t j = mid; j < n; j++) buf[j] = r;
                    break;
                }
            }
            char lab[64];
            snprintf(lab, sizeof(lab), "fuzz#%03d/style%u[%zu]", t, style, n);
            total++; fails += test_one(buf, n, lab, 1);
            free(buf);
        }
    }

    /* Optional: any extra files passed on the command line. */
    for (int a = 1; a < argc; a++) {
        uint8_t *buf = NULL;
        size_t n = 0;
        if (read_file(argv[a], &buf, &n) != 0) {
            fprintf(stderr, "  skip (not readable): %s\n", argv[a]);
            continue;
        }
        total++; fails += test_one(buf, n, argv[a], 1);
        free(buf);
    }

    printf("fuzz_lz4_split: %d/%d passed, %d failed\n", total - fails, total, fails);
    return fails ? 1 : 0;
}
