/* ph_td_test — minimal roundtrip test for the standalone TD slice.
 *
 * Builds a Huffman table from a fixed-skew distribution, encodes a
 * synthetic source buffer, decodes via the TD entry point, and
 * compares byte-for-byte.  Exit code = number of failed cases.
 */

#include "pivco_huffman.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* xorshift64 — small deterministic PRNG; same seed → same data. */
static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void)
{
    uint64_t x = rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng = x;
}
static uint8_t draw_byte(const uint64_t cum[256])
{
    uint64_t r = rng_next() % cum[255];
    /* Binary search would be fine; linear is plenty for tiny tables. */
    for (int i = 0; i < 256; i++) if (r < cum[i]) return (uint8_t)i;
    return 0;
}

/* Run a roundtrip test on `n_blocks` blocks of PIVCO_BLOCK_SIZE each.
 * Returns 0 on success, 1 on failure. */
static int test_one(const char *label, size_t n_blocks,
                    const uint64_t freq[256])
{
    /* Build the Huffman table from frequencies. */
    pivco_huffman_table_t table;
    int rc = pivco_huffman_build_table(freq, &table);
    if (rc != PIVCO_OK) {
        fprintf(stderr, "  build_table: %s rc=%d\n", label, rc);
        return 1;
    }

    uint64_t cum[256] = {0};
    cum[0] = freq[0];
    for (int i = 1; i < 256; i++) cum[i] = cum[i - 1] + freq[i];
    if (cum[255] == 0) {
        fprintf(stderr, "  %s: empty frequency table\n", label);
        return 1;
    }

    const size_t blksz = PIVCO_BLOCK_SIZE;
    uint8_t *src = (uint8_t *)malloc(blksz);
    uint8_t *enc = (uint8_t *)malloc(PIVCO_MAX_ENCODED_SIZE);
    uint8_t *dec = (uint8_t *)malloc(blksz);

    size_t total_enc = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        for (size_t i = 0; i < blksz; i++) src[i] = draw_byte(cum);

        size_t enc_len = PIVCO_MAX_ENCODED_SIZE;
        rc = pivco_huffman_encode(src, &table, enc, &enc_len);
        if (rc != PIVCO_OK) {
            fprintf(stderr, "  encode: %s block %zu rc=%d\n", label, b, rc);
            free(src); free(enc); free(dec); return 1;
        }
        total_enc += enc_len;

        size_t consumed = 0;
        rc = pivco_huffman_decode(enc, enc_len, &table, dec, &consumed);
        if (rc != PIVCO_OK) {
            fprintf(stderr, "  decode: %s block %zu rc=%d\n", label, b, rc);
            free(src); free(enc); free(dec); return 1;
        }
        if (consumed != enc_len) {
            fprintf(stderr,
                    "  consumed mismatch: %s block %zu consumed=%zu enc_len=%zu\n",
                    label, b, consumed, enc_len);
            free(src); free(enc); free(dec); return 1;
        }
        if (memcmp(src, dec, blksz) != 0) {
            size_t i = 0;
            while (i < blksz && src[i] == dec[i]) i++;
            fprintf(stderr,
                    "  roundtrip MISMATCH: %s block %zu at i=%zu (src=0x%02x dec=0x%02x)\n",
                    label, b, i, src[i], dec[i]);
            free(src); free(enc); free(dec); return 1;
        }
    }

    size_t n_bytes = blksz * n_blocks;
    printf("  ok  %-26s blocks=%-4zu  bytes=%-8zu  enc=%-8zu  ratio=%.3f\n",
           label, n_blocks, n_bytes, total_enc,
           (double)total_enc / (double)n_bytes);

    free(src); free(enc); free(dec);
    return 0;
}

int main(void)
{
    int fails = 0;

    /* Cover a few distribution shapes that exercise different tree depths. */
    uint64_t freq_uniform[256];
    for (int i = 0; i < 256; i++) freq_uniform[i] = 1;

    uint64_t freq_zipf[256];
    for (int i = 0; i < 256; i++) freq_zipf[i] = (uint64_t)(1000.0 / (i + 1));

    uint64_t freq_proba80[256] = {0};
    freq_proba80[0]  = 80;   /* one super-common symbol */
    for (int i = 1; i < 256; i++) freq_proba80[i] = 1;

    uint64_t freq_two_sym[256] = {0};
    freq_two_sym[0]   = 1;   /* trivial 2-symbol tree */
    freq_two_sym[255] = 1;

    /* Block counts; each block is PIVCO_BLOCK_SIZE bytes (= 8192 on NEON). */
    size_t n_blocks_set[] = {1, 4, 16};
    const char *names[] = {"uniform", "zipf", "proba80", "two-sym"};
    const uint64_t *freqs[] = {freq_uniform, freq_zipf,
                                freq_proba80, freq_two_sym};

    int n_runs = 0;
    for (size_t si = 0; si < sizeof(n_blocks_set) / sizeof(n_blocks_set[0]); si++) {
        for (size_t di = 0; di < sizeof(freqs) / sizeof(freqs[0]); di++) {
            char label[64];
            snprintf(label, sizeof(label), "%s", names[di]);
            fails += test_one(label, n_blocks_set[si], freqs[di]);
            n_runs++;
        }
    }

    printf("\n%d tests, %d failed\n", n_runs, fails);
    return fails ? 1 : 0;
}
