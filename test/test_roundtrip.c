#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Utilities ---------- */

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

#define FAIL(msg, ...) do { \
    printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
    return 1; \
} while (0)

/* ---------- Test: build table and verify canonical codes ---------- */

static int test_table_build(void)
{
    printf("[test_table_build] ");

    uint64_t freq[PIVCO_MAX_SYMBOLS] = {0};
    freq[0] = 100;
    freq[1] = 50;
    freq[2] = 25;
    freq[3] = 12;
    freq[4] = 6;

    pivco_huffman_table_t table;
    int rc = pivco_huffman_build_table(freq, &table);
    if (rc != PIVCO_OK) FAIL("build_table returned %d", rc);

    /* Verify prefix-free property: no code is a prefix of another */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (table.code_len[i] == 0) continue;
        for (int j = i + 1; j < PIVCO_MAX_SYMBOLS; j++) {
            if (table.code_len[j] == 0) continue;
            int shorter = table.code_len[i] < table.code_len[j] ? i : j;
            int longer  = shorter == i ? j : i;
            int slen = table.code_len[shorter];
            int llen = table.code_len[longer];
            uint16_t prefix = table.code[longer] >> (llen - slen);
            if (prefix == table.code[shorter]) {
                FAIL("code[%d]=%u/%d is prefix of code[%d]=%u/%d",
                     shorter, table.code[shorter], slen,
                     longer, table.code[longer], llen);
            }
        }
    }

    /* Verify decode table round-trips */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (table.code_len[i] == 0) continue;
        uint16_t code = table.code[i];
        int len = table.code_len[i];
        uint32_t idx = (uint32_t)code << (PIVCO_MAX_CODE_LEN - len);
        if (table.decode_sym[idx] != (uint8_t)i) {
            FAIL("decode_sym[%u] = %d, expected %d", idx, table.decode_sym[idx], i);
        }
        if (table.decode_len[idx] != len) {
            FAIL("decode_len[%u] = %d, expected %d", idx, table.decode_len[idx], len);
        }
    }

    printf("PASS\n");
    return 0;
}

/* ---------- Test: single-symbol alphabet ---------- */

static int test_single_symbol(void)
{
    printf("[test_single_symbol] ");

    uint64_t freq[PIVCO_MAX_SYMBOLS] = {0};
    freq[42] = 100;

    pivco_huffman_table_t table;
    pivco_huffman_build_table(freq, &table);

    uint8_t symbols[PIVCO_BLOCK_SIZE];
    memset(symbols, 42, sizeof(symbols));

    /* PIVCO roundtrip */
    uint8_t encoded[PIVCO_MAX_ENCODED_SIZE];
    size_t enc_len;
    int rc = pivco_huffman_encode_scalar(symbols, &table, encoded, &enc_len);
    if (rc != PIVCO_OK) FAIL("encode returned %d", rc);

    uint8_t decoded[PIVCO_BLOCK_SIZE];
    size_t consumed;
    rc = pivco_huffman_decode_scalar(encoded, enc_len, &table, decoded, &consumed);
    if (rc != PIVCO_OK) FAIL("decode returned %d", rc);

    if (memcmp(symbols, decoded, PIVCO_BLOCK_SIZE) != 0) {
        FAIL("PIVCO roundtrip mismatch");
    }

    /* Traditional roundtrip */
    uint8_t trad_enc[PIVCO_BLOCK_SIZE * 2];
    size_t trad_len, trad_bits;
    rc = trad_huffman_encode(symbols, PIVCO_BLOCK_SIZE, &table,
                             trad_enc, &trad_len, &trad_bits);
    if (rc != PIVCO_OK) FAIL("trad encode returned %d", rc);

    uint8_t trad_dec[PIVCO_BLOCK_SIZE];
    rc = trad_huffman_decode(trad_enc, trad_bits, &table,
                             trad_dec, PIVCO_BLOCK_SIZE);
    if (rc != PIVCO_OK) FAIL("trad decode returned %d", rc);

    if (memcmp(symbols, trad_dec, PIVCO_BLOCK_SIZE) != 0) {
        FAIL("trad roundtrip mismatch");
    }

    printf("PASS (encoded %zu bytes pivco, %zu bytes trad)\n", enc_len, trad_len);
    return 0;
}

/* ---------- Helper: roundtrip test with a given frequency distribution ---------- */

static int test_roundtrip_dist(const char *name, const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                uint64_t seed)
{
    printf("[test_roundtrip_%s] ", name);

    pivco_huffman_table_t table;
    int rc = pivco_huffman_build_table(freq, &table);
    if (rc != PIVCO_OK) FAIL("build_table returned %d", rc);

    /* Build CDF for sampling */
    uint64_t total = 0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) total += freq[i];
    if (total == 0) FAIL("empty frequency table");

    /* Generate random symbols from the distribution */
    uint8_t symbols[PIVCO_BLOCK_SIZE];
    uint64_t rng = seed;
    for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
        uint64_t r = xorshift64(&rng) % total;
        uint64_t cum = 0;
        int sym = 0;
        for (sym = 0; sym < PIVCO_MAX_SYMBOLS; sym++) {
            cum += freq[sym];
            if (r < cum) break;
        }
        symbols[i] = (uint8_t)sym;
    }

    /* PIVCO scalar roundtrip */
    uint8_t encoded[PIVCO_MAX_ENCODED_SIZE];
    size_t enc_len;
    rc = pivco_huffman_encode_scalar(symbols, &table, encoded, &enc_len);
    if (rc != PIVCO_OK) FAIL("pivco encode returned %d", rc);

    uint8_t decoded[PIVCO_BLOCK_SIZE];
    size_t consumed;
    rc = pivco_huffman_decode_scalar(encoded, enc_len, &table, decoded, &consumed);
    if (rc != PIVCO_OK) FAIL("pivco decode returned %d", rc);

    for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
        if (symbols[i] != decoded[i]) {
            FAIL("pivco mismatch at position %d: expected %d, got %d",
                 i, symbols[i], decoded[i]);
        }
    }

    if (consumed != enc_len) {
        FAIL("pivco consumed %zu bytes, expected %zu", consumed, enc_len);
    }

    /* Traditional roundtrip */
    uint8_t trad_enc[PIVCO_BLOCK_SIZE * 4];
    size_t trad_len, trad_bits;
    rc = trad_huffman_encode(symbols, PIVCO_BLOCK_SIZE, &table,
                             trad_enc, &trad_len, &trad_bits);
    if (rc != PIVCO_OK) FAIL("trad encode returned %d", rc);

    uint8_t trad_dec[PIVCO_BLOCK_SIZE];
    rc = trad_huffman_decode(trad_enc, trad_bits, &table,
                             trad_dec, PIVCO_BLOCK_SIZE);
    if (rc != PIVCO_OK) FAIL("trad decode returned %d", rc);

    for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
        if (symbols[i] != trad_dec[i]) {
            FAIL("trad mismatch at position %d: expected %d, got %d",
                 i, symbols[i], trad_dec[i]);
        }
    }

#ifdef PIVCO_HAS_NEON
    /* NEON roundtrip */
    uint8_t neon_enc[PIVCO_MAX_ENCODED_SIZE];
    size_t neon_len;
    rc = pivco_huffman_encode_neon(symbols, &table, neon_enc, &neon_len);
    if (rc != PIVCO_OK) FAIL("neon encode returned %d", rc);

    uint8_t neon_dec[PIVCO_BLOCK_SIZE];
    size_t neon_consumed;
    rc = pivco_huffman_decode_neon(neon_enc, neon_len, &table, neon_dec, &neon_consumed);
    if (rc != PIVCO_OK) FAIL("neon decode returned %d", rc);

    for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
        if (symbols[i] != neon_dec[i]) {
            FAIL("neon mismatch at position %d: expected %d, got %d",
                 i, symbols[i], neon_dec[i]);
        }
    }

    /* Cross-check: NEON-encoded stream against scalar decoder.
     * Catches encoder bugs that NEON decode reads symmetrically. */
    {
        uint8_t cross_dec[PIVCO_BLOCK_SIZE];
        size_t cross_consumed;
        rc = pivco_huffman_decode_scalar(neon_enc, neon_len, &table,
                                          cross_dec, &cross_consumed);
        if (rc != PIVCO_OK) FAIL("neon-enc -> scalar-dec rc=%d", rc);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != cross_dec[i]) {
                FAIL("neon-enc / scalar-dec mismatch at %d: "
                     "expected %d, got %d",
                     i, symbols[i], cross_dec[i]);
            }
        }
    }

    /* Bottom-up decoder (experimental).  Decodes the same neon-encoded
     * stream as the top-down decoder; output must match exactly. */
    {
        uint8_t bu_dec[PIVCO_BLOCK_SIZE];
        size_t bu_consumed;
        rc = pivco_huffman_decode_bu_neon(neon_enc, neon_len, &table,
                                           bu_dec, &bu_consumed);
        if (rc != PIVCO_OK) FAIL("bu_neon decode returned %d", rc);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != bu_dec[i]) {
                FAIL("bu_neon mismatch at position %d: expected %d, got %d",
                     i, symbols[i], bu_dec[i]);
            }
        }
        if (bu_consumed != neon_len) {
            FAIL("bu_neon consumed %zu bytes, expected %zu",
                 bu_consumed, neon_len);
        }
    }

    /* Cross-implementation: scalar encode -> neon decode */
    uint8_t cross_dec[PIVCO_BLOCK_SIZE];
    size_t cross_consumed;
    rc = pivco_huffman_decode_neon(encoded, enc_len, &table, cross_dec, &cross_consumed);
    if (rc != PIVCO_OK) FAIL("cross decode returned %d", rc);

    for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
        if (symbols[i] != cross_dec[i]) {
            FAIL("cross mismatch at position %d: expected %d, got %d",
                 i, symbols[i], cross_dec[i]);
        }
    }

#endif

#ifdef PIVCO_HAS_SSE4
    /* SSE4 roundtrip + cross-check against scalar-encoded stream.
     * Was missing for over a year; the e9a668f masked-tail bug went
     * undetected because nothing in this file directly invoked
     * pivco_huffman_decode_x86 on real distribution data. */
    {
        uint8_t sse_enc[PIVCO_MAX_ENCODED_SIZE];
        size_t sse_len;
        rc = pivco_huffman_encode_x86(symbols, &table, sse_enc, &sse_len);
        if (rc != PIVCO_OK) FAIL("sse encode returned %d", rc);

        uint8_t sse_dec[PIVCO_BLOCK_SIZE];
        size_t sse_consumed;
        rc = pivco_huffman_decode_x86(sse_enc, sse_len, &table,
                                       sse_dec, &sse_consumed);
        if (rc != PIVCO_OK) FAIL("sse decode returned %d", rc);

        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != sse_dec[i]) {
                FAIL("sse mismatch at position %d: expected %d, got %d",
                     i, symbols[i], sse_dec[i]);
            }
        }

        /* Cross: scalar encode -> sse decode */
        uint8_t sse_cross[PIVCO_BLOCK_SIZE];
        size_t sse_cross_consumed;
        rc = pivco_huffman_decode_x86(encoded, enc_len, &table,
                                       sse_cross, &sse_cross_consumed);
        if (rc != PIVCO_OK) FAIL("sse cross decode returned %d", rc);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != sse_cross[i]) {
                FAIL("sse cross mismatch at position %d: expected %d, got %d",
                     i, symbols[i], sse_cross[i]);
            }
        }

        /* Bottom-up SSE decoder against the top-down SSE decoder. */
        {
            uint8_t bu_dec[PIVCO_BLOCK_SIZE];
            size_t bu_consumed;
            rc = pivco_huffman_decode_bu_x86(sse_enc, sse_len, &table,
                                              bu_dec, &bu_consumed);
            if (rc != PIVCO_OK) FAIL("bu_x86 decode returned %d", rc);
            for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
                if (symbols[i] != bu_dec[i]) {
                    FAIL("bu_x86 mismatch at position %d: expected %d, got %d",
                         i, symbols[i], bu_dec[i]);
                }
            }
            if (bu_consumed != sse_len) {
                FAIL("bu_x86 consumed %zu bytes, expected %zu",
                     bu_consumed, sse_len);
            }
        }
    }
#endif

#ifdef PIVCO_HAS_AVX512
    /* AVX-512 roundtrip + cross-check against scalar-encoded stream.
     * Was missing; the b136b96 masked-tail bug went undetected for the
     * same reason as the SSE block above. */
    {
        uint8_t avx_enc[PIVCO_MAX_ENCODED_SIZE];
        size_t avx_len;
        rc = pivco_huffman_encode_avx512(symbols, &table, avx_enc, &avx_len);
        if (rc != PIVCO_OK) FAIL("avx512 encode returned %d", rc);

        uint8_t avx_dec[PIVCO_BLOCK_SIZE];
        size_t avx_consumed;
        rc = pivco_huffman_decode_avx512(avx_enc, avx_len, &table,
                                          avx_dec, &avx_consumed);
        if (rc != PIVCO_OK) FAIL("avx512 decode returned %d", rc);

        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != avx_dec[i]) {
                FAIL("avx512 mismatch at position %d: expected %d, got %d",
                     i, symbols[i], avx_dec[i]);
            }
        }

        /* Cross: scalar encode -> avx512 decode */
        uint8_t avx_cross[PIVCO_BLOCK_SIZE];
        size_t avx_cross_consumed;
        rc = pivco_huffman_decode_avx512(encoded, enc_len, &table,
                                          avx_cross, &avx_cross_consumed);
        if (rc != PIVCO_OK) FAIL("avx512 cross decode returned %d", rc);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != avx_cross[i]) {
                FAIL("avx512 cross mismatch at %d: expected %d, got %d",
                     i, symbols[i], avx_cross[i]);
            }
        }
    }
#endif

    printf("PASS (pivco=%zu B, trad=%zu B, ratio=%.2fx)\n",
           enc_len, trad_len, (double)enc_len / (double)trad_len);
    return 0;
}

/* ---------- Distribution generators ---------- */

static void make_uniform(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) freq[i] = 100;
}

static void make_english(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    /* Approximate English character frequencies */
    freq[' '] = 1830; freq['e'] = 1270; freq['t'] = 910;
    freq['a'] = 820;  freq['o'] = 750;  freq['i'] = 700;
    freq['n'] = 670;  freq['s'] = 630;  freq['h'] = 610;
    freq['r'] = 600;  freq['d'] = 430;  freq['l'] = 400;
    freq['c'] = 280;  freq['u'] = 280;  freq['m'] = 240;
    freq['w'] = 240;  freq['f'] = 220;  freq['g'] = 200;
    freq['y'] = 200;  freq['p'] = 190;  freq['b'] = 150;
    freq['v'] = 100;  freq['k'] = 80;   freq['j'] = 15;
    freq['x'] = 15;   freq['q'] = 10;   freq['z'] = 7;
    freq['.'] = 65;   freq[','] = 61;   freq['\n'] = 50;
}

static void make_zipfian(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        freq[i] = (uint64_t)(10000.0 / (double)(i + 1));
        if (freq[i] == 0) freq[i] = 1;
    }
}

static void make_sparse_4(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    freq[0] = 100; freq[1] = 100; freq[2] = 100; freq[3] = 100;
}

static void make_sparse_16(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    for (int i = 0; i < 16; i++) freq[i] = 100;
}

static void make_geometric(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    /* Steep geometric: freq[i] ~= 2^(15-i), capped to ensure 15-bit codes */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        int shift = i < 30 ? 30 - i : 0;
        freq[i] = (uint64_t)1 << shift;
        if (freq[i] == 0) freq[i] = 1;
    }
}

static void make_two_symbol_equal(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    freq[0] = 500; freq[1] = 500;
}

static void make_two_symbol_skewed(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    freq[0] = 900; freq[1] = 100;
}

/* ---------- Main test runner ---------- */

int test_roundtrip_all(void)
{
    int failures = 0;

    failures += test_table_build();
    failures += test_single_symbol();

    uint64_t freq[PIVCO_MAX_SYMBOLS];
    uint64_t seed = 0xDEADBEEFCAFE1234ULL;

    make_uniform(freq);
    failures += test_roundtrip_dist("uniform", freq, seed++);

    make_english(freq);
    failures += test_roundtrip_dist("english", freq, seed++);

    make_zipfian(freq);
    failures += test_roundtrip_dist("zipfian", freq, seed++);

    make_sparse_4(freq);
    failures += test_roundtrip_dist("sparse_4", freq, seed++);

    make_sparse_16(freq);
    failures += test_roundtrip_dist("sparse_16", freq, seed++);

    make_geometric(freq);
    failures += test_roundtrip_dist("geometric", freq, seed++);

    make_two_symbol_equal(freq);
    failures += test_roundtrip_dist("two_sym_eq", freq, seed++);

    make_two_symbol_skewed(freq);
    failures += test_roundtrip_dist("two_sym_skew", freq, seed++);

    /* Multiple blocks with different seeds */
    make_zipfian(freq);
    for (int b = 0; b < 10; b++) {
        char name[32];
        snprintf(name, sizeof(name), "zipf_block_%d", b);
        failures += test_roundtrip_dist(name, freq, seed + (uint64_t)b * 12345);
    }

    return failures;
}
