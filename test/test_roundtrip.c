#include "pivco_huffman.h"
static pivco_encoder_t *g_tenc;
static pivco_decoder_t *g_tdec;
static void t_ctx_init(void) {
    if (!g_tenc) g_tenc = pivco_encoder_create();
    if (!g_tdec) g_tdec = pivco_decoder_create();
}
#include "pivco_huffman_primitives.h"  /* prim_histogram_chunk (arch-selected) */
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

    pivco_table_t table;
    int rc = pivco_build_table(NULL, freq, &table);
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

    pivco_build_traditional_table(&table);
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

    pivco_table_t table;
    pivco_build_table(NULL, freq, &table);

    uint8_t symbols[PIVCO_BLOCK_SIZE];
    memset(symbols, 42, sizeof(symbols));

    /* PIVCO roundtrip */
    uint8_t encoded[PIVCO_MAX_ENCODED_SIZE];
    size_t enc_len;
    int rc = pivco_encode_scalar(g_tenc, &table, symbols, PIVCO_BLOCK_SIZE, encoded, &enc_len);
    if (rc != PIVCO_OK) FAIL("encode returned %d", rc);

    uint8_t decoded[PIVCO_BLOCK_SIZE];
    size_t consumed;
    rc = pivco_decode_scalar(g_tdec, &table, encoded, enc_len, decoded, &consumed);
    if (rc != PIVCO_OK) FAIL("decode returned %d", rc);

    if (memcmp(symbols, decoded, PIVCO_BLOCK_SIZE) != 0) {
        FAIL("PIVCO roundtrip mismatch");
    }

    /* Traditional roundtrip */
    pivco_build_traditional_table(&table);
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

    pivco_table_t table;
    int rc = pivco_build_table(NULL, freq, &table);
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
    rc = pivco_encode_scalar(g_tenc, &table, symbols, PIVCO_BLOCK_SIZE, encoded, &enc_len);
    if (rc != PIVCO_OK) FAIL("pivco encode returned %d", rc);

    uint8_t decoded[PIVCO_BLOCK_SIZE];
    size_t consumed;
    rc = pivco_decode_scalar(g_tdec, &table, encoded, enc_len, decoded, &consumed);
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
    pivco_build_traditional_table(&table);
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
    /* NEON encode + BU NEON decode roundtrip (the production path). */
    uint8_t neon_enc[PIVCO_MAX_ENCODED_SIZE];
    size_t neon_len;
    rc = pivco_encode_neon(g_tenc, &table, symbols, PIVCO_BLOCK_SIZE, neon_enc, &neon_len);
    if (rc != PIVCO_OK) FAIL("neon encode returned %d", rc);

    {
        uint8_t bu_dec[PIVCO_BLOCK_SIZE];
        size_t bu_consumed;
        rc = pivco_decode_bu_neon(g_tdec, &table, neon_enc, neon_len, bu_dec, &bu_consumed);
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

    /* Cross-check: NEON-encoded stream against scalar decoder.
     * Catches encoder bugs that NEON decode reads symmetrically. */
    {
        uint8_t cross_dec[PIVCO_BLOCK_SIZE];
        size_t cross_consumed;
        rc = pivco_decode_scalar(g_tdec, &table, neon_enc, neon_len, cross_dec, &cross_consumed);
        if (rc != PIVCO_OK) FAIL("neon-enc -> scalar-dec rc=%d", rc);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != cross_dec[i]) {
                FAIL("neon-enc / scalar-dec mismatch at %d: "
                     "expected %d, got %d",
                     i, symbols[i], cross_dec[i]);
            }
        }
    }
#endif

#ifdef PIVCO_HAS_SSE4
    /* SSE encode + BU SSE/AVX-512 decode roundtrip. */
    {
        uint8_t sse_enc[PIVCO_MAX_ENCODED_SIZE];
        size_t sse_len;
        rc = pivco_encode_x86(g_tenc, &table, symbols, PIVCO_BLOCK_SIZE, sse_enc, &sse_len);
        if (rc != PIVCO_OK) FAIL("sse encode returned %d", rc);

        uint8_t bu_dec[PIVCO_BLOCK_SIZE];
        size_t bu_consumed;
        rc = pivco_decode_bu_x86(g_tdec, &table, sse_enc, sse_len, bu_dec, &bu_consumed);
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

/* ---------- Test: joint length/shape pass (effort modes) ---------- */

/* 71 live symbols: sigma % gran != 0 for every grouped tier, so the
 * pass ghost-pads with zero-frequency symbols -- those receive real
 * codes the encoder never emits, and the decoder must rebuild the
 * identical (ghost-including) table from the lengths alone. */
static void make_odd_sigma(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    for (int i = 0; i < 71; i++) freq[13 + 2 * i] = 100000 / (i + 3);
}

/* For each effort mode x distribution: build a table (shaped when
 * effort > PLAIN), verify the lengths are Kraft-complete and capped
 * and that every used symbol kept a code, verify the actual coded bits
 * honor the pass's adoption guard against the PLAIN baseline,
 * rebuild the decode table from the transmitted lengths alone (the
 * wire contract), and roundtrip a block through the pair. */
static int test_joint_lengths(void)
{
    printf("[test_joint_lengths] ");

    typedef void (*make_fn)(uint64_t[PIVCO_MAX_SYMBOLS]);
    static const struct { const char *name; make_fn make; } dists[] = {
        {"uniform",   make_uniform},
        {"english",   make_english},
        {"zipfian",   make_zipfian},
        {"geometric", make_geometric},
        {"sparse_16", make_sparse_16},
        {"odd_sigma", make_odd_sigma},
    };
    static const pivco_effort_t efforts[] = {
        PIVCO_EFFORT_PLAIN,
        PIVCO_EFFORT_BALANCED,
        PIVCO_EFFORT_FASTER_DECOMPRESS,
        PIVCO_EFFORT_FASTEST_DECOMPRESS,
        PIVCO_EFFORT_FASTEST_COMPRESS,   /* == BALANCED at build level */
    };
    uint64_t seed = 0x0DDC0FFEE0DDF00DULL;

    for (size_t d = 0; d < sizeof(dists) / sizeof(dists[0]); d++) {
        uint64_t freq[PIVCO_MAX_SYMBOLS];
        dists[d].make(freq);
        uint64_t total = 0;
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) total += freq[i];

        double base_bits = 0;
        for (size_t e = 0; e < sizeof(efforts) / sizeof(efforts[0]); e++) {
            pivco_cfg_t ecfg = pivco_cfg_default; ecfg.effort = efforts[e];
            pivco_table_t table;
            int rc = pivco_build_table(&ecfg, freq, &table);
            if (rc != PIVCO_OK)
                FAIL("%s effort %d: build_table returned %d",
                     dists[d].name, (int)efforts[e], rc);

            /* Lengths must stay a capped, Kraft-COMPLETE code (the
             * shaped set is Kraft-exact by construction; a hole or an
             * overflow here means the deal miscounted). */
            uint64_t kraft = 0;
            double bits = 0;
            for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
                if (freq[s] > 0 && table.code_len[s] == 0)
                    FAIL("%s effort %d: used symbol %d lost its code",
                         dists[d].name, (int)efforts[e], s);
                if (table.code_len[s] == 0) continue;
                if (table.code_len[s] > PIVCO_MAX_CODE_LEN)
                    FAIL("%s effort %d: code_len[%d] = %d exceeds cap",
                         dists[d].name, (int)efforts[e], s, table.code_len[s]);
                kraft += (uint64_t)1 << (PIVCO_MAX_CODE_LEN - table.code_len[s]);
                bits += (double)freq[s] * table.code_len[s];
            }
            if (kraft != (uint64_t)1 << PIVCO_MAX_CODE_LEN)
                FAIL("%s effort %d: Kraft sum %llu != %llu",
                     dists[d].name, (int)efforts[e],
                     (unsigned long long)kraft,
                     (unsigned long long)1 << PIVCO_MAX_CODE_LEN);

            /* The adoption guard bounds the size cost: coded bits stay
             * within 1.5% of the plain-Huffman baseline (efforts[0]). */
            if (e == 0)
                base_bits = bits;
            else if (bits > base_bits * 1.015 * (1 + 1e-9))
                FAIL("%s effort %d: %.0f bits vs baseline %.0f breaks the "
                     "1.5%% guard", dists[d].name, (int)efforts[e],
                     bits, base_bits);

            /* Wire contract: the decoder rebuilds the identical table
             * from the transmitted lengths alone (ghost codes and all). */
            pivco_table_t dtable;
            rc = pivco_build_table_from_code_lens(NULL, table.code_len,
                                                          &dtable);
            if (rc != PIVCO_OK)
                FAIL("%s effort %d: from_code_lens returned %d",
                     dists[d].name, (int)efforts[e], rc);
            if (memcmp(table.code_len, dtable.code_len,
                       sizeof(table.code_len)) != 0
                || memcmp(table.code, dtable.code, sizeof(table.code)) != 0)
                FAIL("%s effort %d: decoder-side rebuild diverged",
                     dists[d].name, (int)efforts[e]);

            /* Roundtrip a block: encode with the freq-built table,
             * decode with the lens-rebuilt one. */
            uint8_t symbols[PIVCO_BLOCK_SIZE];
            uint64_t rng = seed++;
            for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
                uint64_t r = xorshift64(&rng) % total;
                uint64_t cum = 0;
                int sym;
                for (sym = 0; sym < PIVCO_MAX_SYMBOLS; sym++) {
                    cum += freq[sym];
                    if (r < cum) break;
                }
                symbols[i] = (uint8_t)sym;
            }
            uint8_t encoded[PIVCO_MAX_ENCODED_SIZE];
            uint8_t decoded[PIVCO_BLOCK_SIZE];
            size_t enc_len, consumed;
            rc = pivco_encode(g_tenc, &table, symbols, PIVCO_BLOCK_SIZE,
                              encoded, &enc_len);
            if (rc != PIVCO_OK)
                FAIL("%s effort %d: encode returned %d",
                     dists[d].name, (int)efforts[e], rc);
            rc = pivco_decode(g_tdec, &dtable, encoded, enc_len,
                                      decoded, &consumed);
            if (rc != PIVCO_OK)
                FAIL("%s effort %d: decode returned %d",
                     dists[d].name, (int)efforts[e], rc);
            if (consumed != enc_len)
                FAIL("%s effort %d: consumed %zu of %zu bytes",
                     dists[d].name, (int)efforts[e], consumed, enc_len);
            for (int i = 0; i < PIVCO_BLOCK_SIZE; i++)
                if (symbols[i] != decoded[i])
                    FAIL("%s effort %d: mismatch at %d",
                         dists[d].name, (int)efforts[e], i);
        }
    }

    printf("PASS\n");
    return 0;
}


/* ---- histogram primitive: dispatched vs naive, edges + alignment ---- */
static int test_histogram(void)
{
    printf("histogram primitive:\n");
    enum { HCAP = 5 * 1024 * 1024 + 128 };
    uint8_t *buf = malloc(HCAP);
    uint8_t *hscratch = malloc(PIVCO_PRIM_HIST_SCRATCH);
    prim_codec_init();
    uint64_t seed = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < HCAP; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        buf[i] = (seed % 100) < 60 ? 'A' : (uint8_t)(seed >> 32);
    }
    size_t sizes[] = {0, 1, 7, 63, 64, 65, 127, 4095, 4096,
                      65472, 65473, 1 << 20, 5 * 1024 * 1024};
    size_t offs[] = {0, 1, 63};
    int fails = 0;
    for (unsigned si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++)
        for (unsigned oi = 0; oi < sizeof(offs)/sizeof(offs[0]); oi++) {
            size_t n = sizes[si];
            const uint8_t *p = buf + offs[oi];
            uint64_t ref[256] = {0};
            for (size_t i = 0; i < n; i++) ref[p[i]]++;
            uint32_t got32[256] = {0};
            prim_histogram_chunk(p, n, got32, hscratch);
            int ok = 1;
            for (int s = 0; s < 256; s++) ok &= (uint64_t)got32[s] == ref[s];
            if (!ok) {
                printf("  FAIL n=%zu off=%zu\n", n, offs[oi]);
                fails++;
            }
            /* accumulation semantics: second call adds */
            if (n && ok) {
                prim_histogram_chunk(p, n, got32, hscratch);
                for (int s = 0; s < 256; s++) ok &= (uint64_t)got32[s] == 2 * ref[s];
                if (!ok) { printf("  FAIL accum n=%zu\n", n); fails++; }
            }
        }
    /* all-same buffer (bin-overflow flush path) */
    memset(buf, 'Z', 1 << 20);
    { uint32_t got[256] = {0};
      prim_histogram_chunk(buf, 1 << 20, got, hscratch);
      if (got['Z'] != (1u << 20)) { printf("  FAIL all-same\n"); fails++; } }
    free(buf); free(hscratch);
    printf("  %s\n", fails ? "FAILED" : "all ok");
    return fails;
}

/* ---------- Main test runner ---------- */

int test_roundtrip_all(void)
{
    int failures = 0;
    t_ctx_init();

    failures += test_table_build();
    failures += test_single_symbol();
    failures += test_joint_lengths();
    failures += test_histogram();

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
