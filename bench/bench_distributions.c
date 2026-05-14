#include "pivco_huffman.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "dist_real_freqs.h"   /* freq_html_wiki / prose_pride / image_jpeg */

/* ---------- PRNG ---------- */

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* ---------- Distribution definitions ---------- */

typedef struct {
    const char *name;
    int         is_main;   /* 1 = part of the MAIN dev-iteration set */
    uint64_t    freq[PIVCO_MAX_SYMBOLS];
} distribution_t;

/* Sample symbols from a distribution using the CDF */
void dist_sample(const distribution_t *dist, uint8_t *symbols, int n,
                 uint64_t seed)
{
    uint64_t total = 0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) total += dist->freq[i];

    uint64_t rng = seed;
    for (int i = 0; i < n; i++) {
        uint64_t r = xorshift64(&rng) % total;
        uint64_t cum = 0;
        int sym;
        for (sym = 0; sym < PIVCO_MAX_SYMBOLS - 1; sym++) {
            cum += dist->freq[sym];
            if (r < cum) break;
        }
        symbols[i] = (uint8_t)sym;
    }
}

/* ---------- Built-in distributions ---------- */

/* The MAIN set is the curated dev-iteration list: nine distributions
 * that span the regimes we care about (text, JSON, image, near-uniform,
 * flat, skewed).  bench_main.c runs MAIN-only by default; pass --all
 * to include every distribution below (used for paper-grade sweeps). */
static distribution_t distributions[] = {
    { .name = "proba80",       .is_main = 1 },
    { .name = "proba50" },
    { .name = "proba14" },
    { .name = "proba02" },
    { .name = "bell_s10" },
    { .name = "bell_s30" },
    { .name = "bell_s80" },
    { .name = "uniform" },
    { .name = "english",       .is_main = 1 },
    { .name = "zipfian" },
    { .name = "sparse_4" },
    { .name = "sparse_16" },
    { .name = "geometric" },
    { .name = "two_sym_eq" },
    { .name = "two_sym_90/10" },
    /* Flat distributions filling out the M-curve for the prefix backend.
     * Each has 2^M equal-frequency symbols → flat Huffman with all
     * codes of length M. */
    { .name = "flat_M3" },                      /* 8   symbols, M=3 */
    { .name = "flat_M5",       .is_main = 1 },  /* 32  symbols, M=5 */
    { .name = "flat_M6" },                      /* 64  symbols, M=6 */
    { .name = "flat_M7" },                      /* 128 symbols, M=7 */
    /* Real-world byte distributions (extras/datasets/README.md). */
    { .name = "html_wiki",     .is_main = 1 },  /* en.wikipedia.org/wiki/Cat HTML */
    { .name = "prose_pride",   .is_main = 1 },  /* Project Gutenberg Pride and Prejudice */
    { .name = "image_jpeg",    .is_main = 1 },  /* Wikimedia Commons Cat03.jpg */
    { .name = "json_api",      .is_main = 1 },  /* GitHub API commit feed JSON */
    { .name = "source_c" },                     /* zstd_compress.c */
    { .name = "log_apache" },                   /* NASA HTTP / Logstash sample */
    { .name = "dna_fasta" },                    /* E. coli K-12 genome FASTA */
    { .name = "csv_numeric" },                  /* OWID CO2 dataset CSV */
    { .name = "gzip_random",   .is_main = 1 },  /* gzip(cat-wiki.html) — near-uniform */
    { .name = "chinese_text",  .is_main = 1 },  /* Project Gutenberg 紅樓夢 */
    { .name = "calgary_pic",   .is_main = 1 },  /* Calgary corpus 1bpp scanned page — proba80-like real dataset */
};

#define NUM_DISTRIBUTIONS (sizeof(distributions) / sizeof(distributions[0]))

/* FSE-style probability distribution (matches fullbench.c BMK_genData).
   Each symbol gets p% of remaining probability mass.
   Symbol 0: p, Symbol 1: p*(1-p), Symbol 2: p*(1-p)^2, etc. */
static void make_fse_proba(uint64_t *freq, double p)
{
    #define PROBA_TABLE_SIZE 2048
    int table[PROBA_TABLE_SIZE];
    memset(table, 0, sizeof(table));
    int remaining = PROBA_TABLE_SIZE;
    int pos = 0;
    int s = 0;

    if (p < 0.01) p = 0.005;
    if (p > 1.0) p = 1.0;

    while (remaining > 0) {
        int n = (int)(remaining * p);
        if (n == 0) n = 1;
        int end = pos + n;
        if (end > PROBA_TABLE_SIZE) end = PROBA_TABLE_SIZE;
        while (pos < end) table[pos++] = s;
        s++;
        if (s == 255) s = 0;
        remaining -= n;
    }

    /* Count frequencies from the table */
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    for (int i = 0; i < PROBA_TABLE_SIZE; i++) {
        freq[table[i]]++;
    }
    #undef PROBA_TABLE_SIZE
}

static void init_distributions(void)
{
    /* proba80 — FSE benchmark: 80% probability (very skewed) */
    make_fse_proba(distributions[0].freq, 0.80);

    /* proba50 — FSE benchmark: 50% probability */
    make_fse_proba(distributions[1].freq, 0.50);

    /* proba14 — FSE benchmark: 14% probability (moderate) */
    make_fse_proba(distributions[2].freq, 0.14);

    /* proba02 — FSE benchmark: 2% probability (near-uniform) */
    make_fse_proba(distributions[3].freq, 0.02);

    /* bell_s10 — narrow bell curve (σ=10), ~60 effective symbols */
    {
        uint64_t *f = distributions[4].freq;
        memset(f, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
            double x = (double)(i - 128) / 10.0;
            f[i] = (uint64_t)(10000.0 * exp(-0.5 * x * x)) + 1;
        }
    }

    /* bell_s30 — medium bell curve (σ=30), ~180 effective symbols */
    {
        uint64_t *f = distributions[5].freq;
        memset(f, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
            double x = (double)(i - 128) / 30.0;
            f[i] = (uint64_t)(10000.0 * exp(-0.5 * x * x)) + 1;
        }
    }

    /* bell_s80 — wide bell curve (σ=80), nearly all 256 symbols */
    {
        uint64_t *f = distributions[6].freq;
        memset(f, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
            double x = (double)(i - 128) / 80.0;
            f[i] = (uint64_t)(10000.0 * exp(-0.5 * x * x)) + 1;
        }
    }

    /* uniform */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++)
        distributions[7].freq[i] = 100;

    /* english */
    {
        uint64_t *f = distributions[8].freq;
        memset(f, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
        f[' '] = 1830; f['e'] = 1270; f['t'] = 910;
        f['a'] = 820;  f['o'] = 750;  f['i'] = 700;
        f['n'] = 670;  f['s'] = 630;  f['h'] = 610;
        f['r'] = 600;  f['d'] = 430;  f['l'] = 400;
        f['c'] = 280;  f['u'] = 280;  f['m'] = 240;
        f['w'] = 240;  f['f'] = 220;  f['g'] = 200;
        f['y'] = 200;  f['p'] = 190;  f['b'] = 150;
        f['v'] = 100;  f['k'] = 80;   f['j'] = 15;
        f['x'] = 15;   f['q'] = 10;   f['z'] = 7;
        f['.'] = 65;   f[','] = 61;   f['\n'] = 50;
    }

    /* zipfian */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        distributions[9].freq[i] = (uint64_t)(10000.0 / (double)(i + 1));
        if (distributions[9].freq[i] == 0) distributions[9].freq[i] = 1;
    }

    /* sparse_4 */
    memset(distributions[10].freq, 0, sizeof(distributions[10].freq));
    distributions[10].freq[0] = 100;
    distributions[10].freq[1] = 100;
    distributions[10].freq[2] = 100;
    distributions[10].freq[3] = 100;

    /* sparse_16 */
    memset(distributions[11].freq, 0, sizeof(distributions[11].freq));
    for (int i = 0; i < 16; i++)
        distributions[11].freq[i] = 100;

    /* geometric */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        int shift = i < 30 ? 30 - i : 0;
        distributions[12].freq[i] = (uint64_t)1 << shift;
        if (distributions[12].freq[i] == 0) distributions[12].freq[i] = 1;
    }

    /* two_sym_eq */
    memset(distributions[13].freq, 0, sizeof(distributions[13].freq));
    distributions[13].freq[0] = 500;
    distributions[13].freq[1] = 500;

    /* two_sym_90/10 */
    memset(distributions[14].freq, 0, sizeof(distributions[14].freq));
    distributions[14].freq[0] = 900;
    distributions[14].freq[1] = 100;

    /* flat_M3: 8 equal symbols → all codes 3-bit */
    memset(distributions[15].freq, 0, sizeof(distributions[15].freq));
    for (int i = 0; i < 8; i++) distributions[15].freq[i] = 100;

    /* flat_M5: 32 equal symbols → all codes 5-bit */
    memset(distributions[16].freq, 0, sizeof(distributions[16].freq));
    for (int i = 0; i < 32; i++) distributions[16].freq[i] = 100;

    /* flat_M6: 64 equal symbols → all codes 6-bit */
    memset(distributions[17].freq, 0, sizeof(distributions[17].freq));
    for (int i = 0; i < 64; i++) distributions[17].freq[i] = 100;

    /* flat_M7: 128 equal symbols → all codes 7-bit */
    memset(distributions[18].freq, 0, sizeof(distributions[18].freq));
    for (int i = 0; i < 128; i++) distributions[18].freq[i] = 100;

    /* Real-world byte freqs.  Source files + regeneration steps in
     * extras/datasets/README.md. */
    memcpy(distributions[19].freq, freq_html_wiki,    sizeof(freq_html_wiki));
    memcpy(distributions[20].freq, freq_prose_pride,  sizeof(freq_prose_pride));
    memcpy(distributions[21].freq, freq_image_jpeg,   sizeof(freq_image_jpeg));
    memcpy(distributions[22].freq, freq_json_api,     sizeof(freq_json_api));
    memcpy(distributions[23].freq, freq_source_c,     sizeof(freq_source_c));
    memcpy(distributions[24].freq, freq_log_apache,   sizeof(freq_log_apache));
    memcpy(distributions[25].freq, freq_dna_fasta,    sizeof(freq_dna_fasta));
    memcpy(distributions[26].freq, freq_csv_numeric,  sizeof(freq_csv_numeric));
    memcpy(distributions[27].freq, freq_gzip_random,  sizeof(freq_gzip_random));
    memcpy(distributions[28].freq, freq_chinese_text, sizeof(freq_chinese_text));
    memcpy(distributions[29].freq, freq_calgary_pic,  sizeof(freq_calgary_pic));
}

/* ---------- Public API ---------- */

int  bench_num_distributions(void) { return (int)NUM_DISTRIBUTIONS; }

const char *bench_dist_name(int idx) { return distributions[idx].name; }

const uint64_t *bench_dist_freq(int idx) { return distributions[idx].freq; }

int  bench_dist_is_main(int idx) { return distributions[idx].is_main; }

void bench_init(void) { init_distributions(); }

void bench_generate_symbols(int dist_idx, uint8_t *symbols, int n_symbols,
                            uint64_t seed)
{
    dist_sample(&distributions[dist_idx], symbols, n_symbols, seed);
}
