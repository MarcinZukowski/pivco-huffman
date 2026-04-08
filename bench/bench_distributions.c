#include "pivco_huffman.h"
#include <string.h>
#include <stdlib.h>

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

static distribution_t distributions[] = {
    { .name = "uniform" },
    { .name = "english" },
    { .name = "zipfian" },
    { .name = "sparse_4" },
    { .name = "sparse_16" },
    { .name = "geometric" },
    { .name = "two_sym_eq" },
    { .name = "two_sym_90/10" },
};

#define NUM_DISTRIBUTIONS (sizeof(distributions) / sizeof(distributions[0]))

static void init_distributions(void)
{
    /* uniform */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++)
        distributions[0].freq[i] = 100;

    /* english */
    {
        uint64_t *f = distributions[1].freq;
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
        distributions[2].freq[i] = (uint64_t)(10000.0 / (double)(i + 1));
        if (distributions[2].freq[i] == 0) distributions[2].freq[i] = 1;
    }

    /* sparse_4 */
    memset(distributions[3].freq, 0, sizeof(distributions[3].freq));
    distributions[3].freq[0] = 100;
    distributions[3].freq[1] = 100;
    distributions[3].freq[2] = 100;
    distributions[3].freq[3] = 100;

    /* sparse_16 */
    memset(distributions[4].freq, 0, sizeof(distributions[4].freq));
    for (int i = 0; i < 16; i++)
        distributions[4].freq[i] = 100;

    /* geometric */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        int shift = i < 30 ? 30 - i : 0;
        distributions[5].freq[i] = (uint64_t)1 << shift;
        if (distributions[5].freq[i] == 0) distributions[5].freq[i] = 1;
    }

    /* two_sym_eq */
    memset(distributions[6].freq, 0, sizeof(distributions[6].freq));
    distributions[6].freq[0] = 500;
    distributions[6].freq[1] = 500;

    /* two_sym_90/10 */
    memset(distributions[7].freq, 0, sizeof(distributions[7].freq));
    distributions[7].freq[0] = 900;
    distributions[7].freq[1] = 100;
}

/* ---------- Public API ---------- */

int  bench_num_distributions(void) { return (int)NUM_DISTRIBUTIONS; }

const char *bench_dist_name(int idx) { return distributions[idx].name; }

const uint64_t *bench_dist_freq(int idx) { return distributions[idx].freq; }

void bench_init(void) { init_distributions(); }

void bench_generate_symbols(int dist_idx, uint8_t *symbols, int n_symbols,
                            uint64_t seed)
{
    dist_sample(&distributions[dist_idx], symbols, n_symbols, seed);
}
