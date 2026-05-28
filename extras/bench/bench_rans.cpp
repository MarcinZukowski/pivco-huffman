/* Thin C++ wrapper around ryg_rans alias method for benchmarking.
 * Exposes C-callable encode/decode functions. */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

// ryg_rans uses __rdtsc and timer() — stub them out
#ifdef __aarch64__
static inline uint64_t __rdtsc() { uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v; }
#endif

#include "rans_byte.h"

extern "C" {

/* ---- Stats / alias table (matches main_alias.cpp) ---- */

struct RansAlias {
    static const int LOG2NSYMS = 8;
    static const int NSYMS = 1 << LOG2NSYMS;

    uint32_t freqs[NSYMS];
    uint32_t cum_freqs[NSYMS + 1];
    uint32_t divider[NSYMS];
    uint32_t slot_adjust[NSYMS * 2];
    uint32_t slot_freqs[NSYMS * 2];
    uint8_t  sym_id[NSYMS * 2];
    uint32_t *alias_remap;

    uint32_t prob_bits;
};

static void rans_alias_normalize(RansAlias *s, uint32_t target_total)
{
    s->cum_freqs[0] = 0;
    for (int i = 0; i < s->NSYMS; i++)
        s->cum_freqs[i + 1] = s->cum_freqs[i] + s->freqs[i];
    uint32_t cur_total = s->cum_freqs[s->NSYMS];

    for (int i = 1; i <= s->NSYMS; i++)
        s->cum_freqs[i] = ((uint64_t)target_total * s->cum_freqs[i]) / cur_total;

    for (int i = 0; i < s->NSYMS; i++) {
        if (s->freqs[i] && s->cum_freqs[i + 1] == s->cum_freqs[i]) {
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j = 0; j < s->NSYMS; j++) {
                uint32_t freq = s->cum_freqs[j + 1] - s->cum_freqs[j];
                if (freq > 1 && freq < best_freq) { best_freq = freq; best_steal = j; }
            }
            if (best_steal < i) {
                for (int j = best_steal + 1; j <= i; j++) s->cum_freqs[j]--;
            } else {
                for (int j = i + 1; j <= best_steal; j++) s->cum_freqs[j]++;
            }
        }
    }
    for (int i = 0; i < s->NSYMS; i++)
        s->freqs[i] = s->cum_freqs[i + 1] - s->cum_freqs[i];
}

static void rans_alias_make_table(RansAlias *s)
{
    uint32_t sum = s->cum_freqs[s->NSYMS];
    uint32_t tgt_sum = sum / s->NSYMS;

    uint32_t remaining[256];
    for (int i = 0; i < s->NSYMS; i++) {
        remaining[i] = s->freqs[i];
        s->divider[i] = tgt_sum;
        s->sym_id[i * 2 + 0] = (uint8_t)i;
        s->sym_id[i * 2 + 1] = (uint8_t)i;
    }

    int cur_large = 0, cur_small = 0;
    while (cur_large < s->NSYMS && remaining[cur_large] < tgt_sum) cur_large++;
    while (cur_small < s->NSYMS && remaining[cur_small] >= tgt_sum) cur_small++;
    int next_small = cur_small + 1;

    while (cur_large < s->NSYMS && cur_small < s->NSYMS) {
        s->sym_id[cur_small * 2 + 0] = (uint8_t)cur_large;
        s->divider[cur_small] = remaining[cur_small];
        remaining[cur_large] -= tgt_sum - s->divider[cur_small];
        if (remaining[cur_large] >= tgt_sum || next_small <= cur_large) {
            cur_small = next_small;
            while (cur_small < s->NSYMS && remaining[cur_small] >= tgt_sum) cur_small++;
            next_small = cur_small + 1;
        } else {
            cur_small = cur_large;
        }
        while (cur_large < s->NSYMS && remaining[cur_large] < tgt_sum) cur_large++;
    }

    uint32_t assigned[256] = {0};
    s->alias_remap = new uint32_t[sum];

    for (int i = 0; i < s->NSYMS; i++) {
        int j = s->sym_id[i * 2 + 0];
        uint32_t sym0_height = s->divider[i];
        uint32_t sym1_height = tgt_sum - s->divider[i];
        uint32_t base0 = assigned[i];
        uint32_t base1 = assigned[j];
        uint32_t cbase0 = s->cum_freqs[i] + base0;
        uint32_t cbase1 = s->cum_freqs[j] + base1;

        s->divider[i] = i * tgt_sum + sym0_height;

        s->slot_freqs[i * 2 + 1] = s->freqs[i];
        s->slot_freqs[i * 2 + 0] = s->freqs[j];
        s->slot_adjust[i * 2 + 1] = i * tgt_sum - base0;
        s->slot_adjust[i * 2 + 0] = i * tgt_sum - (base1 - sym0_height);
        for (uint32_t k = 0; k < sym0_height; k++)
            s->alias_remap[cbase0 + k] = k + i * tgt_sum;
        for (uint32_t k = 0; k < sym1_height; k++)
            s->alias_remap[cbase1 + k] = (k + sym0_height) + i * tgt_sum;

        assigned[i] += sym0_height;
        assigned[j] += sym1_height;
    }
}

/* Alias rANS decode step */
static inline uint32_t rans_dec_get_alias(RansState *r, RansAlias *s)
{
    RansState x = *r;
    uint32_t mask = (1u << s->prob_bits) - 1;
    uint32_t xm = x & mask;
    uint32_t bucket_id = xm >> (s->prob_bits - RansAlias::LOG2NSYMS);
    uint32_t bucket2 = bucket_id * 2;
    if (xm < s->divider[bucket_id])
        bucket2++;
    *r = s->slot_freqs[bucket2] * (x >> s->prob_bits) + xm - s->slot_adjust[bucket2];
    return s->sym_id[bucket2];
}

/* ---- Public C API ---- */

typedef struct { RansAlias stats; } rans_ctx_t;

void *rans_alias_create(const uint64_t *freq256)
{
    rans_ctx_t *ctx = new rans_ctx_t;
    memset(&ctx->stats, 0, sizeof(ctx->stats));
    ctx->stats.prob_bits = 16;
    ctx->stats.alias_remap = NULL;
    for (int i = 0; i < 256; i++)
        ctx->stats.freqs[i] = (uint32_t)(freq256[i] > 0 ? freq256[i] : 0);
    rans_alias_normalize(&ctx->stats, 1u << ctx->stats.prob_bits);
    rans_alias_make_table(&ctx->stats);
    return ctx;
}

void rans_alias_destroy(void *ctx)
{
    rans_ctx_t *c = (rans_ctx_t *)ctx;
    delete[] c->stats.alias_remap;
    c->stats.alias_remap = NULL;
    delete c;
}

size_t rans_alias_encode(void *ctx, const uint8_t *symbols, size_t n,
                         uint8_t *out, size_t out_cap)
{
    rans_ctx_t *c = (rans_ctx_t *)ctx;
    uint8_t *ptr = out + out_cap;

    RansState rans;
    RansEncInit(&rans);

    for (size_t i = n; i > 0; i--) {
        int s = symbols[i - 1];
        uint32_t freq = c->stats.freqs[s];
        RansState x = RansEncRenorm(rans, &ptr, freq, c->stats.prob_bits);
        rans = ((x / freq) << c->stats.prob_bits) +
               c->stats.alias_remap[(x % freq) + c->stats.cum_freqs[s]];
    }
    RansEncFlush(&rans, &ptr);

    size_t enc_size = (size_t)(out + out_cap - ptr);
    memmove(out, ptr, enc_size);
    return enc_size;
}

size_t rans_alias_decode(void *ctx, const uint8_t *in, size_t in_len,
                         uint8_t *symbols, size_t n)
{
    rans_ctx_t *c = (rans_ctx_t *)ctx;
    uint8_t *ptr = (uint8_t *)in;

    RansState rans;
    RansDecInit(&rans, &ptr);

    for (size_t i = 0; i < n; i++) {
        symbols[i] = (uint8_t)rans_dec_get_alias(&rans, &c->stats);
        RansDecRenorm(&rans, &ptr);
    }
    return n;
}

/* Interleaved 2-stream decode */
size_t rans_alias_decode_x2(void *ctx, const uint8_t *in, size_t in_len,
                            uint8_t *symbols, size_t n)
{
    rans_ctx_t *c = (rans_ctx_t *)ctx;
    uint8_t *ptr = (uint8_t *)in;

    RansState rans0, rans1;
    RansDecInit(&rans0, &ptr);
    RansDecInit(&rans1, &ptr);

    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        symbols[i + 0] = (uint8_t)rans_dec_get_alias(&rans0, &c->stats);
        symbols[i + 1] = (uint8_t)rans_dec_get_alias(&rans1, &c->stats);
        RansDecRenorm(&rans0, &ptr);
        RansDecRenorm(&rans1, &ptr);
    }
    if (i < n) {
        symbols[i] = (uint8_t)rans_dec_get_alias(&rans0, &c->stats);
        RansDecRenorm(&rans0, &ptr);
    }
    return n;
}

/* Interleaved 2-stream encode */
size_t rans_alias_encode_x2(void *ctx, const uint8_t *symbols, size_t n,
                            uint8_t *out, size_t out_cap)
{
    rans_ctx_t *c = (rans_ctx_t *)ctx;
    uint8_t *ptr = out + out_cap;

    RansState rans0, rans1;
    RansEncInit(&rans0);
    RansEncInit(&rans1);

    if (n & 1) {
        int s = symbols[n - 1];
        uint32_t freq = c->stats.freqs[s];
        RansState x = RansEncRenorm(rans0, &ptr, freq, c->stats.prob_bits);
        rans0 = ((x / freq) << c->stats.prob_bits) +
                c->stats.alias_remap[(x % freq) + c->stats.cum_freqs[s]];
    }

    for (size_t i = (n & ~(size_t)1); i > 0; i -= 2) {
        int s1 = symbols[i - 1];
        int s0 = symbols[i - 2];
        {
            uint32_t freq = c->stats.freqs[s1];
            RansState x = RansEncRenorm(rans1, &ptr, freq, c->stats.prob_bits);
            rans1 = ((x / freq) << c->stats.prob_bits) +
                    c->stats.alias_remap[(x % freq) + c->stats.cum_freqs[s1]];
        }
        {
            uint32_t freq = c->stats.freqs[s0];
            RansState x = RansEncRenorm(rans0, &ptr, freq, c->stats.prob_bits);
            rans0 = ((x / freq) << c->stats.prob_bits) +
                    c->stats.alias_remap[(x % freq) + c->stats.cum_freqs[s0]];
        }
    }

    RansEncFlush(&rans1, &ptr);
    RansEncFlush(&rans0, &ptr);

    size_t enc_size = (size_t)(out + out_cap - ptr);
    memmove(out, ptr, enc_size);
    return enc_size;
}

} /* extern "C" */
