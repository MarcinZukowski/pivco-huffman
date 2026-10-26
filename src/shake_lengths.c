/* shake_lengths.c -- size-driven search over code lengths ("shaking").
 *
 * Plain Huffman lengths are optimal for order-0 coding of the symbols,
 * but the per-node bitmap coders (sFSE, k1, k2, nibble, dHuf0, nPH) see
 * the tree's partition bitmaps, whose compressibility depends on which
 * symbols the tree groups together.  Shaking searches Kraft-preserving
 * length moves for a code that makes the encoded segment smaller, priced
 * by real pivco_encode calls on the segment's blocks under the caller's
 * cfg.  Encoder side only: the wire carries plain code lengths.
 *
 * Moves (all keep the code complete):
 *   neighbor swap  a symbol trades lengths with one a level away
 *   directed       split (a L->L-1, c,d L->L+1) or join, partners drawn
 *                  from the needed length
 *   random         swap, split or join on uniformly drawn symbols
 * Symbols holding at least SHAKE_FREEZE_SHARE of the segment never move.
 *
 * Search: a race.  SHAKE_STARTS starting points, the input lengths plus
 * bursts of 1..SHAKE_BURST random moves from them, then rounds of
 * hill-climbing (one move per step, accept if smaller) shared equally
 * among the live candidates, the worse half dropped after each round.
 * A step is first priced on SHAKE_SCREEN spread blocks and only priced
 * in full when those do not get worse.  Lengths already priced are
 * skipped.  The result is never larger than the input lengths', and the
 * search is deterministic (fixed seed). */

#include "pivco_huffman.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SHAKE_STARTS 10
#define SHAKE_BURST 8
#define SHAKE_SCREEN 2
#define SHAKE_FREEZE_SHARE 0.25
#define SHAKE_SEED 0x9E3779B97F4A7C15ull

typedef struct {
    uint8_t lens[PIVCO_MAX_SYMBOLS];
    size_t cost;
    size_t *blk;
    int alive;
} shake_cand_t;

typedef struct {
    pivco_encoder_t *enc;
    const pivco_cfg_t *cfg;
    pivco_table_t *tab;
    const uint8_t *in;
    size_t n;
    size_t block_size;
    int nblk;
    uint8_t *screen;
    uint8_t *out;
    size_t out_cap;
    size_t *cand_blk;
    int used[PIVCO_MAX_SYMBOLS];
    int nu;
    uint64_t rng;
    uint64_t *seen;
    uint32_t seen_mask;
} shake_t;

static uint32_t shake_rnd(shake_t *s, uint32_t n)
{
    s->rng ^= s->rng << 13;
    s->rng ^= s->rng >> 7;
    s->rng ^= s->rng << 17;
    return (uint32_t)((s->rng >> 11) % n);
}

static uint64_t lens_hash(const uint8_t *lens)
{
    uint64_t h = 0xCBF29CE484222325ull;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        h ^= lens[i];
        h *= 0x100000001B3ull;
    }
    return h | 1;
}

/* Inserts h into the set of priced lengths; returns 1 when it was there.
 * A full set (or a hash collision) only lets a duplicate through. */
static int seen_insert(shake_t *s, uint64_t h)
{
    uint32_t i = (uint32_t)h & s->seen_mask;
    for (uint32_t probe = 0; probe <= s->seen_mask; probe++) {
        if (s->seen[i] == h)
            return 1;
        if (s->seen[i] == 0) {
            s->seen[i] = h;
            return 0;
        }
        i = (i + 1) & s->seen_mask;
    }
    return 0;
}

static int random_move(shake_t *s, uint8_t *lens)
{
    int kind = (int)shake_rnd(s, 3);
    if (kind == 0) {
        int a = s->used[shake_rnd(s, (uint32_t)s->nu)];
        int b = s->used[shake_rnd(s, (uint32_t)s->nu)];
        if (lens[a] == lens[b])
            return 0;
        uint8_t t = lens[a];
        lens[a] = lens[b];
        lens[b] = t;
        return 1;
    }
    int a = s->used[shake_rnd(s, (uint32_t)s->nu)];
    int c = s->used[shake_rnd(s, (uint32_t)s->nu)];
    int d = s->used[shake_rnd(s, (uint32_t)s->nu)];
    if (a == c || a == d || c == d)
        return 0;
    if (kind == 1) {
        int L = lens[a];
        if (lens[c] != L || lens[d] != L)
            return 0;
        if (L <= 1 || L + 1 > PIVCO_MAX_CODE_LEN)
            return 0;
        lens[a] = (uint8_t)(L - 1);
        lens[c] = (uint8_t)(L + 1);
        lens[d] = (uint8_t)(L + 1);
        return 1;
    }
    int L = lens[a] + 1;
    if (lens[c] != L + 1 || lens[d] != L + 1)
        return 0;
    lens[a] = (uint8_t)L;
    lens[c] = (uint8_t)L;
    lens[d] = (uint8_t)L;
    return 1;
}

/* Split or join with the two partners drawn from the length the move
 * needs (a uniform triple rarely shares one). */
static int directed_move(shake_t *s, uint8_t *lens)
{
    int a = s->used[shake_rnd(s, (uint32_t)s->nu)];
    int join = (int)shake_rnd(s, 2);
    int want = join ? lens[a] + 2 : lens[a];
    if (!join && (lens[a] <= 1 || lens[a] + 1 > PIVCO_MAX_CODE_LEN))
        return 0;
    if (join && want > PIVCO_MAX_CODE_LEN)
        return 0;
    int cand[PIVCO_MAX_SYMBOLS];
    int nc = 0;
    for (int i = 0; i < s->nu; i++)
        if (s->used[i] != a && lens[s->used[i]] == want)
            cand[nc++] = s->used[i];
    if (nc < 2)
        return 0;
    int i = (int)shake_rnd(s, (uint32_t)nc);
    int j = (int)shake_rnd(s, (uint32_t)(nc - 1));
    if (j >= i)
        j++;
    int c = cand[i];
    int d = cand[j];
    if (join) {
        lens[a] = (uint8_t)(lens[a] + 1);
        lens[c] = (uint8_t)(want - 1);
        lens[d] = (uint8_t)(want - 1);
    } else {
        lens[a] = (uint8_t)(lens[a] - 1);
        lens[c] = (uint8_t)(want + 1);
        lens[d] = (uint8_t)(want + 1);
    }
    return 1;
}

static int neighbor_swap(shake_t *s, uint8_t *lens)
{
    int a = s->used[shake_rnd(s, (uint32_t)s->nu)];
    int want = shake_rnd(s, 2) ? lens[a] + 1 : lens[a] - 1;
    int cand[PIVCO_MAX_SYMBOLS];
    int nc = 0;
    for (int i = 0; i < s->nu; i++)
        if (lens[s->used[i]] == want)
            cand[nc++] = s->used[i];
    if (nc == 0)
        return 0;
    int b = cand[shake_rnd(s, (uint32_t)nc)];
    lens[b] = lens[a];
    lens[a] = (uint8_t)want;
    return 1;
}

/* Applies nmoves moves; returns 1 when at least one applied. */
static int draw_moves(shake_t *s, uint8_t *lens, int nmoves)
{
    int ok = 0;
    for (int mv = 0; mv < nmoves; mv++) {
        uint32_t k = shake_rnd(s, 10);
        if (k < 5)
            ok |= neighbor_swap(s, lens);
        else if (k < 9)
            ok |= directed_move(s, lens);
        else
            ok |= random_move(s, lens);
    }
    return ok;
}

/* Encoded bytes of the blocks selected by sel (all blocks when NULL)
 * under lens, per block into blk; SIZE_MAX when the table does not build
 * or a block does not encode. */
static size_t price(shake_t *s, const uint8_t *lens, const uint8_t *sel, size_t *blk)
{
    if (pivco_build_codec_table(s->cfg, lens, s->tab) != PIVCO_OK)
        return SIZE_MAX;
    size_t total = 0;
    for (int i = 0; i < s->nblk; i++) {
        if (sel && !sel[i])
            continue;
        size_t off = (size_t)i * s->block_size;
        size_t bn = s->n - off < s->block_size ? s->n - off : s->block_size;
        size_t ol = s->out_cap;
        if (pivco_encode(s->enc, s->tab, s->in + off, bn, s->out, &ol) != PIVCO_OK)
            return SIZE_MAX;
        blk[i] = ol;
        total += ol;
    }
    return total;
}

/* One hill-climb step on c. */
static void climb_step(shake_t *s, shake_cand_t *c)
{
    uint8_t cand[PIVCO_MAX_SYMBOLS];
    memcpy(cand, c->lens, sizeof(cand));
    if (!draw_moves(s, cand, 1))
        return;
    if (seen_insert(s, lens_hash(cand)))
        return;
    if (s->screen) {
        size_t sc = price(s, cand, s->screen, s->cand_blk);
        size_t sb = 0;
        for (int i = 0; i < s->nblk; i++)
            if (s->screen[i])
                sb += c->blk[i];
        if (sc == SIZE_MAX || sc > sb)
            return;
    }
    size_t cost = price(s, cand, NULL, s->cand_blk);
    if (cost >= c->cost)
        return;
    c->cost = cost;
    memcpy(c->lens, cand, sizeof(cand));
    memcpy(c->blk, s->cand_blk, (size_t)s->nblk * sizeof(size_t));
}

static void race(shake_t *s, shake_cand_t *c, int steps)
{
    for (int i = 1; i < SHAKE_STARTS; i++) {
        for (int tries = 0; tries < 16 && !c[i].alive; tries++) {
            memcpy(c[i].lens, c[0].lens, sizeof(c[i].lens));
            if (!draw_moves(s, c[i].lens, 1 + (int)shake_rnd(s, SHAKE_BURST)))
                continue;
            if (seen_insert(s, lens_hash(c[i].lens)))
                continue;
            c[i].cost = price(s, c[i].lens, NULL, c[i].blk);
            if (c[i].cost != SIZE_MAX)
                c[i].alive = 1;
        }
    }
    int live = 0;
    for (int i = 0; i < SHAKE_STARTS; i++)
        live += c[i].alive;
    int rounds = 1;
    while ((1 << (rounds - 1)) < SHAKE_STARTS)
        rounds++;
    for (int r = 0; r < rounds; r++) {
        int per = steps / rounds / live;
        for (int i = 0; i < SHAKE_STARTS; i++)
            if (c[i].alive)
                for (int k = 0; k < per; k++)
                    climb_step(s, &c[i]);
        int drop = live / 2;
        for (int d = 0; d < drop; d++) {
            int w = -1;
            for (int i = 0; i < SHAKE_STARTS; i++)
                if (c[i].alive && (w < 0 || c[i].cost > c[w].cost))
                    w = i;
            c[w].alive = 0;
            live--;
        }
    }
}

int pivco_shake_lengths(pivco_encoder_t *enc, const pivco_cfg_t *cfg,
                        const uint8_t *in, size_t n, size_t block_size,
                        uint8_t lengths[PIVCO_MAX_SYMBOLS])
{
    if (!enc || !in || !lengths)
        return PIVCO_ERR_NULL;
    if (!cfg)
        cfg = &pivco_cfg_default;
    if (block_size < 1 || block_size > PIVCO_WIRE_MAX_N)
        return PIVCO_ERR_BAD_CFG;
    if (cfg->shake_steps <= 0 || n == 0)
        return PIVCO_OK;

    uint64_t freq[PIVCO_MAX_SYMBOLS] = { 0 };
    for (size_t i = 0; i < n; i++)
        freq[in[i]]++;

    shake_t s;
    memset(&s, 0, sizeof(s));
    for (int sym = 0; sym < PIVCO_MAX_SYMBOLS; sym++)
        if (freq[sym] && (double)freq[sym] < SHAKE_FREEZE_SHARE * (double)n)
            s.used[s.nu++] = sym;
    if (s.nu < 3)
        return PIVCO_OK;

    s.enc = enc;
    s.cfg = cfg;
    s.in = in;
    s.n = n;
    s.block_size = block_size;
    s.nblk = (int)((n + block_size - 1) / block_size);
    s.out_cap = 2 * block_size + 4096;
    s.rng = SHAKE_SEED;
    uint32_t seen_cap = 1024;
    while (seen_cap < 4u * (uint32_t)(cfg->shake_steps + SHAKE_STARTS) && seen_cap < (1u << 20))
        seen_cap <<= 1;
    s.seen_mask = seen_cap - 1;

    int rc = PIVCO_ERR_NULL;
    shake_cand_t c[SHAKE_STARTS];
    memset(c, 0, sizeof(c));
    s.tab = (pivco_table_t *)calloc(1, sizeof(pivco_table_t));
    s.out = (uint8_t *)malloc(s.out_cap);
    s.cand_blk = (size_t *)malloc((size_t)s.nblk * sizeof(size_t));
    s.seen = (uint64_t *)calloc(seen_cap, sizeof(uint64_t));
    int ok = s.tab && s.out && s.cand_blk && s.seen;
    for (int i = 0; i < SHAKE_STARTS; i++) {
        c[i].blk = (size_t *)malloc((size_t)s.nblk * sizeof(size_t));
        c[i].cost = SIZE_MAX;
        ok = ok && c[i].blk;
    }
    if (!ok)
        goto done;
    if (SHAKE_SCREEN < s.nblk) {
        s.screen = (uint8_t *)calloc((size_t)s.nblk, 1);
        if (!s.screen)
            goto done;
        for (int j = 0; j < SHAKE_SCREEN; j++)
            s.screen[j * s.nblk / SHAKE_SCREEN] = 1;
    }

    /* Pricing runs real encodes; keep them out of the caller's stats. */
    pivco_ctx_stats_t saved = enc->stats;
    memcpy(c[0].lens, lengths, sizeof(c[0].lens));
    seen_insert(&s, lens_hash(c[0].lens));
    c[0].cost = price(&s, c[0].lens, NULL, c[0].blk);
    if (c[0].cost == SIZE_MAX) {
        enc->stats = saved;
        rc = PIVCO_ERR_CORRUPT;
        goto done;
    }
    c[0].alive = 1;
    race(&s, c, cfg->shake_steps);
    enc->stats = saved;

    int b = 0;
    for (int i = 1; i < SHAKE_STARTS; i++)
        if (c[i].cost < c[b].cost)
            b = i;
    memcpy(lengths, c[b].lens, sizeof(c[b].lens));
    rc = PIVCO_OK;

done:
    for (int i = 0; i < SHAKE_STARTS; i++)
        free(c[i].blk);
    free(s.screen);
    free(s.seen);
    free(s.cand_blk);
    free(s.out);
    free(s.tab);
    return rc;
}
