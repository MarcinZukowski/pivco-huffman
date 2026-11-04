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
 * search is deterministic (fixed seed).
 *
 * Pricing runs with a node memo (node_memo.h) attached to the encoder:
 * a node whose symbol sets and depth were priced before adds its known
 * record bytes, summed over the blocks, instead of being coded again. */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "node_memo.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SHAKE_STARTS 10
#define SHAKE_BURST 8
#define SHAKE_SCREEN 2
#define SHAKE_FREEZE_SHARE 0.25
#define SHAKE_SEED 0x9E3779B97F4A7C15ull

/* Node memo size, in entries of 24 bytes.
 *
 * Each step adds one entry per node of the candidate's tree that no
 * earlier tree had.  On silesia that is 0.9 to 8.2 new entries per
 * step, fewer as the search goes on.
 *
 * The memo stops recording at half full.  SHAKE_MEMO_SLOTS_PER_STEP
 * keeps the worst measured case about a quarter full.  The size is a
 * power of two, from SHAKE_MEMO_MIN_SLOTS (96 KiB) to
 * SHAKE_MEMO_MAX_SLOTS (24 MiB).  When a long search fills the memo,
 * new nodes are no longer recorded and are coded each time they come
 * back. */
#define SHAKE_MEMO_SLOTS_PER_STEP 32u
#define SHAKE_MEMO_MIN_SLOTS (1u << 12)
#define SHAKE_MEMO_MAX_SLOTS (1u << 20)
#define SHAKE_MEMO_UNSET UINT32_MAX

/* A node's record bytes summed over the screen blocks and over all
 * blocks (SHAKE_MEMO_UNSET until priced), and the blocks where it is
 * FSE-coded. */
typedef struct {
    uint64_t key;                 /* 0: empty */
    uint64_t coded;
    uint32_t screen_bytes;
    uint32_t all_bytes;
} memo_ent_t;

typedef struct {
    uint8_t lens[PIVCO_MAX_SYMBOLS];
    size_t cost;
    size_t screen_cost;           /* the screen blocks' share of cost */
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
    int used[PIVCO_MAX_SYMBOLS];
    int nu;
    uint64_t rng;
    uint64_t *seen;
    uint32_t seen_mask;
    pivco_node_memo_t *memo;      /* NULL: price without one */
    memo_ent_t *ment;
    uint32_t ment_mask;
    uint32_t ment_count;
    uint64_t key[PIVCO_MAX_SYMBOLS - 1];      /* per sched record */
    memo_ent_t *ent[PIVCO_MAX_SYMBOLS - 1];   /* per sched record */
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

static uint64_t key_mix(uint64_t h, uint64_t v)
{
    h ^= v;
    h *= 0x100000001B3ull;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return h;
}

/* h extended by the set of symbols at ranks [lo, hi) of t. */
static uint64_t set_key(uint64_t h, const pivco_table_t *t, int lo, int hi)
{
    uint64_t w[4] = { 0 };
    for (int r = lo; r < hi; r++) {
        int sym = t->dec.rank_to_sym[r];
        w[sym >> 6] |= 1ull << (sym & 63);
    }
    for (int k = 0; k < 4; k++)
        h = key_mix(h, w[k]);
    return h;
}

/* Keys sched record rec, covering ranks [lo, hi) at depth, and the
 * records below it: a flat region on its set and D, an internal node on
 * its left set, right set and depth (the static table's commit gate
 * reads the depth). */
static void memo_keys(uint64_t *key, const pivco_table_t *t,
                      int rec, int lo, int hi, int depth)
{
    const pivco_sched_rec_t *r = &t->dec.sched[rec];
    pivco_sched_kind_t kind = pivco_sched_kind(r);
    if (kind == PIVCO_SCHED_FLAT) {
        uint64_t h = key_mix(1, (uint64_t)pivco_sched_flat_depth(r));
        key[rec] = set_key(h, t, lo, hi) | 1;
        return;
    }
    int mid = r->param + 1;
    uint64_t h = key_mix(2, (uint64_t)depth);
    h = set_key(h, t, lo, mid);
    h = key_mix(h, 3);
    key[rec] = set_key(h, t, mid, hi) | 1;
    if (kind == PIVCO_SCHED_FULL) {
        memo_keys(key, t, rec + 1, lo, mid, depth + 1);
        memo_keys(key, t, rec + r->right, mid, hi, depth + 1);
    } else if (kind == PIVCO_SCHED_LEAF_LEFT) {
        memo_keys(key, t, rec + 1, mid, hi, depth + 1);
    }
}

/* key's memo entry, a new unpriced one when key is not there yet; NULL
 * when it is not there and the memo is half full. */
static memo_ent_t *memo_entry(shake_t *s, uint64_t key)
{
    uint32_t i = (uint32_t)(key >> 32) & s->ment_mask;
    while (s->ment[i].key) {
        if (s->ment[i].key == key)
            return &s->ment[i];
        i = (i + 1) & s->ment_mask;
    }
    if (s->ment_count >= (s->ment_mask + 1) / 2)
        return NULL;
    memo_ent_t *e = &s->ment[i];
    e->key = key;
    e->coded = 0;
    e->screen_bytes = SHAKE_MEMO_UNSET;
    e->all_bytes = SHAKE_MEMO_UNSET;
    s->ment_count++;
    return e;
}

/* Encodes block i under s->tab; its length (without the records the
 * memo knows), or SIZE_MAX when it does not encode. */
static size_t price_block(shake_t *s, int i)
{
    if (s->memo)
        s->memo->block = i;
    size_t off = (size_t)i * s->block_size;
    size_t bn = s->n - off < s->block_size ? s->n - off : s->block_size;
    size_t ol = s->out_cap;
    if (pivco_encode(s->enc, s->tab, s->in + off, bn, s->out, &ol) != PIVCO_OK)
        return SIZE_MAX;
    return ol;
}

/* Encoded bytes of the segment under lens: of all blocks when full,
 * else of the screen blocks only.  *screen_cost (when not NULL) gets
 * the screen blocks' share.  SIZE_MAX when the table does not build or
 * a block does not encode.
 *
 * The screen blocks are encoded first, so the memo can take each new
 * node's screen sum before the other blocks add to it. */
static size_t price(shake_t *s, const uint8_t *lens, int full, size_t *screen_cost)
{
    if (pivco_build_codec_table(s->cfg, lens, s->tab) != PIVCO_OK)
        return SIZE_MAX;
    pivco_node_memo_t *m = s->memo;
    int nrec = s->tab->dec.sched_len;
    size_t total = 0;
    size_t scr = 0;
    if (m) {
        memo_keys(s->key, s->tab, 0, 0, s->tab->dec.num_ranks, 0);
        for (int r = 0; r < nrec; r++) {
            memo_ent_t *e = memo_entry(s, s->key[r]);
            uint32_t have = !e ? SHAKE_MEMO_UNSET : full ? e->all_bytes : e->screen_bytes;
            s->ent[r] = e;
            m->known[r] = have != SHAKE_MEMO_UNSET;
            m->bytes[r] = 0;
            m->coded[r] = m->known[r] ? e->coded : 0;
            if (m->known[r]) {
                total += have;
                scr += e->screen_bytes;
            }
        }
    }
    for (int i = 0; i < s->nblk; i++) {
        if (!s->screen || !s->screen[i])
            continue;
        size_t ol = price_block(s, i);
        if (ol == SIZE_MAX)
            return SIZE_MAX;
        total += ol;
        scr += ol;
    }
    if (m)
        for (int r = 0; r < nrec; r++)
            if (s->ent[r] && !m->known[r]) {
                s->ent[r]->screen_bytes = m->bytes[r];
                s->ent[r]->coded |= m->coded[r];
            }
    if (full) {
        for (int i = 0; i < s->nblk; i++) {
            if (s->screen && s->screen[i])
                continue;
            size_t ol = price_block(s, i);
            if (ol == SIZE_MAX)
                return SIZE_MAX;
            total += ol;
        }
        if (m)
            for (int r = 0; r < nrec; r++)
                if (s->ent[r] && !m->known[r]) {
                    s->ent[r]->all_bytes = m->bytes[r];
                    s->ent[r]->coded |= m->coded[r];
                }
    }
    if (screen_cost)
        *screen_cost = scr;
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
        size_t sc = price(s, cand, 0, NULL);
        if (sc == SIZE_MAX || sc > c->screen_cost)
            return;
    }
    size_t screen_cost;
    size_t cost = price(s, cand, 1, &screen_cost);
    if (cost >= c->cost)
        return;
    c->cost = cost;
    c->screen_cost = screen_cost;
    memcpy(c->lens, cand, sizeof(cand));
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
            c[i].cost = price(s, c[i].lens, 1, &c[i].screen_cost);
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
    s.seen = (uint64_t *)calloc(seen_cap, sizeof(uint64_t));
    int ok = s.tab && s.out && s.seen;
    /* coded[] has a bit per block: more blocks than that price without
     * a memo. */
    if (s.nblk <= PIVCO_MEMO_MAX_BLOCKS) {
        /* The steps plus the starting points' first pricing. */
        uint64_t memo_want = (uint64_t)SHAKE_MEMO_SLOTS_PER_STEP *
                             (uint64_t)(cfg->shake_steps + SHAKE_STARTS);
        uint32_t memo_cap = SHAKE_MEMO_MIN_SLOTS;
        while (memo_cap < memo_want && memo_cap < SHAKE_MEMO_MAX_SLOTS)
            memo_cap <<= 1;
        s.memo = (pivco_node_memo_t *)calloc(1, sizeof(pivco_node_memo_t));
        s.ment = (memo_ent_t *)calloc(memo_cap, sizeof(memo_ent_t));
        s.ment_mask = memo_cap - 1;
        ok = ok && s.memo && s.ment;
    }
    for (int i = 0; i < SHAKE_STARTS; i++)
        c[i].cost = SIZE_MAX;
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
    pivco_scratch_t *sc = (pivco_scratch_t *)enc->internal;
    sc->memo = s.memo;
    memcpy(c[0].lens, lengths, sizeof(c[0].lens));
    seen_insert(&s, lens_hash(c[0].lens));
    c[0].cost = price(&s, c[0].lens, 1, &c[0].screen_cost);
    if (c[0].cost == SIZE_MAX) {
        sc->memo = NULL;
        enc->stats = saved;
        rc = PIVCO_ERR_CORRUPT;
        goto done;
    }
    c[0].alive = 1;
    race(&s, c, cfg->shake_steps);
    sc->memo = NULL;
    enc->stats = saved;

    int b = 0;
    for (int i = 1; i < SHAKE_STARTS; i++)
        if (c[i].cost < c[b].cost)
            b = i;
    memcpy(lengths, c[b].lens, sizeof(c[b].lens));
    rc = PIVCO_OK;

done:
    free(s.ment);
    free(s.memo);
    free(s.screen);
    free(s.seen);
    free(s.out);
    free(s.tab);
    return rc;
}
