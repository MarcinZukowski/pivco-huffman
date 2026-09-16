/* k=2 bit-context tANS over partition bitmaps and flat regions -- wire
 * id PIVCO_FSE_K2_ID.  The k=1 coder (pivco_k1.c) with one more bit of
 * context; see there for the mechanics they share (min-freq-1
 * normalization, payload layout, interleaved segments, absorbed first
 * symbols, the pricing gate).
 *
 * Model: P(bit | previous two bits).  Four probabilities, each snapped
 * to one of 4 grid values, so a recipe is one byte on the wire: bits
 * 2c..2c+1 hold context c's grid index, where c = (newest previous bit
 * << 1) | the bit before it.  The carry into a byte is the last two
 * bits of the byte before it, in the same order.
 *
 * Complement symmetry halves the catalog.  Complementing a bit string
 * complements every context and every bit, and the grid is closed
 * under p -> 1-p, so P_r(byte | carry) = P_r'(~byte | ~carry) for the
 * mirror recipe r' (context ~c gets the mirrored grid value of context
 * c).  Only the canonical recipes (r <= r') carry tables: four
 * byte-alphabet tANS tables each, one per carry, 2^PIVCO_K2_TABLELOG
 * states.  A region whose recipe is not canonical is coded as its
 * complement under the mirror recipe, from initial carry 3, and the
 * decoder complements the region back after the loop.  The tables are
 * built on first use from the recipes alone.
 *
 * Decoder: e = D[carry][state]; byte = e & 0xFF; carry = byte >> 6;
 * state = (e >> 12) + read((e >> 8) & 0xF). */

#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#include "bitstream.h"

#include "pivco_fse.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define K2_L     PIVCO_K2_TABLELOG
#define K2_TSIZE (1 << K2_L)

/* Probability grid, units of 1/4096, symmetric under p -> 1-p.  A
 * 2-bit index on the wire names one entry; changing this table changes
 * the wire. */
static const uint16_t K2_PTAB[4] = { 128, 1024, 3072, 3968 };

typedef struct { uint32_t deltaNbBits; int32_t deltaFindState; } k2_stt_t;
typedef struct {
    uint32_t dec[4][K2_TSIZE];   /* sym | nbBits << 8 | base << 12 */
    uint16_t st[4][K2_TSIZE];    /* encoder state table */
    k2_stt_t tt[4][256];         /* encoder symbol transform */
    /* estimator: table cost minus model cost of 0x00 under carry 0 and
     * of 0xFF under carry 3, in bits */
    double delta00, deltaFF;
} k2_tabs_t;

/* The mirror recipe: context ~c gets the mirrored grid value of c. */
static inline int k2_mirror(int recipe)
{
    int m = 0;
    for (int c = 0; c < 4; c++) m |= (3 - ((recipe >> (2 * c)) & 3)) << (2 * (3 - c));
    return m;
}

/* The canonical recipe's tables, for every recipe (a non-canonical
 * recipe shares its mirror's).
 * TODO: built from doubles at first use; consider shipping the
 * normalized frequency vectors as pivco_fse_tables.h does for the
 * static schedule and building from integers. */
static k2_tabs_t *g_k2_cat[256];
static pthread_once_t g_k2_once = PTHREAD_ONCE_INIT;
static int g_k2_ok;

static double g_k2_thr[3];                     /* quantizer crossovers on p_hat */
static double g_k2_cost1[4], g_k2_cost0[4];    /* -log2 p, -log2 (1-p) per grid value */

static inline int k2_highbit(uint32_t v) { return 31 - __builtin_clz(v); }

static void k2_init(void)
{
    /* The grid index minimizing n1*(-log2 p) + n0*(-log2(1-p)) depends
     * only on p_hat = n1/(n0+n1); adjacent grid values p_a < p_b cross at
     * T = log((1-p_a)/(1-p_b)) / (log(p_b/p_a) + log((1-p_a)/(1-p_b))). */
    for (int k = 0; k < 3; k++) {
        double pa = K2_PTAB[k] / 4096.0, pb = K2_PTAB[k + 1] / 4096.0;
        double num = log2((1.0 - pa) / (1.0 - pb));
        g_k2_thr[k] = num / (log2(pb / pa) + num);
    }
    for (int k = 0; k < 4; k++) {
        double pk = K2_PTAB[k] / 4096.0;
        g_k2_cost1[k] = -log2(pk);
        g_k2_cost0[k] = -log2(1.0 - pk);
    }
}

/* ---- catalog build ---- */

static void k2_byte_probs(int recipe, int carry, double out[256])
{
    double p1[4];
    for (int c = 0; c < 4; c++) p1[c] = K2_PTAB[(recipe >> (2 * c)) & 3] / 4096.0;
    for (int v = 0; v < 256; v++) {
        double pr = 1.0;
        int c = carry;
        for (int j = 0; j < 8; j++) {
            int bit = (v >> j) & 1;
            pr *= bit ? p1[c] : 1.0 - p1[c];
            c = (bit << 1) | (c >> 1);
        }
        out[v] = pr;
    }
}

/* Min-freq-1 normalization to K2_TSIZE slots (as k1_normalize). */
static void k2_normalize(const double p[256], uint16_t f[256])
{
    double sum = 0.0;
    for (int v = 0; v < 256; v++) sum += p[v];
    int32_t used = 0;
    for (int v = 0; v < 256; v++) {
        uint32_t fs = (uint32_t)(p[v] / sum * (double)K2_TSIZE);
        if (fs == 0) fs = 1;
        f[v] = (uint16_t)fs;
        used += (int32_t)fs;
    }
    int32_t left = (int32_t)K2_TSIZE - used;
    while (left != 0) {
        int big = 0;
        for (int v = 1; v < 256; v++) if (f[v] > f[big]) big = v;
        if (left > 0) { f[big] = (uint16_t)(f[big] + left); break; }
        int32_t d = (int32_t)f[big] - 1;
        if (d > -left) d = -left;
        f[big] = (uint16_t)(f[big] - d);
        left += d;
    }
}

static void k2_build_carry(const double p[256], uint32_t *dec,
                           uint16_t *st, k2_stt_t *tt, uint16_t f[256])
{
    k2_normalize(p, f);

    /* FSE-style spread */
    uint16_t tsym[K2_TSIZE];
    {
        uint32_t step = (K2_TSIZE >> 1) + (K2_TSIZE >> 3) + 3, pos = 0;
        for (int s = 0; s < 256; s++)
            for (int i = 0; i < f[s]; i++) {
                tsym[pos] = (uint16_t)s;
                pos = (pos + step) & (K2_TSIZE - 1);
            }
    }
    /* decode entries */
    {
        uint16_t next[256];
        memcpy(next, f, sizeof(next));
        for (int x = 0; x < K2_TSIZE; x++) {
            int s = tsym[x];
            uint32_t X = next[s]++;
            int nb = K2_L - k2_highbit(X);
            uint32_t base = (X << nb) - K2_TSIZE;
            dec[x] = (uint32_t)s | ((uint32_t)nb << 8) | (base << 12);
        }
    }
    /* encoder tables */
    {
        uint32_t cumul = 0;
        for (int s = 0; s < 256; s++) {
            int maxBits = f[s] == 1 ? K2_L : K2_L - k2_highbit((uint32_t)f[s] - 1);
            tt[s].deltaNbBits    = ((uint32_t)maxBits << 16) - ((uint32_t)f[s] << maxBits);
            tt[s].deltaFindState = (int32_t)cumul - (int32_t)f[s];
            cumul += f[s];
        }
        uint16_t fill[256];
        fill[0] = 0;
        for (int s = 0; s < 255; s++) fill[s + 1] = (uint16_t)(fill[s] + f[s]);
        for (int x = 0; x < K2_TSIZE; x++)
            st[fill[tsym[x]]++] = (uint16_t)(K2_TSIZE + x);
    }
}

static void k2_build_catalog(void)
{
    k2_init();
    for (int recipe = 0; recipe < 256; recipe++) {
        if (k2_mirror(recipe) < recipe) continue;   /* built under its mirror */
        k2_tabs_t *t = (k2_tabs_t *)malloc(sizeof(*t));
        if (!t) return;                    /* g_k2_ok stays 0 */
        double probs[256];
        uint16_t f[256];
        for (int c = 0; c < 4; c++) {
            k2_byte_probs(recipe, c, probs);
            k2_build_carry(probs, t->dec[c], t->st[c], t->tt[c], f);
            if (c == 0) t->delta00 = (K2_L - log2((double)f[0x00])) + log2(probs[0x00]);
            if (c == 3) t->deltaFF = (K2_L - log2((double)f[0xFF])) + log2(probs[0xFF]);
        }
        g_k2_cat[recipe] = t;
        g_k2_cat[k2_mirror(recipe)] = t;
    }
    g_k2_ok = 1;
}

int pivco_k2_prebuild(void)
{
    pthread_once(&g_k2_once, k2_build_catalog);
    return g_k2_ok ? 0 : -1;
}

/* ---- encoder ---- */

#define K2_SEGS PIVCO_K2_SEGS
_Static_assert(PIVCO_K2_SPLIT_MIN >= PIVCO_K2_SEGS * (PIVCO_K2_SEGS - 1),
               "PIVCO_K2_SPLIT_MIN must be at least SEGS*(SEGS-1)");
#if K2_SEGS != 4 && K2_SEGS != 8
#error "PIVCO_K2_SEGS must be 4 or 8"
#endif

static inline void k2_enc_sym(BIT_CStream_t *w, const k2_tabs_t *t, uint32_t *state,
                              int carry, int s)
{
    const k2_stt_t *tt = &t->tt[carry][s];
    uint32_t nb = (*state + tt->deltaNbBits) >> 16;
    BIT_addBits(w, *state, nb);
    *state = t->st[carry][(*state >> nb) + (uint32_t)tt->deltaFindState];
}

static inline uint32_t k2_init_state(const k2_tabs_t *t, int carry, int s)
{
    const k2_stt_t *tt = &t->tt[carry][s];
    uint32_t nb = (tt->deltaNbBits + (1u << 15)) >> 16;
    uint32_t v  = (nb << 16) - tt->deltaNbBits;
    return t->st[carry][(v >> nb) + (uint32_t)tt->deltaFindState];
}

static int k2_quantize(uint32_t n0, uint32_t n1)
{
    /* index = #{k : p_hat > T[k]}, with p_hat > T <=> n1*(1-T) > n0*T */
    int k = 0;
    while (k < 3 && (double)n1 * (1.0 - g_k2_thr[k]) > (double)n0 * g_k2_thr[k]) k++;
    return k;
}

/* The byte coded at j (complemented under a mirrored recipe) and the
 * carry into it: the last two bits of the byte before it, `c0` before
 * the first. */
#define K2_BYTE(j) ((s)[(j)] ^ xm)
#define K2_CARRY_BEFORE(j) ((j) ? K2_BYTE((j) - 1) >> 6 : c0)

pivco_fse_status_t pivco_k2_compress(const void *src, size_t src_len,
                                      void *dst, size_t dst_cap,
                                      size_t max_len, size_t *out_len)
{
    if (src_len == 0) { *out_len = 0; return PIVCO_FSE_OK; }
    if (dst_cap < 1 + sizeof(size_t)) return PIVCO_FSE_ERR_DST_FULL;   /* BIT_initCStream's minimum */
    if (pivco_k2_prebuild() != 0) return PIVCO_FSE_ERR_INTERNAL;
    const uint8_t *s = (const uint8_t *)src;

    /* Recipe: the (two previous bits, bit) triple counts, 64 bits at a
     * time.  sh1 is the stream shifted up one bit and sh2 up two (the
     * previous words' top bits carried in, 0 before the first bit), so
     * per position sh1 is the newest previous bit, sh2 the older one,
     * and popcounts of the eight and-masks are the triple counts. */
    uint32_t cnt[4][2];   /* [context][bit] */
    uint64_t n00 = 0, nFF = 0;   /* 0x00 and 0xFF bytes, for the estimate */
    {
        uint64_t c[4][2] = {{0}};
        uint64_t carry1 = 0, carry2 = 0;
        const uint64_t lo7 = 0x7F7F7F7F7F7F7F7Full;
        size_t j = 0;
        for (; j + 8 <= src_len; j += 8) {
            uint64_t w;
            memcpy(&w, s + j, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            w = __builtin_bswap64(w);
#endif
            uint64_t sh1 = (w << 1) | carry1;
            uint64_t sh2 = (w << 2) | carry2;
            c[0][0] += (uint64_t)__builtin_popcountll(~w & ~sh1 & ~sh2);
            c[0][1] += (uint64_t)__builtin_popcountll( w & ~sh1 & ~sh2);
            c[1][0] += (uint64_t)__builtin_popcountll(~w & ~sh1 &  sh2);
            c[1][1] += (uint64_t)__builtin_popcountll( w & ~sh1 &  sh2);
            c[2][0] += (uint64_t)__builtin_popcountll(~w &  sh1 & ~sh2);
            c[2][1] += (uint64_t)__builtin_popcountll( w &  sh1 & ~sh2);
            c[3][0] += (uint64_t)__builtin_popcountll(~w &  sh1 &  sh2);
            c[3][1] += (uint64_t)__builtin_popcountll( w &  sh1 &  sh2);
            n00 += (uint64_t)__builtin_popcountll(~(((w & lo7) + lo7) | w | lo7));
            nFF += (uint64_t)__builtin_popcountll(~(((~w & lo7) + lo7) | ~w | lo7));
            carry1 = w >> 63;
            carry2 = w >> 62;
        }
        if (j < src_len) {                 /* tail: mask to the real bits */
            size_t r = src_len - j;
            uint64_t w = 0;
            memcpy(&w, s + j, r);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            w = __builtin_bswap64(w);
#endif
            uint64_t mask = (1ull << (8 * r)) - 1;
            uint64_t sh1 = ((w << 1) | carry1) & mask;
            uint64_t sh2 = ((w << 2) | carry2) & mask;
            c[0][0] += (uint64_t)__builtin_popcountll(~w & ~sh1 & ~sh2 & mask);
            c[0][1] += (uint64_t)__builtin_popcountll( w & ~sh1 & ~sh2);
            c[1][0] += (uint64_t)__builtin_popcountll(~w & ~sh1 &  sh2);
            c[1][1] += (uint64_t)__builtin_popcountll( w & ~sh1 &  sh2);
            c[2][0] += (uint64_t)__builtin_popcountll(~w &  sh1 & ~sh2);
            c[2][1] += (uint64_t)__builtin_popcountll( w &  sh1 & ~sh2);
            c[3][0] += (uint64_t)__builtin_popcountll(~w &  sh1 &  sh2);
            c[3][1] += (uint64_t)__builtin_popcountll( w &  sh1 &  sh2);
            n00 += (uint64_t)__builtin_popcountll(~(((w & lo7) + lo7) | w | lo7)) - (8 - r);
            nFF += (uint64_t)__builtin_popcountll(~(((~w & lo7) + lo7) | ~w | lo7) & mask);
        }
        for (int k = 0; k < 4; k++) {
            cnt[k][0] = (uint32_t)c[k][0];
            cnt[k][1] = (uint32_t)c[k][1];
        }
    }
    int recipe = 0;
    for (int k = 0; k < 4; k++) recipe |= k2_quantize(cnt[k][0], cnt[k][1]) << (2 * k);
    const k2_tabs_t *t = g_k2_cat[recipe];
    /* A non-canonical recipe codes the complemented region under its
     * mirror, from carry 3. */
    const int mirrored = k2_mirror(recipe) < recipe;
    const uint8_t xm = mirrored ? 0xFF : 0;
    const int c0 = mirrored ? 3 : 0;

    /* Price before coding (as k1): the ideal bit cost under the recipe,
     * minus the absorbed symbols, plus L per state and the recipe byte. */
    if (PIVCO_K2_EST_SKIP > 0.0 && max_len) {
        const double *c1 = g_k2_cost1, *c0 = g_k2_cost0;
        int gi[4];
        for (int k = 0; k < 4; k++) gi[k] = (recipe >> (2 * k)) & 3;
        double bits = 0.0;
        for (int k = 0; k < 4; k++) bits += cnt[k][1] * c1[gi[k]] + cnt[k][0] * c0[gi[k]];
        int nst = src_len >= PIVCO_K2_SPLIT_MIN ? K2_SEGS : 1;
        size_t q = (src_len + nst - 1) / nst;
        for (int sg = 0; sg < nst; sg++) {
            size_t j = (size_t)sg * q + q - 1;
            if (j >= src_len) continue;
            int ctx = q > 1 ? s[j - 1] >> 6 : 0, b = s[j];
            for (int k = 0; k < 8; k++) {
                int bit = (b >> k) & 1;
                bits -= bit ? c1[gi[ctx]] : c0[gi[ctx]];
                ctx = (bit << 1) | (ctx >> 1);
            }
        }
        if (mirrored) {
            bits += (double)n00 * t->deltaFF + (double)nFF * t->delta00;
        } else {
            bits += (double)n00 * t->delta00 + (double)nFF * t->deltaFF;
        }
        double est = 1.0 + (bits + nst * K2_L) / 8.0;
        if (est * PIVCO_K2_EST_SKIP > (double)max_len) return PIVCO_FSE_FALLBACK;
    }

    uint8_t *d = (uint8_t *)dst;
    d[0] = (uint8_t)recipe;
    BIT_CStream_t w;
    if (FSE_isError(BIT_initCStream(&w, d + 1, dst_cap - 1))) return PIVCO_FSE_ERR_DST_FULL;

    if (src_len >= PIVCO_K2_SPLIT_MIN) {
        size_t q = (src_len + K2_SEGS - 1) / K2_SEGS;
        uint32_t st[K2_SEGS];
        for (int sg = 0; sg < K2_SEGS; sg++) {
            size_t j = (size_t)sg * q + q - 1;
            st[sg] = j < src_len ? k2_init_state(t, q > 1 ? K2_BYTE(j - 1) >> 6 : c0, K2_BYTE(j)) : K2_TSIZE;
        }
        for (size_t i = q - 1; i-- > 0; ) {
            for (int sg = K2_SEGS - 1; sg >= 0; sg--) {
                size_t j = (size_t)sg * q + i;
                if (j >= src_len) continue;
                k2_enc_sym(&w, t, &st[sg], i ? K2_BYTE(j - 1) >> 6 : c0, K2_BYTE(j));
                if (K2_SEGS == 8 && sg == 4) BIT_flushBits(&w);
            }
            BIT_flushBits(&w);
        }
        for (int sg = K2_SEGS - 1; sg >= 0; sg--) {
            BIT_addBits(&w, st[sg] - K2_TSIZE, K2_L);
            if ((sg & 3) == 0) BIT_flushBits(&w);
        }
    } else {
        size_t j = src_len - 1;
        uint32_t state = k2_init_state(t, K2_CARRY_BEFORE(j), K2_BYTE(j));
        for (; j >= 4; j -= 4) {
            k2_enc_sym(&w, t, &state, K2_BYTE(j - 2) >> 6, K2_BYTE(j - 1));
            k2_enc_sym(&w, t, &state, K2_BYTE(j - 3) >> 6, K2_BYTE(j - 2));
            k2_enc_sym(&w, t, &state, K2_BYTE(j - 4) >> 6, K2_BYTE(j - 3));
            k2_enc_sym(&w, t, &state, K2_CARRY_BEFORE(j - 4), K2_BYTE(j - 4));
            BIT_flushBits(&w);
        }
        while (j-- > 0) {
            k2_enc_sym(&w, t, &state, K2_CARRY_BEFORE(j), K2_BYTE(j));
            BIT_flushBits(&w);
        }
        BIT_addBits(&w, state - K2_TSIZE, K2_L);
        BIT_flushBits(&w);
    }
    size_t bits = BIT_closeCStream(&w);          /* 0 = ran out of room */
    if (bits == 0) return PIVCO_FSE_FALLBACK;
    *out_len = 1 + bits;
    if (*out_len >= src_len) return PIVCO_FSE_FALLBACK;
    return PIVCO_FSE_OK;
}

/* ---- decoder ---- */

/* One symbol: table entry by (carry, state), byte out, carry = its top
 * two bits, next state = base + the entry's bits from the stream. */
#define K2_STEP(cr, stv, out)                                           \
    do {                                                                \
        uint32_t e_ = t->dec[cr][stv];                                  \
        (out) = (uint8_t)e_;                                            \
        (cr) = (int)((e_ >> 6) & 3);                                    \
        (stv) = (e_ >> 12) + (uint32_t)BIT_readBits(&bd, (e_ >> 8) & 0xF); \
    } while (0)

pivco_fse_status_t pivco_k2_decompress(const void *src, size_t src_len,
                                        void *dst, size_t dst_cap,
                                        size_t dst_expected,
                                        size_t *out_len)
{
    if (dst_cap < dst_expected) return PIVCO_FSE_ERR_DST_FULL;
    if (dst_expected == 0) { *out_len = 0; return PIVCO_FSE_OK; }
    if (src_len < 2) return PIVCO_FSE_ERR_BAD_INPUT;
    const uint8_t *p = (const uint8_t *)src;
    if (pivco_k2_prebuild() != 0) return PIVCO_FSE_ERR_INTERNAL;
    const k2_tabs_t *t = g_k2_cat[p[0]];
    const int mirrored = k2_mirror(p[0]) < p[0];
    const int c0 = mirrored ? 3 : 0;
    uint8_t *out = (uint8_t *)dst;
    size_t n = dst_expected;
#if PIVCO_K2_PREFETCH
    {
        const char *tb = (const char *)t->dec;
        for (size_t l = 0; l < sizeof(t->dec) / 64; l++) __builtin_prefetch(tb + 64 * l, 0, 3);
    }
#endif

    BIT_DStream_t bd;
    if (FSE_isError(BIT_initDStream(&bd, p + 1, src_len - 1)))
        return PIVCO_FSE_ERR_BAD_INPUT;

    if (n >= PIVCO_K2_SPLIT_MIN) {
        size_t q = (n + K2_SEGS - 1) / K2_SEGS;
        size_t full = n - (K2_SEGS - 1) * q;   /* rounds where every segment has a symbol */
        uint32_t st[K2_SEGS];
        int cr[K2_SEGS];
        uint8_t *o[K2_SEGS];
        for (int sg = 0; sg < K2_SEGS; sg++) {
            if (K2_SEGS == 8 && sg == 4) BIT_reloadDStream(&bd);
            st[sg] = (uint32_t)BIT_readBits(&bd, K2_L);
            cr[sg] = c0;
            o[sg] = out + (size_t)sg * q;
        }
        size_t i = 0;
        for (; i < full; i++) {
            if (BIT_reloadDStream(&bd) != BIT_DStream_unfinished) break;
            K2_STEP(cr[0], st[0], o[0][i]);
            K2_STEP(cr[1], st[1], o[1][i]);
            K2_STEP(cr[2], st[2], o[2][i]);
            K2_STEP(cr[3], st[3], o[3][i]);
#if K2_SEGS == 8
            /* No status check: the round-top reload guarantees the
             * first four steps' bits, the encoder flushed after the same
             * four, and a short stream is caught by the next round-top
             * check (a state stays in range whatever bits come back). */
            BIT_reloadDStream(&bd);
            K2_STEP(cr[4], st[4], o[4][i]);
            K2_STEP(cr[5], st[5], o[5][i]);
            K2_STEP(cr[6], st[6], o[6][i]);
            K2_STEP(cr[7], st[7], o[7][i]);
#endif
        }
        for (; i < q; i++) {
            if (BIT_reloadDStream(&bd) == BIT_DStream_overflow && i + 1 < q) return PIVCO_FSE_ERR_BAD_INPUT;
            for (int sg = 0; sg < K2_SEGS; sg++) {
                size_t j = (size_t)sg * q + i;
                if (j >= n) break;
                if (K2_SEGS == 8 && sg == 4) BIT_reloadDStream(&bd);
                K2_STEP(cr[sg], st[sg], out[j]);
            }
        }
    } else {
        int cr = c0;
        uint32_t st = (uint32_t)BIT_readBits(&bd, K2_L);
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            if (BIT_reloadDStream(&bd) != BIT_DStream_unfinished) break;
            K2_STEP(cr, st, out[j]);
            K2_STEP(cr, st, out[j + 1]);
            K2_STEP(cr, st, out[j + 2]);
            K2_STEP(cr, st, out[j + 3]);
        }
        for (; j < n; j++) {
            if (BIT_reloadDStream(&bd) == BIT_DStream_overflow && j + 1 < n) return PIVCO_FSE_ERR_BAD_INPUT;
            K2_STEP(cr, st, out[j]);
        }
    }
    BIT_reloadDStream(&bd);
    if (bd.ptr != bd.start || bd.bitsConsumed < 64) return PIVCO_FSE_ERR_BAD_INPUT;
    /* A mirrored region was decoded complemented. */
    if (mirrored) {
        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            uint64_t w;
            memcpy(&w, out + j, 8);
            w = ~w;
            memcpy(out + j, &w, 8);
        }
        for (; j < n; j++) out[j] = (uint8_t)~out[j];
    }
    *out_len = n;
    return PIVCO_FSE_OK;
}
