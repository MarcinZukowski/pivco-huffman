/* k=1 bit-context tANS over partition bitmaps and flat regions -- wire
 * id PIVCO_FSE_K1_ID.  See pivco_fse.h for the interface contract and
 * the tuning knobs.
 *
 * Model: P(bit | previous bit).  Two probabilities, each snapped to one
 * of 16 grid values, so a recipe is one byte on the wire (low nibble =
 * context "previous bit 0", high nibble = "previous bit 1").  The 256
 * recipes form a complete catalog: per recipe, two byte-alphabet tANS
 * tables (one per carry = the last bit of the previous byte) with
 * 2^PIVCO_K1_TABLELOG states, every byte value's probability being the
 * chain-rule product of its eight bit probabilities.  Tables are built
 * on first use from the recipe alone; nothing but the recipe byte is
 * transmitted, and the decoder never builds anything per region.
 *
 * Normalization is min-freq-1: a byte value the model calls negligible
 * keeps one slot, and there is no escape symbol.  Measured against a
 * 257-symbol ESC alphabet on the L3 streams and the datasets: the
 * escape fires on 2-4% of bytes wherever this coder carries weight (the
 * two-parameter model is crude), so the floor's one slot beats ESC's
 * 8 raw bits plus its own symbol, and the decode loop loses its
 * compare.  The floor's mass tax shows only on dominant-symbol streams,
 * where commit-if-smaller drops the candidate anyway.
 *
 * Payload: [recipe:u8][FSE bitstream: bits pushed LSB-first, terminator
 * 1-bit, zero-padded to a byte].  The encoder walks the bytes backward;
 * the decoder reads backward from the terminator, through bitstream.h's
 * BIT_CStream / BIT_DStream, the same reader the static path uses.
 * Regions of at least PIVCO_K1_SPLIT_MIN bytes are PIVCO_K1_SEGS
 * contiguous segments with their own carry chain and state, their
 * operations interleaved in the one bitstream so the decode loop
 * overlaps that many dependent-lookup chains; smaller regions use one
 * state.  Each segment's first-encoded symbol is absorbed into its
 * initial state (FSE's initCState2), which is only valid because those
 * symbols are the last reads in stream order.
 *
 * Decoder: e = D[carry][state]; byte = e & 0xFF; carry = byte >> 7;
 * state = (e >> 12) + read((e >> 8) & 0xF).  The region's tables are
 * prefetched up front, since consecutive regions almost always use
 * different recipes.  Encoder: the recipe comes from popcounts of the
 * bit pairs, then the region is priced from those counts and the walk
 * runs only when the estimate can beat the caller's limit. */

#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#include "bitstream.h"

#include "pivco_fse.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define K1_L     PIVCO_K1_TABLELOG
#define K1_TSIZE (1 << K1_L)

/* Probability grid, units of 1/4096.  A 4-bit index on the wire names
 * one entry; changing this table changes the wire. */
static const uint16_t K1_PTAB[16] = {
      64,  128,  256,  512,  768, 1024, 1536, 2048,
    2560, 3072, 3328, 3584, 3840, 3968, 4032, 4064,
};

typedef struct { uint32_t deltaNbBits; int32_t deltaFindState; } k1_stt_t;
typedef struct {
    uint32_t dec[2][K1_TSIZE];   /* sym | nbBits << 8 | base << 12 */
    uint16_t st[2][K1_TSIZE];    /* encoder state table */
    k1_stt_t tt[2][256];         /* encoder per-symbol transform */
    double   delta00, deltaFF;   /* table cost - bit-model cost of 0x00 (carry 0) and 0xFF (carry 1) */
} k1_tabs_t;

/* The catalog: one entry per recipe, all built together on the coder's
 * first use (pthread_once, like pivco_fse_init).
 * TODO: the carry-1 table of a recipe is the carry-0 table of its complement
 * mirror (grid index k <-> 15-k) with the symbols complemented, so half
 * of it is redundant; see IDEAS.md. */
static k1_tabs_t *g_k1_cat[256];
static pthread_once_t g_k1_once = PTHREAD_ONCE_INIT;
static int g_k1_ok;

/* Encoder-side constant, derived once from K1_PTAB. */
static double  g_k1_thr[15];          /* quantizer crossovers on p_hat */
static double  g_k1_cost1[16], g_k1_cost0[16];   /* -log2 p, -log2 (1-p) per grid value */

static inline int k1_highbit(uint32_t v) { return 31 - __builtin_clz(v); }

static void k1_init(void)
{
    /* The recipe index minimizing n1*(-log2 p) + n0*(-log2(1-p)) depends
     * only on p_hat = n1/(n0+n1); adjacent grid values p_a < p_b cross at
     * T = log((1-p_a)/(1-p_b)) / (log(p_b/p_a) + log((1-p_a)/(1-p_b))). */
    for (int k = 0; k < 15; k++) {
        double pa = K1_PTAB[k] / 4096.0, pb = K1_PTAB[k + 1] / 4096.0;
        double num = log2((1.0 - pa) / (1.0 - pb));
        g_k1_thr[k] = num / (log2(pb / pa) + num);
    }
    for (int k = 0; k < 16; k++) {
        double pk = K1_PTAB[k] / 4096.0;
        g_k1_cost1[k] = -log2(pk);
        g_k1_cost0[k] = -log2(1.0 - pk);
    }
}

/* ---- catalog build ---- */

static void k1_byte_probs(int recipe, int carry, double out[256])
{
    const double p1[2] = { K1_PTAB[recipe & 15] / 4096.0,
                           K1_PTAB[recipe >> 4] / 4096.0 };
    for (int v = 0; v < 256; v++) {
        double pr = 1.0;
        int c = carry;
        for (int j = 0; j < 8; j++) {
            int bit = (v >> j) & 1;
            pr *= bit ? p1[c] : 1.0 - p1[c];
            c = bit;
        }
        out[v] = pr;
    }
}

/* Min-freq-1 normalization to K1_TSIZE slots: floor, one forced slot
 * for every value that rounds to zero, the rounding remainder to the
 * largest value.  When the forced ones outnumber the remainder (the
 * mass spread over a few dozen mid-probability values), the deficit
 * comes off the largest values, never below one.  Deterministic, so
 * both sides build identical tables. */
static void k1_normalize(const double p[256], uint16_t f[256])
{
    double sum = 0.0;
    for (int v = 0; v < 256; v++) sum += p[v];
    int32_t used = 0;
    for (int v = 0; v < 256; v++) {
        uint32_t fs = (uint32_t)(p[v] / sum * (double)K1_TSIZE);
        if (fs == 0) fs = 1;
        f[v] = (uint16_t)fs;
        used += (int32_t)fs;
    }
    int32_t left = (int32_t)K1_TSIZE - used;
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

static void k1_build_carry(const double p[256], uint32_t *dec,
                           uint16_t *st, k1_stt_t *tt, uint16_t f[256])
{
    k1_normalize(p, f);

    /* FSE-style spread */
    uint16_t tsym[K1_TSIZE];
    {
        uint32_t step = (K1_TSIZE >> 1) + (K1_TSIZE >> 3) + 3, pos = 0;
        for (int s = 0; s < 256; s++)
            for (int i = 0; i < f[s]; i++) {
                tsym[pos] = (uint16_t)s;
                pos = (pos + step) & (K1_TSIZE - 1);
            }
    }
    /* decode entries */
    {
        uint16_t next[256];
        memcpy(next, f, sizeof(next));
        for (int x = 0; x < K1_TSIZE; x++) {
            int s = tsym[x];
            uint32_t X = next[s]++;
            int nb = K1_L - k1_highbit(X);
            uint32_t base = (X << nb) - K1_TSIZE;
            dec[x] = (uint32_t)s | ((uint32_t)nb << 8) | (base << 12);
        }
    }
    /* encoder tables */
    {
        uint32_t cumul = 0;
        for (int s = 0; s < 256; s++) {
            int maxBits = f[s] == 1 ? K1_L : K1_L - k1_highbit((uint32_t)f[s] - 1);
            tt[s].deltaNbBits    = ((uint32_t)maxBits << 16) - ((uint32_t)f[s] << maxBits);
            tt[s].deltaFindState = (int32_t)cumul - (int32_t)f[s];
            cumul += f[s];
        }
        uint16_t fill[256];
        fill[0] = 0;
        for (int s = 0; s < 255; s++) fill[s + 1] = (uint16_t)(fill[s] + f[s]);
        for (int x = 0; x < K1_TSIZE; x++)
            st[fill[tsym[x]]++] = (uint16_t)(K1_TSIZE + x);
    }
}

static void k1_build_catalog(void)
{
    k1_init();
    for (int recipe = 0; recipe < 256; recipe++) {
        k1_tabs_t *t = (k1_tabs_t *)malloc(sizeof(*t));
        if (!t) return;                    /* g_k1_ok stays 0 */
        double probs[256];
        uint16_t f[256];
        /* Estimator calibration for the run bytes: normalization hands the
         * rounding remainder to the dominant value and forces a slot on
         * every negligible one, so the table prices 0x00 and 0xFF
         * differently from the bit model; regions are mostly those bytes
         * exactly where it matters. */
        for (int c = 0; c < 2; c++) {
            k1_byte_probs(recipe, c, probs);
            k1_build_carry(probs, t->dec[c], t->st[c], t->tt[c], f);
            if (c == 0) t->delta00 = (K1_L - log2((double)f[0x00])) + log2(probs[0x00]);
            else        t->deltaFF = (K1_L - log2((double)f[0xFF])) + log2(probs[0xFF]);
        }
        g_k1_cat[recipe] = t;
    }
    g_k1_ok = 1;
}

int pivco_k1_prebuild(void)
{
    pthread_once(&g_k1_once, k1_build_catalog);
    return g_k1_ok ? 0 : -1;
}

/* ---- encoder ---- */

/* Interleave width: PIVCO_K1_SEGS segments share one bitstream.  A
 * round adds/reads at most SEGS * K1_L bits; the 64-bit container holds
 * 57 usable bits after a reload, so 4 states need one reload per round
 * and 8 need one every 4 symbols. */
#define K1_SEGS PIVCO_K1_SEGS
/* The decoder's main loop runs n - (SEGS-1)*ceil(n/SEGS) rounds with no
 * per-segment bound check, which needs that count >= 0: true for every
 * n >= SEGS*(SEGS-1), so the split must not be set below that. */
_Static_assert(PIVCO_K1_SPLIT_MIN >= PIVCO_K1_SEGS * (PIVCO_K1_SEGS - 1),
               "PIVCO_K1_SPLIT_MIN must be at least SEGS*(SEGS-1)");
#if K1_SEGS != 4 && K1_SEGS != 8
#error "PIVCO_K1_SEGS must be 4 or 8"
#endif

static inline void k1_enc_sym(BIT_CStream_t *w, const k1_tabs_t *t, uint32_t *state,
                              int carry, int s)
{
    const k1_stt_t *tt = &t->tt[carry][s];
    uint32_t nb = (*state + tt->deltaNbBits) >> 16;
    BIT_addBits(w, *state, nb);                  /* addBits masks to nb bits */
    *state = t->st[carry][(*state >> nb) + (uint32_t)tt->deltaFindState];
}

/* Absorb a segment's first-encoded (last-decoded) symbol into its
 * initial state, writing no bits: the decoder still reads that symbol's
 * nbBits, but they are the bits past the start of the stream, which it
 * discards -- so this is only valid for symbols that are the last reads
 * in stream order.  Saves that symbol's coded cost per state (FSE's
 * initCState2). */
static inline uint32_t k1_init_state(const k1_tabs_t *t, int carry, int s)
{
    const k1_stt_t *tt = &t->tt[carry][s];
    uint32_t nb = (tt->deltaNbBits + (1u << 15)) >> 16;
    uint32_t v  = (nb << 16) - tt->deltaNbBits;
    return t->st[carry][(v >> nb) + (uint32_t)tt->deltaFindState];
}

static int k1_quantize(uint32_t n0, uint32_t n1)
{
    /* index = #{k : p_hat > T[k]}, with p_hat > T <=> n1*(1-T) > n0*T;
     * the predicate is monotone in k, so binary search.  Ties go low. */
    int lo = 0, hi = 15;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if ((double)n1 * (1.0 - g_k1_thr[mid]) > (double)n0 * g_k1_thr[mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

pivco_fse_status_t pivco_k1_compress(const void *src, size_t src_len,
                                      void *dst, size_t dst_cap,
                                      size_t max_len, size_t *out_len)
{
    if (src_len == 0) { *out_len = 0; return PIVCO_FSE_OK; }
    if (dst_cap < 1 + sizeof(size_t)) return PIVCO_FSE_ERR_DST_FULL;   /* BIT_initCStream's minimum */
    if (pivco_k1_prebuild() != 0) return PIVCO_FSE_ERR_INTERNAL;
    const uint8_t *s = (const uint8_t *)src;

    /* Recipe: the (previous bit, bit) pair counts, 64 bits at a time.
     * With sh = the stream shifted up one bit (previous word's top bit
     * carried in, 0 before the first bit), popcounts of w & sh and
     * ~w & ~sh are the (1,1) and (0,0) pairs, popcount(sh) the pairs
     * whose previous bit is 1; the other two follow.
     *
     * TODO: one popcount per word would do.  Pairs whose bit is 1 number
     * ones = n_right, pairs whose previous bit is 1 number ones minus the
     * last bit, so with n_right passed in (flat regions have none: keep
     * this form as the fallback) only w & sh needs counting and the rest
     * is arithmetic.  Worth it once the pricing gate makes this pass the
     * bulk of a declined attempt. */
    uint32_t cnt[4];   /* [prev*2 + bit] */
    uint64_t n00 = 0, nFF = 0;   /* 0x00 and 0xFF bytes, for the estimate */
    {
        uint64_t c11 = 0, c00 = 0, p1 = 0, carry = 0;
        const uint64_t lo7 = 0x7F7F7F7F7F7F7F7Full;
        size_t j = 0;
        for (; j + 8 <= src_len; j += 8) {
            uint64_t w;
            memcpy(&w, s + j, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            w = __builtin_bswap64(w);
#endif
            uint64_t sh = (w << 1) | carry;
            c11 += (uint64_t)__builtin_popcountll(w & sh);
            c00 += (uint64_t)__builtin_popcountll(~w & ~sh);
            p1  += (uint64_t)__builtin_popcountll(sh);
            /* zero-byte test: top bit of each byte set iff the byte is 0 */
            n00 += (uint64_t)__builtin_popcountll(~(((w & lo7) + lo7) | w | lo7));
            nFF += (uint64_t)__builtin_popcountll(~(((~w & lo7) + lo7) | ~w | lo7));
            carry = w >> 63;
        }
        if (j < src_len) {                 /* tail: mask to the real bits */
            size_t r = src_len - j;
            uint64_t w = 0;
            memcpy(&w, s + j, r);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            w = __builtin_bswap64(w);
#endif
            uint64_t mask = (1ull << (8 * r)) - 1;
            uint64_t sh = ((w << 1) | carry) & mask;
            c11 += (uint64_t)__builtin_popcountll(w & sh);
            c00 += (uint64_t)__builtin_popcountll(~w & ~sh & mask);
            p1  += (uint64_t)__builtin_popcountll(sh);
            n00 += (uint64_t)__builtin_popcountll(~(((w & lo7) + lo7) | w | lo7)) - (8 - r);
            nFF += (uint64_t)__builtin_popcountll(~(((~w & lo7) + lo7) | ~w | lo7) & mask);
        }
        cnt[3] = (uint32_t)c11;
        cnt[2] = (uint32_t)(p1 - c11);
        cnt[0] = (uint32_t)c00;
        cnt[1] = (uint32_t)(8 * src_len - p1 - c00);
    }
    int recipe = k1_quantize(cnt[0], cnt[1]) | (k1_quantize(cnt[2], cnt[3]) << 4);
    const k1_tabs_t *t = g_k1_cat[recipe];

    /* Price before coding: the ideal bit cost of every bit under the
     * recipe, minus the bits of the absorbed symbols (the last byte of
     * each segment, coded by its initial state instead), plus L per
     * state and the recipe byte.  The table's min-freq-1 floor only adds
     * to this, so the estimate is a lower bound up to rounding. */
    if (PIVCO_K1_EST_SKIP > 0.0 && max_len) {
        const double *c1 = g_k1_cost1, *c0 = g_k1_cost0;
        int i0 = recipe & 15, i1 = recipe >> 4;
        double bits = cnt[1] * c1[i0] + cnt[0] * c0[i0] + cnt[3] * c1[i1] + cnt[2] * c0[i1];
        int nst = src_len >= PIVCO_K1_SPLIT_MIN ? K1_SEGS : 1;
        size_t q = (src_len + nst - 1) / nst;
        for (int sg = 0; sg < nst; sg++) {
            size_t j = (size_t)sg * q + q - 1;
            if (j >= src_len) continue;
            int prev = q > 1 ? s[j - 1] >> 7 : 0, b = s[j];
            for (int k = 0; k < 8; k++) {
                int bit = (b >> k) & 1, ci = prev ? i1 : i0;
                bits -= bit ? c1[ci] : c0[ci];
                prev = bit;
            }
        }
        bits += (double)n00 * t->delta00 + (double)nFF * t->deltaFF;
        double est = 1.0 + (bits + nst * K1_L) / 8.0;
        if (est * PIVCO_K1_EST_SKIP > (double)max_len) return PIVCO_FSE_FALLBACK;
    }

    uint8_t *d = (uint8_t *)dst;
    d[0] = (uint8_t)recipe;
    BIT_CStream_t w;
    if (FSE_isError(BIT_initCStream(&w, d + 1, dst_cap - 1))) return PIVCO_FSE_ERR_DST_FULL;

    if (src_len >= PIVCO_K1_SPLIT_MIN) {
        size_t q = (src_len + K1_SEGS - 1) / K1_SEGS;
        uint32_t st[K1_SEGS];
        /* The last round (i = q-1) is the decoder's final read in every
         * segment that has a symbol there: absorb those into the initial
         * states.  A short last segment ends earlier and is coded normally. */
        for (int sg = 0; sg < K1_SEGS; sg++) {
            size_t j = (size_t)sg * q + q - 1;
            st[sg] = j < src_len ? k1_init_state(t, q > 1 ? s[j - 1] >> 7 : 0, s[j]) : K1_TSIZE;
        }
        for (size_t i = q - 1; i-- > 0; ) {
            for (int sg = K1_SEGS - 1; sg >= 0; sg--) {
                size_t j = (size_t)sg * q + i;
                if (j >= src_len) continue;
                k1_enc_sym(&w, t, &st[sg], i ? s[j - 1] >> 7 : 0, s[j]);
                if (K1_SEGS == 8 && sg == 4) BIT_flushBits(&w);
            }
            BIT_flushBits(&w);
        }
        for (int sg = K1_SEGS - 1; sg >= 0; sg--) {
            BIT_addBits(&w, st[sg] - K1_TSIZE, K1_L);
            if ((sg & 3) == 0) BIT_flushBits(&w);
        }
    } else {
        size_t j = src_len - 1;
        uint32_t state = k1_init_state(t, j ? s[j - 1] >> 7 : 0, s[j]);
        for (; j >= 4; j -= 4) {
            k1_enc_sym(&w, t, &state, s[j - 2] >> 7, s[j - 1]);
            k1_enc_sym(&w, t, &state, s[j - 3] >> 7, s[j - 2]);
            k1_enc_sym(&w, t, &state, s[j - 4] >> 7, s[j - 3]);
            k1_enc_sym(&w, t, &state, j > 4 ? s[j - 5] >> 7 : 0, s[j - 4]);
            BIT_flushBits(&w);
        }
        while (j-- > 0) {
            k1_enc_sym(&w, t, &state, j ? s[j - 1] >> 7 : 0, s[j]);
            BIT_flushBits(&w);
        }
        BIT_addBits(&w, state - K1_TSIZE, K1_L);
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
 * bit, next state = base + the entry's bits from the stream. */
#define K1_STEP(cr, stv, out)                                           \
    do {                                                                \
        uint32_t e_ = t->dec[cr][stv];                                  \
        (out) = (uint8_t)e_;                                            \
        (cr) = (int)(e_ & 0x80) >> 7;                                   \
        (stv) = (e_ >> 12) + (uint32_t)BIT_readBits(&bd, (e_ >> 8) & 0xF); \
    } while (0)

pivco_fse_status_t pivco_k1_decompress(const void *src, size_t src_len,
                                        void *dst, size_t dst_cap,
                                        size_t dst_expected,
                                        size_t *out_len)
{
    if (dst_cap < dst_expected) return PIVCO_FSE_ERR_DST_FULL;
    if (dst_expected == 0) { *out_len = 0; return PIVCO_FSE_OK; }
    if (src_len < 2) return PIVCO_FSE_ERR_BAD_INPUT;
    const uint8_t *p = (const uint8_t *)src;
    if (pivco_k1_prebuild() != 0) return PIVCO_FSE_ERR_INTERNAL;
    const k1_tabs_t *t = g_k1_cat[p[0]];
    uint8_t *out = (uint8_t *)dst;
    size_t n = dst_expected;
    /* Consecutive regions almost always use different recipes, so the
     * tables are cold: fetch every line up front so the misses overlap
     * instead of serializing along the decode chain.  128 lines at
     * L=10.  Whole-file decode, prefetch on vs off: +1-6.5% Granite
     * Rapids, +1-4% Zen 5, a wash on Graviton 4, M4 +1-10% on code
     * streams and -1.5% on literals (large regions amortize the cold
     * table and pay the 128 instructions). */
#if PIVCO_K1_PREFETCH
    {
        const char *tb = (const char *)t->dec;
        for (size_t l = 0; l < sizeof(t->dec) / 64; l++) __builtin_prefetch(tb + 64 * l, 0, 3);
    }
#endif

    BIT_DStream_t bd;
    if (FSE_isError(BIT_initDStream(&bd, p + 1, src_len - 1)))
        return PIVCO_FSE_ERR_BAD_INPUT;

    if (n >= PIVCO_K1_SPLIT_MIN) {
        size_t q = (n + K1_SEGS - 1) / K1_SEGS;
        size_t full = n - (K1_SEGS - 1) * q;   /* rounds where every segment has a symbol */
        uint32_t st[K1_SEGS];
        int cr[K1_SEGS];
        uint8_t *o[K1_SEGS];
        for (int sg = 0; sg < K1_SEGS; sg++) {
            if (K1_SEGS == 8 && sg == 4) BIT_reloadDStream(&bd);
            st[sg] = (uint32_t)BIT_readBits(&bd, K1_L);
            cr[sg] = 0;
            o[sg] = out + (size_t)sg * q;
        }
        size_t i = 0;
        for (; i < full; i++) {
            if (BIT_reloadDStream(&bd) != BIT_DStream_unfinished) break;
            K1_STEP(cr[0], st[0], o[0][i]);
            K1_STEP(cr[1], st[1], o[1][i]);
            K1_STEP(cr[2], st[2], o[2][i]);
            K1_STEP(cr[3], st[3], o[3][i]);
#if K1_SEGS == 8
            BIT_reloadDStream(&bd);
            K1_STEP(cr[4], st[4], o[4][i]);
            K1_STEP(cr[5], st[5], o[5][i]);
            K1_STEP(cr[6], st[6], o[6][i]);
            K1_STEP(cr[7], st[7], o[7][i]);
#endif
        }
        /* tail: the stream's last bytes and/or the ragged final rounds;
         * the absorbed last symbols read past the start, which is fine */
        for (; i < q; i++) {
            if (BIT_reloadDStream(&bd) == BIT_DStream_overflow && i + 1 < q) return PIVCO_FSE_ERR_BAD_INPUT;
            for (int sg = 0; sg < K1_SEGS; sg++) {
                size_t j = (size_t)sg * q + i;
                if (j >= n) break;
                if (K1_SEGS == 8 && sg == 4) BIT_reloadDStream(&bd);
                K1_STEP(cr[sg], st[sg], out[j]);
            }
        }
    } else {
        int cr = 0;
        uint32_t st = (uint32_t)BIT_readBits(&bd, K1_L);
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            if (BIT_reloadDStream(&bd) != BIT_DStream_unfinished) break;
            K1_STEP(cr, st, out[j]);
            K1_STEP(cr, st, out[j + 1]);
            K1_STEP(cr, st, out[j + 2]);
            K1_STEP(cr, st, out[j + 3]);
        }
        for (; j < n; j++) {
            if (BIT_reloadDStream(&bd) == BIT_DStream_overflow && j + 1 < n) return PIVCO_FSE_ERR_BAD_INPUT;
            K1_STEP(cr, st, out[j]);
        }
    }
    /* an intact stream is consumed to its start; the absorbed symbols'
     * final reads may run past it */
    BIT_reloadDStream(&bd);
    if (bd.ptr != bd.start || bd.bitsConsumed < 64) return PIVCO_FSE_ERR_BAD_INPUT;
    *out_len = n;
    return PIVCO_FSE_OK;
}
