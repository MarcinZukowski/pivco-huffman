/* o5bans.c -- BINARY tANS window walk for PHA-style routing bitmaps (M4).
 *
 * Model: PHA per-node bitmaps = Bernoulli(p) bit streams, p_major >= 0.625
 * (production gate).  Binary alphabet lets the window-walk entry emit a
 * PACKED BIT-GROUP (up to 32 bits per dependent lookup) instead of 8 byte
 * symbols; skew guarantees high yield (W / H(p)).
 *
 *  bans_u8    classic 1 bit/step, 8 segment chains, acc/8-bit flush.
 *  bansw_u8   window walk: idx = (state<<W)|window -> {bits u32, cnt, used,
 *             state}; output via branchless 8B RMW-OR at a bit pointer.
 *
 * TLOG=7 (128 states = p quantized to /128), W=8 -> 32K entries x 8B = 256KB.
 * In production: ~16 canonical p levels, one prebuilt table each, shared by
 * all nodes (kills per-node FSE headers too).  Roundtrip-verified.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TLOG 7
#define L (1 << TLOG)
#define W 8
#define WMASK ((1u << W) - 1)
#define MAXE 32
#define SEGBITS 129024
#define AMAX 2

static inline uint64_t ld64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline void st64(void *p, uint64_t v) { memcpy(p, &v, 8); }
static inline int flog2(uint32_t x) { return 31 - __builtin_clz(x); }

static uint8_t  dsym[L]; static uint8_t dnb[L]; static uint16_t dbase[L];
static uint16_t enc_state[AMAX][L]; static uint8_t enc_nb[AMAX][L];
static uint16_t enc_bits[AMAX][L];
typedef struct { uint32_t bits; uint16_t state; uint8_t cnt; uint8_t used; } WENT;
static WENT wtab[L << W];

static void build_bans(uint32_t f1) {          /* f1 = normalized count of bit 1 */
    uint16_t nfreq[2] = {(uint16_t)(L - f1), (uint16_t)f1};
    uint8_t spread[L]; uint32_t step = (L >> 1) + (L >> 3) + 3, pos = 0;
    for (int s = 0; s < 2; s++)
        for (int i = 0; i < nfreq[s]; i++) { spread[pos] = s; pos = (pos + step) & (L - 1); }
    uint16_t nxt[2] = {nfreq[0], nfreq[1]};
    for (int x = 0; x < L; x++) {
        int s = spread[x]; uint32_t X = nxt[s]++;
        int nb = TLOG - flog2(X);
        dsym[x] = s; dnb[x] = nb; dbase[x] = (uint16_t)((X << nb) - L);
    }
    for (int x = 0; x < L; x++)
        for (uint32_t b = 0; b < (1u << dnb[x]); b++) {
            int s = dsym[x];
            enc_state[s][dbase[x] + b] = x; enc_nb[s][dbase[x] + b] = dnb[x];
            enc_bits[s][dbase[x] + b] = b;
        }
    for (uint32_t x = 0; x < L; x++)
        for (uint32_t wv = 0; wv <= WMASK; wv++) {
            uint32_t st = x, rem = W, cnt = 0, bits = 0;
            while (cnt < MAXE) {
                uint32_t nb = dnb[st];
                if (nb > rem) break;
                bits |= (uint32_t)dsym[st] << cnt;
                rem -= nb;
                st = dbase[st] + ((wv >> rem) & ((1u << nb) - 1));
                cnt++;
            }
            WENT *e = &wtab[(x << W) | wv];
            e->bits = bits; e->state = st; e->cnt = cnt; e->used = W - rem;
        }
}

/* container: input = byte-per-bit array; output = packed LSB-first bitmap */
typedef struct {
    int nseg; size_t n;                        /* n = number of BITS */
    uint8_t *arena;
    uint64_t *bitbase, *nbits; uint16_t *state0; size_t *soff;
    double bits_per_bit;
} BENC;

static BENC *bencode(const uint8_t *v, size_t n) {
    BENC *E = calloc(1, sizeof *E);
    int nseg = (int)((n + SEGBITS - 1) / SEGBITS); E->nseg = nseg; E->n = n;
    size_t slot = 16 + SEGBITS / 4;            /* cap 2 bits/bit */
    E->arena = calloc((size_t)nseg * slot + 32, 1);
    E->bitbase = malloc(nseg * 8); E->nbits = malloc(nseg * 8);
    E->state0 = malloc(nseg * 2); E->soff = malloc((nseg + 1) * sizeof(size_t));
    uint64_t totbits = 0;
    for (int s = 0; s < nseg; s++) {
        size_t lo = (size_t)s * SEGBITS, hi = lo + SEGBITS < n ? lo + SEGBITS : n;
        E->soff[s] = lo;
        uint64_t bb = ((uint64_t)s * slot + 16) * 8; E->bitbase[s] = bb;
        uint64_t acc = 0; int nacc = 0; uint64_t bp = bb; uint32_t x = 0;
        for (size_t i = hi; i-- > lo;) {
            int sym = v[i];
            acc |= (uint64_t)enc_bits[sym][x] << nacc; nacc += enc_nb[sym][x];
            x = enc_state[sym][x];
            while (nacc >= 8) { E->arena[bp >> 3] = (uint8_t)acc; acc >>= 8; nacc -= 8; bp += 8; }
        }
        if (nacc) E->arena[bp >> 3] = (uint8_t)acc;
        E->nbits[s] = (bp - bb) + nacc; E->state0[s] = (uint16_t)x;
        totbits += E->nbits[s];
    }
    E->soff[nseg] = n;
    E->bits_per_bit = (double)totbits / n;
    return E;
}
static void benc_free(BENC *E) { free(E->arena); free(E->bitbase); free(E->nbits);
                                 free(E->state0); free(E->soff); free(E); }

#define FOR8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define FOR8_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A) X(4, A) X(5, A) X(6, A) X(7, A)

/* out = packed bitmap buffer; ob = absolute output BIT index */
#define BDECL(k) \
    uint32_t x##k = 0; uint64_t bp##k = 0, ob##k = 0, oe##k = 0; \
    int si##k = k, done##k = 0; \
    BLOAD(k)
#define BLOAD(k) if (si##k < E->nseg) { \
        x##k = E->state0[si##k]; bp##k = E->bitbase[si##k] + E->nbits[si##k]; \
        ob##k = E->soff[si##k]; oe##k = E->soff[si##k + 1]; \
    } else done##k = 1;
#define BSTEPC1(k) { uint32_t nb_ = dnb[x##k]; bp##k -= nb_; \
    uint64_t w_ = ld64(E->arena + (bp##k >> 3)) >> (bp##k & 7); \
    uint64_t cur_ = ld64(out + (ob##k >> 3)); \
    st64(out + (ob##k >> 3), cur_ | ((uint64_t)dsym[x##k] << (ob##k & 7))); \
    ob##k++; \
    x##k = dbase[x##k] + ((uint32_t)w_ & ((1u << nb_) - 1)); }
#define BDRAIN(k) while (ob##k < oe##k) BSTEPC1(k)
#define BTICK(k, NCH) if (!done##k && ob##k + 40 > oe##k) { \
        BDRAIN(k) si##k += NCH; BLOAD(k) }
#define BANY(k) any |= (uintptr_t)!done##k;

#define BSTEP(k) if (!done##k) BSTEPC1(k)
#define BSTEPW(k) if (!done##k) { \
    uint64_t bw_ = bp##k - W; \
    uint32_t wv_ = (uint32_t)(ld64(E->arena + (bw_ >> 3)) >> (bw_ & 7)) & WMASK; \
    WENT e_ = wtab[(x##k << W) | wv_]; \
    uint64_t cur_ = ld64(out + (ob##k >> 3)); \
    st64(out + (ob##k >> 3), cur_ | ((uint64_t)e_.bits << (ob##k & 7))); \
    ob##k += e_.cnt; bp##k -= e_.used; x##k = e_.state; \
    if (!e_.cnt) BSTEPC1(k) }

#define GEN_B(NAME, STEPM) \
static void NAME(const BENC *E, uint8_t *out) { \
    FOR8(BDECL) \
    for (;;) { FOR8_2(BTICK, 8) \
        uintptr_t any = 0; FOR8(BANY) if (!any) break; \
        FOR8(STEPM) } \
}
GEN_B(dec_bans_u8,  BSTEP)
GEN_B(dec_bansw_u8, BSTEPW)

/* ------- honest baseline: byte-alphabet tANS over the PACKED bitmap ------ */
#define T2LOG 10
#define L2 (1 << T2LOG)
static uint32_t dpk2[L2];                       /* base<<16 | nb<<8 | sym */
static uint16_t e2_state[256][L2]; static uint8_t e2_nb[256][L2];
static uint16_t e2_bits[256][L2];

static void build_bans2(const uint64_t *freq) {
    uint64_t tot = 0; int maxs = 0;
    uint16_t nf[256]; int64_t sum = 0;
    for (int s = 0; s < 256; s++) tot += freq[s];
    for (int s = 0; s < 256; s++) {
        if (!freq[s]) { nf[s] = 0; continue; }
        uint32_t x = (uint32_t)(freq[s] * L2 / tot); if (!x) x = 1;
        nf[s] = x; sum += x;
        if (freq[s] > freq[maxs]) maxs = s;
    }
    nf[maxs] += (int)(L2 - sum);
    uint8_t spread[L2]; uint32_t step = (L2 >> 1) + (L2 >> 3) + 3, pos = 0;
    for (int s = 0; s < 256; s++)
        for (int i = 0; i < nf[s]; i++) { spread[pos] = s; pos = (pos + step) & (L2 - 1); }
    uint16_t nxt[256];
    for (int s = 0; s < 256; s++) nxt[s] = nf[s];
    for (int x = 0; x < L2; x++) {
        int s = spread[x]; uint32_t X = nxt[s]++;
        int nb = T2LOG - flog2(X);
        uint16_t base = (uint16_t)((X << nb) - L2);
        dpk2[x] = ((uint32_t)base << 16) | ((uint32_t)nb << 8) | s;
        for (uint32_t b = 0; b < (1u << nb); b++) {
            e2_state[s][base + b] = x; e2_nb[s][base + b] = nb; e2_bits[s][base + b] = b;
        }
    }
}

#define SEG2 16128
typedef struct {
    int nseg; size_t n;
    uint8_t *arena;
    uint64_t *bitbase, *nbits; uint16_t *state0; size_t *soff;
    double bits_per_sym;
} B2ENC;

static B2ENC *bencode2(const uint8_t *v, size_t n) {
    B2ENC *E = calloc(1, sizeof *E);
    int nseg = (int)((n + SEG2 - 1) / SEG2); E->nseg = nseg; E->n = n;
    size_t slot = 16 + (size_t)SEG2 * 2;
    E->arena = calloc((size_t)nseg * slot + 32, 1);
    E->bitbase = malloc(nseg * 8); E->nbits = malloc(nseg * 8);
    E->state0 = malloc(nseg * 2); E->soff = malloc((nseg + 1) * sizeof(size_t));
    uint64_t totbits = 0;
    for (int s = 0; s < nseg; s++) {
        size_t lo = (size_t)s * SEG2, hi = lo + SEG2 < n ? lo + SEG2 : n;
        E->soff[s] = lo;
        uint64_t bb = ((uint64_t)s * slot + 16) * 8; E->bitbase[s] = bb;
        uint64_t acc = 0; int nacc = 0; uint64_t bp = bb; uint32_t x = 0;
        for (size_t i = hi; i-- > lo;) {
            int sym = v[i];
            acc |= (uint64_t)e2_bits[sym][x] << nacc; nacc += e2_nb[sym][x];
            x = e2_state[sym][x];
            while (nacc >= 8) { E->arena[bp >> 3] = (uint8_t)acc; acc >>= 8; nacc -= 8; bp += 8; }
        }
        if (nacc) E->arena[bp >> 3] = (uint8_t)acc;
        E->nbits[s] = (bp - bb) + nacc; E->state0[s] = (uint16_t)x;
        totbits += E->nbits[s];
    }
    E->soff[nseg] = n;
    E->bits_per_sym = (double)totbits / n;
    return E;
}
static void b2_free(B2ENC *E) { free(E->arena); free(E->bitbase); free(E->nbits);
                                free(E->state0); free(E->soff); free(E); }

#define C2DECL(k)     uint32_t y##k = 0; uint64_t cp##k = 0; int ti##k = k, fin##k = 0;     uint8_t *q##k = 0, *ql##k = 0;     C2LOAD(k)
#define C2LOAD(k) if (ti##k < E->nseg) {         y##k = E->state0[ti##k]; cp##k = E->bitbase[ti##k] + E->nbits[ti##k];         q##k = out + E->soff[ti##k]; ql##k = out + E->soff[ti##k + 1] - 16;     } else fin##k = 1;
#define C2STEPC(k) { uint32_t e_ = dpk2[y##k]; uint32_t nb_ = (e_ >> 8) & 15;     cp##k -= nb_;     uint64_t w_ = ld64(E->arena + (cp##k >> 3)) >> (cp##k & 7);     *q##k++ = (uint8_t)e_;     y##k = (e_ >> 16) + ((uint32_t)w_ & ((1u << nb_) - 1)); }
#define C2DRAIN(k) { uint8_t *e_ = out + E->soff[ti##k + 1];     while (q##k < e_) C2STEPC(k) }
#define C2TICK(k, NCH) if (!fin##k && q##k > ql##k) {         C2DRAIN(k) ti##k += NCH; C2LOAD(k) }
#define C2ANY(k) any |= (uintptr_t)!fin##k;
#define C2STEP(k) if (!fin##k) C2STEPC(k)

static void dec_bans2_u8(const B2ENC *E, uint8_t *out) {
    FOR8(C2DECL)
    for (;;) { FOR8_2(C2TICK, 8)
        uintptr_t any = 0; FOR8(C2ANY) if (!any) break;
        FOR8(C2STEP) }
}

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }

typedef void (*bfn)(const BENC *, uint8_t *);
static void run(const char *name, bfn fn, const BENC *E, const uint8_t *ref,
                uint8_t *out, size_t obytes) {
    memset(out, 0, obytes + 64); fn(E, out);
    if (memcmp(out, ref, obytes)) { printf("    %-9s VERIFY FAIL\n", name); return; }
    int R = 1 + (int)(3e9 / E->n); if (R > 3000) R = 3000;
    double best = 1e30;
    for (int i = 0; i < R; i++) {
        memset(out, 0, obytes + 8);            /* OR-accumulating output */
        double t0 = now(); fn(E, out); double t1 = now();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("    %-9s %7.4f ns/bit  %6.2f Gbit/s\n", name, 1e9 * best / E->n, E->n / best / 1e9);
}

int main(void) {
    size_t n = 1 << 23;                        /* 8M bits = 1MB packed */
    uint8_t *v = malloc(n);
    double ps[] = {0.65, 0.80, 0.90, 0.95, 0.98};
    for (int pi = 0; pi < 5; pi++) {
        double p = ps[pi];
        uint32_t thr = (uint32_t)(p * 4294967296.0);
        for (size_t i = 0; i < n; i++) v[i] = xr() < thr ? 0 : 1;   /* 0 = major */
        uint32_t f1 = 0;
        for (size_t i = 0; i < n; i++) f1 += v[i];
        uint32_t nf1 = (uint32_t)((uint64_t)f1 * L / n); if (!nf1) nf1 = 1;
        if (nf1 >= L) nf1 = L - 1;
        build_bans(nf1);
        BENC *E = bencode(v, n);
        double H = 0; double q = (double)f1 / n;
        if (q > 0 && q < 1) H = -q * __builtin_log2(q) - (1 - q) * __builtin_log2(1 - q);
        size_t obytes = n / 8;
        uint8_t *ref = calloc(obytes + 64, 1), *out = malloc(obytes + 128);
        for (size_t i = 0; i < n; i++) ref[i >> 3] |= v[i] << (i & 7);
        printf("p=%.2f  H=%5.3f  tans %5.3f bits/bit  (+%4.1f%% vs H)\n",
               p, H, E->bits_per_bit, 100 * (E->bits_per_bit / H - 1));
        run("bans_u8", dec_bans_u8, E, ref, out, obytes);
        run("bansw_u8", dec_bansw_u8, E, ref, out, obytes);
        { uint64_t bf[256] = {0};
          for (size_t i = 0; i < obytes; i++) bf[ref[i]]++;
          build_bans2(bf);
          B2ENC *E2 = bencode2(ref, obytes);
          memset(out, 0xAA, obytes); dec_bans2_u8(E2, out);
          if (memcmp(out, ref, obytes)) printf("    byte-fse  VERIFY FAIL\n");
          else { int R = 1 + (int)(4e8 / obytes); if (R > 4000) R = 4000;
                 double best = 1e30;
                 for (int i = 0; i < R; i++) { double t0 = now(); dec_bans2_u8(E2, out);
                     double t1 = now(); if (t1 - t0 < best) best = t1 - t0; }
                 printf("    byte-fse  %7.4f ns/bit  %6.2f Gbit/s   (%5.3f bits/bit, +%4.1f%% vs H)\n",
                        1e9 * best / n, n / best / 1e9,
                        E2->bits_per_sym / 8, 100 * (E2->bits_per_sym / 8 / (H > 0 ? H : 1) - 1)); }
          b2_free(E2); }
        benc_free(E); free(ref); free(out);
    }
    free(v);
    return 0;
}
