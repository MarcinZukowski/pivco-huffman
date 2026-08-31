/* o4tans.c -- tANS (FSE-style) roundtrip + serial-chain experiments (M4).
 *
 * Decoders:
 *  tans_u1/u8   classic 1 symbol/step: x -> dtab[x] -> read nb bits -> x'.
 *               u8 = 8 segment chains striding (the standard interleave).
 *  tansw_u1/u8  WINDOW WALK: idx = (x << W) | next-W-bits -> table emits ALL
 *               symbols determined by the window (up to 8), total bits used,
 *               final state.  One dependent lookup per ~W/H(sym) symbols --
 *               the demux walk-table pattern transplanted to entropy decode.
 *               Yield is highest exactly where tANS is needed (skewed
 *               streams, fractional bits/sym).
 *
 * Coder: TLOG=9 (512 states), zstd spread, encoder built by inverting the
 * decode table (enc[sym][x'] = (x, nb, bits)), encode in reverse, decoder
 * reads the bitstream backward (bit-pointer, branchless).  Segments of 16128
 * symbols, each with (bitlen, start-state) header; 8 chains stride segments.
 * Verify: byte-exact roundtrip on every stream.  Reports bits/sym vs Shannon
 * and real Huffman (the sub-bit gain tANS exists for).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TLOG 9
#define L (1 << TLOG)
#define W 7
#define WMASK ((1u << W) - 1)
#define MAXE 8
#define SEGS 16128
#define AMAX 256

static inline uint64_t ld64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline int flog2(uint32_t x) { return 31 - __builtin_clz(x); }

/* ---------------- table set ---------------------------------------------- */
static uint16_t nfreq[AMAX];
static uint8_t  dsym[L]; static uint8_t dnb[L]; static uint16_t dbase[L];
static uint32_t dpk[L];        /* packed entry: base<<16 | nb<<8 | sym (stock-FSE style) */
static uint16_t enc_state[AMAX][L]; static uint8_t enc_nb[AMAX][L];
static uint16_t enc_bits[AMAX][L];
typedef struct { uint64_t syms; uint16_t state; uint8_t cnt; uint8_t used; uint32_t pad; } WENT;
static WENT *wtab;      /* [L << W] */

static double g_wtab_ms;
static uint8_t g_seen[(L << W) >> 3];
static double now2(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                           return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static void build_tans(const uint64_t *freq, int A) {
    uint64_t tot = 0;
    for (int s = 0; s < A; s++) tot += freq[s];
    int64_t sum = 0; int maxs = 0;
    for (int s = 0; s < A; s++) {
        if (!freq[s]) { nfreq[s] = 0; continue; }
        uint32_t nf = (uint32_t)(freq[s] * L / tot); if (!nf) nf = 1;
        nfreq[s] = nf; sum += nf;
        if (freq[s] > freq[maxs]) maxs = s;
    }
    nfreq[maxs] += (int)(L - sum);              /* fix rounding on the mode */
    if ((int)nfreq[maxs] <= 0) { printf("norm fail\n"); exit(1); }
    /* spread (zstd) */
    uint8_t spread[L]; uint32_t step = (L >> 1) + (L >> 3) + 3, pos = 0;
    for (int s = 0; s < A; s++)
        for (int i = 0; i < nfreq[s]; i++) { spread[pos] = s; pos = (pos + step) & (L - 1); }
    /* decode table */
    uint16_t nxt[AMAX];
    for (int s = 0; s < A; s++) nxt[s] = nfreq[s];
    for (int x = 0; x < L; x++) {
        int s = spread[x]; uint32_t X = nxt[s]++;
        int nb = TLOG - flog2(X);
        dsym[x] = s; dnb[x] = nb; dbase[x] = (uint16_t)((X << nb) - L);
        dpk[x] = ((uint32_t)dbase[x] << 16) | ((uint32_t)nb << 8) | s;
    }
    /* encoder = inverse: symbol s's transitions partition [0,L) */
    for (int x = 0; x < L; x++) {
        int s = dsym[x], nb = dnb[x]; uint32_t base = dbase[x];
        for (uint32_t b = 0; b < (1u << nb); b++) {
            enc_state[s][base + b] = x; enc_nb[s][base + b] = nb; enc_bits[s][base + b] = b;
        }
    }
    /* window walk table */
    double t0_ = now2();
    for (uint32_t x = 0; x < L; x++)
        for (uint32_t wv = 0; wv <= WMASK; wv++) {
            uint32_t st = x, rem = W, cnt = 0; uint64_t syms = 0;
            while (cnt < MAXE) {
                uint32_t nb = dnb[st];
                if (nb > rem) break;
                syms |= (uint64_t)dsym[st] << (8 * cnt);
                rem -= nb;
                st = dbase[st] + ((wv >> rem) & ((1u << nb) - 1));
                cnt++;
            }
            WENT *e = &wtab[(x << W) | wv];
            e->syms = syms; e->state = st; e->cnt = cnt; e->used = W - rem;
        }
    g_wtab_ms = (now2() - t0_) * 1e3;
}

/* ---------------- container ---------------------------------------------- */
typedef struct {
    int nseg; size_t n;
    uint8_t *arena;
    uint64_t *bitbase;   /* absolute bit addr of segment stream start */
    uint64_t *nbits;
    uint16_t *state0;
    size_t *soff;        /* symbol offsets */
    double bits_per_sym;
} TENC;

static TENC *tencode(const uint8_t *v, size_t n) {
    TENC *E = calloc(1, sizeof *E);
    int nseg = (int)((n + SEGS - 1) / SEGS); E->nseg = nseg; E->n = n;
    size_t slot = 16 + (size_t)SEGS * 2;
    E->arena = calloc((size_t)nseg * slot + 32, 1);
    E->bitbase = malloc(nseg * 8); E->nbits = malloc(nseg * 8);
    E->state0 = malloc(nseg * 2); E->soff = malloc((nseg + 1) * sizeof(size_t));
    uint64_t totbits = 0;
    for (int s = 0; s < nseg; s++) {
        size_t lo = (size_t)s * SEGS, hi = lo + SEGS < n ? lo + SEGS : n;
        E->soff[s] = lo;
        uint64_t bb = ((uint64_t)s * slot + 16) * 8; E->bitbase[s] = bb;
        uint64_t acc = 0; int nacc = 0; uint64_t bp = bb; uint32_t x = 0;
        for (size_t i = hi; i-- > lo;) {          /* reverse */
            int sym = v[i];
            acc |= (uint64_t)enc_bits[sym][x] << nacc; nacc += enc_nb[sym][x];
            x = enc_state[sym][x];
            while (nacc >= 8) { E->arena[bp >> 3] = (uint8_t)acc; acc >>= 8; nacc -= 8; bp += 8; }
        }
        if (nacc) { E->arena[bp >> 3] = (uint8_t)acc; }
        E->nbits[s] = (bp - bb) + nacc; E->state0[s] = (uint16_t)x;
        totbits += E->nbits[s];
    }
    E->soff[nseg] = n;
    E->bits_per_sym = (double)totbits / n;
    return E;
}
static void tenc_free(TENC *E) { free(E->arena); free(E->bitbase); free(E->nbits);
                                 free(E->state0); free(E->soff); free(E); }

/* ---------------- decoders ------------------------------------------------ */
#define FOR1(X) X(0)
#define FOR8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define FOR1_2(X, A) X(0, A)
#define FOR8_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A) X(4, A) X(5, A) X(6, A) X(7, A)

#define TDECL(k) \
    uint32_t x##k = 0; uint64_t bp##k = 0; int si##k = k, done##k = 0; \
    uint8_t *o##k = 0, *lim##k = 0; \
    TLOAD(k)
#define TLOAD(k) if (si##k < E->nseg) { \
        x##k = E->state0[si##k]; bp##k = E->bitbase[si##k] + E->nbits[si##k]; \
        o##k = out + E->soff[si##k]; lim##k = out + E->soff[si##k + 1] - 16; \
    } else done##k = 1;
#define TSTEPC(k) { uint32_t nb_ = dnb[x##k]; bp##k -= nb_; \
    uint64_t w_ = ld64(E->arena + (bp##k >> 3)) >> (bp##k & 7); \
    *o##k++ = dsym[x##k]; \
    x##k = dbase[x##k] + ((uint32_t)w_ & ((1u << nb_) - 1)); }
#define TDRAIN(k) { uint8_t *e_ = out + E->soff[si##k + 1]; \
    while (o##k < e_) TSTEPC(k) }
#define TTICK(k, NCH) if (!done##k && o##k > lim##k) { \
        TDRAIN(k) si##k += NCH; TLOAD(k) }
#define TANY(k) any |= (uintptr_t)!done##k;

#define TSTEP(k) if (!done##k) TSTEPC(k)
#define TSTEPP(k) if (!done##k) { uint32_t e_ = dpk[x##k]; uint32_t nb_ = (e_ >> 8) & 15; \
    bp##k -= nb_; \
    uint64_t w_ = ld64(E->arena + (bp##k >> 3)) >> (bp##k & 7); \
    *o##k++ = (uint8_t)e_; \
    x##k = (e_ >> 16) + ((uint32_t)w_ & ((1u << nb_) - 1)); }
#ifdef PROF
#define SEEN(ix) g_seen[(ix) >> 3] |= 1u << ((ix) & 7);
#else
#define SEEN(ix)
#endif
#define TSTEPW(k) if (!done##k) { \
    uint64_t bw_ = bp##k - W; \
    uint32_t wv_ = (uint32_t)(ld64(E->arena + (bw_ >> 3)) >> (bw_ & 7)) & WMASK; \
    uint32_t ix_ = (x##k << W) | wv_; SEEN(ix_) \
    WENT e_ = wtab[ix_]; \
    memcpy(o##k, &e_.syms, 8); \
    o##k += e_.cnt; bp##k -= e_.used; x##k = e_.state; \
    if (!e_.cnt) TSTEPC(k) }

#define GEN_T(NAME, FORN, NCH, STEPM) \
static void NAME(const TENC *E, uint8_t *out) { \
    FORN(TDECL) \
    for (;;) { FORN##_2(TTICK, NCH) \
        uintptr_t any = 0; FORN(TANY) if (!any) break; \
        FORN(STEPM) } \
}
GEN_T(dec_tans_u1,  FOR1, 1, TSTEP)
GEN_T(dec_tans_u8,  FOR8, 8, TSTEP)
GEN_T(dec_tansp_u8, FOR8, 8, TSTEPP)
GEN_T(dec_tansw_u1, FOR1, 1, TSTEPW)
GEN_T(dec_tansw_u8, FOR8, 8, TSTEPW)

/* ---------------- harness ------------------------------------------------- */
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ---- idea 1: same bits regrouped into per-width packed arrays ----------- */
typedef struct {
    uint8_t *blob[16];        /* packed k-bit values, decode order, seg-major */
    uint64_t blobbits[16];
    uint32_t *segcnt;         /* [seg*16+k] value counts */
    uint32_t *segbase;        /* [seg*16+k] prefix start into unpacked array */
    uint32_t nk[16];
    uint16_t *vals[16];       /* decoder scratch (unpack target) */
} PK;

/* ---- idea 2 stats: conditional entropy of w --------------------------- */
static uint32_t g_hw[16][1 << TLOG];      /* w | width */
static uint32_t (*g_hx)[1 << TLOG];       /* w | state (L x L) */
static uint32_t (*g_hs)[1 << TLOG];       /* w | (symbol, width): key = s*16+nb */

static double entb(const uint32_t *h, int m) {
    uint64_t n = 0; double e = 0;
    for (int i = 0; i < m; i++) n += h[i];
    if (!n) return 0;
    for (int i = 0; i < m; i++) if (h[i])
        e -= h[i] * __builtin_log2((double)h[i] / n);
    return e;                                  /* total bits */
}

static PK *collect(const TENC *E) {
    PK *P = calloc(1, sizeof *P);
    P->segcnt = calloc((size_t)E->nseg * 16, 4);
    P->segbase = calloc((size_t)E->nseg * 16, 4);
    memset(g_hw, 0, sizeof g_hw);
    if (!g_hx) g_hx = calloc(L, sizeof *g_hx);
    else memset(g_hx, 0, (size_t)L * sizeof *g_hx);
    if (!g_hs) g_hs = calloc(256 * 16, sizeof *g_hs);
    else memset(g_hs, 0, (size_t)256 * 16 * sizeof *g_hs);
    /* pass 1: count per width per segment + stats */
    for (int s = 0; s < E->nseg; s++) {
        uint32_t x = E->state0[s]; uint64_t bp = E->bitbase[s] + E->nbits[s];
        for (size_t j = E->soff[s]; j < E->soff[s + 1]; j++) {
            uint32_t nb = dnb[x]; bp -= nb;
            uint64_t w = (ld64(E->arena + (bp >> 3)) >> (bp & 7)) & ((1u << nb) - 1);
            P->segcnt[s * 16 + nb]++;
            g_hw[nb][w]++; g_hx[x][w]++; g_hs[dsym[x] * 16 + nb][w]++;
            x = dbase[x] + (uint32_t)w;
        }
    }
    for (int k = 0; k < 16; k++) { uint32_t acc = 0;
        for (int s = 0; s < E->nseg; s++) { P->segbase[s * 16 + k] = acc; acc += P->segcnt[s * 16 + k]; }
        P->nk[k] = acc;
        P->blob[k] = calloc((size_t)acc * k / 8 + 16, 1);
        P->vals[k] = malloc((size_t)acc * 2 + 16);
    }
    /* pass 2: pack values (decode order, segment-major) */
    uint64_t wp[16] = {0};
    for (int s = 0; s < E->nseg; s++) {
        uint32_t x = E->state0[s]; uint64_t bp = E->bitbase[s] + E->nbits[s];
        for (size_t j = E->soff[s]; j < E->soff[s + 1]; j++) {
            uint32_t nb = dnb[x]; bp -= nb;
            uint64_t w = (ld64(E->arena + (bp >> 3)) >> (bp & 7)) & ((1u << nb) - 1);
            if (nb) { uint64_t b = wp[nb];
                uint64_t cur = ld64(P->blob[nb] + (b >> 3));
                cur |= w << (b & 7);
                memcpy(P->blob[nb] + (b >> 3), &cur, 8);
                wp[nb] = b + nb; }
            x = dbase[x] + (uint32_t)w;
        }
    }
    for (int k = 0; k < 16; k++) P->blobbits[k] = wp[k];
    return P;
}
static void pk_free(PK *P) {
    for (int k = 0; k < 16; k++) { free(P->blob[k]); free(P->vals[k]); }
    free(P->segcnt); free(P->segbase); free(P);
}

/* decoder: bulk-unpack all width arrays, then walk with array reads */
#define PDECL(k)     uint32_t x##k = 0; int si##k = k, done##k = 0;     uint8_t *o##k = 0, *lim##k = 0; const uint16_t *wp##k[16];     PLOAD(k)
#define PLOAD(k) if (si##k < E->nseg) {         x##k = E->state0[si##k];         for (int q_ = 0; q_ < 16; q_++)             wp##k[q_] = P->vals[q_] + P->segbase[si##k * 16 + q_];         o##k = out + E->soff[si##k]; lim##k = out + E->soff[si##k + 1] - 16;     } else done##k = 1;
#define PSTEPC(k) { uint32_t e_ = dpk[x##k]; uint32_t nb_ = (e_ >> 8) & 15;     uint16_t w_ = *wp##k[nb_]++;     *o##k++ = (uint8_t)e_;     x##k = (e_ >> 16) + w_; }
#define PDRAIN(k) { uint8_t *e_ = out + E->soff[si##k + 1];     while (o##k < e_) PSTEPC(k) }
#define PTICK(k, NCH) if (!done##k && o##k > lim##k) {         PDRAIN(k) si##k += NCH; PLOAD(k) }
#define PSTEP(k) if (!done##k) PSTEPC(k)

static const PK *g_pk;
static void dec_tanspk_u8(const TENC *E, uint8_t *out) {
    const PK *P = g_pk;
    /* unpack pass: charged to decode */
    for (int q = 1; q < 16; q++) {
        const uint8_t *b = P->blob[q]; uint16_t *d = P->vals[q];
        uint64_t bp = 0; uint32_t m = (1u << q) - 1;
        for (uint32_t i = 0; i < P->nk[q]; i++) {
            d[i] = (uint16_t)((ld64(b + (bp >> 3)) >> (bp & 7)) & m); bp += q; }
    }
    memset(P->vals[0], 0, (size_t)P->nk[0] * 2);
    FOR8(PDECL)
    for (;;) { FOR8_2(PTICK, 8)
        uintptr_t any = 0; FOR8(TANY) if (!any) break;
        FOR8(PSTEP) }
}
static double hufcost(const uint64_t *f0, int A) {   /* bits/sym, two-queue */
    uint64_t q1[520], q2[520]; int n1 = 0, n2 = 0, h2 = 0, h1 = 0;
    for (int s = 0; s < A; s++) if (f0[s]) q1[n1++] = f0[s];
    if (n1 <= 1) return n1;
    for (int i = 0; i < n1; i++) for (int j = i + 1; j < n1; j++)
        if (q1[j] < q1[i]) { uint64_t t = q1[i]; q1[i] = q1[j]; q1[j] = t; }
    uint64_t cost = 0, tot = 0;
    for (int i = 0; i < n1; i++) tot += q1[i];
    while ((n1 - h1) + (n2 - h2) > 1) {
        uint64_t a, b;
        a = (h1 < n1 && (h2 >= n2 || q1[h1] <= q2[h2])) ? q1[h1++] : q2[h2++];
        b = (h1 < n1 && (h2 >= n2 || q1[h1] <= q2[h2])) ? q1[h1++] : q2[h2++];
        cost += a + b; q2[n2++] = a + b;
    }
    return (double)cost / tot;
}
static double shannon(const uint64_t *f, int A) {
    uint64_t tot = 0; double h = 0;
    for (int s = 0; s < A; s++) tot += f[s];
    for (int s = 0; s < A; s++) if (f[s]) {
        double p = (double)f[s] / tot; h -= p * __builtin_log2(p); }
    return h;
}

typedef void (*tfn)(const TENC *, uint8_t *);
static void run(const char *name, tfn fn, const TENC *E, const uint8_t *v, uint8_t *out) {
    size_t n = E->n;
    memset(out, 0xAA, n); fn(E, out);
    if (memcmp(out, v, n)) { printf("    %-9s VERIFY FAIL\n", name); return; }
    int R = 1 + (int)(4e8 / n); if (R > 4000) R = 4000;
    double best = 1e30;
    for (int i = 0; i < R; i++) { double t0 = now(); fn(E, out); double t1 = now();
                                  if (t1 - t0 < best) best = t1 - t0; }
    printf("    %-9s %7.3f ns/sym  %6.2f Gsym/s\n", name, 1e9 * best / n, n / best / 1e9);
}

int main(int argc, char **argv) {
    wtab = malloc(sizeof(WENT) << (TLOG + W));
    const char *files[] = {"x-ray", "dickens", "mozilla", "xml"};
    const char *strms[] = {"ml", "ll", "of"};
    for (int fi = 0; fi < 4; fi++) for (int si = 0; si < 3; si++) {
        char path[256];
        snprintf(path, sizeof path, "/tmp/phd_%s/%s", files[fi], strms[si]);
        FILE *f = fopen(path, "rb"); if (!f) continue;
        fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *v = malloc(n + 64); size_t rd = fread(v, 1, n, f); fclose(f);
        if (rd != n) { free(v); continue; }
        int A = 0; for (size_t i = 0; i < n; i++) if (v[i] >= A) A = v[i] + 1;
        uint64_t freq[AMAX] = {0};
        for (size_t i = 0; i < n; i++) freq[v[i]]++;
        build_tans(freq, A);
        TENC *E = tencode(v, n);
        printf("%s-%s: n=%zu A=%d  shannon %5.3f  huf %5.3f  tans %5.3f b/sym"
               "  (tans vs huf: %+.1f%%)\n",
               files[fi], strms[si], n, A, shannon(freq, A), hufcost(freq, A),
               E->bits_per_sym, 100.0 * (E->bits_per_sym / hufcost(freq, A) - 1));
        uint8_t *out = malloc(n + 64);
        run("tans_u1", dec_tans_u1, E, v, out);
        run("tans_u8", dec_tans_u8, E, v, out);
        run("tansp_u8", dec_tansp_u8, E, v, out);
        { PK *P = collect(E); g_pk = P;
          run("tanspk_u8", dec_tanspk_u8, E, v, out);
          /* idea 2: compressibility of the w values */
          double base = 0, condw = 0, condx = 0;
          for (int k = 1; k <= TLOG; k++) { base += (double)P->nk[k] * k;
                                             if (P->nk[k]) condw += entb(g_hw[k], 1 << k); }
          for (int x = 0; x < L; x++) condx += entb(g_hx[x], 1 << TLOG);
          double conds = 0;
          for (int s = 0; s < 256; s++) for (int k = 1; k <= TLOG; k++)
              conds += entb(g_hs[s * 16 + k], 1 << k);
          /* Huffman-REALIZABLE per-group cost (1-bit floor per symbol) */
          double hufx = 0, hufs = 0, t3x = 0;
          { uint64_t tmp[1 << TLOG];
            for (int x = 0; x < L; x++) {
                int nbx = dnb[x];
                if (!nbx) continue;                 /* reads no bits: cost 0 */
                uint64_t m = 0;
                for (int i = 0; i < (1 << nbx); i++) { tmp[i] = g_hx[x][i]; m += tmp[i]; }
                if (!m) continue;
                double c = hufcost(tmp, 1 << nbx);
                hufx += (c < nbx ? c : nbx) * m;    /* per-group raw fallback */
                /* 3-way: add per-group tANS arm (sh + 1% quant + header):
                   nb=1 canonical-p table -> 4-bit level id; else gamma counts */
                double sh_g = entb(g_hx[x], 1 << nbx);
                double hdr = nbx == 1 ? 4.0 : 0;
                if (nbx > 1) { int d = 0;
                    for (int i = 0; i < (1 << nbx); i++) if (g_hx[x][i]) d++;
                    hdr = d * 6.0; }
                double tg = sh_g * 1.01 + hdr;
                double best = (double)nbx * m;
                if ((c < nbx ? c : nbx) * m < best) best = (c < nbx ? c : nbx) * m;
                if (tg < best) best = tg;
                t3x += best;
            }
            for (int s = 0; s < 256; s++) for (int k = 1; k <= TLOG; k++) {
                uint64_t m = 0;
                for (int i = 0; i < (1 << k); i++) { tmp[i] = g_hs[s * 16 + k][i]; m += tmp[i]; }
                if (!m) continue;
                double c = hufcost(tmp, 1 << k);
                hufs += (c < k ? c : k) * m;
            } }
          printf("    wbits/sym %5.3f  |width -%4.2f%%  |(sym,w) sh -%4.2f%% huf -%4.2f%%"
                 "  |state sh -%4.2f%% huf -%4.2f%% 3way -%4.2f%%\n",
                 base / n, 100 * (1 - condw / base), 100 * (1 - conds / base),
                 100 * (1 - hufs / base), 100 * (1 - condx / base),
                 100 * (1 - hufx / base), 100 * (1 - t3x / base));
          pk_free(P); }
        run("tansw_u1", dec_tansw_u1, E, v, out);
        run("tansw_u8", dec_tansw_u8, E, v, out);
#ifdef PROF
        { memset(g_seen, 0, sizeof g_seen);
          dec_tansw_u8(E, out);
          int touched = 0;
          for (size_t i = 0; i < sizeof g_seen; i++) touched += __builtin_popcount(g_seen[i]);
          printf("    wtab build %.3f ms; touched %d / %d entries (%.2f%%)\n",
                 g_wtab_ms, touched, L << W, 100.0 * touched / (L << W)); }
#endif
        tenc_free(E); free(v); free(out);
    }
    return 0;
}
