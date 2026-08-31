/* o6ctans.c -- ORDER-1 (context-switched) tANS: per-context 512-entry tables,
 * decode step  e = dtab[(ctx << TLOG) | x]  with ctx supplied by a pre-decoded
 * route stream (the sequence-code situation: ml routed by llc).  Classic
 * tables only -- K x 2KB, microsecond builds, no window-table economics.
 * Running with route = all-zeros gives the order-0 baseline on identical
 * machinery.  Roundtrip verified.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define TLOG 9
#define L (1 << TLOG)
#define AMAX 64
#define KMAX 64
#define SEGS 16128

static inline uint64_t ld64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline int flog2(uint32_t x) { return 31 - __builtin_clz(x); }

static uint32_t dpk[KMAX << TLOG];           /* base<<16 | nb<<8 | sym */
/* encoder: [ctx][sym][x] flattened */
static uint16_t *enc_state, *enc_bits; static uint8_t *enc_nb;
#define EIX(c, s, x) ((((size_t)(c) * AMAX + (s)) << TLOG) + (x))

/* Elias-gamma cost of one context's normalized counts (header estimate) */
static uint32_t hdr_bits(const uint64_t *freq, int A) {
    uint64_t tot = 0; int maxs = 0;
    for (int s = 0; s < A; s++) { tot += freq[s]; if (freq[s] > freq[maxs]) maxs = s; }
    if (!tot) return 0;
    uint32_t bits = 0;
    for (int s = 0; s < A; s++) {
        uint32_t nf = freq[s] ? (uint32_t)(freq[s] * L / tot) : 0;
        if (freq[s] && !nf) nf = 1;
        uint32_t v = nf + 1;                    /* gamma of count+1 */
        bits += 2 * (31 - __builtin_clz(v)) + 1;
    }
    return bits;
}

static int build_ctx_dec(const uint64_t *freq, int A, int ctx) {
    /* decoder-side only: spread + decode entries, no encoder inversion */
    uint64_t tot = 0; int maxs = -1;
    for (int s = 0; s < A; s++) { tot += freq[s]; if (maxs < 0 || freq[s] > freq[maxs]) maxs = s; }
    if (!tot) return 0;
    uint16_t nf[AMAX]; int64_t sum = 0;
    for (int s = 0; s < A; s++) {
        if (!freq[s]) { nf[s] = 0; continue; }
        uint32_t x = (uint32_t)(freq[s] * L / tot); if (!x) x = 1;
        nf[s] = x; sum += x;
    }
    nf[maxs] += (int)(L - sum);
    uint8_t spread[L]; uint32_t step = (L >> 1) + (L >> 3) + 3, pos = 0;
    for (int s = 0; s < A; s++)
        for (int i = 0; i < nf[s]; i++) { spread[pos] = s; pos = (pos + step) & (L - 1); }
    uint16_t nxt[AMAX];
    for (int s = 0; s < A; s++) nxt[s] = nf[s];
    for (int x = 0; x < L; x++) {
        int s = spread[x]; uint32_t X = nxt[s]++;
        int nb = TLOG - flog2(X);
        dpk[(ctx << TLOG) | x] = (((uint32_t)((X << nb) - L)) << 16) | ((uint32_t)nb << 8) | s;
    }
    return 1;
}

static int build_ctx(const uint64_t *freq, int A, int ctx) {
    uint64_t tot = 0; int maxs = -1;
    for (int s = 0; s < A; s++) { tot += freq[s]; if (maxs < 0 || freq[s] > freq[maxs]) maxs = s; }
    if (!tot) return 0;
    uint16_t nf[AMAX]; int64_t sum = 0;
    for (int s = 0; s < A; s++) {
        if (!freq[s]) { nf[s] = 0; continue; }
        uint32_t x = (uint32_t)(freq[s] * L / tot); if (!x) x = 1;
        nf[s] = x; sum += x;
    }
    nf[maxs] += (int)(L - sum);
    uint8_t spread[L]; uint32_t step = (L >> 1) + (L >> 3) + 3, pos = 0;
    for (int s = 0; s < A; s++)
        for (int i = 0; i < nf[s]; i++) { spread[pos] = s; pos = (pos + step) & (L - 1); }
    uint16_t nxt[AMAX];
    for (int s = 0; s < A; s++) nxt[s] = nf[s];
    for (int x = 0; x < L; x++) {
        int s = spread[x]; uint32_t X = nxt[s]++;
        int nb = TLOG - flog2(X);
        uint16_t base = (uint16_t)((X << nb) - L);
        dpk[(ctx << TLOG) | x] = ((uint32_t)base << 16) | ((uint32_t)nb << 8) | s;
        for (uint32_t b = 0; b < (1u << nb); b++) {
            enc_state[EIX(ctx, s, base + b)] = x;
            enc_nb[EIX(ctx, s, base + b)] = nb;
            enc_bits[EIX(ctx, s, base + b)] = b;
        }
    }
    return 1;
}

typedef struct {
    int nseg; size_t n;
    uint8_t *arena;
    uint64_t *bitbase, *nbits; uint16_t *state0; size_t *soff;
    double bits_per_sym;
} CENC;

static CENC *cencode(const uint8_t *v, const uint8_t *rt, size_t n) {
    CENC *E = calloc(1, sizeof *E);
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
        for (size_t i = hi; i-- > lo;) {
            size_t e = EIX(rt[i], v[i], x);
            acc |= (uint64_t)enc_bits[e] << nacc; nacc += enc_nb[e];
            x = enc_state[e];
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
static void cenc_free(CENC *E) { free(E->arena); free(E->bitbase); free(E->nbits);
                                 free(E->state0); free(E->soff); free(E); }

#define FOR8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define FOR8_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A) X(4, A) X(5, A) X(6, A) X(7, A)

static const uint8_t *g_rt;
#define CDECL(k) \
    uint32_t x##k = 0; uint64_t bp##k = 0; int si##k = k, done##k = 0; \
    uint8_t *o##k = 0, *lim##k = 0; const uint8_t *rt##k = 0; \
    CLOAD(k)
#define CLOAD(k) if (si##k < E->nseg) { \
        x##k = E->state0[si##k]; bp##k = E->bitbase[si##k] + E->nbits[si##k]; \
        o##k = out + E->soff[si##k]; rt##k = g_rt + E->soff[si##k]; \
        lim##k = out + E->soff[si##k + 1] - 16; \
    } else done##k = 1;
#define CSTEPC(k) { uint32_t c_ = *rt##k++; \
    uint32_t e_ = dpk[((uint32_t)c_ << TLOG) | x##k]; \
    uint32_t nb_ = (e_ >> 8) & 15; bp##k -= nb_; \
    uint64_t w_ = ld64(E->arena + (bp##k >> 3)) >> (bp##k & 7); \
    *o##k++ = (uint8_t)e_; \
    x##k = (e_ >> 16) + ((uint32_t)w_ & ((1u << nb_) - 1)); }
#define CDRAIN(k) { uint8_t *e_ = out + E->soff[si##k + 1]; \
    while (o##k < e_) CSTEPC(k) }
#define CTICK(k, NCH) if (!done##k && o##k > lim##k) { \
        CDRAIN(k) si##k += NCH; CLOAD(k) }
#define CANY(k) any |= (uintptr_t)!done##k;
#define CSTEP(k) if (!done##k) CSTEPC(k)

static void dec_ctans_u8(const CENC *E, uint8_t *out) {
    FOR8(CDECL)
    for (;;) { FOR8_2(CTICK, 8)
        uintptr_t any = 0; FOR8(CANY) if (!any) break;
        FOR8(CSTEP) }
}

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static double run(const CENC *E, const uint8_t *v, uint8_t *out) {
    size_t n = E->n;
    memset(out, 0xAA, n); dec_ctans_u8(E, out);
    if (memcmp(out, v, n)) { printf("VERIFY FAIL\n"); exit(1); }
    int R = 1 + (int)(4e8 / n); if (R > 4000) R = 4000;
    double best = 1e30;
    for (int i = 0; i < R; i++) { double t0 = now(); dec_ctans_u8(E, out); double t1 = now();
                                  if (t1 - t0 < best) best = t1 - t0; }
    return 1e9 * best / n;
}

static double cond_shannon(const uint8_t *v, const uint8_t *rt, size_t n, int A, int K) {
    static uint64_t M[KMAX][AMAX];
    memset(M, 0, sizeof M);
    for (size_t i = 0; i < n; i++) M[rt[i]][v[i]]++;
    double h = 0;
    for (int c = 0; c < K; c++) {
        uint64_t m = 0; for (int s = 0; s < A; s++) m += M[c][s];
        if (!m) continue;
        for (int s = 0; s < A; s++) if (M[c][s])
            h -= M[c][s] * log2((double)M[c][s] / m);
    }
    return h / n;
}

int main(void) {
    enc_state = malloc((size_t)KMAX * AMAX * L * 2);
    enc_bits = malloc((size_t)KMAX * AMAX * L * 2);
    enc_nb = malloc((size_t)KMAX * AMAX * L);
    const char *files[] = {"x-ray", "mozilla", "dickens", "xml"};
    const char *pairs[][2] = {{"ml", "ll"}, {"of", "ml"}};
    for (int fi = 0; fi < 4; fi++) for (int pi = 0; pi < 2; pi++) {
        char pt[256], pr[256];
        snprintf(pt, sizeof pt, "/tmp/phd_%s/%s", files[fi], pairs[pi][0]);
        snprintf(pr, sizeof pr, "/tmp/phd_%s/%s", files[fi], pairs[pi][1]);
        FILE *f = fopen(pt, "rb"); if (!f) continue;
        fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *v = malloc(n); size_t rd = fread(v, 1, n, f); fclose(f);
        f = fopen(pr, "rb"); uint8_t *rt = malloc(n); rd += fread(rt, 1, n, f); fclose(f);
        if (rd != 2 * n) { free(v); free(rt); continue; }
        int A = 0, K = 0;
        for (size_t i = 0; i < n; i++) { if (v[i] >= A) A = v[i] + 1; if (rt[i] >= K) K = rt[i] + 1; }
        if (A > AMAX || K > KMAX) { printf("alphabet too big\n"); continue; }
        uint8_t *z = calloc(n, 1), *out = malloc(n + 64);
        /* order-0 baseline: same coder, route = zeros */
        uint64_t fq[AMAX];
        memset(fq, 0, sizeof fq);
        for (size_t i = 0; i < n; i++) fq[v[i]]++;
        memset(dpk, 0, sizeof dpk);
        build_ctx(fq, A, 0);
        g_rt = z;
        CENC *E0 = cencode(v, z, n);
        double t0 = run(E0, v, out);
        /* order-1: per-context tables */
        memset(dpk, 0, sizeof dpk);
        static uint64_t cf[KMAX][AMAX];
        memset(cf, 0, sizeof cf);
        for (size_t i = 0; i < n; i++) cf[rt[i]][v[i]]++;
        double tb0 = now();
        for (int c = 0; c < K; c++) build_ctx(cf[c], A, c);
        double tb = (now() - tb0) * 1e3;
        g_rt = rt;
        CENC *E1 = cencode(v, rt, n);
        double t1 = run(E1, v, out);
        double hc = cond_shannon(v, rt, n, A, K);
        /* decoder-only build timing */
        double td0 = now();
        for (int c = 0; c < K; c++) build_ctx_dec(cf[c], A, c);
        double td = (now() - td0) * 1e3;
        /* header bytes (Elias-gamma of counts) */
        uint32_t hb = 0;
        for (int c = 0; c < K; c++) hb += hdr_bits(cf[c], A);
        double gain_bytes = (E0->bits_per_sym - E1->bits_per_sym) * n / 8;
        /* low-mass merge: contexts with <n/256 syms -> shared fallback */
        uint8_t mapc[KMAX]; int Km = 0; uint64_t thr = n / 256 + 1;
        int misc = -1;
        { uint64_t cn[KMAX] = {0};
          for (size_t i = 0; i < n; i++) cn[rt[i]]++;
          for (int c = 0; c < K; c++)
              if (cn[c] >= thr) mapc[c] = Km++;
          for (int c = 0; c < K; c++)
              if (cn[c] < thr) { if (misc < 0) misc = Km++; mapc[c] = misc; } }
        uint8_t *rt2 = malloc(n);
        for (size_t i = 0; i < n; i++) rt2[i] = mapc[rt[i]];
        static uint64_t cf2[KMAX][AMAX]; memset(cf2, 0, sizeof cf2);
        for (size_t i = 0; i < n; i++) cf2[rt2[i]][v[i]]++;
        memset(dpk, 0, sizeof dpk);
        for (int c = 0; c < Km; c++) build_ctx(cf2[c], A, c);
        g_rt = rt2;
        CENC *E2 = cencode(v, rt2, n);
        double t2 = run(E2, v, out);
        uint32_t hb2 = 0;
        for (int c = 0; c < Km; c++) hb2 += hdr_bits(cf2[c], A);
        double gain2 = (E0->bits_per_sym - E2->bits_per_sym) * n / 8;
        printf("%s %s|%s: n=%zu K=%d  o0 %5.3f -> o1 %5.3f b/sym (condH %5.3f) @%4.2f ns\n"
               "   decbuild %5.2f ms  hdr %5uB  gain %6.0fB  NET %6.0fB\n"
               "   merged K=%2d: %5.3f b/sym @%4.2f ns  hdr %4uB  gain %6.0fB  NET %6.0fB\n",
               files[fi], pairs[pi][0], pairs[pi][1], n, K,
               E0->bits_per_sym, E1->bits_per_sym, hc, t1,
               td, hb / 8, gain_bytes, gain_bytes - hb / 8,
               Km, E2->bits_per_sym, t2, hb2 / 8, gain2, gain2 - hb2 / 8);
        (void)tb; (void)t0;
        cenc_free(E0); cenc_free(E1); cenc_free(E2);
        free(v); free(rt); free(rt2); free(z); free(out);
    }
    return 0;
}
