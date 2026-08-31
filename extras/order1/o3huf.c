/* o3huf.c -- traditional multi-stream Huffman vs fused order-1 hybrid (M4).
 *
 * Decoders (canonical MSB-first, 11-bit table, branchless bit-pointer):
 *  huf0    order-0, one bitstream per segment, 8 chains striding segments.
 *          Loop-carried: ba -> load window -> table -> len -> ba.
 *  huf0x2  same, 2 symbols per window load (traditional fast shape).
 *  huf1k4  FUSED order-1 hybrid (MZ): K=4 per-class bitstreams per segment;
 *          decode from stream[c] with context-c table; class of the decoded
 *          symbol = its top 2 bits (bijective remap) -> next context free.
 *          One table lookup per symbol, no demux pass, no bucket buffers.
 *  huf1k2  same, K=2 (top bit).
 *
 * Also prints achieved bits/B -> first ROUNDTRIP order-1 gain measurement
 * (11-bit length-limited, global tables, per-segment context reset).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MARGIN 8

static inline uint64_t bswap64(uint64_t x) { return __builtin_bswap64(x); }
static inline uint64_t ld64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* ---------- canonical length-limited Huffman ----------------------------- */
typedef struct { uint16_t code[256]; uint8_t len[256]; uint16_t tab[2048]; } HUF;

static void huf_build(HUF *H, const uint64_t *freq) {
    int n = 0, sym[256]; uint64_t f[512]; int parent[512], lens[512];
    for (int s = 0; s < 256; s++) if (freq[s]) { sym[n] = s; f[n] = freq[s]; n++; }
    memset(H->len, 0, 256);
    if (n == 0) return;
    if (n == 1) { H->len[sym[0]] = 1; H->code[sym[0]] = 0;
                  for (int i = 0; i < 1024; i++) H->tab[i] = sym[0] << 4 | 1; return; }
    /* plain Huffman lengths via repeated min-merge (n<=256, O(n^2) fine) */
    int m = n, alive[512];
    for (int i = 0; i < n; i++) { alive[i] = 1; parent[i] = -1; }
    int tot = n;
    while (m > 1) {
        int a = -1, b = -1;
        for (int i = 0; i < tot; i++) if (alive[i]) {
            if (a < 0 || f[i] < f[a]) { b = a; a = i; }
            else if (b < 0 || f[i] < f[b]) b = i;
        }
        alive[a] = alive[b] = 0;
        f[tot] = f[a] + f[b]; parent[a] = parent[b] = tot; parent[tot] = -1;
        alive[tot] = 1; tot++; m--;
    }
    for (int i = 0; i < n; i++) {
        int d = 0, p = i;
        while (parent[p] >= 0) { p = parent[p]; d++; }
        lens[i] = d > 11 ? 11 : d;
    }
    /* Kraft fix after clamping */
    uint32_t kraft = 0;
    for (int i = 0; i < n; i++) kraft += 1u << (11 - lens[i]);
    while (kraft > 2048) {
        int best = -1;
        for (int i = 0; i < n; i++) if (lens[i] < 11 && (best < 0 || f[i] < f[best])) best = i;
        kraft -= 1u << (10 - lens[best]); lens[best]++;
    }
    /* canonical assignment (sort by len then sym) */
    int order[256];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        int a = order[i], b = order[j];
        if (lens[b] < lens[a] || (lens[b] == lens[a] && sym[b] < sym[a])) { order[i] = b; order[j] = a; }
    }
    uint32_t code = 0; int prev = 0;
    memset(H->tab, 0, sizeof H->tab);
    for (int i = 0; i < n; i++) {
        int k = order[i], s = sym[k], l = lens[k];
        code <<= (l - prev); prev = l;
        H->code[s] = code; H->len[s] = l;
        uint32_t base = code << (11 - l), span = 1u << (11 - l);
        for (uint32_t j = 0; j < span; j++) H->tab[base + j] = (uint16_t)(s << 4 | l);
        code++;
    }
}

/* ---------- bit writer (MSB-first) --------------------------------------- */
static void putbits(uint8_t *buf, uint64_t *bitpos, uint32_t code, int len) {
    size_t byte = *bitpos >> 3; int off = *bitpos & 7;
    uint64_t cur = bswap64(ld64(buf + byte));
    cur |= (uint64_t)code << (64 - off - len);
    uint64_t w = bswap64(cur); memcpy(buf + byte, &w, 8);
    *bitpos += len;
}

/* ---------- encoded container -------------------------------------------- */
typedef struct {
    int K, nseg; size_t N;
    size_t *segoff;
    uint8_t *arena;
    uint64_t *sb;            /* [seg*K + c] = start bit-addr in arena */
    uint64_t bits;           /* total encoded bits */
    HUF *H;                  /* K tables */
} HENC;

static HENC *hencode(const uint8_t *v, size_t n, int K, size_t segb, int shift) {
    HENC *E = calloc(1, sizeof *E);
    int nseg = (int)((n + segb - 1) / segb); if (nseg < 1) nseg = 1;
    E->K = K; E->nseg = nseg; E->N = n;
    E->segoff = malloc((nseg + 1) * sizeof *E->segoff);
    for (int s = 0; s < nseg; s++) E->segoff[s] = s * segb;
    E->segoff[nseg] = n;
    E->H = calloc(K, sizeof *E->H);
    uint64_t (*freq)[256] = calloc(K, sizeof *freq);
    {   uint32_t c = 0;                     /* context = class of prev, reset/seg */
        for (int s = 0; s < nseg; s++) { c = 0;
            for (size_t j = E->segoff[s]; j < E->segoff[s + 1]; j++) {
                freq[c][v[j]]++; c = K == 1 ? 0 : v[j] >> shift; } }
    }
    for (int c = 0; c < K; c++) huf_build(&E->H[c], freq[c]);
    free(freq);
    size_t cap = n * 2 + (size_t)nseg * K * 16 + 64;
    E->arena = calloc(cap, 1);
    E->sb = malloc((size_t)nseg * K * sizeof *E->sb);
    uint64_t pos = 0;
    for (int s = 0; s < nseg; s++) {
        /* lay out this segment's K streams: first pass counts bits per class */
        uint64_t need[4] = {0, 0, 0, 0}; uint32_t c = 0;
        for (size_t j = E->segoff[s]; j < E->segoff[s + 1]; j++) {
            need[c] += E->H[c].len[v[j]]; c = K == 1 ? 0 : v[j] >> shift; }
        uint64_t wp[4];
        for (int k = 0; k < K; k++) {
            pos = (pos + 7) & ~7ull;                 /* byte-align stream start */
            E->sb[s * K + k] = pos; wp[k] = pos; pos += need[k];
        }
        c = 0;
        for (size_t j = E->segoff[s]; j < E->segoff[s + 1]; j++) {
            putbits(E->arena, &wp[c], E->H[c].code[v[j]], E->H[c].len[v[j]]);
            c = K == 1 ? 0 : v[j] >> shift;
        }
    }
    E->bits = pos;
    return E;
}
static void henc_free(HENC *E) { free(E->segoff); free(E->arena); free(E->sb); free(E->H); free(E); }

/* ---------- decoders ------------------------------------------------------ */
#define FOR8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define FOR8_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A) X(4, A) X(5, A) X(6, A) X(7, A)

/* order-0: one stream per segment, register bit-addr */
#define H0DECL(k) \
    uint64_t ba##k = 0; uint8_t *o##k = 0, *lim##k = 0; \
    int si##k = k, done##k = 0; H0LOAD(k)
#define H0LOAD(k) if (si##k < E->nseg) { ba##k = E->sb[si##k]; \
        o##k = out + E->segoff[si##k]; lim##k = out + E->segoff[si##k + 1] - MARGIN; \
    } else done##k = 1;
#define H0TICK(k, NCH) if (!done##k && o##k > lim##k) { \
        H0DRAIN(k) si##k += NCH; H0LOAD(k) }
#define H0DRAIN(k) { uint8_t *e_ = out + E->segoff[si##k + 1]; \
    while (o##k < e_) { uint64_t w_ = bswap64(ld64(E->arena + (ba##k >> 3))) << (ba##k & 7); \
        uint16_t t_ = E->H[0].tab[w_ >> 53]; *o##k++ = t_ >> 4; ba##k += t_ & 15; } }
#define H0ANY(k) any |= (uintptr_t)!done##k;

#define H0STEP(k) if (!done##k) { \
    uint64_t w_ = bswap64(ld64(E->arena + (ba##k >> 3))) << (ba##k & 7); \
    uint16_t t_ = E->H[0].tab[w_ >> 53]; \
    *o##k++ = t_ >> 4; ba##k += t_ & 15; }

#define H0STEP2(k) if (!done##k) { \
    uint64_t w_ = bswap64(ld64(E->arena + (ba##k >> 3))) << (ba##k & 7); \
    uint16_t t_ = E->H[0].tab[w_ >> 53]; uint32_t l_ = t_ & 15; \
    uint16_t u_ = E->H[0].tab[(w_ << l_) >> 53]; \
    o##k[0] = t_ >> 4; o##k[1] = u_ >> 4; o##k += 2; ba##k += l_ + (u_ & 15); }

static void dec_huf0(const HENC *E, uint8_t *out) {
    FOR8(H0DECL)
    for (;;) { FOR8_2(H0TICK, 8)
        uintptr_t any = 0; FOR8(H0ANY) if (!any) break;
        FOR8(H0STEP) }
}
static void dec_huf0x2(const HENC *E, uint8_t *out) {
    FOR8(H0DECL)
    for (;;) { FOR8_2(H0TICK, 8)
        uintptr_t any = 0; FOR8(H0ANY) if (!any) break;
        FOR8(H0STEP2) }
}

/* order-1 fused: K per-class streams, context = top bits of prev symbol */
#define H1DECL(k) \
    uint64_t ba##k[4]; uint32_t c##k = 0; uint8_t *o##k = 0, *lim##k = 0; \
    int si##k = k, done##k = 0; H1LOAD(k)
#define H1LOAD(k) if (si##k < E->nseg) { \
        for (int q_ = 0; q_ < E->K; q_++) ba##k[q_] = E->sb[si##k * E->K + q_]; \
        c##k = 0; o##k = out + E->segoff[si##k]; \
        lim##k = out + E->segoff[si##k + 1] - MARGIN; \
    } else done##k = 1;
#define H1TICK(k, NCH) if (!done##k && o##k > lim##k) { \
        H1DRAIN(k) si##k += NCH; H1LOAD(k) }
#define H1DRAIN(k) { uint8_t *e_ = out + E->segoff[si##k + 1]; \
    while (o##k < e_) { uint64_t b_ = ba##k[c##k]; \
        uint64_t w_ = bswap64(ld64(E->arena + (b_ >> 3))) << (b_ & 7); \
        uint16_t t_ = E->H[c##k].tab[w_ >> 53]; \
        ba##k[c##k] = b_ + (t_ & 15); uint8_t s_ = t_ >> 4; \
        *o##k++ = s_; c##k = s_ >> SHIFT_; } }
#define H1ANY(k) any |= (uintptr_t)!done##k;

#define H1STEP(k) if (!done##k) { \
    uint64_t b_ = ba##k[c##k]; \
    uint64_t w_ = bswap64(ld64(E->arena + (b_ >> 3))) << (b_ & 7); \
    uint16_t t_ = E->H[c##k].tab[w_ >> 53]; \
    ba##k[c##k] = b_ + (t_ & 15); uint8_t s_ = t_ >> 4; \
    *o##k++ = s_; c##k = s_ >> SHIFT_; }

#define GEN_H1(NAME, SHIFT) \
static void NAME(const HENC *E, uint8_t *out) { \
    enum { SHIFT_ = SHIFT }; \
    FOR8(H1DECL) \
    for (;;) { FOR8_2(H1TICK, 8) \
        uintptr_t any = 0; FOR8(H1ANY) if (!any) break; \
        FOR8(H1STEP) } \
}
GEN_H1(dec_huf1k4, 6)
GEN_H1(dec_huf1k2, 7)

/* ---------- harness ------------------------------------------------------- */
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
typedef void (*hfn)(const HENC *, uint8_t *);
static void run(const char *name, hfn fn, const HENC *E, const uint8_t *v, uint8_t *out) {
    size_t N = E->N;
    memset(out, 0xAA, N); fn(E, out);
    if (memcmp(out, v, N)) { printf("  %-8s VERIFY FAIL\n", name); return; }
    int R = 1 + (int)(6e8 / N); if (R > 3000) R = 3000;
    double best = 1e30;
    for (int i = 0; i < R; i++) { double t0 = now(); fn(E, out); double t1 = now();
                                  if (t1 - t0 < best) best = t1 - t0; }
    printf("  %-8s %7.3f ns/B  %6.2f GB/s   %5.3f bits/B\n",
           name, 1e9 * best / N, N / best / 1e9, (double)E->bits / N);
}

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }

int main(int argc, char **argv) {
    { /* selftest on skewed random data */
        size_t N = 1 << 20; uint8_t *v = malloc(N), *out = malloc(N + 64);
        for (size_t i = 0; i < N; i++) { uint32_t r = xr(); v[i] = (uint8_t)((r & 0xF) < 12 ? (r >> 8) & 63 : (r >> 8)); }
        HENC *e0 = hencode(v, N, 1, 8192, 0);
        HENC *e4 = hencode(v, N, 4, 8192, 6);
        HENC *e2 = hencode(v, N, 2, 8192, 7);
        uint8_t *o = out;
        memset(o, 0, N); dec_huf0(e0, o);   if (memcmp(o, v, N)) { printf("SELFTEST FAIL huf0\n"); return 1; }
        memset(o, 0, N); dec_huf0x2(e0, o); if (memcmp(o, v, N)) { printf("SELFTEST FAIL huf0x2\n"); return 1; }
        memset(o, 0, N); dec_huf1k4(e4, o); if (memcmp(o, v, N)) { printf("SELFTEST FAIL huf1k4\n"); return 1; }
        memset(o, 0, N); dec_huf1k2(e2, o); if (memcmp(o, v, N)) { printf("SELFTEST FAIL huf1k2\n"); return 1; }
        henc_free(e0); henc_free(e4); henc_free(e2); free(v); free(out);
        printf("selftest ok\n");
    }
    const char *defs[] = {"dickens", "webster", "xml", "samba", "x-ray", "mozilla"};
    int nf = argc > 1 ? argc - 1 : 6;
    for (int fi = 0; fi < nf; fi++) {
        const char *name = argc > 1 ? argv[fi + 1] : defs[fi];
        char path[256], mp4[256], mp2[256];
        snprintf(path, sizeof path, "/tmp/phd_%s/lit", name);
        snprintf(mp4, sizeof mp4, "/tmp/o1maps/%s.map4", name);
        snprintf(mp2, sizeof mp2, "/tmp/o1maps/%s.map2", name);
        FILE *f = fopen(path, "rb"); if (!f) { printf("missing %s\n", path); continue; }
        fseek(f, 0, SEEK_END); size_t N = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *v0 = malloc(N); size_t rd = fread(v0, 1, N, f); fclose(f);
        if (rd != N) { free(v0); continue; }
        uint8_t map4[256], map2[256];
        for (int i = 0; i < 256; i++) map4[i] = map2[i] = (uint8_t)i;
        FILE *m = fopen(mp4, "rb"); if (m) { rd = fread(map4, 1, 256, m); fclose(m); }
        m = fopen(mp2, "rb"); if (m) { rd = fread(map2, 1, 256, m); fclose(m); }
        uint8_t *v4 = malloc(N), *v2 = malloc(N);
        for (size_t i = 0; i < N; i++) { v4[i] = map4[v0[i]]; v2[i] = map2[v0[i]]; }
        uint8_t *out = malloc(N + 64);
        size_t segb = 16384; if (N / 32 < 16384) segb = N / 32 >= 2048 ? N / 32 : 2048;
        printf("%s (%zu bytes)\n", path, N);
        HENC *e0 = hencode(v0, N, 1, segb, 0);
        HENC *e4 = hencode(v4, N, 4, segb, 6);
        HENC *e2 = hencode(v2, N, 2, segb, 7);
        run("huf0", dec_huf0, e0, v0, out);
        run("huf0x2", dec_huf0x2, e0, v0, out);
        run("huf1k2", dec_huf1k2, e2, v2, out);
        run("huf1k4", dec_huf1k4, e4, v4, out);
        henc_free(e0); henc_free(e4); henc_free(e2);
        free(v0); free(v4); free(v2); free(out);
    }
    return 0;
}
