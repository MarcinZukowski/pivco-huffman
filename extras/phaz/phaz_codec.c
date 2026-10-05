/* phaz_codec.c -- buffer API + shared helpers for the pivco-Huffman entropy
 * transplant onto zstd.  See phaz_codec.h.
 *
 * Container layout (little-endian):
 *   "phaz" magic (4) | version u8 |
 *   hdr[5] u64: n, nseq, lits, extrabits, nblk |
 *   blk_ns[nblk] u32 | blk_tl[nblk] u32 | blk_cf[nblk] u8 |
 *   xblen u64 | xb[xblen] |
 *   4x stream: method u8 | blen u64 | blob[blen]   (ll, ml, of, lit)
 */
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define ZSTD_STATIC_LINKING_ONLY   /* ZSTD_sequenceBound, advanced cctx params */
#include "zstd.h"
#include "pivcohuf_file.h"
#include "phaz_codec.h"

#define PHAZ_MAGIC "phaz"
#define PHAZ_VER   2          /* v2: + per-block blk_cf[] repcode-confirmed flags */

const char *const phaz_stream_names[4] = { "ll", "ml", "of", "lit" };

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

void phaz_capture_free(void) {
    free(g_phaz_llc);    free(g_phaz_mlc);    free(g_phaz_ofc);
    free(g_phaz_lit);    free(g_phaz_xb);
    free(g_phaz_blk_ns); free(g_phaz_blk_tl); free(g_phaz_blk_cf);
    g_phaz_llc = g_phaz_mlc = g_phaz_ofc = g_phaz_lit = g_phaz_xb = NULL;
    g_phaz_blk_ns = g_phaz_blk_tl = NULL;
    g_phaz_blk_cf = NULL;
}

size_t phaz_capture_run(const unsigned char *src, size_t n, int level) {
    phaz_capture_free();                       /* drop any previous run's buffers */
    size_t bound = ZSTD_compressBound(n);
    unsigned char *c = malloc(bound);
    if (!c) return (size_t)-1;

    size_t zsize = 0;

    size_t sb = ZSTD_sequenceBound(n);
    g_phaz_llc = malloc(sb); g_phaz_mlc = malloc(sb); g_phaz_ofc = malloc(sb);
    g_phaz_lit = malloc(n + 64);
    g_phaz_xb  = calloc(sb * 8 + 64, 1);
    g_phaz_blk_ns = calloc((n >> 10) + 64, sizeof(unsigned));
    g_phaz_blk_tl = calloc((n >> 10) + 64, sizeof(unsigned));
    g_phaz_blk_cf = calloc((n >> 10) + 64, 1);
    if (!g_phaz_llc || !g_phaz_mlc || !g_phaz_ofc || !g_phaz_lit ||
        !g_phaz_xb || !g_phaz_blk_ns || !g_phaz_blk_tl || !g_phaz_blk_cf) {
        free(c); phaz_capture_free(); return (size_t)-1;
    }
    g_phaz_nseq = 0; g_phaz_lits = 0; g_phaz_extrabits = 0;
    g_phaz_xbpos = 0; g_phaz_nblk = 0; g_phaz_dump = 1;

    /* Stock-configured compress with the capture riding it: the captured
     * parse must be exactly the parse zstd ships.  Any parameter drift
     * changes the parse at the btopt levels (literal pricing feeds the
     * optimal parser's cost model) and breaks phaz-vs-zstd attribution. */
    zsize = ZSTD_compress(c, bound, src, n, level);
    g_phaz_dump = 0;
    free(c);
    if (ZSTD_isError(zsize)) { phaz_capture_free(); return (size_t)-1; }
    return zsize;
}

/* PHA-encode one stream; see header.  PHA (#PHA) gates FSE per node by
 * compressibility, so it dominates plain PH -- no need to try both.  Raw
 * fallback only if PHA expands (tiny streams).  One global table per stream
 * (per-128KB re-table was tried and lost: pivcohuf pays a full header +
 * checksum + table per call, no FSE repeat-mode). */
pivco_cfg_t g_phaz_cfg = {
    .tree_mode = PIVCO_TREE_MODE_OPTIMIZED,
    .effort = PIVCO_EFFORT_PLAIN,
    .fse_enabled = 1,
    .fse_static_enabled = 1,
    .fse_nibble_enabled = 0,
    .fse_k1_enabled = 0,
    .fse_k2_enabled = 0,
    .flat_layout = PIVCO_FLAT_VERTICAL,
};
size_t g_phaz_seg_blocks = (PIVCOHUF_SEGMENT_BYTES_DEFAULT + PIVCO_BLOCK_SIZE - 1) / PIVCO_BLOCK_SIZE;
size_t g_phaz_blk = PIVCO_BLOCK_SIZE;

size_t phaz_pack_stream(unsigned char **cur, unsigned char *end,
                        const unsigned char *raw, size_t rawlen,
                        size_t *best_out, char *tag_out) {
    size_t bound = pivcohuf_compress_bound_seg(rawlen ? rawlen : 1, g_phaz_blk, g_phaz_seg_blocks);
    unsigned char *t = malloc(bound);
    if (!t) return 0;
    size_t l = bound;
    if (getenv("PHAZ_PH")) g_phaz_cfg.fse_enabled = 0;   /* plain PH for embedded runs */
    int ok = rawlen && pivcohuf_compress_seg(raw, rawlen, t, &l, &g_phaz_cfg,
                                             g_phaz_blk, g_phaz_seg_blocks, NULL) == PIVCOHUF_OK;
    const unsigned char *blob = raw; uint64_t blen = rawlen;
    unsigned char method = 0; char tag = 'r';
    if (ok && l < blen) { blob = t; blen = l; method = 1; tag = 'a'; }

    size_t need = 1 + sizeof(blen) + blen;
    if (cur) {
        if (*cur + need > end) { free(t); return 0; }   /* would overflow dst */
        *(*cur)++ = method;
        memcpy(*cur, &blen, sizeof blen); *cur += sizeof blen;
        memcpy(*cur, blob, blen);         *cur += blen;
    }
    free(t);
    if (best_out) *best_out = blen;
    if (tag_out)  *tag_out  = tag;
    return need;
}

/* Inverse of phaz_pack_stream: read method+len+blob from *p, return rawlen
 * decoded bytes (caller frees), or NULL on any malformed/short input. */
static unsigned char *unpack_stream(const unsigned char **p, const unsigned char *end,
                                    size_t rawlen) {
    if (*p + 1 + 8 > end) return NULL;
    unsigned char method = *(*p)++;
    uint64_t blen; memcpy(&blen, *p, 8); *p += 8;
    if (*p + blen > end) return NULL;
    const unsigned char *blob = *p; *p += blen;
    unsigned char *raw = malloc(rawlen ? rawlen + 64 : 64);
    if (!raw) return NULL;
    if (method == 0) {
        if (blen != rawlen) { free(raw); return NULL; }
        memcpy(raw, blob, rawlen);
    } else {
        size_t got = rawlen;
        if (pivcohuf_decompress(blob, blen, raw, &got) != PIVCOHUF_OK || got != rawlen) {
            free(raw); return NULL;
        }
    }
    return raw;
}

/* ---- Context-binned coding of the code streams (Duda's context binning) ----
 *
 * Each sequence code is coded under a context made of other fields of the same
 * and the previous sequence, mapped into a few bins; the stream is demuxed by
 * bin into sub-streams, each coded on its own.  Decode order per sequence:
 * of[i] | (ll[i-1], ml[i-1]);  ml[i] | (of[i], ll[i-1]);  ll[i] | (ml[i], of[i]).
 * Nested maps add the previous sequence's pair as a second stage. */
#include "phaz_ctxmaps.h"
#define NLL 36
#define NML 53
#define NOF 32
enum { CTX_NONE = 0, CTX_BUILTIN = 1, CTX_PAIR = 2, CTX_NESTED = 3 };
/* Chain-free variant (mode | CTX_CF): of stays order-0, ml's contexts use of
 * only, ll's use ml and of; every routing pass then depends only on streams
 * already materialized, so it runs as a plain gather. */
#define CTX_CF 8
#define CTX_KIND(mode) ((mode) & 7)
#define CTX_ISCF(mode) (((mode) & CTX_CF) != 0)
/* Product-form maps (chain-free only): each field of the pair is binned on
 * its own to 8 states, the 8x8 combinations to B (or to 8 states per stage
 * when nested, then the 8x8 stage combinations to B).  Every table has at
 * most 64 entries, so the bin pass runs 16 symbols at a time on NEON. */
#define CTX_PROD  4
#define CTX_PRODN 5
#define CTX_F 8
enum { CTX_A = 64, CTX_STAGE = 32 };
typedef struct {
    int mode, B;
    unsigned char *m1, *m2, *mn;   /* stage maps (transmitted modes) */
    unsigned char *f1, *g1, *j1, *f2, *g2, *j2, *jn;   /* product-form tables, 64 bytes each */
} ctxmodel_t;

/* Raw pair contexts: stage 0 = the near pair, stage 1 = the previous pair. */
static inline int ctx_width(int k, int stage, int cf) {
    static const int w[3][2] = { { NML * NOF, NML * NOF }, { NOF * NLL, NOF * NML }, { NLL * NML, NLL * NML } };
    if (cf && k == 1) return NOF * NOF;
    return w[k][stage];
}
/* History before `base` (a routing lane's first sequence) reads as zero. */
static inline int ctx_raw(int k, int stage, int cf, const unsigned char *ll, const unsigned char *ml,
                          const unsigned char *of, size_t i, size_t base) {
    int h1 = i >= base + 1, h2 = i >= base + 2, h3 = i >= base + 3;
    switch (k) {
    case 0:  return stage == 0 ? ml[i] * NOF + of[i]
                               : (h1 ? ml[i-1] * NOF + of[i-1] : 0);
    case 1:  if (cf) return stage == 0 ? of[i] * NOF + (h1 ? of[i-1] : 0)
                                       : (h2 ? of[i-2] * NOF + (h3 ? of[i-3] : 0) : 0);
             return stage == 0 ? of[i] * NLL + (h1 ? ll[i-1] : 0)
                               : (h1 ? of[i-1] * NML + ml[i-1] : 0);
    default: return stage == 0 ? (h1 ? ll[i-1] * NML + ml[i-1] : 0)
                               : (h2 ? ll[i-2] * NML + ml[i-2] : 0);
    }
}
/* Product-form fields: ll | (ml, of) + (ml-1, of-1); ml | (of, of-1) + (of-2, of-3). */
static inline void ctx_prod_fields(int k, const unsigned char *ll, const unsigned char *ml, const unsigned char *of,
                                   size_t i, size_t base, int *a, int *b, int *a2, int *b2) {
    int h1 = i >= base + 1, h2 = i >= base + 2, h3 = i >= base + 3;
    (void)ll;
    if (k == 0) { *a = ml[i]; *b = of[i]; *a2 = h1 ? ml[i-1] : 0; *b2 = h1 ? of[i-1] : 0; }
    else        { *a = of[i]; *b = h1 ? of[i-1] : 0; *a2 = h2 ? of[i-2] : 0; *b2 = h3 ? of[i-3] : 0; }
}
static inline int ctx_prod_wa(int k, int stage) { return k == 0 ? (stage ? NML : NML) : NOF; }
static inline int ctx_prod_wb(int k, int stage) { (void)k; (void)stage; return NOF; }
static inline int ctx_bin_prod(const ctxmodel_t *m, int k, const unsigned char *ll, const unsigned char *ml,
                               const unsigned char *of, size_t i, size_t base) {
    int a, b, a2, b2;
    ctx_prod_fields(k, ll, ml, of, i, base, &a, &b, &a2, &b2);
    int s1 = m->j1[m->f1[a] * CTX_F + m->g1[b]];
    if (CTX_KIND(m->mode) == CTX_PROD) return s1;
    int s2 = m->j2[m->f2[a2] * CTX_F + m->g2[b2]];
    return m->jn[s1 * CTX_F + s2];
}
static inline int ctx_bin(const ctxmodel_t *m, int k, const unsigned char *ll, const unsigned char *ml,
                          const unsigned char *of, size_t i, size_t base) {
    int cf = CTX_ISCF(m->mode);
    if (CTX_KIND(m->mode) >= CTX_PROD) return ctx_bin_prod(m, k, ll, ml, of, i, base);
    int r0 = ctx_raw(k, 0, cf, ll, ml, of, i, base);
    switch (CTX_KIND(m->mode)) {
    case CTX_BUILTIN: return k == 0 ? phaz_map_ll[r0] : k == 1 ? (cf ? phaz_map_ml_cf[r0] : phaz_map_ml[r0]) : phaz_map_of[r0];
    case CTX_PAIR:    return m->m1[r0];
    case CTX_NESTED:  return m->mn[m->m1[r0] * CTX_STAGE + m->m2[ctx_raw(k, 1, cf, ll, ml, of, i, base)]];
    default:          return 0;
    }
}

/* Greedy context merging: start from one bin per context, merge the pair whose
 * union costs the fewest extra bits, stop at B bins.  Contexts unseen in the
 * counts go to the largest bin.  Returns the bin count. */
static double ctx_cost(const uint32_t *c) {
    double n = 0; for (int y = 0; y < CTX_A; y++) n += c[y];
    if (n == 0) return 0;
    double b = n * log2(n);
    for (int y = 0; y < CTX_A; y++) if (c[y]) b -= c[y] * log2((double)c[y]);
    return b;
}
typedef struct { double d; int a, b; } ctx_cand_t;
static int ctx_fit(const uint32_t *cnt, int V, int B, unsigned char *binof) {
    int maxn = 2 * V;
    uint32_t *nc = calloc((size_t)maxn * CTX_A, sizeof *nc);
    double *cs = calloc(maxn, sizeof *cs);
    int *alive = calloc(maxn, sizeof *alive), *parent = malloc(maxn * sizeof *parent);
    size_t hcap = (size_t)V * V / 2 + V + 16, hn = 0;
    ctx_cand_t *heap = malloc(hcap * sizeof *heap);
    int nn = V, live = 0;
    for (int v = 0; v < V; v++) {
        memcpy(nc + (size_t)v * CTX_A, cnt + (size_t)v * CTX_A, CTX_A * sizeof *nc);
        cs[v] = ctx_cost(nc + (size_t)v * CTX_A);
        parent[v] = v;
        double s = 0; for (int y = 0; y < CTX_A; y++) s += cnt[(size_t)v * CTX_A + y];
        alive[v] = s > 0; live += alive[v];
    }
#define HPUSH(c) do { if (hn == hcap) { hcap *= 2; heap = realloc(heap, hcap * sizeof *heap); } \
        size_t i_ = hn++; while (i_ && heap[(i_-1)/2].d > (c).d) { heap[i_] = heap[(i_-1)/2]; i_ = (i_-1)/2; } heap[i_] = (c); } while (0)
    for (int a = 0; a < V; a++) { if (!alive[a]) continue;
        for (int b = a + 1; b < V; b++) { if (!alive[b]) continue;
            uint32_t u[CTX_A]; for (int y = 0; y < CTX_A; y++) u[y] = nc[(size_t)a*CTX_A+y] + nc[(size_t)b*CTX_A+y];
            ctx_cand_t c = { ctx_cost(u) - cs[a] - cs[b], a, b }; HPUSH(c); } }
    while (live > B && hn) {
        ctx_cand_t c = heap[0];                        /* pop */
        ctx_cand_t last = heap[--hn]; size_t i = 0;
        for (;;) { size_t l = 2*i+1, r = l+1, m = i; double md = last.d;
            if (l < hn && heap[l].d < md) { m = l; md = heap[l].d; }
            if (r < hn && heap[r].d < md) { m = r; }
            if (m == i) break; heap[i] = heap[m]; i = m; }
        heap[i] = last;
        if (!alive[c.a] || !alive[c.b]) continue;
        int m = nn++;
        for (int y = 0; y < CTX_A; y++) nc[(size_t)m*CTX_A+y] = nc[(size_t)c.a*CTX_A+y] + nc[(size_t)c.b*CTX_A+y];
        cs[m] = ctx_cost(nc + (size_t)m * CTX_A);
        alive[c.a] = alive[c.b] = 0; parent[c.a] = parent[c.b] = m; parent[m] = m; alive[m] = 1; live--;
        for (int o = 0; o < nn - 1; o++) { if (!alive[o]) continue;
            uint32_t u[CTX_A]; for (int y = 0; y < CTX_A; y++) u[y] = nc[(size_t)m*CTX_A+y] + nc[(size_t)o*CTX_A+y];
            ctx_cand_t cc = { ctx_cost(u) - cs[m] - cs[o], m, o }; HPUSH(cc); }
    }
#undef HPUSH
    int *lab = malloc(nn * sizeof *lab); int nb = 0; for (int i = 0; i < nn; i++) lab[i] = -1;
    int big = -1; double bigs = -1;
    for (int i = 0; i < nn; i++) { if (!alive[i]) continue; double s = 0; for (int y = 0; y < CTX_A; y++) s += nc[(size_t)i*CTX_A+y]; if (s > bigs) { bigs = s; big = i; } }
    for (int v = 0; v < V; v++) { double s = 0; for (int y = 0; y < CTX_A; y++) s += cnt[(size_t)v*CTX_A+y];
        int r = v; if (s == 0) r = big; else while (parent[r] != r) r = parent[r];
        if (lab[r] < 0) lab[r] = nb++; binof[v] = (unsigned char)lab[r]; }
    free(nc); free(cs); free(alive); free(parent); free(heap); free(lab);
    return nb;
}

/* Fit a transmitted model for stream k on the file's own streams. */
static void ctx_fit_model(ctxmodel_t *m, int k, int mode, int B, const unsigned char *ll,
                          const unsigned char *ml, const unsigned char *of, size_t nseq) {
    const unsigned char *tgt = k == 0 ? ll : k == 1 ? ml : of;
    m->mode = mode; m->m1 = m->m2 = m->mn = NULL;
    int cf = CTX_ISCF(mode), kind = CTX_KIND(mode);
    int V1 = ctx_width(k, 0, cf);
    uint32_t *c1 = calloc((size_t)V1 * CTX_A, sizeof *c1);
    for (size_t i = 0; i < nseq; i++) c1[(size_t)ctx_raw(k, 0, cf, ll, ml, of, i, 0) * CTX_A + tgt[i]]++;
    m->m1 = malloc(V1);
    int nb1 = ctx_fit(c1, V1, kind == CTX_NESTED ? CTX_STAGE : B, m->m1);
    free(c1);
    if (kind == CTX_PAIR) { m->B = nb1; return; }
    int V2 = ctx_width(k, 1, cf);
    uint32_t *c2 = calloc((size_t)V2 * CTX_A, sizeof *c2);
    for (size_t i = 0; i < nseq; i++) c2[(size_t)ctx_raw(k, 1, cf, ll, ml, of, i, 0) * CTX_A + tgt[i]]++;
    m->m2 = malloc(V2);
    int nb2 = ctx_fit(c2, V2, CTX_STAGE, m->m2);
    free(c2); (void)nb2;
    uint32_t *cn = calloc((size_t)CTX_STAGE * CTX_STAGE * CTX_A, sizeof *cn);
    for (size_t i = 0; i < nseq; i++)
        cn[(size_t)(m->m1[ctx_raw(k, 0, cf, ll, ml, of, i, 0)] * CTX_STAGE + m->m2[ctx_raw(k, 1, cf, ll, ml, of, i, 0)]) * CTX_A + tgt[i]]++;
    m->mn = malloc(CTX_STAGE * CTX_STAGE);
    m->B = ctx_fit(cn, CTX_STAGE * CTX_STAGE, B, m->mn);
    free(cn);
}
static void ctx_free_model(ctxmodel_t *m) { free(m->m1); free(m->m2); free(m->mn); m->m1 = m->m2 = m->mn = NULL;
    free(m->f1); free(m->g1); free(m->j1); free(m->f2); free(m->g2); free(m->j2); free(m->jn);
    m->f1 = m->g1 = m->j1 = m->f2 = m->g2 = m->j2 = m->jn = NULL; }

/* Bin one field's values (width w) to CTX_F states by the target's counts. */
static unsigned char *ctx_fit_field(const int *vals, const unsigned char *tgt, size_t nseq, int w) {
    uint32_t *c = calloc((size_t)w * CTX_A, sizeof *c);
    for (size_t i = 0; i < nseq; i++) c[(size_t)vals[i] * CTX_A + tgt[i]]++;
    unsigned char *map = calloc(64, 1);
    ctx_fit(c, w, CTX_F, map);
    free(c);
    return map;
}
static void ctx_fit_prod(ctxmodel_t *m, int k, int mode, int B, const unsigned char *ll,
                         const unsigned char *ml, const unsigned char *of, size_t nseq) {
    const unsigned char *tgt = k == 0 ? ll : ml;
    memset(m, 0, sizeof *m); m->mode = mode;
    int *a = malloc(nseq * sizeof *a), *b = malloc(nseq * sizeof *b), *a2 = malloc(nseq * sizeof *a2), *b2 = malloc(nseq * sizeof *b2);
    for (size_t i = 0; i < nseq; i++) ctx_prod_fields(k, ll, ml, of, i, 0, &a[i], &b[i], &a2[i], &b2[i]);
    m->f1 = ctx_fit_field(a, tgt, nseq, ctx_prod_wa(k, 0));
    m->g1 = ctx_fit_field(b, tgt, nseq, ctx_prod_wb(k, 0));
    int nested = CTX_KIND(mode) == CTX_PRODN;
    uint32_t *c = calloc((size_t)64 * CTX_A, sizeof *c);
    for (size_t i = 0; i < nseq; i++) c[(size_t)(m->f1[a[i]] * CTX_F + m->g1[b[i]]) * CTX_A + tgt[i]]++;
    m->j1 = calloc(64, 1);
    int nb1 = ctx_fit(c, 64, nested ? CTX_F : B, m->j1);
    free(c);
    if (!nested) { m->B = nb1; free(a); free(b); free(a2); free(b2); return; }
    m->f2 = ctx_fit_field(a2, tgt, nseq, ctx_prod_wa(k, 1));
    m->g2 = ctx_fit_field(b2, tgt, nseq, ctx_prod_wb(k, 1));
    c = calloc((size_t)64 * CTX_A, sizeof *c);
    for (size_t i = 0; i < nseq; i++) c[(size_t)(m->f2[a2[i]] * CTX_F + m->g2[b2[i]]) * CTX_A + tgt[i]]++;
    m->j2 = calloc(64, 1);
    ctx_fit(c, 64, CTX_F, m->j2);
    free(c);
    c = calloc((size_t)64 * CTX_A, sizeof *c);
    for (size_t i = 0; i < nseq; i++) {
        int s1 = m->j1[m->f1[a[i]] * CTX_F + m->g1[b[i]]], s2 = m->j2[m->f2[a2[i]] * CTX_F + m->g2[b2[i]]];
        c[(size_t)(s1 * CTX_F + s2) * CTX_A + tgt[i]]++; }
    m->jn = calloc(64, 1);
    m->B = ctx_fit(c, 64, B, m->jn);
    free(c); free(a); free(b); free(a2); free(b2);
}

/* Map bytes: entries packed at a fixed width. */
static int ctx_bits_for(int n) { int b = 0; while ((1 << b) < n) b++; return b < 1 ? 1 : b; }
static size_t ctx_pack_map(unsigned char *out, const unsigned char *map, int V, int width) {
    size_t bits = (size_t)V * width, nbytes = (bits + 7) / 8;
    if (out) { memset(out, 0, nbytes); for (int v = 0; v < V; v++) for (int b = 0; b < width; b++)
        if (map[v] >> b & 1) out[((size_t)v * width + b) >> 3] |= (unsigned char)(1u << (((size_t)v * width + b) & 7)); }
    return nbytes;
}
static void ctx_unpack_map(const unsigned char *in, unsigned char *map, int V, int width) {
    for (int v = 0; v < V; v++) { unsigned x = 0; for (int b = 0; b < width; b++)
        x |= (unsigned)((in[((size_t)v * width + b) >> 3] >> (((size_t)v * width + b) & 7)) & 1) << b; map[v] = (unsigned char)x; }
}
static size_t ctx_prod_map_bytes(const ctxmodel_t *m, int k) {
    int nested = CTX_KIND(m->mode) == CTX_PRODN, wb = ctx_bits_for(m->B);
    size_t n = ctx_pack_map(NULL, NULL, ctx_prod_wa(k, 0), 3) + ctx_pack_map(NULL, NULL, ctx_prod_wb(k, 0), 3)
             + ctx_pack_map(NULL, NULL, 64, nested ? 3 : wb);
    if (nested) n += ctx_pack_map(NULL, NULL, ctx_prod_wa(k, 1), 3) + ctx_pack_map(NULL, NULL, ctx_prod_wb(k, 1), 3)
                   + ctx_pack_map(NULL, NULL, 64, 3) + ctx_pack_map(NULL, NULL, 64, wb);
    return n;
}
static size_t ctx_map_bytes(const ctxmodel_t *m, int k) {
    int cf = CTX_ISCF(m->mode), kind = CTX_KIND(m->mode);
    if (kind >= CTX_PROD) return ctx_prod_map_bytes(m, k);
    if (kind == CTX_PAIR)   return ctx_pack_map(NULL, m->m1, ctx_width(k, 0, cf), ctx_bits_for(m->B));
    if (kind == CTX_NESTED) return ctx_pack_map(NULL, m->m1, ctx_width(k, 0, cf), 5) + ctx_pack_map(NULL, m->m2, ctx_width(k, 1, cf), 5)
                                    + ctx_pack_map(NULL, m->mn, CTX_STAGE * CTX_STAGE, ctx_bits_for(m->B));
    return 0;
}

/* Demux stream k by bin and pack the sub-streams; cur == NULL sizes only.
 * Layout: [method 2][mode][B][map bytes][per bin: u32 rawlen, phaz_pack_stream record]. */
/* Routing lanes: the decoder routes L contiguous ranges of the sequence index
 * at once, so a bin's symbols are stored lane by lane with (L-1) lane counts. */
static int ctx_lanes(size_t nseq) { const char *e = getenv("PHAZ_CTX_LANES"); if (e) return atoi(e); return nseq >= 65536 ? 4 : 1; }
static size_t ctx_pack_stream(unsigned char **cur, unsigned char *end, int k, const ctxmodel_t *m,
                              const unsigned char *ll, const unsigned char *ml, const unsigned char *of,
                              size_t nseq, size_t *enc_out) {
    const unsigned char *tgt = k == 0 ? ll : k == 1 ? ml : of;
    int B = m->B, L = ctx_lanes(nseq);
    size_t *cnt = calloc((size_t)B * L, sizeof *cnt), *off = calloc((size_t)B * L + 1, sizeof *off);
    unsigned char *bins = malloc(nseq ? nseq : 1);
    unsigned char *binof = malloc(nseq ? nseq : 1);
    /* lane l covers sequences [l*nseq/L, (l+1)*nseq/L), the decoder's split */
    for (int l = 0; l < L; l++) for (size_t i = (size_t)l * nseq / L; i < (size_t)(l + 1) * nseq / L; i++) {
        int b = ctx_bin(m, k, ll, ml, of, i, (size_t)l * nseq / L); binof[i] = (unsigned char)b; cnt[(size_t)b * L + l]++; }
    for (size_t x = 0; x < (size_t)B * L; x++) off[x + 1] = off[x] + cnt[x];
    size_t *fill = calloc((size_t)B * L, sizeof *fill);
    for (int l = 0; l < L; l++) for (size_t i = (size_t)l * nseq / L; i < (size_t)(l + 1) * nseq / L; i++) {
        size_t x = (size_t)binof[i] * L + l; bins[off[x] + fill[x]++] = tgt[i]; }
    size_t mapb = ctx_map_bytes(m, k), total = 4 + mapb;
    if (cur) {
        if (*cur + 4 + mapb > end) { free(cnt); free(off); free(bins); free(fill); free(binof); return 0; }
        *(*cur)++ = 2; *(*cur)++ = (unsigned char)m->mode; *(*cur)++ = (unsigned char)B; *(*cur)++ = (unsigned char)L;
        int cf = CTX_ISCF(m->mode), kind = CTX_KIND(m->mode);
        if (kind >= CTX_PROD) { int nested = kind == CTX_PRODN, wb = ctx_bits_for(B);
            *cur += ctx_pack_map(*cur, m->f1, ctx_prod_wa(k, 0), 3); *cur += ctx_pack_map(*cur, m->g1, ctx_prod_wb(k, 0), 3);
            *cur += ctx_pack_map(*cur, m->j1, 64, nested ? 3 : wb);
            if (nested) { *cur += ctx_pack_map(*cur, m->f2, ctx_prod_wa(k, 1), 3); *cur += ctx_pack_map(*cur, m->g2, ctx_prod_wb(k, 1), 3);
                *cur += ctx_pack_map(*cur, m->j2, 64, 3); *cur += ctx_pack_map(*cur, m->jn, 64, wb); } }
        else if (kind == CTX_PAIR) *cur += ctx_pack_map(*cur, m->m1, ctx_width(k, 0, cf), ctx_bits_for(B));
        else if (kind == CTX_NESTED) { *cur += ctx_pack_map(*cur, m->m1, ctx_width(k, 0, cf), 5);
            *cur += ctx_pack_map(*cur, m->m2, ctx_width(k, 1, cf), 5); *cur += ctx_pack_map(*cur, m->mn, CTX_STAGE * CTX_STAGE, ctx_bits_for(B)); }
    }
    size_t enc = 0;
    for (int b = 0; b < B; b++) {
        size_t bl = off[(size_t)(b + 1) * L] - off[(size_t)b * L];
        uint32_t rl = (uint32_t)bl;
        if (cur) { if (*cur + 4 * L > end) { total = 0; break; } memcpy(*cur, &rl, 4); *cur += 4;
            for (int l = 0; l + 1 < L; l++) { uint32_t lc = (uint32_t)cnt[(size_t)b * L + l]; memcpy(*cur, &lc, 4); *cur += 4; } }
        size_t e = 0, need = phaz_pack_stream(cur, end, bins + off[(size_t)b * L], bl, &e, NULL);
        if (need == 0) { total = 0; break; }
        total += 4 * L + need; enc += e;
    }
    free(cnt); free(off); free(bins); free(fill); free(binof);
    if (enc_out) *enc_out = enc;
    return total;
}

/* Inverse: parse a method-2 record into a model plus per-bin buffers. */
typedef struct { ctxmodel_t m; int L; unsigned char **sub; size_t *len, *pos; const unsigned char *t0; } ctxstream_t;
static int ctx_unpack_stream(const unsigned char **p, const unsigned char *end, int k, ctxstream_t *cs) {
    if (*p + 4 > end) return 0;
    ctxmodel_t *m = &cs->m; m->mode = (*p)[1]; m->B = (*p)[2]; cs->L = (*p)[3]; *p += 4; m->m1 = m->m2 = m->mn = NULL;
    int cf = CTX_ISCF(m->mode), kind = CTX_KIND(m->mode);
    m->f1 = m->g1 = m->j1 = m->f2 = m->g2 = m->j2 = m->jn = NULL;
    if (m->B < 1 || m->B > CTX_A || kind < CTX_BUILTIN || kind > CTX_PRODN || cs->L < 1 || cs->L > 8) return 0;
    if (cf && k == 2) return 0;
    if (kind >= CTX_PROD && (!cf || k == 2)) return 0;
    if (kind >= CTX_PROD) { int nested = kind == CTX_PRODN, wb = ctx_bits_for(m->B);
        unsigned char **tabs[7] = { &m->f1, &m->g1, &m->j1, &m->f2, &m->g2, &m->j2, &m->jn };
        int widths[7] = { ctx_prod_wa(k, 0), ctx_prod_wb(k, 0), 64, ctx_prod_wa(k, 1), ctx_prod_wb(k, 1), 64, 64 };
        int bitsw[7] = { 3, 3, nested ? 3 : wb, 3, 3, 3, wb };
        for (int t = 0; t < (nested ? 7 : 3); t++) { size_t nb = ctx_pack_map(NULL, NULL, widths[t], bitsw[t]);
            if (*p + nb > end) return 0; *tabs[t] = calloc(64, 1); ctx_unpack_map(*p, *tabs[t], widths[t], bitsw[t]); *p += nb; }
    }
    else if (kind == CTX_PAIR) { int V = ctx_width(k, 0, cf), w = ctx_bits_for(m->B); size_t nb = ctx_pack_map(NULL, NULL, V, w);
        if (*p + nb > end) return 0; m->m1 = malloc(V); ctx_unpack_map(*p, m->m1, V, w); *p += nb; }
    else if (kind == CTX_NESTED) { int V1 = ctx_width(k, 0, cf), V2 = ctx_width(k, 1, cf), w = ctx_bits_for(m->B);
        size_t n1 = ctx_pack_map(NULL, NULL, V1, 5), n2 = ctx_pack_map(NULL, NULL, V2, 5), n3 = ctx_pack_map(NULL, NULL, CTX_STAGE * CTX_STAGE, w);
        if (*p + n1 + n2 + n3 > end) return 0;
        m->m1 = malloc(V1); ctx_unpack_map(*p, m->m1, V1, 5); *p += n1;
        m->m2 = malloc(V2); ctx_unpack_map(*p, m->m2, V2, 5); *p += n2;
        m->mn = malloc(CTX_STAGE * CTX_STAGE); ctx_unpack_map(*p, m->mn, CTX_STAGE * CTX_STAGE, w); *p += n3; }
    int L = cs->L;
    cs->sub = calloc(m->B, sizeof *cs->sub); cs->len = calloc((size_t)m->B * L, sizeof *cs->len); cs->pos = calloc((size_t)m->B * L, sizeof *cs->pos);
    for (int b = 0; b < m->B; b++) { if (*p + 4 * L > end) return 0; uint32_t rl; memcpy(&rl, *p, 4); *p += 4;
        size_t start = 0;
        for (int l = 0; l + 1 < L; l++) { uint32_t lc; memcpy(&lc, *p, 4); *p += 4; cs->pos[(size_t)b * L + l] = start; start += lc; cs->len[(size_t)b * L + l] = start; }
        cs->pos[(size_t)b * L + L - 1] = start; cs->len[(size_t)b * L + L - 1] = rl;
        if (start > rl) return 0;
        cs->sub[b] = unpack_stream(p, end, rl); if (!cs->sub[b]) return 0; }
    /* stage-0 table as a flat lookup over the raw near-pair context */
    cs->t0 = kind == CTX_BUILTIN ? (k == 0 ? phaz_map_ll : k == 1 ? (cf ? phaz_map_ml_cf : phaz_map_ml) : phaz_map_of) : m->m1;
    return 1;
}
/* Product-form bin pass over [base, end): the first three symbols of a lane
 * scalar (short history), the rest 16 at a time on NEON; scalar elsewhere. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline uint8x16x4_t ctx_tbl64(const unsigned char *t) { uint8x16x4_t r; r.val[0] = vld1q_u8(t); r.val[1] = vld1q_u8(t + 16); r.val[2] = vld1q_u8(t + 32); r.val[3] = vld1q_u8(t + 48); return r; }
#endif
static void ctx_bins_prod(const ctxmodel_t *m, int k, const unsigned char *ll, const unsigned char *ml,
                          const unsigned char *of, size_t base, size_t end, unsigned char *bins) {
    int nested = CTX_KIND(m->mode) == CTX_PRODN;
    size_t i = base;
    for (; i < end && i < base + 3; i++) bins[i] = (unsigned char)ctx_bin_prod(m, k, ll, ml, of, i, base);
#if defined(__ARM_NEON)
    uint8x16x4_t tf1 = ctx_tbl64(m->f1), tg1 = ctx_tbl64(m->g1), tj1 = ctx_tbl64(m->j1);
    uint8x16x4_t tf2, tg2, tj2, tjn;
    if (nested) { tf2 = ctx_tbl64(m->f2); tg2 = ctx_tbl64(m->g2); tj2 = ctx_tbl64(m->j2); tjn = ctx_tbl64(m->jn); }
    const unsigned char *sa = k == 0 ? ml : of, *sb = of, *sa2 = k == 0 ? ml : of, *sb2 = of;
    size_t da = 0, db = k == 0 ? 0 : 1, da2 = k == 0 ? 1 : 2, db2 = k == 0 ? 1 : 3;
    for (; i + 16 <= end; i += 16) {
        uint8x16_t a = vld1q_u8(sa + i - da), b = vld1q_u8(sb + i - db);
        uint8x16_t idx1 = vorrq_u8(vshlq_n_u8(vqtbl4q_u8(tf1, a), 3), vqtbl4q_u8(tg1, b));
        uint8x16_t s1 = vqtbl4q_u8(tj1, idx1);
        if (nested) {
            uint8x16_t a2 = vld1q_u8(sa2 + i - da2), b2 = vld1q_u8(sb2 + i - db2);
            uint8x16_t idx2 = vorrq_u8(vshlq_n_u8(vqtbl4q_u8(tf2, a2), 3), vqtbl4q_u8(tg2, b2));
            uint8x16_t s2 = vqtbl4q_u8(tj2, idx2);
            s1 = vqtbl4q_u8(tjn, vorrq_u8(vshlq_n_u8(s1, 3), s2)); }
        vst1q_u8(bins + i, s1);
    }
#endif
    for (; i < end; i++) bins[i] = (unsigned char)ctx_bin_prod(m, k, ll, ml, of, i, base);
}

/* Chain-free routing: one gather pass per context-coded stream, in the order
 * ml (from of) then ll (from ml, of); contexts are lane-local like the encoder's. */
static int ctx_route_cf(size_t nseq, ctxstream_t *cs, int ctxd[3], unsigned char *ll, unsigned char *ml, unsigned char *of) {
    static const int order[2] = { 1, 0 };
    unsigned char *bins = malloc(nseq ? nseq : 1);
    for (int oi = 0; oi < 2; oi++) { int k = order[oi]; if (!ctxd[k]) continue;
        ctxstream_t *c = &cs[k]; int L = c->L, nb = c->m.B, nested = CTX_KIND(c->m.mode) == CTX_NESTED;
        unsigned char *out = k == 0 ? ll : ml;
        /* pass 1: every symbol's bin, no dependency between symbols */
        size_t lbase[8], lend[8], lane_len = 0;
        int prod = CTX_KIND(c->m.mode) >= CTX_PROD;
        for (int l = 0; l < L; l++) { lbase[l] = (size_t)l * nseq / L; lend[l] = (size_t)(l + 1) * nseq / L; if (lend[l] - lbase[l] > lane_len) lane_len = lend[l] - lbase[l];
            if (prod) { ctx_bins_prod(&c->m, k, ll, ml, of, lbase[l], lend[l], bins); continue; }
            for (size_t i = lbase[l]; i < lend[l]; i++) {
                int b = c->t0[ctx_raw(k, 0, 1, ll, ml, of, i, lbase[l])];
                if (nested) b = c->m.mn[b * CTX_STAGE + c->m.m2[ctx_raw(k, 1, 1, ll, ml, of, i, lbase[l])]];
                bins[i] = (unsigned char)b; } }
        /* pass 2: gather, lanes interleaved so consecutive pulls use different cursors */
        const unsigned char **cur = malloc((size_t)nb * L * sizeof *cur), **stop = malloc((size_t)nb * L * sizeof *stop);
        for (int b = 0; b < nb; b++) for (int l = 0; l < L; l++) { cur[b * L + l] = c->sub[b] + c->pos[(size_t)b * L + l]; stop[b * L + l] = c->sub[b] + c->len[(size_t)b * L + l]; }
        int ok = 1;
        if (L == 4) {
            size_t n0 = lend[0] - lbase[0], n1 = lend[1] - lbase[1], n2 = lend[2] - lbase[2], n3 = lend[3] - lbase[3], nmin = n0;
            if (n1 < nmin) nmin = n1; if (n2 < nmin) nmin = n2; if (n3 < nmin) nmin = n3;
            unsigned char *o0 = out + lbase[0], *o1 = out + lbase[1], *o2 = out + lbase[2], *o3 = out + lbase[3];
            const unsigned char *b0 = bins + lbase[0], *b1 = bins + lbase[1], *b2 = bins + lbase[2], *b3 = bins + lbase[3];
            for (size_t j = 0; j < nmin; j++) {
                int x0 = b0[j] * 4, x1 = b1[j] * 4 + 1, x2 = b2[j] * 4 + 2, x3 = b3[j] * 4 + 3;
                o0[j] = *cur[x0]++; o1[j] = *cur[x1]++; o2[j] = *cur[x2]++; o3[j] = *cur[x3]++; }
            for (int l = 0; l < 4; l++) for (size_t j = nmin; j < lend[l] - lbase[l]; j++) { int x = bins[lbase[l] + j] * 4 + l; out[lbase[l] + j] = *cur[x]++; }
            for (int x = 0; x < nb * 4; x++) if (cur[x] > stop[x]) ok = 0;
        } else {
            for (size_t j = 0; j < lane_len; j++) for (int l = 0; l < L; l++) { size_t i = lbase[l] + j; if (i >= lend[l]) continue;
                int x = bins[i] * L + l; if (cur[x] >= stop[x]) { ok = 0; break; } out[i] = *cur[x]++; }
        }
        free(cur); free(stop);
        if (!ok) { free(bins); return 0; }
    }
    free(bins);
    return 1;
}
static void ctx_free_stream(ctxstream_t *cs) { if (cs->sub) { for (int b = 0; b < cs->m.B; b++) free(cs->sub[b]); } free(cs->sub); free(cs->len); free(cs->pos); ctx_free_model(&cs->m); }

/* Route the three code streams back into flat arrays: L independent lanes of
 * the sequence index advance together, each pulling from its own cursors. */
static inline int ctx_bin_fast(const ctxstream_t *c, int r0, int r1) {
    int s0 = c->t0[r0];
    return CTX_KIND(c->m.mode) == CTX_NESTED ? c->m.mn[s0 * CTX_STAGE + c->m.m2[r1]] : s0;
}
static int ctx_route(size_t nseq, ctxstream_t *cs, int ctxd[3], unsigned char *ll, unsigned char *ml, unsigned char *of) {
    int L = 1; for (int k = 0; k < 3; k++) if (ctxd[k]) L = cs[k].L;
    for (int k = 0; k < 3; k++) if (ctxd[k] && cs[k].L != L) return 0;
    size_t lbase[8], lend[8], lane_len = 0;
    for (int l = 0; l < L; l++) { lbase[l] = (size_t)l * nseq / L; lend[l] = (size_t)(l + 1) * nseq / L; if (lend[l] - lbase[l] > lane_len) lane_len = lend[l] - lbase[l]; }
    /* per (stream, bin, lane) cursor and end, as pointers */
    const unsigned char **cur[3]; const unsigned char **stop[3];
    for (int k = 0; k < 3; k++) { cur[k] = stop[k] = NULL; if (!ctxd[k]) continue; int nb = cs[k].m.B;
        cur[k] = malloc((size_t)nb * L * sizeof *cur[k]); stop[k] = malloc((size_t)nb * L * sizeof *stop[k]);
        for (int b = 0; b < nb; b++) for (int l = 0; l < L; l++) { cur[k][b * L + l] = cs[k].sub[b] + cs[k].pos[(size_t)b * L + l]; stop[k][b * L + l] = cs[k].sub[b] + cs[k].len[(size_t)b * L + l]; } }
    int ok = 1;
    for (size_t j = 0; j < lane_len && ok; j++) {
        for (int l = 0; l < L; l++) {
            size_t i = lbase[l] + j;
            if (i >= lend[l]) continue;
            unsigned pl = j >= 1 ? ll[i-1] : 0, pm = j >= 1 ? ml[i-1] : 0, ppl = j >= 2 ? ll[i-2] : 0, ppm = j >= 2 ? ml[i-2] : 0;
            if (ctxd[2]) { int x = ctx_bin_fast(&cs[2], pl * NML + pm, ppl * NML + ppm) * L + l;
                if (cur[2][x] >= stop[2][x]) { ok = 0; break; } of[i] = *cur[2][x]++; }
            unsigned o = of[i], po = j >= 1 ? of[i-1] : 0;
            if (ctxd[1]) { int x = ctx_bin_fast(&cs[1], o * NLL + pl, po * NML + pm) * L + l;
                if (cur[1][x] >= stop[1][x]) { ok = 0; break; } ml[i] = *cur[1][x]++; }
            unsigned mm = ml[i];
            if (ctxd[0]) { int x = ctx_bin_fast(&cs[0], mm * NOF + o, pm * NOF + po) * L + l;
                if (cur[0][x] >= stop[0][x]) { ok = 0; break; } ll[i] = *cur[0][x]++; }
        }
    }
    for (int k = 0; k < 3; k++) { free(cur[k]); free(stop[k]); }
    return ok;
}

size_t phaz_compress_bound(size_t n) {
    size_t nblk    = (n >> 17) + 2;                          /* ~128KB zstd blocks */
    size_t hdr     = 5 + sizeof(uint64_t) * 5
                     + nblk * (sizeof(unsigned) * 2 + 1) + sizeof(uint64_t);
    size_t xb      = ZSTD_sequenceBound(n) + 64;             /* extra-bits ceiling */
    size_t seqb    = ZSTD_sequenceBound(n);
    size_t streams = 4 * (1 + sizeof(uint64_t))             /* per-stream framing */
                     + pivcohuf_compress_bound(n)            /* lit ~ n */
                     + 3 * pivcohuf_compress_bound(seqb);    /* ll/ml/of ~ nseq */
    size_t ctx = 3 * (3 + 4096 + CTX_A * (4 + 1 + 8));       /* maps + per-bin framing */
    return hdr + xb + streams + ctx + 64;
}

size_t phaz_compress(const void *src_, size_t n, void *dst_, size_t cap,
                     int level, phaz_stats *st) {
    const unsigned char *src = src_;
    unsigned char *dst = dst_, *cur = dst, *end = dst + cap;

    double t0 = now();
    if (phaz_capture_run(src, n, level) == (size_t)-1) return 0;
    if (st) st->capture_ms = (now() - t0) * 1e3;

    uint64_t hdr[5] = { (uint64_t)n, (uint64_t)g_phaz_nseq, (uint64_t)g_phaz_lits,
                        (uint64_t)g_phaz_extrabits, (uint64_t)g_phaz_nblk };
    uint64_t xblen = (g_phaz_xbpos + 7) / 8;
    size_t fixed = 5 + sizeof hdr + g_phaz_nblk * (sizeof(unsigned) * 2 + 1)
                   + sizeof xblen + xblen;
    if (cur + fixed > end) { phaz_capture_free(); return 0; }

    memcpy(cur, PHAZ_MAGIC, 4); cur += 4; *cur++ = PHAZ_VER;
    memcpy(cur, hdr, sizeof hdr); cur += sizeof hdr;
    memcpy(cur, g_phaz_blk_ns, g_phaz_nblk * sizeof(unsigned)); cur += g_phaz_nblk * sizeof(unsigned);
    memcpy(cur, g_phaz_blk_tl, g_phaz_nblk * sizeof(unsigned)); cur += g_phaz_nblk * sizeof(unsigned);
    memcpy(cur, g_phaz_blk_cf, g_phaz_nblk); cur += g_phaz_nblk;
    memcpy(cur, &xblen, sizeof xblen); cur += sizeof xblen;
    memcpy(cur, g_phaz_xb, xblen); cur += xblen;

    const unsigned char *sp[4] = { g_phaz_llc, g_phaz_mlc, g_phaz_ofc, g_phaz_lit };
    size_t srl[4] = { g_phaz_nseq, g_phaz_nseq, g_phaz_nseq, g_phaz_lits };
    int ctx_on = getenv("PHAZ_CTX") != NULL;   /* context-binned code streams, off by default */
    for (int i = 0; i < 4; i++) {
        double a = now(); size_t enc = 0;
        if (st) { st->ctx_mode[i] = 0; st->ctx_bins[i] = 1; st->ctx_map_bytes[i] = 0; }
        if (i < 3 && ctx_on && g_phaz_nseq > 0) {
            /* pick the smallest of order-0, the built-in pair map, and fitted
             * pair / nested maps at 16..64 bins, by actual packed size */
            size_t best = phaz_pack_stream(NULL, NULL, sp[i], srl[i], NULL, NULL);
            ctxmodel_t bestm; memset(&bestm, 0, sizeof bestm); bestm.mode = CTX_NONE; bestm.B = 1;
            const int cand_mode[13] = { CTX_BUILTIN, CTX_PAIR, CTX_PAIR, CTX_PAIR, CTX_NESTED, CTX_NESTED, CTX_NESTED,
                                        CTX_PROD, CTX_PROD, CTX_PROD, CTX_PRODN, CTX_PRODN, CTX_PRODN };
            const int cand_B[13]    = { PHAZ_CTX_B, 16, 32, 64, 16, 32, 64, 16, 32, 64, 16, 32, 64 };
            int cf = getenv("PHAZ_CTX_CF") != NULL;            /* chain-free contexts */
            int prodonly = getenv("PHAZ_CTX_PRODONLY") != NULL;
            int ncand = getenv("PHAZ_CTX_BUILTIN") ? 1 : (cf ? 13 : 7);   /* product kinds are chain-free only */
            int maxb = getenv("PHAZ_CTX_MAXB") ? atoi(getenv("PHAZ_CTX_MAXB")) : 64;
            for (int c = 0; c < ncand && !(cf && i == 2); c++) {
                if (cand_B[c] > maxb) continue;
                if (prodonly && cand_mode[c] < CTX_PROD) continue;
                ctxmodel_t m; memset(&m, 0, sizeof m); m.mode = cand_mode[c] | (cf ? CTX_CF : 0); m.B = cand_B[c];
                if (CTX_KIND(m.mode) >= CTX_PROD) ctx_fit_prod(&m, i, m.mode, m.B, g_phaz_llc, g_phaz_mlc, g_phaz_ofc, g_phaz_nseq);
                else if (CTX_KIND(m.mode) != CTX_BUILTIN) ctx_fit_model(&m, i, m.mode, m.B, g_phaz_llc, g_phaz_mlc, g_phaz_ofc, g_phaz_nseq);
                size_t sz = ctx_pack_stream(NULL, NULL, i, &m, g_phaz_llc, g_phaz_mlc, g_phaz_ofc, g_phaz_nseq, NULL);
                if (sz && sz < best) { ctx_free_model(&bestm); best = sz; bestm = m; } else ctx_free_model(&m);
            }
            if (bestm.mode != CTX_NONE) {
                size_t sz = ctx_pack_stream(&cur, end, i, &bestm, g_phaz_llc, g_phaz_mlc, g_phaz_ofc, g_phaz_nseq, &enc);
                if (st) { st->ctx_mode[i] = bestm.mode; st->ctx_bins[i] = bestm.B; st->ctx_map_bytes[i] = ctx_map_bytes(&bestm, i); }
                ctx_free_model(&bestm);
                if (sz == 0) { phaz_capture_free(); return 0; }
                if (st) { st->pack_ms[i] = (now() - a) * 1e3; st->stream_raw[i] = srl[i]; st->stream_enc[i] = enc; }
                continue;
            }
        }
        if (phaz_pack_stream(&cur, end, sp[i], srl[i], &enc, NULL) == 0) {
            phaz_capture_free(); return 0;
        }
        if (st) { st->pack_ms[i] = (now() - a) * 1e3;
                  st->stream_raw[i] = srl[i]; st->stream_enc[i] = enc; }
    }
    size_t total = (size_t)(cur - dst);
    phaz_capture_free();
    return total;
}

size_t phaz_decompress(const void *src_, size_t fn, void *dst_, size_t cap,
                       phaz_stats *st) {
    const unsigned char *buf = src_, *p = buf, *end = buf + fn;
    if (fn < 5 + sizeof(uint64_t) * 5 ||
        memcmp(p, PHAZ_MAGIC, 4) != 0 || p[4] != PHAZ_VER) return 0;
    p += 5;
    uint64_t hdr[5]; memcpy(hdr, p, sizeof hdr); p += sizeof hdr;
    size_t n = hdr[0], nseq = hdr[1], lits = hdr[2], nblk = hdr[4];
    if (n > cap) return 0;

    size_t na = (nblk ? nblk : 1) * sizeof(unsigned);
    unsigned *bns = malloc(na), *btl = malloc(na);
    unsigned char *bcf = malloc(nblk ? nblk : 1);
    if (!bns || !btl || !bcf) { free(bns); free(btl); free(bcf); return 0; }
    if (p + nblk * (sizeof(unsigned) * 2 + 1) > end) { free(bns); free(btl); free(bcf); return 0; }
    memcpy(bns, p, nblk * sizeof(unsigned)); p += nblk * sizeof(unsigned);
    memcpy(btl, p, nblk * sizeof(unsigned)); p += nblk * sizeof(unsigned);
    memcpy(bcf, p, nblk); p += nblk;
    if (p + 8 > end) { free(bns); free(btl); free(bcf); return 0; }
    uint64_t xblen; memcpy(&xblen, p, 8); p += 8;
    if (p + xblen > end) { free(bns); free(btl); free(bcf); return 0; }
    const unsigned char *xb = p; p += xblen;

    size_t srl[4] = { nseq, nseq, nseq, lits };
    unsigned char *str[4] = { 0, 0, 0, 0 };
    ctxstream_t cs[3]; memset(cs, 0, sizeof cs); int ctxd[3] = { 0, 0, 0 };
    for (int i = 0; i < 4; i++) {
        double a = now();
        if (i < 3 && p < end && *p == 2) {
            ctxd[i] = 1;
            if (!ctx_unpack_stream(&p, end, i, &cs[i])) {
                for (int j = 0; j < 3; j++) ctx_free_stream(&cs[j]); for (int j = 0; j < i; j++) free(str[j]);
                free(bns); free(btl); free(bcf); return 0; }
            str[i] = malloc(nseq ? nseq + 64 : 64);
            if (st) { st->ctx_mode[i] = cs[i].m.mode; st->ctx_bins[i] = cs[i].m.B; }
        } else {
            str[i] = unpack_stream(&p, end, srl[i]);
            if (st) { st->ctx_mode[i] = 0; st->ctx_bins[i] = 1; }
        }
        if (!str[i]) { for (int j = 0; j < 3; j++) ctx_free_stream(&cs[j]); for (int j = 0; j < i; j++) free(str[j]);
                       free(bns); free(btl); free(bcf); return 0; }
        if (st) { st->entropy_ms[i] = (now() - a) * 1e3; st->stream_raw[i] = srl[i]; }
    }
    double trt = now();
    int cf = 0; for (int k = 0; k < 3; k++) if (ctxd[k] && CTX_ISCF(cs[k].m.mode)) cf = 1;
    if (ctxd[0] || ctxd[1] || ctxd[2]) {
        if (!(cf ? ctx_route_cf(nseq, cs, ctxd, str[0], str[1], str[2]) : ctx_route(nseq, cs, ctxd, str[0], str[1], str[2]))) {
            for (int j = 0; j < 3; j++) ctx_free_stream(&cs[j]); for (int j = 0; j < 4; j++) free(str[j]);
            free(bns); free(btl); free(bcf); return 0; }
    }
    for (int j = 0; j < 3; j++) ctx_free_stream(&cs[j]);
    if (st) st->route_ms = (now() - trt) * 1e3;
    double tr = now();
    size_t got = ZSTD_phazDecode(dst_, cap, str[0], str[1], str[2], xb, str[3],
                                 lits, bns, btl, bcf, nblk);
    if (st) st->reconstruct_ms = (now() - tr) * 1e3;
    for (int i = 0; i < 4; i++) free(str[i]);
    free(bns); free(btl); free(bcf);
    return got == n ? got : 0;
}
