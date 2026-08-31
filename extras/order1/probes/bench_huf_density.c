/* bench_huf_density: issue #29 prototype.
 *
 * Static per-density BYTE-HUFFMAN tables for routing bitmaps, on the
 * same density schedule as the static FSE tables (pivco_fse_tables.h:
 * p = 0.50..0.99 step 0.01, selected via pivco_fse_select_table with
 * the same XOR-flip normalization).  A density-p table's byte
 * probabilities are the iid-bit binomial P(b) = p^(8-pop) (1-p)^pop,
 * so code lengths depend only on popcount(b); lengths are optimal
 * under a 12-bit cap (package-merge), canonical codes.
 *
 * Harvests REAL merge bitmaps: chunks the input 32K, builds the pivco
 * table (PH, FSE off), encodes, walks the wire (post-order records per
 * src/pivco_huffman_wire.h, PR#30 layout: flat regions carry a marker
 * too) and collects every merge bitmap.  For each bitmap >= 8 bytes it
 * prices: raw, static FSE, dynamic nibble FSE (PR#30), and byte-huf --
 * all with the same +2B length framing, commit-if-smaller vs raw.
 *
 * Speed: per-bitmap encode and decode of the committed sets, byte-huf
 * (single-stream 12-bit-LUT decoder) vs pivco_fse, roundtrip-verified.
 * (The Huffman table build here is bench-local table construction for
 * the 50 density tables, not a copy of a production primitive.)
 */
#include "pivco_huffman.h"
#include "pivco_fse.h"
#include "pivco_fse_tables.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#define CHUNK    32768
#define HUF_LOG  12
#define MINB     8          /* min bitmap bytes to attempt any coder */

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- per-density byte-huffman tables ---------- */

typedef struct {
    uint8_t  len[256];
    uint16_t code[256];         /* canonical, LSB-first convenient form */
    uint8_t  dec_sym[1 << HUF_LOG];
    uint8_t  dec_len[1 << HUF_LOG];
} huftab_t;

static huftab_t g_huf[PIVCO_FSE_NUM_TABLES + 1];

/* package-merge: optimal lengths under HUF_LOG cap for 256 weights.
   Items carry their symbol multiset as an index list; 256 syms x 12
   levels is small enough to brute force. */
typedef struct { double w; uint16_t nsyms; uint16_t *syms; } pk_t;
static int pk_cmp(const void *a, const void *b)
{
    double d = ((const pk_t *)a)->w - ((const pk_t *)b)->w;
    return d < 0 ? -1 : d > 0 ? 1 : 0;
}
static void pm_lengths(const double *w, uint8_t *len)
{
        enum { N = 256 };
    static uint16_t pool[1 << 22];
    size_t pool_off = 0;
    int cnt[N]; memset(cnt, 0, sizeof(cnt));
    pk_t *prev = NULL; int nprev = 0;
    pk_t *lists[HUF_LOG + 1] = {0}; int nlist[HUF_LOG + 1] = {0};
    for (int l = HUF_LOG; l >= 1; l--) {
        /* items at this level: the N singletons + packages of pairs from prev */
        int cap = N + nprev / 2;
        pk_t *cur = malloc(sizeof(pk_t) * (size_t)cap);
        int n = 0;
        for (int s = 0; s < N; s++) {
            cur[n].w = w[s]; cur[n].nsyms = 1;
            cur[n].syms = &pool[pool_off]; pool[pool_off++] = (uint16_t)s;
            n++;
        }
        for (int i = 0; i + 1 < nprev; i += 2) {
            cur[n].w = prev[i].w + prev[i + 1].w;
            cur[n].nsyms = (uint16_t)(prev[i].nsyms + prev[i + 1].nsyms);
            cur[n].syms = &pool[pool_off];
            memcpy(&pool[pool_off], prev[i].syms, sizeof(uint16_t) * prev[i].nsyms);
            pool_off += prev[i].nsyms;
            memcpy(&pool[pool_off], prev[i + 1].syms, sizeof(uint16_t) * prev[i + 1].nsyms);
            pool_off += prev[i + 1].nsyms;
            n++;
        }
        qsort(cur, (size_t)n, sizeof(pk_t), pk_cmp);
        lists[l] = cur; nlist[l] = n;
        prev = cur; nprev = n;
    }
    /* take the first 2N-2 items of the level-1 list; count symbol occurrences */
    int take = 2 * N - 2;
    for (int i = 0; i < take && i < nlist[1]; i++)
        for (int k = 0; k < lists[1][i].nsyms; k++)
            cnt[lists[1][i].syms[k]]++;
    for (int s = 0; s < N; s++) len[s] = (uint8_t)cnt[s];
    for (int l = 1; l <= HUF_LOG; l++) free(lists[l]);
}

static void build_huftab_w(huftab_t *t, const double *w)
{
    pm_lengths(w, t->len);
    /* canonical codes, MSB-first assignment, stored bit-reversed so the
       encoder can append LSB-first and the decoder can index the LUT
       with the low HUF_LOG bits of the window. */
    int hist[HUF_LOG + 1] = {0};
    for (int b = 0; b < 256; b++) hist[t->len[b]]++;
    uint32_t next[HUF_LOG + 1]; uint32_t code = 0;
    for (int l = 1; l <= HUF_LOG; l++) { next[l] = code; code = (code + (uint32_t)hist[l]) << 1; }
    for (int b = 0; b < 256; b++) {
        int l = t->len[b];
        uint32_t c = next[l]++;
        uint32_t rev = 0;
        for (int i = 0; i < l; i++) rev |= ((c >> i) & 1u) << (l - 1 - i);
        t->code[b] = (uint16_t)rev;
    }
    /* decode LUT: for every HUF_LOG-bit window, which symbol + length */
    for (int b = 0; b < 256; b++) {
        int l = t->len[b];
        uint32_t c = t->code[b];
        for (uint32_t fill = 0; fill < (1u << (HUF_LOG - l)); fill++) {
            uint32_t e = c | (fill << l);
            t->dec_sym[e] = (uint8_t)b;
            t->dec_len[e] = (uint8_t)l;
        }
    }
}

/* canonical-code + LUT rebuild from lengths alone (the decode side's
 * per-group cost in the shared-table scheme) */
static void huftab_from_lengths(huftab_t *t, const uint8_t *len)
{
    memcpy(t->len, len, 256);
    int hist[HUF_LOG + 1] = {0};
    for (int b = 0; b < 256; b++) hist[t->len[b]]++;
    uint32_t next[HUF_LOG + 1]; uint32_t code = 0;
    for (int l = 1; l <= HUF_LOG; l++) { next[l] = code; code = (code + (uint32_t)hist[l]) << 1; }
    for (int b = 0; b < 256; b++) {
        int l = t->len[b];
        uint32_t c = next[l]++;
        uint32_t rev = 0;
        for (int i = 0; i < l; i++) rev |= ((c >> i) & 1u) << (l - 1 - i);
        t->code[b] = (uint16_t)rev;
        for (uint32_t fill = 0; fill < (1u << (HUF_LOG - l)); fill++) {
            uint32_t e = t->code[b] | (fill << l);
            t->dec_sym[e] = (uint8_t)b;
            t->dec_len[e] = (uint8_t)l;
        }
    }
}

static void build_huftab(int idx)
{
    double p = (double)pivco_fse_freq[idx];    /* frequent-bit prob */
    double q = 1.0 - p;
    double w[256];
    for (int b = 0; b < 256; b++) {
        int pop = __builtin_popcount((unsigned)b);
        w[b] = pow(p, 8 - pop) * pow(q, pop);
        if (w[b] < 1e-12) w[b] = 1e-12;
    }
    build_huftab_w(&g_huf[idx], w);
}

/* ---------- bitstream encode/decode ---------- */

static size_t huf_encode(const huftab_t *t, const uint8_t *src, size_t n, uint8_t *dst)
{
    uint64_t acc = 0; int nbits = 0; uint8_t *d = dst;
    for (size_t i = 0; i < n; i++) {
        acc |= (uint64_t)t->code[src[i]] << nbits;
        nbits += t->len[src[i]];
        while (nbits >= 8) { *d++ = (uint8_t)acc; acc >>= 8; nbits -= 8; }
    }
    if (nbits) *d++ = (uint8_t)acc;
    return (size_t)(d - dst);
}

static void huf_decode(const huftab_t *t, const uint8_t *src, size_t src_len,
                       uint8_t *dst, size_t n)
{
    /* single stream, 4 symbols per refill: refill to >= 56 bits (one
     * unaligned 64-bit load), then 4 lookups of <= HUF_LOG bits each
     * (4 x 12 = 48 <= 56) with no branch in between. */
    uint64_t acc = 0; int nbits = 0; size_t sp = 0, i = 0;
    const uint32_t M = (1u << HUF_LOG) - 1;
    while (i + 4 <= n && sp + 8 <= src_len) {
        uint64_t w; memcpy(&w, src + sp, 8);
        acc |= w << nbits;
        int take = (63 - nbits) >> 3;
        sp += (size_t)take; nbits += take << 3;
        for (int k = 0; k < 4; k++) {
            uint32_t e = (uint32_t)acc & M;
            dst[i + (size_t)k] = t->dec_sym[e];
            int l = t->dec_len[e];
            acc >>= l; nbits -= l;
        }
        i += 4;
    }
    for (; i < n; i++) {
        while (nbits <= 56 && sp < src_len) { acc |= (uint64_t)src[sp++] << nbits; nbits += 8; }
        uint32_t e = (uint32_t)acc & M;
        dst[i] = t->dec_sym[e];
        int l = t->dec_len[e];
        acc >>= l; nbits -= l;
    }
}

/* ---- 4-stream form: [len0:u16][len1:u16][len2:u16][s0][s1][s2][s3] ---- */

static size_t huf_encode4(const huftab_t *t, const uint8_t *src, size_t n, uint8_t *dst)
{
    size_t n4 = n / 4;
    size_t cut[5] = {0, n4, 2 * n4, 3 * n4, n};
    uint8_t *d = dst + 6;
    size_t lens[4];
    for (int k = 0; k < 4; k++) {
        lens[k] = huf_encode(t, src + cut[k], cut[k + 1] - cut[k], d);
        d += lens[k];
    }
    dst[0] = (uint8_t)lens[0]; dst[1] = (uint8_t)(lens[0] >> 8);
    dst[2] = (uint8_t)lens[1]; dst[3] = (uint8_t)(lens[1] >> 8);
    dst[4] = (uint8_t)lens[2]; dst[5] = (uint8_t)(lens[2] >> 8);
    return (size_t)(d - dst);
}

typedef struct { uint64_t acc; int nbits; const uint8_t *p, *end; } hstr_t;
static inline void hs_refill(hstr_t *s)
{
    if (s->nbits <= 55 && s->p + 8 <= s->end) {
        uint64_t w; memcpy(&w, s->p, 8);
        s->acc |= w << s->nbits;
        int take = (63 - s->nbits) >> 3;
        s->p += take; s->nbits += take << 3;
    } else {
        while (s->nbits <= 56 && s->p < s->end) { s->acc |= (uint64_t)*s->p++ << s->nbits; s->nbits += 8; }
    }
}
static void huf_decode4(const huftab_t *t, const uint8_t *src, size_t src_len,
                        uint8_t *dst, size_t n)
{
    const uint32_t M = (1u << HUF_LOG) - 1;
    size_t n4 = n / 4;
    size_t cut[5] = {0, n4, 2 * n4, 3 * n4, n};
    size_t l0 = src[0] | ((size_t)src[1] << 8);
    size_t l1 = src[2] | ((size_t)src[3] << 8);
    size_t l2 = src[4] | ((size_t)src[5] << 8);
    const uint8_t *base = src + 6;
    hstr_t st[4];
    st[0] = (hstr_t){0, 0, base, base + l0};
    st[1] = (hstr_t){0, 0, base + l0, base + l0 + l1};
    st[2] = (hstr_t){0, 0, base + l0 + l1, base + l0 + l1 + l2};
    st[3] = (hstr_t){0, 0, base + l0 + l1 + l2, src + src_len};
    size_t i = 0, m = cut[1];          /* lockstep over the shortest quarter */
    /* fast loop: one refill per stream per iteration, then 4 symbols per
     * stream (4 x HUF_LOG = 48 <= 56 buffered bits), 16 symbols/iter */
    for (; i + 4 <= m; i += 4) {
        for (int k = 0; k < 4; k++) {
            hstr_t *s2 = &st[k];
            hs_refill(s2);
            uint64_t a = s2->acc; int nb = s2->nbits;
            uint8_t *o = dst + cut[k] + i;
            for (int j = 0; j < 4; j++) {
                uint32_t e = (uint32_t)a & M;
                o[j] = t->dec_sym[e];
                int l = t->dec_len[e];
                a >>= l; nb -= l;
            }
            s2->acc = a; s2->nbits = nb;
        }
    }
    for (; i < m; i++) {
        for (int k = 0; k < 4; k++) {
            hstr_t *s2 = &st[k];
            hs_refill(s2);
            uint32_t e = (uint32_t)s2->acc & M;
            dst[cut[k] + i] = t->dec_sym[e];
            int l = t->dec_len[e];
            s2->acc >>= l; s2->nbits -= l;
        }
    }
    for (size_t i2 = cut[3] + n4; i2 < n; i2++) {   /* stream 3 tail */
        hstr_t *s2 = &st[3];
        hs_refill(s2);
        uint32_t e = (uint32_t)s2->acc & M;
        dst[i2] = t->dec_sym[e];
        int l = t->dec_len[e];
        s2->acc >>= l; s2->nbits -= l;
    }
}

/* size in bits without materializing */
static uint64_t huf_bits(const huftab_t *t, const uint8_t *src, size_t n)
{
    uint64_t bits = 0;
    for (size_t i = 0; i < n; i++) bits += t->len[src[i]];
    return bits;
}

/* ---------- wire walk: collect merge bitmaps ---------- */

typedef struct { const uint8_t *bm; int K; int depth; int16_t node; } bmref_t;
static bmref_t g_bms[65536];
static int g_nbm;
static uint64_t g_flat_bytes;   /* raw flat-region payload in this chunk */

static const uint8_t *walk(const pivco_table_t *t, int16_t id, int K, const uint8_t *p, int depth)
{
    const pivco_tree_node_t *n = &t->tree[id];
    if (K == 0 || n->symbol >= 0) return p;
    if (t->flat_depth[id] >= 2) {                  /* flat region: marker + raw body */
        p += 1;                                    /* marker (0: raw, fse off) */
        size_t fb = ((size_t)K * t->flat_depth[id] + 7) >> 3;
        g_flat_bytes += fb;
        p += fb;
        return p;
    }
    const pivco_tree_node_t *L = &t->tree[n->left], *R = &t->tree[n->right];
    int left_leaf = L->symbol >= 0, right_leaf = R->symbol >= 0;
    int nb = (K + 7) >> 3;
    if (left_leaf && right_leaf) {
        p += 1;                                     /* marker */
        if (g_nbm < 65536) { g_bms[g_nbm].bm = p; g_bms[g_nbm].K = K; g_bms[g_nbm].depth = depth; g_bms[g_nbm].node = id; g_nbm++; }
        return p + nb;
    }
    int kr = p[0] | (p[1] << 8); p += 2;
    int kl = K - kr;
    if (left_leaf) {                                /* right child internal */
        p = walk(t, n->right, kr, p, depth + 1);
    } else if (right_leaf) {
        p = walk(t, n->left, kl, p, depth + 1);
    } else if (kr > kl) {
        p = walk(t, n->right, kr, p, depth + 1);
        p = walk(t, n->left, kl, p, depth + 1);
    } else {
        p = walk(t, n->left, kl, p, depth + 1);
        p = walk(t, n->right, kr, p, depth + 1);
    }
    p += 1;                                         /* marker */
    if (g_nbm < 65536) { g_bms[g_nbm].bm = p; g_bms[g_nbm].K = K; g_bms[g_nbm].depth = depth; g_bms[g_nbm].node = id; g_nbm++; }
    return p + nb;
}

/* ---------- shared-table groups (per chunk x density bucket) ---------- */

typedef struct { uint8_t *lens; int first; int n; } grp_t;
static grp_t g_grp[32768]; static int g_ngrp;
typedef struct { uint8_t *cbuf; size_t clen; uint8_t *norm; size_t len; int fourx; } gsamp_t;
static gsamp_t g_gs[131072]; static int g_ngs; static size_t g_gs_bytes;
static uint64_t g_shr_groups, g_shr_members, g_shr_committed;

/* ---------- speed-bucket storage (normalized bitmap copies) ---------- */

typedef struct { uint8_t *data; uint8_t *raw; size_t len; int idx; } sample_t;
static sample_t g_samples[65536];
static int g_nsamples;
static size_t g_sample_bytes;

/* ---------- main ---------- */

/* ---------- within-tier grouping experiment (--order=...) ----------
 * The wire's within-tier tree order is symbol-value, so a value
 * permutation IS a grouping choice.  We relabel each length tier's
 * symbols onto that tier's own (ascending) value slots per heuristic,
 * remap the chunk, and run the normal pipeline on the remapped data. */
enum { ORD_ID, ORD_FREQDESC, ORD_RAND, ORD_CHAIN, ORD_SCHAIN, ORD_BISECT };
static int g_climb = 0;          /* --climb=N proposals per chunk */

/* ---- --wscan: symbol-width sweep over committed bitmaps ---- */
static int g_wscan = 0;
static uint64_t g_pm[5], g_pmbytes[5];  /* p_major bands: <.55 <.625 <.75 <.90 >=.90 */
#define NW 10
static const int WLIST[NW] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16};
static uint64_t w_tot[NW], w_oracle, w_wins[NW], w_rawwins, w_nscan;
static uint64_t w_ktot[NW], w_koracle, w_kwins[NW], w_krawwins;
/* per-bitmap KT records for the size/depth analysis */
static int wb_K[65536], wb_depth[65536], wb_n;
static uint32_t wb_kt[65536][NW];
static uint64_t w_mktot[6], w_phtot, w_alloracle;
static uint64_t w_bci[2], w_bch[2];   /* blkctx o=2,4: ideal vs huffman */
static uint64_t w_pmi[2], w_pmh[2];   /* pmkv o=2,4: ideal vs huffman */
static uint64_t q_sep[3], q_fus[3], q_traw, q_n;   /* quad-fusion probe */
/* oct (3-level fusion) probe, by parent depth (3 = >=3) */
static uint64_t od_n[4], od_all[4], od_raw[4];
static uint64_t od_sep[4][3], od_qchain[4][3], od_cquad[4][3], od_fus[4][2];
/* dynamic nibble-FSE (id 51) on packed quad stream vs separate bitmaps */
static uint64_t qnd_sep[4], qnd_fus[4], qnd_raw[4], qnd_n;
static uint64_t qd_sep[4][3], qd_fus[4][3], qd_raw[4], qd_n[4];  /* by parent depth */
/* root-stream vs fixed value-split comparison: H0 and kt(q=3) of the
 * 4-ary class stream, tree-root pairs vs byte bits 7-6 vs bits 1-0 */
static uint64_t rv_h0[3], rv_kt[3], rv_n[3];
static uint64_t qh_sep[4], qh_fus[4];   /* pure byte-huffman: 3 bitmaps vs fused */

static size_t h0_4ary(const uint8_t *syms, int n)
{
    uint32_t c[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) c[syms[i]]++;
    double bits = 0;
    for (int s = 0; s < 4; s++)
        if (c[s]) bits += (double)c[s] * -log2((double)c[s] / (double)n);
    return (size_t)((bits + 7.0) / 8.0) + 2;
}

/* KT price of an order-q model over a 4-ary symbol stream (fused
 * parent+child route pairs).  Context = previous q pair-symbols. */
static size_t kt4_price(const uint8_t *syms, int n, int q)
{
    int nctx = 1 << (2 * q);
    uint32_t c[64][4], tot[64];
    memset(c, 0, sizeof(uint32_t) * (size_t)nctx * 4);
    memset(tot, 0, sizeof(uint32_t) * (size_t)nctx);
    uint32_t ctx = 0, cmask = (uint32_t)nctx - 1;
    double bits = 0;
    for (int i = 0; i < n; i++) {
        int s = syms[i];
        bits -= log2(((double)c[ctx][s] + 0.5) / ((double)tot[ctx] + 2.0));
        c[ctx][s]++; tot[ctx]++;
        ctx = ((ctx << 2) | (uint32_t)s) & cmask;
    }
    return (size_t)((bits + 7.0) / 8.0) + 2;
}

/* KT price of an order-q model over an 8-ary symbol stream (fused
 * parent+child+grandchild route triples).  Context = previous q
 * oct-symbols (3q bits). */
static size_t kt8_price(const uint8_t *syms, int n, int q)
{
    int nctx = 1 << (3 * q);
    static uint32_t c[64][8], tot[64];
    memset(c, 0, sizeof(uint32_t) * (size_t)nctx * 8);
    memset(tot, 0, sizeof(uint32_t) * (size_t)nctx);
    uint32_t ctx = 0, cmask = (uint32_t)nctx - 1;
    double bits = 0;
    for (int i = 0; i < n; i++) {
        int sy = syms[i];
        bits -= log2(((double)c[ctx][sy] + 0.5) / ((double)tot[ctx] + 4.0));
        c[ctx][sy]++; tot[ctx]++;
        ctx = ((ctx << 3) | (uint32_t)sy) & cmask;
    }
    return (size_t)((bits + 7.0) / 8.0) + 2;
}
static uint64_t w_mwins[7], w_awins;   /* model wins vs best-W in the all-oracle */

/* carry-conditioned W=8 block model: one table per carry (last o bits
 * of previous byte, LSB-first so those are the byte's HIGH bits).
 * Prices the SAME model two ways: ideal (ANS, fractional bits) and
 * package-merge Huffman lengths (12-bit cap).  Both pay nt*6+16 bits
 * of table charge per context + tail bits + 2B framing. */
static void blkctx_price(const uint8_t *bm, int K, int o, size_t *ideal, size_t *huf)
{
    int ns = K / 8;
    *ideal = *huf = SIZE_MAX;
    if (ns < 8) return;
    int nctx = 1 << o;
    static uint32_t cnt[16][256];
    memset(cnt, 0, sizeof(uint32_t) * (size_t)nctx * 256);
    uint32_t carry = 0;
    for (int s = 0; s < ns; s++) {
        uint8_t sym = bm[s];
        cnt[carry][sym]++;
        carry = (uint32_t)sym >> (8 - o);
    }
    double ib = 0, hb = 0;
    for (int c = 0; c < nctx; c++) {
        uint64_t tc = 0; int nt = 0;
        for (int v = 0; v < 256; v++) { tc += cnt[c][v]; nt += cnt[c][v] > 0; }
        if (!tc) continue;
        double w[256]; uint8_t len[256];
        for (int v = 0; v < 256; v++)
            w[v] = cnt[c][v] ? (double)cnt[c][v] : 1e-12;
        pm_lengths(w, len);
        for (int v = 0; v < 256; v++) if (cnt[c][v]) {
            ib += (double)cnt[c][v] * log2((double)tc / (double)cnt[c][v]);
            hb += (double)cnt[c][v] * (double)len[v];
        }
        double charge = (double)nt * 6.0 + 16.0;
        ib += charge; hb += charge;
    }
    double tail = (double)(K - ns * 8);
    *ideal = (size_t)((ib + tail + 7.0) / 8.0) + 2;
    *huf   = (size_t)((hb + tail + 7.0) / 8.0) + 2;
}

/* PARAMETRIC per-carry tables (the mkv production design): estimate the
 * 2^o bit-context probs from the bitmap, derive each carry's 256-block
 * distribution by chain rule, price the payload (a) ideally under the
 * model (tANS expansion) and (b) with package-merge Huffman lengths
 * built on the MODEL distribution.  Transmitted: 2^o quantized probs
 * (charged 4 bits each + 16) — same charge both ways.  +2B framing. */
static void pmkv_price(const uint8_t *bm, int K, int o, size_t *ideal, size_t *huf)
{
    int ns = K / 8;
    *ideal = *huf = SIZE_MAX;
    if (ns < 8) return;
    int nctx = 1 << o; uint32_t cmask = (uint32_t)nctx - 1;
    uint32_t c[16][2]; memset(c, 0, sizeof(uint32_t) * (size_t)nctx * 2);
    uint32_t ctx = 0;
    for (int i = 0; i < ns * 8; i++) {
        int bit = (bm[i >> 3] >> (i & 7)) & 1;
        c[ctx][bit]++;
        ctx = ((ctx << 1) | (uint32_t)bit) & cmask;
    }
    double p1[16];
    double ib = 0;
    for (int cx = 0; cx < nctx; cx++) {
        double n0 = (double)c[cx][0], n1 = (double)c[cx][1];
        p1[cx] = (n1 + 0.5) / (n0 + n1 + 1.0);
        if (n1) ib -= n1 * log2(p1[cx]);
        if (n0) ib -= n0 * log2(1.0 - p1[cx]);
    }
    /* per-carry block counts (byte-aligned pass, same ctx convention) */
    static uint32_t cnt2[16][256];
    memset(cnt2, 0, sizeof(uint32_t) * (size_t)nctx * 256);
    ctx = 0;
    for (int s = 0; s < ns; s++) {
        cnt2[ctx][bm[s]]++;
        for (int j = 0; j < 8; j++)
            ctx = ((ctx << 1) | (uint32_t)((bm[s] >> j) & 1)) & cmask;
    }
    double hb = 0;
    for (int cx = 0; cx < nctx; cx++) {
        uint64_t tc = 0;
        for (int v = 0; v < 256; v++) tc += cnt2[cx][v];
        if (!tc) continue;
        double w[256]; uint8_t len[256];
        for (int v = 0; v < 256; v++) {           /* model prob of block v */
            double pr = 1.0; uint32_t cc = (uint32_t)cx;
            for (int j = 0; j < 8; j++) {
                int bit = (v >> j) & 1;
                pr *= bit ? p1[cc] : 1.0 - p1[cc];
                cc = ((cc << 1) | (uint32_t)bit) & cmask;
            }
            w[v] = pr > 1e-12 ? pr : 1e-12;
        }
        pm_lengths(w, len);
        for (int v = 0; v < 256; v++)
            if (cnt2[cx][v]) hb += (double)cnt2[cx][v] * (double)len[v];
    }
    double charge = (double)nctx * 4.0 + 16.0;
    double tail = (double)(K - ns * 8);
    *ideal = (size_t)((ib + charge + tail + 7.0) / 8.0) + 2;
    *huf   = (size_t)((hb + charge + tail + 7.0) / 8.0) + 2;
}

/* KT price of a parametric bit-context model over the bitmap:
 * ctx = previous `order` bits (mkv), or position mod s (phase).
 * Adaptive per-context binary KT, no transmitted params. +2B framing. */
static size_t bitctx_price(const uint8_t *bm, int K, int order, int s)
{
    uint32_t c[256][2]; memset(c, 0, sizeof(c));
    uint32_t ctx = 0, cmask = (1u << order) - 1;
    double bits = 0;
    for (int i = 0; i < K; i++) {
        int bit = (bm[i >> 3] >> (i & 7)) & 1;
        uint32_t cx = s ? (uint32_t)(i % s) : ctx;
        double n0 = (double)c[cx][0], n1 = (double)c[cx][1];
        double p = ((double)(bit ? n1 : n0) + 0.5) / (n0 + n1 + 1.0);
        bits -= log2(p);
        c[cx][bit]++;
        if (!s) ctx = ((ctx << 1) | (uint32_t)bit) & cmask;
    }
    return (size_t)((bits + 7.0) / 8.0) + 2;
}

/* two prices for an order-0 coder at width W over the K/W
 * non-overlapping W-bit symbols (LSB-first), leftover tail bits raw:
 *  - tab: empirical entropy + 6 bits per present symbol table charge
 *    + 16 bits header (optimistic for sparse wide alphabets: symbol
 *    identities not charged)
 *  - kt:  Krichevsky-Trofimov sequential cost, no table at all (the
 *    adaptive-coder / MDL-honest number; novel symbols pay ~W bits)
 * Both + 2B framing.  Capped at nb by the caller. */
static void wprice(const uint8_t *bm, int K, int W, size_t *tab, size_t *kt)
{
    int ns = K / W;
    *tab = *kt = SIZE_MAX;
    if (ns < 8) return;
    static uint32_t hist[1 << 16];
    static uint32_t touched[1 << 16];
    int nt = 0;
    uint32_t mask = (1u << W) - 1;
    double half_A = (double)(1u << (W - 1));
    double ktbits = 0;
    for (int s = 0; s < ns; s++) {
        int bit = s * W;
        uint64_t win;
        memcpy(&win, bm + (bit >> 3), 8);
        uint32_t sym = (uint32_t)(win >> (bit & 7)) & mask;
        ktbits -= log2(((double)hist[sym] + 0.5) / ((double)s + half_A));
        if (!hist[sym]) touched[nt++] = sym;
        hist[sym]++;
    }
    double bits = 0;
    for (int t = 0; t < nt; t++) {
        uint32_t c = hist[touched[t]];
        bits += (double)c * -log2((double)c / (double)ns);
        hist[touched[t]] = 0;
    }
    double tail = (double)(K - ns * W);    /* tail bits raw */
    bits += tail + (double)nt * 6.0 + 16.0;
    ktbits += tail;
    *tab = (size_t)((bits + 7.0) / 8.0) + 2;
    *kt = (size_t)((ktbits + 7.0) / 8.0) + 2;
}
static int g_pre = 0;            /* 0 none, 1 delta1, 2 stride-delta */
static int g_order = ORD_ID;

/* recursive balanced min-cut ordering over a symbol set, affinity A */
static void bisect_order(uint32_t A[256][256], const int *set, int cnt, int *out)
{
    if (cnt <= 2) { for (int i = 0; i < cnt; i++) out[i] = set[i]; return; }
    /* seeds: the pair among the first up-to-12 symbols with minimal affinity */
    int lim = cnt < 12 ? cnt : 12, s1 = set[0], s2 = set[1];
    uint64_t worst = UINT64_MAX;
    for (int i = 0; i < lim; i++)
        for (int j = i + 1; j < lim; j++) {
            uint64_t a = A[set[i]][set[j]];
            if (a < worst) { worst = a; s1 = set[i]; s2 = set[j]; }
        }
    int half = cnt / 2;
    int la[256], lb[256], na = 0, nb = 0;
    la[na++] = s1; lb[nb++] = s2;
    /* greedy: highest affinity-difference first */
    int rest[256], nr = 0;
    for (int i = 0; i < cnt; i++)
        if (set[i] != s1 && set[i] != s2) rest[nr++] = set[i];
    for (int r = 0; r < nr; r++) {
        int bi = -1; long long bd = -1;
        for (int i = 0; i < nr; i++) {
            if (rest[i] < 0) continue;
            long long aa = 0, ab = 0;
            for (int k = 0; k < na; k++) aa += A[rest[i]][la[k]];
            for (int k = 0; k < nb; k++) ab += A[rest[i]][lb[k]];
            long long d = aa > ab ? aa - ab : ab - aa;
            if (d > bd) { bd = d; bi = i; }
        }
        int v = rest[bi]; rest[bi] = -1;
        long long aa = 0, ab = 0;
        for (int k = 0; k < na; k++) aa += A[v][la[k]];
        for (int k = 0; k < nb; k++) ab += A[v][lb[k]];
        if ((aa >= ab && na < half) || nb >= cnt - half) la[na++] = v;
        else lb[nb++] = v;
    }
    bisect_order(A, la, na, out);
    bisect_order(A, lb, nb, out + na);
}

static void make_perm(const uint8_t *chunk, size_t n,
                      const uint64_t *freq, const uint8_t *lens,
                      uint8_t perm[256])
{
    for (int i = 0; i < 256; i++) perm[i] = (uint8_t)i;
    if (g_order == ORD_ID) return;
    static uint32_t A[256][256];
    if (g_order == ORD_CHAIN || g_order == ORD_SCHAIN || g_order == ORD_BISECT) {
        size_t lag = 1;
        if (g_order == ORD_SCHAIN || g_order == ORD_BISECT) {
            /* dominant stride: byte-equality autocorrelation over candidate lags */
            size_t bestc = 0;
            static const size_t lags[] = {1, 2, 3, 4, 6, 8, 16};
            for (unsigned li = 0; li < sizeof(lags) / sizeof(lags[0]); li++) {
                size_t L2 = lags[li], c = 0;
                for (size_t i = L2; i < n; i += 7) c += (chunk[i] == chunk[i - L2]);
                if (c > bestc) { bestc = c; lag = L2; }
            }
        }
        memset(A, 0, sizeof(A));
        for (size_t i = lag; i < n; i++) {
            A[chunk[i - lag]][chunk[i]] += 2; A[chunk[i]][chunk[i - lag]] += 2;
        }
        if (g_order == ORD_CHAIN)
            for (size_t i = 2; i < n; i++) { A[chunk[i - 2]][chunk[i]]++; A[chunk[i]][chunk[i - 2]]++; }
    }
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (int L = 1; L <= 15; L++) {
        int tier[256], tn = 0;
        for (int v = 0; v < 256; v++)
            if (freq[v] && lens[v] == L) tier[tn++] = v;
        if (tn < 3) continue;
        int order[256];
        if (g_order == ORD_FREQDESC) {
            memcpy(order, tier, sizeof(int) * (size_t)tn);
            for (int i = 1; i < tn; i++) {          /* insertion sort, freq desc */
                int x = order[i], j = i;
                while (j > 0 && freq[order[j - 1]] < freq[x]) { order[j] = order[j - 1]; j--; }
                order[j] = x;
            }
        } else if (g_order == ORD_RAND) {
            memcpy(order, tier, sizeof(int) * (size_t)tn);
            for (int i = tn - 1; i > 0; i--) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                int j = (int)(rng % (uint64_t)(i + 1));
                int t = order[i]; order[i] = order[j]; order[j] = t;
            }
        } else if (g_order == ORD_BISECT) {
            bisect_order(A, tier, tn, order);
        } else {                                     /* chain/schain: greedy affinity */
            int used[256] = {0};
            int cur = tier[0];                       /* start: heaviest */
            for (int i = 1; i < tn; i++) if (freq[tier[i]] > freq[cur]) cur = tier[i];
            order[0] = cur; used[cur] = 1;
            for (int k = 1; k < tn; k++) {
                int best = -1; uint64_t ba = 0;
                for (int i = 0; i < tn; i++) {
                    int v = tier[i];
                    if (used[v]) continue;
                    uint64_t a = A[cur][v];
                    if (best < 0 || a > ba || (a == ba && freq[v] > freq[best])) { best = v; ba = a; }
                }
                order[k] = best; used[best] = 1; cur = best;
            }
        }
        /* assign heuristic order onto the tier's ascending value slots */
        for (int k = 0; k < tn; k++) perm[order[k]] = (uint8_t)tier[k];
    }
}

/* entropy-priced chunk cost: encode remapped chunk with forced lengths,
 * walk, price each merge bitmap at empirical byte entropy + 12B table
 * charge (capped at raw), flats + small bitmaps raw.  Returns SIZE_MAX
 * on failure. */
static size_t climb_cost(pivco_encoder_t *enc, const uint8_t *chunk, size_t n,
                         const uint8_t *lens)
{
    static pivco_table_t ct;
    static uint8_t ebuf[PIVCO_MAX_ENCODED_SIZE];
    pivco_cfg_t cfg = pivco_cfg_default; cfg.fse_enabled = 0;
    /* the builder CRASHES (PIVCO_CHECK) on non-Kraft-tight lens; verify
     * sum(2^-len) == 1 exactly before handing it anything */
    uint64_t ksum = 0;
    for (int v = 0; v < 256; v++)
        if (lens[v]) ksum += 1ull << (24 - lens[v]);
    if (ksum != 1ull << 24) return SIZE_MAX;
    if (pivco_build_table_from_code_lens(&cfg, lens, &ct) != PIVCO_OK) return SIZE_MAX;
    size_t elen = 0;
    if (pivco_encode(enc, &ct, chunk, n, ebuf, &elen) != PIVCO_OK) return SIZE_MAX;
    if (ct.tree[ct.tree_root].symbol >= 0) return SIZE_MAX;
    g_nbm = 0; g_flat_bytes = 0;
    const uint8_t *q = ebuf + 2;
    if ((size_t)(walk(&ct, ct.tree_root, (int)n, q, 0) - ebuf) != elen) return SIZE_MAX;
    double bits = 0;
    size_t total = g_flat_bytes;
    for (int b = 0; b < g_nbm; b++) {
        int nb = (g_bms[b].K + 7) >> 3;
        if (nb < MINB) { total += (size_t)nb; continue; }
        unsigned cnt[256] = {0};
        for (int i = 0; i < nb; i++) cnt[g_bms[b].bm[i]]++;
        bits = 0;
        for (int v = 0; v < 256; v++)
            if (cnt[v]) bits += (double)cnt[v] * -log2((double)cnt[v] / (double)nb);
        size_t hb = (size_t)(bits / 8.0) + 12;
        total += hb < (size_t)nb ? hb : (size_t)nb;
    }
    return total;
}

/* hill-climb per chunk: moves = same-tier value swaps (ordering) and
 * length swaps between two used symbols (co-tiering).  Greedy accept.
 * Returns the climbed (perm, lens) in place. */
static void climb_chunk(pivco_encoder_t *enc, const uint8_t *orig, size_t n,
                        uint8_t perm[256], uint8_t lens[256], int iters)
{
    static uint8_t buf[65536];
    int used[256], nu = 0;
    for (int v = 0; v < 256; v++) if (lens[v]) used[nu++] = v;
    if (nu < 3) return;
    for (size_t i = 0; i < n; i++) buf[i] = perm[orig[i]];
    size_t best = climb_cost(enc, buf, n, lens);
    if (best == SIZE_MAX) return;
    uint64_t rng = 0xC0FFEE123456789ull;
    for (int it = 0; it < iters; it++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        int a = used[rng % (uint64_t)nu];
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        int b = used[rng % (uint64_t)nu];
        if (a == b) continue;
        /* move mix: 0-4 ordering swap, 5-7 co-tier len swap,
         * 8 Kraft merge {l,l+2}->{l+1,l+1}, 9 Kraft split {l,l,l}->{l-1,l+1,l+1} */
        int mv = (int)((rng >> 32) % 10);
        uint8_t la = lens[a], lb = lens[b];
        int pa = -1, pb = -1, c3 = -1; uint8_t lc = 0;
        if (mv < 5) {
            if (la != lb) continue;
            /* swap the data-labels: find preimages of a and b under perm */
            for (int v = 0; v < 256; v++) {
                if (perm[v] == a) pa = v;
                if (perm[v] == b) pb = v;
            }
            uint8_t t = perm[pa]; perm[pa] = perm[pb]; perm[pb] = t;
        } else if (mv < 8) {
            if (la == lb) continue;
            lens[a] = lb; lens[b] = la;
        } else if (mv == 8) {
            /* merge: {la, la+2, la+2} -> {la+1, la+1, la+1} (Kraft-exact) */
            if (la + 2 > PIVCO_MAX_CODE_LEN) continue;
            b = -1; c3 = -1;
            unsigned st = (unsigned)(rng % (uint64_t)nu);
            for (int k = 0; k < nu; k++) {
                int v = used[(st + k) % nu];
                if (v == a || lens[v] != la + 2) continue;
                if (b < 0) b = v; else { c3 = v; break; }
            }
            if (c3 < 0) continue;
            lb = lens[b]; lc = lens[c3];
            lens[a] = (uint8_t)(la + 1);
            lens[b] = (uint8_t)(la + 1); lens[c3] = (uint8_t)(la + 1);
        } else {
            /* split: a,b,c all at l; a -> l-1, b,c -> l+1 (Kraft-exact) */
            if (la < 2 || la + 1 > PIVCO_MAX_CODE_LEN) continue;
            b = -1;
            unsigned st = (unsigned)(rng % (uint64_t)nu);
            for (int k = 0; k < nu; k++) {
                int v = used[(st + k) % nu];
                if (v == a || lens[v] != la) continue;
                if (b < 0) b = v; else { c3 = v; break; }
            }
            if (c3 < 0) continue;
            lb = lens[b]; lc = lens[c3];
            lens[a] = (uint8_t)(la - 1);
            lens[b] = (uint8_t)(la + 1); lens[c3] = (uint8_t)(la + 1);
        }
        for (size_t i = 0; i < n; i++) buf[i] = perm[orig[i]];
        size_t c = climb_cost(enc, buf, n, lens);
        if (c < best) best = c;
        else {          /* revert */
            if (mv < 5) { uint8_t t = perm[pa]; perm[pa] = perm[pb]; perm[pb] = t; }
            else { lens[a] = la; lens[b] = lb; if (c3 >= 0) lens[c3] = lc; }
        }
    }
}

int main(int argc, char **argv)
{
    pivco_fse_init();
    for (int i = 1; i <= PIVCO_FSE_NUM_TABLES; i++) build_huftab(i);

    pivco_encoder_t *enc = pivco_encoder_create();
    static uint8_t encbuf[PIVCO_MAX_ENCODED_SIZE];
    static pivco_table_t table;

    printf("%-18s %9s | %6s %6s %6s %6s %6s %6s   (commit-if-smaller vs raw; PH0 = nested PH w/o table cost; huf0/shHu pay tables in-stream; shHu = one table per chunk x density bucket, commit iff >=0.5%% bucket saving)\n",
           "file", "raw", "sFSE", "bHuf", "dANS", "huf0", "PH0", "shHu");

    int argstart = 1;
    if (argc > 1 && !strncmp(argv[1], "--order=", 8)) {
        const char *m = argv[1] + 8;
        g_order = !strcmp(m, "freqdesc") ? ORD_FREQDESC
                : !strcmp(m, "rand")     ? ORD_RAND
                : !strcmp(m, "chain")    ? ORD_CHAIN
                : !strcmp(m, "schain")   ? ORD_SCHAIN
                : !strcmp(m, "bisect")   ? ORD_BISECT : ORD_ID;
        argstart = 2;
    }
    while (argc > argstart && !strncmp(argv[argstart], "--climb=", 8)) {
        g_climb = atoi(argv[argstart] + 8); argstart++;
    }
    while (argc > argstart && !strncmp(argv[argstart], "--pre=", 6)) {
        g_pre = !strcmp(argv[argstart] + 6, "delta1") ? 1
              : !strcmp(argv[argstart] + 6, "dstride") ? 2 : 0;
        argstart++;
    }
    while (argc > argstart && !strcmp(argv[argstart], "--wscan")) {
        g_wscan = 1; argstart++;
    }
    for (int a = argstart; a < argc; a++) {
        struct stat st;
        if (stat(argv[a], &st)) { fprintf(stderr, "stat %s\n", argv[a]); continue; }
        uint8_t *data = malloc((size_t)st.st_size);
        FILE *f = fopen(argv[a], "rb");
        if (fread(data, 1, (size_t)st.st_size, f) != (size_t)st.st_size) return 1;
        fclose(f);

        uint64_t t_raw = 0, t_sfse = 0, t_bhuf = 0, t_dans = 0, t_huf0 = 0, t_ph = 0, t_shr = 0;
        memset(g_pm, 0, sizeof(g_pm)); memset(g_pmbytes, 0, sizeof(g_pmbytes));
        memset(w_tot, 0, sizeof(w_tot)); memset(w_wins, 0, sizeof(w_wins));
        memset(w_ktot, 0, sizeof(w_ktot)); memset(w_kwins, 0, sizeof(w_kwins));
        w_oracle = 0; w_rawwins = 0; w_nscan = 0; w_koracle = 0; w_krawwins = 0;
        wb_n = 0;
        memset(w_mktot, 0, sizeof(w_mktot)); w_phtot = 0; w_alloracle = 0;
        memset(w_mwins, 0, sizeof(w_mwins)); w_awins = 0;
        memset(w_bci, 0, sizeof(w_bci)); memset(w_bch, 0, sizeof(w_bch));
        memset(w_pmi, 0, sizeof(w_pmi)); memset(w_pmh, 0, sizeof(w_pmh));
        memset(q_sep, 0, sizeof(q_sep)); memset(q_fus, 0, sizeof(q_fus));
        q_traw = 0; q_n = 0;
        memset(qd_sep, 0, sizeof(qd_sep)); memset(qd_fus, 0, sizeof(qd_fus));
        memset(qd_raw, 0, sizeof(qd_raw)); memset(qd_n, 0, sizeof(qd_n));
        memset(rv_h0, 0, sizeof(rv_h0)); memset(rv_kt, 0, sizeof(rv_kt));
        memset(rv_n, 0, sizeof(rv_n));
        memset(qh_sep, 0, sizeof(qh_sep)); memset(qh_fus, 0, sizeof(qh_fus));
        for (size_t off = 0; off < (size_t)st.st_size; off += CHUNK) {
            size_t n = (size_t)st.st_size - off; if (n > CHUNK) n = CHUNK;
            uint64_t freq[256] = {0};
            for (size_t i = 0; i < n; i++) freq[data[off + i]]++;
            pivco_cfg_t cfg = pivco_cfg_default;
            cfg.fse_enabled = 0;
            static uint8_t remap[65536];
            static uint8_t pre[65536];
            const uint8_t *chunk = data + off;
            if (g_pre) {
                size_t lag = 1;
                if (g_pre == 2) {
                    size_t bestc = 0;
                    static const size_t lags[] = {1, 2, 3, 4, 6, 8, 16};
                    for (unsigned li = 0; li < 7; li++) {
                        size_t L2 = lags[li], c = 0;
                        for (size_t i = L2; i < n; i += 7) c += (chunk[i] == chunk[i - L2]);
                        if (c > bestc) { bestc = c; lag = L2; }
                    }
                }
                for (size_t i = 0; i < n; i++)
                    pre[i] = (uint8_t)(chunk[i] - (i >= lag ? chunk[i - lag] : 0));
                chunk = pre;
                memset(freq, 0, sizeof(uint64_t) * 256);
                for (size_t i = 0; i < n; i++) freq[chunk[i]]++;
            }
            if (g_order != ORD_ID) {
                if (pivco_build_table(&cfg, freq, &table) != PIVCO_OK) continue;
                uint8_t perm[256];
                make_perm(chunk, n, freq, table.code_len, perm);
                for (size_t i = 0; i < n; i++) remap[i] = perm[chunk[i]];
                chunk = remap;
                memset(freq, 0, sizeof(freq));
                for (size_t i = 0; i < n; i++) freq[chunk[i]]++;
            }
            if (pivco_build_table(&cfg, freq, &table) != PIVCO_OK) continue;
            if (g_climb > 0) {
                uint8_t cperm[256], clens[256];
                for (int v = 0; v < 256; v++) cperm[v] = (uint8_t)v;
                memcpy(clens, table.code_len, 256);
                climb_chunk(enc, chunk, n, cperm, clens, g_climb);
                static uint8_t climbed[65536];
                for (size_t i = 0; i < n; i++) climbed[i] = cperm[chunk[i]];
                memcpy(remap, climbed, n); chunk = remap;
                if (pivco_build_table_from_code_lens(&cfg, clens, &table) != PIVCO_OK) continue;
            }
            size_t elen = 0;
            if (pivco_encode(enc, &table, chunk, n, encbuf, &elen) != PIVCO_OK) continue;
            if (table.tree[table.tree_root].symbol >= 0) continue;  /* constant chunk */

            g_nbm = 0; g_flat_bytes = 0;
            const uint8_t *p = encbuf;
            int N = p[0] | (p[1] << 8); p += 2;
            (void)N;
            const uint8_t *end = walk(&table, table.tree_root, (int)n, p, 0);
            if ((size_t)(end - encbuf) != elen) {
                fprintf(stderr, "WIRE WALK MISMATCH: consumed %zu of %zu (chunk @%zu)\n",
                        (size_t)(end - encbuf), elen, off);
                return 1;
            }

            t_raw += g_flat_bytes; t_sfse += g_flat_bytes; t_bhuf += g_flat_bytes;
            t_dans += g_flat_bytes; t_huf0 += g_flat_bytes; t_ph += g_flat_bytes;
            t_shr += g_flat_bytes;
            for (int wi = 0; wi < NW; wi++) { w_tot[wi] += g_flat_bytes; w_ktot[wi] += g_flat_bytes; }
            w_oracle += g_flat_bytes; w_koracle += g_flat_bytes;
            for (int oi = 0; oi < 6; oi++) w_mktot[oi] += g_flat_bytes;
            w_phtot += g_flat_bytes; w_alloracle += g_flat_bytes;
            for (int oi = 0; oi < 2; oi++) { w_bci[oi] += g_flat_bytes; w_bch[oi] += g_flat_bytes; w_pmi[oi] += g_flat_bytes; w_pmh[oi] += g_flat_bytes; }
            static uint8_t *ch_norm[65536]; static int ch_len[65536]; static int ch_idx[65536];
            int ch_n = 0;
            for (int b = 0; b < g_nbm; b++) {
                int K = g_bms[b].K, nb = (K + 7) >> 3;
                if (nb < MINB) { t_raw += (uint64_t)nb; t_sfse += (uint64_t)nb; t_bhuf += (uint64_t)nb; t_dans += (uint64_t)nb; t_huf0 += (uint64_t)nb; t_ph += (uint64_t)nb; t_shr += (uint64_t)nb; for (int wi = 0; wi < NW; wi++) { w_tot[wi] += (uint64_t)nb; w_ktot[wi] += (uint64_t)nb; } w_oracle += (uint64_t)nb; w_koracle += (uint64_t)nb; for (int oi = 0; oi < 6; oi++) w_mktot[oi] += (uint64_t)nb; w_phtot += (uint64_t)nb; w_alloracle += (uint64_t)nb; for (int oi = 0; oi < 2; oi++) { w_bci[oi] += (uint64_t)nb; w_bch[oi] += (uint64_t)nb; w_pmi[oi] += (uint64_t)nb; w_pmh[oi] += (uint64_t)nb; } continue; }
                const uint8_t *bm = g_bms[b].bm;
                int ones = 0;
                for (int i = 0; i < nb - 1; i++) ones += __builtin_popcount(bm[i]);
                for (int i = (nb - 1) * 8; i < K; i++) ones += (bm[i >> 3] >> (i & 7)) & 1;
                int flip = 2 * ones > K;
                double p_major = (double)(flip ? ones : K - ones) / (double)K;
                int idx = pivco_fse_select_table(p_major);
                g_pm[p_major < 0.55 ? 0 : p_major < 0.625 ? 1 :
                     p_major < 0.75 ? 2 : p_major < 0.90 ? 3 : 4]++;
                g_pmbytes[p_major < 0.55 ? 0 : p_major < 0.625 ? 1 :
                     p_major < 0.75 ? 2 : p_major < 0.90 ? 3 : 4] += (uint64_t)nb;
                uint8_t norm[CHUNK / 8 + 16];
                if (flip) for (int i = 0; i < nb; i++) norm[i] = (uint8_t)~bm[i];
                else memcpy(norm, bm, (size_t)nb);
                /* mask tail bits of the last byte so coders see zeros there */
                if (K & 7) norm[nb - 1] &= (uint8_t)((1u << (K & 7)) - 1);

                t_raw += (uint64_t)nb;
                /* static FSE */
                uint8_t out[CHUNK / 8 + 64]; size_t olen = 0;
                size_t best_s = (size_t)nb;
                if (idx >= 1 &&
                    pivco_fse_compress(idx, norm, (size_t)nb, out, sizeof(out), &olen) == PIVCO_FSE_OK
                    && olen + 2 < (size_t)nb)
                    best_s = olen + 2;
                t_sfse += best_s;
                /* dynamic nibble FSE (original, unflipped, like the PR) */
                size_t best_d = (size_t)nb;
                if (pivco_fse_compress(PIVCO_FSE_DYNAMIC_ID, bm, (size_t)nb, out, sizeof(out), &olen) == PIVCO_FSE_OK
                    && olen + 2 < (size_t)nb)
                    best_d = olen + 2;
                t_dans += best_d;
                /* byte huffman */
                size_t hb = (size_t)((huf_bits(&g_huf[idx], norm, (size_t)nb) + 7) >> 3);
                size_t best_h = (hb + 2 < (size_t)nb) ? hb + 2 : (size_t)nb;
                t_bhuf += best_h;
                /* huf0: dynamic per-bitmap Huffman table, transmitted in-stream */
                size_t best_u = (size_t)nb;
                {
                    size_t cs = HUF_compress(out, sizeof(out), bm, (size_t)nb);
                    if (cs > 0 && !HUF_isError(cs) && cs + 2 < (size_t)nb) best_u = cs + 2;
                }
                t_huf0 += best_u;
                /* nested PH on the bitmap bytes (NO table-transmission cost) */
                size_t best_p = (size_t)nb;
                {
                    uint64_t bfreq[256] = {0};
                    for (int i = 0; i < nb; i++) bfreq[bm[i]]++;
                    static pivco_table_t bt;
                    pivco_cfg_t bcfg = pivco_cfg_default; bcfg.fse_enabled = 0;
                    size_t pl = 0;
                    static uint8_t pout[PIVCO_MAX_ENCODED_SIZE];
                    if (pivco_build_table(&bcfg, bfreq, &bt) == PIVCO_OK &&
                        bt.tree[bt.tree_root].symbol < 0 &&
                        pivco_encode(enc, &bt, bm, (size_t)nb, pout, &pl) == PIVCO_OK &&
                        pl + 2 < (size_t)nb)
                        best_p = pl + 2;
                }
                t_ph += best_p;
                if (g_wscan) {
                    size_t bestw = (size_t)nb; int besti = -1;
                    size_t bestk = (size_t)nb; int bestki = -1;
                    for (int wi = 0; wi < NW; wi++) {
                        size_t c, ck;
                        wprice(bm, K, WLIST[wi], &c, &ck);
                        if (c > (size_t)nb) c = (size_t)nb;
                        if (ck > (size_t)nb) ck = (size_t)nb;
                        w_tot[wi] += c; w_ktot[wi] += ck;
                        if (c < bestw) { bestw = c; besti = wi; }
                        if (ck < bestk) { bestk = ck; bestki = wi; }
                        if (wb_n < 65536) wb_kt[wb_n][wi] = (uint32_t)ck;
                    }
                    if (wb_n < 65536) {
                        wb_K[wb_n] = K; wb_depth[wb_n] = g_bms[b].depth; wb_n++;
                    }
                    static const int ORD[6] = {1, 2, 3, 4, 6, 8};
                    size_t mk[6];
                    for (int oi = 0; oi < 6; oi++) {
                        mk[oi] = bitctx_price(bm, K, ORD[oi], 0);
                        if (mk[oi] > (size_t)nb) mk[oi] = (size_t)nb;
                        w_mktot[oi] += mk[oi];
                    }
                    for (int oi = 0; oi < 2; oi++) {
                        size_t bi, bh;
                        blkctx_price(bm, K, oi ? 4 : 2, &bi, &bh);
                        if (bi > (size_t)nb) bi = (size_t)nb;
                        if (bh > (size_t)nb) bh = (size_t)nb;
                        w_bci[oi] += bi; w_bch[oi] += bh;
                        pmkv_price(bm, K, oi ? 4 : 2, &bi, &bh);
                        if (bi > (size_t)nb) bi = (size_t)nb;
                        if (bh > (size_t)nb) bh = (size_t)nb;
                        w_pmi[oi] += bi; w_pmh[oi] += bh;
                    }
                    size_t ph = (size_t)nb;
                    static const int SL[5] = {2, 3, 4, 6, 8};
                    for (int si = 0; si < 5; si++) {
                        size_t c = bitctx_price(bm, K, 0, SL[si]);
                        if (c < ph) ph = c;
                    }
                    w_phtot += ph;
                    size_t ao = bestk; int aw = -1;   /* -1=W, 0..5=mkv, 6=ph */
                    for (int oi = 0; oi < 6; oi++)
                        if (mk[oi] < ao) { ao = mk[oi]; aw = oi; }
                    if (ph < ao) { ao = ph; aw = 6; }
                    w_alloracle += ao;
                    if (aw < 0) w_awins++; else w_mwins[aw]++;
                    w_oracle += bestw; w_koracle += bestk; w_nscan++;
                    if (besti < 0) w_rawwins++; else w_wins[besti]++;
                    if (bestki < 0) w_krawwins++; else w_kwins[bestki]++;
                }
                if (ch_n < 65536) {
                    ch_norm[ch_n] = malloc((size_t)nb); memcpy(ch_norm[ch_n], norm, (size_t)nb);
                    ch_len[ch_n] = nb; ch_idx[ch_n] = idx; ch_n++;
                }
                /* keep a copy for the speed phase when byte-huf commits */
                if (best_h < (size_t)nb && g_nsamples < 65536 && g_sample_bytes < (48u << 20)) {
                    sample_t *s = &g_samples[g_nsamples++];
                    s->data = malloc((size_t)nb); memcpy(s->data, norm, (size_t)nb);
                    s->raw = malloc((size_t)nb); memcpy(s->raw, bm, (size_t)nb);
                    s->len = (size_t)nb; s->idx = idx;
                    g_sample_bytes += (size_t)nb;
                }
            }
            /* ---- quad-fusion probe: nodes whose both children are
             *      internal with bitmaps.  Fused 4-ary route stream vs
             *      the three bitmaps priced separately, matched context
             *      memory (o bits vs q=o/2 pair-symbols). ---- */
            if (g_wscan) {
                static int idx_of[1024];
                memset(idx_of, -1, sizeof(idx_of));
                for (int b = 0; b < g_nbm; b++)
                    if (g_bms[b].node >= 0 && g_bms[b].node < 1024) idx_of[g_bms[b].node] = b;
                for (int b = 0; b < g_nbm; b++) {
                    int16_t nd = g_bms[b].node;
                    const pivco_tree_node_t *tn = &table.tree[nd];
                    if (tn->symbol >= 0) continue;
                    int li = idx_of[tn->left], ri = idx_of[tn->right];
                    if (li < 0 || ri < 0) continue;
                    int K = g_bms[b].K, kl = g_bms[li].K, kr = g_bms[ri].K;
                    if (kl + kr != K || (K >> 3) < MINB) continue;
                    const uint8_t *pb = g_bms[b].bm, *lb = g_bms[li].bm, *rb = g_bms[ri].bm;
                    int ones = 0;
                    for (int i = 0; i < K; i++) ones += (pb[i >> 3] >> (i & 7)) & 1;
                    int inv;                       /* bit sense: 1 = right? */
                    if (ones == kr) inv = 0; else if (ones == kl) inv = 1; else continue;
                    static uint8_t qs[65536];
                    int z = 0, on = 0;
                    for (int i = 0; i < K; i++) {
                        int b1 = ((pb[i >> 3] >> (i & 7)) & 1) ^ inv;
                        int b2 = b1 ? (rb[on >> 3] >> (on & 7)) & 1
                                    : (lb[z >> 3] >> (z & 7)) & 1;
                        if (b1) on++; else z++;
                        qs[i] = (uint8_t)((b1 << 1) | b2);
                    }
                    uint64_t traw = (uint64_t)(((K + 7) >> 3) + ((kl + 7) >> 3) + ((kr + 7) >> 3));
                    int db = g_bms[b].depth > 3 ? 3 : g_bms[b].depth;
                    if (nd == table.tree_root) {
                        rv_h0[0] += h0_4ary(qs, K);
                        rv_kt[0] += kt4_price(qs, K, 3);
                        rv_n[0]++;
                    }
                    q_traw += traw; q_n++;
                    qd_raw[db] += traw; qd_n[db]++;
                    static const int QO[3] = {2, 4, 6};
                    for (int oi = 0; oi < 3; oi++) {
                        size_t s = bitctx_price(pb, K, QO[oi], 0)
                                 + bitctx_price(lb, kl, QO[oi], 0)
                                 + bitctx_price(rb, kr, QO[oi], 0);
                        size_t f = kt4_price(qs, K, oi + 1);
                        q_sep[oi] += s; q_fus[oi] += f;
                        qd_sep[db][oi] += s; qd_fus[db][oi] += f;
                    }
                    /* pure byte-huffman arm: huf0 on each object, commit-if-smaller */
                    {
                        static uint8_t qpk[16384], qout[24576];
                        const uint8_t *sbm[3] = {pb, lb, rb};
                        int slen[3] = {(K + 7) >> 3, (kl + 7) >> 3, (kr + 7) >> 3};
                        size_t sh = 0;
                        for (int t = 0; t < 3; t++) {
                            size_t cs = HUF_compress(qout, sizeof(qout), sbm[t], (size_t)slen[t]);
                            sh += (cs > 0 && !HUF_isError(cs) && cs + 2 < (size_t)slen[t])
                                  ? cs + 2 : (size_t)slen[t];
                        }
                        int flen = (2 * K + 7) >> 3;
                        memset(qpk, 0, (size_t)flen);
                        for (int i = 0; i < K; i++)
                            qpk[i >> 2] |= (uint8_t)(qs[i] << ((i & 3) * 2));
                        size_t cs = HUF_compress(qout, sizeof(qout), qpk, (size_t)flen);
                        size_t fh = (cs > 0 && !HUF_isError(cs) && cs + 2 < (size_t)flen)
                                    ? cs + 2 : (size_t)flen;
                        qh_sep[db] += sh; qh_fus[db] += fh;
                    }
                    /* ---- dynamic nibble-FSE (id 51's coder): packed quad
                     *      stream vs the three bitmaps separately,
                     *      commit-if-smaller vs raw on each object ---- */
                    {
                        static uint8_t qpk[16384], dout[40960];
                        const uint8_t *sbm[3] = {pb, lb, rb};
                        int slen[3] = {(K + 7) >> 3, (kl + 7) >> 3, (kr + 7) >> 3};
                        size_t sep = 0;
                        for (int t = 0; t < 3; t++) {
                            size_t dl = 0;
                            sep += (pivco_fse_compress_dynamic(sbm[t], (size_t)slen[t],
                                        dout, sizeof(dout), &dl) == PIVCO_FSE_OK
                                    && dl + 2 < (size_t)slen[t]) ? dl + 2 : (size_t)slen[t];
                        }
                        int flen = (2 * K + 7) >> 3;
                        memset(qpk, 0, (size_t)flen);
                        for (int i = 0; i < K; i++)
                            qpk[i >> 2] |= (uint8_t)(qs[i] << ((i & 3) * 2));
                        size_t dl = 0;
                        size_t fus = (pivco_fse_compress_dynamic(qpk, (size_t)flen,
                                          dout, sizeof(dout), &dl) == PIVCO_FSE_OK
                                      && dl + 2 < (size_t)flen) ? dl + 2 : (size_t)flen;
                        qnd_sep[db] += sep; qnd_fus[db] += fus;
                        qnd_raw[db] += traw; qnd_n++;
                    }
                    /* ---- 3-level (oct) fusion: parent + children +
                     *      grandchildren as one 8-ary route stream.
                     *      Requires all four grandchildren internal with
                     *      harvested bitmaps. ---- */
                    od_all[db]++;
                    {
                        const pivco_tree_node_t *ln = &table.tree[tn->left];
                        const pivco_tree_node_t *rn = &table.tree[tn->right];
                        int gid[4] = {ln->left, ln->right, rn->left, rn->right};
                        int gi[4]; int ok = 1;
                        for (int t = 0; t < 4; t++) {
                            gi[t] = (gid[t] >= 0 && gid[t] < 1024) ? idx_of[gid[t]] : -1;
                            if (gi[t] < 0 || table.tree[gid[t]].symbol >= 0) ok = 0;
                        }
                        int kgc[4], linv = 0, rinv = 0;
                        if (ok) {
                            for (int t = 0; t < 4; t++) kgc[t] = g_bms[gi[t]].K;
                            if (kgc[0] + kgc[1] != kl || kgc[2] + kgc[3] != kr) ok = 0;
                        }
                        if (ok) {
                            int ol = 0, orr = 0;
                            for (int i = 0; i < kl; i++) ol  += (lb[i >> 3] >> (i & 7)) & 1;
                            for (int i = 0; i < kr; i++) orr += (rb[i >> 3] >> (i & 7)) & 1;
                            if (ol == kgc[1]) linv = 0; else if (ol == kgc[0]) linv = 1; else ok = 0;
                            if (orr == kgc[3]) rinv = 0; else if (orr == kgc[2]) rinv = 1; else ok = 0;
                        }
                        if (ok) {
                            static uint8_t os8[65536], cqL[65536], cqR[65536];
                            int zz = 0, oo = 0, gcur[4] = {0, 0, 0, 0};
                            for (int i = 0; i < K; i++) {
                                int b1 = ((pb[i >> 3] >> (i & 7)) & 1) ^ inv;
                                int pos2 = b1 ? oo : zz;
                                const uint8_t *cb = b1 ? rb : lb;
                                int b2 = ((cb[pos2 >> 3] >> (pos2 & 7)) & 1) ^ (b1 ? rinv : linv);
                                int g = (b1 << 1) | b2;
                                const uint8_t *gb = g_bms[gi[g]].bm;
                                int pos3 = gcur[g]++;
                                int b3 = (gb[pos3 >> 3] >> (pos3 & 7)) & 1;
                                os8[i] = (uint8_t)((b1 << 2) | (b2 << 1) | b3);
                                if (b1) { cqR[oo] = (uint8_t)((b2 << 1) | b3); oo++; }
                                else    { cqL[zz] = (uint8_t)((b2 << 1) | b3); zz++; }
                            }
                            uint64_t graw = 0;
                            for (int t = 0; t < 4; t++) graw += (uint64_t)((kgc[t] + 7) >> 3);
                            od_raw[db] += traw + graw; od_n[db]++;
                            static const int QO2[3] = {2, 4, 6};
                            for (int oi = 0; oi < 3; oi++) {
                                int o = QO2[oi];
                                size_t gc = 0;
                                for (int t = 0; t < 4; t++)
                                    gc += bitctx_price(g_bms[gi[t]].bm, kgc[t], o, 0);
                                od_sep[db][oi]    += bitctx_price(pb, K, o, 0)
                                                   + bitctx_price(lb, kl, o, 0)
                                                   + bitctx_price(rb, kr, o, 0) + gc;
                                od_qchain[db][oi] += kt4_price(qs, K, oi + 1) + gc;
                                od_cquad[db][oi]  += bitctx_price(pb, K, o, 0)
                                                   + kt4_price(cqL, kl, oi + 1)
                                                   + kt4_price(cqR, kr, oi + 1);
                            }
                            od_fus[db][0] += kt8_price(os8, K, 1);
                            od_fus[db][1] += kt8_price(os8, K, 2);
                        }
                    }
                }
            }
            if (g_wscan) {          /* fixed value-split class streams */
                static uint8_t vs[65536];
                for (size_t i = 0; i < n; i++) vs[i] = chunk[i] >> 6;
                rv_h0[1] += h0_4ary(vs, (int)n); rv_kt[1] += kt4_price(vs, (int)n, 3); rv_n[1]++;
                for (size_t i = 0; i < n; i++) vs[i] = chunk[i] & 3;
                rv_h0[2] += h0_4ary(vs, (int)n); rv_kt[2] += kt4_price(vs, (int)n, 3); rv_n[2]++;
            }
            /* ---- shared-table pass: one dynamic byte-huffman table per
             *      (chunk, coarse density class), committed only if it saves
             *      >= 0.5% of the class's bytes.  6 log-spaced classes over
             *      the 50-entry density schedule. ---- */
#define DCLASS(idx) ((idx) <= 4 ? 0 : (idx) <= 10 ? 1 : (idx) <= 18 ? 2 : (idx) <= 28 ? 3 : (idx) <= 38 ? 4 : 5)
            for (int bucket = 0; bucket < 6; bucket++) {
                unsigned cnt[256] = {0};
                size_t braw = 0; int nmem = 0;
                for (int i = 0; i < ch_n; i++)
                    if (DCLASS(ch_idx[i]) == bucket) {
                        for (int k = 0; k < ch_len[i]; k++) cnt[ch_norm[i][k]]++;
                        braw += (size_t)ch_len[i]; nmem++;
                    }
                if (!nmem) continue;
                g_shr_groups++; g_shr_members += (uint64_t)nmem;
                HUF_CREATE_STATIC_CTABLE(celt, 255); uint8_t tab[512];
                unsigned tlog = HUF_optimalTableLog(11, braw, 255);
                size_t mb = HUF_buildCTable(celt, cnt, 255, tlog);
                size_t tlen = (HUF_isError(mb) || mb == 0) ? 0
                            : HUF_writeCTable(tab, sizeof(tab), celt, 255, (unsigned)mb);
                if (!tlen || HUF_isError(tlen)) tlen = 24;   /* conservative charge */
                double wts[256]; huftab_t *ht = malloc(sizeof(huftab_t));
                for (int b = 0; b < 256; b++)
                    wts[b] = cnt[b] ? (double)cnt[b] : 1e-12;
                build_huftab_w(ht, wts);
                size_t cand = tlen;
                static uint8_t cb[65536 / 8 + 512];
                static size_t clens[65536]; static int fourx[65536]; int m = 0;
                static uint8_t *cbufs[65536];
                for (int i = 0; i < ch_n; i++) {
                    if (DCLASS(ch_idx[i]) != bucket) continue;
                    int fx = ch_len[i] >= 64;
                    size_t cl = fx ? huf_encode4(ht, ch_norm[i], (size_t)ch_len[i], cb)
                                   : huf_encode(ht, ch_norm[i], (size_t)ch_len[i], cb);
                    if (cl + 2 >= (size_t)ch_len[i]) {
                        cand += (size_t)ch_len[i]; clens[m] = 0;
                    } else {
                        cand += cl + 2; clens[m] = cl;
                        cbufs[m] = malloc(cl); memcpy(cbufs[m], cb, cl);
                    }
                    fourx[m] = fx; m++;
                }
                int commit = braw > cand && (double)(braw - cand) >= 0.005 * (double)braw;
                if (!commit) {
                    t_shr += braw;
                    for (int i = 0; i < m; i++) if (clens[i]) free(cbufs[i]);
                } else {
                    t_shr += cand; g_shr_committed++;
                    /* store the group for the speed phase */
                    if (g_ngrp < 32768 && g_ngs + m < 131072 && g_gs_bytes < (48u << 20)) {
                        grp_t *g = &g_grp[g_ngrp];
                        g->lens = malloc(256); memcpy(g->lens, ht->len, 256);
                        g->first = g_ngs; g->n = 0;
                        int mi = 0;
                        for (int i = 0; i < ch_n; i++) {
                            if (DCLASS(ch_idx[i]) != bucket) continue;
                            if (clens[mi]) {
                                gsamp_t *sm = &g_gs[g_ngs++];
                                sm->cbuf = cbufs[mi]; sm->clen = clens[mi];
                                sm->norm = ch_norm[i]; sm->len = (size_t)ch_len[i];
                                sm->fourx = fourx[mi];
                                g_gs_bytes += sm->len; g->n++;
                            }
                            mi++;
                        }
                        if (g->n) g_ngrp++; else { free(g->lens); }
                    } else {
                        for (int i = 0; i < m; i++) if (clens[i]) free(cbufs[i]);
                    }
                }
                free(ht);
            }
            for (int i = 0; i < ch_n; i++) { /* norm copies owned by g_gs where stored */ }
        }
        double r = (double)t_raw;
        if (g_wscan) {
            printf("WSCAN %-14s (%llu bitmaps scanned; ideal coder + 6b/sym table; abs bytes incl. flats)\n",
                   argv[a], (unsigned long long)w_nscan);
            printf("   W:   ");
            for (int wi = 0; wi < NW; wi++) printf(" %8d", WLIST[wi]);
            printf(" |   oracle\n   tab: ");
            for (int wi = 0; wi < NW; wi++) printf(" %8llu", (unsigned long long)w_tot[wi]);
            printf(" | %8llu\n   wins:", (unsigned long long)w_oracle);
            for (int wi = 0; wi < NW; wi++) printf(" %8llu", (unsigned long long)w_wins[wi]);
            printf(" | raw %llu\n   kt:  ", (unsigned long long)w_rawwins);
            for (int wi = 0; wi < NW; wi++) printf(" %8llu", (unsigned long long)w_ktot[wi]);
            printf(" | %8llu\n   kwin:", (unsigned long long)w_koracle);
            for (int wi = 0; wi < NW; wi++) printf(" %8llu", (unsigned long long)w_kwins[wi]);
            printf(" | raw %llu\n", (unsigned long long)w_krawwins);
            printf("   blkctx W8 o=2: ideal=%llu huf=%llu | o=4: ideal=%llu huf=%llu\n",
                   (unsigned long long)w_bci[0], (unsigned long long)w_bch[0],
                   (unsigned long long)w_bci[1], (unsigned long long)w_bch[1]);
            printf("   quad-fuse: %llu triples, raw=%llu | sep o=2,4,6: %llu %llu %llu | fused q=1,2,3: %llu %llu %llu\n",
                   (unsigned long long)q_n, (unsigned long long)q_traw,
                   (unsigned long long)q_sep[0], (unsigned long long)q_sep[1],
                   (unsigned long long)q_sep[2],
                   (unsigned long long)q_fus[0], (unsigned long long)q_fus[1],
                   (unsigned long long)q_fus[2]);
            printf("   class-stream ctx gain (H0 -> kt q=3): tree-root n=%llu %llu->%llu | val-top2 %llu->%llu | val-low2 %llu->%llu\n",
                   (unsigned long long)rv_n[0],
                   (unsigned long long)rv_h0[0], (unsigned long long)rv_kt[0],
                   (unsigned long long)rv_h0[1], (unsigned long long)rv_kt[1],
                   (unsigned long long)rv_h0[2], (unsigned long long)rv_kt[2]);
            for (int db = 0; db < 4; db++) {
                if (!qd_n[db]) continue;
                uint64_t bs = qd_sep[db][0], bf = qd_fus[db][0];
                for (int oi = 1; oi < 3; oi++) {
                    if (qd_sep[db][oi] < bs) bs = qd_sep[db][oi];
                    if (qd_fus[db][oi] < bf) bf = qd_fus[db][oi];
                }
                printf("   quad d%s%d: n=%4llu raw=%8llu sep=%8llu fused=%8llu (%+.1f%%)\n",
                       db == 3 ? ">=" : "=", db, (unsigned long long)qd_n[db],
                       (unsigned long long)qd_raw[db], (unsigned long long)bs,
                       (unsigned long long)bf,
                       100.0 * ((double)bf / (double)bs - 1.0));
            }
            {
                uint64_t ts = 0, tf = 0;
                for (int db = 0; db < 4; db++) { ts += qh_sep[db]; tf += qh_fus[db]; }
                if (ts)
                    printf("   quad pure-huf0: sep=%llu fused=%llu (%+.1f%%) | d0 %llu->%llu d1 %llu->%llu d2 %llu->%llu d3+ %llu->%llu\n",
                           (unsigned long long)ts, (unsigned long long)tf,
                           100.0 * ((double)tf / (double)ts - 1.0),
                           (unsigned long long)qh_sep[0], (unsigned long long)qh_fus[0],
                           (unsigned long long)qh_sep[1], (unsigned long long)qh_fus[1],
                           (unsigned long long)qh_sep[2], (unsigned long long)qh_fus[2],
                           (unsigned long long)qh_sep[3], (unsigned long long)qh_fus[3]);
            }
            {
                uint64_t tn2 = 0, ta = 0, tr = 0, ts[3] = {0,0,0}, tq[3] = {0,0,0},
                         tc[3] = {0,0,0}, tf[2] = {0,0};
                for (int d2 = 0; d2 < 4; d2++) {
                    tn2 += od_n[d2]; ta += od_all[d2]; tr += od_raw[d2];
                    for (int oi = 0; oi < 3; oi++) {
                        ts[oi] += od_sep[d2][oi]; tq[oi] += od_qchain[d2][oi];
                        tc[oi] += od_cquad[d2][oi];
                    }
                    tf[0] += od_fus[d2][0]; tf[1] += od_fus[d2][1];
                }
                if (tn2)
                    printf("   oct3: %llu/%llu triples, raw7=%llu | sep7 o246: %llu %llu %llu | quad+gc: %llu %llu %llu | par+2cq: %llu %llu %llu | oct q12: %llu %llu\n",
                           (unsigned long long)tn2, (unsigned long long)ta,
                           (unsigned long long)tr,
                           (unsigned long long)ts[0], (unsigned long long)ts[1], (unsigned long long)ts[2],
                           (unsigned long long)tq[0], (unsigned long long)tq[1], (unsigned long long)tq[2],
                           (unsigned long long)tc[0], (unsigned long long)tc[1], (unsigned long long)tc[2],
                           (unsigned long long)tf[0], (unsigned long long)tf[1]);
                for (int d2 = 0; d2 < 4; d2++) {
                    if (!od_n[d2]) continue;
                    uint64_t bs = od_sep[d2][0], bq = od_qchain[d2][0], bc = od_cquad[d2][0];
                    for (int oi = 1; oi < 3; oi++) {
                        if (od_sep[d2][oi] < bs) bs = od_sep[d2][oi];
                        if (od_qchain[d2][oi] < bq) bq = od_qchain[d2][oi];
                        if (od_cquad[d2][oi] < bc) bc = od_cquad[d2][oi];
                    }
                    uint64_t bf = od_fus[d2][0] < od_fus[d2][1] ? od_fus[d2][0] : od_fus[d2][1];
                    printf("   oct3 d%s%d: n=%4llu raw=%8llu sep=%8llu quad+gc=%8llu par+2cq=%8llu oct=%8llu (oct vs quad+gc %+.1f%%)\n",
                           d2 == 3 ? ">=" : "=", d2, (unsigned long long)od_n[d2],
                           (unsigned long long)od_raw[d2], (unsigned long long)bs,
                           (unsigned long long)bq, (unsigned long long)bc,
                           (unsigned long long)bf,
                           100.0 * ((double)bf / (double)bq - 1.0));
                }
                uint64_t qs2 = 0, qf2 = 0, qr2 = 0;
                for (int d2 = 0; d2 < 4; d2++) { qs2 += qnd_sep[d2]; qf2 += qnd_fus[d2]; qr2 += qnd_raw[d2]; }
                if (qnd_n)
                    printf("   quad-dyn51: %llu triples raw=%llu sep=%llu fused=%llu (%+.1f%%) | d0 %llu->%llu d1 %llu->%llu d2 %llu->%llu d3+ %llu->%llu\n",
                           (unsigned long long)qnd_n, (unsigned long long)qr2,
                           (unsigned long long)qs2, (unsigned long long)qf2,
                           100.0 * ((double)qf2 / (double)qs2 - 1.0),
                           (unsigned long long)qnd_sep[0], (unsigned long long)qnd_fus[0],
                           (unsigned long long)qnd_sep[1], (unsigned long long)qnd_fus[1],
                           (unsigned long long)qnd_sep[2], (unsigned long long)qnd_fus[2],
                           (unsigned long long)qnd_sep[3], (unsigned long long)qnd_fus[3]);
            }
            printf("   pmkv W8 o=2: tans=%llu huf=%llu | o=4: tans=%llu huf=%llu\n",
                   (unsigned long long)w_pmi[0], (unsigned long long)w_pmh[0],
                   (unsigned long long)w_pmi[1], (unsigned long long)w_pmh[1]);
            printf("   mkv o=1,2,3,4,6,8:");
            for (int oi = 0; oi < 6; oi++) printf(" %8llu", (unsigned long long)w_mktot[oi]);
            printf("  phase=%llu\n   all-oracle=%llu (wins: W %llu, mkv1..8 %llu/%llu/%llu/%llu/%llu/%llu, ph %llu)\n",
                   (unsigned long long)w_phtot, (unsigned long long)w_alloracle,
                   (unsigned long long)w_awins,
                   (unsigned long long)w_mwins[0], (unsigned long long)w_mwins[1],
                   (unsigned long long)w_mwins[2], (unsigned long long)w_mwins[3],
                   (unsigned long long)w_mwins[4], (unsigned long long)w_mwins[5],
                   (unsigned long long)w_mwins[6]);

            /* ---- size/depth decomposition of the oracle-vs-fixed gap ---- */
            static const char *KB[4] = {"K<=512", "K<=2k", "K<=8k", "K>8k"};
            static const char *DB[5] = {"d=0", "d=1", "d=2", "d=3", "d>=4"};
            uint64_t kt_tot[4][NW], kt_orc[4]; int kt_cnt[4][NW], kt_n[4];
            uint64_t dt_tot[5][NW], dt_orc[5]; int dt_cnt[5][NW], dt_n[5];
            memset(kt_tot, 0, sizeof(kt_tot)); memset(kt_orc, 0, sizeof(kt_orc));
            memset(kt_cnt, 0, sizeof(kt_cnt)); memset(kt_n, 0, sizeof(kt_n));
            memset(dt_tot, 0, sizeof(dt_tot)); memset(dt_orc, 0, sizeof(dt_orc));
            memset(dt_cnt, 0, sizeof(dt_cnt)); memset(dt_n, 0, sizeof(dt_n));
            uint64_t fixed8 = 0, orc = 0, regsum = 0;
            static uint32_t regs[65536]; int nreg5 = 0; uint64_t reg5b = 0;
            for (int i = 0; i < wb_n; i++) {
                int kbk = wb_K[i] <= 512 ? 0 : wb_K[i] <= 2048 ? 1 : wb_K[i] <= 8192 ? 2 : 3;
                int dbk = wb_depth[i] > 4 ? 4 : wb_depth[i];
                uint32_t mn = wb_kt[i][0]; int mi = 0;
                for (int wi = 1; wi < NW; wi++)
                    if (wb_kt[i][wi] < mn) { mn = wb_kt[i][wi]; mi = wi; }
                uint32_t r8 = wb_kt[i][6] - mn;    /* WLIST[6] == 8 */
                fixed8 += wb_kt[i][6]; orc += mn; regsum += r8;
                regs[i] = r8;
                int nbb = (wb_K[i] + 7) >> 3;
                if (r8 * 20 >= (uint32_t)nbb) { nreg5++; reg5b += r8; }
                kt_n[kbk]++; dt_n[dbk]++;
                kt_cnt[kbk][mi]++; dt_cnt[dbk][mi]++;
                kt_orc[kbk] += mn; dt_orc[dbk] += mn;
                for (int wi = 0; wi < NW; wi++) {
                    kt_tot[kbk][wi] += wb_kt[i][wi];
                    dt_tot[dbk][wi] += wb_kt[i][wi];
                }
            }
            uint64_t ruleK = 0, ruleD = 0;
            for (int bkt = 0; bkt < 4; bkt++) {
                if (!kt_n[bkt]) continue;
                uint64_t mn = kt_tot[bkt][0]; int mi = 0;
                for (int wi = 1; wi < NW; wi++)
                    if (kt_tot[bkt][wi] < mn) { mn = kt_tot[bkt][wi]; mi = wi; }
                ruleK += mn;
                printf("   %-7s n=%4d bestW=%-2d tot=%8llu oracle=%8llu | argmin:",
                       KB[bkt], kt_n[bkt], WLIST[mi],
                       (unsigned long long)mn, (unsigned long long)kt_orc[bkt]);
                for (int wi = 0; wi < NW; wi++)
                    if (kt_cnt[bkt][wi]) printf(" %d:%d", WLIST[wi], kt_cnt[bkt][wi]);
                printf("\n");
            }
            for (int bkt = 0; bkt < 5; bkt++) {
                if (!dt_n[bkt]) continue;
                uint64_t mn = dt_tot[bkt][0]; int mi = 0;
                for (int wi = 1; wi < NW; wi++)
                    if (dt_tot[bkt][wi] < mn) { mn = dt_tot[bkt][wi]; mi = wi; }
                ruleD += mn;
                printf("   %-7s n=%4d bestW=%-2d tot=%8llu oracle=%8llu | argmin:",
                       DB[bkt], dt_n[bkt], WLIST[mi],
                       (unsigned long long)mn, (unsigned long long)dt_orc[bkt]);
                for (int wi = 0; wi < NW; wi++)
                    if (dt_cnt[bkt][wi]) printf(" %d:%d", WLIST[wi], dt_cnt[bkt][wi]);
                printf("\n");
            }
            /* top-20 regret concentration */
            uint64_t top20 = 0;
            for (int t = 0; t < 20 && t < wb_n; t++) {
                int mx = 0;
                for (int i = 1; i < wb_n; i++) if (regs[i] > regs[mx]) mx = i;
                top20 += regs[mx]; regs[mx] = 0;
            }
            printf("   fixed8=%llu ruleK=%llu ruleD=%llu oracle=%llu | regret8: total=%llu top20=%llu >=5%%raw on %d bms (%llu B)\n",
                   (unsigned long long)fixed8, (unsigned long long)ruleK,
                   (unsigned long long)ruleD, (unsigned long long)orc,
                   (unsigned long long)regsum, (unsigned long long)top20,
                   nreg5, (unsigned long long)reg5b);
        }
        printf("   p_major bands <.55/<.625/<.75/<.90/>=.90: n %llu/%llu/%llu/%llu/%llu  bytes %llu/%llu/%llu/%llu/%llu\n",
               (unsigned long long)g_pm[0], (unsigned long long)g_pm[1],
               (unsigned long long)g_pm[2], (unsigned long long)g_pm[3],
               (unsigned long long)g_pm[4],
               (unsigned long long)g_pmbytes[0], (unsigned long long)g_pmbytes[1],
               (unsigned long long)g_pmbytes[2], (unsigned long long)g_pmbytes[3],
               (unsigned long long)g_pmbytes[4]);
        printf("%-18s %9llu | %9llu %9llu | %9llu %9llu %9llu   (abs bytes incl. raw flat regions: raw | sFSE bHuf | dANS huf0 shHu)\n",
               argv[a], (unsigned long long)t_raw,
               (unsigned long long)t_sfse, (unsigned long long)t_bhuf,
               (unsigned long long)t_dans, (unsigned long long)t_huf0,
               (unsigned long long)t_shr);
        if (0) printf("%-18s %9llu | %9llu %9llu %9llu   (abs bytes incl. raw flat regions: raw | dANS huf0 shHu)\n",
               strrchr(argv[a], '/') ? strrchr(argv[a], '/') + 1 : argv[a],
               (unsigned long long)t_raw,
               (unsigned long long)t_dans, (unsigned long long)t_huf0,
               (unsigned long long)t_shr);
        free(data);
    }

    /* ---------- speed phase: byte-huf vs static FSE on the same samples ---------- */
    printf("\nspeed (per-bitmap calls over %d committed bitmaps, %.1f MB; roundtrip-verified):\n",
           g_nsamples, g_sample_bytes / 1048576.0);
    if (g_nsamples == 0) return 0;

    /* pre-encode both forms */
    static uint8_t *huf_enc_buf[65536]; static size_t huf_enc_len[65536];
    static uint8_t *huf4_enc_buf[65536]; static size_t huf4_enc_len[65536];
    static uint8_t *fse_enc_buf[65536]; static size_t fse_enc_len[65536];
    uint8_t tmp[CHUNK / 8 + 512], back[CHUNK / 8 + 512];
    size_t fse_bytes = 0;
    for (int i = 0; i < g_nsamples; i++) {
        sample_t *s = &g_samples[i];
        size_t hl = huf_encode(&g_huf[s->idx], s->data, s->len, tmp);
        huf_enc_buf[i] = malloc(hl); memcpy(huf_enc_buf[i], tmp, hl); huf_enc_len[i] = hl;
        huf_decode(&g_huf[s->idx], huf_enc_buf[i], hl, back, s->len);
        if (memcmp(back, s->data, s->len)) { fprintf(stderr, "HUF ROUNDTRIP FAIL sample %d\n", i); return 1; }
        size_t h4 = huf_encode4(&g_huf[s->idx], s->data, s->len, tmp);
        huf4_enc_buf[i] = malloc(h4); memcpy(huf4_enc_buf[i], tmp, h4); huf4_enc_len[i] = h4;
        memset(back, 0xAB, s->len);
        huf_decode4(&g_huf[s->idx], huf4_enc_buf[i], h4, back, s->len);
        if (memcmp(back, s->data, s->len)) { fprintf(stderr, "HUF4 ROUNDTRIP FAIL sample %d\n", i); return 1; }
        size_t fl = 0;
        if (pivco_fse_compress(s->idx, s->data, s->len, tmp, sizeof(tmp), &fl) == PIVCO_FSE_OK) {
            fse_enc_buf[i] = malloc(fl); memcpy(fse_enc_buf[i], tmp, fl); fse_enc_len[i] = fl;
            size_t ol = 0;
            pivco_fse_decompress(s->idx, fse_enc_buf[i], fl, back, sizeof(back), s->len, &ol);
            if (ol != s->len || memcmp(back, s->data, s->len)) { fprintf(stderr, "FSE ROUNDTRIP FAIL %d\n", i); return 1; }
            fse_bytes += s->len;
        } else fse_enc_buf[i] = NULL;
    }
    /* huf0 + nested-PH pre-encode for the speed phase */
    static uint8_t *h0_buf[65536]; static size_t h0_len[65536];
    static uint8_t *ph_buf[65536]; static size_t ph_len[65536];
    static uint8_t *ph_lens[65536];
    size_t h0_bytes = 0, ph_bytes = 0;
    pivco_decoder_t *dec = pivco_decoder_create();
    {
        static pivco_table_t bt;
        pivco_cfg_t bcfg = pivco_cfg_default; bcfg.fse_enabled = 0;
        static uint8_t pout[PIVCO_MAX_ENCODED_SIZE];
        for (int i = 0; i < g_nsamples; i++) {
            sample_t *sm = &g_samples[i];
            size_t cs = HUF_compress(tmp, sizeof(tmp), sm->raw, sm->len);
            if (cs > 0 && !HUF_isError(cs) && cs < sm->len) {
                h0_buf[i] = malloc(cs); memcpy(h0_buf[i], tmp, cs); h0_len[i] = cs;
                HUF_decompress(back, sm->len, h0_buf[i], cs);
                if (memcmp(back, sm->raw, sm->len)) { fprintf(stderr, "HUF0 RT FAIL %d\n", i); return 1; }
                h0_bytes += sm->len;
            } else h0_buf[i] = NULL;
            uint64_t bfreq[256] = {0};
            for (size_t k = 0; k < sm->len; k++) bfreq[sm->raw[k]]++;
            size_t pl = 0;
            if (pivco_build_table(&bcfg, bfreq, &bt) == PIVCO_OK &&
                bt.tree[bt.tree_root].symbol < 0 &&
                pivco_encode(enc, &bt, sm->raw, sm->len, pout, &pl) == PIVCO_OK) {
                ph_buf[i] = malloc(pl); memcpy(ph_buf[i], pout, pl); ph_len[i] = pl;
                ph_lens[i] = malloc(256); memcpy(ph_lens[i], bt.code_len, 256);
                size_t consumed = 0;
                memset(back, 0xAB, sm->len);
                if (pivco_decode(dec, &bt, ph_buf[i], pl, back, &consumed) != PIVCO_OK ||
                    memcmp(back, sm->raw, sm->len)) { fprintf(stderr, "PH RT FAIL %d\n", i); return 1; }
                ph_bytes += sm->len;
            } else { ph_buf[i] = NULL; ph_lens[i] = NULL; }
        }
    }
    const int R = 12;
    double t0, tt;
    /* decode */
    t0 = now_sec();
    for (int r = 0; r < R; r++)
        for (int i = 0; i < g_nsamples; i++)
            huf_decode(&g_huf[g_samples[i].idx], huf_enc_buf[i], huf_enc_len[i], back, g_samples[i].len);
    tt = now_sec() - t0;
    printf("  byteHuf decode (1-stream %d-bit LUT): %7.0f MB/s\n", HUF_LOG,
           g_sample_bytes * (double)R / tt / 1048576.0);
    t0 = now_sec();
    for (int r = 0; r < R; r++)
        for (int i = 0; i < g_nsamples; i++)
            huf_decode4(&g_huf[g_samples[i].idx], huf4_enc_buf[i], huf4_enc_len[i], back, g_samples[i].len);
    tt = now_sec() - t0;
    printf("  byteHuf decode (4-stream, +6B hdr):   %7.0f MB/s\n",
           g_sample_bytes * (double)R / tt / 1048576.0);
    t0 = now_sec();
    for (int r = 0; r < R; r++)
        for (int i = 0; i < g_nsamples; i++)
            if (fse_enc_buf[i]) {
                size_t ol = 0;
                pivco_fse_decompress(g_samples[i].idx, fse_enc_buf[i], fse_enc_len[i], back, sizeof(back), g_samples[i].len, &ol);
            }
    tt = now_sec() - t0;
    printf("  static FSE decode (production path):  %7.0f MB/s  (over %.1f MB it accepted)\n",
           fse_bytes * (double)R / tt / 1048576.0, fse_bytes / 1048576.0);
    t0 = now_sec();
    for (int r = 0; r < R; r++)
        for (int i = 0; i < g_nsamples; i++)
            if (h0_buf[i]) HUF_decompress(back, g_samples[i].len, h0_buf[i], h0_len[i]);
    tt = now_sec() - t0;
    printf("  huf0 decode (4X, dynamic table):      %7.0f MB/s  (over %.1f MB)\n",
           h0_bytes * (double)R / tt / 1048576.0, h0_bytes / 1048576.0);
    {
        huftab_t *gt = malloc(sizeof(huftab_t));
        /* verify once */
        for (int g = 0; g < g_ngrp; g++) {
            huftab_from_lengths(gt, g_grp[g].lens);
            for (int i = 0; i < g_grp[g].n; i++) {
                gsamp_t *sm = &g_gs[g_grp[g].first + i];
                memset(back, 0xAB, sm->len);
                if (sm->fourx) huf_decode4(gt, sm->cbuf, sm->clen, back, sm->len);
                else           huf_decode(gt, sm->cbuf, sm->clen, back, sm->len);
                if (memcmp(back, sm->norm, sm->len)) {
                    fprintf(stderr, "SHARED RT FAIL grp %d member %d\n", g, i); exit(1);
                }
            }
        }
        t0 = now_sec();
        for (int r = 0; r < R; r++)
            for (int g = 0; g < g_ngrp; g++) {
                huftab_from_lengths(gt, g_grp[g].lens);
                for (int i = 0; i < g_grp[g].n; i++) {
                    gsamp_t *sm = &g_gs[g_grp[g].first + i];
                    if (sm->fourx) huf_decode4(gt, sm->cbuf, sm->clen, back, sm->len);
                    else           huf_decode(gt, sm->cbuf, sm->clen, back, sm->len);
                }
            }
        tt = now_sec() - t0;
        printf("  shared-table dyn byteHuf decode:      %7.0f MB/s  (over %.1f MB, %.1f bitmaps/table avg incl. LUT rebuild)\n",
               g_gs_bytes * (double)R / tt / 1048576.0, g_gs_bytes / 1048576.0,
               g_shr_groups ? (double)g_shr_members / (double)g_shr_groups : 0.0);
        free(gt);
        static pivco_table_t bt2;
        pivco_cfg_t bcfg2 = pivco_cfg_default; bcfg2.fse_enabled = 0;
        t0 = now_sec();
        for (int r = 0; r < R; r++)
            for (int i = 0; i < g_nsamples; i++)
                if (ph_buf[i]) {
                    size_t consumed = 0;
                    pivco_build_table_from_code_lens(&bcfg2, ph_lens[i], &bt2);
                    pivco_decode(dec, &bt2, ph_buf[i], ph_len[i], back, &consumed);
                }
        tt = now_sec() - t0;
        printf("  nested-PH decode (rebuild + decode):  %7.0f MB/s  (over %.1f MB)\n",
               ph_bytes * (double)R / tt / 1048576.0, ph_bytes / 1048576.0);
        t0 = now_sec();
        for (int r = 0; r < R; r++)
            for (int i = 0; i < g_nsamples; i++)
                if (ph_buf[i]) pivco_build_table_from_code_lens(&bcfg2, ph_lens[i], &bt2);
        tt = now_sec() - t0;
        printf("  nested-PH table rebuild alone:        %7.0f MB/s-equivalent\n",
               ph_bytes * (double)R / tt / 1048576.0);
    }
    /* encode */
    t0 = now_sec();
    for (int r = 0; r < R; r++)
        for (int i = 0; i < g_nsamples; i++)
            huf_encode(&g_huf[g_samples[i].idx], g_samples[i].data, g_samples[i].len, tmp);
    tt = now_sec() - t0;
    printf("  byteHuf encode:                       %7.0f MB/s\n", g_sample_bytes * (double)R / tt / 1048576.0);
    t0 = now_sec();
    for (int r = 0; r < R; r++)
        for (int i = 0; i < g_nsamples; i++) {
            size_t fl = 0;
            pivco_fse_compress(g_samples[i].idx, g_samples[i].data, g_samples[i].len, tmp, sizeof(tmp), &fl);
        }
    tt = now_sec() - t0;
    printf("  static FSE encode:                    %7.0f MB/s\n", g_sample_bytes * (double)R / tt / 1048576.0);
    return 0;
}
