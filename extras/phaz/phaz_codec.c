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

size_t phaz_capture_run(const unsigned char *src, size_t n, int level, int want_stock) {
    phaz_capture_free();                       /* drop any previous run's buffers */
    size_t bound = ZSTD_compressBound(n);
    unsigned char *c = malloc(bound);
    if (!c) return (size_t)-1;

    size_t zsize = 0;
    if (want_stock) {
        zsize = ZSTD_compress(c, bound, src, n, level);
        if (ZSTD_isError(zsize)) { free(c); return (size_t)-1; }
    }

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

    /* phaz re-codes literals itself, so skip zstd's HUF on the (discarded)
     * literal section -- recovers ~9 ms.  The parse is unchanged at the lazy
     * strategies; at btopt levels (>=16) disabling literal compression shifts
     * the parser's cost model, so the captured parse differs slightly from a
     * stock zstd compress (still byte-exact, just a different parse). */
    ZSTD_CCtx *cc = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(cc, ZSTD_c_compressionLevel, level);
    ZSTD_CCtx_setParameter(cc, ZSTD_c_literalCompressionMode, ZSTD_lcm_uncompressed);
    size_t z2 = ZSTD_compress2(cc, c, bound, src, n);
    g_phaz_dump = 0;
    ZSTD_freeCCtx(cc); free(c);
    if (ZSTD_isError(z2)) { phaz_capture_free(); return (size_t)-1; }
    return zsize;
}

/* PHA-encode one stream; see header.  PHA (#PHA) gates FSE per node by
 * compressibility, so it dominates plain PH -- no need to try both.  Raw
 * fallback only if PHA expands (tiny streams).  One global table per stream
 * (per-128KB re-table was tried and lost: pivcohuf pays a full header +
 * checksum + table per call, no FSE repeat-mode). */
size_t phaz_pack_stream(unsigned char **cur, unsigned char *end,
                        const unsigned char *raw, size_t rawlen,
                        size_t *best_out, char *tag_out) {
    size_t bound = pivcohuf_compress_bound(rawlen ? rawlen : 1);
    unsigned char *t = malloc(bound);
    if (!t) return 0;
    size_t l = bound;
    int ok = rawlen && pivcohuf_compress_ex(raw, rawlen, t, &l, 1) == PIVCOHUF_OK;
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

size_t phaz_compress_bound(size_t n) {
    size_t nblk    = (n >> 17) + 2;                          /* ~128KB zstd blocks */
    size_t hdr     = 5 + sizeof(uint64_t) * 5
                     + nblk * (sizeof(unsigned) * 2 + 1) + sizeof(uint64_t);
    size_t xb      = ZSTD_sequenceBound(n) + 64;             /* extra-bits ceiling */
    size_t seqb    = ZSTD_sequenceBound(n);
    size_t streams = 4 * (1 + sizeof(uint64_t))             /* per-stream framing */
                     + pivcohuf_compress_bound(n)            /* lit ~ n */
                     + 3 * pivcohuf_compress_bound(seqb);    /* ll/ml/of ~ nseq */
    return hdr + xb + streams + 64;
}

size_t phaz_compress(const void *src_, size_t n, void *dst_, size_t cap,
                     int level, phaz_stats *st) {
    const unsigned char *src = src_;
    unsigned char *dst = dst_, *cur = dst, *end = dst + cap;

    double t0 = now();
    if (phaz_capture_run(src, n, level, 0) == (size_t)-1) return 0;
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
    for (int i = 0; i < 4; i++) {
        double a = now(); size_t enc = 0;
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
    for (int i = 0; i < 4; i++) {
        double a = now();
        str[i] = unpack_stream(&p, end, srl[i]);
        if (!str[i]) { for (int j = 0; j < i; j++) free(str[j]);
                       free(bns); free(btl); free(bcf); return 0; }
        if (st) { st->entropy_ms[i] = (now() - a) * 1e3; st->stream_raw[i] = srl[i]; }
    }
    double tr = now();
    size_t got = ZSTD_phazDecode(dst_, cap, str[0], str[1], str[2], xb, str[3],
                                 lits, bns, btl, bcf, nblk);
    if (st) st->reconstruct_ms = (now() - tr) * 1e3;
    for (int i = 0; i < 4; i++) free(str[i]);
    free(bns); free(btl); free(bcf);
    return got == n ? got : 0;
}
