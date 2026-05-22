/* bench_fair.c -- fair head-to-head: ph / pha(ph+FSE) / huf0 / FSE on a
 * fixed 1 MB byte buffer, in TWO modes:
 *
 *   opaque   : realistic per-call cost -- each codec (re)builds its
 *              entropy table at its own granularity G (table-refresh
 *              bytes) and the table bytes count toward the ratio.
 *   prebuilt : one table built once up front and reused across the
 *              whole 1 MB (huf0 via usingDTable/usingCTable, FSE via
 *              usingC/DTable, ph via its static table) -- isolates raw
 *              kernel throughput.
 *
 * ph's table-refresh granularity G is decoupled from PIVCO_BLOCK_SIZE
 * (the 8 KB decode sub-block): in opaque mode ph rebuilds its Huffman
 * tree every G bytes (default 128 KB, matching huf0's hard chunk cap),
 * which models the parked block-structured file format's "format block".
 *
 * Methodology: best of RUNS runs x REPEATS passes over the 1 MB buffer.
 * Reports enc/dec MB/s (input bytes) for each mode + compression ratio
 * + table builds per 1 MB.  Oodle columns (opaque-only) added separately
 * under PIVCO_HAS_OODLE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "pivco_huffman.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#include "bitstream.h"
#include "fse_xy_codec.h"   /* encode_x + decode_x8_y1 (tuned shape) */

/* From bench_distributions.c */
extern void         bench_init(void);
extern int          bench_num_distributions(void);
extern const char  *bench_dist_name(int idx);
extern int          bench_dist_is_main(int idx);
extern void         bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                           int n_symbols, uint64_t seed);

/* ---- config ---- */
#define TOTAL      (1 << 20)            /* 1 MB working buffer */
#define BLK        PIVCO_BLOCK_SIZE     /* ph decode sub-block (4-8 KB) */
#define HUF_CHUNK  (128 * 1024)         /* huf0 hard cap; FSE controlled chunk */
#define RUNS       5
#define REPEATS    10
#define SEED       0xBEEFCAFE12345678ULL
#define MAXLOG     12                   /* FSE/HUF table log */

static size_t g_table_G = 128 * 1024;   /* ph table-refresh granularity */

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* best (max) MB/s over RUNS, each run = REPEATS passes over TOTAL bytes */
#define BEST_MBPS(...) do {                                             \
        best = 0.0;                                                     \
        for (int _r = 0; _r < RUNS; _r++) {                             \
            double _t0 = now_ns();                                      \
            for (int _rep = 0; _rep < REPEATS; _rep++) { __VA_ARGS__; } \
            double _el = now_ns() - _t0;                                \
            double _mb = 1000.0 * (double)TOTAL * REPEATS / _el;        \
            if (_mb > best) best = _mb;                                 \
        }                                                               \
    } while (0)

static void histo_u64(const uint8_t *s, size_t n, uint64_t f[256]) {
    memset(f, 0, 256 * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++) f[s[i]]++;
}
static void histo_u(const uint8_t *s, size_t n, unsigned f[256], unsigned *maxSym) {
    memset(f, 0, 256 * sizeof(unsigned));
    for (size_t i = 0; i < n; i++) f[s[i]]++;
    unsigned m = 255; while (m > 0 && f[m] == 0) m--;
    *maxSym = m;
}

typedef struct {
    int    ok;
    double enc_op, enc_pb, dec_op, dec_pb;  /* MB/s */
    double ratio_op, ratio_pb;              /* TOTAL / comp_bytes */
    int    builds;                          /* table builds / 1 MB (opaque) */
} result_t;

/* ============================ ph / pha ============================ */
static result_t measure_ph(const uint8_t *sym, size_t n, int fse_on) {
    result_t R; memset(&R, 0, sizeof R);
    size_t nblk = n / BLK;
    size_t G    = g_table_G;
    size_t nwin = n / G;
    size_t bpw  = G / BLK;            /* sub-blocks per table window */
    R.builds = (int)nwin;

    pivco_huffman_set_fse_enabled(fse_on);

    pivco_huffman_table_t *gtbl = NULL, *wtbl = NULL, *wtbls = NULL;
    uint8_t (*win_clen)[256] = NULL;
    uint8_t *enc = NULL, *enco = NULL, *dec = NULL;
    size_t  *off = NULL, *offo = NULL;

    gtbl = malloc(sizeof *gtbl);
    wtbl = malloc(sizeof *wtbl);
    if (!gtbl || !wtbl) goto done_fail;
    uint64_t f[256];
    histo_u64(sym, n, f);
    if (pivco_huffman_build_table(f, gtbl) != 0) goto done_fail;

    /* per-window tables + their code_lens, for opaque enc/dec */
    win_clen = malloc(nwin * 256);
    wtbls = malloc(nwin * sizeof *wtbls);
    if (!win_clen || !wtbls) goto done_fail;
    for (size_t w = 0; w < nwin; w++) {
        uint64_t wf[256]; histo_u64(sym + w * G, G, wf);
        if (pivco_huffman_build_table(wf, &wtbls[w]) != 0) goto done_fail;
        memcpy(win_clen[w], wtbls[w].code_len, 256);
    }

    enc = malloc(n + n / 2 + 4096);
    off = malloc((nblk + 1) * sizeof(size_t));   /* prebuilt stream offsets */
    offo= malloc((nblk + 1) * sizeof(size_t));   /* opaque   stream offsets */
    enco= malloc(n + n / 2 + 4096);
    dec = malloc(n);
    if (!enc || !off || !offo || !enco || !dec) goto done_fail;

    /* pre-encode prebuilt stream (global table) */
    off[0] = 0;
    for (size_t b = 0; b < nblk; b++) {
        size_t L = 0;
        if (pivco_huffman_encode(sym + b * BLK, gtbl, enc + off[b], &L) != 0) goto done_fail;
        off[b + 1] = off[b] + L;
    }
    /* pre-encode opaque stream (per-window tables) */
    offo[0] = 0;
    for (size_t w = 0; w < nwin; w++)
        for (size_t i = 0; i < bpw; i++) {
            size_t b = w * bpw + i, L = 0;
            if (pivco_huffman_encode(sym + b * BLK, &wtbls[w], enco + offo[b], &L) != 0) goto done_fail;
            offo[b + 1] = offo[b] + L;
        }

    /* correctness check (prebuilt + opaque) */
    for (size_t b = 0; b < nblk; b++) {
        size_t c = 0;
        pivco_huffman_decode(enc + off[b], off[b+1]-off[b], gtbl, dec, &c);
        if (memcmp(sym + b * BLK, dec, BLK) != 0) { fprintf(stderr,"ph PB mismatch blk %zu\n",b); goto done_fail; }
    }
    for (size_t w = 0; w < nwin; w++)
        for (size_t i = 0; i < bpw; i++) {
            size_t b = w*bpw+i, c = 0;
            pivco_huffman_decode(enco + offo[b], offo[b+1]-offo[b], &wtbls[w], dec, &c);
            if (memcmp(sym + b * BLK, dec, BLK) != 0) { fprintf(stderr,"ph OP mismatch blk %zu\n",b); goto done_fail; }
        }

    double best;
    /* ---- encode prebuilt: global table, just emit ---- */
    BEST_MBPS({
        for (size_t b = 0; b < nblk; b++) { size_t L=0; pivco_huffman_encode(sym + b*BLK, gtbl, enc + off[b], &L); }
    });
    R.enc_pb = best;
    /* ---- encode opaque: rebuild table per window + emit ---- */
    BEST_MBPS({
        for (size_t w = 0; w < nwin; w++) {
            uint64_t wf[256]; histo_u64(sym + w*G, G, wf);
            pivco_huffman_build_table(wf, wtbl);
            for (size_t i = 0; i < bpw; i++) { size_t b=w*bpw+i, L=0; pivco_huffman_encode(sym + b*BLK, wtbl, enco + offo[b], &L); }
        }
    });
    R.enc_op = best;
    /* ---- decode prebuilt: global table ---- */
    BEST_MBPS({
        for (size_t b = 0; b < nblk; b++) { size_t c=0; pivco_huffman_decode(enc + off[b], off[b+1]-off[b], gtbl, dec, &c); }
    });
    R.dec_pb = best;
    /* ---- decode opaque: rebuild table-from-codelens per window ---- */
    BEST_MBPS({
        for (size_t w = 0; w < nwin; w++) {
            pivco_huffman_build_table_from_code_lens(win_clen[w], NULL, wtbl);
            for (size_t i = 0; i < bpw; i++) { size_t b=w*bpw+i, c=0; pivco_huffman_decode(enco + offo[b], offo[b+1]-offo[b], wtbl, dec, &c); }
        }
    });
    R.dec_op = best;

    /* ratios: opaque adds ~128 B code-len header per window; prebuilt adds one */
    {
        size_t comp_pb = off[nblk] + 128;             /* one table header */
        size_t comp_op = offo[nblk] + 128 * nwin;     /* one per window */
        R.ratio_pb = (double)n / (double)comp_pb;
        R.ratio_op = (double)n / (double)comp_op;
    }
    R.ok = 1;
done_fail:
    free(gtbl); free(wtbl); free(wtbls); free(win_clen);
    free(enc); free(off); free(offo); free(enco); free(dec);
    return R;
}

/* ============================ huf0 (4X2) ============================ */
static result_t measure_huf0(const uint8_t *sym, size_t n) {
    result_t R; memset(&R, 0, sizeof R);
    size_t nch = (n + HUF_CHUNK - 1) / HUF_CHUNK;
    R.builds = (int)nch;

    unsigned cnt[256], maxSym; histo_u(sym, n, cnt, &maxSym);
    HUF_CREATE_STATIC_CTABLE(ctable, 255);
    size_t huffLog = HUF_buildCTable(ctable, cnt, maxSym, MAXLOG);  /* returns actual maxNbBits */
    if (HUF_isError(huffLog)) return R;

    uint8_t *enc = malloc(n + n/2 + 4096);        /* opaque stream (HUF_compress, w/ header) */
    uint8_t *encp= malloc(n + n/2 + 4096);        /* prebuilt stream (usingCTable body only) */
    size_t  *off = malloc((nch+1)*sizeof(size_t));
    size_t  *offp= malloc((nch+1)*sizeof(size_t));
    uint8_t *dec = malloc(n);
    void    *wksp= malloc(1<<16);
    HUF_DTable *dt   = malloc(HUF_DTABLE_SIZE(MAXLOG) * sizeof(HUF_DTable)); /* opaque scratch */
    HUF_DTable *dtpb = malloc(HUF_DTABLE_SIZE(MAXLOG) * sizeof(HUF_DTable)); /* prebuilt, global */
    uint8_t  hdr[512]; size_t hdrSize = HUF_writeCTable(hdr, sizeof hdr, ctable, maxSym, (unsigned)huffLog);
    if (!enc||!encp||!off||!offp||!dec||!wksp||!dt||!dtpb||HUF_isError(hdrSize)) goto fail;
    dt[0]   = (HUF_DTable)(MAXLOG * 0x01000001);   /* set max table log */
    dtpb[0] = (HUF_DTable)(MAXLOG * 0x01000001);

    /* pre-encode opaque (each chunk builds its own table, header inline) */
    off[0]=0;
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=HUF_compress(enc+off[c], sz+1024, sym+c*HUF_CHUNK, sz);
        if (HUF_isError(r)||r==0) goto fail; off[c+1]=off[c]+r; }
    /* pre-encode prebuilt (shared CTable, body only) */
    offp[0]=0;
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=HUF_compress4X_usingCTable(encp+offp[c], sz+1024, sym+c*HUF_CHUNK, sz, ctable);
        if (HUF_isError(r)||r==0) goto fail; offp[c+1]=offp[c]+r; }

    /* prebuilt DTable from the GLOBAL table header (matches usingCTable enc) */
    if (HUF_isError(HUF_readDTableX2(dtpb, hdr, hdrSize))) goto fail;

    /* correctness: opaque path (DCtx rebuilds per-chunk from inline header) */
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=HUF_decompress4X2_DCtx_wksp(dt, dec, sz, enc+off[c], off[c+1]-off[c], wksp, 1<<16);
        if (HUF_isError(r)||memcmp(sym+c*HUF_CHUNK,dec,sz)!=0){fprintf(stderr,"huf0 OP mismatch ch %zu\n",c);goto fail;} }
    /* correctness: prebuilt path (shared global DTable on header-less bodies) */
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=HUF_decompress4X2_usingDTable(dec, sz, encp+offp[c], offp[c+1]-offp[c], dtpb);
        if (HUF_isError(r)||memcmp(sym+c*HUF_CHUNK,dec,sz)!=0){fprintf(stderr,"huf0 PB mismatch ch %zu\n",c);goto fail;} }

    double best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        HUF_compress(enc+off[c], sz+1024, sym+c*HUF_CHUNK, sz); } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        HUF_compress4X_usingCTable(encp+offp[c], sz+1024, sym+c*HUF_CHUNK, sz, ctable); } });
    R.enc_pb = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        HUF_decompress4X2_DCtx_wksp(dt, dec, sz, enc+off[c], off[c+1]-off[c], wksp, 1<<16); } });
    R.dec_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        HUF_decompress4X2_usingDTable(dec, sz, encp+offp[c], offp[c+1]-offp[c], dtpb); } });
    R.dec_pb = best;

    R.ratio_op = (double)n / (double)off[nch];               /* headers inline */
    R.ratio_pb = (double)n / (double)(offp[nch] + hdrSize);  /* one shared header */
    R.ok = 1;
fail:
    free(enc); free(encp); free(off); free(offp); free(dec); free(wksp); free(dt); free(dtpb);
    return R;
}

/* ============================ FSE ============================ */
static result_t measure_fse(const uint8_t *sym, size_t n) {
    result_t R; memset(&R, 0, sizeof R);
    size_t nch = (n + HUF_CHUNK - 1) / HUF_CHUNK;
    R.builds = (int)nch;

    unsigned cnt[256], maxSym; histo_u(sym, n, cnt, &maxSym);
    unsigned tlog = FSE_optimalTableLog(MAXLOG, n, maxSym);
    short norm[256];
    if (FSE_isError(FSE_normalizeCount(norm, tlog, cnt, n, maxSym))) return R;

    FSE_CTable *ct = NULL; FSE_DTable *dt = NULL;
    uint8_t *enc = NULL, *encp = NULL, *dec = NULL;
    size_t  *off = NULL, *offp = NULL;
    ct = malloc(FSE_CTABLE_SIZE(MAXLOG, 255));
    dt = malloc(FSE_DTABLE_SIZE(MAXLOG));
    if (!ct||!dt) goto fail;
    if (FSE_isError(FSE_buildCTable(ct, norm, maxSym, tlog))) goto fail;
    if (FSE_isError(FSE_buildDTable(dt, norm, maxSym, tlog))) goto fail;
    uint8_t ncbuf[512]; size_t ncSize = FSE_writeNCount(ncbuf, sizeof ncbuf, norm, maxSym, tlog);

    enc = malloc(n + n/2 + 4096);    /* opaque (FSE_compress, w/ NCount) */
    encp= malloc(n + n/2 + 4096);    /* prebuilt (usingCTable, body only) */
    off = malloc((nch+1)*sizeof(size_t));
    offp= malloc((nch+1)*sizeof(size_t));
    dec = malloc(n);
    if (!enc||!encp||!off||!offp||!dec||FSE_isError(ncSize)) goto fail;

    off[0]=0;
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=FSE_compress(enc+off[c], sz+1024, sym+c*HUF_CHUNK, sz);
        if (FSE_isError(r)||r==0) goto fail; off[c+1]=off[c]+r; }
    offp[0]=0;
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=FSE_compress_usingCTable(encp+offp[c], sz+1024, sym+c*HUF_CHUNK, sz, ct);
        if (FSE_isError(r)||r==0) goto fail; offp[c+1]=offp[c]+r; }

    /* correctness: prebuilt decode (usingDTable on body) */
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=FSE_decompress_usingDTable(dec, sz, encp+offp[c], offp[c+1]-offp[c], dt);
        if (FSE_isError(r)||memcmp(sym+c*HUF_CHUNK,dec,sz)!=0){fprintf(stderr,"FSE PB mismatch ch %zu\n",c);goto fail;} }

    double best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        FSE_compress(enc+off[c], sz+1024, sym+c*HUF_CHUNK, sz); } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        FSE_compress_usingCTable(encp+offp[c], sz+1024, sym+c*HUF_CHUNK, sz, ct); } });
    R.enc_pb = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        FSE_decompress(dec, sz, enc+off[c], off[c+1]-off[c]); } });
    R.dec_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        FSE_decompress_usingDTable(dec, sz, encp+offp[c], offp[c+1]-offp[c], dt); } });
    R.dec_pb = best;

    R.ratio_op = (double)n / (double)off[nch];
    R.ratio_pb = (double)n / (double)(offp[nch] + ncSize);
    R.ok = 1;
fail:
    free(ct); free(dt); free(enc); free(encp); free(off); free(offp); free(dec);
    return R;
}

/* ===================== tuned FSE: x8y1 wide-cursor ===================== */
/* Same byte data + 128 KB chunking as measure_fse, but the entropy
 * stage is the x=8 cursors / y=1 unroll decoder (encode_x(8)/decode_x8_y1
 * from fse_xy_codec.h) -- the shape picked by the 2026-05-22 cross-host
 * sweep as "decent but almost always > stock".  Falls back to n/a when
 * P(max symbol) > 50% (FSE_decodeSymbolFast unsafe). */
static result_t measure_fse_tuned(const uint8_t *sym, size_t n) {
    result_t R; memset(&R, 0, sizeof R);
    size_t nch = (n + HUF_CHUNK - 1) / HUF_CHUNK;
    R.builds = (int)nch;
    if (HUF_CHUNK % 8 != 0) return R;          /* x=8 must divide the chunk */

    unsigned gcnt[256], gmax; histo_u(sym, n, gcnt, &gmax);
    unsigned gtlog = FSE_optimalTableLog(MAXLOG, n, gmax);
    short gnorm[256];
    if (FSE_isError(FSE_normalizeCount(gnorm, gtlog, gcnt, n, gmax))) return R;
    int gMaxNorm = 0; for (int i=0;i<=(int)gmax;i++) if (gnorm[i]>gMaxNorm) gMaxNorm=gnorm[i];
    if (gMaxNorm > (1 << (gtlog - 1))) return R;  /* P>50%: tuned path unsafe */

    FSE_CTable *gct = malloc(FSE_CTABLE_SIZE(MAXLOG,255));
    FSE_DTable *gdt = malloc(FSE_DTABLE_SIZE(MAXLOG));
    FSE_CTable *ct  = malloc(FSE_CTABLE_SIZE(MAXLOG,255));
    FSE_DTable *dt  = malloc(FSE_DTABLE_SIZE(MAXLOG));
    short  (*cnorm)[256] = malloc(nch * sizeof *cnorm);
    unsigned *cmax = malloc(nch*sizeof(unsigned)), *ctlog = malloc(nch*sizeof(unsigned));
    uint8_t *enc = malloc(n + n/2 + 4096), *encp = malloc(n + n/2 + 4096), *dec = malloc(n);
    size_t  *off = malloc((nch+1)*sizeof(size_t)), *offp = malloc((nch+1)*sizeof(size_t));
    if (!gct||!gdt||!ct||!dt||!cnorm||!cmax||!ctlog||!enc||!encp||!dec||!off||!offp) goto fail;

    FSE_buildCTable(gct, gnorm, gmax, gtlog);
    FSE_buildDTable(gdt, gnorm, gmax, gtlog);
    uint8_t gnc[512]; size_t gncSize = FSE_writeNCount(gnc, sizeof gnc, gnorm, gmax, gtlog);
    if (FSE_isError(gncSize)) goto fail;

    /* per-chunk normalized counts (opaque) + pre-encode both streams */
    off[0]=offp[0]=0;
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        unsigned cc[256], cm; histo_u(sym+c*HUF_CHUNK, sz, cc, &cm);
        unsigned tl = FSE_optimalTableLog(MAXLOG, sz, cm);
        if (FSE_isError(FSE_normalizeCount(cnorm[c], tl, cc, sz, cm))) goto fail;
        cmax[c]=cm; ctlog[c]=tl;
        FSE_buildCTable(ct, cnorm[c], cm, tl);
        size_t e = encode_x(8, sym+c*HUF_CHUNK, sz, enc+off[c], sz+1024, ct);
        size_t ep= encode_x(8, sym+c*HUF_CHUNK, sz, encp+offp[c], sz+1024, gct);
        if (e==0||ep==0) goto fail;
        off[c+1]=off[c]+e; offp[c+1]=offp[c]+ep;
    }
    /* correctness */
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        FSE_buildDTable(dt, cnorm[c], cmax[c], ctlog[c]);
        if (decode_x8_y1(enc+off[c], off[c+1]-off[c], dec, sz, dt)!=sz || memcmp(sym+c*HUF_CHUNK,dec,sz)){fprintf(stderr,"fse_x8y1 OP mismatch ch %zu\n",c);goto fail;}
        if (decode_x8_y1(encp+offp[c], offp[c+1]-offp[c], dec, sz, gdt)!=sz || memcmp(sym+c*HUF_CHUNK,dec,sz)){fprintf(stderr,"fse_x8y1 PB mismatch ch %zu\n",c);goto fail;}
    }

    double best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        unsigned cc[256], cm; histo_u(sym+c*HUF_CHUNK, sz, cc, &cm);
        unsigned tl=FSE_optimalTableLog(MAXLOG,sz,cm); short nm[256]; FSE_normalizeCount(nm,tl,cc,sz,cm);
        FSE_buildCTable(ct, nm, cm, tl); encode_x(8, sym+c*HUF_CHUNK, sz, enc+off[c], sz+1024, ct); } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        encode_x(8, sym+c*HUF_CHUNK, sz, encp+offp[c], sz+1024, gct); } });
    R.enc_pb = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        FSE_buildDTable(dt, cnorm[c], cmax[c], ctlog[c]); decode_x8_y1(enc+off[c], off[c+1]-off[c], dec, sz, dt); } });
    R.dec_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        decode_x8_y1(encp+offp[c], offp[c+1]-offp[c], dec, sz, gdt); } });
    R.dec_pb = best;

    R.ratio_op = (double)n / (double)(off[nch]  + gncSize * nch);  /* one NCount per chunk */
    R.ratio_pb = (double)n / (double)(offp[nch] + gncSize);        /* one shared NCount */
    R.ok = 1;
fail:
    free(gct); free(gdt); free(ct); free(dt); free(cnorm); free(cmax); free(ctlog);
    free(enc); free(encp); free(dec); free(off); free(offp);
    return R;
}

/* ===================== Oodle (opaque-only reference) ===================== */
#ifdef PIVCO_HAS_OODLE
#include "bench_oodle_wrapper.h"
/* Oodle exposes no prebuilt-table API, so it appears in the opaque
 * columns only; prebuilt fields are left n/a (-1).  Full per-call
 * (header read + table build + decode), per 128 KB chunk. */
static result_t measure_oodle(const uint8_t *sym, size_t n, int is_tans) {
    result_t R; memset(&R, 0, sizeof R);
    R.enc_pb = R.dec_pb = R.ratio_pb = -1.0;     /* no prebuilt mode */
    size_t nch = (n + HUF_CHUNK - 1) / HUF_CHUNK;
    R.builds = (int)nch;
    uint8_t *enc = malloc(n + n/2 + 4096), *dec = malloc(n);
    size_t  *off = malloc((nch+1)*sizeof(size_t));
    int     *ht  = malloc(nch*sizeof(int));
    if (!enc||!dec||!off||!ht) goto fail;

    off[0]=0;
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK; int r;
        if (is_tans) { r = oodle_tans_encode(sym+c*HUF_CHUNK, sz, enc+off[c], sz+1024); ht[c]=0; }
        else         { r = oodle_huff_encode(sym+c*HUF_CHUNK, sz, enc+off[c], sz+1024, &ht[c]); }
        if (r <= 0 || r > (int)sz) goto fail;     /* declined / incompressible / fail */
        off[c+1]=off[c]+(size_t)r;
    }
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        memset(dec,0xCC,sz);
        if (is_tans) oodle_tans_decode(enc+off[c], off[c+1]-off[c], dec, sz);
        else         oodle_huff_decode(enc+off[c], off[c+1]-off[c], dec, sz, ht[c]);
        if (memcmp(sym+c*HUF_CHUNK,dec,sz)!=0){fprintf(stderr,"oodle-%s mismatch ch %zu\n",is_tans?"tans":"huff",c);goto fail;}
    }
    double best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK; int t;
        if (is_tans) oodle_tans_encode(sym+c*HUF_CHUNK, sz, enc+off[c], sz+1024);
        else         oodle_huff_encode(sym+c*HUF_CHUNK, sz, enc+off[c], sz+1024, &t); } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        if (is_tans) oodle_tans_decode(enc+off[c], off[c+1]-off[c], dec, sz);
        else         oodle_huff_decode(enc+off[c], off[c+1]-off[c], dec, sz, ht[c]); } });
    R.dec_op = best;
    R.ratio_op = (double)n / (double)off[nch];
    R.ok = 1;
fail:
    free(enc); free(dec); free(off); free(ht);
    return R;
}
#endif

static void f5(double v) { if (v < 0) printf(" %7s", "  -  "); else printf(" %7.0f", v); }
static void r5(double v) { if (v < 0) printf(" %5s", "  -  "); else printf(" %5.2f", v); }
static void print_row(const char *name, result_t R) {
    if (!R.ok) { printf("%-8s   (n/a)\n", name); return; }
    printf("%-8s |", name);
    f5(R.enc_op); f5(R.enc_pb); printf(" |"); f5(R.dec_op); f5(R.dec_pb);
    printf(" |"); r5(R.ratio_op); r5(R.ratio_pb); printf(" | %3d\n", R.builds);
}

int main(int argc, char **argv) {
    int run_all = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) run_all = 1;
        else if (!strncmp(argv[i], "--G=", 4)) g_table_G = (size_t)atoi(argv[i]+4) * 1024;
    }
    bench_init();
    printf("fair-bench: %d MB-class buffer = %d KB, best of %dx%d, ph table-G=%zu KB, BLK=%d\n",
           TOTAL/(1<<20), TOTAL/1024, RUNS, REPEATS, g_table_G/1024, BLK);
    printf("columns: enc(opaque prebuilt)  dec(opaque prebuilt)  MB/s | ratio(op pb) | builds/1MB\n\n");

    uint8_t *sym = malloc(TOTAL);
    int nd = bench_num_distributions();
    for (int d = 0; d < nd; d++) {
        if (!run_all && !bench_dist_is_main(d)) continue;
        bench_generate_symbols(d, sym, TOTAL, SEED);
        printf("== %-16s ==        enc_op  enc_pb   dec_op  dec_pb |  r_op  r_pb | blds\n", bench_dist_name(d));
        print_row("ph",      measure_ph(sym, TOTAL, 0));
        print_row("pha",     measure_ph(sym, TOTAL, 1));
        print_row("huf0",    measure_huf0(sym, TOTAL));
        print_row("fse_stk", measure_fse(sym, TOTAL));
        print_row("fse_x8y1",measure_fse_tuned(sym, TOTAL));
#ifdef PIVCO_HAS_OODLE
        print_row("oo_huff", measure_oodle(sym, TOTAL, 0));
        print_row("oo_tans", measure_oodle(sym, TOTAL, 1));
#endif
        printf("\n");
    }
    free(sym);
    return 0;
}
