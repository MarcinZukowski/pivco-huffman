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

/* Independent namespaced top-down TD library (phtd_*), for the TD grid.
 * The SIMD ISA macro (PIVCO_HAS_NEON / PIVCO_HAS_AVX512) is passed by
 * CMake to match the ph_td lib build, so phtd.h exposes the right
 * prototypes and the right simd grid rows compile in. */
#include "phtd.h"

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

/* ===================== top-down TD grid (phtd_* lib) ===================== */
/* Generic over (build, encode, decode) so the 2x2 grid -- tree {naive,opt}
 * x prims {scalar,simd} -- reuses one driver.  Mirrors measure_ph's
 * opaque (rebuild table per G window) vs prebuilt (one static table) split,
 * but on the namespaced TD library with its opaque table type. */
typedef int (*phtd_build_fn)(const uint64_t*, phtd_table_t*);
typedef int (*phtd_enc_fn)(const uint8_t*, const phtd_table_t*, uint8_t*, size_t*);
typedef int (*phtd_dec_fn)(const uint8_t*, size_t, const phtd_table_t*, uint8_t*, size_t*);

static result_t measure_phtd(phtd_build_fn B, phtd_enc_fn E, phtd_dec_fn D,
                             const uint8_t *sym, size_t n) {
    result_t R; memset(&R, 0, sizeof R);
    const size_t TB = PHTD_BLOCK_SIZE, tsz = phtd_table_size();
    size_t nblk = n / TB, nwin = n / g_table_G, bpw = g_table_G / TB;
    R.builds = (int)nwin;
    char *gt = malloc(tsz), *wt = malloc(tsz), *wts = malloc(nwin * tsz);
    uint8_t *enc = malloc(n + n/2 + 4096), *eno = malloc(n + n/2 + 4096), *dec = malloc(n);
    size_t *off = malloc((nblk+1)*sizeof(size_t)), *ofo = malloc((nblk+1)*sizeof(size_t));
    if (!gt||!wt||!wts||!enc||!eno||!dec||!off||!ofo) goto done;
#define WT(k) ((phtd_table_t*)(wts + (k)*tsz))
    uint64_t f[256]; histo_u64(sym, n, f);
    if (B(f, (phtd_table_t*)gt) != 0) goto done;
    for (size_t k=0;k<nwin;k++){ uint64_t wf[256]; histo_u64(sym+k*g_table_G, g_table_G, wf);
        if (B(wf, WT(k)) != 0) goto done; }

    off[0]=0; for (size_t b=0;b<nblk;b++){ size_t L=0; if (E(sym+b*TB,(phtd_table_t*)gt,enc+off[b],&L)!=0) goto done; off[b+1]=off[b]+L; }
    ofo[0]=0; for (size_t k=0;k<nwin;k++) for (size_t i=0;i<bpw;i++){ size_t b=k*bpw+i,L=0;
        if (E(sym+b*TB,WT(k),eno+ofo[b],&L)!=0) goto done; ofo[b+1]=ofo[b]+L; }

    for (size_t b=0;b<nblk;b++){ size_t c=0; D(enc+off[b],off[b+1]-off[b],(phtd_table_t*)gt,dec,&c);
        if (memcmp(sym+b*TB,dec,TB)){fprintf(stderr,"phtd PB mismatch blk %zu\n",b);goto done;} }
    for (size_t k=0;k<nwin;k++) for (size_t i=0;i<bpw;i++){ size_t b=k*bpw+i,c=0;
        D(eno+ofo[b],ofo[b+1]-ofo[b],WT(k),dec,&c);
        if (memcmp(sym+b*TB,dec,TB)){fprintf(stderr,"phtd OP mismatch blk %zu\n",b);goto done;} }

    double best;
    BEST_MBPS({ for (size_t b=0;b<nblk;b++){ size_t L=0; E(sym+b*TB,(phtd_table_t*)gt,enc+off[b],&L);} });
    R.enc_pb = best;
    BEST_MBPS({ for (size_t k=0;k<nwin;k++){ uint64_t wf[256]; histo_u64(sym+k*g_table_G,g_table_G,wf); B(wf,(phtd_table_t*)wt);
        for (size_t i=0;i<bpw;i++){ size_t b=k*bpw+i,L=0; E(sym+b*TB,(phtd_table_t*)wt,eno+ofo[b],&L);} } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t b=0;b<nblk;b++){ size_t c=0; D(enc+off[b],off[b+1]-off[b],(phtd_table_t*)gt,dec,&c);} });
    R.dec_pb = best;
    BEST_MBPS({ for (size_t k=0;k<nwin;k++){ uint64_t wf[256]; histo_u64(sym+k*g_table_G,g_table_G,wf); B(wf,(phtd_table_t*)wt);
        for (size_t i=0;i<bpw;i++){ size_t b=k*bpw+i,c=0; D(eno+ofo[b],ofo[b+1]-ofo[b],(phtd_table_t*)wt,dec,&c);} } });
    R.dec_op = best;
    R.ratio_pb = (double)n / (double)(off[nblk] + 128);
    R.ratio_op = (double)n / (double)(ofo[nblk] + 128 * nwin);
    R.ok = 1;
#undef WT
done:
    free(gt); free(wt); free(wts); free(enc); free(eno); free(dec); free(off); free(ofo);
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

/* ===== stock huf0: the top-level one-liner API a user would reach for =====
 * HUF_compress / HUF_decompress (auto-dispatch X1/X2, RLE/uncompressed
 * handling, table built+read per call).  Opaque-only -- the stock API
 * exposes no prebuilt-table path.  Contrast with the `huf0` row above,
 * which is the tuned 4X2 + usingD/CTable path (we gave SoTA every
 * advantage there; this shows the realistic default). */
static result_t measure_huf0_stk(const uint8_t *sym, size_t n) {
    result_t R; memset(&R, 0, sizeof R);
    R.enc_pb = R.dec_pb = R.ratio_pb = -1.0;          /* no prebuilt API */
    size_t nch = (n + HUF_CHUNK - 1) / HUF_CHUNK;
    R.builds = (int)nch;
    uint8_t *enc = malloc(n + n/2 + 4096), *dec = malloc(n);
    size_t  *off = malloc((nch+1)*sizeof(size_t));
    if (!enc||!dec||!off) goto fail;
    off[0]=0;
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=HUF_compress(enc+off[c], sz+1024, sym+c*HUF_CHUNK, sz);
        if (HUF_isError(r)||r==0) goto fail; off[c+1]=off[c]+r; }
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        size_t r=HUF_decompress(dec, sz, enc+off[c], off[c+1]-off[c]);
        if (HUF_isError(r)||memcmp(sym+c*HUF_CHUNK,dec,sz)!=0){fprintf(stderr,"huf0_stk mismatch ch %zu\n",c);goto fail;} }
    double best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        HUF_compress(enc+off[c], sz+1024, sym+c*HUF_CHUNK, sz); } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        HUF_decompress(dec, sz, enc+off[c], off[c+1]-off[c]); } });
    R.dec_op = best;
    R.ratio_op = (double)n / (double)off[nch];
    R.ok = 1;
fail:
    free(enc); free(dec); free(off);
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

/* ---- engine registry: uniform (sym,n)->result_t thunks ---- */
static result_t e_ph (const uint8_t*s,size_t n){ return measure_ph(s,n,0); }
static result_t e_pha(const uint8_t*s,size_t n){ return measure_ph(s,n,1); }
static result_t e_td_naive (const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table_naive, phtd_encode_naive,      phtd_decode_naive,      s,n); }
static result_t e_td_scl   (const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table,       phtd_encode_scalar_opt, phtd_decode_scalar_opt, s,n); }
#if defined(PIVCO_HAS_NEON)
static result_t e_td_nvsimd(const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table_naive, phtd_encode_naive, phtd_decode_naive_simd_neon, s,n); }
static result_t e_td_simdopt(const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table,      phtd_encode_neon,  phtd_decode_neon,            s,n); }
#elif defined(PIVCO_HAS_AVX512)
static result_t e_td_nvsimd(const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table_naive, phtd_encode_naive,  phtd_decode_naive_simd_avx512, s,n); }
static result_t e_td_simdopt(const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table,      phtd_encode_avx512, phtd_decode_avx512,            s,n); }
#endif
static result_t e_huf0    (const uint8_t*s,size_t n){ return measure_huf0(s,n); }
static result_t e_huf0_stk(const uint8_t*s,size_t n){ return measure_huf0_stk(s,n); }
static result_t e_fse_stk (const uint8_t*s,size_t n){ return measure_fse(s,n); }
static result_t e_fse_x8y1(const uint8_t*s,size_t n){ return measure_fse_tuned(s,n); }
#ifdef PIVCO_HAS_OODLE
static result_t e_oo_huff (const uint8_t*s,size_t n){ return measure_oodle(s,n,0); }
static result_t e_oo_tans (const uint8_t*s,size_t n){ return measure_oodle(s,n,1); }
#endif

typedef result_t (*engine_fn)(const uint8_t*, size_t);
static const struct { const char *name; engine_fn fn; } ENGINES[] = {
    {"ph", e_ph}, {"pha", e_pha},
    {"td_naive", e_td_naive}, {"td_scl_opt", e_td_scl},
#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_AVX512)
    {"td_nv_simd", e_td_nvsimd}, {"td_simdopt", e_td_simdopt},
#endif
    {"huf0", e_huf0}, {"huf0_stk", e_huf0_stk}, {"fse_stk", e_fse_stk}, {"fse_x8y1", e_fse_x8y1},
#ifdef PIVCO_HAS_OODLE
    {"oo_huff", e_oo_huff}, {"oo_tans", e_oo_tans},
#endif
};
#define N_ENGINES (int)(sizeof(ENGINES)/sizeof(ENGINES[0]))

/* membership in a comma-separated list; NULL list = match everything */
static int in_csv(const char *csv, const char *name){
    if (!csv) return 1;
    size_t nl = strlen(name);
    for (const char *p = csv; *p; ) {
        const char *c = strchr(p, ',');
        size_t len = c ? (size_t)(c - p) : strlen(p);
        if (len == nl && strncmp(p, name, nl) == 0) return 1;
        p += len; if (*p) p++;
    }
    return 0;
}

int main(int argc, char **argv) {
    int run_all = 0;
    const char *eng_filter = NULL, *dist_filter = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) run_all = 1;
        else if (!strncmp(argv[i], "--G=", 4)) g_table_G = (size_t)atoi(argv[i]+4) * 1024;
        else if (!strncmp(argv[i], "--engines=", 10)) eng_filter = argv[i] + 10;
        else if (!strncmp(argv[i], "--dist=", 7)) { dist_filter = argv[i] + 7; }
        else if (!strcmp(argv[i], "--list") || !strcmp(argv[i], "--help")) {
            bench_init();
            printf("usage: pivco_fair_bench [--all] [--G=KB] [--engines=a,b] [--dist=x,y]\n\n");
            printf("engines:");
            for (int e = 0; e < N_ENGINES; e++) printf(" %s", ENGINES[e].name);
            printf("\n\ndistributions (* = in default 'main' set):\n");
            for (int d = 0; d < bench_num_distributions(); d++)
                printf("  %s%s\n", bench_dist_name(d), bench_dist_is_main(d) ? " *" : "");
            return 0;
        }
    }
    bench_init();
    phtd_set_fse_enabled(0);   /* TD grid: raw bitmaps, isolate tree x prims */
    printf("fair-bench: %d MB-class buffer = %d KB, best of %dx%d, ph table-G=%zu KB, BLK=%d\n",
           TOTAL/(1<<20), TOTAL/1024, RUNS, REPEATS, g_table_G/1024, BLK);
    if (eng_filter)  printf("  engines: %s\n", eng_filter);
    if (dist_filter) printf("  dists:   %s\n", dist_filter);
    printf("columns: enc(opaque prebuilt)  dec(opaque prebuilt)  MB/s | ratio(op pb) | builds/1MB\n\n");

    uint8_t *sym = malloc(TOTAL);
    int nd = bench_num_distributions();
    for (int d = 0; d < nd; d++) {
        int include = dist_filter ? in_csv(dist_filter, bench_dist_name(d))
                                  : (run_all || bench_dist_is_main(d));
        if (!include) continue;
        bench_generate_symbols(d, sym, TOTAL, SEED);
        printf("== %-16s ==        enc_op  enc_pb   dec_op  dec_pb |  r_op  r_pb | blds\n", bench_dist_name(d));
        for (int e = 0; e < N_ENGINES; e++)
            if (in_csv(eng_filter, ENGINES[e].name))
                print_row(ENGINES[e].name, ENGINES[e].fn(sym, TOTAL));
        printf("\n");
    }
    free(sym);
    return 0;
}
