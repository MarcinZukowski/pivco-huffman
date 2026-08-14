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
 * Methodology: adaptive best-of-N x REPEATS passes over the 1 MB buffer.
 *   Start with 20 samples (each = 20 inner passes); if the top two are
 *   within 2%, stop and report the best.  Otherwise add 2 more samples
 *   and re-check, up to 40 samples total.
 * Reports enc/dec MB/s (input bytes) for each mode + compression ratio
 * + table builds per 1 MB.  Oodle columns (opaque-only) added separately
 * under PIVCO_HAS_OODLE.
 */

#include <stdio.h>
#include "bench_ctx.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* Optional cycle-counter timer (Linux only; falls back to wall-clock
 * on hosts without perf_event_open).  Enable with `--timer=cyc`.  The
 * BEST_MBPS macro then computes "MB/Gcyc" instead of MB/s — useful for
 * frequency-stable A/B regardless of turbo / DVFS. */
#if defined(__linux__)
#  include <errno.h>
#  include <unistd.h>
#  include <linux/perf_event.h>
#  include <sys/syscall.h>
#  include <sys/ioctl.h>
#endif

#include "pivco_huffman.h"
#include "bench_canary.h"
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
extern int          bench_dist_size(int dist_idx, int min_n, int block_align);

/* ---- config ---- */
#define TOTAL      (1 << 20)            /* MIN buffer size (1 MB).  For
                                           real-file datasets the natural
                                           size may be larger (file used
                                           as-is if >= 1 MB; cycled to
                                           >= 1 MB if smaller). */
#define TOTAL_MAX  (16 * 1024 * 1024)   /* generous alloc ceiling */
/* ph decode sub-block.  Runtime-selectable via --blk= (the codec reads N
 * from the wire header, so no recompile is needed); defaults to the
 * compiled PIVCO_BLOCK_SIZE.  Capped at PIVCO_BLOCK_SIZE because the codec's
 * internal scratch buffers are sized at compile time. */
static size_t g_blk = PIVCO_BLOCK_SIZE;
#define BLK        g_blk
#define HUF_CHUNK  (128 * 1024)         /* huf0 hard cap; FSE controlled chunk */

/* Per-distribution buffer size, set in main() before each engine row is
 * measured.  Used by BEST_MBPS for the MB/s conversion. */
static size_t g_total = TOTAL;
#define RUNS       10
#define REPEATS    20
#define SEED       0xBEEFCAFE12345678ULL
#define MAXLOG     12                   /* FSE tableLog (tANS state-table log) */
#define HUFLOG     PIVCO_MAX_CODE_LEN    /* HUF max code length = ph's budget (11);
                                            12 == HUF_TABLELOG_MAX, where a 1-bit
                                            dominant code yields an invalid weight-12
                                            header that HUF_readStats rejects */

static size_t g_table_G = 128 * 1024;   /* ph table-refresh granularity */

static double now_ns_wall(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

#if defined(__linux__)
static int    g_perf_fd  = -1;
static int    g_use_cyc  = 0;   /* set by --timer=cyc */

static void perf_init_cyc(void) {
    struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_HARDWARE; pe.size = sizeof pe;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 0;  /* count continuously; we read deltas */
    pe.exclude_kernel = 1; pe.exclude_hv = 1;
    g_perf_fd = (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    if (g_perf_fd < 0) {
        fprintf(stderr, "perf_event_open failed (%s); falling back to wall-clock\n",
                strerror(errno));
        g_use_cyc = 0;
    }
}
static inline double now_ticks(void) {
    if (g_use_cyc && g_perf_fd >= 0) {
        uint64_t v = 0;
        if (read(g_perf_fd, &v, sizeof v) == sizeof v) return (double)v;
    }
    return now_ns_wall();
}
#else
static inline void perf_init_cyc(void) {
    fprintf(stderr, "--timer=cyc not supported on this OS; using wall-clock\n");
}
static int g_use_cyc = 0;
static inline double now_ticks(void) { return now_ns_wall(); }
#endif

/* now_ns(): returns "ticks" — either real ns (wall-clock) or CPU
 * cycles (when --timer=cyc).  The BEST_MBPS formula is identical
 * in both modes; reported units are MB/s vs MB/Gcyc. */
static inline double now_ns(void) { return now_ticks(); }

/* Adaptive best-MB/s measurement: each sample times REPEATS consecutive
 * passes over the buffer; collect samples; keep going until the top two
 * samples differ by <= 2% (good convergence) or we hit _BMA_MAX.
 *
 * Schedule: 20 samples up front, then 2 more at a time up to 40.  Budget
 * skewed toward outer samples (independent draws) rather than inner reps
 * (which past ~10 just amortize the same jitter on a longer window). */
#define _BMA_MIN     20
#define _BMA_STEP    2
#define _BMA_MAX     40
#define _BMA_TOL     0.02
#define BEST_MBPS(...) do {                                             \
        double _bma_smp[_BMA_MAX];                                      \
        int    _bma_n = 0;                                              \
        for (int _r = 0; _r < _BMA_MIN; _r++) {                         \
            double _t0 = now_ns();                                      \
            for (int _rep = 0; _rep < REPEATS; _rep++) { __VA_ARGS__; } \
            double _el = now_ns() - _t0;                                \
            _bma_smp[_bma_n++] = 1000.0 * (double)g_total * REPEATS / _el;\
        }                                                               \
        for (;;) {                                                      \
            double _m1 = 0, _m2 = 0;                                    \
            for (int _i = 0; _i < _bma_n; _i++) {                       \
                double _v = _bma_smp[_i];                               \
                if      (_v > _m1) { _m2 = _m1; _m1 = _v; }             \
                else if (_v > _m2) { _m2 = _v; }                        \
            }                                                           \
            int _converged = (_m1 > 0 &&                                \
                              (_m1 - _m2) <= _BMA_TOL * _m1);           \
            if (_converged || _bma_n >= _BMA_MAX) { best = _m1; break; }\
            for (int _r = 0; _r < _BMA_STEP && _bma_n < _BMA_MAX; _r++){\
                double _t0 = now_ns();                                  \
                for (int _rep = 0; _rep < REPEATS; _rep++) { __VA_ARGS__; }\
                double _el = now_ns() - _t0;                            \
                _bma_smp[_bma_n++] = 1000.0 * (double)g_total * REPEATS / _el;\
            }                                                           \
        }                                                               \
    } while (0)

/* 4 interleaved partial tables: breaks the same-counter store->load-forward
 * dependency chain that throttles a single-table histogram on skewed data
 * (e.g. proba80, byte0 ~80%: 1-table ~0.6 GB/s -> 4-table ~2.3-2.6 GB/s; 3-4x).
 * 4 cursors is the cross-uarch sweet spot (8 regresses on Xeon/Graviton); the
 * FSE word-load + read-ahead trick adds only marginal gains on non-skewed data,
 * so we keep the simple byte form.  Dominates enc_op setup on skewed inputs. */
static void histo_u64(const uint8_t *s, size_t n, uint64_t f[256]) {
    uint64_t f0[256]={0}, f1[256]={0}, f2[256]={0}, f3[256]={0};
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        f0[s[i]]++; f1[s[i+1]]++; f2[s[i+2]]++; f3[s[i+3]]++;
    }
    for (; i < n; i++) f0[s[i]]++;
    for (int b = 0; b < 256; b++) f[b] = f0[b] + f1[b] + f2[b] + f3[b];
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
static result_t measure_ph(const uint8_t *sym, size_t n, int fse_on,
                             pivco_tree_mode_t tree_mode) {
    result_t R; memset(&R, 0, sizeof R);
    size_t nblk = n / BLK;
    size_t G    = g_table_G;
    size_t nwin = (n + G - 1) / G;    /* ceiling: last window may be short */
    size_t bpw  = G / BLK;            /* full-window sub-block count */
    /* Per-window block count: bpw for all but possibly the last. */
    #define WBPW(w) (((w) + 1 < nwin) ? bpw : (nblk - (w) * bpw))
    #define WSZ(w)  (((w) + 1 < nwin) ? G   : (n    - (w) * G))
    R.builds = (int)nwin;

    bench_cfg()->fse_enabled = (fse_on);
    bench_cfg()->tree_mode = (tree_mode);

    pivco_table_t *gtbl = NULL, *wtbl = NULL, *wtbls = NULL;
    uint8_t (*win_clen)[256] = NULL;
    uint8_t *enc = NULL, *enco = NULL, *dec = NULL;
    size_t  *off = NULL, *offo = NULL;

    gtbl = malloc(sizeof *gtbl);
    wtbl = malloc(sizeof *wtbl);
    if (!gtbl || !wtbl) goto done_fail;
    uint64_t f[256];
    histo_u64(sym, n, f);
    if (pivco_build_table(bench_cfg(), f, gtbl) != 0) goto done_fail;

    /* per-window tables + their code_lens, for opaque enc/dec */
    win_clen = malloc(nwin * 256);
    wtbls = malloc(nwin * sizeof *wtbls);
    if (!win_clen || !wtbls) goto done_fail;
    for (size_t w = 0; w < nwin; w++) {
        uint64_t wf[256]; histo_u64(sym + w * G, WSZ(w), wf);
        if (pivco_build_table(bench_cfg(), wf, &wtbls[w]) != 0) goto done_fail;
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
        if (pivco_encode(bench_enc_ctx(), gtbl, sym + b * BLK, BLK, enc + off[b], &L) != 0) goto done_fail;
        off[b + 1] = off[b] + L;
    }
    /* pre-encode opaque stream (per-window tables) */
    offo[0] = 0;
    for (size_t w = 0; w < nwin; w++)
        for (size_t i = 0, wb = WBPW(w); i < wb; i++) {
            size_t b = w * bpw + i, L = 0;
            if (pivco_encode(bench_enc_ctx(), &wtbls[w], sym + b * BLK, BLK, enco + offo[b], &L) != 0) goto done_fail;
            offo[b + 1] = offo[b] + L;
        }

    /* correctness check (prebuilt + opaque) */
    for (size_t b = 0; b < nblk; b++) {
        size_t c = 0;
        pivco_decode(bench_dec_ctx(), gtbl, enc + off[b], off[b+1]-off[b], dec, &c);
        if (memcmp(sym + b * BLK, dec, BLK) != 0) { fprintf(stderr,"ph PB mismatch blk %zu\n",b); goto done_fail; }
    }
    for (size_t w = 0; w < nwin; w++)
        for (size_t i = 0, wb = WBPW(w); i < wb; i++) {
            size_t b = w*bpw+i, c = 0;
            pivco_decode(bench_dec_ctx(), &wtbls[w], enco + offo[b], offo[b+1]-offo[b], dec, &c);
            if (memcmp(sym + b * BLK, dec, BLK) != 0) { fprintf(stderr,"ph OP mismatch blk %zu\n",b); goto done_fail; }
        }

    double best;
    /* ---- encode prebuilt: global table, just emit ---- */
    BEST_MBPS({
        for (size_t b = 0; b < nblk; b++) { size_t L=0; pivco_encode(bench_enc_ctx(), gtbl, sym + b*BLK, BLK, enc + off[b], &L); }
    });
    R.enc_pb = best;
    /* ---- encode opaque: rebuild table per window + emit ---- */
    BEST_MBPS({
        for (size_t w = 0; w < nwin; w++) {
            uint64_t wf[256]; histo_u64(sym + w*G, WSZ(w), wf);
            pivco_build_table(bench_cfg(), wf, wtbl);
            for (size_t i = 0, wb = WBPW(w); i < wb; i++) { size_t b=w*bpw+i, L=0; pivco_encode(bench_enc_ctx(), wtbl, sym + b*BLK, BLK, enco + offo[b], &L); }
        }
    });
    R.enc_op = best;
    /* ---- decode prebuilt: global table ---- */
    BEST_MBPS({
        for (size_t b = 0; b < nblk; b++) { size_t c=0; pivco_decode(bench_dec_ctx(), gtbl, enc + off[b], off[b+1]-off[b], dec, &c); }
    });
    R.dec_pb = best;
    /* ---- decode opaque: rebuild table-from-codelens per window ---- */
    BEST_MBPS({
        for (size_t w = 0; w < nwin; w++) {
            pivco_build_table_from_code_lens(bench_cfg(), win_clen[w], wtbl);
            for (size_t i = 0, wb = WBPW(w); i < wb; i++) { size_t b=w*bpw+i, c=0; pivco_decode(bench_dec_ctx(), wtbl, enco + offo[b], offo[b+1]-offo[b], dec, &c); }
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
#undef WBPW
#undef WSZ

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
    size_t G = g_table_G;
    size_t nblk = n / TB, nwin = (n + G - 1) / G, bpw = G / TB;
    #define WBPW(w) (((w) + 1 < nwin) ? bpw : (nblk - (w) * bpw))
    #define WSZ(w)  (((w) + 1 < nwin) ? G   : (n    - (w) * G))
    R.builds = (int)nwin;
    char *gt = malloc(tsz), *wt = malloc(tsz), *wts = malloc(nwin * tsz);
    uint8_t *enc = malloc(n + n/2 + 4096), *eno = malloc(n + n/2 + 4096), *dec = malloc(n);
    size_t *off = malloc((nblk+1)*sizeof(size_t)), *ofo = malloc((nblk+1)*sizeof(size_t));
    if (!gt||!wt||!wts||!enc||!eno||!dec||!off||!ofo) goto done;
#define WT(k) ((phtd_table_t*)(wts + (k)*tsz))
    uint64_t f[256]; histo_u64(sym, n, f);
    if (B(f, (phtd_table_t*)gt) != 0) goto done;
    for (size_t k=0;k<nwin;k++){ uint64_t wf[256]; histo_u64(sym+k*G, WSZ(k), wf);
        if (B(wf, WT(k)) != 0) goto done; }

    off[0]=0; for (size_t b=0;b<nblk;b++){ size_t L=0; if (E(sym+b*TB,(phtd_table_t*)gt,enc+off[b],&L)!=0) goto done; off[b+1]=off[b]+L; }
    ofo[0]=0; for (size_t k=0;k<nwin;k++) for (size_t i=0,wb=WBPW(k);i<wb;i++){ size_t b=k*bpw+i,L=0;
        if (E(sym+b*TB,WT(k),eno+ofo[b],&L)!=0) goto done; ofo[b+1]=ofo[b]+L; }

    for (size_t b=0;b<nblk;b++){ size_t c=0; D(enc+off[b],off[b+1]-off[b],(phtd_table_t*)gt,dec,&c);
        if (memcmp(sym+b*TB,dec,TB)){fprintf(stderr,"phtd PB mismatch blk %zu\n",b);goto done;} }
    for (size_t k=0;k<nwin;k++) for (size_t i=0,wb=WBPW(k);i<wb;i++){ size_t b=k*bpw+i,c=0;
        D(eno+ofo[b],ofo[b+1]-ofo[b],WT(k),dec,&c);
        if (memcmp(sym+b*TB,dec,TB)){fprintf(stderr,"phtd OP mismatch blk %zu\n",b);goto done;} }

    double best;
    BEST_MBPS({ for (size_t b=0;b<nblk;b++){ size_t L=0; E(sym+b*TB,(phtd_table_t*)gt,enc+off[b],&L);} });
    R.enc_pb = best;
    BEST_MBPS({ for (size_t k=0;k<nwin;k++){ uint64_t wf[256]; histo_u64(sym+k*G,WSZ(k),wf); B(wf,(phtd_table_t*)wt);
        for (size_t i=0,wb=WBPW(k);i<wb;i++){ size_t b=k*bpw+i,L=0; E(sym+b*TB,(phtd_table_t*)wt,eno+ofo[b],&L);} } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t b=0;b<nblk;b++){ size_t c=0; D(enc+off[b],off[b+1]-off[b],(phtd_table_t*)gt,dec,&c);} });
    R.dec_pb = best;
    /* opaque decode = pure decode against the per-window tables (built once, up
     * front -- NOT inside the timer).  A decoder never re-histograms or rebuilds
     * the tree from frequencies, so that work must not pollute the decode timer.
     * (Tree-construction cost is measured separately, in a dedicated bench.) */
    BEST_MBPS({ for (size_t k=0;k<nwin;k++){ for (size_t i=0,wb=WBPW(k);i<wb;i++){ size_t b=k*bpw+i,c=0; D(eno+ofo[b],ofo[b+1]-ofo[b],WT(k),dec,&c);} } });
    R.dec_op = best;
    R.ratio_pb = (double)n / (double)(off[nblk] + 128);
    R.ratio_op = (double)n / (double)(ofo[nblk] + 128 * nwin);
    R.ok = 1;
#undef WT
#undef WBPW
#undef WSZ
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
    size_t huffLog = HUF_buildCTable(ctable, cnt, maxSym, HUFLOG);  /* returns actual maxNbBits */
    if (HUF_isError(huffLog)) return R;

    uint8_t *enc = malloc(n + n/2 + 4096);        /* opaque stream (HUF_compress, w/ header) */
    uint8_t *encp= malloc(n + n/2 + 4096);        /* prebuilt stream (usingCTable body only) */
    size_t  *off = malloc((nch+1)*sizeof(size_t));
    size_t  *offp= malloc((nch+1)*sizeof(size_t));
    uint8_t *dec = malloc(n);
    void    *wksp= malloc(1<<16);
    HUF_DTable *dt   = malloc(HUF_DTABLE_SIZE(HUFLOG) * sizeof(HUF_DTable)); /* opaque scratch */
    HUF_DTable *dtpb = malloc(HUF_DTABLE_SIZE(HUFLOG) * sizeof(HUF_DTable)); /* prebuilt, global */
    uint8_t  hdr[512]; size_t hdrSize = HUF_writeCTable(hdr, sizeof hdr, ctable, maxSym, (unsigned)huffLog);
    if (!enc||!encp||!off||!offp||!dec||!wksp||!dt||!dtpb||HUF_isError(hdrSize)) goto fail;
    dt[0]   = (HUF_DTable)(HUFLOG * 0x01000001);   /* set max table log */
    dtpb[0] = (HUF_DTable)(HUFLOG * 0x01000001);

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
    size_t CHK = HUF_CHUNK;
    size_t nch = (n + CHK - 1) / CHK;
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
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?CHK:n-c*CHK;
        size_t r=FSE_compress(enc+off[c], sz+1024, sym+c*CHK, sz);
        if (FSE_isError(r)||r==0) goto fail; off[c+1]=off[c]+r; }
    offp[0]=0;
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?CHK:n-c*CHK;
        size_t r=FSE_compress_usingCTable(encp+offp[c], sz+1024, sym+c*CHK, sz, ct);
        if (FSE_isError(r)||r==0) goto fail; offp[c+1]=offp[c]+r; }

    /* correctness: prebuilt decode (usingDTable on body) */
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?CHK:n-c*CHK;
        size_t r=FSE_decompress_usingDTable(dec, sz, encp+offp[c], offp[c+1]-offp[c], dt);
        if (FSE_isError(r)||memcmp(sym+c*CHK,dec,sz)!=0){fprintf(stderr,"FSE PB mismatch ch %zu\n",c);goto fail;} }

    double best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?CHK:n-c*CHK;
        FSE_compress(enc+off[c], sz+1024, sym+c*CHK, sz); } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?CHK:n-c*CHK;
        FSE_compress_usingCTable(encp+offp[c], sz+1024, sym+c*CHK, sz, ct); } });
    R.enc_pb = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?CHK:n-c*CHK;
        FSE_decompress(dec, sz, enc+off[c], off[c+1]-off[c]); } });
    R.dec_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?CHK:n-c*CHK;
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
/* Is an FSE table fast-mode-safe?  decode_x8_y1 uses FSE_decodeSymbolFast,
 * which is UB on a zero-bit read -- a DTable entry gets nbBits==0 exactly
 * when a symbol's normalized count >= tableSize/2 (the same condition
 * FSE_buildDTable uses to clear its own fastMode flag).  Mirror it
 * precisely (>=, not >): e.g. `geometric`'s top symbol lands at exactly
 * 2^(tableLog-1) and would slip past a `>` check, then segfault. */
static int fse_norm_fast_safe(const short *norm, unsigned maxSym, unsigned tlog) {
    int lim = 1 << (tlog - 1);
    for (unsigned i = 0; i <= maxSym; i++) if (norm[i] >= lim) return 0;
    return 1;
}

/* Encode/decode one chunk: wide x8y1 when the table is fast-safe, else
 * fall back to stock FSE (which handles zero-bit symbols correctly).
 * enc returns 0 on error/incompressible so callers can bail. */
static size_t enc_chunk(int safe, const uint8_t *src, size_t sz,
                        uint8_t *dst, size_t cap, const FSE_CTable *ct) {
    if (safe) return encode_x(8, src, sz, dst, cap, ct);
    size_t r = FSE_compress_usingCTable(dst, cap, src, sz, ct);
    return (FSE_isError(r) || r < 2) ? 0 : r;
}
static size_t dec_chunk(int safe, const void *src, size_t srclen,
                        uint8_t *dst, size_t sz, const FSE_DTable *dt) {
    return safe ? decode_x8_y1(src, srclen, dst, sz, dt)
                : FSE_decompress_usingDTable(dst, sz, src, srclen, dt);
}

/* Same byte data + 128 KB chunking as measure_fse, but the entropy
 * stage is the x=8 cursors / y=1 unroll decoder (encode_x(8)/decode_x8_y1
 * from fse_xy_codec.h) -- the shape picked by the 2026-05-22 cross-host
 * sweep as "decent but almost always > stock".  Per-table fast-safe gate:
 * tables with a >=50% symbol (FSE_decodeSymbolFast unsafe) fall back to
 * stock FSE for that chunk instead of n/a-ing the whole engine. */
static result_t measure_fse_tuned(const uint8_t *sym, size_t n) {
    result_t R; memset(&R, 0, sizeof R);
    size_t nch = (n + HUF_CHUNK - 1) / HUF_CHUNK;
    R.builds = (int)nch;
    if (HUF_CHUNK % 8 != 0) return R;          /* x=8 must divide the chunk */

    unsigned gcnt[256], gmax; histo_u(sym, n, gcnt, &gmax);
    unsigned gtlog = FSE_optimalTableLog(MAXLOG, n, gmax);
    short gnorm[256];
    if (FSE_isError(FSE_normalizeCount(gnorm, gtlog, gcnt, n, gmax))) return R;
    int safe_pb = fse_norm_fast_safe(gnorm, gmax, gtlog);  /* prebuilt-table path */

    FSE_CTable *gct = malloc(FSE_CTABLE_SIZE(MAXLOG,255));
    FSE_DTable *gdt = malloc(FSE_DTABLE_SIZE(MAXLOG));
    FSE_CTable *ct  = malloc(FSE_CTABLE_SIZE(MAXLOG,255));
    FSE_DTable *dt  = malloc(FSE_DTABLE_SIZE(MAXLOG));
    short  (*cnorm)[256] = malloc(nch * sizeof *cnorm);
    unsigned *cmax = malloc(nch*sizeof(unsigned)), *ctlog = malloc(nch*sizeof(unsigned));
    int *safe_op = malloc(nch*sizeof(int));     /* per-chunk opaque-table fast-safe */
    uint8_t *enc = malloc(n + n/2 + 4096), *encp = malloc(n + n/2 + 4096), *dec = malloc(n);
    size_t  *off = malloc((nch+1)*sizeof(size_t)), *offp = malloc((nch+1)*sizeof(size_t));
    if (!gct||!gdt||!ct||!dt||!cnorm||!cmax||!ctlog||!safe_op||!enc||!encp||!dec||!off||!offp) goto fail;

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
        safe_op[c] = fse_norm_fast_safe(cnorm[c], cm, tl);
        FSE_buildCTable(ct, cnorm[c], cm, tl);
        size_t e = enc_chunk(safe_op[c], sym+c*HUF_CHUNK, sz, enc+off[c], sz+1024, ct);
        size_t ep= enc_chunk(safe_pb,    sym+c*HUF_CHUNK, sz, encp+offp[c], sz+1024, gct);
        if (e==0||ep==0) goto fail;
        off[c+1]=off[c]+e; offp[c+1]=offp[c]+ep;
    }
    /* correctness */
    for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        FSE_buildDTable(dt, cnorm[c], cmax[c], ctlog[c]);
        if (dec_chunk(safe_op[c], enc+off[c], off[c+1]-off[c], dec, sz, dt)!=sz || memcmp(sym+c*HUF_CHUNK,dec,sz)){fprintf(stderr,"fse_x8y1 OP mismatch ch %zu\n",c);goto fail;}
        if (dec_chunk(safe_pb, encp+offp[c], offp[c+1]-offp[c], dec, sz, gdt)!=sz || memcmp(sym+c*HUF_CHUNK,dec,sz)){fprintf(stderr,"fse_x8y1 PB mismatch ch %zu\n",c);goto fail;}
    }

    double best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        unsigned cc[256], cm; histo_u(sym+c*HUF_CHUNK, sz, cc, &cm);
        unsigned tl=FSE_optimalTableLog(MAXLOG,sz,cm); short nm[256]; FSE_normalizeCount(nm,tl,cc,sz,cm);
        FSE_buildCTable(ct, nm, cm, tl); enc_chunk(safe_op[c], sym+c*HUF_CHUNK, sz, enc+off[c], sz+1024, ct); } });
    R.enc_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        enc_chunk(safe_pb, sym+c*HUF_CHUNK, sz, encp+offp[c], sz+1024, gct); } });
    R.enc_pb = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        FSE_buildDTable(dt, cnorm[c], cmax[c], ctlog[c]); dec_chunk(safe_op[c], enc+off[c], off[c+1]-off[c], dec, sz, dt); } });
    R.dec_op = best;
    BEST_MBPS({ for (size_t c=0;c<nch;c++){ size_t sz=(c<nch-1)?HUF_CHUNK:n-c*HUF_CHUNK;
        dec_chunk(safe_pb, encp+offp[c], offp[c+1]-offp[c], dec, sz, gdt); } });
    R.dec_pb = best;

    R.ratio_op = (double)n / (double)(off[nch]  + gncSize * nch);  /* one NCount per chunk */
    R.ratio_pb = (double)n / (double)(offp[nch] + gncSize);        /* one shared NCount */
    R.ok = 1;
fail:
    free(gct); free(gdt); free(ct); free(dt); free(cnorm); free(cmax); free(ctlog);
    free(safe_op);
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
static result_t e_ph       (const uint8_t*s,size_t n){ return measure_ph(s,n,0,PIVCO_TREE_MODE_OPTIMIZED); }
static result_t e_pha      (const uint8_t*s,size_t n){ return measure_ph(s,n,1,PIVCO_TREE_MODE_OPTIMIZED); }
static result_t e_ph_naive (const uint8_t*s,size_t n){ return measure_ph(s,n,0,PIVCO_TREE_MODE_NAIVE); }
static result_t e_ph_flat  (const uint8_t*s,size_t n){ return measure_ph(s,n,0,PIVCO_TREE_MODE_CANONICAL_FLAT); }
static result_t e_td_naive (const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table_naive, phtd_encode_naive,      phtd_decode_naive,      s,n); }
static result_t e_td_scl   (const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table,       phtd_encode_scalar_opt, phtd_decode_scalar_opt, s,n); }
#if defined(PIVCO_HAS_NEON)
static result_t e_td_nvsimd(const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table_naive, phtd_encode_naive, phtd_decode_naive_simd_neon, s,n); }
static result_t e_td_simdopt(const uint8_t*s,size_t n){ return measure_phtd(phtd_build_table,      phtd_encode_neon,  phtd_decode_neon,            s,n); }
#elif defined(PIVCO_HAS_AVX512)
/* The CMake AVX-512 probe is compile-time; the run host may still lack
 * the hardware (e.g. Zen 3).  Guard at runtime so those rows show n/a
 * instead of SIGILL. */
static int avx512_runtime_ok(void){ return __builtin_cpu_supports("avx512vbmi2"); }
static result_t e_td_nvsimd(const uint8_t*s,size_t n){ result_t na={0};
    return avx512_runtime_ok() ? measure_phtd(phtd_build_table_naive, phtd_encode_naive,  phtd_decode_naive_simd_avx512, s,n) : na; }
static result_t e_td_simdopt(const uint8_t*s,size_t n){ result_t na={0};
    return avx512_runtime_ok() ? measure_phtd(phtd_build_table,      phtd_encode_avx512, phtd_decode_avx512,            s,n) : na; }
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
    /* Tree-shape ablation variants — same codec as `ph`, different chunk-
     * decomposition at build_table.  See bench_cfg()->tree_mode = ().
     * Used by paper/plots/tree_modes_*.svg and <tab-tree-modes>. */
    {"ph_naive", e_ph_naive}, {"ph_flat", e_ph_flat},
    {"td_naive", e_td_naive}, {"td_scl_opt", e_td_scl},
#if defined(PIVCO_HAS_NEON) || defined(PIVCO_HAS_AVX512)
    {"td_nv_simd", e_td_nvsimd}, {"td_simdopt", e_td_simdopt},
#endif
    /* "huf0" is stock HUF_decompress (auto-dispatch) -- the canonical baseline.
     * "huf0_4x2" is forced HUF_decompress4X2, kept for ad-hoc reference only
     * (excluded from the reported set; stock is equal-or-better on all hosts). */
    {"huf0", e_huf0_stk}, {"huf0_4x2", e_huf0}, {"fse_stk", e_fse_stk}, {"fse_x8y1", e_fse_x8y1},
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
    int run_all = 0, canary = 1;
    const char *eng_filter = NULL, *dist_filter = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) run_all = 1;
        else if (!strncmp(argv[i], "--canary=", 9)) canary = atoi(argv[i]+9);
        else if (!strncmp(argv[i], "--G=", 4)) g_table_G = (size_t)atoi(argv[i]+4) * 1024;
        else if (!strncmp(argv[i], "--blk=", 6)) {
            long v = atol(argv[i] + 6);
            if (v < 1 || v > PIVCO_WIRE_MAX_N) {
                fprintf(stderr, "--blk must be in 1..%d (uint16 wire N limit)\n",
                        PIVCO_WIRE_MAX_N);
                return 1;
            }
            g_blk = (size_t)v;
        }
        else if (!strncmp(argv[i], "--engines=", 10)) eng_filter = argv[i] + 10;
        else if (!strncmp(argv[i], "--dist=", 7)) { dist_filter = argv[i] + 7; }
        else if (!strcmp(argv[i], "--timer=cyc")) {
#if defined(__linux__)
            g_use_cyc = 1;
#else
            fprintf(stderr, "--timer=cyc requires Linux perf_event_open; ignoring\n");
#endif
        }
        else if (!strcmp(argv[i], "--list") || !strcmp(argv[i], "--help")
                 || !strcmp(argv[i], "-h")) {
            bench_init();
            printf("usage: pivco_fair_bench [--all] [--G=KB] [--blk=N] [--engines=a,b] [--dist=x,y] [--timer=cyc] [--canary=0|1]\n\n");
            printf("engines:");
            for (int e = 0; e < N_ENGINES; e++) printf(" %s", ENGINES[e].name);
            printf("\n\ndistributions (* = in default 'main' set):\n");
            for (int d = 0; d < bench_num_distributions(); d++)
                printf("  %s%s\n", bench_dist_name(d), bench_dist_is_main(d) ? " *" : "");
            return 0;
        }
        else {
            fprintf(stderr, "unknown arg: %s (try --help)\n", argv[i]);
            return 1;
        }
    }
    bench_init();
#if defined(__linux__)
    if (g_use_cyc) perf_init_cyc();
#endif
    phtd_set_fse_enabled(0);   /* TD grid: raw bitmaps, isolate tree x prims */
    printf("fair-bench: buffer >= %d KB (real-file dists may be larger), "
           "adaptive %d-%d runs x %d reps (top-2 within %d%% stops), "
           "ph table-G=%zu KB, BLK=%zu  timer=%s\n",
           TOTAL/1024, _BMA_MIN, _BMA_MAX, REPEATS,
           (int)(_BMA_TOL*100), g_table_G/1024, (size_t)BLK,
           g_use_cyc ? "CPU_CYCLES (units: MB/Gcyc)" : "wall ns (units: MB/s)");
    if (eng_filter)  printf("  engines: %s\n", eng_filter);
    if (dist_filter) printf("  dists:   %s\n", dist_filter);
    printf("columns: enc(opaque prebuilt)  dec(opaque prebuilt)  %s | ratio(op pb) | builds/1MB\n\n",
           g_use_cyc ? "MB/Gcyc" : "MB/s");

    if (canary >= 1) bench_canary("start");
    uint8_t *sym = malloc(TOTAL_MAX);
    int nd = bench_num_distributions();
    for (int d = 0; d < nd; d++) {
        int include = dist_filter ? in_csv(dist_filter, bench_dist_name(d))
                                  : (run_all || bench_dist_is_main(d));
        if (!include) continue;
        /* Align only to the codec's sub-block size BLK.  measure_ph /
           measure_phtd handle a possibly-short last window via WBPW/WSZ. */
        int align = (int)BLK;
        int n = bench_dist_size(d, TOTAL, align);
        if (n > TOTAL_MAX) n = TOTAL_MAX - (TOTAL_MAX % align);
        g_total = (size_t)n;
        bench_generate_symbols(d, sym, n, SEED);
        printf("== %-16s == n=%d KB enc_op  enc_pb   dec_op  dec_pb |  r_op  r_pb | blds\n",
               bench_dist_name(d), n/1024);
        for (int e = 0; e < N_ENGINES; e++)
            if (in_csv(eng_filter, ENGINES[e].name))
                print_row(ENGINES[e].name, ENGINES[e].fn(sym, (size_t)n));
        printf("\n");
    }
    if (canary >= 1) { bench_canary("end"); bench_canary_summary(); }
    free(sym);
    return 0;
}
