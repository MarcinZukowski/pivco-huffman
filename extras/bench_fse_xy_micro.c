/* FSE cursor-count × unroll-factor microbench (standalone).
 *
 * Hypothesis being tested: FSE's existing decoder (x=2, y=2 — 2
 * interleaved FSE_DState_t advancing in the hot loop, 4 symbols
 * per loop body) saturates the per-symbol ILP at our typical
 * bitmap sizes.  Adding more cursors (x) should add independent
 * dep chains the OOO core can pipeline; adding more unroll (y)
 * should reduce per-iter loop overhead but not add ILP.
 *
 * Sweep: x ∈ {2, 4, 6, 8, 10, 12, 16} × y ∈ {1, 2, 4}.  x=1 is
 * omitted because it needs a different post-overflow tail
 * pattern (no "other cursor" to decode after the main reload
 * signals overflow); not worth the special case since it'd just
 * be a slower x=2 anyway.
 *
 * Pure microbench -- doesn't touch the codec wire format or any
 * runtime API.  Hand-rolled encoder + decoder over FSE's
 * static-inline primitives (FSE_initCState / FSE_encodeSymbol /
 * FSE_flushCState + BIT_initCStream / BIT_addBits /
 * BIT_flushBits / BIT_closeCStream and the matching decode
 * primitives).  ext/fse/ untouched.
 *
 * Build:
 *   cmake --build build --target pivco_fse_xy_micro
 * Run:
 *   ./build/pivco_fse_xy_micro              # 50k iters/cell
 *   ./build/pivco_fse_xy_micro 250000
 *
 * Restrictions: bitmap size must be a multiple of x (encoder
 * rejects misaligned sizes).  Test cells are picked so all four x
 * values can encode each size.
 */

#include "pivco_fse.h"
#include "pivco_fse_tables.h"
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#include "bitstream.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include "hist.h"
#ifdef PIVCO_HAS_OODLE
#include "bench_oodle_wrapper.h"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================
 *  Table setup.  We use the same normalized counts as
 *  pivco_fse_tables.h, but build local CTables/DTables (the ones
 *  inside src/pivco_fse.c are file-static).
 * ============================================================ */
static FSE_CTable *g_ct;
static FSE_DTable *g_dt;

static void build_tables_for_p(double p_major)
{
    int t_id = pivco_fse_select_table(p_major);
    if (t_id < 1) { g_ct = NULL; g_dt = NULL; return; }
    g_ct = FSE_createCTable(PIVCO_FSE_MAX_SYMBOL, PIVCO_FSE_TABLE_LOG);
    g_dt = FSE_createDTable(PIVCO_FSE_TABLE_LOG);
    FSE_buildCTable(g_ct, pivco_fse_norm[t_id],
                     PIVCO_FSE_MAX_SYMBOL, PIVCO_FSE_TABLE_LOG);
    FSE_buildDTable(g_dt, pivco_fse_norm[t_id],
                     PIVCO_FSE_MAX_SYMBOL, PIVCO_FSE_TABLE_LOG);
}

static void free_tables(void)
{
    if (g_ct) { FSE_freeCTable(g_ct); g_ct = NULL; }
    if (g_dt) { FSE_freeDTable(g_dt); g_dt = NULL; }
}


/* ============================================================
 *  FSE "full per-call setup" measurement.
 *
 *  ph's actual FSE usage avoids per-call setup: 12 static tables
 *  are loaded once at process start and picked by a 1-byte table
 *  id from the wire.  The FSE x*y decode columns measure that
 *  steady-state cost (pre-built DTable).
 *
 *  But to compare apples-to-apples with huf0/Oodle (which both
 *  read the table header + build DTable per call), we also time
 *  what FSE would cost if it had to do per-call setup.  That work
 *  is: FSE_readNCount (parse the counts header) + FSE_buildDTable
 *  (build the lookup table).  Per cell we serialize the chosen
 *  pivco_fse_norm[t_id] to a header bytes buffer via FSE_write
 *  NCount, then time read+build per call.  The result is per-call
 *  setup time in nanoseconds; printed in the decode table as the
 *  "su_ns" column so the reader can compute the "full per call"
 *  FSE throughput as 1 / (su_ns/1e9 + bytes/steady_mbps).
 * ============================================================ */
static unsigned char g_fse_header[1024];
static size_t        g_fse_header_size;

static int fse_setup_header(double p_major)
{
    int t_id = pivco_fse_select_table(p_major);
    if (t_id < 1) { g_fse_header_size = 0; return 0; }
    size_t r = FSE_writeNCount(g_fse_header, sizeof(g_fse_header),
                                pivco_fse_norm[t_id],
                                PIVCO_FSE_MAX_SYMBOL,
                                PIVCO_FSE_TABLE_LOG);
    if (FSE_isError(r)) { g_fse_header_size = 0; return 0; }
    g_fse_header_size = r;
    return 1;
}
/* time_fse_setup_ns_per_call defined later (after now_ns + N_BATCHES). */


/* ============================================================
 *  Generic x-cursor encoder.
 *
 *  Layout convention (mirrors FSE's reference x=2 encoder
 *  generalised to N cursors):
 *
 *    init cursors x-1, x-2, ..., 0 each consuming one input byte
 *    via FSE_initCState2 (so cursor 0 ends up flushed last → read
 *    by decoder first).
 *
 *    per round: encode x symbols, cursor x-1 first, cursor 0
 *    last.  Flush bits every 5 symbols (so a single flush stays
 *    under the 64-bit container).
 *
 *    flush cursors x-1, x-2, ..., 0 — cursor 0 is now closest to
 *    the bitstream end.
 *
 *  Decoder reads init states from the end (cursor 0 first), then
 *  decodes in cursor 0, 1, 2, ..., x-1 order per round, producing
 *  output in original input order.
 * ============================================================ */
static size_t encode_x(int x, const uint8_t *src, size_t n,
                       void *dst, size_t dst_cap,
                       const FSE_CTable *ct)
{
    if (x < 2 || x > 16) return 0;
    if (n % (size_t)x != 0) return 0;   /* bench restriction */

    BIT_CStream_t bitC;
    if (FSE_isError(BIT_initCStream(&bitC, dst, dst_cap))) return 0;

    FSE_CState_t st[16];
    size_t i = n;
    for (int k = x - 1; k >= 0; k--) {
        FSE_initCState2(&st[k], ct, src[--i]);
    }

    while (i > 0) {
        int pushed = 0;
        for (int k = x - 1; k >= 0; k--) {
            FSE_encodeSymbol(&bitC, &st[k], src[--i]);
            pushed++;
            if (pushed == 5 && i > 0) {
                BIT_flushBitsFast(&bitC);
                pushed = 0;
            }
        }
        if (pushed > 0) BIT_flushBitsFast(&bitC);
    }

    for (int k = x - 1; k >= 0; k--) {
        FSE_flushCState(&bitC, &st[k]);
    }
    return BIT_closeCStream(&bitC);
}


/* ============================================================
 *  Per-(x, y) decoder template.
 *
 *  Termination correctness is the tricky bit.  My first attempt
 *  used a fixed iteration count derived from dst_expected; that
 *  over-reads when the bitstream is "tight" (no slack between
 *  bits consumed and bits emitted, common at high skew), and
 *  garbage bits push state to invalid table indices → segfault.
 *
 *  Mirror FSE's reference instead:
 *
 *    1. Main fast loop runs while reload says `unfinished` AND
 *       there's room for x*y output bytes.  BODY decodes x*y
 *       symbols and reloads as needed by its cadence.
 *
 *    2. Tail: per round (x decodes), reload-check BETWEEN
 *       cursors.  When reload returns overflow, the remaining
 *       cursors of the current round decode one symbol each
 *       (post-overflow — they read 0 bits because the symbol is
 *       determined by the current state, and the now-overflowed
 *       bit reader supplies 0 bits to the state transition; the
 *       state ends up garbage but we don't use it again).
 *
 *    3. After the tail breaks, dst_expected % x bytes may still
 *       be left; decode them.
 *
 *  This is what `FSE_decompress_usingDTable_generic` does in
 *  ext/fse/lib/fse_decompress.c.
 *
 *  BODY is a sequence of per-cursor decode macros (D{X}RND for
 *  one round of X decodes) repeated Y times, with reloads
 *  inserted between rounds for x*tableLog > 64.
 * ============================================================ */

#define MK_DECODE_FN(NAME, X, Y, BODY) \
static size_t NAME(const void *src, size_t src_len, \
                    uint8_t *dst, size_t dst_expected, \
                    const FSE_DTable *dt) \
{ \
    BIT_DStream_t bitD; \
    if (FSE_isError(BIT_initDStream(&bitD, src, src_len))) return 0; \
    FSE_DState_t s[16]; \
    for (int k = 0; k < (X); k++) FSE_initDState(&s[k], &bitD, dt); \
    uint8_t *op = dst; \
    uint8_t * const olim = dst + dst_expected; \
    /* Main fast loop. */ \
    while ((BIT_reloadDStream(&bitD) == BIT_DStream_unfinished) \
            & (op + (X) * (Y) <= olim)) { \
        BODY; \
        op += (X) * (Y); \
    } \
    /* Tail: FSE-reference pattern. */ \
    while (op + (X) <= olim) { \
        int overflowed = 0; \
        for (int k = 0; k < (X); k++) { \
            *op++ = FSE_decodeSymbol(&s[k], &bitD); \
            if (BIT_reloadDStream(&bitD) == BIT_DStream_overflow) { \
                for (int kk = k + 1; kk < (X) && op < olim; kk++) \
                    *op++ = FSE_decodeSymbol(&s[kk], &bitD); \
                overflowed = 1; \
                break; \
            } \
        } \
        if (overflowed) break; \
    } \
    /* Partial final round (only fires if dst_expected isn't a \
     * multiple of x; bench sizes are aligned so this is unused \
     * in practice). */ \
    for (int k = 0; k < (X) && op < olim; k++) \
        *op++ = FSE_decodeSymbol(&s[k], &bitD); \
    return op - dst; \
}


/* Per-round decode macros (X decodes, with mid-round reloads
 * inserted for X * PIVCO_FSE_TABLE_LOG > 64 bits).  At our
 * tableLog=12, 5 decodes = 60 bits fits one container; 6+ decodes
 * needs a reload mid-round. */

#define D2RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD);
#define D4RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD);
#define D6RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD); \
    op[(base)+4] = FSE_decodeSymbolFast(&s[4], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+5] = FSE_decodeSymbolFast(&s[5], &bitD);
#define D8RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+4] = FSE_decodeSymbolFast(&s[4], &bitD); \
    op[(base)+5] = FSE_decodeSymbolFast(&s[5], &bitD); \
    op[(base)+6] = FSE_decodeSymbolFast(&s[6], &bitD); \
    op[(base)+7] = FSE_decodeSymbolFast(&s[7], &bitD);

/* x = 2 family.  At y=1 the main BODY is 2 decodes; at y=2 it's
 * 4 (matches FSE's reference shipping decoder shape exactly); at
 * y=4 it's 8, with one mid-body reload. */
MK_DECODE_FN(decode_x2_y1, 2, 1, D2RND(0))
MK_DECODE_FN(decode_x2_y2, 2, 2,
    D2RND(0)
    D2RND(2))
MK_DECODE_FN(decode_x2_y4, 2, 4,
    D2RND(0)
    D2RND(2)
    BIT_reloadDStream(&bitD);
    D2RND(4)
    D2RND(6))

/* x = 4 family. */
MK_DECODE_FN(decode_x4_y1, 4, 1, D4RND(0))
MK_DECODE_FN(decode_x4_y2, 4, 2,
    D4RND(0)
    BIT_reloadDStream(&bitD);
    D4RND(4))
MK_DECODE_FN(decode_x4_y4, 4, 4,
    D4RND(0)
    BIT_reloadDStream(&bitD);
    D4RND(4)
    BIT_reloadDStream(&bitD);
    D4RND(8)
    BIT_reloadDStream(&bitD);
    D4RND(12))

/* x = 6 family.  D6RND inserts its own mid-round reload after
 * the 5th decode, so no extra reload between rounds. */
MK_DECODE_FN(decode_x6_y1, 6, 1, D6RND(0))
MK_DECODE_FN(decode_x6_y2, 6, 2,
    D6RND(0)
    BIT_reloadDStream(&bitD);
    D6RND(6))
MK_DECODE_FN(decode_x6_y4, 6, 4,
    D6RND(0)
    BIT_reloadDStream(&bitD);
    D6RND(6)
    BIT_reloadDStream(&bitD);
    D6RND(12)
    BIT_reloadDStream(&bitD);
    D6RND(18))

/* x = 8 family.  D8RND inserts a reload after the 4th decode. */
MK_DECODE_FN(decode_x8_y1, 8, 1, D8RND(0))
MK_DECODE_FN(decode_x8_y2, 8, 2,
    D8RND(0)
    BIT_reloadDStream(&bitD);
    D8RND(8))
MK_DECODE_FN(decode_x8_y4, 8, 4,
    D8RND(0)
    BIT_reloadDStream(&bitD);
    D8RND(8)
    BIT_reloadDStream(&bitD);
    D8RND(16)
    BIT_reloadDStream(&bitD);
    D8RND(24))

/* x = 10 / 12 / 16 families.  Per-round macros insert internal
 * reloads to keep ≤ 5 decodes between reload calls (5 × tableLog
 * = 60 bits fits one 64-bit container). */
#define D10RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD); \
    op[(base)+4] = FSE_decodeSymbolFast(&s[4], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+5] = FSE_decodeSymbolFast(&s[5], &bitD); \
    op[(base)+6] = FSE_decodeSymbolFast(&s[6], &bitD); \
    op[(base)+7] = FSE_decodeSymbolFast(&s[7], &bitD); \
    op[(base)+8] = FSE_decodeSymbolFast(&s[8], &bitD); \
    op[(base)+9] = FSE_decodeSymbolFast(&s[9], &bitD);

#define D12RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD); \
    op[(base)+4] = FSE_decodeSymbolFast(&s[4], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+5] = FSE_decodeSymbolFast(&s[5], &bitD); \
    op[(base)+6] = FSE_decodeSymbolFast(&s[6], &bitD); \
    op[(base)+7] = FSE_decodeSymbolFast(&s[7], &bitD); \
    op[(base)+8] = FSE_decodeSymbolFast(&s[8], &bitD); \
    op[(base)+9] = FSE_decodeSymbolFast(&s[9], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+10] = FSE_decodeSymbolFast(&s[10], &bitD); \
    op[(base)+11] = FSE_decodeSymbolFast(&s[11], &bitD);

#define D16RND(base) \
    op[(base)+0]  = FSE_decodeSymbolFast(&s[0],  &bitD); \
    op[(base)+1]  = FSE_decodeSymbolFast(&s[1],  &bitD); \
    op[(base)+2]  = FSE_decodeSymbolFast(&s[2],  &bitD); \
    op[(base)+3]  = FSE_decodeSymbolFast(&s[3],  &bitD); \
    op[(base)+4]  = FSE_decodeSymbolFast(&s[4],  &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+5]  = FSE_decodeSymbolFast(&s[5],  &bitD); \
    op[(base)+6]  = FSE_decodeSymbolFast(&s[6],  &bitD); \
    op[(base)+7]  = FSE_decodeSymbolFast(&s[7],  &bitD); \
    op[(base)+8]  = FSE_decodeSymbolFast(&s[8],  &bitD); \
    op[(base)+9]  = FSE_decodeSymbolFast(&s[9],  &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+10] = FSE_decodeSymbolFast(&s[10], &bitD); \
    op[(base)+11] = FSE_decodeSymbolFast(&s[11], &bitD); \
    op[(base)+12] = FSE_decodeSymbolFast(&s[12], &bitD); \
    op[(base)+13] = FSE_decodeSymbolFast(&s[13], &bitD); \
    op[(base)+14] = FSE_decodeSymbolFast(&s[14], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+15] = FSE_decodeSymbolFast(&s[15], &bitD);

MK_DECODE_FN(decode_x10_y1, 10, 1, D10RND(0))
MK_DECODE_FN(decode_x10_y2, 10, 2,
    D10RND(0)
    BIT_reloadDStream(&bitD);
    D10RND(10))
MK_DECODE_FN(decode_x10_y4, 10, 4,
    D10RND(0)
    BIT_reloadDStream(&bitD);
    D10RND(10)
    BIT_reloadDStream(&bitD);
    D10RND(20)
    BIT_reloadDStream(&bitD);
    D10RND(30))

MK_DECODE_FN(decode_x12_y1, 12, 1, D12RND(0))
MK_DECODE_FN(decode_x12_y2, 12, 2,
    D12RND(0)
    BIT_reloadDStream(&bitD);
    D12RND(12))
MK_DECODE_FN(decode_x12_y4, 12, 4,
    D12RND(0)
    BIT_reloadDStream(&bitD);
    D12RND(12)
    BIT_reloadDStream(&bitD);
    D12RND(24)
    BIT_reloadDStream(&bitD);
    D12RND(36))

MK_DECODE_FN(decode_x16_y1, 16, 1, D16RND(0))
MK_DECODE_FN(decode_x16_y2, 16, 2,
    D16RND(0)
    BIT_reloadDStream(&bitD);
    D16RND(16))
MK_DECODE_FN(decode_x16_y4, 16, 4,
    D16RND(0)
    BIT_reloadDStream(&bitD);
    D16RND(16)
    BIT_reloadDStream(&bitD);
    D16RND(32)
    BIT_reloadDStream(&bitD);
    D16RND(48))


/* ============================================================
 *  Bench driver.
 * ============================================================ */

typedef size_t (*decode_fn_t)(const void *, size_t,
                                uint8_t *, size_t,
                                const FSE_DTable *);

typedef struct { int x, y; const char *name; decode_fn_t fn; } cfg_t;

static const cfg_t cfgs[] = {
    { 2,1,"x2y1", decode_x2_y1 }, { 2,2,"x2y2", decode_x2_y2 }, { 2,4,"x2y4", decode_x2_y4 },
    { 4,1,"x4y1", decode_x4_y1 }, { 4,2,"x4y2", decode_x4_y2 }, { 4,4,"x4y4", decode_x4_y4 },
    { 6,1,"x6y1", decode_x6_y1 }, { 6,2,"x6y2", decode_x6_y2 }, { 6,4,"x6y4", decode_x6_y4 },
    { 8,1,"x8y1", decode_x8_y1 }, { 8,2,"x8y2", decode_x8_y2 }, { 8,4,"x8y4", decode_x8_y4 },
    {10,1,"x10y1",decode_x10_y1},{10,2,"x10y2",decode_x10_y2},{10,4,"x10y4",decode_x10_y4},
    {12,1,"x12y1",decode_x12_y1},{12,2,"x12y2",decode_x12_y2},{12,4,"x12y4",decode_x12_y4},
    {16,1,"x16y1",decode_x16_y1},{16,2,"x16y2",decode_x16_y2},{16,4,"x16y4",decode_x16_y4},
};
#define N_CFGS (sizeof(cfgs)/sizeof(cfgs[0]))

static uint64_t xs_state = 0x123456789ABCDEF0ULL;
static uint64_t xs(void) {
    uint64_t v = xs_state; v ^= v<<13; v ^= v>>7; v ^= v<<17;
    return (xs_state = v);
}

/* Bytes whose bits are drawn IID with P(bit = 0) = p_major.
 * Matches the codec's per-node partition bitmap: high p_major
 * = one branch dominates = skewed = compresses tightly. */
static void fill_pmajor(uint8_t *buf, size_t len, double p_major)
{
    for (size_t i = 0; i < len; i++) {
        uint8_t b = 0;
        for (int j = 0; j < 8; j++) {
            int one = ((double)(xs() & 0xFFFF) / 65535.0) > p_major;
            b |= ((uint8_t)one) << j;
        }
        buf[i] = b;
    }
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Min-of-N-batches timing.  One large run is sensitive to a single
 * OS preemption / freq dip; min across N short batches drops to the
 * fastest one, which approximates the "no-interference" speed.
 *
 * Each batch ends with a roundtrip verification: the decoded
 * output must match the original source bytes (decode timer) or
 * the re-encoded payload must match the known-good payload
 * (encode timer).  Catches any drift mid-run.  On any mismatch we
 * print a diagnostic and exit nonzero. */
#define N_BATCHES 5

/* Time the cost of doing FSE table setup per call (FSE_readNCount
 * + FSE_buildDTable from the cell's pre-serialized header).  Returns
 * nanoseconds per call.  Used to derive "FSE full-per-call" rates
 * apples-to-apples with huf0/Oodle, which both pay this cost per
 * call.  ph's actual FSE usage avoids it entirely. */
static double time_fse_setup_ns_per_call(int iters)
{
    short norm[256];
    unsigned ms, tl;
    FSE_DTable *dt = FSE_createDTable(PIVCO_FSE_TABLE_LOG);
    for (int w = 0; w < 256; w++) {
        ms = 255;
        (void)FSE_readNCount(norm, &ms, &tl, g_fse_header, g_fse_header_size);
        (void)FSE_buildDTable(dt, norm, ms, tl);
    }
    double best_ns = 1e18;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile size_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            ms = 255;
            sink ^= FSE_readNCount(norm, &ms, &tl, g_fse_header, g_fse_header_size);
            sink ^= FSE_buildDTable(dt, norm, ms, tl);
        }
        double t1 = now_ns();
        (void)sink;
        double ns_per_call = (t1 - t0) / (double)iters;
        if (ns_per_call < best_ns) best_ns = ns_per_call;
    }
    FSE_freeDTable(dt);
    return best_ns;
}

static double time_decode_min(decode_fn_t fn, const void *enc, size_t enc_l,
                               uint8_t *dec, size_t bytes,
                               const FSE_DTable *dt, int iters,
                               const uint8_t *expect_src,
                               const char *cfg_name, double pmaj_for_msg)
{
    for (int w = 0; w < 256; w++) fn(enc, enc_l, dec, bytes, dt);
    double best_mbps = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            fn(enc, enc_l, dec, bytes, dt);
            sink ^= dec[0] ^ dec[bytes/2];
        }
        double t1 = now_ns();
        (void)sink;
        if (memcmp(expect_src, dec, bytes) != 0) {
            fprintf(stderr, "DECODE MISMATCH mid-timing: cfg=%s "
                    "size=%zu pmaj=%.2f batch=%d\n",
                    cfg_name, bytes, pmaj_for_msg, b);
            exit(2);
        }
        double mbps = 1000.0 * ((double)bytes * (double)iters) / (t1 - t0);
        if (mbps > best_mbps) best_mbps = mbps;
    }
    return best_mbps;
}

static double time_encode_min(int x, const uint8_t *src, size_t bytes,
                               uint8_t *enc_scratch, size_t enc_cap,
                               const FSE_CTable *ct, int iters,
                               const uint8_t *expect_enc, size_t expect_enc_len,
                               double pmaj_for_msg)
{
    for (int w = 0; w < 64; w++)
        (void)encode_x(x, src, bytes, enc_scratch, enc_cap, ct);
    double best_mbps = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile size_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++)
            sink ^= encode_x(x, src, bytes, enc_scratch, enc_cap, ct);
        double t1 = now_ns();
        (void)sink;
        size_t last_len = encode_x(x, src, bytes, enc_scratch, enc_cap, ct);
        if (last_len != expect_enc_len ||
            memcmp(expect_enc, enc_scratch, expect_enc_len) != 0) {
            fprintf(stderr, "ENCODE MISMATCH mid-timing: x=%d "
                    "size=%zu pmaj=%.2f batch=%d "
                    "(expected len=%zu got len=%zu)\n",
                    x, bytes, pmaj_for_msg, b,
                    expect_enc_len, last_len);
            exit(2);
        }
        double mbps = 1000.0 * ((double)bytes * (double)iters) / (t1 - t0);
        if (mbps > best_mbps) best_mbps = mbps;
    }
    return best_mbps;
}

/* ============================================================
 *  huff0 reference path (for comparison against FSE x*y).
 *
 *  3 decode variants are timed:
 *    huf1X1  =  1 stream  + X1 (single-symbol) decode table
 *    huf4X1  =  4 streams + X1 decode table
 *    huf4X2  =  4 streams + X2 (double-symbol) decode table — zstd's
 *               hot path; usually fastest on x86
 *  2 encode variants (the compressed payload differs by stream count
 *  only; X1/X2 is a decoder-table choice, not an encoder choice):
 *    hufC1   =  HUF_compress1X
 *    hufC4   =  HUF_compress4X
 *
 *  Same min-of-N protocol as FSE, with per-batch roundtrip verify.
 *  Table setup (HUF_compress + HUF_readDTable*) is done once per cell
 *  outside the timed loops — we measure steady-state throughput only.
 * ============================================================ */

static HUF_CREATE_STATIC_DTABLEX1(g_huf_dt_x1, 11);
static HUF_CREATE_STATIC_DTABLEX2(g_huf_dt_x2, 12);
static uint32_t g_huf_ct_storage[HUF_CTABLE_SIZE_U32(255)];
static HUF_CElt * const g_huf_ct = (HUF_CElt *)g_huf_ct_storage;
static unsigned char g_huf_wksp[HUF_WORKSPACE_SIZE];

/* g_huf*X_buf: full payload from HUF_compress*X_wksp (header + body),
 * used by the DECODE timing path which strips the header via the
 * g_huf*X_hdr offset.  Note: the body here may use a different
 * tableLog than g_huf_ct (the _wksp variant may call
 * HUF_optimalTableLog internally for small inputs), so the body
 * bytes are NOT guaranteed to equal HUF_compress*X_usingCTable's
 * output — we capture that separately for encode-timing verify. */
static uint8_t g_huf1X_buf[131072];
static uint8_t g_huf4X_buf[131072];
static size_t  g_huf1X_total, g_huf4X_total;
static size_t  g_huf1X_hdr,   g_huf4X_hdr;
/* g_huf*X_uct_buf: reference output of HUF_compress*X_usingCTable
 * (no header), used to verify the encode-timing loop's output. */
static uint8_t g_huf1X_uct_buf[131072];
static uint8_t g_huf4X_uct_buf[131072];
static size_t  g_huf1X_uct_total, g_huf4X_uct_total;

/* Returns 1 on success, 0 if huf0 declined (incompressible / single-
 * symbol / oversized — none of which the bench's pmaj-distributed
 * inputs at our cell sizes should hit at high pmaj, but pmaj near
 * 0.50 is roughly uniform-byte territory where huf0 may bail). */
static int huf_setup(const uint8_t *src, size_t n)
{
    /* Build the CTable directly from src counts; the encode-timing
     * loop uses HUF_compress*X_usingCTable (no table build inside)
     * to match FSE's "table setup outside timing" methodology. */
    unsigned counts[256];
    unsigned maxSym = 255;
    size_t largest = HIST_count(counts, &maxSym, src, n);
    if (HIST_isError(largest)) return 0;
    if (largest == n) return 0;          /* single-symbol */

    size_t ctbuild = HUF_buildCTable_wksp(g_huf_ct, counts, maxSym, 11,
                                            g_huf_wksp, sizeof(g_huf_wksp));
    if (HUF_isError(ctbuild)) return 0;

    /* Self-describing payloads (header + body): keeps the decode-side
     * setup straightforward via HUF_readDTable*_wksp. */
    g_huf1X_total = HUF_compress1X_wksp(g_huf1X_buf, sizeof(g_huf1X_buf),
                                         src, n, 255, 11,
                                         g_huf_wksp, sizeof(g_huf_wksp));
    g_huf4X_total = HUF_compress4X_wksp(g_huf4X_buf, sizeof(g_huf4X_buf),
                                         src, n, 255, 11,
                                         g_huf_wksp, sizeof(g_huf_wksp));
    if (HUF_isError(g_huf1X_total) || HUF_isError(g_huf4X_total)
        || g_huf1X_total <= 1 || g_huf4X_total <= 1) return 0;

    g_huf1X_hdr = HUF_readDTableX1_wksp(g_huf_dt_x1, g_huf1X_buf,
                                         g_huf1X_total,
                                         g_huf_wksp, sizeof(g_huf_wksp));
    g_huf4X_hdr = HUF_readDTableX1_wksp(g_huf_dt_x1, g_huf4X_buf,
                                         g_huf4X_total,
                                         g_huf_wksp, sizeof(g_huf_wksp));
    size_t x2hdr = HUF_readDTableX2_wksp(g_huf_dt_x2, g_huf4X_buf,
                                          g_huf4X_total,
                                          g_huf_wksp, sizeof(g_huf_wksp));
    if (HUF_isError(g_huf1X_hdr) || HUF_isError(g_huf4X_hdr)
        || HUF_isError(x2hdr)) return 0;

    /* Reference encoded payloads using OUR CTable (matches what the
     * timed encode loop will produce). */
    g_huf1X_uct_total = HUF_compress1X_usingCTable(g_huf1X_uct_buf,
                                                     sizeof(g_huf1X_uct_buf),
                                                     src, n, g_huf_ct);
    g_huf4X_uct_total = HUF_compress4X_usingCTable(g_huf4X_uct_buf,
                                                     sizeof(g_huf4X_uct_buf),
                                                     src, n, g_huf_ct);
    if (HUF_isError(g_huf1X_uct_total) || HUF_isError(g_huf4X_uct_total))
        return 0;
    return 1;
}

/* Decode wrappers.  Use the _DCtx_wksp family which reads the
 * table header from the payload + builds the DTable + decodes -
 * matches what zstd does at runtime per call.  This makes the
 * huf0 column apples-to-apples with the Oodle column (both
 * include per-call table setup); FSE is steady-state because
 * ph's actual FSE usage picks from 12 pre-built static tables
 * via a 1-byte table-id and doesn't build per call. */
static size_t huf_dec_1X1(uint8_t *dec, size_t dec_cap)
{
    return HUF_decompress1X1_DCtx_wksp(g_huf_dt_x1, dec, dec_cap,
                                        g_huf1X_buf, g_huf1X_total,
                                        g_huf_wksp, sizeof(g_huf_wksp));
}
static size_t huf_dec_4X1(uint8_t *dec, size_t dec_cap)
{
    return HUF_decompress4X1_DCtx_wksp(g_huf_dt_x1, dec, dec_cap,
                                        g_huf4X_buf, g_huf4X_total,
                                        g_huf_wksp, sizeof(g_huf_wksp));
}
static size_t huf_dec_4X2(uint8_t *dec, size_t dec_cap)
{
    return HUF_decompress4X2_DCtx_wksp(g_huf_dt_x2, dec, dec_cap,
                                        g_huf4X_buf, g_huf4X_total,
                                        g_huf_wksp, sizeof(g_huf_wksp));
}

/* Encode timing uses the pre-built CTable — matches FSE's
 * "table setup outside the timing loop" methodology.  Output has
 * no table header; it's the body that HUF_compress*X_wksp would
 * produce after stripping its written-CTable prefix. */
static size_t huf_enc_1X(const uint8_t *src, size_t n,
                          uint8_t *dst, size_t cap)
{
    return HUF_compress1X_usingCTable(dst, cap, src, n, g_huf_ct);
}
static size_t huf_enc_4X(const uint8_t *src, size_t n,
                          uint8_t *dst, size_t cap)
{
    return HUF_compress4X_usingCTable(dst, cap, src, n, g_huf_ct);
}

typedef size_t (*huf_dec_fn_t)(uint8_t *, size_t);
typedef size_t (*huf_enc_fn_t)(const uint8_t *, size_t, uint8_t *, size_t);

static double time_huf_decode_min(huf_dec_fn_t fn, uint8_t *dec, size_t bytes,
                                    int iters, const uint8_t *expect_src,
                                    const char *name, double pmaj_for_msg)
{
    for (int w = 0; w < 256; w++) (void)fn(dec, bytes);
    double best_mbps = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            (void)fn(dec, bytes);
            sink ^= dec[0] ^ dec[bytes/2];
        }
        double t1 = now_ns();
        (void)sink;
        if (memcmp(expect_src, dec, bytes) != 0) {
            fprintf(stderr, "HUF DECODE MISMATCH mid-timing: %s "
                    "size=%zu pmaj=%.2f batch=%d\n",
                    name, bytes, pmaj_for_msg, b);
            exit(2);
        }
        double mbps = 1000.0 * ((double)bytes * (double)iters) / (t1 - t0);
        if (mbps > best_mbps) best_mbps = mbps;
    }
    return best_mbps;
}

static double time_huf_encode_min(huf_enc_fn_t fn,
                                    const uint8_t *src, size_t bytes,
                                    uint8_t *scratch, size_t scratch_cap,
                                    int iters,
                                    const uint8_t *expect_enc,
                                    size_t expect_enc_len,
                                    const char *name, double pmaj_for_msg)
{
    for (int w = 0; w < 64; w++)
        (void)fn(src, bytes, scratch, scratch_cap);
    double best_mbps = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile size_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++)
            sink ^= fn(src, bytes, scratch, scratch_cap);
        double t1 = now_ns();
        (void)sink;
        size_t last_len = fn(src, bytes, scratch, scratch_cap);
        if (last_len != expect_enc_len ||
            memcmp(expect_enc, scratch, expect_enc_len) != 0) {
            fprintf(stderr, "HUF ENCODE MISMATCH mid-timing: %s "
                    "size=%zu pmaj=%.2f batch=%d "
                    "(expected len=%zu got len=%zu)\n",
                    name, bytes, pmaj_for_msg, b,
                    expect_enc_len, last_len);
            exit(2);
        }
        double mbps = 1000.0 * ((double)bytes * (double)iters) / (t1 - t0);
        if (mbps > best_mbps) best_mbps = mbps;
    }
    return best_mbps;
}


#ifdef PIVCO_HAS_OODLE
/* ============================================================
 *  Oodle reference path (gated on PIVCO_HAS_OODLE).
 *
 *  Single decode column ("oh3" or "oh6" depending on which
 *  variant the tuner picked) and single encode column ("ohC").
 *  Note: Oodle's newlz_get_array_huff includes table-header read
 *  per call (no _usingDTable variant exported).  For FSE / huff0
 *  we factored that cost out; for Oodle we accept it — matches
 *  the shipping shape and the header is small relative to body.
 * ============================================================ */

static uint8_t g_oodle_buf[131072];
static int     g_oodle_total;
static int     g_oodle_huff_type;
static int     g_oodle_ok;

static int oodle_setup(const uint8_t *src, size_t n)
{
    g_oodle_huff_type = 0;
    g_oodle_total = oodle_huff_encode(src, n,
                                       g_oodle_buf, sizeof(g_oodle_buf),
                                       &g_oodle_huff_type);
    if (g_oodle_total <= 0 || g_oodle_total > (int)n) {
        g_oodle_ok = 0; return 0;
    }
    g_oodle_ok = 1;
    return 1;
}

static size_t oodle_dec(uint8_t *dec, size_t dec_cap)
{
    int r = oodle_huff_decode(g_oodle_buf, (size_t)g_oodle_total,
                               dec, dec_cap, g_oodle_huff_type);
    return r > 0 ? (size_t)r : 0;
}

static size_t oodle_enc(const uint8_t *src, size_t n,
                         uint8_t *dst, size_t cap)
{
    int huff_type = 0;
    int r = oodle_huff_encode(src, n, dst, cap, &huff_type);
    return r > 0 ? (size_t)r : 0;
}

static double time_oodle_decode_min(uint8_t *dec, size_t bytes,
                                      int iters, const uint8_t *expect_src,
                                      double pmaj_for_msg)
{
    for (int w = 0; w < 256; w++) (void)oodle_dec(dec, bytes);
    double best_mbps = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            (void)oodle_dec(dec, bytes);
            sink ^= dec[0] ^ dec[bytes/2];
        }
        double t1 = now_ns();
        (void)sink;
        if (memcmp(expect_src, dec, bytes) != 0) {
            fprintf(stderr, "OODLE DECODE MISMATCH mid-timing: "
                    "size=%zu pmaj=%.2f batch=%d\n",
                    bytes, pmaj_for_msg, b);
            exit(2);
        }
        double mbps = 1000.0 * ((double)bytes * (double)iters) / (t1 - t0);
        if (mbps > best_mbps) best_mbps = mbps;
    }
    return best_mbps;
}

static double time_oodle_encode_min(const uint8_t *src, size_t bytes,
                                      uint8_t *scratch, size_t scratch_cap,
                                      int iters, double pmaj_for_msg)
{
    for (int w = 0; w < 64; w++) (void)oodle_enc(src, bytes, scratch, scratch_cap);
    double best_mbps = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile size_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++)
            sink ^= oodle_enc(src, bytes, scratch, scratch_cap);
        double t1 = now_ns();
        (void)sink;
        size_t last_len = oodle_enc(src, bytes, scratch, scratch_cap);
        if ((int)last_len != g_oodle_total ||
            memcmp(g_oodle_buf, scratch, last_len) != 0) {
            fprintf(stderr, "OODLE ENCODE MISMATCH mid-timing: "
                    "size=%zu pmaj=%.2f batch=%d "
                    "(expected len=%d got len=%zu)\n",
                    bytes, pmaj_for_msg, b,
                    g_oodle_total, last_len);
            exit(2);
        }
        double mbps = 1000.0 * ((double)bytes * (double)iters) / (t1 - t0);
        if (mbps > best_mbps) best_mbps = mbps;
    }
    return best_mbps;
}

/* ---- Oodle tANS (newlz_arrays_tans): the entropy stage we compare
 *      head-to-head against FSE.  Same per-call shape as the huff
 *      path (encode-once in setup, then time decode/encode). ---- */

static uint8_t g_oodle_tans_buf[131072];
static int     g_oodle_tans_total;
static int     g_oodle_tans_ok;

static int oodle_tans_setup(const uint8_t *src, size_t n)
{
    g_oodle_tans_total = oodle_tans_encode(src, n,
                                            g_oodle_tans_buf,
                                            sizeof(g_oodle_tans_buf));
    if (g_oodle_tans_total <= 0 || g_oodle_tans_total > (int)n) {
        g_oodle_tans_ok = 0; return 0;
    }
    g_oodle_tans_ok = 1;
    return 1;
}

static size_t oodle_tans_dec(uint8_t *dec, size_t dec_cap)
{
    int r = oodle_tans_decode(g_oodle_tans_buf, (size_t)g_oodle_tans_total,
                               dec, dec_cap);
    return r > 0 ? (size_t)r : 0;
}

static size_t oodle_tans_enc(const uint8_t *src, size_t n,
                             uint8_t *dst, size_t cap)
{
    int r = oodle_tans_encode(src, n, dst, cap);
    return r > 0 ? (size_t)r : 0;
}

static double time_oodle_tans_decode_min(uint8_t *dec, size_t bytes,
                                          int iters, const uint8_t *expect_src,
                                          double pmaj_for_msg)
{
    for (int w = 0; w < 256; w++) (void)oodle_tans_dec(dec, bytes);
    double best_mbps = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            (void)oodle_tans_dec(dec, bytes);
            sink ^= dec[0] ^ dec[bytes/2];
        }
        double t1 = now_ns();
        (void)sink;
        if (memcmp(expect_src, dec, bytes) != 0) {
            fprintf(stderr, "OODLE TANS DECODE MISMATCH mid-timing: "
                    "size=%zu pmaj=%.2f batch=%d\n",
                    bytes, pmaj_for_msg, b);
            exit(2);
        }
        double mbps = 1000.0 * ((double)bytes * (double)iters) / (t1 - t0);
        if (mbps > best_mbps) best_mbps = mbps;
    }
    return best_mbps;
}

static double time_oodle_tans_encode_min(const uint8_t *src, size_t bytes,
                                          uint8_t *scratch, size_t scratch_cap,
                                          int iters, double pmaj_for_msg)
{
    for (int w = 0; w < 64; w++) (void)oodle_tans_enc(src, bytes, scratch, scratch_cap);
    double best_mbps = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile size_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++)
            sink ^= oodle_tans_enc(src, bytes, scratch, scratch_cap);
        double t1 = now_ns();
        (void)sink;
        size_t last_len = oodle_tans_enc(src, bytes, scratch, scratch_cap);
        if ((int)last_len != g_oodle_tans_total ||
            memcmp(g_oodle_tans_buf, scratch, last_len) != 0) {
            fprintf(stderr, "OODLE TANS ENCODE MISMATCH mid-timing: "
                    "size=%zu pmaj=%.2f batch=%d "
                    "(expected len=%d got len=%zu)\n",
                    bytes, pmaj_for_msg, b,
                    g_oodle_tans_total, last_len);
            exit(2);
        }
        double mbps = 1000.0 * ((double)bytes * (double)iters) / (t1 - t0);
        if (mbps > best_mbps) best_mbps = mbps;
    }
    return best_mbps;
}
#endif  /* PIVCO_HAS_OODLE */


int main(int argc, char **argv)
{
    int iters = 50000;
    if (argc > 1) iters = atoi(argv[1]);
    if (iters < 1000) iters = 1000;

    pivco_fse_init();

    /* Cells: bytes is the # of source bytes to encode/decode.
     * p_major is the bit-1 probability used to fill source bytes
     * (mirrors ph's partition-bitmap distribution: high p_major
     * = highly-skewed bitmap = ~2 leaf symbols dominating).
     *
     * Sizes divisible by all x ∈ {2,4,6,8,10,12,16}: multiples of
     * LCM(...)=240.  Where the size doesn't divide a given x, the
     * bench prints "-" for those columns.  48/96 stay (skip x=10
     * only) because they're useful small-size data points. */
    static const struct { size_t size; double p_major; } cells[] = {
        /* Size sweep at pmaj=0.80 (one axis at a time).  Sizes are
         * multiples of LCM(2,4,6,8,10,12,16)=240 so every x value
         * divides cleanly.  Small sizes (48/96) skip x=10.
         *
         * Larger cells (4080+) added to amortize per-call table-
         * read overhead - Oodle's newlz_get_array_huff re-reads
         * the table header every call (no _usingDTable equivalent
         * in the public API), so at small sizes the comparison is
         * unfair to it.  At >= 16320 B the setup is well-amortized
         * and Oodle should approach its ~1.3-1.5 cyc/sym shipping
         * asymptote (ryg's quote).  See IDEAS.md "Oodle's
         * newlz_arrays_huff" entry. */
        {   48, 0.80 },
        {   96, 0.80 },
        {  240, 0.80 },
        {  480, 0.80 },
        {  960, 0.80 },
        { 1440, 0.80 },
        { 2880, 0.80 },
        { 4080, 0.80 },    /* ~ 4 KB */
        { 8160, 0.80 },    /* ~ 8 KB */
        {16320, 0.80 },    /* ~16 KB */
        {32640, 0.80 },    /* ~32 KB */
        {65280, 0.80 },    /* ~64 KB */
        /* Skew sweep at size=960 (size held fixed).  pmaj=0.50 is
         * the lowest table threshold (uniform-ish bitmap). */
        {  960, 0.50 },
        {  960, 0.55 },
        {  960, 0.60 },
        {  960, 0.70 },
        {  960, 0.90 },
    };
    const int n_cells = sizeof(cells)/sizeof(cells[0]);

    /* Per-x encode results: indexed by xi in [0..n_x). */
    const int x_values[] = {2, 4, 6, 8, 10, 12, 16};
    const int n_x = sizeof(x_values)/sizeof(x_values[0]);
    uint8_t enc_buf[7][131072];
    size_t  enc_len[7];

    uint8_t src[131072];
    uint8_t dec[131072];

    printf("FSE x[2,4,6,8,10,12,16] × y[1,2,4] microbench, "
           "min of %d batches × %d iters/batch per cell\n",
           N_BATCHES, iters);
    printf("(numbers are MB/s; higher = faster.  x2y2 = FSE's "
           "shipping decode shape.)\n");
    printf("size = source-byte count.  pmaj = P(bit = 0) = major-"
           "symbol probability (each bit drawn IID).\n");
    printf("high pmaj = more zeros = skewed bitmap = tighter "
           "FSE compression.\n");
    printf("Per-call cost model:\n");
    printf("  FSE columns - steady-state (pre-built DTable).  Matches\n");
    printf("                ph's real usage: 12 static tables picked\n");
    printf("                at runtime by 1-byte table-id from the wire.\n");
    printf("  su_ns       - FSE per-call setup cost in ns (FSE_readNCount\n");
    printf("                + FSE_buildDTable).  Add to derive 'FSE full\n");
    printf("                per-call' MB/s for any x*y as:\n");
    printf("                bytes / (su_ns + bytes * 1000/steady_mbps).\n");
    printf("  huf0 columns - FULL per-call (HUF_decompress*X*_DCtx_wksp:\n");
    printf("                 reads table header + builds DTable + decodes).\n");
    printf("                 Matches what zstd does at runtime per call.\n");
#ifdef PIVCO_HAS_OODLE
    printf("  oodle column - FULL per-call (newlz_get_array_huff:\n");
    printf("                 reads table header + builds tab + decodes).\n");
    printf("                 No _usingDTable variant in Oodle's public API.\n");
    printf("                 Tuner picks huff6 (6-stream).\n");
    printf("  o_tans column- FULL per-call Oodle tANS (newlz_*_array_tans:\n");
    printf("                 2 bitstreams x 5-way interleave).  Compare\n");
    printf("                 head-to-head vs the FSE x*y columns.\n");
    printf("                 Both decode kernels are ASM, wired in via\n");
    printf("                 extras/oodle_build_patches/ (huff + tANS).\n");
    printf("                 Set -DOODLE_LIB_VARIANT=shipped to link\n");
    printf("                 RAD's prebuilt lib instead of our build-out.\n");
#endif
    fflush(stdout);

    double enc_mbps[24][8];          /* [cell][xi] */
    double dec_mbps[24][32];         /* [cell][cfg_index] */
    int    x_ok_mat[24][8] = {{0}};  /* [cell][xi] */
    /* huff0 reference (2 encode + 3 decode columns). */
    double huf_enc_mbps[24][2];      /* [cell][hufC1, hufC4] */
    double huf_dec_mbps[24][3];      /* [cell][1X1, 4X1, 4X2] */
    int    huf_ok_mat[24] = {0};
    /* FSE per-call setup cost (FSE_readNCount + FSE_buildDTable);
     * ph's actual usage avoids this via pre-built static tables,
     * but it's shown here for apples-to-apples vs huf0/Oodle. */
    double fse_setup_ns[24];
#ifdef PIVCO_HAS_OODLE
    /* Oodle reference (1 encode + 1 decode column; tuner picks
     * huff3 vs huff6 internally, label reflects choice per cell). */
    double oodle_enc_mbps[24];
    double oodle_dec_mbps[24];
    int    oodle_type_mat[24] = {0}; /* OODLE_HUFF_TYPE_HUFF{3,6} or 0 */
    /* Oodle tANS — head-to-head vs FSE (the entropy stage, not huff). */
    double oodle_tans_enc_mbps[24];
    double oodle_tans_dec_mbps[24];
#endif

    for (int ci = 0; ci < n_cells; ci++) {
        size_t bytes = cells[ci].size;
        double p = cells[ci].p_major;
        build_tables_for_p(p);
        fse_setup_ns[ci] = -1.0;
        if (!g_ct) {
            for (int xi = 0; xi < n_x; xi++) enc_mbps[ci][xi] = -1.0;
            for (size_t k = 0; k < N_CFGS; k++) dec_mbps[ci][k] = -1.0;
            continue;
        }
        if (bytes > sizeof(src)) { free_tables(); continue; }
        /* Time the per-call FSE setup cost (read header + build
         * DTable).  Same number used by every x*y config. */
        if (fse_setup_header(p)) {
            fse_setup_ns[ci] = time_fse_setup_ns_per_call(iters);
        }
        fill_pmajor(src, bytes, p);

        /* Encode each x; sanity-check by decoding back with x*y=1. */
        for (int xi = 0; xi < n_x; xi++) {
            int x = x_values[xi];
            if (bytes % (size_t)x != 0) { enc_mbps[ci][xi] = -1.0; continue; }
            size_t elen = encode_x(x, src, bytes,
                                    enc_buf[xi], sizeof(enc_buf[xi]), g_ct);
            if (elen == 0) { enc_mbps[ci][xi] = -1.0; continue; }
            enc_len[xi] = elen;
            x_ok_mat[ci][xi] = 1;
        }

        /* Sanity: every config round-trips. */
        for (size_t k = 0; k < N_CFGS; k++) {
            int xi = -1;
            for (int j = 0; j < n_x; j++)
                if (x_values[j] == cfgs[k].x) { xi = j; break; }
            if (xi < 0 || !x_ok_mat[ci][xi]) { dec_mbps[ci][k] = -1.0; continue; }
            memset(dec, 0xCC, bytes);
            cfgs[k].fn(enc_buf[xi], enc_len[xi], dec, bytes, g_dt);
            if (memcmp(src, dec, bytes) != 0) {
                fprintf(stderr, "MISMATCH size=%zu p=%.2f cfg=%s\n",
                        bytes, p, cfgs[k].name);
                free_tables();
                return 1;
            }
        }

        /* Time encode (min of N batches, with per-batch verify). */
        for (int xi = 0; xi < n_x; xi++) {
            if (!x_ok_mat[ci][xi]) { enc_mbps[ci][xi] = -1.0; continue; }
            uint8_t enc_scratch[131072];
            enc_mbps[ci][xi] = time_encode_min(x_values[xi], src, bytes,
                                                enc_scratch, sizeof(enc_scratch),
                                                g_ct, iters,
                                                enc_buf[xi], enc_len[xi], p);
        }

        /* Time decode (min of N batches, with per-batch verify). */
        for (size_t k = 0; k < N_CFGS; k++) {
            int xi = -1;
            for (int j = 0; j < n_x; j++)
                if (x_values[j] == cfgs[k].x) { xi = j; break; }
            if (xi < 0 || !x_ok_mat[ci][xi]) { dec_mbps[ci][k] = -1.0; continue; }
            dec_mbps[ci][k] = time_decode_min(cfgs[k].fn, enc_buf[xi],
                                                enc_len[xi], dec, bytes,
                                                g_dt, iters, src,
                                                cfgs[k].name, p);
        }

        /* huff0 reference path: setup, sanity-check, time. */
        huf_ok_mat[ci] = huf_setup(src, bytes);
        if (!huf_ok_mat[ci]) {
            for (int j = 0; j < 2; j++) huf_enc_mbps[ci][j] = -1.0;
            for (int j = 0; j < 3; j++) huf_dec_mbps[ci][j] = -1.0;
        } else {
            /* Sanity-check each huff0 decoder. */
            memset(dec, 0xCC, bytes);
            (void)huf_dec_1X1(dec, bytes);
            if (memcmp(src, dec, bytes) != 0) {
                fprintf(stderr, "HUF 1X1 SANITY MISMATCH size=%zu p=%.2f\n",
                        bytes, p);
                free_tables();
                return 1;
            }
            memset(dec, 0xCC, bytes);
            (void)huf_dec_4X1(dec, bytes);
            if (memcmp(src, dec, bytes) != 0) {
                fprintf(stderr, "HUF 4X1 SANITY MISMATCH size=%zu p=%.2f\n",
                        bytes, p);
                free_tables();
                return 1;
            }
            memset(dec, 0xCC, bytes);
            (void)huf_dec_4X2(dec, bytes);
            if (memcmp(src, dec, bytes) != 0) {
                fprintf(stderr, "HUF 4X2 SANITY MISMATCH size=%zu p=%.2f\n",
                        bytes, p);
                free_tables();
                return 1;
            }

            huf_dec_mbps[ci][0] = time_huf_decode_min(huf_dec_1X1, dec, bytes,
                                                       iters, src, "huf1X1", p);
            huf_dec_mbps[ci][1] = time_huf_decode_min(huf_dec_4X1, dec, bytes,
                                                       iters, src, "huf4X1", p);
            huf_dec_mbps[ci][2] = time_huf_decode_min(huf_dec_4X2, dec, bytes,
                                                       iters, src, "huf4X2", p);

            /* Verify reference is HUF_compress*X_usingCTable output
             * with our pre-built CTable (captured in setup). */
            uint8_t huf_scratch[131072];
            huf_enc_mbps[ci][0] = time_huf_encode_min(huf_enc_1X, src, bytes,
                                                       huf_scratch, sizeof(huf_scratch),
                                                       iters,
                                                       g_huf1X_uct_buf,
                                                       g_huf1X_uct_total,
                                                       "hufC1", p);
            huf_enc_mbps[ci][1] = time_huf_encode_min(huf_enc_4X, src, bytes,
                                                       huf_scratch, sizeof(huf_scratch),
                                                       iters,
                                                       g_huf4X_uct_buf,
                                                       g_huf4X_uct_total,
                                                       "hufC4", p);
        }

#ifdef PIVCO_HAS_OODLE
        /* Oodle reference path: setup, sanity, time. */
        if (oodle_setup(src, bytes)) {
            oodle_type_mat[ci] = g_oodle_huff_type;
            memset(dec, 0xCC, bytes);
            if (oodle_dec(dec, bytes) == 0 ||
                memcmp(src, dec, bytes) != 0) {
                fprintf(stderr, "OODLE SANITY MISMATCH size=%zu p=%.2f\n",
                        bytes, p);
                free_tables();
                return 1;
            }
            oodle_dec_mbps[ci] = time_oodle_decode_min(dec, bytes, iters, src, p);
            uint8_t oodle_scratch[131072];
            oodle_enc_mbps[ci] = time_oodle_encode_min(src, bytes,
                                                       oodle_scratch,
                                                       sizeof(oodle_scratch),
                                                       iters, p);
        } else {
            oodle_type_mat[ci] = 0;
            oodle_enc_mbps[ci] = -1.0;
            oodle_dec_mbps[ci] = -1.0;
        }

        /* Oodle tANS reference path: setup, sanity, time. */
        if (oodle_tans_setup(src, bytes)) {
            memset(dec, 0xCC, bytes);
            if (oodle_tans_dec(dec, bytes) == 0 ||
                memcmp(src, dec, bytes) != 0) {
                fprintf(stderr, "OODLE TANS SANITY MISMATCH size=%zu p=%.2f\n",
                        bytes, p);
                free_tables();
                return 1;
            }
            oodle_tans_dec_mbps[ci] = time_oodle_tans_decode_min(dec, bytes,
                                                                 iters, src, p);
            uint8_t otans_scratch[131072];
            oodle_tans_enc_mbps[ci] = time_oodle_tans_encode_min(src, bytes,
                                                                 otans_scratch,
                                                                 sizeof(otans_scratch),
                                                                 iters, p);
        } else {
            oodle_tans_enc_mbps[ci] = -1.0;
            oodle_tans_dec_mbps[ci] = -1.0;
        }
#endif

        free_tables();
    }

    /* Print encode table (FSE varies only with x, huff0 = 2 ref cols). */
    printf("\n--- ENCODE (MB/s, varies only with x for FSE; huff0 = ref) ---\n");
    printf("%5s %5s |", "size", "pmaj");
    for (int xi = 0; xi < n_x; xi++) {
        char lbl[8]; snprintf(lbl, sizeof(lbl), "x%d", x_values[xi]);
        printf(" %6s", lbl);
    }
    printf("  %6s %6s", "hufC1", "hufC4");
#ifdef PIVCO_HAS_OODLE
    printf("  %6s %6s", "oodle", "o_tans");
#endif
    printf("\n%5s %5s-+", "-----", "----");
    for (int xi = 0; xi < n_x; xi++) printf("-------");
    printf("---------------");
#ifdef PIVCO_HAS_OODLE
    printf("---------------");
#endif
    printf("\n");
    for (int ci = 0; ci < n_cells; ci++) {
        printf("%5zu %5.2f |", cells[ci].size, cells[ci].p_major);
        for (int xi = 0; xi < n_x; xi++) {
            if (enc_mbps[ci][xi] < 0) printf(" %6s", "  -  ");
            else printf(" %6.1f", enc_mbps[ci][xi]);
        }
        printf(" ");
        for (int j = 0; j < 2; j++) {
            if (huf_enc_mbps[ci][j] < 0) printf(" %6s", "  -  ");
            else printf(" %6.1f", huf_enc_mbps[ci][j]);
        }
#ifdef PIVCO_HAS_OODLE
        printf(" ");
        if (oodle_enc_mbps[ci] < 0) printf(" %6s", "  -  ");
        else printf(" %6.1f", oodle_enc_mbps[ci]);
        if (oodle_tans_enc_mbps[ci] < 0) printf(" %6s", "  -  ");
        else printf(" %6.1f", oodle_tans_enc_mbps[ci]);
#endif
        printf("\n");
    }

    /* Print decode table (full x × y grid + 3 huff0 ref cols).
     * Visual gap between x-groups (every 3 cfgs, since y ∈ {1,2,4}). */
    printf("\n--- DECODE (MB/s; huff0 = ref) ---\n");
    printf("%5s %5s |", "size", "pmaj");
    for (size_t c = 0; c < N_CFGS; c++) {
        if (c > 0 && (c % 3) == 0) printf(" ");
        printf(" %6s", cfgs[c].name);
    }
    /* su_ns = FSE per-call setup cost in ns.  Reader can compute
     * "FSE full per-call" MB/s for any x*y as:
     *     1 / (su_ns/1e9 + bytes/steady_mbps/1e3)  (in GB/s)
     * Equivalently:
     *     bytes / (su_ns + bytes * 1000/steady_mbps)  (MB/s) */
    printf("  %6s", " su_ns");
    printf("  %6s %6s %6s", "huf1X1", "huf4X1", "huf4X2");
#ifdef PIVCO_HAS_OODLE
    printf("  %6s %6s", "oodle", "o_tans");
#endif
    printf("\n%5s %5s-+", "-----", "----");
    for (size_t c = 0; c < N_CFGS; c++) {
        if (c > 0 && (c % 3) == 0) printf("-");
        printf("-------");
    }
    printf("--------");
    printf("-----------------------");
#ifdef PIVCO_HAS_OODLE
    printf("---------------");
#endif
    printf("\n");
    for (int ci = 0; ci < n_cells; ci++) {
        printf("%5zu %5.2f |", cells[ci].size, cells[ci].p_major);
        for (size_t k = 0; k < N_CFGS; k++) {
            if (k > 0 && (k % 3) == 0) printf(" ");
            if (dec_mbps[ci][k] < 0) printf(" %6s", "  -  ");
            else printf(" %6.1f", dec_mbps[ci][k]);
        }
        if (fse_setup_ns[ci] < 0) printf("  %6s", "  -  ");
        else printf("  %6.0f", fse_setup_ns[ci]);
        printf(" ");
        for (int j = 0; j < 3; j++) {
            if (huf_dec_mbps[ci][j] < 0) printf(" %6s", "  -  ");
            else printf(" %6.1f", huf_dec_mbps[ci][j]);
        }
#ifdef PIVCO_HAS_OODLE
        printf(" ");
        if (oodle_dec_mbps[ci] < 0) printf(" %6s", "  -  ");
        else printf(" %6.1f", oodle_dec_mbps[ci]);
        if (oodle_tans_dec_mbps[ci] < 0) printf(" %6s", "  -  ");
        else printf(" %6.1f", oodle_tans_dec_mbps[ci]);
#endif
        printf("\n");
    }
#ifdef PIVCO_HAS_OODLE
    /* Show which Oodle variant the tuner picked per cell. */
    printf("\nOodle huff variant per cell (tuner pick):\n");
    for (int ci = 0; ci < n_cells; ci++) {
        const char *lbl;
        switch (oodle_type_mat[ci]) {
            case OODLE_HUFF_TYPE_HUFF3: lbl = "huff3 (3-stream)"; break;
            case OODLE_HUFF_TYPE_HUFF6: lbl = "huff6 (6-stream)"; break;
            default: lbl = "(declined)"; break;
        }
        printf("  size=%4zu pmaj=%.2f -> %s\n",
                cells[ci].size, cells[ci].p_major, lbl);
    }
#endif

    return 0;
}
