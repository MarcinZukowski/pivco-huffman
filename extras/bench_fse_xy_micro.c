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
        /* Size sweep at pmaj=0.80 (one axis at a time).  2880 added
         * to see when ILP scaling flattens at large sizes. */
        {   48, 0.80 },
        {   96, 0.80 },
        {  240, 0.80 },
        {  480, 0.80 },
        {  960, 0.80 },
        { 1440, 0.80 },
        { 2880, 0.80 },
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
    uint8_t enc_buf[7][16384];
    size_t  enc_len[7];

    uint8_t src[8192];
    uint8_t dec[8192];

    printf("FSE x[2,4,6,8,10,12,16] × y[1,2,4] microbench, "
           "min of %d batches × %d iters/batch per cell\n",
           N_BATCHES, iters);
    printf("(numbers are MB/s; higher = faster.  x2y2 = FSE's "
           "shipping decode shape.)\n");
    printf("size = source-byte count.  pmaj = P(bit = 0) = major-"
           "symbol probability (each bit drawn IID).\n");
    printf("high pmaj = more zeros = skewed bitmap = tighter "
           "FSE compression.\n");
    fflush(stdout);

    double enc_mbps[16][8];          /* [cell][xi] */
    double dec_mbps[16][32];         /* [cell][cfg_index] */
    int    x_ok_mat[16][8] = {{0}};  /* [cell][xi] */

    for (int ci = 0; ci < n_cells; ci++) {
        size_t bytes = cells[ci].size;
        double p = cells[ci].p_major;
        build_tables_for_p(p);
        if (!g_ct) {
            for (int xi = 0; xi < n_x; xi++) enc_mbps[ci][xi] = -1.0;
            for (size_t k = 0; k < N_CFGS; k++) dec_mbps[ci][k] = -1.0;
            continue;
        }
        if (bytes > sizeof(src)) { free_tables(); continue; }
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
            uint8_t enc_scratch[16384];
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

        free_tables();
    }

    /* Print encode table (varies only with x). */
    printf("\n--- ENCODE (MB/s, varies only with x) ---\n");
    printf("%5s %5s |", "size", "pmaj");
    for (int xi = 0; xi < n_x; xi++) {
        char lbl[8]; snprintf(lbl, sizeof(lbl), "x%d", x_values[xi]);
        printf(" %6s", lbl);
    }
    printf("\n%5s %5s-+", "-----", "----");
    for (int xi = 0; xi < n_x; xi++) printf("-------");
    printf("\n");
    for (int ci = 0; ci < n_cells; ci++) {
        printf("%5zu %5.2f |", cells[ci].size, cells[ci].p_major);
        for (int xi = 0; xi < n_x; xi++) {
            if (enc_mbps[ci][xi] < 0) printf(" %6s", "  -  ");
            else printf(" %6.1f", enc_mbps[ci][xi]);
        }
        printf("\n");
    }

    /* Print decode table (full x × y grid).  Visual gap between
     * x-groups (every 3 cfgs, since y ∈ {1,2,4}). */
    printf("\n--- DECODE (MB/s) ---\n");
    printf("%5s %5s |", "size", "pmaj");
    for (size_t c = 0; c < N_CFGS; c++) {
        if (c > 0 && (c % 3) == 0) printf(" ");
        printf(" %6s", cfgs[c].name);
    }
    printf("\n%5s %5s-+", "-----", "----");
    for (size_t c = 0; c < N_CFGS; c++) {
        if (c > 0 && (c % 3) == 0) printf("-");
        printf("-------");
    }
    printf("\n");
    for (int ci = 0; ci < n_cells; ci++) {
        printf("%5zu %5.2f |", cells[ci].size, cells[ci].p_major);
        for (size_t k = 0; k < N_CFGS; k++) {
            if (k > 0 && (k % 3) == 0) printf(" ");
            if (dec_mbps[ci][k] < 0) printf(" %6s", "  -  ");
            else printf(" %6.1f", dec_mbps[ci][k]);
        }
        printf("\n");
    }

    return 0;
}
