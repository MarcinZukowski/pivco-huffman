#define FSE_STATIC_LINKING_ONLY
#include "pivco_fse.h"
#include "pivco_fse_tables.h"

#include "fse.h"
#include "bitstream.h"
/* Wide-cursor (multi-state) FSE codec used for the per-node bitmaps:
 * ~1.5-1.6x faster decode than stock 2-state FSE on M4 + c8i.  Pulls in
 * several unused static helpers (the other x*y shapes); silence those.
 *
 * To experiment with a different cursor/unroll shape, override the two
 * macros below together (must be one of the decode_x{N}_y{M} functions
 * in fse_xy_codec.h, with X == N): e.g. -DPIVCO_FSE_XY_X=16
 * -DPIVCO_FSE_XY_DECODE=decode_x16_y2.  Any bitmap byte length >= 64
 * works on the wide path (below that it falls back to stock FSE). */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "fse_xy_codec.h"
#pragma GCC diagnostic pop

#ifndef PIVCO_FSE_XY_X
#define PIVCO_FSE_XY_X       8
#define PIVCO_FSE_XY_DECODE  decode_x8_y1
#endif

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "pivco_huffman.h"   /* PIVCO_FSE_STATS_SLOTS */
/* The FSE stats arrays are indexed by t_id in [0, PIVCO_FSE_NIBBLE_ID]
 * (0 = reject, 1..N = pivco_fse_select_table(), N+1 = the nibble
 * table).  Guard against the slot count drifting behind the table count
 * (it did once: 26 slots vs 50 tables). */
_Static_assert(PIVCO_FSE_STATS_SLOTS >= PIVCO_FSE_NIBBLE_ID + 1,
               "PIVCO_FSE_STATS_SLOTS must cover every FSE table id (>= NIBBLE_ID + 1)");
/* The nibble id sits immediately past the static schedule, and the wire
 * marker byte only has 7 bits for it (bit 7 is the xor flag). */
_Static_assert(PIVCO_FSE_NIBBLE_ID == PIVCO_FSE_NUM_TABLES + 1,
               "PIVCO_FSE_NIBBLE_ID must follow the last static table id");
_Static_assert(PIVCO_FSE_NIBBLE_ID <= 0x7F,
               "PIVCO_FSE_NIBBLE_ID must fit the wire marker's 7-bit table field");
_Static_assert(PIVCO_FSE_NIB_TABLELOG <= PIVCO_FSE_NIB_TABLELOG_MAX,
               "nibble encoder tableLog must not exceed what the decoder accepts");
_Static_assert(PIVCO_FSE_STATS_SLOTS >= PIVCO_FSE_K1_ID + 1,
               "PIVCO_FSE_STATS_SLOTS must cover the k=1 table id");
_Static_assert(PIVCO_FSE_K1_ID == PIVCO_FSE_NIBBLE_ID + 1 && PIVCO_FSE_K1_ID <= 0x7F,
               "PIVCO_FSE_K1_ID must follow the nibble id and fit the 7-bit marker field");
_Static_assert(PIVCO_FSE_STATS_SLOTS >= PIVCO_FSE_K2_ID + 1,
               "PIVCO_FSE_STATS_SLOTS must cover the k=2 table id");
_Static_assert(PIVCO_FSE_K2_ID == PIVCO_FSE_K1_ID + 1 && PIVCO_FSE_K2_ID <= 0x7F,
               "PIVCO_FSE_K2_ID must follow the k=1 id and fit the 7-bit marker field");

/* One CTable + one DTable per pre-built distribution.  Slot 0 is
 * reserved (matches marker 0 = "no FSE").  Allocated by
 * FSE_createCTable / FSE_createDTable on first use. */
static FSE_CTable *g_ctables[PIVCO_FSE_NUM_TABLES + 1];
static FSE_DTable *g_dtables[PIVCO_FSE_NUM_TABLES + 1];

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_init_ok = 0;

/* Wide-cursor FSE on by default (~1.5-1.6x faster decode); PIVCO_FSE_WIDE=0
 * forces stock 2-state (for A/B benchmarking).  Per-table safety: the wide
 * decoder uses FSE_decodeSymbolFast, so it only runs on tables FSE itself
 * marks fast-mode-safe (no zero-bit DTable entries) -- see g_wide_safe in
 * do_init.  The decision is deterministic from (table_id, nbytes), so
 * encode and decode agree without a wire flag. */
static int g_wide_on = 1;
static int g_wide_safe[PIVCO_FSE_NUM_TABLES + 1];

static void do_init(void)
{
    for (int i = 1; i <= PIVCO_FSE_NUM_TABLES; i++) {
        g_ctables[i] = FSE_createCTable(PIVCO_FSE_MAX_SYMBOL,
                                         PIVCO_FSE_TABLE_LOG);
        g_dtables[i] = FSE_createDTable(PIVCO_FSE_TABLE_LOG);
        if (!g_ctables[i] || !g_dtables[i]) return;
        size_t rc;
        rc = FSE_buildCTable(g_ctables[i], pivco_fse_norm[i],
                              PIVCO_FSE_MAX_SYMBOL,
                              PIVCO_FSE_TABLE_LOG);
        if (FSE_isError(rc)) return;
        rc = FSE_buildDTable(g_dtables[i], pivco_fse_norm[i],
                              PIVCO_FSE_MAX_SYMBOL,
                              PIVCO_FSE_TABLE_LOG);
        if (FSE_isError(rc)) return;
    }
    /* The wide decoder uses FSE_decodeSymbolFast / BIT_readBitsFast, which
     * is UB for a zero-bit read.  A DTable entry gets nbBits==0 exactly
     * when a symbol's normalized count >= largeLimit (= tableSize/2) -- the
     * same condition FSE_buildDTable uses to clear its own fastMode flag
     * (fse_decompress.c).  Mirror it precisely (>=, not >), or the wide
     * path mis-decodes high-entropy bitmaps (two_sym 50/50, json, csv). */
    const short large_limit = (short)(1 << (PIVCO_FSE_TABLE_LOG - 1));
    for (int i = 1; i <= PIVCO_FSE_NUM_TABLES; i++) {
        int fast = 1;
        for (int s = 0; s <= PIVCO_FSE_MAX_SYMBOL; s++)
            if (pivco_fse_norm[i][s] >= large_limit) { fast = 0; break; }
        g_wide_safe[i] = fast;
    }
    { const char *e = getenv("PIVCO_FSE_WIDE"); if (e) g_wide_on = (e[0] == '1'); }
    g_init_ok = 1;
}

/* Should this node use the wide-cursor path?  Same deterministic decision
 * on encode + decode (toggle + table-safety + minimum size).
 *
 * The former `(nbytes % PIVCO_FSE_XY_X) == 0` term was a bench-era
 * restriction with an outsized cost: bitmap lengths are ceil(K/8) with
 * data-dependent K, so ~ (X-1)/X of all FSE'd bitmaps silently fell
 * back to stock 2-state FSE (~1.6x slower) — and WHICH ones flipped
 * with any size change, the main source of the FSE-heavy dists'
 * "cursed" M4 variance (proba80 oscillated ±42% with period 64 in the
 * block size: root bitmap = N/8 bytes, divisible by 8 iff N % 64 == 0).
 * encode_x/the decode template now handle any length >= X. */
static inline int use_wide(int table_id, size_t nbytes)
{
    return g_wide_on && g_wide_safe[table_id] && nbytes >= 64;
}

void pivco_fse_init(void)
{
    pthread_once(&g_once, do_init);
}

int pivco_fse_select_table(double p_major)
{
    /* Largest table index whose tabulated frequency <= p_major.
     * Linear scan top-down -- only PIVCO_FSE_NUM_TABLES entries, no point in being
     * clever; called per non-flat internal node so it should be cheap
     * but not at the cost of code clarity. */
    for (int i = PIVCO_FSE_NUM_TABLES; i >= 1; i--) {
        if (pivco_fse_freq[i] <= p_major) return i;
    }
    return 0;  /* below table 1's threshold -- no FSE */
}

/* ---------- Nibble path ----------
 *
 * Payload: [FSE_writeNCount of the 16 nibble counts][bitstream].  The
 * bitstream is the wide-cursor layout of the static path (encode_x /
 * the decode template of fse_xy_codec.h, PIVCO_FSE_NIB_X cursors), so
 * the decoder keeps that many dependent table-load chains in flight
 * instead of stock FSE's two, and it packs each cursor pair's nibbles
 * straight into the output byte: no nibble buffer, no repack pass. */

#ifndef PIVCO_FSE_NIB_X
#define PIVCO_FSE_NIB_X 8
#endif
_Static_assert(PIVCO_FSE_NIB_X == 4 || PIVCO_FSE_NIB_X == 8, "the nibble decoder is written for 4 or 8 cursors");
_Static_assert(4 * PIVCO_FSE_NIB_TABLELOG_MAX <= 64 - 7,
               "four decodes between reloads must fit the bit container");

/* The nibble decoder: one round of PIVCO_FSE_NIB_X decodes is half as
 * many output bytes, low nibble from the even cursor.  DEC is
 * FSE_decodeSymbolFast for a table where no symbol takes half the
 * states or more, else the checked FSE_decodeSymbol.  Termination
 * mirrors the fse_xy_codec.h template. */
#if PIVCO_FSE_NIB_X == 8
#define NIB_ROUND(DEC, base)                                                  \
    op[(base) + 0] = (uint8_t)(DEC(&s[0], &bitD) | (DEC(&s[1], &bitD) << 4)); \
    op[(base) + 1] = (uint8_t)(DEC(&s[2], &bitD) | (DEC(&s[3], &bitD) << 4)); \
    BIT_reloadDStream(&bitD);                                                 \
    op[(base) + 2] = (uint8_t)(DEC(&s[4], &bitD) | (DEC(&s[5], &bitD) << 4)); \
    op[(base) + 3] = (uint8_t)(DEC(&s[6], &bitD) | (DEC(&s[7], &bitD) << 4));
#else
#define NIB_ROUND(DEC, base)                                                  \
    op[(base) + 0] = (uint8_t)(DEC(&s[0], &bitD) | (DEC(&s[1], &bitD) << 4)); \
    op[(base) + 1] = (uint8_t)(DEC(&s[2], &bitD) | (DEC(&s[3], &bitD) << 4));
#endif
#define NIB_RB (PIVCO_FSE_NIB_X / 2)   /* bytes per round */

#define MK_NIB_DECODE(NAME, DEC)                                              \
static size_t NAME(const void *src, size_t src_len,                          \
                   uint8_t *dst, size_t dst_expected,                        \
                   const FSE_DTable *dt)                                     \
{                                                                            \
    BIT_DStream_t bitD;                                                      \
    if (FSE_isError(BIT_initDStream(&bitD, src, src_len))) return 0;         \
    FSE_DState_t s[PIVCO_FSE_NIB_X];                                         \
    for (int k = 0; k < PIVCO_FSE_NIB_X; k++) FSE_initDState(&s[k], &bitD, dt); \
    uint8_t *op = dst;                                                       \
    uint8_t * const olim = dst + dst_expected;                               \
    while ((BIT_reloadDStream(&bitD) == BIT_DStream_unfinished)              \
            & (op + NIB_RB <= olim)) {                                       \
        NIB_ROUND(DEC, 0)                                                    \
        op += NIB_RB;                                                        \
    }                                                                        \
    /* Tail rounds, reload-checked between bytes; once the                \
     * reader overflows the remaining symbols are fixed by the states     \
     * alone (the template's rule), so the round and the partial final    \
     * round finish without checks. */                                    \
    while (op + NIB_RB <= olim) {                                            \
        int overflowed = 0;                                                  \
        for (int j = 0; j < NIB_RB; j++) {                                   \
            uint8_t lo = FSE_decodeSymbol(&s[2 * j], &bitD);                 \
            *op++ = (uint8_t)(lo | (FSE_decodeSymbol(&s[2 * j + 1], &bitD) << 4)); \
            if (BIT_reloadDStream(&bitD) == BIT_DStream_overflow) {          \
                for (int jj = j + 1; jj < NIB_RB && op < olim; jj++) {       \
                    lo = FSE_decodeSymbol(&s[2 * jj], &bitD);                \
                    *op++ = (uint8_t)(lo | (FSE_decodeSymbol(&s[2 * jj + 1], &bitD) << 4)); \
                }                                                            \
                overflowed = 1;                                              \
                break;                                                       \
            }                                                                \
        }                                                                    \
        if (overflowed) break;                                               \
    }                                                                        \
    for (int j = 0; j < NIB_RB && op < olim; j++) {                          \
        uint8_t lo = FSE_decodeSymbol(&s[2 * j], &bitD);                     \
        *op++ = (uint8_t)(lo | (FSE_decodeSymbol(&s[2 * j + 1], &bitD) << 4)); \
    }                                                                        \
    return (size_t)(op - dst);                                               \
}
MK_NIB_DECODE(nib_decode_fast, FSE_decodeSymbolFast)
MK_NIB_DECODE(nib_decode_safe, FSE_decodeSymbol)

pivco_fse_status_t pivco_fse_compress_nibble(const void *src, size_t src_len,
                                               void *dst, size_t dst_cap,
                                               size_t *out_len)
{
    if (src_len == 0) { *out_len = 0; return PIVCO_FSE_OK; }
    if (src_len * 2 < PIVCO_FSE_NIB_X) return PIVCO_FSE_FALLBACK;   /* one nibble per cursor */

    const uint8_t *s = (const uint8_t *)src;
    uint8_t *nib = (uint8_t *)malloc(src_len * 2);
    if (!nib) return PIVCO_FSE_ERR_INTERNAL;
    unsigned cnt[16] = {0};
    for (size_t i = 0; i < src_len; i++) {
        nib[2 * i]     = (uint8_t)(s[i] & 0x0F);
        nib[2 * i + 1] = (uint8_t)(s[i] >> 4);
        cnt[s[i] & 0x0F]++;
        cnt[s[i] >> 4]++;
    }
    /* A single nibble value: nothing to code (stock FSE's RLE case). */
    int used = 0;
    for (int v = 0; v < 16; v++) used += cnt[v] > 0;
    if (used < 2) { free(nib); return PIVCO_FSE_FALLBACK; }

    unsigned tl = FSE_optimalTableLog(PIVCO_FSE_NIB_TABLELOG, src_len * 2, PIVCO_FSE_NIB_MAX_SYMBOL);
    short nc[PIVCO_FSE_NIB_MAX_SYMBOL + 1];
    size_t rc = FSE_normalizeCount(nc, tl, cnt, src_len * 2, PIVCO_FSE_NIB_MAX_SYMBOL);
    if (FSE_isError(rc)) { free(nib); return PIVCO_FSE_FALLBACK; }
    tl = (unsigned)rc;
    uint8_t *d = (uint8_t *)dst;
    size_t hlen = FSE_writeNCount(d, dst_cap, nc, PIVCO_FSE_NIB_MAX_SYMBOL, tl);
    if (FSE_isError(hlen)) { free(nib); return PIVCO_FSE_FALLBACK; }

    FSE_CTable ct[FSE_CTABLE_SIZE_U32(PIVCO_FSE_NIB_TABLELOG_MAX, PIVCO_FSE_NIB_MAX_SYMBOL)];
    rc = FSE_buildCTable(ct, nc, PIVCO_FSE_NIB_MAX_SYMBOL, tl);
    if (FSE_isError(rc)) { free(nib); return PIVCO_FSE_ERR_INTERNAL; }
    size_t blen = encode_x(PIVCO_FSE_NIB_X, nib, src_len * 2, d + hlen, dst_cap - hlen, ct);
    free(nib);
    if (blen == 0) return PIVCO_FSE_FALLBACK;
    /* Header included -- this is where the nibble table pays for itself
     * or doesn't. */
    if (hlen + blen >= src_len) return PIVCO_FSE_FALLBACK;
    *out_len = hlen + blen;
    return PIVCO_FSE_OK;
}

pivco_fse_status_t pivco_fse_decompress_nibble(const void *src, size_t src_len,
                                                 void *dst, size_t dst_cap,
                                                 size_t dst_expected,
                                                 size_t *out_len)
{
    if (dst_cap < dst_expected)  return PIVCO_FSE_ERR_DST_FULL;
    if (dst_expected == 0) { *out_len = 0; return PIVCO_FSE_OK; }

    short nc[PIVCO_FSE_NIB_MAX_SYMBOL + 1];
    unsigned max_sym = PIVCO_FSE_NIB_MAX_SYMBOL, tl = 0;
    size_t hlen = FSE_readNCount(nc, &max_sym, &tl, src, src_len);
    if (FSE_isError(hlen) || max_sym > PIVCO_FSE_NIB_MAX_SYMBOL
        || tl > PIVCO_FSE_NIB_TABLELOG_MAX || hlen >= src_len)
        return PIVCO_FSE_ERR_BAD_INPUT;
    FSE_DTable dt[FSE_DTABLE_SIZE_U32(PIVCO_FSE_NIB_TABLELOG_MAX)];
    if (FSE_isError(FSE_buildDTable(dt, nc, max_sym, tl))) return PIVCO_FSE_ERR_BAD_INPUT;
    /* The fast decode step reads at least one bit per symbol; a symbol
     * holding half the states or more takes none (FSE's own fastMode
     * rule), so such a table decodes through the checked step. */
    int fast = 1;
    const short large_limit = (short)(1 << (tl - 1));
    for (unsigned v = 0; v <= max_sym; v++) if (nc[v] >= large_limit) fast = 0;
    size_t r = fast ? nib_decode_fast((const uint8_t *)src + hlen, src_len - hlen,
                                      (uint8_t *)dst, dst_expected, dt)
                    : nib_decode_safe((const uint8_t *)src + hlen, src_len - hlen,
                                      (uint8_t *)dst, dst_expected, dt);
    if (r != dst_expected) return PIVCO_FSE_ERR_BAD_INPUT;
    *out_len = dst_expected;
    return PIVCO_FSE_OK;
}

pivco_fse_status_t pivco_fse_compress(int table_id,
                                       const void *src, size_t src_len,
                                       void *dst, size_t dst_cap,
                                       size_t *out_len)
{
    if (table_id == PIVCO_FSE_NIBBLE_ID)
        return pivco_fse_compress_nibble(src, src_len, dst, dst_cap, out_len);
    if (table_id == PIVCO_FSE_K1_ID)
        return pivco_k1_compress(src, src_len, dst, dst_cap, 0, out_len);
    if (table_id == PIVCO_FSE_K2_ID)
        return pivco_k2_compress(src, src_len, dst, dst_cap, 0, out_len);

    pivco_fse_init();
    if (!g_init_ok) return PIVCO_FSE_ERR_INTERNAL;
    if (table_id < 1 || table_id > PIVCO_FSE_NUM_TABLES)
        return PIVCO_FSE_ERR_BAD_TABLE;
    if (src_len == 0) { *out_len = 0; return PIVCO_FSE_OK; }

    if (use_wide(table_id, src_len)) {
        size_t r = encode_x(PIVCO_FSE_XY_X, (const uint8_t *)src, src_len,
                            dst, dst_cap, g_ctables[table_id]);
        if (r == 0 || r >= src_len) return PIVCO_FSE_FALLBACK;
        *out_len = r;
        return PIVCO_FSE_OK;
    }

    size_t rc = FSE_compress_usingCTable(dst, dst_cap, src, src_len,
                                          g_ctables[table_id]);
    if (FSE_isError(rc)) return PIVCO_FSE_ERR_INTERNAL;
    /* rc == 0 means "input is not compressible" (RLE/incompressible).
     * For our purposes that's a fallback. */
    if (rc == 0 || rc == 1) return PIVCO_FSE_FALLBACK;
    if (rc >= src_len)      return PIVCO_FSE_FALLBACK;
    *out_len = rc;
    return PIVCO_FSE_OK;
}

pivco_fse_status_t pivco_fse_decompress(int table_id,
                                         const void *src, size_t src_len,
                                         void *dst, size_t dst_cap,
                                         size_t dst_expected,
                                         size_t *out_len)
{
    if (table_id == PIVCO_FSE_NIBBLE_ID)
        return pivco_fse_decompress_nibble(src, src_len, dst, dst_cap,
                                             dst_expected, out_len);
    if (table_id == PIVCO_FSE_K1_ID)
        return pivco_k1_decompress(src, src_len, dst, dst_cap,
                                   dst_expected, out_len);
    if (table_id == PIVCO_FSE_K2_ID)
        return pivco_k2_decompress(src, src_len, dst, dst_cap,
                                   dst_expected, out_len);

    pivco_fse_init();
    if (!g_init_ok) return PIVCO_FSE_ERR_INTERNAL;
    if (table_id < 1 || table_id > PIVCO_FSE_NUM_TABLES)
        return PIVCO_FSE_ERR_BAD_TABLE;
    if (dst_cap < dst_expected) return PIVCO_FSE_ERR_DST_FULL;

    if (use_wide(table_id, dst_expected)) {
        size_t r = PIVCO_FSE_XY_DECODE((const void *)src, src_len,
                                       (uint8_t *)dst, dst_expected,
                                       g_dtables[table_id]);
        if (r != dst_expected) return PIVCO_FSE_ERR_BAD_INPUT;
        *out_len = r;
        return PIVCO_FSE_OK;
    }

    size_t rc = FSE_decompress_usingDTable(dst, dst_cap, src, src_len,
                                            g_dtables[table_id]);
    if (FSE_isError(rc)) return PIVCO_FSE_ERR_BAD_INPUT;
    if (rc != dst_expected) return PIVCO_FSE_ERR_BAD_INPUT;
    *out_len = rc;
    return PIVCO_FSE_OK;
}

void pivco_fse_flip_bits(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)~buf[i];
}
