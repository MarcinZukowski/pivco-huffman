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
/* The FSE stats arrays are indexed by t_id in [0, PIVCO_FSE_DYNAMIC_ID]
 * (0 = reject, 1..N = pivco_fse_select_table(), N+1 = the dynamic nibble
 * table).  Guard against the slot count drifting behind the table count
 * (it did once: 26 slots vs 50 tables). */
_Static_assert(PIVCO_FSE_STATS_SLOTS >= PIVCO_FSE_DYNAMIC_ID + 1,
               "PIVCO_FSE_STATS_SLOTS must cover every FSE table id (>= DYNAMIC_ID + 1)");
/* The dynamic id sits immediately past the static schedule, and the wire
 * marker byte only has 7 bits for it (bit 7 is the xor flag). */
_Static_assert(PIVCO_FSE_DYNAMIC_ID == PIVCO_FSE_NUM_TABLES + 1,
               "PIVCO_FSE_DYNAMIC_ID must follow the last static table id");
_Static_assert(PIVCO_FSE_DYNAMIC_ID <= 0x7F,
               "PIVCO_FSE_DYNAMIC_ID must fit the wire marker's 7-bit table field");

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

/* ---------- Dynamic nibble path ----------
 *
 * Deliberately unoptimized: it mallocs a nibble buffer per call and
 * runs stock FSE_compress2 / FSE_decompress_wksp over it.  The point is
 * to establish the wire format and measure the ratio win; the split /
 * merge and the table build are all obvious targets for later work. */

pivco_fse_status_t pivco_fse_compress_dynamic(const void *src, size_t src_len,
                                               void *dst, size_t dst_cap,
                                               size_t *out_len)
{
    if (src_len == 0) { *out_len = 0; return PIVCO_FSE_OK; }

    const uint8_t *s = (const uint8_t *)src;
    /* calloc, not malloc: the loop below writes every byte, but GCC
     * can't see that through the *2 and warns on the FSE_compress2 read. */
    uint8_t *nib = (uint8_t *)calloc(src_len, 2);
    if (!nib) return PIVCO_FSE_ERR_INTERNAL;
    for (size_t i = 0; i < src_len; i++) {
        nib[2 * i]     = (uint8_t)(s[i] & 0x0F);
        nib[2 * i + 1] = (uint8_t)(s[i] >> 4);
    }

    size_t rc = FSE_compress2(dst, dst_cap, nib, src_len * 2,
                              PIVCO_FSE_NIB_MAX_SYMBOL,
                              PIVCO_FSE_NIB_TABLELOG);
    free(nib);

    if (FSE_isError(rc)) return PIVCO_FSE_ERR_INTERNAL;
    /* 0 = FSE judged the nibbles incompressible, 1 = single-symbol RLE.
     * Neither has a payload we could hand back, so both are fallbacks. */
    if (rc <= 1)         return PIVCO_FSE_FALLBACK;
    /* Header included -- this is where the dynamic table pays for itself
     * or doesn't. */
    if (rc >= src_len)   return PIVCO_FSE_FALLBACK;
    *out_len = rc;
    return PIVCO_FSE_OK;
}

pivco_fse_status_t pivco_fse_decompress_dynamic(const void *src, size_t src_len,
                                                 void *dst, size_t dst_cap,
                                                 size_t dst_expected,
                                                 size_t *out_len)
{
    if (dst_cap < dst_expected)  return PIVCO_FSE_ERR_DST_FULL;
    if (dst_expected == 0) { *out_len = 0; return PIVCO_FSE_OK; }

    uint8_t *nib = (uint8_t *)calloc(dst_expected, 2);
    if (!nib) return PIVCO_FSE_ERR_INTERNAL;

    FSE_DTable dt[FSE_DTABLE_SIZE_U32(PIVCO_FSE_NIB_TABLELOG)];
    size_t rc = FSE_decompress_wksp(nib, dst_expected * 2, src, src_len,
                                    dt, PIVCO_FSE_NIB_TABLELOG);
    if (FSE_isError(rc) || rc != dst_expected * 2) {
        free(nib);
        return PIVCO_FSE_ERR_BAD_INPUT;
    }

    uint8_t *d = (uint8_t *)dst;
    for (size_t i = 0; i < dst_expected; i++)
        d[i] = (uint8_t)((nib[2 * i] & 0x0F) | ((nib[2 * i + 1] & 0x0F) << 4));
    free(nib);

    *out_len = dst_expected;
    return PIVCO_FSE_OK;
}

pivco_fse_status_t pivco_fse_compress(int table_id,
                                       const void *src, size_t src_len,
                                       void *dst, size_t dst_cap,
                                       size_t *out_len)
{
    if (table_id == PIVCO_FSE_DYNAMIC_ID)
        return pivco_fse_compress_dynamic(src, src_len, dst, dst_cap, out_len);

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
    if (table_id == PIVCO_FSE_DYNAMIC_ID)
        return pivco_fse_decompress_dynamic(src, src_len, dst, dst_cap,
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
