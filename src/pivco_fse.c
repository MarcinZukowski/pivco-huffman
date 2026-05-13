#include "pivco_fse.h"
#include "pivco_fse_tables.h"

#include "fse.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* One CTable + one DTable per pre-built distribution.  Slot 0 is
 * reserved (matches marker 0 = "no FSE").  Allocated by
 * FSE_createCTable / FSE_createDTable on first use. */
static FSE_CTable *g_ctables[PIVCO_FSE_NUM_TABLES + 1];
static FSE_DTable *g_dtables[PIVCO_FSE_NUM_TABLES + 1];

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_init_ok = 0;

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
    g_init_ok = 1;
}

void pivco_fse_init(void)
{
    pthread_once(&g_once, do_init);
}

int pivco_fse_select_table(double p_major)
{
    /* Largest table index whose tabulated frequency <= p_major.
     * Linear scan top-down -- only 25 entries, no point in being
     * clever; called per non-flat internal node so it should be cheap
     * but not at the cost of code clarity. */
    for (int i = PIVCO_FSE_NUM_TABLES; i >= 1; i--) {
        if (pivco_fse_freq[i] <= p_major) return i;
    }
    return 0;  /* below table 1's threshold -- no FSE */
}

pivco_fse_status_t pivco_fse_compress(int table_id,
                                       const void *src, size_t src_len,
                                       void *dst, size_t dst_cap,
                                       size_t *out_len)
{
    pivco_fse_init();
    if (!g_init_ok) return PIVCO_FSE_ERR_INTERNAL;
    if (table_id < 1 || table_id > PIVCO_FSE_NUM_TABLES)
        return PIVCO_FSE_ERR_BAD_TABLE;
    if (src_len == 0) { *out_len = 0; return PIVCO_FSE_OK; }

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
    pivco_fse_init();
    if (!g_init_ok) return PIVCO_FSE_ERR_INTERNAL;
    if (table_id < 1 || table_id > PIVCO_FSE_NUM_TABLES)
        return PIVCO_FSE_ERR_BAD_TABLE;
    if (dst_cap < dst_expected) return PIVCO_FSE_ERR_DST_FULL;

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
