/* pivco_huffman.c — top-down decoder dispatcher.
 *
 * Standalone NEON-only TD slice extracted from the historical
 * pivco-huffman codebase at SHA 31cbf75 (2026-05-14, just before the
 * TD entry points were retired from the public API).  See
 * extras/ph-td/README.md for context.
 *
 * Differences from upstream:
 *   - decode dispatches to the TD entry pivco_huffman_decode_neon
 *     (the production tree was switched to BU tree_merge in 2026-05-12).
 *   - x86 / AVX-512 / SVE branches removed.
 *   - FSE per-table-id stats / event log removed (FSE codepath is
 *     disabled in this slice — built without PIVCO_HAS_FSE). */

#include "pivco_huffman.h"

#include <string.h>

static pivco_impl_t g_impl = PIVCO_IMPL_AUTO;

void pivco_huffman_set_impl(pivco_impl_t impl)
{
    g_impl = impl;
}

pivco_impl_t pivco_huffman_get_impl(void)
{
    return g_impl;
}

/* FSE toggle kept as a no-op for API compatibility with the upstream
 * public header.  The TD slice is built without PIVCO_HAS_FSE so the
 * encoder's FSE attempt path is fully #ifdef'd out anyway. */
static int g_fse_enabled = 0;

void pivco_huffman_set_fse_enabled(int enabled) { (void)enabled; }
int  pivco_huffman_get_fse_enabled(void)        { return g_fse_enabled; }

int pivco_huffman_encode(const uint8_t *symbols,
                         const pivco_huffman_table_t *table,
                         uint8_t *out, size_t *out_len)
{
#ifdef PIVCO_HAS_NEON
    return pivco_huffman_encode_neon(symbols, table, out, out_len);
#else
    (void)symbols; (void)table; (void)out; (void)out_len;
    return PIVCO_ERR_NULL;
#endif
}

int pivco_huffman_decode(const uint8_t *in, size_t in_len,
                         const pivco_huffman_table_t *table,
                         uint8_t *symbols, size_t *consumed)
{
#ifdef PIVCO_HAS_NEON
    /* Top-down stream-scatter decode (the historical primary path). */
    return pivco_huffman_decode_neon(in, in_len, table, symbols, consumed);
#else
    (void)in; (void)in_len; (void)table; (void)symbols; (void)consumed;
    return PIVCO_ERR_NULL;
#endif
}
