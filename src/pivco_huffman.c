#include "pivco_huffman.h"

static pivco_impl_t g_impl = PIVCO_IMPL_AUTO;

void pivco_huffman_set_impl(pivco_impl_t impl)
{
    g_impl = impl;
}

pivco_impl_t pivco_huffman_get_impl(void)
{
    return g_impl;
}

static pivco_impl_t resolve_impl(void)
{
    if (g_impl != PIVCO_IMPL_AUTO) return g_impl;
#ifdef PIVCO_HAS_NEON
    return PIVCO_IMPL_NEON;
#else
    return PIVCO_IMPL_SCALAR;
#endif
}

int pivco_huffman_encode(const uint8_t *symbols,
                         const pivco_huffman_table_t *table,
                         uint8_t *out, size_t *out_len)
{
    switch (resolve_impl()) {
#ifdef PIVCO_HAS_NEON
    case PIVCO_IMPL_NEON:
        return pivco_huffman_encode_neon(symbols, table, out, out_len);
#endif
    default:
        return pivco_huffman_encode_scalar(symbols, table, out, out_len);
    }
}

int pivco_huffman_decode(const uint8_t *in, size_t in_len,
                         const pivco_huffman_table_t *table,
                         uint8_t *symbols, size_t *consumed)
{
    switch (resolve_impl()) {
#ifdef PIVCO_HAS_NEON
    case PIVCO_IMPL_NEON:
        return pivco_huffman_decode_neon(in, in_len, table, symbols, consumed);
#endif
    default:
        return pivco_huffman_decode_scalar(in, in_len, table, symbols, consumed);
    }
}
