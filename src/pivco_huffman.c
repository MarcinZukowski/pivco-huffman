#include "pivco_huffman.h"

#include <string.h>

static pivco_impl_t g_impl = PIVCO_IMPL_AUTO;

/* ---------- FSE per-table-id stats storage ----------
 *
 * Backend-neutral home for the FSE-encode instrumentation counters.
 * Defined here so codec.c (compiled per-backend) and any legacy
 * backend-specific .c files all link against the same storage; before
 * this lived in pivco_huffman_neon.c as static, which broke
 * pivco_bench_fse_table_use on x86 hosts where neon.c isn't compiled.
 *
 * Slot 0 of `commit` counts "FSE attempted but rejected" (codeword-cost
 * gate refused or the FSE library returned fallback).  Slots 1..25 of
 * commit/bytes_in/bytes_out are per-table-id committed FSE encodes.
 * attempt[t_id] counts every call to pivco_fse_compress for table t_id
 * whether or not it committed.  Not thread-safe -- debug instrumentation
 * only; the codec mutates these inline during encode. */
uint64_t g_pivco_fse_commit  [PIVCO_FSE_STATS_SLOTS];
uint64_t g_pivco_fse_attempt [PIVCO_FSE_STATS_SLOTS];
uint64_t g_pivco_fse_bytes_in [PIVCO_FSE_STATS_SLOTS];
uint64_t g_pivco_fse_bytes_out[PIVCO_FSE_STATS_SLOTS];

#define PIVCO_FSE_ROOT_LOG_MAX 65536
pivco_huffman_fse_root_event_t g_pivco_fse_root_log[PIVCO_FSE_ROOT_LOG_MAX];
int g_pivco_fse_root_n;

void pivco_huffman_fse_stats_reset(void)
{
    memset(g_pivco_fse_commit,    0, sizeof(g_pivco_fse_commit));
    memset(g_pivco_fse_attempt,   0, sizeof(g_pivco_fse_attempt));
    memset(g_pivco_fse_bytes_in,  0, sizeof(g_pivco_fse_bytes_in));
    memset(g_pivco_fse_bytes_out, 0, sizeof(g_pivco_fse_bytes_out));
    g_pivco_fse_root_n = 0;
}

void pivco_huffman_fse_stats_get(uint64_t commit[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t attempt[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t bytes_in[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t bytes_out[PIVCO_FSE_STATS_SLOTS])
{
    memcpy(commit,    g_pivco_fse_commit,    sizeof(g_pivco_fse_commit));
    memcpy(attempt,   g_pivco_fse_attempt,   sizeof(g_pivco_fse_attempt));
    memcpy(bytes_in,  g_pivco_fse_bytes_in,  sizeof(g_pivco_fse_bytes_in));
    memcpy(bytes_out, g_pivco_fse_bytes_out, sizeof(g_pivco_fse_bytes_out));
}

int pivco_huffman_fse_root_count(void)
{
    return g_pivco_fse_root_n;
}

void pivco_huffman_fse_root_get(int idx, pivco_huffman_fse_root_event_t *out)
{
    if (idx < 0 || idx >= g_pivco_fse_root_n) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = g_pivco_fse_root_log[idx];
}

void pivco_huffman_set_impl(pivco_impl_t impl)
{
    g_impl = impl;
}

pivco_impl_t pivco_huffman_get_impl(void)
{
    return g_impl;
}

/* Runtime FSE encode-dispatch toggle.  Default = enabled.  When 0, the
 * encoder's FSE dispatch path is skipped entirely -- marker byte stays
 * 0, raw bitmap emitted, decoder reads it unchanged.  Wire format is
 * always v0.2+ compatible regardless of this setting. */
static int g_fse_enabled = 1;

void pivco_huffman_set_fse_enabled(int enabled)
{
    g_fse_enabled = enabled ? 1 : 0;
}

int pivco_huffman_get_fse_enabled(void)
{
    return g_fse_enabled;
}

static pivco_impl_t resolve_impl(void)
{
    if (g_impl != PIVCO_IMPL_AUTO) return g_impl;
#ifdef PIVCO_HAS_AVX512
    return PIVCO_IMPL_NEON; /* reuse enum — best SIMD path */
#elif defined(PIVCO_HAS_SVE)
    return PIVCO_IMPL_NEON;
#elif defined(PIVCO_HAS_NEON)
    return PIVCO_IMPL_NEON;
#elif defined(PIVCO_HAS_SSE4)
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
    case PIVCO_IMPL_NEON:
#ifdef PIVCO_HAS_AVX512
        return pivco_huffman_encode_avx512(symbols, table, out, out_len);
#elif defined(PIVCO_HAS_SVE)
        return pivco_huffman_encode_sve(symbols, table, out, out_len);
#elif defined(PIVCO_HAS_NEON)
        return pivco_huffman_encode_neon(symbols, table, out, out_len);
#elif defined(PIVCO_HAS_SSE4)
        return pivco_huffman_encode_x86(symbols, table, out, out_len);
#endif
    default:
        return pivco_huffman_encode_scalar(symbols, table, out, out_len);
    }
}

int pivco_huffman_decode(const uint8_t *in, size_t in_len,
                         const pivco_huffman_table_t *table,
                         uint8_t *symbols, size_t *consumed)
{
    /* Bottom-up tree_merge is the primary decode path (2026-05-12 K_right
     * landing -- see results/SUMMARY-20260512-kr-landing.md).  Top-down
     * stream-scatter is retired but kept compiled in for ratio comparisons
     * via pivco_huffman_decode_{neon,x86,avx512} direct entry points. */
    switch (resolve_impl()) {
    case PIVCO_IMPL_NEON:
#if defined(PIVCO_HAS_AVX512) || defined(PIVCO_HAS_SSE4)
        return pivco_huffman_decode_bu_x86(in, in_len, table, symbols, consumed);
#elif defined(PIVCO_HAS_NEON)
        return pivco_huffman_decode_bu_neon(in, in_len, table, symbols, consumed);
#endif
    default:
        return pivco_huffman_decode_scalar(in, in_len, table, symbols, consumed);
    }
}
