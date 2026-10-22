#include "pivco_huffman.h"

#include <string.h>

/* PIVCO_CHECK failure path (see pivco_check.h): print and crash --
 * internal invariants must fail loudly in every build type. */
#include "pivco_check.h"
#include <stdio.h>
#include <stdlib.h>
void pivco_check_fail(const char *expr, const char *file, int line)
{
    fprintf(stderr, "PIVCO_CHECK failed: %s (%s:%d)\n", expr, file, line);
    fflush(NULL);
    abort();
}

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
pivco_fse_root_event_t g_pivco_fse_root_log[PIVCO_FSE_ROOT_LOG_MAX];
int g_pivco_fse_root_n;

void pivco_fse_stats_reset(void)
{
    memset(g_pivco_fse_commit,    0, sizeof(g_pivco_fse_commit));
    memset(g_pivco_fse_attempt,   0, sizeof(g_pivco_fse_attempt));
    memset(g_pivco_fse_bytes_in,  0, sizeof(g_pivco_fse_bytes_in));
    memset(g_pivco_fse_bytes_out, 0, sizeof(g_pivco_fse_bytes_out));
    g_pivco_fse_root_n = 0;
}

void pivco_fse_stats_get(uint64_t commit[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t attempt[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t bytes_in[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t bytes_out[PIVCO_FSE_STATS_SLOTS])
{
    memcpy(commit,    g_pivco_fse_commit,    sizeof(g_pivco_fse_commit));
    memcpy(attempt,   g_pivco_fse_attempt,   sizeof(g_pivco_fse_attempt));
    memcpy(bytes_in,  g_pivco_fse_bytes_in,  sizeof(g_pivco_fse_bytes_in));
    memcpy(bytes_out, g_pivco_fse_bytes_out, sizeof(g_pivco_fse_bytes_out));
}

int pivco_fse_root_count(void)
{
    return g_pivco_fse_root_n;
}

void pivco_fse_root_get(int idx, pivco_fse_root_event_t *out)
{
    if (idx < 0 || idx >= g_pivco_fse_root_n) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = g_pivco_fse_root_log[idx];
}

/* Compile-time backend choice for the dispatched entries: the build
 * enables exactly one SIMD tier (or none), so there is nothing to
 * select at runtime. */
int pivco_encode(pivco_encoder_t *enc, const pivco_table_t *table,
                 const uint8_t *symbols, size_t n,
                 uint8_t *out, size_t *out_len)
{
    int rc;
#ifdef PIVCO_HAS_AVX512
    rc = pivco_encode_avx512(enc, table, symbols, n, out, out_len);
#elif defined(PIVCO_HAS_NEON)
    rc = pivco_encode_neon(enc, table, symbols, n, out, out_len);
#elif defined(PIVCO_HAS_SSE4)
    rc = pivco_encode_x86(enc, table, symbols, n, out, out_len);
#else
    rc = pivco_encode_scalar(enc, table, symbols, n, out, out_len);
#endif
    if (rc == PIVCO_OK) {
        enc->stats.blocks++;
        enc->stats.bytes_in += n;
        enc->stats.bytes_out += *out_len;
    }
    return rc;
}

int pivco_decode(pivco_decoder_t *dec, const pivco_table_t *table,
                 const uint8_t *in, size_t in_len,
                 uint8_t *symbols, size_t *consumed)
{
    int rc;
#ifdef PIVCO_HAS_AVX512
    rc = pivco_decode_bu_avx512(dec, table, in, in_len, symbols, consumed);
#elif defined(PIVCO_HAS_NEON)
    rc = pivco_decode_bu_neon(dec, table, in, in_len, symbols, consumed);
#elif defined(PIVCO_HAS_SSE4)
    rc = pivco_decode_bu_x86(dec, table, in, in_len, symbols, consumed);
#else
    rc = pivco_decode_scalar(dec, table, in, in_len, symbols, consumed);
#endif
    if (rc == PIVCO_OK) {
        dec->stats.blocks++;
        dec->stats.bytes_in += in_len;
        dec->stats.bytes_out += *consumed ? *consumed : 0;
    }
    return rc;
}

/* ---------------- contexts ---------------- */

const pivco_cfg_t pivco_cfg_default = {
    .tree_mode   = PIVCO_TREE_MODE_OPTIMIZED,
    .effort      = PIVCO_EFFORT_PLAIN,
    .fse_enabled = 1,
    .fse_static_enabled = 1,
    .fse_nibble_enabled = 0,
    .fse_k1_enabled = 0,
    .fse_k2_enabled = 0,
    .fse_dhuf0_enabled = 0,
    .fse_dnph_enabled = 0,
    .fse_snph_enabled = 0,
    .flat_layout = PIVCO_FLAT_VERTICAL,
    .krt = 0,
};

#include "pivco_huffman_common.h"
#include "pivco_huffman_primitives.h"   /* arch-selected; prim_histogram_chunk */
#include "pivco_nph.h"

static pivco_scratch_t *scratch_create(int max_n)
{
    pivco_scratch_t *sc = (pivco_scratch_t *)calloc(1, sizeof(*sc));
    if (!sc) return NULL;
    sc->enc_cap = PIVCO_ENC_SCRATCH_BYTES(max_n);
    sc->dec_cap = PIVCO_DEC_SCRATCH_BYTES(max_n)
                + DECODE_SCRATCH_ALIGN + DECODE_SCRATCH_SHIFT;
    /* PH_CTX_ALLOC=1 placement A/B probe: start from a deliberately
     * small 256 KiB cap instead of the full preallocation, mimicking
     * the retired TLS arenas' grow-on-demand footprint (the ensure
     * helpers realloc up on first oversized block). */
#define PH_CTX_ALLOC_PROBE_CAP ((size_t)256 * 1024)
    const char *am = getenv("PH_CTX_ALLOC");
    if (am && am[0] == '1') {
        sc->enc_cap = PH_CTX_ALLOC_PROBE_CAP;
        sc->dec_cap = PH_CTX_ALLOC_PROBE_CAP
                    + DECODE_SCRATCH_ALIGN + DECODE_SCRATCH_SHIFT;
        sc->enc = (uint8_t *)malloc(sc->enc_cap);
        sc->dec = (uint8_t *)malloc(sc->dec_cap);
    } else {
        sc->enc = (uint8_t *)malloc(sc->enc_cap);
        sc->dec = (uint8_t *)malloc(sc->dec_cap);
    }
    if (!sc->enc || !sc->dec) {
        free(sc->enc); free(sc->dec); free(sc);
        return NULL;
    }
    return sc;
}

static void scratch_free(pivco_scratch_t *sc)
{
    if (!sc) return;
    if (sc->nested) {
        pivco_encoder_free(sc->nested->enc);
        pivco_decoder_free(sc->nested->dec);
        for (int t = 0; t <= PIVCO_SNPH_NUM_TABLES; t++)
            free(sc->nested->snph[t]);
        free(sc->nested->out);
        free(sc->nested->flipped_bitmap);
        free(sc->nested);
    }
    free(sc->enc); free(sc->dec); free(sc);
}

static pivco_encoder_t *encoder_create(int max_n)
{
    pivco_encoder_t *e = (pivco_encoder_t *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->internal = scratch_create(max_n);
    if (!e->internal) { free(e); return NULL; }
    prim_codec_init();
    return e;
}

static pivco_decoder_t *decoder_create(int max_n)
{
    pivco_decoder_t *d = (pivco_decoder_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->internal = scratch_create(max_n);
    if (!d->internal) { free(d); return NULL; }
    prim_codec_init();
    return d;
}

pivco_encoder_t *pivco_encoder_create(void)
{
    return encoder_create(PIVCO_WIRE_MAX_N);
}

pivco_decoder_t *pivco_decoder_create(void)
{
    return decoder_create(PIVCO_WIRE_MAX_N);
}

/* Staging for one inner block: what the encoder writes for an N-byte
 * region (PIVCO_MAX_ENCODED_SIZE's 2 bytes per symbol), plus room for
 * the code-length record in front. */
#define PIVCO_NPH_OUT_BYTES ((size_t)PIVCO_NPH_MAX_N * 2 + 4096)

pivco_nested_t *pivco_nested_for_encoder(pivco_scratch_t *sc)
{
    if (sc->nested) return sc->nested;
    if (sc->is_child) return NULL;
    pivco_nested_t *nested = (pivco_nested_t *)calloc(1, sizeof(*nested));
    if (!nested) return NULL;
    nested->enc = encoder_create(PIVCO_NPH_MAX_N);
    nested->out = (uint8_t *)malloc(PIVCO_NPH_OUT_BYTES);
    nested->flipped_bitmap = (uint8_t *)malloc(PIVCO_NPH_MAX_N);
    if (!nested->enc || !nested->out || !nested->flipped_bitmap) {
        pivco_encoder_free(nested->enc);
        free(nested->out);
        free(nested->flipped_bitmap);
        free(nested);
        return NULL;
    }
    ((pivco_scratch_t *)nested->enc->internal)->is_child = 1;
    sc->nested = nested;
    return nested;
}

pivco_nested_t *pivco_nested_for_decoder(pivco_scratch_t *sc)
{
    if (sc->nested) return sc->nested;
    if (sc->is_child) return NULL;
    pivco_nested_t *nested = (pivco_nested_t *)calloc(1, sizeof(*nested));
    if (!nested) return NULL;
    nested->dec = decoder_create(PIVCO_NPH_MAX_N);
    if (!nested->dec) {
        free(nested);
        return NULL;
    }
    ((pivco_scratch_t *)nested->dec->internal)->is_child = 1;
    sc->nested = nested;
    return nested;
}

void pivco_encoder_free(pivco_encoder_t *e)
{
    if (!e) return;
    scratch_free((pivco_scratch_t *)e->internal);
    free(e);
}

void pivco_decoder_free(pivco_decoder_t *d)
{
    if (!d) return;
    scratch_free((pivco_scratch_t *)d->internal);
    free(d);
}

/* ---------------- histogram ---------------- */

int pivco_histogram(pivco_encoder_t *enc, const uint8_t *in, size_t n,
                    uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    if (!enc || (!in && n) || !freq) return PIVCO_ERR_NULL;
    pivco_scratch_t *sc = (pivco_scratch_t *)enc->internal;
    /* hist bins live in the tail of the encode arena (never concurrent
     * with an encode call: contexts are single-threaded) */
    uint8_t *scratch = sc->enc + sc->enc_cap - PIVCO_PRIM_HIST_SCRATCH_MAX;
    size_t off = 0;
    while (off < n) {
        size_t len = n - off;
        if (len > PIVCO_PRIM_HIST_CHUNK) len = PIVCO_PRIM_HIST_CHUNK;
        uint32_t h32[256] = {0};
        prim_histogram_chunk(in + off, len, h32, scratch);
        for (int sym = 0; sym < 256; sym++) freq[sym] += h32[sym];
        off += len;
    }
    return PIVCO_OK;
}
