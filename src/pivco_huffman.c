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

#ifdef PIVCO_CODEC_DISPATCH
#include "pivco_cpu.h"
#include "pivco_prims.h"

/* SIMD tiers, each a superset of the one below.  A codec or primitive
 * table runs only on a CPU at or above its tier. */
enum { PIVCO_TIER_BASE, PIVCO_TIER_SSE4, PIVCO_TIER_AVX2, PIVCO_TIER_AVX512, PIVCO_TIER_COUNT };
static const char *const pivco_tier_names[] = { "base", "sse4.1", "avx2", "avx512-vbmi2" };
_Static_assert(sizeof pivco_tier_names / sizeof pivco_tier_names[0] == PIVCO_TIER_COUNT,
               "one name per tier");

static int pivco_cpu_tier(void)
{
#if defined(__x86_64__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vbmi")
        && __builtin_cpu_supports("avx512vbmi2") && __builtin_cpu_supports("avx512vpopcntdq")
        && __builtin_cpu_supports("gfni") && __builtin_cpu_supports("avx512bitalg"))
        return PIVCO_TIER_AVX512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2")
        && __builtin_cpu_supports("popcnt"))
        return PIVCO_TIER_AVX2;
    if (__builtin_cpu_supports("sse4.1") && __builtin_cpu_supports("popcnt"))
        return PIVCO_TIER_SSE4;
#endif
    return PIVCO_TIER_BASE;
}

#if defined(PIVCO_BUILT_AVX2)
extern const pivco_prims_t *const pivco_prims_x86_avx2;
extern const pivco_prims_t *const pivco_prims_x86_avx2_intel;
#endif
static const struct {
    const char *name;
    int tier;
    const pivco_prims_t *const *prims;
} pivco_prims_tables[] = {
    { "scalar", PIVCO_TIER_BASE, &pivco_prims_scalar },
#if defined(PIVCO_HAS_NEON)
    { "neon", PIVCO_TIER_BASE, &pivco_prims_neon },
#endif
#if defined(PIVCO_HAS_SSE4)
    { "x86", PIVCO_TIER_SSE4, &pivco_prims_x86 },
#endif
#if defined(PIVCO_BUILT_AVX2)
    { "x86_avx2", PIVCO_TIER_AVX2, &pivco_prims_x86_avx2 },
    { "x86_avx2_intel", PIVCO_TIER_AVX2, &pivco_prims_x86_avx2_intel },
#endif
#if defined(PIVCO_HAS_AVX512)
    { "avx512", PIVCO_TIER_AVX512, &pivco_prims_avx512 },
#endif
};
#define PIVCO_N_PRIMS_TABLES (int)(sizeof pivco_prims_tables / sizeof pivco_prims_tables[0])
const pivco_prims_t *g_pivco_prims;

typedef int (*pivco_encode_fn)(pivco_encoder_t *, const pivco_table_t *,
                               const uint8_t *, size_t, uint8_t *, size_t *);
typedef int (*pivco_decode_fn)(pivco_decoder_t *, const pivco_table_t *,
                               const uint8_t *, size_t, uint8_t *, size_t *);

/* A codec: its entry points and the SIMD tier it needs. */
typedef struct {
    const char *name;
    int tier;
    pivco_encode_fn enc;
    pivco_decode_fn dec;
} pivco_codec_t;

#define PIVCO_DECLARE_CODEC(s)                                              \
    int pivco_encode_##s(pivco_encoder_t *, const pivco_table_t *,          \
                         const uint8_t *, size_t, uint8_t *, size_t *);     \
    int pivco_decode_bu_##s(pivco_decoder_t *, const pivco_table_t *,       \
                            const uint8_t *, size_t, uint8_t *, size_t *);
PIVCO_DECLARE_CODEC(prims)
#if defined(PIVCO_BUILT_AVX2)
PIVCO_DECLARE_CODEC(x86_avx2)
PIVCO_DECLARE_CODEC(x86_avx2_intel)
#endif
static const pivco_codec_t pivco_codecs[] = {
    { "scalar", PIVCO_TIER_BASE, pivco_encode_scalar, pivco_decode_scalar },
#if defined(PIVCO_HAS_NEON)
    { "neon", PIVCO_TIER_BASE, pivco_encode_neon, pivco_decode_bu_neon },
#endif
#if defined(PIVCO_HAS_SSE4)
    { "x86", PIVCO_TIER_SSE4, pivco_encode_x86, pivco_decode_bu_x86 },
#endif
#if defined(PIVCO_BUILT_AVX2)
    { "x86_avx2", PIVCO_TIER_AVX2, pivco_encode_x86_avx2, pivco_decode_bu_x86_avx2 },
    { "x86_avx2_intel", PIVCO_TIER_AVX2, pivco_encode_x86_avx2_intel, pivco_decode_bu_x86_avx2_intel },
#endif
#if defined(PIVCO_HAS_AVX512)
    { "avx512", PIVCO_TIER_AVX512, pivco_encode_avx512, pivco_decode_bu_avx512 },
#endif
    { "prims", PIVCO_TIER_BASE, pivco_encode_prims, pivco_decode_bu_prims },
};
#define PIVCO_N_CODECS (int)(sizeof pivco_codecs / sizeof pivco_codecs[0])
static int pivco_codec_idx;

const char *pivco_codec_name(void)
{
    PIVCO_CHECK(pivco_codec_idx >= 0 && pivco_codec_idx < PIVCO_N_CODECS);
    return pivco_codecs[pivco_codec_idx].name;
}

const char *pivco_prims_name(void)
{
    PIVCO_CHECK(g_pivco_prims != NULL);
    return g_pivco_prims->name;
}

/* Index of `name` among the primitive tables, or -1. */
static int pivco_find_prims(const char *name)
{
    for (int i = 0; i < PIVCO_N_PRIMS_TABLES; i++)
        if (!strcmp(pivco_prims_tables[i].name, name))
            return i;
    return -1;
}

/* Index of `name` among the codecs, or -1. */
static int pivco_find_codec(const char *name)
{
    for (int i = 0; i < PIVCO_N_CODECS; i++)
        if (!strcmp(pivco_codecs[i].name, name))
            return i;
    return -1;
}

/* The best table this CPU runs: its highest built tier, refined by
 * vendor.  Codec and table names match, so it names the codec too. */
static const char *pivco_default_name(int cpu_tier)
{
    int intel = pivco_cpu()->vendor == PIVCO_VENDOR_INTEL;
    const char *order[] = {
        "avx512",
        intel ? "x86_avx2_intel" : "x86_avx2",
        "x86",
        "neon",
        "scalar",
    };
    for (int i = 0; i < (int)(sizeof order / sizeof order[0]); i++) {
        int t = pivco_find_prims(order[i]);
        if (t >= 0 && pivco_prims_tables[t].tier <= cpu_tier)
            return order[i];
    }
    return "scalar";
}

/* PIVCO_CODEC=<name> picks the codec, PIVCO_PRIMS=<name> the table the
 * prims codec calls through; both default to pivco_default_name().  A
 * name the CPU can't run aborts. */
__attribute__((constructor)) static void pivco_codec_select(void)
{
    int cpu_tier = pivco_cpu_tier();
    PIVCO_CHECK(cpu_tier >= 0 && cpu_tier < PIVCO_TIER_COUNT);
    const char *want = getenv("PIVCO_CODEC");
    if (!want)
        want = pivco_default_name(cpu_tier);
    int c = pivco_find_codec(want);
    if (c < 0) {
        fprintf(stderr, "PIVCO_CODEC=%s: unknown or not built\n", want);
        abort();
    }
    if (pivco_codecs[c].tier > cpu_tier) {
        PIVCO_CHECK(pivco_codecs[c].tier < PIVCO_TIER_COUNT);
        fprintf(stderr, "PIVCO_CODEC=%s needs %s; this CPU has %s\n", want,
                pivco_tier_names[pivco_codecs[c].tier], pivco_tier_names[cpu_tier]);
        abort();
    }
    pivco_codec_idx = c;
    const char *tab = getenv("PIVCO_PRIMS");
    if (!tab)
        tab = pivco_default_name(cpu_tier);
    int t = pivco_find_prims(tab);
    if (t < 0) {
        fprintf(stderr, "PIVCO_PRIMS=%s: unknown or not built\n", tab);
        abort();
    }
    if (pivco_prims_tables[t].tier > cpu_tier) {
        PIVCO_CHECK(pivco_prims_tables[t].tier < PIVCO_TIER_COUNT);
        fprintf(stderr, "PIVCO_PRIMS=%s needs %s; this CPU has %s\n", tab,
                pivco_tier_names[pivco_prims_tables[t].tier], pivco_tier_names[cpu_tier]);
        abort();
    }
    g_pivco_prims = *pivco_prims_tables[t].prims;
    PIVCO_CHECK(g_pivco_prims != NULL);
}
#endif  /* PIVCO_CODEC_DISPATCH*/

/* Compile-time backend choice for the dispatched entries: the build
 * enables exactly one SIMD tier (or none), so there is nothing to
 * select at runtime. */
int pivco_encode(pivco_encoder_t *enc, const pivco_table_t *table,
                 const uint8_t *symbols, size_t n,
                 uint8_t *out, size_t *out_len)
{
    int rc;
#if defined(PIVCO_CODEC_DISPATCH)
    rc = pivco_codecs[pivco_codec_idx].enc(enc, table, symbols, n, out, out_len);
#elif defined(PIVCO_HAS_AVX512)
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
#if defined(PIVCO_CODEC_DISPATCH)
    rc = pivco_codecs[pivco_codec_idx].dec(dec, table, in, in_len, symbols, consumed);
#elif defined(PIVCO_HAS_AVX512)
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
    .shake_steps = 0,
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
#if defined(PIVCO_CODEC_DISPATCH)
        g_pivco_prims->histogram_chunk(in + off, len, h32, scratch);
#else
        prim_histogram_chunk(in + off, len, h32, scratch);
#endif
        for (int sym = 0; sym < 256; sym++) freq[sym] += h32[sym];
        off += len;
    }
    return PIVCO_OK;
}
