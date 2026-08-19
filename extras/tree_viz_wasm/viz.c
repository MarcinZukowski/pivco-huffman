/* tree_viz WASM entry points: expose the REAL library length pipeline
 * (plain Huffman build + limit + joint shaping) to figures/tree_viz.html,
 * so the viz shows exactly what pivco_build_table adopts -- no JS port
 * to drift.  Compiles natively too (extras/tree_viz_wasm/native_dump.c
 * uses the same entry points for the fidelity A/B). */
#include "pivco_huffman.h"
#include <string.h>

#ifdef __wasm__
#define TVW_EXPORT(name) __attribute__((export_name(name), visibility("default")))
void tvw_heap_reset(void);
#else
#define TVW_EXPORT(name)
static void tvw_heap_reset(void) {}
#endif

/* I/O buffers.  Frequencies cross the JS boundary as f64 (every bench
 * distribution fits; values above 2^53 aren't representable in the viz
 * data anyway) and are converted here. */
static double   g_freq_f64[PIVCO_MAX_SYMBOLS];
static uint64_t g_freq[PIVCO_MAX_SYMBOLS];
static uint8_t  g_lengths[PIVCO_MAX_SYMBOLS];
static pivco_table_t g_table;

TVW_EXPORT("viz_freq_buf")
double *viz_freq_buf(void) { return g_freq_f64; }

TVW_EXPORT("viz_len_buf")
uint8_t *viz_len_buf(void) { return g_lengths; }

static void load_freq(void)
{
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        double v = g_freq_f64[i];
        g_freq[i] = v <= 0.0 ? 0 : (uint64_t)v;
    }
}

/* Plain Huffman lengths (two-queue build + length-limiting to
 * PIVCO_MAX_CODE_LEN), via the real pivco_build_table.  Returns
 * PIVCO_OK / PIVCO_ERR_*. */
TVW_EXPORT("viz_plain_lengths")
int viz_plain_lengths(void)
{
    tvw_heap_reset();
    load_freq();
    pivco_cfg_t cfg = pivco_cfg_default;
    cfg.effort = PIVCO_EFFORT_PLAIN;
    int rc = pivco_build_table(&cfg, g_freq, &g_table);
    if (rc != PIVCO_OK) return rc;
    memcpy(g_lengths, g_table.code_len, PIVCO_MAX_SYMBOLS);
    return PIVCO_OK;
}

/* Joint shaping on top of the plain lengths, exactly as
 * pivco_build_table runs it.  Returns 1 = shaped set adopted
 * (viz_len_buf updated), 0 = guard kept the plain lengths,
 * negative = build error. */
TVW_EXPORT("viz_joint_lengths")
int viz_joint_lengths(int effort, int fse_enabled)
{
    int rc = viz_plain_lengths();
    if (rc != PIVCO_OK) return rc;
    pivco_cfg_t cfg = pivco_cfg_default;
    cfg.effort      = (pivco_effort_t)effort;
    cfg.fse_enabled = fse_enabled;
    return pivco_joint_optimize_lengths(g_freq, g_lengths, &cfg) == 0 ? 1 : 0;
}
