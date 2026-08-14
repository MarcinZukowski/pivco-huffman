/* bench_ctx.h — shared lazily-created contexts + config for benchmarks.
 * Benches are single-threaded; one ctx pair per process is fine.  The
 * mutable bench_cfg() replaces the retired process-global setters:
 * set flags before building tables, pass bench_cfg() to the builds. */
#ifndef PIVCO_BENCH_CTX_H
#define PIVCO_BENCH_CTX_H
#include "pivco_huffman.h"
static inline pivco_encoder_t *bench_enc_ctx(void) {
    static pivco_encoder_t *e; if (!e) e = pivco_encoder_create(); return e;
}
static inline pivco_decoder_t *bench_dec_ctx(void) {
    static pivco_decoder_t *d; if (!d) d = pivco_decoder_create(); return d;
}
static inline pivco_cfg_t *bench_cfg(void) {
    static pivco_cfg_t c; static int init;
    if (!init) { c = pivco_cfg_default; init = 1; }
    return &c;
}
#endif
