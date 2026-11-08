/* pivco_prims.h — the codec's primitives as a struct of function
 * pointers, one const instance per table (prims/tables.h).
 *
 * The prims codec (the codec built with PIVCO_PRIMS_INDIRECT) calls
 * every prim_* through g_pivco_prims instead of inlining a backend's
 * header. The signatures and contracts are the ones documented in
 * pivco_huffman_primitives.h.
 *
 * Internal header.  Not part of the public API. */

#ifndef PIVCO_PRIMS_H
#define PIVCO_PRIMS_H

#include <stddef.h>
#include <stdint.h>
#include "pivco_huffman.h"

typedef struct {
    const char *name;
    void (*codec_init)(void);
    void (*enc_init)(uint8_t *ranks, int n, const uint8_t *symbols,
                     const uint8_t *sym_to_rank,
                     const pivco_enc_init_aux_t *aux);
    int  (*enc_partition_full)(uint8_t *ranks, int n, uint8_t thr,
                               uint8_t *bm, uint8_t *right_out);
    int  (*enc_partition_right)(uint8_t *ranks, int n, uint8_t thr,
                                uint8_t *bm, uint8_t *right_out);
    int  (*enc_partition_none)(uint8_t *ranks, int n, uint8_t thr,
                               uint8_t *bm);
    int  (*popcount_K_right)(const uint8_t *bm, int K);
    void (*merge_cst_cst)(const uint8_t *bm, int K, uint8_t left_sym,
                          uint8_t right_sym, uint8_t *out);
    void (*merge_cst_vec)(const uint8_t *bm, int K, uint8_t left_sym,
                          const uint8_t *right_buf, uint8_t *out);
    void (*merge_vec_vec)(const uint8_t *bm, int K,
                          const uint8_t *left_buf,
                          const uint8_t *right_buf, uint8_t *out);
    /* Flat regions, one slot per span (prims/entries.h): the 512- and
     * 128-lane vertical blocks and the natural tail. */
    void (*merge_v512)(uint8_t *out, int n, const uint8_t *bm, int D,
                       const uint8_t *c2s);
    void (*merge_v128)(uint8_t *out, int n, const uint8_t *bm, int D,
                       const uint8_t *c2s);
    void (*merge_nat)(uint8_t *out, int n, const uint8_t *bm, int D,
                      const uint8_t *c2s);
    void (*pack_v512)(uint8_t *out, const uint8_t *ranks, int n, int D,
                      uint8_t base);
    void (*pack_v128)(uint8_t *out, const uint8_t *ranks, int n, int D,
                      uint8_t base);
    void (*pack_nat)(uint8_t *out, const uint8_t *ranks, int n, int D,
                     uint8_t base);
    void (*histogram_chunk)(const uint8_t *in, size_t n, uint32_t hist[256],
                            uint8_t *scratch);
} pivco_prims_t;

/* The widest merge overread (PIVCO_PRIM_MERGE_OVERREAD) of any backend.
 * The prims codec can run any table's merges, so its buffers need this
 * much slack; each table's backend is checked against it. */
#define PIVCO_PRIMS_MERGE_OVERREAD 16

extern const pivco_prims_t *const pivco_prims_scalar;
extern const pivco_prims_t *const pivco_prims_neon;
extern const pivco_prims_t *const pivco_prims_x86;
extern const pivco_prims_t *const pivco_prims_avx512;

/* The table the prims codec calls through.  Defaults to the build's
 * tier refined by CPU vendor; PIVCO_PRIMS=<name> in the environment
 * overrides it (developers only). */
extern const pivco_prims_t *g_pivco_prims;

/* Names of the codec and the table chosen at startup (dispatch builds). */
const char *pivco_codec_name(void);
const char *pivco_prims_name(void);

#endif /* PIVCO_PRIMS_H */
