/* prims/tables.c — the primitive tables: one pivco_prims_t per kernel
 * choice, binding each slot to a kernel.
 *
 * A table's kernels need its backend's header and compile flags, so
 * this file is compiled once per backend and SIMD tier (x86 and
 * x86_avx2 are two compiles), defining the tables its PIVCO_BACKEND_*
 * and tier defines select.  One compile can define more than one table,
 * for one microarchitecture each (pivco_prims_x86_avx2_intel,
 * pivco_prims_avx512_zen45).
 *
 * Kernels whose signature differs from their slot's get a slot_<slot>_
 * <backend> adapter next to their table.
 *
 * Each backend also sets PIVCO_PRIM_MERGE_OVERREAD: the widest load a
 * merge kernel issues at a child-buffer cursor.  The cursor can rest AT
 * `size` on the exhausted side, so buffers a merge reads need this much
 * trailing slack.  The codec's buffers cover the widest of them
 * (PIVCO_PRIMS_MERGE_OVERREAD, checked at the bottom). */

#include "../pivco_prims.h"

#if defined(PIVCO_BACKEND_SCALAR)

#include "scalar.h"

#define PIVCO_PRIM_MERGE_OVERREAD 1   /* byte loads */

static inline __attribute__((always_inline))
void slot_enc_init_scalar(uint8_t *ranks, int n, const uint8_t *symbols,
                          const uint8_t *sym_to_rank,
                          const pivco_enc_init_aux_t *aux)
{
    (void)aux;
    enc_init_scalar(ranks, n, symbols, sym_to_rank);
}

static inline __attribute__((always_inline))
int slot_enc_partition_full_scalar(uint8_t *ranks, int n, uint8_t thr,
                                   uint8_t *bm, uint8_t *right_out)
{ return part_core_scalar(ranks, n, thr, bm, right_out, 1, 1); }

static inline __attribute__((always_inline))
int slot_enc_partition_right_scalar(uint8_t *ranks, int n, uint8_t thr,
                                    uint8_t *bm, uint8_t *right_out)
{ return part_core_scalar(ranks, n, thr, bm, right_out, 1, 0); }

static inline __attribute__((always_inline))
int slot_enc_partition_none_scalar(uint8_t *ranks, int n, uint8_t thr,
                                   uint8_t *bm)
{ return part_core_scalar(ranks, n, thr, bm, NULL, 0, 0); }

static inline __attribute__((always_inline))
int slot_popcount_K_right_scalar(const uint8_t *bm, int K)
{ return popcount_K_right_scalar(bm, bitmap_bytes(K), K); }

const pivco_prims_t pivco_prims_scalar = {
    .name                = "scalar",
    .codec_init          = codec_init_scalar,
    .histogram_chunk     = histogram_chunk_scalar,
    .enc_init            = slot_enc_init_scalar,
    .enc_partition_full  = slot_enc_partition_full_scalar,
    .enc_partition_right = slot_enc_partition_right_scalar,
    .enc_partition_none  = slot_enc_partition_none_scalar,
    .popcount_K_right    = slot_popcount_K_right_scalar,
    .merge_cst_cst       = merge_cst_cst_scalar,
    .merge_cst_vec       = merge_cst_vec_scalar,
    .merge_vec_vec       = merge_vec_vec_scalar,
    .merge_v512          = vert512_merge_scalar,
    .merge_v128          = vert_merge_scalar,
    .merge_nat           = merge_flat_scalar,
    .pack_v512           = vert512_pack_scalar,
    .pack_v128           = vert_pack_scalar,
    .pack_nat            = pack_dN_scalar,
};

#elif defined(PIVCO_BACKEND_NEON)

#include "neon.h"

#define PIVCO_PRIM_MERGE_OVERREAD 16  /* 16B vld1q at child cursors */

static inline __attribute__((always_inline))
void slot_enc_init_neon(uint8_t *ranks, int n, const uint8_t *symbols,
                        const uint8_t *sym_to_rank,
                        const pivco_enc_init_aux_t *aux)
{
    (void)aux;
    init_neon(ranks, n, symbols, sym_to_rank);
}

static inline __attribute__((always_inline))
int slot_enc_partition_right_neon(uint8_t *ranks, int n, uint8_t thr,
                                  uint8_t *bm, uint8_t *right_out)
{ return part_core_neon(ranks, n, thr, bm, right_out, 1); }

static inline __attribute__((always_inline))
int slot_enc_partition_none_neon(uint8_t *ranks, int n, uint8_t thr,
                                 uint8_t *bm)
{ return part_core_neon(ranks, n, thr, bm, NULL, 0); }

static inline __attribute__((always_inline))
int slot_popcount_K_right_neon(const uint8_t *bm, int K)
{ return popcount_K_right_neon(bm, bitmap_bytes(K), K); }

const pivco_prims_t pivco_prims_neon = {
    .name                = "neon",
    .codec_init          = codec_init_neon,
    /* NEON has no histogram win over the shared scalar core (measured). */
    .histogram_chunk     = histogram_chunk_scalar,
    .enc_init            = slot_enc_init_neon,
    .enc_partition_full  = part_full_neon,
    .enc_partition_right = slot_enc_partition_right_neon,
    .enc_partition_none  = slot_enc_partition_none_neon,
    .popcount_K_right    = slot_popcount_K_right_neon,
    .merge_cst_cst       = merge_cst_cst_neon,
    .merge_cst_vec       = merge_cst_vec_neon,
    .merge_vec_vec       = merge_vec_vec_neon,
    .merge_v512          = vert512_merge_neon,
    .merge_v128          = vert_merge_neon,
    .merge_nat           = merge_flat_neon,
    .pack_v512           = vert512_pack_neon,
    .pack_v128           = vert_pack_neon,
    .pack_nat            = pack_dN_neon,
};

#elif defined(PIVCO_BACKEND_X86)

#include "x86.h"

#define PIVCO_PRIM_MERGE_OVERREAD 16  /* 16B loadu at child cursors */

static inline __attribute__((always_inline))
int slot_enc_partition_right_x86(uint8_t *ranks, int n, uint8_t thr,
                                 uint8_t *bm, uint8_t *right_out)
{ return part_core_x86(ranks, n, thr, bm, right_out, 1); }

static inline __attribute__((always_inline))
int slot_enc_partition_none_x86(uint8_t *ranks, int n, uint8_t thr,
                                uint8_t *bm)
{ return part_core_x86(ranks, n, thr, bm, NULL, 0); }

static inline __attribute__((always_inline))
int slot_popcount_K_right_x86(const uint8_t *bm, int K)
{ return popcount_K_right_x86(bm, bitmap_bytes(K), K); }

/* All x86 tables run the shared xmm 16-lane cores on 128-lane blocks:
 * the wide forms (ymm two-block, zmm srlv/multishift) only covered the
 * 128..511 mid-band (<= 4% of elements at 32K blocks, ~0.2% E2E) at
 * ~12 KB of text per TU; they are in git @ 87da7ed (see IDEAS.md). */
#if !defined(PIVCO_HAS_AVX2)
const pivco_prims_t pivco_prims_x86 = {
    .name                = "x86",
    .codec_init          = codec_init_x86,
    /* SSE4.1/AVX2 has no histogram win over the shared scalar core
     * (measured). */
    .histogram_chunk     = histogram_chunk_scalar,
    .enc_init            = enc_init_2tab_x86,
    .enc_partition_full  = part_full_x86,
    .enc_partition_right = slot_enc_partition_right_x86,
    .enc_partition_none  = slot_enc_partition_none_x86,
    .popcount_K_right    = slot_popcount_K_right_x86,
    .merge_cst_cst       = merge_cst_cst_x86,
    .merge_cst_vec       = merge_cst_vec_x86,
    .merge_vec_vec       = merge_vec_vec_x86,
    .merge_v128          = vert_merge_x86v,
    .merge_nat           = merge_flat_x86,
    .pack_v128           = vert_pack_x86v,
    /* The 512-lane blocks as four interleaved 16-lane quarters. */
    .merge_v512          = vert512_merge_x86v,
    .pack_v512           = vert512_pack_x86v,
    .pack_nat            = pack_dN_x86,
};
#else
const pivco_prims_t pivco_prims_x86_avx2 = {
    .name                = "x86_avx2",
    .codec_init          = codec_init_x86,
    /* SSE4.1/AVX2 has no histogram win over the shared scalar core
     * (measured). */
    .histogram_chunk     = histogram_chunk_scalar,
    .enc_init            = enc_init_2tab_x86,
    .enc_partition_full  = part_full_x86,
    .enc_partition_right = slot_enc_partition_right_x86,
    .enc_partition_none  = slot_enc_partition_none_x86,
    .popcount_K_right    = slot_popcount_K_right_x86,
    .merge_cst_cst       = merge_cst_cst_x86,
    .merge_cst_vec       = merge_cst_vec_x86,
    .merge_vec_vec       = merge_vec_vec_x86,
    .merge_v128          = vert_merge_x86v,
    .merge_nat           = merge_flat_x86,
    .pack_v128           = vert_pack_x86v,
    .pack_nat            = pack_dN_avx2_x86,
    .merge_v512          = vert512_merge_ymm,
    /* D=7 packs walk 16-lane quarters (Zen splits ymm ops in two). */
    .pack_v512           = vert512_pack_ymm,
};

const pivco_prims_t pivco_prims_x86_avx2_intel = {
    .name                = "x86_avx2_intel",
    .codec_init          = codec_init_x86,
    /* SSE4.1/AVX2 has no histogram win over the shared scalar core
     * (measured). */
    .histogram_chunk     = histogram_chunk_scalar,
    .enc_init            = enc_init_2tab_x86,
    .enc_partition_full  = part_full_x86,
    .enc_partition_right = slot_enc_partition_right_x86,
    .enc_partition_none  = slot_enc_partition_none_x86,
    .popcount_K_right    = slot_popcount_K_right_x86,
    .merge_cst_cst       = merge_cst_cst_x86,
    .merge_cst_vec       = merge_cst_vec_x86,
    .merge_vec_vec       = merge_vec_vec_x86,
    .merge_v128          = vert_merge_x86v,
    .merge_nat           = merge_flat_x86,
    .pack_v128           = vert_pack_x86v,
    .pack_nat            = pack_dN_avx2_x86,
    .merge_v512          = vert512_merge_ymm,
    /* D=7 packs in ymm rows, as D=2..6 (Intel runs ymm ops whole). */
    .pack_v512           = vert512_pack_ymm_intel,
};
#endif

#elif defined(PIVCO_BACKEND_AVX512)

#include "avx512.h"

#define PIVCO_PRIM_MERGE_OVERREAD 8   /* 8B expandloadu tail window */

static inline __attribute__((always_inline))
void slot_enc_init_avx512(uint8_t *ranks, int n, const uint8_t *symbols,
                          const uint8_t *sym_to_rank,
                          const pivco_enc_init_aux_t *aux)
{
    (void)aux;
    init_avx512(ranks, n, symbols, sym_to_rank);
}

static inline __attribute__((always_inline))
int slot_popcount_K_right_avx512(const uint8_t *bm, int K)
{ return popcount_K_right_avx512(bm, bitmap_bytes(K), K); }

const pivco_prims_t pivco_prims_avx512 = {
    .name                = "avx512",
    .codec_init          = codec_init_avx512,
    .histogram_chunk     = histogram_chunk_avx512,
    .enc_init            = slot_enc_init_avx512,
    .enc_partition_full  = part_full_avx512,
    .enc_partition_right = part_right_avx512,
    .enc_partition_none  = part_none_avx512,
    .popcount_K_right    = slot_popcount_K_right_avx512,
    .merge_cst_cst       = merge_cst_cst_avx512,
    .merge_cst_vec       = merge_cst_vec_avx512,
    .merge_vec_vec       = merge_vec_vec_avx512,
    .merge_v512          = vert512_merge_zmm,
    .merge_v128          = vert_merge_x86v,
    .merge_nat           = merge_flat_avx512,
    .pack_v512           = vert512_pack_zmm,
    .pack_v128           = vert_pack_x86v,
    .pack_nat            = pack_dN_avx512,
};

/* Zen 4 and Zen 5: the merges with the vpexpandb encoding pinned to
 * disp8 and the c8a loop shapes (the _asm merges). */
const pivco_prims_t pivco_prims_avx512_zen45 = {
    .name                = "avx512_zen45",
    .codec_init          = codec_init_avx512,
    .histogram_chunk     = histogram_chunk_avx512,
    .enc_init            = slot_enc_init_avx512,
    .enc_partition_full  = part_full_avx512,
    .enc_partition_right = part_right_avx512,
    .enc_partition_none  = part_none_avx512,
    .popcount_K_right    = slot_popcount_K_right_avx512,
    .merge_cst_cst       = merge_cst_cst_avx512,
    .merge_cst_vec       = merge_cst_vec_avx512_asm,
    .merge_vec_vec       = merge_vec_vec_avx512_asm,
    .merge_v512          = vert512_merge_zmm,
    .merge_v128          = vert_merge_x86v,
    .merge_nat           = merge_flat_avx512,
    .pack_v512           = vert512_pack_zmm,
    .pack_v128           = vert_pack_x86v,
    .pack_nat            = pack_dN_avx512,
};

#else
#  error "prims/tables.c: no PIVCO_BACKEND_* defined"
#endif

_Static_assert(PIVCO_PRIM_MERGE_OVERREAD <= PIVCO_PRIMS_MERGE_OVERREAD,
               "the codec's buffers must cover this table's merges");
