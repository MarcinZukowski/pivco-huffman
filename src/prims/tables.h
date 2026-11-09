/* prims/tables.h — the primitive tables: one pivco_prims_t per kernel
 * choice, binding each slot to a kernel.
 *
 * A table's kernels need its backend's header and compile flags, so each
 * compile defines the one table its PIVCO_BACKEND_* and variant defines
 * select, as the static const prims_tab.  pivco_prims_table.c exports it
 * for the codec.  A bench that includes one backend calls through it
 * (prims/entries.h), and the compiler resolves and inlines every call.
 * The codec itself (PIVCO_PRIMS_INDIRECT) defines a table of
 * trampolines: each slot calls the same slot of g_pivco_prims, the
 * table picked at runtime.
 *
 * A backend can define more tables with the same flags, for one
 * microarchitecture each (prims_tab_<name>).  Only the codec runs them,
 * so they are defined only where PIVCO_PRIMS_ALL_TABLES is set
 * (pivco_prims_table.c).
 *
 * Kernels whose signature differs from their slot's get a slot_<slot>_
 * <backend> adapter next to their table.
 *
 * Each table also sets PIVCO_PRIM_MERGE_OVERREAD: the widest load a merge
 * kernel issues at a child-buffer cursor.  The cursor can rest AT `size`
 * on the exhausted side, so buffers a merge reads need this much
 * trailing slack.  Consumed by the decode placement logic (scratch_carve
 * / place_tail).
 *
 * Internal header.  Not part of the public API. */

#ifndef PIVCO_PRIMS_TABLES_H
#define PIVCO_PRIMS_TABLES_H

#include "../pivco_prims.h"

#if defined(PIVCO_PRIMS_INDIRECT)

/* Any table's merges can run here: the widest overread of them all. */
#define PIVCO_PRIM_MERGE_OVERREAD PIVCO_PRIMS_MERGE_OVERREAD

/* Trampolines: slot_<slot>_prims calls <slot> of g_pivco_prims, the
 * table pivco_prims_select() picked for this CPU (or PIVCO_PRIMS).
 *
 * Why: they exist only so a specialized codec can still be built,
 * without the dispatch machinery.  The codec always calls prims_tab.
 * A specialized build binds it to one backend's kernels, which the
 * compiler inlines; the prims codec binds it to these trampolines.
 * Without specialized builds, the codec could call g_pivco_prims
 * directly.
 *
 * The compiler folds prims_tab and inlines the trampolines, so each
 * prim_* call is one indirect call through g_pivco_prims, with no
 * trampoline frame. */
static inline __attribute__((always_inline))
void slot_codec_init_prims(void)
{ g_pivco_prims->codec_init(); }

static inline __attribute__((always_inline))
void slot_histogram_chunk_prims(const uint8_t *in, size_t n,
                                uint32_t hist[256], uint8_t *scratch)
{ g_pivco_prims->histogram_chunk(in, n, hist, scratch); }

static inline __attribute__((always_inline))
void slot_enc_init_prims(uint8_t *ranks, int n, const uint8_t *symbols,
                         const uint8_t *sym_to_rank,
                         const pivco_enc_init_aux_t *aux)
{ g_pivco_prims->enc_init(ranks, n, symbols, sym_to_rank, aux); }

static inline __attribute__((always_inline))
int slot_enc_partition_full_prims(uint8_t *ranks, int n, uint8_t thr,
                                  uint8_t *bm, uint8_t *right_out)
{ return g_pivco_prims->enc_partition_full(ranks, n, thr, bm, right_out); }

static inline __attribute__((always_inline))
int slot_enc_partition_right_prims(uint8_t *ranks, int n, uint8_t thr,
                                   uint8_t *bm, uint8_t *right_out)
{ return g_pivco_prims->enc_partition_right(ranks, n, thr, bm, right_out); }

static inline __attribute__((always_inline))
int slot_enc_partition_none_prims(uint8_t *ranks, int n, uint8_t thr,
                                  uint8_t *bm)
{ return g_pivco_prims->enc_partition_none(ranks, n, thr, bm); }

static inline __attribute__((always_inline))
int slot_popcount_K_right_prims(const uint8_t *bm, int K)
{ return g_pivco_prims->popcount_K_right(bm, K); }

static inline __attribute__((always_inline))
void slot_merge_cst_cst_prims(const uint8_t *bm, int K, uint8_t left_sym,
                              uint8_t right_sym, uint8_t *out)
{ g_pivco_prims->merge_cst_cst(bm, K, left_sym, right_sym, out); }

static inline __attribute__((always_inline))
void slot_merge_cst_vec_prims(const uint8_t *bm, int K, uint8_t left_sym,
                              const uint8_t *right_buf, uint8_t *out)
{ g_pivco_prims->merge_cst_vec(bm, K, left_sym, right_buf, out); }

static inline __attribute__((always_inline))
void slot_merge_vec_vec_prims(const uint8_t *bm, int K,
                              const uint8_t *left_buf,
                              const uint8_t *right_buf, uint8_t *out)
{ g_pivco_prims->merge_vec_vec(bm, K, left_buf, right_buf, out); }

static inline __attribute__((always_inline))
void slot_merge_v512_prims(uint8_t *out, int n, const uint8_t *bm, int D,
                           const uint8_t *c2s)
{ g_pivco_prims->merge_v512(out, n, bm, D, c2s); }

static inline __attribute__((always_inline))
void slot_merge_v128_prims(uint8_t *out, int n, const uint8_t *bm, int D,
                           const uint8_t *c2s)
{ g_pivco_prims->merge_v128(out, n, bm, D, c2s); }

static inline __attribute__((always_inline))
void slot_merge_nat_prims(uint8_t *out, int n, const uint8_t *bm, int D,
                          const uint8_t *c2s)
{ g_pivco_prims->merge_nat(out, n, bm, D, c2s); }

static inline __attribute__((always_inline))
void slot_pack_v512_prims(uint8_t *out, const uint8_t *ranks, int n, int D,
                          uint8_t base)
{ g_pivco_prims->pack_v512(out, ranks, n, D, base); }

static inline __attribute__((always_inline))
void slot_pack_v128_prims(uint8_t *out, const uint8_t *ranks, int n, int D,
                          uint8_t base)
{ g_pivco_prims->pack_v128(out, ranks, n, D, base); }

static inline __attribute__((always_inline))
void slot_pack_nat_prims(uint8_t *out, const uint8_t *ranks, int n, int D,
                         uint8_t base)
{ g_pivco_prims->pack_nat(out, ranks, n, D, base); }

/* The prims codec's table: every slot a trampoline, so the prims codec
 * runs whichever table g_pivco_prims points to. */
static const pivco_prims_t prims_tab = {
    .name                = "prims",
    .codec_init          = slot_codec_init_prims,
    .histogram_chunk     = slot_histogram_chunk_prims,
    .enc_init            = slot_enc_init_prims,
    .enc_partition_full  = slot_enc_partition_full_prims,
    .enc_partition_right = slot_enc_partition_right_prims,
    .enc_partition_none  = slot_enc_partition_none_prims,
    .popcount_K_right    = slot_popcount_K_right_prims,
    .merge_cst_cst       = slot_merge_cst_cst_prims,
    .merge_cst_vec       = slot_merge_cst_vec_prims,
    .merge_vec_vec       = slot_merge_vec_vec_prims,
    .merge_v512          = slot_merge_v512_prims,
    .merge_v128          = slot_merge_v128_prims,
    .merge_nat           = slot_merge_nat_prims,
    .pack_v512           = slot_pack_v512_prims,
    .pack_v128           = slot_pack_v128_prims,
    .pack_nat            = slot_pack_nat_prims,
};

#elif defined(PIVCO_BACKEND_SCALAR)

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

static const pivco_prims_t prims_tab = {
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

static const pivco_prims_t prims_tab = {
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
static const pivco_prims_t prims_tab = {
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
#if !defined(PIVCO_HAS_AVX2)
    .name                = "x86",
    /* The 512-lane blocks as four interleaved 16-lane quarters. */
    .merge_v512          = vert512_merge_x86v,
    .pack_v512           = vert512_pack_x86v,
    .pack_nat            = pack_dN_x86,
#else
#  if defined(PIVCO_X86_INTEL)
    .name                = "x86_avx2_intel",
#  else
    .name                = "x86_avx2",
#  endif
    .pack_nat            = pack_dN_avx2_x86,
    .merge_v512          = vert512_merge_ymm,
#  if defined(PIVCO_X86_INTEL)
    /* D=7 packs in ymm rows, as D=2..6 (Intel runs ymm ops whole). */
    .pack_v512           = vert512_pack_ymm_intel,
#  else
    /* D=7 packs walk 16-lane quarters (Zen splits ymm ops in two). */
    .pack_v512           = vert512_pack_ymm,
#  endif
#endif
};

#elif defined(PIVCO_BACKEND_AVX512)

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

static const pivco_prims_t prims_tab = {
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

#if defined(PIVCO_PRIMS_ALL_TABLES)
/* Zen 4 and Zen 5: the merges with the vpexpandb encoding pinned to
 * disp8 and the c8a loop shapes (the _asm merges). */
static const pivco_prims_t prims_tab_avx512_zen45 = {
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
#endif

#endif

#endif  /* PIVCO_PRIMS_TABLES_H */
