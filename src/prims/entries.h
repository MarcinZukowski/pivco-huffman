/* prims/entries.h — the prim_* entry points the codec calls, built from
 * this compile's primitive table, prims_tab (prims/tables.h).
 *
 * Each prim_* calls its slot.  The flat-region entries (prim_merge_flat,
 * prim_enc_pack_dN) are built from the three span slots: the 512-lane
 * vertical blocks, the 128-lane vertical blocks, and the natural tail
 * (pivco_huffman_vertical.h).
 *
 * Internal header.  Not part of the public API. */

#ifndef PIVCO_PRIMS_ENTRIES_H
#define PIVCO_PRIMS_ENTRIES_H

#include "../pivco_huffman_vertical.h"

#define PIVCO_PRIM_ALWAYS_INLINE __attribute__((always_inline)) static inline

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ prims_tab.codec_init(); }

PIVCO_PRIM_ALWAYS_INLINE void prim_histogram_chunk(const uint8_t *in, size_t n,
                                                   uint32_t hist[256],
                                                   uint8_t *scratch)
{ prims_tab.histogram_chunk(in, n, hist, scratch); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint8_t *ranks, int n,
                                            const uint8_t *symbols,
                                            const uint8_t *sym_to_rank,
                                            const pivco_enc_init_aux_t *aux)
{ prims_tab.enc_init(ranks, n, symbols, sym_to_rank, aux); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_full(uint8_t *ranks, int n,
                                                     uint8_t thr, uint8_t *bm,
                                                     uint8_t *right_out)
{ return prims_tab.enc_partition_full(ranks, n, thr, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_right(uint8_t *ranks, int n,
                                                      uint8_t thr, uint8_t *bm,
                                                      uint8_t *right_out)
{ return prims_tab.enc_partition_right(ranks, n, thr, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_none(uint8_t *ranks, int n,
                                                     uint8_t thr, uint8_t *bm)
{ return prims_tab.enc_partition_none(ranks, n, thr, bm); }

PIVCO_PRIM_ALWAYS_INLINE int prim_popcount_K_right(const uint8_t *bm, int K)
{ return prims_tab.popcount_K_right(bm, K); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_cst(const uint8_t *bm, int K,
                                                 uint8_t left_sym,
                                                 uint8_t right_sym,
                                                 uint8_t *out)
{ prims_tab.merge_cst_cst(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_vec(const uint8_t *bm, int K,
                                                 uint8_t left_sym,
                                                 const uint8_t *right_buf,
                                                 uint8_t *out)
{ prims_tab.merge_cst_vec(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_vec_vec(const uint8_t *bm, int K,
                                                 const uint8_t *left_buf,
                                                 const uint8_t *right_buf,
                                                 uint8_t *out)
{ prims_tab.merge_vec_vec(bm, K, left_buf, right_buf, out); }

/* ---------- Flat regions: the three span slots ---------- */

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_v512(uint8_t *out, int n,
                                              const uint8_t *bm, int D,
                                              const uint8_t *c2s)
{ prims_tab.merge_v512(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_v128(uint8_t *out, int n,
                                              const uint8_t *bm, int D,
                                              const uint8_t *c2s)
{ prims_tab.merge_v128(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_nat(uint8_t *out, int n,
                                             const uint8_t *bm, int D,
                                             const uint8_t *c2s)
{ prims_tab.merge_nat(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_pack_v512(uint8_t *out, const uint8_t *ranks,
                                             int n, int D, uint8_t base)
{ prims_tab.pack_v512(out, ranks, n, D, base); }

PIVCO_PRIM_ALWAYS_INLINE void prim_pack_v128(uint8_t *out, const uint8_t *ranks,
                                             int n, int D, uint8_t base)
{ prims_tab.pack_v128(out, ranks, n, D, base); }

PIVCO_PRIM_ALWAYS_INLINE void prim_pack_nat(uint8_t *out, const uint8_t *ranks,
                                            int n, int D, uint8_t base)
{ prims_tab.pack_nat(out, ranks, n, D, base); }

/* Natural-layout entries for bench_prim's ST_PACK/ST_MERGE_FLAT rows
 * (the entries below produce the layout `vertical` selects: the hybrid
 * vertical wire, or natural when 0). */
PIVCO_PRIM_ALWAYS_INLINE void prim_enc_pack_dN_natural(const uint8_t *ranks,
                                                       int n, int D, uint8_t base,
                                                       uint8_t *out_packed)
{ prim_pack_nat(out_packed, ranks, n, D, base); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat_natural(uint8_t *out, int n,
                                                      const uint8_t *bm, int D,
                                                      const uint8_t *c2s)
{ prim_merge_nat(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_pack_dN(const uint8_t *ranks,
                                               int n, int D, uint8_t base,
                                               uint8_t *out_packed, int vertical)
{
    int n5 = vertical == PIVCO_FLAT_VERTICAL ? pivco_vert_n512(n, D) : 0;
    if (n5) prim_pack_v512(out_packed, ranks, n5, D, base);
    int r = n - n5, nv = vertical ? pivco_vert_n(r, D) : 0;
    uint8_t *o1 = out_packed + (((size_t)n5 * D) >> 3);
    if (nv) prim_pack_v128(o1, ranks + n5, nv, D, base);
    if (r > nv) prim_pack_nat(o1 + (((size_t)nv * D) >> 3),
                              ranks + n5 + nv, r - nv, D, base);
}

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat(uint8_t *out, int n,
                                              const uint8_t *bm, int D,
                                              const uint8_t *c2s, int vertical)
{
    int n5 = vertical == PIVCO_FLAT_VERTICAL ? pivco_vert_n512(n, D) : 0;
    if (n5) prim_merge_v512(out, n5, bm, D, c2s);
    int r = n - n5, nv = vertical ? pivco_vert_n(r, D) : 0;
    const uint8_t *bm1 = bm + (((size_t)n5 * D) >> 3);
    if (nv) prim_merge_v128(out + n5, nv, bm1, D, c2s);
    if (r > nv) prim_merge_nat(out + n5 + nv, r - nv,
                               bm1 + (((size_t)nv * D) >> 3), D, c2s);
}

#endif  /* PIVCO_PRIMS_ENTRIES_H */
