/* pivco_primitives_scalar.h — scalar implementations of the
 * codec-primitive interface (see pivco_primitives.h).
 *
 * Specialized names end in `_scalar`; the codec calls the aliases
 * `prim_*` defined at the bottom as always-inline wrappers.
 *
 * Internal header.  Included by pivco_primitives.h when
 * PIVCO_BACKEND_SCALAR is defined.  Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_PRIMITIVES_SCALAR_H
#define PIVCO_HUFFMAN_PRIMITIVES_SCALAR_H

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"

#include <stdint.h>
#include <string.h>

/* Backend lifecycle.  Scalar has no runtime tables to lazy-init. */
static inline void codec_init_scalar(void) { /* no-op */ }

/* ---------- Encode primitives: rank-based encoding (8-bit in-order ranks) ---------- *
 * Partition compares a per-node threshold (split_rank) to the leaf rank, so
 * the values are 8-bit and partition routing is byte-identical to the code_la
 * bit-test.  Flat pack subtracts flat_base_rank to get the local D-bit code. */
static inline void enc_init_scalar(uint8_t *ranks, int n,
                                        const uint8_t *symbols, const uint8_t *sym_to_rank)
{ for (int i = 0; i < n; i++) ranks[i] = sym_to_rank[symbols[i]]; }

static inline int part_core_scalar(uint8_t *ranks, int n, uint8_t thr,
                                        uint8_t *bm, uint8_t *right_out,
                                        int EMIT_RIGHT, int EMIT_LEFT)
{
    memset(bm, 0, (size_t)bitmap_bytes(n));
    int n_left = 0, n_right = 0;
    for (int j = 0; j < n; j++) {
        uint8_t v = ranks[j];
        if (v > thr) {
            bm[j >> 3] |= (uint8_t)(1u << (j & 7));
            if (EMIT_RIGHT) right_out[n_right] = v;
            n_right++;
        } else {
            if (EMIT_LEFT) ranks[n_left] = v;
            n_left++;
        }
    }
    return n_right;
}

static inline void pack_dN_scalar(uint8_t *out, const uint8_t *ranks,
                                       int n, int D, uint8_t base)
{
    uint64_t buf = 0;
    int bits_in_buf = 0, byte_idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t local = (uint32_t)(uint8_t)(ranks[i] - base);  /* code in [0,2^D); no mask */
        buf |= (uint64_t)local << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xFFu);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
}

/* ---------- Decode primitives ---------- */

/* Extract D bits at bit position `bit_pos` from a packed-bit region. */
static inline uint32_t extract_D_bits_scalar(const uint8_t *in,
                                              int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3;
    int bit_off  = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8)  val |= ((uint32_t)in[byte_idx + 1]) << 8;
    if (bit_off + D > 16) val |= ((uint32_t)in[byte_idx + 2]) << 16;
    return (val >> bit_off) & ((1u << D) - 1);
}

/* Unpack n D-bit codes, look up in c2s, write to out[0..n). */
static inline void merge_flat_scalar(uint8_t *out, int n,
                                                  const uint8_t *bm, int D,
                                                  const uint8_t *c2s)
{
    if (D == 8) {           /* full-alphabet flat: c2s is the identity, codes
                             * ARE the symbols (see merge_flat_d8_neon) */
        memcpy(out, bm, (size_t)n);
        return;
    }
    for (int i = 0; i < n; i++) {
        uint32_t code = extract_D_bits_scalar(bm, i * D, D);
        out[i] = c2s[code];
    }
}

/* Both-leaves merge: per bit, pick left_sym or right_sym. */
static inline void merge_cst_cst_scalar(const uint8_t *bm, int K,
                                             uint8_t left_sym,
                                             uint8_t right_sym,
                                             uint8_t *out)
{
    for (int j = 0; j < K; j++) {
        int bit = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = bit ? right_sym : left_sym;
    }
}

/* Half-leaf merge, constant left: out[j] = (bit_j ? right_buf[r++] : left_sym). */
static inline void merge_cst_vec_scalar(const uint8_t *bm, int K,
                                                  uint8_t left_sym,
                                                  const uint8_t *right_buf,
                                                  uint8_t *out)
{
    int r = 0;
    for (int j = 0; j < K; j++) {
        int bit = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = bit ? right_buf[r++] : left_sym;
    }
}

/* Full BU merge: out[j] = (bit_j ? right_buf[r++] : left_buf[l++]). */
static inline void merge_vec_vec_scalar(const uint8_t *bm, int K,
                                       const uint8_t *left_buf,
                                       const uint8_t *right_buf,
                                       uint8_t *out)
{
    int l = 0, r = 0;
    for (int j = 0; j < K; j++) {
        int bit = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = bit ? right_buf[r++] : left_buf[l++];
    }
}

/* ---------- Aliases consumed by codec.c ---------- */

#define PIVCO_PRIM_ALWAYS_INLINE __attribute__((always_inline)) static inline

/* Widest load a merge kernel issues at a child-buffer cursor (byte loads);
 * the cursor can rest AT `size` on the exhausted side, so buffers a
 * merge reads need this much trailing slack.  Consumed by the decode
 * placement logic (scratch_carve / place_tail). */
#define PIVCO_PRIM_MERGE_OVERREAD 1

#include "pivco_huffman_hist_scalar.h"

PIVCO_PRIM_ALWAYS_INLINE void prim_histogram_chunk(const uint8_t *in, size_t n,
                                                   uint32_t hist[256],
                                                   uint8_t *scratch)
{ histogram_chunk_scalar(in, n, hist, scratch); }

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ codec_init_scalar(); }

/* rank-based encode aliases (consumed by codec.c) */
PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint8_t *ranks, int n,
                                             const uint8_t *symbols, const uint8_t *sym_to_rank,
                                             const pivco_enc_init_aux_t *aux)
{ (void)aux; enc_init_scalar(ranks, n, symbols, sym_to_rank); }
PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_full(uint8_t *ranks, int n,
                                             uint8_t thr, uint8_t *bm, uint8_t *right_out)
{ return part_core_scalar(ranks, n, thr, bm, right_out, 1, 1); }
PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_right(uint8_t *ranks, int n,
                                             uint8_t thr, uint8_t *bm, uint8_t *right_out)
{ return part_core_scalar(ranks, n, thr, bm, right_out, 1, 0); }
PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_none(uint8_t *ranks, int n,
                                             uint8_t thr, uint8_t *bm)
{ return part_core_scalar(ranks, n, thr, bm, NULL, 0, 0); }
PIVCO_PRIM_ALWAYS_INLINE void prim_enc_pack_dN(const uint8_t *ranks,
                                             int n, int D, uint8_t base, uint8_t *out_packed)
{ pack_dN_scalar(out_packed, ranks, n, D, base); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat(uint8_t *out, int n,
                                                           const uint8_t *bm, int D,
                                                           const uint8_t *c2s)
{ merge_flat_scalar(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_cst(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_cst_cst_scalar(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_vec(const uint8_t *bm, int K,
                                                           uint8_t left_sym,
                                                           const uint8_t *right_buf,
                                                           uint8_t *out)
{ merge_cst_vec_scalar(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_vec_vec(const uint8_t *bm, int K,
                                                const uint8_t *left_buf,
                                                const uint8_t *right_buf,
                                                uint8_t *out)
{ merge_vec_vec_scalar(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_SCALAR_H */
