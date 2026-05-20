/* pivco_huffman_primitives_scalar.h — scalar implementations of the
 * codec-primitive interface (see pivco_huffman_primitives.h).
 *
 * Specialized names end in `_scalar`; the codec calls the aliases
 * `prim_*` defined at the bottom as always-inline wrappers.
 *
 * Internal header.  Included by pivco_huffman_primitives.h when
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

/* ---------- Encode primitives ---------- */

/* Build per-block codes_la array.  Just an indexed gather; the LUT
 * is `table->code_la`. */
static inline void enc_init_scalar(uint16_t *codes_la, int n,
                                    const uint8_t *symbols,
                                    const uint16_t *code_la_lut)
{
    for (int i = 0; i < n; i++) codes_la[i] = code_la_lut[symbols[i]];
}

/* Build the n-bit partition bitmap from codes_la[0..n) and partition
 * codes_la in place.  See pivco_huffman_primitives.h for the contract.
 *
 * codes_la is not mutated across recursion levels; the current-depth
 * partition bit lives at position `15 - depth` of each codes_la[j]. */
static inline int build_bitmap_partition_scalar(uint16_t *codes_la, int n,
                                                  int depth,
                                                  uint8_t *bm,
                                                  uint16_t *tmp)
{
    int nbytes = bitmap_bytes(n);
    memset(bm, 0, (size_t)nbytes);
    int bit_shift = 15 - depth;

    /* Partition codes_la in place; values are left UNSHIFTED so children
     * read their own depth's partition bit from the same codes_la lane. */
    int n_left = 0, n_right = 0;
    for (int j = 0; j < n; j++) {
        uint16_t v = codes_la[j];
        int bit = (v >> bit_shift) & 1;
        if (bit) {
            bm[j >> 3] |= (uint8_t)(1u << (j & 7));
            tmp[n_right++] = v;
        } else {
            codes_la[n_left++] = v;
        }
    }
    return n_right;
}

/* Flat-subtree path: pack the D bits at positions [15-depth ..
 * 15-depth-D+1] of each codes_la[i] LSB-first into out. */
static inline void pack_dN_scalar(uint8_t *out,
                                   const uint16_t *codes_la,
                                   int n, int D, int depth)
{
    uint32_t mask = (1u << D) - 1;
    int right_shift = 16 - depth - D;
    uint64_t buf = 0;
    int bits_in_buf = 0;
    int byte_idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t local = ((uint32_t)codes_la[i] >> right_shift) & mask;
        buf |= (uint64_t)local << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xFFu);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) {
        out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
    }
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
static inline void flat_decode_to_buffer_scalar(uint8_t *out, int n,
                                                  const uint8_t *bm, int D,
                                                  const uint8_t *c2s)
{
    for (int i = 0; i < n; i++) {
        uint32_t code = extract_D_bits_scalar(bm, i * D, D);
        out[i] = c2s[code];
    }
}

/* Both-leaves merge: per bit, pick left_sym or right_sym. */
static inline void merge_both_const_scalar(const uint8_t *bm, int K,
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
static inline void tree_merge_bcast_left_scalar(const uint8_t *bm, int K,
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

/* Half-leaf merge, constant right: out[j] = (bit_j ? right_sym : left_buf[l++]). */
static inline void tree_merge_bcast_right_scalar(const uint8_t *bm, int K,
                                                   const uint8_t *left_buf,
                                                   uint8_t right_sym,
                                                   uint8_t *out)
{
    int l = 0;
    for (int j = 0; j < K; j++) {
        int bit = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = bit ? right_sym : left_buf[l++];
    }
}

/* Full BU merge: out[j] = (bit_j ? right_buf[r++] : left_buf[l++]). */
static inline void tree_merge_scalar(const uint8_t *bm, int K,
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

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ codec_init_scalar(); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ enc_init_scalar(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int prim_partition(uint16_t *codes_la, int n,
                                                           int depth,
                                                           uint8_t *bm,
                                                           uint16_t *tmp)
{ return build_bitmap_partition_scalar(codes_la, n, depth, bm, tmp); }

PIVCO_PRIM_ALWAYS_INLINE void prim_pack_dN(uint8_t *out,
                                             const uint16_t *codes_la,
                                             int n, int D, int depth)
{ pack_dN_scalar(out, codes_la, n, D, depth); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat(uint8_t *out, int n,
                                                           const uint8_t *bm, int D,
                                                           const uint8_t *c2s)
{ flat_decode_to_buffer_scalar(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_two(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_both_const_scalar(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_constant_left(const uint8_t *bm, int K,
                                                           uint8_t left_sym,
                                                           const uint8_t *right_buf,
                                                           uint8_t *out)
{ tree_merge_bcast_left_scalar(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_constant_right(const uint8_t *bm, int K,
                                                            const uint8_t *left_buf,
                                                            uint8_t right_sym,
                                                            uint8_t *out)
{ tree_merge_bcast_right_scalar(bm, K, left_buf, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge(const uint8_t *bm, int K,
                                                const uint8_t *left_buf,
                                                const uint8_t *right_buf,
                                                uint8_t *out)
{ tree_merge_scalar(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_SCALAR_H */
