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

/* ---------- Encode primitives ---------- */

/* Build per-block codes_la array.  Just an indexed gather; the LUT
 * is `table->code_la`. */
static inline void enc_init_scalar(uint16_t *codes_la, int n,
                                    const uint8_t *symbols,
                                    const uint16_t *code_la_lut)
{
    for (int i = 0; i < n; i++) codes_la[i] = code_la_lut[symbols[i]];
}

/* At a non-flat internal node: write the n-bit partition bitmap and
 * partition codes_la in place.  Each codes_la value is shifted left
 * by 1 in the partitioned outputs, so the next level reads the
 * next-depth partition bit from bit 15 again.
 *
 *   - left (top bit == 0): codes_la[n_left++] = codes_la[j] << 1
 *   - right (top bit == 1): tmp[n_right++] = codes_la[j] << 1
 *
 * Returns n_right; caller derives n_left = n - n_right. */
static inline int encode_partition_scalar(uint16_t *codes_la, int n,
                                            uint8_t *bm, uint16_t *tmp)
{
    int nbytes = bitmap_bytes(n);
    memset(bm, 0, (size_t)nbytes);
    int n_left = 0, n_right = 0;
    for (int j = 0; j < n; j++) {
        uint16_t v = codes_la[j];
        int bit = (v >> 15) & 1;
        if (bit) {
            bm[j >> 3] |= (uint8_t)(1u << (j & 7));
            tmp[n_right++] = (uint16_t)(v << 1);
        } else {
            codes_la[n_left++] = (uint16_t)(v << 1);
        }
    }
    return n_right;
}

/* Flat-subtree path: pack the top D bits of each codes_la[i] LSB-first
 * into out[ceil(n*D/8)] bytes. */
static inline void pack_dN_scalar(uint8_t *out,
                                   const uint16_t *codes_la,
                                   int n, int D)
{
    uint32_t mask = (1u << D) - 1;
    uint64_t buf = 0;
    int bits_in_buf = 0;
    int byte_idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t local = ((uint32_t)codes_la[i] >> (16 - D)) & mask;
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

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ enc_init_scalar(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int prim_encode_partition(uint16_t *codes_la, int n,
                                                     uint8_t *bm, uint16_t *tmp)
{ return encode_partition_scalar(codes_la, n, bm, tmp); }

PIVCO_PRIM_ALWAYS_INLINE void prim_pack_dN(uint8_t *out,
                                             const uint16_t *codes_la,
                                             int n, int D)
{ pack_dN_scalar(out, codes_la, n, D); }

PIVCO_PRIM_ALWAYS_INLINE void prim_flat_decode_to_buffer(uint8_t *out, int n,
                                                           const uint8_t *bm, int D,
                                                           const uint8_t *c2s)
{ flat_decode_to_buffer_scalar(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_both_const(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_both_const_scalar(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge_bcast_left(const uint8_t *bm, int K,
                                                           uint8_t left_sym,
                                                           const uint8_t *right_buf,
                                                           uint8_t *out)
{ tree_merge_bcast_left_scalar(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge_bcast_right(const uint8_t *bm, int K,
                                                            const uint8_t *left_buf,
                                                            uint8_t right_sym,
                                                            uint8_t *out)
{ tree_merge_bcast_right_scalar(bm, K, left_buf, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge(const uint8_t *bm, int K,
                                                const uint8_t *left_buf,
                                                const uint8_t *right_buf,
                                                uint8_t *out)
{ tree_merge_scalar(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_SCALAR_H */
