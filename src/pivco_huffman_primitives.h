/* pivco_huffman_primitives.h — backend-primitive interface (router header).
 *
 * pivco_huffman_codec.c is compiled once per backend tier; this header
 * pulls in the right primitive implementation header based on the
 * PIVCO_BACKEND_* macro that CMake passes to that translation unit.
 *
 * Every backend primitive header MUST provide static-inline
 * implementations under specialized names (e.g. `enc_init_scalar`,
 * `enc_init_neon`, etc.) and declare the aliases the codec uses
 * (`prim_enc_init`, ...).  The aliases forward to the specialized
 * name via always-inline static-inline wrappers, so that:
 *
 *   - codec.c reads cleanly (calls `prim_X` consistently)
 *   - stepping into the alias drops you on the specialized name
 *   - grep for `enc_init_scalar` finds exactly the scalar impl
 *
 * ---------- Primitive contract (every backend must implement these) ----
 *
 * void prim_enc_init(uint16_t codes_la[n], int n,
 *                     const uint8_t *symbols,
 *                     const uint16_t code_la_lut[256]);
 *
 *   Build the per-block left-aligned-codes array:
 *     codes_la[i] = code_la_lut[symbols[i]]
 *   `code_la_lut` is table->code_la — the per-symbol code shifted up
 *   so the topmost bit of the uint16 is the root partition bit.
 *
 * int prim_encode_partition(uint16_t *codes_la, int n, uint8_t *bm,
 *                            uint16_t *tmp);
 *
 *   At a non-flat internal node, given n codes_la values whose top bit
 *   is the current partition decision:
 *     - Write the n-bit partition bitmap into bm[ceil(n/8)].
 *     - Partition codes_la in-place: left (top-bit==0) stays in
 *       codes_la[0..n_left); right (top-bit==1) goes to tmp[0..n_right).
 *     - All written values are pre-shifted left by 1, so the next call
 *       (one level deeper) reads the next-depth partition bit from the
 *       top of each lane.
 *   Returns n_right (caller derives n_left = n - n_right).
 *
 * void prim_pack_dN(uint8_t *out, const uint16_t *codes_la, int n, int D);
 *
 *   Flat-subtree path.  Pack the top D bits of each codes_la[i] LSB-first
 *   into out[ceil(n*D/8)] bytes.  The top D bits of codes_la at the flat
 *   node are exactly the local D-bit code that the decoder reads.
 *
 * void prim_flat_decode_to_buffer(uint8_t *out, int n,
 *                                  const uint8_t *bm, int D,
 *                                  const uint8_t *c2s);
 *
 *   Flat-subtree decode.  Unpack n D-bit codes from bm[], look each up
 *   in c2s[2^D], write the resulting symbols to out[0..n).
 *
 * void prim_merge_both_const(const uint8_t *bm, int K,
 *                             uint8_t left_sym, uint8_t right_sym,
 *                             uint8_t *out);
 *
 *   Both-leaves merge: for j in [0..K), out[j] = (bit_j ? right_sym
 *   : left_sym).
 *
 * void prim_tree_merge_bcast_left(const uint8_t *bm, int K,
 *                                  uint8_t left_sym,
 *                                  const uint8_t *right_buf,
 *                                  uint8_t *out);
 *
 *   Half-leaf merge with constant on the LEFT: for j in [0..K), out[j]
 *   = (bit_j ? right_buf[right_idx++] : left_sym).  Used by HALF_RIGHT
 *   when the left child is a leaf (or prefilled).
 *
 * void prim_tree_merge_bcast_right(const uint8_t *bm, int K,
 *                                   const uint8_t *left_buf,
 *                                   uint8_t right_sym,
 *                                   uint8_t *out);
 *
 *   Mirror of the above: out[j] = (bit_j ? right_sym
 *   : left_buf[left_idx++]).
 *
 * void prim_tree_merge(const uint8_t *bm, int K,
 *                       const uint8_t *left_buf, const uint8_t *right_buf,
 *                       uint8_t *out);
 *
 *   Full BU merge: for j in [0..K), out[j] = (bit_j ?
 *   right_buf[right_idx++] : left_buf[left_idx++]).
 *
 * Internal header.  Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_PRIMITIVES_H
#define PIVCO_HUFFMAN_PRIMITIVES_H

#if defined(PIVCO_BACKEND_SCALAR)
#  include "pivco_huffman_primitives_scalar.h"
#elif defined(PIVCO_BACKEND_NEON)
#  include "pivco_huffman_primitives_neon.h"
#elif defined(PIVCO_BACKEND_X86)
#  include "pivco_huffman_primitives_x86.h"
#elif defined(PIVCO_BACKEND_AVX512)
#  include "pivco_huffman_primitives_avx512.h"
#else
#  error "pivco_huffman_primitives.h requires PIVCO_BACKEND_{SCALAR,NEON,X86,AVX512} to be defined."
#endif

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_H */
