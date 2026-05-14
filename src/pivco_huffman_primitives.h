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
 * ===========================================================================
 *  Primitive contract — every backend must implement these
 * ===========================================================================
 *
 *  Boundary convention.  The codec.c tree walk owns wire-record framing
 *  that is purely structural (the K_right header, the recursion order).
 *  Backend primitives own everything that depends on partition skew or
 *  data layout: the FSE marker byte, the bitmap or FSE payload, the
 *  partition itself.  This split lets each backend make its own decision
 *  on whether (and when) to attempt FSE-coding of the bitmap, without
 *  any feature-flag plumbing through codec.c.
 *
 * ---------------------------------------------------------------------------
 *  ENCODE PRIMITIVES
 * ---------------------------------------------------------------------------
 *
 *  void prim_enc_init(uint16_t codes_la[n], int n,
 *                      const uint8_t *symbols,
 *                      const uint16_t code_la_lut[256]);
 *
 *    Build the per-block left-aligned-codes array, gathering from
 *    `code_la_lut` (= table->code_la) indexed by each input symbol:
 *
 *        codes_la[i] = code_la_lut[symbols[i]]
 *
 *    `code_la_lut[s]` holds the per-symbol Huffman code shifted up so
 *    bit 15 is the root partition bit.  After this call, codes_la[i]
 *    has its current-depth partition bit at position 15 for the root
 *    visit; subsequent visits shift each surviving value left by 1
 *    so bit 15 stays canonical.
 *
 *  int prim_encode_node(uint16_t *codes_la, int n, int depth,
 *                        uint8_t **out_ptr, uint16_t *tmp);
 *
 *    Emit the wire BODY of one non-flat internal node and partition
 *    codes_la.  The codec has already reserved the (optional) K_right
 *    header preceding this call; the K_right value is committed by
 *    the codec after the primitive returns.
 *
 *    Writes, starting at *out_ptr and advancing *out_ptr past it:
 *
 *        [marker: u8]
 *        either   [bitmap: ceil(n/8) bytes]                  if marker == 0
 *        or       [fse_len: u16 LE][fse_payload: fse_len B]  if marker != 0
 *
 *    Partitions codes_la in place:
 *
 *        - codes_la[0..n_left)  left  (top bit was 0), each shifted << 1
 *        - tmp[0..n_right)      right (top bit was 1), each shifted << 1
 *
 *    Returns n_right (caller derives n_left = n - n_right).
 *
 *    `depth` is supplied for the per-codeword FSE commit gate (FSE wins
 *    only when (depth + fse_frac) <= MIN_RATIO * (depth + 1) — see
 *    FSE-V0.md).  Backends that don't attempt FSE may ignore it.
 *
 *    Backends today:
 *      - scalar:        always emits marker=0 + raw bitmap.  No FSE attempt.
 *      - NEON:          attempts FSE when partition skew >= MIN_THRESHOLD;
 *                       commits via marker | xor_flag if the codeword gate
 *                       accepts (see primitives_neon.h).
 *      - x86, AVX-512:  same as scalar today (FSE-encode port pending).
 *
 *  void prim_pack_dN(uint8_t *out, const uint16_t *codes_la, int n, int D);
 *
 *    Flat-subtree path.  Pack the top D bits of each codes_la[i]
 *    LSB-first into out[ceil(n*D/8)] bytes.  The top D bits of codes_la
 *    at a flat-subtree node are exactly the local D-bit code that the
 *    decoder reads.
 *
 * ---------------------------------------------------------------------------
 *  DECODE PRIMITIVES (bottom-up)
 * ---------------------------------------------------------------------------
 *
 *  void prim_flat_decode_to_buffer(uint8_t *out, int n,
 *                                   const uint8_t *bm, int D,
 *                                   const uint8_t *c2s);
 *
 *    Unpack n D-bit codes from bm[], look each up in c2s[2^D], write
 *    the resulting symbols to out[0..n).  bm is `ceil(n*D/8)` bytes.
 *
 *  void prim_merge_both_const(const uint8_t *bm, int K,
 *                              uint8_t left_sym, uint8_t right_sym,
 *                              uint8_t *out);
 *
 *    Both-leaves merge: for j in [0..K),
 *        out[j] = (bit_j ? right_sym : left_sym).
 *
 *  void prim_tree_merge_bcast_left(const uint8_t *bm, int K,
 *                                   uint8_t left_sym,
 *                                   const uint8_t *right_buf,
 *                                   uint8_t *out);
 *
 *    Half-leaf merge, constant LEFT: out[j] = (bit_j ? right_buf[r++]
 *    : left_sym).  Used by HALF_RIGHT when the left child is a leaf
 *    (or prefilled).
 *
 *  void prim_tree_merge_bcast_right(const uint8_t *bm, int K,
 *                                    const uint8_t *left_buf,
 *                                    uint8_t right_sym,
 *                                    uint8_t *out);
 *
 *    Mirror of the above: out[j] = (bit_j ? right_sym : left_buf[l++]).
 *
 *  void prim_tree_merge(const uint8_t *bm, int K,
 *                        const uint8_t *left_buf,
 *                        const uint8_t *right_buf,
 *                        uint8_t *out);
 *
 *    Full BU merge: out[j] = (bit_j ? right_buf[r++] : left_buf[l++]).
 *
 * ===========================================================================
 *  Internal header.  Not part of the public API.
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
