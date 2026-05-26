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
 *  Boundary convention.  Primitives own only the SIMD-bound work:
 *  building the raw partition bitmap, partitioning codes_la, packing
 *  N·D-bit flat regions, the BU merge kernels.  Everything else --
 *  the K_right header, the FSE marker byte, the optional FSE-attempt
 *  on the raw bitmap, the per-stats bookkeeping -- is arch-agnostic
 *  glue and lives in codec.c.  This split is structural: when a new
 *  backend is added, it inherits all of the wire-format and FSE logic
 *  automatically by going through codec.c; there is no per-backend
 *  feature flag to "remember to wire FSE here too" (the original cause
 *  of the scalar↔SSE wire-format-drift bug this refactor exists to
 *  fix).
 *
 *  Lifecycle:
 *
 *  void prim_codec_init(void);
 *
 *    Idempotent lazy-init for any backend-specific runtime tables
 *    (e.g. NEON's compress_tab + expand_tab pre-bakes).  codec.c
 *    calls this once at every encode/decode entry.  Scalar's
 *    implementation is empty; NEON's calls init_compress_table and
 *    init_expand_table.
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
 *    bit 15 is the root partition bit.  codes_la is built once per block
 *    and DOES NOT mutate across the recursion -- the current-depth
 *    partition bit is at position `15 - depth`, which the primitives
 *    derive from the `depth` argument they receive.
 *
 *  int prim_enc_partition(uint16_t *codes_la, int n, int depth,
 *                                   uint8_t *bm, uint16_t *right_out);
 *
 *    Build the n-bit partition bitmap from codes_la[0..n) and partition
 *    codes_la in place.  This is the only arch-specific encode-side
 *    operation; everything around it (marker byte, optional FSE attempt
 *    that may rewrite the marker+bitmap region) is handled in codec.c.
 *
 *    Writes ceil(n/8) bytes into bm.  Bit j (for j in [0..n)) is bit
 *    (15 - depth) of codes_la[j] at the moment of call; ends up at bit
 *    (j & 7) of byte bm[j >> 3].
 *
 *    Partitions codes_la (values unchanged -- no shift across levels):
 *
 *        - codes_la[0..n_left)  left  (bit was 0)
 *        - right_out[0..n_right)      right (bit was 1)
 *
 *    Returns n_right (caller derives n_left = n - n_right).
 *
 *    codec.c wraps this primitive at every non-flat internal node:
 *
 *        marker_slot = *out_ptr;  *marker_slot = 0;  *out_ptr += 1;
 *        bm = *out_ptr;  *out_ptr += bitmap_bytes(n);
 *        n_right = prim_enc_partition(codes_la, n, depth, bm, right_out);
 *        codec_maybe_fse_attempt(...);  // may rewrite marker + bm,
 *                                       // adjust *out_ptr on commit
 *        wire_commit_kr_header(kr_slot, n_right);
 *
 *  void prim_enc_pack_dN(const uint16_t *codes_la, int n, int D, int depth,
 *                     uint8_t *out_packed);
 *
 *    Flat-subtree path.  Pack the D bits at positions [15-depth ..
 *    15-depth-D+1] of each codes_la[i] LSB-first into out_packed[ceil(n*D/8)]
 *    bytes.  Equivalent to a right-shift by `(16 - depth - D)` and a
 *    `(1 << D) - 1` mask before packing.
 *
 *    The depth-threaded representation (rather than shifting codes_la
 *    per recursion level) matches the NEON encoder's SIMD-tuned
 *    ergonomic: vshlq_u16 with a runtime vector amount is one op,
 *    paid once per pack_dN call rather than n times per partition
 *    pass across the recursion.
 *
 * ---------------------------------------------------------------------------
 *  DECODE PRIMITIVES (bottom-up)
 * ---------------------------------------------------------------------------
 *
 *  void prim_merge_flat(uint8_t *out, int n,
 *                                   const uint8_t *bm, int D,
 *                                   const uint8_t *c2s);
 *
 *    Unpack n D-bit codes from bm[], look each up in c2s[2^D], write
 *    the resulting symbols to out[0..n).  bm is `ceil(n*D/8)` bytes.
 *
 *  void prim_merge_two(const uint8_t *bm, int K,
 *                              uint8_t left_sym, uint8_t right_sym,
 *                              uint8_t *out);
 *
 *    Both-leaves merge: for j in [0..K),
 *        out[j] = (bit_j ? right_sym : left_sym).
 *
 *  void prim_merge_constant_left(const uint8_t *bm, int K,
 *                                   uint8_t left_sym,
 *                                   const uint8_t *right_buf,
 *                                   uint8_t *out);
 *
 *    Half-leaf merge, constant LEFT: out[j] = (bit_j ? right_buf[r++]
 *    : left_sym).  Used by HALF_RIGHT when the left child is a leaf
 *    (or prefilled).
 *
 *  void prim_merge_constant_right(const uint8_t *bm, int K,
 *                                    const uint8_t *left_buf,
 *                                    uint8_t right_sym,
 *                                    uint8_t *out);
 *
 *    Mirror of the above: out[j] = (bit_j ? right_sym : left_buf[l++]).
 *
 *  void prim_merge(const uint8_t *bm, int K,
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
