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
 *  ENCODE PARTITION FAMILY  (prim_enc_partition_{full,left,right,none})
 *
 *    Every non-flat internal node builds the same n-bit partition bitmap
 *    from codes_la[0..n); the four members differ only in how many code
 *    halves they additionally scatter.  Building the bitmap is mandatory
 *    (it goes on the wire); the scatter outputs feed further recursion,
 *    so a side is scattered ONLY when that child is itself a subtree.
 *    The codec picks the member by node_type, mirroring the decode-side
 *    prim_merge_* family 1:1:
 *
 *      node_type        primitive                     scatters   outputs
 *      INTERNAL_FULL    prim_enc_partition_full        both       left->codes_la,
 *                                                                 right->right_out
 *      HALF_RIGHT       prim_enc_partition_right       right only right->right_out
 *      HALF_LEFT        prim_enc_partition_left        left only  left->left_out
 *      BOTH_LEAVES      prim_enc_partition_none        neither    (bitmap only)
 *
 *    SUFFIX CONVENTION: the suffix names the NON-TRIVIAL child subtree —
 *    the side whose codes are emitted for further recursion — identical
 *    to the HALF_* node_type meaning (HALF_RIGHT's right child is the
 *    subtree, left is a leaf, so prim_enc_partition_right emits the right
 *    codes).  `_none` = zero CODE outputs (both children leaves); it still
 *    writes the bitmap, so it is exactly the bitmap-build step.
 *
 *    Common contract (all four):
 *      Writes ceil(n/8) bytes into bm.  Bit j (j in [0..n)) is bit
 *      (15 - depth) of codes_la[j]; lands at bit (j & 7) of bm[j >> 3].
 *      codes_la values are unchanged across levels (depth-threaded, no
 *      shift) — a scattered side carries the full uint16 codes onward.
 *
 *    Signatures:
 *      int  prim_enc_partition_full (uint16_t *codes_la, int n, int depth,
 *                                    uint8_t *bm, uint16_t *right_out);
 *           // left stays in codes_la[0..n_left); right->right_out[0..n_right)
 *      int  prim_enc_partition_right(uint16_t *codes_la, int n, int depth,
 *                                    uint8_t *bm, uint16_t *right_out);
 *           // emits right_out[0..n_right); left side not produced
 *      int  prim_enc_partition_left (uint16_t *codes_la, int n, int depth,
 *                                    uint8_t *bm);
 *           // left compacted IN PLACE into codes_la[0..n_left); right not produced
 *      int  prim_enc_partition_none (uint16_t *codes_la, int n,
 *                                    int depth, uint8_t *bm);
 *           // bitmap only (no scatter)
 *    All four return n_right (caller derives n_left = n - n_right).  _left
 *    keeps its codes in place in codes_la, matching _full, so the codec's
 *    left child recurses on codes_la either way; _right emits to right_out
 *    (codes_la untouched); _none emits no codes.
 *
 *    SHARED SCATTER CORE: the compress-table scatter used here is the
 *    same operation the top-down decoder needs (read bitmap + scatter vs.
 *    build bitmap + scatter).  Keep the scatter core factored so a future
 *    prim_dec_partition_* family (TD decode, once ph-td is de-forked) can
 *    reuse it rather than re-implementing — that reuse is the main reason
 *    to land the half/none split as named members now.
 *
 *    codec.c wraps the chosen member at every non-flat internal node:
 *
 *        marker_slot = *out_ptr;  *marker_slot = 0;  *out_ptr += 1;
 *        bm = *out_ptr;  *out_ptr += bitmap_bytes(n);
 *        n_right = prim_enc_partition_<m>(codes_la, n, depth, bm, ...);
 *        codec_maybe_fse_attempt(...);  // may rewrite marker + bm,
 *                                       // adjust *out_ptr on commit
 *        wire_commit_kr_header(kr_slot, n_right);
 *
 *    IMPLEMENTATION (2026-05-26): all four members live in every backend.
 *    _right/_left/_none share one parameterized core (part_core_<backend>,
 *    BUILD + EMIT_RIGHT/EMIT_LEFT compile-time flags) that also covers the
 *    from-bitmap (BUILD=0) form for the future TD-decode share.  _full stays
 *    HAND-WRITTEN (build_bitmap_partition_<backend>) because the generic
 *    core's 1,1,1 specialization scheduled ~8% slower on the hot common path
 *    (measured on M4).  bench_prim numbers that motivated the split (M4/NEON):
 *    _none (bitmap only) ~-54% vs _full, fused build+half (_right/_left) ~-26%
 *    vs _full — the *unfused* "build then partition-half" route is a wash
 *    (the re-read eats the one-sided-scatter saving), so _right/_left are
 *    fused build+scatter.  End-to-end encode gain lands on skewed dists
 *    (AVX-512 calgary/proba80 +16-18%, dna +8%; smaller on NEON/SSE); balanced
 *    inputs (english) are flat.
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
 *  void prim_merge_cst_cst(const uint8_t *bm, int K,
 *                              uint8_t left_sym, uint8_t right_sym,
 *                              uint8_t *out);
 *
 *    Both-leaves merge: for j in [0..K),
 *        out[j] = (bit_j ? right_sym : left_sym).
 *
 *  void prim_merge_cst_vec(const uint8_t *bm, int K,
 *                                   uint8_t left_sym,
 *                                   const uint8_t *right_buf,
 *                                   uint8_t *out);
 *
 *    Half-leaf merge, constant LEFT: out[j] = (bit_j ? right_buf[r++]
 *    : left_sym).  Used by HALF_RIGHT when the left child is a leaf
 *    (or prefilled).
 *
 *  void prim_merge_vec_cst(const uint8_t *bm, int K,
 *                                    const uint8_t *left_buf,
 *                                    uint8_t right_sym,
 *                                    uint8_t *out);
 *
 *    Mirror of the above: out[j] = (bit_j ? right_sym : left_buf[l++]).
 *
 *  void prim_merge_vec_vec(const uint8_t *bm, int K,
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
