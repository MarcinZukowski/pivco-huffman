/* pivco_huffman_primitives.h — backend-primitive interface (router header).
 *
 * pivco_huffman_codec.c is compiled once per backend tier; this header
 * pulls in the right primitive implementation header based on the
 * PIVCO_BACKEND_* macro that CMake passes to that translation unit.
 *
 * Every backend primitive header MUST provide static-inline
 * implementations under specialized names (e.g. `part_core_scalar`,
 * `part_core_neon`, etc.) and declare the aliases the codec uses
 * (`prim_enc_init`, ...).  The aliases forward to the specialized
 * name via always-inline static-inline wrappers, so that:
 *
 *   - codec.c reads cleanly (calls `prim_X` consistently)
 *   - stepping into the alias drops you on the specialized name
 *   - grep for `part_core_scalar` finds exactly the scalar impl
 *
 * ===========================================================================
 *  Primitive contract — every backend must implement these
 * ===========================================================================
 *
 *  Boundary convention.  Primitives own only the SIMD-bound work:
 *  building the raw partition bitmap, partitioning the rank array, packing
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
 *  void prim_histogram_chunk(const uint8_t *in, size_t n,
 *                            uint32_t hist[256], uint8_t *scratch);
 *
 *    Adds the byte counts of in[0..n) into hist[256].  Caller
 *    guarantees n <= PIVCO_PRIM_HIST_CHUNK (so no u32 counter can
 *    overflow) and provides PIVCO_PRIM_HIST_SCRATCH bytes of scratch
 *    (the AVX-512 bin buffers; other backends ignore it).  No
 *    alignment requirements; reads and writes are exact.  Backends
 *    without their own implementation alias the shared scalar core
 *    (pivco_huffman_hist_scalar.h).
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
 *  void prim_enc_init(uint8_t ranks[n], int n,
 *                      const uint8_t *symbols,
 *                      const uint8_t sym_to_rank[256],
 *                      const pivco_enc_init_aux_t *aux);
 *
 *    Build the per-block in-order rank array, gathering from `sym_to_rank`
 *    (= table->sym_to_rank) indexed by each input symbol:
 *
 *        ranks[i] = sym_to_rank[symbols[i]]
 *
 *    `aux` (= &table->enc_init_aux) carries arch-specific precomputed gather
 *    tables — its fields are non-NULL only on arches that use them (x86 SSE/AVX2
 *    uses `s2r_hi` = sym_to_rank<<8 for the 2tab no-OR merge).  A backend that
 *    consumes a field asserts it is non-NULL; backends that don't need it ignore
 *    `aux`.
 *
 *    Each leaf's rank is its left-to-right position among the tree's leaves
 *    (partbyrank).  A subtree's leaves form a contiguous rank range, so
 *    the per-node routing test reduces to an 8-bit compare against the node's
 *    split_rank (below).  ranks is built once per block; the partition mutates
 *    it in place across the recursion (the left half stays, the right half is
 *    compacted into a scratch buffer for the right child to recurse on).
 *
 *  ENCODE PARTITION FAMILY  (prim_enc_partition_{full,left,right,none})
 *
 *    Every non-flat internal node builds the same n-bit partition bitmap from
 *    ranks[0..n): bit j = (ranks[j] > split_rank), where split_rank is the max
 *    rank in the node's left subtree.  Because in-order rank order == the old
 *    left-aligned-code order, this is byte-identical to the former code_la
 *    bit-test — wire format and decoder are unchanged.  The four members
 *    differ only in how many halves they additionally scatter; the codec picks
 *    by node_type, mirroring the decode-side prim_merge_* family 1:1:
 *
 *      node_type        primitive                     scatters   outputs
 *      INTERNAL_FULL    prim_enc_partition_full        both       left in place,
 *                                                                 right->right_out
 *      LEAF_LEFT        prim_enc_partition_right       right only right->right_out
 *      BOTH_LEAVES      prim_enc_partition_none        neither    (bitmap only)
 *
 *    SUFFIX CONVENTION: the suffix names the NON-TRIVIAL child subtree —
 *    the side whose ranks are emitted for further recursion (LEAF_LEFT's
 *    right child is the subtree, left is a leaf, so
 *    prim_enc_partition_right emits the right ranks).  `_none` = zero
 *    outputs (both children leaves); it still writes the bitmap, so it
 *    is exactly the bitmap-build step.
 *
 *    Common contract (all four):
 *      Writes ceil(n/8) bytes into bm.  Bit j (j in [0..n)) is
 *      (ranks[j] > thr); lands at bit (j & 7) of bm[j >> 3].
 *
 *    Signatures (thr = table->split_rank[node]):
 *      int  prim_enc_partition_full (uint8_t *ranks, int n, uint8_t thr,
 *                                    uint8_t *bm, uint8_t *right_out);
 *           // left stays in ranks[0..n_left); right->right_out[0..n_right)
 *      int  prim_enc_partition_right(uint8_t *ranks, int n, uint8_t thr,
 *                                    uint8_t *bm, uint8_t *right_out);
 *           // emits right_out[0..n_right); left side not produced
 *      int  prim_enc_partition_none (uint8_t *ranks, int n, uint8_t thr,
 *                                    uint8_t *bm);
 *           // bitmap only (no scatter)
 *    All three return n_right (caller derives n_left = n - n_right).
 *    _full keeps the left ranks in place in ranks[0..n_left); _right
 *    emits to right_out (ranks untouched); _none emits no ranks.
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
 *        n_right = prim_enc_partition_<m>(ranks, n, split_rank, bm, ...);
 *        codec_maybe_fse_attempt(...);  // may rewrite marker + bm,
 *                                       // adjust *out_ptr on commit
 *        wire_commit_kr_header(kr_slot, n_right);
 *
 *    IMPLEMENTATION: _right/_none share one parameterized core
 *    (part_core_<backend>, EMIT_RIGHT compile-time flag; the scalar core
 *    additionally carries EMIT_LEFT because scalar _full rides it too).
 *    On NEON/x86, _full stays HAND-WRITTEN (part_full_<backend>) because
 *    the generic core's both-sides specialization scheduled ~8% slower on
 *    the hot common path (measured on M4 for the former code_la
 *    partition).  bench_prim numbers that motivated the split (M4/NEON):
 *    _none (bitmap only) ~-54% vs _full, fused build+half (_right) ~-26%
 *    vs _full — the *unfused* "build then partition-half" route is a wash
 *    (the re-read eats the one-sided-scatter saving), so _right is
 *    fused build+scatter.  End-to-end encode gain lands on skewed dists
 *    (AVX-512 calgary/proba80 +16-18%, dna +8%; smaller on NEON/SSE); balanced
 *    inputs (english) are flat.
 *
 *  void prim_enc_pack_dN(const uint8_t *ranks, int n, int D, uint8_t base,
 *                     uint8_t *out_packed);
 *
 *    Flat-subtree path.  In a flat subtree (all 2^D leaves at the same depth),
 *    the in-subtree local code is `ranks[i] - base`, already a D-bit value
 *    (base = table->flat_base_rank[node] = the min rank in the subtree).  Pack
 *    those local codes LSB-first into out_packed[ceil(n*D/8)] bytes.
 *
 *    Because (rank - base) is already an 8-bit value in the low bits, the
 *    packers read straight from the u8 rank array — no u16 load + shift +
 *    narrow round-trip that the former code_la pack needed.
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
 *    : left_sym).  Used by LEAF_LEFT (the left child is a leaf).
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

/* Backend selection.  The per-backend codec TUs pass an explicit
 * PIVCO_BACKEND_* define (CMake) and keep full control.  Any OTHER TU
 * may simply include this header: with no explicit define, the build
 * architecture picks the backend (the same tier CMake's codec dispatch
 * prefers for the build), so callers of prim_* never name a backend. */
#if !defined(PIVCO_BACKEND_SCALAR) && !defined(PIVCO_BACKEND_NEON) \
 && !defined(PIVCO_BACKEND_X86)    && !defined(PIVCO_BACKEND_AVX512)
#  if defined(__aarch64__) || defined(__ARM_NEON)
#    define PIVCO_BACKEND_NEON 1
#  elif defined(__AVX512VBMI2__)
#    define PIVCO_BACKEND_AVX512 1
#  elif defined(__SSE4_1__)
#    define PIVCO_BACKEND_X86 1
#  else
#    define PIVCO_BACKEND_SCALAR 1
#  endif
#endif

#if defined(PIVCO_BACKEND_SCALAR)
#  include "pivco_huffman_primitives_scalar.h"
#  define PIVCO_PRIM_BACKEND_NAME "scalar"
#elif defined(PIVCO_BACKEND_NEON)
#  include "pivco_huffman_primitives_neon.h"
#  define PIVCO_PRIM_BACKEND_NAME "neon"
#elif defined(PIVCO_BACKEND_X86)
#  include "pivco_huffman_primitives_x86.h"
#  define PIVCO_PRIM_BACKEND_NAME "sse/avx2"
#elif defined(PIVCO_BACKEND_AVX512)
#  include "pivco_huffman_primitives_avx512.h"
#  define PIVCO_PRIM_BACKEND_NAME "avx512"
#endif

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_H */
