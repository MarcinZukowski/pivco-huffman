#ifndef PIVCO_HUFFMAN_H
#define PIVCO_HUFFMAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Constants ---------- */

#ifndef PIVCO_BLOCK_SIZE
#if defined(__aarch64__)
#define PIVCO_BLOCK_SIZE 8192   /* 128KB L1D on Apple M-series */
#elif defined(__AVX512F__)
#define PIVCO_BLOCK_SIZE 8192   /* 48KB+ L1D on Intel Granite Rapids etc. */
#else
#define PIVCO_BLOCK_SIZE 4096   /* 32KB L1D on x86 (Zen, etc.) */
#endif
#endif

#define PIVCO_MAX_SYMBOLS   256

/* Maximum Huffman code length (length-limited Huffman, like huf0).
 * Capping at 11 matches zstd's huf0 max and bounds the canonical
 * decode_sym/decode_len tables at 2KB each (fits L1 cleanly).  Trees
 * for distributions with naturally deeper Huffman (e.g. prose_pride at
 * natural max 16) get reshaped: rare deep leaves are pulled up and
 * other leaves get longer codes per Kraft.  Compression cost is small
 * (~0.5-1% on text-like data, 0% on most distributions whose natural
 * max is <=11 anyway).  Decode is slightly faster on text-like
 * distributions, slightly slower on geometric — net wash to small win
 * on real workloads.  Override at build time if you need
 * different trade-off: -DPIVCO_MAX_CODE_LEN=15. */
#ifndef PIVCO_MAX_CODE_LEN
#define PIVCO_MAX_CODE_LEN  11
#endif

/* Maximum encoded size for one block (generous upper bound):
   Sum of code bits across all symbols. Worst case: all 8-bit codes
   => N bytes. Plus rounding overhead per tree node. */
#define PIVCO_MAX_ENCODED_SIZE (PIVCO_BLOCK_SIZE * 2)

/* ---------- Error codes ---------- */

#define PIVCO_OK            0
#define PIVCO_ERR_NULL      (-1)
#define PIVCO_ERR_OVERFLOW  (-2)
#define PIVCO_ERR_CORRUPT   (-3)
#define PIVCO_ERR_EMPTY     (-4)

/* ---------- Huffman tree node (for PIVCO tree-walk) ---------- */

/* Compact tree: nodes stored in array, indexed by node ID.
   Max nodes = 2 * MAX_SYMBOLS - 1 = 511.
   Leaf: symbol >= 0.  Internal: symbol = -1, left/right are children. */
#define PIVCO_MAX_TREE_NODES (2 * PIVCO_MAX_SYMBOLS - 1)

typedef struct {
    int16_t symbol;   /* >= 0 for leaf, -1 for internal */
    int16_t left;     /* child node index (bit=0) */
    int16_t right;    /* child node index (bit=1) */
} pivco_tree_node_t;

/* ---------- Per-node decode dispatch ----------
 *
 * Classifies each tree node at build_table time so the decoder can
 * dispatch via a single switch on table->node_type[node_id] instead of
 * the chain of conditional checks (skip_node? leaf? flat? both-leaves?
 * half-prefilled?) that decode_node_neon used to do per call.
 *
 * Same classification applies to all backends (scalar, NEON, AVX-512,
 * SSE).  Priority order matches decode_node_neon's existing logic:
 * HALF_RIGHT/LEFT > BOTH_LEAVES > FULL_PARTITION.
 */
typedef enum {
    PIVCO_NODE_INTERNAL_FULL = 0,  /* general partition path (default) */
    PIVCO_NODE_INTERNAL_FLAT,      /* flat_depth[i] >= 2 — flat-subtree fast path */
    PIVCO_NODE_BOTH_LEAVES,        /* both children are leaves, NEITHER prefilled */
    PIVCO_NODE_HALF_RIGHT,         /* left child IS the prefilled leaf — half-partition right + recurse right */
    PIVCO_NODE_HALF_LEFT,          /* right child IS the prefilled leaf — half-partition left + recurse left */
    PIVCO_NODE_LEAF,               /* leaf, not the prefilled symbol — scatter_sym */
    PIVCO_NODE_SKIP,               /* prefilled leaf — early return (memset already wrote it) */
} pivco_node_type_t;

/* ---------- Huffman table ---------- */

/* Arch-specific enc_init gather tables (see the production header).  ph-td is a
 * retired backend that never calls prim_enc_init, but it compiles the shared
 * primitives headers, so the type must exist to parse the declaration. */
typedef struct {
    const uint16_t *s2r_hi;
} pivco_huffman_enc_init_aux_t;

typedef struct {
    /* Per-symbol encode info */
    uint16_t code[PIVCO_MAX_SYMBOLS];       /* canonical Huffman code */
    uint8_t  code_len[PIVCO_MAX_SYMBOLS];   /* code length (0 = unused) */
    /* Left-aligned code: code << (16 - code_len).  Used by the dense
     * tree-walk encoder so that the bit at tree-depth d is at fixed
     * position 15-d across all symbols, eliminating the per-element
     * shift-amount variance.  Populated by pivco_huffman_build_table. */
    uint16_t code_la[PIVCO_MAX_SYMBOLS];

    /* Tree for PIVCO tree-walk encode/decode */
    pivco_tree_node_t tree[PIVCO_MAX_TREE_NODES];
    int16_t tree_root;
    int16_t tree_node_count;

    /* Canonical decode info (for traditional decoder) */
    uint16_t first_code[PIVCO_MAX_CODE_LEN + 1];
    uint16_t first_sym_idx[PIVCO_MAX_CODE_LEN + 1];
    uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1];
    uint8_t  sorted_symbols[PIVCO_MAX_SYMBOLS];

    /* Flat decode table: 2^MAX_CODE_LEN entries (for traditional decoder) */
    uint8_t  decode_sym[1 << PIVCO_MAX_CODE_LEN];
    uint8_t  decode_len[1 << PIVCO_MAX_CODE_LEN];

    uint8_t  max_len;
    uint8_t  min_len;
    uint16_t num_symbols;

    /* Most frequent symbol (shortest code). PIVCO decode prefills the
       output with this symbol via memset and skips its leaf scatter. */
    uint8_t  prefill_sym;
    int16_t  prefill_node;      /* tree node ID of the prefill leaf */

    /* Flat-subtree fast path: per-node, if flat_depth[i] >= 2 then node i
       is the root of a MAXIMAL flat subtree of depth D = flat_depth[i]
       (all 2^D leaves at the same relative depth).  Encoder emits N*D
       packed bits at this node instead of D levels of bitmaps; decoder
       reads N*D bits and uses flat_code_to_sym[flat_offset[i] + code]
       per element.  Pool sum of 2^D across flat subtrees <= num_symbols. */
    uint8_t  flat_depth[PIVCO_MAX_TREE_NODES];
    uint16_t flat_offset[PIVCO_MAX_TREE_NODES];
    uint8_t  flat_code_to_sym[PIVCO_MAX_SYMBOLS];

    /* Max leaf depth in the subtree rooted at this node, relative to
     * the global tree.  At runtime, the encoder checks
     * `max_leaf_depth[node] - depth <= 8` to decide whether to repack
     * codes_la from uint16 to uint8 and run subsequent partitions on
     * byte-wide SIMD. */
    uint8_t  max_leaf_depth[PIVCO_MAX_TREE_NODES];

    /* Decode dispatch type per node — see pivco_node_type_t.  Set by
     * build_table after tree, prefill_node, and flat_depth are all
     * finalized.  Decoders switch on this instead of running a chain
     * of conditional checks. */
    uint8_t  node_type[PIVCO_MAX_TREE_NODES];
} pivco_huffman_table_t;

/* Compat aliases: the main tree's shared headers (pivco_huffman_common.h,
 * pivco_huffman_primitives_*.h) use the short pivco_ names since the
 * 2026-07 prefix rename; this frozen snapshot keeps its long names and
 * maps them here for the shared-header TUs. */
typedef pivco_huffman_table_t pivco_table_t;
typedef pivco_huffman_enc_init_aux_t pivco_enc_init_aux_t;

/* ---------- Implementation selection ---------- */

typedef enum {
    PIVCO_IMPL_AUTO = 0,
    PIVCO_IMPL_SCALAR,
    PIVCO_IMPL_NEON
} pivco_impl_t;

void         pivco_huffman_set_impl(pivco_impl_t impl);
pivco_impl_t pivco_huffman_get_impl(void);

/* Runtime toggle for the encoder's FSE-dispatch path (v0.2+ wire
 * format).  Default is enabled.  When set to 0, encode_node_* skip
 * the FSE-compress attempt and always emit raw bitmaps with marker=0.
 * Useful for benchmarking the no-FSE codec path without rebuilding,
 * and for cases where the FSE overhead isn't worth its ratio gain on
 * a specific dataset (e.g. proba80-like distributions where the
 * partition bitmaps are too small for FSE to beat marker overhead).
 * The decoder always supports both marker=0 (raw) and marker!=0
 * (FSE) so files produced with FSE enabled can be decoded with FSE
 * disabled and vice versa. */
void pivco_huffman_set_fse_enabled(int enabled);
int  pivco_huffman_get_fse_enabled(void);

/* ---------- FSE table-usage stats (debug instrumentation) ----------
 *
 * Per-table-id counters incremented inside the encoder every time an
 * FSE-coded bitmap is committed.  Indexed 0..25 (slot 0 = "FSE attempted
 * but did not commit"; slots 1..25 = pivco_fse_freq[] table picked).
 * Not thread-safe; intended for single-threaded analysis runs. */
#define PIVCO_FSE_STATS_SLOTS 26
void pivco_huffman_fse_stats_reset(void);
void pivco_huffman_fse_stats_get(uint64_t commit_count[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t attempt_count[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t bytes_in[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t bytes_out[PIVCO_FSE_STATS_SLOTS]);

/* Per-root-event log: one entry per block's root-node visit.
 * Captures table_id chosen (0 if below threshold / no table), the
 * observed p_major, whether the FSE commit succeeded, and byte counts.
 * Useful for showing how a single tree position (the root) adapts
 * across blocks of the same file. */
typedef struct {
    int    table_id;     /* 0 if below MIN_THRESHOLD / no FSE attempt */
    double p_major;      /* observed max(n_left,n_right)/n */
    int    committed;    /* 1 if FSE emitted, 0 otherwise */
    int    nbytes_in;    /* raw bitmap byte count */
    int    nbytes_out;   /* fse_len if committed; nbytes_in if not */
} pivco_huffman_fse_root_event_t;

int  pivco_huffman_fse_root_count(void);
void pivco_huffman_fse_root_get(int idx, pivco_huffman_fse_root_event_t *out);

/* ---------- Table construction ---------- */

int pivco_huffman_build_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table);

/* Build a Huffman table whose node_type[] classification reflects a
 * naive decoder: every internal -> PIVCO_NODE_INTERNAL_FULL, every
 * leaf -> PIVCO_NODE_LEAF.  No flat-subtree path, no half-partition
 * variant, no fused both-leaves scatter, no prefill.  Pair with
 * pivco_huffman_decode_naive. */
int pivco_huffman_build_table_naive(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                     pivco_huffman_table_t *table);

/* Naive TD encoder paired with the naive decoder below.  Emits only
 * raw bitmaps in DFS-preorder of internal nodes (no FSE marker, no
 * K_right header).  Pair with pivco_huffman_decode_naive. */
int pivco_huffman_encode_naive(const uint8_t *symbols,
                                const pivco_huffman_table_t *table,
                                uint8_t *out, size_t *out_len);

/* Naive scalar TD decoder built from exactly two primitives:
 *   P  -- scalar partition
 *   S1 -- scalar scatter-symbol
 * Reads the slim naive wire format produced by
 * pivco_huffman_encode_naive (raw bitmaps only). */
int pivco_huffman_decode_naive(const uint8_t *in, size_t in_len,
                                const pivco_huffman_table_t *table,
                                uint8_t *symbols, size_t *consumed);

/* Scalar-opt TD decoder: every tree-shape optimisation enabled
 * (constant-prefill, flat-subtree path, both-leaves fused scatter,
 * half-partition variants, K_right header), every primitive in
 * scalar C (no SIMD).  Reads the FULL ph wire format produced by
 * pivco_huffman_encode_scalar_opt or pivco_huffman_encode_neon. */
int pivco_huffman_decode_scalar_opt(const uint8_t *in, size_t in_len,
                                     const pivco_huffman_table_t *table,
                                     uint8_t *symbols, size_t *consumed);

/* Scalar-opt encoder: same wire format as pivco_huffman_encode_neon
 * but pure scalar C (no NEON).  Bit-for-bit equivalent output.  Used
 * on non-NEON hosts so the scalar-opt decoder has data to read. */
int pivco_huffman_encode_scalar_opt(const uint8_t *symbols,
                                      const pivco_huffman_table_t *table,
                                      uint8_t *out, size_t *out_len);

/* Build a Huffman table from already-known code lengths plus an
 * optional explicit within-tier ordering.  This is the path used by
 * decoders that recovered code_lens from a wire format and want to
 * reproduce a specific encoder-supplied within-tier order (so the
 * chunk-assignment optimization in build_table -- top-frequency byte
 * in the largest chunk per tier -- survives a code-len-only
 * serialization).
 *
 * rank_within_tier[s] = 0-based position of symbol s within its
 * code-length tier in the encoder's intended order (lower = earlier).
 * Pass -1 for any tier whose ordering should follow the default
 * (smaller-sym-first tie-break, matching v0.2 behavior).  Pass NULL
 * to use defaults for all tiers (equivalent to build_table with
 * uniform-within-tier synth_freqs).
 *
 * Internally synthesises rank-aware frequencies and runs the same
 * build pipeline as pivco_huffman_build_table -- the caller does
 * not see the synth_freq construction. */
int pivco_huffman_build_table_from_code_lens(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    const int16_t *rank_within_tier,
    pivco_huffman_table_t *table);

/* ---------- PIVCO Huffman encode/decode (block of PIVCO_BLOCK_SIZE symbols) ---------- */

int pivco_huffman_encode(const uint8_t *symbols,
                         const pivco_huffman_table_t *table,
                         uint8_t *out, size_t *out_len);

int pivco_huffman_decode(const uint8_t *in, size_t in_len,
                         const pivco_huffman_table_t *table,
                         uint8_t *symbols, size_t *consumed);

int pivco_huffman_encode_scalar(const uint8_t *symbols,
                                const pivco_huffman_table_t *table,
                                uint8_t *out, size_t *out_len);

int pivco_huffman_decode_scalar(const uint8_t *in, size_t in_len,
                                const pivco_huffman_table_t *table,
                                uint8_t *symbols, size_t *consumed);

#ifdef PIVCO_HAS_NEON
int pivco_huffman_encode_neon(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);

int pivco_huffman_decode_neon(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed);

/* Naive-tree + NEON-SIMD-primitives decoder.  Reads the slim wire
 * format from pivco_huffman_encode_naive (no FSE marker, no K_right). */
int pivco_huffman_decode_naive_simd_neon(
        const uint8_t *in, size_t in_len,
        const pivco_huffman_table_t *table,
        uint8_t *symbols, size_t *consumed);

/* Experimental: bottom-up tree_merge decode (NEON). */
int pivco_huffman_decode_bu_neon(const uint8_t *in, size_t in_len,
                                  const pivco_huffman_table_t *table,
                                  uint8_t *symbols, size_t *consumed);
#endif

#ifdef PIVCO_HAS_SSE4
int pivco_huffman_encode_x86(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);

int pivco_huffman_decode_x86(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed);

/* Experimental: bottom-up tree_merge decode (x86 SSE4.1 / AVX-512 VBMI2). */
int pivco_huffman_decode_bu_x86(const uint8_t *in, size_t in_len,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *symbols, size_t *consumed);
#endif

/* Prior experimental NEON variants (neon2, neon2b, neon_fused_1leaf)
 * are preserved under extras/ as negative results. See extras/README_*
 * files for writeups. */

/* Prefix-radix research backend retired to extras/pivco_huffman_neon_prefix.c
 * (alongside its bench_prefix_profile.c and pivco_huffman_neon_common.h).
 * BU on the standard 2-way wire format beats it on all 29 distributions
 * across all 7 EC2 test hosts; no production caller remained.  See
 * PREFIX_RADIX.md for the historical design record. */

#ifdef PIVCO_HAS_SVE
int pivco_huffman_encode_sve(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);

int pivco_huffman_decode_sve(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed);
#endif

#ifdef PIVCO_HAS_AVX512
int pivco_huffman_encode_avx512(const uint8_t *symbols,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *out, size_t *out_len);

int pivco_huffman_decode_avx512(const uint8_t *in, size_t in_len,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *symbols, size_t *consumed);

/* Naive-tree + AVX-512-SIMD-primitives decoder.  Same slim wire as
 * pivco_huffman_decode_naive_simd_neon. */
int pivco_huffman_decode_naive_simd_avx512(
        const uint8_t *in, size_t in_len,
        const pivco_huffman_table_t *table,
        uint8_t *symbols, size_t *consumed);
#endif

/* ---------- Traditional Huffman encode/decode (for comparison) ---------- */

int trad_huffman_encode(const uint8_t *symbols, size_t n_symbols,
                        const pivco_huffman_table_t *table,
                        uint8_t *out, size_t *out_len, size_t *out_bits);

int trad_huffman_decode(const uint8_t *in, size_t in_bits,
                        const pivco_huffman_table_t *table,
                        uint8_t *symbols, size_t n_symbols);

/* SotA 4-stream encode/decode (huff0-style) */
int trad_huffman_encode_4s(const uint8_t *symbols, size_t n_symbols,
                           const pivco_huffman_table_t *table,
                           uint8_t *out, size_t *out_len);

int trad_huffman_decode_4s(const uint8_t *in, size_t in_len,
                           const pivco_huffman_table_t *table,
                           uint8_t *symbols, size_t n_symbols);

/* ---------- Instrumentation ---------- */
void pivco_instrument_node_size(int n);
void pivco_dump_node_size_hist(void);

#ifdef __cplusplus
}
#endif

#endif /* PIVCO_HUFFMAN_H */
