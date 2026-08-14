#ifndef PIVCO_HUFFMAN_H
#define PIVCO_HUFFMAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Constants ---------- */

/* PIVCO_BLOCK_SIZE is the *default* per-block symbol count chosen by the
 * file codec / CLI / benchmarks.  It is no longer a hard codec limit: the
 * codec sizes its scratch dynamically off the runtime N (carried in the
 * per-block uint16 wire header), so any block size in [1, PIVCO_WIRE_MAX_N]
 * works without a recompile.  Defaults are the per-arch sweet spots measured
 * across M4 / Granite Rapids / Zen5 (see issue #2): bigger blocks amortise
 * per-block table/tree reload, which dominates on the smaller-L1 x86 parts. */
#ifndef PIVCO_BLOCK_SIZE
/* 32K is the cross-arch sweet spot measured across the full fleet (12 EC2
 * parts + M4): every uarch peaks at or near 32K, and the fast modern AVX-512
 * parts regress past it (cache cliff).  See docs/BLOCK_SIZE.md.
 *
 * Apple M-series is the exception: 32K regresses its text dists (its wide
 * L1/L2 already absorbs the per-block cost at 16K, after which the larger
 * working set only hurts), so it defaults to 16K.  Gated compile-time on
 * macOS/arm64 — a macOS arm64 binary is always Apple Silicon, and a macOS
 * binary's ISA is fixed at build time, so the gate is exact.  An explicit
 * -DPIVCO_BLOCK_SIZE still wins (this whole block is #ifndef-guarded).
 * Only M4 was measured; M1–M3 are assumed to share the wide-L1 behaviour.
 * A principled runtime gate keyed on cache size (which is the real cause)
 * could supersede this later. */
#if defined(__APPLE__) && defined(__aarch64__)
#define PIVCO_BLOCK_SIZE 16384
#else
#define PIVCO_BLOCK_SIZE 32768
#endif
#endif

/* Hard upper bound on a block's symbol count: the per-block wire header
 * stores N as a uint16 little-endian field, so N must fit in 16 bits. */
#define PIVCO_WIRE_MAX_N 65535

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
 * per-call conditional chains.  Classification is by "leafness" of the
 * children alone: bottom-up merges consume a leaf child's symbol
 * directly, so a leaf node itself is never dispatched — the parent's
 * merge materializes it.
 *
 * Same classification applies to all backends (scalar, NEON, AVX-512,
 * SSE).
 */
typedef enum {
    PIVCO_NODE_INTERNAL_FULL = 0,  /* both children internal — general partition/merge */
    PIVCO_NODE_INTERNAL_FLAT,      /* flat_depth[i] >= 2 — flat-subtree fast path */
    PIVCO_NODE_BOTH_LEAVES,        /* both children leaves — merge_cst_cst, partition_none */
    PIVCO_NODE_LEAF_LEFT,          /* left child leaf, right internal — merge_cst_vec, partition_right */
    PIVCO_NODE_LEAF,               /* leaf — consumed by the parent merge, never dispatched */
} pivco_node_type_t;

/* ---------- Huffman table ---------- */

/* Arch-specific precomputed gather tables for prim_enc_init.  Every pointer is
 * NULL unless the host arch fills it (only x86 SSE/AVX2 today, for the 4tab
 * no-shift merge).  Backends that don't need it ignore the struct; a backend
 * that does asserts the fields it uses are non-NULL.  The struct type is
 * arch-invariant (always these fields) — only the backing storage in the table
 * is arch-gated — so it never degenerates to an empty struct. */
typedef struct {
    const uint16_t *s2r_hi;      /* sym_to_rank[s] << 8 (u16) — x86 2tab merge */
} pivco_enc_init_aux_t;

typedef struct {
    /* Per-symbol encode info */
    uint16_t code[PIVCO_MAX_SYMBOLS];       /* canonical Huffman code */
    uint8_t  code_len[PIVCO_MAX_SYMBOLS];   /* code length (0 = unused) */

    /* "partbyrank" encode: a subtree's leaves are a contiguous rank range, so
     * per-node routing is `rank > split_rank` (8-bit, vs a 16-bit code bit-test)
     * and a flat subtree's local code is `rank - flat_base_rank`.  Filled by
     * pivco_build_table; byte-identical wire output. */
    uint8_t  sym_to_rank[PIVCO_MAX_SYMBOLS];        /* in-order leaf rank per symbol */
#if defined(__x86_64__) || defined(__i386__)
    /* Backing storage for enc_init_aux — the x86 2tab merge hi table (sym_to_rank
     * << 8).  Other arches don't allocate it.  Filled by pivco_build_table. */
    uint16_t enc_init_hi[PIVCO_MAX_SYMBOLS];
#endif
    /* Aux gather tables: pointers into the arch-gated storage above (x86) or all
     * NULL (other arches).  Self-referential — rebuild, don't bitwise-copy, a
     * table after pivco_build_table. */
    pivco_enc_init_aux_t enc_init_aux;
    uint8_t  split_rank[PIVCO_MAX_TREE_NODES];      /* max rank in node's left subtree */
    uint8_t  flat_base_rank[PIVCO_MAX_TREE_NODES];  /* min rank in a flat subtree */

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
    uint8_t  fse_enabled;      /* baked from pivco_cfg_t at build */

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
     * build_table after tree and flat_depth are finalized.  Decoders
     * switch on this instead of running per-call conditional chains. */
    uint8_t  node_type[PIVCO_MAX_TREE_NODES];
} pivco_table_t;

/* ---------- Tree-shape mode (build-time) ----------
 *
 * Experimental knob for paper-style ablations.  Changes how the chunks
 * are decomposed inside pivco_build_table; the codec downstream
 * picks up the resulting table->flat_depth/flat_offset/code[] uniformly.
 *
 *   OPTIMIZED      production: per length L, decompose c_L by its set
 *                  bits.  Produces non-canonical codes that maximize
 *                  flat-D>=2 subtree coverage.
 *   NAIVE          every symbol is a D=0 singleton.  Tree shape ==
 *                  pure canonical Huffman; no leaf fusion, no flat
 *                  subtrees.  Slowest decode; best baseline for "ph
 *                  without any tree optimizations vs Huff0".
 *   FUSED          allow D=1 sibling pairs but no D>=2 flats.  Tree
 *                  shape == canonical with `scatter_two` / `merge_two`
 *                  leaf fusion only.
 *   CANONICAL_FLAT chunks are derived from canonical code positions:
 *                  greedy peel the largest 2^k chunk such that the
 *                  canonical start code is 2^k-aligned and 2^k <=
 *                  remaining.  Produces canonical codes that happen
 *                  to contain flat subtrees; isolates the gain from
 *                  the OPTIMIZED non-canonical reorganization.
 *
 * Set via pivco_cfg_t.tree_mode at table build; both encoder and
 * decoder side must build with the same value (the wire format
 * carries only code lengths, not tree shape).  Default = OPTIMIZED. */
typedef enum {
    PIVCO_TREE_MODE_OPTIMIZED       = 0,
    PIVCO_TREE_MODE_NAIVE           = 1,
    PIVCO_TREE_MODE_FUSED           = 2,
    PIVCO_TREE_MODE_CANONICAL_FLAT  = 3,
} pivco_tree_mode_t;

/* ---------- Compression effort (encoder-side, build-time) ----------
 *
 * How much table-build time pivco_build_table spends shaping
 * the code lengths for DECOMPRESSION speed (the joint length/shape
 * pass in src/joint_lengths.c).  More shaping: slower table build,
 * faster decompression, ~same compressed size -- an adoption guard
 * only accepts shapes whose modeled bits stay within 1.5% of the
 * Huffman baseline AND whose modeled decode time improves by at least
 * 10%; on any reject the plain Huffman lengths are kept.  Encoder
 * side only: the wire carries plain code lengths, so ANY decoder
 * reads the output and both sides rebuild identical tables.
 *
 * The cost is per build_table CALL -- the file codec builds one table
 * per file -- so it only matters for small inputs or high call rates.
 * The superlatives are the extremes; most callers want the middle.
 * Set via pivco_cfg_t.effort at table build.  Default = PLAIN
 * (shaping is opt-in until the lambda/guard tuning settles; see
 * issue #20). */
typedef enum {
    PIVCO_EFFORT_PLAIN              = 0,  /* plain Huffman lengths: no
                                             shaping, no shaping time --
                                             the default */
    PIVCO_EFFORT_BALANCED           = 1,  /* a coarse grouped solve buys
                                             most of the decompress win */
    PIVCO_EFFORT_FASTER_DECOMPRESS  = 2,  /* auto-tier solve: nearly all
                                             of the win */
    PIVCO_EFFORT_FASTEST_DECOMPRESS = 3,  /* exact DP, provably optimal
                                             shape in-model: encode-once-
                                             decode-forever data */
    PIVCO_EFFORT_FASTEST_COMPRESS   = 4,  /* PLAIN below 256 KiB of
                                             input, BALANCED above --
                                             resolved by input size in the
                                             pivcohuf file codec; a bare
                                             build_table (no size known)
                                             treats it as BALANCED */
} pivco_effort_t;

/* ---------- Build configuration ----------
 *
 * The one user-settable configuration object.  Consumed only by the
 * table builds (pivco_build_table / _from_code_lens); every
 * field bakes into the resulting table, so config plays no role after
 * build -- encode/decode read everything they need from the table.
 * Pass NULL to a build to get pivco_cfg_default. */
typedef struct {
    pivco_tree_mode_t tree_mode;    /* default PIVCO_TREE_MODE_OPTIMIZED */
    pivco_effort_t    effort;       /* default PIVCO_EFFORT_PLAIN */
    int               fse_enabled;  /* default 1: per-node FSE attempts */
} pivco_cfg_t;

extern const pivco_cfg_t pivco_cfg_default;

/* ---------- Encoder / decoder contexts ----------
 *
 * A context owns the scratch memory one encode (or decode) stream
 * needs, plus running stats.  Single-threaded objects: create one per
 * thread, reuse it across blocks; create/free are malloc-priced,
 * everything between is allocation-free (scratch is preallocated for
 * PIVCO_WIRE_MAX_N at create and only grows on a larger need).  Holds
 * no config -- everything user-settable lives in pivco_cfg_t
 * at table build.
 *
 * `stats` accumulate over successful pivco_encode/_decode
 * calls (blocks, payload bytes in, stream bytes out -- and the mirror
 * for decode); the caller may zero the struct at any time.  `internal`
 * is the opaque scratch arena. */
typedef struct {
    uint64_t blocks;
    uint64_t bytes_in;
    uint64_t bytes_out;
} pivco_ctx_stats_t;

typedef struct {
    pivco_ctx_stats_t stats;
    void             *internal;
} pivco_encoder_t;

typedef struct {
    pivco_ctx_stats_t stats;
    void             *internal;
} pivco_decoder_t;

pivco_encoder_t *pivco_encoder_create(void);
pivco_decoder_t *pivco_decoder_create(void);
void             pivco_encoder_free(pivco_encoder_t *enc);
void             pivco_decoder_free(pivco_decoder_t *dec);

/* Byte histogram: ADDS counts of in[0..n) into freq[256] (caller
 * zeroes; accumulate across buffers/chunks freely).  SIMD on capable
 * hosts; scratch from the encoder context. */
int pivco_histogram(pivco_encoder_t *enc, const uint8_t *in, size_t n,
                    uint64_t freq[PIVCO_MAX_SYMBOLS]);

/* The pass itself (called by pivco_build_table between length
 * derivation and table construction; exposed for tests/benchmarks).
 * Rewrites lengths[] -- Huffman lengths for freq[], already limited to
 * PIVCO_MAX_CODE_LEN -- in place.  cfg as in the builds (NULL means
 * pivco_cfg_default).  Returns 0 if a shaped set was adopted, -1 if
 * the baseline was kept (PLAIN effort, non-OPTIMIZED tree mode, guard
 * reject, or internal failure such as malloc). */
int pivco_joint_optimize_lengths(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                 uint8_t lengths[PIVCO_MAX_SYMBOLS],
                                 const pivco_cfg_t *cfg);

/* ---------- FSE table-usage stats (debug instrumentation) ----------
 *
 * Per-table-id counters incremented inside the encoder every time an
 * FSE-coded bitmap is committed.  Slot 0 = "FSE attempted but did not
 * commit"; slots 1..PIVCO_FSE_NUM_TABLES = pivco_fse_freq[] table picked.
 * MUST be >= PIVCO_FSE_NUM_TABLES + 1 (static-asserted in pivco_fse.c).
 * Not thread-safe; intended for single-threaded analysis runs. */
#define PIVCO_FSE_STATS_SLOTS 51
void pivco_fse_stats_reset(void);
void pivco_fse_stats_get(uint64_t commit_count[PIVCO_FSE_STATS_SLOTS],
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
} pivco_fse_root_event_t;

int  pivco_fse_root_count(void);
void pivco_fse_root_get(int idx, pivco_fse_root_event_t *out);

/* ---------- Table construction ---------- */

int pivco_build_table(const pivco_cfg_t *cfg,
                              const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              pivco_table_t *table);

/* Build a Huffman table from already-known code lengths (the path used by
 * decoders that recovered code_lens from a wire format).  The tree is fully
 * determined by the lengths -- within-tier order is symbol-value ascending --
 * so encoder and decoder reconstruct identical tables with no extra wire info.
 *
 * Internally synthesises power-of-two frequencies that reproduce the lengths
 * and runs the same build pipeline as pivco_build_table.
 *
 * (Through wire v0.3 this also took a rank_within_tier array to reproduce a
 * frequency-based within-tier order; that ordering was dropped in v0.4 -- it
 * required extra wire bytes and only masked a frequency-blind FSE commit
 * policy.  See the reshape note in huffman_table.c.) */
int pivco_build_table_from_code_lens(
    const pivco_cfg_t *cfg,
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    pivco_table_t *table);

/* Fill the 2^MAX_CODE_LEN flat decode table (decode_sym/decode_len) used only
 * by the traditional flat-table decoder (trad_huffman_decode*).  Call after
 * building the table; pivco_build_table no longer fills it (the
 * production tree-walk decoder does not need it). */
void pivco_build_traditional_table(pivco_table_t *table);

/* ---------- PIVCO Huffman encode/decode (variable-N blocks, N ≤ PIVCO_BLOCK_SIZE) ----------
 *
 * The production entry points.  One block per call:
 *
 *   pivco_encode — encodes symbols[0..n) with `table` into
 *     `out`, sets *out_len.  `out` must hold PIVCO_MAX_ENCODED_SIZE
 *     bytes.  Writes a 2-byte LE N header at the start of the stream
 *     (see pivco_huffman_wire.h).  N must satisfy
 *     1 ≤ n ≤ PIVCO_BLOCK_SIZE; values outside that range return error.
 *
 *   pivco_decode — decodes one block from in[0..in_len) with
 *     `table` into `symbols`, sets *consumed to the stream bytes read.
 *     N comes from the wire — no `n` parameter — and `symbols` must
 *     have room for the worst case, typically PIVCO_BLOCK_SIZE.
 *
 * Both compile-time-dispatch to the best backend built into this
 * binary (the workers below) and accumulate ctx->stats on success;
 * that is their only difference from the workers. */

int pivco_encode(pivco_encoder_t *enc, const pivco_table_t *table,
                 const uint8_t *symbols, size_t n,
                 uint8_t *out, size_t *out_len);
int pivco_decode(pivco_decoder_t *dec, const pivco_table_t *table,
                 const uint8_t *in, size_t in_len,
                 uint8_t *symbols, size_t *consumed);

/* Per-backend workers: same contract as pivco_encode/_decode
 * minus the stats accounting.  Exposed for benches and tests; normal
 * callers use the dispatching pair above. */

int pivco_encode_scalar(pivco_encoder_t *enc, const pivco_table_t *table,
                        const uint8_t *symbols, size_t n,
                        uint8_t *out, size_t *out_len);


int pivco_decode_scalar(pivco_decoder_t *dec, const pivco_table_t *table,
                        const uint8_t *in, size_t in_len,
                        uint8_t *symbols, size_t *consumed);

#ifdef PIVCO_HAS_NEON
int pivco_encode_neon(pivco_encoder_t *enc, const pivco_table_t *table,
                      const uint8_t *symbols, size_t n,
                      uint8_t *out, size_t *out_len);

/* Bottom-up merge decode (NEON). */
int pivco_decode_bu_neon(pivco_decoder_t *dec, const pivco_table_t *table,
                         const uint8_t *in, size_t in_len,
                         uint8_t *symbols, size_t *consumed);
#endif

#ifdef PIVCO_HAS_SSE4
int pivco_encode_x86(pivco_encoder_t *enc, const pivco_table_t *table,
                     const uint8_t *symbols, size_t n,
                     uint8_t *out, size_t *out_len);

/* Bottom-up merge decode (x86 SSE4.1 / AVX-512 VBMI2). */
int pivco_decode_bu_x86(pivco_decoder_t *dec, const pivco_table_t *table,
                        const uint8_t *in, size_t in_len,
                        uint8_t *symbols, size_t *consumed);
#endif

/* Prior experimental NEON variants (neon2, neon2b, neon_fused_1leaf)
 * are preserved under extras/ as negative results. See extras/README_*
 * files for writeups. */

/* Prefix-radix research backend retired to extras/pivco_huffman_neon_prefix.c
 * (alongside its bench_prefix_profile.c and pivco_huffman_neon_common.h).
 * BU on the standard 2-way wire format beats it on all 29 distributions
 * across all 7 EC2 test hosts; no production caller remained.  See
 * docs/PREFIX_RADIX.md for the historical design record. */

#ifdef PIVCO_HAS_AVX512
int pivco_encode_avx512(pivco_encoder_t *enc, const pivco_table_t *table,
                        const uint8_t *symbols, size_t n,
                        uint8_t *out, size_t *out_len);

/* Bottom-up merge decode (AVX-512 VBMI2). */
int pivco_decode_bu_avx512(pivco_decoder_t *dec, const pivco_table_t *table,
                           const uint8_t *in, size_t in_len,
                           uint8_t *symbols, size_t *consumed);
#endif

/* Top-down (TD) decode entry points have been retired (2026-05-14).
 * BU is the production decoder on every platform.  TD implementations
 * still live in the legacy .c files as now-unreachable static functions;
 * step 3.8 of the unify-framework refactor retires them along with the
 * legacy .c files when codec.c takes over the encode/decode entries.
 * See extras/legacy_td/README.md for the git-archaeology pointer. */

/* ---------- Traditional Huffman encode/decode (for comparison) ---------- */

int trad_huffman_encode(const uint8_t *symbols, size_t n_symbols,
                        const pivco_table_t *table,
                        uint8_t *out, size_t *out_len, size_t *out_bits);

int trad_huffman_decode(const uint8_t *in, size_t in_bits,
                        const pivco_table_t *table,
                        uint8_t *symbols, size_t n_symbols);

/* SotA 4-stream encode/decode (huff0-style) */
int trad_huffman_encode_4s(const uint8_t *symbols, size_t n_symbols,
                           const pivco_table_t *table,
                           uint8_t *out, size_t *out_len);

int trad_huffman_decode_4s(const uint8_t *in, size_t in_len,
                           const pivco_table_t *table,
                           uint8_t *symbols, size_t n_symbols);

/* ---------- Instrumentation ---------- */
void pivco_instrument_node_size(int n);
void pivco_dump_node_size_hist(void);

#ifdef __cplusplus
}
#endif

#endif /* PIVCO_HUFFMAN_H */
