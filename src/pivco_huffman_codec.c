/* pivco_huffman_codec.c — unified encode + bottom-up decode.
 *
 * One source file, compiled once per backend tier (CMake passes
 * -DPIVCO_BACKEND_{SCALAR,NEON,X86,AVX512}).  The tree walk + dispatch
 * + wire format are identical across backends; the per-node SIMD work
 * lives in pivco_huffman_primitives_<backend>.h, selected via the
 * router pivco_huffman_primitives.h.
 *
 * Two responsibilities only:
 *
 *   1. Walk the Huffman tree (encode recursion + BU decode recursion).
 *   2. Read/write per-node wire records via pivco_huffman_wire.h.
 *
 * Everything backend-shaped (bitmap build, partition, flat-decode,
 * merge, etc.) is a `prim_*` call.  No vector types here.
 *
 * The bottom-up decoder is the production path (top-down has been
 * parked).  Encode is shared.
 */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_wire.h"
#include "pivco_huffman_primitives.h"
#include "pivco_prof.h"
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Thread-local growable decode scratch arena.  Replaces the former fixed
 * `static __thread` array sized at PIVCO_BLOCK_SIZE so the block size is a
 * pure runtime parameter (the wire N header carries the per-block count).
 * Grows on demand and is reused across blocks; never shrinks.  One instance
 * per backend translation unit, which is exactly what we want. */
static __thread uint8_t *g_decode_scratch     = NULL;
static __thread size_t   g_decode_scratch_cap = 0;

static uint8_t *decode_scratch_ensure(size_t need)
{
    if (need > g_decode_scratch_cap) {
        uint8_t *p = (uint8_t *)realloc(g_decode_scratch, need);
        if (!p) return NULL;
        g_decode_scratch     = p;
        g_decode_scratch_cap = need;
    }
    return g_decode_scratch;
}

/* Thread-local growable encode scratch arena.  Mirrors the decode arena
 * above: holds the per-block ranks buffer + the tree-walk's right-half
 * recursion scratch, grown on demand and reused across blocks (never shrinks),
 * so a block-loop encode doesn't malloc/free per block.
 * @todo stopgap: this per-thread global should become part of an explicit
 * encoder API context (a pivco_huffman_encoder_t handle) so the scratch's
 * ownership and lifetime are caller-controlled rather than a hidden
 * thread_local.  See IDEAS.md. */
static __thread uint8_t *g_encode_scratch     = NULL;
static __thread size_t   g_encode_scratch_cap = 0;

static uint8_t *encode_scratch_ensure(size_t need)
{
    if (need > g_encode_scratch_cap) {
        uint8_t *p = (uint8_t *)realloc(g_encode_scratch, need);
        if (!p) return NULL;
        g_encode_scratch     = p;
        g_encode_scratch_cap = need;
    }
    return g_encode_scratch;
}

/* ---------- FSE dispatch parameters ----------
 *
 * The thresholds match the NEON encoder's settings so the wire format
 * is byte-identical across backends: same skew threshold, same per-
 * codeword cost gate, same minimum bitmap size.  See docs/FSE-V0.md for the
 * derivation of each value.  Overridable at build time via -D... . */
#ifndef PIVCO_FSE_MIN_THRESHOLD
#define PIVCO_FSE_MIN_THRESHOLD 0.625
#endif
#ifndef PIVCO_FSE_MIN_RATIO
#define PIVCO_FSE_MIN_RATIO     0.95
#endif
#ifndef PIVCO_FSE_MIN_BITMAP_BYTES
#define PIVCO_FSE_MIN_BITMAP_BYTES 32
#endif

/* FSE per-table-id stats live in src/pivco_huffman.c (backend-neutral
 * TU) so the symbols resolve regardless of backend.  codec.c writes
 * the counters every time it commits or rejects an FSE attempt. */
extern uint64_t g_pivco_fse_commit  [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_attempt [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_bytes_in [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_bytes_out[PIVCO_FSE_STATS_SLOTS];

/* ---------- Backend → entry-point name ---------- */

#if defined(PIVCO_BACKEND_SCALAR)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_scalar
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_scalar
#elif defined(PIVCO_BACKEND_NEON)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_neon
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_neon
#elif defined(PIVCO_BACKEND_X86)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_x86
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_x86
#elif defined(PIVCO_BACKEND_AVX512)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_avx512
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_avx512
#else
#  error "pivco_huffman_codec.c needs PIVCO_BACKEND_{SCALAR,NEON,X86,AVX512}"
#endif

/* ---------- Encode tree walk ---------- *
 *
 * DFS, pre-order: emit the partition bitmap at this node, then recurse left,
 * then right.  At each non-flat internal node, `ranks[0..n)` holds the
 * surviving leaves' in-order ranks; partition routes each by `rank >
 * split_rank[node]`, leaving the left half in place in `ranks[0..n_left)` and
 * compacting the right half into `tmp[0..n_right)`.  The recursion descends
 * left on `ranks`, right on `tmp`. */

/* Arch-agnostic FSE attempt on a freshly-built raw bitmap.
 *
 * Inputs:
 *   marker_slot  — points at the 1-byte FSE marker (currently 0 = raw)
 *   bm           — points at the ceil(n/8)-byte raw bitmap region
 *                  immediately after the marker
 *   nbytes       — bitmap_bytes(n)
 *   n / n_left / n_right — partition counts (for the skew test)
 *   depth        — for the codeword-cost gate
 *   out_ptr      — cursor; advanced past the FSE payload on commit
 *
 * On commit: rewrites *marker_slot, replaces bm with [fse_len:u16
 * LE][fse_payload], advances *out_ptr to one past the payload.  Stats
 * (g_pivco_fse_*) are bumped.
 *
 * On no-commit / no-attempt: stream and stats untouched.
 *
 * No-op when PIVCO_HAS_FSE is not defined. */
static inline void codec_maybe_fse_attempt(uint8_t *marker_slot,
                                            uint8_t *bm, int nbytes,
                                            int n, int n_left, int n_right,
                                            int depth, uint8_t **out_ptr)
{
#ifdef PIVCO_HAS_FSE
    if (!pivco_huffman_get_fse_enabled()) return;
    if (nbytes < PIVCO_FSE_MIN_BITMAP_BYTES) return;

    int n_major = (n_left >= n_right) ? n_left : n_right;
    double p_major = (n > 0) ? (double)n_major / (double)n : 0.0;
    if (p_major < PIVCO_FSE_MIN_THRESHOLD) return;

    int t_id = pivco_fse_select_table(p_major);
    if (t_id < 1) return;
    assert(t_id < PIVCO_FSE_STATS_SLOTS);  /* guards the g_pivco_fse_* indexing */

    int xor_flag = (n_right > n_left);
    uint8_t scratch[(size_t)nbytes + 16];
    if (xor_flag) {
        for (int i = 0; i < nbytes; i++) scratch[i] = (uint8_t)~bm[i];
    } else {
        memcpy(scratch, bm, (size_t)nbytes);
    }

    uint8_t fse_out[(size_t)nbytes + 64];
    size_t fse_len = 0;
    pivco_fse_status_t rc = pivco_fse_compress(t_id, scratch, (size_t)nbytes,
                                                fse_out, sizeof(fse_out),
                                                &fse_len);
    g_pivco_fse_attempt[t_id]++;
    if (rc != PIVCO_FSE_OK) {
        g_pivco_fse_commit[0]++;     /* slot 0 = attempted, rejected */
        return;
    }
    /* Per-codeword commit gate (see docs/FSE-V0.md):
     *   raw: every codeword through this node costs (depth + 1) bits
     *   fse: (depth + (fse_len + 2 wire-prefix) * 8 / n) bits
     * Commit iff (depth + fse_frac) <= MIN_RATIO * (depth + 1). */
    double fse_frac = (double)(fse_len + 2) * 8.0 / (double)n;
    double codeword_ratio = ((double)depth + fse_frac) /
                              ((double)depth + 1.0);
    if (codeword_ratio > (double)PIVCO_FSE_MIN_RATIO) {
        g_pivco_fse_commit[0]++;
        return;
    }

    /* Commit: rewrite marker + bitmap region with [fse_len][payload],
     * adjust the wire cursor to one past the payload. */
    *marker_slot = (uint8_t)((xor_flag ? 0x80 : 0) | t_id);
    uint8_t *p = bm;
    *p++ = (uint8_t)( fse_len       & 0xFF);
    *p++ = (uint8_t)((fse_len >> 8) & 0xFF);
    memcpy(p, fse_out, fse_len);
    *out_ptr = p + fse_len;

    g_pivco_fse_commit  [t_id]++;
    g_pivco_fse_bytes_in [t_id] += (uint64_t)nbytes;
    g_pivco_fse_bytes_out[t_id] += (uint64_t)(fse_len + 3);
#else
    (void)marker_slot; (void)bm; (void)nbytes;
    (void)n; (void)n_left; (void)n_right; (void)depth; (void)out_ptr;
#endif
}

static void codec_encode_node(const pivco_huffman_table_t *table,
                               int16_t node_id,
                               uint8_t *ranks, int n,
                               int depth,
                               uint8_t **out_ptr,
                               uint8_t *tmp)
{
    if (n == 0) return;
    PROF_COUNT_ONLY(PROF_ENC_NODE_VISIT, n);

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;  /* leaf — nothing to emit */

    /* Flat-subtree fast path: pack n*D bits, no marker, no K_right. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        PROF_TIC();
        prim_enc_pack_dN(ranks, n, D, table->flat_base_rank[node_id], *out_ptr);
        PROF_TOC(PROF_ENC_FLAT, n);
        *out_ptr += total_bytes;
        return;
    }

    /* Non-flat internal node.  codec.c owns: K_right header reservation,
     * FSE marker byte, optional FSE-attempt on the raw bitmap.  The
     * arch-specific primitive does only the SIMD-bound work: build the
     * raw bitmap and partition codes_la. */
    uint8_t *kr_slot = wire_reserve_kr_header(table, node_id, out_ptr);

    /* Reserve marker (default = 0, raw bitmap). */
    uint8_t *marker_slot = *out_ptr;
    *marker_slot = 0;
    *out_ptr += 1;

    /* Reserve the bitmap region; primitive fills it in. */
    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    *out_ptr += nbytes;

    /* Pick the partition variant by node_type, mirroring the decode-side
     * dispatch.  The bitmap (and thus the wire bytes) is identical across
     * variants; only the encode-internal scatter work differs — a leaf child
     * never reads its scattered side, so that side's scatter is skipped:
     * BOTH_LEAVES stores nothing, LEAF_LEFT only the right (compacted into
     * tmp), LEAF_RIGHT only the left, FULL both. */
    uint8_t thr = table->split_rank[node_id];
    int n_right;
    PROF_TIC();
    switch ((pivco_node_type_t)table->node_type[node_id]) {
    case PIVCO_NODE_BOTH_LEAVES:
        n_right = prim_enc_partition_none(ranks, n, thr, bm);        break;
    case PIVCO_NODE_LEAF_LEFT:
        n_right = prim_enc_partition_right(ranks, n, thr, bm, tmp);  break;
    case PIVCO_NODE_LEAF_RIGHT:
        n_right = prim_enc_partition_left(ranks, n, thr, bm);        break;
    default:
        n_right = prim_enc_partition_full(ranks, n, thr, bm, tmp);   break;
    }
    PROF_TOC(PROF_ENC_NODE_FULL, n);
    int n_left  = n - n_right;

    /* Optional FSE attempt on the raw bitmap.  On commit, marker_slot
     * and bm region are rewritten in place and *out_ptr is advanced to
     * the end of the FSE payload (which may be shorter than the raw
     * bitmap region we already reserved).  No-op otherwise. */
    codec_maybe_fse_attempt(marker_slot, bm, nbytes,
                             n, n_left, n_right, depth, out_ptr);

    wire_commit_kr_header(kr_slot, n_right);

    codec_encode_node(table, node->left,  ranks, n_left,  depth + 1,
                       out_ptr, tmp + n_right);
    codec_encode_node(table, node->right, tmp,   n_right, depth + 1,
                       out_ptr, tmp + n_right);
}

int CODEC_ENCODE_ENTRY(const uint8_t *symbols, size_t n,
                       const pivco_huffman_table_t *table,
                       uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;
    if (n == 0 || n > PIVCO_WIRE_MAX_N) return PIVCO_ERR_OVERFLOW;
    prim_codec_init();

    const int N = (int)n;

    /* Block header: write N as the first 2 bytes so the decoder can
     * recover it without an out-of-band channel. */
    uint8_t *ptr = out;
    wire_write_block_n(ptr, N);
    ptr += PIVCO_BLOCK_N_BYTES;

    /* One heap block: the per-block ranks buffer + the recursion's right-half
     * scratch (see the tree-walk note above).  +64 slack on ranks absorbs the
     * SIMD partition's over-wide (16/64-byte) tail store at end-of-buffer; the
     * scratch holds one right-half per recursion level, hence (MAX_CODE_LEN+2)*N. */
    const size_t ranks_capacity = (size_t)N + 64;
    const size_t tmp_capacity   = (size_t)N * (PIVCO_MAX_CODE_LEN + 2);
    uint8_t *ranks = encode_scratch_ensure(ranks_capacity + tmp_capacity);
    if (!ranks) return PIVCO_ERR_NULL;
    uint8_t *tmp = ranks + ranks_capacity;

    /* ranks[i] = in-order rank of symbols[i] (gather table->sym_to_rank). */
    PROF_COUNT_ONLY(PROF_ENC_ENTRY, N);
    PROF_TIC();
    prim_enc_init(ranks, N, symbols, table->sym_to_rank, &table->enc_init_aux);
    PROF_TOC(PROF_ENC_INIT, N);

    codec_encode_node(table, table->tree_root, ranks, N, 0, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- Bottom-up decode tree walk ---------- *
 *
 * Bottom-up: each call decodes a subtree into a contiguous K-byte
 * output buffer.  Internal nodes recurse into their children (which
 * write to scratch arenas), then merge per the bitmap.  The flat-
 * subtree fast path bypasses recursion entirely.
 *
 * Dispatch on node_type, computed at build-table time (by children's
 * leafness — a leaf child's symbol goes straight into the parent's
 * merge, so the walk never recurses into a leaf):
 *
 *   INTERNAL_FLAT   — packed-bits flat decode into out_buf
 *   BOTH_LEAVES     — both children leaves, merge_cst_cst directly
 *   LEAF_LEFT       — left child leaf, recurse right, merge_cst_vec
 *   LEAF_RIGHT      — mirror (structurally absent under canonical codes)
 *   INTERNAL_FULL   — both children internal: recurse both, merge_vec_vec
 *
 * `scratch_top` is the arena pointer for child output buffers; each
 * caller bumps it past its own K bytes when calling further down. */

static void codec_decode_subtree(const pivco_huffman_table_t *table,
                                   int16_t node_id, int K,
                                   uint8_t *out_buf,
                                   const uint8_t **in_ptr,
                                   uint8_t *scratch_top)
{
    if (K == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];

    switch ((pivco_node_type_t)table->node_type[node_id]) {

    case PIVCO_NODE_LEAF:
        /* Unreachable: every parent consumes a leaf child via its
         * cst_* merge instead of recursing into it. */
        assert(!"codec_decode_subtree dispatched on a leaf");
        return;

    case PIVCO_NODE_INTERNAL_FLAT: {
        int D = table->flat_depth[node_id];
        int total_bytes = (K * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s =
            &table->flat_code_to_sym[table->flat_offset[node_id]];
        prim_merge_flat(out_buf, K, bm, D, c2s);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        /* No K_right header (kr_header_needed returns false). */
        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);
        prim_merge_cst_cst(bm, K,
                           (uint8_t)table->tree[node->left].symbol,
                           (uint8_t)table->tree[node->right].symbol,
                           out_buf);
        return;
    }

    case PIVCO_NODE_LEAF_LEFT: {
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        uint8_t *right_buf = scratch_top;
        codec_decode_subtree(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top + K_right);
        prim_merge_cst_vec(bm, K,
                           (uint8_t)table->tree[node->left].symbol,
                           right_buf, out_buf);
        return;
    }

    case PIVCO_NODE_LEAF_RIGHT: {
        /* Structurally absent under canonical codes (leaf sorts left);
         * kept dispatchable for safety. */
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        int K_left = K - K_right;
        uint8_t *left_buf = scratch_top;
        codec_decode_subtree(table, node->left, K_left,
                              left_buf, in_ptr, scratch_top + K_left);
        prim_merge_vec_cst(bm, K, left_buf,
                           (uint8_t)table->tree[node->right].symbol,
                           out_buf);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        /* Both children internal.  Recurse into both with disjoint
         * scratch slices, then merge. */
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        int K_left = K - K_right;
        uint8_t *left_buf  = scratch_top;
        uint8_t *right_buf = scratch_top + K_left;
        uint8_t *new_scratch_top = scratch_top + K;

        codec_decode_subtree(table, node->left,  K_left,
                              left_buf,  in_ptr, new_scratch_top);
        codec_decode_subtree(table, node->right, K_right,
                              right_buf, in_ptr, new_scratch_top);
        prim_merge_vec_vec(bm, K, left_buf, right_buf, out_buf);
        return;
    }
    }
}

int CODEC_DECODE_ENTRY(const uint8_t *in, size_t in_len,
                       const pivco_huffman_table_t *table,
                       uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;
    prim_codec_init();

    /* Block header: first 2 bytes are N (symbol count for this block). */
    const uint8_t *ptr = in;
    const int N = wire_read_block_n(&ptr);
    if (N <= 0 || N > PIVCO_WIRE_MAX_N) return PIVCO_ERR_CORRUPT;
    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root-is-leaf: fill everything with the single symbol. */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Fast path: BOTH_LEAVES at root — a 2-symbol (or single-symbol)
     * tree, where the whole block collapses to "read the K-bit
     * partition, blend two symbols".  Skips the recursive
     * codec_decode_subtree machinery (switch dispatch + bm_scratch
     * stack frame + scratch TLS reference / arena ensure).  Worth −26%
     * on two_sym decode on older narrow x86 (IvyBridge), noise on
     * modern hosts.  TODO: consider removing this extreme-case
     * optimization. */
    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_BOTH_LEAVES) {
        uint8_t bm_scratch[(size_t)bitmap_bytes(N) + 16];
        const uint8_t *bm = wire_read_bitmap(&ptr, N, bm_scratch);
        const pivco_tree_node_t *left_child  = &table->tree[root->left];
        const pivco_tree_node_t *right_child = &table->tree[root->right];
        prim_merge_cst_cst(bm, N,
                               (uint8_t)left_child->symbol,
                               (uint8_t)right_child->symbol,
                               symbols);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Scratch arena.  Worst case at a heavily-skewed node, the
     * partition is one-sided so a single recursion can consume up to
     * N bytes.  Bounded by (MAX_CODE_LEN+2) * N.  Grown on demand from a
     * thread-local heap buffer so block size is a runtime parameter. */
    uint8_t *scratch =
        decode_scratch_ensure((size_t)N * (PIVCO_MAX_CODE_LEN + 2));
    if (!scratch) return PIVCO_ERR_NULL;

    codec_decode_subtree(table, table->tree_root, N,
                          symbols, &ptr, scratch);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}
