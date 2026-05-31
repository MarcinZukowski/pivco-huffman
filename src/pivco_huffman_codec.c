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
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
 * DFS, in-order: emit the partition bitmap at this node, then recurse
 * left, then right.  At each non-flat internal node, `codes_la[0..n)`
 * holds the surviving codes (left-aligned: top bit = current depth's
 * partition).  After partition, left stays in codes_la[0..n_left) and
 * right goes to tmp[0..n_right); the shift-by-1 in the primitive lets
 * the next recursion read bit 15 again. */

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
    uint8_t scratch[PIVCO_BLOCK_SIZE / 8 + 16];
    if (xor_flag) {
        for (int i = 0; i < nbytes; i++) scratch[i] = (uint8_t)~bm[i];
    } else {
        memcpy(scratch, bm, (size_t)nbytes);
    }

    uint8_t fse_out[PIVCO_BLOCK_SIZE];
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
                               uint16_t *codes_la, int n,
                               int depth,
                               uint8_t **out_ptr,
                               uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;  /* leaf — nothing to emit */

    /* Flat-subtree fast path: pack n*D bits, no marker, no K_right. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        prim_enc_pack_dN(codes_la, n, D, depth, *out_ptr);
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
     * never reads its scattered side, so HALF/BOTH_LEAVES skip that scatter. */
    int n_right;
    switch ((pivco_node_type_t)table->node_type[node_id]) {
    case PIVCO_NODE_BOTH_LEAVES:
        n_right = prim_enc_partition_none(codes_la, n, depth, bm);        break;
    case PIVCO_NODE_HALF_RIGHT:
        n_right = prim_enc_partition_right(codes_la, n, depth, bm, tmp);  break;
    case PIVCO_NODE_HALF_LEFT:
        n_right = prim_enc_partition_left(codes_la, n, depth, bm);        break;
    default:
        n_right = prim_enc_partition_full(codes_la, n, depth, bm, tmp);   break;
    }
    int n_left  = n - n_right;

    /* Optional FSE attempt on the raw bitmap.  On commit, marker_slot
     * and bm region are rewritten in place and *out_ptr is advanced to
     * the end of the FSE payload (which may be shorter than the raw
     * bitmap region we already reserved).  No-op otherwise. */
    codec_maybe_fse_attempt(marker_slot, bm, nbytes,
                             n, n_left, n_right, depth, out_ptr);

    wire_commit_kr_header(kr_slot, n_right);

    codec_encode_node(table, node->left,  codes_la, n_left,  depth + 1,
                       out_ptr, tmp + n_right);
    codec_encode_node(table, node->right, tmp,      n_right, depth + 1,
                       out_ptr, tmp + n_right);
}

int CODEC_ENCODE_ENTRY(const uint8_t *symbols,
                       const pivco_huffman_table_t *table,
                       uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;
    prim_codec_init();

    const int N = PIVCO_BLOCK_SIZE;

    /* Per-block left-aligned codes.  Built once per block via
     * prim_enc_init (a gather from table->code_la[symbols[i]]). */
    /* +16 slack: NEON's build_bitmap_partition primitive does a stride-8
     * SIMD partition whose 16-byte vst1q_u8 store can land up to 8 uint16
     * elements past the cursor at end-of-buffer.  The scalar primitive
     * doesn't need the padding but it costs us 16 bytes of stack per
     * call.  Same trick + same rationale as the legacy NEON encoder
     * carried since dense-codes_la landed (b31d269 2026-05-11). */
    uint16_t codes_la[PIVCO_BLOCK_SIZE + 16];
    prim_enc_init(codes_la, N, symbols, table->code_la);

    /* Scratch arena for the right-half partitions at each recursion
     * level.  Worst-case usage on a skewed tree: every recursion
     * pushes ~n_right elements which can be near n if the partition
     * is one-sided.  Size for (MAX_CODE_LEN+2) * BLOCK_SIZE — matches
     * the bound used elsewhere in the codec. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)malloc(tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;

    uint8_t *ptr = out;
    codec_encode_node(table, table->tree_root, codes_la, N, 0, &ptr, tmp);

    free(tmp);
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
 * Dispatch on node_type, computed at build-table time:
 *
 *   SKIP            — prefilled leaf, memset prefill_sym
 *   LEAF            — non-prefill leaf, memset node->symbol
 *   INTERNAL_FLAT   — packed-bits flat decode into out_buf
 *   BOTH_LEAVES     — both children leaves, merge_cst_cst directly
 *   HALF_RIGHT      — left child is the prefilled leaf, recurse right
 *   HALF_LEFT       — right child is the prefilled leaf, recurse left
 *   INTERNAL_FULL   — general merge: recurse both, merge
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

    case PIVCO_NODE_SKIP:
        memset(out_buf, table->prefill_sym, (size_t)K);
        return;

    case PIVCO_NODE_LEAF:
        memset(out_buf, (uint8_t)node->symbol, (size_t)K);
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
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);
        prim_merge_cst_cst(bm, K,
                               (uint8_t)table->tree[node->left].symbol,
                               (uint8_t)table->tree[node->right].symbol,
                               out_buf);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        /* If the right child is also a leaf, no recursion needed:
         * merge_cst_cst with prefill_sym on the left and the
         * right child's symbol on the right. */
        if (table->node_type[node->right] == (uint8_t)PIVCO_NODE_LEAF) {
            prim_merge_cst_cst(bm, K, table->prefill_sym,
                                   (uint8_t)table->tree[node->right].symbol,
                                   out_buf);
            return;
        }
        uint8_t *right_buf = scratch_top;
        codec_decode_subtree(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top + K_right);
        prim_merge_cst_vec(bm, K, table->prefill_sym,
                                    right_buf, out_buf);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        if (table->node_type[node->left] == (uint8_t)PIVCO_NODE_LEAF) {
            prim_merge_cst_cst(bm, K,
                                   (uint8_t)table->tree[node->left].symbol,
                                   table->prefill_sym, out_buf);
            return;
        }
        int K_left = K - K_right;
        uint8_t *left_buf = scratch_top;
        codec_decode_subtree(table, node->left, K_left,
                              left_buf, in_ptr, scratch_top + K_left);
        prim_merge_vec_cst(bm, K, left_buf,
                                     table->prefill_sym, out_buf);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);

        int left_kind  = table->node_type[node->left];
        int right_kind = table->node_type[node->right];

        if (left_kind == (uint8_t)PIVCO_NODE_LEAF
            && right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            prim_merge_cst_cst(bm, K,
                                   (uint8_t)table->tree[node->left].symbol,
                                   (uint8_t)table->tree[node->right].symbol,
                                   out_buf);
            return;
        }
        if (left_kind == (uint8_t)PIVCO_NODE_LEAF) {
            uint8_t *right_buf = scratch_top;
            codec_decode_subtree(table, node->right, K_right,
                                  right_buf, in_ptr, scratch_top + K_right);
            prim_merge_cst_vec(bm, K,
                                        (uint8_t)table->tree[node->left].symbol,
                                        right_buf, out_buf);
            return;
        }
        if (right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            int K_left = K - K_right;
            uint8_t *left_buf = scratch_top;
            codec_decode_subtree(table, node->left, K_left,
                                  left_buf, in_ptr, scratch_top + K_left);
            prim_merge_vec_cst(bm, K, left_buf,
                                         (uint8_t)table->tree[node->right].symbol,
                                         out_buf);
            return;
        }

        /* General case: both children non-leaf.  Recurse into both
         * with disjoint scratch slices, then merge. */
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

    const int N = PIVCO_BLOCK_SIZE;
    const pivco_tree_node_t *root = &table->tree[table->tree_root];
    const uint8_t *ptr = in;

    /* Root-is-leaf: fill everything with the single symbol. */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Fast path: BOTH_LEAVES at root.  Common on heavily-skewed
     * distributions (proba80, two_sym_eq, calgary_pic) where one
     * symbol dominates and its tree has just a 1-bit code: hot blocks
     * collapse to "read the K-bit partition, blend two symbols".  The
     * legacy bu_neon / bu_x86 entries kept this fast path and it
     * accounted for the proba80 win: skipping the recursive
     * codec_decode_subtree machinery (switch dispatch + bm_scratch
     * stack frame + scratch TLS reference) saves ~1.5 us per block on
     * Apple M4 and Xeon Granite Rapids, where the actual merge is
     * only ~200 ns.  Lost during the unify-framework refactor;
     * restored 2026-05-14 after a ~3-4x regression on proba80 across
     * all hosts. */
    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_BOTH_LEAVES) {
        uint8_t bm_scratch[PIVCO_BLOCK_SIZE / 8 + 16];
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
     * N bytes.  Bounded by (MAX_CODE_LEN+2) * N. */
    static __thread uint8_t scratch[(size_t)PIVCO_BLOCK_SIZE *
                                     (PIVCO_MAX_CODE_LEN + 2)];

    codec_decode_subtree(table, table->tree_root, N,
                          symbols, &ptr, scratch);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}
