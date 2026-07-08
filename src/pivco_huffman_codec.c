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
#include "pivco_check.h"

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

/* ---------- Runtime placement flags (issue #10 integration) ----------
 *
 * PIVCO_DEC_CARVE=1    page-hazard-aware scratch carving: child slices
 *                      are nudged off page boundaries (issue #8's
 *                      split-load penalty on a parked merge cursor).
 * PIVCO_DEC_INPLACE=1  in-place merge: an internal node's longer child
 *                      decodes into out_buf's tail instead of scratch,
 *                      halving the merge working set.
 *
 * Both default OFF: placement is then byte-identical to the plain bump
 * allocator.  Read once per process (per backend TU). */
static int g_dec_carve   = -1;
static int g_dec_inplace = -1;
static inline int dec_flag(const char *name, int *cache)
{
    int v = *cache;
    if (v < 0) {
        const char *e = getenv(name);
        v = (e && e[0] == '1');
        *cache = v;
    }
    return v;
}

/* SCRATCH_PAGE is the page-split-load hazard granularity used by
 * scratch_carve below: the hardware page size (16 KB Apple Silicon,
 * 4 KB elsewhere in the fleet; a 64 KB-page Linux arm64 kernel gets
 * needless-but-harmless 4 KB padding). */
#if defined(__APPLE__) && defined(__aarch64__)
#define SCRATCH_PAGE   ((uintptr_t)16384)
#else
#define SCRATCH_PAGE   ((uintptr_t)4096)
#endif
#define MERGE_OVERREAD ((uintptr_t)PIVCO_PRIM_MERGE_OVERREAD)
/* Keep the first ~cache line of every slice straddle-free: a merge
 * cursor lingers NEAR offset 0 (not just at it) while the other side
 * drains. */
#define START_GUARD    ((uintptr_t)64)

/* ALL padding draws from a per-block budget (reset in the decode
 * entry): when it runs out, slices are placed unpadded -- a perf
 * hazard reachable only by adversarial page phases, never a safety
 * issue.  Absolute (not per-page) so the arena bound is identical on
 * every platform. */
#define PIVCO_SCRATCH_PAD_BUDGET ((size_t)16384)
static __thread size_t g_scratch_pad_left;

/* Carve a child slice off the arena.  With PIVCO_DEC_CARVE off this is
 * a plain bump.  With it on, page-hazard-aware placement (issue #8):
 *   - sub-page slice: never crosses a page boundary (its cursor dwells
 *     everywhere inside it, so a mid-slice straddle is the hammer);
 *   - page-crossing slice: interior straddles are unavoidable (and its
 *     cursor moves fast anyway); small nudges keep the START zone and
 *     the exhausted-cursor END position straddle-free (the end lands ON
 *     a boundary, so the end-position load reads the next page without
 *     straddling). */
static inline uint8_t *scratch_carve(uint8_t **top, int size)
{
    uintptr_t p = (uintptr_t)*top;
    if (g_dec_carve) {
        uintptr_t rem = SCRATCH_PAGE - (p & (SCRATCH_PAGE - 1));  /* 1..PAGE */
        uintptr_t pad = 0;
        if ((uintptr_t)size + MERGE_OVERREAD <= SCRATCH_PAGE) {
            if ((uintptr_t)size + MERGE_OVERREAD > rem)
                pad = rem;                              /* sub-page: never cross */
        } else {
            if (rem < START_GUARD) pad = rem;           /* clean start */
            uintptr_t end  = p + pad + (uintptr_t)size;
            uintptr_t erem = SCRATCH_PAGE - (end & (SCRATCH_PAGE - 1));
            if (erem < MERGE_OVERREAD) {
                pad += erem;                            /* end onto boundary */
                uintptr_t srem = SCRATCH_PAGE - ((p + pad) & (SCRATCH_PAGE - 1));
                if (srem < START_GUARD) pad += srem;    /* re-clean start */
            }
        }
        if (pad <= g_scratch_pad_left) g_scratch_pad_left -= pad; else pad = 0;
        p += pad;
    }
    *top = (uint8_t *)(p + (uintptr_t)size);
    return (uint8_t *)p;
}

/* Place an internal node's tail child at out_buf + offset (in-place
 * merge; only called when tail_ok && PIVCO_DEC_INPLACE).  A tail's
 * position is fixed by the merge invariant, so it cannot be nudged;
 * the one dangerous placement is a SUB-PAGE tail that would cross a
 * page boundary (a small slice's cursor is parked everywhere in it) --
 * that one falls back to a no-cross carve, charging the extra scratch
 * content to the same pad budget. */
static inline uint8_t *place_tail(uint8_t *out_buf, int offset, int size,
                                  uint8_t **top)
{
    uint8_t *p = out_buf + offset;
    if (g_dec_carve && (uintptr_t)size + MERGE_OVERREAD <= SCRATCH_PAGE) {
        uintptr_t rem = SCRATCH_PAGE - ((uintptr_t)p & (SCRATCH_PAGE - 1));
        if ((uintptr_t)size + MERGE_OVERREAD > rem &&
            (size_t)size <= g_scratch_pad_left) {
            g_scratch_pad_left -= (size_t)size;   /* charge the content... */
            return scratch_carve(top, size);      /* ...the carve pads itself */
        }
    }
    return p;
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
 * DFS, emitting records in decompression order (an Euler walk): the
 * partition runs at node entry (it routes the ranks the recursion
 * needs) and the K_right header is written there too, but the node's
 * marker+bitmap record is emitted after the children's regions —
 * exactly where the decoder's merge consumes it, so the decoder reads
 * the stream strictly forward.  At each non-flat internal node,
 * `ranks[0..n)` holds the surviving leaves' in-order ranks; partition
 * routes each by `rank > split_rank[node]`, leaving the left half in
 * place in `ranks[0..n_left)` and compacting the right half into
 * `tmp[0..n_right)`.  The recursion descends left on `ranks`, right on
 * `tmp`.  The bitmap is staged in a stack buffer across the recursion
 * (its final stream position depends on the children's — FSE-variable —
 * encoded sizes, so it can't be written in place up front);
 * ≤ bitmap_bytes(N)+64 per level, tree height ≤ PIVCO_MAX_CODE_LEN
 * levels. */

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
    PIVCO_CHECK(t_id < PIVCO_FSE_STATS_SLOTS);  /* guards the g_pivco_fse_* indexing */

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

    /* Non-flat internal node.  codec.c owns: K_right header, FSE marker
     * byte, optional FSE-attempt on the raw bitmap.  The arch-specific
     * primitive does only the SIMD-bound work: build the raw bitmap and
     * partition the ranks.
     *
     * The bitmap is built into a stack staging buffer (+64 slack
     * absorbs the SIMD partitions' over-wide tail stores) and copied
     * into the stream after the children's regions. */
    int nbytes = bitmap_bytes(n);
    uint8_t bm_stage[(size_t)nbytes + 64];

    /* Pick the partition variant by node_type, mirroring the decode-side
     * dispatch.  The bitmap (and thus the wire bytes) is identical across
     * variants; only the encode-internal scatter work differs — a leaf child
     * never reads its scattered side, so that side's scatter is skipped:
     * BOTH_LEAVES stores nothing, LEAF_LEFT only the right (compacted into
     * tmp), FULL both. */
    uint8_t thr = table->split_rank[node_id];
    int n_right;
    PROF_TIC();
    switch ((pivco_node_type_t)table->node_type[node_id]) {
    case PIVCO_NODE_BOTH_LEAVES:
        n_right = prim_enc_partition_none(ranks, n, thr, bm_stage);        break;
    case PIVCO_NODE_LEAF_LEFT:
        n_right = prim_enc_partition_right(ranks, n, thr, bm_stage, tmp);  break;
    default:
        n_right = prim_enc_partition_full(ranks, n, thr, bm_stage, tmp);   break;
    }
    PROF_TOC(PROF_ENC_NODE_FULL, n);
    int n_left  = n - n_right;

    /* One K_right header per recursion site, consumed by the decoder
     * at node entry so it can size both children before their regions
     * arrive. */
    wire_write_kr_header(table, node_id, out_ptr, n_right);

    codec_encode_node(table, node->left,  ranks, n_left,  depth + 1,
                       out_ptr, tmp + n_right);
    codec_encode_node(table, node->right, tmp,   n_right, depth + 1,
                       out_ptr, tmp + n_right);

    /* Emit this node's record: marker + staged bitmap.  The FSE attempt
     * may rewrite marker+bm in place with [fse_len][payload] and pull
     * *out_ptr back to the payload end.  No-op otherwise. */
    uint8_t *marker_slot = *out_ptr;
    *marker_slot = 0;
    *out_ptr += 1;
    uint8_t *bm = *out_ptr;
    memcpy(bm, bm_stage, (size_t)nbytes);
    *out_ptr += nbytes;
    codec_maybe_fse_attempt(marker_slot, bm, nbytes,
                             n, n_left, n_right, depth, out_ptr);
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
 * The wire is in decompression order, so the input cursor is consumed
 * strictly forward and each record is loaded exactly where it is
 * used: the K_right header at node entry, the children's regions
 * during their recursion, the node's bitmap right before its merge.
 * The bm_scratch VLA for FSE-coded bitmaps is thus also allocated
 * after the recursion, so its lifetime doesn't span the subtree.
 *
 * Dispatch on node_type, computed at build-table time (by children's
 * leafness — a leaf child's symbol goes straight into the parent's
 * merge, so the walk never recurses into a leaf):
 *
 *   INTERNAL_FLAT   — packed-bits flat decode into out_buf
 *   BOTH_LEAVES     — both children leaves, merge_cst_cst directly
 *   LEAF_LEFT       — left child leaf, recurse right, merge_cst_vec
 *   INTERNAL_FULL   — both children internal: recurse both, merge_vec_vec
 *
 * `scratch_top` is the arena pointer for child output buffers; each
 * caller carves its children's slices off it (see scratch_carve) when
 * calling further down.
 *
 * `tail_ok` is nonzero when out_buf is arena-backed (a carve or a
 * nested tail) and so has MERGE_OVERREAD trailing slack: a child may
 * then be placed in its tail (see place_tail) when PIVCO_DEC_INPLACE
 * is on.  The root call passes 0 -- the caller's output buffer has no
 * over-read slack. */

static void codec_decode_subtree(const pivco_huffman_table_t *table,
                                   int16_t node_id, int K,
                                   uint8_t *out_buf,
                                   const uint8_t **in_ptr,
                                   uint8_t *scratch_top, int tail_ok)
{
    if (K == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];

    switch ((pivco_node_type_t)table->node_type[node_id]) {

    case PIVCO_NODE_LEAF:
        /* Unreachable: every parent consumes a leaf child via its
         * cst_* merge instead of recursing into it. */
        pivco_check_fail("codec_decode_subtree dispatched on a leaf",
                         __FILE__, __LINE__);

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
        /* K_right header at entry sizes the right child; recurse right;
         * then the [marker][bitmap] record — which the merge consumes —
         * is read right before the merge (post-order). */
        int K_right = wire_read_kr_header(table, node_id, in_ptr);

        uint8_t *right_buf = tail_ok
            ? place_tail(out_buf, K - K_right, K_right, &scratch_top)
            : scratch_carve(&scratch_top, K_right);
        codec_decode_subtree(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top, g_dec_inplace);

        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);
        prim_merge_cst_vec(bm, K,
                           (uint8_t)table->tree[node->left].symbol,
                           right_buf, out_buf);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        /* Both children internal.  The K_right header at entry sizes
         * both children so the recursion can carve disjoint scratch
         * slices; the bitmap is read after both children decode, right
         * before the merge (post-order). */
        int K_right = wire_read_kr_header(table, node_id, in_ptr);

        int K_left = K - K_right;
        uint8_t *left_buf, *right_buf;
        if (tail_ok && K_left >= K_right) {
            /* Longer child decodes into out_buf's tail (free -- see
             * place_tail), the shorter into a scratch carve.  Decode
             * order stays left-then-right (the wire is pre-order) --
             * only buffer placement differs. */
            left_buf  = place_tail(out_buf, K_right, K_left, &scratch_top);
            right_buf = scratch_carve(&scratch_top, K_right);
        } else if (tail_ok) {
            right_buf = place_tail(out_buf, K_left, K_right, &scratch_top);
            left_buf  = scratch_carve(&scratch_top, K_left);
        } else {
            left_buf  = scratch_carve(&scratch_top, K_left);
            right_buf = scratch_carve(&scratch_top, K_right);
        }

        codec_decode_subtree(table, node->left,  K_left,
                              left_buf,  in_ptr, scratch_top, g_dec_inplace);
        codec_decode_subtree(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top, g_dec_inplace);

        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);
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
     * thread-local heap buffer so block size is a runtime parameter.
     * With carve or in-place placement on, double it for carve-bump
     * waste and add page + over-read slack (the exact 2N + budget
     * bound comes with the arena-shrink step). */
    dec_flag("PIVCO_DEC_CARVE",   &g_dec_carve);
    dec_flag("PIVCO_DEC_INPLACE", &g_dec_inplace);
    size_t need = (size_t)N * (PIVCO_MAX_CODE_LEN + 2);
    if (g_dec_carve || g_dec_inplace)
        need = 2 * need + SCRATCH_PAGE + MERGE_OVERREAD;
    uint8_t *scratch = decode_scratch_ensure(need);
    if (!scratch) return PIVCO_ERR_NULL;
    g_scratch_pad_left = PIVCO_SCRATCH_PAD_BUDGET;

    codec_decode_subtree(table, table->tree_root, N,
                          symbols, &ptr, scratch, /*tail_ok=*/0);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}
