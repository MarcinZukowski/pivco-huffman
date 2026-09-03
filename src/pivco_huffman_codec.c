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

/* Decode scratch arena, owned by the caller's pivco_decoder_t (the
 * pivco_scratch_t behind its `internal` pointer).  Preallocated at
 * context create for PIVCO_WIRE_MAX_N and reused across blocks (the
 * ensure below still grows it if ever needed; never shrinks) -- block
 * size stays a pure runtime parameter (the wire N header carries the
 * per-block count). */


/* The returned base is aligned to a 16 KiB boundary plus 64*39 bytes —
 * see DECODE_SCRATCH_ALIGN / DECODE_SCRATCH_SHIFT in
 * pivco_huffman_common.h for the page-split rationale. */
static uint8_t *decode_scratch_ensure(pivco_scratch_t *sc, size_t need)
{
    need += DECODE_SCRATCH_ALIGN + DECODE_SCRATCH_SHIFT;
    if (need > sc->dec_cap) {
        uint8_t *p = (uint8_t *)realloc(sc->dec, need);
        if (!p) return NULL;
        sc->dec     = p;
        sc->dec_cap = need;
    }
    uintptr_t p = (uintptr_t)sc->dec;
    p = ((p + DECODE_SCRATCH_ALIGN - 1) & ~(DECODE_SCRATCH_ALIGN - 1))
        + DECODE_SCRATCH_SHIFT;
    return (uint8_t *)p;
}

/* MERGE_OVERREAD: the SIMD merges load their source buffers in
 * full-vector chunks, so they may read (never write) up to this many
 * bytes past a source's end.  Every buffer the decode walk hands to a
 * merge therefore needs this much trailing slack inside the arena; the
 * caller's `symbols` buffer, which guarantees none, is only ever a
 * merge destination (writes are exact). */
#define MERGE_OVERREAD ((size_t)PIVCO_PRIM_MERGE_OVERREAD)
_Static_assert(PIVCO_PRIM_MERGE_OVERREAD <= PIVCO_FLAT_FSE_SLACK,
               "wire_read_flat_region's slack must cover the merges' overread");

/* Growable encode scratch arena, owned by the caller's pivco_encoder_t
 * (the pivco_scratch_t behind its `internal` pointer): holds the
 * per-block ranks buffer + the tree-walk's right-half recursion
 * scratch, preallocated at context create and reused across blocks
 * (never shrinks), so a block-loop encode doesn't malloc/free per
 * block.  The former thread_local stopgap (and its thread-death leak)
 * is gone -- ownership and lifetime are the context's. */


static uint8_t *encode_scratch_ensure(pivco_scratch_t *sc, size_t need)
{
    if (need > sc->enc_cap) {
        uint8_t *p = (uint8_t *)realloc(sc->enc, need);
        if (!p) return NULL;
        sc->enc     = p;
        sc->enc_cap = need;
    }
    return sc->enc;
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
#  define CODEC_ENCODE_ENTRY pivco_encode_scalar
#  define CODEC_DECODE_ENTRY pivco_decode_scalar
#elif defined(PIVCO_BACKEND_NEON)
#  define CODEC_ENCODE_ENTRY pivco_encode_neon
#  define CODEC_DECODE_ENTRY pivco_decode_bu_neon
#elif defined(PIVCO_BACKEND_X86)
#  define CODEC_ENCODE_ENTRY pivco_encode_x86
#  define CODEC_DECODE_ENTRY pivco_decode_bu_x86
#elif defined(PIVCO_BACKEND_AVX512)
#  define CODEC_ENCODE_ENTRY pivco_encode_avx512
#  define CODEC_DECODE_ENTRY pivco_decode_bu_avx512
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

#ifdef PIVCO_HAS_FSE
/* Overwrite an already-emitted raw region with its FSE form:
 * marker := [xor_flag:1][t_id:7], region := [fse_len:u16 LE][payload],
 * and the wire cursor pulled back to one past the payload.  Callers have
 * already decided the record is worth writing; this only does the write
 * and the stats. */
static inline void codec_fse_commit(uint8_t *marker_slot, uint8_t *body,
                                     int nbytes, int t_id, int xor_flag,
                                     const uint8_t *payload, size_t fse_len,
                                     uint8_t **out_ptr)
{
    *marker_slot = (uint8_t)((xor_flag ? 0x80 : 0) | t_id);
    uint8_t *p = body;
    *p++ = (uint8_t)( fse_len       & 0xFF);
    *p++ = (uint8_t)((fse_len >> 8) & 0xFF);
    memcpy(p, payload, fse_len);
    *out_ptr = p + fse_len;

    g_pivco_fse_commit  [t_id]++;
    g_pivco_fse_bytes_in [t_id] += (uint64_t)nbytes;
    g_pivco_fse_bytes_out[t_id] += (uint64_t)(fse_len + 3);
}
#endif

/* Arch-agnostic FSE attempt on a freshly-built raw bitmap.
 *
 * Two candidates are tried and the smaller payload wins:
 *
 *   static  — one of the PIVCO_FSE_NUM_TABLES pre-built byte-alphabet
 *             tables, picked from the partition skew.  Zero header
 *             cost, but only applies to bitmaps skewed enough
 *             (p_major >= PIVCO_FSE_MIN_THRESHOLD) for the schedule to
 *             model them.
 *   dynamic — PIVCO_FSE_DYNAMIC_ID: the bitmap's bytes split into
 *             nibbles and coded with a table fitted to *this* bitmap's
 *             nibble histogram, table description included in the
 *             payload.  Applies to any bitmap, and pays for its own
 *             header inside the length it reports — so comparing raw
 *             payload lengths already accounts for it.
 *
 * Inputs:
 *   fse_on       — table->fse_enabled; 0 disables both candidates
 *   dyn_on       — table->fse_dynamic; 0 leaves only the static one
 *                  (decode is unaffected either way — it dispatches on
 *                  the wire marker and always knows both forms)
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
static inline void codec_maybe_fse_attempt(int fse_on, int dyn_on,
                                            uint8_t *marker_slot,
                                            uint8_t *bm, int nbytes,
                                            int n, int n_left, int n_right,
                                            int depth, uint8_t **out_ptr)
{
#ifdef PIVCO_HAS_FSE
    if (!fse_on) return;

    int     best_id  = 0;
    int     best_xor = 0;
    size_t  best_len = 0;
    /* Either surviving candidate is strictly shorter than the raw region
     * (that is each one's commit gate), so nbytes + 64 holds it. */
    uint8_t best_out[(size_t)nbytes + 64];

    /* ---- candidate 1: static table ----
     * Needs a skewed bitmap; the fixed schedule bottoms out at
     * pivco_fse_freq[1] and modelling a near-50/50 bitmap with it costs
     * more than it saves.  Also size-gated: below MIN_BITMAP_BYTES the
     * attempt is not worth the encoder time. */
    int static_id = (nbytes >= PIVCO_FSE_MIN_BITMAP_BYTES &&
                     n > 0 &&
                     (double)((n_left >= n_right) ? n_left : n_right) /
                       (double)n >= PIVCO_FSE_MIN_THRESHOLD)
                      ? pivco_fse_select_table(
                            (double)((n_left >= n_right) ? n_left : n_right) /
                              (double)n)
                      : 0;
    if (static_id >= 1) {
        PIVCO_CHECK(static_id < PIVCO_FSE_STATS_SLOTS);  /* guards the g_pivco_fse_* indexing */
        /* Flip when the right side is the majority: the tables are all
         * tuned for "0 is the frequent bit". */
        int xor_flag = (n_right > n_left);
        uint8_t scratch[(size_t)nbytes + 16];
        if (xor_flag) {
            for (int i = 0; i < nbytes; i++) scratch[i] = (uint8_t)~bm[i];
        } else {
            memcpy(scratch, bm, (size_t)nbytes);
        }
        size_t len = 0;
        pivco_fse_status_t rc = pivco_fse_compress(static_id, scratch,
                                                    (size_t)nbytes,
                                                    best_out, sizeof(best_out),
                                                    &len);
        g_pivco_fse_attempt[static_id]++;
        /* Per-codeword commit gate (see docs/FSE-V0.md):
         *   raw: every codeword through this node costs (depth + 1) bits
         *   fse: (depth + (len + 2 wire-prefix) * 8 / n) bits
         * Commit iff (depth + fse_frac) <= MIN_RATIO * (depth + 1).  This
         * is a SPEED gate, not a size one: stock-table FSE decode is slow
         * enough that a marginal byte saving is a bad trade. */
        double fse_frac = (double)(len + 2) * 8.0 / (double)n;
        double codeword_ratio = ((double)depth + fse_frac) /
                                  ((double)depth + 1.0);
        if (rc == PIVCO_FSE_OK && codeword_ratio <= (double)PIVCO_FSE_MIN_RATIO) {
            best_id  = static_id;
            best_xor = xor_flag;
            best_len = len;
        }
    }

    /* ---- candidate 2: dynamic nibble table ----
     * No skew or size precondition: a bitmap the fixed byte-alphabet
     * schedule cannot model may still have a lopsided nibble histogram.
     * Coded from `bm` directly, so no xor flip, and the reported length
     * already includes the table description.
     *
     * Gated purely on size (see codec_try_fse_dynamic) -- deliberately
     * NOT the static path's per-codeword speed gate.  Applying that gate
     * here was why this codec lost to plain FSE on literal streams: it
     * declines most deep nodes, where the ratio tends to 1 with depth no
     * matter how many bytes the coding actually saves. */
    if (dyn_on) {
        /* Staging is 2*nbytes+64, not nbytes+64: FSE's internal
         * incompressibility test is against the nibble count (2*nbytes),
         * so it can hand back a payload larger than the raw region,
         * which pivco_fse_compress_dynamic then rejects. */
        uint8_t dyn_out[2 * (size_t)nbytes + 64];
        size_t dyn_len = 0;
        pivco_fse_status_t rc = pivco_fse_compress_dynamic(bm, (size_t)nbytes,
                                                            dyn_out,
                                                            sizeof(dyn_out),
                                                            &dyn_len);
        g_pivco_fse_attempt[PIVCO_FSE_DYNAMIC_ID]++;
        if (rc == PIVCO_FSE_OK && dyn_len + 2 < (size_t)nbytes &&
            (best_id < 1 || dyn_len < best_len)) {
            best_id  = PIVCO_FSE_DYNAMIC_ID;
            best_xor = 0;
            best_len = dyn_len;
            memcpy(best_out, dyn_out, dyn_len);
        }
    }

    if (best_id < 1) {
        g_pivco_fse_commit[0]++;     /* slot 0 = attempted, rejected */
        return;
    }
    codec_fse_commit(marker_slot, bm, nbytes, best_id, best_xor,
                      best_out, best_len, out_ptr);
#else
    (void)fse_on; (void)dyn_on; (void)marker_slot; (void)bm; (void)nbytes;
    (void)n; (void)n_left; (void)n_right; (void)depth; (void)out_ptr;
#endif
}

/* Is a flat region worth a dynamic-nibble attempt?  Split out from the
 * attempt itself because the caller has to know the answer *before* it
 * packs: an FSE'd region is packed naturally, a raw one in the
 * configured layout (see codec_maybe_fse_flat). */
static inline int codec_flat_fse_eligible(const pivco_table_t *table, int nbytes)
{
#ifdef PIVCO_HAS_FSE
    /* Below ~8 bytes a 16-symbol NCount cannot pay for itself. */
    return table->fse_enabled && table->fse_dynamic && nbytes >= 8;
#else
    (void)table; (void)nbytes;
    return 0;
#endif
}

/* Dynamic-nibble attempt on a flat-subtree region.  Returns 1 on commit.
 *
 * Flat regions are the other half of the source codec's design that the
 * first port left out: a depth-D flat subtree gives all 2^D of its
 * symbols the same code length *by construction*, so plain Huffman
 * models their true frequencies not at all.  Nibble-FSE over the packed
 * n*D bits is what recovers that -- and on literal streams, where the
 * bulk of the data sits in flat subtrees, it is where most of the win
 * is.  Same [marker][fse_len:u16][payload] record as an internal node's
 * bitmap; only the dynamic table is tried (there is no partition skew
 * for the static schedule to key on).
 *
 * `body` must be packed in PIVCO_FLAT_NATURAL, NOT table->flat_layout.
 * When D does not divide 4 a nibble straddles code boundaries, so its
 * value encodes a tuple of neighbouring codes -- and under natural
 * packing those are *adjacent source symbols*, which makes runs collapse
 * the tuple distribution onto a few values.  That is order-1 structure
 * an order-0 nibble coder gets to capture for free.  The vertical
 * layouts gather at lane stride 16, which breaks the runs and reverts
 * the tuple distribution to the product of the marginals: measured cost
 * 0.30% over the corpus, 1.69% on x-ray.serial, and exactly 0 at D == 4
 * (a nibble is then one whole code, so grouping cannot matter) and for
 * i.i.d. input.  The marker byte tells the decoder which layout to
 * unpack with, so the raw path keeps the configured layout and its
 * decode speed.
 *
 * No-op when PIVCO_HAS_FSE is not defined -- the marker byte is still
 * written by the caller, so the wire layout does not depend on it. */
static inline int codec_maybe_fse_flat(uint8_t *marker_slot,
                                        uint8_t *body, int nbytes,
                                        uint8_t **out_ptr)
{
#ifdef PIVCO_HAS_FSE
    /* Heap, not a VLA: a depth-8 flat root over a full block packs to
     * 32 KiB, so the 2*nbytes staging would be a 64 KiB stack frame. */
    size_t cap = 2 * (size_t)nbytes + 64;
    uint8_t *dyn_out = (uint8_t *)malloc(cap);
    if (!dyn_out) return 0;
    size_t dyn_len = 0;
    pivco_fse_status_t rc = pivco_fse_compress_dynamic(body, (size_t)nbytes,
                                                        dyn_out, cap, &dyn_len);
    g_pivco_fse_attempt[PIVCO_FSE_DYNAMIC_ID]++;
    /* Raw record costs 1 + nbytes, coded costs 3 + dyn_len. */
    if (rc != PIVCO_FSE_OK || dyn_len + 2 >= (size_t)nbytes) {
        g_pivco_fse_commit[0]++;
        free(dyn_out);
        return 0;
    }
    codec_fse_commit(marker_slot, body, nbytes, PIVCO_FSE_DYNAMIC_ID, 0,
                      dyn_out, dyn_len, out_ptr);
    free(dyn_out);
    return 1;
#else
    (void)marker_slot; (void)body; (void)nbytes; (void)out_ptr;
    return 0;
#endif
}

static void codec_encode_node(const pivco_table_t *table,
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

    /* Flat-subtree path: a marker byte then n*D packed bits (no K_right).
     * The marker is written unconditionally, exactly like an internal
     * node's, so the wire layout does not depend on the encoder's FSE
     * settings; codec_maybe_fse_flat may then replace the packed bits
     * with an FSE record in place. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        uint8_t base = table->flat_base_rank[node_id];
        int total_bytes = (n * D + 7) >> 3;
        uint8_t *marker_slot = *out_ptr;
        *marker_slot = 0;
        *out_ptr += 1;
        uint8_t *body = *out_ptr;
        *out_ptr += total_bytes;

        /* Pack naturally first when an FSE attempt is on the table: the
         * nibble coder wants adjacent codes sharing a byte (see
         * codec_maybe_fse_flat).  If the attempt does not commit we
         * re-pack in the configured layout over the same bytes, which
         * costs an extra pack only on regions FSE declined. */
        int try_fse = codec_flat_fse_eligible(table, total_bytes);
        PROF_TIC();
        if (try_fse) prim_enc_pack_dN_natural(ranks, n, D, base, body);
        else         prim_enc_pack_dN(ranks, n, D, base, body,
                                      table->flat_layout);
        PROF_TOC(PROF_ENC_FLAT, n);

        if (try_fse) {
            if (codec_maybe_fse_flat(marker_slot, body, total_bytes, out_ptr))
                return;                     /* committed, natural-packed */
            if (table->flat_layout != PIVCO_FLAT_NATURAL) {
                PROF_TIC();
                prim_enc_pack_dN(ranks, n, D, base, body, table->flat_layout);
                PROF_TOC(PROF_ENC_FLAT, n);
            }
        }
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

    /* Emit the larger-K child's region first: the decoder can then
     * decode it into scratch that the smaller, not-yet-decoded
     * sibling's buffer overlaps (hole-reuse), shrinking the arena
     * high-water.  The two rank buffers (ranks=left, tmp=right) and
     * the shared deeper scratch tmp+n_right are mutually disjoint, so
     * the call order is free.  A leaf child emits nothing, so this
     * only changes the stream at INTERNAL_FULL nodes — exactly where
     * the decoder reorders. */
    if (n_right > n_left) {
        codec_encode_node(table, node->right, tmp,   n_right, depth + 1,
                           out_ptr, tmp + n_right);
        codec_encode_node(table, node->left,  ranks, n_left,  depth + 1,
                           out_ptr, tmp + n_right);
    } else {
        codec_encode_node(table, node->left,  ranks, n_left,  depth + 1,
                           out_ptr, tmp + n_right);
        codec_encode_node(table, node->right, tmp,   n_right, depth + 1,
                           out_ptr, tmp + n_right);
    }

    /* Emit this node's record: marker + staged bitmap.  The FSE attempt
     * may rewrite marker+bm in place with [fse_len][payload] and pull
     * *out_ptr back to the payload end.  No-op otherwise. */
    uint8_t *marker_slot = *out_ptr;
    *marker_slot = 0;
    *out_ptr += 1;
    uint8_t *bm = *out_ptr;
    memcpy(bm, bm_stage, (size_t)nbytes);
    *out_ptr += nbytes;
    codec_maybe_fse_attempt(table->fse_enabled, table->fse_dynamic,
                             marker_slot, bm, nbytes,
                             n, n_left, n_right, depth, out_ptr);
}

int CODEC_ENCODE_ENTRY(pivco_encoder_t *enc_ctx, const pivco_table_t *table, const uint8_t *symbols, size_t n, uint8_t *out, size_t *out_len)
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
    uint8_t *ranks = encode_scratch_ensure((pivco_scratch_t *)enc_ctx->internal,
                                       ranks_capacity + tmp_capacity);
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

/* ---------- Bottom-up decode tree walk (ping-pong scratch) ---------- *
 *
 * Each call decodes a subtree's K symbols into out[0,K).  Internal
 * nodes recurse into their children, then merge per the node's bitmap.
 * The flat-subtree fast path bypasses recursion entirely.
 *
 * The wire is in decompression order — larger-K child first — so the
 * input cursor is consumed strictly forward and each record is loaded
 * exactly where it is used: the K_right header at node entry, the
 * children's regions during their recursion, the node's bitmap right
 * before its merge.
 *
 * Scratch placement is a two-buffer ping-pong (out, tmp):
 *
 *   - the larger child decodes in place into out's tail
 *     out[K_small, K): safe under the merge, whose write cursor can
 *     never overtake its tail-side read cursor (by the time it writes
 *     out[i] it has consumed at least i - K_small tail bytes);
 *   - the smaller child decodes into tmp[0, K_small), and its own
 *     recursion uses out's still-empty prefix out[0, K_small) as its
 *     partner — the pair (tmp, out-prefix) ping-pongs down the
 *     smaller-child spine.
 *
 * The caller guarantees tmp capacity floor(K/2): a node's smaller child
 * is at most floor(K/2), and everything a smaller child's subtree puts
 * in its partner stays inside out[0, K_small).  The walk's footprint is
 * therefore out[0,K), plus at most floor(K/2) bytes past tmp (the
 * largest smaller-child on the larger-child spine), plus MERGE_OVERREAD
 * read slack past whichever region ends last.
 *
 * Dispatch on node_type, computed at build-table time (by children's
 * leafness — a leaf child's symbol goes straight into the parent's
 * merge, so the walk never recurses into a leaf):
 *
 *   INTERNAL_FLAT   — packed-bits flat decode into out
 *   BOTH_LEAVES     — both children leaves, merge_cst_cst directly
 *   LEAF_LEFT       — left child leaf, recurse right (in place, into
 *                     out's tail), merge_cst_vec
 *   INTERNAL_FULL   — both children internal: larger child in place,
 *                     smaller via the ping-pong partner, merge_vec_vec */

static void codec_decode_subtree(const pivco_table_t *table,
                                   int16_t node_id, int K,
                                   uint8_t *out, uint8_t *tmp,
                                   const uint8_t **in_ptr)
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
        uint8_t *owned = NULL;
        int fse_coded = 0;
        const uint8_t *bm = wire_read_flat_region(in_ptr, total_bytes,
                                                   &owned, &fse_coded);
        if (!bm) return;   /* allocation failure; caller sees short output */
        const uint8_t *c2s =
            &table->flat_code_to_sym[table->flat_offset[node_id]];
        /* An FSE-coded region is always natural-packed, whatever the
         * table's configured layout (see codec_maybe_fse_flat). */
        if (fse_coded) prim_merge_flat_natural(out, K, bm, D, c2s);
        else           prim_merge_flat(out, K, bm, D, c2s, table->flat_layout);
        free(owned);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        /* No K_right header (kr_header_needed returns false). */
        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);
        prim_merge_cst_cst(bm, K,
                           (uint8_t)table->tree[node->left].symbol,
                           (uint8_t)table->tree[node->right].symbol,
                           out);
        return;
    }

    case PIVCO_NODE_LEAF_LEFT: {
        /* One internal child (right); the leaf contributes the K_left
         * symbols the merge fills into out's prefix.  The right child
         * decodes in place into out's tail, whatever its share of K. */
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t *right_buf = out + (K - K_right);
        codec_decode_subtree(table, node->right, K_right,
                              right_buf, tmp, in_ptr);

        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);
        prim_merge_cst_vec(bm, K,
                           (uint8_t)table->tree[node->left].symbol,
                           right_buf, out);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        /* Both children internal.  Stream order == decode order ==
         * larger first (strict >, ties left-first — must match the
         * encoder). */
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        int K_left  = K - K_right;
        uint8_t *left_buf, *right_buf;
        if (K_right > K_left) {
            right_buf = out + K_left;            /* larger, in place    */
            left_buf  = tmp;                     /* smaller, ping-pong  */
            codec_decode_subtree(table, node->right, K_right,
                                  right_buf, tmp, in_ptr);
            codec_decode_subtree(table, node->left,  K_left,
                                  left_buf,  out, in_ptr);
        } else {
            left_buf  = out + K_right;           /* larger, in place    */
            right_buf = tmp;                     /* smaller, ping-pong  */
            codec_decode_subtree(table, node->left,  K_left,
                                  left_buf,  tmp, in_ptr);
            codec_decode_subtree(table, node->right, K_right,
                                  right_buf, out, in_ptr);
        }

        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch);
        prim_merge_vec_vec(bm, K, left_buf, right_buf, out);
        return;
    }
    }
}

int CODEC_DECODE_ENTRY(pivco_decoder_t *dec_ctx, const pivco_table_t *table, const uint8_t *in, size_t in_len, uint8_t *symbols, size_t *consumed)
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

    /* Flat root: the whole tree is one packed-bits region, decoded
     * straight into symbols (exact writes, no merge, no scratch). */
    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_INTERNAL_FLAT) {
        int D = table->flat_depth[table->tree_root];
        int total_bytes = (N * D + 7) >> 3;
        uint8_t *owned = NULL;
        int fse_coded = 0;
        const uint8_t *bm = wire_read_flat_region(&ptr, total_bytes,
                                                   &owned, &fse_coded);
        if (!bm) return PIVCO_ERR_CORRUPT;   /* allocation failure */
        const uint8_t *root_c2s =
            &table->flat_code_to_sym[table->flat_offset[table->tree_root]];
        if (fse_coded) prim_merge_flat_natural(symbols, N, bm, D, root_c2s);
        else           prim_merge_flat(symbols, N, bm, D, root_c2s,
                                       table->flat_layout);
        free(owned);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Remaining root shapes recurse.  The ping-pong walk parks children
     * in its out buffer's tail and prefix and the merges read their
     * sources with up-to-MERGE_OVERREAD slack past the end — guarantees
     * the caller's `symbols` doesn't offer.  So the root's children
     * decode into the context's arena (preallocated, reused
     * across blocks) and only the root's own merge, whose writes are
     * exact, targets `symbols`.
     *
     * Arena bound: (MAX_CODE_LEN+2)·N, loose.  The ping-pong walk's
     * true high-water is under 1.5·N — but only on valid streams, so
     * that figure is not a safe allocation target.  Against invalid
     * data (a bitmap popcount disagreeing with the declared child
     * counts) the floor is 2·N plus kernel-overtouch pad: with splits
     * bounded (0 <= KR <= K — not currently checked), every recursive
     * (out, tmp) placement satisfies out + 2K <= 2·N and tmp + K <=
     * 2·N as arena offsets, which caps hostile cursor excursions too. */
    size_t need = (size_t)N * (PIVCO_MAX_CODE_LEN + 2) + MERGE_OVERREAD;

    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_LEAF_LEFT) {
        /* One internal child: it decodes at the arena base with the
         * space after it as ping-pong partner; the cst_vec merge fills
         * symbols. */
        int K_right = wire_read_kr_header(table, table->tree_root, &ptr);
        uint8_t *scratch = decode_scratch_ensure((pivco_scratch_t *)dec_ctx->internal, need);
        if (!scratch) return PIVCO_ERR_NULL;
        codec_decode_subtree(table, root->right, K_right,
                              scratch, scratch + K_right, &ptr);

        uint8_t bm_scratch[(size_t)bitmap_bytes(N) + 16];
        const uint8_t *bm = wire_read_bitmap(&ptr, N, bm_scratch);
        prim_merge_cst_vec(bm, N,
                           (uint8_t)table->tree[root->left].symbol,
                           scratch, symbols);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* INTERNAL_FULL root — hybrid hole-reuse placement.  Both children
     * decode into the arena's first N bytes, [larger | smaller], and
     * the final merge writes symbols.  The larger child (first on the
     * wire) uses the smaller sibling's still-empty slot as its
     * ping-pong partner — hole-reuse; when a smaller-child on its spine
     * outgrows that slot, the partner writes spill past N into fresh
     * arena.  The smaller root child then decodes into its slot with a
     * fresh partner beyond N. */
    int K_right = wire_read_kr_header(table, table->tree_root, &ptr);
    int K_left  = N - K_right;
    uint8_t *scratch = decode_scratch_ensure((pivco_scratch_t *)dec_ctx->internal, need);
    if (!scratch) return PIVCO_ERR_NULL;

    uint8_t *buf_left, *buf_right;
    if (K_right > K_left) {                  /* right larger -> first on the wire */
        buf_right = scratch;
        buf_left  = scratch + K_right;
        codec_decode_subtree(table, root->right, K_right,
                              buf_right, /*tmp=*/buf_left, &ptr);
        codec_decode_subtree(table, root->left,  K_left,
                              buf_left,  /*tmp=*/scratch + N, &ptr);
    } else {
        buf_left  = scratch;
        buf_right = scratch + K_left;
        codec_decode_subtree(table, root->left,  K_left,
                              buf_left,  /*tmp=*/buf_right, &ptr);
        codec_decode_subtree(table, root->right, K_right,
                              buf_right, /*tmp=*/scratch + N, &ptr);
    }

    uint8_t bm_scratch[(size_t)bitmap_bytes(N) + 16];
    const uint8_t *bm = wire_read_bitmap(&ptr, N, bm_scratch);
    prim_merge_vec_vec(bm, N, buf_left, buf_right, symbols);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}
