/* pivco_huffman_wire.h — single source of truth for the per-node wire format.
 *
 * All backends MUST consume/produce the per-non-flat-internal-node wire
 * record through these helpers.  Previously each backend hand-rolled
 * the read/write of K_right header + FSE marker byte + bitmap bytes,
 * which led to silent drift (scalar+NEON added the FSE marker byte in
 * 2026-05-13, x86+AVX-512 didn't — broke scalar↔SSE cross-decoding).
 *
 * Wire format (v0.8): file data in decompression order, larger-K
 * child first.  The layout is an Euler walk of the tree — each node's
 * K_right split header lands at its pre-order position (on the way
 * down), its bitmap record at its post-order position (on the way up,
 * after its children's regions); its FSE marker is hoisted out to the
 * block-level marker prefix:
 *
 * Per-block header (once, at the very start of each encoded block):
 *   [block_N: uint16 LE, 2 bytes]
 *        symbol count N for this block; read before the walk (N up to 65535).
 *   [marker prefix]
 *        the FSE markers for all M non-flat internal nodes, hoisted out of the
 *        per-node records.  Empty when M == 0.  Otherwise an (M+1)-bit presence
 *        bitmap: bit 0 is the "any node FSE-coded" flag, bit i+1 set = node i
 *        is FSE-coded; when any is set, one marker value byte per set node bit
 *        follows.  When no node is FSE the whole prefix is that single byte
 *        (bit 0 == 0) -- all of #PH, and any #PHA block FSE never fired on.
 *        i indexes the nodes in the table's ascending pre-order
 *        (marker_positions); M and the order derive from the tree on both
 *        sides, so nothing else on the wire names them.
 *
 * Per non-flat internal node (marker-less: the marker is in the prefix above):
 *   [optional K_right: uint16 LE, 2 bytes]
 *        if kr_header_needed(); at node entry.
 *   [larger-K child region][smaller child region]
 *        recursively, same layout; larger first (strict >, ties left-first),
 *        keyed on the K_right header -- no extra bits.
 *   [bitmap body]
 *        marker == 0 (raw): n-bit bitmap, ceil(n/8) bytes.  marker != 0 (FSE):
 *        2-byte LE fse_len + fse_len bytes of FSE-compressed bytes.  The marker
 *        is looked up from the prefix by node id.
 *
 * The marker byte is [xor_flag:1][table_id:7].
 * table_id == 1..PIVCO_FSE_NUM_TABLES
 *   selects a pre-built static table (xor_flag says
 *   whether the encoder bit-inverted the bitmap first, so those tables
 *   always see "0 is the frequent bit").
 * table_id == PIVCO_FSE_NIBBLE_ID
 *   selects the nibble table: the payload
 *   starts with an FSE_writeNCount table description fitted to this
 *   bitmap's nibble histogram, followed by the coded nibbles (low nibble
 *   of each raw byte first); xor_flag is always 0 there.
 * table_id == PIVCO_FSE_K1_ID
 *   selects the k=1 bit-context table: the payload is one recipe byte
 *   (two 4-bit grid indices for P(bit | previous bit)) followed by the
 *   tANS bits, coded from a prebuilt catalog; xor_flag is always 0.
 *
 * All forms are decoded by pivco_fse_decompress(), which dispatches on
 * table_id.
 *
 * The order above is exactly the order the BU decoder consumes bytes.
 * It needs both child counts up front to size the children's buffers
 * - and forward parsing of variable-size regions requires the sizing
 * information in prefix position anyway — but it consumes a node's
 * bitmap only at merge time, after both children are decoded.  So the
 * stream is read strictly forward, each byte touched once, and the
 * decoder's L1 working set is a moving window.
 *
 * The larger-K child goes first so the decoder meets each node's
 * dominant half while the smaller sibling's buffer is still empty —
 * the decoder can overlap the larger child's working scratch with
 * that hole, which is what lets the scratch arena stay near N (see
 * the decode tree walk in pivco_huffman_codec.c).  Both sides key the
 * order on the K_right header already on the wire, so it costs no
 * bits.
 *
 * The K_right header occupies one slot per recursion site into a
 * non-leaf child (kr_header_needed()): leaf-only nodes (BOTH_LEAVES,
 * LEAF_LEFT's leaf side) carry no count, and empty subtrees (n == 0)
 * emit nothing at all.
 *
 * Flat-subtree nodes carry no K_right (they have no children, so pre-
 * and post-order coincide), but they DO carry the same marker + body
 * record as an internal node's bitmap:
 *
 *   [FSE marker byte: uint8, 1 byte]  always
 *   [region body]                     marker == 0: n·D packed bits,
 *                                     ceil(n·D/8) bytes
 *                                     marker != 0: 2-byte LE fse_len +
 *                                     fse_len payload bytes
 *
 * Only the nibble table is ever used there (marker ==
 * PIVCO_FSE_NIBBLE_ID): a flat region has no partition skew for the
 * static schedule to key on.  Coding it matters because a depth-D flat
 * subtree gives all 2^D of its symbols the same code length by
 * construction — Huffman models their real frequencies not at all, and
 * on literal streams that is where most of the residual redundancy is.
 * See wire_read_flat_region below and codec_fse_try.
 *
 * The RAW form's bit layout follows table->flat_layout — natural,
 * hybrid vertical, or 128-only vertical (see pivco_huffman_vertical.h);
 * the region byte size is the same in every layout.  The FSE form is
 * ALWAYS natural, whatever the table says, because the vertical layouts
 * gather codes at lane stride 16 and that destroys the adjacency the
 * nibble table feeds on (worth 0.30% over the corpus, 1.69% on
 * x-ray.serial).  The marker byte therefore selects the unpack kernel,
 * not table->flat_layout — see wire_read_flat_region and
 * codec_fse_try.  Vertical keeps its decode speed on the raw
 * path, which is where it matters.  See pivco_huffman.h:flat_depth.
 *
 * Internal header, not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_WIRE_H
#define PIVCO_HUFFMAN_WIRE_H

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_check.h"
#include "pivco_prof.h"
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PIVCO_BLOCK_N_BYTES 2  /* per-block N header: uint16 little-endian */

/* ---------- Per-block N header ---------- */

/* Encode: write the block's symbol count N as the first 2 bytes of the
 * encoded stream.  N <= 65535 (the existing PIVCO_BLOCK_SIZE of 8192/4096
 * leaves plenty of headroom; uint16 caps any future variable-block work
 * at the same 65535 limit). */
static inline void wire_write_block_n(uint8_t *out_ptr, int n)
{
    out_ptr[0] = (uint8_t)(n & 0xFF);
    out_ptr[1] = (uint8_t)((n >> 8) & 0xFF);
}

/* Decode: read the block's symbol count N from the first 2 bytes and
 * advance *in_ptr. */
static inline int wire_read_block_n(const uint8_t **in_ptr)
{
    uint16_t v;
    memcpy(&v, *in_ptr, 2);
    *in_ptr += PIVCO_BLOCK_N_BYTES;
    return (int)v;
}

/* ---------- Encode side: K_right header ----------
 *
 * The encoder's partition runs before anything is emitted for the
 * node, so the header value is known and written directly at node
 * entry.  No-op when the node carries no header (kr_header_needed()). */
static inline void wire_write_kr_header(const pivco_table_t *table,
                                         int16_t node_id,
                                         uint8_t **out_ptr, int n_right)
{
    if (!kr_header_needed(table, node_id)) return;
    (*out_ptr)[0] = (uint8_t)(n_right & 0xFF);
    (*out_ptr)[1] = (uint8_t)((n_right >> 8) & 0xFF);
    *out_ptr += KR_HEADER_BYTES;
}

/* Note: the FSE marker byte + bitmap (or FSE payload) is emitted by
 * the backend's `prim_encode_node` primitive, not by a helper here.
 * Backends that attempt FSE-coding of the bitmap need to make that
 * decision after building the raw bitmap, which is intrinsically
 * backend-specific; threading a wire-helper through that flow would
 * be more complexity than win.  The wire FORMAT — 1 byte marker
 * followed by raw bitmap (marker == 0) or [fse_len:u16][fse_payload]
 * (marker != 0) — is still authoritative here in the header doc, and
 * `wire_read_bitmap` below is the corresponding decoder. */

/* ---------- Decode side ---------- */

/* Read the K_right header at node entry.  Every decode-side call site
 * dispatches on node_type first (LEAF_LEFT / INTERNAL_FULL), where the
 * header is present by construction, so kr_header_needed() -- three
 * dependent tree loads re-deriving a statically known truth -- is only
 * consulted in debug builds.  (The encoder's write side still uses it:
 * its walk visits header-less nodes too.) */
static inline int wire_read_kr_header(const pivco_table_t *table,
                                       int16_t node_id,
                                       const uint8_t **in_ptr)
{
    PIVCO_CHECK_DEBUG(kr_header_needed(table, node_id));
    (void)table; (void)node_id;
    PROF_TIC();
    uint16_t v;
    memcpy(&v, *in_ptr, 2);
    *in_ptr += KR_HEADER_BYTES;
    PROF_TOC(PROF_WIRE_KR, 1);
    return (int)v;
}

/* Read the per-node bitmap body (payload only).  `marker` is this node's
 * marker, looked up by the caller from the block's marker prefix (0 = raw,
 * else FSE table id, high bit = XOR flip) -- it is no longer inline in the
 * region.  Returns a pointer to the usable n-bit bitmap (into the input
 * stream for marker==0, or into `scratch` for the FSE path).  Advances
 * *in_ptr past the payload.
 *
 * scratch must hold at least bitmap_bytes(n) bytes and stay live for
 * the entire span where the returned pointer is dereferenced. */
static inline const uint8_t *wire_read_bitmap(const uint8_t **in_ptr,
                                                int n,
                                                uint8_t *scratch,
                                                uint8_t marker)
{
    PROF_TIC();
    int nbytes = bitmap_bytes(n);
    if (marker == 0) {
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        PROF_TOC(PROF_WIRE_BITMAP_RAW, n);
        return bm;
    }
#ifdef PIVCO_HAS_FSE
    int t_id = marker & 0x7F;
    int xor_flag = (marker >> 7) & 1;
    uint16_t fse_len;
    memcpy(&fse_len, *in_ptr, 2);
    *in_ptr += 2;
    size_t out_len = 0;
    (void)pivco_fse_decompress(t_id, *in_ptr, fse_len,
                                scratch, (size_t)nbytes,
                                (size_t)nbytes, &out_len);
    *in_ptr += fse_len;
    if (xor_flag) pivco_fse_flip_bits(scratch, (size_t)nbytes);
    PROF_TOC(PROF_WIRE_BITMAP_FSE, n);
    return scratch;
#else
    /* FSE not built but stream uses it — best-effort fallback.  The
     * caller will produce wrong output; the file codec will catch the
     * mismatch.  We don't fault, just advance and return zeros. */
    (void)scratch;
    *in_ptr += nbytes;
    PROF_TOC(PROF_WIRE_BITMAP_FSE, n);
    return *in_ptr - nbytes;
#endif
}

/* Slack allocated past an FSE-decoded flat region: the flat merges load
 * their source in full vectors, so they may read (never write) past the
 * region's end.  Comfortably above every backend's
 * PIVCO_PRIM_MERGE_OVERREAD (max 16); static-asserted in the codec. */
#define PIVCO_FLAT_FSE_SLACK 128

/* Read a flat-subtree region: [marker][n*D packed bits] when marker == 0,
 * or [marker][fse_len:u16 LE][payload] for a transmitted table (nibble
 * or k=1).
 *
 * Returns a pointer to `nbytes` usable packed bytes -- straight into the
 * input stream for the raw form (the common case, no copy), or into a
 * freshly malloc'd buffer for the FSE form, in which case *owned is set
 * and the caller must free it once the merge has run.
 *
 * *fse_coded reports which form was read, and with it which bit layout
 * the bytes are in: the raw form uses table->flat_layout, the FSE form
 * is ALWAYS PIVCO_FLAT_NATURAL (the encoder packs it that way so the
 * nibble table sees adjacent codes sharing a byte -- see
 * codec_fse_try).  Callers must pick the unpack kernel from it,
 * not from the table.
 *
 * *owned and *fse_coded are always written (*owned NULL when nothing was
 * allocated).  Returns NULL only on allocation failure. */
static inline const uint8_t *wire_read_flat_region(const uint8_t **in_ptr,
                                                    int nbytes,
                                                    uint8_t **owned,
                                                    int *fse_coded)
{
    *owned = NULL;
    *fse_coded = 0;
    uint8_t marker = **in_ptr;
    *in_ptr += 1;
    if (marker == 0) {
        const uint8_t *body = *in_ptr;
        *in_ptr += nbytes;
        return body;
    }
    *fse_coded = 1;
    uint16_t fse_len;
    memcpy(&fse_len, *in_ptr, 2);
    *in_ptr += 2;
#ifdef PIVCO_HAS_FSE
    uint8_t *buf = (uint8_t *)malloc((size_t)nbytes + PIVCO_FLAT_FSE_SLACK);
    if (!buf) { *in_ptr += fse_len; return NULL; }
    memset(buf + nbytes, 0, PIVCO_FLAT_FSE_SLACK);
    size_t out_len = 0;
    (void)pivco_fse_decompress(marker & 0x7F, *in_ptr, fse_len,
                                buf, (size_t)nbytes, (size_t)nbytes, &out_len);
    *in_ptr += fse_len;
    *owned = buf;
    return buf;
#else
    /* FSE not built but the stream uses it -- same best-effort fallback
     * as wire_read_bitmap: advance and hand back zeros. */
    *in_ptr += fse_len;
    uint8_t *buf = (uint8_t *)calloc((size_t)nbytes + PIVCO_FLAT_FSE_SLACK, 1);
    if (!buf) return NULL;   /* *owned stays NULL; caller sees short output */
    *owned = buf;
    return buf;
#endif
}

#endif  /* PIVCO_HUFFMAN_WIRE_H */
