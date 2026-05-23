/* pivco_huffman_wire.h — single source of truth for the per-node wire format.
 *
 * All backends MUST consume/produce the per-non-flat-internal-node wire
 * record through these helpers.  Previously each backend hand-rolled
 * the read/write of K_right header + FSE marker byte + bitmap bytes,
 * which led to silent drift (scalar+NEON added the FSE marker byte in
 * 2026-05-13, x86+AVX-512 didn't — broke scalar↔SSE cross-decoding).
 *
 * Wire format (v0.2+):
 *
 *   [optional K_right_header: uint16 LE, 2 bytes]   if kr_header_needed()
 *   [FSE marker byte:        uint8,    1 byte]    always
 *   [bitmap body]                                  marker == 0: raw n-bit
 *                                                  bitmap, ceil(n/8) bytes
 *                                                  marker != 0: 2-byte LE
 *                                                  fse_len + fse_len bytes
 *                                                  of FSE-compressed bytes
 *
 * Flat-subtree nodes do NOT use this header — they emit n·D packed bits
 * directly.  See pivco_huffman.h:flat_depth.
 *
 * Internal header, not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_WIRE_H
#define PIVCO_HUFFMAN_WIRE_H

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif

#include <stdint.h>
#include <string.h>

/* ---------- Encode side: reserve / commit slots ----------
 *
 * The encoder reserves the header slot(s) BEFORE knowing n_right, then
 * commits the value afterwards.  Returns pointer to where the K_right
 * uint16 should be written (NULL if no header was reserved). */
static inline uint8_t *wire_reserve_kr_header(const pivco_huffman_table_t *table,
                                               int16_t node_id,
                                               uint8_t **out_ptr)
{
    if (!kr_header_needed(table, node_id)) return NULL;
    uint8_t *slot = *out_ptr;
    *out_ptr += KR_HEADER_BYTES;
    return slot;
}

/* Write the K_right value into a previously-reserved slot.  No-op if
 * `slot` is NULL (header wasn't reserved for this node). */
static inline void wire_commit_kr_header(uint8_t *slot, int n_right)
{
    if (!slot) return;
    slot[0] = (uint8_t)(n_right & 0xFF);
    slot[1] = (uint8_t)((n_right >> 8) & 0xFF);
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

/* Skip the K_right header bytes, returning the value as an int.  If no
 * header is present for this node, returns -1.  (Top-down decoders
 * don't use the value; bottom-up ones do.) */
static inline int wire_read_kr_header(const pivco_huffman_table_t *table,
                                       int16_t node_id,
                                       const uint8_t **in_ptr)
{
    if (!kr_header_needed(table, node_id)) return -1;
    PROF_TIC();
    uint16_t v;
    memcpy(&v, *in_ptr, 2);
    *in_ptr += KR_HEADER_BYTES;
    PROF_TOC(PROF_WIRE_KR, 1);
    return (int)v;
}

/* Read the per-node bitmap body (marker + payload).  Returns a pointer
 * to the usable n-bit bitmap (either pointing into the input stream
 * for marker==0, or into the caller-provided `scratch` for the FSE
 * path).  Advances *in_ptr past the whole record.
 *
 * scratch must hold at least bitmap_bytes(n) bytes and stay live for
 * the entire span where the returned pointer is dereferenced. */
static inline const uint8_t *wire_read_bitmap(const uint8_t **in_ptr,
                                                int n,
                                                uint8_t *scratch)
{
    PROF_TIC();
    int nbytes = bitmap_bytes(n);
    uint8_t marker = **in_ptr;
    *in_ptr += 1;
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

#endif  /* PIVCO_HUFFMAN_WIRE_H */
