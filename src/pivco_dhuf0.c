/* Dynamic byte Huffman over partition bitmaps and flat regions -- wire
 * id PIVCO_FSE_DHUF0_ID ("dHuf0").  The region's bytes are coded with
 * huf0 (the Huffman coder vendored with FSE in ext/fse) under a table
 * fitted to this region's byte histogram.
 *
 * Payload: huf0's own 4-stream format, as HUF_compress4X_wksp writes it:
 * the FSE-coded weight header, a 6-byte jump table and four interleaved
 * bitstreams.  The decoder needs only the expected byte count.
 *
 * huf0's single-symbol (RLE) and "not compressible" outputs are not on
 * the wire; both are a fallback to raw (or to another candidate). */

#define HUF_STATIC_LINKING_ONLY
#include "huf.h"

#include "pivco_fse.h"

#include <stdint.h>
#include <string.h>

pivco_fse_status_t pivco_dhuf0_compress(const void *src, size_t src_len,
                                         void *dst, size_t dst_cap,
                                         size_t *out_len)
{
    if (src_len == 0 || src_len > HUF_BLOCKSIZE_MAX)
        return PIVCO_FSE_FALLBACK;
    uint32_t wksp[HUF_WORKSPACE_SIZE_U32];
    size_t r = HUF_compress4X_wksp(dst, dst_cap, src, src_len,
                                   255, HUF_TABLELOG_DEFAULT,
                                   wksp, sizeof(wksp));
    if (HUF_isError(r)) return PIVCO_FSE_FALLBACK;
    /* 0: not compressible; 1: single-symbol RLE, not a wire form. */
    if (r <= 1 || r >= src_len) return PIVCO_FSE_FALLBACK;
    *out_len = r;
    return PIVCO_FSE_OK;
}

pivco_fse_status_t pivco_dhuf0_decompress(const void *src, size_t src_len,
                                           void *dst, size_t dst_cap,
                                           size_t dst_expected,
                                           size_t *out_len)
{
    if (dst_cap < dst_expected) return PIVCO_FSE_ERR_DST_FULL;
    if (dst_expected == 0 || dst_expected > HUF_BLOCKSIZE_MAX)
        return PIVCO_FSE_ERR_BAD_INPUT;
    /* Sized for the double-symbol decoder at the largest table log, so
     * huf0 is free to pick either decoder for this region.  Only the
     * descriptor in dt[0] is set: the table read writes every entry the
     * decode then looks up, so the rest needs no zero fill. */
    HUF_DTable dt[HUF_DTABLE_SIZE(HUF_TABLELOG_MAX)];
    dt[0] = (HUF_DTable)((uint32_t)HUF_TABLELOG_MAX * 0x01000001u);
    uint32_t wksp[HUF_DECOMPRESS_WORKSPACE_SIZE_U32];
    size_t r = HUF_decompress4X_hufOnly_wksp(dt, dst, dst_expected,
                                             src, src_len,
                                             wksp, sizeof(wksp));
    if (HUF_isError(r) || r != dst_expected) return PIVCO_FSE_ERR_BAD_INPUT;
    *out_len = dst_expected;
    return PIVCO_FSE_OK;
}
