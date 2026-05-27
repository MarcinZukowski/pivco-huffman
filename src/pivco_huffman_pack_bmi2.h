/* pivco_huffman_pack_bmi2.h — BMI2 pext flat-subtree pack.
 *
 * The contiguous-wire pack (concatenate n D-bit codes LSB-first into one
 * bitstream) maps perfectly onto BMI2 `pext`: each code's D bits live at
 * [right_shift, right_shift+D) within its 16-bit codes_la lane, and pext
 * gathers the mask-selected bits of a 64-bit window into a contiguous low
 * field.  One pext packs 4 codes (a 4×u16 window); two cover a group of 8.
 *
 * Unlike the AVX-512 vector pack (widen u16->u64, sllv, cross-lane reduce),
 * this needs NO cross-lane reduction and keeps the contiguous wire format
 * (no transpose) — the fast path for short contiguous packs (cf. Lemire's
 * LittleIntPacker / TurboPFor, which are scalar BMI2 for short arrays).
 *
 * Shared by the x86 + AVX-512 primitive backends (both imply BMI2 on the
 * hosts we target; NEON has no pext and keeps its uint32-lane pack).
 * Not part of the public API.
 */
#ifndef PIVCO_HUFFMAN_PACK_BMI2_H
#define PIVCO_HUFFMAN_PACK_BMI2_H

#if defined(__BMI2__)
#include <stdint.h>
#include <string.h>
#include <immintrin.h>   /* _pext_u64 */

/* Pack n D-bit codes (D in 2..8) from codes_la into out, contiguous LSB-first.
 * Returns the number of codes packed (a multiple of 8); the caller's scalar
 * tail finishes the residual.  Writes exactly ceil(8D/8)=D bytes per group of
 * 8, byte-aligned, so consecutive groups tile without overlap or over-read. */
static inline int pack_dN_bmi2(uint8_t *out, const uint16_t *codes_la,
                               int n, int D, int right_shift)
{
    /* D bits at [right_shift, right_shift+D) replicated across the four
     * 16-bit lanes of a 64-bit window. */
    uint64_t field = (((uint64_t)1 << D) - 1) << right_shift;
    uint64_t mask  = field | (field << 16) | (field << 32) | (field << 48);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w0, w1;
        memcpy(&w0, codes_la + i,     8);   /* codes i   .. i+3 */
        memcpy(&w1, codes_la + i + 4, 8);   /* codes i+4 .. i+7 */
        uint64_t packed = _pext_u64(w0, mask)
                        | (_pext_u64(w1, mask) << (4 * D));   /* 8D bits */
        memcpy(out + ((i * D) >> 3), &packed, (size_t)D);     /* 8D bits = D bytes */
    }
    return i;
}
#endif /* __BMI2__ */

#endif /* PIVCO_HUFFMAN_PACK_BMI2_H */
