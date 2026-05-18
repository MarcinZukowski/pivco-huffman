/* lz4_split_decode — custom 4-stream LZ4 decoder.
 *
 * Reads (literals, tokens, offsets, overflow) and writes the reconstructed
 * source buffer.  The match-copy follows LZ4's wildcopy-back semantics
 * (offset < match_length is allowed; copies byte-by-byte in that case).
 *
 * Termination: when output buffer is fully written, we stop.  The
 * final token in the tokens stream is a literals-only sequence (no
 * offset).
 *
 * This decoder does NOT reconstruct the standard LZ4 wire format.  It
 * walks the 4 streams in lockstep, advancing each as it consumes from
 * it.  This is the actual perf-relevant path for the LZ4+ph hybrid. */

#include "lz4_split.h"

#include <string.h>

#define MINMATCH 4
#define RUN_MASK 15
#define ML_MASK  15
#define ML_BITS  4

/* Wildcopy: copy `n` bytes from `src` to `dst`, byte-by-byte when
 * src+n > dst (overlapping match, e.g., RLE pattern via offset=1).
 * Otherwise use word-sized copies for speed. */
static inline void wildcopy_back(uint8_t *dst, const uint8_t *src, size_t n)
{
    /* Distance from src to dst — if < n we MUST copy byte-by-byte. */
    if ((uintptr_t)(dst - src) < (uintptr_t)n) {
        for (size_t i = 0; i < n; i++) dst[i] = src[i];
    } else {
        memcpy(dst, src, n);
    }
}

int lz4_split_decompress(const uint8_t *literals, size_t literals_len,
                          const uint8_t *tokens,   size_t tokens_len,
                          const uint8_t *offsets,  size_t offsets_len,
                          const uint8_t *overflow, size_t overflow_len,
                          uint8_t *out, size_t out_size)
{
    (void)literals_len; (void)offsets_len;  /* used only for asserts */

    const uint8_t *lit_p = literals;
    const uint8_t *tok_p = tokens;
    const uint8_t *off_p = offsets;
    const uint8_t *ovf_p = overflow;
    const uint8_t *tok_end = tokens   + tokens_len;
    const uint8_t *ovf_end = overflow + overflow_len;

    uint8_t       *out_p   = out;
    uint8_t * const out_end = out + out_size;

    while (tok_p < tok_end && out_p < out_end) {
        uint8_t token = *tok_p++;

        /* Decode literal-run length. */
        size_t lit_len = (size_t)(token >> 4);
        if (lit_len == 15) {
            while (ovf_p < ovf_end && *ovf_p == 255) {
                lit_len += 255;
                ovf_p++;
            }
            if (ovf_p >= ovf_end) return -1;
            lit_len += *ovf_p++;
        }

        /* Copy literals into output. */
        if (out_p + lit_len > out_end) return -2;
        memcpy(out_p, lit_p, lit_len);
        out_p += lit_len;
        lit_p += lit_len;

        /* End-of-output check: this was the final literals-only token. */
        if (out_p >= out_end) break;

        /* Read offset (2 bytes LE). */
        if (off_p + 2 > offsets + offsets_len) return -3;
        uint16_t offset = (uint16_t)off_p[0] | ((uint16_t)off_p[1] << 8);
        off_p += 2;
        /* Offset bounds check: must be in (0, output-so-far].  Note:
         * `offset` is u16 (max 65535) but `out_p - out` can be much
         * larger than 64 KB on long files — a naïve `(uint16_t)`
         * cast truncates and produces false positives past the
         * 64-KB boundary.  Compare in size_t. */
        if (offset == 0 || (size_t)offset > (size_t)(out_p - out)) return -4;

        /* Decode match length (token lo-nibble + 4, plus overflow). */
        size_t match_len = (size_t)(token & 0xf);
        if (match_len == 15) {
            while (ovf_p < ovf_end && *ovf_p == 255) {
                match_len += 255;
                ovf_p++;
            }
            if (ovf_p >= ovf_end) return -5;
            match_len += *ovf_p++;
        }
        match_len += MINMATCH;

        if (out_p + match_len > out_end) return -6;
        wildcopy_back(out_p, out_p - offset, match_len);
        out_p += match_len;
    }

    if (out_p != out_end) return -7;
    return 0;
}
