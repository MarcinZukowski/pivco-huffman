/* lz4_split_decode — custom 4-stream LZ4 decoder.
 *
 * Two-loop design mirroring upstream LZ4_decompress_safe:
 *
 *  - FAST LOOP: one hoisted output-side check at the loop head
 *    (`out_p + FASTLOOP_SAFE <= out_end`).  Input streams are
 *    NOT bounds-checked per sequence; the caller's slack contract
 *    guarantees the few overshooting reads (16-byte literal slurp,
 *    wildCopy32) land in trailing pad.  Two more checks fire inside
 *    the body for the long-literal / long-match cases that would
 *    cross the output tail — each jumps to a per-sequence safe
 *    completion via `tail_literal:` / `tail_match:`.
 *
 *  - SAFE LOOP: handles the last ~64 bytes of output, plus the
 *    initial sequences when the input is too small to bother with
 *    the fast loop.  Every cursor is bounds-checked.
 *
 * CALLER SLACK CONTRACT (required for the fast path):
 *    - out:      ≥64 B writable trailing pad
 *    - literals: ≥32 B readable trailing pad
 *    (tokens/offsets are consumed exactly, no overshoot.  overflow is
 *    consumed bytewise; well-formed data terminates each 255-run with
 *    a < 255 byte, so we won't read past the end.)
 *
 * The bench harness allocates `+64` on every buffer, so this holds.
 */

#include "lz4_split.h"

#include <stdint.h>
#include <string.h>

#define MINMATCH       4
#define FASTLOOP_SAFE 64

static const unsigned inc32table[8] = {0, 1, 2,  1,  0,  4, 4, 4};
static const int      dec64table[8] = {0, 0, 0, -1, -4,  1, 2, 3};

static inline void wildCopy8(uint8_t *dst, const uint8_t *src, uint8_t *end)
{
    do { memcpy(dst, src, 8); dst += 8; src += 8; } while (dst < end);
}

static inline void wildCopy32(uint8_t *dst, const uint8_t *src, uint8_t *end)
{
    do {
        memcpy(dst,      src,      16);
        memcpy(dst + 16, src + 16, 16);
        dst += 32;
        src += 32;
    } while (dst < end);
}

static inline void match_copy_small_offset(uint8_t *dst, const uint8_t *src,
                                            uint8_t *end, size_t offset)
{
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
    src += inc32table[offset];
    memcpy(dst + 4, src, 4);
    src -= dec64table[offset];
    dst += 8;
    if (dst < end) wildCopy8(dst, src, end);
}

int lz4_split_decompress(const uint8_t *literals, size_t literals_len,
                          const uint8_t *tokens,   size_t tokens_len,
                          const uint8_t *offsets,  size_t offsets_len,
                          const uint8_t *overflow, size_t overflow_len,
                          uint8_t *out, size_t out_size)
{
    const uint8_t *lit_p = literals;
    const uint8_t *tok_p = tokens;
    const uint8_t *off_p = offsets;
    const uint8_t *ovf_p = overflow;
    const uint8_t *const lit_end = literals + literals_len;
    const uint8_t *const tok_end = tokens   + tokens_len;
    const uint8_t *const off_end = offsets  + offsets_len;
    const uint8_t *const ovf_end = overflow + overflow_len;

    uint8_t       *out_p   = out;
    uint8_t * const out_end = out + out_size;

    /* Hoisted per-iter scratch (gotos cross block scopes). */
    uint8_t  token     = 0;
    size_t   lit_len   = 0;
    size_t   match_len = 0;
    uint16_t offset    = 0;

    if (out_size < 128 || literals_len < 32) goto safe_loop;

    /* === FAST LOOP =========================================================
     * Only checks the output tail; input cursors are trusted to fit within
     * their caller-provided slack. */
    while (out_p <= out_end - FASTLOOP_SAFE && tok_p < tok_end) {
        token   = *tok_p++;
        lit_len = (size_t)(token >> 4);

        /* --- literal copy --- */
        if (lit_len == 15) {
            while (*ovf_p == 255) { lit_len += 255; ovf_p++; }
            lit_len += *ovf_p++;
            if (out_p + lit_len > out_end - 32) goto tail_literal;
            wildCopy32(out_p, lit_p, out_p + lit_len);
        } else {
            /* short lit (0..14): single 16-byte slurp, harmless overshoot. */
            memcpy(out_p, lit_p, 16);
        }
        out_p += lit_len;
        lit_p += lit_len;

        if (out_p >= out_end) goto done;

        /* --- offset + match length --- */
        offset = (uint16_t)off_p[0] | ((uint16_t)off_p[1] << 8);
        off_p += 2;

        match_len = (size_t)(token & 0xf);
        if (match_len == 15) {
            while (*ovf_p == 255) { match_len += 255; ovf_p++; }
            match_len += *ovf_p++;
        }
        match_len += MINMATCH;

        if (out_p + match_len > out_end - FASTLOOP_SAFE) goto tail_match;

        /* --- match copy --- */
        {
            const uint8_t *match = out_p - offset;
            if (match_len <= 18 && offset >= 8) {
                memcpy(out_p,      match,      8);
                memcpy(out_p + 8,  match + 8,  8);
                memcpy(out_p + 16, match + 16, 2);
            } else if (offset >= 16) {
                wildCopy32(out_p, match, out_p + match_len);
            } else if (offset >= 8) {
                wildCopy8(out_p, match, out_p + match_len);
            } else {
                match_copy_small_offset(out_p, match, out_p + match_len, offset);
            }
        }
        out_p += match_len;
    }
    goto safe_loop;

    /* ----- Tail completion paths (one-shot, then fall to safe_loop) ----- */

tail_literal:
    if (out_p + lit_len > out_end || lit_p + lit_len > lit_end) return -2;
    memcpy(out_p, lit_p, lit_len);
    out_p += lit_len;
    lit_p += lit_len;
    if (out_p >= out_end) goto done;

    if (off_p + 2 > off_end) return -3;
    offset = (uint16_t)off_p[0] | ((uint16_t)off_p[1] << 8);
    off_p += 2;

    match_len = (size_t)(token & 0xf);
    if (match_len == 15) {
        while (ovf_p < ovf_end && *ovf_p == 255) { match_len += 255; ovf_p++; }
        if (ovf_p >= ovf_end) return -5;
        match_len += *ovf_p++;
    }
    match_len += MINMATCH;
    /* fall through */

tail_match:
    if (offset == 0 || (size_t)offset > (size_t)(out_p - out)) return -4;
    if (out_p + match_len > out_end) return -6;
    {
        const uint8_t *m = out_p - offset;
        if ((size_t)(out_p - m) < match_len) {
            for (size_t i = 0; i < match_len; i++) out_p[i] = m[i];
        } else {
            memcpy(out_p, m, match_len);
        }
    }
    out_p += match_len;
    /* fall through to safe loop for remaining sequences */

    /* === SAFE LOOP ========================================================= */
safe_loop:
    while (tok_p < tok_end && out_p < out_end) {
        token = *tok_p++;

        lit_len = (size_t)(token >> 4);
        if (lit_len == 15) {
            while (ovf_p < ovf_end && *ovf_p == 255) {
                lit_len += 255;
                ovf_p++;
            }
            if (ovf_p >= ovf_end) return -1;
            lit_len += *ovf_p++;
        }
        if (out_p + lit_len > out_end || lit_p + lit_len > lit_end) return -2;
        memcpy(out_p, lit_p, lit_len);
        out_p += lit_len;
        lit_p += lit_len;

        if (out_p >= out_end) break;

        if (off_p + 2 > off_end) return -3;
        offset = (uint16_t)off_p[0] | ((uint16_t)off_p[1] << 8);
        off_p += 2;
        if (offset == 0 || (size_t)offset > (size_t)(out_p - out)) return -4;

        match_len = (size_t)(token & 0xf);
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

        {
            const uint8_t *match = out_p - offset;
            if ((size_t)(out_p - match) < match_len) {
                for (size_t i = 0; i < match_len; i++) out_p[i] = match[i];
            } else {
                memcpy(out_p, match, match_len);
            }
        }
        out_p += match_len;
    }

done:
    if (out_p != out_end) return -7;
    return 0;
}

/* =========================================================================
 *  TRUST-MODE DECODER — diagnostic only.
 *
 *  Same copy primitives, same control flow as the fast path, but with
 *  ZERO bounds checks anywhere.  Used to measure the upper bound on
 *  4-stream decode speed.  Caller must guarantee every input is
 *  well-formed and that all buffers have ≥64 B of slack.
 * ========================================================================= */
int lz4_split_decompress_trust(const uint8_t *literals,
                                const uint8_t *tokens, size_t tokens_len,
                                const uint8_t *offsets,
                                const uint8_t *overflow,
                                uint8_t *out, size_t out_size)
{
    const uint8_t *lit_p = literals;
    const uint8_t *tok_p = tokens;
    const uint8_t *off_p = offsets;
    const uint8_t *ovf_p = overflow;
    const uint8_t *const tok_end = tokens + tokens_len;

    uint8_t       *out_p   = out;
    uint8_t * const out_end = out + out_size;

    while (tok_p < tok_end && out_p < out_end) {
        uint8_t token = *tok_p++;

        size_t lit_len = (size_t)(token >> 4);
        if (lit_len == 15) {
            while (*ovf_p == 255) { lit_len += 255; ovf_p++; }
            lit_len += *ovf_p++;
        }
        if (lit_len <= 14) {
            memcpy(out_p, lit_p, 16);
        } else {
            wildCopy32(out_p, lit_p, out_p + lit_len);
        }
        out_p += lit_len;
        lit_p += lit_len;

        if (out_p >= out_end) break;

        uint16_t offset = (uint16_t)off_p[0] | ((uint16_t)off_p[1] << 8);
        off_p += 2;

        size_t match_len = (size_t)(token & 0xf);
        if (match_len == 15) {
            while (*ovf_p == 255) { match_len += 255; ovf_p++; }
            match_len += *ovf_p++;
        }
        match_len += MINMATCH;

        const uint8_t *match = out_p - offset;
        if (match_len <= 18 && offset >= 8) {
            memcpy(out_p,      match,      8);
            memcpy(out_p + 8,  match + 8,  8);
            memcpy(out_p + 16, match + 16, 2);
        } else if (offset >= 16) {
            wildCopy32(out_p, match, out_p + match_len);
        } else if (offset >= 8) {
            wildCopy8(out_p, match, out_p + match_len);
        } else {
            match_copy_small_offset(out_p, match, out_p + match_len, offset);
        }
        out_p += match_len;
    }

    return 0;
}
