/* lz4_split — pivco fork of LZ4-HC that emits 4 separate streams
 *   - literals       : verbatim source bytes from each literal run
 *   - tokens         : 1 byte per sequence (incl. the final lits-only seq)
 *   - offsets        : 2 bytes per regular sequence (LE uint16)
 *   - overflow       : variable bytes per overflow event
 *
 * The custom decoder consumes those 4 streams directly and produces
 * output without reconstructing the standard LZ4 wire format.
 *
 * See extras/lz4hc_split.c for the encoder (lightly-patched copy of
 * upstream lib/lz4hc.c) and extras/lz4_split_decode.c for the decoder. */

#ifndef LZ4_SPLIT_H
#define LZ4_SPLIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *literals; size_t lit_pos; size_t lit_cap;
    uint8_t *tokens;   size_t tok_pos; size_t tok_cap;
    uint8_t *offsets;  size_t off_pos; size_t off_cap;
    uint8_t *overflow; size_t ovf_pos; size_t ovf_cap;
    int      ok;       /* 0 on capacity exhaustion */
} lz4_split_ctx_t;

/* Encode `src` of size `src_size` into the 4 streams set up in `split`
 * (caller allocates buffers + caps).  Returns the size of the standard
 * LZ4 output (which is also produced into `throwaway_dst` and then
 * ignored).  On success, the 4 *_pos fields hold the split-stream
 * lengths and split->ok == 1.
 *
 * compression_level: 1..12 (matches upstream LZ4HC). */
int phsplit_LZ4_compress_HC_split(const char *src, int src_size,
                                   void *throwaway_dst, int throwaway_cap,
                                   int compression_level,
                                   lz4_split_ctx_t *split);

/* Decode the 4 streams back into a contiguous output buffer.
 * out_capacity must equal the original source size (we already know
 * it from the outer wire format).  Returns 0 on success, < 0 on error. */
int lz4_split_decompress(const uint8_t *literals, size_t literals_len,
                          const uint8_t *tokens,   size_t tokens_len,
                          const uint8_t *offsets,  size_t offsets_len,
                          const uint8_t *overflow, size_t overflow_len,
                          uint8_t *out, size_t out_size);

/* Trust-mode decoder: identical primitives, but ZERO bounds checks on
 * inputs.  Caller must guarantee the streams are well-formed and have
 * ≥64 B trailing pad.  Used only for benchmark diagnostics — never
 * call in production.  Returns 0 always. */
int lz4_split_decompress_trust(const uint8_t *literals,
                                const uint8_t *tokens, size_t tokens_len,
                                const uint8_t *offsets,
                                const uint8_t *overflow,
                                uint8_t *out, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* LZ4_SPLIT_H */
