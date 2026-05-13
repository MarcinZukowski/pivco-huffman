/* pivcohuf file format (v0.1) -- standalone file-level codec built on
 * top of the pivco-huffman block primitives.
 *
 *   WIRE FORMAT (little-endian throughout)
 *
 *   HEADER (26 bytes, fixed across versions)
 *      0-7   "PIVCOHUF" magic
 *      8     MAJOR_VERSION (0)
 *      9     MINOR_VERSION (1)
 *     10-17  BODY_LENGTH (uint64) -- length of BODY in bytes
 *     18-21  BODY_CHECKSUM (XXH32 of BODY bytes, seed 0)
 *     22-25  HEADER_CHECKSUM (XXH32 of bytes 0..21, seed 0)
 *
 *   The HEADER_CHECKSUM specifically protects BODY_LENGTH: a corrupted
 *   length read from untrusted memory could cause OOB reads.  Verify
 *   header checksum BEFORE trusting BODY_LENGTH.
 *
 *   BODY (variable, length = HEADER.BODY_LENGTH)
 *      0-7   UNCOMPRESSED_SIZE (uint64) -- total bytes the decoder produces
 *      8-9   BLOCK_SIZE (uint16) -- codec block size in symbols; valid range
 *            [1024, 65535].  Decoder rejects if it can't handle this size.
 *     10-137 CODE_LENGTHS[256] packed as 4-bit nibbles, LSB first
 *            (symbol 2i in low nibble of byte i, symbol 2i+1 in high nibble)
 *     138...  Concatenated per-block records:
 *               4 bytes ENCODED_LEN (uint32)
 *               ENCODED_LEN bytes encoded block (pivco-Huffman stream)
 *
 *   The final block may have fewer than BLOCK_SIZE input symbols.  The
 *   encoder pads the input to BLOCK_SIZE with the file's first byte
 *   (always present in the alphabet); the decoder truncates output
 *   based on UNCOMPRESSED_SIZE.
 */
#ifndef PIVCOHUF_FILE_H
#define PIVCOHUF_FILE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PIVCOHUF_MAGIC          "PIVCOHUF"
#define PIVCOHUF_VERSION_MAJOR  0
#define PIVCOHUF_VERSION_MINOR  2  /* 0.2: per-node FSE marker + payload (see FSE-V0.md) */
#define PIVCOHUF_HEADER_SIZE    26

typedef enum {
    PIVCOHUF_OK = 0,
    PIVCOHUF_ERR_NULL = -1,
    PIVCOHUF_ERR_TOO_SHORT = -2,
    PIVCOHUF_ERR_BAD_MAGIC = -3,
    PIVCOHUF_ERR_BAD_VERSION = -4,
    PIVCOHUF_ERR_BAD_HEADER_CHECKSUM = -5,
    PIVCOHUF_ERR_BAD_BODY_CHECKSUM = -6,
    PIVCOHUF_ERR_BAD_BLOCK_SIZE = -7,
    PIVCOHUF_ERR_OUTPUT_TOO_SMALL = -8,
    PIVCOHUF_ERR_INTERNAL = -9,
} pivcohuf_status_t;

/* Worst-case output size given input size.  Overestimates; never lies low. */
size_t pivcohuf_compress_bound(size_t in_len);

/* Compress in[0..in_len) into out (capacity *out_len).  On success,
 * sets *out_len to the actual encoded length and returns PIVCOHUF_OK. */
int pivcohuf_compress(const uint8_t *in, size_t in_len,
                      uint8_t *out, size_t *out_len);

/* Decompress in[0..in_len) into out (capacity *out_len).  Verifies
 * header and body checksums.  On success, sets *out_len to the actual
 * uncompressed length and returns PIVCOHUF_OK. */
int pivcohuf_decompress(const uint8_t *in, size_t in_len,
                        uint8_t *out, size_t *out_len);

/* Peek the uncompressed size from a compressed stream's header.
 * Used to allocate the output buffer before calling decompress. */
int pivcohuf_peek_uncompressed_size(const uint8_t *in, size_t in_len,
                                     size_t *uncompressed_size);

#ifdef __cplusplus
}
#endif

#endif /* PIVCOHUF_FILE_H */
