/* pivcohuf file format -- standalone file-level codec built on
 * top of the pivco-huffman block primitives.
 *
 *   WIRE FORMAT (little-endian throughout)
 *
 *   HEADER (26 bytes, fixed across versions)
 *      0-7   "PIVCOHUF" magic
 *      8     MAJOR_VERSION (PIVCOHUF_VERSION_MAJOR below)
 *      9     MINOR_VERSION (PIVCOHUF_VERSION_MINOR below)
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
 *     138... Concatenated per-block records:
 *               4 bytes ENCODED_LEN (uint32)
 *               ENCODED_LEN bytes encoded block (pivco-Huffman stream)
 *
 *   v0.4 vs v0.3: drops the within-tier ORDERING section.  The decode tree
 *   is fully determined by the code lengths (within-tier order is symbol-
 *   value), so nothing beyond the lengths is transmitted.  v0.3 streams are
 *   not readable by v0.4 decoders.
 *
 *   v0.5: per-block uint16 N header for arbitrary block sizes (see
 *   pivco_huffman_wire.h).  This wire change actually shipped on main
 *   without a MINOR bump; it is recorded here so the version line is
 *   honest, and 0.5 is folded into the 0.6 gate below rather than
 *   emitted on its own.
 *
 *   v0.6 vs v0.4/0.5: the FSE (PHA) bitmap path now uses the wide 8-cursor
 *   format for any bitmap length, not just multiples of 8.  Streams with
 *   n % 8 == 0 bitmaps are byte-identical to the prior format; those with
 *   unaligned FSE bitmaps switch format, so earlier decoders mis-decode
 *   unaligned FSE streams -- hence the minor bump.
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
#include "pivco_huffman.h"   /* pivco_cfg_t */

#ifdef __cplusplus
extern "C" {
#endif

#define PIVCOHUF_MAGIC          "PIVCOHUF"
#define PIVCOHUF_VERSION_MAJOR  0
#define PIVCOHUF_VERSION_MINOR  8
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

/* Worst-case output size given input size.  Overestimates; never lies low.
 * Uses the default block size (PIVCO_BLOCK_SIZE). */
size_t pivcohuf_compress_bound(size_t in_len);

/* As pivcohuf_compress_bound, but for a specific block size.  Smaller blocks
 * carry more per-block overhead and need a larger bound, so callers of
 * pivcohuf_compress_blk must size the output buffer with this. */
size_t pivcohuf_compress_bound_blk(size_t in_len, size_t block_size);

/* Compress in[0..in_len) into out (capacity *out_len).  On success,
 * sets *out_len to the actual encoded length and returns PIVCOHUF_OK.
 * Plain Huffman (#PH). */
int pivcohuf_compress(const uint8_t *in, size_t in_len,
                      uint8_t *out, size_t *out_len);

/* As pivcohuf_compress, but `use_ans != 0` selects #PHA: per-block partition
 * bitmaps may be ANS(FSE)-coded for a better ratio on skewed data, at some
 * decode cost.  Same wire format and decoder — pivcohuf_decompress auto-detects
 * the ANS-coded blocks, so pha and ph streams decompress identically. */
int pivcohuf_compress_ex(const uint8_t *in, size_t in_len,
                         uint8_t *out, size_t *out_len, int use_ans);

/* Decompress in[0..in_len) into out (capacity *out_len).  Verifies
 * header and body checksums.  On success, sets *out_len to the actual
 * uncompressed length and returns PIVCOHUF_OK. */
int pivcohuf_decompress(const uint8_t *in, size_t in_len,
                        uint8_t *out, size_t *out_len);

/* Peek the uncompressed size from a compressed stream's header.
 * Used to allocate the output buffer before calling decompress. */
int pivcohuf_peek_uncompressed_size(const uint8_t *in, size_t in_len,
                                     size_t *uncompressed_size);

/* Per-phase wall-clock breakdown (nanoseconds) filled by the *_timed
 * variants.  Phases not relevant to the call stay 0 (e.g. freq_ns on
 * decompress).  freq_ns and build_ns are distinct: a caller who already
 * has symbol frequencies can skip the histogram (freq_ns) and build the
 * table directly via the block API in pivco_huffman.h.  Timing is coarse
 * (never inside hot inner loops); pass NULL to skip it entirely. */
typedef struct {
    double freq_ns;    /* build frequencies (symbol histogram) -- compress only */
    double build_ns;   /* build codes/tree (Huffman table) */
    double codec_ns;   /* encode (compress) or decode (decompress) block loop */
    double malloc_ns;  /* internal scratch allocations */
} pivcohuf_timing_t;

/* Full-parameter compress: cfg (NULL = defaults; fse_enabled selects
 * #PHA), explicit block size, optional timing. */
int pivcohuf_compress_cfg(const uint8_t *in, size_t in_len,
                          uint8_t *out, size_t *out_len,
                          const pivco_cfg_t *cfg, size_t block_size,
                          pivcohuf_timing_t *timing);


/* As pivcohuf_compress_ex / pivcohuf_decompress, but fill *timing (nullable)
 * with the per-phase breakdown above.  The struct is zeroed on entry. */
int pivcohuf_compress_timed(const uint8_t *in, size_t in_len,
                            uint8_t *out, size_t *out_len,
                            int use_ans, pivcohuf_timing_t *timing);

/* As pivcohuf_compress_timed, but with a caller-chosen block size (symbol
 * count per block, 1..PIVCO_WIRE_MAX_N).  The block size is recorded in the
 * stream header, so pivcohuf_decompress reads it back automatically — no
 * matching build flag required.  Larger blocks amortise per-block table/tree
 * reload (a big decode win on small-L1 x86; see issue #2).  Size the output
 * buffer with pivcohuf_compress_bound_blk(in_len, block_size).  timing may be
 * NULL. */
int pivcohuf_compress_blk(const uint8_t *in, size_t in_len,
                          uint8_t *out, size_t *out_len,
                          int use_ans, size_t block_size,
                          pivcohuf_timing_t *timing);
int pivcohuf_decompress_timed(const uint8_t *in, size_t in_len,
                              uint8_t *out, size_t *out_len,
                              pivcohuf_timing_t *timing);

#ifdef __cplusplus
}
#endif

#endif /* PIVCOHUF_FILE_H */
