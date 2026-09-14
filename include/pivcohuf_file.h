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
 *     10     FLAGS (uint8, v0.9+) -- format feature bits:
 *              bits0-1  FLAT_LAYOUT (pivco_flat_layout_t): 0 = natural,
 *                       1 = hybrid vertical (512- then 128-lane blocks
 *                       + natural tail), 2 = 128-lane vertical only,
 *                       3 = reserved (rejected)
 *              bit2     QUAD_NODES: reserved, must be 0
 *              bits3..7 reserved, must be 0
 *            The decoder rejects any value or set bit it does not
 *            implement (BAD_VERSION), so reserved bits are assignable
 *            later without risking silent mis-decode.
 *              bits3..7 reserved, must be 0
 *     11-138 CODE_LENGTHS[256] packed as 4-bit nibbles, LSB first
 *            (symbol 2i in low nibble of byte i, symbol 2i+1 in high nibble)
 *     139... Concatenated per-block records:
 *               3 bytes ENCODED_LEN (uint24) -- payload length
 *               1 byte  BLOCK_FLAGS (uint8, v0.10+; 0 before):
 *                         bit0  NEW_TABLE: 128 bytes CODE_LENGTHS follow,
 *                               the table for this and the following blocks
 *                         bits1..7 reserved, must be 0
 *               [128 bytes CODE_LENGTHS when NEW_TABLE is set]
 *               ENCODED_LEN bytes encoded block (pivco-Huffman stream)
 *            When and how often the encoder switches tables is its own
 *            policy (see PIVCOHUF_SEGMENT_BYTES_DEFAULT); the decoder only
 *            follows the flags.
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
 *   v0.9 vs v0.8: inserts the FLAGS byte at body offset 10 (shifting the
 *   code-length nibbles and block records by one) and introduces the
 *   selectable flat-region layout (FLAGS bits0-1; default hybrid
 *   vertical, see pivco_cfg_t.flat_layout).  Decode still accepts v0.8
 *   streams: no FLAGS byte, flat regions natural.
 *
 *   v0.10 vs v0.9: the block record's 32-bit ENCODED_LEN becomes a
 *   24-bit length plus a BLOCK_FLAGS byte, whose NEW_TABLE bit lets a
 *   block carry its own code-length table (see above); the default entry
 *   points write a new table per PIVCOHUF_SEGMENT_BYTES_DEFAULT of input
 *   when it pays for itself.  A stream with no flagged block is
 *   byte-identical to v0.9, and v0.9 / v0.8 streams still decode.
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
#define PIVCOHUF_VERSION_MINOR  10
#define PIVCOHUF_HEADER_SIZE    26

/* BODY FLAGS byte (v0.9+).  Bits0-1 carry the pivco_flat_layout_t value
 * (3 is reserved); any bit outside the layout field (including the
 * reserved QUAD_NODES bit) makes the decoder return BAD_VERSION. */
#define PIVCOHUF_FLAGS_LAYOUT_MASK   0x03u  /* bits0-1: flat layout */
#define PIVCOHUF_FLAG_QUAD_NODES     0x04u  /* bit2: reserved, must be 0 */
/* Block record header: a 24-bit ENCODED_LEN and a BLOCK_FLAGS byte
 * (v0.10+; the byte was the always-zero top of a 32-bit length before). */
#define PIVCOHUF_BLOCK_LEN_MASK      0x00FFFFFFu
#define PIVCOHUF_BLOCK_FLAGS_SHIFT   24
#define PIVCOHUF_BLOCK_FLAG_NEW_TABLE 0x01u  /* 128-byte code-length table precedes the payload */

/* Input bytes per Huffman table for the entry points that take no
 * seg_blocks: one table per 128 KiB (rounded to whole blocks).  128 KiB
 * is the knee of the segment-size sweep: -1.7% (PH) / -1.1% (PHA) of output
 * over one table per file at ~5% / 0% compress cost and ~2% / -8%
 * decode cost, against -1.9% / -1.2% at 64 KiB for twice the table
 * builds.  Per-file streams stay available via pivcohuf_compress_seg
 * with seg_blocks == 0. */
#define PIVCOHUF_SEGMENT_BYTES_DEFAULT (128u * 1024u)

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
 * Uses the default block size (PIVCO_BLOCK_SIZE) and segment size. */
size_t pivcohuf_compress_bound(size_t in_len);

/* As pivcohuf_compress_bound, but for a specific block size.  Smaller blocks
 * carry more per-block overhead and need a larger bound, so callers of
 * pivcohuf_compress_blk must size the output buffer with this.  Assumes
 * the default segment size. */
size_t pivcohuf_compress_bound_blk(size_t in_len, size_t block_size);

/* As pivcohuf_compress_bound_blk, for a stream with a table every
 * seg_blocks blocks (0 = one table for the whole input). */
size_t pivcohuf_compress_bound_seg(size_t in_len, size_t block_size,
                                   size_t seg_blocks);

/* Blocks per segment for the default segment size at a given block size
 * (at least 1, at most 255). */
size_t pivcohuf_default_seg_blocks(size_t block_size);

/* Compress in[0..in_len) into out (capacity *out_len).  On success,
 * sets *out_len to the actual encoded length and returns PIVCOHUF_OK.
 * Plain Huffman (#PH). */
int pivcohuf_compress(const uint8_t *in, size_t in_len,
                      uint8_t *out, size_t *out_len);

/* As pivcohuf_compress, but `use_ans != 0` selects #PHA: per-block partition
 * bitmaps may be ANS(FSE)-coded with the static table schedule for a better
 * ratio on skewed data, at some decode cost (the transmitted-table
 * candidates, pivco_cfg_t.fse_nibble_enabled / fse_k1_enabled, are opt-in
 * through pivcohuf_compress_cfg).  Same wire format and decoder —
 * pivcohuf_decompress auto-detects the ANS-coded blocks, so pha and ph
 * streams decompress identically. */
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

/* As pivcohuf_compress_cfg, with the Huffman table rebuilt from the
 * input every seg_blocks blocks and transmitted when it pays for itself
 * (0 = one table for the whole input).  seg_blocks is at most 255.
 * Size the output with pivcohuf_compress_bound_seg.  The other compress
 * entry points use pivcohuf_default_seg_blocks(block_size). */
int pivcohuf_compress_seg(const uint8_t *in, size_t in_len,
                          uint8_t *out, size_t *out_len,
                          const pivco_cfg_t *cfg, size_t block_size,
                          size_t seg_blocks, pivcohuf_timing_t *timing);


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
