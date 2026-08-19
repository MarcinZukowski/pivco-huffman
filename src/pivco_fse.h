/* Thin wrapper around Yann Collet's FSE library for pivco-huffman's
 * per-node partition-bitmap compression path.  See docs/FSE-V0.md.
 *
 * Owns PIVCO_FSE_NUM_TABLES pre-built CTable + DTable globals (one per
 * frequent-bit probability on a linear 0.50..0.99 / 0.01 schedule),
 * populated lazily on first use from the normalized counts in
 * pivco_fse_tables.h.
 *
 * On top of those static tables there is one extra table id,
 * PIVCO_FSE_DYNAMIC_ID, that means "dynamic nibble table": instead of a
 * pre-built byte-alphabet distribution, the bitmap's bytes are split
 * into 4-bit nibbles, an FSE table is fitted to *that* bitmap's nibble
 * histogram, and the table description is written into the payload
 * ahead of the coded nibbles.  It costs a header (~15-30 bytes) but
 * adapts to bitmaps the fixed schedule models badly.
 *
 * Decoupled from the FSE library types so the FSE includes stay out
 * of the rest of the codec. */

#ifndef PIVCO_FSE_H
#define PIVCO_FSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PIVCO_FSE_OK            =  0,
    PIVCO_FSE_FALLBACK      =  1,  /* compressed >= raw; caller should emit raw */
    PIVCO_FSE_ERR_BAD_TABLE = -1,
    PIVCO_FSE_ERR_INTERNAL  = -2,
    PIVCO_FSE_ERR_DST_FULL  = -3,
    PIVCO_FSE_ERR_BAD_INPUT = -4,
} pivco_fse_status_t;

/* Table id meaning "dynamic nibble table" (see the file header).  It
 * sits one past the last static table id, so it still fits the 7 bits
 * the wire marker byte gives the table id.  Static-asserted against
 * PIVCO_FSE_NUM_TABLES in pivco_fse.c. */
#define PIVCO_FSE_DYNAMIC_ID 51

/* Nibble alphabet parameters for the dynamic path.  A 16-symbol
 * alphabet does not need a big table, and a smaller tableLog also
 * shrinks the FSE_writeNCount header we have to pay for per bitmap. */
#define PIVCO_FSE_NIB_MAX_SYMBOL 15
#ifndef PIVCO_FSE_NIB_TABLELOG
#define PIVCO_FSE_NIB_TABLELOG   10
#endif

/* Idempotent.  Safe to call multiple times; first call builds the
 * CTables + DTables; subsequent calls are no-ops. */
void pivco_fse_init(void);

/* Select the table index (1..25) whose tabulated frequency is the
 * largest value <= p_major.  Returns 0 if p_major is below the
 * smallest tabulated frequency (caller emits raw bitmap).
 *
 * p_major is the empirical frequency of whichever bit (0 or 1) is
 * the majority in the bitmap to be encoded.  Must be in [0.5, 1.0]. */
int pivco_fse_select_table(double p_major);

/* Compress src[0..src_len) into dst (capacity dst_cap).
 * On PIVCO_FSE_OK: *out_len holds the compressed length.
 * On PIVCO_FSE_FALLBACK: FSE-compressed output was >= src_len; the
 *   caller should emit the raw bitmap instead.  *out_len is undefined.
 * Otherwise: error.  *out_len is undefined.
 *
 * table_id must be in [1, PIVCO_FSE_NUM_TABLES] (via
 * pivco_fse_select_table) or PIVCO_FSE_DYNAMIC_ID, in which case the
 * call is forwarded to pivco_fse_compress_dynamic(). */
pivco_fse_status_t pivco_fse_compress(int table_id,
                                       const void *src, size_t src_len,
                                       void *dst, size_t dst_cap,
                                       size_t *out_len);

/* Decompress src[0..src_len) into dst (capacity dst_cap, expected size
 * is dst_expected).  On PIVCO_FSE_OK, *out_len == dst_expected.
 * table_id == PIVCO_FSE_DYNAMIC_ID forwards to
 * pivco_fse_decompress_dynamic(). */
pivco_fse_status_t pivco_fse_decompress(int table_id,
                                         const void *src, size_t src_len,
                                         void *dst, size_t dst_cap,
                                         size_t dst_expected,
                                         size_t *out_len);

/* ---- Dynamic nibble path ----
 *
 * Splits every src byte into two 4-bit symbols (low nibble first:
 * nib[2i] = src[i] & 0xF, nib[2i+1] = src[i] >> 4), fits an FSE table
 * to the resulting 16-symbol histogram and emits
 * [FSE_writeNCount table description][coded nibbles].  The payload is
 * self-describing: the decoder only needs the expected byte count.
 *
 * Returns PIVCO_FSE_FALLBACK when the result would not be smaller than
 * src_len (header included) or when FSE reports the nibbles as
 * incompressible / single-symbol RLE (the RLE form is not on the wire).
 *
 * dst_cap should be at least 2 * src_len + 64: FSE's own "did this
 * help?" test is against the *nibble* count, so an unhelpful table can
 * legitimately produce up to ~2 * src_len bytes before we reject it. */
pivco_fse_status_t pivco_fse_compress_dynamic(const void *src, size_t src_len,
                                               void *dst, size_t dst_cap,
                                               size_t *out_len);

pivco_fse_status_t pivco_fse_decompress_dynamic(const void *src, size_t src_len,
                                                 void *dst, size_t dst_cap,
                                                 size_t dst_expected,
                                                 size_t *out_len);

/* Helper: byte-wise XOR-flip a buffer (all 1s become 0s and vice
 * versa).  Used when the right side is the majority -- we flip the
 * bitmap so the encoder always sees the "0 is frequent" distribution
 * that the tables are tuned for. */
void pivco_fse_flip_bits(uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif  /* PIVCO_FSE_H */
