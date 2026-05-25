/* Thin wrapper around Yann Collet's FSE library for pivco-huffman's
 * per-node partition-bitmap compression path.  See docs/FSE-V0.md.
 *
 * Owns PIVCO_FSE_NUM_TABLES pre-built CTable + DTable globals (one per
 * frequent-bit probability on a linear 0.50..0.99 / 0.01 schedule),
 * populated lazily on first use from the normalized counts in
 * pivco_fse_tables.h.
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
 * table_id must be in [1, PIVCO_FSE_NUM_TABLES] (via pivco_fse_select_table). */
pivco_fse_status_t pivco_fse_compress(int table_id,
                                       const void *src, size_t src_len,
                                       void *dst, size_t dst_cap,
                                       size_t *out_len);

/* Decompress src[0..src_len) into dst (capacity dst_cap, expected size
 * is dst_expected).  On PIVCO_FSE_OK, *out_len == dst_expected. */
pivco_fse_status_t pivco_fse_decompress(int table_id,
                                         const void *src, size_t src_len,
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
