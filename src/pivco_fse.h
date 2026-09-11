/* Thin wrapper around Yann Collet's FSE library for pivco-huffman's
 * per-node partition-bitmap compression path.  See docs/FSE-V0.md.
 *
 * Owns PIVCO_FSE_NUM_TABLES pre-built CTable + DTable globals (one per
 * frequent-bit probability on a linear 0.50..0.99 / 0.01 schedule),
 * populated lazily on first use from the normalized counts in
 * pivco_fse_tables.h.
 *
 * On top of those static tables there are two transmitted-table ids.
 * PIVCO_FSE_NIBBLE_ID means "nibble table": instead of a
 * pre-built byte-alphabet distribution, the bitmap's bytes are split
 * into 4-bit nibbles, an FSE table is fitted to *that* bitmap's nibble
 * histogram, and the table description is written into the payload
 * ahead of the coded nibbles.  It costs a header (~15-30 bytes) but
 * adapts to bitmaps the fixed schedule models badly.
 * PIVCO_FSE_K1_ID means "k=1 bit-context table": the bitmap's bits are
 * modeled as P(bit | previous bit), two grid-quantized probabilities that
 * travel as one recipe byte and select prebuilt tables from a catalog (see
 * the PIVCO_K1_* knobs below and pivco_k1.c).
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

/* Table id for "nibble table" (see the file header).
 * Static-asserted against PIVCO_FSE_NUM_TABLES in pivco_fse.c. */
#define PIVCO_FSE_NIBBLE_ID 51

/* Nibble alphabet parameters for the nibble path.  A 16-symbol
 * alphabet does not need a big table: the per-region table build is
 * O(2^tableLog) at both ends, a smaller tableLog also shrinks the
 * FSE_writeNCount header we have to pay for per bitmap, and 2^7
 * states resolve a nibble histogram well enough (2^6 starts losing
 * on extreme skew).  The encoder cap is overridable for sweeps; the
 * decoder accepts any accuracy log up to PIVCO_FSE_NIB_TABLELOG_MAX,
 * since NCount carries the log per region. */
#define PIVCO_FSE_NIB_MAX_SYMBOL 15
#ifndef PIVCO_FSE_NIB_TABLELOG
#define PIVCO_FSE_NIB_TABLELOG   7
#endif
#define PIVCO_FSE_NIB_TABLELOG_MAX 10

/* Table id for the k=1 bit-context table (see pivco_k1.c): the bitmap
 * is coded with a byte-alphabet tANS whose distribution is derived
 * from P(bit | previous bit), two grid-quantized (16 groups)
 * probabilities that travel as one recipe byte.
 * The 256 recipes are a complete catalog
 * built on first use, so the decoder builds nothing per region.
 *
 * The knobs below are #ifndef-guarded for sweeps; both sides of the
 * wire derive everything from them, so a build with different values
 * reads different streams. */
#define PIVCO_FSE_K1_ID 52

/* States per table.  10 halves the catalog of 11 (2 MB of decode
 * tables) for 18% faster decode on the streams at the same size. */
#ifndef PIVCO_K1_TABLELOG
#define PIVCO_K1_TABLELOG 10
#endif

/* Regions of at least this many bytes are coded as PIVCO_K1_SEGS
 * interleaved segments (own state and carry chain each) so the decode
 * loop overlaps the dependent-lookup chains; smaller regions use one
 * state. */
#ifndef PIVCO_K1_SPLIT_MIN
#define PIVCO_K1_SPLIT_MIN 128
#endif

/* How many interleaved segments we use.
 * More is faster, but adds some data to the wire (a few bytes amortized). */
#ifndef PIVCO_K1_SEGS
#define PIVCO_K1_SEGS 8
#endif

/* Pre-encode skip: the encoder prices a region from its bit counts and
 * skips the tANS walk when estimate * PIVCO_K1_EST_SKIP exceeds the
 * largest useful output.  The estimate runs high on regions dominated
 * by one byte value even after the 0x00/0xFF correction (normalization
 * prices that value differently from the bit model), so the factor is
 * a ratio-vs-compress-speed dial, measured on the L3 streams + datasets
 * (M4) against no skip:  0.80: no size change, compress 1.3-1.5x;
 * 0.85: +0.02% (x-ray lit +0.12%), 1.3-1.8x;  0.90: +0.03% (x-ray lit
 * +0.21%), 1.4-1.9x.  0 disables the skip. */
#ifndef PIVCO_K1_EST_SKIP
#define PIVCO_K1_EST_SKIP 0.85
#endif

/* Prefetch the region's tables at region start (1) or not (0); see the
 * A/B numbers at the prefetch in pivco_k1.c. */
#ifndef PIVCO_K1_PREFETCH
#define PIVCO_K1_PREFETCH 1
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
 * pivco_fse_select_table) or PIVCO_FSE_NIBBLE_ID, in which case the
 * call is forwarded to pivco_fse_compress_nibble(). */
pivco_fse_status_t pivco_fse_compress(int table_id,
                                       const void *src, size_t src_len,
                                       void *dst, size_t dst_cap,
                                       size_t *out_len);

/* Decompress src[0..src_len) into dst (capacity dst_cap, expected size
 * is dst_expected).  On PIVCO_FSE_OK, *out_len == dst_expected.
 * table_id == PIVCO_FSE_NIBBLE_ID forwards to
 * pivco_fse_decompress_nibble(). */
pivco_fse_status_t pivco_fse_decompress(int table_id,
                                         const void *src, size_t src_len,
                                         void *dst, size_t dst_cap,
                                         size_t dst_expected,
                                         size_t *out_len);

/* ---- Nibble path ----
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
pivco_fse_status_t pivco_fse_compress_nibble(const void *src, size_t src_len,
                                               void *dst, size_t dst_cap,
                                               size_t *out_len);

pivco_fse_status_t pivco_fse_decompress_nibble(const void *src, size_t src_len,
                                                 void *dst, size_t dst_cap,
                                                 size_t dst_expected,
                                                 size_t *out_len);

/* ---- k=1 bit-context path (pivco_k1.c) ----
 *
 * Payload: [recipe:u8][tANS bits, backward, LSB-first][terminator].
 * The decoder needs only the expected byte count.  Returns
 * PIVCO_FSE_FALLBACK when the result would not be smaller than src_len
 * or dst_cap is too small; dst_cap >= src_len + 16 is always enough.
 *
 * max_len: the largest output the caller can use (0 = no limit).  The
 * encoder prices the region from its bit counts before coding and
 * returns PIVCO_FSE_FALLBACK without encoding when the estimate is
 * clearly above it (see PIVCO_K1_EST_SKIP). */
pivco_fse_status_t pivco_k1_compress(const void *src, size_t src_len,
                                      void *dst, size_t dst_cap,
                                      size_t max_len, size_t *out_len);

pivco_fse_status_t pivco_k1_decompress(const void *src, size_t src_len,
                                        void *dst, size_t dst_cap,
                                        size_t dst_expected,
                                        size_t *out_len);

/* Build the whole catalog (pthread_once; the coders call this on first
 * use).  Returns 0, or -1 on allocation failure. */
int pivco_k1_prebuild(void);

/* Helper: byte-wise XOR-flip a buffer (all 1s become 0s and vice
 * versa).  Used when the right side is the majority -- we flip the
 * bitmap so the encoder always sees the "0 is frequent" distribution
 * that the tables are tuned for. */
void pivco_fse_flip_bits(uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif  /* PIVCO_FSE_H */
