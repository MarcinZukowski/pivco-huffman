/* Nested PH: a partition bitmap's bytes coded as an inner PH block.
 *
 * Dynamic nPH (dnPH, PIVCO_FSE_DNPH_ID) fits the inner block's table to
 * the region, so its payload starts with the table's code lengths:
 *
 *   [code-length record][inner PH block]
 *
 * The inner block is an ordinary PH block over the region's bytes,
 * [block_N][marker prefix][regions], with block_N == the region's byte
 * count.  It cannot hold nested regions itself.
 *
 * Code-length record: a mode byte, then one of
 *   PIVCO_NPH_LENS_FULL      128 B: the 256 lengths as nibbles, symbol
 *                            2i in the low nibble of byte i, 2i+1 in
 *                            the high one; 0 = symbol absent.
 *   PIVCO_NPH_LENS_PRESENCE  a 32 B presence bitmap over the 256 byte
 *                            values (LSB-first), then the k present
 *                            symbols' lengths as nibbles in symbol
 *                            order, low nibble first: 32 + ceil(k/2) B.
 * The writer picks the smaller form.  Lengths are 1..PIVCO_MAX_CODE_LEN.
 *
 * The inner table takes everything the decoder needs from the code
 * lengths and the outer table (its flat layout); the rest of its
 * configuration is the encoder's choice (pivco_nph_inner_cfg). */

#ifndef PIVCO_NPH_H
#define PIVCO_NPH_H

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"

#include <stddef.h>
#include <stdint.h>

/* Largest inner block: the bitmap of the largest outer block. */
#define PIVCO_NPH_MAX_N (PIVCO_WIRE_MAX_N / 8 + 1)

/* What a context needs to code nested regions: a child context for the
 * inner block's encode or decode (the parent's arenas are in use by the
 * outer walk), the inner block's table, rebuilt per region, and on the
 * encoder side a staging buffer for the inner block. */
typedef struct pivco_nested {
    pivco_encoder_t *enc;
    pivco_decoder_t *dec;
    pivco_table_t    table;
    uint8_t         *out;
} pivco_nested_t;

/* The context's nested-region state, created on first use with a child
 * context of the same kind.  NULL on allocation failure, or when sc is
 * itself a child. */
pivco_nested_t *pivco_nested_for_encoder(pivco_scratch_t *sc);
pivco_nested_t *pivco_nested_for_decoder(pivco_scratch_t *sc);

/* The code-length record's mode byte (the forms are described above). */
#define PIVCO_NPH_LENS_FULL     0   /* 128 B of nibbles, all 256 symbols */
#define PIVCO_NPH_LENS_PRESENCE 1   /* 32 B presence bitmap + present symbols' nibbles */

/* Largest code-length record: the mode byte plus the full form. */
#define PIVCO_NPH_LENS_MAX_BYTES (1 + PIVCO_MAX_SYMBOLS / 2)

/* KrT of the inner blocks the encoder writes.  The block carries its
 * own KrT, so this is encoder policy, not wire. */
#ifndef PIVCO_NPH_KRT
#define PIVCO_NPH_KRT 256
#endif

/* Write the record for lens[] at out; returns its length (at most
 * PIVCO_NPH_LENS_MAX_BYTES). */
size_t pivco_nph_write_lens(const uint8_t lens[PIVCO_MAX_SYMBOLS], uint8_t *out);

/* Read a record from in[0..avail) into lens[]; returns the bytes read,
 * or 0 when the record is truncated, has an unknown mode, or carries a
 * length outside 1..PIVCO_MAX_CODE_LEN.  Whether the lengths form a
 * complete code is left to the table build. */
size_t pivco_nph_read_lens(const uint8_t *in, size_t avail,
                           uint8_t lens[PIVCO_MAX_SYMBOLS]);

/* The configuration of an inner block's table: no FSE inside, KrT
 * PIVCO_NPH_KRT, and the outer table's flat layout. */
void pivco_nph_inner_cfg(pivco_cfg_t *cfg, int flat_layout);

#endif  /* PIVCO_NPH_H */
