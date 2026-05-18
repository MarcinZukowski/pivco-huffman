/* pivco_huffman_neon_tables.h — TD-slice subset: encoder partition
 * compress table only.  BU expand tables stripped — see .c file.
 */

#ifndef PIVCO_HUFFMAN_NEON_TABLES_H
#define PIVCO_HUFFMAN_NEON_TABLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Encoder partition (8 elements per call) shuffle / popcount tables.
 * Indexed by an 8-bit mask; bytes 0..15 give vqtbl1q indices for the
 * right half, bytes 16..31 for the left half. */
extern uint8_t compress_tab    [256][32];
extern uint8_t compress_popcnt [256];
extern int     compress_table_ready;
void init_compress_table(void);

#ifdef __cplusplus
}
#endif

#endif  /* PIVCO_HUFFMAN_NEON_TABLES_H */
