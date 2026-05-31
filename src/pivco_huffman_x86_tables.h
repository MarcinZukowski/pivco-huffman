/* pivco_huffman_x86_tables.h — shared x86 (SSE4.1 / AVX2) lookup tables.
 *
 * Two table families:
 *
 *   compress_tab + compress_popcnt + init_compress_table_x86
 *      Encoder partition shuffle table for the 8-element SSE pshufb-
 *      based `partition_8_sse` primitive.  Bytes 0..15 of compress_tab
 *      [mask] are the pshufb indices that pack right-going (bit==1)
 *      uint16 lanes to the front of the destination; bytes 16..31 are
 *      the complementary indices for the left half.  Out-of-range
 *      slots are set to 0x80 (pshufb zero-fill sentinel).
 *
 *   expand_tab + expand_popcnt + init_expand_table_x86
 *      BU merge per-mask-byte shuffle pattern for the 8-element
 *      pshufb merge over `_mm_unpacklo_epi64(L8, R8)`.  expand_tab[m][k]
 *      gives the lane index (0..15) for output position k controlled
 *      by mask byte m -- values 0..7 select from L, 8..15 from R.
 *
 * Both init functions are idempotent.  Not thread-safe; callers are
 * expected to invoke prim_codec_init from a single thread before
 * recursing.
 *
 * Internal header.  Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_X86_TABLES_H
#define PIVCO_HUFFMAN_X86_TABLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Encoder partition (8 elements per call) shuffle / popcount tables. */
extern uint8_t compress_tab    [256][32];
extern uint8_t compress_popcnt [256];
extern int     compress_table_ready;
void init_compress_table_x86(void);

/* BU merge expand / popcount tables. */
extern uint8_t expand_tab    [256][8];
extern uint8_t expand_popcnt [256];
extern int     expand_table_ready;
void init_expand_table_x86(void);

#ifdef __cplusplus
}
#endif

#endif  /* PIVCO_HUFFMAN_X86_TABLES_H */
