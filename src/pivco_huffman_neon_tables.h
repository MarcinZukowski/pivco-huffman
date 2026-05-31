/* pivco_huffman_neon_tables.h — shared NEON lookup tables.
 *
 * Two table families live here so both the legacy NEON .c files and
 * the codec.c-compiled-as-NEON object library can share one runtime
 * copy each (the largest, `expand_tab_pre`, is 18 KB — material to
 * avoid duplicating per-TU).
 *
 *   compress_tab + compress_popcnt + init_compress_table
 *      Encoder partition shuffle table for the 8-element NEON
 *      `partition_8` primitive.  Indexed by an 8-bit partition mask;
 *      bytes  0..15 give the vqtbl1q indices that pack the right half
 *      (bit==1) to the front of the destination lane; bytes 16..31
 *      give the indices for the complement (bit==0 → left half).
 *      Both halves loaded with one `ldp q0, q1` (32 bytes contiguous).
 *
 *   expand_tab / expand_tab_pre / expand_popcnt + init_expand_table
 *      BU `merge` V4 strategy: per (nr0, m1) precomputed shuf
 *      vectors for the 32-byte (L_full, R_full) source register pair.
 *      18 432 bytes; fits L1d on every target.  See the long comment
 *      at the definition for the (nr0, m1) algebra.
 *
 * Both init functions are idempotent and lazy: the first caller pays
 * the construction cost, subsequent callers no-op.  Not thread-safe;
 * the caller must guarantee a single init thread (in practice the
 * encode/decode entry points call init at top before recursing).
 *
 * Internal header.  Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_NEON_TABLES_H
#define PIVCO_HUFFMAN_NEON_TABLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Encoder partition (8 elements per call) shuffle / popcount tables. */
extern uint8_t compress_tab    [256][32];
extern uint8_t compress_popcnt [256];
extern int     compress_table_ready;
void init_compress_table(void);

/* BU merge V4 expand / popcount tables. */
extern uint8_t expand_tab     [256][8];
extern uint8_t expand_tab_pre [9][256][8];
extern uint8_t expand_popcnt  [256];
extern int     expand_table_ready;
void init_expand_table(void);

#ifdef __cplusplus
}
#endif

#endif  /* PIVCO_HUFFMAN_NEON_TABLES_H */
