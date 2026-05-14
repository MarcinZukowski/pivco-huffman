/* pivco_huffman_neon_common.h — internal NEON symbols shared between
 * pivco_huffman_neon.c (the main 2-way tree-walk backend) and
 * pivco_huffman_neon_prefix.c (the prefix-radix backend).
 *
 * These symbols have external linkage so the prefix backend can delegate
 * subtree work to the neon backend without going through a public header.
 * They are not part of the library's public API.
 */

#ifndef PIVCO_HUFFMAN_NEON_COMMON_H
#define PIVCO_HUFFMAN_NEON_COMMON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Combined shuffle table: [256][32] where bytes 0-15 are the shuffle
 * for mask (right partition) and bytes 16-31 are for ~mask (left
 * partition).  Both loaded with a single ldp q0, q1 on ARM — one cache
 * line access instead of two separate lookups at unrelated addresses.
 *
 * Defined in pivco_huffman_neon.c; populated lazily by init_compress_table(). */
extern uint8_t compress_tab[256][32];
extern uint8_t compress_popcnt[256];

/* Lazily initialise compress_tab / compress_popcnt.  Cheap no-op after
 * the first call.  MUST be called before invoking any function that
 * reads from those tables. */
void init_compress_table(void);

#ifdef __cplusplus
}
#endif

#endif /* PIVCO_HUFFMAN_NEON_COMMON_H */
