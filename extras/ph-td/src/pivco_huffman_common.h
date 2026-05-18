#ifndef PIVCO_HUFFMAN_COMMON_H
#define PIVCO_HUFFMAN_COMMON_H

#include "pivco_huffman.h"
#include <string.h>

/* ---------- Bitmap utilities ---------- */

/* Popcount of a bitmap stored as bytes. n_bytes = ceil(count / 8). */
static inline int fast_popcount(const uint8_t *bitmap, int n_bytes)
{
    int count = 0;
    int i = 0;
    /* Process 8 bytes at a time */
    for (; i + 8 <= n_bytes; i += 8) {
        uint64_t w;
        memcpy(&w, bitmap + i, 8);
        count += __builtin_popcountll(w);
    }
    /* Remaining bytes */
    for (; i < n_bytes; i++) {
        count += __builtin_popcount(bitmap[i]);
    }
    return count;
}

/* Extract indices of set bits in bitmap into active[].
   Returns number of set bits. */
static inline int bitmap_extract(const uint8_t *bitmap, int n_bytes,
                                 const uint16_t *old_active,
                                 uint16_t *new_active)
{
    int out = 0;
    int bit_idx = 0;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t byte = bitmap[i];
        while (byte) {
            int bit = __builtin_ctz(byte);
            new_active[out++] = old_active[bit_idx + bit];
            byte &= byte - 1; /* clear lowest set bit */
        }
        bit_idx += 8;
    }
    return out;
}

/* Same but active[j] == j (identity mapping) */
static inline int bitmap_extract_identity(const uint8_t *bitmap, int n_bytes,
                                          uint16_t *new_active)
{
    int out = 0;
    int bit_idx = 0;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t byte = bitmap[i];
        while (byte) {
            int bit = __builtin_ctz(byte);
            new_active[out++] = (uint16_t)(bit_idx + bit);
            byte &= byte - 1;
        }
        bit_idx += 8;
    }
    return out;
}

/* Set bit j in bitmap */
static inline void bitmap_set(uint8_t *bitmap, int j)
{
    bitmap[j >> 3] |= (1u << (j & 7));
}

/* Get bit j from bitmap */
static inline int bitmap_get(const uint8_t *bitmap, int j)
{
    return (bitmap[j >> 3] >> (j & 7)) & 1;
}

/* Bytes needed for a bitmap of n bits */
static inline int bitmap_bytes(int n)
{
    return (n + 7) >> 3;
}

/* K_right wire-format header decision (2026-05-12).
 *
 * Wire format: at each non-flat internal node whose bitmap is followed by
 * recursion into at least one non-leaf child, the encoder writes a 2-byte
 * little-endian uint16 K_right header immediately before the bitmap.  The
 * BU decoder reads this directly instead of running popcount; the TD
 * decoder skips it (still computes splits inline per stride).
 *
 * Condition: node has at least one child that's NOT a leaf.  Encodes the
 * exact set of popcount call sites in the BU decoder.  Both-leaf cases
 * and HALF_*-with-leaf cases get no header (decoder uses merge_both_const
 * directly).
 *
 * The "needs header" decision is a pure function of the tree topology and
 * matches across encoder and decoder via this shared helper. */
static inline int kr_header_needed(const pivco_huffman_table_t *table,
                                    int16_t node_id)
{
    const pivco_tree_node_t *n = &table->tree[node_id];
    if (n->symbol >= 0) return 0;                /* leaf */
    if (table->flat_depth[node_id] >= 2) return 0; /* flat path */
    return (table->tree[n->left].symbol < 0)
        || (table->tree[n->right].symbol < 0);
}

#define KR_HEADER_BYTES 2  /* uint16 little-endian */

#endif /* PIVCO_HUFFMAN_COMMON_H */
