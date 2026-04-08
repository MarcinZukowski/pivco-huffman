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

#endif /* PIVCO_HUFFMAN_COMMON_H */
