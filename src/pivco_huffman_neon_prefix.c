/* pivco_huffman_neon_prefix.c — experimental "prefix-stream" backend.
 *
 * Encodes the first M bits of each element's code as a contiguous
 * per-element stream (M = table->min_len), rather than PIVCO's standard
 * bitmap-per-level layout.  Intended to beat the current decoder on
 * deep, uniform-ish trees where M is large.
 *
 * v1 scope: handles only flat trees (min_code_len == max_code_len).
 *   - Applies to: uniform, sparse_4, sparse_16, two_sym_eq.
 *   - Encoder:  pack M bits per element.
 *   - Decoder:  bit-unpack per element → symbols[k] = code_to_sym[prefix[k]].
 *   - No subtree recursion, no radix partition.  Just permutation.
 *
 * Non-flat trees return PIVCO_ERR_CORRUPT from both encode and decode
 * so callers can gate at the block level and fall back to the neon
 * backend.
 */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>

#ifdef PIVCO_HAS_NEON

/* ---------- Bit pack / unpack helpers ----------
 *
 * Stream layout: M-bit values packed LSB-first.  Element k occupies bits
 * [k*M, k*M + M) of the stream. */

static inline void pack_m_bits(uint8_t *out, int k, int M, uint32_t value)
{
    size_t bit_pos = (size_t)k * (size_t)M;
    size_t byte_pos = bit_pos >> 3;
    int shift = (int)(bit_pos & 7);
    /* Up to 3 bytes touched for M ≤ 15. */
    out[byte_pos    ] |= (uint8_t)(value << shift);
    if (shift + M > 8) {
        out[byte_pos + 1] |= (uint8_t)(value >> (8 - shift));
        if (shift + M > 16)
            out[byte_pos + 2] |= (uint8_t)(value >> (16 - shift));
    }
}

static inline uint32_t unpack_m_bits(const uint8_t *in, int k, int M)
{
    size_t bit_pos = (size_t)k * (size_t)M;
    size_t byte_pos = bit_pos >> 3;
    int shift = (int)(bit_pos & 7);
    /* Read up to 24 bits straddling up to 3 bytes. */
    uint32_t w = (uint32_t)in[byte_pos]
               | ((uint32_t)in[byte_pos + 1] << 8)
               | ((uint32_t)in[byte_pos + 2] << 16);
    return (w >> shift) & (((uint32_t)1 << M) - 1);
}

/* Number of bytes needed for N elements × M bits (+2 byte slack so the
 * 3-byte unpack read is always in-bounds). */
static inline size_t prefix_stream_bytes(int N, int M)
{
    return (size_t)(((size_t)N * (size_t)M + 7) >> 3) + 2;
}

/* ---------- Encode ---------- */

int pivco_huffman_encode_neon_prefix(const uint8_t *symbols,
                                      const pivco_huffman_table_t *table,
                                      uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    /* v1: flat trees only. */
    if (table->min_len != table->max_len) return PIVCO_ERR_CORRUPT;

    const int N = PIVCO_BLOCK_SIZE;
    const int M = table->min_len;
    if (M < 1 || M > 15) return PIVCO_ERR_CORRUPT;

    size_t nbytes = prefix_stream_bytes(N, M);
    memset(out, 0, nbytes);

    for (int k = 0; k < N; k++) {
        uint16_t code = table->code[symbols[k]];
        pack_m_bits(out, k, M, code);
    }
    /* Trim trailing slack bytes from reported length — decoder's 2-byte
     * oversize buffer isn't part of the logical stream. */
    *out_len = (size_t)(((size_t)N * (size_t)M + 7) >> 3);
    return PIVCO_OK;
}

/* ---------- Decode ---------- */

int pivco_huffman_decode_neon_prefix(const uint8_t *in, size_t in_len,
                                      const pivco_huffman_table_t *table,
                                      uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;

    if (table->min_len != table->max_len) return PIVCO_ERR_CORRUPT;

    const int N = PIVCO_BLOCK_SIZE;
    const int M = table->min_len;
    if (M < 1 || M > 15) return PIVCO_ERR_CORRUPT;

    /* Build code → symbol lookup table. For canonical Huffman with a flat
     * tree, code c corresponds to sorted_symbols[first_sym_idx[M] + (c - first_code[M])].
     * first_code[M] is 0 for canonical codes in the flat case (all symbols
     * have the same length), but we compute defensively. */
    int K = 1 << M;
    uint8_t code_to_sym[1 << 15];  /* worst-case M=15 ⇒ 32K — ok on stack */
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (table->code_len[s] == M) {
            code_to_sym[table->code[s]] = (uint8_t)s;
        }
    }
    (void)K;

    /* Fast path for byte-aligned M values. */
    if (M == 8) {
        for (int k = 0; k < N; k++) {
            symbols[k] = code_to_sym[in[k]];
        }
        *consumed = (size_t)N;
        return PIVCO_OK;
    }
    if (M == 4) {
        /* Two codes per byte: low nibble = element 2k, high nibble = 2k+1. */
        for (int k = 0; k < N; k += 2) {
            uint8_t b = in[k >> 1];
            symbols[k    ] = code_to_sym[b & 0x0F];
            symbols[k + 1] = code_to_sym[b >> 4];
        }
        *consumed = (size_t)(N >> 1);
        return PIVCO_OK;
    }
    if (M == 2) {
        /* Four codes per byte. */
        for (int k = 0; k < N; k += 4) {
            uint8_t b = in[k >> 2];
            symbols[k    ] = code_to_sym[(b     ) & 3];
            symbols[k + 1] = code_to_sym[(b >> 2) & 3];
            symbols[k + 2] = code_to_sym[(b >> 4) & 3];
            symbols[k + 3] = code_to_sym[(b >> 6) & 3];
        }
        *consumed = (size_t)(N >> 2);
        return PIVCO_OK;
    }
    if (M == 1) {
        /* 8 codes per byte. */
        for (int k = 0; k < N; k += 8) {
            uint8_t b = in[k >> 3];
            symbols[k    ] = code_to_sym[(b     ) & 1];
            symbols[k + 1] = code_to_sym[(b >> 1) & 1];
            symbols[k + 2] = code_to_sym[(b >> 2) & 1];
            symbols[k + 3] = code_to_sym[(b >> 3) & 1];
            symbols[k + 4] = code_to_sym[(b >> 4) & 1];
            symbols[k + 5] = code_to_sym[(b >> 5) & 1];
            symbols[k + 6] = code_to_sym[(b >> 6) & 1];
            symbols[k + 7] = code_to_sym[(b >> 7) & 1];
        }
        *consumed = (size_t)(N >> 3);
        return PIVCO_OK;
    }

    /* Generic (slower) path for M ∈ {3,5,6,7,...}. */
    for (int k = 0; k < N; k++) {
        uint32_t code = unpack_m_bits(in, k, M);
        symbols[k] = code_to_sym[code];
    }
    *consumed = (size_t)(((size_t)N * (size_t)M + 7) >> 3);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
