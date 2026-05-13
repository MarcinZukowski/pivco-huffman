/* pivcohuf file format codec.  See include/pivcohuf_file.h for the
 * wire-format specification. */

#include "pivcohuf_file.h"
#include "pivco_huffman.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * XXH32 -- 32-bit xxHash, tiny self-contained implementation.
 * Used for header + body integrity (not crypto).  Seed 0.
 * Algorithm reference: github.com/Cyan4973/xxHash (BSD-2).
 * ============================================================ */

#define XXH_PRIME32_1  0x9E3779B1U
#define XXH_PRIME32_2  0x85EBCA77U
#define XXH_PRIME32_3  0xC2B2AE3DU
#define XXH_PRIME32_4  0x27D4EB2FU
#define XXH_PRIME32_5  0x165667B1U

static inline uint32_t rotl32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

static uint32_t xxh32(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    const uint8_t *end = p + len;
    uint32_t h;

    if (len >= 16) {
        uint32_t v1 = 0 + XXH_PRIME32_1 + XXH_PRIME32_2;
        uint32_t v2 = 0 + XXH_PRIME32_2;
        uint32_t v3 = 0;
        uint32_t v4 = 0 - XXH_PRIME32_1;
        const uint8_t *limit = end - 16;
        while (p <= limit) {
            uint32_t k;
            memcpy(&k, p, 4); p += 4;
            v1 = rotl32(v1 + k * XXH_PRIME32_2, 13) * XXH_PRIME32_1;
            memcpy(&k, p, 4); p += 4;
            v2 = rotl32(v2 + k * XXH_PRIME32_2, 13) * XXH_PRIME32_1;
            memcpy(&k, p, 4); p += 4;
            v3 = rotl32(v3 + k * XXH_PRIME32_2, 13) * XXH_PRIME32_1;
            memcpy(&k, p, 4); p += 4;
            v4 = rotl32(v4 + k * XXH_PRIME32_2, 13) * XXH_PRIME32_1;
        }
        h = rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18);
    } else {
        h = 0 + XXH_PRIME32_5;
    }
    h += (uint32_t)len;

    while (p + 4 <= end) {
        uint32_t k;
        memcpy(&k, p, 4); p += 4;
        h += k * XXH_PRIME32_3;
        h = rotl32(h, 17) * XXH_PRIME32_4;
    }
    while (p < end) {
        h += (uint32_t)(*p++) * XXH_PRIME32_5;
        h = rotl32(h, 11) * XXH_PRIME32_1;
    }
    h ^= h >> 15; h *= XXH_PRIME32_2;
    h ^= h >> 13; h *= XXH_PRIME32_3;
    h ^= h >> 16;
    return h;
}

/* ============================================================
 * Little-endian field readers/writers.
 * ============================================================ */
static inline void put_u8 (uint8_t *p, uint8_t v)  { p[0] = v; }
static inline void put_u16(uint8_t *p, uint16_t v) { p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; }
static inline void put_u32(uint8_t *p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v>>8) & 0xff;
    p[2] = (v>>16) & 0xff; p[3] = (v>>24) & 0xff;
}
static inline void put_u64(uint8_t *p, uint64_t v) {
    put_u32(p, (uint32_t)v);
    put_u32(p + 4, (uint32_t)(v >> 32));
}
static inline uint8_t  get_u8 (const uint8_t *p) { return p[0]; }
static inline uint16_t get_u16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static inline uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint64_t get_u64(const uint8_t *p) {
    return (uint64_t)get_u32(p) | ((uint64_t)get_u32(p + 4) << 32);
}

/* ============================================================
 * compress_bound + compress
 * ============================================================ */

size_t pivcohuf_compress_bound(size_t in_len)
{
    /* Per-block worst case is the full block size (no compression) plus
     * a small overhead for the encoded format.  We bound generously at
     * 2x block size; the K_right header adds <1%. */
    const size_t B = PIVCO_BLOCK_SIZE;
    size_t nblocks = (in_len + B - 1) / B;
    if (nblocks == 0) nblocks = 1;  /* zero-byte input still produces one header */
    size_t worst_per_block = 4 /* length prefix */ + PIVCO_MAX_ENCODED_SIZE;
    return PIVCOHUF_HEADER_SIZE      /* header */
         + 8 + 2 + 128                /* body header */
         + nblocks * worst_per_block;
}

int pivcohuf_compress(const uint8_t *in, size_t in_len,
                      uint8_t *out, size_t *out_len)
{
    if (!in && in_len > 0) return PIVCOHUF_ERR_NULL;
    if (!out || !out_len) return PIVCOHUF_ERR_NULL;
    if (*out_len < pivcohuf_compress_bound(in_len))
        return PIVCOHUF_ERR_OUTPUT_TOO_SMALL;

    const size_t B = PIVCO_BLOCK_SIZE;

    /* Build histogram over real input. */
    uint64_t real_freq[256] = {0};
    for (size_t i = 0; i < in_len; i++) real_freq[in[i]]++;
    if (in_len == 0) real_freq[0] = 1;

    pivco_huffman_table_t real_table;
    if (pivco_huffman_build_table(real_freq, &real_table) != PIVCO_OK)
        return PIVCOHUF_ERR_INTERNAL;

    /* Rebuild the table from synthesised exponential frequencies derived
     * from the code lengths -- the decoder will do exactly the same
     * synthesis from the serialised code-length nibbles, so both sides
     * deterministically produce identical code[]/code_la[]/tree[] from
     * the same inputs.  Lengths drive compression; the resulting tree
     * shape is what matters for round-trip. */
    uint64_t freq[256] = {0};
    int max_len = 0;
    for (int s = 0; s < 256; s++)
        if (real_table.code_len[s] > max_len) max_len = real_table.code_len[s];
    for (int s = 0; s < 256; s++)
        if (real_table.code_len[s] > 0)
            freq[s] = (uint64_t)1 << (max_len - real_table.code_len[s]);

    pivco_huffman_table_t table;
    if (pivco_huffman_build_table(freq, &table) != PIVCO_OK)
        return PIVCOHUF_ERR_INTERNAL;

    /* Pad with prefill_sym (the most-frequent symbol -- always has the
     * shortest code).  Padding with arbitrary bytes can hit pathological
     * deep-recursion paths in the encoder when blk_in << B. */
    const uint8_t pad_byte = table.prefill_sym;

    uint8_t *p = out;
    /* === Reserve HEADER bytes; fill at end. === */
    uint8_t *hdr = p;
    p += PIVCOHUF_HEADER_SIZE;

    /* === BODY start. === */
    uint8_t *body = p;

    /* UNCOMPRESSED_SIZE */
    put_u64(p, (uint64_t)in_len); p += 8;

    /* BLOCK_SIZE (uint16, 1024..65535). */
    put_u16(p, (uint16_t)B); p += 2;

    /* CODE_LENGTHS packed as 4-bit nibbles, sym 2i in low nibble. */
    for (int i = 0; i < 128; i++) {
        uint8_t lo = table.code_len[2*i]     & 0x0F;
        uint8_t hi = table.code_len[2*i + 1] & 0x0F;
        p[i] = (uint8_t)(lo | (hi << 4));
    }
    p += 128;

    /* === Encode block-by-block. === */
    size_t off = 0;
    uint8_t *block_buf = (uint8_t *)malloc(B);
    if (!block_buf) return PIVCOHUF_ERR_INTERNAL;
    while (off < in_len) {
        size_t blk_in = in_len - off;
        const uint8_t *blk_src;
        if (blk_in >= B) {
            blk_src = in + off;
            off += B;
        } else {
            /* Final (short) block: pad to full B with prefill_sym. */
            if (blk_in > 0) memcpy(block_buf, in + off, blk_in);
            memset(block_buf + blk_in, pad_byte, B - blk_in);
            blk_src = block_buf;
            off = in_len;
        }
        uint8_t *len_field = p; p += 4;
        size_t enc_len = 0;
        if (pivco_huffman_encode(blk_src, &table, p, &enc_len) != PIVCO_OK) {
            free(block_buf);
            return PIVCOHUF_ERR_INTERNAL;
        }
        put_u32(len_field, (uint32_t)enc_len);
        p += enc_len;
    }
    free(block_buf);

    size_t body_len = (size_t)(p - body);

    /* Body checksum (covers BODY only). */
    uint32_t body_csum = xxh32(body, body_len);

    /* Write HEADER (positions are fixed). */
    memcpy(hdr + 0, PIVCOHUF_MAGIC, 8);
    hdr[8] = PIVCOHUF_VERSION_MAJOR;
    hdr[9] = PIVCOHUF_VERSION_MINOR;
    put_u64(hdr + 10, (uint64_t)body_len);
    put_u32(hdr + 18, body_csum);
    /* Header checksum covers bytes 0..21 (before the checksum field itself). */
    uint32_t hdr_csum = xxh32(hdr, 22);
    put_u32(hdr + 22, hdr_csum);

    *out_len = (size_t)(p - out);
    return PIVCOHUF_OK;
}

/* ============================================================
 * peek + decompress
 * ============================================================ */

static int parse_header(const uint8_t *in, size_t in_len, uint64_t *body_len)
{
    if (in_len < PIVCOHUF_HEADER_SIZE) return PIVCOHUF_ERR_TOO_SHORT;
    if (memcmp(in, PIVCOHUF_MAGIC, 8) != 0) return PIVCOHUF_ERR_BAD_MAGIC;
    if (in[8] != PIVCOHUF_VERSION_MAJOR || in[9] != PIVCOHUF_VERSION_MINOR)
        return PIVCOHUF_ERR_BAD_VERSION;
    uint32_t got = get_u32(in + 22);
    uint32_t expect = xxh32(in, 22);
    if (got != expect) return PIVCOHUF_ERR_BAD_HEADER_CHECKSUM;
    *body_len = get_u64(in + 10);
    return PIVCOHUF_OK;
}

int pivcohuf_peek_uncompressed_size(const uint8_t *in, size_t in_len,
                                     size_t *uncompressed_size)
{
    if (!in || !uncompressed_size) return PIVCOHUF_ERR_NULL;
    uint64_t body_len;
    int rc = parse_header(in, in_len, &body_len);
    if (rc != PIVCOHUF_OK) return rc;
    if (in_len < PIVCOHUF_HEADER_SIZE + 8) return PIVCOHUF_ERR_TOO_SHORT;
    *uncompressed_size = (size_t)get_u64(in + PIVCOHUF_HEADER_SIZE);
    return PIVCOHUF_OK;
}

int pivcohuf_decompress(const uint8_t *in, size_t in_len,
                        uint8_t *out, size_t *out_len)
{
    if (!in || !out || !out_len) return PIVCOHUF_ERR_NULL;
    uint64_t body_len_u64;
    int rc = parse_header(in, in_len, &body_len_u64);
    if (rc != PIVCOHUF_OK) return rc;
    if (in_len < PIVCOHUF_HEADER_SIZE + body_len_u64)
        return PIVCOHUF_ERR_TOO_SHORT;
    size_t body_len = (size_t)body_len_u64;
    const uint8_t *body = in + PIVCOHUF_HEADER_SIZE;

    uint32_t body_csum_got = get_u32(in + 18);
    uint32_t body_csum_expect = xxh32(body, body_len);
    if (body_csum_got != body_csum_expect) return PIVCOHUF_ERR_BAD_BODY_CHECKSUM;

    /* Parse body header. */
    if (body_len < 8 + 2 + 128) return PIVCOHUF_ERR_TOO_SHORT;
    size_t uncomp_size = (size_t)get_u64(body);
    uint16_t file_blk = get_u16(body + 8);
    if (file_blk != (uint16_t)PIVCO_BLOCK_SIZE) return PIVCOHUF_ERR_BAD_BLOCK_SIZE;
    const size_t B = (size_t)file_blk;

    if (*out_len < uncomp_size) return PIVCOHUF_ERR_OUTPUT_TOO_SMALL;

    /* Reconstruct Huffman table from code lengths. */
    uint8_t code_lens[256];
    const uint8_t *nibbles = body + 10;
    for (int i = 0; i < 128; i++) {
        code_lens[2*i]     = nibbles[i] & 0x0F;
        code_lens[2*i + 1] = (nibbles[i] >> 4) & 0x0F;
    }
    /* Synthesize a frequency histogram that yields the same code lengths
     * via build_table.  Trick: a symbol with length L gets frequency
     * 2^(MAX_LEN - L) -- pure exponential weights produce a length-
     * limited canonical Huffman tree with those exact lengths. */
    uint64_t freq[256] = {0};
    int max_len = 0;
    for (int s = 0; s < 256; s++) {
        if (code_lens[s] > max_len) max_len = code_lens[s];
    }
    for (int s = 0; s < 256; s++) {
        if (code_lens[s] > 0) {
            freq[s] = (uint64_t)1 << (max_len - code_lens[s]);
        }
    }
    pivco_huffman_table_t table;
    if (pivco_huffman_build_table(freq, &table) != PIVCO_OK)
        return PIVCOHUF_ERR_INTERNAL;
    /* Sanity check: rebuilt code lengths must match. */
    for (int s = 0; s < 256; s++) {
        if (table.code_len[s] != code_lens[s]) {
            return PIVCOHUF_ERR_INTERNAL;
        }
    }

    /* Decode blocks.  block_buf is on heap (avoids large stack frames; also
     * sized B which is read from the file). */
    uint8_t *block_buf = (uint8_t *)malloc(B);
    if (!block_buf) return PIVCOHUF_ERR_INTERNAL;
    const uint8_t *p = body + 10 + 128;
    const uint8_t *body_end = body + body_len;
    size_t written = 0;
    int err = 0;
    while (p < body_end && written < uncomp_size) {
        if (p + 4 > body_end) { err = PIVCOHUF_ERR_TOO_SHORT; break; }
        uint32_t blk_enc_len = get_u32(p); p += 4;
        if (p + blk_enc_len > body_end) { err = PIVCOHUF_ERR_TOO_SHORT; break; }
        size_t blk_remaining = uncomp_size - written;
        uint8_t *blk_out = (blk_remaining >= B) ? (out + written) : block_buf;
        size_t consumed = 0;
        if (pivco_huffman_decode(p, blk_enc_len, &table,
                                 blk_out, &consumed) != PIVCO_OK) {
            err = PIVCOHUF_ERR_INTERNAL; break;
        }
        if (blk_remaining < B) {
            memcpy(out + written, block_buf, blk_remaining);
            written = uncomp_size;
        } else {
            written += B;
        }
        p += blk_enc_len;
    }
    free(block_buf);
    if (err) return err;

    if (written != uncomp_size) return PIVCOHUF_ERR_INTERNAL;
    *out_len = uncomp_size;
    return PIVCOHUF_OK;
}
