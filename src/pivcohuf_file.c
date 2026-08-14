/* pivcohuf file format codec.  See include/pivcohuf_file.h for the
 * wire-format specification. */

#include "pivcohuf_file.h"
#include "pivco_huffman.h"
#include "pivco_prof.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Monotonic wall-clock in nanoseconds, for the *_timed phase breakdown.
 * Coarse (phase-level, never inside hot inner loops), so always-on cost is
 * a handful of clock_gettime calls per compress/decompress. */
static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
#define TIC(t)        ((t) ? now_ns() : 0.0)
#define TOC(t, fld, s) do { if (t) (t)->fld += now_ns() - (s); } while (0)

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

__attribute__((unused))
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
__attribute__((unused)) static inline void put_u8 (uint8_t *p, uint8_t v)  { p[0] = v; }
static inline void put_u16(uint8_t *p, uint16_t v) { p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; }
static inline void put_u32(uint8_t *p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v>>8) & 0xff;
    p[2] = (v>>16) & 0xff; p[3] = (v>>24) & 0xff;
}
static inline void put_u64(uint8_t *p, uint64_t v) {
    put_u32(p, (uint32_t)v);
    put_u32(p + 4, (uint32_t)(v >> 32));
}
__attribute__((unused)) static inline uint8_t  get_u8 (const uint8_t *p) { return p[0]; }
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

size_t pivcohuf_compress_bound_blk(size_t in_len, size_t block_size)
{
    /* Per-block worst case is the full block size (no compression) plus
     * a small overhead for the encoded format.  We bound generously at
     * 2x block size; the K_right header adds <1%.  Smaller blocks mean
     * more per-block overhead, so the bound must use the actual block
     * size the caller will compress with. */
    if (block_size < 1) block_size = 1;
    if (block_size > PIVCO_WIRE_MAX_N) block_size = PIVCO_WIRE_MAX_N;
    const size_t B = block_size;
    size_t nblocks = (in_len + B - 1) / B;
    if (nblocks == 0) nblocks = 1;  /* zero-byte input still produces one header */
    size_t worst_per_block = 4 /* length prefix */ + 2 * B + 64;
    return PIVCOHUF_HEADER_SIZE      /* header */
         + 8 + 2 + 128                /* body header: usize + blk + code-len nibbles */
         + nblocks * worst_per_block;
}

size_t pivcohuf_compress_bound(size_t in_len)
{
    return pivcohuf_compress_bound_blk(in_len, PIVCO_BLOCK_SIZE);
}


static int pivcohuf_compress_impl(pivco_encoder_t *enc_ctx,
                                  const pivco_cfg_t *cfg,
                                  const uint8_t *in, size_t in_len,
                                  uint8_t *out, size_t *out_len,
                                  size_t block_size,
                                  pivcohuf_timing_t *tm)
{
    if (!in && in_len > 0) return PIVCOHUF_ERR_NULL;
    if (!out || !out_len) return PIVCOHUF_ERR_NULL;
    if (block_size < 1 || block_size > PIVCO_WIRE_MAX_N)
        return PIVCOHUF_ERR_BAD_BLOCK_SIZE;
    if (*out_len < pivcohuf_compress_bound_blk(in_len, block_size))
        return PIVCOHUF_ERR_OUTPUT_TOO_SMALL;

    const size_t B = block_size;

    /* Build histogram over real input via file_histogram above --
     * prim_histogram_chunk under a chunked u32 -> u64 wrapper. */
    uint64_t real_freq[256] = {0};
    { double _t = TIC(tm);
      if (pivco_histogram(enc_ctx, in, in_len, real_freq) != PIVCO_OK)
          return PIVCOHUF_ERR_INTERNAL;
      if (in_len == 0) real_freq[0] = 1;
      TOC(tm, freq_ns, _t); }

    pivco_table_t real_table;
    { PROF_TIC(); double _t = TIC(tm);
      if (pivco_build_table(cfg, real_freq, &real_table) != PIVCO_OK)
          return PIVCOHUF_ERR_INTERNAL;
      PROF_TOC(PROF_FILE_BUILD_TABLE_REAL, 1); TOC(tm, build_ns, _t); }

    /* Rebuild the encode-time table via the code-lens builder, so encode
     * uses the exact table the decoder reconstructs from the wire.  The tree
     * is fully determined by the code lengths (within-tier order is symbol-
     * value), so nothing beyond the lengths is transmitted. */
    pivco_table_t table;
    { PROF_TIC(); double _t = TIC(tm);
      if (pivco_build_table_from_code_lens(cfg, real_table.code_len,
                                                    &table) != PIVCO_OK)
          return PIVCOHUF_ERR_INTERNAL;
      PROF_TOC(PROF_FILE_BUILD_TABLE_SYN, 1); TOC(tm, build_ns, _t); }

    /* Pad with the most-frequent symbol (sorted_symbols[0] -- always has
     * the shortest code).  Padding with arbitrary bytes can hit pathological
     * deep-recursion paths in the encoder when blk_in << B. */
    const uint8_t pad_byte = table.sorted_symbols[0];

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
    double _tm = TIC(tm);
    uint8_t *block_buf = (uint8_t *)malloc(B);
    TOC(tm, malloc_ns, _tm);
    if (!block_buf) return PIVCOHUF_ERR_INTERNAL;
    double _te = TIC(tm);
    while (off < in_len) {
        size_t blk_in = in_len - off;
        size_t this_n;
        const uint8_t *blk_src;
        uint8_t *len_field;
        { PROF_TIC();
          if (blk_in >= B) {
              blk_src = in + off;
              this_n  = B;
              off += B;
          } else {
              /* Final (short) block: encode it at its actual size.  The
               * codec writes a 2-byte N header at the start of the encoded
               * stream so the decoder recovers the count without any
               * out-of-band channel. */
              blk_src = in + off;
              this_n  = blk_in;
              off = in_len;
          }
          (void)pad_byte; (void)block_buf;  /* padding path retired */
          len_field = p; p += 4;
          PROF_TOC(PROF_FILE_BLOCK_PROLOGUE, (uint64_t)this_n); }

        { PROF_TIC();
          size_t enc_len = 0;
          if (pivco_encode(enc_ctx, &table, blk_src, this_n, p, &enc_len) != PIVCO_OK) {
              free(block_buf);
              return PIVCOHUF_ERR_INTERNAL;
          }
          put_u32(len_field, (uint32_t)enc_len);
          p += enc_len;
          PROF_TOC(PROF_FILE_BLOCK_ENCODE, (uint64_t)this_n); }
    }
    TOC(tm, codec_ns, _te);
    free(block_buf);

    size_t body_len = (size_t)(p - body);

    /* Write HEADER (positions are fixed).  Checksums temporarily disabled
     * -- always zero (2026-05-12).  Format byte positions preserved so a
     * later commit can turn them back on without a wire-format break. */
    memcpy(hdr + 0, PIVCOHUF_MAGIC, 8);
    hdr[8] = PIVCOHUF_VERSION_MAJOR;
    hdr[9] = PIVCOHUF_VERSION_MINOR;
    put_u64(hdr + 10, (uint64_t)body_len);
    put_u32(hdr + 18, 0);   /* BODY_CHECKSUM = 0 (disabled) */
    put_u32(hdr + 22, 0);   /* HEADER_CHECKSUM = 0 (disabled) */

    *out_len = (size_t)(p - out);
    return PIVCOHUF_OK;
}

/* pha (#PHA): same wire/decoder, but per-block bitmaps may be ANS(FSE)-coded.
 * The FSE path is selected per build via pivco_cfg_t.fse_enabled (baked
 * into the table).  Decompress needs no flag — it auto-detects FSE
 * markers per block. */
static int compress_dispatch(const uint8_t *in, size_t in_len,
                             uint8_t *out, size_t *out_len,
                             const pivco_cfg_t *cfg_in, int use_ans,
                             size_t block_size, pivcohuf_timing_t *tm)
{
    if (tm) memset(tm, 0, sizeof(*tm));
    pivco_cfg_t cfg = cfg_in ? *cfg_in : pivco_cfg_default;
    cfg.fse_enabled = use_ans;
    /* FASTEST_COMPRESS is the one effort mode the bare table build
     * cannot resolve (it needs the input size): below 256 KiB plain
     * Huffman lengths encode fastest; above, a flatter tree ENCODES
     * faster than the BALANCED shaping solve costs, and the solve's
     * cost keeps shrinking as 1/n. */
    if (cfg.effort == PIVCO_EFFORT_FASTEST_COMPRESS)
        cfg.effort = in_len < (size_t)262144 ? PIVCO_EFFORT_PLAIN
                                             : PIVCO_EFFORT_BALANCED;
    pivco_encoder_t *enc_ctx = pivco_encoder_create();
    if (!enc_ctx) return PIVCOHUF_ERR_INTERNAL;
    int r = pivcohuf_compress_impl(enc_ctx, &cfg, in, in_len, out, out_len,
                                   block_size, tm);
    pivco_encoder_free(enc_ctx);
    return r;
}

int pivcohuf_compress_blk(const uint8_t *in, size_t in_len,
                          uint8_t *out, size_t *out_len,
                          int use_ans, size_t block_size,
                          pivcohuf_timing_t *timing)
{
    return compress_dispatch(in, in_len, out, out_len, NULL, use_ans,
                             block_size, timing);
}

int pivcohuf_compress_cfg(const uint8_t *in, size_t in_len,
                          uint8_t *out, size_t *out_len,
                          const pivco_cfg_t *cfg, size_t block_size,
                          pivcohuf_timing_t *timing)
{
    int use_ans = cfg ? cfg->fse_enabled : 0;
    return compress_dispatch(in, in_len, out, out_len, cfg, use_ans,
                             block_size, timing);
}

int pivcohuf_compress_ex(const uint8_t *in, size_t in_len,
                         uint8_t *out, size_t *out_len, int use_ans)
{
    return compress_dispatch(in, in_len, out, out_len, NULL, use_ans,
                             PIVCO_BLOCK_SIZE, NULL);
}

int pivcohuf_compress(const uint8_t *in, size_t in_len,
                      uint8_t *out, size_t *out_len)
{
    return compress_dispatch(in, in_len, out, out_len, NULL, 0,
                             PIVCO_BLOCK_SIZE, NULL);
}

int pivcohuf_compress_timed(const uint8_t *in, size_t in_len,
                            uint8_t *out, size_t *out_len,
                            int use_ans, pivcohuf_timing_t *timing)
{
    return compress_dispatch(in, in_len, out, out_len, NULL, use_ans,
                             PIVCO_BLOCK_SIZE, timing);
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
    /* HEADER_CHECKSUM verification disabled (2026-05-12) -- bytes are
     * still in the format at offset 22..25, currently always zero. */
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

static int pivcohuf_decompress_impl(pivco_decoder_t *dec_ctx,
                                    const uint8_t *in, size_t in_len,
                                    uint8_t *out, size_t *out_len,
                                    pivcohuf_timing_t *tm)
{
    if (!in || !out || !out_len) return PIVCOHUF_ERR_NULL;
    uint64_t body_len_u64;
    int rc = parse_header(in, in_len, &body_len_u64);
    if (rc != PIVCOHUF_OK) return rc;
    if (in_len < PIVCOHUF_HEADER_SIZE + body_len_u64)
        return PIVCOHUF_ERR_TOO_SHORT;
    size_t body_len = (size_t)body_len_u64;
    const uint8_t *body = in + PIVCOHUF_HEADER_SIZE;
    /* BODY_CHECKSUM verification disabled (2026-05-12). */

    /* Parse body header: UNCOMPRESSED_SIZE(8) + BLOCK_SIZE(2) + nibbles(128). */
    if (body_len < 8 + 2 + 128) return PIVCOHUF_ERR_TOO_SHORT;
    size_t uncomp_size = (size_t)get_u64(body);
    uint16_t file_blk = get_u16(body + 8);
    /* The block size is read from the file, not fixed at compile time: the
     * codec sizes its scratch dynamically off the per-block wire N header,
     * so any block size the encoder could write is decodable here.  Only a
     * zero block size (impossible from a valid encoder) is rejected. */
    if (file_blk == 0) return PIVCOHUF_ERR_BAD_BLOCK_SIZE;
    const size_t B = (size_t)file_blk;

    if (*out_len < uncomp_size) return PIVCOHUF_ERR_OUTPUT_TOO_SMALL;

    /* Reconstruct Huffman table from code lengths. */
    uint8_t code_lens[256];
    const uint8_t *nibbles = body + 10;
    for (int i = 0; i < 128; i++) {
        code_lens[2*i]     = nibbles[i] & 0x0F;
        code_lens[2*i + 1] = (nibbles[i] >> 4) & 0x0F;
    }

    pivco_table_t table;
    { PROF_TIC(); double _t = TIC(tm);
      if (pivco_build_table_from_code_lens(NULL, code_lens, &table) != PIVCO_OK)
          return PIVCOHUF_ERR_INTERNAL;
      PROF_TOC(PROF_FILE_BUILD_TABLE_SYN, 1); TOC(tm, build_ns, _t); }
    /* Sanity check: rebuilt code lengths must match. */
    for (int s = 0; s < 256; s++) {
        if (table.code_len[s] != code_lens[s]) {
            return PIVCOHUF_ERR_INTERNAL;
        }
    }

    /* Decode blocks.  block_buf is on heap (avoids large stack frames; also
     * sized B which is read from the file). */
    double _tm = TIC(tm);
    uint8_t *block_buf = (uint8_t *)malloc(B);
    TOC(tm, malloc_ns, _tm);
    if (!block_buf) return PIVCOHUF_ERR_INTERNAL;
    const uint8_t *p = body + 10 + 128;
    const uint8_t *body_end = body + body_len;
    size_t written = 0;
    int err = 0;
    double _td = TIC(tm);
    while (p < body_end && written < uncomp_size) {
        uint32_t blk_enc_len;
        uint8_t *blk_out;
        size_t blk_remaining;
        { PROF_TIC();
          if (p + 4 > body_end) { err = PIVCOHUF_ERR_TOO_SHORT; break; }
          blk_enc_len = get_u32(p); p += 4;
          if (p + blk_enc_len > body_end) { err = PIVCOHUF_ERR_TOO_SHORT; break; }
          blk_remaining = uncomp_size - written;
          blk_out = (blk_remaining >= B) ? (out + written) : block_buf;
          PROF_TOC(PROF_FILE_BLOCK_PROLOGUE, (uint64_t)B); }

        { PROF_TIC();
          size_t consumed = 0;
          if (pivco_decode(dec_ctx, &table, p, blk_enc_len,
                                   blk_out, &consumed) != PIVCO_OK) {
              err = PIVCOHUF_ERR_INTERNAL; break;
          }
          PROF_TOC(PROF_FILE_BLOCK_DECODE, (uint64_t)B); }
        if (blk_remaining < B) {
            memcpy(out + written, block_buf, blk_remaining);
            written = uncomp_size;
        } else {
            written += B;
        }
        p += blk_enc_len;
    }
    TOC(tm, codec_ns, _td);
    free(block_buf);
    if (err) return err;

    if (written != uncomp_size) return PIVCOHUF_ERR_INTERNAL;
    *out_len = uncomp_size;
    return PIVCOHUF_OK;
}

int pivcohuf_decompress(const uint8_t *in, size_t in_len,
                        uint8_t *out, size_t *out_len)
{
    {
        pivco_decoder_t *dec_ctx = pivco_decoder_create();
        if (!dec_ctx) return PIVCOHUF_ERR_INTERNAL;
        int r = pivcohuf_decompress_impl(dec_ctx, in, in_len, out, out_len, NULL);
        pivco_decoder_free(dec_ctx);
        return r;
    }
}

int pivcohuf_decompress_timed(const uint8_t *in, size_t in_len,
                              uint8_t *out, size_t *out_len,
                              pivcohuf_timing_t *timing)
{
    if (timing) memset(timing, 0, sizeof(*timing));
    {
        pivco_decoder_t *dec_ctx = pivco_decoder_create();
        if (!dec_ctx) return PIVCOHUF_ERR_INTERNAL;
        int r = pivcohuf_decompress_impl(dec_ctx, in, in_len, out, out_len, timing);
        pivco_decoder_free(dec_ctx);
        return r;
    }
}
