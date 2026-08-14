#include "pivco_huffman.h"
#include <string.h>

/* ================================================================
 * Bitstream reader — branchless 64-bit accumulator, MSB-aligned.
 *
 * bits: [valid bits (nbits)] [empty (64-nbits)]
 *        ^MSB                                 ^LSB
 *
 * Peek extracts the top N bits.  Consume shifts them out.
 * Refill loads up to 7 new bytes via a single unaligned 64-bit read
 * (caller must ensure 8 bytes of readable padding past the stream).
 * ================================================================ */

typedef struct {
    uint64_t       bits;
    int            nbits;
    const uint8_t *ptr;
    const uint8_t *end;
} bitreader_t;

static inline void br_init(bitreader_t *br, const uint8_t *data, size_t len)
{
    br->bits  = 0;
    br->nbits = 0;
    br->ptr   = data;
    br->end   = data + len;
}

/* Branchless refill: one 64-bit unaligned load, always safe if
   8 bytes of padding exist after the stream. */
static inline void br_refill(bitreader_t *br)
{
    uint64_t next;
    memcpy(&next, br->ptr, 8);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    next = __builtin_bswap64(next);
#endif
    br->bits |= next >> br->nbits;
    int bytes = (64 - br->nbits) >> 3;
    br->ptr  += bytes;
    br->nbits += bytes << 3;
}

/* Slow refill for streams without padding */
static inline void br_refill_safe(bitreader_t *br)
{
    while (br->nbits <= 56 && br->ptr < br->end) {
        br->bits |= (uint64_t)(*br->ptr++) << (56 - br->nbits);
        br->nbits += 8;
    }
}

static inline uint32_t br_peek(const bitreader_t *br, int n)
{
    return (uint32_t)(br->bits >> (64 - n));
}

static inline void br_consume(bitreader_t *br, int n)
{
    br->bits <<= n;
    br->nbits -= n;
}

/* ================================================================
 * Bitstream writer
 * ================================================================ */

typedef struct {
    uint8_t *buf;
    uint64_t bits;
    int      nbits;
    size_t   byte_pos;
} bitwriter_t;

static inline void bw_init(bitwriter_t *bw, uint8_t *buf)
{
    bw->buf = buf; bw->bits = 0; bw->nbits = 0; bw->byte_pos = 0;
}

static inline void bw_flush(bitwriter_t *bw)
{
    while (bw->nbits >= 8) {
        bw->buf[bw->byte_pos++] = (uint8_t)(bw->bits >> 56);
        bw->bits <<= 8;
        bw->nbits -= 8;
    }
}

static inline void bw_put(bitwriter_t *bw, uint16_t code, uint8_t len)
{
    bw->bits |= (uint64_t)code << (64 - bw->nbits - len);
    bw->nbits += len;
    if (bw->nbits >= 32) bw_flush(bw);
}

static inline void bw_finish(bitwriter_t *bw)
{
    bw_flush(bw);
    if (bw->nbits > 0) {
        bw->buf[bw->byte_pos++] = (uint8_t)(bw->bits >> 56);
        bw->nbits = 0;
        bw->bits = 0;
    }
}

/* ================================================================
 * Packed decode table — adaptive table_log = max_code_len.
 *
 * Each entry packs (symbol, nbits) into one uint16_t for cache density.
 * Table size = 2^max_code_len entries.  For typical distributions:
 *   max 8 bits → 256 entries (512 B)
 *   max 11 bits → 2048 entries (4 KB)
 *   max 15 bits → 32768 entries (64 KB)
 * ================================================================ */

typedef uint16_t dtentry_t;   /* low 8 = symbol, high 8 = nbits */

#define DTE_SYM(e)  ((uint8_t)((e) & 0xFF))
#define DTE_BITS(e) ((uint8_t)((e) >> 8))
#define DTE_MAKE(sym, bits) ((uint16_t)(sym) | ((uint16_t)(bits) << 8))

typedef struct {
    dtentry_t entries[1 << PIVCO_MAX_CODE_LEN];
    int       table_log;
    int       table_size;
} fast_table_t;

static void build_fast_table(const pivco_table_t *ht, fast_table_t *ft)
{
    ft->table_log  = ht->max_len;
    ft->table_size = 1 << ft->table_log;

    int shift = PIVCO_MAX_CODE_LEN - ft->table_log;
    for (int i = 0; i < ft->table_size; i++) {
        int full_idx = i << shift;
        ft->entries[i] = DTE_MAKE(ht->decode_sym[full_idx],
                                  ht->decode_len[full_idx]);
    }
}

/* Decode one symbol from a bitreader using the fast table */
static inline uint8_t fast_decode(bitreader_t *br, const fast_table_t *ft)
{
    uint32_t idx = br_peek(br, ft->table_log);
    dtentry_t e = ft->entries[idx];
    br_consume(br, DTE_BITS(e));
    return DTE_SYM(e);
}

/* ================================================================
 * Traditional Huffman Encode (single-stream, used for both modes)
 * ================================================================ */

int trad_huffman_encode(const uint8_t *symbols, size_t n_symbols,
                        const pivco_table_t *table,
                        uint8_t *out, size_t *out_len, size_t *out_bits)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    bitwriter_t bw;
    bw_init(&bw, out);

    size_t total_bits = 0;
    for (size_t i = 0; i < n_symbols; i++) {
        uint8_t sym = symbols[i];
        bw_put(&bw, table->code[sym], table->code_len[sym]);
        total_bits += table->code_len[sym];
    }

    bw_finish(&bw);
    *out_len = bw.byte_pos;
    if (out_bits) *out_bits = total_bits;
    return PIVCO_OK;
}

/* ================================================================
 * Traditional Huffman Decode — single-stream, 15-bit flat table.
 * Kept for backward compatibility / reference.
 * ================================================================ */

int trad_huffman_decode(const uint8_t *in, size_t in_bits,
                        const pivco_table_t *table,
                        uint8_t *symbols, size_t n_symbols)
{
    if (!in || !table || !symbols) return PIVCO_ERR_NULL;

    size_t in_len = (in_bits + 7) / 8;
    bitreader_t br;
    br_init(&br, in, in_len);
    br_refill_safe(&br);

    for (size_t i = 0; i < n_symbols; i++) {
        if (br.nbits < PIVCO_MAX_CODE_LEN)
            br_refill_safe(&br);
        uint32_t idx = br_peek(&br, PIVCO_MAX_CODE_LEN);
        br_consume(&br, table->decode_len[idx]);
        symbols[i] = table->decode_sym[idx];
    }

    return PIVCO_OK;
}

/* ================================================================
 * SotA Huffman — 4-stream encode/decode (huff0-style)
 *
 * Input is split into 4 equal-size streams for instruction-level
 * parallelism.  The OoO CPU can overlap the serial dependency
 * chains of all 4 streams, roughly 2-3x faster than single-stream.
 *
 * Encoded format:
 *   [stream1_bytes: 2B] [stream2_bytes: 2B] [stream3_bytes: 2B]
 *   [stream1 data] [stream2 data] [stream3 data] [stream4 data]
 *   (+ 8 bytes padding for branchless refill safety)
 *
 * Each stream is encoded MSB-first, same as single-stream.
 * Stream 4 size is implicit (remainder).
 * ================================================================ */

#define STREAM_HEADER_SIZE 6   /* 3 × uint16_t */

int trad_huffman_encode_4s(const uint8_t *symbols, size_t n_symbols,
                           const pivco_table_t *table,
                           uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    size_t quarter = n_symbols / 4;
    /* Streams: [0..q-1] [q..2q-1] [2q..3q-1] [3q..n-1] */
    const uint8_t *src[4] = {
        symbols, symbols + quarter, symbols + 2*quarter, symbols + 3*quarter
    };
    size_t slen[4] = { quarter, quarter, quarter, n_symbols - 3*quarter };

    /* Encode each stream into a temporary buffer */
    uint8_t tmp[4][PIVCO_BLOCK_SIZE]; /* generous per-stream buffer */
    size_t  stream_bytes[4];
    size_t  stream_bits[4];

    for (int s = 0; s < 4; s++) {
        bitwriter_t bw;
        bw_init(&bw, tmp[s]);
        size_t bits = 0;
        for (size_t i = 0; i < slen[s]; i++) {
            uint8_t sym = src[s][i];
            bw_put(&bw, table->code[sym], table->code_len[sym]);
            bits += table->code_len[sym];
        }
        bw_finish(&bw);
        stream_bytes[s] = bw.byte_pos;
        stream_bits[s] = bits;
    }

    /* Write header: 3 stream sizes */
    uint8_t *ptr = out;
    for (int s = 0; s < 3; s++) {
        ptr[0] = (uint8_t)(stream_bytes[s] & 0xFF);
        ptr[1] = (uint8_t)(stream_bytes[s] >> 8);
        ptr += 2;
    }

    /* Write stream data */
    for (int s = 0; s < 4; s++) {
        memcpy(ptr, tmp[s], stream_bytes[s]);
        ptr += stream_bytes[s];
    }

    /* 8 bytes of zero padding for branchless refill safety */
    memset(ptr, 0, 8);
    ptr += 8;

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

int trad_huffman_decode_4s(const uint8_t *in, size_t in_len,
                           const pivco_table_t *table,
                           uint8_t *symbols, size_t n_symbols)
{
    if (!in || !table || !symbols) return PIVCO_ERR_NULL;
    if (in_len < STREAM_HEADER_SIZE) return PIVCO_ERR_CORRUPT;

    /* Build adaptive packed table */
    fast_table_t ft;
    build_fast_table(table, &ft);
    int tlog = ft.table_log;

    /* Read stream sizes from header */
    const uint8_t *hdr = in;
    size_t s_bytes[4];
    s_bytes[0] = (size_t)hdr[0] | ((size_t)hdr[1] << 8);
    s_bytes[1] = (size_t)hdr[2] | ((size_t)hdr[3] << 8);
    s_bytes[2] = (size_t)hdr[4] | ((size_t)hdr[5] << 8);

    const uint8_t *s_data[4];
    s_data[0] = in + STREAM_HEADER_SIZE;
    s_data[1] = s_data[0] + s_bytes[0];
    s_data[2] = s_data[1] + s_bytes[1];
    s_data[3] = s_data[2] + s_bytes[2];
    /* Stream 4 size = remainder (minus 8 bytes padding) */
    s_bytes[3] = (size_t)((in + in_len - 8) - s_data[3]);

    size_t quarter = n_symbols / 4;
    size_t slen[4] = { quarter, quarter, quarter, n_symbols - 3*quarter };

    /* Output pointers for each stream */
    uint8_t *op[4] = {
        symbols,
        symbols + quarter,
        symbols + 2*quarter,
        symbols + 3*quarter
    };

    /* Init 4 bit readers */
    bitreader_t br[4];
    for (int s = 0; s < 4; s++) {
        br_init(&br[s], s_data[s], s_bytes[s]);
        br_refill(&br[s]);
    }

    /* Interleaved 4-stream decode loop.
       Decode 2 symbols per stream per round (8 total), then refill all 4.
       With tlog <= 15, each symbol consumes at most 15 bits.
       2 rounds × 15 bits = 30 bits max consumed before refill.
       After refill we have >= 56 bits, so 2 rounds is safe. */
    size_t min_len = slen[0]; /* all quarters are equal except possibly stream 3 */
    size_t i = 0;

    for (; i + 2 <= min_len; i += 2) {
        /* Round 1: 4 decodes */
        op[0][i]   = fast_decode(&br[0], &ft);
        op[1][i]   = fast_decode(&br[1], &ft);
        op[2][i]   = fast_decode(&br[2], &ft);
        op[3][i]   = fast_decode(&br[3], &ft);

        /* Round 2: 4 decodes */
        op[0][i+1] = fast_decode(&br[0], &ft);
        op[1][i+1] = fast_decode(&br[1], &ft);
        op[2][i+1] = fast_decode(&br[2], &ft);
        op[3][i+1] = fast_decode(&br[3], &ft);

        /* Refill all 4 */
        br_refill(&br[0]);
        br_refill(&br[1]);
        br_refill(&br[2]);
        br_refill(&br[3]);
    }

    /* Remainder: main loop decoded i symbols per stream.
       Streams 0-2 have exactly quarter symbols, stream 3 may have more. */
    for (int s = 0; s < 4; s++) {
        for (size_t j = i; j < slen[s]; j++) {
            if (br[s].nbits < tlog) br_refill_safe(&br[s]);
            op[s][j] = fast_decode(&br[s], &ft);
        }
    }

    return PIVCO_OK;
}
