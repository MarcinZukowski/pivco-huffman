/* bench_brotli.c -- encode/decode throughput of Brotli's pure-Huffman
 * primitives, alongside pivco and huf0.
 *
 * Brotli is the algorithm family libjxl ported its Huffman from (libjxl's
 * `dec_huffman.cc` is essentially a Brotli re-implementation).  We vendor
 * Brotli rather than libjxl because the C surface is ~5x smaller for the
 * same data point.  Single-stream decode, matches huf0_x1's shape.
 *
 * Reuses the existing 8192-symbol block layout.  Per block: build the
 * Brotli decode table (HuffmanCode[]), then encode + decode.  Bit
 * reader/writer is local (Brotli's own writer is tangled with the
 * higher-level meta-block format).
 */
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include "pivco_huffman.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Brotli vendored headers. */
#include "huffman.h"          /* HuffmanCode, BrotliBuildHuffmanTable */
#include "entropy_encode.h"   /* BrotliConvertBitDepthsToSymbols */

/* bench distribution API (defined in bench_distributions.c). */
extern void           bench_init(void);
extern int            bench_num_distributions(void);
extern const char    *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int            bench_dist_is_main(int idx);
extern void           bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                              int n_symbols, uint64_t seed);

#define BLK 8192
#define HUF0_CHUNK (128 * 1024)
#define TOTAL_SYMBOLS (4 * 1024 * 1024)
#define ROOT_BITS 8
#define TABLE_SIZE 2048   /* worst-case 2-level table size at root_bits=8 */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* --- Brotli table build (symbol_lists is a per-length linked-list
 * structure; see BrotliBuildHuffmanTable). ---------------------- */
typedef struct {
    HuffmanCode table[TABLE_SIZE];
    uint16_t bits[256];        /* canonical codes, LSB-first */
    uint8_t  depth[256];       /* code length, 0 = unused */
} brotli_table_t;

static void brotli_table_build(brotli_table_t *bt,
                                const pivco_huffman_table_t *pt) {
    memcpy(bt->depth, pt->code_len, sizeof bt->depth);
    BrotliConvertBitDepthsToSymbols(bt->depth, 256, bt->bits);

    uint16_t storage[16 + 256];
    uint16_t *symbol_lists = storage + 16;  /* allows symbol_lists[-16..255] */
    for (int i = -16; i < 256; i++) symbol_lists[i] = 0xFFFF;
    uint16_t count[16] = {0};
    int next_symbol[16];
    for (int i = 0; i < 16; i++) next_symbol[i] = i - 16;

    for (int s = 0; s < 256; s++) {
        uint8_t L = bt->depth[s];
        if (L == 0 || L > 15) continue;
        symbol_lists[next_symbol[L]] = (uint16_t)s;
        next_symbol[L] = s;
        count[L]++;
    }
    BrotliBuildHuffmanTable(bt->table, ROOT_BITS, symbol_lists, count);
}

/* --- LSB-first bit writer.  Accumulate up to 56 bits in a uint64_t,
 * drain a byte at a time as we go.  Avoids the "next write clobbers
 * upper bytes of previous write" hazard of a memory-direct writer.  --- */
typedef struct {
    uint8_t *buf;
    size_t   pos;
    uint64_t bit_buf;
    int      n_bits;
} bitwriter_t;

static inline void bw_init(bitwriter_t *w, uint8_t *buf) {
    w->buf = buf; w->pos = 0; w->bit_buf = 0; w->n_bits = 0;
}
static inline void bw_write(bitwriter_t *w, uint32_t value, int nbits) {
    w->bit_buf |= (uint64_t)value << w->n_bits;
    w->n_bits += nbits;
    while (w->n_bits >= 8) {
        w->buf[w->pos++] = (uint8_t)(w->bit_buf & 0xFF);
        w->bit_buf >>= 8;
        w->n_bits -= 8;
    }
}
static inline size_t bw_finish(bitwriter_t *w) {
    if (w->n_bits > 0) {
        w->buf[w->pos++] = (uint8_t)(w->bit_buf & ((1u << w->n_bits) - 1));
    }
    return w->pos;
}

/* --- LSB-first bit reader.  Matches brotli's read order. ------------- */
typedef struct {
    const uint8_t *buf;
    size_t         byte_pos;
    int            bit_pos;
} bitreader_t;

static inline void br_init(bitreader_t *r, const uint8_t *buf) {
    r->buf = buf; r->byte_pos = 0; r->bit_pos = 0;
}
static inline uint32_t br_peek(const bitreader_t *r, int nbits) {
    uint64_t v;
    memcpy(&v, r->buf + r->byte_pos, sizeof v);
    return (uint32_t)((v >> r->bit_pos) & ((1u << nbits) - 1));
}
static inline void br_consume(bitreader_t *r, int nbits) {
    int total = r->bit_pos + nbits;
    r->byte_pos += total >> 3;
    r->bit_pos   = total & 7;
}

/* --- Encode / decode one block via brotli's primitives. -------------- */
static size_t brotli_encode_block(const uint8_t *symbols, int n,
                                   const brotli_table_t *bt,
                                   uint8_t *out)
{
    bitwriter_t w;
    bw_init(&w, out);
    for (int i = 0; i < n; i++) {
        uint8_t s = symbols[i];
        bw_write(&w, bt->bits[s], bt->depth[s]);
    }
    return bw_finish(&w);
}

static void brotli_decode_block(const uint8_t *in, int n,
                                 const brotli_table_t *bt,
                                 uint8_t *out)
{
    bitreader_t r;
    br_init(&r, in);
    const HuffmanCode *root = bt->table;
    for (int i = 0; i < n; i++) {
        uint32_t bits = br_peek(&r, 16);
        int idx = bits & ((1u << ROOT_BITS) - 1);
        HuffmanCode hc = root[idx];
        if (hc.bits > ROOT_BITS) {
            /* The root entry says "subtable width = hc.bits - ROOT_BITS"
             * and "subtable at offset hc.value (relative to idx)".
             * Subtable entries store bits = actual_code_length - ROOT_BITS,
             * which for codes shorter than the subtable max is < the
             * subtable width.  Total to consume = ROOT_BITS + sub.bits. */
            int sub_width = hc.bits - ROOT_BITS;
            hc = root[idx + hc.value + ((bits >> ROOT_BITS) & ((1u << sub_width) - 1))];
            br_consume(&r, ROOT_BITS + hc.bits);
        } else {
            br_consume(&r, hc.bits);
        }
        out[i] = (uint8_t)hc.value;
    }
}

/* --- bench driver --------------------------------------------------- */
typedef struct {
    const char *name;
    double pivco_enc_mps, pivco_dec_mps;
    double brotli_enc_mps, brotli_dec_mps;
    double huf0_enc_mps, huf0_dec_mps;
    int roundtrip_ok;
    size_t pivco_bytes, brotli_bytes, huf0_bytes;
} dist_result_t;

static void bench_one_dist(int dist_idx, int repeats, dist_result_t *out)
{
    const char *name = bench_dist_name(dist_idx);
    const uint64_t *freq = bench_dist_freq(dist_idx);
    out->name = name;
    out->roundtrip_ok = 1;

    pivco_huffman_table_t table;
    if (pivco_huffman_build_table(freq, &table) != PIVCO_OK) {
        fprintf(stderr, "  %s: build_table failed\n", name);
        out->roundtrip_ok = 0;
        return;
    }

    uint8_t *symbols = (uint8_t *)malloc(TOTAL_SYMBOLS);
    bench_generate_symbols(dist_idx, symbols, TOTAL_SYMBOLS, 0xBEEFCAFE);

    brotli_table_t bt;
    brotli_table_build(&bt, &table);

    /* Output buffers per block.  Brotli output worst-case = 11 bits/sym
     * for our length cap, so 11*8192/8 + slack = ~11.5 KB. */
    size_t enc_slot = 16384;
    int blocks = TOTAL_SYMBOLS / BLK;
    uint8_t *pivco_enc  = (uint8_t *)malloc((size_t)blocks * enc_slot);
    uint8_t *brotli_enc = (uint8_t *)malloc((size_t)blocks * enc_slot);
    size_t   *pivco_lens  = (size_t *)malloc((size_t)blocks * sizeof(size_t));
    size_t   *brotli_lens = (size_t *)malloc((size_t)blocks * sizeof(size_t));

    /* Pre-encode for decode timing + roundtrip sanity. */
    for (int b = 0; b < blocks; b++) {
        size_t enc_len;
        if (pivco_huffman_encode(symbols + (size_t)b * BLK, &table,
                                 pivco_enc + (size_t)b * enc_slot, &enc_len) != PIVCO_OK)
        { out->roundtrip_ok = 0; goto cleanup; }
        pivco_lens[b] = enc_len;
        brotli_lens[b] = brotli_encode_block(symbols + (size_t)b * BLK, BLK, &bt,
                                              brotli_enc + (size_t)b * enc_slot);
    }
    /* Roundtrip sanity check on first block. */
    {
        uint8_t dec[BLK];
        brotli_decode_block(brotli_enc, BLK, &bt, dec);
        if (memcmp(symbols, dec, BLK) != 0) {
            fprintf(stderr, "  %s: brotli roundtrip FAILED\n", name);
            out->roundtrip_ok = 0;
            goto cleanup;
        }
    }
    out->pivco_bytes  = 0;
    out->brotli_bytes = 0;
    for (int b = 0; b < blocks; b++) {
        out->pivco_bytes  += pivco_lens[b];
        out->brotli_bytes += brotli_lens[b];
    }

    /* --- ENCODE timing ---------------------------------------------- */
    double t0, t1;

    t0 = now_sec();
    for (int r = 0; r < repeats; r++) {
        for (int b = 0; b < blocks; b++) {
            size_t enc_len;
            pivco_huffman_encode(symbols + (size_t)b * BLK, &table,
                                 pivco_enc + (size_t)b * enc_slot, &enc_len);
        }
    }
    t1 = now_sec();
    out->pivco_enc_mps = (double)blocks * BLK * repeats / (t1 - t0) / 1e6;

    t0 = now_sec();
    for (int r = 0; r < repeats; r++) {
        for (int b = 0; b < blocks; b++) {
            brotli_encode_block(symbols + (size_t)b * BLK, BLK, &bt,
                                 brotli_enc + (size_t)b * enc_slot);
        }
    }
    t1 = now_sec();
    out->brotli_enc_mps = (double)blocks * BLK * repeats / (t1 - t0) / 1e6;

    /* huf0 reference: HUF_compress1X over 128KB chunks. */
    int huf0_nchunks = TOTAL_SYMBOLS / HUF0_CHUNK;
    size_t huf0_slot = HUF0_CHUNK + 1024;
    uint8_t *huf0_enc = (uint8_t *)malloc((size_t)huf0_nchunks * huf0_slot);
    size_t huf0_total = 0;
    for (int c = 0; c < huf0_nchunks; c++) {
        size_t r = HUF_compress1X(huf0_enc + (size_t)c * huf0_slot, huf0_slot,
                                   symbols + (size_t)c * HUF0_CHUNK,
                                   HUF0_CHUNK, 255, 11);
        if (HUF_isError(r) || r == 0) { huf0_total = 0; break; }
        huf0_total += r;
    }
    out->huf0_bytes = huf0_total;
    t0 = now_sec();
    for (int r = 0; r < repeats; r++) {
        for (int c = 0; c < huf0_nchunks; c++) {
            HUF_compress1X(huf0_enc + (size_t)c * huf0_slot, huf0_slot,
                           symbols + (size_t)c * HUF0_CHUNK,
                           HUF0_CHUNK, 255, 11);
        }
    }
    t1 = now_sec();
    out->huf0_enc_mps = (double)TOTAL_SYMBOLS * repeats / (t1 - t0) / 1e6;

    /* --- DECODE timing ---------------------------------------------- */
    uint8_t *dec_buf = (uint8_t *)malloc(TOTAL_SYMBOLS);

    t0 = now_sec();
    for (int r = 0; r < repeats; r++) {
        for (int b = 0; b < blocks; b++) {
            size_t consumed;
            pivco_huffman_decode(pivco_enc + (size_t)b * enc_slot, pivco_lens[b],
                                 &table, dec_buf + (size_t)b * BLK, &consumed);
        }
    }
    t1 = now_sec();
    out->pivco_dec_mps = (double)blocks * BLK * repeats / (t1 - t0) / 1e6;

    t0 = now_sec();
    for (int r = 0; r < repeats; r++) {
        for (int b = 0; b < blocks; b++) {
            brotli_decode_block(brotli_enc + (size_t)b * enc_slot, BLK, &bt,
                                 dec_buf + (size_t)b * BLK);
        }
    }
    t1 = now_sec();
    out->brotli_dec_mps = (double)blocks * BLK * repeats / (t1 - t0) / 1e6;

    if (huf0_total > 0) {
        size_t *huf0_off = (size_t *)malloc((size_t)huf0_nchunks * sizeof(size_t));
        size_t acc = 0;
        for (int c = 0; c < huf0_nchunks; c++) {
            huf0_off[c] = acc;
            size_t r = HUF_compress1X(huf0_enc + acc, huf0_slot,
                                       symbols + (size_t)c * HUF0_CHUNK,
                                       HUF0_CHUNK, 255, 11);
            acc += r;
        }
        t0 = now_sec();
        for (int r = 0; r < repeats; r++) {
            for (int c = 0; c < huf0_nchunks; c++) {
                size_t enc_len = (c + 1 < huf0_nchunks ? huf0_off[c+1] - huf0_off[c]
                                                       : acc - huf0_off[c]);
                HUF_decompress1X1(dec_buf + (size_t)c * HUF0_CHUNK, HUF0_CHUNK,
                                  huf0_enc + huf0_off[c], enc_len);
            }
        }
        t1 = now_sec();
        out->huf0_dec_mps = (double)TOTAL_SYMBOLS * repeats / (t1 - t0) / 1e6;
        free(huf0_off);
    } else {
        out->huf0_dec_mps = 0;
    }
    free(dec_buf);
    free(huf0_enc);

cleanup:
    free(pivco_enc); free(brotli_enc);
    free(pivco_lens); free(brotli_lens);
    free(symbols);
}

int main(int argc, char **argv)
{
    int repeats = (argc > 1) ? atoi(argv[1]) : 10;

    bench_init();
    int n_dists = bench_num_distributions();

    printf("=== Brotli vs pivco vs huf0_x1 (single-stream Huffman) ===\n");
    printf("Sequence: %d × %d-symbol blocks, repeats=%d\n\n",
           TOTAL_SYMBOLS / BLK, BLK, repeats);
    printf("%-13s | ENC M/s pvc / brt / h0_1s | DEC M/s pvc / brt / h0_1s | "
           "b/h0_1s | bytes/sym pvc / brt / h0_1s\n", "dist");
    printf("--------------+--------------------------+--------------------------+"
           "---------+----------------------------\n");

    for (int i = 0; i < n_dists; i++) {
        if (!bench_dist_is_main(i)) continue;
        dist_result_t r;
        bench_one_dist(i, repeats, &r);
        if (!r.roundtrip_ok) {
            printf("%-13s | (skipped)\n", r.name);
            continue;
        }
        double dec_ratio = r.huf0_dec_mps > 0 ? r.brotli_dec_mps / r.huf0_dec_mps : 0;
        printf("%-13s | %6.0f / %5.0f / %5.0f | %6.0f / %5.0f / %5.0f | "
               "%5.2fx  | %4.2f / %4.2f / %4.2f\n",
               r.name,
               r.pivco_enc_mps, r.brotli_enc_mps, r.huf0_enc_mps,
               r.pivco_dec_mps, r.brotli_dec_mps, r.huf0_dec_mps,
               dec_ratio,
               (double)r.pivco_bytes  / TOTAL_SYMBOLS,
               (double)r.brotli_bytes / TOTAL_SYMBOLS,
               (double)r.huf0_bytes   / TOTAL_SYMBOLS);
    }
    return 0;
}
