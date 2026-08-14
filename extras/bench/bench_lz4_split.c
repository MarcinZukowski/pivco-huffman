/* bench_lz4_split — LZ4 output stream analyser + per-stream entropy
 * comparison.  Motivated by the IDEAS "LZ4 + ph" research direction
 * (results/lz4-vs-raw-m4-20260517.md).
 *
 * For each input file:
 *
 *   1. LZ4-HC compress (level 9, same as `lz4 -9`).
 *   2. Walk the LZ4 block format and split the bytestream into:
 *        - LITERALS: verbatim source bytes inside each sequence.
 *        - HEADERS : everything else — token bytes + literal-length
 *                    overflow + 16-bit match offsets + match-length
 *                    overflow.
 *   3. For each of {full lz4, literals, headers}: report size, Shannon
 *      byte-entropy H0 (bits/byte), and the actual byte-Huffman ratio
 *      (length-limited at PIVCO_MAX_CODE_LEN to match what ph would
 *      ship).
 *   4. Bench huff0 and ph (--no-fse) decode throughput on each stream.
 *      Compressed sizes already covered by step 3; here we measure
 *      decode MB/s for both libraries on the same byte buffer.
 *
 * This is the "analyser-only" first step in the LZ4+ph integration:
 * tells us per-dataset which stream is worth entropy-coding and which
 * isn't, and how huf0 vs ph compare on each.  No persistent
 * wire-format work yet — just observation.
 *
 * Build:   cmake --build build --target pivco_bench_lz4_split
 * Run:     ./build/pivco_bench_lz4_split [iters]    # default 200
 *
 *          # or override the dataset list:
 *          ./build/pivco_bench_lz4_split 500 path/to/file ...
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#define FSE_STATIC_LINKING_ONLY
#define HUF_STATIC_LINKING_ONLY
#include "fse.h"
#include "huf.h"

#include "lz4.h"
#include "lz4hc.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================
 *  LZ4 block-format walker.
 *
 *  Per-sequence layout (LZ4 block format, see lz4_block_format.md):
 *    token   : 1 byte  (high nibble = literal len; low nibble =
 *                       match len - MINMATCH).
 *    lit_ext : 0+ bytes of literal-length overflow when high
 *              nibble == 15.
 *    lits    : `lit_len` literal bytes.
 *    offset  : 2 bytes LE (skipped on the final sequence — last
 *              sequence is literals-only).
 *    mat_ext : 0+ bytes of match-length overflow when low nibble
 *              == 15.
 * ============================================================ */

/* Per-sequence semantic-field arrays.  Headers stream decomposed
 * into: token-bytes (one per seq), low/high offset bytes (two per
 * seq), and 8-bit-capped length-CODES (the high-nibble of the token,
 * which is the bucketed prefix of the actual length; this is what
 * zstd-style entropy coders work with rather than the raw length-
 * overflow byte stream).  Match-length code is the low nibble of the
 * token, similarly. */
typedef struct {
    uint8_t *tokens;       /* one byte per sequence */
    uint8_t *off_lo;       /* low byte of each offset */
    uint8_t *off_hi;       /* high byte of each offset */
    uint8_t *lit_codes;    /* high-nibble buckets 0..15 */
    uint8_t *mat_codes;    /* low-nibble buckets 0..15 */
    size_t   n;            /* number of valid entries */
    size_t   cap;
} hdr_fields_t;

static int lz4_split(const uint8_t *lz4_buf, size_t lz4_size,
                      uint8_t *lits_out, size_t lits_cap,
                      uint8_t *hdrs_out, size_t hdrs_cap,
                      size_t *lits_len_out, size_t *hdrs_len_out,
                      size_t *n_seqs_out,
                      hdr_fields_t *hf /* optional */)
{
    const uint8_t *p   = lz4_buf;
    const uint8_t *end = lz4_buf + lz4_size;
    size_t lits = 0, hdrs = 0, n_seqs = 0;

    while (p < end) {
        /* token */
        if (hdrs >= hdrs_cap) return -1;
        hdrs_out[hdrs++] = *p;
        uint8_t token = *p++;

        if (hf && hf->n < hf->cap) {
            hf->tokens   [hf->n] = token;
            hf->lit_codes[hf->n] = token >> 4;
            hf->mat_codes[hf->n] = token & 0xf;
        }

        /* literal length (with overflow) */
        size_t lit_len = token >> 4;
        if (lit_len == 15) {
            while (p < end && *p == 255) {
                if (hdrs >= hdrs_cap) return -1;
                hdrs_out[hdrs++] = *p;
                lit_len += 255;
                p++;
            }
            if (p < end) {
                if (hdrs >= hdrs_cap) return -1;
                hdrs_out[hdrs++] = *p;
                lit_len += *p++;
            }
        }

        /* literal bytes */
        if (lit_len > 0) {
            if (lits + lit_len > lits_cap) return -2;
            if (p + lit_len > end)         return -3;
            memcpy(lits_out + lits, p, lit_len);
            lits += lit_len;
            p    += lit_len;
        }

        if (p >= end) break;   /* last sequence — literals only */

        /* offset (2 bytes LE) */
        if (p + 2 > end) return -4;
        if (hf && hf->n < hf->cap) {
            hf->off_lo[hf->n] = p[0];
            hf->off_hi[hf->n] = p[1];
        }
        if (hdrs + 2 > hdrs_cap) return -1;
        hdrs_out[hdrs++] = *p++;
        hdrs_out[hdrs++] = *p++;

        /* match length (with overflow) */
        size_t mat_len = token & 0xf;
        if (mat_len == 15) {
            while (p < end && *p == 255) {
                if (hdrs >= hdrs_cap) return -1;
                hdrs_out[hdrs++] = *p;
                mat_len += 255;
                p++;
            }
            if (p < end) {
                if (hdrs >= hdrs_cap) return -1;
                hdrs_out[hdrs++] = *p;
                mat_len += *p++;
            }
        }

        if (hf && hf->n < hf->cap) hf->n++;
        n_seqs++;
    }
    *lits_len_out = lits;
    *hdrs_len_out = hdrs;
    *n_seqs_out   = n_seqs;
    return 0;
}

/* ============================================================
 *  Byte-frequency analyser: entropy + length-limited Huffman ratio.
 * ============================================================ */

static double shannon_entropy(const uint8_t *buf, size_t len, uint64_t freq[256])
{
    memset(freq, 0, 256 * sizeof(uint64_t));
    for (size_t i = 0; i < len; i++) freq[buf[i]]++;
    double H = 0.0;
    for (int s = 0; s < 256; s++) {
        if (freq[s] == 0) continue;
        double p = (double)freq[s] / (double)len;
        H -= p * log2(p);
    }
    return H;
}

/* Byte-Huffman ratio: build ph's length-limited Huffman tree on the
 * actual byte frequencies and return encoded_bits / (8 * len). */
static double huffman_ratio(const uint8_t *buf, size_t len)
{
    if (len == 0) return 0.0;
    uint64_t freq[256];
    (void)shannon_entropy(buf, len, freq);
    pivco_table_t table;
    if (pivco_build_table(bench_cfg(), freq, &table) != PIVCO_OK) return -1.0;
    uint64_t total_bits = 0;
    for (int s = 0; s < 256; s++) {
        if (freq[s] > 0)
            total_bits += freq[s] * (uint64_t)table.code_len[s];
    }
    return (double)total_bits / (8.0 * (double)len);
}

/* FSE byte-level ratio: encode with FSE_compress (Yann's tANS, within
 * ~0.01 bits/byte of entropy), return encoded_bytes / len.  Returns
 * negative if FSE bails (incompressible). */
static double fse_byte_ratio(const uint8_t *buf, size_t len)
{
    if (len == 0) return 0.0;
    size_t cap = FSE_compressBound(len);
    uint8_t *dst = (uint8_t *)malloc(cap);
    size_t r = FSE_compress(dst, cap, buf, len);
    double ratio;
    if (FSE_isError(r))     ratio = -1.0;
    else if (r == 0)        ratio = -2.0;   /* incompressible */
    else if (r == 1)        ratio = -3.0;   /* RLE-only */
    else                    ratio = (double)r / (double)len;
    free(dst);
    return ratio;
}

/* ============================================================
 *  Per-stream decode-throughput bench (huff0 vs ph, no FSE).
 *
 *  Both codecs are timed on the same byte buffer.  ph encodes in
 *  PIVCO_BLOCK_SIZE-byte blocks; non-multiples are padded with the
 *  last byte (preserves the dominant-symbol skew so the encoded size
 *  approximates the unpadded stream).  huff0 chunks at 128 KB.
 *
 *  Returns: encoded_size + decode MB/s for each codec.
 * ============================================================ */

#define HUF0_CHUNK    (128 * 1024)
#define N_BATCHES     5

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

typedef struct {
    size_t enc_size;
    double dec_mbps;
    int    ok;
    const char *note;
} codec_result_t;

static codec_result_t bench_huf0(const uint8_t *src, size_t len, int iters)
{
    codec_result_t r = {0};
    if (len < 32) { r.note = "too small"; return r; }

    int n_chunks = (int)((len + HUF0_CHUNK - 1) / HUF0_CHUNK);
    uint8_t *enc = (uint8_t *)malloc((size_t)n_chunks * (HUF0_CHUNK + 1024));
    size_t  *enc_off = (size_t *)calloc((size_t)(n_chunks + 1), sizeof(size_t));
    for (int c = 0; c < n_chunks; c++) {
        size_t chunk_sz = (c < n_chunks - 1) ? HUF0_CHUNK
                         : len - (size_t)c * HUF0_CHUNK;
        size_t cr = HUF_compress(enc + enc_off[c], chunk_sz + 1024,
                                  src + (size_t)c * HUF0_CHUNK, chunk_sz);
        if (HUF_isError(cr) || cr == 0) {
            r.note = "incompressible";
            free(enc); free(enc_off);
            return r;
        }
        enc_off[c + 1] = enc_off[c] + cr;
    }
    r.enc_size = enc_off[n_chunks];

    uint8_t *dec = (uint8_t *)malloc(len + 64);
    double best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            for (int c = 0; c < n_chunks; c++) {
                size_t chunk_sz = (c < n_chunks - 1) ? HUF0_CHUNK
                                 : len - (size_t)c * HUF0_CHUNK;
                HUF_decompress(dec + (size_t)c * HUF0_CHUNK, chunk_sz,
                                enc + enc_off[c],
                                enc_off[c + 1] - enc_off[c]);
            }
            sink ^= dec[0] ^ dec[len - 1];
        }
        double t1 = now_ns();
        (void)sink;
        double mb = (double)len * (double)iters / (t1 - t0) * 1e3;
        if (mb > best) best = mb;
    }
    r.dec_mbps = best;
    r.ok = 1;
    free(enc); free(enc_off); free(dec);
    return r;
}

static codec_result_t bench_ph(const uint8_t *src, size_t len, int iters)
{
    codec_result_t r = {0};
    if (len < 32) { r.note = "too small"; return r; }

    size_t padded = ((len + PIVCO_BLOCK_SIZE - 1) / PIVCO_BLOCK_SIZE)
                    * PIVCO_BLOCK_SIZE;
    int n_blocks = (int)(padded / PIVCO_BLOCK_SIZE);

    uint8_t *padded_src = (uint8_t *)malloc(padded);
    memcpy(padded_src, src, len);
    /* Tile-fill the pad region with bytes from src so the byte
     * distribution is preserved (preserves ph's measured ratio on
     * short streams, which would otherwise see a giant run of zeros). */
    for (size_t off = len; off < padded; off++) {
        padded_src[off] = src[off % len];
    }

    uint64_t freq[256];
    (void)shannon_entropy(padded_src, padded, freq);
    pivco_table_t table;
    if (pivco_build_table(bench_cfg(), freq, &table) != PIVCO_OK) {
        r.note = "build_table failed";
        free(padded_src);
        return r;
    }

    size_t enc_cap_per = PIVCO_BLOCK_SIZE * 2 + 256;
    uint8_t *enc = (uint8_t *)malloc((size_t)n_blocks * enc_cap_per);
    size_t  *enc_off = (size_t *)calloc((size_t)(n_blocks + 1), sizeof(size_t));
    for (int b = 0; b < n_blocks; b++) {
        size_t out_len = enc_cap_per;
        int rc = pivco_encode(bench_enc_ctx(), &table, padded_src + (size_t)b * PIVCO_BLOCK_SIZE, PIVCO_BLOCK_SIZE, enc + enc_off[b], &out_len);
        if (rc != PIVCO_OK) {
            r.note = "encode failed";
            free(enc); free(enc_off); free(padded_src);
            return r;
        }
        enc_off[b + 1] = enc_off[b] + out_len;
    }
    r.enc_size = enc_off[n_blocks];

    uint8_t *dec = (uint8_t *)malloc(padded + 64);
    double best = 0.0;
    for (int batch = 0; batch < N_BATCHES; batch++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            for (int b = 0; b < n_blocks; b++) {
                size_t consumed = 0;
                pivco_decode(bench_dec_ctx(), &table, enc + enc_off[b], enc_off[b + 1] - enc_off[b], dec + (size_t)b * PIVCO_BLOCK_SIZE, &consumed);
            }
            sink ^= dec[0] ^ dec[padded - 1];
        }
        double t1 = now_ns();
        (void)sink;
        double mb = (double)padded * (double)iters / (t1 - t0) * 1e3;
        if (mb > best) best = mb;
    }
    r.dec_mbps = best;
    r.ok = 1;
    free(enc); free(enc_off); free(dec); free(padded_src);
    return r;
}


/* ============================================================
 *  Main driver
 * ============================================================ */

static const char *DEFAULT_FILES[] = {
    "extras/datasets/cat-wiki.html",
    "extras/datasets/pride.txt",
    "extras/datasets/cat-image.jpg",
    "extras/datasets/json_api.json",
    "extras/datasets/chinese_text.txt",
    "extras/datasets/calgary_pic",
    "extras/datasets/gzip_random.gz",
};

static int read_file(const char *path, uint8_t **buf, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return -2; }
    fseek(f, 0, SEEK_SET);
    *buf = (uint8_t *)malloc((size_t)sz);
    if (!*buf) { fclose(f); return -3; }
    size_t got = fread(*buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(*buf); return -4; }
    *len = (size_t)sz;
    return 0;
}

/* Pretty short dataset label from path. */
static const char *basename_of(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

int main(int argc, char **argv)
{
    int iters = 200;
    int first_path_arg = 1;
    if (argc > 1) {
        char *e;
        long n = strtol(argv[1], &e, 10);
        if (*e == '\0' && n > 0) {
            iters = (int)n;
            first_path_arg = 2;
        }
    }

    const char **paths;
    int n_paths;
    if (first_path_arg < argc) {
        paths = (const char **)&argv[first_path_arg];
        n_paths = argc - first_path_arg;
    } else {
        paths = DEFAULT_FILES;
        n_paths = (int)(sizeof(DEFAULT_FILES) / sizeof(DEFAULT_FILES[0]));
    }

    printf("# bench_lz4_split — LZ4 stream split + per-stream entropy/decode bench\n");
    printf("# LZ4 compression: HC level 9 (matches `lz4 -9`).\n");
    printf("# H0 = Shannon byte entropy (bits/byte).  HuffRatio = length-\n");
    printf("#   limited byte-Huffman ratio (PIVCO_MAX_CODE_LEN cap).\n");
    printf("# Decode MB/s = min of %d batches × %d iters/batch.\n", N_BATCHES, iters);
    printf("#\n");

    for (int i = 0; i < n_paths; i++) {
        const char *path = paths[i];
        uint8_t *src = NULL; size_t src_len = 0;
        if (read_file(path, &src, &src_len) != 0) {
            printf("=== %s ===\n  ERROR: cannot read file\n\n", basename_of(path));
            continue;
        }

        /* LZ4-HC compress. */
        int lz4_cap = LZ4_compressBound((int)src_len);
        uint8_t *lz4 = (uint8_t *)malloc((size_t)lz4_cap);
        int lz4_size = LZ4_compress_HC((const char *)src, (char *)lz4,
                                        (int)src_len, lz4_cap, 9);
        if (lz4_size <= 0) {
            printf("=== %s ===\n  ERROR: LZ4_compress_HC failed\n\n",
                    basename_of(path));
            free(src); free(lz4);
            continue;
        }

        /* Split into literals + headers (and collect per-sequence
         * semantic fields for the post-table analysis). */
        uint8_t *lits = (uint8_t *)malloc(src_len);     /* upper bound */
        uint8_t *hdrs = (uint8_t *)malloc((size_t)lz4_size);
        size_t lits_len = 0, hdrs_len = 0, n_seqs = 0;
        size_t seq_cap = (size_t)lz4_size;  /* upper bound on n_seqs */
        hdr_fields_t hf = {
            .tokens    = (uint8_t *)malloc(seq_cap),
            .off_lo    = (uint8_t *)malloc(seq_cap),
            .off_hi    = (uint8_t *)malloc(seq_cap),
            .lit_codes = (uint8_t *)malloc(seq_cap),
            .mat_codes = (uint8_t *)malloc(seq_cap),
            .n = 0, .cap = seq_cap,
        };
        int srt = lz4_split(lz4, (size_t)lz4_size,
                             lits, src_len, hdrs, (size_t)lz4_size,
                             &lits_len, &hdrs_len, &n_seqs, &hf);
        if (srt != 0) {
            printf("=== %s ===\n  ERROR: lz4_split rc=%d\n\n",
                    basename_of(path), srt);
            free(src); free(lz4); free(lits); free(hdrs);
            continue;
        }

        printf("=== %s ===\n", basename_of(path));
        printf("  raw=%zu  lz4=%d  sequences=%zu  (lits=%zu, hdrs=%zu, sum=%zu)\n",
                src_len, lz4_size, n_seqs, lits_len, hdrs_len,
                lits_len + hdrs_len);

        /* Per-stream stats + per-codec decode bench. */
        printf("  %-9s | %8s %6s %8s %8s | %8s %8s | %8s %8s\n",
               "stream", "size", "H0", "huf-rat", "fse-rat",
               "huf0 enc", "huf0 dec", "ph enc", "ph dec");
        printf("  %s\n",
               "--------------------------------------------------------------------------------------------");

        struct { const char *name; const uint8_t *buf; size_t len; } streams[] = {
            { "full lz4", lz4,  (size_t)lz4_size },
            { "literals", lits, lits_len },
            { "headers",  hdrs, hdrs_len },
        };
        for (int s = 0; s < 3; s++) {
            const char *name = streams[s].name;
            const uint8_t *buf = streams[s].buf;
            size_t len = streams[s].len;
            uint64_t freq[256];
            double H  = shannon_entropy(buf, len, freq);
            double hr = huffman_ratio(buf, len);
            double fr = fse_byte_ratio(buf, len);
            codec_result_t hr0 = bench_huf0(buf, len, iters);
            codec_result_t hph = bench_ph  (buf, len, iters);

            char fr_s[16];
            if      (fr ==  0.0) snprintf(fr_s, sizeof(fr_s), "    -   ");
            else if (fr == -2.0) snprintf(fr_s, sizeof(fr_s), "  incmpr");
            else if (fr <    0.0) snprintf(fr_s, sizeof(fr_s), "  err   ");
            else                  snprintf(fr_s, sizeof(fr_s), "%8.4f", fr);

            printf("  %-9s | %8zu %6.3f  %7.4f %s | ",
                    name, len, H, hr, fr_s);
            if (hr0.ok) printf("%8zu %7.0f  | ", hr0.enc_size, hr0.dec_mbps);
            else         printf("%8s %8s | ", "-", hr0.note ? hr0.note : "-");
            if (hph.ok) printf("%8zu %7.0f\n",     hph.enc_size, hph.dec_mbps);
            else         printf("%8s %8s\n",        "-", hph.note ? hph.note : "-");
        }

        /* Semantic-field decomposition of the headers stream — this
         * is the regime where zstd's FSE choice actually pays off:
         * tokens and length-codes have strongly-skewed integer
         * distributions even though their byte-encoding is mixed
         * into a near-uniform combined headers stream. */
        if (hf.n > 0) {
            /* Compute log₂-bucketed offset codes a la zstd: code =
             * floor(log2(offset)) for offsets ≥ 1.  Code distribution
             * is heavily skewed (small offsets are common); the bias
             * bits are uniform within each bucket and stored raw at
             * `code` bits each.  Total raw bias cost = sum(code). */
            uint8_t  *off_codes = (uint8_t *)malloc(hf.n);
            uint64_t  bias_bits_total = 0;
            for (size_t i = 0; i < hf.n; i++) {
                uint32_t off = (uint32_t)hf.off_lo[i]
                             | ((uint32_t)hf.off_hi[i] << 8);
                int code = (off > 0) ? (31 - __builtin_clz(off)) : 0;
                off_codes[i] = (uint8_t)code;
                bias_bits_total += (uint64_t)code;
            }

            printf("\n  hdr fields (n_seqs=%zu).  enc-B = approximate bytes\n", hf.n);
            printf("                              after byte-Huffman coding\n");
            printf("                              of this field's value stream.\n");
            printf("  %-12s | %8s %8s %6s %8s %8s | %8s\n",
                   "field", "n_vals", "raw-B", "H0", "huf-rat", "fse-rat",
                   "enc-B");
            struct { const char *name; const uint8_t *buf; int raw_bits; } fields[] = {
                { "tokens",     hf.tokens,    8 },
                { " lit_codes", hf.lit_codes, 4 },   /* indented = sub-field of tokens */
                { " mat_codes", hf.mat_codes, 4 },
                { "off_lo",     hf.off_lo,    8 },
                { "off_hi",     hf.off_hi,    8 },
                { "off_codes",  off_codes,    5 },   /* zstd-style log2 buckets */
            };
            size_t off_codes_enc_bytes = 0;
            size_t off_lo_enc_bytes    = 0;
            size_t off_hi_enc_bytes    = 0;
            for (int f = 0; f < 6; f++) {
                const uint8_t *buf = fields[f].buf;
                size_t len = hf.n;
                size_t raw_bytes = (len * (size_t)fields[f].raw_bits + 7) / 8;
                uint64_t freq[256];
                double H  = shannon_entropy(buf, len, freq);
                double hr = huffman_ratio(buf, len);
                double fr = fse_byte_ratio(buf, len);
                size_t enc_bytes = (size_t)(len * hr + 0.5);
                if (f == 3) off_lo_enc_bytes    = enc_bytes;
                if (f == 4) off_hi_enc_bytes    = enc_bytes;
                if (f == 5) off_codes_enc_bytes = enc_bytes;
                char fr_s[16];
                if      (fr ==  0.0) snprintf(fr_s, sizeof(fr_s), "    -   ");
                else if (fr == -2.0) snprintf(fr_s, sizeof(fr_s), "  incmpr");
                else if (fr <    0.0) snprintf(fr_s, sizeof(fr_s), "  err   ");
                else                  snprintf(fr_s, sizeof(fr_s), "%8.4f", fr);
                printf("  %-12s | %8zu %8zu %6.3f %8.4f %s | %8zu\n",
                       fields[f].name, len, raw_bytes,
                       H, hr, fr_s, enc_bytes);
            }
            size_t bias_bytes = (size_t)((bias_bits_total + 7) / 8);
            size_t zstd_off_total = off_codes_enc_bytes + bias_bytes;
            size_t raw_off_total  = hf.n * 2;
            size_t huf_off_total  = off_lo_enc_bytes + off_hi_enc_bytes;
            printf("\n  offset summary (%zu sequences):\n", hf.n);
            printf("    raw 2-byte offsets ......................... %8zu B\n",
                   raw_off_total);
            printf("    byte-Huffman on off_lo + off_hi ............ %8zu B\n",
                   huf_off_total);
            printf("    zstd-style: Huffman(off_codes) + raw bias .. %8zu B "
                   "(%zu codes + %zu bias)\n",
                   zstd_off_total, off_codes_enc_bytes, bias_bytes);
            free(off_codes);
        }
        printf("\n");

        free(src); free(lz4); free(lits); free(hdrs);
        free(hf.tokens); free(hf.off_lo); free(hf.off_hi);
        free(hf.lit_codes); free(hf.mat_codes);
    }

    return 0;
}
