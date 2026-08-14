/* bench_lz4_ph — minimal "LZ4 + ph" prototype, no metadata splitting.
 *
 * For each input file:
 *   - LZ4 alone     : LZ4_compress_HC(level=9) → LZ4_decompress_safe
 *   - LZ4 + ph      : LZ4 → pivcohuf_compress.  Decode reverses.
 *   - LZ4 + huf0    : LZ4 → HUF_compress (128 KB chunks).  Decode reverses.
 *   - zstd          : ZSTD_compress(level=9) for the size/speed reference.
 *
 * The "+ph" / "+huf0" variants entropy-code the WHOLE LZ4 byte
 * stream as a single buffer (no semantic splitting yet).  This is
 * the first-cut prototype to see where the stacked LZ4+ph codec
 * lands on the LZ4 ↔ zstd Pareto frontier.
 *
 * Output: per-dataset row with compressed size + ratio + decompress
 * MB/s for each codec.
 *
 * Build:   cmake --build build --target pivco_bench_lz4_ph
 * Run:     ./build/pivco_bench_lz4_ph [iters]   # default 50
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#include "pivcohuf_file.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"

#include "lz4.h"
#include "lz4hc.h"
#include "zstd.h"

#include "../lz4_split.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HUF0_CHUNK    (128 * 1024)
#define N_BATCHES     5

/* Length-limited byte-Huffman ratio (length cap = PIVCO_MAX_CODE_LEN).
 * Same primitive used in bench_lz4_split.c — duplicated here so this
 * file stays self-contained.  Returns encoded_bits / (8 * len). */
static double huff_ratio(const uint8_t *buf, size_t len)
{
    if (len == 0) return 0.0;
    uint64_t freq[256] = {0};
    for (size_t i = 0; i < len; i++) freq[buf[i]]++;
    pivco_table_t table;
    if (pivco_build_table(bench_cfg(), freq, &table) != PIVCO_OK) return 1.0;
    uint64_t total_bits = 0;
    for (int s = 0; s < 256; s++) {
        if (freq[s] > 0)
            total_bits += freq[s] * (uint64_t)table.code_len[s];
    }
    return (double)total_bits / (8.0 * (double)len);
}

/* Walk an LZ4 block-format bytestream, decomposing into semantic
 * fields.  Returns the total projected encoded size if each field
 * were byte-Huffman-coded independently.  No decoder is built (this
 * is a size-only projection — the user's "if we knew how to do it,
 * what ratio could we hit" question).
 *
 * Fields:
 *   - literals     : verbatim source bytes inside each sequence
 *   - tokens       : 1 byte per sequence (lit_code + mat_code nibbles)
 *   - offsets      : 2 bytes per sequence (LE uint16) — kept split
 *                    into off_lo + off_hi byte arrays, Huffman'd
 *                    independently (matches zstd's no-bucket option;
 *                    bucketing barely helped at LZ4 window sizes
 *                    per the 2026-05-17 bench_lz4_split run)
 *   - overflows    : literal-length and match-length overflow bytes
 *                    (mostly 255s + occasional low byte)
 *
 * Adds a small fixed per-stream header allowance (~32 B × 5 streams)
 * for the per-field code-length tables. */
__attribute__((unused))
static size_t lz4_semantic_split_projected(const uint8_t *lz4, size_t lz4_size,
                                            size_t src_len)
{
    /* Upper-bound allocations.  Literals: at most src_len.  Tokens
     * + overflow + offsets: at most lz4_size. */
    uint8_t *literals  = (uint8_t *)malloc(src_len);
    uint8_t *tokens    = (uint8_t *)malloc(lz4_size);
    uint8_t *off_lo    = (uint8_t *)malloc(lz4_size);
    uint8_t *off_hi    = (uint8_t *)malloc(lz4_size);
    uint8_t *overflow  = (uint8_t *)malloc(lz4_size);
    size_t  nlit = 0, ntok = 0, noff = 0, nov = 0;

    const uint8_t *p   = lz4;
    const uint8_t *end = lz4 + lz4_size;
    while (p < end) {
        uint8_t tok = *p++;
        tokens[ntok++] = tok;

        size_t lit_len = tok >> 4;
        if (lit_len == 15) {
            while (p < end && *p == 255) {
                overflow[nov++] = *p; p++; lit_len += 255;
            }
            if (p < end) { overflow[nov++] = *p; lit_len += *p++; }
        }
        if (lit_len > 0) {
            if (p + lit_len > end) lit_len = end - p;
            memcpy(literals + nlit, p, lit_len);
            nlit += lit_len;
            p    += lit_len;
        }
        if (p >= end) break;   /* last seq: literals only */

        off_lo[noff] = p[0];
        off_hi[noff] = p[1];
        p += 2; noff++;

        size_t mat_len = tok & 0xf;
        if (mat_len == 15) {
            while (p < end && *p == 255) {
                overflow[nov++] = *p; p++; mat_len += 255;
            }
            if (p < end) { overflow[nov++] = *p; mat_len += *p++; }
        }
    }

    double r_lit = huff_ratio(literals, nlit);
    double r_tok = huff_ratio(tokens,   ntok);
    double r_olo = huff_ratio(off_lo,   noff);
    double r_ohi = huff_ratio(off_hi,   noff);
    double r_ov  = huff_ratio(overflow, nov);

    size_t enc_lit = (size_t)(nlit * r_lit + 0.5);
    size_t enc_tok = (size_t)(ntok * r_tok + 0.5);
    size_t enc_off = (size_t)(noff * (r_olo + r_ohi) + 0.5);
    size_t enc_ov  = (size_t)(nov  * r_ov  + 0.5);

    /* Per-field code-length table cost: ~32 B each in a tiny-input-
     * optimised wire format (most fields have small alphabets after
     * decomposition; lit_codes and mat_codes are 4-bit, so their
     * length tables are <16 entries).  Use 32 B as a generous floor. */
    size_t per_field_hdr = 32 * 5;

    /* Tiny outer header (block sizes + count). */
    size_t outer_hdr = 24;

    free(literals); free(tokens); free(off_lo); free(off_hi); free(overflow);
    return enc_lit + enc_tok + enc_off + enc_ov + per_field_hdr + outer_hdr;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

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
    *len = (size_t)sz;
    return (got == (size_t)sz) ? 0 : -4;
}

static const char *basename_of(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

/* ============================================================
 *  Per-codec encode + decode + timing helpers.
 * ============================================================ */

typedef struct {
    size_t enc_size;     /* compressed bytes */
    double dec_mbps;     /* decode throughput in source-bytes / sec */
    double enc_mbps;     /* encode throughput in source-bytes / sec */
    int    ok;
    const char *note;
} bench_result_t;

/* Encode iters per batch: kept small because HC level 9 + zstd level 9
 * are slow (single-digit-MB/s class), and we only need 2-3 samples to
 * pick min-of-batches. */
#define ENC_ITERS 3

/* ---------- LZ4 alone ---------- */

/* Level convention to match the LZ4 CLI:
 *   level 1   -> LZ4_compress_default       (fast greedy)
 *   level >=3 -> LZ4_compress_HC(level)     (HC chain at given depth) */
static int lz4_encode_at_level(const uint8_t *src, int src_len,
                                uint8_t *dst, int dst_cap, int level)
{
    if (level <= 1) {
        return LZ4_compress_default((const char *)src, (char *)dst,
                                     src_len, dst_cap);
    }
    return LZ4_compress_HC((const char *)src, (char *)dst,
                            src_len, dst_cap, level);
}

static bench_result_t bench_lz4_lvl(const uint8_t *src, size_t src_len,
                                     int iters, int level)
{
    bench_result_t r = {0};
    int cap = LZ4_compressBound((int)src_len);
    uint8_t *enc = (uint8_t *)malloc((size_t)cap);
    int enc_size = lz4_encode_at_level(src, (int)src_len, enc, cap, level);
    if (enc_size <= 0) { r.note = "LZ4 encode failed"; free(enc); return r; }
    r.enc_size = (size_t)enc_size;

    /* Encode timing — re-encode into the same buffer each iter. */
    double enc_best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        double t0 = now_ns();
        for (int i = 0; i < ENC_ITERS; i++) {
            lz4_encode_at_level(src, (int)src_len, enc, cap, level);
        }
        double t1 = now_ns();
        double mb = (double)src_len * (double)ENC_ITERS / (t1 - t0) * 1e3;
        if (mb > enc_best) enc_best = mb;
    }
    r.enc_mbps = enc_best;

    uint8_t *dec = (uint8_t *)malloc(src_len + 64);
    /* Warm up + sanity check. */
    int dsz = LZ4_decompress_safe((const char *)enc, (char *)dec,
                                    enc_size, (int)src_len);
    if (dsz != (int)src_len || memcmp(dec, src, src_len) != 0) {
        r.note = "LZ4 decode mismatch";
        free(enc); free(dec);
        return r;
    }

    double best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            LZ4_decompress_safe((const char *)enc, (char *)dec,
                                 enc_size, (int)src_len);
            sink ^= dec[0] ^ dec[src_len - 1];
        }
        double t1 = now_ns();
        (void)sink;
        double mb = (double)src_len * (double)iters / (t1 - t0) * 1e3;
        if (mb > best) best = mb;
    }
    r.dec_mbps = best;
    r.ok = 1;
    free(enc); free(dec);
    return r;
}

/* ---------- LZ4 then ph (stacked, no metadata split) ---------- */

static bench_result_t bench_lz4_ph(const uint8_t *src, size_t src_len, int iters)
{
    bench_result_t r = {0};
    int cap = LZ4_compressBound((int)src_len);
    uint8_t *lz4_enc = (uint8_t *)malloc((size_t)cap);
    int lz4_size = LZ4_compress_HC((const char *)src, (char *)lz4_enc,
                                    (int)src_len, cap, 9);
    if (lz4_size <= 0) {
        r.note = "LZ4 encode failed";
        free(lz4_enc);
        return r;
    }

    size_t ph_cap = pivcohuf_compress_bound((size_t)lz4_size);
    uint8_t *ph_enc = (uint8_t *)malloc(ph_cap);
    size_t ph_size = ph_cap;
    if (pivcohuf_compress(lz4_enc, (size_t)lz4_size, ph_enc, &ph_size)
        != PIVCOHUF_OK) {
        r.note = "ph encode failed";
        free(lz4_enc); free(ph_enc);
        return r;
    }
    r.enc_size = ph_size;

    /* Sanity round-trip. */
    uint8_t *lz4_dec_buf = (uint8_t *)malloc((size_t)lz4_size);
    size_t lz4_dec_len = (size_t)lz4_size;
    if (pivcohuf_decompress(ph_enc, ph_size, lz4_dec_buf, &lz4_dec_len)
        != PIVCOHUF_OK
        || lz4_dec_len != (size_t)lz4_size
        || memcmp(lz4_dec_buf, lz4_enc, lz4_size) != 0) {
        r.note = "ph roundtrip mismatch";
        free(lz4_enc); free(ph_enc); free(lz4_dec_buf);
        return r;
    }
    uint8_t *dec = (uint8_t *)malloc(src_len + 64);
    int dsz = LZ4_decompress_safe((const char *)lz4_dec_buf, (char *)dec,
                                    lz4_size, (int)src_len);
    if (dsz != (int)src_len || memcmp(dec, src, src_len) != 0) {
        r.note = "LZ4 decode mismatch";
        free(lz4_enc); free(ph_enc); free(lz4_dec_buf); free(dec);
        return r;
    }

    /* Encode timing — LZ4 + ph stacked. */
    {
        double enc_best = 0.0;
        for (int b = 0; b < N_BATCHES; b++) {
            double t0 = now_ns();
            for (int i = 0; i < ENC_ITERS; i++) {
                LZ4_compress_HC((const char *)src, (char *)lz4_enc,
                                 (int)src_len, cap, 9);
                size_t s = ph_cap;
                pivcohuf_compress(lz4_enc, (size_t)lz4_size, ph_enc, &s);
            }
            double t1 = now_ns();
            double mb = (double)src_len * (double)ENC_ITERS / (t1 - t0) * 1e3;
            if (mb > enc_best) enc_best = mb;
        }
        r.enc_mbps = enc_best;
    }

    double best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            size_t lz4_out = (size_t)lz4_size;
            pivcohuf_decompress(ph_enc, ph_size, lz4_dec_buf, &lz4_out);
            LZ4_decompress_safe((const char *)lz4_dec_buf, (char *)dec,
                                 lz4_size, (int)src_len);
            sink ^= dec[0] ^ dec[src_len - 1];
        }
        double t1 = now_ns();
        (void)sink;
        double mb = (double)src_len * (double)iters / (t1 - t0) * 1e3;
        if (mb > best) best = mb;
    }
    r.dec_mbps = best;
    r.ok = 1;
    free(lz4_enc); free(ph_enc); free(lz4_dec_buf); free(dec);
    return r;
}

/* ---------- LZ4 then huf0 (same pattern; reference) ---------- */

static bench_result_t bench_lz4_huf0(const uint8_t *src, size_t src_len, int iters)
{
    bench_result_t r = {0};
    int cap = LZ4_compressBound((int)src_len);
    uint8_t *lz4_enc = (uint8_t *)malloc((size_t)cap);
    int lz4_size = LZ4_compress_HC((const char *)src, (char *)lz4_enc,
                                    (int)src_len, cap, 9);
    if (lz4_size <= 0) {
        r.note = "LZ4 encode failed";
        free(lz4_enc);
        return r;
    }

    /* huf0 in 128 KB chunks. */
    int n_chunks = (int)(((size_t)lz4_size + HUF0_CHUNK - 1) / HUF0_CHUNK);
    uint8_t *huf_enc = (uint8_t *)malloc((size_t)n_chunks * (HUF0_CHUNK + 1024));
    size_t  *huf_off = (size_t *)calloc((size_t)(n_chunks + 1), sizeof(size_t));
    for (int c = 0; c < n_chunks; c++) {
        size_t chunk_sz = (c < n_chunks - 1) ? HUF0_CHUNK
                         : (size_t)lz4_size - (size_t)c * HUF0_CHUNK;
        size_t cr = HUF_compress(huf_enc + huf_off[c], chunk_sz + 1024,
                                  lz4_enc + (size_t)c * HUF0_CHUNK, chunk_sz);
        if (HUF_isError(cr) || cr == 0) {
            /* huf0 bails: emit the raw LZ4 chunk. */
            memcpy(huf_enc + huf_off[c],
                   lz4_enc + (size_t)c * HUF0_CHUNK, chunk_sz);
            cr = chunk_sz;
        }
        huf_off[c + 1] = huf_off[c] + cr;
    }
    r.enc_size = huf_off[n_chunks];

    uint8_t *lz4_dec_buf = (uint8_t *)malloc((size_t)lz4_size);
    uint8_t *dec = (uint8_t *)malloc(src_len + 64);

    /* Warm up sanity. */
    for (int c = 0; c < n_chunks; c++) {
        size_t chunk_sz = (c < n_chunks - 1) ? HUF0_CHUNK
                         : (size_t)lz4_size - (size_t)c * HUF0_CHUNK;
        size_t enc_sz = huf_off[c + 1] - huf_off[c];
        if (enc_sz == chunk_sz) {
            /* uncompressed chunk */
            memcpy(lz4_dec_buf + (size_t)c * HUF0_CHUNK,
                   huf_enc + huf_off[c], chunk_sz);
        } else {
            HUF_decompress(lz4_dec_buf + (size_t)c * HUF0_CHUNK, chunk_sz,
                            huf_enc + huf_off[c], enc_sz);
        }
    }
    int dsz = LZ4_decompress_safe((const char *)lz4_dec_buf, (char *)dec,
                                    lz4_size, (int)src_len);
    if (dsz != (int)src_len || memcmp(dec, src, src_len) != 0) {
        r.note = "huf0 roundtrip mismatch";
        free(lz4_enc); free(huf_enc); free(huf_off);
        free(lz4_dec_buf); free(dec);
        return r;
    }

    double best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            for (int c = 0; c < n_chunks; c++) {
                size_t chunk_sz = (c < n_chunks - 1) ? HUF0_CHUNK
                                 : (size_t)lz4_size - (size_t)c * HUF0_CHUNK;
                size_t enc_sz = huf_off[c + 1] - huf_off[c];
                if (enc_sz == chunk_sz) {
                    memcpy(lz4_dec_buf + (size_t)c * HUF0_CHUNK,
                           huf_enc + huf_off[c], chunk_sz);
                } else {
                    HUF_decompress(lz4_dec_buf + (size_t)c * HUF0_CHUNK, chunk_sz,
                                    huf_enc + huf_off[c], enc_sz);
                }
            }
            LZ4_decompress_safe((const char *)lz4_dec_buf, (char *)dec,
                                 lz4_size, (int)src_len);
            sink ^= dec[0] ^ dec[src_len - 1];
        }
        double t1 = now_ns();
        (void)sink;
        double mb = (double)src_len * (double)iters / (t1 - t0) * 1e3;
        if (mb > best) best = mb;
    }
    r.dec_mbps = best;
    r.ok = 1;
    free(lz4_enc); free(huf_enc); free(huf_off); free(lz4_dec_buf); free(dec);
    return r;
}

/* ---------- LZ4-split + ph (the real hacked-LZ4 path) ----------
 *
 * Uses phsplit_LZ4_compress_HC_split to emit 4 streams via the modified
 * LZ4HC encoder, ph-encodes each independently, and writes a small
 * outer wire format.  Decode reverses: ph-decompress each section then
 * lz4_split_decompress directly off the 4 streams (no LZ4 byte-stream
 * reconstruction).
 *
 * Wire format:
 *   [outer hdr 16 B]: magic "LPS\0", version 1, src_size u32, n_sections=4 u32
 *   [section 0]: u32 len + pivcohuf-compressed literals
 *   [section 1]: u32 len + pivcohuf-compressed tokens
 *   [section 2]: u32 len + pivcohuf-compressed offsets
 *   [section 3]: u32 len + pivcohuf-compressed overflow
 */

static bench_result_t bench_lz4_split_ph_lvl(const uint8_t *src, size_t src_len, int iters, int lz4_level)
{
    bench_result_t r = {0};

    /* Allocate worst-case sized buffers for the 4 streams + the
     * throwaway dst.  Caps a bit oversized for safety. */
    int lz4_cap = LZ4_compressBound((int)src_len);
    uint8_t *throwaway = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_lit = (uint8_t *)malloc(src_len + 64);
    uint8_t *s_tok = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_off = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_ovf = (uint8_t *)malloc((size_t)lz4_cap);

    lz4_split_ctx_t split = {
        .literals = s_lit, .lit_cap = src_len + 64,
        .tokens   = s_tok, .tok_cap = (size_t)lz4_cap,
        .offsets  = s_off, .off_cap = (size_t)lz4_cap,
        .overflow = s_ovf, .ovf_cap = (size_t)lz4_cap,
    };

    int lz4_size = phsplit_LZ4_compress_HC_split(
        (const char *)src, (int)src_len, throwaway, lz4_cap, lz4_level, &split);
    if (lz4_size <= 0 || !split.ok) {
        r.note = "LZ4-split encode failed";
        free(throwaway); free(s_lit); free(s_tok); free(s_off); free(s_ovf);
        return r;
    }

    /* Sanity check: the SUM of stream sizes (tokens + literals + offsets
     * + overflow) should equal the LZ4 output size, since each byte
     * written to op also went into exactly one of the 4 streams. */
    size_t split_total = split.lit_pos + split.tok_pos
                       + split.off_pos + split.ovf_pos;
    if (split_total != (size_t)lz4_size) {
        fprintf(stderr, "  split-stream byte-count mismatch: total=%zu lz4=%d "
                       "(lit=%zu tok=%zu off=%zu ovf=%zu)\n",
                split_total, lz4_size,
                split.lit_pos, split.tok_pos,
                split.off_pos, split.ovf_pos);
    }

    /* Strong sanity check: walk the standard LZ4 output buffer and
     * produce the 4 streams the way bench_lz4_split.c does.  Compare
     * byte-for-byte against what the modified encoder pushed.  Any
     * mismatch tells us EXACTLY which stream is wrong and at which
     * byte the encoder's side-channel diverges from the LZ4 it
     * actually emitted. */
    {
        uint8_t *w_lit = (uint8_t *)malloc(src_len + 64);
        uint8_t *w_tok = (uint8_t *)malloc((size_t)lz4_size);
        uint8_t *w_off = (uint8_t *)malloc((size_t)lz4_size);
        uint8_t *w_ovf = (uint8_t *)malloc((size_t)lz4_size);
        size_t   w_lit_n = 0, w_tok_n = 0, w_off_n = 0, w_ovf_n = 0;
        const uint8_t *p   = throwaway;
        const uint8_t *end = throwaway + lz4_size;
        while (p < end) {
            uint8_t token = *p++;
            w_tok[w_tok_n++] = token;
            size_t lit_len = token >> 4;
            if (lit_len == 15) {
                while (p < end && *p == 255) { w_ovf[w_ovf_n++] = *p; lit_len += 255; p++; }
                if (p < end) { w_ovf[w_ovf_n++] = *p; lit_len += *p++; }
            }
            if (lit_len > 0) {
                if (p + lit_len > end) lit_len = end - p;
                memcpy(w_lit + w_lit_n, p, lit_len);
                w_lit_n += lit_len;
                p       += lit_len;
            }
            if (p >= end) break;
            w_off[w_off_n++] = *p++;
            w_off[w_off_n++] = *p++;
            size_t mat_len = token & 0xf;
            if (mat_len == 15) {
                while (p < end && *p == 255) { w_ovf[w_ovf_n++] = *p; mat_len += 255; p++; }
                if (p < end) { w_ovf[w_ovf_n++] = *p; mat_len += *p++; }
            }
        }
        int diff = 0;
        if (w_lit_n != split.lit_pos) { fprintf(stderr, "  literals count differ: walker=%zu split=%zu\n", w_lit_n, split.lit_pos); diff = 1; }
        if (w_tok_n != split.tok_pos) { fprintf(stderr, "  tokens count differ:   walker=%zu split=%zu\n", w_tok_n, split.tok_pos); diff = 1; }
        if (w_off_n != split.off_pos) { fprintf(stderr, "  offsets count differ:  walker=%zu split=%zu\n", w_off_n, split.off_pos); diff = 1; }
        if (w_ovf_n != split.ovf_pos) { fprintf(stderr, "  overflow count differ: walker=%zu split=%zu\n", w_ovf_n, split.ovf_pos); diff = 1; }
        if (!diff && memcmp(w_lit, s_lit, w_lit_n) != 0) {
            size_t i = 0; while (i < w_lit_n && w_lit[i] == s_lit[i]) i++;
            fprintf(stderr, "  literals content differ at idx %zu/%zu (walker=0x%02x split=0x%02x)\n",
                   i, w_lit_n, w_lit[i], s_lit[i]);
        }
        if (!diff && memcmp(w_tok, s_tok, w_tok_n) != 0) {
            size_t i = 0; while (i < w_tok_n && w_tok[i] == s_tok[i]) i++;
            fprintf(stderr, "  tokens content differ at idx %zu/%zu (walker=0x%02x split=0x%02x)\n",
                   i, w_tok_n, w_tok[i], s_tok[i]);
        }
        if (!diff && memcmp(w_off, s_off, w_off_n) != 0) {
            size_t i = 0; while (i < w_off_n && w_off[i] == s_off[i]) i++;
            fprintf(stderr, "  offsets content differ at idx %zu/%zu (walker=0x%02x split=0x%02x)\n",
                   i, w_off_n, w_off[i], s_off[i]);
        }
        if (!diff && memcmp(w_ovf, s_ovf, w_ovf_n) != 0) {
            size_t i = 0; while (i < w_ovf_n && w_ovf[i] == s_ovf[i]) i++;
            fprintf(stderr, "  overflow content differ at idx %zu/%zu (walker=0x%02x split=0x%02x)\n",
                   i, w_ovf_n, w_ovf[i], s_ovf[i]);
        }
        free(w_lit); free(w_tok); free(w_off); free(w_ovf);
    }

    /* ph-encode each section. */
    size_t cap_lit = pivcohuf_compress_bound(split.lit_pos + 1);
    size_t cap_tok = pivcohuf_compress_bound(split.tok_pos + 1);
    size_t cap_off = pivcohuf_compress_bound(split.off_pos + 1);
    size_t cap_ovf = pivcohuf_compress_bound(split.ovf_pos + 1);
    uint8_t *enc_lit = (uint8_t *)malloc(cap_lit);
    uint8_t *enc_tok = (uint8_t *)malloc(cap_tok);
    uint8_t *enc_off = (uint8_t *)malloc(cap_off);
    uint8_t *enc_ovf = (uint8_t *)malloc(cap_ovf);
    size_t enc_lit_size = cap_lit, enc_tok_size = cap_tok;
    size_t enc_off_size = cap_off, enc_ovf_size = cap_ovf;
    if (pivcohuf_compress(s_lit, split.lit_pos, enc_lit, &enc_lit_size) != PIVCOHUF_OK ||
        pivcohuf_compress(s_tok, split.tok_pos, enc_tok, &enc_tok_size) != PIVCOHUF_OK ||
        pivcohuf_compress(s_off, split.off_pos, enc_off, &enc_off_size) != PIVCOHUF_OK ||
        pivcohuf_compress(s_ovf, split.ovf_pos, enc_ovf, &enc_ovf_size) != PIVCOHUF_OK) {
        r.note = "ph encode (split) failed";
        goto cleanup;
    }

    /* Encode timing — full pipeline: phsplit + 4× ph_compress. */
    {
        double enc_best = 0.0;
        for (int b = 0; b < N_BATCHES; b++) {
            double t0 = now_ns();
            for (int i = 0; i < ENC_ITERS; i++) {
                lz4_split_ctx_t rs = {
                    .literals = s_lit, .lit_cap = src_len + 64,
                    .tokens   = s_tok, .tok_cap = (size_t)lz4_cap,
                    .offsets  = s_off, .off_cap = (size_t)lz4_cap,
                    .overflow = s_ovf, .ovf_cap = (size_t)lz4_cap,
                };
                phsplit_LZ4_compress_HC_split(
                    (const char *)src, (int)src_len, throwaway, lz4_cap, lz4_level, &rs);
                size_t a = cap_lit, b2 = cap_tok, c = cap_off, d = cap_ovf;
                pivcohuf_compress(s_lit, rs.lit_pos, enc_lit, &a);
                pivcohuf_compress(s_tok, rs.tok_pos, enc_tok, &b2);
                pivcohuf_compress(s_off, rs.off_pos, enc_off, &c);
                pivcohuf_compress(s_ovf, rs.ovf_pos, enc_ovf, &d);
            }
            double t1 = now_ns();
            double mb = (double)src_len * (double)ENC_ITERS / (t1 - t0) * 1e3;
            if (mb > enc_best) enc_best = mb;
        }
        r.enc_mbps = enc_best;
    }

    /* Outer wire: 16 B header + 4× (4 B len + payload). */
    size_t wire_size = 16 + 4 * 4 + enc_lit_size + enc_tok_size
                                  + enc_off_size + enc_ovf_size;
    uint8_t *wire = (uint8_t *)malloc(wire_size);
    {
        uint8_t *p = wire;
        memcpy(p, "LPS\0", 4); p += 4;
        p[0] = 1; p[1] = 0; p[2] = 0; p[3] = 0; p += 4;     /* version + pad */
        p[0] = (uint8_t)(src_len & 0xff);
        p[1] = (uint8_t)((src_len >> 8) & 0xff);
        p[2] = (uint8_t)((src_len >> 16) & 0xff);
        p[3] = (uint8_t)((src_len >> 24) & 0xff);
        p[4] = 0; p[5] = 0; p[6] = 0; p[7] = 0;
        p += 8;
        #define WRITE_SECTION(buf, sz)                                          \
            do {                                                                \
                p[0] = (uint8_t)((sz)       & 0xff);                            \
                p[1] = (uint8_t)(((sz) >> 8) & 0xff);                           \
                p[2] = (uint8_t)(((sz) >> 16)& 0xff);                           \
                p[3] = (uint8_t)(((sz) >> 24)& 0xff);                           \
                p += 4;                                                         \
                memcpy(p, (buf), (sz));                                          \
                p += (sz);                                                       \
            } while (0)
        WRITE_SECTION(enc_lit, enc_lit_size);
        WRITE_SECTION(enc_tok, enc_tok_size);
        WRITE_SECTION(enc_off, enc_off_size);
        WRITE_SECTION(enc_ovf, enc_ovf_size);
        #undef WRITE_SECTION
    }
    r.enc_size = wire_size;

    /* Decode + sanity check. */
    uint8_t *dec_lit = (uint8_t *)malloc(split.lit_pos + 64);
    uint8_t *dec_tok = (uint8_t *)malloc(split.tok_pos + 64);
    uint8_t *dec_off = (uint8_t *)malloc(split.off_pos + 64);
    uint8_t *dec_ovf = (uint8_t *)malloc(split.ovf_pos + 64);
    uint8_t *dec = (uint8_t *)malloc(src_len + 64);

    /* Reusable function to decode all 4 sections + run lz4_split_decompress.
     * Inlined since this is the timed inner loop. */
    #define DECODE_ROUND()                                                  \
        do {                                                                \
            const uint8_t *p = wire + 16;                                   \
            size_t s_lit_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4; \
            size_t l_out = split.lit_pos + 64;                              \
            (void)pivcohuf_decompress(p, s_lit_size, dec_lit, &l_out);       \
            p += s_lit_size;                                                \
            size_t s_tok_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4; \
            size_t t_out = split.tok_pos + 64;                              \
            (void)pivcohuf_decompress(p, s_tok_size, dec_tok, &t_out);       \
            p += s_tok_size;                                                \
            size_t s_off_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4; \
            size_t o_out = split.off_pos + 64;                              \
            (void)pivcohuf_decompress(p, s_off_size, dec_off, &o_out);       \
            p += s_off_size;                                                \
            size_t s_ovf_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4; \
            size_t v_out = split.ovf_pos + 64;                              \
            (void)pivcohuf_decompress(p, s_ovf_size, dec_ovf, &v_out);       \
            (void)lz4_split_decompress(dec_lit, l_out, dec_tok, t_out,       \
                                        dec_off, o_out, dec_ovf, v_out,      \
                                        dec, src_len);                       \
        } while (0)

    DECODE_ROUND();
    if (memcmp(dec, src, src_len) != 0) {
        /* Find first divergence for debugging. */
        size_t i = 0;
        while (i < src_len && dec[i] == src[i]) i++;
        fprintf(stderr,
                "  split decode mismatch at offset %zu/%zu  (src=0x%02x dec=0x%02x)\n"
                "    src_len=%zu lit_pos=%zu tok_pos=%zu off_pos=%zu ovf_pos=%zu\n",
                i, src_len, i < src_len ? src[i] : 0, i < src_len ? dec[i] : 0,
                src_len, split.lit_pos, split.tok_pos, split.off_pos, split.ovf_pos);
        r.note = "split decode mismatch";
        goto cleanup_dec;
    }

    double best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            DECODE_ROUND();
            sink ^= dec[0] ^ dec[src_len - 1];
        }
        double t1 = now_ns();
        (void)sink;
        double mb = (double)src_len * (double)iters / (t1 - t0) * 1e3;
        if (mb > best) best = mb;
    }
    r.dec_mbps = best;
    r.ok = 1;

cleanup_dec:
    free(dec_lit); free(dec_tok); free(dec_off); free(dec_ovf); free(dec);
    #undef DECODE_ROUND
    free(wire);
cleanup:
    free(enc_lit); free(enc_tok); free(enc_off); free(enc_ovf);
    free(throwaway); free(s_lit); free(s_tok); free(s_off); free(s_ovf);
    return r;
}


/* ---------- LZ4-split RAW (no entropy coding) ----------
 *
 * Isolates the cost of the 4-stream LZ4 path itself, separate from
 * the 4 ph passes.  Encode: hacked LZ4 emits 4 streams; we just
 * concatenate them with a tiny header.  Decode: lz4_split_decompress
 * directly on the raw streams, no pivcohuf in the loop.  Lets us
 * answer "is the 4-stream LZ4 decoder itself competitive with
 * upstream LZ4_decompress_safe?". */

static bench_result_t bench_lz4_split_raw_lvl(const uint8_t *src, size_t src_len, int iters, int lz4_level)
{
    bench_result_t r = {0};
    int lz4_cap = LZ4_compressBound((int)src_len);
    uint8_t *throwaway = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_lit = (uint8_t *)malloc(src_len + 64);
    uint8_t *s_tok = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_off = (uint8_t *)malloc((size_t)lz4_cap);
    uint8_t *s_ovf = (uint8_t *)malloc((size_t)lz4_cap);

    lz4_split_ctx_t split = {
        .literals = s_lit, .lit_cap = src_len + 64,
        .tokens   = s_tok, .tok_cap = (size_t)lz4_cap,
        .offsets  = s_off, .off_cap = (size_t)lz4_cap,
        .overflow = s_ovf, .ovf_cap = (size_t)lz4_cap,
    };
    int lz4_size = phsplit_LZ4_compress_HC_split(
        (const char *)src, (int)src_len, throwaway, lz4_cap, lz4_level, &split);
    if (lz4_size <= 0 || !split.ok) {
        r.note = "raw-split encode failed";
        free(throwaway); free(s_lit); free(s_tok); free(s_off); free(s_ovf);
        return r;
    }

    /* Encode timing — phsplit re-encode into the same 4-stream buffers. */
    {
        double enc_best = 0.0;
        for (int b = 0; b < N_BATCHES; b++) {
            double t0 = now_ns();
            for (int i = 0; i < ENC_ITERS; i++) {
                lz4_split_ctx_t rs = {
                    .literals = s_lit, .lit_cap = src_len + 64,
                    .tokens   = s_tok, .tok_cap = (size_t)lz4_cap,
                    .offsets  = s_off, .off_cap = (size_t)lz4_cap,
                    .overflow = s_ovf, .ovf_cap = (size_t)lz4_cap,
                };
                phsplit_LZ4_compress_HC_split(
                    (const char *)src, (int)src_len, throwaway, lz4_cap, lz4_level, &rs);
            }
            double t1 = now_ns();
            double mb = (double)src_len * (double)ENC_ITERS / (t1 - t0) * 1e3;
            if (mb > enc_best) enc_best = mb;
        }
        r.enc_mbps = enc_best;
    }

    /* Wire = 16 B outer header + 4× (4 B len + raw section). */
    size_t wire_size = 16 + 4 * 4 + split.lit_pos + split.tok_pos
                                  + split.off_pos + split.ovf_pos;
    uint8_t *wire = (uint8_t *)malloc(wire_size);
    {
        uint8_t *p = wire;
        memcpy(p, "LSR\0", 4); p += 4;
        p[0] = 1; p[1] = 0; p[2] = 0; p[3] = 0; p += 4;
        p[0] = (uint8_t)(src_len & 0xff);
        p[1] = (uint8_t)((src_len >> 8) & 0xff);
        p[2] = (uint8_t)((src_len >> 16) & 0xff);
        p[3] = (uint8_t)((src_len >> 24) & 0xff);
        p[4] = 0; p[5] = 0; p[6] = 0; p[7] = 0;
        p += 8;
        #define WRITE_RAW(buf, sz)                                              \
            do {                                                                \
                p[0] = (uint8_t)((sz)       & 0xff);                            \
                p[1] = (uint8_t)(((sz) >> 8) & 0xff);                           \
                p[2] = (uint8_t)(((sz) >> 16)& 0xff);                           \
                p[3] = (uint8_t)(((sz) >> 24)& 0xff);                           \
                p += 4;                                                         \
                memcpy(p, (buf), (sz));                                          \
                p += (sz);                                                       \
            } while (0)
        WRITE_RAW(s_lit, split.lit_pos);
        WRITE_RAW(s_tok, split.tok_pos);
        WRITE_RAW(s_off, split.off_pos);
        WRITE_RAW(s_ovf, split.ovf_pos);
        #undef WRITE_RAW
    }
    r.enc_size = wire_size;

    uint8_t *dec = (uint8_t *)malloc(src_len + 64);
    /* Sanity check. */
    {
        const uint8_t *p = wire + 16;
        size_t s_lit_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4;
        const uint8_t *p_lit = p; p += s_lit_size;
        size_t s_tok_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4;
        const uint8_t *p_tok = p; p += s_tok_size;
        size_t s_off_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4;
        const uint8_t *p_off = p; p += s_off_size;
        size_t s_ovf_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4;
        const uint8_t *p_ovf = p;
        int rc = lz4_split_decompress(p_lit, s_lit_size,
                                       p_tok, s_tok_size,
                                       p_off, s_off_size,
                                       p_ovf, s_ovf_size,
                                       dec, src_len);
        if (rc != 0 || memcmp(dec, src, src_len) != 0) {
            r.note = "raw-split roundtrip failed";
            free(wire); free(dec);
            free(throwaway); free(s_lit); free(s_tok); free(s_off); free(s_ovf);
            return r;
        }
    }

    double best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            const uint8_t *p = wire + 16;
            size_t s_lit_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4;
            const uint8_t *p_lit = p; p += s_lit_size;
            size_t s_tok_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4;
            const uint8_t *p_tok = p; p += s_tok_size;
            size_t s_off_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4;
            const uint8_t *p_off = p; p += s_off_size;
            size_t s_ovf_size = (size_t)(p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24)); p += 4;
            const uint8_t *p_ovf = p;
            lz4_split_decompress(p_lit, s_lit_size,
                                  p_tok, s_tok_size,
                                  p_off, s_off_size,
                                  p_ovf, s_ovf_size,
                                  dec, src_len);
            sink ^= dec[0] ^ dec[src_len - 1];
        }
        double t1 = now_ns();
        (void)sink;
        double mb = (double)src_len * (double)iters / (t1 - t0) * 1e3;
        if (mb > best) best = mb;
    }
    r.dec_mbps = best;
    r.ok = 1;
    free(wire); free(dec);
    free(throwaway); free(s_lit); free(s_tok); free(s_off); free(s_ovf);
    return r;
}


/* ---------- zstd reference (level 9 to match LZ4-HC level 9) ---------- */

static bench_result_t bench_zstd_lvl(const uint8_t *src, size_t src_len,
                                      int iters, int level)
{
    bench_result_t r = {0};
    size_t cap = ZSTD_compressBound(src_len);
    uint8_t *enc = (uint8_t *)malloc(cap);
    size_t enc_size = ZSTD_compress(enc, cap, src, src_len, level);
    if (ZSTD_isError(enc_size)) {
        r.note = "zstd encode failed";
        free(enc); return r;
    }
    r.enc_size = enc_size;

    /* Encode timing. */
    double enc_best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        double t0 = now_ns();
        for (int i = 0; i < ENC_ITERS; i++) {
            ZSTD_compress(enc, cap, src, src_len, level);
        }
        double t1 = now_ns();
        double mb = (double)src_len * (double)ENC_ITERS / (t1 - t0) * 1e3;
        if (mb > enc_best) enc_best = mb;
    }
    r.enc_mbps = enc_best;

    uint8_t *dec = (uint8_t *)malloc(src_len + 64);
    size_t dsz = ZSTD_decompress(dec, src_len, enc, enc_size);
    if (ZSTD_isError(dsz) || dsz != src_len || memcmp(dec, src, src_len) != 0) {
        r.note = "zstd roundtrip mismatch";
        free(enc); free(dec); return r;
    }

    double best = 0.0;
    for (int b = 0; b < N_BATCHES; b++) {
        volatile uint8_t sink = 0;
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            ZSTD_decompress(dec, src_len, enc, enc_size);
            sink ^= dec[0] ^ dec[src_len - 1];
        }
        double t1 = now_ns();
        (void)sink;
        double mb = (double)src_len * (double)iters / (t1 - t0) * 1e3;
        if (mb > best) best = mb;
    }
    r.dec_mbps = best;
    r.ok = 1;
    free(enc); free(dec);
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
    "extras/datasets/source_c.c",
    "extras/datasets/log_apache.log",
    "extras/datasets/dna_fasta.fa",
    "extras/datasets/csv_numeric.csv",
};

int main(int argc, char **argv)
{
    int iters = 50;
    int first_arg = 1;
    if (argc > 1) {
        char *e;
        long n = strtol(argv[1], &e, 10);
        if (*e == '\0' && n > 0) { iters = (int)n; first_arg = 2; }
    }
    const char **paths;
    int n_paths;
    if (first_arg < argc) {
        paths = (const char **)&argv[first_arg];
        n_paths = argc - first_arg;
    } else {
        paths = DEFAULT_FILES;
        n_paths = (int)(sizeof(DEFAULT_FILES) / sizeof(DEFAULT_FILES[0]));
    }

    /* ph FSE toggled per-variant: split (Huffman-only, apples-to-apples
     * with +ph/+huf columns) vs split-fse (FSE-enabled, the proper ph
     * config for entropy coding skewed streams like offsets/overflow). */
    bench_cfg()->fse_enabled = (0);

    printf("# bench_lz4_ph — stacked codec prototype\n");
    printf("# LZ4 = LZ4_compress_HC(level=9).  zstd@N = ZSTD_compress(level=N).\n");
    printf("# ph runs with --no-fse for +ph/split, with FSE on for split-fse.\n");
    printf("# huf0 chunks at 128 KB.\n");
    printf("# Decode MB/s = source bytes / total decode time (incl. inner LZ4 stage).\n");
    printf("# min of %d batches × %d iters/batch.\n", N_BATCHES, iters);
    printf("#\n");
    printf("%-22s  %10s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s | %8s %5s %7s\n",
            "dataset", "raw",
            "lz4@1 sz", "rat", "MB/s",
            "lz4@3 sz", "rat", "MB/s",
            "lz4@9 sz", "rat", "MB/s",
            "lzx@9 sz", "rat", "MB/s",
            "lzx@1 sz", "rat", "MB/s",
            "+ph sz", "rat", "MB/s",
            "+huf sz", "rat", "MB/s",
            "zstd@9 sz", "rat", "MB/s",
            "zstd@3 sz", "rat", "MB/s",
            "zstd@1 sz", "rat", "MB/s",
            "lzxph@9", "rat", "MB/s",
            "lzxph+fse", "rat", "MB/s",
            "lzxph@1", "rat", "MB/s",
            "lzxph@1+fse", "rat", "MB/s");
    printf("%-22s  %10s + %s\n", "", "",
            "------------------------------------------------------------------------"
            "------------------------------------------------------------------------"
            "------------------------------------------------------------------------");
    printf("%-22s  %10s |   lz4-r    = LZ4-split RAW (4-stream encoder, custom decoder, NO entropy).\n"
           "%-22s  %10s     split    = LZ4-split + per-stream ph (Huffman-only).\n"
           "%-22s  %10s     split-fse= LZ4-split + per-stream ph with FSE enabled (proper entropy coding).\n",
           "", "", "", "", "", "");

    for (int i = 0; i < n_paths; i++) {
        const char *path = paths[i];
        uint8_t *src = NULL; size_t src_len = 0;
        if (read_file(path, &src, &src_len) != 0) {
            printf("%-22s  ERROR: cannot read file\n", basename_of(path));
            continue;
        }

        bench_result_t lz1_  = bench_lz4_lvl       (src, src_len, iters, 1);
        bench_result_t lz3_  = bench_lz4_lvl       (src, src_len, iters, 3);
        bench_result_t lz    = bench_lz4_lvl       (src, src_len, iters, 9);
        bench_result_t lzr   = bench_lz4_split_raw_lvl (src, src_len, iters, 9);
        bench_result_t lzr1  = bench_lz4_split_raw_lvl (src, src_len, iters, 1);

        bench_cfg()->fse_enabled = (0);
        bench_result_t lzp   = bench_lz4_ph        (src, src_len, iters);
        bench_result_t lzh   = bench_lz4_huf0      (src, src_len, iters);
        bench_result_t lzsp  = bench_lz4_split_ph_lvl(src, src_len, iters, 9);
        bench_result_t lzsp1 = bench_lz4_split_ph_lvl(src, src_len, iters, 1);

        bench_cfg()->fse_enabled = (1);
        bench_result_t lzsp_fse  = bench_lz4_split_ph_lvl(src, src_len, iters, 9);
        bench_result_t lzsp1_fse = bench_lz4_split_ph_lvl(src, src_len, iters, 1);
        bench_cfg()->fse_enabled = (0);

        bench_result_t zs9   = bench_zstd_lvl      (src, src_len, iters, 9);
        bench_result_t zs3   = bench_zstd_lvl      (src, src_len, iters, 3);
        bench_result_t zs1   = bench_zstd_lvl      (src, src_len, iters, 1);

        printf("%-22s  %10zu | ", basename_of(path), src_len);

        #define PRINT_COL(R, term) do {                                         \
            if ((R).ok) printf("%8zu %5.3f %7.0f" term,                          \
                                (R).enc_size, (double)(R).enc_size / src_len,    \
                                (R).dec_mbps);                                   \
            else         printf("%8s %5s %7s" term, "-", "-",                    \
                                (R).note ? (R).note : "-");                      \
        } while (0)

        PRINT_COL(lz1_,        " | ");
        PRINT_COL(lz3_,        " | ");
        PRINT_COL(lz,          " | ");
        PRINT_COL(lzr,         " | ");
        PRINT_COL(lzr1,        " | ");
        PRINT_COL(lzp,         " | ");
        PRINT_COL(lzh,         " | ");
        PRINT_COL(zs9,         " | ");
        PRINT_COL(zs3,         " | ");
        PRINT_COL(zs1,         " | ");
        PRINT_COL(lzsp,        " | ");
        PRINT_COL(lzsp_fse,    " | ");
        PRINT_COL(lzsp1,       " | ");
        PRINT_COL(lzsp1_fse,   "\n");

        #undef PRINT_COL

        /* Encode-rate continuation row. */
        printf("%-22s  %10s   enc MB/s: lz4@1=%.0f lz4@3=%.0f lz4@9=%.0f "
               "lzx@9=%.0f lzx@1=%.0f +ph=%.0f +huf=%.0f "
               "zs9=%.0f zs3=%.0f zs1=%.0f "
               "lzxph@9=%.0f lzxph@9+fse=%.0f lzxph@1=%.0f lzxph@1+fse=%.0f\n",
               "", "",
               lz1_.enc_mbps, lz3_.enc_mbps, lz.enc_mbps,
               lzr.enc_mbps, lzr1.enc_mbps, lzp.enc_mbps, lzh.enc_mbps,
               zs9.enc_mbps, zs3.enc_mbps, zs1.enc_mbps,
               lzsp.enc_mbps, lzsp_fse.enc_mbps,
               lzsp1.enc_mbps, lzsp1_fse.enc_mbps);

        fflush(stdout);
        free(src);
    }

    return 0;
}
