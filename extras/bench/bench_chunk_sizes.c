/* Per-chunk compressed-size comparison across four entropy coders.
 *
 * Splits every input file into fixed-size chunks (32 KiB by default) and
 * compresses each chunk independently with:
 *
 *   pivco        PivCo-Huffman, per-node FSE off -- raw partition bitmaps
 *   pivco+sANS   ... + static ANS: the pre-built PIVCO_FSE_NUM_TABLES
 *                byte-alphabet FSE tables, selected per node from the
 *                partition skew (no per-node table header)
 *   pivco+dANS   ... + dynamic ANS: the static schedule *and* the dynamic
 *                nibble table (PIVCO_FSE_DYNAMIC_ID), smaller payload wins
 *   fse          Yann's TANS straight over the chunk's bytes
 *                (FSE_compress -- order-0, self-describing)
 *
 * plus an `entropy` reference row: the order-0 (per-symbol, memoryless)
 * entropy of each chunk, summed over chunks, with NO table or framing
 * cost of any kind.  That is the floor an order-0 coder with a free,
 * perfectly-adapted per-chunk model would reach, so a codec landing
 * below it has beaten order-0 -- it is exploiting structure a symbol
 * histogram cannot see.  All the "vs entropy" percentages are measured
 * against this row.
 *
 * Sizes only; this bench answers "what does it cost", not "how fast".
 * For throughput see pivco_bench / pivco_bench_4way.
 *
 * Accounting is header-inclusive and per chunk, so the numbers are what a
 * real container storing independently-decodable chunks would pay:
 *
 *   - every chunk carries a 1-byte block-type tag: CODED, RAW, or RLE --
 *     the three types zstd has, and the minimum any container needs
 *   - the three pivco columns add CODE_LEN_BYTES = 128 for the chunk's
 *     Huffman code lengths, nibble-packed exactly as pivcohuf_file.c
 *     writes them.  Nothing else about the tree goes on the wire
 *   - fse's table description is already inside FSE_compress's output
 *   - a chunk whose coded form is not smaller than its raw bytes is
 *     stored raw, and charged as such, for every codec
 *   - a constant chunk is RLE for every codec (tag + the repeated byte),
 *     so it drops out of the comparison instead of scoring whichever
 *     codec happens to have a special case for it.  Without this, FSE
 *     looks far worse than it is on files with long runs of one byte:
 *     FSE_compress reports constant input as RLE and emits nothing,
 *     which would otherwise be charged a full raw copy
 *
 * Every chunk is decoded and compared against the input, so a size here
 * is a size for output that actually round-trips.
 *
 * Usage: pivco_bench_chunk_sizes [--chunk BYTES] [--flat LAYOUT] FILE [...]
 *
 * --flat picks the flat-subtree bit layout (natural / vertical /
 * vertical128, default vertical as in pivco_cfg_default).  It matters
 * here beyond decode speed: the dynamic nibble table reads the packed
 * flat region as bytes, and vertical packing scatters each symbol's bits
 * across lanes, so a byte gathers one bit from each of eight different
 * symbols instead of holding whole codes.  That flattens the nibble
 * histogram the table depends on. */

#include "pivco_huffman.h"
#include "pivco_fse.h"
#include "bench_ctx.h"
#include "fse.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Per-chunk Huffman code lengths: 256 symbols at 4 bits each, as
 * pivcohuf_file.c's CODE_LENGTHS field. */
#define CODE_LEN_BYTES  128
/* Per-chunk block-type tag (CODED / RAW / RLE). */
#define BLOCK_TAG_BYTES 1

enum { C_PIVCO = 0, C_PIVCO_SANS, C_PIVCO_DANS, C_FSE, N_CODECS };

static const char *codec_name[N_CODECS] = {
    "pivco", "pivco+sANS", "pivco+dANS", "fse",
};

static pivco_flat_layout_t g_flat_layout = PIVCO_FLAT_VERTICAL;
static const char *g_flat_name = "vertical";
static int g_per_file_table = 0;

typedef struct {
    uint64_t bytes;        /* header-inclusive compressed size */
    uint64_t raw_chunks;   /* chunks that fell back to a stored copy */
    uint64_t dyn_commits;  /* dynamic-nibble bitmaps committed (pivco+dANS) */
} codec_totals_t;

/* ---------- helpers ---------- */

static uint8_t *read_file(const char *path, size_t *out_len)
{
    struct stat st;
    if (stat(path, &st) != 0) return NULL;
    if (!S_ISREG(st.st_mode)) { errno = EISDIR; return NULL; }
    *out_len = (size_t)st.st_size;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t *buf = (uint8_t *)malloc(*out_len ? *out_len : 1);
    if (!buf) { fclose(f); return NULL; }
    if (*out_len && fread(buf, 1, *out_len, f) != *out_len) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    return buf;
}

/* Order-0 entropy of one chunk, in bits: -sum p_s log2 p_s scaled by n.
 *
 * No table cost and no framing -- this is the idealized floor, not a
 * size any real coder can hit.  A constant chunk contributes 0. */
static double chunk_entropy_bits(const uint8_t *src, size_t n)
{
    uint64_t f[256] = {0};
    for (size_t i = 0; i < n; i++) f[src[i]]++;
    double bits = 0.0;
    for (int sym = 0; sym < 256; sym++)
        if (f[sym])
            bits += (double)f[sym] * log2((double)n / (double)f[sym]);
    return bits;
}

/* An all-one-byte chunk: RLE for every codec (see the file header). */
static int chunk_is_constant(const uint8_t *src, size_t n)
{
    for (size_t i = 1; i < n; i++) if (src[i] != src[0]) return 0;
    return 1;
}

/* Cost of putting a pivco table on the wire: the 256 code lengths
 * nibble-packed into CODE_LEN_BYTES, then entropy-coded.
 *
 * pivcohuf_file.c stores those 128 bytes raw, which is fine when one
 * table covers a whole file but is a real overcharge per 32 KiB chunk --
 * and an unfair one here, since FSE's own table description inside
 * FSE_compress is entropy-coded.  zstd's huf0 compresses its weight
 * table for exactly this reason, so this charges what a per-chunk pivco
 * container would actually pay: a 1-byte raw/coded flag plus whichever
 * of the two forms is smaller. */
static size_t pivco_header_bytes(const pivco_table_t *table)
{
    uint8_t packed[CODE_LEN_BYTES];
    for (int i = 0; i < CODE_LEN_BYTES; i++) {
        packed[i] = (uint8_t)((table->code_len[2 * i] & 0x0F) |
                              ((table->code_len[2 * i + 1] & 0x0F) << 4));
    }
    uint8_t out[2 * CODE_LEN_BYTES + 512];
    size_t cr = FSE_compress(out, sizeof(out), packed, sizeof(packed));
    if (FSE_isError(cr) || cr <= 1 || cr >= CODE_LEN_BYTES)
        return 1 + CODE_LEN_BYTES;
    return 1 + cr;
}

/* Build the pivco table for `freq` under the given FSE settings.
 *
 * Two builds, as a real container does: the frequency-based one, then a
 * rebuild from its code lengths alone -- the decoder only ever sees the
 * lengths, so the encoder must use the table they reconstruct. */
static int pivco_make_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                            int fse_enabled, int fse_dynamic,
                            pivco_table_t *out)
{
    pivco_cfg_t cfg = pivco_cfg_default;
    cfg.fse_enabled = fse_enabled;
    cfg.fse_dynamic = fse_dynamic;
    cfg.flat_layout = g_flat_layout;

    pivco_table_t real_table;
    if (pivco_build_table(&cfg, freq, &real_table) != PIVCO_OK) return 0;
    if (pivco_build_table_from_code_lens(&cfg, real_table.code_len,
                                         out) != PIVCO_OK) return 0;
    return 1;
}

/* Compress one chunk with an already-built pivco table.  `hdr` is the
 * table cost attributed to this chunk (CODE_LEN_BYTES under per-chunk
 * tables, 0 under a per-file table, which is charged once by the caller).
 * Round-trips the result before reporting a size.
 *
 * Returns the size, or 0 on failure.  *stored is set when the coded form
 * lost to a raw copy. */
static size_t pivco_chunk_size(const pivco_table_t *table,
                               const uint8_t *src, size_t n, size_t hdr,
                               uint8_t *enc, uint8_t *dec, int *stored)
{
    size_t enc_len = 0;
    if (pivco_encode(bench_enc_ctx(), table, src, n, enc, &enc_len) != PIVCO_OK)
        return 0;

    size_t consumed = 0;
    if (pivco_decode(bench_dec_ctx(), table, enc, enc_len, dec, &consumed) != PIVCO_OK)
        return 0;
    if (memcmp(src, dec, n) != 0) return 0;

    size_t coded = BLOCK_TAG_BYTES + hdr + enc_len;
    size_t raw   = BLOCK_TAG_BYTES + n;
    if (coded >= raw) { *stored = 1; return raw; }
    *stored = 0;
    return coded;
}

/* Compress one chunk with stock FSE.  Same accounting and the same
 * round-trip check. */
static size_t fse_chunk_size(const uint8_t *src, size_t n,
                             uint8_t *enc, size_t enc_cap,
                             uint8_t *dec, int *stored)
{
    size_t raw = BLOCK_TAG_BYTES + n;
    size_t cr = FSE_compress(enc, enc_cap, src, n);
    /* 0 = incompressible, 1 = single-symbol RLE; FSE_decompress handles
     * neither.  Constant chunks are already routed to the RLE block type
     * by the caller, so what lands here is genuinely incompressible. */
    if (FSE_isError(cr) || cr <= 1 || BLOCK_TAG_BYTES + cr >= raw) {
        *stored = 1;
        return raw;
    }
    size_t dr = FSE_decompress(dec, n, enc, cr);
    if (FSE_isError(dr) || dr != n || memcmp(src, dec, n) != 0) return 0;
    *stored = 0;
    return BLOCK_TAG_BYTES + cr;
}

/* ---------- reporting ---------- */

/* One results table.  `entropy_bytes` anchors the "vs entropy" column;
 * negative means the codec came in under the order-0 floor. */
static void print_table(uint64_t raw_bytes, double entropy_bytes,
                        const codec_totals_t tot[N_CODECS])
{
    printf("%-12s  %12s  %8s  %10s  %7s\n",
           "codec", "bytes", "%raw", "vs entropy", "stored");
    printf("%-12s  %12s  %8s  %10s  %7s\n",
           "------------", "------------", "--------", "----------", "-------");

    const double e = entropy_bytes > 0.0 ? entropy_bytes : 0.0;
    #define VS(b) (e > 0.0 ? 100.0 * ((double)(b) / e - 1.0) : 0.0)

    printf("%-12s  %12llu  %7.2f%%  ", "raw",
           (unsigned long long)raw_bytes, 100.0);
    if (e > 0.0) printf("%+9.2f%%", VS(raw_bytes)); else printf("%10s", "-");
    printf("  %7s\n", "-");

    printf("%-12s  %12.0f  %7.2f%%  %10s  %7s\n", "entropy O0", e,
           raw_bytes ? 100.0 * e / (double)raw_bytes : 0.0, "--", "-");

    for (int c = 0; c < N_CODECS; c++) {
        double pct = raw_bytes ? 100.0 * (double)tot[c].bytes / (double)raw_bytes : 0.0;
        printf("%-12s  %12llu  %7.2f%%  ",
               codec_name[c], (unsigned long long)tot[c].bytes, pct);
        if (e > 0.0) printf("%+9.2f%%", VS(tot[c].bytes)); else printf("%10s", "-");
        printf("  %7llu\n", (unsigned long long)tot[c].raw_chunks);
    }
    #undef VS
}

/* ---------- per-file run ---------- */

static int run_one(const char *path, size_t chunk, codec_totals_t grand[N_CODECS],
                   uint64_t *grand_raw, double *grand_entropy_bits)
{
    size_t len;
    uint8_t *buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        return 1;
    }
    if (len == 0) {
        fprintf(stderr, "%s: empty, skipped\n", path);
        free(buf);
        return 1;
    }

    /* pivco_encode's contract is PIVCO_MAX_ENCODED_SIZE of output room and
     * PIVCO_BLOCK_SIZE of decode room, independent of the chunk size we
     * happen to feed it -- so size to those, not to `chunk`. */
    size_t enc_cap = FSE_compressBound(chunk);
    if (enc_cap < (size_t)PIVCO_MAX_ENCODED_SIZE) enc_cap = PIVCO_MAX_ENCODED_SIZE;
    uint8_t *enc = (uint8_t *)malloc(enc_cap);
    uint8_t *dec = (uint8_t *)malloc(PIVCO_BLOCK_SIZE);
    if (!enc || !dec) { free(enc); free(dec); free(buf); return 1; }

    codec_totals_t tot[N_CODECS] = {{0, 0, 0}};
    size_t nchunks = 0, last_chunk = 0, nrle = 0;
    double entropy_bits = 0.0;
    int failed = 0;

    /* Table policy.  Per-file is the default because it is what both real
     * containers do: pivcohuf_file.c writes CODE_LENGTHS once in the body
     * header and then blocks at PIVCO_BLOCK_SIZE, and the openzl codec
     * carries a single `weights` stream for the whole input.  Charging
     * 128 bytes per chunk instead misrepresents both, and on files with
     * many chunks that overhead alone can swamp the codec difference
     * being measured.  Default is nonetheless per-chunk, matching what
     * FSE_compress does (a fresh table per chunk) so both sides adapt at
     * the same rate; --table per-file shows the container's policy. */
    struct { int fse, dyn; } variant[3] = { {0, 0}, {1, 0}, {1, 1} };
    pivco_table_t file_table[3];
    size_t per_chunk_hdr = 0;   /* set per chunk when tables are per-chunk */
    if (g_per_file_table) {
        uint64_t freq[PIVCO_MAX_SYMBOLS] = {0};
        if (pivco_histogram(bench_enc_ctx(), buf, len, freq) != PIVCO_OK) {
            free(enc); free(dec); free(buf); return 1;
        }
        for (int v = 0; v < 3; v++) {
            if (!pivco_make_table(freq, variant[v].fse, variant[v].dyn,
                                  &file_table[v])) {
                fprintf(stderr, "%s: table build failed\n", path);
                free(enc); free(dec); free(buf); return 1;
            }
            tot[v].bytes += pivco_header_bytes(&file_table[v]);  /* once per file */
        }
    }

    for (size_t off = 0; off < len; off += chunk) {
        size_t n = (len - off < chunk) ? len - off : chunk;
        const uint8_t *src = buf + off;
        nchunks++;
        last_chunk = n;
        entropy_bits += chunk_entropy_bits(src, n);

        if (chunk_is_constant(src, n)) {
            nrle++;
            for (int c = 0; c < N_CODECS; c++) tot[c].bytes += BLOCK_TAG_BYTES + 1;
            continue;
        }

        for (int v = 0; v < 3; v++) {
            /* Reset per chunk so the dynamic-commit count is attributable
             * to this variant alone. */
            pivco_fse_stats_reset();
            pivco_table_t chunk_table;
            const pivco_table_t *table = &file_table[v];
            if (!g_per_file_table) {
                uint64_t freq[PIVCO_MAX_SYMBOLS] = {0};
                if (pivco_histogram(bench_enc_ctx(), src, n, freq) != PIVCO_OK ||
                    !pivco_make_table(freq, variant[v].fse, variant[v].dyn,
                                      &chunk_table)) {
                    fprintf(stderr, "%s: table build failed at offset %zu\n",
                            path, off);
                    failed = 1;
                    break;
                }
                table = &chunk_table;
                per_chunk_hdr = pivco_header_bytes(table);
            }
            int stored = 0;
            size_t sz = pivco_chunk_size(table, src, n, per_chunk_hdr,
                                         enc, dec, &stored);
            if (sz == 0) {
                fprintf(stderr, "%s: %s failed on chunk at offset %zu\n",
                        path, codec_name[v], off);
                failed = 1;
                break;
            }
            tot[v].bytes += sz;
            tot[v].raw_chunks += (uint64_t)stored;
            if (v == C_PIVCO_DANS) {
                uint64_t commit[PIVCO_FSE_STATS_SLOTS], attempt[PIVCO_FSE_STATS_SLOTS];
                uint64_t bin[PIVCO_FSE_STATS_SLOTS], bout[PIVCO_FSE_STATS_SLOTS];
                pivco_fse_stats_get(commit, attempt, bin, bout);
                tot[v].dyn_commits += commit[PIVCO_FSE_DYNAMIC_ID];
            }
        }
        if (failed) break;

        int stored = 0;
        size_t sz = fse_chunk_size(src, n, enc, enc_cap, dec, &stored);
        if (sz == 0) {
            fprintf(stderr, "%s: fse failed on chunk at offset %zu\n", path, off);
            failed = 1;
            break;
        }
        tot[C_FSE].bytes += sz;
        tot[C_FSE].raw_chunks += (uint64_t)stored;
    }

    free(enc); free(dec); free(buf);
    if (failed) return 1;

    printf("=== %s ===\n", path);
    printf("%zu bytes, %zu chunk%s of %zu (last %zu), %zu constant (RLE)\n",
           len, nchunks, nchunks == 1 ? "" : "s", chunk, last_chunk, nrle);
    print_table((uint64_t)len, entropy_bits / 8.0, tot);
    printf("pivco+dANS committed the dynamic nibble table on %llu bitmap%s\n\n",
           (unsigned long long)tot[C_PIVCO_DANS].dyn_commits,
           tot[C_PIVCO_DANS].dyn_commits == 1 ? "" : "s");

    for (int c = 0; c < N_CODECS; c++) {
        grand[c].bytes       += tot[c].bytes;
        grand[c].raw_chunks  += tot[c].raw_chunks;
        grand[c].dyn_commits += tot[c].dyn_commits;
    }
    *grand_raw += len;
    *grand_entropy_bits += entropy_bits;
    return 0;
}

int main(int argc, char **argv)
{
    size_t chunk = 32768;
    int first_file = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--table") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if      (!strcmp(v, "per-file"))  g_per_file_table = 1;
            else if (!strcmp(v, "per-chunk")) g_per_file_table = 0;
            else { fprintf(stderr, "--table must be per-file|per-chunk\n"); return 2; }
            first_file = i + 1;
        } else if (strcmp(argv[i], "--flat") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if      (!strcmp(v, "natural"))     g_flat_layout = PIVCO_FLAT_NATURAL;
            else if (!strcmp(v, "vertical"))    g_flat_layout = PIVCO_FLAT_VERTICAL;
            else if (!strcmp(v, "vertical128")) g_flat_layout = PIVCO_FLAT_VERTICAL_128;
            else { fprintf(stderr, "--flat must be natural|vertical|vertical128\n"); return 2; }
            g_flat_name = v;
            first_file = i + 1;
        } else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 0);
            if (v < 1 || v > PIVCO_BLOCK_SIZE) {
                fprintf(stderr, "--chunk must be in [1, %d] (the codec's "
                                "max block size)\n", PIVCO_BLOCK_SIZE);
                return 2;
            }
            chunk = (size_t)v;
            first_file = i + 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--chunk BYTES] [--flat LAYOUT] [--table POLICY] FILE [...]\n"
                   "  Compresses each file in independent chunks (default 32768 B)\n"
                   "  with pivco / pivco+static-ANS / pivco+dynamic-ANS / fse and\n"
                   "  reports header-inclusive sizes.\n"
                   "  --flat  natural|vertical|vertical128 (default vertical)\n"
                   "  --table per-file|per-chunk           (default per-file)\n",
                   argv[0]);
            return 0;
        } else {
            first_file = i;
            break;
        }
    }

    if (first_file >= argc) {
        fprintf(stderr, "Usage: %s [--chunk BYTES] [--flat LAYOUT] FILE [FILE ...]\n",
                argv[0]);
        return 2;
    }

    codec_totals_t grand[N_CODECS] = {{0, 0, 0}};
    uint64_t grand_raw = 0;
    double grand_entropy_bits = 0.0;
    int nfiles = 0, nfail = 0;

    for (int i = first_file; i < argc; i++) {
        if (run_one(argv[i], chunk, grand, &grand_raw, &grand_entropy_bits) == 0)
            nfiles++;
        else
            nfail++;
    }

    if (nfiles > 1) {
        printf("=== TOTAL (%d files) ===\n", nfiles);
        print_table(grand_raw, grand_entropy_bits / 8.0, grand);
        printf("\n");
    }

    printf("# flat layout: %s   huffman table: %s\n",
           g_flat_name, g_per_file_table ? "per-file" : "per-chunk");
    printf("# sizes include a %d-byte block tag per chunk for every codec, plus an\n"
           "# entropy-coded Huffman code-length table for the pivco columns (%d\n"
           "# bytes nibble-packed, then FSE'd, as huf0 does with its weights) --\n"
           "# once per chunk by default, once per file under --table per-file.\n"
           "# fse's own table description is inside its payload, per chunk always.\n"
           "# 'stored' counts chunks that lost to a raw copy; constant chunks are\n"
           "# RLE for every codec alike.  Every reported chunk round-trips.\n"
           "# 'entropy O0' is the summed per-chunk order-0 entropy with no table\n"
           "# or framing cost at all -- an unreachable floor for an order-0 coder,\n"
           "# so a negative 'vs entropy' means that codec beat order-0 outright.\n",
           BLOCK_TAG_BYTES, CODE_LEN_BYTES);

    return nfail ? 1 : 0;
}
