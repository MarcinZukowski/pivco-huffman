/* phaz_codec.h -- buffer API + shared helpers for the pivco-Huffman entropy
 * transplant onto zstd.  Implemented in phaz_codec.c against the patched
 * libzstd (capture hook + ZSTD_phazDecode) and libpivco_huffman (PH/PHA stream
 * codec).  The CLI (tools/phaz.c) and the TurboBench plugin both build on this;
 * the TurboBench blob exports only phaz_compress/phaz_decompress (everything
 * else -- zstd, FSE/HUF, the g_phaz_* capture globals -- is localized).
 */
#ifndef PHAZ_CODEC_H
#define PHAZ_CODEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Optional per-step timing + size report (nullable in the calls below). */
typedef struct {
    double capture_ms;       /* compress: zstd parse + capture hook */
    double pack_ms[4];       /* compress: PH-encode each stream (ll/ml/of/lit) */
    double entropy_ms[4];    /* decompress: PH-decode each stream */
    double reconstruct_ms;   /* decompress: ZSTD_phazDecode (seq replay + copy) */
    size_t stream_raw[4];    /* raw per-stream byte counts */
    size_t stream_enc[4];    /* encoded per-stream byte counts */
} phaz_stats;

extern const char *const phaz_stream_names[4];   /* {"ll","ml","of","lit"} */

/* Worst-case container size for an `n`-byte input. */
size_t phaz_compress_bound(size_t n);

/* Compress src[0..n) into dst (capacity cap) as a phaz container at the given
 * zstd `level`.  Returns container size, or 0 on error / insufficient capacity.
 * `st` (nullable) receives per-step timing + sizes. */
size_t phaz_compress(const void *src, size_t n, void *dst, size_t cap,
                     int level, phaz_stats *st);

/* Decompress a phaz container src[0..n) into dst (capacity cap).  Returns the
 * original byte count, or 0 on error.  `st` (nullable) receives timing. */
size_t phaz_decompress(const void *src, size_t n, void *dst, size_t cap,
                       phaz_stats *st);

/* ---- Lower-level pieces, exposed for the CLI's analysis commands (stats /
 *      dump).  Plain compress/decompress users can ignore everything below. ---- */

/* Patched-libzstd capture-hook globals: a ZSTD_compress2 with g_phaz_dump=1
 * fills these with the pivoted ll/ml/of/lit streams + per-block metadata. */
extern int g_phaz_dump;
extern unsigned char *g_phaz_llc, *g_phaz_mlc, *g_phaz_ofc, *g_phaz_lit, *g_phaz_xb;
extern unsigned long long g_phaz_xbpos;
extern unsigned *g_phaz_blk_ns, *g_phaz_blk_tl;
extern unsigned char *g_phaz_blk_cf;             /* per block: repcodes confirmed? */
extern size_t g_phaz_nblk, g_phaz_nseq, g_phaz_lits;
extern unsigned long long g_phaz_extrabits;

/* Reconstruct: replay the captured sequences via zstd's exec/copy engine.
 * blkCf[nblk] flags blocks whose repcodes zstd confirmed (raw/RLE blocks revert
 * the repcode state); pass NULL for legacy continuous-carry behaviour. */
extern size_t ZSTD_phazDecode(void *dst, size_t dstCap,
    const unsigned char *llc, const unsigned char *mlc, const unsigned char *ofc,
    const unsigned char *xb, const unsigned char *lit, size_t litSize,
    const unsigned *blkNs, const unsigned *blkTl, const unsigned char *blkCf, size_t nblk);

/* Run zstd's compressor with the capture hook, (re)allocating + filling the
 * g_phaz_* globals.  want_stock!=0 also does a plain compress and returns the
 * stock zstd size; else returns 0.  Returns (size_t)-1 on error.  Frees any
 * globals from a previous call first, so it is safe to call repeatedly. */
size_t phaz_capture_run(const unsigned char *src, size_t n, int level, int want_stock);

/* Free + null the g_phaz_* globals (idempotent). */
void phaz_capture_free(void);

/* PHA-encode raw[0..rawlen) into one self-describing blob (method+len+blob).
 * If cur!=NULL, append to *cur (advanced), failing with 0 if it would pass end.
 * cur==NULL = size only.  Returns container bytes; best_out + tag_out (nullable)
 * report the chosen encoded size + tag ('a'=PHA, 'r'=raw). */
size_t phaz_pack_stream(unsigned char **cur, unsigned char *end,
                        const unsigned char *raw, size_t rawlen,
                        size_t *best_out, char *tag_out);

#ifdef __cplusplus
}
#endif

#endif /* PHAZ_CODEC_H */
