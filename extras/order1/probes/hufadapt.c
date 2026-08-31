/* hufadapt: zstd-style entropy arms for the phaz-vs-zstd waterfall.
 *
 *   hufadapt lit|code FILE
 *
 * Z arm (zstd-style adaptive): fresh table per 128 K chunk, table
 * transmission included in the coder's own output (+3 B section hdr,
 * raw fallback per chunk).
 * G arm (phaz-style global): one table built on the whole file,
 * transmitted once, used on every chunk.  Z - G isolates what
 * per-block table adaptation is worth on this stream, inside the
 * same coder family zstd uses (HUF for literals, FSE for codes).
 */
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include "fse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define CHUNK 131072

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: hufadapt lit|code FILE\n"); return 1; }
    int lit = !strcmp(argv[1], "lit");
    struct stat st;
    if (stat(argv[2], &st)) { perror("stat"); return 1; }
    size_t n = (size_t)st.st_size;
    uint8_t *buf = malloc(n ? n : 1);
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(buf, 1, n, f) != n) { perror("read"); return 1; }
    fclose(f);

    uint8_t *dst = malloc(CHUNK * 2 + 4096);

    /* ---- Z arm: fresh table per chunk ---- */
    uint64_t Z = 0;
    for (size_t off = 0; off < n; off += CHUNK) {
        size_t len = n - off < CHUNK ? n - off : CHUNK;
        size_t c = lit ? HUF_compress(dst, CHUNK * 2 + 4096, buf + off, len)
                       : FSE_compress(dst, CHUNK * 2 + 4096, buf + off, len);
        int err = lit ? HUF_isError(c) : FSE_isError(c);
        Z += (!err && c > 1 && c < len) ? c + 3 : len + 3;
    }

    /* ---- G arm: one global table ---- */
    unsigned count[256] = {0};
    unsigned maxSym = 0;
    for (size_t i = 0; i < n; i++) count[buf[i]]++;
    for (unsigned s2 = 0; s2 < 256; s2++) if (count[s2]) maxSym = s2;

    uint64_t G = 0, tbl = 0;
    if (lit) {
        HUF_CREATE_STATIC_CTABLE(ct, 255);
        size_t maxbits = HUF_buildCTable(ct, count, maxSym, 11);
        if (HUF_isError(maxbits)) { fprintf(stderr, "buildCTable fail\n"); return 1; }
        uint8_t tb[512];
        size_t tw = HUF_writeCTable(tb, sizeof(tb), ct, maxSym, (unsigned)maxbits);
        tbl = HUF_isError(tw) ? 128 : tw;
        for (size_t off = 0; off < n; off += CHUNK) {
            size_t len = n - off < CHUNK ? n - off : CHUNK;
            size_t c = HUF_compress1X_usingCTable(dst, CHUNK * 2 + 4096,
                                                  buf + off, len, ct);
            G += (!HUF_isError(c) && c > 0 && c < len) ? c + 3 : len + 3;
        }
    } else {
        unsigned tlog = FSE_optimalTableLog(0, n, maxSym);
        short norm[256];
        if (FSE_isError(FSE_normalizeCount(norm, tlog, count, n, maxSym))) {
            fprintf(stderr, "normalize fail\n"); return 1;
        }
        uint8_t nc[512];
        size_t nw = FSE_writeNCount(nc, sizeof(nc), norm, maxSym, tlog);
        tbl = FSE_isError(nw) ? 64 : nw;
        FSE_CTable *ct = FSE_createCTable(maxSym, tlog);
        if (FSE_isError(FSE_buildCTable(ct, norm, maxSym, tlog))) {
            fprintf(stderr, "buildCTable fail\n"); return 1;
        }
        for (size_t off = 0; off < n; off += CHUNK) {
            size_t len = n - off < CHUNK ? n - off : CHUNK;
            size_t c = FSE_compress_usingCTable(dst, CHUNK * 2 + 4096,
                                                buf + off, len, ct);
            G += (!FSE_isError(c) && c > 0 && c < len) ? c + 3 : len + 3;
        }
    }
    G += tbl;
    printf("HA %s raw %zu Z %llu G %llu tbl %llu\n", argv[1], n,
           (unsigned long long)Z, (unsigned long long)G,
           (unsigned long long)tbl);
    return 0;
}
