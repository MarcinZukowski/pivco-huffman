/* try.c — minimal demo of pivco-huffman as a library.
 *
 * Reads a file into memory, compresses it with PH and PHA, verifies the
 * roundtrip, and prints compression ratio + encode/decode throughput for each.
 * Nothing here touches the wire format — it's just the public buffer API in
 * include/pivcohuf_file.h, so you can drop the same four calls into your own
 * code to measure pivco-huffman on your own data.
 *
 * Build (after `cmake --build build`):
 *   cc -O3 -I include examples/try.c build/libpivco_huffman.a -lm -o try
 *   ./try <file>
 * Or via CMake: build target `pivco_try`, then `./build/pivco_try <file>`.
 */
#include "pivcohuf_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* MB/s (SI, 1e6 B/s) for `n` bytes processed in `ns` nanoseconds. */
static double mbps(size_t n, double ns) { return (double)n * 1e3 / ns; }

static void run(const char *label, int use_ans, const uint8_t *in, size_t n) {
    /* compress */
    size_t cap = pivcohuf_compress_bound(n);
    uint8_t *comp = malloc(cap);
    size_t clen = cap;
    double t0 = now_ns();
    int rc = pivcohuf_compress_ex(in, n, comp, &clen, use_ans);
    double t1 = now_ns();
    if (rc != PIVCOHUF_OK) { fprintf(stderr, "%s: compress failed (%d)\n", label, rc); exit(1); }

    /* decompress (same call for ph and pha — the decoder auto-detects) */
    size_t usz = 0;
    pivcohuf_peek_uncompressed_size(comp, clen, &usz);
    uint8_t *dec = malloc(usz);
    size_t dlen = usz;
    double t2 = now_ns();
    rc = pivcohuf_decompress(comp, clen, dec, &dlen);
    double t3 = now_ns();
    if (rc != PIVCOHUF_OK) { fprintf(stderr, "%s: decompress failed (%d)\n", label, rc); exit(1); }

    int ok = (dlen == n && memcmp(in, dec, n) == 0);
    printf("  %-4s  %.2fx  (%zu -> %zu)   enc %.0f MB/s   dec %.0f MB/s   %s\n",
           label, (double)n / (double)clen, n, clen,
           mbps(n, t1 - t0), mbps(n, t3 - t2),
           ok ? "roundtrip ok" : "ROUNDTRIP MISMATCH");
    free(comp);
    free(dec);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file>\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fprintf(stderr, "empty or unreadable file\n"); return 1; }
    uint8_t *buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { perror("read"); return 1; }
    fclose(f);

    printf("%s (%ld bytes)   [ratio = in/out, higher = better]\n", argv[1], n);
    run("ph",  0, buf, (size_t)n);   /* plain Huffman              */
    run("pha", 1, buf, (size_t)n);   /* + ANS-coded bitmaps        */
    free(buf);
    return 0;
}
