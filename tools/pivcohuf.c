/* pivcohuf -- CLI for compress/decompress using the pivco-huffman
 * file format (see include/pivcohuf_file.h).
 *
 * Usage:
 *   pivcohuf c IN [OUT]     compress IN to OUT (default IN.ph)
 *   pivcohuf d IN [OUT]     decompress IN to OUT (default IN with .ph stripped)
 *   pivcohuf c -            read stdin, write stdout
 *   pivcohuf d -            same
 *   -k                      keep input file (default: keep)
 *   -f                      overwrite output if it exists
 *
 * Always prints: input size, output size, ratio, time, bandwidth.
 */

#include "pivcohuf_file.h"
#include "pivco_prof.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define EXT ".ph"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "pivcohuf: out of memory (%zu bytes)\n", n); exit(2); }
    return p;
}

static int read_all(const char *path, uint8_t **out_buf, size_t *out_len) {
    int from_stdin = (strcmp(path, "-") == 0);

    if (!from_stdin) {
        /* Fast path: stat the file, allocate exact size, single fread.
         * Avoids the doubling-realloc memcpy churn (which costs O(N) of
         * extra memcpy work on top of the actual read for a 1 GB file). */
        struct stat st;
        if (stat(path, &st) != 0) {
            fprintf(stderr, "pivcohuf: cannot stat '%s': %s\n", path, strerror(errno));
            return -1;
        }
        if (S_ISREG(st.st_mode)) {
            size_t len = (size_t)st.st_size;
            uint8_t *buf = (uint8_t *)xmalloc(len > 0 ? len : 1);
            FILE *f = fopen(path, "rb");
            if (!f) {
                fprintf(stderr, "pivcohuf: cannot open '%s' for read: %s\n",
                        path, strerror(errno));
                free(buf);
                return -1;
            }
            size_t got = fread(buf, 1, len, f);
            fclose(f);
            if (got != len) {
                fprintf(stderr, "pivcohuf: short read on '%s' (%zu / %zu)\n",
                        path, got, len);
                free(buf);
                return -1;
            }
            *out_buf = buf;
            *out_len = len;
            return 0;
        }
        /* Non-regular file (FIFO, char device, etc.): fall through to
         * the doubling-buffer path below. */
    }

    /* Stdin or non-regular file: size unknown, grow buffer dynamically. */
    FILE *f = from_stdin ? stdin : fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "pivcohuf: cannot open '%s' for read: %s\n",
                path, strerror(errno));
        return -1;
    }
    size_t cap = 1 << 20, len = 0;
    uint8_t *buf = (uint8_t *)xmalloc(cap);
    for (;;) {
        if (len == cap) {
            cap *= 2;
            buf = (uint8_t *)realloc(buf, cap);
            if (!buf) { fprintf(stderr, "pivcohuf: OOM growing read buffer\n"); return -1; }
        }
        size_t got = fread(buf + len, 1, cap - len, f);
        len += got;
        if (got == 0) break;
    }
    if (!from_stdin) fclose(f);
    *out_buf = buf;
    *out_len = len;
    return 0;
}

static int write_all(const char *path, const uint8_t *buf, size_t len, int force) {
    int to_stdout = (strcmp(path, "-") == 0);
    FILE *f;
    if (to_stdout) {
        f = stdout;
    } else {
        if (!force) {
            struct stat st;
            if (stat(path, &st) == 0) {
                fprintf(stderr, "pivcohuf: '%s' already exists (use -f to overwrite)\n", path);
                return -1;
            }
        }
        f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "pivcohuf: cannot open '%s' for write: %s\n",
                    path, strerror(errno));
            return -1;
        }
    }
    size_t wrote = fwrite(buf, 1, len, f);
    if (!to_stdout) fclose(f);
    if (wrote != len) {
        fprintf(stderr, "pivcohuf: short write (%zu / %zu)\n", wrote, len);
        return -1;
    }
    return 0;
}

static const char *err_msg(int rc) {
    switch (rc) {
    case PIVCOHUF_OK:                      return "ok";
    case PIVCOHUF_ERR_NULL:                return "null pointer";
    case PIVCOHUF_ERR_TOO_SHORT:           return "input too short / truncated";
    case PIVCOHUF_ERR_BAD_MAGIC:           return "bad magic (not a pivcohuf file)";
    case PIVCOHUF_ERR_BAD_VERSION:         return "unsupported version";
    case PIVCOHUF_ERR_BAD_HEADER_CHECKSUM: return "header checksum mismatch";
    case PIVCOHUF_ERR_BAD_BODY_CHECKSUM:   return "body checksum mismatch (data corruption)";
    case PIVCOHUF_ERR_BAD_BLOCK_SIZE:      return "encoded block size does not match this build's codec";
    case PIVCOHUF_ERR_OUTPUT_TOO_SMALL:    return "output buffer too small";
    case PIVCOHUF_ERR_INTERNAL:            return "internal error";
    default:                                return "unknown error";
    }
}

static void print_stats(const char *op, size_t in_bytes, size_t out_bytes, double secs) {
    /* "compress" uses input as the throughput basis; "decompress" uses
     * output (the produced bytes are the work).  Both are reported. */
    double ratio = (in_bytes > 0) ? (double)out_bytes / (double)in_bytes : 0.0;
    double in_mb = (double)in_bytes / 1.0e6;
    double out_mb = (double)out_bytes / 1.0e6;
    double bw_in = secs > 0 ? in_mb / secs : 0.0;
    double bw_out = secs > 0 ? out_mb / secs : 0.0;
    fprintf(stderr, "%-10s in=%zu out=%zu  ratio=%.4f  time=%.3f ms  "
            "in=%.1f MB/s  out=%.1f MB/s\n",
            op, in_bytes, out_bytes, ratio, secs * 1000.0, bw_in, bw_out);
}

static void usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  pivcohuf c IN [OUT]   compress (default OUT = IN" EXT ")\n"
        "  pivcohuf d IN [OUT]   decompress (default OUT = IN with " EXT " stripped)\n"
        "  pivcohuf c -          stdin/stdout\n"
        "Flags:\n"
        "  -f                    overwrite OUT if it exists\n");
}

int main(int argc, char **argv)
{
    int force = 0;
    /* First pass: pluck flags anywhere on the command line. */
    const char *positionals[4] = {0};
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] != '\0' && argv[i][2] == '\0'
            && argv[i][1] == 'f') {
            force = 1;
        } else if (npos < (int)(sizeof positionals / sizeof positionals[0])) {
            positionals[npos++] = argv[i];
        }
    }
    if (npos < 2) { usage(); return 1; }
    const char *cmd = positionals[0];
    const char *in_path = positionals[1];
    char default_out[4096];
    const char *out_path;
    if (npos >= 3) {
        out_path = positionals[2];
    } else if (strcmp(in_path, "-") == 0) {
        (void)default_out;
        out_path = "-";
    } else if (cmd[0] == 'c') {
        snprintf(default_out, sizeof default_out, "%s%s", in_path, EXT);
        out_path = default_out;
    } else if (cmd[0] == 'd') {
        size_t n = strlen(in_path);
        size_t ext_len = strlen(EXT);
        if (n > ext_len && strcmp(in_path + n - ext_len, EXT) == 0) {
            memcpy(default_out, in_path, n - ext_len);
            default_out[n - ext_len] = '\0';
        } else {
            snprintf(default_out, sizeof default_out, "%s.out", in_path);
        }
        out_path = default_out;
    } else {
        usage(); return 1;
    }

    uint8_t *in_buf = NULL;
    size_t in_len = 0;
    if (read_all(in_path, &in_buf, &in_len) != 0) return 2;

    if (cmd[0] == 'c') {
        size_t bound = pivcohuf_compress_bound(in_len);
        uint8_t *out_buf = (uint8_t *)xmalloc(bound);
        size_t out_len = bound;
        double t0 = now_sec();
        int rc = pivcohuf_compress(in_buf, in_len, out_buf, &out_len);
        double t1 = now_sec();
        if (rc != PIVCOHUF_OK) {
            fprintf(stderr, "pivcohuf: compress failed: %s\n", err_msg(rc));
            return 2;
        }
        if (write_all(out_path, out_buf, out_len, force) != 0) return 2;
        print_stats("compress", in_len, out_len, t1 - t0);
#ifdef PIVCO_PROF
        pivco_prof_dump("pivcohuf compress", t1 - t0,
                         pivco_prof_probe_tick_freq(),
                         (uint64_t)((in_len + 8191) / 8192));
#endif
        free(out_buf);
    } else if (cmd[0] == 'd') {
        size_t uncomp_size = 0;
        int rc = pivcohuf_peek_uncompressed_size(in_buf, in_len, &uncomp_size);
        if (rc != PIVCOHUF_OK) {
            fprintf(stderr, "pivcohuf: cannot peek header: %s\n", err_msg(rc));
            return 2;
        }
        uint8_t *out_buf = (uint8_t *)xmalloc(uncomp_size > 0 ? uncomp_size : 1);
        size_t out_len = uncomp_size;
        double t0 = now_sec();
        rc = pivcohuf_decompress(in_buf, in_len, out_buf, &out_len);
        double t1 = now_sec();
        if (rc != PIVCOHUF_OK) {
            fprintf(stderr, "pivcohuf: decompress failed: %s\n", err_msg(rc));
            return 2;
        }
        if (write_all(out_path, out_buf, out_len, force) != 0) return 2;
        print_stats("decompress", in_len, out_len, t1 - t0);
#ifdef PIVCO_PROF
        pivco_prof_dump("pivcohuf decompress", t1 - t0,
                         pivco_prof_probe_tick_freq(),
                         (uint64_t)((out_len + 8191) / 8192));
#endif
        free(out_buf);
    } else {
        usage(); return 1;
    }
    free(in_buf);
    return 0;
}
