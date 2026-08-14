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
#include "pivco_huffman.h"
#include "pivco_prof.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
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
    case PIVCOHUF_ERR_BAD_BLOCK_SIZE:      return "invalid block size (must be 1..65535)";
    case PIVCOHUF_ERR_OUTPUT_TOO_SMALL:    return "output buffer too small";
    case PIVCOHUF_ERR_INTERNAL:            return "internal error";
    default:                                return "unknown error";
    }
}

static void print_stats(const char *op, size_t in_bytes, size_t out_bytes,
                         double secs_codec, double secs_total) {
    double ratio = (in_bytes > 0) ? (double)out_bytes / (double)in_bytes : 0.0;
    double bw_in  = secs_codec > 0 ? (double)in_bytes  / 1.0e6 / secs_codec : 0.0;
    double bw_out = secs_codec > 0 ? (double)out_bytes / 1.0e6 / secs_codec : 0.0;
    int total_ms    = (int)(secs_total * 1000.0 + 0.5);
    int comp_ms     = (int)(secs_codec * 1000.0 + 0.5);
    int overhead_ms = total_ms - comp_ms;
    if (overhead_ms < 0) overhead_ms = 0;
    fprintf(stderr, "%-10s in=%zu out=%zu  ratio=%.4f\n",
            op, in_bytes, out_bytes, ratio);
    fprintf(stderr, "           total_time:%dms overhead:%dms comp:%dms  "
                    "comp_bw in=%d MB/s out=%d MB/s\n",
            total_ms, overhead_ms, comp_ms,
            (int)(bw_in + 0.5), (int)(bw_out + 0.5));
}

/* Detailed per-phase breakdown (ms).  io/malloc are measured by the CLI;
 * freq/build/codec come from the library's pivcohuf_timing_t.  malloc folds
 * the CLI's output-buffer alloc with the codec's internal scratch.  freq is
 * the symbol histogram (compress only) -- separate from build (tree/codes),
 * since a caller with known frequencies skips it. */
static void print_phases(const char *codec_label, double io_read_ms,
                         double io_write_ms, double cli_malloc_ms,
                         const pivcohuf_timing_t *tm) {
    double malloc_ms = cli_malloc_ms + tm->malloc_ns / 1.0e6;
    fprintf(stderr, "           phases: io(rd=%.2f wr=%.2f) malloc=%.2f",
            io_read_ms, io_write_ms, malloc_ms);
    if (tm->freq_ns > 0.0)
        fprintf(stderr, " freq=%.2f", tm->freq_ns / 1.0e6);
    fprintf(stderr, " build=%.2f %s=%.2f  (ms)\n",
            tm->build_ns / 1.0e6, codec_label, tm->codec_ns / 1.0e6);
}

static void usage(FILE *out) {
    fprintf(out,
        "Usage:\n"
        "  pivcohuf c IN [OUT]   compress (default OUT = IN" EXT ")\n"
        "  pivcohuf d IN [OUT]   decompress (default OUT = IN with " EXT " stripped)\n"
        "  pivcohuf c -          stdin/stdout\n"
        "Flags:\n"
        "  -a, --ans             compress with PHA (ANS-coded bitmaps; better\n"
        "                        ratio on skewed data).  decompress auto-detects.\n"
        "  -b, --block-size N    symbols per block (1..65535; default per-arch).\n"
        "                        recorded in the stream; decompress reads it back.\n"
        "  -e, --effort N        compress-time shaping effort (0..4, default 1):\n"
        "                        0 simplest, 1 balanced, 2 faster-decompress,\n"
        "                        3 fastest-decompress, 4 fastest-compress\n"
        "                        (simplest under 256 KiB, else balanced).\n"
        "                        wire-compatible; decompress needs no flag.\n"
        "  -f                    overwrite OUT if it exists\n"
        "  -r N                  re-run codec N times into the same buffer\n"
        "                        (no extra I/O); reports per-iter timing\n"
        "  -h, --help            show this help and exit\n");
}

int main(int argc, char **argv)
{
    pivco_cfg_t cli_cfg = { PIVCO_TREE_MODE_OPTIMIZED, PIVCO_EFFORT_PLAIN, 0 };
    int force = 0;
    int repeat = 1;
    int use_ans = 0;   /* -a / --ans : compress with #PHA (ANS-coded bitmaps) */
    size_t block_size = PIVCO_BLOCK_SIZE;  /* -b / --block-size : symbols/block */
    /* First pass: pluck flags anywhere on the command line. */
    const char *positionals[4] = {0};
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] != '\0' && argv[i][2] == '\0'
            && argv[i][1] == 'f') {
            force = 1;
        } else if ((argv[i][0] == '-' && argv[i][1] == 'a' && argv[i][2] == '\0')
                   || strcmp(argv[i], "--ans") == 0) {
            use_ans = 1;
        } else if (argv[i][0] == '-' && argv[i][1] == 'r' && argv[i][2] == '\0'
                   && i + 1 < argc) {
            repeat = atoi(argv[i + 1]);
            if (repeat < 1) repeat = 1;
            i++;   /* skip the N */
        } else if ((argv[i][0] == '-' && argv[i][1] == 'b' && argv[i][2] == '\0'
                    && i + 1 < argc)) {
            block_size = (size_t)strtoul(argv[i + 1], NULL, 0);
            i++;   /* skip the N */
        } else if (strncmp(argv[i], "--block-size=", 13) == 0) {
            block_size = (size_t)strtoul(argv[i] + 13, NULL, 0);
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            block_size = (size_t)strtoul(argv[i + 1], NULL, 0);
            i++;
        } else if ((argv[i][0] == '-' && argv[i][1] == 'e' && argv[i][2] == '\0'
                    && i + 1 < argc)) {
            cli_cfg.effort = (pivco_effort_t)atoi(argv[i + 1]);
            i++;   /* skip the N */
        } else if (strncmp(argv[i], "--effort=", 9) == 0) {
            cli_cfg.effort = (pivco_effort_t)atoi(argv[i] + 9);
        } else if (strcmp(argv[i], "--effort") == 0 && i + 1 < argc) {
            cli_cfg.effort = (pivco_effort_t)atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            /* Any other dash-prefixed token is an unknown option.  ("-" alone
             * is the stdin/stdout sentinel and falls through to positionals.) */
            fprintf(stderr, "pivcohuf: unknown option '%s'\n", argv[i]);
            usage(stderr);
            return 1;
        } else if (npos < (int)(sizeof positionals / sizeof positionals[0])) {
            positionals[npos++] = argv[i];
        } else {
            fprintf(stderr, "pivcohuf: too many arguments ('%s')\n", argv[i]);
            usage(stderr);
            return 1;
        }
    }
    if (npos < 2) { usage(stderr); return 1; }
    if (block_size < 1 || block_size > PIVCO_WIRE_MAX_N) {
        fprintf(stderr, "pivcohuf: --block-size must be in 1..%d\n",
                PIVCO_WIRE_MAX_N);
        return 1;
    }
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
        usage(stderr); return 1;
    }

    /* Total wall covers the whole user-visible operation: read input,
     * allocate output, prep (madvise on decompress), codec, write
     * output.  overhead = total - codec. */
    double t_total_start = now_sec();

    uint8_t *in_buf = NULL;
    size_t in_len = 0;
    double _rd0 = now_sec();
    if (read_all(in_path, &in_buf, &in_len) != 0) return 2;
    double io_read_ms = (now_sec() - _rd0) * 1000.0;

    if (cmd[0] == 'c') {
        size_t bound = pivcohuf_compress_bound_blk(in_len, block_size);
        double _m0 = now_sec();
        uint8_t *out_buf = (uint8_t *)xmalloc(bound);
        double cli_malloc_ms = (now_sec() - _m0) * 1000.0;
        size_t out_len = bound;
        pivcohuf_timing_t tm;
        double t0 = now_sec();
        cli_cfg.fse_enabled = use_ans;
        int rc = pivcohuf_compress_cfg(in_buf, in_len, out_buf, &out_len, &cli_cfg, block_size, &tm);
        double t1 = now_sec();
        if (rc != PIVCOHUF_OK) {
            fprintf(stderr, "pivcohuf: compress failed: %s\n", err_msg(rc));
            return 2;
        }
        double _w0 = now_sec();
        if (write_all(out_path, out_buf, out_len, force) != 0) return 2;
        double io_write_ms = (now_sec() - _w0) * 1000.0;
        double t_total_end = now_sec();
        print_stats(use_ans ? "compress (pha)" : "compress", in_len, out_len, t1 - t0, t_total_end - t_total_start);
        print_phases("encode", io_read_ms, io_write_ms, cli_malloc_ms, &tm);
        if (repeat > 1) {
            fprintf(stderr, "  -- replaying compress %d more times into same buffer --\n", repeat - 1);
            for (int r = 1; r < repeat; r++) {
                size_t rep_out_len = bound;
                double rt0 = now_sec();
                pivcohuf_compress_cfg(in_buf, in_len, out_buf, &rep_out_len, &cli_cfg, block_size, NULL);
                double rt1 = now_sec();
                int ms = (int)((rt1 - rt0) * 1000.0 + 0.5);
                fprintf(stderr, "  iter %2d: comp:%dms  comp_bw in=%d MB/s out=%d MB/s\n",
                        r + 1, ms,
                        (int)((double)in_len     / 1.0e6 / (rt1 - rt0) + 0.5),
                        (int)((double)rep_out_len / 1.0e6 / (rt1 - rt0) + 0.5));
            }
        }
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
        double _m0 = now_sec();
        uint8_t *out_buf = (uint8_t *)xmalloc(uncomp_size > 0 ? uncomp_size : 1);
        double cli_malloc_ms = (now_sec() - _m0) * 1000.0;
        size_t out_len = uncomp_size;
        /* madvise WILLNEED on the output buffer: hints the kernel to
         * populate pages so the codec doesn't take minor page faults
         * inside its timed loop.  ~2% wall improvement on 1 GB. */
        madvise(out_buf, uncomp_size, MADV_WILLNEED);
        pivcohuf_timing_t tm;
        double t0 = now_sec();
        rc = pivcohuf_decompress_timed(in_buf, in_len, out_buf, &out_len, &tm);
        double t1 = now_sec();
        if (rc != PIVCOHUF_OK) {
            fprintf(stderr, "pivcohuf: decompress failed: %s\n", err_msg(rc));
            return 2;
        }
        double _w0 = now_sec();
        if (write_all(out_path, out_buf, out_len, force) != 0) return 2;
        double io_write_ms = (now_sec() - _w0) * 1000.0;
        double t_total_end = now_sec();
        print_stats("decompress", in_len, out_len, t1 - t0, t_total_end - t_total_start);
        print_phases("decode", io_read_ms, io_write_ms, cli_malloc_ms, &tm);
        if (repeat > 1) {
            fprintf(stderr, "  -- replaying decompress %d more times into same buffer --\n", repeat - 1);
            for (int r = 1; r < repeat; r++) {
                size_t rep_out_len = uncomp_size;
                double rt0 = now_sec();
                pivcohuf_decompress(in_buf, in_len, out_buf, &rep_out_len);
                double rt1 = now_sec();
                int ms = (int)((rt1 - rt0) * 1000.0 + 0.5);
                fprintf(stderr, "  iter %2d: comp:%dms  comp_bw in=%d MB/s out=%d MB/s\n",
                        r + 1, ms,
                        (int)((double)in_len      / 1.0e6 / (rt1 - rt0) + 0.5),
                        (int)((double)rep_out_len / 1.0e6 / (rt1 - rt0) + 0.5));
            }
        }
#ifdef PIVCO_PROF
        pivco_prof_dump("pivcohuf decompress", t1 - t0,
                         pivco_prof_probe_tick_freq(),
                         (uint64_t)((out_len + 8191) / 8192));
#endif
        free(out_buf);
    } else {
        usage(stderr); return 1;
    }
    free(in_buf);
    return 0;
}
