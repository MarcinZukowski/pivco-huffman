/* bench_zstd_breakdown: measure where zstd spends its time on encode + decode.
 *
 * Patched libzstd accumulators:
 *   ENCODE:
 *     g_zstd_prof_enc_lz         LZ match-find (ZSTD_buildSeqStore)
 *     g_zstd_prof_enc_huff       Huffman literals encode (ZSTD_compressLiterals)
 *     g_zstd_prof_enc_fse_build  FSE stat tables for LL/OF/ML
 *     g_zstd_prof_enc_fse_emit   FSE sequence emission (ZSTD_encodeSequences)
 *   DECODE:
 *     g_zstd_prof_huff           HUF_decompress*
 *     g_zstd_prof_fse_build      FSE LL/OF/ML DTable construction
 *     g_zstd_prof_seq_loop       sequence decode loop (FSE state + LZ exec)
 *
 * Decode LZ exec split: skip-exec pass (g_zstd_prof_skip_exec=1)
 * short-circuits ZSTD_execSequence; lz_exec = seq_loop_A - seq_loop_B.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <mach/mach_time.h>

#include "../../ext/zstd/lib/zstd.h"
#include "../../ext/zstd/lib/decompress/zstd_prof.h"

#define DEC_REPEATS 500
#define ENC_REPEATS 50

static double mt_seconds_per_tick(void) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return (double)tb.numer / (double)tb.denom * 1e-9;
}

static void* slurp(const char* path, size_t* out_len) {
    struct stat st;
    if (stat(path, &st) != 0) { fprintf(stderr, "stat %s: failed\n", path); exit(1); }
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: failed\n", path); exit(1); }
    void* buf = malloc((size_t)st.st_size);
    if (fread(buf, 1, (size_t)st.st_size, f) != (size_t)st.st_size) {
        fprintf(stderr, "read %s: short\n", path); exit(1);
    }
    fclose(f);
    *out_len = (size_t)st.st_size;
    return buf;
}

typedef struct {
    uint64_t total_ticks;
    uint64_t huff_ticks;
    uint64_t fse_build_ticks;
    uint64_t seq_loop_ticks;
    size_t   decompressed_bytes;
} dec_pass_t;

typedef struct {
    uint64_t total_ticks;
    uint64_t lz_ticks;
    uint64_t huff_ticks;
    uint64_t fse_build_ticks;
    uint64_t fse_emit_ticks;
    size_t   csize;
} enc_pass_t;

static dec_pass_t run_decode_pass(const void* compressed, size_t csize,
                                  void* dst, size_t dst_cap, int skip_exec) {
    dec_pass_t p = {0};
    g_zstd_prof_skip_exec = skip_exec;
    for (int r = 0; r < 3; r++) {
        size_t s = ZSTD_decompress(dst, dst_cap, compressed, csize);
        (void)s;
    }
    zstd_prof_reset();
    uint64_t t0 = mach_absolute_time();
    for (int r = 0; r < DEC_REPEATS; r++) {
        size_t s = ZSTD_decompress(dst, dst_cap, compressed, csize);
        if (ZSTD_isError(s) && !skip_exec) {
            fprintf(stderr, "decompress err: %s\n", ZSTD_getErrorName(s));
            exit(1);
        }
        p.decompressed_bytes = s;
    }
    uint64_t t1 = mach_absolute_time();
    p.total_ticks = t1 - t0;
    p.huff_ticks = g_zstd_prof_huff;
    p.fse_build_ticks = g_zstd_prof_fse_build;
    p.seq_loop_ticks = g_zstd_prof_seq_loop;
    return p;
}

static enc_pass_t run_encode_pass(const void* raw, size_t raw_len,
                                  void* compressed, size_t cbound, int level) {
    enc_pass_t p = {0};
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
    /* warmup — let internal hash tables populate */
    for (int r = 0; r < 3; r++) {
        size_t s = ZSTD_compress2(cctx, compressed, cbound, raw, raw_len);
        (void)s;
    }
    zstd_prof_reset();
    uint64_t t0 = mach_absolute_time();
    for (int r = 0; r < ENC_REPEATS; r++) {
        size_t s = ZSTD_compress2(cctx, compressed, cbound, raw, raw_len);
        if (ZSTD_isError(s)) {
            fprintf(stderr, "compress err: %s\n", ZSTD_getErrorName(s));
            exit(1);
        }
        p.csize = s;
    }
    uint64_t t1 = mach_absolute_time();
    p.total_ticks = t1 - t0;
    p.lz_ticks = g_zstd_prof_enc_lz;
    p.huff_ticks = g_zstd_prof_enc_huff;
    p.fse_build_ticks = g_zstd_prof_enc_fse_build;
    p.fse_emit_ticks = g_zstd_prof_enc_fse_emit;
    ZSTD_freeCCtx(cctx);
    return p;
}

int main(int argc, char** argv) {
    int level = 3;
    int argstart = 1;
    if (argc > 2 && argv[1][0] == '-' && argv[1][1] == 'L') {
        level = atoi(argv[1]+2);
        argstart = 2;
    }
    if (argc - argstart < 1) {
        fprintf(stderr, "usage: %s [-L<level>] <file> [<file>...]\n", argv[0]);
        return 1;
    }
    double tick = mt_seconds_per_tick();

    printf("== ENCODE (level %d) ==\n", level);
    printf("%-16s %8s %8s %5s | %7s %7s %7s %7s %7s | %8s\n",
           "dataset", "raw_KB", "csize_KB", "ratio",
           "lz%", "huff%", "fseB%", "fseE%", "other%",
           "MB/s");

    /* Pre-encode all files so we have compressed buffers for decode pass. */
    int nfiles = argc - argstart;
    void** raws = calloc(nfiles, sizeof(void*));
    size_t* raw_lens = calloc(nfiles, sizeof(size_t));
    void** comps = calloc(nfiles, sizeof(void*));
    size_t* csizes = calloc(nfiles, sizeof(size_t));

    for (int i = 0; i < nfiles; i++) {
        const char* path = argv[argstart + i];
        raws[i] = slurp(path, &raw_lens[i]);
        size_t cbound = ZSTD_compressBound(raw_lens[i]);
        comps[i] = malloc(cbound);

        enc_pass_t E = run_encode_pass(raws[i], raw_lens[i], comps[i], cbound, level);
        csizes[i] = E.csize;

        double sec = E.total_ticks * tick;
        double lz   = E.lz_ticks * tick;
        double hu   = E.huff_ticks * tick;
        double fb   = E.fse_build_ticks * tick;
        double fe   = E.fse_emit_ticks * tick;
        double oth  = sec - lz - hu - fb - fe;
        if (oth < 0) oth = 0;
        double total_bytes = (double)raw_lens[i] * ENC_REPEATS;
        double mbps = total_bytes / (1024.0*1024.0) / sec;

        const char* base = strrchr(path, '/');
        base = base ? base + 1 : path;
        printf("%-16s %8.0f %8.0f %5.2f | %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% | %8.0f\n",
               base, raw_lens[i]/1024.0, E.csize/1024.0, (double)raw_lens[i]/E.csize,
               100*lz/sec, 100*hu/sec, 100*fb/sec, 100*fe/sec, 100*oth/sec, mbps);
        fflush(stdout);
    }

    printf("\n== DECODE ==\n");
    printf("%-16s %5s | %7s %7s %7s %7s %7s | %8s\n",
           "dataset", "ratio",
           "huff%", "fseB%", "fseD%", "lzX%", "other%",
           "MB/s");

    for (int i = 0; i < nfiles; i++) {
        const char* path = argv[argstart + i];
        size_t dst_cap = raw_lens[i] + 64;
        void* dstA = malloc(dst_cap);
        void* dstB = malloc(dst_cap);
        for (int r = 0; r < 50; r++) {
            ZSTD_decompress(dstA, dst_cap, comps[i], csizes[i]);
            ZSTD_decompress(dstB, dst_cap, comps[i], csizes[i]);
        }
        dec_pass_t A = run_decode_pass(comps[i], csizes[i], dstA, dst_cap, 0);
        dec_pass_t B = run_decode_pass(comps[i], csizes[i], dstB, dst_cap, 1);
        dec_pass_t A2 = run_decode_pass(comps[i], csizes[i], dstA, dst_cap, 0);
        if (A2.total_ticks < A.total_ticks) A = A2;

        double sec = A.total_ticks * tick;
        double hu  = A.huff_ticks * tick;
        double fb  = A.fse_build_ticks * tick;
        double seqA = A.seq_loop_ticks * tick;
        double seqB = B.seq_loop_ticks * tick;
        double lzx = seqA - seqB; if (lzx < 0) lzx = 0;
        double fsd = seqB;
        double oth = sec - hu - fb - lzx - fsd; if (oth < 0) oth = 0;
        double total_bytes = (double)raw_lens[i] * DEC_REPEATS;
        double mbps = total_bytes / (1024.0*1024.0) / sec;

        const char* base = strrchr(path, '/');
        base = base ? base + 1 : path;
        printf("%-16s %5.2f | %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% | %8.0f\n",
               base, (double)raw_lens[i]/csizes[i],
               100*hu/sec, 100*fb/sec, 100*fsd/sec, 100*lzx/sec, 100*oth/sec, mbps);
        fflush(stdout);
        free(dstA); free(dstB);
    }
    for (int i = 0; i < nfiles; i++) { free(raws[i]); free(comps[i]); }
    free(raws); free(raw_lens); free(comps); free(csizes);
    return 0;
}
