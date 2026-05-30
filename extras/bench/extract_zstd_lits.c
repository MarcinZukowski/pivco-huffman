/* extract_zstd_lits: dump all 4 streams zstd hands to its entropy coders
 *   - literals (HUF)            -> <base>.lits
 *   - LL codes (FSE LL)         -> <base>.ll
 *   - OF codes (FSE OF)         -> <base>.of
 *   - ML codes (FSE ML)         -> <base>.ml
 * Plus per-stream Shannon entropy.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>

#include "../../ext/zstd/lib/zstd.h"
#include "../../ext/zstd/lib/decompress/zstd_prof.h"

static void* slurp(const char* path, size_t* out_len) {
    struct stat st;
    if (stat(path, &st) != 0) { *out_len = 0; return NULL; }
    FILE* f = fopen(path, "rb");
    if (!f) { *out_len = 0; return NULL; }
    void* buf = malloc((size_t)st.st_size);
    if (fread(buf, 1, (size_t)st.st_size, f) != (size_t)st.st_size) {
        fclose(f); free(buf); *out_len = 0; return NULL;
    }
    fclose(f);
    *out_len = (size_t)st.st_size;
    return buf;
}

static double shannon_H(const uint8_t* buf, size_t n) {
    if (n == 0) return 0.0;
    uint64_t cnt[256] = {0};
    for (size_t i = 0; i < n; i++) cnt[buf[i]]++;
    double H = 0.0; double total = (double)n;
    for (int s = 0; s < 256; s++) {
        if (cnt[s] == 0) continue;
        double p = cnt[s] / total;
        H -= p * log2(p);
    }
    return H;
}

static int alphabet_size(const uint8_t* buf, size_t n) {
    if (n == 0) return 0;
    int seen[256] = {0};
    for (size_t i = 0; i < n; i++) seen[buf[i]] = 1;
    int c = 0;
    for (int s = 0; s < 256; s++) c += seen[s];
    return c;
}

int main(int argc, char** argv) {
    int level = 3;
    int argi = 1;
    if (argc > 2 && argv[1][0] == '-' && argv[1][1] == 'L') {
        level = atoi(argv[1]+2);
        argi = 2;
    }
    if (argc - argi < 2) {
        fprintf(stderr, "usage: %s [-L<level>] <out_dir> <file>...\n", argv[0]);
        return 1;
    }
    const char* out_dir = argv[argi++];

    printf("zstd-L%d stream extraction\n", level);
    printf("%-16s %8s %4s | %8s %5s %5s | %8s %5s %5s | %8s %5s %5s | %8s %5s %5s\n",
           "dataset", "raw_KB", "H_r",
           "lit_KB","|A|","H",
           "LL_KB", "|A|","H",
           "OF_KB", "|A|","H",
           "ML_KB", "|A|","H");

    for (; argi < argc; argi++) {
        const char* path = argv[argi];
        size_t raw_len;
        void* raw = slurp(path, &raw_len);
        if (!raw) { fprintf(stderr, "read %s failed\n", path); continue; }

        const char* base = strrchr(path, '/');
        base = base ? base + 1 : path;

        char p_lit[1024], p_ll[1024], p_of[1024], p_ml[1024];
        snprintf(p_lit, sizeof(p_lit), "%s/%s.lits", out_dir, base);
        snprintf(p_ll,  sizeof(p_ll),  "%s/%s.ll",   out_dir, base);
        snprintf(p_of,  sizeof(p_of),  "%s/%s.of",   out_dir, base);
        snprintf(p_ml,  sizeof(p_ml),  "%s/%s.ml",   out_dir, base);

        FILE* f_lit = fopen(p_lit, "wb");
        FILE* f_ll  = fopen(p_ll,  "wb");
        FILE* f_of  = fopen(p_of,  "wb");
        FILE* f_ml  = fopen(p_ml,  "wb");
        if (!f_lit || !f_ll || !f_of || !f_ml) {
            fprintf(stderr, "open output failed for %s\n", base); return 1;
        }
        g_zstd_prof_lit_dump_fp = f_lit;
        g_zstd_prof_ll_dump_fp  = f_ll;
        g_zstd_prof_of_dump_fp  = f_of;
        g_zstd_prof_ml_dump_fp  = f_ml;

        size_t cbound = ZSTD_compressBound(raw_len);
        void* compressed = malloc(cbound);
        size_t csize = ZSTD_compress(compressed, cbound, raw, raw_len, level);
        if (ZSTD_isError(csize)) {
            fprintf(stderr, "compress err: %s\n", ZSTD_getErrorName(csize)); return 1;
        }
        g_zstd_prof_lit_dump_fp = NULL;
        g_zstd_prof_ll_dump_fp = NULL;
        g_zstd_prof_of_dump_fp = NULL;
        g_zstd_prof_ml_dump_fp = NULL;
        fclose(f_lit); fclose(f_ll); fclose(f_of); fclose(f_ml);

        size_t n_lit, n_ll, n_of, n_ml;
        uint8_t* d_lit = slurp(p_lit, &n_lit);
        uint8_t* d_ll  = slurp(p_ll,  &n_ll);
        uint8_t* d_of  = slurp(p_of,  &n_of);
        uint8_t* d_ml  = slurp(p_ml,  &n_ml);

        printf("%-16s %8.0f %4.2f | %8.0f %5d %5.3f | %8.0f %5d %5.3f | %8.0f %5d %5.3f | %8.0f %5d %5.3f\n",
               base, raw_len/1024.0, shannon_H((uint8_t*)raw, raw_len),
               n_lit/1024.0, alphabet_size(d_lit, n_lit), shannon_H(d_lit, n_lit),
               n_ll /1024.0, alphabet_size(d_ll,  n_ll),  shannon_H(d_ll, n_ll),
               n_of /1024.0, alphabet_size(d_of,  n_of),  shannon_H(d_of, n_of),
               n_ml /1024.0, alphabet_size(d_ml,  n_ml),  shannon_H(d_ml, n_ml));

        free(raw); free(d_lit); free(d_ll); free(d_of); free(d_ml); free(compressed);
    }
    return 0;
}
