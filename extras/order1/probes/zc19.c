#include <zstd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
int main(int argc, char **argv) {
    struct stat st; stat(argv[1], &st);
    size_t n = st.st_size;
    char *b = malloc(n); FILE *f = fopen(argv[1], "rb");
    if (fread(b, 1, n, f) != n) return 1;
    size_t cap = ZSTD_compressBound(n); char *c = malloc(cap);
    size_t cl = ZSTD_compress(c, cap, b, n, atoi(argv[3]));
    if (ZSTD_isError(cl)) { fprintf(stderr, "%s\n", ZSTD_getErrorName(cl)); return 1; }
    FILE *o = fopen(argv[2], "wb"); fwrite(c, 1, cl, o); fclose(o);
    printf("%zu\n", cl);
    return 0;
}
