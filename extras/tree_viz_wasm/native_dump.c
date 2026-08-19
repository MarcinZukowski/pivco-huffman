/* Native twin of the tree_viz wasm module, for the fidelity A/B: reads
 * "name f0 f1 ... f255" lines on stdin, emits the same report format as
 * test_fidelity.js does for the wasm build.  Link with viz.c +
 * joint_lengths.c + huffman_table.c (see build.sh --test). */
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double *viz_freq_buf(void);
uint8_t *viz_len_buf(void);
int viz_plain_lengths(void);
int viz_joint_lengths(int effort, int fse_enabled);

const pivco_cfg_t pivco_cfg_default = {
    .tree_mode   = PIVCO_TREE_MODE_OPTIMIZED,
    .effort      = PIVCO_EFFORT_PLAIN,
    .fse_enabled = 1,
    .flat_layout = PIVCO_FLAT_VERTICAL,
};

__attribute__((noreturn))
void pivco_check_fail(const char *expr, const char *file, int line)
{
    fprintf(stderr, "PIVCO_CHECK failed: %s (%s:%d)\n", expr, file, line);
    abort();
}

static void print_lengths(const char *tag)
{
    const uint8_t *l = viz_len_buf();
    printf("%s ", tag);
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) printf("%x", l[i]);
    printf("\n");
}

int main(void)
{
    char line[16384];
    while (fgets(line, sizeof(line), stdin)) {
        char name[256];
        char *p = line;
        if (sscanf(p, "%255s", name) != 1) continue;
        p = strchr(p, ' ');
        double *freq = viz_freq_buf();
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++)
            freq[i] = strtod(p, &p);
        printf("== %s\n", name);
        if (viz_plain_lengths() != PIVCO_OK) { printf("plain ERR\n"); continue; }
        print_lengths("plain");
        for (int e = 1; e <= 3; e++)
            for (int f = 0; f <= 1; f++) {
                /* reload: joint mutates the shared length buffer */
                char tag[32];
                int rc = viz_joint_lengths(e, f);
                snprintf(tag, sizeof(tag), "joint_e%d_f%d_a%d", e, f, rc);
                print_lengths(tag);
            }
    }
    return 0;
}
