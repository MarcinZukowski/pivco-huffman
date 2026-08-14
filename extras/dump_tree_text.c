/* dump_tree_text -- emit the production decode tree as text, in the same
 * format as figures/tree_viz.html's "download tree text" button, so the two
 * can be diffed to confirm the figure reflects the real C-built tree.
 *
 * Uses bench_dist_freq() -- the exact frequency arrays that
 * extras/dump_distributions bakes into figures/tree_viz_data.js -- so the C
 * builder and the JS figure get identical input; any diff is a real
 * build-logic drift, not a sampling artifact.
 *
 * Usage:  ./pivco_dump_tree_text <dist-name>   (e.g. image_jpeg)
 *
 * Two sections, both deterministic from the code lengths:
 *   ## codes  -- per-symbol (symbol length code), sorted by symbol
 *   ## tree   -- pre-order walk (0=left,1=right, "." = root): each LEAF and
 *                each maximal FLAT subtree (D>=2), members in local-code order.
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

static void print_code(uint16_t code, int len)
{
    for (int b = len - 1; b >= 0; b--)
        putchar(((code >> b) & 1) ? '1' : '0');
}

static void emit_path(const char *buf, int depth)
{
    if (depth == 0) putchar('.');
    else            fwrite(buf, 1, (size_t)depth, stdout);
}

static void walk(const pivco_table_t *T, int16_t node,
                 char *buf, int depth)
{
    if (T->flat_depth[node] >= 2) {
        int D = T->flat_depth[node];
        uint16_t off = T->flat_offset[node];
        emit_path(buf, depth);
        printf(" FLAT D=%d syms=", D);
        for (int i = 0; i < (1 << D); i++) {
            if (i) putchar(',');
            printf("%d", T->flat_code_to_sym[off + i]);
        }
        putchar('\n');
        return;                       /* maximal -- don't descend */
    }
    if (T->tree[node].symbol >= 0) {
        emit_path(buf, depth);
        printf(" LEAF sym=%d len=%d\n", T->tree[node].symbol, depth);
        return;
    }
    buf[depth] = '0'; walk(T, T->tree[node].left,  buf, depth + 1);
    buf[depth] = '1'; walk(T, T->tree[node].right, buf, depth + 1);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <dist-name>\n", argv[0]);
        return 2;
    }
    bench_init();
    int d = -1;
    for (int k = 0; k < bench_num_distributions(); k++)
        if (strcmp(bench_dist_name(k), argv[1]) == 0) { d = k; break; }
    if (d < 0) { fprintf(stderr, "unknown dist '%s'\n", argv[1]); return 2; }

    const uint64_t *f = bench_dist_freq(d);
    pivco_table_t *T = malloc(sizeof *T);
    if (!T || pivco_build_table(bench_cfg(), f, T) != PIVCO_OK) {
        fprintf(stderr, "build_table failed\n");
        return 1;
    }

    int nsyms = 0;
    for (int s = 0; s < 256; s++) if (T->code_len[s]) nsyms++;

    printf("# tree_viz tree layout dump\n");
    printf("# dist=%s phOpt=1 maxSkew=0 keysSort=symbol maxLen=%d nsyms=%d\n",
           argv[1], T->max_len, nsyms);
    printf("## codes  (symbol length code)\n");
    for (int s = 0; s < 256; s++) {
        if (!T->code_len[s]) continue;
        printf("%d %d ", s, T->code_len[s]);
        print_code(T->code[s], T->code_len[s]);
        putchar('\n');
    }
    printf("## tree  (path kind detail; path 0=left 1=right, \".\" = root)\n");
    char buf[64];
    walk(T, T->tree_root, buf, 0);

    free(T);
    return 0;
}
