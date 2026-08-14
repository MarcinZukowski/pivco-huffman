/* bench_node_bitmap_stats — per-node routing-bitmap structure analyzer.
 *
 * For each distribution: build the table, block-encode-walk the tree exactly
 * like the encoder (rank partition per internal node), and for every non-flat
 * internal node's bitmap accumulate, at 16/32/64-bit chunk granularity, how
 * many chunks are all-0 / all-1 — i.e. how often a skew-specialized merge easy
 * path (constant fill / straight copy) would fire on REAL data — plus the
 * u64 zero-delta gate count (v[w] == v[w-1]; see IDEAS "Skew-specialized
 * merge").  gate%% = share of u64 words living in nodes where zero-deltas
 * exceed 25%% of the node's words (nodes where the hybrid would take the
 * fused path); gate-easy%% = the all-0/all-1 u64 share within those nodes.
 *
 * Usage: ./pivco_node_bitmap_stats [--all]   (default: the main dist set)
 */
#include "pivco_huffman.h"
#include "bench_ctx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void bench_init(void);
extern int bench_num_distributions(void);
extern const char *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int bench_dist_is_main(int idx);
extern int bench_dist_size(int dist_idx, int min_n, int block_align);
extern void bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                   int n_symbols, uint64_t seed);

#define TOTAL   (4 * 1024 * 1024)
#define SEED    0x5EEDBA5Eull

typedef struct {
    unsigned long long c16, c16_0, c16_1;    /* u16 chunks: total / all-0 / all-1 */
    unsigned long long c32, c32_0, c32_1;
    unsigned long long c64, c64_0, c64_1;
    unsigned long long zd64;                 /* u64 zero-delta pairs (the gate count) */
    unsigned long long gate_w, gate_easy;    /* u64 words in gate-fired nodes / easy among them */
    unsigned long long nodes, elems;
} stats_t;

/* Build the node's bitmap words from ranks, accumulate stats, partition,
 * recurse.  Mirrors the encode walk; scalar throughout (analysis tool). */
static void walk(const pivco_table_t *t, int node,
                 uint8_t *ranks, int n, uint8_t *tmp, stats_t *s)
{
    pivco_node_type_t nt = (pivco_node_type_t)t->node_type[node];
    if (nt == PIVCO_NODE_LEAF) return;
    if (nt == PIVCO_NODE_INTERNAL_FLAT) return;          /* packed codes, no bitmap */
    if (n <= 0) return;

    uint8_t thr = t->split_rank[node];
    s->nodes++; s->elems += (unsigned long long)n;

    int nw = n >> 6;
    unsigned long long node_zd = 0, node_easy = 0;
    uint64_t prev = 0; int have_prev = 0;
    for (int w = 0; w < nw; w++) {
        uint64_t v = 0;
        for (int b = 0; b < 64; b++)
            v |= (uint64_t)(ranks[(w << 6) + b] > thr) << b;
        s->c64++;
        int easy64 = (v == 0) || (v == ~0ull);
        s->c64_0 += (v == 0); s->c64_1 += (v == ~0ull);
        node_easy += (unsigned long long)easy64;
        if (have_prev) node_zd += (v == prev);
        prev = v; have_prev = 1;
        for (int h = 0; h < 4; h++) {                    /* u16 granularity */
            uint16_t x = (uint16_t)(v >> (16 * h));
            s->c16++; s->c16_0 += (x == 0); s->c16_1 += (x == 0xFFFF);
        }
        for (int h = 0; h < 2; h++) {                    /* u32 granularity */
            uint32_t x = (uint32_t)(v >> (32 * h));
            s->c32++; s->c32_0 += (x == 0); s->c32_1 += (x == 0xFFFFFFFFu);
        }
    }
    s->zd64 += node_zd;
    if (nw > 0 && node_zd * 4 > (unsigned long long)nw) {   /* gate: zd > 25% of words */
        s->gate_w += (unsigned long long)nw;
        s->gate_easy += node_easy;
    }

    /* partition (scalar) and recurse into internal children */
    int nl = 0, nr = 0;
    for (int j = 0; j < n; j++) {
        uint8_t r = ranks[j];
        if (r > thr) tmp[nr++] = r; else ranks[nl++] = r;
    }
    walk(t, t->tree[node].left,  ranks, nl, tmp + nr, s);
    walk(t, t->tree[node].right, tmp,   nr, tmp + nr, s);
}

int main(int argc, char **argv)
{
    int run_all = (argc > 1 && !strcmp(argv[1], "--all"));
    bench_init();

    printf("%-14s %6s | %7s %6s %6s | %6s %6s | %6s %6s | %6s | %6s %8s\n",
           "dist", "nodes", "words64", "w64_0%", "w64_1%",
           "w32_0%", "w32_1%", "w16_0%", "w16_1%", "zd64%", "gate%", "gateEZ%");
    for (int d = 0; d < bench_num_distributions(); d++) {
        if (!run_all && !bench_dist_is_main(d)) continue;
        int n = bench_dist_size(d, TOTAL, PIVCO_BLOCK_SIZE);
        uint8_t *sym = malloc((size_t)n);
        bench_generate_symbols(d, sym, n, SEED);

        uint64_t freq[PIVCO_MAX_SYMBOLS];
        memset(freq, 0, sizeof(freq));
        for (int i = 0; i < n; i++) freq[sym[i]]++;
        pivco_table_t *table = malloc(sizeof(*table));
        if (pivco_build_table(bench_cfg(), freq, table) != 0) {
            printf("%-14s (build_table failed)\n", bench_dist_name(d));
            free(sym); free(table); continue;
        }

        /* ranks buffer + right-half recursion scratch, like the encoder */
        uint8_t *ranks = malloc((size_t)PIVCO_BLOCK_SIZE + 64);
        uint8_t *tmp = malloc((size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2));
        stats_t s; memset(&s, 0, sizeof(s));
        for (int off = 0; off + PIVCO_BLOCK_SIZE <= n; off += PIVCO_BLOCK_SIZE) {
            for (int i = 0; i < PIVCO_BLOCK_SIZE; i++)
                ranks[i] = table->sym_to_rank[sym[off + i]];
            walk(table, table->tree_root, ranks, PIVCO_BLOCK_SIZE, tmp, &s);
        }

#define PCT(a, b) ((b) ? 100.0 * (double)(a) / (double)(b) : 0.0)
        printf("%-14s %6llu | %7llu %6.2f %6.2f | %6.2f %6.2f | %6.2f %6.2f | %6.2f | %6.2f %8.2f\n",
               bench_dist_name(d), s.nodes,
               s.c64, PCT(s.c64_0, s.c64), PCT(s.c64_1, s.c64),
               PCT(s.c32_0, s.c32), PCT(s.c32_1, s.c32),
               PCT(s.c16_0, s.c16), PCT(s.c16_1, s.c16),
               PCT(s.zd64, s.c64),
               PCT(s.gate_w, s.c64), PCT(s.gate_easy, s.gate_w));
#undef PCT
        free(sym); free(table); free(ranks); free(tmp);
    }
    return 0;
}
