/* bench_mask_pair_degeneracy — quantify partition-mask byte degeneracy
 * on real-world byte distributions.
 *
 * Per real distribution, samples N blocks of PIVCO_BLOCK_SIZE symbols,
 * walks the tree-walk partition (skipping flat-subtree nodes — those
 * never call partition_8), and accumulates:
 *
 *   Singleton stats — informs single-mask early-exit:
 *     P(m == 0)    : all-left   (all 8 indices go left ; left_out gets 16B memcpy)
 *     P(m == 0xFF) : all-right  (all 8 indices go right; right_out gets 16B memcpy)
 *
 *   Pair stats (consecutive masks within one partition node, at the
 *   j+16 loop boundary) — informs the pair-degenerate fast path:
 *     P(m0,m1) ∈ {(0,0), (0xFF,0xFF)} : 32B memcpy to one side
 *     P(m0,m1) ∈ {(0,0xFF), (0xFF,0)} : two 16B vst, no TBL
 *
 *   Pair store-save histogram — informs the pair-merge variant:
 *     s = popcnt(m0) + popcnt(m1) ∈ [0,16]
 *       s ∈ {0, 16}  → 32B memcpy possible. SAVE 2 stores, no TBL.
 *       s == 8       → pair-merge fits both sides in 1 store each. SAVE 2 stores.
 *       s ∈ [1,7] ∪ [9,15] → pair-merge saves 1 store on the smaller side.
 *
 * Caveat: real-text bytes carry local correlations (bigrams, sequence
 * structure) that this i.i.d. sampling does not reproduce, so true
 * rates on a raw byte stream may differ.
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
extern void            bench_generate_symbols(int dist_idx,
                                               uint8_t *symbols,
                                               int n_symbols, uint64_t seed);

static uint8_t pop8_tab[256];
static void init_pop8(void) {
    for (int i = 0; i < 256; i++) {
        int c = 0;
        for (int b = 0; b < 8; b++) c += (i >> b) & 1;
        pop8_tab[i] = (uint8_t)c;
    }
}

typedef struct {
    uint64_t total_bytes;
    uint64_t m_zero;
    uint64_t m_ones;
    uint64_t total_pairs;
    uint64_t pair_zz;        /* (0, 0)        */
    uint64_t pair_oo;        /* (0xFF, 0xFF)  */
    uint64_t pair_zo_oz;     /* (0,0xFF) + (0xFF,0) */
    uint64_t s_hist[17];     /* popcount(m0) + popcount(m1) */
} stats_t;

/* Walk one partition node like encode_node_neon — build mask byte for
 * each 8-group and partition into left/right children — but accumulate
 * mask stats instead of writing a bitstream. */
static void walk_partition(const pivco_table_t *t,
                            int16_t node_id,
                            uint16_t *indices, int n,
                            int depth,
                            const uint16_t *codes,
                            const uint8_t *lens,
                            uint16_t *tmp,
                            stats_t *st)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &t->tree[node_id];
    if (node->symbol >= 0) return;
    if (t->flat_depth[node_id] >= 2) return;  /* skipped by partition_8 path */

    int n_left = 0, n_right = 0;
    int j = 0;
    uint8_t pair_buf[2];
    int pair_phase = 0;

    /* Buffer 8 indices first to avoid in-place overwrite hazards
     * (n_left can reach j during partition). */
    while (j + 8 <= n) {
        uint16_t buf[8];
        for (int k = 0; k < 8; k++) buf[k] = indices[j + k];

        uint8_t mask = 0;
        for (int k = 0; k < 8; k++) {
            int bit = (codes[buf[k]] >> (lens[buf[k]] - 1 - depth)) & 1;
            mask |= (uint8_t)(bit << k);
        }

        st->total_bytes++;
        if (mask == 0)         st->m_zero++;
        else if (mask == 0xFF) st->m_ones++;

        pair_buf[pair_phase] = mask;
        pair_phase ^= 1;
        if (pair_phase == 0) {
            uint8_t m0 = pair_buf[0], m1 = pair_buf[1];
            st->total_pairs++;
            if      (m0 == 0    && m1 == 0)    st->pair_zz++;
            else if (m0 == 0xFF && m1 == 0xFF) st->pair_oo++;
            else if ((m0 == 0    && m1 == 0xFF) ||
                     (m0 == 0xFF && m1 == 0))   st->pair_zo_oz++;
            int s = pop8_tab[m0] + pop8_tab[m1];
            st->s_hist[s]++;
        }

        for (int k = 0; k < 8; k++) {
            int bit = (mask >> k) & 1;
            if (bit) tmp[n_right++] = buf[k];
            else     indices[n_left++] = buf[k];
        }
        j += 8;
    }
    /* Scalar tail — no pair contribution (j+16 path is over). */
    for (; j < n; j++) {
        int idx = indices[j];
        int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
        if (bit) tmp[n_right++] = idx;
        else     indices[n_left++] = idx;
    }

    walk_partition(t, node->left,  indices, n_left,  depth + 1,
                   codes, lens, tmp + n_right, st);
    walk_partition(t, node->right, tmp,     n_right, depth + 1,
                   codes, lens, tmp + n_right, st);
}

static void run_dist(int idx, int n_blocks)
{
    const char *name = bench_dist_name(idx);
    const uint64_t *freq = bench_dist_freq(idx);

    pivco_table_t t;
    if (pivco_build_table(bench_cfg(), freq, &t) != PIVCO_OK) {
        printf("%-15s  build_table failed\n", name);
        return;
    }

    /* Set up codes/lens lookup the encoder uses. */
    static uint16_t codes[PIVCO_MAX_SYMBOLS];
    static uint8_t  lens[PIVCO_MAX_SYMBOLS];
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        codes[s] = t.code[s];
        lens[s]  = t.code_len[s];
    }

    static uint8_t  symbols[PIVCO_BLOCK_SIZE];
    static uint16_t indices[PIVCO_BLOCK_SIZE];
    static uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    static uint16_t blk_codes[PIVCO_BLOCK_SIZE];
    static uint8_t  blk_lens[PIVCO_BLOCK_SIZE];

    stats_t st = {0};

    for (int b = 0; b < n_blocks; b++) {
        bench_generate_symbols(idx, symbols, PIVCO_BLOCK_SIZE,
                               0xC0FFEEULL + b * 0x9E3779B97F4A7C15ULL);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            blk_codes[i] = codes[symbols[i]];
            blk_lens[i]  = lens[symbols[i]];
            indices[i]   = (uint16_t)i;
        }
        walk_partition(&t, t.tree_root, indices, PIVCO_BLOCK_SIZE, 0,
                       blk_codes, blk_lens, tmp, &st);
    }

    if (st.total_bytes == 0) {
        printf("%-15s  no partition_8 calls (fully flat-subtree)\n", name);
        return;
    }

    double tb = (double)st.total_bytes;
    double tp = (double)st.total_pairs;

    /* Singleton degeneracy. */
    double pZ  = 100.0 * st.m_zero / tb;
    double pO  = 100.0 * st.m_ones / tb;
    double pD  = pZ + pO;

    /* Pair fully-degenerate. */
    double ppZZ = (tp > 0) ? 100.0 * st.pair_zz / tp : 0;
    double ppOO = (tp > 0) ? 100.0 * st.pair_oo / tp : 0;
    double ppZO = (tp > 0) ? 100.0 * st.pair_zo_oz / tp : 0;
    double ppDeg = ppZZ + ppOO + ppZO;

    /* Idea-1 pair-merge store-save buckets. */
    uint64_t save2_zerofull = st.s_hist[0] + st.s_hist[16];
    uint64_t save2_balanced = st.s_hist[8];
    uint64_t save2 = save2_zerofull + save2_balanced;
    uint64_t save1 = st.total_pairs - save2;
    double pSave2zf  = (tp > 0) ? 100.0 * save2_zerofull / tp : 0;
    double pSave2bal = (tp > 0) ? 100.0 * save2_balanced / tp : 0;
    double pSave2    = (tp > 0) ? 100.0 * save2 / tp : 0;
    double pSave1    = (tp > 0) ? 100.0 * save1 / tp : 0;

    printf("%-13s | %9llu | %5.1f %5.1f %5.1f | %5.1f %5.1f %5.1f %5.1f | %5.1f %5.1f %5.1f | %5.1f\n",
           name,
           (unsigned long long)st.total_bytes,
           pZ, pO, pD,
           ppZZ, ppOO, ppZO, ppDeg,
           pSave2zf, pSave2bal, pSave2, pSave1);
}

int main(int argc, char **argv)
{
    int n_blocks = (argc > 1) ? atoi(argv[1]) : 64;

    bench_init();
    init_pop8();

    /* Real-world byte distributions only.  english (idx 8) uses real
     * English letter frequencies; the rest are file-derived. */
    static const char *real_names[] = {
        "english",
        "html_wiki", "prose_pride", "image_jpeg", "json_api",
        "source_c",  "log_apache",  "dna_fasta",  "csv_numeric",
        "gzip_random", "chinese_text",
    };
    int n_real = sizeof(real_names) / sizeof(real_names[0]);

    printf("Mask-byte degeneracy on real distributions (i.i.d. samples,\n");
    printf("%d blocks × %d symbols).  Tail-mask scalar path excluded.\n\n",
           n_blocks, PIVCO_BLOCK_SIZE);
    printf("                               singleton (%%)        pair fully-degenerate (%%)         pair store-save (%%)\n");
    printf("distribution  | total_bytes |  m=0   m=FF   any  | (0,0) (FF,FF) split   any | s∈{0,16} s=8  save2 | save1\n");
    printf("--------------+-------------+--------------------+--------------------------+--------------------+------\n");

    int n_dist = bench_num_distributions();
    for (int r = 0; r < n_real; r++) {
        int idx = -1;
        for (int i = 0; i < n_dist; i++) {
            if (strcmp(bench_dist_name(i), real_names[r]) == 0) { idx = i; break; }
        }
        if (idx < 0) {
            printf("%-13s | (not found)\n", real_names[r]);
            continue;
        }
        run_dist(idx, n_blocks);
    }

    printf("\nLegend:\n");
    printf("  singleton:     %% of mask bytes that are 0x00 / 0xFF / either\n");
    printf("  pair fully-degenerate: %% of consecutive (m0,m1) pairs that fall in\n");
    printf("                         (0,0), (0xFF,0xFF), or split (one all-0, one all-1)\n");
    printf("  pair store-save: idea-1 pair-merge buckets, where s = popcnt(m0)+popcnt(m1)\n");
    printf("                   s∈{0,16}: 32B memcpy possible — save 2 stores, no TBL\n");
    printf("                   s=8:      pair-merge fits both sides in one 16B store each — save 2\n");
    printf("                   save1:    everything else — pair-merge saves 1 store on the smaller side\n");
    return 0;
}
