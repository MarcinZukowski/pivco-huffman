/* bench_entropy_breakdown — per-dataset entropy + FSE-per-node analysis
 *
 * For each test distribution: sample BLK bytes via dist_sample, build the
 * Huffman tree, walk every internal node and compute:
 *   - the bitmap entropy at that node (H of the left/right split)
 *   - the FSE-on-bytes compressed size of the (synthesized IID) bitmap
 *   - the FSE-on-bits compressed size (expand each bit to 0x00/0x01 byte
 *     then FSE)
 *   - bits saved by FSE at that node vs the 1-bit-per-symbol baseline
 *
 * Aggregate across all internal nodes and emit one row per dataset with:
 *
 *   N      H        huf      fse_byte fse_bit  top1_b  top1_d
 *
 *   N         source byte count (BLK)
 *   H         byte entropy = Σ p_i · -log2(p_i)  (bits/byte)
 *   huf       weighted Huffman avg code length = Σ p_i · len_i (bits/byte)
 *   fse_byte  Σ over internal nodes of FSE-on-bytes(node bitmap) payload
 *             length (in bits, excludes FSE counts header), divided by N
 *             — bits per source byte using ph-style FSE-on-bitmap-bytes
 *   fse_bit   Σ over internal nodes of FSE-on-bits(expand bitmap to 1
 *             byte per bit, then FSE) payload length, divided by N
 *             — bits per source byte using FSE-on-each-bitmap-bit
 *   top1_b    max single-node bits saved per source byte
 *             = max over nodes of (n_node · (1 − H_node)) / N
 *   top1_d    depth of that top-1 node (root = 0)
 *
 * Theories being tested (from user spec):
 *   - FSE-on-bytes lands very close to the Shannon entropy H.
 *   - A single node accounts for most of the saving.
 *   - That node is usually the root.
 *
 * Synthetic-IID-bitmap caveat: we synthesize each node's bitmap as IID
 * bits with the right p_left, rather than encoding a real source byte
 * sequence.  For dist_sample (which itself draws IID bytes) the two
 * approaches are statistically equivalent for FSE; we could not exploit
 * bit-to-bit correlation even if it were there.
 */

#define FSE_STATIC_LINKING_ONLY
#include "pivco_huffman.h"
#include "bench_ctx.h"
#include "fse.h"
#include "hist.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);
extern int             bench_dist_is_main(int idx);
extern void            bench_generate_symbols(int idx, uint8_t *symbols,
                                                int n, uint64_t seed);

#define BLK 8192

static double h_binary(double p)
{
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p);
}

/* Returns FSE-compressed payload length in BYTES (excludes the FSE
 * counts header).  Falls back to the source byte count for inputs
 * FSE refuses to compress (uniform, RLE, etc.). */
static size_t fse_payload_bytes(const uint8_t *src, size_t n,
                                  uint8_t *scratch, size_t scratch_cap)
{
    if (n == 0) return 0;
    if (n == 1) return 1;
    size_t total = FSE_compress(scratch, scratch_cap, src, n);
    if (FSE_isError(total)) return n;
    if (total == 0) return n;     /* incompressible, "store raw" */
    if (total == 1) return 1;     /* single-symbol RLE */
    /* Strip the counts header so we measure payload only. */
    short normCount[256];
    unsigned maxSym = 255, tableLog = 0;
    size_t hdr = FSE_readNCount(normCount, &maxSym, &tableLog,
                                  scratch, total);
    if (FSE_isError(hdr)) return total;
    return total - hdr;
}

typedef struct {
    double total_fse_byte_bits;
    double total_fse_bit_bits;
    double total_entropy_bits;   /* Σ n_node · H_node — should equal N·H */
    double top1_saved_bits;
    int    top1_depth;
    int    n_internal;
    uint64_t xs_state;
    /* Flat-subtree accounting (D ≥ 2 maximal subtrees with all
     * leaves at the same depth from the subtree root). */
    double flat_raw_bits;        /* Σ n_subtree · D — what ph emits today */
    double flat_entropy_bits;    /* Σ n_subtree · H_conditional — floor */
    int    n_flat_subtrees;
} stats_t;

static uint64_t xs_next(stats_t *s)
{
    uint64_t v = s->xs_state;
    v ^= v << 13; v ^= v >> 7; v ^= v << 17;
    return (s->xs_state = v);
}

/* Returns the depth (from this node) at which all leaves sit if the
 * subtree rooted here is flat, or -1 otherwise.  A leaf returns 0. */
static int subtree_flat_depth(const pivco_table_t *t, int16_t node)
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) return 0;
    int ld = subtree_flat_depth(t, n->left);
    if (ld < 0) return -1;
    int rd = subtree_flat_depth(t, n->right);
    if (rd < 0 || rd != ld) return -1;
    return ld + 1;
}

/* Collect leaf frequencies (in source-order) into `out`, returning
 * count of leaves filled. */
static int collect_leaf_freqs(const pivco_table_t *t, int16_t node,
                                const uint64_t *freq, uint64_t *out, int cap)
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) {
        if (cap <= 0) return 0;
        out[0] = freq[n->symbol];
        return 1;
    }
    int got = collect_leaf_freqs(t, n->left, freq, out, cap);
    got += collect_leaf_freqs(t, n->right, freq, out + got, cap - got);
    return got;
}

/* Walk the tree returning total subtree count.  Accumulates FSE
 * measurements into `s`.  Detects maximal flat subtrees (D ≥ 2)
 * and accounts them separately into s->flat_*. */
static uint64_t walk(const pivco_table_t *t, int16_t node, int depth,
                     const uint64_t *freq, stats_t *s,
                     uint8_t *bitmap_buf, uint8_t *expand_buf,
                     uint8_t *scratch, size_t scratch_cap)
{
    const pivco_tree_node_t *n = &t->tree[node];
    if (n->symbol >= 0) return freq[n->symbol];

    /* Maximal-flat check.  `subtree_flat_depth` returns D if this
     * subtree is flat; we treat D ≥ 2 as the ph fast-path trigger.
     * Caller has already verified the parent wasn't flat-enough
     * (otherwise we wouldn't have descended here). */
    int flat_d = subtree_flat_depth(t, node);
    if (flat_d >= 2) {
        /* Collect leaf freqs to compute the local entropy. */
        uint64_t leaf_freqs[1 << 16];  /* depth up to 16 = 65536 leaves */
        int n_leaves = collect_leaf_freqs(t, node, freq, leaf_freqs,
                                            (int)(sizeof(leaf_freqs) / sizeof(leaf_freqs[0])));
        uint64_t sum = 0;
        for (int i = 0; i < n_leaves; i++) sum += leaf_freqs[i];
        double H_cond = 0.0;
        if (sum > 0) {
            for (int i = 0; i < n_leaves; i++) {
                if (leaf_freqs[i] == 0) continue;
                double p = (double)leaf_freqs[i] / (double)sum;
                H_cond -= p * log2(p);
            }
        }
        s->n_flat_subtrees++;
        s->flat_raw_bits     += (double)sum * (double)flat_d;
        s->flat_entropy_bits += (double)sum * H_cond;
        /* Return without recursing — flat subtrees are accounted as
         * a single unit; we don't double-count the internal nodes
         * inside them in the regular per-node sums. */
        return sum;
    }

    uint64_t left  = walk(t, n->left,  depth + 1, freq, s,
                          bitmap_buf, expand_buf, scratch, scratch_cap);
    uint64_t right = walk(t, n->right, depth + 1, freq, s,
                          bitmap_buf, expand_buf, scratch, scratch_cap);
    uint64_t total = left + right;
    if (total == 0) return 0;

    s->n_internal++;
    double p_left = (double)left / (double)total;
    double H_node = h_binary(p_left);
    s->total_entropy_bits += (double)total * H_node;

    double saved = (double)total * (1.0 - H_node);
    if (saved > s->top1_saved_bits) {
        s->top1_saved_bits = saved;
        s->top1_depth      = depth;
    }

    /* Synthesize IID bitmap with the right p_left, FSE-compress on
     * bytes and on expanded bits. */
    size_t bitmap_bits  = (size_t)total;
    size_t bitmap_bytes = (bitmap_bits + 7) / 8;
    memset(bitmap_buf, 0, bitmap_bytes);
    uint64_t threshold = (uint64_t)(p_left * (double)UINT64_MAX);
    for (size_t i = 0; i < bitmap_bits; i++) {
        int bit = (xs_next(s) < threshold) ? 1 : 0;
        if (bit) bitmap_buf[i >> 3] |= (uint8_t)(1u << (i & 7));
    }
    size_t fse_byte = fse_payload_bytes(bitmap_buf, bitmap_bytes,
                                          scratch, scratch_cap);
    s->total_fse_byte_bits += (double)fse_byte * 8.0;

    for (size_t i = 0; i < bitmap_bits; i++)
        expand_buf[i] = (bitmap_buf[i >> 3] >> (i & 7)) & 1u;
    size_t fse_bit = fse_payload_bytes(expand_buf, bitmap_bits,
                                         scratch, scratch_cap);
    s->total_fse_bit_bits += (double)fse_bit * 8.0;

    return total;
}

static void print_row(const char *name, int N, double H, double huf,
                       double fse_byte_bpB,
                       double top1_bpB, int top1_depth,
                       double flat_today_bpB, double flat_floor_bpB,
                       int n_flat, int is_main)
{
    char marker = is_main ? '*' : ' ';
    printf("%c %-22s %6d  %5.3f  %5.3f  %5.3f  %5.3f  %3d  %5.3f  %5.3f  %3d\n",
            marker, name, N, H, huf, fse_byte_bpB,
            top1_bpB, top1_depth,
            flat_today_bpB, flat_floor_bpB, n_flat);
}

int main(int argc, char **argv)
{
    bench_init();
    int n_dist = bench_num_distributions();

    static uint8_t symbols[BLK];
    static uint8_t bitmap_buf[BLK + 16];
    static uint8_t expand_buf[BLK * 8 + 16];
    static uint8_t scratch[BLK + 1024];

    printf("Per-dataset entropy + FSE-per-node breakdown (N = %d B "
            "synthetic IID sample, seed=1).\n", BLK);
    printf("All bits/byte values normalised to bits per source byte.\n");
    printf("  H          = byte entropy = Σ p · −log2 p\n");
    printf("  huf        = weighted Huffman avg code length\n");
    printf("  fse_byte   = Σ FSE(bitmap bytes) for non-flat internal nodes,\n");
    printf("               bits / N.  (Excludes nodes inside flat subtrees.)\n");
    printf("  top1_b     = max single-node bits saved per source byte\n");
    printf("               = max[ n_node · (1 − H_node) ] / N\n");
    printf("  top1_d     = depth of that top-1 node (root = 0)\n");
    printf("  flat_today = bits / source byte emitted by ph's flat-subtree\n");
    printf("               fast path today (= Σ n_subtree · D / N, where\n");
    printf("               D is the flat-subtree depth, raw bits / symbol).\n");
    printf("  flat_floor = entropy floor for those same flat-subtree bits\n");
    printf("               = Σ n_subtree · H_conditional_local / N.\n");
    printf("               (flat_today − flat_floor) is what we'd save by\n");
    printf("               FSE-ing the flat regions instead of raw-emitting.\n");
    printf("  n_flat     = number of maximal flat subtrees (D ≥ 2).\n");
    printf("  * marks MAIN distributions.\n\n");

    printf("  %-22s     N       H    huf  fseB    top1   top1  flat_  flat_  n\n",
            "dataset");
    printf("  %-22s                                _byte    _b     _d  today  floor flat\n",
            "");
    printf("  %-22s  ------  -----  -----  -----  -----  ----  -----  -----  ---\n",
            "----------------------");

    for (int d = 0; d < n_dist; d++) {
        const char *name = bench_dist_name(d);
        const uint64_t *freq_global = bench_dist_freq(d);
        (void)freq_global;

        bench_generate_symbols(d, symbols, BLK, /*seed=*/1ULL);

        /* Per-sample histogram + entropy. */
        uint64_t hist[256] = {0};
        for (int i = 0; i < BLK; i++) hist[symbols[i]]++;
        double H = 0.0;
        for (int i = 0; i < 256; i++) {
            if (hist[i] == 0) continue;
            double p = (double)hist[i] / (double)BLK;
            H -= p * log2(p);
        }

        /* Build the Huffman table; pivco's build takes a table* by
         * reference, returns 0 on success. */
        pivco_table_t table_storage;
        if (pivco_build_table(bench_cfg(), freq_global, &table_storage) != 0) {
            printf("  %-22s  build_table failed\n", name);
            continue;
        }
        const pivco_table_t *table = &table_storage;

        /* Weighted Huffman avg code length from leaf depths in the table.
         * pivco's tree stores leaf depth implicitly via the symbol's
         * path; reconstruct by walking. */
        double huf_total_bits = 0.0;
        /* Simple BFS to record per-leaf depth. */
        int leaf_depth[256];
        for (int i = 0; i < 256; i++) leaf_depth[i] = -1;
        /* Use a stack-based DFS. */
        int16_t stk[2 * 256];
        int     dep[2 * 256];
        int sp = 0;
        stk[sp] = table->tree_root;
        dep[sp] = 0;
        sp++;
        while (sp > 0) {
            sp--;
            int16_t cur = stk[sp];
            int     d2  = dep[sp];
            const pivco_tree_node_t *nn = &table->tree[cur];
            if (nn->symbol >= 0) {
                leaf_depth[nn->symbol] = d2;
            } else {
                stk[sp] = nn->left;  dep[sp] = d2 + 1; sp++;
                stk[sp] = nn->right; dep[sp] = d2 + 1; sp++;
            }
        }
        for (int i = 0; i < 256; i++) {
            if (hist[i] == 0 || leaf_depth[i] < 0) continue;
            double p = (double)hist[i] / (double)BLK;
            huf_total_bits += p * (double)leaf_depth[i];
        }

        stats_t s = { .xs_state = 0x9E3779B97F4A7C15ULL ^ (uint64_t)d };
        walk(table, table->tree_root, 0, hist, &s,
              bitmap_buf, expand_buf, scratch, sizeof(scratch));

        double fse_byte_bpB  = s.total_fse_byte_bits  / (double)BLK;
        double top1_bpB      = s.top1_saved_bits      / (double)BLK;
        double flat_today_bpB = s.flat_raw_bits       / (double)BLK;
        double flat_floor_bpB = s.flat_entropy_bits   / (double)BLK;

        print_row(name, BLK, H, huf_total_bits, fse_byte_bpB,
                   top1_bpB, s.top1_depth,
                   flat_today_bpB, flat_floor_bpB,
                   s.n_flat_subtrees, bench_dist_is_main(d));
    }

    return 0;
    (void)argc; (void)argv;
}
