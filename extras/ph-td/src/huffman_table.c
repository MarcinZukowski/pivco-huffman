#include "pivco_huffman.h"
#include <string.h>
#include <stdlib.h>

/* ---------- Flat-subtree detection ---------- */

/* Shortest leaf depth below node_id (0 if node_id is itself a leaf). */
static int flat_local_min(const pivco_tree_node_t *tree, int16_t node_id)
{
    const pivco_tree_node_t *n = &tree[node_id];
    if (n->symbol >= 0) return 0;
    int l = flat_local_min(tree, n->left);
    int r = flat_local_min(tree, n->right);
    return 1 + (l < r ? l : r);
}

/* Deepest leaf depth below node_id. */
static int flat_local_max(const pivco_tree_node_t *tree, int16_t node_id)
{
    const pivco_tree_node_t *n = &tree[node_id];
    if (n->symbol >= 0) return 0;
    int l = flat_local_max(tree, n->left);
    int r = flat_local_max(tree, n->right);
    return 1 + (l > r ? l : r);
}

/* Populate code_to_sym for a flat subtree of depth D rooted at node_id.
   For each leaf symbol reachable from this root, local_code = low D bits
   of code[sym] (canonical Huffman codes are MSB-first; low D bits are
   the in-subtree part).  `out_base` is the pool offset for this subtree;
   table->flat_code_to_sym[out_base + local_code] = sym. */
static void flat_fill_code_to_sym(pivco_huffman_table_t *t,
                                   int16_t node_id, int D,
                                   uint16_t out_base)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) {
        int sym = n->symbol;
        int local_code = t->code[sym] & ((1 << D) - 1);
        t->flat_code_to_sym[out_base + local_code] = (uint8_t)sym;
        return;
    }
    flat_fill_code_to_sym(t, n->left,  D, out_base);
    flat_fill_code_to_sym(t, n->right, D, out_base);
}

/* DFS from root.  At every internal node, if subtree is flat with depth
   >= 2, mark node as a maximal flat-subtree root and stop (its descendants
   are not separately flagged).  Otherwise recurse into children. */
static void flat_mark_subtrees(pivco_huffman_table_t *t,
                                int16_t node_id,
                                uint16_t *pool_cursor)
{
    const pivco_tree_node_t *n = &t->tree[node_id];
    if (n->symbol >= 0) return;
    int lmin = flat_local_min(t->tree, node_id);
    int lmax = flat_local_max(t->tree, node_id);
    if (lmin == lmax && lmin >= 2) {
        int D = lmin;
        int size = 1 << D;
        t->flat_depth[node_id]  = (uint8_t)D;
        t->flat_offset[node_id] = *pool_cursor;
        flat_fill_code_to_sym(t, node_id, D, *pool_cursor);
        *pool_cursor = (uint16_t)(*pool_cursor + size);
        return;  /* maximal — don't descend */
    }
    flat_mark_subtrees(t, n->left,  pool_cursor);
    flat_mark_subtrees(t, n->right, pool_cursor);
}

/* ---------- Rank-aware synthetic-frequency construction ----------
 *
 * Given just `code_lens` (one length per symbol) and an optional
 * `rank_within_tier` (per-symbol 0-based rank, -1 for default), build
 * a synth_freq array that:
 *  - reproduces the same code lengths through pivco_huffman_build_table
 *    (because inter-tier ratios are exact powers of 2)
 *  - preserves the encoder's intended within-tier ordering as the
 *    primary sort key (instead of the default smaller-sym tiebreak)
 *
 * Formula: synth_freq[s] = (1 << (max_len - L)) * BIG + (K_L - rank).
 *   BIG = 1024: large enough that the within-tier offset (max 256)
 *               can't cross the inter-tier step (factor of 2).
 *   K_L:  count of symbols at code length L.
 *   rank: 0 = top, K_L - 1 = bottom.  When < 0 (default), use 0
 *         (uniform within tier, identical to original synth_freq
 *         behavior).
 *
 * Both encoder and decoder feed the same code_lens + rank_within_tier
 * into this builder, so they construct identical synth_freqs and
 * therefore identical Huffman tables. */
static void build_rank_aware_synth_freq(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    const int16_t *rank_within_tier,
    uint64_t freq_out[PIVCO_MAX_SYMBOLS])
{
    int max_len = 0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (code_lens[s] > max_len) max_len = code_lens[s];
    }
    int sym_count_per_len[PIVCO_MAX_CODE_LEN + 1] = {0};
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (code_lens[s] > 0 && code_lens[s] <= PIVCO_MAX_CODE_LEN)
            sym_count_per_len[code_lens[s]]++;
    }
    const uint64_t BIG = 1024;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        freq_out[s] = 0;
        int L = code_lens[s];
        if (L == 0) continue;
        uint64_t base = ((uint64_t)1 << (max_len - L)) * BIG;
        int rank = (rank_within_tier && rank_within_tier[s] >= 0)
                   ? rank_within_tier[s] : -1;
        if (rank < 0) {
            freq_out[s] = base;
        } else {
            int K = sym_count_per_len[L];
            freq_out[s] = base + (uint64_t)(K - rank);
        }
    }
}

/* ---------- Min-heap for Huffman tree construction ---------- */

typedef struct {
    uint64_t freq;
    int      symbol;  /* >= 0 for leaf, < 0 for internal (-1 - index) */
    int      left;
    int      right;
} huff_node_t;

typedef struct {
    int      indices[PIVCO_MAX_SYMBOLS * 2];
    int      size;
    huff_node_t *nodes;
} min_heap_t;

static void heap_swap(min_heap_t *h, int a, int b)
{
    int tmp = h->indices[a];
    h->indices[a] = h->indices[b];
    h->indices[b] = tmp;
}

static void heap_sift_up(min_heap_t *h, int i)
{
    while (i > 0) {
        int parent = (i - 1) / 2;
        uint64_t fi = h->nodes[h->indices[i]].freq;
        uint64_t fp = h->nodes[h->indices[parent]].freq;
        if (fi < fp || (fi == fp && h->indices[i] < h->indices[parent])) {
            heap_swap(h, i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

static void heap_sift_down(min_heap_t *h, int i)
{
    while (1) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < h->size) {
            uint64_t fl = h->nodes[h->indices[left]].freq;
            uint64_t fs = h->nodes[h->indices[smallest]].freq;
            if (fl < fs || (fl == fs && h->indices[left] < h->indices[smallest]))
                smallest = left;
        }
        if (right < h->size) {
            uint64_t fr = h->nodes[h->indices[right]].freq;
            uint64_t fs = h->nodes[h->indices[smallest]].freq;
            if (fr < fs || (fr == fs && h->indices[right] < h->indices[smallest]))
                smallest = right;
        }
        if (smallest == i) break;
        heap_swap(h, i, smallest);
        i = smallest;
    }
}

static void heap_push(min_heap_t *h, int node_idx)
{
    h->indices[h->size] = node_idx;
    heap_sift_up(h, h->size);
    h->size++;
}

static int heap_pop(min_heap_t *h)
{
    int result = h->indices[0];
    h->size--;
    h->indices[0] = h->indices[h->size];
    if (h->size > 0) heap_sift_down(h, 0);
    return result;
}

/* ---------- Code length extraction via DFS ---------- */

static void extract_lengths(const huff_node_t *nodes, int idx, int depth,
                            uint8_t *lengths)
{
    if (nodes[idx].symbol >= 0) {
        /* Leaf */
        lengths[nodes[idx].symbol] = (uint8_t)(depth > 0 ? depth : 1);
        return;
    }
    extract_lengths(nodes, nodes[idx].left, depth + 1, lengths);
    extract_lengths(nodes, nodes[idx].right, depth + 1, lengths);
}

/* ---------- Code length limiting (DEFLATE-style, RFC 1951) ---------- */

static void limit_code_lengths(uint8_t *lengths, int n_symbols, int max_len)
{
    /* Count symbols at each length */
    int count[64] = {0}; /* support original lengths up to 63 */
    int max_orig = 0;
    for (int i = 0; i < n_symbols; i++) {
        if (lengths[i] > 0) {
            count[lengths[i]]++;
            if (lengths[i] > max_orig) max_orig = lengths[i];
        }
    }
    if (max_orig <= max_len) return; /* nothing to do */

    /* Move all symbols longer than max_len down to max_len */
    for (int i = max_orig; i > max_len; i--) {
        count[max_len] += count[i];
        count[i] = 0;
    }

    /* Now Kraft sum may exceed 1.0. Fix by moving symbols from max_len
       to shorter lengths. Each time we move one symbol from length L
       to length L-1, the Kraft delta is: 2^(max-L+1) - 2^(max-L) = 2^(max-L).
       But that creates a "debt" at length L-1 which may also overflow.

       Work bottom-up: for each length from max_len down, if we have
       overflow, push pairs up to parent (length-1). */

    /* Compute Kraft sum in units of 2^(-max_len) */
    uint64_t kraft = 0;
    for (int i = 1; i <= max_len; i++) {
        kraft += (uint64_t)count[i] << (max_len - i);
    }
    uint64_t target = (uint64_t)1 << max_len;

    /* While over-full, increase the longest codes */
    while (kraft > target) {
        /* Find a symbol at a length < max_len and increase it by 1.
           This reduces Kraft by 2^(max_len - len) - 2^(max_len - len - 1)
           = 2^(max_len - len - 1). Pick the longest such length to
           minimize Kraft reduction per step. */
        int best = -1;
        for (int len = max_len - 1; len >= 1; len--) {
            if (count[len] > 0) {
                best = len;
                break;
            }
        }
        if (best < 0) break; /* shouldn't happen */

        count[best]--;
        count[best + 1]++;
        kraft -= (uint64_t)1 << (max_len - best - 1);
    }

    /* While under-full, decrease some max_len codes to shorter lengths.
       This fills unused Kraft capacity. */
    while (kraft < target && count[max_len] > 0) {
        /* Find the shortest length where we can add capacity */
        for (int len = max_len - 1; len >= 1; len--) {
            uint64_t gain = ((uint64_t)1 << (max_len - len)) -
                            ((uint64_t)1 << (max_len - len - 1));
            /* gain = 2^(max_len-len) - 2^(max_len-len-1) = 2^(max_len-len-1) */
            /* But moving from max_len to len changes kraft by:
               +2^(max_len-len) - 2^(max_len-max_len) = 2^(max_len-len) - 1 */
            uint64_t delta = ((uint64_t)1 << (max_len - len)) - 1;
            if (kraft + delta <= target && count[max_len] > 0) {
                count[max_len]--;
                count[len]++;
                kraft += delta;
                break;
            }
        }
        /* If we couldn't shorten anything, done */
        if (kraft < target) {
            /* Try filling one slot at max_len-1 at a time */
            uint64_t delta = ((uint64_t)1 << 1) - 1; /* moving max_len to max_len-1 */
            if (kraft + delta <= target && count[max_len] >= 2) {
                /* Move one from max_len to max_len-1: net = +2 - 1 = +1 */
                count[max_len]--;
                count[max_len - 1]++;
                kraft += 1;
            } else {
                break;
            }
        }
    }

    /* Reassign lengths based on new counts.
       Sort symbols by original length (as proxy for frequency),
       assign shortest new lengths to the most frequent symbols. */
    /* Build sorted list of (original_length, symbol_index) */
    typedef struct { uint8_t len; uint8_t sym; } ls_t;
    ls_t sorted[PIVCO_MAX_SYMBOLS];
    int ns = 0;
    for (int i = 0; i < n_symbols; i++) {
        if (lengths[i] > 0) {
            sorted[ns].len = lengths[i] > max_len ? (uint8_t)max_len : lengths[i];
            sorted[ns].sym = (uint8_t)i;
            ns++;
        }
    }
    /* Sort by original length (shorter = more frequent = should get shorter code) */
    for (int i = 1; i < ns; i++) {
        ls_t tmp = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j].len > tmp.len) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = tmp;
    }

    /* Assign new lengths from count array */
    int si = 0;
    for (int len = 1; len <= max_len && si < ns; len++) {
        for (int c = 0; c < count[len] && si < ns; c++) {
            lengths[sorted[si].sym] = (uint8_t)len;
            si++;
        }
    }
}

/* ---------- Canonical Huffman code assignment ---------- */

int pivco_huffman_build_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table)
{
    if (!freq || !table) return PIVCO_ERR_NULL;

    memset(table, 0, sizeof(*table));

    /* Count symbols with nonzero frequency */
    int n_used = 0;
    int used[PIVCO_MAX_SYMBOLS];
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (freq[i] > 0) {
            used[n_used++] = i;
        }
    }

    if (n_used == 0) return PIVCO_ERR_EMPTY;

    table->num_symbols = (uint16_t)n_used;

    if (n_used == 1) {
        /* Single symbol: code = 0, length = 1 */
        int sym = used[0];
        table->code[sym] = 0;
        table->code_len[sym] = 1;
        table->max_len = 1;
        table->min_len = 1;
        table->sym_count[1] = 1;
        table->first_code[1] = 0;
        table->first_sym_idx[1] = 0;
        table->sorted_symbols[0] = (uint8_t)sym;
        /* (no 2^MAX_CODE_LEN flat table -- unused in this TD library) */
        /* Build tree: root (internal) -> left child (leaf) */
        table->tree[0].symbol = -1;
        table->tree[0].left = 1;
        table->tree[0].right = 2;
        table->tree[1].symbol = (int16_t)sym;
        table->tree[1].left = -1;
        table->tree[1].right = -1;
        table->tree[2].symbol = (int16_t)sym; /* both children = same symbol */
        table->tree[2].left = -1;
        table->tree[2].right = -1;
        table->tree_root = 0;
        table->tree_node_count = 3;
        table->prefill_sym = (uint8_t)sym;
        table->prefill_node = 1;
        /* Classify the 3 nodes for the single-symbol tree:
         *   node 0 (root, internal): both children leaves, left=skip → HALF_RIGHT
         *   node 1 (left leaf, prefilled): SKIP
         *   node 2 (right leaf, same sym): LEAF (functionally equivalent to SKIP
         *           since output is already filled, but classify per the rule). */
        table->node_type[0] = PIVCO_NODE_HALF_RIGHT;
        table->node_type[1] = PIVCO_NODE_SKIP;
        table->node_type[2] = PIVCO_NODE_LEAF;
        return PIVCO_OK;
    }

    /* Build Huffman tree using min-heap */
    huff_node_t nodes[PIVCO_MAX_SYMBOLS * 2];
    memset(nodes, 0, sizeof(nodes));
    min_heap_t heap;
    heap.size = 0;
    heap.nodes = nodes;

    int next_node = 0;
    for (int i = 0; i < n_used; i++) {
        nodes[next_node].freq = freq[used[i]];
        nodes[next_node].symbol = used[i];
        nodes[next_node].left = -1;
        nodes[next_node].right = -1;
        heap_push(&heap, next_node);
        next_node++;
    }

    while (heap.size > 1) {
        int a = heap_pop(&heap);
        int b = heap_pop(&heap);
        nodes[next_node].freq = nodes[a].freq + nodes[b].freq;
        nodes[next_node].symbol = -1; /* internal */
        nodes[next_node].left = a;
        nodes[next_node].right = b;
        heap_push(&heap, next_node);
        next_node++;
    }

    int root = heap_pop(&heap);

    /* Extract code lengths */
    uint8_t lengths[PIVCO_MAX_SYMBOLS];
    memset(lengths, 0, sizeof(lengths));
    extract_lengths(nodes, root, 0, lengths);

    /* Limit code lengths to PIVCO_MAX_CODE_LEN */
    limit_code_lengths(lengths, PIVCO_MAX_SYMBOLS, PIVCO_MAX_CODE_LEN);

    /* Copy lengths to table */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        table->code_len[i] = lengths[i];
    }

    /* Count symbols per length */
    uint8_t max_len = 0, min_len = PIVCO_MAX_CODE_LEN + 1;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (lengths[i] > 0) {
            table->sym_count[lengths[i]]++;
            if (lengths[i] > max_len) max_len = lengths[i];
            if (lengths[i] < min_len) min_len = lengths[i];
        }
    }
    table->max_len = max_len;
    table->min_len = min_len;

    /* ---------- Flat-aware code assignment ----------
     *
     * Goal: give each symbol a code of its assigned length such that the
     * resulting binary tree has as many large flat-D>=2 subtrees as
     * possible (consolidates the partition path during tree-walk decode).
     * Compression is unaffected — code lengths match the Huffman result.
     *
     * Algorithm: per length L, decompose c_L by its binary representation
     * into "chunks": bits >= 2 form D>=2 flat subtrees of size 2^D rooted
     * at depth L-D; bit 1 forms a D=1 sibling pair (handled by stage
     * fusion at decode); bit 0 is a singleton.  Sort chunks by their
     * tree-depth asc (depth = L-D for D>=2 chunks, L-1 for D=1, L for
     * singletons), then canonical-assign codes to chunks.  Within each
     * chunk, top-freq-first symbols of length L are assigned to its
     * 2^bit suffix slots (highest freqs go to the largest-D chunk per
     * length, where the partition-path savings are deepest).
     *
     * See IDEAS.md "Flat-aware Huffman tree restructurer" for the gap
     * analysis (extras/bench/bench_flat_optimal.c).
     */

    /* Per-length: collect symbols sorted by frequency desc (ties by
       symbol value asc for determinism). */
    typedef struct {
        uint8_t  sym;
        uint64_t freq;
    } sf_t;
    sf_t  flat_items[PIVCO_MAX_SYMBOLS];
    int   per_len_start[PIVCO_MAX_CODE_LEN + 2];
    {
        int cursor = 0;
        for (int L = 1; L <= max_len; L++) {
            per_len_start[L] = cursor;
            int seg_start = cursor;
            for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
                if (lengths[s] == (uint8_t)L) {
                    flat_items[cursor].sym  = (uint8_t)s;
                    flat_items[cursor].freq = freq[s];
                    cursor++;
                }
            }
            /* Insertion sort: highest freq first; tie-break by smaller sym. */
            for (int i = seg_start + 1; i < cursor; i++) {
                sf_t cur = flat_items[i];
                int j = i - 1;
                while (j >= seg_start &&
                       (flat_items[j].freq < cur.freq ||
                        (flat_items[j].freq == cur.freq && flat_items[j].sym > cur.sym))) {
                    flat_items[j + 1] = flat_items[j];
                    j--;
                }
                flat_items[j + 1] = cur;
            }
        }
        per_len_start[max_len + 1] = cursor;
    }

    /* Decompose each c_L into chunks (one chunk per set bit). */
    typedef struct {
        uint16_t L;
        uint16_t bit;       /* 0..PIVCO_MAX_CODE_LEN */
        uint16_t depth;     /* tree-depth of chunk root */
        uint16_t n_syms;    /* 1 << bit */
        int      sym_idx;   /* index into flat_items */
    } chunk_t;
    chunk_t chunks[PIVCO_MAX_SYMBOLS];   /* upper bound: one chunk per symbol */
    int n_chunks = 0;
    {
        for (int L = 1; L <= max_len; L++) {
            int c = table->sym_count[L];
            int cur = per_len_start[L];
            /* Iterate set bits high-to-low so larger chunks come first
               within the length (matters only for top-freq-first symbol
               assignment within the length). */
            for (int bit = PIVCO_MAX_CODE_LEN; bit >= 0; bit--) {
                if (c & (1 << bit)) {
                    int n = 1 << bit;
                    int depth;
                    if      (bit >= 2) depth = L - bit;
                    else if (bit == 1) depth = L - 1;
                    else               depth = L;
                    chunks[n_chunks].L      = (uint16_t)L;
                    chunks[n_chunks].bit    = (uint16_t)bit;
                    chunks[n_chunks].depth  = (uint16_t)depth;
                    chunks[n_chunks].n_syms = (uint16_t)n;
                    chunks[n_chunks].sym_idx = cur;
                    cur += n;
                    n_chunks++;
                }
            }
        }
    }

    /* Sort chunks by depth asc (stable; ties keep their natural order
       which is L asc by length, larger-bit-first within length). */
    for (int i = 1; i < n_chunks; i++) {
        chunk_t cur = chunks[i];
        int j = i - 1;
        while (j >= 0 && chunks[j].depth > cur.depth) {
            chunks[j + 1] = chunks[j];
            j--;
        }
        chunks[j + 1] = cur;
    }

    /* Canonical-assign codes to chunks (chunk-level Kraft sum = 1).
       Each chunk gets a code prefix of length `chunk.depth`.  Within
       the chunk, symbol i takes suffix i for i in [0, 2^bit). */
    {
        uint32_t code = 0;
        int prev_depth = 0;
        for (int ci = 0; ci < n_chunks; ci++) {
            int d = chunks[ci].depth;
            if (d > prev_depth) code <<= (d - prev_depth);
            int bit = chunks[ci].bit;
            int n   = chunks[ci].n_syms;
            for (int i = 0; i < n; i++) {
                uint8_t sym = flat_items[chunks[ci].sym_idx + i].sym;
                table->code[sym] = (uint16_t)((code << bit) | (uint32_t)i);
            }
            code += 1;
            prev_depth = d;
        }
    }

    /* Populate sorted_symbols / first_sym_idx / first_code from the
       new code assignment.  These fields are not used by runtime
       decoders (only by the tree-walk pass below), but we keep them
       in length-asc order for compatibility with anyone inspecting
       the table. */
    int sorted_idx = 0;
    for (int len = 1; len <= max_len; len++) {
        table->first_sym_idx[len] = (uint16_t)sorted_idx;
        uint16_t min_code = 0xFFFF;
        int seg_start = sorted_idx;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            if (lengths[s] == (uint8_t)len) {
                if (table->code[s] < min_code) min_code = table->code[s];
                table->sorted_symbols[sorted_idx++] = (uint8_t)s;
            }
        }
        (void)seg_start;
        table->first_code[len] = (min_code == 0xFFFF) ? 0 : min_code;
    }

    /* The 2^MAX_CODE_LEN flat decode table (decode_sym/decode_len) has no
     * reader in this TD library (the tree-walk decoders don't use it), so it
     * is not built -- it was dead work inflating build measurements. */

    /* Build canonical Huffman tree for PIVCO tree-walk.
       Insert each symbol's canonical code into the tree by walking
       bits MSB-first, creating internal nodes as needed. */
    {
        int16_t nc = 0; /* node count */
        /* Root node */
        table->tree[nc].symbol = -1;
        table->tree[nc].left = -1;
        table->tree[nc].right = -1;
        nc++;
        table->tree_root = 0;

        for (int si = 0; si < sorted_idx; si++) {
            uint8_t sym = table->sorted_symbols[si];
            uint16_t c = table->code[sym];
            int len = table->code_len[sym];
            int16_t cur = 0; /* start at root */

            for (int b = len - 1; b >= 0; b--) {
                int bit = (c >> b) & 1;
                int16_t *child = bit ? &table->tree[cur].right
                                     : &table->tree[cur].left;
                if (*child < 0) {
                    /* Create new node */
                    *child = nc;
                    table->tree[nc].symbol = -1;
                    table->tree[nc].left = -1;
                    table->tree[nc].right = -1;
                    nc++;
                }
                cur = *child;
            }
            table->tree[cur].symbol = (int16_t)sym;
        }
        table->tree_node_count = nc;
    }

    /* Find the most frequent symbol (shortest code) for prefill.
       Walk the tree to find its node ID. */
    {
        uint8_t best_sym = 0;
        uint8_t best_len = 255;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            if (table->code_len[s] > 0 && table->code_len[s] < best_len) {
                best_len = table->code_len[s];
                best_sym = (uint8_t)s;
            }
        }
        table->prefill_sym = best_sym;
        /* Find the tree node for this symbol */
        table->prefill_node = -1;
        for (int16_t i = 0; i < table->tree_node_count; i++) {
            if (table->tree[i].symbol == (int16_t)best_sym) {
                table->prefill_node = i;
                break;
            }
        }
    }

    /* Flat-subtree detection.  Mark every internal node that is the root
       of a maximal flat subtree with depth >= 2 and fill its
       code_to_sym lookup.  The root itself is eligible — when the whole
       tree is flat with depth >= 2, the root gets flat_depth = max_len
       and the decoder handles the whole block directly via its
       INTERNAL_FLAT dispatch arm. */
    {
        uint16_t pool_cursor = 0;
        flat_mark_subtrees(table, table->tree_root, &pool_cursor);
    }

    /* Classify each node for decode-dispatch.  Mirrors the existing
     * conditional priority in decode_node_neon:
     *   FLAT (subtree, D>=2)  >  HALF_RIGHT/LEFT  >  BOTH_LEAVES  >  FULL.
     * Leaves are SKIP if prefilled, LEAF otherwise. */
    for (int16_t i = 0; i < table->tree_node_count; i++) {
        const pivco_tree_node_t *node = &table->tree[i];

        if (node->symbol >= 0) {
            /* Leaf */
            table->node_type[i] = (i == table->prefill_node)
                                ? (uint8_t)PIVCO_NODE_SKIP
                                : (uint8_t)PIVCO_NODE_LEAF;
            continue;
        }

        /* Internal node */
        if (table->flat_depth[i] >= 2) {
            table->node_type[i] = (uint8_t)PIVCO_NODE_INTERNAL_FLAT;
            continue;
        }

        int16_t left_id  = node->left;
        int16_t right_id = node->right;
        int left_leaf  = (table->tree[left_id].symbol  >= 0);
        int right_leaf = (table->tree[right_id].symbol >= 0);
        int left_skip  = (left_id  == table->prefill_node);
        int right_skip = (right_id == table->prefill_node);

        if (left_leaf && left_skip) {
            table->node_type[i] = (uint8_t)PIVCO_NODE_HALF_RIGHT;
        } else if (right_leaf && right_skip) {
            table->node_type[i] = (uint8_t)PIVCO_NODE_HALF_LEFT;
        } else if (left_leaf && right_leaf) {
            table->node_type[i] = (uint8_t)PIVCO_NODE_BOTH_LEAVES;
        } else {
            table->node_type[i] = (uint8_t)PIVCO_NODE_INTERNAL_FULL;
        }
    }

    /* Populate code_la (left-aligned code) for the dense tree-walk
     * encoder.  Bit-d of the canonical code lives at position 15-d of
     * code_la (for d < code_len[sym]). */
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        uint8_t len = table->code_len[s];
        table->code_la[s] = len > 0
            ? (uint16_t)(table->code[s] << (16 - len))
            : 0;
    }

    /* Populate max_leaf_depth[node] for every internal node.  Used by
     * the encoder to detect when a subtree's remaining bits fit in a
     * byte and can be processed with uint8-wide partitions.  Iterative
     * post-order via tree_node_count traversal: tree nodes are
     * allocated in order of construction (children before parents in
     * our build), so a single pass from node 0 to tree_node_count
     * fills max_leaf_depth bottom-up.
     *
     * BUT: that ordering is not guaranteed in general.  Use recursion
     * for correctness; the depth is small (<= PIVCO_MAX_CODE_LEN). */
    {
        /* Iterative DFS via an explicit small stack.  Simpler than
         * thinking about node-allocation order, and recursion-free. */
        int stack[2 * PIVCO_MAX_TREE_NODES];
        int top = 0;
        stack[top++] = table->tree_root;
        /* First pass: count children visited per node, leaf := 0. */
        memset(table->max_leaf_depth, 0, sizeof(table->max_leaf_depth));
        int order[PIVCO_MAX_TREE_NODES];
        int order_n = 0;
        while (top > 0) {
            int16_t id = (int16_t)stack[--top];
            order[order_n++] = id;
            const pivco_tree_node_t *n = &table->tree[id];
            if (n->symbol < 0) {
                stack[top++] = n->left;
                stack[top++] = n->right;
            }
        }
        /* Process in reverse (children before parents). */
        for (int oi = order_n - 1; oi >= 0; oi--) {
            int16_t id = (int16_t)order[oi];
            const pivco_tree_node_t *n = &table->tree[id];
            if (n->symbol >= 0) {
                table->max_leaf_depth[id] = 0;
            } else {
                uint8_t l = table->max_leaf_depth[n->left];
                uint8_t r = table->max_leaf_depth[n->right];
                table->max_leaf_depth[id] = (uint8_t)(1 + (l > r ? l : r));
            }
        }
    }

    return PIVCO_OK;
}

/* Public API: build a table from code lengths + optional within-tier
 * ordering.  See the comment in pivco_huffman.h. */
int pivco_huffman_build_table_from_code_lens(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    const int16_t *rank_within_tier,
    pivco_huffman_table_t *table)
{
    if (!code_lens || !table) return PIVCO_ERR_NULL;
    uint64_t freq[PIVCO_MAX_SYMBOLS];
    build_rank_aware_synth_freq(code_lens, rank_within_tier, freq);
    return pivco_huffman_build_table(freq, table);
}
