#include "pivco_huffman.h"
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#ifdef PIVCO_BUILD_PROFILE
#include <mach/mach_time.h>            /* M4-native counter; no clock syscall */
double      pivco_prof_ticks[32];      /* accumulated raw mach ticks per phase */
const char *pivco_prof_name[32];
int         pivco_prof_n;
static uint64_t pivco_prof_t0;
static int      pivco_prof_i;
#define PROF_RESET()  do { pivco_prof_t0 = mach_absolute_time(); pivco_prof_i = 0; } while (0)
#define PROF_MARK(nm) do { uint64_t _t = mach_absolute_time();                  \
    pivco_prof_ticks[pivco_prof_i] += (double)(_t - pivco_prof_t0);             \
    pivco_prof_name[pivco_prof_i] = (nm); pivco_prof_t0 = _t;                    \
    if (++pivco_prof_i > pivco_prof_n) pivco_prof_n = pivco_prof_i; } while (0)
#else
#define PROF_RESET()
#define PROF_MARK(nm)
#endif

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

/* Builds everything downstream of the code lengths (canonical assignment,
 * tree, flat-subtree detection, aux tables).  Shared by the encode path
 * (after the min-heap derives lengths from frequencies) and the decode path
 * (lengths come straight off the wire -- no heap needed).  Assumes `table` is
 * already zeroed and table->num_symbols is set; caller handles n_used <= 1. */
static int build_table_finish(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table);

/* Single-symbol degenerate tree: root -> two leaves of the same symbol, with
 * the left leaf prefilled.  Assumes `table` is zeroed. */
static void build_single_symbol_table(int sym, pivco_huffman_table_t *table)
{
    table->code[sym] = 0;
    table->code_len[sym] = 1;
    table->max_len = 1;
    table->min_len = 1;
    table->sym_count[1] = 1;
    table->first_code[1] = 0;
    table->first_sym_idx[1] = 0;
    table->sorted_symbols[0] = (uint8_t)sym;
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
    /* node 0 (root): both children leaves, left=skip -> HALF_RIGHT;
     * node 1 (left leaf, prefilled): SKIP; node 2 (right, same sym): LEAF. */
    table->node_type[0] = PIVCO_NODE_HALF_RIGHT;
    table->node_type[1] = PIVCO_NODE_SKIP;
    table->node_type[2] = PIVCO_NODE_LEAF;
}

int pivco_huffman_build_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table)
{
    if (!freq || !table) return PIVCO_ERR_NULL;
    PROF_RESET();   /* the freq (encode) path was previously un-instrumented */

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
        build_single_symbol_table(used[0], table);
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

    PROF_MARK("huff_heap");   /* freq->lengths: heap tree build + extract + limit */
    return build_table_finish(lengths, table);
}

static int build_table_finish(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table)
{
    /* Copy lengths to table */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        table->code_len[i] = lengths[i];
    }
    PROF_MARK("copy_len");

    /* Histogram code lengths.  The len>0 guard isn't about correctness
       (sym_count[0] is an unread scratch bin) -- it keeps the unused symbols
       from all piling onto bin 0, whose serial store-to-load-forward chain
       was 5x slower than the (well-predicted) branch on sparse alphabets.
       min/max are derived from the bins below, not inline here. */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (lengths[i] > 0)
            table->sym_count[lengths[i]]++;
    }
    PROF_MARK("count_len");

    /* Derive min/max code length from the (<=11) length bins. */
    uint8_t max_len = 0, min_len = PIVCO_MAX_CODE_LEN + 1;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        if (table->sym_count[L]) {
            if (L < min_len) min_len = (uint8_t)L;
            max_len = (uint8_t)L;
        }
    }
    table->max_len = max_len;
    table->min_len = min_len;
    PROF_MARK("minmax");

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

    /* Per-length: collect symbols in symbol-value order.
     *
     * We used to sort within a tier by frequency-desc so the heaviest
     * symbols landed in the largest flat chunk.  That was dropped: it is
     * the only thing that made the tree depend on within-tier frequency
     * order, which in turn forced a within-tier ordering onto the wire
     * (the v0.3 ORDERING section + rank_within_tier) so the decoder could
     * reproduce it.  On FSE-coded blocks the freq-order "win" only *masked*
     * a bad FSE commit policy (the gate ignores FSE decode cost).  Plain
     * symbol-value order is deterministic from the code lengths alone, so
     * encoder and decoder agree with no rank info transmitted. */
    typedef struct {
        uint8_t  sym;
    } sf_t;
    sf_t  flat_items[PIVCO_MAX_SYMBOLS];
    int   per_len_start[PIVCO_MAX_CODE_LEN + 2];
    {
        /* Counting sort by length: prefix-sum the per-length counts, then a
           single symbol-order pass places each symbol.  Equivalent to the
           old nested for-L/for-s scan but O(256) instead of O(max_len*256). */
        int acc = 0;
        int cursor[PIVCO_MAX_CODE_LEN + 2];
        for (int L = 1; L <= max_len; L++) {
            per_len_start[L] = acc;
            cursor[L] = acc;
            acc += table->sym_count[L];
        }
        per_len_start[max_len + 1] = acc;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            uint8_t L = lengths[s];
            if (L) flat_items[cursor[L]++].sym = (uint8_t)s;
        }
    }
    PROF_MARK("gather_2a");

    /* Decompose each c_L into chunks.  Strategy depends on tree mode --
       see pivco_huffman_set_tree_mode().  Default OPTIMIZED matches the
       original production behavior (decompose c_L by its set bits). */
    typedef struct {
        uint16_t L;
        uint16_t bit;       /* 0..PIVCO_MAX_CODE_LEN */
        uint16_t depth;     /* tree-depth of chunk root */
        uint16_t n_syms;    /* 1 << bit */
        uint16_t root_code; /* canonical code of the chunk root (depth bits) */
        int      sym_idx;   /* index into flat_items */
    } chunk_t;
    chunk_t chunks[PIVCO_MAX_SYMBOLS];   /* upper bound: one chunk per symbol */
    int n_chunks = 0;
    pivco_tree_mode_t tree_mode = pivco_huffman_get_tree_mode();

    if (tree_mode == PIVCO_TREE_MODE_NAIVE) {
        /* Every symbol is its own D=0 chunk at depth L. */
        for (int L = 1; L <= max_len; L++) {
            int c = table->sym_count[L];
            int cur = per_len_start[L];
            for (int i = 0; i < c; i++) {
                chunks[n_chunks].L       = (uint16_t)L;
                chunks[n_chunks].bit     = 0;
                chunks[n_chunks].depth   = (uint16_t)L;
                chunks[n_chunks].n_syms  = 1;
                chunks[n_chunks].sym_idx = cur + i;
                n_chunks++;
            }
        }
    } else if (tree_mode == PIVCO_TREE_MODE_FUSED) {
        /* D=1 sibling pairs first within each length, then a D=0 singleton
           for the odd-tail symbol.  Sequential reassign in the standard
           depth-sort step gives canonical Huffman codes. */
        for (int L = 1; L <= max_len; L++) {
            int c = table->sym_count[L];
            int cur = per_len_start[L];
            int n_pairs = c / 2;
            int n_singletons = c & 1;
            for (int i = 0; i < n_pairs; i++) {
                chunks[n_chunks].L       = (uint16_t)L;
                chunks[n_chunks].bit     = 1;
                chunks[n_chunks].depth   = (uint16_t)(L - 1);
                chunks[n_chunks].n_syms  = 2;
                chunks[n_chunks].sym_idx = cur;
                cur += 2;
                n_chunks++;
            }
            for (int i = 0; i < n_singletons; i++) {
                chunks[n_chunks].L       = (uint16_t)L;
                chunks[n_chunks].bit     = 0;
                chunks[n_chunks].depth   = (uint16_t)L;
                chunks[n_chunks].n_syms  = 1;
                chunks[n_chunks].sym_idx = cur;
                cur++;
                n_chunks++;
            }
        }
    } else if (tree_mode == PIVCO_TREE_MODE_CANONICAL_FLAT) {
        /* Compute canonical first_code[L] = (first_code[L-1] + c_{L-1}) << 1
           starting from min_len.  For each length, greedy-peel the largest
           2^k chunk such that the canonical start code C is 2^k-aligned and
           2^k <= remaining.  root_code = C >> k. */
        uint32_t fc[PIVCO_MAX_CODE_LEN + 2] = {0};
        uint32_t code = 0;
        int last_L = 0;
        for (int L = 1; L <= max_len; L++) {
            if (table->sym_count[L]) {
                if (last_L) code = (code + (uint32_t)table->sym_count[last_L]) << (L - last_L);
                fc[L] = code;
                last_L = L;
            }
        }
        for (int L = 1; L <= max_len; L++) {
            int c = table->sym_count[L];
            if (c == 0) continue;
            int cur = per_len_start[L];
            uint32_t C = fc[L];
            int remaining = c;
            while (remaining > 0) {
                int max_k_align = (C == 0) ? PIVCO_MAX_CODE_LEN : __builtin_ctz(C);
                int max_k_count = (remaining > 1) ? (31 - __builtin_clz((unsigned)remaining)) : 0;
                int k = max_k_align < max_k_count ? max_k_align : max_k_count;
                /* Safety: chunk depth = L-k must be >= 0; since k <= log2(remaining) <= log2(c) <= L-1
                   under any valid Kraft length distribution, this is always true. */
                int n = 1 << k;
                chunks[n_chunks].L        = (uint16_t)L;
                chunks[n_chunks].bit      = (uint16_t)k;
                chunks[n_chunks].depth    = (uint16_t)(L - k);
                chunks[n_chunks].n_syms   = (uint16_t)n;
                chunks[n_chunks].root_code = (uint16_t)(C >> k);
                chunks[n_chunks].sym_idx  = cur;
                cur += n;
                n_chunks++;
                C += (uint32_t)n;
                remaining -= n;
            }
        }
    } else {
        /* OPTIMIZED (default): original bit-decomposition of c_L. */
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

    PROF_MARK("decompose_2b");

    /* For CANONICAL_FLAT, chunks already carry canonical root_codes; assign
       symbol codes directly from them and skip the depth-sort + sequential
       reassign step (sequential reassign would clobber the canonical
       prefixes when chunks span multiple depths).  All other modes use
       the standard pipeline. */
    if (tree_mode == PIVCO_TREE_MODE_CANONICAL_FLAT) {
        for (int ci = 0; ci < n_chunks; ci++) {
            int bit = chunks[ci].bit;
            int n   = chunks[ci].n_syms;
            uint16_t root = chunks[ci].root_code;
            for (int i = 0; i < n; i++) {
                uint8_t sym = flat_items[chunks[ci].sym_idx + i].sym;
                table->code[sym] = (uint16_t)(((uint32_t)root << bit) | (uint32_t)i);
            }
        }
        PROF_MARK("assign_2d");
    } else {
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

        PROF_MARK("sort_2c");

        /* Canonical-assign codes to chunks (chunk-level Kraft sum = 1).
           Each chunk gets a code prefix of length `chunk.depth`.  Within
           the chunk, symbol i takes suffix i for i in [0, 2^bit). */
        {
            uint32_t code = 0;
            int prev_depth = 0;
            for (int ci = 0; ci < n_chunks; ci++) {
                int d = chunks[ci].depth;
                if (d > prev_depth) code <<= (d - prev_depth);
                chunks[ci].root_code = (uint16_t)code;
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

        PROF_MARK("assign_2d");
    }

    /* Populate sorted_symbols / first_sym_idx / first_code from the
       new code assignment.  These fields are not used by runtime
       decoders (only by the tree-walk pass below), but we keep them
       in length-asc order for compatibility with anyone inspecting
       the table. */
    int sorted_idx = per_len_start[max_len + 1];
    for (int i = 0; i < sorted_idx; i++)
        table->sorted_symbols[i] = flat_items[i].sym;
    for (int len = 1; len <= max_len; len++) {
        table->first_sym_idx[len] = (uint16_t)per_len_start[len];
        uint16_t min_code = 0xFFFF;
        for (int i = per_len_start[len]; i < per_len_start[len + 1]; i++) {
            uint16_t c = table->code[flat_items[i].sym];
            if (c < min_code) min_code = c;
        }
        table->first_code[len] = (min_code == 0xFFFF) ? 0 : min_code;
    }

    /* The 2^MAX_CODE_LEN flat decode table (decode_sym/decode_len) is used
     * ONLY by the traditional flat-table decoder (trad_huffman_decode*), never
     * by the production tree-walk path.  It is built on demand via
     * pivco_huffman_build_traditional_table() so the normal build -- and the
     * decode-side rebuild from code lengths -- don't pay for the 2 KB fill. */

    PROF_MARK("sorted_syms_3");

    /* Build the PIVCO tree-walk tree, one node-creating walk per chunk.
       A flat subtree (D>=2) stops at its root: the decoder reaches its 2^D
       symbols via flat_code_to_sym (filled here), so we never materialize
       the 2^D leaves nor the internal nodes below the root -- a large node
       saving on full alphabets, which also shrinks the classify and
       max_leaf_depth passes.  Singletons (D=0) and sibling pairs (D=1)
       build their leaves. */
    {
        int16_t nc = 0; /* node count */
        table->tree[0].symbol = -1;
        table->tree[0].left   = -1;
        table->tree[0].right  = -1;
        nc++;
        table->tree_root = 0;
        uint16_t pool = 0;

        for (int ci = 0; ci < n_chunks; ci++) {
            int D = chunks[ci].bit;
            int d = chunks[ci].depth;
            uint16_t rc = chunks[ci].root_code;
            int base = chunks[ci].sym_idx;

            /* Walk rc's d bits MSB-first, creating spine nodes as needed. */
            int16_t cur = 0;
            for (int b = d - 1; b >= 0; b--) {
                int16_t *child = ((rc >> b) & 1) ? &table->tree[cur].right
                                                 : &table->tree[cur].left;
                if (*child < 0) {
                    *child = nc;
                    table->tree[nc].symbol = -1;
                    table->tree[nc].left   = -1;
                    table->tree[nc].right  = -1;
                    nc++;
                }
                cur = *child;
            }

            if (D >= 2) {
                /* Flat root: mark + fill code_to_sym; no children built.
                   Leaf i of the chunk has in-subtree code i (low D bits of
                   its canonical code), so flat_code_to_sym[base+i] is its
                   i-th symbol. */
                assert(table->tree[cur].left == -1 &&
                       table->tree[cur].right == -1);
                table->flat_depth[cur]  = (uint8_t)D;
                table->flat_offset[cur] = pool;
                int n = 1 << D;
                for (int i = 0; i < n; i++)
                    table->flat_code_to_sym[pool + i] = flat_items[base + i].sym;
                pool = (uint16_t)(pool + n);
            } else if (D == 1) {
                /* Sibling pair: two leaf children (suffix 0 -> left). */
                table->tree[cur].left = nc;
                table->tree[nc].symbol = (int16_t)flat_items[base].sym;
                table->tree[nc].left = -1; table->tree[nc].right = -1; nc++;
                table->tree[cur].right = nc;
                table->tree[nc].symbol = (int16_t)flat_items[base + 1].sym;
                table->tree[nc].left = -1; table->tree[nc].right = -1; nc++;
            } else {
                /* Singleton: cur is the leaf at depth d. */
                table->tree[cur].symbol = (int16_t)flat_items[base].sym;
            }
        }
        table->tree_node_count = nc;
    }

    PROF_MARK("tree_build_4");

    /* Find the most frequent symbol (shortest code) for prefill.
       Walk the tree to find its node ID. */
    {
        /* Most-frequent symbol = shortest code = smallest-index symbol at
           min_len, which is the first entry of min_len's symbol-order run. */
        uint8_t best_sym = flat_items[per_len_start[min_len]].sym;
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

    PROF_MARK("prefill");

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

    PROF_MARK("classify");

    /* Populate code_la (left-aligned code) for the dense tree-walk
     * encoder.  Bit-d of the canonical code lives at position 15-d of
     * code_la (for d < code_len[sym]). */
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        uint8_t len = table->code_len[s];
        table->code_la[s] = len > 0
            ? (uint16_t)(table->code[s] << (16 - len))
            : 0;
    }

    PROF_MARK("code_la");

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
            /* Flat roots have no materialized children -- treat as terminal. */
            if (n->symbol < 0 && table->flat_depth[id] < 2) {
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
            } else if (table->flat_depth[id] >= 2) {
                /* All 2^D leaves sit D levels below this flat root. */
                table->max_leaf_depth[id] = table->flat_depth[id];
            } else {
                uint8_t l = table->max_leaf_depth[n->left];
                uint8_t r = table->max_leaf_depth[n->right];
                table->max_leaf_depth[id] = (uint8_t)(1 + (l > r ? l : r));
            }
        }
    }
    PROF_MARK("max_leaf_depth");

    return PIVCO_OK;
}

/* Public API: build a table from code lengths alone.  The tree is fully
 * determined by the lengths (within-tier order is symbol-value), so encoder
 * and decoder reconstruct identical tables with no extra wire info.  Goes
 * straight to build_table_finish -- no synthetic frequencies, no Huffman
 * heap (the lengths are already final). */
int pivco_huffman_build_table_from_code_lens(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    pivco_huffman_table_t *table)
{
    if (!code_lens || !table) return PIVCO_ERR_NULL;
    PROF_RESET();
    /* Clear everything except the 4 KB decode_sym/decode_len pair: those are
     * filled independently by pivco_huffman_build_traditional_table() and are
     * never read by the bulk decoder, so zeroing them here is wasted work. */
    {
        size_t skip_end = offsetof(pivco_huffman_table_t, decode_len)
                        + sizeof(table->decode_len);
        memset(table, 0, offsetof(pivco_huffman_table_t, decode_sym));
        memset((char *)table + skip_end, 0, sizeof(*table) - skip_end);
    }
    PROF_MARK("memset");

    int n_used = 0, last = 0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++)
        if (code_lens[i] > 0) { n_used++; last = i; }
    PROF_MARK("count_used");
    if (n_used == 0) return PIVCO_ERR_EMPTY;
    table->num_symbols = (uint16_t)n_used;

    if (n_used == 1) {
        build_single_symbol_table(last, table);
        return PIVCO_OK;
    }
    return build_table_finish(code_lens, table);
}

/* Fill the 2^MAX_CODE_LEN flat decode table (decode_sym/decode_len) read by
 * the traditional flat-table decoder (trad_huffman_decode*).  Call once after
 * the table is built; the production tree-walk decoder does not need it, so
 * pivco_huffman_build_table no longer fills it automatically. */
void pivco_huffman_build_traditional_table(pivco_huffman_table_t *table)
{
    if (!table) return;
    /* Defensive base fill covers any gap for incomplete codes (single sym). */
    memset(table->decode_sym, table->prefill_sym, sizeof(table->decode_sym));
    memset(table->decode_len, 1, sizeof(table->decode_len));
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        int len = table->code_len[s];
        if (len <= 0) continue;
        int shift = PIVCO_MAX_CODE_LEN - len;
        uint32_t base  = (uint32_t)table->code[s] << shift;
        uint32_t count = (uint32_t)1 << shift;
        memset(&table->decode_sym[base], s, count);
        memset(&table->decode_len[base], len, count);
    }
}
