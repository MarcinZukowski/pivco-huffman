#include "pivco_huffman.h"
#include <string.h>
#include <stdlib.h>

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
        /* Fill decode table */
        memset(table->decode_sym, (uint8_t)sym, sizeof(table->decode_sym));
        memset(table->decode_len, 1, sizeof(table->decode_len));
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

    /* Sort symbols by (length, value) for canonical assignment */
    int sorted_idx = 0;
    for (int len = 1; len <= max_len; len++) {
        table->first_sym_idx[len] = (uint16_t)sorted_idx;
        for (int sym = 0; sym < PIVCO_MAX_SYMBOLS; sym++) {
            if (lengths[sym] == len) {
                table->sorted_symbols[sorted_idx++] = (uint8_t)sym;
            }
        }
    }

    /* Assign canonical codes */
    uint16_t code = 0;
    for (int len = 1; len <= max_len; len++) {
        table->first_code[len] = code;
        int idx = table->first_sym_idx[len];
        for (int i = 0; i < table->sym_count[len]; i++) {
            int sym = table->sorted_symbols[idx + i];
            table->code[sym] = code;
            code++;
        }
        code <<= 1;
    }

    /* Build flat decode table (2^MAX_CODE_LEN entries) */
    for (int len = 1; len <= max_len; len++) {
        int idx = table->first_sym_idx[len];
        for (int i = 0; i < table->sym_count[len]; i++) {
            uint8_t sym = table->sorted_symbols[idx + i];
            uint16_t c = table->code[sym];
            /* Left-align code to MAX_CODE_LEN bits */
            int shift = PIVCO_MAX_CODE_LEN - len;
            uint32_t base = (uint32_t)c << shift;
            uint32_t count = (uint32_t)1 << shift;
            for (uint32_t j = 0; j < count; j++) {
                table->decode_sym[base + j] = sym;
                table->decode_len[base + j] = (uint8_t)len;
            }
        }
    }

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

    return PIVCO_OK;
}
