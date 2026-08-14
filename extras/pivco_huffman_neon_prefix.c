/* pivco_neon_prefix.c — prefix-radix backend.
 *
 * Format: first M = table->min_len bits of every element's code are
 * packed as a contiguous per-element stream (LSB-first within bytes).
 * For non-flat trees (min_len < max_len), the prefix stream is followed
 * by standard 2-way PIVCO subtree bitmaps in DFS order (by bin index
 * v = 0..2^M-1), one per bin that lands at an internal node.
 *
 * Decoder:
 *   1. memset output with prefill_sym.
 *   2. Extract the M-bit prefix of every element.
 *   3. K-way radix partition (histogram → prefix-sum → place).
 *   4. For each bin v:
 *        - leaf bin == prefill_sym  → no work (memset covered it).
 *        - leaf bin, other symbol   → scatter_sym to bin's elements.
 *        - subtree bin              → hand off to pivco_neon_decode_subtree_
 *                                     starting at the bin's tree node at depth M.
 *
 * Gated by table shape — for stick trees with min_len = 1, the radix
 * phase would degenerate to a single-bit partition and lose to the
 * current neon decoder.  Callers should pick max(pivco_n, pivco_p).
 */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_neon_common.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef PIVCO_HAS_NEON

/* ---------- Bit pack / unpack helpers ----------
 * Stream layout: M-bit values packed LSB-first.  Element k occupies
 * bits [k*M, k*M + M) of the stream, little-endian within each byte. */

static inline void pack_m_bits(uint8_t *out, int k, int M, uint32_t value)
{
    size_t bit_pos = (size_t)k * (size_t)M;
    size_t byte_pos = bit_pos >> 3;
    int shift = (int)(bit_pos & 7);
    out[byte_pos    ] |= (uint8_t)(value << shift);
    if (shift + M > 8) {
        out[byte_pos + 1] |= (uint8_t)(value >> (8 - shift));
        if (shift + M > 16)
            out[byte_pos + 2] |= (uint8_t)(value >> (16 - shift));
    }
}

static inline size_t prefix_stream_bytes(int N, int M)
{
    return (size_t)(((size_t)N * (size_t)M + 7) >> 3);
}

/* ---------- Prefix-extraction fast paths (scalar unrolled) ----------
 *
 * For each M, unroll to consume an integer number of bytes per
 * iteration so the inner loop has no cross-iteration bit carry. */

static void extract_M1(const uint8_t *in, uint8_t *out, int N)
{
    for (int k = 0; k < N; k += 8) {
        uint8_t b = in[k >> 3];
        out[k    ] = (b >> 0) & 1;
        out[k + 1] = (b >> 1) & 1;
        out[k + 2] = (b >> 2) & 1;
        out[k + 3] = (b >> 3) & 1;
        out[k + 4] = (b >> 4) & 1;
        out[k + 5] = (b >> 5) & 1;
        out[k + 6] = (b >> 6) & 1;
        out[k + 7] = (b >> 7) & 1;
    }
}

static void extract_M2(const uint8_t *in, uint8_t *out, int N)
{
    for (int k = 0; k < N; k += 4) {
        uint8_t b = in[k >> 2];
        out[k    ] = (b >> 0) & 3;
        out[k + 1] = (b >> 2) & 3;
        out[k + 2] = (b >> 4) & 3;
        out[k + 3] = (b >> 6) & 3;
    }
}

static void extract_M3(const uint8_t *in, uint8_t *out, int N)
{
    /* 8 elements = 24 bits = 3 bytes */
    for (int k = 0; k < N; k += 8) {
        uint32_t w = (uint32_t)in[0]
                   | ((uint32_t)in[1] << 8)
                   | ((uint32_t)in[2] << 16);
        in += 3;
        out[k    ] = (uint8_t)((w >>  0) & 7);
        out[k + 1] = (uint8_t)((w >>  3) & 7);
        out[k + 2] = (uint8_t)((w >>  6) & 7);
        out[k + 3] = (uint8_t)((w >>  9) & 7);
        out[k + 4] = (uint8_t)((w >> 12) & 7);
        out[k + 5] = (uint8_t)((w >> 15) & 7);
        out[k + 6] = (uint8_t)((w >> 18) & 7);
        out[k + 7] = (uint8_t)((w >> 21) & 7);
    }
}

static void extract_M4(const uint8_t *in, uint8_t *out, int N)
{
    for (int k = 0; k < N; k += 2) {
        uint8_t b = in[k >> 1];
        out[k    ] = b & 0x0F;
        out[k + 1] = b >> 4;
    }
}

static void extract_M5(const uint8_t *in, uint8_t *out, int N)
{
    /* 8 elements = 40 bits = 5 bytes */
    for (int k = 0; k < N; k += 8) {
        uint64_t w = (uint64_t)in[0]
                   | ((uint64_t)in[1] << 8)
                   | ((uint64_t)in[2] << 16)
                   | ((uint64_t)in[3] << 24)
                   | ((uint64_t)in[4] << 32);
        in += 5;
        out[k    ] = (uint8_t)((w >>  0) & 0x1F);
        out[k + 1] = (uint8_t)((w >>  5) & 0x1F);
        out[k + 2] = (uint8_t)((w >> 10) & 0x1F);
        out[k + 3] = (uint8_t)((w >> 15) & 0x1F);
        out[k + 4] = (uint8_t)((w >> 20) & 0x1F);
        out[k + 5] = (uint8_t)((w >> 25) & 0x1F);
        out[k + 6] = (uint8_t)((w >> 30) & 0x1F);
        out[k + 7] = (uint8_t)((w >> 35) & 0x1F);
    }
}

static void extract_M6(const uint8_t *in, uint8_t *out, int N)
{
    /* 4 elements = 24 bits = 3 bytes */
    for (int k = 0; k < N; k += 4) {
        uint32_t w = (uint32_t)in[0]
                   | ((uint32_t)in[1] << 8)
                   | ((uint32_t)in[2] << 16);
        in += 3;
        out[k    ] = (uint8_t)((w >>  0) & 0x3F);
        out[k + 1] = (uint8_t)((w >>  6) & 0x3F);
        out[k + 2] = (uint8_t)((w >> 12) & 0x3F);
        out[k + 3] = (uint8_t)((w >> 18) & 0x3F);
    }
}

static void extract_M7(const uint8_t *in, uint8_t *out, int N)
{
    /* 8 elements = 56 bits = 7 bytes */
    for (int k = 0; k < N; k += 8) {
        uint64_t w = (uint64_t)in[0]
                   | ((uint64_t)in[1] << 8)
                   | ((uint64_t)in[2] << 16)
                   | ((uint64_t)in[3] << 24)
                   | ((uint64_t)in[4] << 32)
                   | ((uint64_t)in[5] << 40)
                   | ((uint64_t)in[6] << 48);
        in += 7;
        out[k    ] = (uint8_t)((w >>  0) & 0x7F);
        out[k + 1] = (uint8_t)((w >>  7) & 0x7F);
        out[k + 2] = (uint8_t)((w >> 14) & 0x7F);
        out[k + 3] = (uint8_t)((w >> 21) & 0x7F);
        out[k + 4] = (uint8_t)((w >> 28) & 0x7F);
        out[k + 5] = (uint8_t)((w >> 35) & 0x7F);
        out[k + 6] = (uint8_t)((w >> 42) & 0x7F);
        out[k + 7] = (uint8_t)((w >> 49) & 0x7F);
    }
}

static void extract_M8(const uint8_t *in, uint8_t *out, int N)
{
    memcpy(out, in, (size_t)N);
}

typedef void (*extract_fn)(const uint8_t *, uint8_t *, int);

static extract_fn pick_extract(int M)
{
    switch (M) {
    case 1: return extract_M1;
    case 2: return extract_M2;
    case 3: return extract_M3;
    case 4: return extract_M4;
    case 5: return extract_M5;
    case 6: return extract_M6;
    case 7: return extract_M7;
    case 8: return extract_M8;
    default: return NULL;
    }
}

/* ---------- Per-bin metadata precomputation ----------
 *
 * For each of K = 2^M possible M-bit prefixes v, walk M bits from the
 * tree root.  Record whether v lands on a leaf or an internal node, the
 * leaf symbol if applicable, and the tree node id (needed for subtree
 * recursion). */
typedef struct {
    uint8_t is_leaf;      /* 1 if bin is a leaf at depth M */
    uint8_t leaf_sym;     /* valid iff is_leaf */
    int16_t node_id;      /* tree node id landed at after M steps */
} bin_info_t;

static void precompute_bins(const pivco_table_t *t, int M,
                             bin_info_t *bins)
{
    int K = 1 << M;
    for (int v = 0; v < K; v++) {
        int16_t node_id = t->tree_root;
        for (int b = M - 1; b >= 0; b--) {
            const pivco_tree_node_t *n = &t->tree[node_id];
            if (n->symbol >= 0) break;     /* shouldn't happen since M <= min_len */
            int bit = (v >> b) & 1;
            node_id = bit ? n->right : n->left;
        }
        const pivco_tree_node_t *n = &t->tree[node_id];
        bins[v].node_id = node_id;
        if (n->symbol >= 0) {
            bins[v].is_leaf = 1;
            bins[v].leaf_sym = (uint8_t)n->symbol;
        } else {
            bins[v].is_leaf = 0;
            bins[v].leaf_sym = 0;
        }
    }
}

/* ---------- Scatter (duplicated from neon backend) ---------- */

#include <arm_neon.h>

static inline void scatter_sym(uint8_t *symbols,
                                const uint16_t *indices, int n, uint8_t sym)
{
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        uint16x8_t idx = vld1q_u16(indices + j);
        symbols[vgetq_lane_u16(idx, 0)] = sym;
        symbols[vgetq_lane_u16(idx, 1)] = sym;
        symbols[vgetq_lane_u16(idx, 2)] = sym;
        symbols[vgetq_lane_u16(idx, 3)] = sym;
        symbols[vgetq_lane_u16(idx, 4)] = sym;
        symbols[vgetq_lane_u16(idx, 5)] = sym;
        symbols[vgetq_lane_u16(idx, 6)] = sym;
        symbols[vgetq_lane_u16(idx, 7)] = sym;
    }
    for (; j < n; j++)
        symbols[indices[j]] = sym;
}

/* ---------- Encode ---------- */

int pivco_encode_neon_prefix(const uint8_t *symbols,
                                      const pivco_table_t *table,
                                      uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    /* The subtree encoder (delegated via pivco_neon_encode_subtree_) relies
     * on the NEON compress_tab/compress_popcnt globals.  Make sure they're
     * initialised — no-op after the first call. */
    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    const int M = table->min_len;
    if (M < 1 || M > 8) return PIVCO_ERR_CORRUPT;
    const int K = 1 << M;

    /* Prefix stream = first N*M bits. */
    size_t prefix_bytes = prefix_stream_bytes(N, M);
    memset(out, 0, prefix_bytes);

    /* Per-position codes / lens for the subtree encoder.  Heap-allocated
     * so their lifetime isn't confused by the compiler's inline/scope
     * analysis with other locals further down the function. */
    uint16_t *codes_pos = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    uint8_t  *lens_pos  = (uint8_t *) malloc((size_t)N);
    if (!codes_pos || !lens_pos) { free(codes_pos); free(lens_pos); return PIVCO_ERR_OVERFLOW; }
    for (int k = 0; k < N; k++) {
        uint8_t s = symbols[k];
        codes_pos[k] = table->code[s];
        lens_pos[k]  = table->code_len[s];
    }

    /* Pack M-bit prefixes into the stream; record per-stream histogram
     * so we can do 8-way parallel bucket placement to match the decoder.
     * The decoder's 8-way bucket produces an order-within-bin where
     * stream s's contributions appear together; the encoder must emit
     * subtree bitmaps in the SAME order so bit positions match at decode
     * time. */
    int bc[8][1 << 8] = {{0}};
    {
        int k = 0;
        for (; k + 8 <= N; k += 8) {
            int L0 = lens_pos[k];     uint32_t p0 = (uint32_t)codes_pos[k    ] >> (L0 - M);
            int L1 = lens_pos[k + 1]; uint32_t p1 = (uint32_t)codes_pos[k + 1] >> (L1 - M);
            int L2 = lens_pos[k + 2]; uint32_t p2 = (uint32_t)codes_pos[k + 2] >> (L2 - M);
            int L3 = lens_pos[k + 3]; uint32_t p3 = (uint32_t)codes_pos[k + 3] >> (L3 - M);
            int L4 = lens_pos[k + 4]; uint32_t p4 = (uint32_t)codes_pos[k + 4] >> (L4 - M);
            int L5 = lens_pos[k + 5]; uint32_t p5 = (uint32_t)codes_pos[k + 5] >> (L5 - M);
            int L6 = lens_pos[k + 6]; uint32_t p6 = (uint32_t)codes_pos[k + 6] >> (L6 - M);
            int L7 = lens_pos[k + 7]; uint32_t p7 = (uint32_t)codes_pos[k + 7] >> (L7 - M);
            pack_m_bits(out, k    , M, p0);
            pack_m_bits(out, k + 1, M, p1);
            pack_m_bits(out, k + 2, M, p2);
            pack_m_bits(out, k + 3, M, p3);
            pack_m_bits(out, k + 4, M, p4);
            pack_m_bits(out, k + 5, M, p5);
            pack_m_bits(out, k + 6, M, p6);
            pack_m_bits(out, k + 7, M, p7);
            bc[0][p0]++; bc[1][p1]++; bc[2][p2]++; bc[3][p3]++;
            bc[4][p4]++; bc[5][p5]++; bc[6][p6]++; bc[7][p7]++;
        }
        for (; k < N; k++) {
            int L = lens_pos[k];
            uint32_t prefix = (uint32_t)codes_pos[k] >> (L - M);
            pack_m_bits(out, k, M, prefix);
            bc[0][prefix]++;
        }
    }
    int bin_count[1 << 8];
    for (int v = 0; v < K; v++)
        bin_count[v] = bc[0][v] + bc[1][v] + bc[2][v] + bc[3][v]
                     + bc[4][v] + bc[5][v] + bc[6][v] + bc[7][v];

    if (M == table->max_len) {
        /* Flat: no subtree bitmaps. */
        *out_len = prefix_bytes;
        free(codes_pos); free(lens_pos);
        return PIVCO_OK;
    }

    int bin_offset[(1 << 8) + 1];
    bin_offset[0] = 0;
    for (int v = 0; v < K; v++) bin_offset[v+1] = bin_offset[v] + bin_count[v];

    /* bin_elements[] holds original positions sorted by bin.  Populate
     * via 8-way parallel placement (matching decoder order).
     * +8 slots of slack — encode_node_neon's 16-byte TBL stores can
     * write up to 7 uint16 past the end of a bin's segment. */
    uint16_t bin_elements[PIVCO_BLOCK_SIZE + 8];
    int place[8][1 << 8];
    for (int v = 0; v < K; v++) {
        place[0][v] = bin_offset[v];
        place[1][v] = place[0][v] + bc[0][v];
        place[2][v] = place[1][v] + bc[1][v];
        place[3][v] = place[2][v] + bc[2][v];
        place[4][v] = place[3][v] + bc[3][v];
        place[5][v] = place[4][v] + bc[4][v];
        place[6][v] = place[5][v] + bc[5][v];
        place[7][v] = place[6][v] + bc[6][v];
    }
    {
        int k = 0;
        for (; k + 8 <= N; k += 8) {
            int L0 = lens_pos[k];     int v0 = (int)((uint32_t)codes_pos[k    ] >> (L0 - M));
            int L1 = lens_pos[k + 1]; int v1 = (int)((uint32_t)codes_pos[k + 1] >> (L1 - M));
            int L2 = lens_pos[k + 2]; int v2 = (int)((uint32_t)codes_pos[k + 2] >> (L2 - M));
            int L3 = lens_pos[k + 3]; int v3 = (int)((uint32_t)codes_pos[k + 3] >> (L3 - M));
            int L4 = lens_pos[k + 4]; int v4 = (int)((uint32_t)codes_pos[k + 4] >> (L4 - M));
            int L5 = lens_pos[k + 5]; int v5 = (int)((uint32_t)codes_pos[k + 5] >> (L5 - M));
            int L6 = lens_pos[k + 6]; int v6 = (int)((uint32_t)codes_pos[k + 6] >> (L6 - M));
            int L7 = lens_pos[k + 7]; int v7 = (int)((uint32_t)codes_pos[k + 7] >> (L7 - M));
            bin_elements[place[0][v0]++] = (uint16_t)(k    );
            bin_elements[place[1][v1]++] = (uint16_t)(k + 1);
            bin_elements[place[2][v2]++] = (uint16_t)(k + 2);
            bin_elements[place[3][v3]++] = (uint16_t)(k + 3);
            bin_elements[place[4][v4]++] = (uint16_t)(k + 4);
            bin_elements[place[5][v5]++] = (uint16_t)(k + 5);
            bin_elements[place[6][v6]++] = (uint16_t)(k + 6);
            bin_elements[place[7][v7]++] = (uint16_t)(k + 7);
        }
        for (; k < N; k++) {
            int L = lens_pos[k];
            int v = (int)((uint32_t)codes_pos[k] >> (L - M));
            bin_elements[place[0][v]++] = (uint16_t)k;
        }
    }

    /* Per-bin metadata. */
    bin_info_t bins[1 << 8];
    precompute_bins(table, M, bins);

    uint8_t *ptr = out + prefix_bytes;
    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    uint16_t scratch_indices[PIVCO_BLOCK_SIZE + 8];

    for (int v = 0; v < K; v++) {
        if (bins[v].is_leaf) continue;
        int n = bin_count[v];
        if (n == 0) continue;
        /* Copy bin's elements into a private scratch so the subtree
         * encoder's in-place partitioning doesn't touch other bins. */
        memcpy(scratch_indices, bin_elements + bin_offset[v],
               (size_t)n * sizeof(uint16_t));
        pivco_neon_encode_subtree_(table, bins[v].node_id,
                                   scratch_indices, n,
                                   /*depth=*/M,
                                   codes_pos, lens_pos,
                                   &ptr, tmp);
    }
    *out_len = (size_t)(ptr - out);
    free(codes_pos); free(lens_pos);
    return PIVCO_OK;
}

/* ---------- Decode ---------- */

int pivco_decode_neon_prefix(const uint8_t *in, size_t in_len,
                                      const pivco_table_t *table,
                                      uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    const int M = table->min_len;
    if (M < 1 || M > 8) return PIVCO_ERR_CORRUPT;
    const int K = 1 << M;

    /* ---------- Flat-tree fast path (min_len == max_len) ----------
     * Every bin is a leaf, so this collapses to a direct permutation:
     *   symbols[k] = code_to_sym[prefix[k]]
     * — no histogram, no bucketing, no subtree recursion. */
    if (M == table->max_len) {
        uint8_t code_to_sym[1 << 8];
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            if (table->code_len[s] == M)
                code_to_sym[table->code[s]] = (uint8_t)s;
        }
        if (M == 8) {
            for (int k = 0; k < N; k++) symbols[k] = code_to_sym[in[k]];
            *consumed = (size_t)N;
            return PIVCO_OK;
        }
        if (M == 4) {
            for (int k = 0; k < N; k += 2) {
                uint8_t b = in[k >> 1];
                symbols[k    ] = code_to_sym[b & 0x0F];
                symbols[k + 1] = code_to_sym[b >> 4];
            }
            *consumed = (size_t)(N >> 1);
            return PIVCO_OK;
        }
        if (M == 2) {
            for (int k = 0; k < N; k += 4) {
                uint8_t b = in[k >> 2];
                symbols[k    ] = code_to_sym[(b     ) & 3];
                symbols[k + 1] = code_to_sym[(b >> 2) & 3];
                symbols[k + 2] = code_to_sym[(b >> 4) & 3];
                symbols[k + 3] = code_to_sym[(b >> 6) & 3];
            }
            *consumed = (size_t)(N >> 2);
            return PIVCO_OK;
        }
        if (M == 1) {
            for (int k = 0; k < N; k += 8) {
                uint8_t b = in[k >> 3];
                symbols[k    ] = code_to_sym[(b     ) & 1];
                symbols[k + 1] = code_to_sym[(b >> 1) & 1];
                symbols[k + 2] = code_to_sym[(b >> 2) & 1];
                symbols[k + 3] = code_to_sym[(b >> 3) & 1];
                symbols[k + 4] = code_to_sym[(b >> 4) & 1];
                symbols[k + 5] = code_to_sym[(b >> 5) & 1];
                symbols[k + 6] = code_to_sym[(b >> 6) & 1];
                symbols[k + 7] = code_to_sym[(b >> 7) & 1];
            }
            *consumed = (size_t)(N >> 3);
            return PIVCO_OK;
        }
        /* Generic M ∈ {3, 5, 6, 7} flat path. */
        uint8_t prefix_buf[PIVCO_BLOCK_SIZE];
        extract_fn ef0 = pick_extract(M);
        ef0(in, prefix_buf, N);
        for (int k = 0; k < N; k++) symbols[k] = code_to_sym[prefix_buf[k]];
        *consumed = prefix_stream_bytes(N, M);
        return PIVCO_OK;
    }

    /* ---------- Non-flat path ---------- */

    /* Prefill output with the most frequent symbol — same trick as neon. */
    memset(symbols, table->prefill_sym, (size_t)N);

    /* Phase 1: extract the M-bit prefix per element. */
    uint8_t prefix[PIVCO_BLOCK_SIZE];
    extract_fn ef = pick_extract(M);
    if (!ef) return PIVCO_ERR_CORRUPT;
    ef(in, prefix, N);

    const uint8_t *ptr = in + prefix_stream_bytes(N, M);

    /* Phase 2: histogram — 8-way parallel to break the serial dep chain
     * that `bin_count[prefix[k]]++` has when prefix values cluster.
     * Eight independent counter arrays, each updated by one of every
     * eight elements; summed at the end.  4 streams weren't enough to
     * hide the load→increment→store dependency on clustered inputs;
     * 8 streams provide enough instruction-level parallelism for OoO
     * to keep the memory ports fed. */
    int bc[8][1 << 8] = {{0}};
    {
        int k = 0;
        for (; k + 8 <= N; k += 8) {
            bc[0][prefix[k    ]]++;
            bc[1][prefix[k + 1]]++;
            bc[2][prefix[k + 2]]++;
            bc[3][prefix[k + 3]]++;
            bc[4][prefix[k + 4]]++;
            bc[5][prefix[k + 5]]++;
            bc[6][prefix[k + 6]]++;
            bc[7][prefix[k + 7]]++;
        }
        for (; k < N; k++) bc[0][prefix[k]]++;
    }
    int bin_count[1 << 8];
    for (int v = 0; v < K; v++)
        bin_count[v] = bc[0][v] + bc[1][v] + bc[2][v] + bc[3][v]
                     + bc[4][v] + bc[5][v] + bc[6][v] + bc[7][v];

    /* Phase 3: prefix-sum for offsets. */
    int bin_offset[(1 << 8) + 1];
    bin_offset[0] = 0;
    for (int v = 0; v < K; v++) bin_offset[v+1] = bin_offset[v] + bin_count[v];

    /* Phase 4: bucket element ids by bin — 8-way parallel placement.
     * Each stream s ∈ {0..7} places elements at positions k where
     * k%8 == s, using its own place[s][] offset per bin.  place[s][v]
     * starts at bin_offset[v] + sum_{s' < s} bc[s'][v] so streams don't
     * overlap.  Order within a bin is interleaved by stream rather than
     * pure k-order — irrelevant for subtree decode correctness.
     * +8 slack on bin_elements for the subtree partition's overflow.
     *
     * Earlier analysis claimed phase 4 was memory-port-limited at 4-way;
     * that turned out to be wrong — the real bottleneck is the load→add→
     * store dependency chain on place[s][v], and 8 streams are needed
     * to hide its latency for clustered distributions.  Empirical:
     * 8-way gives measurable gains over 4-way on english/zipfian/proba14. */
    uint16_t bin_elements[PIVCO_BLOCK_SIZE + 8];
    int place[8][1 << 8];
    for (int v = 0; v < K; v++) {
        place[0][v] = bin_offset[v];
        place[1][v] = place[0][v] + bc[0][v];
        place[2][v] = place[1][v] + bc[1][v];
        place[3][v] = place[2][v] + bc[2][v];
        place[4][v] = place[3][v] + bc[3][v];
        place[5][v] = place[4][v] + bc[4][v];
        place[6][v] = place[5][v] + bc[5][v];
        place[7][v] = place[6][v] + bc[6][v];
    }
    {
        int k = 0;
        for (; k + 8 <= N; k += 8) {
            bin_elements[place[0][prefix[k    ]]++] = (uint16_t)(k    );
            bin_elements[place[1][prefix[k + 1]]++] = (uint16_t)(k + 1);
            bin_elements[place[2][prefix[k + 2]]++] = (uint16_t)(k + 2);
            bin_elements[place[3][prefix[k + 3]]++] = (uint16_t)(k + 3);
            bin_elements[place[4][prefix[k + 4]]++] = (uint16_t)(k + 4);
            bin_elements[place[5][prefix[k + 5]]++] = (uint16_t)(k + 5);
            bin_elements[place[6][prefix[k + 6]]++] = (uint16_t)(k + 6);
            bin_elements[place[7][prefix[k + 7]]++] = (uint16_t)(k + 7);
        }
        for (; k < N; k++)
            bin_elements[place[0][prefix[k]]++] = (uint16_t)k;
    }

    /* Phase 5: per-bin metadata + dispatch.  Each subtree decode is
     * handed a scratch copy of the bin's indices to prevent in-place
     * partitioning from bleeding into neighbouring bins. */
    bin_info_t bins[1 << 8];
    precompute_bins(table, M, bins);
    int16_t skip_node = table->prefill_node;

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    uint16_t scratch_indices[PIVCO_BLOCK_SIZE + 8];
    for (int v = 0; v < K; v++) {
        int n = bin_count[v];
        if (n == 0) continue;
        if (bins[v].is_leaf) {
            if (bins[v].node_id == skip_node) continue;   /* prefill covered */
            scatter_sym(symbols, bin_elements + bin_offset[v], n,
                        bins[v].leaf_sym);
        } else {
            memcpy(scratch_indices, bin_elements + bin_offset[v],
                   (size_t)n * sizeof(uint16_t));
            pivco_neon_decode_subtree_(table, bins[v].node_id,
                                       scratch_indices, n,
                                       symbols, &ptr, tmp, skip_node);
        }
    }

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_NEON */
