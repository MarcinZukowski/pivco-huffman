/* Bottom-up PIVCO-Huffman decoder (x86: SSE4.1 + optional AVX-512 VBMI2).
 *
 * See src/pivco_huffman_bu_neon.c for the algorithm description.  This
 * file is the x86 port.  Same recursive DFS structure, same expand_tab
 * layout — only the SIMD primitives differ.
 *
 * SSE 8-byte chunk per iter:
 *   pshufb on _mm_unpacklo_epi64(L8, R8) with expand_tab[mask] -> 8 bytes.
 *
 * AVX-512 VBMI2 (when available): 64-byte chunks via two
 *   _mm512_maskz_expandloadu_epi8 calls (one with mask, one with ~mask)
 *   ORed together.  ~0.023 ns/byte on Xeon Ice Lake+ per microbench.
 *
 * For now we use AVX-512 only for the LARGE merges (K >= 64) and fall
 * back to SSE for smaller merges; the per-call overhead of vpexpandb's
 * mask setup is amortised only when K is reasonably large. */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
#include <string.h>

#ifdef PIVCO_HAS_SSE4
#include <smmintrin.h>  /* SSE4.1 */
#include <immintrin.h>  /* AVX/AVX-512 umbrella; harmless on SSE-only builds */

/* Popcount K bits from bm using POPCNT (BMI1/SSE4.2 era; -mpopcnt
 * enabled for both AVX-512 and SSE-only builds in CMakeLists).
 * 4-way unrolled 64-bit loop: each independent POPCNT lands on its own
 * exec unit (Zen3 has 4 popcnt ports, Xeon ICX has 1 — both benefit
 * from 4 independent reductions accumulated into separate ints, which
 * the compiler then sums at the end).  Mirrors the structure of the
 * NEON 64-byte chunk path in pivco_huffman_bu_neon.c. */
static inline int popcount_K_right_x86(const uint8_t *bm, int nbytes, int K) {
    (void)nbytes;   /* derivable from K; kept for API stability */
    PROF_TIC();
    int full_bytes = K >> 3;
    int partial_bits = K & 7;
    int b = 0;

    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (; b + 32 <= full_bytes; b += 32) {
        uint64_t v0, v1, v2, v3;
        memcpy(&v0, bm + b,      8);
        memcpy(&v1, bm + b + 8,  8);
        memcpy(&v2, bm + b + 16, 8);
        memcpy(&v3, bm + b + 24, 8);
        a0 += __builtin_popcountll(v0);
        a1 += __builtin_popcountll(v1);
        a2 += __builtin_popcountll(v2);
        a3 += __builtin_popcountll(v3);
    }
    int K_right = (int)(a0 + a1 + a2 + a3);

    /* 8-byte mop-up (1..3 leftover 8-byte groups). */
    for (; b + 8 <= full_bytes; b += 8) {
        uint64_t v;
        memcpy(&v, bm + b, 8);
        K_right += __builtin_popcountll(v);
    }
    /* Byte tail (0..7 leftover full bytes). */
    for (; b < full_bytes; b++) {
        K_right += __builtin_popcount(bm[b]);
    }
    /* Optional partial byte (K & 7 valid bits). */
    if (partial_bits) {
        uint8_t valid_mask = (uint8_t)((1u << partial_bits) - 1);
        K_right += __builtin_popcount(bm[full_bytes] & valid_mask);
    }
    PROF_TOC(PROF_BU_POPCOUNT_K, K);
    return K_right;
}

/* ---------- expand_tab: per-mask-byte shuffle pattern ---------- */
static uint8_t expand_tab[256][8] __attribute__((aligned(32)));
static uint8_t expand_popcnt[256]  __attribute__((aligned(64)));
static int expand_table_ready = 0;

static void init_expand_table_x86(void) {
    if (expand_table_ready) return;
    for (int m = 0; m < 256; m++) {
        int n_zeros = 0, n_ones = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) {
                expand_tab[m][k] = (uint8_t)(8 + n_ones);
                n_ones++;
            } else {
                expand_tab[m][k] = (uint8_t)n_zeros;
                n_zeros++;
            }
        }
        expand_popcnt[m] = (uint8_t)n_ones;
    }
    expand_table_ready = 1;
}

/* ---------- SSE 8-byte tree_merge ---------- */
static inline void tree_merge(const uint8_t *bm, int K,
                               const uint8_t *left,
                               const uint8_t *right,
                               uint8_t *out) {
    PROF_TIC();
    int lc = 0, rc = 0;
    int j = 0;
#ifdef __AVX512VBMI2__
    /* Fast path: process 64-byte chunks via vpexpandb (single
     * load+expand per side, OR'd together).  Microbench shows
     * ~0.023 ns/byte on Xeon Ice Lake+. */
    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __mmask64 nm = ~m;
        __m512i L = _mm512_maskz_expandloadu_epi8(nm, left + lc);
        __m512i R = _mm512_maskz_expandloadu_epi8(m,  right + rc);
        __m512i o = _mm512_or_si512(L, R);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        int nr = __builtin_popcountll(mask);
        rc += nr; lc += (64 - nr);
    }
#endif
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(L, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        int nr = expand_popcnt[m];
        rc += nr; lc += (8 - nr);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
    PROF_TOC(PROF_BU_TREE_MERGE, K);
}

/* ---------- broadcast-left (constant left input) ---------- */
static inline void tree_merge_bcast_left(const uint8_t *bm, int K,
                                          uint8_t left_sym,
                                          const uint8_t *right,
                                          uint8_t *out) {
    PROF_TIC();
    int rc = 0;
    int j = 0;
    __m128i Lbcast8 = _mm_set1_epi8((char)left_sym);
#ifdef __AVX512VBMI2__
    __m512i Lbcast64 = _mm512_set1_epi8((char)left_sym);
    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __mmask64 nm = ~m;
        /* For left bytes: just broadcast at 0-positions (vpexpandb of
         * a broadcast register works, but mask-blending is simpler). */
        __m512i R = _mm512_maskz_expandloadu_epi8(m, right + rc);
        __m512i o = _mm512_mask_mov_epi8(Lbcast64, m, R);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        rc += __builtin_popcountll(mask);
    }
#endif
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(Lbcast8, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        rc += expand_popcnt[m];
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left_sym;
    }
    PROF_TOC(PROF_BU_TREE_MERGE_BCAST_LEFT, K);
}

/* ---------- broadcast-right (constant right input) ---------- */
static inline void tree_merge_bcast_right(const uint8_t *bm, int K,
                                           const uint8_t *left,
                                           uint8_t right_sym,
                                           uint8_t *out) {
    PROF_TIC();
    int lc = 0;
    int j = 0;
    __m128i Rbcast8 = _mm_set1_epi8((char)right_sym);
#ifdef __AVX512VBMI2__
    __m512i Rbcast64 = _mm512_set1_epi8((char)right_sym);
    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __mmask64 nm = ~m;
        __m512i L = _mm512_maskz_expandloadu_epi8(nm, left + lc);
        __m512i o = _mm512_mask_mov_epi8(L, m, Rbcast64);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        lc += 64 - __builtin_popcountll(mask);
    }
#endif
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i both = _mm_unpacklo_epi64(L, Rbcast8);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        lc += (8 - expand_popcnt[m]);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left[lc++];
    }
    PROF_TOC(PROF_BU_TREE_MERGE_BCAST_RIGHT, K);
}

/* ---------- both-leaves (both inputs constants) ----------
 * vpblendvb-style: for each bit in mask, output is right_sym or left_sym.
 * Mirrors the existing x86 BOTH_LEAVES sequential-store code in
 * src/pivco_huffman_x86.c (pshufb-broadcast of bitmap bits + pblendvb). */
static inline void merge_both_const(const uint8_t *bm, int K,
                                     uint8_t left_sym, uint8_t right_sym,
                                     uint8_t *out) {
    PROF_TIC();
    __m128i vsym0 = _mm_set1_epi8((char)left_sym);
    __m128i vsym1 = _mm_set1_epi8((char)right_sym);
    __m128i bits  = _mm_setr_epi8(1,2,4,8,16,32,64,(char)128,
                                   1,2,4,8,16,32,64,(char)128);
    __m128i shuf  = _mm_setr_epi8(0,0,0,0,0,0,0,0,
                                   1,1,1,1,1,1,1,1);
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        __m128i bm_pair = _mm_cvtsi32_si128(*(const uint16_t *)(bm + (j >> 3)));
        __m128i bm_dup  = _mm_shuffle_epi8(bm_pair, shuf);
        __m128i masked  = _mm_and_si128(bm_dup, bits);
        __m128i mask8   = _mm_cmpeq_epi8(masked, bits);
        __m128i o       = _mm_blendv_epi8(vsym0, vsym1, mask8);
        _mm_storeu_si128((__m128i *)(out + j), o);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left_sym;
    }
    PROF_TOC(PROF_BU_MERGE_BOTH_CONST, K);
}

/* ---------- D-bit flat decode ----------
 * On AVX-512 VBMI2 hosts route through the AVX-512 unpack (D=5/6 have
 * dedicated fast paths there).  Without AVX-512 fall back to the SSE
 * D=4 vectorised path.  The bu_x86 backend is the only x86 caller of
 * these wrappers; the TD decoders inline them directly. */
extern void pivco_huffman_flat_decode_direct_x86_(uint8_t *symbols, int n,
                                                   const uint8_t *bm, int D,
                                                   const uint8_t *c2s);
#ifdef PIVCO_HAS_AVX512
extern void pivco_huffman_flat_decode_direct_avx512_(uint8_t *symbols, int n,
                                                      const uint8_t *bm, int D,
                                                      const uint8_t *c2s);
#endif
static inline void flat_decode_to_buffer(uint8_t *out, int n,
                                          const uint8_t *bm, int D,
                                          const uint8_t *c2s) {
    PROF_TIC();
#ifdef PIVCO_HAS_AVX512
    pivco_huffman_flat_decode_direct_avx512_(out, n, bm, D, c2s);
#else
    pivco_huffman_flat_decode_direct_x86_(out, n, bm, D, c2s);
#endif
    PROF_TOC(PROF_BU_FLAT_DECODE, n);
}

/* ---------- Recursive bottom-up decode ---------- */
static void decode_subtree_bu(const pivco_huffman_table_t *table,
                               int16_t node_id, int K,
                               uint8_t *out_buf,
                               const uint8_t **in_ptr,
                               uint8_t *scratch_top)
{
    if (K == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];

    switch ((pivco_node_type_t)table->node_type[node_id]) {

    case PIVCO_NODE_SKIP:
        { PROF_TIC();
          memset(out_buf, table->prefill_sym, (size_t)K);
          PROF_TOC(PROF_BU_LEAF_MEMSET, K); }
        return;

    case PIVCO_NODE_LEAF:
        { PROF_TIC();
          memset(out_buf, (uint8_t)node->symbol, (size_t)K);
          PROF_TOC(PROF_BU_LEAF_MEMSET, K); }
        return;

    case PIVCO_NODE_INTERNAL_FLAT: {
        int D = table->flat_depth[node_id];
        int total_bytes = (K * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[node_id]];
        flat_decode_to_buffer(out_buf, K, bm, D, c2s);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        const pivco_tree_node_t *left_child  = &table->tree[node->left];
        const pivco_tree_node_t *right_child = &table->tree[node->right];
        merge_both_const(bm, K,
                          (uint8_t)left_child->symbol,
                          (uint8_t)right_child->symbol,
                          out_buf);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        if (table->node_type[node->right] == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const(bm, K, table->prefill_sym,
                              (uint8_t)table->tree[node->right].symbol,
                              out_buf);
            return;
        }
        int K_right = popcount_K_right_x86(bm, nbytes, K);
        uint8_t *right_buf = scratch_top;
        decode_subtree_bu(table, node->right, K_right,
                          right_buf, in_ptr, scratch_top + K_right);
        tree_merge_bcast_left(bm, K, table->prefill_sym, right_buf, out_buf);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        if (table->node_type[node->left] == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const(bm, K,
                              (uint8_t)table->tree[node->left].symbol,
                              table->prefill_sym, out_buf);
            return;
        }
        int K_right = popcount_K_right_x86(bm, nbytes, K);
        int K_left = K - K_right;
        uint8_t *left_buf = scratch_top;
        decode_subtree_bu(table, node->left, K_left,
                          left_buf, in_ptr, scratch_top + K_left);
        tree_merge_bcast_right(bm, K, left_buf, table->prefill_sym, out_buf);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        int nbytes = bitmap_bytes(K);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        int left_kind  = table->node_type[node->left];
        int right_kind = table->node_type[node->right];

        if (left_kind == (uint8_t)PIVCO_NODE_LEAF
            && right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            merge_both_const(bm, K,
                              (uint8_t)table->tree[node->left].symbol,
                              (uint8_t)table->tree[node->right].symbol,
                              out_buf);
            return;
        }
        if (left_kind == (uint8_t)PIVCO_NODE_LEAF) {
            int K_right = popcount_K_right_x86(bm, nbytes, K);
            uint8_t *right_buf = scratch_top;
            decode_subtree_bu(table, node->right, K_right,
                              right_buf, in_ptr, scratch_top + K_right);
            tree_merge_bcast_left(bm, K,
                                   (uint8_t)table->tree[node->left].symbol,
                                   right_buf, out_buf);
            return;
        }
        if (right_kind == (uint8_t)PIVCO_NODE_LEAF) {
            int K_right = popcount_K_right_x86(bm, nbytes, K);
            int K_left = K - K_right;
            uint8_t *left_buf = scratch_top;
            decode_subtree_bu(table, node->left, K_left,
                              left_buf, in_ptr, scratch_top + K_left);
            tree_merge_bcast_right(bm, K, left_buf,
                                    (uint8_t)table->tree[node->right].symbol,
                                    out_buf);
            return;
        }

        int K_right = popcount_K_right_x86(bm, nbytes, K);
        int K_left = K - K_right;
        uint8_t *left_buf  = scratch_top;
        uint8_t *right_buf = scratch_top + K_left;
        uint8_t *new_scratch_top = scratch_top + K;
        decode_subtree_bu(table, node->left,  K_left,
                          left_buf,  in_ptr, new_scratch_top);
        decode_subtree_bu(table, node->right, K_right,
                          right_buf, in_ptr, new_scratch_top);
        tree_merge(bm, K, left_buf, right_buf, out_buf);
        return;
    }
    }
}

/* ---------- Entry point ---------- */
int pivco_huffman_decode_bu_x86(const uint8_t *in, size_t in_len,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    init_expand_table_x86();

    (void)in_len;
    const uint8_t *ptr = in;
    const int N = PIVCO_BLOCK_SIZE;
    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Fast path: root is BOTH_LEAVES. */
    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_BOTH_LEAVES) {
        int nbytes = bitmap_bytes(N);
        const uint8_t *bm = ptr;
        ptr += nbytes;
        const pivco_tree_node_t *left_child  = &table->tree[root->left];
        const pivco_tree_node_t *right_child = &table->tree[root->right];
        merge_both_const(bm, N,
                          (uint8_t)left_child->symbol,
                          (uint8_t)right_child->symbol,
                          symbols);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    static uint8_t scratch[3 * PIVCO_BLOCK_SIZE + 64]
        __attribute__((aligned(64)));

    decode_subtree_bu(table, table->tree_root, N,
                      symbols, &ptr, scratch);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_SSE4 */
