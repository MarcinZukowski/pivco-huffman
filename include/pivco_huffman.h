#ifndef PIVCO_HUFFMAN_H
#define PIVCO_HUFFMAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Constants ---------- */

#ifndef PIVCO_BLOCK_SIZE
#if defined(__aarch64__)
#define PIVCO_BLOCK_SIZE 8192   /* 128KB L1D on Apple M-series */
#elif defined(__AVX512F__)
#define PIVCO_BLOCK_SIZE 8192   /* 48KB+ L1D on Intel Granite Rapids etc. */
#else
#define PIVCO_BLOCK_SIZE 4096   /* 32KB L1D on x86 (Zen, etc.) */
#endif
#endif

#define PIVCO_MAX_SYMBOLS   256
#define PIVCO_MAX_CODE_LEN  15

/* Maximum encoded size for one block (generous upper bound):
   Sum of code bits across all symbols. Worst case: all 8-bit codes
   => N bytes. Plus rounding overhead per tree node. */
#define PIVCO_MAX_ENCODED_SIZE (PIVCO_BLOCK_SIZE * 2)

/* ---------- Error codes ---------- */

#define PIVCO_OK            0
#define PIVCO_ERR_NULL      (-1)
#define PIVCO_ERR_OVERFLOW  (-2)
#define PIVCO_ERR_CORRUPT   (-3)
#define PIVCO_ERR_EMPTY     (-4)

/* ---------- Huffman tree node (for PIVCO tree-walk) ---------- */

/* Compact tree: nodes stored in array, indexed by node ID.
   Max nodes = 2 * MAX_SYMBOLS - 1 = 511.
   Leaf: symbol >= 0.  Internal: symbol = -1, left/right are children. */
#define PIVCO_MAX_TREE_NODES (2 * PIVCO_MAX_SYMBOLS - 1)

typedef struct {
    int16_t symbol;   /* >= 0 for leaf, -1 for internal */
    int16_t left;     /* child node index (bit=0) */
    int16_t right;    /* child node index (bit=1) */
} pivco_tree_node_t;

/* ---------- Huffman table ---------- */

typedef struct {
    /* Per-symbol encode info */
    uint16_t code[PIVCO_MAX_SYMBOLS];       /* canonical Huffman code */
    uint8_t  code_len[PIVCO_MAX_SYMBOLS];   /* code length (0 = unused) */

    /* Tree for PIVCO tree-walk encode/decode */
    pivco_tree_node_t tree[PIVCO_MAX_TREE_NODES];
    int16_t tree_root;
    int16_t tree_node_count;

    /* Canonical decode info (for traditional decoder) */
    uint16_t first_code[PIVCO_MAX_CODE_LEN + 1];
    uint16_t first_sym_idx[PIVCO_MAX_CODE_LEN + 1];
    uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1];
    uint8_t  sorted_symbols[PIVCO_MAX_SYMBOLS];

    /* Flat decode table: 2^MAX_CODE_LEN entries (for traditional decoder) */
    uint8_t  decode_sym[1 << PIVCO_MAX_CODE_LEN];
    uint8_t  decode_len[1 << PIVCO_MAX_CODE_LEN];

    uint8_t  max_len;
    uint8_t  min_len;
    uint16_t num_symbols;

    /* Most frequent symbol (shortest code). PIVCO decode prefills the
       output with this symbol via memset and skips its leaf scatter. */
    uint8_t  prefill_sym;
    int16_t  prefill_node;      /* tree node ID of the prefill leaf */

    /* Flat-subtree fast path: per-node, if flat_depth[i] >= 2 then node i
       is the root of a MAXIMAL flat subtree of depth D = flat_depth[i]
       (all 2^D leaves at the same relative depth).  Encoder emits N*D
       packed bits at this node instead of D levels of bitmaps; decoder
       reads N*D bits and uses flat_code_to_sym[flat_offset[i] + code]
       per element.  Pool sum of 2^D across flat subtrees <= num_symbols. */
    uint8_t  flat_depth[PIVCO_MAX_TREE_NODES];
    uint16_t flat_offset[PIVCO_MAX_TREE_NODES];
    uint8_t  flat_code_to_sym[PIVCO_MAX_SYMBOLS];
} pivco_huffman_table_t;

/* ---------- Implementation selection ---------- */

typedef enum {
    PIVCO_IMPL_AUTO = 0,
    PIVCO_IMPL_SCALAR,
    PIVCO_IMPL_NEON
} pivco_impl_t;

void         pivco_huffman_set_impl(pivco_impl_t impl);
pivco_impl_t pivco_huffman_get_impl(void);

/* ---------- Table construction ---------- */

int pivco_huffman_build_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table);

/* ---------- PIVCO Huffman encode/decode (block of PIVCO_BLOCK_SIZE symbols) ---------- */

int pivco_huffman_encode(const uint8_t *symbols,
                         const pivco_huffman_table_t *table,
                         uint8_t *out, size_t *out_len);

int pivco_huffman_decode(const uint8_t *in, size_t in_len,
                         const pivco_huffman_table_t *table,
                         uint8_t *symbols, size_t *consumed);

int pivco_huffman_encode_scalar(const uint8_t *symbols,
                                const pivco_huffman_table_t *table,
                                uint8_t *out, size_t *out_len);

int pivco_huffman_decode_scalar(const uint8_t *in, size_t in_len,
                                const pivco_huffman_table_t *table,
                                uint8_t *symbols, size_t *consumed);

#ifdef PIVCO_HAS_NEON
int pivco_huffman_encode_neon(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);

int pivco_huffman_decode_neon(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed);
#endif

#ifdef PIVCO_HAS_SSE4
int pivco_huffman_encode_x86(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);

int pivco_huffman_decode_x86(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed);
#endif

/* Prior experimental NEON variants (neon2, neon2b, neon_fused_1leaf)
 * are preserved under extras/ as negative results. See extras/README_*
 * files for writeups. */

/* Prefix-stream backend.  Uses a different bitstream format: first M
 * bits of every element's code packed contiguously (M = table->min_len),
 * followed (for non-flat trees) by standard 2-way PIVCO subtree bitmaps
 * in DFS order.  Works on both flat (min == max) and non-flat tables. */
int pivco_huffman_encode_neon_prefix(const uint8_t *symbols,
                                      const pivco_huffman_table_t *table,
                                      uint8_t *out, size_t *out_len);

int pivco_huffman_decode_neon_prefix(const uint8_t *in, size_t in_len,
                                      const pivco_huffman_table_t *table,
                                      uint8_t *symbols, size_t *consumed);

/* Internal entry points used by the prefix backend to delegate the
 * subtree work to the standard neon encoder/decoder.  Not intended for
 * direct use. */
void pivco_neon_decode_subtree_(const pivco_huffman_table_t *table,
                                 int16_t node_id,
                                 uint16_t *indices, int n,
                                 uint8_t *symbols,
                                 const uint8_t **in_ptr,
                                 uint16_t *tmp,
                                 int16_t skip_node);

void pivco_neon_encode_subtree_(const pivco_huffman_table_t *table,
                                 int16_t node_id,
                                 uint16_t *indices, int n,
                                 int depth,
                                 const uint16_t *codes, const uint8_t *lens,
                                 uint8_t **out_ptr,
                                 uint16_t *tmp);

#ifdef PIVCO_HAS_SVE
int pivco_huffman_encode_sve(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);

int pivco_huffman_decode_sve(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed);
#endif

#ifdef PIVCO_HAS_AVX512
int pivco_huffman_encode_avx512(const uint8_t *symbols,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *out, size_t *out_len);

int pivco_huffman_decode_avx512(const uint8_t *in, size_t in_len,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *symbols, size_t *consumed);
#endif

/* ---------- Traditional Huffman encode/decode (for comparison) ---------- */

int trad_huffman_encode(const uint8_t *symbols, size_t n_symbols,
                        const pivco_huffman_table_t *table,
                        uint8_t *out, size_t *out_len, size_t *out_bits);

int trad_huffman_decode(const uint8_t *in, size_t in_bits,
                        const pivco_huffman_table_t *table,
                        uint8_t *symbols, size_t n_symbols);

/* SotA 4-stream encode/decode (huff0-style) */
int trad_huffman_encode_4s(const uint8_t *symbols, size_t n_symbols,
                           const pivco_huffman_table_t *table,
                           uint8_t *out, size_t *out_len);

int trad_huffman_decode_4s(const uint8_t *in, size_t in_len,
                           const pivco_huffman_table_t *table,
                           uint8_t *symbols, size_t n_symbols);

#ifdef __cplusplus
}
#endif

#endif /* PIVCO_HUFFMAN_H */
