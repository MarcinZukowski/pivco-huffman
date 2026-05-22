/* phtd.h -- public consumer header for the namespaced top-down (TD)
 * library.  Lets the main bench drive ph-td's TD decoder grid alongside
 * the main pivco_huffman (bottom-up) library in one binary.
 *
 * phtd_table_t is opaque (ph-td's struct layout differs from the main
 * lib and may differ per host); allocate phtd_table_size() bytes. */
#ifndef PHTD_H
#define PHTD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PHTD_BLOCK_SIZE 8192   /* ph-td forces 8192 on every host */

typedef struct phtd_table phtd_table_t;   /* opaque */
size_t phtd_table_size(void);

void phtd_set_fse_enabled(int enabled);

/* Tree builds: naive (every internal node FULL, no opt) vs opt (flat
 * subtrees, fused scatter, prefill, K_right header). */
int phtd_build_table(const uint64_t freq[256], phtd_table_t *t);
int phtd_build_table_naive(const uint64_t freq[256], phtd_table_t *t);

/* naive tree -- slim wire (raw bitmaps, no FSE marker / K_right) */
int phtd_encode_naive(const uint8_t *sym, const phtd_table_t *t, uint8_t *out, size_t *out_len);
int phtd_decode_naive(const uint8_t *in, size_t in_len, const phtd_table_t *t, uint8_t *sym, size_t *consumed);

/* opt tree -- full ph wire */
int phtd_encode_scalar_opt(const uint8_t *sym, const phtd_table_t *t, uint8_t *out, size_t *out_len);
int phtd_decode_scalar_opt(const uint8_t *in, size_t in_len, const phtd_table_t *t, uint8_t *sym, size_t *consumed);

#ifdef PIVCO_HAS_NEON
int phtd_encode_neon(const uint8_t *sym, const phtd_table_t *t, uint8_t *out, size_t *out_len);
int phtd_decode_neon(const uint8_t *in, size_t in_len, const phtd_table_t *t, uint8_t *sym, size_t *consumed);          /* opt tree, NEON prims */
int phtd_decode_naive_simd_neon(const uint8_t *in, size_t in_len, const phtd_table_t *t, uint8_t *sym, size_t *consumed); /* naive tree, NEON prims */
int phtd_decode_bu_neon(const uint8_t *in, size_t in_len, const phtd_table_t *t, uint8_t *sym, size_t *consumed);
#endif

#ifdef PIVCO_HAS_AVX512
int phtd_encode_avx512(const uint8_t *sym, const phtd_table_t *t, uint8_t *out, size_t *out_len);
int phtd_decode_avx512(const uint8_t *in, size_t in_len, const phtd_table_t *t, uint8_t *sym, size_t *consumed);             /* opt tree, AVX-512 prims */
int phtd_decode_naive_simd_avx512(const uint8_t *in, size_t in_len, const phtd_table_t *t, uint8_t *sym, size_t *consumed);  /* naive tree, AVX-512 prims */
#endif

#ifdef __cplusplus
}
#endif

#endif /* PHTD_H */
