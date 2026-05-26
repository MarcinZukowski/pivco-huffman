/* pivco_huffman_primitives_avx512.h — AVX-512 VBMI2 primitive
 * implementations of the codec primitive interface (see
 * pivco_huffman_primitives.h).
 *
 * Specialized names end in `_avx512`; the codec calls the aliases
 * `prim_*` defined at the bottom as always-inline wrappers.
 *
 * Floor: AVX-512 F + BW + VBMI + VBMI2 + VPOPCNTDQ.  Pivco's AVX-512
 * tier is built with all of these (-mavx512f -mavx512bw -mavx512vbmi
 * -mavx512vbmi2 -mavx512vpopcntdq via CMakeLists.txt), and the runtime
 * dispatcher (pivco_huffman.c::resolve_impl) only routes to this
 * backend when /proc/cpuinfo advertises avx512_vbmi2.  The header errors
 * out if the macro contract isn't met -- catches misconfigured builds
 * before they produce silently-incorrect output.
 *
 * Notable kernels (all kept symmetric with the legacy
 * pivco_huffman_avx512.c bodies they replace):
 *
 *   - enc_init_avx512: 64-char vpermex2var_epi8 byte-split table lookup
 *     (chunked over a 256-entry uint16 LUT split into 8 byte-half
 *     chunks).  ~0.19 ops/char vs scalar's ~1.0.  See the comment block
 *     in the function for the lookup geometry.
 *
 *   - build_bitmap_partition_avx512: stride-32 vpcompressw main loop
 *     (one ZMM = 32 uint16 codes per iter), with an SSE-stride-8 tail
 *     also using vpcompressw via VL.  No shuffle table -- compressw is
 *     the table-free analog of the SSE pshufb + compress_tab dance.
 *
 *   - BU tree_merge_avx512 family: 64-byte vpexpandb main loop (one
 *     ZMM load+expand per side, OR'd together), SSE stride-16 tail
 *     using expand_tab from pivco_huffman_x86_tables (the only x86
 *     table this backend depends on; codec_init_avx512 inits it).
 *
 *   - flat_decode_to_buffer_avx512: D=2/3/4/5/6 vector unpacks via the
 *     shared pivco_huffman_avx512_flat.h helpers (D=5/6 use
 *     vpmultishiftqb / vpermb -- the AVX-512-only fast paths).
 *
 *   - pack_dN_avx512: D=2..7 via vpcompressw + sllv across 8 uint64
 *     lanes per iter; D=8 via vpmovwb byte narrow.
 *
 * Internal header.  Included by pivco_huffman_primitives.h when
 * PIVCO_BACKEND_AVX512 is defined.  Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_PRIMITIVES_AVX512_H
#define PIVCO_HUFFMAN_PRIMITIVES_AVX512_H

#if !defined(PIVCO_HAS_AVX512)
#error "pivco_huffman_primitives_avx512.h requires PIVCO_HAS_AVX512"
#endif
#if !defined(__AVX512VBMI2__) || !defined(__AVX512VPOPCNTDQ__)
#error "pivco_huffman_primitives_avx512.h requires AVX-512 VBMI2 + VPOPCNTDQ"
#endif

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_x86_tables.h"      /* expand_tab for BU tail */
#include "pivco_huffman_avx512_flat.h"     /* flat_d{2,3,4,5,6}_unpack_avx512* */
#include "pivco_prof.h"

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/* Backend lifecycle.  Only the BU tree_merge SSE-stride tail needs a
 * runtime table (expand_tab in pivco_huffman_x86_tables.c).  AVX-512
 * partition and encode are entirely table-free (vpcompressw is the
 * "table" -- it's hardware). */
static inline void codec_init_avx512(void)
{
    init_expand_table_x86();
}

/* ---------- Decode primitives (bottom-up) ---------- */

/* popcount_K_right_avx512 — count "1" bits in the first K bits of bm.
 * 64-byte main loop with VPOPCNTQ; 1c throughput.  No codec.c caller
 * (codec uses wire_read_kr_header for the value at read time); kept
 * for symmetry with primitives_x86.h + primitives_neon.h.  `nbytes`
 * is derivable from K. */
static inline int popcount_K_right_avx512(const uint8_t *bm,
                                            int nbytes, int K)
{
    (void)nbytes;
    PROF_TIC();
    int full_bytes = K >> 3;
    int partial_bits = K & 7;
    int b = 0;

    __m512i acc = _mm512_setzero_si512();
    for (; b + 64 <= full_bytes; b += 64) {
        __m512i v = _mm512_loadu_si512((const __m512i *)(bm + b));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
    }
    int K_right = (int)_mm512_reduce_add_epi64(acc);

    for (; b + 8 <= full_bytes; b += 8) {
        uint64_t v;
        memcpy(&v, bm + b, 8);
        K_right += __builtin_popcountll(v);
    }
    for (; b < full_bytes; b++) {
        K_right += __builtin_popcount(bm[b]);
    }
    if (partial_bits) {
        uint8_t valid_mask = (uint8_t)((1u << partial_bits) - 1);
        K_right += __builtin_popcount(bm[full_bytes] & valid_mask);
    }
    PROF_TOC(PROF_BU_POPCOUNT_K, K);
    return K_right;
}

/* tree_merge_avx512 — VBMI2 64-byte main loop via vpexpandb (two
 * masked expand-loads OR'd together), SSE stride-16/-8 tails using
 * expand_tab from x86_tables.  ~0.023 ns/byte on Xeon Ice Lake+. */
static inline void tree_merge_avx512(const uint8_t *bm, int K,
                                       const uint8_t *left,
                                       const uint8_t *right,
                                       uint8_t *out)
{
    PROF_TIC();
    int lc = 0, rc = 0;
    int j = 0;
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
    /* 2x-unrolled SSE stride-16: see primitives_x86.h. */
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i L0 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(L0, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        int nr0 = expand_popcnt[m0];
        rc += nr0; lc += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i L1 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(L1, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        int nr1 = expand_popcnt[m1];
        rc += nr1; lc += (8 - nr1);
    }
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

/* tree_merge_bcast_left_avx512 — left input is a broadcast constant. */
static inline void tree_merge_bcast_left_avx512(const uint8_t *bm, int K,
                                                  uint8_t left_sym,
                                                  const uint8_t *right,
                                                  uint8_t *out)
{
    PROF_TIC();
    int rc = 0;
    int j = 0;
    __m128i Lbcast8 = _mm_set1_epi8((char)left_sym);
    __m512i Lbcast64 = _mm512_set1_epi8((char)left_sym);
    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __m512i R = _mm512_maskz_expandloadu_epi8(m, right + rc);
        __m512i o = _mm512_mask_mov_epi8(Lbcast64, m, R);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        rc += __builtin_popcountll(mask);
    }
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(Lbcast8, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        rc += expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(Lbcast8, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        rc += expand_popcnt[m1];
    }
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

/* tree_merge_bcast_right_avx512 — mirror of the bcast_left variant. */
static inline void tree_merge_bcast_right_avx512(const uint8_t *bm, int K,
                                                   const uint8_t *left,
                                                   uint8_t right_sym,
                                                   uint8_t *out)
{
    PROF_TIC();
    int lc = 0;
    int j = 0;
    __m128i Rbcast8 = _mm_set1_epi8((char)right_sym);
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
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i L0 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i both0 = _mm_unpacklo_epi64(L0, Rbcast8);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        lc += (8 - expand_popcnt[m0]);

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i L1 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i both1 = _mm_unpacklo_epi64(L1, Rbcast8);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        lc += (8 - expand_popcnt[m1]);
    }
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

/* merge_both_const_avx512 — both inputs are constants.  AVX2 stride-32
 * pblendvb main path (lifted as-is from primitives_x86.h since AVX-512
 * has no faster sequence for this); SSE 16-byte tail. */
static inline void merge_both_const_avx512(const uint8_t *bm, int K,
                                             uint8_t left_sym, uint8_t right_sym,
                                             uint8_t *out)
{
    PROF_TIC();
    __m128i vsym0 = _mm_set1_epi8((char)left_sym);
    __m128i vsym1 = _mm_set1_epi8((char)right_sym);
    __m128i bits  = _mm_setr_epi8(1,2,4,8,16,32,64,(char)128,
                                   1,2,4,8,16,32,64,(char)128);
    __m128i shuf  = _mm_setr_epi8(0,0,0,0,0,0,0,0,
                                   1,1,1,1,1,1,1,1);
    int j = 0;
    __m256i vsym0_256 = _mm256_set1_epi8((char)left_sym);
    __m256i vsym1_256 = _mm256_set1_epi8((char)right_sym);
    __m256i bits_256  = _mm256_broadcastsi128_si256(bits);
    __m256i shuf_256  = _mm256_broadcastsi128_si256(shuf);
    for (; j + 32 <= K; j += 32) {
        uint32_t four;
        memcpy(&four, bm + (j >> 3), 4);
        __m256i bm_quad = _mm256_set_epi32(0, 0, 0, (int)(four >> 16),
                                           0, 0, 0, (int)(four & 0xFFFF));
        __m256i bm_dup  = _mm256_shuffle_epi8(bm_quad, shuf_256);
        __m256i masked  = _mm256_and_si256(bm_dup, bits_256);
        __m256i mask8   = _mm256_cmpeq_epi8(masked, bits_256);
        __m256i o       = _mm256_blendv_epi8(vsym0_256, vsym1_256, mask8);
        _mm256_storeu_si256((__m256i *)(out + j), o);
    }
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

/* ---------- Flat-subtree decode (contiguous output) ---------- */

/* Extract D bits at bit position `bit_pos` from `in`.  D <= 16. */
static inline uint32_t extract_D_bits_avx512(const uint8_t *in,
                                               int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3;
    int bit_off  = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8)  val |= ((uint32_t)in[byte_idx + 1]) << 8;
    if (bit_off + D > 16) val |= ((uint32_t)in[byte_idx + 2]) << 16;
    return (val >> bit_off) & ((1u << D) - 1);
}

/* Per-D scalar unrolled switch (used as tail for D=7/8 and any
 * unhandled D).  `DST(k)` is the destination expression. */
#define AVX512_FLAT_UNPACK_SWITCH(DST)                                         \
    int i = 0;                                                                 \
    switch (D) {                                                               \
    case 2:                                                                    \
        for (; i + 4 <= n; i += 4) {                                           \
            uint8_t b = bm[i >> 2];                                            \
            DST(i    ) = c2s[(b     ) & 3];                                    \
            DST(i + 1) = c2s[(b >> 2) & 3];                                    \
            DST(i + 2) = c2s[(b >> 4) & 3];                                    \
            DST(i + 3) = c2s[(b >> 6) & 3];                                    \
        } break;                                                               \
    case 3:                                                                    \
        for (; i + 8 <= n; i += 8) {                                           \
            const uint8_t *p = bm + ((i * 3) >> 3);                            \
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16); \
            DST(i    ) = c2s[(w      ) & 7];                                   \
            DST(i + 1) = c2s[(w >>  3) & 7];                                   \
            DST(i + 2) = c2s[(w >>  6) & 7];                                   \
            DST(i + 3) = c2s[(w >>  9) & 7];                                   \
            DST(i + 4) = c2s[(w >> 12) & 7];                                   \
            DST(i + 5) = c2s[(w >> 15) & 7];                                   \
            DST(i + 6) = c2s[(w >> 18) & 7];                                   \
            DST(i + 7) = c2s[(w >> 21) & 7];                                   \
        } break;                                                               \
    case 4:                                                                    \
        for (; i + 2 <= n; i += 2) {                                           \
            uint8_t b = bm[i >> 1];                                            \
            DST(i    ) = c2s[b & 0x0F];                                        \
            DST(i + 1) = c2s[b >> 4];                                          \
        } break;                                                               \
    case 5:                                                                    \
        for (; i + 8 <= n; i += 8) {                                           \
            const uint8_t *p = bm + ((i * 5) >> 3);                            \
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)                \
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)       \
                       | ((uint64_t)p[4] << 32);                               \
            DST(i    ) = c2s[(w      ) & 0x1F];                                \
            DST(i + 1) = c2s[(w >>  5) & 0x1F];                                \
            DST(i + 2) = c2s[(w >> 10) & 0x1F];                                \
            DST(i + 3) = c2s[(w >> 15) & 0x1F];                                \
            DST(i + 4) = c2s[(w >> 20) & 0x1F];                                \
            DST(i + 5) = c2s[(w >> 25) & 0x1F];                                \
            DST(i + 6) = c2s[(w >> 30) & 0x1F];                                \
            DST(i + 7) = c2s[(w >> 35) & 0x1F];                                \
        } break;                                                               \
    case 6:                                                                    \
        for (; i + 4 <= n; i += 4) {                                           \
            const uint8_t *p = bm + ((i * 6) >> 3);                            \
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16); \
            DST(i    ) = c2s[(w      ) & 0x3F];                                \
            DST(i + 1) = c2s[(w >>  6) & 0x3F];                                \
            DST(i + 2) = c2s[(w >> 12) & 0x3F];                                \
            DST(i + 3) = c2s[(w >> 18) & 0x3F];                                \
        } break;                                                               \
    case 7:                                                                    \
        for (; i + 8 <= n; i += 8) {                                           \
            const uint8_t *p = bm + ((i * 7) >> 3);                            \
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)                \
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)       \
                       | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)       \
                       | ((uint64_t)p[6] << 48);                               \
            DST(i    ) = c2s[(w      ) & 0x7F];                                \
            DST(i + 1) = c2s[(w >>  7) & 0x7F];                                \
            DST(i + 2) = c2s[(w >> 14) & 0x7F];                                \
            DST(i + 3) = c2s[(w >> 21) & 0x7F];                                \
            DST(i + 4) = c2s[(w >> 28) & 0x7F];                                \
            DST(i + 5) = c2s[(w >> 35) & 0x7F];                                \
            DST(i + 6) = c2s[(w >> 42) & 0x7F];                                \
            DST(i + 7) = c2s[(w >> 49) & 0x7F];                                \
        } break;                                                               \
    case 8:                                                                    \
        for (; i < n; i++) DST(i) = c2s[bm[i]];                                \
        break;                                                                 \
    }                                                                          \
    for (; i < n; i++) {                                                       \
        uint32_t code = extract_D_bits_avx512(bm, i * D, D);                   \
        DST(i) = c2s[code];                                                    \
    }

/* flat_decode_direct_avx512_inner — D=2/3/4/5/6 SIMD; scalar tail
 * for D=7/8 and unhandled.  D=5/D=6 use vpermb / vpermw over 16/32/64-
 * entry c2s registers (the AVX-512-only fast paths). */
static inline void flat_decode_direct_avx512_inner(uint8_t *symbols, int n,
                                                     const uint8_t *bm, int D,
                                                     const uint8_t *c2s)
{
    if (D == 6) {
        /* c2s has 64 entries — fits in zmm, use vpermb. */
        __m512i c2s_vec = _mm512_loadu_si512((const __m512i *)c2s);
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3));
            __m512i codes_ext = _mm512_castsi128_si512(codes);
            __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm512_castsi512_si128(syms_full));
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d6_unpack_avx512_safe(bm + ((i * 6) >> 3));
            __m512i codes_ext = _mm512_castsi128_si512(codes);
            __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm512_castsi512_si128(syms_full));
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 7) {
        /* c2s has 128 entries — two zmm tables, one vpermi2b (indices 0..127
         * pick from c2s_lo[0..63] / c2s_hi[64..127]). */
        __m512i c2s_lo = _mm512_loadu_si512((const __m512i *)c2s);
        __m512i c2s_hi = _mm512_loadu_si512((const __m512i *)(c2s + 64));
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d7_unpack_avx512_fast(bm + ((i * 7) >> 3));
            __m512i syms = _mm512_permutex2var_epi8(c2s_lo,
                               _mm512_castsi128_si512(codes), c2s_hi);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm512_castsi512_si128(syms));
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d7_unpack_avx512_safe(bm + ((i * 7) >> 3));
            __m512i syms = _mm512_permutex2var_epi8(c2s_lo,
                               _mm512_castsi128_si512(codes), c2s_hi);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm512_castsi512_si128(syms));
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 5) {
        /* c2s has 32 entries — needs vpermb over ymm. */
        __m256i c2s_vec = _mm256_loadu_si256((const __m256i *)c2s);
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3));
            __m256i codes_ext = _mm256_zextsi128_si256(codes);
            __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm256_castsi256_si128(syms_full));
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d5_unpack_avx512_safe(bm + ((i * 5) >> 3));
            __m256i codes_ext = _mm256_zextsi128_si256(codes);
            __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm256_castsi256_si128(syms_full));
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 3) {
        uint64_t c2s_lo;
        memcpy(&c2s_lo, c2s, 8);
        __m128i c2s_vec = _mm_cvtsi64_si128((int64_t)c2s_lo);
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d3_unpack_avx512_safe(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 4) {
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_avx512(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 2) {
        uint32_t c2s_lo;
        memcpy(&c2s_lo, c2s, 4);
        __m128i c2s_vec = _mm_set1_epi32((int32_t)c2s_lo);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d2_unpack_avx512(bm + (i >> 2));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            symbols[i    ] = c2s[(b     ) & 3];
            symbols[i + 1] = c2s[(b >> 2) & 3];
            symbols[i + 2] = c2s[(b >> 4) & 3];
            symbols[i + 3] = c2s[(b >> 6) & 3];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
#define DST_DIRECT(k) symbols[k]
    AVX512_FLAT_UNPACK_SWITCH(DST_DIRECT)
#undef DST_DIRECT
}

/* flat_decode_to_buffer_avx512 — D-bit flat-subtree decode into a
 * contiguous output buffer. */
static inline void flat_decode_to_buffer_avx512(uint8_t *out, int n,
                                                  const uint8_t *bm, int D,
                                                  const uint8_t *c2s)
{
    PROF_TIC();
    flat_decode_direct_avx512_inner(out, n, bm, D, c2s);
    PROF_TOC(PROF_BU_FLAT_DECODE, n);
}

/* ---------- Encode primitives (bitmap + partition) ----------
 *
 * Stride-32 main loop: load 32 left-aligned codes, build the 32-bit
 * mask via vpsllw + vpmovw2m (the AVX-512 analog of SSE's vpsllw +
 * vpacksw + vpmovmskb sequence -- single instruction).  Partition with
 * vpcompressw, two stores.  SSE-stride-8 tail uses _mm_maskz_compress_
 * epi16 (VL).  No shuffle table needed at either tier.
 *
 * codes_la is depth-threaded: not shifted across recursion levels.
 * The current-depth partition bit is at position (15 - depth) of each
 * lane; we left-shift by `depth` to move it to bit 15 (sign bit of
 * int16) so vpmovw2m reads it directly. */

static inline uint32_t enc_mask32_codes_la_avx512(__m512i code_vec, int depth)
{
    __m512i shifted = _mm512_slli_epi16(code_vec, depth);
    return (uint32_t)_mm512_movepi16_mask(shifted);
}

static inline int build_bitmap_partition_avx512(uint16_t *codes_la, int n,
                                                  int depth,
                                                  uint8_t *bm,
                                                  uint16_t *tmp)
{
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 32 <= n; j += 32) {
        __m512i code_vec = _mm512_loadu_si512((const __m512i *)(codes_la + j));
        uint32_t mask = enc_mask32_codes_la_avx512(code_vec, depth);
        memcpy(bm + (j >> 3), &mask, 4);

        __m512i right_v = _mm512_maskz_compress_epi16((__mmask32) mask, code_vec);
        __m512i left_v  = _mm512_maskz_compress_epi16((__mmask32)~mask, code_vec);
        _mm512_storeu_si512((__m512i *)(tmp      + n_right), right_v);
        _mm512_storeu_si512((__m512i *)(codes_la + n_left ), left_v);
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += (32 - nr);
    }
    /* SSE-stride-8 remainder via _mm_maskz_compress_epi16 (VL). */
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted  = _mm_sll_epi16(code_vec, shift_count);
        __m128i bytes    = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask     = (uint8_t)_mm_movemask_epi8(bytes);
        bm[j >> 3] = mask;

        __m128i right_v = _mm_maskz_compress_epi16((__mmask8) mask, code_vec);
        __m128i left_v  = _mm_maskz_compress_epi16((__mmask8)~mask, code_vec);
        _mm_storeu_si128((__m128i *)(tmp      + n_right), right_v);
        _mm_storeu_si128((__m128i *)(codes_la + n_left ), left_v);
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += (8 - nr);
    }
    /* Scalar tail. */
    if (j < n) {
        int tail = n - j;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask = 0;
        int shift_d = 15 - depth;
        for (int k = 0; k < tail; k++) {
            int bit = (tail_buf[k] >> shift_d) & 1;
            mask |= (uint8_t)(bit << k);
        }
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k))
                tmp[n_right++] = tail_buf[k];
            else
                codes_la[n_left++] = tail_buf[k];
        }
    }
    return n_right;
}

/* ---------- Encode primitives (init) ----------
 *
 * enc_init_avx512 — gather per-symbol left-aligned codes via byte-split
 * vpermex2var_epi8 (AVX-512 VBMI).  64 chars per iter, 12 ops, ~0.19
 * ops/char.  See the comment block for the lookup geometry.
 *
 * Lifted verbatim (minus the surrounding encoder driver) from the
 * legacy pivco_huffman_avx512.c.  All design notes preserved. */
static inline void enc_init_avx512(uint16_t *codes_la, int n,
                                     const uint8_t *symbols,
                                     const uint16_t *code_la_lut)
{
    /* Build lo/hi byte tables from the uint16 code_la table.  Each
     * byte-table chunk holds 64 sequential entries' lo (or hi) bytes;
     * two chunks per byte-half pair the chars [0,128) and [128,256)
     * regions for vpermex2var_epi8. */
    __m512i u0 = _mm512_loadu_si512((const __m512i *)&code_la_lut[  0]);
    __m512i u1 = _mm512_loadu_si512((const __m512i *)&code_la_lut[ 32]);
    __m512i u2 = _mm512_loadu_si512((const __m512i *)&code_la_lut[ 64]);
    __m512i u3 = _mm512_loadu_si512((const __m512i *)&code_la_lut[ 96]);
    __m512i u4 = _mm512_loadu_si512((const __m512i *)&code_la_lut[128]);
    __m512i u5 = _mm512_loadu_si512((const __m512i *)&code_la_lut[160]);
    __m512i u6 = _mm512_loadu_si512((const __m512i *)&code_la_lut[192]);
    __m512i u7 = _mm512_loadu_si512((const __m512i *)&code_la_lut[224]);

    __m512i lo_c0p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u0)),
        _mm512_cvtepi16_epi8(u1), 1);
    __m512i lo_c0p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u2)),
        _mm512_cvtepi16_epi8(u3), 1);
    __m512i lo_c1p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u4)),
        _mm512_cvtepi16_epi8(u5), 1);
    __m512i lo_c1p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u6)),
        _mm512_cvtepi16_epi8(u7), 1);

    __m512i hi_c0p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u0, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u1, 8)), 1);
    __m512i hi_c0p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u2, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u3, 8)), 1);
    __m512i hi_c1p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u4, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u5, 8)), 1);
    __m512i hi_c1p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u6, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u7, 8)), 1);

    static const uint8_t inter_sel0_tab[64] __attribute__((aligned(64))) = {
         0, 64,  1, 65,  2, 66,  3, 67,  4, 68,  5, 69,  6, 70,  7, 71,
         8, 72,  9, 73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79,
        16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87,
        24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95
    };
    static const uint8_t inter_sel1_tab[64] __attribute__((aligned(64))) = {
        32, 96, 33, 97, 34, 98, 35, 99, 36,100, 37,101, 38,102, 39,103,
        40,104, 41,105, 42,106, 43,107, 44,108, 45,109, 46,110, 47,111,
        48,112, 49,113, 50,114, 51,115, 52,116, 53,117, 54,118, 55,119,
        56,120, 57,121, 58,122, 59,123, 60,124, 61,125, 62,126, 63,127
    };
    __m512i sel0 = _mm512_load_si512((const __m512i *)inter_sel0_tab);
    __m512i sel1 = _mm512_load_si512((const __m512i *)inter_sel1_tab);

    PROF_TIC();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i chars = _mm512_loadu_si512((const __m512i *)(symbols + i));
        __mmask64 hi_chunk = _mm512_movepi8_mask(chars);

        __m512i lo0 = _mm512_permutex2var_epi8(lo_c0p1, chars, lo_c0p2);
        __m512i lo1 = _mm512_permutex2var_epi8(lo_c1p1, chars, lo_c1p2);
        __m512i lo  = _mm512_mask_blend_epi8(hi_chunk, lo0, lo1);

        __m512i hi0 = _mm512_permutex2var_epi8(hi_c0p1, chars, hi_c0p2);
        __m512i hi1 = _mm512_permutex2var_epi8(hi_c1p1, chars, hi_c1p2);
        __m512i hi  = _mm512_mask_blend_epi8(hi_chunk, hi0, hi1);

        __m512i out0 = _mm512_permutex2var_epi8(lo, sel0, hi);
        __m512i out1 = _mm512_permutex2var_epi8(lo, sel1, hi);

        _mm512_storeu_si512((__m512i *)(codes_la + i     ), out0);
        _mm512_storeu_si512((__m512i *)(codes_la + i + 32), out1);
    }
    /* Scalar tail (PIVCO_BLOCK_SIZE is a multiple of 64 on AVX-512
     * hosts so this is currently dead, but kept defensively). */
    for (; i < n; i++) codes_la[i] = code_la_lut[symbols[i]];
    PROF_TOC(PROF_ENC_INIT, n);
}

/* ---------- Encode primitives (flat-subtree pack) ----------
 *
 * D=2..7: 8 codes per iter via uint64 lanes (max shift 7*7=49 < 64),
 * one vpcompressw + sllv per call.
 * D=8: byte-aligned, 32 codes per iter via vpmovwb narrow + store. */

#define PACK_DN_AVX512_UNIFIED(NAME, D_VAL, BITS_OUT)                          \
static inline int NAME(uint8_t *out, const uint16_t *codes_la,                 \
                       int n, int right_shift)                                 \
{                                                                              \
    static const int64_t shifts[8] = {                                         \
        0, D_VAL, 2*D_VAL, 3*D_VAL, 4*D_VAL, 5*D_VAL, 6*D_VAL, 7*D_VAL         \
    };                                                                         \
    __m512i shift_vec = _mm512_loadu_si512((const __m512i *)shifts);           \
    __m512i mask_vec  = _mm512_set1_epi64((1ULL << D_VAL) - 1);                \
    int i = 0;                                                                 \
    for (; i + 8 <= n; i += 8) {                                               \
        __m128i v16 = _mm_loadu_si128((const __m128i *)(codes_la + i));        \
        __m512i v64 = _mm512_cvtepu16_epi64(v16);                              \
        v64 = _mm512_srli_epi64(v64, right_shift);                             \
        v64 = _mm512_and_si512(v64, mask_vec);                                 \
        v64 = _mm512_sllv_epi64(v64, shift_vec);                               \
        uint64_t packed = _mm512_reduce_add_epi64(v64);                        \
        int bi = i * D_VAL / 8;                                                \
        memcpy(out + bi, &packed, (BITS_OUT + 7) / 8);                         \
    }                                                                          \
    return i;                                                                  \
}
PACK_DN_AVX512_UNIFIED(pack_d2_avx512, 2, 16)
PACK_DN_AVX512_UNIFIED(pack_d3_avx512, 3, 24)
PACK_DN_AVX512_UNIFIED(pack_d4_avx512, 4, 32)
PACK_DN_AVX512_UNIFIED(pack_d5_avx512, 5, 40)
PACK_DN_AVX512_UNIFIED(pack_d6_avx512, 6, 48)
PACK_DN_AVX512_UNIFIED(pack_d7_avx512, 7, 56)
#undef PACK_DN_AVX512_UNIFIED

static inline int pack_d8_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512i v = _mm512_loadu_si512((const __m512i *)(codes_la + i));
        v = _mm512_srli_epi16(v, right_shift);
        __m256i bytes = _mm512_cvtepi16_epi8(v);
        _mm256_storeu_si256((__m256i *)(out + i), bytes);
    }
    return i;
}

/* Dispatcher: pack n D-bit codes from codes_la into out[].  Selects
 * the SIMD per-D pack helper; scalar tail picks up the residual. */
static inline void pack_dN_avx512(uint8_t *out, const uint16_t *codes_la,
                                    int n, int D, int depth)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int right_shift = 16 - depth - D;

    int i = 0;
    switch (D) {
    case 2: i = pack_d2_avx512(out, codes_la, n, right_shift); break;
    case 3: i = pack_d3_avx512(out, codes_la, n, right_shift); break;
    case 4: i = pack_d4_avx512(out, codes_la, n, right_shift); break;
    case 5: i = pack_d5_avx512(out, codes_la, n, right_shift); break;
    case 6: i = pack_d6_avx512(out, codes_la, n, right_shift); break;
    case 7: i = pack_d7_avx512(out, codes_la, n, right_shift); break;
    case 8: i = pack_d8_avx512(out, codes_la, n, right_shift); break;
    default: break;
    }

    int simd_n = i > n ? n : i;
    PROF_COUNT_ONLY(PROF_ENC_FLAT_SIMD_ELEMS, simd_n);
    PROF_COUNT_ONLY(PROF_ENC_FLAT_TAIL_ELEMS, n - simd_n);
    (void)simd_n;  /* unused when PIVCO_PROF=0 */

    if (i >= n) return;

    /* Scalar tail (fires only for D >= 9, currently impossible with
     * PIVCO_MAX_CODE_LEN = 11 and flat-D <= depth bound). */
    uint32_t mask = (1u << D) - 1;
    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = ((uint32_t)codes_la[i] >> right_shift) & mask;
        buf |= ((uint64_t)local) << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) {
        out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
    }
}

/* ---------- Aliases consumed by codec.c ---------- */

#define PIVCO_PRIM_ALWAYS_INLINE __attribute__((always_inline)) static inline

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ codec_init_avx512(); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ enc_init_avx512(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int prim_partition(uint16_t *codes_la,
                                                           int n, int depth,
                                                           uint8_t *bm,
                                                           uint16_t *tmp)
{ return build_bitmap_partition_avx512(codes_la, n, depth, bm, tmp); }

PIVCO_PRIM_ALWAYS_INLINE void prim_pack_dN(uint8_t *out,
                                             const uint16_t *codes_la,
                                             int n, int D, int depth)
{ pack_dN_avx512(out, codes_la, n, D, depth); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat(uint8_t *out, int n,
                                                          const uint8_t *bm, int D,
                                                          const uint8_t *c2s)
{ flat_decode_to_buffer_avx512(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_two(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_both_const_avx512(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_constant_left(const uint8_t *bm, int K,
                                                          uint8_t left_sym,
                                                          const uint8_t *right_buf,
                                                          uint8_t *out)
{ tree_merge_bcast_left_avx512(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_constant_right(const uint8_t *bm, int K,
                                                           const uint8_t *left_buf,
                                                           uint8_t right_sym,
                                                           uint8_t *out)
{ tree_merge_bcast_right_avx512(bm, K, left_buf, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge(const uint8_t *bm, int K,
                                               const uint8_t *left_buf,
                                               const uint8_t *right_buf,
                                               uint8_t *out)
{ tree_merge_avx512(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_AVX512_H */
