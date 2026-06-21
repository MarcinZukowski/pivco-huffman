/* pivco_huffman_primitives_x86.h — x86 (SSE4.1 + optional AVX2) primitive
 * implementations of the codec primitive interface (see
 * pivco_huffman_primitives.h).
 *
 * Specialized names end in `_x86`; the codec calls the aliases `prim_*`
 * defined at the bottom as always-inline wrappers.  Two implementation
 * tiers gated by PIVCO_HAS_AVX2: the AVX2 tier widens pack_dN to 64-bit
 * per-lane shifts via _mm256_sllv_epi64 (D=3/5/6/7) and gives flat
 * decode a 32-byte D=4 fast path.  The SSE4.1 floor handles D=2/4/8
 * with hand-rolled tricks (_mm_maddubs_epi16 weighted pair-add for D=2/4,
 * _mm_mullo_epi32 multiply-as-shift for D=3) and falls back to scalar
 * for D=5/6/7 (no uint64 per-lane shift in SSE).
 *
 * AVX-512 VBMI2 fast paths live in primitives_avx512.h (Phase 5
 * landed 2026-05-14).  On AVX-512 hosts the runtime dispatcher routes
 * to codec_avx512, so this file does NOT need to gate __AVX512* fast
 * paths internally.  Even when the codec_x86 OBJECT lib is compiled
 * on an AVX-512 host (with -mavx512vbmi2 enabled globally), it's
 * never reached at runtime there.
 *
 * Internal header.  Included by pivco_huffman_primitives.h when
 * PIVCO_BACKEND_X86 is defined.  Not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_PRIMITIVES_X86_H
#define PIVCO_HUFFMAN_PRIMITIVES_X86_H

#if !defined(PIVCO_HAS_SSE4)
#error "pivco_huffman_primitives_x86.h requires PIVCO_HAS_SSE4"
#endif

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_x86_tables.h"   /* compress_tab*, expand_tab* */
#include "pivco_huffman_x86_flat.h"     /* flat_d{2,3,4,5,6}_unpack_x86 */
#ifdef PIVCO_HAS_AVX2
#include "pivco_huffman_avx2_pack.h"    /* pack_d{2,3,5,6,7}_avx2_x86 (ryg pack) */
#endif
#include "pivco_prof.h"

#include <smmintrin.h>                  /* SSE4.1 */
#include <immintrin.h>                  /* AVX/AVX2/AVX-512 umbrella;
                                         * gated paths drop out cleanly
                                         * on SSE-only builds. */
#include <stdint.h>
#include <string.h>

/* Backend lifecycle.  Lazily build the compress_tab + expand_tab pre-
 * bake tables that the x86 partition / merge primitives index
 * into.  Idempotent and cheap after the first call. */
static inline void codec_init_x86(void)
{
    init_compress_table_x86();
    init_expand_table_x86();
}

/* ---------- Decode primitives (bottom-up) ---------- */

/* popcount_K_right_x86 — count "1" bits in the first K bits of bm.
 * Scalar 64-bit POPCNT, 4-way unrolled.  No codec.c caller (codec uses
 * wire_read_kr_header for the value at read time); kept for signature
 * stability with the NEON BU backend.  `nbytes` is derivable from K.
 * VPOPCNTQ fast path lives in primitives_avx512.h. */
static inline int popcount_K_right_x86(const uint8_t *bm, int nbytes, int K)
{
    (void)nbytes;
    PROF_TIC();
    int full_bytes = K >> 3;
    int partial_bits = K & 7;
    int b = 0;
    int K_right = 0;

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
    K_right = (int)(a0 + a1 + a2 + a3);

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

/* merge_vec_vec_x86 — SSE 8-byte chunk via pshufb on
 * _mm_unpacklo_epi64(L8, R8) with expand_tab[mask].  2x-unrolled
 * stride-16 main path: two independent 8-byte merges per iter so OOO
 * overlaps loads / shuffles / stores.  Only lc/rc cursor adds carry a
 * real dep, and that's short-latency add.  AVX-512 VBMI2 64-byte
 * vpexpandb fast path lives in primitives_avx512.h. */
static inline void merge_vec_vec_x86(const uint8_t *bm, int K,
                                    const uint8_t *left,
                                    const uint8_t *right,
                                    uint8_t *out)
{
    PROF_TIC();
    int lc = 0, rc = 0;
    int j = 0;
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
    PROF_TOC(PROF_BU_MERGE_VEC_VEC, K);
}

/* merge_cst_vec_x86 — left input is a broadcast constant.
 * Same 2x-unrolled structure; the L lane is a duplicated 16-byte
 * register holding left_sym. */
static inline void merge_cst_vec_x86(const uint8_t *bm, int K,
                                               uint8_t left_sym,
                                               const uint8_t *right,
                                               uint8_t *out)
{
    PROF_TIC();
    int rc = 0;
    int j = 0;
    __m128i Lbcast8 = _mm_set1_epi8((char)left_sym);
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
    PROF_TOC(PROF_BU_MERGE_CST_VEC, K);
}

/* merge_vec_cst_x86 — mirror of merge_cst_vec_x86. */
static inline void merge_vec_cst_x86(const uint8_t *bm, int K,
                                                const uint8_t *left,
                                                uint8_t right_sym,
                                                uint8_t *out)
{
    PROF_TIC();
    int lc = 0;
    int j = 0;
    __m128i Rbcast8 = _mm_set1_epi8((char)right_sym);
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
    PROF_TOC(PROF_BU_MERGE_VEC_CST, K);
}

/* merge_cst_cst_x86 — both inputs are constants.  vpblendvb-style:
 * for each bit in mask, output is right_sym or left_sym.  AVX2 widens
 * to 32 bytes per iter; SSE4.1 floor handles 16. */
static inline void merge_cst_cst_x86(const uint8_t *bm, int K,
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
#ifdef PIVCO_HAS_AVX2
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
#endif
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
    PROF_TOC(PROF_BU_MERGE_CST_CST, K);
}

/* ---------- Flat-subtree decode (contiguous output) ----------
 *
 * Reads n*D packed bits, looks up each D-bit code in c2s, writes the
 * resulting bytes to out[0..n).  D=4 has a SIMD path (AVX2 32-byte or
 * SSE 16-byte); all other D values use the per-D scalar unrolled
 * switch below.  D=2/3/5/6 require either per-byte variable shifts
 * (AVX2's _mm_srlv_*) or vpmultishiftqb (AVX-512 VBMI2) to build per-
 * byte codes efficiently, and the scalar unrolled forms win without
 * those.  AVX-512 VBMI2 D=5/6 fast paths live in primitives_avx512.h. */

/* Extract D bits at bit position `bit_pos` from `in`.  D <= 16. */
static inline uint32_t extract_D_bits_x86(const uint8_t *in,
                                            int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3;
    int bit_off  = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8)  val |= ((uint32_t)in[byte_idx + 1]) << 8;
    if (bit_off + D > 16) val |= ((uint32_t)in[byte_idx + 2]) << 16;
    return (val >> bit_off) & ((1u << D) - 1);
}

/* Per-D switch for D=2/3/5/6/7/8 (and any D > 8).  `DST(k)` is the
 * destination expression for output element k. */
#define X86_FLAT_UNPACK_SWITCH(DST)                                            \
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
        uint32_t code = extract_D_bits_x86(bm, i * D, D);                      \
        DST(i) = c2s[code];                                                    \
    }

/* merge_flat_x86_impl — write n D-bit symbols contiguously to
 * out[].  D=4 SIMD specialisation, scalar unrolled tail for everything
 * else. */
static inline void merge_flat_x86_impl(uint8_t *symbols, int n,
                                                  const uint8_t *bm, int D,
                                                  const uint8_t *c2s)
{
    if (D == 4) {
#ifdef PIVCO_HAS_AVX2
        /* AVX2 32-byte fast path: D=4 means c2s has 16 entries → fits
         * in a 128-bit lane, broadcast to both 256-bit lanes. */
        __m128i c2s_lo = _mm_loadu_si128((const __m128i *)c2s);
        __m256i c2s_v  = _mm256_broadcastsi128_si256(c2s_lo);
        __m128i lo_mask128 = _mm_set1_epi8(0x0F);
        int i = 0;
        for (; i + 32 <= n; i += 32) {
            __m128i raw   = _mm_loadu_si128((const __m128i *)(bm + (i >> 1)));
            __m128i lo    = _mm_and_si128(raw, lo_mask128);
            __m128i hi    = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask128);
            __m128i codes_lo = _mm_unpacklo_epi8(lo, hi);  /* codes 0..15  */
            __m128i codes_hi = _mm_unpackhi_epi8(lo, hi);  /* codes 16..31 */
            __m256i codes = _mm256_set_m128i(codes_hi, codes_lo);
            __m256i syms = _mm256_shuffle_epi8(c2s_v, codes);
            _mm256_storeu_si256((__m256i *)(symbols + i), syms);
        }
        /* 16-byte SSE fallback for the trailing < 32 elements. */
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_x86(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
#else
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_x86(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
#endif
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[i    ] = c2s[b & 0x0F];
            symbols[i + 1] = c2s[b >> 4];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_x86(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    /* D=2/3/5/6 SIMD paths via ryg's PSHUFB+PMULLO unpack (SSE4.1+, no AVX2
     * needed).  Replaced the previous AVX2 vpsrlvd-based unpack: ryg's path
     * measured uniformly faster on c3 (Ivy Bridge, no AVX2 — gains full SIMD
     * coverage instead of scalar), c4/c5 (Haswell/Cascade Lake, -21%),
     * c5a (Zen 2, -43% to -54%), c6a (Zen 3, -25%). */
    if (D == 2) {
        /* 16 codes/iter: D=2 unpack + 4-entry pshufb scatter. */
        __m128i c2s_vec = _mm_loadl_epi64((const __m128i *)c2s);  /* 4 entries */
        int i = 0;
#if defined(PIVCO_HAS_AVX2)
        /* AVX2: one vpsrlvd transpose unpack per 16 codes (terrelln PR #1),
         * ~1.3-1.9x faster than two ryg calls.  Reads exactly 4 bytes/iter, so
         * no over-read slop is needed beyond the last group. */
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d2_unpack_avx2(bm + ((i * 2) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
#else
        /* SSE4.1 fallback: 2x ryg D=2 unpack.  Each reads a 16-byte window
         * (slop), so stop the fast loop a few groups early. */
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i lo_codes = flat_d2_unpack_x86(bm + ((i      * 2) >> 3));
            __m128i hi_codes = flat_d2_unpack_x86(bm + (((i + 8) * 2) >> 3));
            __m128i codes    = _mm_unpacklo_epi64(lo_codes, hi_codes);
            __m128i syms     = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
#endif
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_x86(bm, i * 2, 2);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 3) {
        /* 8 codes/iter: ryg D=3 unpack + 8-entry pshufb scatter.  The unpack
         * reads a 16-byte window — keep a generous scalar tail. */
        __m128i c2s_vec = _mm_loadl_epi64((const __m128i *)c2s);  /* 8 entries */
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 8 <= fast_end; i += 8) {
            __m128i codes = flat_d3_unpack_x86(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storel_epi64((__m128i *)(symbols + i), syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_x86(bm, i * 3, 3);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 5) {
        /* 8 codes/iter: ryg D=5 unpack + 2-table pshufb scatter (codes 0-31).
         * pshufb on either table uses code&15; blend by bit 4. */
        __m128i lo = _mm_loadu_si128((const __m128i *)c2s);        /* c2s[0..15]  */
        __m128i hi = _mm_loadu_si128((const __m128i *)(c2s + 16)); /* c2s[16..31] */
        const __m128i b4 = _mm_set1_epi8(0x10);
        int i = 0;
        int fast_end = n >= 24 ? n - 24 : 0;
        for (; i + 8 <= fast_end; i += 8) {
            __m128i codes = flat_d5_unpack_x86(bm + ((i * 5) >> 3));
            __m128i rlo = _mm_shuffle_epi8(lo, codes);
            __m128i rhi = _mm_shuffle_epi8(hi, codes);
            __m128i sel = _mm_cmpeq_epi8(_mm_and_si128(codes, b4), b4);
            __m128i syms = _mm_blendv_epi8(rlo, rhi, sel);
            _mm_storel_epi64((__m128i *)(symbols + i), syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_x86(bm, i * 5, 5);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 6) {
        /* 8 codes/iter: ryg D=6 unpack + 4-table pshufb scatter (codes 0-63).
         * Four pshufb (code&15 into each quarter) then a 2-level blend by
         * bits 5,4 selects the right quarter. */
        __m128i t0 = _mm_loadu_si128((const __m128i *)c2s);
        __m128i t1 = _mm_loadu_si128((const __m128i *)(c2s + 16));
        __m128i t2 = _mm_loadu_si128((const __m128i *)(c2s + 32));
        __m128i t3 = _mm_loadu_si128((const __m128i *)(c2s + 48));
        const __m128i b4 = _mm_set1_epi8(0x10);
        const __m128i b5 = _mm_set1_epi8(0x20);
        int i = 0;
        int fast_end = n >= 24 ? n - 24 : 0;
        for (; i + 8 <= fast_end; i += 8) {
            __m128i codes = flat_d6_unpack_x86(bm + ((i * 6) >> 3));
            __m128i r0 = _mm_shuffle_epi8(t0, codes);
            __m128i r1 = _mm_shuffle_epi8(t1, codes);
            __m128i r2 = _mm_shuffle_epi8(t2, codes);
            __m128i r3 = _mm_shuffle_epi8(t3, codes);
            __m128i s4 = _mm_cmpeq_epi8(_mm_and_si128(codes, b4), b4);
            __m128i s5 = _mm_cmpeq_epi8(_mm_and_si128(codes, b5), b5);
            __m128i a = _mm_blendv_epi8(r0, r1, s4);  /* bit5=0: t0/t1 */
            __m128i b = _mm_blendv_epi8(r2, r3, s4);  /* bit5=1: t2/t3 */
            __m128i syms = _mm_blendv_epi8(a, b, s5);
            _mm_storel_epi64((__m128i *)(symbols + i), syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_x86(bm, i * 6, 6);
            symbols[i] = c2s[code];
        }
        return;
    }
    /* D=7 has no x86 SIMD path: pshufb is only 16-wide, so the 128-entry
     * scatter needs 8 sub-tables + a 3-level blend tree that measures no
     * faster than the scalar-unrolled inner (~0.38 ns/elem on Zen 3).  Unlike
     * NEON (vqtbl4) and AVX-512 (vpermi2b), x86 has no cheap wide table lookup.
     * Falls through to the scalar X86_FLAT_UNPACK_SWITCH below. */
#define DST_DIRECT(k) symbols[k]
    X86_FLAT_UNPACK_SWITCH(DST_DIRECT)
#undef DST_DIRECT
}

/* merge_flat_x86 — D-bit flat-subtree decode into a
 * contiguous output buffer.  AVX2 D=4 32-byte path or SSE D=4 16-byte
 * path; scalar unrolled tail for other D.  AVX-512 D=5/D=6 fast paths
 * live in primitives_avx512.h. */
static inline void merge_flat_x86(uint8_t *out, int n,
                                               const uint8_t *bm, int D,
                                               const uint8_t *c2s)
{
    PROF_TIC();
    merge_flat_x86_impl(out, n, bm, D, c2s);
    PROF_TOC(PROF_BU_MERGE_FLAT, n);
}

/* ---------- Encode primitives: rank-based encoding (8-bit in-order ranks) ----------
 * Partition 8-bit ranks against split_rank, a u8 port of the code_la partition
 * (part_core_x86): per-8-rank chunk, movemask routing mask, compress_tab pshufb,
 * 8-byte storel compaction.  No unsigned byte-compare on SSE, so the routing
 * mask uses the MIN trick: rank > thr  <=>  min(rank, thr+1) == thr+1. */
static uint8_t x86_pc8[256];
static uint8_t x86_ctab_r[256][16], x86_ctab_l[256][16];
static uint8_t x86_pre_r[9][256][16], x86_pre_l[9][256][16];
static int     x86_tabs_ready = 0;
static void x86_build_tabs(void)
{
    if (x86_tabs_ready) return;
    for (int m = 0; m < 256; m++) {
        x86_pc8[m] = (uint8_t)__builtin_popcount(m);
        memset(x86_ctab_r[m], 0x80, 16);
        memset(x86_ctab_l[m], 0x80, 16);
        int pr = 0, pl = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) x86_ctab_r[m][pr++] = (uint8_t)k;  /* right -> [0:n_right) */
            else              x86_ctab_l[m][pl++] = (uint8_t)k;  /* left  -> [0:n_left) */
        }
    }
    /* High-half (lanes 8..15) source positions pre-shifted to output offset
     * nlo, for the dense 16-wide compaction (min-merge with the low-half ctab). */
    for (int nlo = 0; nlo <= 8; nlo++) {
        for (int m = 0; m < 256; m++) {
            memset(x86_pre_r[nlo][m], 0x80, 16);
            memset(x86_pre_l[nlo][m], 0x80, 16);
            int pr = nlo, pl = nlo;
            for (int k = 0; k < 8; k++) {
                if (m & (1 << k)) x86_pre_r[nlo][m][pr++] = (uint8_t)(8 + k);
                else              x86_pre_l[nlo][m][pl++] = (uint8_t)(8 + k);
            }
        }
    }
    x86_tabs_ready = 1;
}

/* 8-bit mask of (rank > thr) for the 8 ranks in the low 8 lanes of `ids8`. */
static inline uint8_t x86_mask8(__m128i ids8, __m128i thr1)
{
    __m128i ge = _mm_cmpeq_epi8(_mm_min_epu8(ids8, thr1), thr1);
    return (uint8_t)_mm_movemask_epi8(ge);
}

/* Compact the 8 ranks in the low 8 lanes of `v` (chunk mask `m`): right ranks
 * to tmp[ro), left ranks in place to ranks[lo).  pshufb gathers each side
 * contiguously; storel writes exactly 8 bytes (the chunk), the (8 - popcount)
 * trailing zeros get overwritten by the next chunk's compaction.  The 8-byte
 * width (vs the code_la 16-byte store) keeps the in-place left write from
 * clobbering the next iter's not-yet-loaded ranks. */
/* Dense 16-wide compaction: one pshufb + one 16-byte store per side over all 16
 * ranks in `v`.  The low-half ctab (lanes 0..7 of chunk mlo) is min-merged with
 * the high-half pre table (lanes 8..15 of chunk mhi, pre-shifted to output
 * offset rlo): both use 0x80 fill, so min picks the real index at each output
 * lane.  Half the pshufb + store traffic of the per-8 form — the binding
 * resource on a port-bound SSE loop (SSE has no native byte-compress).  The
 * in-place left 16-byte store is safe: n_left <= j so n_left+16 <= j+16 = the next
 * iter's load, no clobber. */
#define X86_COMPACT16(v, mlo, mhi, rlo, ldst, rdst)                         \
    do {                                                                        \
        __m128i ridx_ = _mm_min_epu8(                                           \
            _mm_load_si128((const __m128i *)x86_ctab_r[mlo]),               \
            _mm_load_si128((const __m128i *)x86_pre_r[rlo][mhi]));          \
        _mm_storeu_si128((__m128i *)(tmp + (rdst)), _mm_shuffle_epi8((v), ridx_)); \
        int llo_ = 8 - (rlo);                                                   \
        __m128i lidx_ = _mm_min_epu8(                                           \
            _mm_load_si128((const __m128i *)x86_ctab_l[mlo]),               \
            _mm_load_si128((const __m128i *)x86_pre_l[llo_][mhi]));         \
        _mm_storeu_si128((__m128i *)(ranks + (ldst)), _mm_shuffle_epi8((v), lidx_)); \
    } while (0)

/* full: 16-wide dense compaction (one pshufb + one 16-byte store per side),
 * 32 ranks per main-loop iter.  Dense-16 beats the per-8 port (the u16 code_la
 * partition shape) by halving the pshufb + store count — the binding resource
 * on a port-bound SSE loop with no byte-compress.  Doing 32/iter then folds the
 * loop, mask-build and bitmap-write overhead in half again (~10-12% over the
 * 16/iter form on Zen 3 / Skylake-X / Haswell).  The pshufb + store traffic per
 * byte is unchanged — two X86_COMPACT16 per 32 ranks. */
static inline int part_full_x86(uint8_t *ranks, int n, uint8_t thr,
                                   uint8_t *bm, uint8_t *tmp)
{
    x86_build_tabs();
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    /* 32 ranks/iter: two SSE movemasks OR'd into a 32-bit routing mask (one
     * 4-byte bitmap write), two 16-wide compactions, POPCNT for the cursor
     * advance.  Both 16-byte halves are loaded before any in-place left store,
     * so the dense left write can't clobber an un-loaded rank. */
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1));
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1));
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        int r0 = x86_pc8[(uint8_t)mlo];
        X86_COMPACT16(v0, (uint8_t)mlo, (uint8_t)(mlo >> 8), r0, n_left, n_right);
        int nr01 = __builtin_popcount(mlo);
        n_right += nr01; n_left += 16 - nr01;
        int r2 = x86_pc8[(uint8_t)mhi];
        X86_COMPACT16(v1, (uint8_t)mhi, (uint8_t)(mhi >> 8), r2, n_left, n_right);
        int nr23 = __builtin_popcount(mhi);
        n_right += nr23; n_left += 16 - nr23;
    }
    /* 16-rank tail of the [32k, 32k+31] remainder: one movemask, one compaction. */
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        uint8_t mlo = (uint8_t)mm;
        uint8_t mhi = (uint8_t)(mm >> 8);
        memcpy(bm + (j >> 3), &mm, 2);
        int rlo = x86_pc8[mlo], rhi = x86_pc8[mhi];
        X86_COMPACT16(v, mlo, mhi, rlo, n_left, n_right);
        n_right += rlo + rhi;
        n_left += (16 - rlo - rhi);
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}

/* right/left/none: u8 port of part_core_x86 — stride-8, movemask mask,
 * compress_tab pshufb.  EMIT_RIGHT/EMIT_LEFT compile-time -> the unused side's
 * pshufb + store fold away. */
__attribute__((always_inline)) static inline
int part_core_x86(uint8_t *ranks, int n, uint8_t thr,
                     uint8_t *bm, uint8_t *tmp, int EMIT_RIGHT, int EMIT_LEFT)
{
    x86_build_tabs();
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    for (; j + 8 <= n; j += 8) {
        __m128i v = _mm_loadl_epi64((const __m128i *)(ranks + j));
        uint8_t m = x86_mask8(v, thr1);
        bm[j >> 3] = m;
        if (EMIT_RIGHT)
            _mm_storel_epi64((__m128i *)(tmp + n_right),
                _mm_shuffle_epi8(v, _mm_load_si128((const __m128i *)x86_ctab_r[m])));
        if (EMIT_LEFT)
            _mm_storel_epi64((__m128i *)(ranks + n_left),
                _mm_shuffle_epi8(v, _mm_load_si128((const __m128i *)x86_ctab_l[m])));
        int rc = x86_pc8[m];
        n_right += rc;
        n_left += 8 - rc;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7));
                       if (EMIT_RIGHT) tmp[n_right] = r; n_right++; }
        else         { if (EMIT_LEFT) ranks[n_left] = r; n_left++; }
    }
    return n_right;
}

/* Native u8 rank packers (SSE4.1).  The flat local code is (rank - base),
 * already a D-bit byte, so the byte-laid intermediate comes from a u8 load +
 * sub + mask — no u16 srli + saturating narrow.  The bit-stitch backend mirrors
 * the code_la pack_d{2,3,4,8}_sse_x86 helpers above. */

/* SSE4.1 D=2: 16 ranks -> 4 bytes.  _mm_maddubs_epi16 weighted pair-add
 * with weights {1, 4, 16, 64} (int8 max 127, so 64 fits). */
static inline int pack_d2_sse_x86(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    const __m128i weights = _mm_setr_epi8(1, 4, 16, 64, 1, 4, 16, 64,
                                           1, 4, 16, 64, 1, 4, 16, 64);
    const __m128i vb = _mm_set1_epi8((char)base);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i bytes = _mm_and_si128(
            _mm_sub_epi8(_mm_loadu_si128((const __m128i *)(ranks + i)), vb),
            _mm_set1_epi8(0x3));
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        __m128i step2 = _mm_hadd_epi16(step1, _mm_setzero_si128());
        __m128i out_bytes = _mm_packus_epi16(step2, _mm_setzero_si128());
        uint32_t packed4 = (uint32_t)_mm_cvtsi128_si32(out_bytes);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

/* SSE4.1 D=4: 16 ranks -> 8 bytes.  _mm_maddubs_epi16 with weights {1, 16}. */
static inline int pack_d4_sse_x86(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    const __m128i weights = _mm_setr_epi8(1, 16, 1, 16, 1, 16, 1, 16,
                                           1, 16, 1, 16, 1, 16, 1, 16);
    const __m128i vb = _mm_set1_epi8((char)base);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i bytes = _mm_and_si128(
            _mm_sub_epi8(_mm_loadu_si128((const __m128i *)(ranks + i)), vb),
            _mm_set1_epi8(0xF));
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        __m128i out_bytes = _mm_packus_epi16(step1, _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)(out + (i * 4 / 8)), out_bytes);
    }
    return i;
}

/* SSE4.1 D=8: 16 ranks -> 16 bytes, byte-aligned. */
static inline int pack_d8_sse_x86(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    const __m128i vb = _mm_set1_epi8((char)base);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm_storeu_si128((__m128i *)(out + i),
            _mm_sub_epi8(_mm_loadu_si128((const __m128i *)(ranks + i)), vb));
    }
    return i;
}

/* SSE4.1 D=3: 8 ranks -> 24 bits via _mm_mullo_epi32 multiply-as-shift.
 * SSE4.1 lacks _mm_sllv_epi32; multiplying uint32 by 2^k achieves the
 * same per-lane left shift. */
static inline int pack_d3_sse_x86(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    const __m128i mlo = _mm_setr_epi32(1, 8, 64, 512);
    const __m128i mhi = _mm_setr_epi32(4096, 32768, 262144, 2097152);
    const __m128i vb = _mm_set1_epi8((char)base);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v8 = _mm_and_si128(
            _mm_sub_epi8(_mm_loadl_epi64((const __m128i *)(ranks + i)), vb),
            _mm_set1_epi8(0x7));
        __m128i v = _mm_cvtepu8_epi16(v8);                 /* 8 ranks -> u16 */
        __m128i vlo = _mm_unpacklo_epi16(v, _mm_setzero_si128());
        __m128i vhi = _mm_unpackhi_epi16(v, _mm_setzero_si128());
        vlo = _mm_mullo_epi32(vlo, mlo);
        vhi = _mm_mullo_epi32(vhi, mhi);
        __m128i s = _mm_add_epi32(vlo, vhi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        uint32_t packed = (uint32_t)_mm_cvtsi128_si32(s);
        int bi = i * 3 / 8;
        out[bi    ] = (uint8_t)(packed      );
        out[bi + 1] = (uint8_t)(packed >>  8);
        out[bi + 2] = (uint8_t)(packed >> 16);
    }
    return i;
}

/* Dispatcher: native SIMD per-D path (mirrors pack_dN_x86) + scalar tail. */
static inline void pack_dN_x86(uint8_t *out, const uint8_t *ranks,
                                  int n, int D, uint8_t base)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;

    int i = 0;
    switch (D) {
    case 4: i = pack_d4_sse_x86(out, ranks, n, base); break;
    case 8: i = pack_d8_sse_x86(out, ranks, n, base); break;
#ifdef PIVCO_HAS_AVX2
    case 2: i = pack_d2_avx2_x86(out, ranks, n, base); break;
    case 3: i = pack_d3_avx2_x86(out, ranks, n, base); break;
    case 5: i = pack_d5_avx2_x86(out, ranks, n, base); break;
    case 6: i = pack_d6_avx2_x86(out, ranks, n, base); break;
    case 7: i = pack_d7_avx2_x86(out, ranks, n, base); break;
#else
    case 2: i = pack_d2_sse_x86(out, ranks, n, base); break;
    case 3: i = pack_d3_sse_x86(out, ranks, n, base); break;
    /* D=5,6,7 fall through to the scalar tail on SSE4.1-only hosts. */
#endif
    default: break;
    }

    if (i >= n) return;

    /* Scalar tail. */
    uint32_t mask = (1u << D) - 1;
    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = (uint32_t)(uint8_t)(ranks[i] - base) & mask;
        buf |= (uint64_t)local << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
}

/* ---------- Aliases consumed by codec.c ---------- */

#define PIVCO_PRIM_ALWAYS_INLINE __attribute__((always_inline)) static inline

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ codec_init_x86(); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint8_t *ranks, int n,
                                              const uint8_t *symbols,
                                              const uint8_t *sym_to_rank)
{ for (int i = 0; i < n; i++) ranks[i] = sym_to_rank[symbols[i]]; }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_full(uint8_t *ranks,
                                                      int n, uint8_t thr,
                                                      uint8_t *bm,
                                                      uint8_t *right_out)
{ return part_full_x86(ranks, n, thr, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_right(uint8_t *ranks,
                                                      int n, uint8_t thr,
                                                      uint8_t *bm,
                                                      uint8_t *right_out)
{ return part_core_x86(ranks, n, thr, bm, right_out, 1, 0); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_left(uint8_t *ranks,
                                                     int n, uint8_t thr,
                                                     uint8_t *bm)
{ return part_core_x86(ranks, n, thr, bm, NULL, 0, 1); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_none(uint8_t *ranks,
                                                     int n, uint8_t thr,
                                                     uint8_t *bm)
{ return part_core_x86(ranks, n, thr, bm, NULL, 0, 0); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_pack_dN(const uint8_t *ranks,
                                             int n, int D, uint8_t base,
                                             uint8_t *out_packed)
{ pack_dN_x86(out_packed, ranks, n, D, base); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat(uint8_t *out, int n,
                                                          const uint8_t *bm, int D,
                                                          const uint8_t *c2s)
{ merge_flat_x86(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_cst(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_cst_cst_x86(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_vec(const uint8_t *bm, int K,
                                                          uint8_t left_sym,
                                                          const uint8_t *right_buf,
                                                          uint8_t *out)
{ merge_cst_vec_x86(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_vec_cst(const uint8_t *bm, int K,
                                                           const uint8_t *left_buf,
                                                           uint8_t right_sym,
                                                           uint8_t *out)
{ merge_vec_cst_x86(bm, K, left_buf, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_vec_vec(const uint8_t *bm, int K,
                                               const uint8_t *left_buf,
                                               const uint8_t *right_buf,
                                               uint8_t *out)
{ merge_vec_vec_x86(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_X86_H */
