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
#include "pivco_huffman_x86_flat.h"     /* flat_d4_unpack_x86 */
#include "pivco_prof.h"

#include <smmintrin.h>                  /* SSE4.1 */
#include <immintrin.h>                  /* AVX/AVX2/AVX-512 umbrella;
                                         * gated paths drop out cleanly
                                         * on SSE-only builds. */
#include <stdint.h>
#include <string.h>

/* Backend lifecycle.  Lazily build the compress_tab + expand_tab pre-
 * bake tables that the x86 partition / tree_merge primitives index
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

/* tree_merge_x86 — SSE 8-byte chunk via pshufb on
 * _mm_unpacklo_epi64(L8, R8) with expand_tab[mask].  2x-unrolled
 * stride-16 main path: two independent 8-byte merges per iter so OOO
 * overlaps loads / shuffles / stores.  Only lc/rc cursor adds carry a
 * real dep, and that's short-latency add.  AVX-512 VBMI2 64-byte
 * vpexpandb fast path lives in primitives_avx512.h. */
static inline void tree_merge_x86(const uint8_t *bm, int K,
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
    PROF_TOC(PROF_BU_TREE_MERGE, K);
}

/* tree_merge_bcast_left_x86 — left input is a broadcast constant.
 * Same 2x-unrolled structure; the L lane is a duplicated 16-byte
 * register holding left_sym. */
static inline void tree_merge_bcast_left_x86(const uint8_t *bm, int K,
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
    PROF_TOC(PROF_BU_TREE_MERGE_BCAST_LEFT, K);
}

/* tree_merge_bcast_right_x86 — mirror of tree_merge_bcast_left_x86. */
static inline void tree_merge_bcast_right_x86(const uint8_t *bm, int K,
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
    PROF_TOC(PROF_BU_TREE_MERGE_BCAST_RIGHT, K);
}

/* merge_both_const_x86 — both inputs are constants.  vpblendvb-style:
 * for each bit in mask, output is right_sym or left_sym.  AVX2 widens
 * to 32 bytes per iter; SSE4.1 floor handles 16. */
static inline void merge_both_const_x86(const uint8_t *bm, int K,
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
    PROF_TOC(PROF_BU_MERGE_BOTH_CONST, K);
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

/* flat_decode_direct_x86_inner — write n D-bit symbols contiguously to
 * out[].  D=4 SIMD specialisation, scalar unrolled tail for everything
 * else. */
static inline void flat_decode_direct_x86_inner(uint8_t *symbols, int n,
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
#define DST_DIRECT(k) symbols[k]
    X86_FLAT_UNPACK_SWITCH(DST_DIRECT)
#undef DST_DIRECT
}

/* flat_decode_to_buffer_x86 — D-bit flat-subtree decode into a
 * contiguous output buffer.  AVX2 D=4 32-byte path or SSE D=4 16-byte
 * path; scalar unrolled tail for other D.  AVX-512 D=5/D=6 fast paths
 * live in primitives_avx512.h. */
static inline void flat_decode_to_buffer_x86(uint8_t *out, int n,
                                               const uint8_t *bm, int D,
                                               const uint8_t *c2s)
{
    PROF_TIC();
    flat_decode_direct_x86_inner(out, n, bm, D, c2s);
    PROF_TOC(PROF_BU_FLAT_DECODE, n);
}

/* ---------- Encode primitives (bitmap + partition) ----------
 *
 * Dense-codes mask build via the classic SSE movemask trick.
 * code_vec holds 8 left-aligned 16-bit Huffman codes.  At tree depth d,
 * bit d of the original code is at position (15 - d) of code_la; we
 * shift LEFT by d to move that bit to position 15 (= sign bit of each
 * int16 lane).  _mm_packs_epi16 with signed saturation then collapses
 * each int16 lane to an int8 byte where bit 7 is the sign bit, and
 * _mm_movemask_epi8 reads bit 7 of each byte into an 8-bit bitmask --
 * the per-element bit slice we want.
 *
 * Cost: vpsllw (1) + vpacksw (1) + vpmovmskb (1) = 3 SSE ops + 1 mask. */
static inline uint8_t enc_mask8_codes_la_x86(__m128i code_vec,
                                               __m128i shift_count)
{
    __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
    __m128i bytes   = _mm_packs_epi16(shifted, _mm_setzero_si128());
    return (uint8_t)_mm_movemask_epi8(bytes);
}

/* Stride-8 SIMD main path: load 8 left-aligned codes, build mask byte
 * via the dense movemask, partition the SAME register into left/right
 * halves using compress_tab[mask].  In-place write of the LEFT half
 * over codes_la; RIGHT half goes to tmp.  Per 8 elements: 1 vld, 3 SSE
 * mask ops, 2 vld (shuf), 2 pshufb, 2 vst.  Scalar tail handles the
 * residual 1..7 elements. */
static inline int build_bitmap_partition_x86(uint16_t *codes_la, int n,
                                               int depth,
                                               uint8_t *bm,
                                               uint16_t *tmp)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);

    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        uint8_t mask = enc_mask8_codes_la_x86(code_vec, shift_count);
        bm[j >> 3] = mask;

        const uint8_t *tab = compress_tab[mask];
        __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
        __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));
        __m128i right  = _mm_shuffle_epi8(code_vec, shuf_r);
        __m128i left   = _mm_shuffle_epi8(code_vec, shuf_l);
        int nr = compress_popcnt[mask];
        _mm_storeu_si128((__m128i *)(tmp      + n_right), right);
        _mm_storeu_si128((__m128i *)(codes_la + n_left ), left);
        n_right += nr;
        n_left  += (8 - nr);
    }

    /* Scalar tail.  Read all tail codes into a temporary before writing
     * back, since the in-place left write can overlap the read when
     * n_left + 8 > j. */
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

/* ---------- Encode primitives (init) ---------- */

/* enc_init_x86 — gather per-symbol left-aligned codes into codes_la.
 * Today this is a straight scalar loop; the compiler auto-vectorises
 * it well enough on AVX2 hosts that a hand-rolled vpermi2w / vpgatherq
 * version doesn't materially help (the LSU is the bottleneck either
 * way).  AVX-512 has an actual SIMD win via vpermi2w -- see
 * primitives_avx512.h. */
static inline void enc_init_x86(uint16_t *codes_la, int n,
                                  const uint8_t *symbols,
                                  const uint16_t *code_la_lut)
{
    for (int i = 0; i < n; i++) codes_la[i] = code_la_lut[symbols[i]];
}

/* ---------- Encode primitives (flat-subtree pack) ----------
 *
 * Per-D SIMD bit-pack helpers.  Each reads D-bit codes from codes_la
 * (each lane holds the left-aligned Huffman code -- bit-d of the
 * original code is at position 15-d) and packs them LSB-first into
 * the output byte stream.  Each helper OVERPACKS (processes
 * ceil(n / stride) * stride elements).  The dispatcher pack_dN_x86
 * below handles the residual scalar tail. */

#ifdef PIVCO_HAS_AVX2
/* AVX2 D=3,5,6,7: 8 codes per iter via __m256i uint64 lanes (4 lanes
 * × 2 halves = 8 codes; max shift 7*7 = 49 < 64). */
#define PACK_DN_AVX2_UNIFIED_X86(NAME, D_VAL, BITS_OUT)                        \
static inline int NAME(uint8_t *out, const uint16_t *codes_la,                 \
                       int n, int right_shift)                                 \
{                                                                              \
    static const int64_t shifts_lo[4] = {0,         D_VAL,   2*D_VAL, 3*D_VAL};\
    static const int64_t shifts_hi[4] = {4*D_VAL, 5*D_VAL, 6*D_VAL, 7*D_VAL};  \
    __m256i sl = _mm256_loadu_si256((const __m256i *)shifts_lo);               \
    __m256i sh = _mm256_loadu_si256((const __m256i *)shifts_hi);               \
    __m256i mask_vec = _mm256_set1_epi64x((1LL << D_VAL) - 1);                 \
    int i = 0;                                                                 \
    for (; i + 8 <= n; i += 8) {                                               \
        __m128i v16 = _mm_loadu_si128((const __m128i *)(codes_la + i));        \
        __m256i lo = _mm256_cvtepu16_epi64(v16);                               \
        __m256i hi = _mm256_cvtepu16_epi64(_mm_unpackhi_epi64(v16, v16));      \
        lo = _mm256_and_si256(_mm256_srli_epi64(lo, right_shift), mask_vec);   \
        hi = _mm256_and_si256(_mm256_srli_epi64(hi, right_shift), mask_vec);   \
        lo = _mm256_sllv_epi64(lo, sl);                                        \
        hi = _mm256_sllv_epi64(hi, sh);                                        \
        __m256i sum = _mm256_add_epi64(lo, hi);                                \
        __m128i s128 = _mm_add_epi64(_mm256_castsi256_si128(sum),              \
                                      _mm256_extracti128_si256(sum, 1));       \
        uint64_t packed = _mm_cvtsi128_si64(s128)                              \
                        + _mm_cvtsi128_si64(_mm_unpackhi_epi64(s128, s128));   \
        int bi = i * D_VAL / 8;                                                \
        memcpy(out + bi, &packed, (BITS_OUT + 7) / 8);                         \
    }                                                                          \
    return i;                                                                  \
}
PACK_DN_AVX2_UNIFIED_X86(pack_d3_avx2_x86, 3, 24)
PACK_DN_AVX2_UNIFIED_X86(pack_d5_avx2_x86, 5, 40)
PACK_DN_AVX2_UNIFIED_X86(pack_d6_avx2_x86, 6, 48)
PACK_DN_AVX2_UNIFIED_X86(pack_d7_avx2_x86, 7, 56)
#undef PACK_DN_AVX2_UNIFIED_X86
#endif  /* PIVCO_HAS_AVX2 */

/* SSE4.1 D=2: 16 codes -> 4 bytes.  _mm_maddubs_epi16 weighted pair-add
 * with weights {1, 4, 16, 64} (int8 max 127, so 64 fits). */
static inline int pack_d2_sse_x86(uint8_t *out, const uint16_t *codes_la,
                                    int n, int right_shift)
{
    const __m128i weights = _mm_setr_epi8(1, 4, 16, 64, 1, 4, 16, 64,
                                           1, 4, 16, 64, 1, 4, 16, 64);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(codes_la + i    ));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(codes_la + i + 8));
        v0 = _mm_srli_epi16(v0, right_shift);
        v1 = _mm_srli_epi16(v1, right_shift);
        v0 = _mm_and_si128(v0, _mm_set1_epi16(0x3));
        v1 = _mm_and_si128(v1, _mm_set1_epi16(0x3));
        __m128i bytes = _mm_packus_epi16(v0, v1);
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        __m128i step2 = _mm_hadd_epi16(step1, _mm_setzero_si128());
        __m128i out_bytes = _mm_packus_epi16(step2, _mm_setzero_si128());
        uint32_t packed4 = (uint32_t)_mm_cvtsi128_si32(out_bytes);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

/* SSE4.1 D=4: 16 codes -> 8 bytes.  _mm_maddubs_epi16 with weights {1, 16}. */
static inline int pack_d4_sse_x86(uint8_t *out, const uint16_t *codes_la,
                                    int n, int right_shift)
{
    const __m128i weights = _mm_setr_epi8(1, 16, 1, 16, 1, 16, 1, 16,
                                           1, 16, 1, 16, 1, 16, 1, 16);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(codes_la + i    ));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(codes_la + i + 8));
        v0 = _mm_srli_epi16(v0, right_shift);
        v1 = _mm_srli_epi16(v1, right_shift);
        v0 = _mm_and_si128(v0, _mm_set1_epi16(0xF));
        v1 = _mm_and_si128(v1, _mm_set1_epi16(0xF));
        __m128i bytes = _mm_packus_epi16(v0, v1);
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        __m128i out_bytes = _mm_packus_epi16(step1, _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)(out + (i * 4 / 8)), out_bytes);
    }
    return i;
}

/* SSE4.1 D=8: 16 codes -> 16 bytes, byte-aligned. */
static inline int pack_d8_sse_x86(uint8_t *out, const uint16_t *codes_la,
                                    int n, int right_shift)
{
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(codes_la + i    ));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(codes_la + i + 8));
        v0 = _mm_srli_epi16(v0, right_shift);
        v1 = _mm_srli_epi16(v1, right_shift);
        __m128i bytes = _mm_packus_epi16(v0, v1);
        _mm_storeu_si128((__m128i *)(out + i), bytes);
    }
    return i;
}

/* SSE4.1 D=3: 8 codes -> 24 bits via _mm_mullo_epi32 multiply-as-shift.
 * SSE4.1 lacks _mm_sllv_epi32; multiplying uint32 by 2^k achieves the
 * same per-lane left shift. */
static inline int pack_d3_sse_x86(uint8_t *out, const uint16_t *codes_la,
                                    int n, int right_shift)
{
    const __m128i mlo = _mm_setr_epi32(1, 8, 64, 512);
    const __m128i mhi = _mm_setr_epi32(4096, 32768, 262144, 2097152);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128((const __m128i *)(codes_la + i));
        v = _mm_srli_epi16(v, right_shift);
        v = _mm_and_si128(v, _mm_set1_epi16(0x7));
        __m128i vlo = _mm_unpacklo_epi16(v, _mm_setzero_si128());
        __m128i vhi = _mm_unpackhi_epi16(v, _mm_setzero_si128());
        vlo = _mm_mullo_epi32(vlo, mlo);
        vhi = _mm_mullo_epi32(vhi, mhi);
        __m128i s = _mm_add_epi32(vlo, vhi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        uint32_t packed = (uint32_t)_mm_cvtsi128_si32(s);
        int bi = i * 3 / 8;
        out[bi    ] = (uint8_t)(packed       );
        out[bi + 1] = (uint8_t)(packed >>  8);
        out[bi + 2] = (uint8_t)(packed >> 16);
    }
    return i;
}

/* Dispatcher: pack n D-bit codes from codes_la into out[].  Selects the
 * SIMD per-D pack helper, then handles any residual scalar tail.
 *
 * D=2/4/8 use SSE4.1 directly; D=3/5/6/7 use AVX2 sllv where available,
 * falling back to scalar on SSE4.1-only hosts (D=3 has an SSE multiply-
 * as-shift version, D=5/6/7 stay scalar since SSE has no uint64 per-
 * lane shift). */
static inline void pack_dN_x86(uint8_t *out, const uint16_t *codes_la,
                                 int n, int D, int depth)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int right_shift = 16 - depth - D;

    int i = 0;
    switch (D) {
    case 2: i = pack_d2_sse_x86(out, codes_la, n, right_shift); break;
    case 4: i = pack_d4_sse_x86(out, codes_la, n, right_shift); break;
    case 8: i = pack_d8_sse_x86(out, codes_la, n, right_shift); break;
#ifdef PIVCO_HAS_AVX2
    case 3: i = pack_d3_avx2_x86(out, codes_la, n, right_shift); break;
    case 5: i = pack_d5_avx2_x86(out, codes_la, n, right_shift); break;
    case 6: i = pack_d6_avx2_x86(out, codes_la, n, right_shift); break;
    case 7: i = pack_d7_avx2_x86(out, codes_la, n, right_shift); break;
#else
    case 3: i = pack_d3_sse_x86(out, codes_la, n, right_shift); break;
    /* D=5,6,7 fall through to scalar tail on SSE4.1-only hosts. */
#endif
    default: break;
    }

    int simd_n = i > n ? n : i;
    PROF_COUNT_ONLY(PROF_ENC_FLAT_SIMD_ELEMS, simd_n);
    PROF_COUNT_ONLY(PROF_ENC_FLAT_TAIL_ELEMS, n - simd_n);
    (void)simd_n;  /* unused when PIVCO_PROF=0 (PROF_COUNT_ONLY expands away) */

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
{ codec_init_x86(); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ enc_init_x86(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int prim_build_bitmap_partition(uint16_t *codes_la,
                                                           int n, int depth,
                                                           uint8_t *bm,
                                                           uint16_t *tmp)
{ return build_bitmap_partition_x86(codes_la, n, depth, bm, tmp); }

PIVCO_PRIM_ALWAYS_INLINE void prim_pack_dN(uint8_t *out,
                                             const uint16_t *codes_la,
                                             int n, int D, int depth)
{ pack_dN_x86(out, codes_la, n, D, depth); }

PIVCO_PRIM_ALWAYS_INLINE void prim_flat_decode_to_buffer(uint8_t *out, int n,
                                                          const uint8_t *bm, int D,
                                                          const uint8_t *c2s)
{ flat_decode_to_buffer_x86(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_both_const(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_both_const_x86(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge_bcast_left(const uint8_t *bm, int K,
                                                          uint8_t left_sym,
                                                          const uint8_t *right_buf,
                                                          uint8_t *out)
{ tree_merge_bcast_left_x86(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge_bcast_right(const uint8_t *bm, int K,
                                                           const uint8_t *left_buf,
                                                           uint8_t right_sym,
                                                           uint8_t *out)
{ tree_merge_bcast_right_x86(bm, K, left_buf, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_tree_merge(const uint8_t *bm, int K,
                                               const uint8_t *left_buf,
                                               const uint8_t *right_buf,
                                               uint8_t *out)
{ tree_merge_x86(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_X86_H */
