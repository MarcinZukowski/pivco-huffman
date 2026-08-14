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
#include "pivco_check.h"

/* Backend lifecycle.  Lazily build the compress_tab + expand_tab pre-
 * bake tables that the x86 partition / merge primitives index
 * into.  Idempotent and cheap after the first call. */
static void init_x86_merge_tables(void);   /* two-table merge_vec_vec shuffles (below) */
static inline void codec_init_x86(void)
{
    init_compress_table_x86();
    init_expand_table_x86();
    init_x86_merge_tables();
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

/* ---- merge_vec_vec_x86: two-table merge (PSHUFB-complement + OR) ----
 *
 * x86 has no 2-source PSHUFB, but PSHUFB zeroes any lane whose index MSB is
 * set: shuffle R with the merged index (R lanes valid, L lanes -> 0 via the
 * 255-off indices), shuffle L with the complemented index (L valid, R -> 0),
 * then OR.  The high half's +pop0 offset is folded in by replicating pop0
 * across shuf0's top 8 bytes and adding shuf1 (L entries stored 247-off).
 * Two index tables (g_x86_merge_shuf0[.][16] + g_x86_merge_shuf1[.][8]) are
 * built once in codec_init_x86.  With AVX2 the main loop runs 32 B/iter as two
 * 128-bit lanes (broadcast + vpblendd index assembly, asm-pinned for
 * llvm#203132); the SSE 16 B form handles the residual (and all of K on
 * SSE4.1-only hosts).  AVX-512 VBMI2 uses the vpexpandb path in
 * primitives_avx512.h instead. */
static uint8_t g_x86_merge_shuf0[256][16] __attribute__((aligned(16)));
static uint8_t g_x86_merge_shuf1[256][8];
static void init_x86_merge_tables(void)
{
    static int built = 0;
    if (built) return;
    for (int m = 0; m < 256; m++) {
        int rset = 0, rclr = 0, pop = __builtin_popcount(m);
        for (int i = 0; i < 8; i++)
            g_x86_merge_shuf0[m][i] = ((m >> i) & 1) ? (uint8_t)(rset++)
                                                     : (uint8_t)(255 - rclr++);
        for (int i = 8; i < 16; i++) g_x86_merge_shuf0[m][i] = (uint8_t)pop;
        rset = 0; rclr = 0;
        for (int t = 0; t < 8; t++)
            g_x86_merge_shuf1[m][t] = ((m >> t) & 1) ? (uint8_t)(rset++)
                                                     : (uint8_t)(247 - rclr++);
    }
    built = 1;
}
#if defined(__AVX2__)
static inline __m256i x86_merge_bcastq(const void *src)
{ return _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)src)); }
static inline __m256i x86_merge_load_halves(const void *s0, const void *s1)
{
    __m256i v = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)s0));
    return _mm256_inserti128_si256(v, _mm_loadu_si128((const __m128i *)s1), 1);
}
#endif
static inline void merge_vec_vec_x86(const uint8_t *bm, int K,
                                    const uint8_t *left,
                                    const uint8_t *right,
                                    uint8_t *out)
{
    PROF_TIC();
    int lc = 0, rc = 0, j = 0;
#if defined(__AVX2__)
    {
        const __m256i ones = _mm256_set1_epi8(-1), zeros = _mm256_setzero_si256();
        for (; j + 32 <= K; j += 32) {
            uint32_t mask; memcpy(&mask, bm + (j >> 3), 4);
            unsigned m0 = mask & 0xff, m1 = (mask >> 8) & 0xff,
                     m2 = (mask >> 16) & 0xff, m3 = (mask >> 24) & 0xff;
            __m256i vShuf02 = x86_merge_load_halves(g_x86_merge_shuf0[m0], g_x86_merge_shuf0[m2]);
            __m256i vShuf1  = x86_merge_bcastq(g_x86_merge_shuf1[m1]);
            __m256i vShuf3  = x86_merge_bcastq(g_x86_merge_shuf1[m3]);
            __asm__("" : "+x"(vShuf1)); __asm__("" : "+x"(vShuf3));   /* dodge llvm#203132 */
            __m256i vShuf13  = _mm256_blend_epi32(vShuf1, vShuf3, 0xf0);
            __m256i vShuf13M = _mm256_blend_epi32(zeros, vShuf13, 0xcc);
            __m256i vShuf    = _mm256_add_epi8(vShuf02, vShuf13M);
            int lo_pop = _mm_popcnt_u32(mask & 0xffff);
            __m256i vR = x86_merge_load_halves(right + rc, right + rc + lo_pop);
            __m256i vL = x86_merge_load_halves(left  + lc, left  + lc + 16 - lo_pop);
            __m256i rr = _mm256_shuffle_epi8(vR, vShuf);
            __m256i rl = _mm256_shuffle_epi8(vL, _mm256_xor_si256(vShuf, ones));
            _mm256_storeu_si256((__m256i *)(out + j), _mm256_or_si256(rl, rr));
            int pr = _mm_popcnt_u32(mask);
            rc += pr; lc += 32 - pr;
        }
    }
#endif
    {
        const __m128i ones = _mm_set1_epi8(-1);
        for (; j + 16 <= K; j += 16) {
            unsigned lo = bm[j >> 3], hi = bm[(j >> 3) + 1];
            __m128i shuf0 = _mm_load_si128((const __m128i *)g_x86_merge_shuf0[lo]);
            __m128i shuf1 = _mm_slli_si128(_mm_loadl_epi64((const __m128i *)g_x86_merge_shuf1[hi]), 8);
            __m128i merged = _mm_add_epi8(shuf0, shuf1);
            __m128i R16 = _mm_loadu_si128((const __m128i *)(right + rc));
            __m128i L16 = _mm_loadu_si128((const __m128i *)(left  + lc));
            __m128i rr = _mm_shuffle_epi8(R16, merged);
            __m128i rl = _mm_shuffle_epi8(L16, _mm_xor_si128(merged, ones));
            _mm_storeu_si128((__m128i *)(out + j), _mm_or_si128(rr, rl));
            int pr = __builtin_popcount(lo) + __builtin_popcount(hi);
            rc += pr; lc += 16 - pr;
        }
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
/* merge_cst_vec_x86 — two-table merge, L = broadcast const (no L load/cursor);
 * only R advances.  AVX2 32B main + SSE 16B residual.  See merge_vec_vec_x86. */
static inline void merge_cst_vec_x86(const uint8_t *bm, int K,
                                     uint8_t left_sym,
                                     const uint8_t *right,
                                     uint8_t *out)
{
    PROF_TIC();
    int rc = 0, j = 0;
#if defined(__AVX2__)
    {
        const __m256i ones = _mm256_set1_epi8(-1), zeros = _mm256_setzero_si256();
        const __m256i vLb = _mm256_set1_epi8((char)left_sym);
        for (; j + 32 <= K; j += 32) {
            uint32_t mask; memcpy(&mask, bm + (j >> 3), 4);
            unsigned m0=mask&0xff, m1=(mask>>8)&0xff, m2=(mask>>16)&0xff, m3=(mask>>24)&0xff;
            __m256i vShuf02 = x86_merge_load_halves(g_x86_merge_shuf0[m0], g_x86_merge_shuf0[m2]);
            __m256i vShuf1  = x86_merge_bcastq(g_x86_merge_shuf1[m1]);
            __m256i vShuf3  = x86_merge_bcastq(g_x86_merge_shuf1[m3]);
            __asm__("" : "+x"(vShuf1)); __asm__("" : "+x"(vShuf3));
            __m256i vShuf13  = _mm256_blend_epi32(vShuf1, vShuf3, 0xf0);
            __m256i vShuf13M = _mm256_blend_epi32(zeros, vShuf13, 0xcc);
            __m256i vShuf    = _mm256_add_epi8(vShuf02, vShuf13M);
            int lo_pop = _mm_popcnt_u32(mask & 0xffff);
            __m256i vR = x86_merge_load_halves(right + rc, right + rc + lo_pop);
            __m256i rr = _mm256_shuffle_epi8(vR,  vShuf);
            __m256i rl = _mm256_shuffle_epi8(vLb, _mm256_xor_si256(vShuf, ones));
            _mm256_storeu_si256((__m256i *)(out + j), _mm256_or_si256(rl, rr));
            rc += _mm_popcnt_u32(mask);
        }
    }
#endif
    {
        const __m128i ones = _mm_set1_epi8(-1), Lb = _mm_set1_epi8((char)left_sym);
        for (; j + 16 <= K; j += 16) {
            unsigned lo = bm[j >> 3], hi = bm[(j >> 3) + 1];
            __m128i shuf0 = _mm_load_si128((const __m128i *)g_x86_merge_shuf0[lo]);
            __m128i shuf1 = _mm_slli_si128(_mm_loadl_epi64((const __m128i *)g_x86_merge_shuf1[hi]), 8);
            __m128i merged = _mm_add_epi8(shuf0, shuf1);
            __m128i R16 = _mm_loadu_si128((const __m128i *)(right + rc));
            __m128i rr = _mm_shuffle_epi8(R16, merged);
            __m128i rl = _mm_shuffle_epi8(Lb, _mm_xor_si128(merged, ones));
            _mm_storeu_si128((__m128i *)(out + j), _mm_or_si128(rr, rl));
            rc += __builtin_popcount(lo) + __builtin_popcount(hi);
        }
    }
    for (; j < K; j++) { int mb = (bm[j >> 3] >> (j & 7)) & 1; out[j] = mb ? right[rc++] : left_sym; }
    PROF_TOC(PROF_BU_MERGE_CST_VEC, K);
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

/* Generic scalar mop-up shared by every merge_flat_dN_x86: decode codes
 * [i, n) one at a time.  D is always a literal at the call sites, so it
 * constant-folds. */
static inline void merge_flat_tail_x86(uint8_t *symbols, int i, int n,
                                       const uint8_t *bm, int D,
                                       const uint8_t *c2s)
{
    for (; i < n; i++)
        symbols[i] = c2s[extract_D_bits_x86(bm, i * D, D)];
}

/* merge_flat_dN_x86 — one static inline per supported D (mirrors the NEON
 * file's structure); merge_flat_x86 below dispatches.  Each writes n D-bit
 * symbols contiguously to symbols[]. */

/* D=2: 16 codes/iter, unpack + 4-entry pshufb scatter. */
static inline void merge_flat_d2_x86(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
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
    /* SSE4.1: TL/TH prepped nibble tables (issue #5, dougallj x86 port):
     * TL[nib]=c2s[nib&3], TH[nib]=c2s[(nib>>2)&3] map input nibbles straight
     * to symbol pairs -- no unpack pass; 64 codes/iter, the 4-way interleave
     * is a 2-level punpck tree.  (On AVX2 builds the vpsrlvd unpack above is
     * faster on Intel; this form wins on AMD too -- vendor-dispatch note in
     * IDEAS "enc_init 4tab / bc2".) */
    if (n >= 64) {
        uint32_t w; memcpy(&w, c2s, 4);
        const __m128i TL = _mm_set1_epi32((int)w);                       /* c2s[nib&3] */
        const __m128i TH = _mm_shuffle_epi8(TL,
            _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3));            /* c2s[(nib>>2)&3] */
        const __m128i m  = _mm_set1_epi8(0x0F);
        for (; i + 64 <= n; i += 64) {
            __m128i v  = _mm_loadu_si128((const __m128i *)(bm + (i >> 2)));
            __m128i lo = _mm_and_si128(v, m);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), m);
            __m128i a = _mm_shuffle_epi8(TL, lo);   /* code0 of each byte */
            __m128i b = _mm_shuffle_epi8(TH, lo);   /* code1 */
            __m128i c = _mm_shuffle_epi8(TL, hi);   /* code2 */
            __m128i d = _mm_shuffle_epi8(TH, hi);   /* code3 */
            __m128i ab_lo = _mm_unpacklo_epi8(a, b), ab_hi = _mm_unpackhi_epi8(a, b);
            __m128i cd_lo = _mm_unpacklo_epi8(c, d), cd_hi = _mm_unpackhi_epi8(c, d);
            _mm_storeu_si128((__m128i *)(symbols + i),      _mm_unpacklo_epi16(ab_lo, cd_lo));
            _mm_storeu_si128((__m128i *)(symbols + i + 16), _mm_unpackhi_epi16(ab_lo, cd_lo));
            _mm_storeu_si128((__m128i *)(symbols + i + 32), _mm_unpacklo_epi16(ab_hi, cd_hi));
            _mm_storeu_si128((__m128i *)(symbols + i + 48), _mm_unpackhi_epi16(ab_hi, cd_hi));
        }
    }
    /* 2x ryg D=2 unpack remainder.  Each reads a 16-byte window (slop), so
     * stop the fast loop a few groups early. */
    int fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        __m128i lo_codes = flat_d2_unpack_x86(bm + ((i      * 2) >> 3));
        __m128i hi_codes = flat_d2_unpack_x86(bm + (((i + 8) * 2) >> 3));
        __m128i codes    = _mm_unpacklo_epi64(lo_codes, hi_codes);
        __m128i syms     = _mm_shuffle_epi8(c2s_vec, codes);
        _mm_storeu_si128((__m128i *)(symbols + i), syms);
    }
#endif
    merge_flat_tail_x86(symbols, i, n, bm, 2, c2s);
}

/* D=3: 32 codes/iter 6-bit pair-gather (issue #5, dougallj; x86 port of the
 * NEON kernel): one pshufb positions two adjacent 3-bit codes per byte, the
 * bidirectional u16 shift is pmullw-as-shift + one uniform psrlw, and the
 * 8-entry c2s is duplicated so any 4-bit index works.  Falls through to the
 * stock 8-wide ryg path + scalar tail for the remainder. */
static inline void merge_flat_d3_x86(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    int i = 0;
    int fast_end = n >= 16 ? n - 16 : 0;
    if (n >= 48) {
        const uint8_t *bp = bm;
        __m128i c2s8  = _mm_loadl_epi64((const __m128i *)c2s);
        __m128i c2s16 = _mm_unpacklo_epi64(c2s8, c2s8);                  /* 8 entries x2 */
        const __m128i pair6_shuf = _mm_setr_epi8(0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11);
        const __m128i mul6     = _mm_setr_epi16(16,1, 16,1, 16,1, 16,1); /* <<(4-o) */
        const __m128i m3f_even = _mm_set1_epi16(0x003F);
        const __m128i m3f_odd  = _mm_set1_epi16(0x3F00);
        const __m128i m7       = _mm_set1_epi8(7);
        for (; i + 32 <= fast_end; i += 32, bp += 12) {
            __m128i packed = _mm_loadu_si128((const __m128i *)bp);
            __m128i x = _mm_mullo_epi16(_mm_shuffle_epi8(packed, pair6_shuf), mul6);
            /* 12-bit group at bits 4..15: even 6-bit half at 4..9, odd at 10..15 */
            __m128i pair6 = _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(x, 4), m3f_even),
                _mm_and_si128(_mm_srli_epi16(x, 2), m3f_odd));
            __m128i lo3 = _mm_and_si128(pair6, m7);
            __m128i hi3 = _mm_and_si128(_mm_srli_epi16(pair6, 3), m7);
            __m128i s_lo = _mm_shuffle_epi8(c2s16, lo3);
            __m128i s_hi = _mm_shuffle_epi8(c2s16, hi3);
            _mm_storeu_si128((__m128i *)(symbols + i),      _mm_unpacklo_epi8(s_lo, s_hi));
            _mm_storeu_si128((__m128i *)(symbols + i + 16), _mm_unpackhi_epi8(s_lo, s_hi));
        }
    }
    /* stock 8-wide ryg path + scalar tail */
    __m128i c2s_vec = _mm_loadl_epi64((const __m128i *)c2s);  /* 8 entries */
    for (; i + 8 <= fast_end; i += 8) {
        __m128i codes = flat_d3_unpack_x86(bm + ((i * 3) >> 3));
        __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
        _mm_storel_epi64((__m128i *)(symbols + i), syms);
    }
    merge_flat_tail_x86(symbols, i, n, bm, 3, c2s);
}

/* D=4: nibble codes, 32/iter (AVX2) or 16/iter (SSE). */
static inline void merge_flat_d4_x86(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
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
    merge_flat_tail_x86(symbols, i, n, bm, 4, c2s);
}

/* D=5: 16 codes/iter pair-gather (issue #5, dougallj; x86 port of the NEON
 * kernel): one pshufb gathers two adjacent 5-bit codes into each u16 lane,
 * pmullw-as-shift aligns the 10-bit pair to the lane top, and two shift+mask
 * place code0/code1 in the even/odd byte -- the interleave is free.  The
 * 32-entry scatter is 2 pshufb + blendv, with bit 4 moved to the sign bit by
 * one psllw (idx bytes are pre-masked, so the cross-byte spill is clean).
 * Falls through to the stock 8-wide ryg path + scalar tail. */
/* ---- AVX2 ymm widening of the D=5/6/7 flat decoders ----
 * 32 codes/iter: each 128-bit lane runs the SSE kernel's constants on its
 * own packed window (lane1 loads at +S bytes, S = 10/12/14); c2s tables
 * broadcast per lane; blendv select tree is byte-wise so lanes never
 * interact.  Returns codes decoded (multiple of 32); the callers shift
 * their frame and let the 128-bit body + scalar tail finish.  Measured
 * 1.5-1.9x over the 128-bit kernels on c4/c5/c5a/c6a (bench_prim). */
#ifdef PIVCO_HAS_AVX2
static inline __m256i flat_bc128_x86(const uint8_t *p)
{
    return _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)p));
}
static inline __m256i flat_load2_x86(const uint8_t *p, int stride)
{
    return _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)p)),
        _mm_loadu_si128((const __m128i *)(p + stride)), 1);
}

static inline int merge_flat_d5_ymm_x86(uint8_t *symbols, int n,
                                        const uint8_t *bm, const uint8_t *c2s)
{
    const __m256i lo = flat_bc128_x86(c2s), hi = flat_bc128_x86(c2s + 16);
    const __m256i pair5_shuf = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0,1, 1,2, 2,3, 3,4, 5,6, 6,7, 7,8, 8,9));
    const __m256i mul5     = _mm256_set1_epi64x(0x0001000400100040ll); /* {64,16,4,1} */
    const __m256i m1f_even = _mm256_set1_epi16(0x001F);
    const __m256i m1f_odd  = _mm256_set1_epi16(0x1F00);
    int pb = (5 * n) >> 3;
    int blocks = pb >= 26 ? (pb - 26) / 20 + 1 : 0;
    if (blocks > (n >> 5)) blocks = n >> 5;
    for (int b = 0; b < blocks; ++b) {
        __m256i packed = flat_load2_x86(bm + b * 20, 10);
        __m256i x = _mm256_mullo_epi16(_mm256_shuffle_epi8(packed, pair5_shuf), mul5);
        __m256i idx = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(x, 6), m1f_even),
            _mm256_and_si256(_mm256_srli_epi16(x, 3), m1f_odd));
        __m256i rlo = _mm256_shuffle_epi8(lo, idx);
        __m256i rhi = _mm256_shuffle_epi8(hi, idx);
        __m256i sel = _mm256_slli_epi16(idx, 3);   /* bit4 -> sign bit */
        _mm256_storeu_si256((__m256i *)(symbols + (b << 5)),
                            _mm256_blendv_epi8(rlo, rhi, sel));
    }
    return blocks << 5;
}

static inline int merge_flat_d6_ymm_x86(uint8_t *symbols, int n,
                                        const uint8_t *bm, const uint8_t *c2s)
{
    const __m256i t0 = flat_bc128_x86(c2s),      t1 = flat_bc128_x86(c2s + 16);
    const __m256i t2 = flat_bc128_x86(c2s + 32), t3 = flat_bc128_x86(c2s + 48);
    const __m256i pair6_shuf = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11));
    const __m256i mul6     = _mm256_set1_epi32(0x00010010);            /* {16,1} */
    const __m256i m3f_even = _mm256_set1_epi16(0x003F);
    const __m256i m3f_odd  = _mm256_set1_epi16(0x3F00);
    int pb = (6 * n) >> 3;
    int blocks = pb >= 28 ? (pb - 28) / 24 + 1 : 0;
    if (blocks > (n >> 5)) blocks = n >> 5;
    for (int b = 0; b < blocks; ++b) {
        __m256i packed = flat_load2_x86(bm + b * 24, 12);
        __m256i x = _mm256_mullo_epi16(_mm256_shuffle_epi8(packed, pair6_shuf), mul6);
        __m256i idx = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(x, 4), m3f_even),
            _mm256_and_si256(_mm256_srli_epi16(x, 2), m3f_odd));
        __m256i r0 = _mm256_shuffle_epi8(t0, idx);
        __m256i r1 = _mm256_shuffle_epi8(t1, idx);
        __m256i r2 = _mm256_shuffle_epi8(t2, idx);
        __m256i r3 = _mm256_shuffle_epi8(t3, idx);
        __m256i s4 = _mm256_slli_epi16(idx, 3);   /* bit4 -> sign bit */
        __m256i s5 = _mm256_slli_epi16(idx, 2);   /* bit5 -> sign bit */
        __m256i a  = _mm256_blendv_epi8(r0, r1, s4);
        __m256i b2 = _mm256_blendv_epi8(r2, r3, s4);
        _mm256_storeu_si256((__m256i *)(symbols + (b << 5)),
                            _mm256_blendv_epi8(a, b2, s5));
    }
    return blocks << 5;
}

static inline int merge_flat_d7_ymm_x86(uint8_t *symbols, int n,
                                        const uint8_t *bm, const uint8_t *c2s)
{
    const __m256i t0 = flat_bc128_x86(c2s),      t1 = flat_bc128_x86(c2s + 16);
    const __m256i t2 = flat_bc128_x86(c2s + 32), t3 = flat_bc128_x86(c2s + 48);
    const __m256i t4 = flat_bc128_x86(c2s + 64), t5 = flat_bc128_x86(c2s + 80);
    const __m256i t6 = flat_bc128_x86(c2s + 96), t7 = flat_bc128_x86(c2s + 112);
    const __m256i g_lo = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0,1,2,3, 1,2,3,4, 3,4,5,6, 5,6,7,8));
    const __m256i g_hi = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(7,8,9,10, 8,9,10,11, 10,11,12,13, 12,13,14,15));
    const __m256i mul7 = _mm256_broadcastsi128_si256(
        _mm_setr_epi32(64,1,4,16));   /* <<(6-o), o={0,6,4,2} */
    const __m256i m7f_even = _mm256_set1_epi32(0x0000007F);
    const __m256i m7f_odd  = _mm256_set1_epi32(0x00007F00);
    int pb = (7 * n) >> 3;
    int blocks = pb >= 30 ? (pb - 30) / 28 + 1 : 0;
    if (blocks > (n >> 5)) blocks = n >> 5;
    for (int b = 0; b < blocks; ++b) {
        __m256i packed = flat_load2_x86(bm + b * 28, 14);
        __m256i xl = _mm256_mullo_epi32(_mm256_shuffle_epi8(packed, g_lo), mul7);
        __m256i xh = _mm256_mullo_epi32(_mm256_shuffle_epi8(packed, g_hi), mul7);
        __m256i cl = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi32(xl, 6), m7f_even),
            _mm256_and_si256(_mm256_srli_epi32(xl, 5), m7f_odd));
        __m256i ch = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi32(xh, 6), m7f_even),
            _mm256_and_si256(_mm256_srli_epi32(xh, 5), m7f_odd));
        __m256i idx = _mm256_packus_epi32(cl, ch);   /* per-lane, in order */
        __m256i r0 = _mm256_shuffle_epi8(t0, idx);
        __m256i r1 = _mm256_shuffle_epi8(t1, idx);
        __m256i r2 = _mm256_shuffle_epi8(t2, idx);
        __m256i r3 = _mm256_shuffle_epi8(t3, idx);
        __m256i r4 = _mm256_shuffle_epi8(t4, idx);
        __m256i r5 = _mm256_shuffle_epi8(t5, idx);
        __m256i r6 = _mm256_shuffle_epi8(t6, idx);
        __m256i r7 = _mm256_shuffle_epi8(t7, idx);
        __m256i s4 = _mm256_slli_epi16(idx, 3);   /* bit4 -> sign bit */
        __m256i s5 = _mm256_slli_epi16(idx, 2);   /* bit5 -> sign bit */
        __m256i s6 = _mm256_slli_epi16(idx, 1);   /* bit6 -> sign bit */
        __m256i a0 = _mm256_blendv_epi8(r0, r1, s4);
        __m256i a1 = _mm256_blendv_epi8(r2, r3, s4);
        __m256i a2 = _mm256_blendv_epi8(r4, r5, s4);
        __m256i a3 = _mm256_blendv_epi8(r6, r7, s4);
        __m256i b0 = _mm256_blendv_epi8(a0, a1, s5);
        __m256i b1 = _mm256_blendv_epi8(a2, a3, s5);
        _mm256_storeu_si256((__m256i *)(symbols + (b << 5)),
                            _mm256_blendv_epi8(b0, b1, s6));
    }
    return blocks << 5;
}
#endif /* PIVCO_HAS_AVX2 */

static inline void merge_flat_d5_x86(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
#ifdef PIVCO_HAS_AVX2
    {   /* ymm blocks first; shift the frame so the 128-bit body and the
         * scalar tail below run unchanged on the remainder. */
        int done = merge_flat_d5_ymm_x86(symbols, n, bm, c2s);
        symbols += done; bm += (done * 5) >> 3; n -= done;
    }
#endif
    /* pshufb on either table uses code&15; blend by bit 4. */
    __m128i lo = _mm_loadu_si128((const __m128i *)c2s);        /* c2s[0..15]  */
    __m128i hi = _mm_loadu_si128((const __m128i *)(c2s + 16)); /* c2s[16..31] */
    int i = 0;
    if (n >= 25) {
        const __m128i pair5_shuf = _mm_setr_epi8(0,1, 1,2, 2,3, 3,4, 5,6, 6,7, 7,8, 8,9);
        const __m128i mul5       = _mm_setr_epi16(64,16,4,1, 64,16,4,1);  /* <<(6-o) */
        const __m128i m1f_even   = _mm_set1_epi16(0x001F);
        const __m128i m1f_odd    = _mm_set1_epi16(0x1F00);
        int blocks = (n - 9) >> 4;
        for (int b = 0; b < blocks; ++b) {
            __m128i packed = _mm_loadu_si128((const __m128i *)(bm + b * 10));
            __m128i x = _mm_mullo_epi16(_mm_shuffle_epi8(packed, pair5_shuf), mul5);
            /* code0 at bits 6..10, code1 at bits 11..15 of each u16 */
            __m128i idx = _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(x, 6), m1f_even),
                _mm_and_si128(_mm_srli_epi16(x, 3), m1f_odd));
            __m128i rlo = _mm_shuffle_epi8(lo, idx);
            __m128i rhi = _mm_shuffle_epi8(hi, idx);
            __m128i sel = _mm_slli_epi16(idx, 3);   /* bit4 -> sign bit */
            _mm_storeu_si128((__m128i *)(symbols + (b << 4)),
                             _mm_blendv_epi8(rlo, rhi, sel));
        }
        i = blocks << 4;
    }
    const __m128i b4 = _mm_set1_epi8(0x10);
    int fast_end = n >= 24 ? n - 24 : 0;
    for (; i + 8 <= fast_end; i += 8) {
        __m128i codes = flat_d5_unpack_x86(bm + ((i * 5) >> 3));
        __m128i rlo = _mm_shuffle_epi8(lo, codes);
        __m128i rhi = _mm_shuffle_epi8(hi, codes);
        __m128i sel = _mm_cmpeq_epi8(_mm_and_si128(codes, b4), b4);
        __m128i syms = _mm_blendv_epi8(rlo, rhi, sel);
        _mm_storel_epi64((__m128i *)(symbols + i), syms);
    }
    merge_flat_tail_x86(symbols, i, n, bm, 5, c2s);
}

/* D=6: 16 codes/iter pair-gather (issue #5, dougallj; x86 port of the NEON
 * kernel): same gather as D=5 but on 12-bit pairs (the shuffle/mul constants
 * match the D=3 pair-gather, which works the same 6-bit grid).  The 64-entry
 * scatter is 4 pshufb + a 2-level blend with bits 4/5 psllw'd to the sign
 * bit.  Falls through to the stock 8-wide ryg path + scalar tail. */
static inline void merge_flat_d6_x86(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
#ifdef PIVCO_HAS_AVX2
    {   /* ymm blocks first; shift the frame so the 128-bit body and the
         * scalar tail below run unchanged on the remainder. */
        int done = merge_flat_d6_ymm_x86(symbols, n, bm, c2s);
        symbols += done; bm += (done * 6) >> 3; n -= done;
    }
#endif
    /* four pshufb (code&15 into each quarter) then a 2-level blend by
     * bits 5,4 selects the right quarter. */
    __m128i t0 = _mm_loadu_si128((const __m128i *)c2s);
    __m128i t1 = _mm_loadu_si128((const __m128i *)(c2s + 16));
    __m128i t2 = _mm_loadu_si128((const __m128i *)(c2s + 32));
    __m128i t3 = _mm_loadu_si128((const __m128i *)(c2s + 48));
    int i = 0;
    if (n >= 24) {
        const __m128i pair6_shuf = _mm_setr_epi8(0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11);
        const __m128i mul6       = _mm_setr_epi16(16,1, 16,1, 16,1, 16,1);  /* <<(4-o) */
        const __m128i m3f_even   = _mm_set1_epi16(0x003F);
        const __m128i m3f_odd    = _mm_set1_epi16(0x3F00);
        int blocks = (n - 8) >> 4;
        for (int b = 0; b < blocks; ++b) {
            __m128i packed = _mm_loadu_si128((const __m128i *)(bm + b * 12));
            __m128i x = _mm_mullo_epi16(_mm_shuffle_epi8(packed, pair6_shuf), mul6);
            /* code0 at bits 4..9, code1 at bits 10..15 of each u16 */
            __m128i idx = _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(x, 4), m3f_even),
                _mm_and_si128(_mm_srli_epi16(x, 2), m3f_odd));
            __m128i r0 = _mm_shuffle_epi8(t0, idx);
            __m128i r1 = _mm_shuffle_epi8(t1, idx);
            __m128i r2 = _mm_shuffle_epi8(t2, idx);
            __m128i r3 = _mm_shuffle_epi8(t3, idx);
            __m128i s4 = _mm_slli_epi16(idx, 3);   /* bit4 -> sign bit */
            __m128i s5 = _mm_slli_epi16(idx, 2);   /* bit5 -> sign bit */
            __m128i a  = _mm_blendv_epi8(r0, r1, s4);
            __m128i b2 = _mm_blendv_epi8(r2, r3, s4);
            _mm_storeu_si128((__m128i *)(symbols + (b << 4)),
                             _mm_blendv_epi8(a, b2, s5));
        }
        i = blocks << 4;
    }
    const __m128i b4 = _mm_set1_epi8(0x10);
    const __m128i b5 = _mm_set1_epi8(0x20);
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
    merge_flat_tail_x86(symbols, i, n, bm, 6, c2s);
}

/* D=7: 16 codes/iter u32-lane pair-gather.  14-bit pairs at offsets {0,6,4,2}
 * don't fit the u16 windows the D<=6 kernels use (up to 20 bits), so each
 * pair gets a 4-byte window in a u32 lane (2 pshufb gathers x 4 pairs),
 * pmulld by 2^(6-o) normalizes to bits 6..19, shift+mask place code0/code1
 * in the lane's low bytes, and packus_epi32 compacts to 16 in-order codes.
 * The 128-entry scatter is 8 pshufb quarters + a 3-level blend on bits 4/5/6
 * (psllw into blendv's sign bit).
 *
 * Kept as its own function: the first cut, written directly inside
 * merge_flat_d7_x86, cost c3 (IvyBridge) ~12% E2E on every flat-using dist
 * via codegen/layout; as a named function the codegen is fine whether or
 * not the compiler inlines it (inline policy revisits with dynamic
 * dispatch).  Decodes the largest 16-multiple prefix that gating allows;
 * returns the count of codes written. */
static inline int merge_flat_d7_pair_x86(uint8_t *symbols,
                                                int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    __m128i t0 = _mm_loadu_si128((const __m128i *)c2s);
    __m128i t1 = _mm_loadu_si128((const __m128i *)(c2s + 16));
    __m128i t2 = _mm_loadu_si128((const __m128i *)(c2s + 32));
    __m128i t3 = _mm_loadu_si128((const __m128i *)(c2s + 48));
    __m128i t4 = _mm_loadu_si128((const __m128i *)(c2s + 64));
    __m128i t5 = _mm_loadu_si128((const __m128i *)(c2s + 80));
    __m128i t6 = _mm_loadu_si128((const __m128i *)(c2s + 96));
    __m128i t7 = _mm_loadu_si128((const __m128i *)(c2s + 112));
    const __m128i g_lo = _mm_setr_epi8(0,1,2,3, 1,2,3,4, 3,4,5,6, 5,6,7,8);
    const __m128i g_hi = _mm_setr_epi8(7,8,9,10, 8,9,10,11, 10,11,12,13, 12,13,14,15);
    const __m128i mul7 = _mm_setr_epi32(64,1,4,16);   /* <<(6-o), o={0,6,4,2} */
    const __m128i m7f_even = _mm_set1_epi32(0x0000007F);
    const __m128i m7f_odd  = _mm_set1_epi32(0x00007F00);
    int blocks = (n - 3) >> 4;
    for (int b = 0; b < blocks; ++b) {
        __m128i packed = _mm_loadu_si128((const __m128i *)(bm + b * 14));
        __m128i xl = _mm_mullo_epi32(_mm_shuffle_epi8(packed, g_lo), mul7);
        __m128i xh = _mm_mullo_epi32(_mm_shuffle_epi8(packed, g_hi), mul7);
        /* code0 at bits 6..12, code1 at bits 13..19 of each u32 */
        __m128i cl = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32(xl, 6), m7f_even),
            _mm_and_si128(_mm_srli_epi32(xl, 5), m7f_odd));
        __m128i ch = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32(xh, 6), m7f_even),
            _mm_and_si128(_mm_srli_epi32(xh, 5), m7f_odd));
        __m128i idx = _mm_packus_epi32(cl, ch);   /* 16 codes, in order */
        __m128i r0 = _mm_shuffle_epi8(t0, idx);
        __m128i r1 = _mm_shuffle_epi8(t1, idx);
        __m128i r2 = _mm_shuffle_epi8(t2, idx);
        __m128i r3 = _mm_shuffle_epi8(t3, idx);
        __m128i r4 = _mm_shuffle_epi8(t4, idx);
        __m128i r5 = _mm_shuffle_epi8(t5, idx);
        __m128i r6 = _mm_shuffle_epi8(t6, idx);
        __m128i r7 = _mm_shuffle_epi8(t7, idx);
        __m128i s4 = _mm_slli_epi16(idx, 3);   /* bit4 -> sign bit */
        __m128i s5 = _mm_slli_epi16(idx, 2);   /* bit5 -> sign bit */
        __m128i s6 = _mm_slli_epi16(idx, 1);   /* bit6 -> sign bit */
        __m128i a0 = _mm_blendv_epi8(r0, r1, s4);
        __m128i a1 = _mm_blendv_epi8(r2, r3, s4);
        __m128i a2 = _mm_blendv_epi8(r4, r5, s4);
        __m128i a3 = _mm_blendv_epi8(r6, r7, s4);
        __m128i b0 = _mm_blendv_epi8(a0, a1, s5);
        __m128i b1 = _mm_blendv_epi8(a2, a3, s5);
        _mm_storeu_si128((__m128i *)(symbols + (b << 4)),
                         _mm_blendv_epi8(b0, b1, s6));
    }
    return blocks << 4;
}

static inline void merge_flat_d7_x86(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
#ifdef PIVCO_HAS_AVX2
    {   /* ymm blocks first; shift the frame so the 128-bit body and the
         * scalar tail below run unchanged on the remainder. */
        int done = merge_flat_d7_ymm_x86(symbols, n, bm, c2s);
        symbols += done; bm += (done * 7) >> 3; n -= done;
    }
#endif
    int i = 0;
    if (n >= 19)
        i = merge_flat_d7_pair_x86(symbols, n, bm, c2s);
    for (; i + 8 <= n; i += 8) {
        const uint8_t *p = bm + ((i * 7) >> 3);
        uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                   | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                   | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)
                   | ((uint64_t)p[6] << 48);
        symbols[i    ] = c2s[(w      ) & 0x7F];
        symbols[i + 1] = c2s[(w >>  7) & 0x7F];
        symbols[i + 2] = c2s[(w >> 14) & 0x7F];
        symbols[i + 3] = c2s[(w >> 21) & 0x7F];
        symbols[i + 4] = c2s[(w >> 28) & 0x7F];
        symbols[i + 5] = c2s[(w >> 35) & 0x7F];
        symbols[i + 6] = c2s[(w >> 42) & 0x7F];
        symbols[i + 7] = c2s[(w >> 49) & 0x7F];
    }
    merge_flat_tail_x86(symbols, i, n, bm, 7, c2s);
}

/* D=8: a depth-8 flat region is the full 256-symbol alphabet at equal code
 * length, whose canonical c2s is the identity permutation -- the byte-aligned
 * codes ARE the symbols, so the whole decode is a memcpy.  See the derivation
 * at merge_flat_d8_neon in pivco_huffman_primitives_neon.h. */
static inline void merge_flat_d8_x86(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    (void)c2s;
    memcpy(symbols, bm, (size_t)n);
}

/* merge_flat_x86 — D-bit flat-subtree decode into a contiguous output
 * buffer.  Per-D SIMD paths above; scalar unrolled for the rest.  AVX-512
 * D=5/D=6 fast paths live in primitives_avx512.h. */
static inline void merge_flat_x86(uint8_t *out, int n,
                                               const uint8_t *bm, int D,
                                               const uint8_t *c2s)
{
    PROF_TIC();
    switch (D) {
    case 2: merge_flat_d2_x86(out, n, bm, c2s); break;
    case 3: merge_flat_d3_x86(out, n, bm, c2s); break;
    case 4: merge_flat_d4_x86(out, n, bm, c2s); break;
    case 5: merge_flat_d5_x86(out, n, bm, c2s); break;
    case 6: merge_flat_d6_x86(out, n, bm, c2s); break;
    case 7: merge_flat_d7_x86(out, n, bm, c2s); break;
    case 8: merge_flat_d8_x86(out, n, bm, c2s); break;
    default:
        pivco_check_fail("merge_flat_x86: D out of range (flat_depth is 2..8)",
                         __FILE__, __LINE__);
        break;
    }
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

/* p16rev partition LUTs (part_full_x86).  One combined index per 16-lane group
 * packs {left, forward, front} | {right, reversed, back}; left+right tile the
 * 16 lanes so the OR of two disjoint-support tables is exact.  A fraction of
 * the LUT footprint of the ctab_r/l + pre_r/l pair above (12 KB vs 80 KB) —
 * the win on x86's 32-48 KB L1.
 *   x86_p16rev_tabA[m0]   low-byte (positions 0..7)
 *   x86_p16rev_tabB0[m1]  high-byte (positions 8..15), pc0=0 layout only: the
 *                       pc0>0 layout is this one shifted left by pc0 lanes, so
 *                       tabB[pc0][m1] is recovered as a byte-offset load
 *                       `tabB0[m1] + pc0` (8 KB vs the former 36 KB).  32 B
 *                       rows keep the offset-16 load inside one cache line.
 * The right side is recovered with a single loop-invariant full-reverse
 * constant in part_full_x86. */
static uint8_t x86_p16rev_tabA[256][16]  __attribute__((aligned(16)));
static uint8_t x86_p16rev_tabB0[256][32] __attribute__((aligned(32)));
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
    /* p16rev combined-index tables (low byte over m0, high byte over [pc0][m1]),
     * 0-fill on the non-owned lanes so the OR is exact (every lane owned once). */
    for (int m0 = 0; m0 < 256; m0++) {
        memset(x86_p16rev_tabA[m0], 0, 16);
        int lp = 0, rp = 15;
        for (int k = 0; k < 8; k++) {
            if ((m0 >> k) & 1) x86_p16rev_tabA[m0][rp--] = (uint8_t)k;
            else               x86_p16rev_tabA[m0][lp++] = (uint8_t)k;
        }
    }
    for (int m1 = 0; m1 < 256; m1++) {
        memset(x86_p16rev_tabB0[m1], 0, 32);
        int lp = 8, rp = 15;   /* pc0 = 0 layout; pc0 > 0 handled by the load offset */
        for (int k = 0; k < 8; k++) {
            if ((m1 >> k) & 1) x86_p16rev_tabB0[m1][rp--] = (uint8_t)(8 + k);
            else               x86_p16rev_tabB0[m1][lp++] = (uint8_t)(8 + k);
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

/* full: p16rev — per 16-lane group, ONE combined index (OR of the two disjoint
 * x86_p16rev_tabA/tabB0) feeds one pshufb that yields {left fwd | right reversed}
 * in a single register; that register IS the left output (store it), and the
 * right output is recovered with a second pshufb over the SAME register using
 * the loop-invariant full-reverse constant.  vs the prior dense X86_COMPACT16
 * (two independent min-merged indices): one OR + one table-pair load instead of
 * two pminub + four loads per 16 lanes, and the LUTs shrink (12 KB vs 80 KB) —
 * the win on x86's 32-48 KB L1.  32 ranks per iter; the 16-byte tail overstore
 * is absorbed by the ranks +64 / tmp +2N scratch slack reserved in codec.c. */
static inline int part_full_x86(uint8_t *ranks, int n, uint8_t thr,
                                   uint8_t *bm, uint8_t *tmp)
{
    x86_build_tabs();
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    /* Right recovery via one loop-invariant full-reverse constant: reversing the
     * whole comb register lands the top-pc reversed right lanes at output [0,pc)
     * (the tail is left-reversed garbage the next group overwrites). */
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    const __m128i rev16 = _mm_loadu_si128((const __m128i *)rev16_a);
    /* cl_/cr_ = lefts/rights already emitted by earlier groups THIS iter, so
     * both groups' store addresses are known up front (no serial cursor chain
     * between them); n_left/n_right advance once per iter.  (issue #5) */
#define _P16REV(v, mlo_, mhi_, cl_, cr_) do {                                  \
        uint32_t pc0_ = (uint32_t)__builtin_popcount((unsigned)(mlo_));        \
        __m128i cidx_ = _mm_or_si128(                                          \
            _mm_load_si128((const __m128i *)x86_p16rev_tabA[(mlo_)]),          \
            _mm_loadu_si128((const __m128i *)&x86_p16rev_tabB0[(mhi_)][pc0_])); \
        __m128i comb_ = _mm_shuffle_epi8((v), cidx_);                         \
        _mm_storeu_si128((__m128i *)(ranks + n_left + (cl_)), comb_);         \
        _mm_storeu_si128((__m128i *)(tmp + n_right + (cr_)),                   \
            _mm_shuffle_epi8(comb_, rev16));                                   \
    } while (0)
    /* 32 ranks/iter: two SSE movemasks OR'd into a 32-bit routing mask (one
     * 4-byte bitmap write), two combined-shuffle compactions.  Both 16-byte
     * halves are loaded before any in-place left store, so the dense left write
     * can't clobber an un-loaded rank. */
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1));
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1));
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        uint32_t cr1   = (uint32_t)__builtin_popcount(mlo);  /* rights in group 0 */
        uint32_t total = (uint32_t)__builtin_popcount(mm);
        _P16REV(v0, (uint8_t)mlo, (uint8_t)(mlo >> 8), 0, 0);
        _P16REV(v1, (uint8_t)mhi, (uint8_t)(mhi >> 8), 16 - cr1, cr1);
        n_right += (int)total; n_left += 32 - (int)total;
    }
    /* 16-rank tail of the [32k, 32k+31] remainder: one movemask, one compaction. */
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        memcpy(bm + (j >> 3), &mm, 2);
        uint32_t pc16 = (uint32_t)__builtin_popcount((unsigned)mm);
        _P16REV(v, (uint8_t)mm, (uint8_t)(mm >> 8), 0, 0);
        n_right += (int)pc16; n_left += 16 - (int)pc16;
    }
#undef _P16REV
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}

/* right/none: 16-wide one-sided compaction, 32 ranks/iter.  Per 16-lane
 * group, the emitted side's index is the matching half of X86_COMPACT16 (RIGHT:
 * min(ctab_r[m0], pre_r[pc0][m1]); LEFT: min(ctab_l[m0], pre_l[8-pc0][m1])), one
 * pshufb + one 16-byte store — vs the prior stride-8 form's pshufb + 8-byte store
 * every 8 lanes.  Reuses the production ctab/pre tables.  EMIT_RIGHT is
 * compile-time, so the none form folds to a pure bitmap build; the left
 * side is never scattered (a leaf child's ranks are dead). */
__attribute__((always_inline)) static inline
int part_core_x86(uint8_t *ranks, int n, uint8_t thr,
                     uint8_t *bm, uint8_t *tmp, int EMIT_RIGHT)
{
    x86_build_tabs();
    int n_right = 0;
    int j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
#define _PC16(v, mlo_, mhi_) do {                                             \
        uint32_t pc0_ = x86_pc8[(mlo_)];                                      \
        uint32_t pc_  = pc0_ + x86_pc8[(mhi_)];                               \
        if (EMIT_RIGHT) _mm_storeu_si128((__m128i *)(tmp + n_right),          \
            _mm_shuffle_epi8((v), _mm_min_epu8(                               \
                _mm_load_si128((const __m128i *)x86_ctab_r[(mlo_)]),          \
                _mm_load_si128((const __m128i *)x86_pre_r[pc0_][(mhi_)]))));  \
        n_right += pc_;                                                       \
    } while (0)
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1));
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1));
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        _PC16(v0, (uint8_t)mlo, (uint8_t)(mlo >> 8));
        _PC16(v1, (uint8_t)mhi, (uint8_t)(mhi >> 8));
    }
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        memcpy(bm + (j >> 3), &mm, 2);
        _PC16(v, (uint8_t)mm, (uint8_t)(mm >> 8));
    }
#undef _PC16
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7));
                       if (EMIT_RIGHT) tmp[n_right] = r; n_right++; }
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
        __m128i bytes = _mm_sub_epi8(_mm_loadu_si128((const __m128i *)(ranks + i)), vb);
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);   /* local code in [0,2^D); no mask */
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
        __m128i bytes = _mm_sub_epi8(_mm_loadu_si128((const __m128i *)(ranks + i)), vb);
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);   /* local code in [0,2^D); no mask */
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

/* SSE4.1 D=3/5/6/7: 128-bit twin of the AVX2 ryg/pyramid pack (see
 * pivco_huffman_avx2_pack.h for the op-by-op story).  16 codes per xmm
 * iter: maddubs (2D-bit byte pairs) -> madd (4D-bit dword pairs) ->
 * qword fuse (srlq + and/andn/or) -> pshufb compact -> one 16-byte
 * store.  The store's trailing junk (16 - 2D bytes) is overwritten by
 * the next iter; the LAST iter's junk needs slack past the packed
 * stream, which PIVCO_MAX_ENCODED_SIZE provides (same contract as the
 * AVX2 kernels).  Replaces the mullo+hadd 8-codes/iter D=3 form and
 * the D=5/6/7 scalar fallback on non-AVX2 hosts: c3/IvyBridge measures
 * 5.1x (d3) and 19-22x (d5-d7); the old d3 is asof-b8bf472 in
 * bench/prim_variants.  D=2/4 keep the simpler maddubs forms above
 * (faster than a full pyramid at those widths on every host tested). */
#define PIVCO_PACK_SSE_DN(NAME, D_VAL, C0,C1,C2, C3,C4,C5, C6,C7,C8, C9,C10,C11, C12,C13) \
static inline int NAME(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)          \
{                                                                                         \
    const __m128i c0 = _mm_set1_epi16((int16_t)(((1 << (D_VAL)) << 8) | 1));              \
    const __m128i c1 = _mm_set1_epi32((int32_t)(((int32_t)1 << (2*(D_VAL))) << 16) | 1);  \
    const __m128i c3m = _mm_set1_epi64x((int64_t)(((int64_t)1 << (4*(D_VAL))) - 1));      \
    const __m128i compact = _mm_setr_epi8(C0,C1,C2, C3,C4,C5, C6,C7,C8, C9,C10,C11,       \
                                          C12,C13, -1,-1);                                \
    const __m128i vb = _mm_set1_epi8((char)base);                                         \
    int i = 0;                                                                            \
    for (; i + 16 <= n; i += 16) {                                                        \
        __m128i cb = _mm_sub_epi8(_mm_loadu_si128((const __m128i *)(ranks + i)), vb);     \
        __m128i x  = _mm_maddubs_epi16(c0, cb);                                           \
        x = _mm_madd_epi16(x, c1);                                                        \
        __m128i xs = _mm_srli_epi64(x, 32 - 4*(D_VAL));                                   \
        x = _mm_or_si128(_mm_and_si128(x, c3m), _mm_andnot_si128(c3m, xs));               \
        _mm_storeu_si128((__m128i *)(out + ((i * (D_VAL)) >> 3)),                         \
                          _mm_shuffle_epi8(x, compact));                                  \
    }                                                                                     \
    return i;                                                                             \
}
/* compact patterns: bytes [0..D-1] from qword0, [D..2D-1] from qword1 (pos 8+). */
PIVCO_PACK_SSE_DN(pack_d3_sse_x86, 3, 0,1,2,  8,9,10,  -1,-1,-1, -1,-1,-1, -1,-1)
PIVCO_PACK_SSE_DN(pack_d5_sse_x86, 5, 0,1,2,  3,4,8,   9,10,11,  12,-1,-1, -1,-1)
PIVCO_PACK_SSE_DN(pack_d6_sse_x86, 6, 0,1,2,  3,4,5,   8,9,10,   11,12,13, -1,-1)
PIVCO_PACK_SSE_DN(pack_d7_sse_x86, 7, 0,1,2,  3,4,5,   6,8,9,    10,11,12, 13,14)
#undef PIVCO_PACK_SSE_DN

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
    case 5: i = pack_d5_sse_x86(out, ranks, n, base); break;
    case 6: i = pack_d6_sse_x86(out, ranks, n, base); break;
    case 7: i = pack_d7_sse_x86(out, ranks, n, base); break;
#endif
    default: break;
    }

    if (i >= n) return;

    /* Scalar tail. */
    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = (uint32_t)(uint8_t)(ranks[i] - base);  /* code in [0,2^D); no mask */
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

#include "pivco_huffman_hist_scalar.h"

/* SSE4.1/AVX2 has no histogram win over the shared scalar core (measured);
 * alias it explicitly. */
PIVCO_PRIM_ALWAYS_INLINE void prim_histogram_chunk(const uint8_t *in, size_t n,
                                                   uint32_t hist[256],
                                                   uint8_t *scratch)
{ histogram_chunk_scalar(in, n, hist, scratch); }


/* Widest load a merge kernel issues at a child-buffer cursor (16B loadu at child cursors);
 * the cursor can rest AT `size` on the exhausted side, so buffers a
 * merge reads need this much trailing slack.  Consumed by the decode
 * placement logic (scratch_carve / place_tail). */
#define PIVCO_PRIM_MERGE_OVERREAD 16

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ codec_init_x86(); }

/* enc_init 2tab no-OR gather: read 16 input symbols as 2x u64 (frees the load
 * ports for the dependent table loads) and merge each rank pair as
 * (u16)sym_to_rank[s0] + hi[s1], where hi[s] = sym_to_rank[s]<<8 (aux->s2r_hi,
 * built once in the table).  Disjoint byte lanes -> + is a single add, no shift
 * and the hi load folds in as a memory operand -- the shift+or that x86 can't
 * fuse is gone.  ~1.6x the naive byte loop across the SSE/AVX2 tier, and the
 * only variant with no pathological host (4tab regresses on Skylake, bc2 on all
 * Intel).  x86-only: on AArch64 the shift folds into orr, so NEON keeps its SIMD
 * gather.  See IDEAS.md ("enc_init 4tab / bc2") for the A/B/C that chose 2tab. */
PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint8_t *restrict ranks, int n,
                                              const uint8_t *restrict symbols,
                                              const uint8_t *sym_to_rank,
                                              const pivco_enc_init_aux_t *aux)
{
    PIVCO_CHECK(aux && aux->s2r_hi);
    const uint16_t *restrict hi = aux->s2r_hi;
#define PIVCO_LO(x) ((uint16_t)sym_to_rank[(uint8_t)(x)])
#define PIVCO_HI(x) hi[(uint8_t)(x)]
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t a, b;
        memcpy(&a, symbols + i,     8);
        memcpy(&b, symbols + i + 8, 8);
        uint16_t h0 = PIVCO_LO(a)       + PIVCO_HI(a >> 8);
        uint16_t h1 = PIVCO_LO(a >> 16) + PIVCO_HI(a >> 24);
        uint16_t h2 = PIVCO_LO(a >> 32) + PIVCO_HI(a >> 40);
        uint16_t h3 = PIVCO_LO(a >> 48) + PIVCO_HI(a >> 56);
        uint16_t h4 = PIVCO_LO(b)       + PIVCO_HI(b >> 8);
        uint16_t h5 = PIVCO_LO(b >> 16) + PIVCO_HI(b >> 24);
        uint16_t h6 = PIVCO_LO(b >> 32) + PIVCO_HI(b >> 40);
        uint16_t h7 = PIVCO_LO(b >> 48) + PIVCO_HI(b >> 56);
        memcpy(ranks + i,      &h0, 2); memcpy(ranks + i + 2,  &h1, 2);
        memcpy(ranks + i + 4,  &h2, 2); memcpy(ranks + i + 6,  &h3, 2);
        memcpy(ranks + i + 8,  &h4, 2); memcpy(ranks + i + 10, &h5, 2);
        memcpy(ranks + i + 12, &h6, 2); memcpy(ranks + i + 14, &h7, 2);
    }
    for (; i < n; i++) ranks[i] = sym_to_rank[symbols[i]];
#undef PIVCO_LO
#undef PIVCO_HI
}

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_full(uint8_t *ranks,
                                                      int n, uint8_t thr,
                                                      uint8_t *bm,
                                                      uint8_t *right_out)
{ return part_full_x86(ranks, n, thr, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_right(uint8_t *ranks,
                                                      int n, uint8_t thr,
                                                      uint8_t *bm,
                                                      uint8_t *right_out)
{ return part_core_x86(ranks, n, thr, bm, right_out, 1); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_none(uint8_t *ranks,
                                                     int n, uint8_t thr,
                                                     uint8_t *bm)
{ return part_core_x86(ranks, n, thr, bm, NULL, 0); }

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

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_vec_vec(const uint8_t *bm, int K,
                                               const uint8_t *left_buf,
                                               const uint8_t *right_buf,
                                               uint8_t *out)
{ merge_vec_vec_x86(bm, K, left_buf, right_buf, out); }

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_X86_H */
