/* pivco_huffman_x86_vertical.h — vertical-128 flat kernels, SSE4.1.
 *
 * Shared by the x86 (SSE4.1/AVX2) and AVX-512 backends: the vertical
 * block layout is 16-lane-native (see pivco_huffman_vertical.h), so a
 * 128-bit kernel is the natural shape; the AVX-512 backend reuses it
 * as-is (wider forms — two blocks per ymm/zmm, vpsrlvw — are possible
 * later if profiles ask).
 *
 * SSE has no per-byte shifts; each step's field extraction synthesizes
 * them from u16-lane shifts plus byte masks.  With DV and the step
 * index compile-time constants (per-D macro instantiation + unrolled
 * s-loop), the three cases (off == 0 / field within byte / field
 * spanning two columns) fold to straight-line code.
 *
 * Maps are the same 1/2/4/8-table pshufb(+pblendvb select tree) forms
 * used elsewhere in the x86 flat path.
 */
#ifndef PIVCO_HUFFMAN_X86_VERTICAL_H
#define PIVCO_HUFFMAN_X86_VERTICAL_H

#include <stdint.h>
#include <smmintrin.h>
#include "pivco_huffman_vertical.h"

#define PIVCO_VERT_MERGE_X86_BODY(DV, MAPEXPR, BLK, O, CS, OS)                \
    do {                                                                       \
        __m128i cols[8];                                                       \
        for (int j = 0; j < DV; j++)                                           \
            cols[j] = _mm_loadu_si128((const __m128i *)((BLK) + (CS) * j));    \
        for (int s = 0; s < 8; s++) {                                          \
            const int bit = s * DV, j = bit >> 3, off = bit & 7;               \
            __m128i w;                                                         \
            if (off == 0) {                                                    \
                w = cols[j];                                                   \
            } else if (off + DV <= 8) {                                        \
                w = _mm_srli_epi16(cols[j], off);                              \
            } else {                                                           \
                __m128i lo = _mm_and_si128(_mm_srli_epi16(cols[j], off),       \
                                 _mm_set1_epi8((char)(0xFFu >> off)));         \
                __m128i hi = _mm_and_si128(_mm_slli_epi16(cols[j + 1], 8 - off),\
                                 _mm_set1_epi8((char)(0xFFu << (8 - off))));   \
                w = _mm_or_si128(lo, hi);                                      \
            }                                                                  \
            __m128i codes = _mm_and_si128(w, maskv);                           \
            _mm_storeu_si128((__m128i *)((O) + (OS) * s), MAPEXPR);            \
        }                                                                      \
    } while (0)
#define PIVCO_VERT_MERGE_X86(DV, SETUP, MAPEXPR)                               \
static void vert_merge_x86_d##DV(uint8_t *out, int n_v, const uint8_t *bm,     \
                                 const uint8_t *c2s)                           \
{                                                                              \
    SETUP                                                                      \
    const __m128i maskv = _mm_set1_epi8((char)((1u << DV) - 1));               \
    for (int b = 0; b < n_v >> 7; b++)                                         \
        PIVCO_VERT_MERGE_X86_BODY(DV, MAPEXPR,                                 \
                                  bm + (size_t)b * 16 * DV,                    \
                                  out + ((size_t)b << 7), 16, 16);             \
}                                                                              \
static void vert512_merge_x86_d##DV(uint8_t *out, int n_v, const uint8_t *bm,  \
                                    const uint8_t *c2s)                        \
{                                                                              \
    SETUP                                                                      \
    const __m128i maskv = _mm_set1_epi8((char)((1u << DV) - 1));               \
    for (int b = 0; b < n_v >> 9; b++)                                         \
        for (int qt = 0; qt < 4; qt++)                                         \
            PIVCO_VERT_MERGE_X86_BODY(DV, MAPEXPR,                             \
                                      bm + (size_t)b * 64 * DV + 16 * qt,      \
                                      out + ((size_t)b << 9) + 16 * qt,        \
                                      64, 64);                                 \
}

PIVCO_VERT_MERGE_X86(2,
    const __m128i t = _mm_loadu_si128((const __m128i *)c2s);,
    _mm_shuffle_epi8(t, codes))
PIVCO_VERT_MERGE_X86(3,
    const __m128i t = _mm_loadu_si128((const __m128i *)c2s);,
    _mm_shuffle_epi8(t, codes))
PIVCO_VERT_MERGE_X86(4,
    const __m128i t = _mm_loadu_si128((const __m128i *)c2s);,
    _mm_shuffle_epi8(t, codes))
PIVCO_VERT_MERGE_X86(5,
    const __m128i t0 = _mm_loadu_si128((const __m128i *)c2s);
    const __m128i t1 = _mm_loadu_si128((const __m128i *)(c2s + 16));,
    _mm_blendv_epi8(_mm_shuffle_epi8(t0, codes),
                    _mm_shuffle_epi8(t1, codes), _mm_slli_epi16(codes, 3)))
PIVCO_VERT_MERGE_X86(6,
    const __m128i t0 = _mm_loadu_si128((const __m128i *)c2s);
    const __m128i t1 = _mm_loadu_si128((const __m128i *)(c2s + 16));
    const __m128i t2 = _mm_loadu_si128((const __m128i *)(c2s + 32));
    const __m128i t3 = _mm_loadu_si128((const __m128i *)(c2s + 48));,
    _mm_blendv_epi8(
        _mm_blendv_epi8(_mm_shuffle_epi8(t0, codes),
                        _mm_shuffle_epi8(t1, codes), _mm_slli_epi16(codes, 3)),
        _mm_blendv_epi8(_mm_shuffle_epi8(t2, codes),
                        _mm_shuffle_epi8(t3, codes), _mm_slli_epi16(codes, 3)),
        _mm_slli_epi16(codes, 2)))
PIVCO_VERT_MERGE_X86(7,
    __m128i t[8];
    for (int q = 0; q < 8; q++)
        t[q] = _mm_loadu_si128((const __m128i *)(c2s + 16 * q));,
    _mm_blendv_epi8(
        _mm_blendv_epi8(
            _mm_blendv_epi8(_mm_shuffle_epi8(t[0], codes),
                            _mm_shuffle_epi8(t[1], codes), _mm_slli_epi16(codes, 3)),
            _mm_blendv_epi8(_mm_shuffle_epi8(t[2], codes),
                            _mm_shuffle_epi8(t[3], codes), _mm_slli_epi16(codes, 3)),
            _mm_slli_epi16(codes, 2)),
        _mm_blendv_epi8(
            _mm_blendv_epi8(_mm_shuffle_epi8(t[4], codes),
                            _mm_shuffle_epi8(t[5], codes), _mm_slli_epi16(codes, 3)),
            _mm_blendv_epi8(_mm_shuffle_epi8(t[6], codes),
                            _mm_shuffle_epi8(t[7], codes), _mm_slli_epi16(codes, 3)),
            _mm_slli_epi16(codes, 2)),
        _mm_slli_epi16(codes, 1)))
#undef PIVCO_VERT_MERGE_X86
#undef PIVCO_VERT_MERGE_X86_BODY

static inline void vert_merge_x86v(uint8_t *out, int n_v, const uint8_t *bm,
                                   int D, const uint8_t *c2s)
{
    switch (D) {
    case 2: vert_merge_x86_d2(out, n_v, bm, c2s); break;
    case 3: vert_merge_x86_d3(out, n_v, bm, c2s); break;
    case 4: vert_merge_x86_d4(out, n_v, bm, c2s); break;
    case 5: vert_merge_x86_d5(out, n_v, bm, c2s); break;
    case 6: vert_merge_x86_d6(out, n_v, bm, c2s); break;
    default: vert_merge_x86_d7(out, n_v, bm, c2s); break;
    }
}

static inline void vert512_merge_x86v(uint8_t *out, int n_v, const uint8_t *bm,
                                      int D, const uint8_t *c2s)
{
    switch (D) {
    case 2: vert512_merge_x86_d2(out, n_v, bm, c2s); break;
    case 3: vert512_merge_x86_d3(out, n_v, bm, c2s); break;
    case 4: vert512_merge_x86_d4(out, n_v, bm, c2s); break;
    case 5: vert512_merge_x86_d5(out, n_v, bm, c2s); break;
    case 6: vert512_merge_x86_d6(out, n_v, bm, c2s); break;
    default: vert512_merge_x86_d7(out, n_v, bm, c2s); break;
    }
}

/* Encoder mirror: accumulate byte-columns with the same synthesized
 * per-byte shifts.  Per-D instantiation for constant folding. */
#define PIVCO_VERT_PACK_X86_BODY(DV, BLK, R, CS, OS)                          \
    do {                                                                       \
        __m128i cols[8];                                                       \
        for (int j = 0; j < DV; j++) cols[j] = _mm_setzero_si128();            \
        for (int s = 0; s < 8; s++) {                                          \
            __m128i val = _mm_sub_epi8(                                        \
                _mm_loadu_si128((const __m128i *)((R) + (OS) * s)), basev);    \
            const int bit = s * DV, j = bit >> 3, off = bit & 7;               \
            if (off == 0) {                                                    \
                cols[j] = _mm_or_si128(cols[j], val);                          \
            } else {                                                           \
                cols[j] = _mm_or_si128(cols[j],                                \
                    _mm_and_si128(_mm_slli_epi16(val, off),                    \
                                  _mm_set1_epi8((char)(0xFFu << off))));       \
                if (off + DV > 8)                                              \
                    cols[j + 1] = _mm_or_si128(cols[j + 1],                    \
                        _mm_and_si128(_mm_srli_epi16(val, 8 - off),            \
                                      _mm_set1_epi8((char)(0xFFu >> (8 - off)))));\
            }                                                                  \
        }                                                                      \
        for (int j = 0; j < DV; j++)                                           \
            _mm_storeu_si128((__m128i *)((BLK) + (CS) * j), cols[j]);          \
    } while (0)
#define PIVCO_VERT_PACK_X86(DV)                                                \
static void vert_pack_x86_d##DV(uint8_t *out, const uint8_t *ranks,            \
                                int n_v, uint8_t base)                         \
{                                                                              \
    const __m128i basev = _mm_set1_epi8((char)base);                           \
    for (int b = 0; b < n_v >> 7; b++)                                         \
        PIVCO_VERT_PACK_X86_BODY(DV, out + (size_t)b * 16 * DV,                \
                                 ranks + ((size_t)b << 7), 16, 16);            \
}                                                                              \
static void vert512_pack_x86_d##DV(uint8_t *out, const uint8_t *ranks,         \
                                   int n_v, uint8_t base)                      \
{                                                                              \
    const __m128i basev = _mm_set1_epi8((char)base);                           \
    for (int b = 0; b < n_v >> 9; b++)                                         \
        for (int qt = 0; qt < 4; qt++)                                         \
            PIVCO_VERT_PACK_X86_BODY(DV, out + (size_t)b * 64 * DV + 16 * qt,  \
                                     ranks + ((size_t)b << 9) + 16 * qt,       \
                                     64, 64);                                  \
}
PIVCO_VERT_PACK_X86(2)
PIVCO_VERT_PACK_X86(3)
PIVCO_VERT_PACK_X86(4)
PIVCO_VERT_PACK_X86(5)
PIVCO_VERT_PACK_X86(6)
PIVCO_VERT_PACK_X86(7)
#undef PIVCO_VERT_PACK_X86
#undef PIVCO_VERT_PACK_X86_BODY

static inline void vert_pack_x86v(uint8_t *out, const uint8_t *ranks,
                                  int n_v, int D, uint8_t base)
{
    switch (D) {
    case 2: vert_pack_x86_d2(out, ranks, n_v, base); break;
    case 3: vert_pack_x86_d3(out, ranks, n_v, base); break;
    case 4: vert_pack_x86_d4(out, ranks, n_v, base); break;
    case 5: vert_pack_x86_d5(out, ranks, n_v, base); break;
    case 6: vert_pack_x86_d6(out, ranks, n_v, base); break;
    default: vert_pack_x86_d7(out, ranks, n_v, base); break;
    }
}

static inline void vert512_pack_x86v(uint8_t *out, const uint8_t *ranks,
                                     int n_v, int D, uint8_t base)
{
    switch (D) {
    case 2: vert512_pack_x86_d2(out, ranks, n_v, base); break;
    case 3: vert512_pack_x86_d3(out, ranks, n_v, base); break;
    case 4: vert512_pack_x86_d4(out, ranks, n_v, base); break;
    case 5: vert512_pack_x86_d5(out, ranks, n_v, base); break;
    case 6: vert512_pack_x86_d6(out, ranks, n_v, base); break;
    default: vert512_pack_x86_d7(out, ranks, n_v, base); break;
    }
}


/* Best-available dispatcher for the vertical prefix.  All x86 tiers
 * run the shared xmm 16-lane cores here: the wide 128-block forms (ymm
 * two-block, zmm srlv/multishift) were removed 2026-08 -- they only
 * covered the 128..511 mid-band (<= 4% of elements at 32K blocks, ~0.2%
 * E2E) at ~12 KB of text per TU.  Resurrect from git @ 87da7ed if that
 * band ever matters (see IDEAS.md). */
static inline void vert_merge_x86_best(uint8_t *out, int n_v, const uint8_t *bm,
                                       int D, const uint8_t *c2s)
{
    vert_merge_x86v(out, n_v, bm, D, c2s);
}


/* Best-available vertical pack (xmm cores on all tiers, as above). */
static inline void vert_pack_x86_best(uint8_t *out, const uint8_t *ranks,
                                      int n_v, int D, uint8_t base)
{
    vert_pack_x86v(out, ranks, n_v, D, base);
}


/* ==== 512-value / 64-lane blocks (see pivco_huffman_vertical.h) ====
 * With 64 lanes, one step fills a whole zmm row: extraction is a uniform
 * immediate shift + byte masks -- no srlv, no multishift, no reorder, no
 * vendor split.  AVX2 processes 32-lane half-rows the same way; the SSE
 * tier walks the four interleaved 16-lane quarters with the group cores
 * above.  Pack mirrors merge (contiguous rank loads, monotonic per-row
 * offsets let the first touch skip the zero-init). */
#if defined(__AVX512BW__) && defined(__AVX512VBMI__) && defined(__AVX512VL__)


static void vert512_merge_zmm_d2(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
    const __m512i tab = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)c2s));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 2;
        uint8_t *o = out + ((size_t)b << 9);
        __m512i r0 = _mm512_loadu_si512((const void *)(blk + 0));
        __m512i r1 = _mm512_loadu_si512((const void *)(blk + 64));
        __m512i v0 = _mm512_and_si512(r0, _mm512_set1_epi8((char)0x03));
        _mm512_storeu_si512((void *)(o + 0), _mm512_permutexvar_epi8(v0, tab));
        __m512i v1 = _mm512_and_si512(_mm512_srli_epi16(r0, 2), _mm512_set1_epi8((char)0x03));
        _mm512_storeu_si512((void *)(o + 64), _mm512_permutexvar_epi8(v1, tab));
        __m512i v2 = _mm512_and_si512(_mm512_srli_epi16(r0, 4), _mm512_set1_epi8((char)0x03));
        _mm512_storeu_si512((void *)(o + 128), _mm512_permutexvar_epi8(v2, tab));
        __m512i v3 = _mm512_and_si512(_mm512_srli_epi16(r0, 6), _mm512_set1_epi8((char)0x03));
        _mm512_storeu_si512((void *)(o + 192), _mm512_permutexvar_epi8(v3, tab));
        __m512i v4 = _mm512_and_si512(r1, _mm512_set1_epi8((char)0x03));
        _mm512_storeu_si512((void *)(o + 256), _mm512_permutexvar_epi8(v4, tab));
        __m512i v5 = _mm512_and_si512(_mm512_srli_epi16(r1, 2), _mm512_set1_epi8((char)0x03));
        _mm512_storeu_si512((void *)(o + 320), _mm512_permutexvar_epi8(v5, tab));
        __m512i v6 = _mm512_and_si512(_mm512_srli_epi16(r1, 4), _mm512_set1_epi8((char)0x03));
        _mm512_storeu_si512((void *)(o + 384), _mm512_permutexvar_epi8(v6, tab));
        __m512i v7 = _mm512_and_si512(_mm512_srli_epi16(r1, 6), _mm512_set1_epi8((char)0x03));
        _mm512_storeu_si512((void *)(o + 448), _mm512_permutexvar_epi8(v7, tab));
    }
}

static void vert512_merge_zmm_d3(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
    const __m512i tab = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)c2s));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 3;
        uint8_t *o = out + ((size_t)b << 9);
        __m512i r0 = _mm512_loadu_si512((const void *)(blk + 0));
        __m512i r1 = _mm512_loadu_si512((const void *)(blk + 64));
        __m512i r2 = _mm512_loadu_si512((const void *)(blk + 128));
        __m512i v0 = _mm512_and_si512(r0, _mm512_set1_epi8((char)0x07));
        _mm512_storeu_si512((void *)(o + 0), _mm512_permutexvar_epi8(v0, tab));
        __m512i v1 = _mm512_and_si512(_mm512_srli_epi16(r0, 3), _mm512_set1_epi8((char)0x07));
        _mm512_storeu_si512((void *)(o + 64), _mm512_permutexvar_epi8(v1, tab));
        __m512i v2 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r0, 6), _mm512_set1_epi8((char)0x03)),
                _mm512_and_si512(_mm512_slli_epi16(r1, 2), _mm512_set1_epi8((char)0xFC))),
                _mm512_set1_epi8((char)0x07));
        _mm512_storeu_si512((void *)(o + 128), _mm512_permutexvar_epi8(v2, tab));
        __m512i v3 = _mm512_and_si512(_mm512_srli_epi16(r1, 1), _mm512_set1_epi8((char)0x07));
        _mm512_storeu_si512((void *)(o + 192), _mm512_permutexvar_epi8(v3, tab));
        __m512i v4 = _mm512_and_si512(_mm512_srli_epi16(r1, 4), _mm512_set1_epi8((char)0x07));
        _mm512_storeu_si512((void *)(o + 256), _mm512_permutexvar_epi8(v4, tab));
        __m512i v5 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r1, 7), _mm512_set1_epi8((char)0x01)),
                _mm512_and_si512(_mm512_slli_epi16(r2, 1), _mm512_set1_epi8((char)0xFE))),
                _mm512_set1_epi8((char)0x07));
        _mm512_storeu_si512((void *)(o + 320), _mm512_permutexvar_epi8(v5, tab));
        __m512i v6 = _mm512_and_si512(_mm512_srli_epi16(r2, 2), _mm512_set1_epi8((char)0x07));
        _mm512_storeu_si512((void *)(o + 384), _mm512_permutexvar_epi8(v6, tab));
        __m512i v7 = _mm512_and_si512(_mm512_srli_epi16(r2, 5), _mm512_set1_epi8((char)0x07));
        _mm512_storeu_si512((void *)(o + 448), _mm512_permutexvar_epi8(v7, tab));
    }
}

static void vert512_merge_zmm_d4(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
    const __m512i tab = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)c2s));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 4;
        uint8_t *o = out + ((size_t)b << 9);
        __m512i r0 = _mm512_loadu_si512((const void *)(blk + 0));
        __m512i r1 = _mm512_loadu_si512((const void *)(blk + 64));
        __m512i r2 = _mm512_loadu_si512((const void *)(blk + 128));
        __m512i r3 = _mm512_loadu_si512((const void *)(blk + 192));
        __m512i v0 = _mm512_and_si512(r0, _mm512_set1_epi8((char)0x0F));
        _mm512_storeu_si512((void *)(o + 0), _mm512_permutexvar_epi8(v0, tab));
        __m512i v1 = _mm512_and_si512(_mm512_srli_epi16(r0, 4), _mm512_set1_epi8((char)0x0F));
        _mm512_storeu_si512((void *)(o + 64), _mm512_permutexvar_epi8(v1, tab));
        __m512i v2 = _mm512_and_si512(r1, _mm512_set1_epi8((char)0x0F));
        _mm512_storeu_si512((void *)(o + 128), _mm512_permutexvar_epi8(v2, tab));
        __m512i v3 = _mm512_and_si512(_mm512_srli_epi16(r1, 4), _mm512_set1_epi8((char)0x0F));
        _mm512_storeu_si512((void *)(o + 192), _mm512_permutexvar_epi8(v3, tab));
        __m512i v4 = _mm512_and_si512(r2, _mm512_set1_epi8((char)0x0F));
        _mm512_storeu_si512((void *)(o + 256), _mm512_permutexvar_epi8(v4, tab));
        __m512i v5 = _mm512_and_si512(_mm512_srli_epi16(r2, 4), _mm512_set1_epi8((char)0x0F));
        _mm512_storeu_si512((void *)(o + 320), _mm512_permutexvar_epi8(v5, tab));
        __m512i v6 = _mm512_and_si512(r3, _mm512_set1_epi8((char)0x0F));
        _mm512_storeu_si512((void *)(o + 384), _mm512_permutexvar_epi8(v6, tab));
        __m512i v7 = _mm512_and_si512(_mm512_srli_epi16(r3, 4), _mm512_set1_epi8((char)0x0F));
        _mm512_storeu_si512((void *)(o + 448), _mm512_permutexvar_epi8(v7, tab));
    }
}

static void vert512_merge_zmm_d5(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
    const __m512i tab = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)c2s));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 5;
        uint8_t *o = out + ((size_t)b << 9);
        __m512i r0 = _mm512_loadu_si512((const void *)(blk + 0));
        __m512i r1 = _mm512_loadu_si512((const void *)(blk + 64));
        __m512i r2 = _mm512_loadu_si512((const void *)(blk + 128));
        __m512i r3 = _mm512_loadu_si512((const void *)(blk + 192));
        __m512i r4 = _mm512_loadu_si512((const void *)(blk + 256));
        __m512i v0 = _mm512_and_si512(r0, _mm512_set1_epi8((char)0x1F));
        _mm512_storeu_si512((void *)(o + 0), _mm512_permutexvar_epi8(v0, tab));
        __m512i v1 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r0, 5), _mm512_set1_epi8((char)0x07)),
                _mm512_and_si512(_mm512_slli_epi16(r1, 3), _mm512_set1_epi8((char)0xF8))),
                _mm512_set1_epi8((char)0x1F));
        _mm512_storeu_si512((void *)(o + 64), _mm512_permutexvar_epi8(v1, tab));
        __m512i v2 = _mm512_and_si512(_mm512_srli_epi16(r1, 2), _mm512_set1_epi8((char)0x1F));
        _mm512_storeu_si512((void *)(o + 128), _mm512_permutexvar_epi8(v2, tab));
        __m512i v3 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r1, 7), _mm512_set1_epi8((char)0x01)),
                _mm512_and_si512(_mm512_slli_epi16(r2, 1), _mm512_set1_epi8((char)0xFE))),
                _mm512_set1_epi8((char)0x1F));
        _mm512_storeu_si512((void *)(o + 192), _mm512_permutexvar_epi8(v3, tab));
        __m512i v4 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r2, 4), _mm512_set1_epi8((char)0x0F)),
                _mm512_and_si512(_mm512_slli_epi16(r3, 4), _mm512_set1_epi8((char)0xF0))),
                _mm512_set1_epi8((char)0x1F));
        _mm512_storeu_si512((void *)(o + 256), _mm512_permutexvar_epi8(v4, tab));
        __m512i v5 = _mm512_and_si512(_mm512_srli_epi16(r3, 1), _mm512_set1_epi8((char)0x1F));
        _mm512_storeu_si512((void *)(o + 320), _mm512_permutexvar_epi8(v5, tab));
        __m512i v6 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r3, 6), _mm512_set1_epi8((char)0x03)),
                _mm512_and_si512(_mm512_slli_epi16(r4, 2), _mm512_set1_epi8((char)0xFC))),
                _mm512_set1_epi8((char)0x1F));
        _mm512_storeu_si512((void *)(o + 384), _mm512_permutexvar_epi8(v6, tab));
        __m512i v7 = _mm512_and_si512(_mm512_srli_epi16(r4, 3), _mm512_set1_epi8((char)0x1F));
        _mm512_storeu_si512((void *)(o + 448), _mm512_permutexvar_epi8(v7, tab));
    }
}

static void vert512_merge_zmm_d6(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
    const __m512i tab = _mm512_loadu_si512((const void *)c2s);
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 6;
        uint8_t *o = out + ((size_t)b << 9);
        __m512i r0 = _mm512_loadu_si512((const void *)(blk + 0));
        __m512i r1 = _mm512_loadu_si512((const void *)(blk + 64));
        __m512i r2 = _mm512_loadu_si512((const void *)(blk + 128));
        __m512i r3 = _mm512_loadu_si512((const void *)(blk + 192));
        __m512i r4 = _mm512_loadu_si512((const void *)(blk + 256));
        __m512i r5 = _mm512_loadu_si512((const void *)(blk + 320));
        __m512i v0 = _mm512_and_si512(r0, _mm512_set1_epi8((char)0x3F));
        _mm512_storeu_si512((void *)(o + 0), _mm512_permutexvar_epi8(v0, tab));
        __m512i v1 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r0, 6), _mm512_set1_epi8((char)0x03)),
                _mm512_and_si512(_mm512_slli_epi16(r1, 2), _mm512_set1_epi8((char)0xFC))),
                _mm512_set1_epi8((char)0x3F));
        _mm512_storeu_si512((void *)(o + 64), _mm512_permutexvar_epi8(v1, tab));
        __m512i v2 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r1, 4), _mm512_set1_epi8((char)0x0F)),
                _mm512_and_si512(_mm512_slli_epi16(r2, 4), _mm512_set1_epi8((char)0xF0))),
                _mm512_set1_epi8((char)0x3F));
        _mm512_storeu_si512((void *)(o + 128), _mm512_permutexvar_epi8(v2, tab));
        __m512i v3 = _mm512_and_si512(_mm512_srli_epi16(r2, 2), _mm512_set1_epi8((char)0x3F));
        _mm512_storeu_si512((void *)(o + 192), _mm512_permutexvar_epi8(v3, tab));
        __m512i v4 = _mm512_and_si512(r3, _mm512_set1_epi8((char)0x3F));
        _mm512_storeu_si512((void *)(o + 256), _mm512_permutexvar_epi8(v4, tab));
        __m512i v5 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r3, 6), _mm512_set1_epi8((char)0x03)),
                _mm512_and_si512(_mm512_slli_epi16(r4, 2), _mm512_set1_epi8((char)0xFC))),
                _mm512_set1_epi8((char)0x3F));
        _mm512_storeu_si512((void *)(o + 320), _mm512_permutexvar_epi8(v5, tab));
        __m512i v6 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r4, 4), _mm512_set1_epi8((char)0x0F)),
                _mm512_and_si512(_mm512_slli_epi16(r5, 4), _mm512_set1_epi8((char)0xF0))),
                _mm512_set1_epi8((char)0x3F));
        _mm512_storeu_si512((void *)(o + 384), _mm512_permutexvar_epi8(v6, tab));
        __m512i v7 = _mm512_and_si512(_mm512_srli_epi16(r5, 2), _mm512_set1_epi8((char)0x3F));
        _mm512_storeu_si512((void *)(o + 448), _mm512_permutexvar_epi8(v7, tab));
    }
}

static void vert512_merge_zmm_d7(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
    const __m512i tlo = _mm512_loadu_si512((const void *)c2s);
    const __m512i thi = _mm512_loadu_si512((const void *)(c2s + 64));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 7;
        uint8_t *o = out + ((size_t)b << 9);
        __m512i r0 = _mm512_loadu_si512((const void *)(blk + 0));
        __m512i r1 = _mm512_loadu_si512((const void *)(blk + 64));
        __m512i r2 = _mm512_loadu_si512((const void *)(blk + 128));
        __m512i r3 = _mm512_loadu_si512((const void *)(blk + 192));
        __m512i r4 = _mm512_loadu_si512((const void *)(blk + 256));
        __m512i r5 = _mm512_loadu_si512((const void *)(blk + 320));
        __m512i r6 = _mm512_loadu_si512((const void *)(blk + 384));
        __m512i v0 = _mm512_and_si512(r0, _mm512_set1_epi8((char)0x7F));
        _mm512_storeu_si512((void *)(o + 0), _mm512_permutex2var_epi8(tlo, v0, thi));
        __m512i v1 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r0, 7), _mm512_set1_epi8((char)0x01)),
                _mm512_and_si512(_mm512_slli_epi16(r1, 1), _mm512_set1_epi8((char)0xFE))),
                _mm512_set1_epi8((char)0x7F));
        _mm512_storeu_si512((void *)(o + 64), _mm512_permutex2var_epi8(tlo, v1, thi));
        __m512i v2 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r1, 6), _mm512_set1_epi8((char)0x03)),
                _mm512_and_si512(_mm512_slli_epi16(r2, 2), _mm512_set1_epi8((char)0xFC))),
                _mm512_set1_epi8((char)0x7F));
        _mm512_storeu_si512((void *)(o + 128), _mm512_permutex2var_epi8(tlo, v2, thi));
        __m512i v3 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r2, 5), _mm512_set1_epi8((char)0x07)),
                _mm512_and_si512(_mm512_slli_epi16(r3, 3), _mm512_set1_epi8((char)0xF8))),
                _mm512_set1_epi8((char)0x7F));
        _mm512_storeu_si512((void *)(o + 192), _mm512_permutex2var_epi8(tlo, v3, thi));
        __m512i v4 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r3, 4), _mm512_set1_epi8((char)0x0F)),
                _mm512_and_si512(_mm512_slli_epi16(r4, 4), _mm512_set1_epi8((char)0xF0))),
                _mm512_set1_epi8((char)0x7F));
        _mm512_storeu_si512((void *)(o + 256), _mm512_permutex2var_epi8(tlo, v4, thi));
        __m512i v5 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r4, 3), _mm512_set1_epi8((char)0x1F)),
                _mm512_and_si512(_mm512_slli_epi16(r5, 5), _mm512_set1_epi8((char)0xE0))),
                _mm512_set1_epi8((char)0x7F));
        _mm512_storeu_si512((void *)(o + 320), _mm512_permutex2var_epi8(tlo, v5, thi));
        __m512i v6 = _mm512_and_si512(_mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(r5, 2), _mm512_set1_epi8((char)0x3F)),
                _mm512_and_si512(_mm512_slli_epi16(r6, 6), _mm512_set1_epi8((char)0xC0))),
                _mm512_set1_epi8((char)0x7F));
        _mm512_storeu_si512((void *)(o + 384), _mm512_permutex2var_epi8(tlo, v6, thi));
        __m512i v7 = _mm512_and_si512(_mm512_srli_epi16(r6, 1), _mm512_set1_epi8((char)0x7F));
        _mm512_storeu_si512((void *)(o + 448), _mm512_permutex2var_epi8(tlo, v7, thi));
    }
}

static inline void vert512_merge_zmm(uint8_t *out, int n_v, const uint8_t *bm, int D, const uint8_t *c2s)
{
    switch (D) {
    case 2: vert512_merge_zmm_d2(out, n_v, bm, c2s); break;
    case 3: vert512_merge_zmm_d3(out, n_v, bm, c2s); break;
    case 4: vert512_merge_zmm_d4(out, n_v, bm, c2s); break;
    case 5: vert512_merge_zmm_d5(out, n_v, bm, c2s); break;
    case 6: vert512_merge_zmm_d6(out, n_v, bm, c2s); break;
    default: vert512_merge_zmm_d7(out, n_v, bm, c2s); break;
    }
}

static void vert512_pack_zmm_d2(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m512i basev = _mm512_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 2;
        const uint8_t *r = ranks + ((size_t)b << 9);
        __m512i c0, c1;
        __m512i v0 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 0)), basev);
        c0 = v0;
        __m512i v1 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 64)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v1, 2), _mm512_set1_epi8((char)0xFC)));
        __m512i v2 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 128)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v2, 4), _mm512_set1_epi8((char)0xF0)));
        __m512i v3 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 192)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v3, 6), _mm512_set1_epi8((char)0xC0)));
        __m512i v4 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 256)), basev);
        c1 = v4;
        __m512i v5 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 320)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v5, 2), _mm512_set1_epi8((char)0xFC)));
        __m512i v6 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 384)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v6, 4), _mm512_set1_epi8((char)0xF0)));
        __m512i v7 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 448)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v7, 6), _mm512_set1_epi8((char)0xC0)));
        _mm512_storeu_si512((void *)(blk + 0), c0);
        _mm512_storeu_si512((void *)(blk + 64), c1);
    }
}

static void vert512_pack_zmm_d3(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m512i basev = _mm512_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 3;
        const uint8_t *r = ranks + ((size_t)b << 9);
        __m512i c0, c1, c2;
        __m512i v0 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 0)), basev);
        c0 = v0;
        __m512i v1 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 64)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v1, 3), _mm512_set1_epi8((char)0xF8)));
        __m512i v2 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 128)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v2, 6), _mm512_set1_epi8((char)0xC0)));
        c1 = _mm512_and_si512(_mm512_srli_epi16(v2, 2), _mm512_set1_epi8((char)0x3F));
        __m512i v3 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 192)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v3, 1), _mm512_set1_epi8((char)0xFE)));
        __m512i v4 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 256)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v4, 4), _mm512_set1_epi8((char)0xF0)));
        __m512i v5 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 320)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v5, 7), _mm512_set1_epi8((char)0x80)));
        c2 = _mm512_and_si512(_mm512_srli_epi16(v5, 1), _mm512_set1_epi8((char)0x7F));
        __m512i v6 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 384)), basev);
        c2 = _mm512_or_si512(c2, _mm512_and_si512(_mm512_slli_epi16(v6, 2), _mm512_set1_epi8((char)0xFC)));
        __m512i v7 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 448)), basev);
        c2 = _mm512_or_si512(c2, _mm512_and_si512(_mm512_slli_epi16(v7, 5), _mm512_set1_epi8((char)0xE0)));
        _mm512_storeu_si512((void *)(blk + 0), c0);
        _mm512_storeu_si512((void *)(blk + 64), c1);
        _mm512_storeu_si512((void *)(blk + 128), c2);
    }
}

static void vert512_pack_zmm_d4(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m512i basev = _mm512_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 4;
        const uint8_t *r = ranks + ((size_t)b << 9);
        __m512i c0, c1, c2, c3;
        __m512i v0 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 0)), basev);
        c0 = v0;
        __m512i v1 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 64)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v1, 4), _mm512_set1_epi8((char)0xF0)));
        __m512i v2 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 128)), basev);
        c1 = v2;
        __m512i v3 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 192)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v3, 4), _mm512_set1_epi8((char)0xF0)));
        __m512i v4 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 256)), basev);
        c2 = v4;
        __m512i v5 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 320)), basev);
        c2 = _mm512_or_si512(c2, _mm512_and_si512(_mm512_slli_epi16(v5, 4), _mm512_set1_epi8((char)0xF0)));
        __m512i v6 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 384)), basev);
        c3 = v6;
        __m512i v7 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 448)), basev);
        c3 = _mm512_or_si512(c3, _mm512_and_si512(_mm512_slli_epi16(v7, 4), _mm512_set1_epi8((char)0xF0)));
        _mm512_storeu_si512((void *)(blk + 0), c0);
        _mm512_storeu_si512((void *)(blk + 64), c1);
        _mm512_storeu_si512((void *)(blk + 128), c2);
        _mm512_storeu_si512((void *)(blk + 192), c3);
    }
}

static void vert512_pack_zmm_d5(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m512i basev = _mm512_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 5;
        const uint8_t *r = ranks + ((size_t)b << 9);
        __m512i c0, c1, c2, c3, c4;
        __m512i v0 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 0)), basev);
        c0 = v0;
        __m512i v1 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 64)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v1, 5), _mm512_set1_epi8((char)0xE0)));
        c1 = _mm512_and_si512(_mm512_srli_epi16(v1, 3), _mm512_set1_epi8((char)0x1F));
        __m512i v2 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 128)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v2, 2), _mm512_set1_epi8((char)0xFC)));
        __m512i v3 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 192)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v3, 7), _mm512_set1_epi8((char)0x80)));
        c2 = _mm512_and_si512(_mm512_srli_epi16(v3, 1), _mm512_set1_epi8((char)0x7F));
        __m512i v4 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 256)), basev);
        c2 = _mm512_or_si512(c2, _mm512_and_si512(_mm512_slli_epi16(v4, 4), _mm512_set1_epi8((char)0xF0)));
        c3 = _mm512_and_si512(_mm512_srli_epi16(v4, 4), _mm512_set1_epi8((char)0x0F));
        __m512i v5 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 320)), basev);
        c3 = _mm512_or_si512(c3, _mm512_and_si512(_mm512_slli_epi16(v5, 1), _mm512_set1_epi8((char)0xFE)));
        __m512i v6 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 384)), basev);
        c3 = _mm512_or_si512(c3, _mm512_and_si512(_mm512_slli_epi16(v6, 6), _mm512_set1_epi8((char)0xC0)));
        c4 = _mm512_and_si512(_mm512_srli_epi16(v6, 2), _mm512_set1_epi8((char)0x3F));
        __m512i v7 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 448)), basev);
        c4 = _mm512_or_si512(c4, _mm512_and_si512(_mm512_slli_epi16(v7, 3), _mm512_set1_epi8((char)0xF8)));
        _mm512_storeu_si512((void *)(blk + 0), c0);
        _mm512_storeu_si512((void *)(blk + 64), c1);
        _mm512_storeu_si512((void *)(blk + 128), c2);
        _mm512_storeu_si512((void *)(blk + 192), c3);
        _mm512_storeu_si512((void *)(blk + 256), c4);
    }
}

static void vert512_pack_zmm_d6(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m512i basev = _mm512_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 6;
        const uint8_t *r = ranks + ((size_t)b << 9);
        __m512i c0, c1, c2, c3, c4, c5;
        __m512i v0 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 0)), basev);
        c0 = v0;
        __m512i v1 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 64)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v1, 6), _mm512_set1_epi8((char)0xC0)));
        c1 = _mm512_and_si512(_mm512_srli_epi16(v1, 2), _mm512_set1_epi8((char)0x3F));
        __m512i v2 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 128)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v2, 4), _mm512_set1_epi8((char)0xF0)));
        c2 = _mm512_and_si512(_mm512_srli_epi16(v2, 4), _mm512_set1_epi8((char)0x0F));
        __m512i v3 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 192)), basev);
        c2 = _mm512_or_si512(c2, _mm512_and_si512(_mm512_slli_epi16(v3, 2), _mm512_set1_epi8((char)0xFC)));
        __m512i v4 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 256)), basev);
        c3 = v4;
        __m512i v5 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 320)), basev);
        c3 = _mm512_or_si512(c3, _mm512_and_si512(_mm512_slli_epi16(v5, 6), _mm512_set1_epi8((char)0xC0)));
        c4 = _mm512_and_si512(_mm512_srli_epi16(v5, 2), _mm512_set1_epi8((char)0x3F));
        __m512i v6 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 384)), basev);
        c4 = _mm512_or_si512(c4, _mm512_and_si512(_mm512_slli_epi16(v6, 4), _mm512_set1_epi8((char)0xF0)));
        c5 = _mm512_and_si512(_mm512_srli_epi16(v6, 4), _mm512_set1_epi8((char)0x0F));
        __m512i v7 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 448)), basev);
        c5 = _mm512_or_si512(c5, _mm512_and_si512(_mm512_slli_epi16(v7, 2), _mm512_set1_epi8((char)0xFC)));
        _mm512_storeu_si512((void *)(blk + 0), c0);
        _mm512_storeu_si512((void *)(blk + 64), c1);
        _mm512_storeu_si512((void *)(blk + 128), c2);
        _mm512_storeu_si512((void *)(blk + 192), c3);
        _mm512_storeu_si512((void *)(blk + 256), c4);
        _mm512_storeu_si512((void *)(blk + 320), c5);
    }
}

static void vert512_pack_zmm_d7(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m512i basev = _mm512_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 7;
        const uint8_t *r = ranks + ((size_t)b << 9);
        __m512i c0, c1, c2, c3, c4, c5, c6;
        __m512i v0 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 0)), basev);
        c0 = v0;
        __m512i v1 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 64)), basev);
        c0 = _mm512_or_si512(c0, _mm512_and_si512(_mm512_slli_epi16(v1, 7), _mm512_set1_epi8((char)0x80)));
        c1 = _mm512_and_si512(_mm512_srli_epi16(v1, 1), _mm512_set1_epi8((char)0x7F));
        __m512i v2 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 128)), basev);
        c1 = _mm512_or_si512(c1, _mm512_and_si512(_mm512_slli_epi16(v2, 6), _mm512_set1_epi8((char)0xC0)));
        c2 = _mm512_and_si512(_mm512_srli_epi16(v2, 2), _mm512_set1_epi8((char)0x3F));
        __m512i v3 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 192)), basev);
        c2 = _mm512_or_si512(c2, _mm512_and_si512(_mm512_slli_epi16(v3, 5), _mm512_set1_epi8((char)0xE0)));
        c3 = _mm512_and_si512(_mm512_srli_epi16(v3, 3), _mm512_set1_epi8((char)0x1F));
        __m512i v4 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 256)), basev);
        c3 = _mm512_or_si512(c3, _mm512_and_si512(_mm512_slli_epi16(v4, 4), _mm512_set1_epi8((char)0xF0)));
        c4 = _mm512_and_si512(_mm512_srli_epi16(v4, 4), _mm512_set1_epi8((char)0x0F));
        __m512i v5 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 320)), basev);
        c4 = _mm512_or_si512(c4, _mm512_and_si512(_mm512_slli_epi16(v5, 3), _mm512_set1_epi8((char)0xF8)));
        c5 = _mm512_and_si512(_mm512_srli_epi16(v5, 5), _mm512_set1_epi8((char)0x07));
        __m512i v6 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 384)), basev);
        c5 = _mm512_or_si512(c5, _mm512_and_si512(_mm512_slli_epi16(v6, 2), _mm512_set1_epi8((char)0xFC)));
        c6 = _mm512_and_si512(_mm512_srli_epi16(v6, 6), _mm512_set1_epi8((char)0x03));
        __m512i v7 = _mm512_sub_epi8(_mm512_loadu_si512((const void *)(r + 448)), basev);
        c6 = _mm512_or_si512(c6, _mm512_and_si512(_mm512_slli_epi16(v7, 1), _mm512_set1_epi8((char)0xFE)));
        _mm512_storeu_si512((void *)(blk + 0), c0);
        _mm512_storeu_si512((void *)(blk + 64), c1);
        _mm512_storeu_si512((void *)(blk + 128), c2);
        _mm512_storeu_si512((void *)(blk + 192), c3);
        _mm512_storeu_si512((void *)(blk + 256), c4);
        _mm512_storeu_si512((void *)(blk + 320), c5);
        _mm512_storeu_si512((void *)(blk + 384), c6);
    }
}

static inline void vert512_pack_zmm(uint8_t *out, const uint8_t *ranks, int n_v, int D, uint8_t base)
{
    switch (D) {
    case 2: vert512_pack_zmm_d2(out, ranks, n_v, base); break;
    case 3: vert512_pack_zmm_d3(out, ranks, n_v, base); break;
    case 4: vert512_pack_zmm_d4(out, ranks, n_v, base); break;
    case 5: vert512_pack_zmm_d5(out, ranks, n_v, base); break;
    case 6: vert512_pack_zmm_d6(out, ranks, n_v, base); break;
    default: vert512_pack_zmm_d7(out, ranks, n_v, base); break;
    }
}

#endif /* AVX512BW + VBMI + VL */


#if defined(__AVX2__)

static void vert512_merge_ymm_d2(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
        const __m256i t0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)c2s));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 2;
        uint8_t *o = out + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            const uint8_t *bh = blk + 32 * h;
            uint8_t *oh = o + 32 * h;
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(bh + 0));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(bh + 64));
            __m256i v0 = _mm256_and_si256(r0, _mm256_set1_epi8((char)0x03));
            _mm256_storeu_si256((__m256i *)(oh + 0),
                _mm256_shuffle_epi8(t0, v0));
            __m256i v1 = _mm256_and_si256(_mm256_srli_epi16(r0, 2), _mm256_set1_epi8((char)0x03));
            _mm256_storeu_si256((__m256i *)(oh + 64),
                _mm256_shuffle_epi8(t0, v1));
            __m256i v2 = _mm256_and_si256(_mm256_srli_epi16(r0, 4), _mm256_set1_epi8((char)0x03));
            _mm256_storeu_si256((__m256i *)(oh + 128),
                _mm256_shuffle_epi8(t0, v2));
            __m256i v3 = _mm256_and_si256(_mm256_srli_epi16(r0, 6), _mm256_set1_epi8((char)0x03));
            _mm256_storeu_si256((__m256i *)(oh + 192),
                _mm256_shuffle_epi8(t0, v3));
            __m256i v4 = _mm256_and_si256(r1, _mm256_set1_epi8((char)0x03));
            _mm256_storeu_si256((__m256i *)(oh + 256),
                _mm256_shuffle_epi8(t0, v4));
            __m256i v5 = _mm256_and_si256(_mm256_srli_epi16(r1, 2), _mm256_set1_epi8((char)0x03));
            _mm256_storeu_si256((__m256i *)(oh + 320),
                _mm256_shuffle_epi8(t0, v5));
            __m256i v6 = _mm256_and_si256(_mm256_srli_epi16(r1, 4), _mm256_set1_epi8((char)0x03));
            _mm256_storeu_si256((__m256i *)(oh + 384),
                _mm256_shuffle_epi8(t0, v6));
            __m256i v7 = _mm256_and_si256(_mm256_srli_epi16(r1, 6), _mm256_set1_epi8((char)0x03));
            _mm256_storeu_si256((__m256i *)(oh + 448),
                _mm256_shuffle_epi8(t0, v7));
        }
    }
}

static void vert512_merge_ymm_d3(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
        const __m256i t0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)c2s));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 3;
        uint8_t *o = out + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            const uint8_t *bh = blk + 32 * h;
            uint8_t *oh = o + 32 * h;
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(bh + 0));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(bh + 64));
            __m256i r2 = _mm256_loadu_si256((const __m256i *)(bh + 128));
            __m256i v0 = _mm256_and_si256(r0, _mm256_set1_epi8((char)0x07));
            _mm256_storeu_si256((__m256i *)(oh + 0),
                _mm256_shuffle_epi8(t0, v0));
            __m256i v1 = _mm256_and_si256(_mm256_srli_epi16(r0, 3), _mm256_set1_epi8((char)0x07));
            _mm256_storeu_si256((__m256i *)(oh + 64),
                _mm256_shuffle_epi8(t0, v1));
            __m256i v2 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r0, 6), _mm256_set1_epi8((char)0x03)),
                _mm256_and_si256(_mm256_slli_epi16(r1, 2), _mm256_set1_epi8((char)0xFC))),
                _mm256_set1_epi8((char)0x07));
            _mm256_storeu_si256((__m256i *)(oh + 128),
                _mm256_shuffle_epi8(t0, v2));
            __m256i v3 = _mm256_and_si256(_mm256_srli_epi16(r1, 1), _mm256_set1_epi8((char)0x07));
            _mm256_storeu_si256((__m256i *)(oh + 192),
                _mm256_shuffle_epi8(t0, v3));
            __m256i v4 = _mm256_and_si256(_mm256_srli_epi16(r1, 4), _mm256_set1_epi8((char)0x07));
            _mm256_storeu_si256((__m256i *)(oh + 256),
                _mm256_shuffle_epi8(t0, v4));
            __m256i v5 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r1, 7), _mm256_set1_epi8((char)0x01)),
                _mm256_and_si256(_mm256_slli_epi16(r2, 1), _mm256_set1_epi8((char)0xFE))),
                _mm256_set1_epi8((char)0x07));
            _mm256_storeu_si256((__m256i *)(oh + 320),
                _mm256_shuffle_epi8(t0, v5));
            __m256i v6 = _mm256_and_si256(_mm256_srli_epi16(r2, 2), _mm256_set1_epi8((char)0x07));
            _mm256_storeu_si256((__m256i *)(oh + 384),
                _mm256_shuffle_epi8(t0, v6));
            __m256i v7 = _mm256_and_si256(_mm256_srli_epi16(r2, 5), _mm256_set1_epi8((char)0x07));
            _mm256_storeu_si256((__m256i *)(oh + 448),
                _mm256_shuffle_epi8(t0, v7));
        }
    }
}

static void vert512_merge_ymm_d4(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
        const __m256i t0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)c2s));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 4;
        uint8_t *o = out + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            const uint8_t *bh = blk + 32 * h;
            uint8_t *oh = o + 32 * h;
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(bh + 0));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(bh + 64));
            __m256i r2 = _mm256_loadu_si256((const __m256i *)(bh + 128));
            __m256i r3 = _mm256_loadu_si256((const __m256i *)(bh + 192));
            __m256i v0 = _mm256_and_si256(r0, _mm256_set1_epi8((char)0x0F));
            _mm256_storeu_si256((__m256i *)(oh + 0),
                _mm256_shuffle_epi8(t0, v0));
            __m256i v1 = _mm256_and_si256(_mm256_srli_epi16(r0, 4), _mm256_set1_epi8((char)0x0F));
            _mm256_storeu_si256((__m256i *)(oh + 64),
                _mm256_shuffle_epi8(t0, v1));
            __m256i v2 = _mm256_and_si256(r1, _mm256_set1_epi8((char)0x0F));
            _mm256_storeu_si256((__m256i *)(oh + 128),
                _mm256_shuffle_epi8(t0, v2));
            __m256i v3 = _mm256_and_si256(_mm256_srli_epi16(r1, 4), _mm256_set1_epi8((char)0x0F));
            _mm256_storeu_si256((__m256i *)(oh + 192),
                _mm256_shuffle_epi8(t0, v3));
            __m256i v4 = _mm256_and_si256(r2, _mm256_set1_epi8((char)0x0F));
            _mm256_storeu_si256((__m256i *)(oh + 256),
                _mm256_shuffle_epi8(t0, v4));
            __m256i v5 = _mm256_and_si256(_mm256_srli_epi16(r2, 4), _mm256_set1_epi8((char)0x0F));
            _mm256_storeu_si256((__m256i *)(oh + 320),
                _mm256_shuffle_epi8(t0, v5));
            __m256i v6 = _mm256_and_si256(r3, _mm256_set1_epi8((char)0x0F));
            _mm256_storeu_si256((__m256i *)(oh + 384),
                _mm256_shuffle_epi8(t0, v6));
            __m256i v7 = _mm256_and_si256(_mm256_srli_epi16(r3, 4), _mm256_set1_epi8((char)0x0F));
            _mm256_storeu_si256((__m256i *)(oh + 448),
                _mm256_shuffle_epi8(t0, v7));
        }
    }
}

static void vert512_merge_ymm_d5(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
        const __m256i t0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 0)));
        const __m256i t1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 16)));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 5;
        uint8_t *o = out + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            const uint8_t *bh = blk + 32 * h;
            uint8_t *oh = o + 32 * h;
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(bh + 0));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(bh + 64));
            __m256i r2 = _mm256_loadu_si256((const __m256i *)(bh + 128));
            __m256i r3 = _mm256_loadu_si256((const __m256i *)(bh + 192));
            __m256i r4 = _mm256_loadu_si256((const __m256i *)(bh + 256));
            __m256i v0 = _mm256_and_si256(r0, _mm256_set1_epi8((char)0x1F));
            _mm256_storeu_si256((__m256i *)(oh + 0),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v0),
                _mm256_shuffle_epi8(t1, v0), _mm256_slli_epi16(v0, 3)));
            __m256i v1 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r0, 5), _mm256_set1_epi8((char)0x07)),
                _mm256_and_si256(_mm256_slli_epi16(r1, 3), _mm256_set1_epi8((char)0xF8))),
                _mm256_set1_epi8((char)0x1F));
            _mm256_storeu_si256((__m256i *)(oh + 64),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v1),
                _mm256_shuffle_epi8(t1, v1), _mm256_slli_epi16(v1, 3)));
            __m256i v2 = _mm256_and_si256(_mm256_srli_epi16(r1, 2), _mm256_set1_epi8((char)0x1F));
            _mm256_storeu_si256((__m256i *)(oh + 128),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v2),
                _mm256_shuffle_epi8(t1, v2), _mm256_slli_epi16(v2, 3)));
            __m256i v3 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r1, 7), _mm256_set1_epi8((char)0x01)),
                _mm256_and_si256(_mm256_slli_epi16(r2, 1), _mm256_set1_epi8((char)0xFE))),
                _mm256_set1_epi8((char)0x1F));
            _mm256_storeu_si256((__m256i *)(oh + 192),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v3),
                _mm256_shuffle_epi8(t1, v3), _mm256_slli_epi16(v3, 3)));
            __m256i v4 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r2, 4), _mm256_set1_epi8((char)0x0F)),
                _mm256_and_si256(_mm256_slli_epi16(r3, 4), _mm256_set1_epi8((char)0xF0))),
                _mm256_set1_epi8((char)0x1F));
            _mm256_storeu_si256((__m256i *)(oh + 256),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v4),
                _mm256_shuffle_epi8(t1, v4), _mm256_slli_epi16(v4, 3)));
            __m256i v5 = _mm256_and_si256(_mm256_srli_epi16(r3, 1), _mm256_set1_epi8((char)0x1F));
            _mm256_storeu_si256((__m256i *)(oh + 320),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v5),
                _mm256_shuffle_epi8(t1, v5), _mm256_slli_epi16(v5, 3)));
            __m256i v6 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r3, 6), _mm256_set1_epi8((char)0x03)),
                _mm256_and_si256(_mm256_slli_epi16(r4, 2), _mm256_set1_epi8((char)0xFC))),
                _mm256_set1_epi8((char)0x1F));
            _mm256_storeu_si256((__m256i *)(oh + 384),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v6),
                _mm256_shuffle_epi8(t1, v6), _mm256_slli_epi16(v6, 3)));
            __m256i v7 = _mm256_and_si256(_mm256_srli_epi16(r4, 3), _mm256_set1_epi8((char)0x1F));
            _mm256_storeu_si256((__m256i *)(oh + 448),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v7),
                _mm256_shuffle_epi8(t1, v7), _mm256_slli_epi16(v7, 3)));
        }
    }
}

static void vert512_merge_ymm_d6(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
        const __m256i t0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 0)));
        const __m256i t1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 16)));
        const __m256i t2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 32)));
        const __m256i t3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 48)));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 6;
        uint8_t *o = out + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            const uint8_t *bh = blk + 32 * h;
            uint8_t *oh = o + 32 * h;
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(bh + 0));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(bh + 64));
            __m256i r2 = _mm256_loadu_si256((const __m256i *)(bh + 128));
            __m256i r3 = _mm256_loadu_si256((const __m256i *)(bh + 192));
            __m256i r4 = _mm256_loadu_si256((const __m256i *)(bh + 256));
            __m256i r5 = _mm256_loadu_si256((const __m256i *)(bh + 320));
            __m256i v0 = _mm256_and_si256(r0, _mm256_set1_epi8((char)0x3F));
            _mm256_storeu_si256((__m256i *)(oh + 0),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v0),
                _mm256_shuffle_epi8(t1, v0), _mm256_slli_epi16(v0, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v0),
                _mm256_shuffle_epi8(t3, v0), _mm256_slli_epi16(v0, 3)), _mm256_slli_epi16(v0, 2)));
            __m256i v1 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r0, 6), _mm256_set1_epi8((char)0x03)),
                _mm256_and_si256(_mm256_slli_epi16(r1, 2), _mm256_set1_epi8((char)0xFC))),
                _mm256_set1_epi8((char)0x3F));
            _mm256_storeu_si256((__m256i *)(oh + 64),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v1),
                _mm256_shuffle_epi8(t1, v1), _mm256_slli_epi16(v1, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v1),
                _mm256_shuffle_epi8(t3, v1), _mm256_slli_epi16(v1, 3)), _mm256_slli_epi16(v1, 2)));
            __m256i v2 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r1, 4), _mm256_set1_epi8((char)0x0F)),
                _mm256_and_si256(_mm256_slli_epi16(r2, 4), _mm256_set1_epi8((char)0xF0))),
                _mm256_set1_epi8((char)0x3F));
            _mm256_storeu_si256((__m256i *)(oh + 128),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v2),
                _mm256_shuffle_epi8(t1, v2), _mm256_slli_epi16(v2, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v2),
                _mm256_shuffle_epi8(t3, v2), _mm256_slli_epi16(v2, 3)), _mm256_slli_epi16(v2, 2)));
            __m256i v3 = _mm256_and_si256(_mm256_srli_epi16(r2, 2), _mm256_set1_epi8((char)0x3F));
            _mm256_storeu_si256((__m256i *)(oh + 192),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v3),
                _mm256_shuffle_epi8(t1, v3), _mm256_slli_epi16(v3, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v3),
                _mm256_shuffle_epi8(t3, v3), _mm256_slli_epi16(v3, 3)), _mm256_slli_epi16(v3, 2)));
            __m256i v4 = _mm256_and_si256(r3, _mm256_set1_epi8((char)0x3F));
            _mm256_storeu_si256((__m256i *)(oh + 256),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v4),
                _mm256_shuffle_epi8(t1, v4), _mm256_slli_epi16(v4, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v4),
                _mm256_shuffle_epi8(t3, v4), _mm256_slli_epi16(v4, 3)), _mm256_slli_epi16(v4, 2)));
            __m256i v5 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r3, 6), _mm256_set1_epi8((char)0x03)),
                _mm256_and_si256(_mm256_slli_epi16(r4, 2), _mm256_set1_epi8((char)0xFC))),
                _mm256_set1_epi8((char)0x3F));
            _mm256_storeu_si256((__m256i *)(oh + 320),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v5),
                _mm256_shuffle_epi8(t1, v5), _mm256_slli_epi16(v5, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v5),
                _mm256_shuffle_epi8(t3, v5), _mm256_slli_epi16(v5, 3)), _mm256_slli_epi16(v5, 2)));
            __m256i v6 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r4, 4), _mm256_set1_epi8((char)0x0F)),
                _mm256_and_si256(_mm256_slli_epi16(r5, 4), _mm256_set1_epi8((char)0xF0))),
                _mm256_set1_epi8((char)0x3F));
            _mm256_storeu_si256((__m256i *)(oh + 384),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v6),
                _mm256_shuffle_epi8(t1, v6), _mm256_slli_epi16(v6, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v6),
                _mm256_shuffle_epi8(t3, v6), _mm256_slli_epi16(v6, 3)), _mm256_slli_epi16(v6, 2)));
            __m256i v7 = _mm256_and_si256(_mm256_srli_epi16(r5, 2), _mm256_set1_epi8((char)0x3F));
            _mm256_storeu_si256((__m256i *)(oh + 448),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v7),
                _mm256_shuffle_epi8(t1, v7), _mm256_slli_epi16(v7, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v7),
                _mm256_shuffle_epi8(t3, v7), _mm256_slli_epi16(v7, 3)), _mm256_slli_epi16(v7, 2)));
        }
    }
}

static void vert512_merge_ymm_d7(uint8_t *out, int n_v, const uint8_t *bm,
                                   const uint8_t *c2s)
{
        const __m256i t0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 0)));
        const __m256i t1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 16)));
        const __m256i t2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 32)));
        const __m256i t3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 48)));
        const __m256i t4 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 64)));
        const __m256i t5 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 80)));
        const __m256i t6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 96)));
        const __m256i t7 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(c2s + 112)));
    for (int b = 0; b < n_v >> 9; b++) {
        const uint8_t *blk = bm + (size_t)b * 64 * 7;
        uint8_t *o = out + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            const uint8_t *bh = blk + 32 * h;
            uint8_t *oh = o + 32 * h;
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(bh + 0));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(bh + 64));
            __m256i r2 = _mm256_loadu_si256((const __m256i *)(bh + 128));
            __m256i r3 = _mm256_loadu_si256((const __m256i *)(bh + 192));
            __m256i r4 = _mm256_loadu_si256((const __m256i *)(bh + 256));
            __m256i r5 = _mm256_loadu_si256((const __m256i *)(bh + 320));
            __m256i r6 = _mm256_loadu_si256((const __m256i *)(bh + 384));
            __m256i v0 = _mm256_and_si256(r0, _mm256_set1_epi8((char)0x7F));
            _mm256_storeu_si256((__m256i *)(oh + 0),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v0),
                _mm256_shuffle_epi8(t1, v0), _mm256_slli_epi16(v0, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v0),
                _mm256_shuffle_epi8(t3, v0), _mm256_slli_epi16(v0, 3)), _mm256_slli_epi16(v0, 2)),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t4, v0),
                _mm256_shuffle_epi8(t5, v0), _mm256_slli_epi16(v0, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t6, v0),
                _mm256_shuffle_epi8(t7, v0), _mm256_slli_epi16(v0, 3)), _mm256_slli_epi16(v0, 2)), _mm256_slli_epi16(v0, 1)));
            __m256i v1 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r0, 7), _mm256_set1_epi8((char)0x01)),
                _mm256_and_si256(_mm256_slli_epi16(r1, 1), _mm256_set1_epi8((char)0xFE))),
                _mm256_set1_epi8((char)0x7F));
            _mm256_storeu_si256((__m256i *)(oh + 64),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v1),
                _mm256_shuffle_epi8(t1, v1), _mm256_slli_epi16(v1, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v1),
                _mm256_shuffle_epi8(t3, v1), _mm256_slli_epi16(v1, 3)), _mm256_slli_epi16(v1, 2)),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t4, v1),
                _mm256_shuffle_epi8(t5, v1), _mm256_slli_epi16(v1, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t6, v1),
                _mm256_shuffle_epi8(t7, v1), _mm256_slli_epi16(v1, 3)), _mm256_slli_epi16(v1, 2)), _mm256_slli_epi16(v1, 1)));
            __m256i v2 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r1, 6), _mm256_set1_epi8((char)0x03)),
                _mm256_and_si256(_mm256_slli_epi16(r2, 2), _mm256_set1_epi8((char)0xFC))),
                _mm256_set1_epi8((char)0x7F));
            _mm256_storeu_si256((__m256i *)(oh + 128),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v2),
                _mm256_shuffle_epi8(t1, v2), _mm256_slli_epi16(v2, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v2),
                _mm256_shuffle_epi8(t3, v2), _mm256_slli_epi16(v2, 3)), _mm256_slli_epi16(v2, 2)),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t4, v2),
                _mm256_shuffle_epi8(t5, v2), _mm256_slli_epi16(v2, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t6, v2),
                _mm256_shuffle_epi8(t7, v2), _mm256_slli_epi16(v2, 3)), _mm256_slli_epi16(v2, 2)), _mm256_slli_epi16(v2, 1)));
            __m256i v3 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r2, 5), _mm256_set1_epi8((char)0x07)),
                _mm256_and_si256(_mm256_slli_epi16(r3, 3), _mm256_set1_epi8((char)0xF8))),
                _mm256_set1_epi8((char)0x7F));
            _mm256_storeu_si256((__m256i *)(oh + 192),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v3),
                _mm256_shuffle_epi8(t1, v3), _mm256_slli_epi16(v3, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v3),
                _mm256_shuffle_epi8(t3, v3), _mm256_slli_epi16(v3, 3)), _mm256_slli_epi16(v3, 2)),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t4, v3),
                _mm256_shuffle_epi8(t5, v3), _mm256_slli_epi16(v3, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t6, v3),
                _mm256_shuffle_epi8(t7, v3), _mm256_slli_epi16(v3, 3)), _mm256_slli_epi16(v3, 2)), _mm256_slli_epi16(v3, 1)));
            __m256i v4 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r3, 4), _mm256_set1_epi8((char)0x0F)),
                _mm256_and_si256(_mm256_slli_epi16(r4, 4), _mm256_set1_epi8((char)0xF0))),
                _mm256_set1_epi8((char)0x7F));
            _mm256_storeu_si256((__m256i *)(oh + 256),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v4),
                _mm256_shuffle_epi8(t1, v4), _mm256_slli_epi16(v4, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v4),
                _mm256_shuffle_epi8(t3, v4), _mm256_slli_epi16(v4, 3)), _mm256_slli_epi16(v4, 2)),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t4, v4),
                _mm256_shuffle_epi8(t5, v4), _mm256_slli_epi16(v4, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t6, v4),
                _mm256_shuffle_epi8(t7, v4), _mm256_slli_epi16(v4, 3)), _mm256_slli_epi16(v4, 2)), _mm256_slli_epi16(v4, 1)));
            __m256i v5 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r4, 3), _mm256_set1_epi8((char)0x1F)),
                _mm256_and_si256(_mm256_slli_epi16(r5, 5), _mm256_set1_epi8((char)0xE0))),
                _mm256_set1_epi8((char)0x7F));
            _mm256_storeu_si256((__m256i *)(oh + 320),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v5),
                _mm256_shuffle_epi8(t1, v5), _mm256_slli_epi16(v5, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v5),
                _mm256_shuffle_epi8(t3, v5), _mm256_slli_epi16(v5, 3)), _mm256_slli_epi16(v5, 2)),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t4, v5),
                _mm256_shuffle_epi8(t5, v5), _mm256_slli_epi16(v5, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t6, v5),
                _mm256_shuffle_epi8(t7, v5), _mm256_slli_epi16(v5, 3)), _mm256_slli_epi16(v5, 2)), _mm256_slli_epi16(v5, 1)));
            __m256i v6 = _mm256_and_si256(_mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(r5, 2), _mm256_set1_epi8((char)0x3F)),
                _mm256_and_si256(_mm256_slli_epi16(r6, 6), _mm256_set1_epi8((char)0xC0))),
                _mm256_set1_epi8((char)0x7F));
            _mm256_storeu_si256((__m256i *)(oh + 384),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v6),
                _mm256_shuffle_epi8(t1, v6), _mm256_slli_epi16(v6, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v6),
                _mm256_shuffle_epi8(t3, v6), _mm256_slli_epi16(v6, 3)), _mm256_slli_epi16(v6, 2)),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t4, v6),
                _mm256_shuffle_epi8(t5, v6), _mm256_slli_epi16(v6, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t6, v6),
                _mm256_shuffle_epi8(t7, v6), _mm256_slli_epi16(v6, 3)), _mm256_slli_epi16(v6, 2)), _mm256_slli_epi16(v6, 1)));
            __m256i v7 = _mm256_and_si256(_mm256_srli_epi16(r6, 1), _mm256_set1_epi8((char)0x7F));
            _mm256_storeu_si256((__m256i *)(oh + 448),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t0, v7),
                _mm256_shuffle_epi8(t1, v7), _mm256_slli_epi16(v7, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t2, v7),
                _mm256_shuffle_epi8(t3, v7), _mm256_slli_epi16(v7, 3)), _mm256_slli_epi16(v7, 2)),
                _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t4, v7),
                _mm256_shuffle_epi8(t5, v7), _mm256_slli_epi16(v7, 3)),
                _mm256_blendv_epi8(
                _mm256_shuffle_epi8(t6, v7),
                _mm256_shuffle_epi8(t7, v7), _mm256_slli_epi16(v7, 3)), _mm256_slli_epi16(v7, 2)), _mm256_slli_epi16(v7, 1)));
        }
    }
}

static inline void vert512_merge_ymm(uint8_t *out, int n_v, const uint8_t *bm, int D, const uint8_t *c2s)
{
    switch (D) {
    case 2: vert512_merge_ymm_d2(out, n_v, bm, c2s); break;
    case 3: vert512_merge_ymm_d3(out, n_v, bm, c2s); break;
    case 4: vert512_merge_ymm_d4(out, n_v, bm, c2s); break;
    case 5: vert512_merge_ymm_d5(out, n_v, bm, c2s); break;
    case 6: vert512_merge_ymm_d6(out, n_v, bm, c2s); break;
    default: vert512_merge_ymm_d7(out, n_v, bm, c2s); break;
    }
}

static void vert512_pack_ymm_d2(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m256i basev = _mm256_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 2;
        const uint8_t *r = ranks + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            uint8_t *bh = blk + 32 * h;
            const uint8_t *rh = r + 32 * h;
            __m256i c0, c1;
            __m256i v0 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 0)), basev);
            c0 = v0;
            __m256i v1 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 64)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v1, 2), _mm256_set1_epi8((char)0xFC)));
            __m256i v2 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 128)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v2, 4), _mm256_set1_epi8((char)0xF0)));
            __m256i v3 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 192)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v3, 6), _mm256_set1_epi8((char)0xC0)));
            __m256i v4 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 256)), basev);
            c1 = v4;
            __m256i v5 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 320)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v5, 2), _mm256_set1_epi8((char)0xFC)));
            __m256i v6 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 384)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v6, 4), _mm256_set1_epi8((char)0xF0)));
            __m256i v7 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 448)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v7, 6), _mm256_set1_epi8((char)0xC0)));
            _mm256_storeu_si256((__m256i *)(bh + 0), c0);
            _mm256_storeu_si256((__m256i *)(bh + 64), c1);
        }
    }
}

static void vert512_pack_ymm_d3(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m256i basev = _mm256_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 3;
        const uint8_t *r = ranks + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            uint8_t *bh = blk + 32 * h;
            const uint8_t *rh = r + 32 * h;
            __m256i c0, c1, c2;
            __m256i v0 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 0)), basev);
            c0 = v0;
            __m256i v1 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 64)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v1, 3), _mm256_set1_epi8((char)0xF8)));
            __m256i v2 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 128)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v2, 6), _mm256_set1_epi8((char)0xC0)));
            c1 = _mm256_and_si256(_mm256_srli_epi16(v2, 2), _mm256_set1_epi8((char)0x3F));
            __m256i v3 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 192)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v3, 1), _mm256_set1_epi8((char)0xFE)));
            __m256i v4 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 256)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v4, 4), _mm256_set1_epi8((char)0xF0)));
            __m256i v5 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 320)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v5, 7), _mm256_set1_epi8((char)0x80)));
            c2 = _mm256_and_si256(_mm256_srli_epi16(v5, 1), _mm256_set1_epi8((char)0x7F));
            __m256i v6 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 384)), basev);
            c2 = _mm256_or_si256(c2, _mm256_and_si256(_mm256_slli_epi16(v6, 2), _mm256_set1_epi8((char)0xFC)));
            __m256i v7 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 448)), basev);
            c2 = _mm256_or_si256(c2, _mm256_and_si256(_mm256_slli_epi16(v7, 5), _mm256_set1_epi8((char)0xE0)));
            _mm256_storeu_si256((__m256i *)(bh + 0), c0);
            _mm256_storeu_si256((__m256i *)(bh + 64), c1);
            _mm256_storeu_si256((__m256i *)(bh + 128), c2);
        }
    }
}

static void vert512_pack_ymm_d4(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m256i basev = _mm256_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 4;
        const uint8_t *r = ranks + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            uint8_t *bh = blk + 32 * h;
            const uint8_t *rh = r + 32 * h;
            __m256i c0, c1, c2, c3;
            __m256i v0 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 0)), basev);
            c0 = v0;
            __m256i v1 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 64)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v1, 4), _mm256_set1_epi8((char)0xF0)));
            __m256i v2 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 128)), basev);
            c1 = v2;
            __m256i v3 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 192)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v3, 4), _mm256_set1_epi8((char)0xF0)));
            __m256i v4 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 256)), basev);
            c2 = v4;
            __m256i v5 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 320)), basev);
            c2 = _mm256_or_si256(c2, _mm256_and_si256(_mm256_slli_epi16(v5, 4), _mm256_set1_epi8((char)0xF0)));
            __m256i v6 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 384)), basev);
            c3 = v6;
            __m256i v7 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 448)), basev);
            c3 = _mm256_or_si256(c3, _mm256_and_si256(_mm256_slli_epi16(v7, 4), _mm256_set1_epi8((char)0xF0)));
            _mm256_storeu_si256((__m256i *)(bh + 0), c0);
            _mm256_storeu_si256((__m256i *)(bh + 64), c1);
            _mm256_storeu_si256((__m256i *)(bh + 128), c2);
            _mm256_storeu_si256((__m256i *)(bh + 192), c3);
        }
    }
}

static void vert512_pack_ymm_d5(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m256i basev = _mm256_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 5;
        const uint8_t *r = ranks + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            uint8_t *bh = blk + 32 * h;
            const uint8_t *rh = r + 32 * h;
            __m256i c0, c1, c2, c3, c4;
            __m256i v0 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 0)), basev);
            c0 = v0;
            __m256i v1 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 64)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v1, 5), _mm256_set1_epi8((char)0xE0)));
            c1 = _mm256_and_si256(_mm256_srli_epi16(v1, 3), _mm256_set1_epi8((char)0x1F));
            __m256i v2 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 128)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v2, 2), _mm256_set1_epi8((char)0xFC)));
            __m256i v3 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 192)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v3, 7), _mm256_set1_epi8((char)0x80)));
            c2 = _mm256_and_si256(_mm256_srli_epi16(v3, 1), _mm256_set1_epi8((char)0x7F));
            __m256i v4 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 256)), basev);
            c2 = _mm256_or_si256(c2, _mm256_and_si256(_mm256_slli_epi16(v4, 4), _mm256_set1_epi8((char)0xF0)));
            c3 = _mm256_and_si256(_mm256_srli_epi16(v4, 4), _mm256_set1_epi8((char)0x0F));
            __m256i v5 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 320)), basev);
            c3 = _mm256_or_si256(c3, _mm256_and_si256(_mm256_slli_epi16(v5, 1), _mm256_set1_epi8((char)0xFE)));
            __m256i v6 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 384)), basev);
            c3 = _mm256_or_si256(c3, _mm256_and_si256(_mm256_slli_epi16(v6, 6), _mm256_set1_epi8((char)0xC0)));
            c4 = _mm256_and_si256(_mm256_srli_epi16(v6, 2), _mm256_set1_epi8((char)0x3F));
            __m256i v7 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 448)), basev);
            c4 = _mm256_or_si256(c4, _mm256_and_si256(_mm256_slli_epi16(v7, 3), _mm256_set1_epi8((char)0xF8)));
            _mm256_storeu_si256((__m256i *)(bh + 0), c0);
            _mm256_storeu_si256((__m256i *)(bh + 64), c1);
            _mm256_storeu_si256((__m256i *)(bh + 128), c2);
            _mm256_storeu_si256((__m256i *)(bh + 192), c3);
            _mm256_storeu_si256((__m256i *)(bh + 256), c4);
        }
    }
}

static void vert512_pack_ymm_d6(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m256i basev = _mm256_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 6;
        const uint8_t *r = ranks + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            uint8_t *bh = blk + 32 * h;
            const uint8_t *rh = r + 32 * h;
            __m256i c0, c1, c2, c3, c4, c5;
            __m256i v0 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 0)), basev);
            c0 = v0;
            __m256i v1 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 64)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v1, 6), _mm256_set1_epi8((char)0xC0)));
            c1 = _mm256_and_si256(_mm256_srli_epi16(v1, 2), _mm256_set1_epi8((char)0x3F));
            __m256i v2 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 128)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v2, 4), _mm256_set1_epi8((char)0xF0)));
            c2 = _mm256_and_si256(_mm256_srli_epi16(v2, 4), _mm256_set1_epi8((char)0x0F));
            __m256i v3 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 192)), basev);
            c2 = _mm256_or_si256(c2, _mm256_and_si256(_mm256_slli_epi16(v3, 2), _mm256_set1_epi8((char)0xFC)));
            __m256i v4 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 256)), basev);
            c3 = v4;
            __m256i v5 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 320)), basev);
            c3 = _mm256_or_si256(c3, _mm256_and_si256(_mm256_slli_epi16(v5, 6), _mm256_set1_epi8((char)0xC0)));
            c4 = _mm256_and_si256(_mm256_srli_epi16(v5, 2), _mm256_set1_epi8((char)0x3F));
            __m256i v6 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 384)), basev);
            c4 = _mm256_or_si256(c4, _mm256_and_si256(_mm256_slli_epi16(v6, 4), _mm256_set1_epi8((char)0xF0)));
            c5 = _mm256_and_si256(_mm256_srli_epi16(v6, 4), _mm256_set1_epi8((char)0x0F));
            __m256i v7 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 448)), basev);
            c5 = _mm256_or_si256(c5, _mm256_and_si256(_mm256_slli_epi16(v7, 2), _mm256_set1_epi8((char)0xFC)));
            _mm256_storeu_si256((__m256i *)(bh + 0), c0);
            _mm256_storeu_si256((__m256i *)(bh + 64), c1);
            _mm256_storeu_si256((__m256i *)(bh + 128), c2);
            _mm256_storeu_si256((__m256i *)(bh + 192), c3);
            _mm256_storeu_si256((__m256i *)(bh + 256), c4);
            _mm256_storeu_si256((__m256i *)(bh + 320), c5);
        }
    }
}

static void vert512_pack_ymm_d7(uint8_t *out, const uint8_t *ranks,
                                  int n_v, uint8_t base)
{
    const __m256i basev = _mm256_set1_epi8((char)base);
    for (int b = 0; b < n_v >> 9; b++) {
        uint8_t *blk = out + (size_t)b * 64 * 7;
        const uint8_t *r = ranks + ((size_t)b << 9);
        for (int h = 0; h < 2; h++) {
            uint8_t *bh = blk + 32 * h;
            const uint8_t *rh = r + 32 * h;
            __m256i c0, c1, c2, c3, c4, c5, c6;
            __m256i v0 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 0)), basev);
            c0 = v0;
            __m256i v1 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 64)), basev);
            c0 = _mm256_or_si256(c0, _mm256_and_si256(_mm256_slli_epi16(v1, 7), _mm256_set1_epi8((char)0x80)));
            c1 = _mm256_and_si256(_mm256_srli_epi16(v1, 1), _mm256_set1_epi8((char)0x7F));
            __m256i v2 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 128)), basev);
            c1 = _mm256_or_si256(c1, _mm256_and_si256(_mm256_slli_epi16(v2, 6), _mm256_set1_epi8((char)0xC0)));
            c2 = _mm256_and_si256(_mm256_srli_epi16(v2, 2), _mm256_set1_epi8((char)0x3F));
            __m256i v3 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 192)), basev);
            c2 = _mm256_or_si256(c2, _mm256_and_si256(_mm256_slli_epi16(v3, 5), _mm256_set1_epi8((char)0xE0)));
            c3 = _mm256_and_si256(_mm256_srli_epi16(v3, 3), _mm256_set1_epi8((char)0x1F));
            __m256i v4 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 256)), basev);
            c3 = _mm256_or_si256(c3, _mm256_and_si256(_mm256_slli_epi16(v4, 4), _mm256_set1_epi8((char)0xF0)));
            c4 = _mm256_and_si256(_mm256_srli_epi16(v4, 4), _mm256_set1_epi8((char)0x0F));
            __m256i v5 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 320)), basev);
            c4 = _mm256_or_si256(c4, _mm256_and_si256(_mm256_slli_epi16(v5, 3), _mm256_set1_epi8((char)0xF8)));
            c5 = _mm256_and_si256(_mm256_srli_epi16(v5, 5), _mm256_set1_epi8((char)0x07));
            __m256i v6 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 384)), basev);
            c5 = _mm256_or_si256(c5, _mm256_and_si256(_mm256_slli_epi16(v6, 2), _mm256_set1_epi8((char)0xFC)));
            c6 = _mm256_and_si256(_mm256_srli_epi16(v6, 6), _mm256_set1_epi8((char)0x03));
            __m256i v7 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(rh + 448)), basev);
            c6 = _mm256_or_si256(c6, _mm256_and_si256(_mm256_slli_epi16(v7, 1), _mm256_set1_epi8((char)0xFE)));
            _mm256_storeu_si256((__m256i *)(bh + 0), c0);
            _mm256_storeu_si256((__m256i *)(bh + 64), c1);
            _mm256_storeu_si256((__m256i *)(bh + 128), c2);
            _mm256_storeu_si256((__m256i *)(bh + 192), c3);
            _mm256_storeu_si256((__m256i *)(bh + 256), c4);
            _mm256_storeu_si256((__m256i *)(bh + 320), c5);
            _mm256_storeu_si256((__m256i *)(bh + 384), c6);
        }
    }
}

static inline void vert512_pack_ymm(uint8_t *out, const uint8_t *ranks, int n_v, int D, uint8_t base)
{
    switch (D) {
    case 2: vert512_pack_ymm_d2(out, ranks, n_v, base); break;
    case 3: vert512_pack_ymm_d3(out, ranks, n_v, base); break;
    case 4: vert512_pack_ymm_d4(out, ranks, n_v, base); break;
    case 5: vert512_pack_ymm_d5(out, ranks, n_v, base); break;
    case 6: vert512_pack_ymm_d6(out, ranks, n_v, base); break;
#if defined(PIVCO_X86_INTEL)
    default: vert512_pack_ymm_d7(out, ranks, n_v, base); break;
#else
    /* Zen 3 splits 256-bit ops in two; the 16-lane quarter walk wins
     * there (1.12 vs 0.89 on c6a) while Intel AVX2 prefers ymm (1.54
     * vs 1.22 on c5). */
    default: vert512_pack_x86_d7(out, ranks, n_v, base); break;
#endif
    }
}

#endif /* __AVX2__ */


/* Best-available 512-block kernels. */
static inline void vert512_merge_x86_best(uint8_t *out, int n_v, const uint8_t *bm,
                                          int D, const uint8_t *c2s)
{
#if defined(__AVX512BW__) && defined(__AVX512VBMI__) && defined(__AVX512VL__)
    vert512_merge_zmm(out, n_v, bm, D, c2s);
#elif defined(__AVX2__)
    vert512_merge_ymm(out, n_v, bm, D, c2s);
#else
    vert512_merge_x86v(out, n_v, bm, D, c2s);
#endif
}

static inline void vert512_pack_x86_best(uint8_t *out, const uint8_t *ranks,
                                         int n_v, int D, uint8_t base)
{
#if defined(__AVX512BW__) && defined(__AVX512VBMI__) && defined(__AVX512VL__)
    vert512_pack_zmm(out, ranks, n_v, D, base);
#elif defined(__AVX2__)
    vert512_pack_ymm(out, ranks, n_v, D, base);
#else
    vert512_pack_x86v(out, ranks, n_v, D, base);
#endif
}

#endif /* PIVCO_HUFFMAN_X86_VERTICAL_H */
