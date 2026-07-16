/* bench/prim_variants/prims-pack.h — flat-subtree pack variant graveyard.
 *
 * Logical primitive: enc_pack_dN (ST_PACK).  Pack n D-bit codes (LSB-first)
 * from codes_la[] into a contiguous bitstream.  See prims.h for the contract +
 * naming (PV_ = constants/macros, pv_ = plumbing, prim_ = kernels).
 *
 * Per-D variants (registered with PV_VARIANT_D so each row only runs for the D
 * the bench sweeps):
 *   multishift     — vpmultishiftqb pack, 64 codes/iter (extras/bench/
 *                    bench_pack_v2.c pack_v2_d{2..7}).
 *   asof-cd119a6   — BMI2 _pext_u64 pack, 8 codes/iter (git show
 *                    cd119a6:src/pivco_huffman_pack_bmi2.h pack_dN_bmi2).
 *   asof-2f80076   — sllv + reduce_add pack, 8 codes/iter (the PACK macro at
 *                    cd119a6~1:src/pivco_huffman_primitives_avx512.h).
 *   asof-f9974f5   — pre-PR#22 NEON per-D packs (git show
 *                    f9974f5:src/pivco_huffman_primitives_neon.h pack_d{2,3,4}
 *                    + f9974f5:src/pivco_huffman_neon_pack.h pack_d{5,6,7}).
 *
 * Each per-D SIMD kernel returns the number of codes it packed (a multiple of
 * its stride); a shared scalar tail (pv_pack_scalar_tail) finishes the
 * n % stride residual.  Production right_shift = 16 - depth - D, matching
 * scalar_pack in bench_prim.c (the correctness reference).
 *
 * Gated `#if defined(__AVX512VBMI2__)` (multishift / sllv), `#if
 * defined(__BMI2__)` (pext) — all three ISAs are present on the AVX-512
 * host — and `#if defined(USE_NEON_KERNELS)` (asof-f9974f5).
 */
#ifndef PIVCO_PRIM_VARIANTS_PACK_H
#define PIVCO_PRIM_VARIANTS_PACK_H

#if defined(__AVX512VBMI2__) || defined(__BMI2__)
#include <immintrin.h>
#include <string.h>

/* Shared scalar tail: pack codes [start, n) starting at bit start*D.
 * Matches scalar_pack's LSB-first byte layout (the bench reference).
 * NB: these u16 codes_la kernels register under ST_U16_PACK — the
 * rank-based ST_PACK slot fills u8 ranks, not la_work (they sat on
 * ST_PACK from before the slot split and failed its check). */
static inline void pv_pack_scalar_tail(uint8_t *out, const uint16_t *codes_la,
                                       int start, int n, int D, int right_shift) {
    uint32_t mask = (1u << D) - 1;
    for (int i = start; i < n; i++) {
        uint32_t code = ((uint32_t)codes_la[i] >> right_shift) & mask;
        int bit_pos = i * D, byte_idx = bit_pos >> 3, bit_off = bit_pos & 7;
        out[byte_idx] |= (uint8_t)(code << bit_off);
        if (bit_off + D > 8)  out[byte_idx + 1] |= (uint8_t)(code >> (8 - bit_off));
        if (bit_off + D > 16) out[byte_idx + 2] |= (uint8_t)(code >> (16 - bit_off));
    }
}

#endif /* __AVX512VBMI2__ || __BMI2__ */

/* ============================================================================
 * multishift — vpmultishiftqb pack, 64 codes/iter
 *   From extras/bench/bench_pack_v2.c (pack_v2_d{2..7}).  D=2/4 use a plain
 *   vpermb gather + shift + OR (codes don't cross byte boundaries); D=3/5/6/7
 *   use _mm512_multishift_epi64_epi8 to place each code at its bit offset
 *   within a 64-bit lane, then vpermb to compact the valid bytes.
 * ========================================================================== */
#if defined(__AVX512VBMI2__) && defined(__AVX512VBMI__)

/* Load 64 u16 codes, right-shift, narrow to 1 byte/code, mask to low D bits. */
static inline __m512i pv_load_codes_byte(const uint16_t *codes_la,
                                         int right_shift, uint8_t code_mask) {
    __m512i lo16 = _mm512_loadu_si512((const __m512i *)(codes_la));
    __m512i hi16 = _mm512_loadu_si512((const __m512i *)(codes_la + 32));
    __m256i lo_b = _mm512_cvtepi16_epi8(_mm512_srli_epi16(lo16, right_shift));
    __m256i hi_b = _mm512_cvtepi16_epi8(_mm512_srli_epi16(hi16, right_shift));
    __m512i cb = _mm512_inserti64x4(_mm512_castsi256_si512(lo_b), hi_b, 1);
    return _mm512_and_si512(cb, _mm512_set1_epi8(code_mask));
}

static const uint8_t pv_compact_d3_tab[64] __attribute__((aligned(64))) = {
     0, 1, 2,  8, 9,10, 16,17,18, 24,25,26, 32,33,34, 40,41,42, 48,49,50, 56,57,58,
     0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0 };
static const uint8_t pv_compact_d5_tab[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4,  8, 9,10,11,12, 16,17,18,19,20, 24,25,26,27,28,
    32,33,34,35,36, 40,41,42,43,44, 48,49,50,51,52, 56,57,58,59,60,
     0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0 };
static const uint8_t pv_compact_d6_tab[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4, 5,  8, 9,10,11,12,13, 16,17,18,19,20,21, 24,25,26,27,28,29,
    32,33,34,35,36,37, 40,41,42,43,44,45, 48,49,50,51,52,53, 56,57,58,59,60,61,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0 };
static const uint8_t pv_compact_d7_tab[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4, 5, 6,  8, 9,10,11,12,13,14, 16,17,18,19,20,21,22, 24,25,26,27,28,29,30,
    32,33,34,35,36,37,38, 40,41,42,43,44,45,46, 48,49,50,51,52,53,54, 56,57,58,59,60,61,62,  0,0,0,0,0,0,0,0 };

static int pv_pack_ms_d2(uint8_t *out, const uint16_t *codes_la, int n, int rs) {
    const __m512i shuf0 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 60,56,52,48,44,40,36,32, 28,24,20,16,12,8,4,0);
    const __m512i shuf1 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 61,57,53,49,45,41,37,33, 29,25,21,17,13,9,5,1);
    const __m512i shuf2 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 62,58,54,50,46,42,38,34, 30,26,22,18,14,10,6,2);
    const __m512i shuf3 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 63,59,55,51,47,43,39,35, 31,27,23,19,15,11,7,3);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pv_load_codes_byte(codes_la + i, rs, 0x03);
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i g2 = _mm512_permutexvar_epi8(shuf2, cb);
        __m512i g3 = _mm512_permutexvar_epi8(shuf3, cb);
        __m512i packed = _mm512_or_si512(
            _mm512_or_si512(g0, _mm512_slli_epi32(g1, 2)),
            _mm512_or_si512(_mm512_slli_epi32(g2, 4), _mm512_slli_epi32(g3, 6)));
        _mm512_storeu_si512(out + ((i * 2) >> 3), packed);
    }
    return i;
}
static int pv_pack_ms_d4(uint8_t *out, const uint16_t *codes_la, int n, int rs) {
    const __m512i shuf0 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        62,60,58,56,54,52,50,48, 46,44,42,40,38,36,34,32,
        30,28,26,24,22,20,18,16, 14,12,10, 8, 6, 4, 2, 0);
    const __m512i shuf1 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        63,61,59,57,55,53,51,49, 47,45,43,41,39,37,35,33,
        31,29,27,25,23,21,19,17, 15,13,11, 9, 7, 5, 3, 1);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pv_load_codes_byte(codes_la + i, rs, 0x0F);
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i packed = _mm512_or_si512(g0, _mm512_slli_epi32(g1, 4));
        _mm512_storeu_si512(out + ((i * 4) >> 3), packed);
    }
    return i;
}
static int pv_pack_ms_d3(uint8_t *out, const uint16_t *codes_la, int n, int rs) {
    const __m512i mA = _mm512_set1_epi64((int64_t)0x0000000700000007ULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x0000070000000700ULL);
    const __m512i mC = _mm512_set1_epi64((int64_t)0x0007000000070000ULL);
    const __m512i mD = _mm512_set1_epi64((int64_t)0x0700000007000000ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x0000000000081C00ULL);
    const __m512i cB = _mm512_set1_epi64((int64_t)0x0000000000292105ULL);
    const __m512i cC = _mm512_set1_epi64((int64_t)0x00000000002E120AULL);
    const __m512i cD = _mm512_set1_epi64((int64_t)0x0000000000331700ULL);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pv_load_codes_byte(codes_la + i, rs, 0x07);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i c = _mm512_multishift_epi64_epi8(cC, _mm512_and_si512(cb, mC));
        __m512i d = _mm512_multishift_epi64_epi8(cD, _mm512_and_si512(cb, mD));
        __m512i packed = _mm512_or_si512(_mm512_or_si512(a, b), _mm512_or_si512(c, d));
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pv_compact_d3_tab), packed);
        _mm512_storeu_si512(out + ((i * 3) >> 3), compact);
    }
    return i;
}
static int pv_pack_ms_d5(uint8_t *out, const uint16_t *codes_la, int n, int rs) {
    const __m512i mA = _mm512_set1_epi64((int64_t)0x001F00001F00001FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x1F00001F00001F00ULL);
    const __m512i mC = _mm512_set1_epi64((int64_t)0x00001F00001F0000ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x000000322A191100ULL);
    const __m512i cB = _mm512_set1_epi64((int64_t)0x00000035241C0B03ULL);
    const __m512i cC = _mm512_set1_epi64((int64_t)0x0000000027000E00ULL);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pv_load_codes_byte(codes_la + i, rs, 0x1F);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i c = _mm512_multishift_epi64_epi8(cC, _mm512_and_si512(cb, mC));
        __m512i packed = _mm512_or_si512(_mm512_or_si512(a, b), c);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pv_compact_d5_tab), packed);
        _mm512_storeu_si512(out + ((i * 5) >> 3), compact);
    }
    return i;
}
static int pv_pack_ms_d6(uint8_t *out, const uint16_t *codes_la, int n, int rs) {
    const __m512i mA = _mm512_set1_epi64((int64_t)0x003F003F003F003FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x3F003F003F003F00ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x0000342C20140C00ULL);
    const __m512i cB = _mm512_set1_epi64((int64_t)0x0000362A22160A02ULL);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pv_load_codes_byte(codes_la + i, rs, 0x3F);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i packed = _mm512_or_si512(a, b);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pv_compact_d6_tab), packed);
        _mm512_storeu_si512(out + ((i * 6) >> 3), compact);
    }
    return i;
}
static int pv_pack_ms_d7(uint8_t *out, const uint16_t *codes_la, int n, int rs) {
    const __m512i mA = _mm512_set1_epi64((int64_t)0x007F007F007F007FULL);
    const __m512i mB = _mm512_set1_epi64((int64_t)0x7F007F007F007F00ULL);
    const __m512i cA = _mm512_set1_epi64((int64_t)0x00362E241C120A00ULL);
    const __m512i cB = _mm512_set1_epi64((int64_t)0x00372D251B130901ULL);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pv_load_codes_byte(codes_la + i, rs, 0x7F);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i packed = _mm512_or_si512(a, b);
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pv_compact_d7_tab), packed);
        _mm512_storeu_si512(out + ((i * 7) >> 3), compact);
    }
    return i;
}

static void prim_pack_multishift(const ctx_t *c) {
    int D = c->D, rs = 16 - c->depth - D, i = 0;
    int total = (c->n * D + 7) >> 3;
    memset(c->pack_out, 0, total);
    switch (D) {
    case 2: i = pv_pack_ms_d2(c->pack_out, c->la_work, c->n, rs); break;
    case 3: i = pv_pack_ms_d3(c->pack_out, c->la_work, c->n, rs); break;
    case 4: i = pv_pack_ms_d4(c->pack_out, c->la_work, c->n, rs); break;
    case 5: i = pv_pack_ms_d5(c->pack_out, c->la_work, c->n, rs); break;
    case 6: i = pv_pack_ms_d6(c->pack_out, c->la_work, c->n, rs); break;
    case 7: i = pv_pack_ms_d7(c->pack_out, c->la_work, c->n, rs); break;
    default: break;
    }
    pv_pack_scalar_tail(c->pack_out, c->la_work, i, c->n, D, rs);
}

#endif /* __AVX512VBMI2__ && __AVX512VBMI__ */

/* ============================================================================
 * asof-cd119a6 — BMI2 _pext_u64 pack, 8 codes/iter
 *   git show cd119a6:src/pivco_huffman_pack_bmi2.h (pack_dN_bmi2).  One pext
 *   packs 4 codes (a 4×u16 window); two cover a group of 8.  D bytes per 8.
 * ========================================================================== */
#if defined(__BMI2__)
static inline int pv_pack_bmi2_dN(uint8_t *out, const uint16_t *codes_la,
                                  int n, int D, int right_shift) {
    uint64_t field = (((uint64_t)1 << D) - 1) << right_shift;
    uint64_t mask  = field | (field << 16) | (field << 32) | (field << 48);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w0, w1;
        memcpy(&w0, codes_la + i,     8);
        memcpy(&w1, codes_la + i + 4, 8);
        uint64_t packed = _pext_u64(w0, mask) | (_pext_u64(w1, mask) << (4 * D));
        memcpy(out + ((i * D) >> 3), &packed, (size_t)D);
    }
    return i;
}
static void prim_pack_bmi2(const ctx_t *c) {
    int D = c->D, rs = 16 - c->depth - D;
    int total = (c->n * D + 7) >> 3;
    memset(c->pack_out, 0, total);
    int i = pv_pack_bmi2_dN(c->pack_out, c->la_work, c->n, D, rs);
    pv_pack_scalar_tail(c->pack_out, c->la_work, i, c->n, D, rs);
}
#endif /* __BMI2__ */

/* ============================================================================
 * asof-2f80076 — sllv + reduce_add pack, 8 codes/iter
 *   The PACK_DN_AVX512_UNIFIED macro at
 *   cd119a6~1:src/pivco_huffman_primitives_avx512.h.  Widen 8 u16 -> u64,
 *   srli + and to isolate the code, sllv by {0,D,2D,..7D}, reduce_add across
 *   8 lanes -> one 8D-bit qword; store the low ceil(8D/8)=D bytes.
 * ========================================================================== */
#if defined(__AVX512F__)
#define PV_PACK_SLLV_DN(NAME, D_VAL, BITS_OUT)                                  \
static int NAME(uint8_t *out, const uint16_t *codes_la, int n, int rs) {        \
    static const int64_t shifts[8] = {                                         \
        0, D_VAL, 2*D_VAL, 3*D_VAL, 4*D_VAL, 5*D_VAL, 6*D_VAL, 7*D_VAL };       \
    __m512i shift_vec = _mm512_loadu_si512((const __m512i *)shifts);            \
    __m512i mask_vec  = _mm512_set1_epi64((1ULL << D_VAL) - 1);                 \
    int i = 0;                                                                  \
    for (; i + 8 <= n; i += 8) {                                               \
        __m128i v16 = _mm_loadu_si128((const __m128i *)(codes_la + i));         \
        __m512i v64 = _mm512_cvtepu16_epi64(v16);                               \
        v64 = _mm512_srli_epi64(v64, rs);                                       \
        v64 = _mm512_and_si512(v64, mask_vec);                                  \
        v64 = _mm512_sllv_epi64(v64, shift_vec);                                \
        uint64_t packed = _mm512_reduce_add_epi64(v64);                         \
        int bi = i * D_VAL / 8;                                                 \
        memcpy(out + bi, &packed, (BITS_OUT + 7) / 8);                          \
    }                                                                           \
    return i;                                                                   \
}
PV_PACK_SLLV_DN(pv_pack_sllv_d2, 2, 16)
PV_PACK_SLLV_DN(pv_pack_sllv_d3, 3, 24)
PV_PACK_SLLV_DN(pv_pack_sllv_d4, 4, 32)
PV_PACK_SLLV_DN(pv_pack_sllv_d5, 5, 40)
PV_PACK_SLLV_DN(pv_pack_sllv_d6, 6, 48)
PV_PACK_SLLV_DN(pv_pack_sllv_d7, 7, 56)
#undef PV_PACK_SLLV_DN

static void prim_pack_sllv(const ctx_t *c) {
    int D = c->D, rs = 16 - c->depth - D, i = 0;
    int total = (c->n * D + 7) >> 3;
    memset(c->pack_out, 0, total);
    switch (D) {
    case 2: i = pv_pack_sllv_d2(c->pack_out, c->la_work, c->n, rs); break;
    case 3: i = pv_pack_sllv_d3(c->pack_out, c->la_work, c->n, rs); break;
    case 4: i = pv_pack_sllv_d4(c->pack_out, c->la_work, c->n, rs); break;
    case 5: i = pv_pack_sllv_d5(c->pack_out, c->la_work, c->n, rs); break;
    case 6: i = pv_pack_sllv_d6(c->pack_out, c->la_work, c->n, rs); break;
    case 7: i = pv_pack_sllv_d7(c->pack_out, c->la_work, c->n, rs); break;
    default: break;
    }
    pv_pack_scalar_tail(c->pack_out, c->la_work, i, c->n, D, rs);
}
#endif /* __AVX512F__ */

/* ============================================================================
 * asof-f9974f5 — pre-PR#22 NEON per-D packs (prior production)
 *   The rank-based (u8, out/ranks/n/base) per-D kernels production shipped
 *   before PR #22's variable-shift pyramid rework: non-unrolled paired-add
 *   d2/d4 + distributed-base-free vsubq, u32-horadd d3, and the ryg
 *   multiply-as-shift d5/d6/d7 with UNBOUNDED 16-byte stores (16-2D trailing
 *   junk bytes per store; the bench pack_out's +16 slack absorbs the last
 *   iter's junk, as PIVCO_MAX_ENCODED_SIZE did in production).  Frozen
 *   verbatim; d7's compact shuffle is byte-identical to today's so it reuses
 *   the production pivco_pack_compact_d7_neon, d5/d6 keep the old (byte-0
 *   based) tables below.  The adapter mirrors the production pack_dN_neon
 *   dispatcher (unchanged by #22): zero the last byte, per-D SIMD kernel,
 *   scalar bit tail for the residual codes.
 * ========================================================================== */
#if defined(USE_NEON_KERNELS)

/* Load 16 ranks, subtract base (1 rank/byte) — the byte-laid intermediate the
 * multiply-as-shift pack expects, with no u16 narrow.  The local code is already
 * in [0,2^D) (rank - flat_base_rank over a depth-D flat subtree), so no mask to D
 * bits is needed. */
static inline uint8x16_t
pv_pack_load_byte_asof_f9974f5_neon(const uint8_t *ranks, uint8_t base)
{
    return vsubq_u8(vld1q_u8(ranks), vdupq_n_u8(base));
}

static const uint8_t pv_pack_compact_d5_asof_f9974f5_neon[16] = {
    0, 1, 2, 3, 4,   8, 9, 10, 11, 12,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
static const uint8_t pv_pack_compact_d6_asof_f9974f5_neon[16] = {
    0, 1, 2, 3, 4, 5,   8, 9, 10, 11, 12, 13,
    0xff, 0xff, 0xff, 0xff
};

static inline int pv_pack_asof_f9974f5_d2(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d2[16] = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    uint8x16_t vb = vdupq_n_u8(base);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t b = vsubq_u8(vld1q_u8(ranks + i), vb);  /* local code in [0,2^D); no mask needed */
        b = vshlq_u8(b, vld1q_s8(shifts_d2));
        uint8x16_t s1 = vpaddq_u8(b, b);
        uint8x16_t s2 = vpaddq_u8(s1, s1);
        uint32_t packed4 = vgetq_lane_u32(vreinterpretq_u32_u8(s2), 0);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

/* D=3: 8 ranks -> 24 bits, u32 horizontal accumulator. */
static inline int pv_pack_asof_f9974f5_d3(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int32_t shifts_lo[4] = { 0, 3, 6, 9 };
    static const int32_t shifts_hi[4] = { 12, 15, 18, 21 };
    uint8x8_t vb = vdup_n_u8(base);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t b8 = vsub_u8(vld1_u8(ranks + i), vb);    /* local code in [0,2^D); no mask needed */
        uint16x8_t v = vmovl_u8(b8);
        uint32x4_t lo = vshlq_u32(vmovl_u16(vget_low_u16(v)),  vld1q_s32(shifts_lo));
        uint32x4_t hi = vshlq_u32(vmovl_u16(vget_high_u16(v)), vld1q_s32(shifts_hi));
        uint32_t packed = vaddvq_u32(vaddq_u32(lo, hi));
        int bi = i * 3 / 8;
        out[bi]     = (uint8_t)(packed       & 0xff);
        out[bi + 1] = (uint8_t)((packed >> 8 ) & 0xff);
        out[bi + 2] = (uint8_t)((packed >> 16) & 0xff);
    }
    return i;
}

static inline int pv_pack_asof_f9974f5_d4(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d4[16] = { 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4 };
    uint8x16_t vb = vdupq_n_u8(base);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t b = vsubq_u8(vld1q_u8(ranks + i), vb);  /* local code in [0,2^D); no mask needed */
        b = vshlq_u8(b, vld1q_s8(shifts_d4));
        uint8x16_t paired = vpaddq_u8(b, b);
        vst1_u8(out + (i * 4 / 8), vget_low_u8(paired));
    }
    return i;
}

/* D as compile-time constant so vshrq_n_u64 / mask constants fold. */
#define PV_PACK_NEON_DN_ASOF_F9974F5(NAME, D_VAL, COMPACT_TAB)                  \
static inline int NAME(uint8_t *out, const uint8_t *ranks,                     \
                       int n, uint8_t base)                                     \
{                                                                                \
    const uint8x16_t c0 = vreinterpretq_u8_u16(                                  \
        vdupq_n_u16((uint16_t)(((1u << (D_VAL)) << 8) | 1u)));                   \
    const uint16x8_t c1 = vreinterpretq_u16_u32(                                 \
        vdupq_n_u32((uint32_t)(((1u << (2*(D_VAL))) << 16) | 1u)));              \
    const uint64x2_t c3 = vdupq_n_u64(((uint64_t)1 << (4*(D_VAL))) - 1);         \
    const uint8x16_t compact = vld1q_u8(COMPACT_TAB);                            \
    int i = 0;                                                                   \
    for (; i + 16 <= n; i += 16) {                                               \
        uint8x16_t cb = pv_pack_load_byte_asof_f9974f5_neon(ranks + i, base);    \
        /* Step 1: word[i] = cb[2i] + cb[2i+1] * 2^D  (8 u16 lanes)   */         \
        uint16x8_t prod_lo = vmull_u8(vget_low_u8(cb),  vget_low_u8(c0));        \
        uint16x8_t prod_hi = vmull_high_u8(cb, c0);                              \
        uint16x8_t w = vpaddq_u16(prod_lo, prod_hi);                             \
        /* Step 2: dword[i] = word[2i] + word[2i+1] * 2^(2D)  (4 u32 lanes) */   \
        uint32x4_t prod32_lo = vmull_u16(vget_low_u16(w),  vget_low_u16(c1));    \
        uint32x4_t prod32_hi = vmull_high_u16(w, c1);                            \
        uint32x4_t d  = vpaddq_u32(prod32_lo, prod32_hi);                        \
        /* Step 3: per-u64 lane, merge dword[2i+1] (right-shifted) with         \
         * dword[2i].  After srli by (32 - 4D): the high-32 dword sits at       \
         * bits [4D..4D+31].  Mask keeps low 4D bits of x, takes high 4D bits  \
         * from xs — together 8D bits per u64.                                  */ \
        uint64x2_t x  = vreinterpretq_u64_u32(d);                                \
        uint64x2_t xs = vshrq_n_u64(x, 32 - 4*(D_VAL));                          \
        uint64x2_t m  = vorrq_u64(vandq_u64(x, c3),                              \
                                   vbicq_u64(xs, c3));                           \
        /* Step 4: compact 2D consecutive bytes per 128-bit lane.   */           \
        uint8x16_t packed = vqtbl1q_u8(vreinterpretq_u8_u64(m), compact);        \
        vst1q_u8(out + ((i * (D_VAL)) >> 3), packed);                            \
    }                                                                            \
    return i;                                                                    \
}
PV_PACK_NEON_DN_ASOF_F9974F5(pv_pack_asof_f9974f5_d5, 5, pv_pack_compact_d5_asof_f9974f5_neon)
PV_PACK_NEON_DN_ASOF_F9974F5(pv_pack_asof_f9974f5_d6, 6, pv_pack_compact_d6_asof_f9974f5_neon)
PV_PACK_NEON_DN_ASOF_F9974F5(pv_pack_asof_f9974f5_d7, 7, pivco_pack_compact_d7_neon)
#undef PV_PACK_NEON_DN_ASOF_F9974F5

/* Adapter: same call shape as the production ST_PACK row (simd_pack ->
 * prim_enc_pack_dN(ranks, n, D, rank_base, pack_out)); the dispatcher body
 * mirrors pack_dN_neon (unchanged by #22) with the frozen kernels switched in. */
static void prim_pack_asof_f9974f5(const ctx_t *c) {
    uint8_t *out = c->pack_out; const uint8_t *ranks = c->ranks;
    int n = c->n, D = c->D; uint8_t base = c->rank_base;
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;

    int i = 0;
    switch (D) {
    case 2: i = pv_pack_asof_f9974f5_d2(out, ranks, n, base); break;
    case 3: i = pv_pack_asof_f9974f5_d3(out, ranks, n, base); break;
    case 4: i = pv_pack_asof_f9974f5_d4(out, ranks, n, base); break;
    case 5: i = pv_pack_asof_f9974f5_d5(out, ranks, n, base); break;
    case 6: i = pv_pack_asof_f9974f5_d6(out, ranks, n, base); break;
    case 7: i = pv_pack_asof_f9974f5_d7(out, ranks, n, base); break;
    default: break;
    }
    if (i >= n) return;

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
#endif /* USE_NEON_KERNELS */


/* ============================================================================
 * pack : pyramid-sse — 128-bit twin of the production AVX2 ryg/pyramid pack
 *   (pivco_huffman_avx2_pack.h).  16 codes per xmm iter: maddubs (2D-bit
 *   byte pairs) -> madd (4D-bit dword pairs) -> qword fuse (srlq +
 *   and/andn/or) -> pshufb compact -> one 16-byte store whose trailing
 *   junk (16 - 2D bytes) is overwritten by the next iter; the bench
 *   pack_out +16 slack absorbs the last iter's junk (as
 *   PIVCO_MAX_ENCODED_SIZE would in production).  Candidate production
 *   fallback for the non-AVX2 tier (c3), where d5/6/7 currently pack
 *   fully scalar and d3 used the mullo+hadd 8-codes/iter form.
 *   PROMOTED to production for d3/5/6/7 on the non-AVX2 tier (2026-07-16);
 *   the rows stay as the xmm-vs-ymm / xmm-vs-multishift comparison on
 *   AVX2/AVX-512 builds.  The replaced hadd d3 is asof-b8bf472 below.
 * ========================================================================== */
#if defined(__SSE4_1__)
#define PV_PACK_SSE_PYR(NAME, D_VAL, C0,C1,C2, C3,C4,C5, C6,C7,C8, C9,C10,C11, C12,C13)  \
static inline int NAME##_k(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)      \
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
PV_PACK_SSE_PYR(pv_pack_pyr_sse_d2, 2, 0,1,8,  9,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1)
PV_PACK_SSE_PYR(pv_pack_pyr_sse_d3, 3, 0,1,2,  8,9,10,  -1,-1,-1, -1,-1,-1, -1,-1)
PV_PACK_SSE_PYR(pv_pack_pyr_sse_d4, 4, 0,1,2,  3,8,9,   10,11,-1, -1,-1,-1, -1,-1)
PV_PACK_SSE_PYR(pv_pack_pyr_sse_d5, 5, 0,1,2,  3,4,8,   9,10,11,  12,-1,-1, -1,-1)
PV_PACK_SSE_PYR(pv_pack_pyr_sse_d6, 6, 0,1,2,  3,4,5,   8,9,10,   11,12,13, -1,-1)
PV_PACK_SSE_PYR(pv_pack_pyr_sse_d7, 7, 0,1,2,  3,4,5,   6,8,9,    10,11,12, 13,14)
#undef PV_PACK_SSE_PYR

/* ============================================================================
 * pack : asof-b8bf472 — the pre-pyramid SSE4.1 D=3 (prior production on the
 *   non-AVX2 tier).  8 ranks -> 24 bits via _mm_mullo_epi32 multiply-as-shift
 *   + two hadds; frozen verbatim from b8bf472:src/pivco_huffman_primitives_x86.h.
 * ========================================================================== */
static inline int pv_pack_d3_asof_b8bf472(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    const __m128i mlo = _mm_setr_epi32(1, 8, 64, 512);
    const __m128i mhi = _mm_setr_epi32(4096, 32768, 262144, 2097152);
    const __m128i vb = _mm_set1_epi8((char)base);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v8 = _mm_sub_epi8(_mm_loadl_epi64((const __m128i *)(ranks + i)), vb);
        __m128i v = _mm_cvtepu8_epi16(v8);                 /* 8 ranks -> u16; code in [0,2^D) */
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
static void prim_pack_d3_asof_b8bf472(const ctx_t *c) {
    uint8_t *out = c->pack_out; const uint8_t *ranks = c->ranks;
    int n = c->n, D = c->D; uint8_t base = c->rank_base;
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int i = D == 3 ? pv_pack_d3_asof_b8bf472(out, ranks, n, base) : 0;
    if (i >= n) return;
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

/* Adapter: same call shape as the production ST_PACK row; dispatcher body
 * mirrors pack_dN_x86 (zero last byte, per-D kernel, scalar bit tail). */
static void prim_pack_pyr_sse(const ctx_t *c) {
    uint8_t *out = c->pack_out; const uint8_t *ranks = c->ranks;
    int n = c->n, D = c->D; uint8_t base = c->rank_base;
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;

    int i = 0;
    switch (D) {
    case 2: i = pv_pack_pyr_sse_d2_k(out, ranks, n, base); break;
    case 3: i = pv_pack_pyr_sse_d3_k(out, ranks, n, base); break;
    case 4: i = pv_pack_pyr_sse_d4_k(out, ranks, n, base); break;
    case 5: i = pv_pack_pyr_sse_d5_k(out, ranks, n, base); break;
    case 6: i = pv_pack_pyr_sse_d6_k(out, ranks, n, base); break;
    case 7: i = pv_pack_pyr_sse_d7_k(out, ranks, n, base); break;
    default: break;
    }
    if (i >= n) return;

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
#endif /* __SSE4_1__ */

/* ============================================================================
 * Registry — pack family (no-op where the ISA is unavailable)
 * ========================================================================== */
static void pv_register_pack(void) {
    for (int d = 2; d <= 7; d++) {
        PV_VARIANT_D(ST_U16_PACK, "multishift", d, PV_ISA_AVX512,
                     "bench_pack_v2.c pack_v2_dN",
                     "vpmultishiftqb, 64 codes/iter", 0, PV_FN_VBMI2(prim_pack_multishift));
        PV_VARIANT_D(ST_U16_PACK, "asof-cd119a6", d, PV_ISA_AVX512,
                     "cd119a6 pack_dN_bmi2",
                     "BMI2 pext pack, 8 codes/iter", 0, PV_FN_BMI2(prim_pack_bmi2));
        PV_VARIANT_D(ST_U16_PACK, "asof-2f80076", d, PV_ISA_AVX512,
                     "cd119a6~1 PACK_DN_AVX512_UNIFIED",
                     "sllv + reduce_add pack, 8 codes/iter", 0, PV_FN_AVX512F(prim_pack_sllv));
        if (d == 3)
            PV_VARIANT_D(ST_PACK, "asof-b8bf472", d, PV_ISA_SSE4,
                         "b8bf472 (prior non-AVX2 production)",
                         "mullo multiply-as-shift + 2x hadd, 8 codes/iter; replaced by the sse pyramid", 0, PV_FN_SSE(prim_pack_d3_asof_b8bf472));
        PV_VARIANT_D(ST_PACK, "pyramid-sse", d, PV_ISA_SSE4,
                     "128-bit twin of the AVX2 ryg/pyramid pack",
                     "maddubs->madd->qword-fuse->pshufb, 16 codes/iter; candidate non-AVX2 production (c3 tier packs d5/6/7 scalar today)", 0, PV_FN_SSE(prim_pack_pyr_sse));
        PV_VARIANT_D(ST_PACK, "asof-f9974f5", d, PV_ISA_NEON,
                     "f9974f5 (prior production)",
                     "pre-PR#22 per-D packs: paired-add d2/d4, u32-horadd d3, ryg multiply-as-shift d5/d6/d7", 0, PV_FN_NEON(prim_pack_asof_f9974f5));
    }
}

#endif /* PIVCO_PRIM_VARIANTS_PACK_H */
