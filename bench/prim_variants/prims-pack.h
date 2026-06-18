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
 *
 * Each per-D SIMD kernel returns the number of codes it packed (a multiple of
 * its stride); a shared scalar tail (pv_pack_scalar_tail) finishes the
 * n % stride residual.  Production right_shift = 16 - depth - D, matching
 * scalar_pack in bench_prim.c (the correctness reference).
 *
 * Gated `#if defined(__AVX512VBMI2__)` (multishift / sllv) and `#if
 * defined(__BMI2__)` (pext); all three ISAs are present on the AVX-512 host.
 */
#ifndef PIVCO_PRIM_VARIANTS_PACK_H
#define PIVCO_PRIM_VARIANTS_PACK_H

#if defined(__AVX512VBMI2__) || defined(__BMI2__)
#include <immintrin.h>
#include <string.h>

/* Shared scalar tail: pack codes [start, n) starting at bit start*D.
 * Matches scalar_pack's LSB-first byte layout (the bench reference). */
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
 * Registry — pack family (no-op where the ISA is unavailable)
 * ========================================================================== */
static void pv_register_pack(void) {
    for (int d = 2; d <= 7; d++) {
        PV_VARIANT_D(ST_PACK, "multishift", d, PV_ISA_AVX512,
                     "bench_pack_v2.c pack_v2_dN",
                     "vpmultishiftqb, 64 codes/iter", 0, PV_FN_VBMI2(prim_pack_multishift));
        PV_VARIANT_D(ST_PACK, "asof-cd119a6", d, PV_ISA_AVX512,
                     "cd119a6 pack_dN_bmi2",
                     "BMI2 pext pack, 8 codes/iter", 0, PV_FN_BMI2(prim_pack_bmi2));
        PV_VARIANT_D(ST_PACK, "asof-2f80076", d, PV_ISA_AVX512,
                     "cd119a6~1 PACK_DN_AVX512_UNIFIED",
                     "sllv + reduce_add pack, 8 codes/iter", 0, PV_FN_AVX512F(prim_pack_sllv));
    }
}

#endif /* PIVCO_PRIM_VARIANTS_PACK_H */
