/* bench/prim_variants/prims-flat.h — flat-subtree decode variant graveyard.
 *
 * Logical primitives: flat_dN_unpack (ST_UNPACK, per-D) and merge_flat
 * (ST_MERGE_FLAT, per-D).  See prims.h for the contract + naming
 * (PV_ = constants/macros, pv_ = plumbing, prim_ = kernels); per-D rows use
 * PV_VARIANT_D.  Production flat unpack helpers (flat_dN_unpack[_safe]) from
 * src/pivco_huffman_neon_flat.h are in scope here.
 *
 * What's here vs deliberately omitted:
 *   fl_natural (D=2,4)  — row-major shift+mask unpack with vst4q/vst2q
 *                         deinterleave.  Output IS sequential row-major, so
 *                         it verifies against the production LSB-first layout.
 *   asof-6dc5632 (D=5)  — the 2-register-TBL flat decode as first shipped:
 *                         unpack via memcpy(5)+vsetq_lane_u64 (later replaced
 *                         by the byte-wise vsetq_lane_u8 form to dodge a
 *                         Neoverse-V2 store-forward stall) + vqtbl2 c2s lookup.
 *                         Identical output to production, different load.
 *
 *   fl_layout (D=2..7, incl. the only D=7 NEON unpack): the FastLanes
 *     1024-vector TRANSPOSED layout.  It reads/writes a different bit
 *     ordering than pivco's row-major LSB-first packed stream, so it does
 *     NOT match scalar_unpack of the same bytes and cannot be verified
 *     byte-exact in this harness — omitted.  Code lives in
 *     extras/bench/bench_unpack_fl_layout.c.
 *
 *   asof-460709b (flat_d3/d5/d6 byte-wise vsetq_lane_u8): byte-for-byte
 *     identical to the production flat_d{3,5,6}_unpack_safe already used by
 *     neon_unpack / simd_merge_flat — a pure duplicate, omitted.
 */
#ifndef PIVCO_PRIM_VARIANTS_FLAT_H
#define PIVCO_PRIM_VARIANTS_FLAT_H

#if defined(USE_NEON_KERNELS)

/* ============================================================================
 * flat_dN_unpack : fl_natural — row-major shift+mask + vstKq deinterleave.
 *   Defined only when D | 8 (D=2 -> vst4q, D=4 -> vst2q).  From
 *   bench_unpack_fl_layout.c (fl_natural_d2 / fl_natural_d4).  The vstKq
 *   interleave restores sequential row-major order, so codes[i] matches the
 *   scalar reference.  64 codes/iter (D=2) / 32 codes/iter (D=4).
 * ========================================================================== */
static void prim_flat_unpack_fl_natural_d2(const ctx_t *c) {
    uint8_t *out = c->codes; const uint8_t *in = c->bm; int n = c->n;
    uint8x16_t mask3 = vdupq_n_u8(0x03);
    for (int i = 0; i + 64 <= n; i += 64) {
        uint8x16_t reg = vld1q_u8(in + (i >> 2));
        uint8x16_t g0 = vandq_u8(reg, mask3);
        uint8x16_t g1 = vandq_u8(vshrq_n_u8(reg, 2), mask3);
        uint8x16_t g2 = vandq_u8(vshrq_n_u8(reg, 4), mask3);
        uint8x16_t g3 = vandq_u8(vshrq_n_u8(reg, 6), mask3);
        uint8x16x4_t v = {{g0, g1, g2, g3}};
        vst4q_u8(out + i, v);
    }
}
static void prim_flat_unpack_fl_natural_d4(const ctx_t *c) {
    uint8_t *out = c->codes; const uint8_t *in = c->bm; int n = c->n;
    uint8x16_t maskF = vdupq_n_u8(0x0F);
    for (int i = 0; i + 32 <= n; i += 32) {
        uint8x16_t reg = vld1q_u8(in + (i >> 1));
        uint8x16_t g0 = vandq_u8(reg, maskF);
        uint8x16_t g1 = vandq_u8(vshrq_n_u8(reg, 4), maskF);
        uint8x16x2_t v = {{g0, g1}};
        vst2q_u8(out + i, v);
    }
}

/* ============================================================================
 * merge_flat : asof-6dc5632 — D=5 flat decode as first shipped.  Unpack 8
 *   codes / 5 bytes via memcpy(&packed,5)+vsetq_lane_u64 (the v0.1 load,
 *   replaced at 460709b by byte-wise vsetq_lane_u8 to avoid a Neoverse-V2
 *   int-store->vector-load forward stall) + 32-entry c2s via vqtbl2q.  Reuses
 *   the production flat_d5 shuffle/shift tables.  Output is byte-identical to
 *   the current production merge_flat D=5; this isolates the load strategy.
 * ========================================================================== */
static inline uint8x8_t pv_flat_d5_unpack_memcpy(const uint8_t *bm_ptr) {
    uint64_t packed = 0;
    memcpy(&packed, bm_ptr, 5);
    uint8x16_t bm_lo = vreinterpretq_u8_u64(vsetq_lane_u64(packed, vdupq_n_u64(0), 0));
    uint8x16_t shuffled = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d5_shuf_tab));
    uint16x8_t w = vreinterpretq_u16_u8(shuffled);
    uint16x8_t shifted = vshlq_u16(w, vld1q_s16(flat_d5_shift_tab));
    uint16x8_t masked = vandq_u16(shifted, vdupq_n_u16(0x1F));
    return vmovn_u16(masked);
}
static void prim_merge_flat_asof_6dc5632_d5(const ctx_t *c) {
    uint8_t *out = c->out; int n = c->n; const uint8_t *bm = c->bm; const uint8_t *c2s = c->c2s;
    uint8x16x2_t c2s_vec; c2s_vec.val[0] = vld1q_u8(c2s); c2s_vec.val[1] = vld1q_u8(c2s + 16);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x8_t lo = pv_flat_d5_unpack_memcpy(bm + ((i      * 5) >> 3));
        uint8x8_t hi = pv_flat_d5_unpack_memcpy(bm + (((i + 8) * 5) >> 3));
        uint8x16_t codes = vcombine_u8(lo, hi);
        vst1q_u8(out + i, vqtbl2q_u8(c2s_vec, codes));
    }
    for (; i + 8 <= n; i += 8) {
        uint8x8_t codes = pv_flat_d5_unpack_memcpy(bm + ((i * 5) >> 3));
        vst1_u8(out + i, vqtbl2_u8(c2s_vec, codes));
    }
    for (; i < n; i++) {
        const uint8_t *p = bm + ((i * 5) >> 3); int sh = (i * 5) & 7;
        uint16_t w; memcpy(&w, p, 2);
        out[i] = c2s[(w >> sh) & 0x1F];
    }
}

#endif /* USE_NEON_KERNELS */

/* ============================================================================
 * x86 (AVX2) flat unpack — "asof-d580b16": the pre-ryg vpsrlvd AVX2 flat
 * unpackers from git d580b16~1:src/pivco_huffman_x86_flat.h, restored verbatim
 * and composed with a scalar c2s gather into merge_flat(out,n,bm,c2s) for
 * D=2,3,5,6.  Production has since replaced these with ryg's PSHUFB+PMULLO
 * unpack (a1aa6b9); kept for the record / cross-uarch comparison.
 * ========================================================================== */
#if defined(__AVX2__)
#include <immintrin.h>
#include <string.h>

static inline __m128i pv_flat_d2_unpack_avx2(const uint8_t *bm_ptr) {
    uint32_t packed; memcpy(&packed, bm_ptr, 4);
    __m256i v = _mm256_set1_epi32((int)packed);
    const __m256i s0 = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i s1 = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    const __m256i m  = _mm256_set1_epi32(0x3);
    __m256i v0 = _mm256_and_si256(_mm256_srlv_epi32(v, s0), m);
    __m256i v1 = _mm256_and_si256(_mm256_srlv_epi32(v, s1), m);
    const __m256i bshuf = _mm256_setr_epi8(
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    __m256i p0 = _mm256_shuffle_epi8(v0, bshuf);
    __m256i p1 = _mm256_shuffle_epi8(v1, bshuf);
    __m128i c0 = _mm_unpacklo_epi32(_mm256_castsi256_si128(p0),
                                    _mm256_extracti128_si256(p0, 1));
    __m128i c1 = _mm_unpacklo_epi32(_mm256_castsi256_si128(p1),
                                    _mm256_extracti128_si256(p1, 1));
    return _mm_unpacklo_epi64(c0, c1);
}
static inline __m128i pv_flat_d3_unpack_avx2(const uint8_t *bm_ptr) {
    uint32_t packed; memcpy(&packed, bm_ptr, 4);
    __m256i v = _mm256_set1_epi32((int)packed);
    const __m256i sh = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    v = _mm256_and_si256(_mm256_srlv_epi32(v, sh), _mm256_set1_epi32(0x7));
    const __m256i bshuf = _mm256_setr_epi8(
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    __m256i s = _mm256_shuffle_epi8(v, bshuf);
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(s),
                              _mm256_extracti128_si256(s, 1));
}
static inline __m128i pv_flat_d5_unpack_avx2(const uint8_t *bm_ptr) {
    __m128i raw128 = _mm_loadu_si128((const __m128i *)bm_ptr);
    __m256i src = _mm256_broadcastsi128_si256(raw128);
    const __m256i byteidx = _mm256_setr_epi8(
        0,1,-1,-1, 0,1,-1,-1, 1,2,-1,-1, 1,2,-1,-1,
        2,3,-1,-1, 3,4,-1,-1, 3,4,-1,-1, 4,5,-1,-1);
    __m256i bytes = _mm256_shuffle_epi8(src, byteidx);
    const __m256i sub = _mm256_setr_epi32(0, 5, 2, 7, 4, 1, 6, 3);
    __m256i v = _mm256_and_si256(_mm256_srlv_epi32(bytes, sub),
                                 _mm256_set1_epi32(0x1F));
    const __m256i bshuf = _mm256_setr_epi8(
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    __m256i p = _mm256_shuffle_epi8(v, bshuf);
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(p),
                              _mm256_extracti128_si256(p, 1));
}
static inline __m128i pv_flat_d6_unpack_avx2(const uint8_t *bm_ptr) {
    __m128i raw128 = _mm_loadu_si128((const __m128i *)bm_ptr);
    __m256i src = _mm256_broadcastsi128_si256(raw128);
    const __m256i byteidx = _mm256_setr_epi8(
        0,1,-1,-1, 0,1,-1,-1, 1,2,-1,-1, 2,3,-1,-1,
        3,4,-1,-1, 3,4,-1,-1, 4,5,-1,-1, 5,6,-1,-1);
    __m256i bytes = _mm256_shuffle_epi8(src, byteidx);
    const __m256i sub = _mm256_setr_epi32(0, 6, 4, 2, 0, 6, 4, 2);
    __m256i v = _mm256_and_si256(_mm256_srlv_epi32(bytes, sub),
                                 _mm256_set1_epi32(0x3F));
    const __m256i bshuf = _mm256_setr_epi8(
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    __m256i p = _mm256_shuffle_epi8(v, bshuf);
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(p),
                              _mm256_extracti128_si256(p, 1));
}
static inline void pv_merge_flat_d2_avx2(uint8_t *out, int n, const uint8_t *bm,
                                         const uint8_t *c2s) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8_t cd[16];
        _mm_storeu_si128((__m128i *)cd, pv_flat_d2_unpack_avx2(bm + (i >> 4) * 4));
        for (int k = 0; k < 16; k++) out[i + k] = c2s[cd[k]];
    }
    for (; i < n; i++) { int bo = i * 2; out[i] = c2s[(bm[bo>>3] >> (bo&7)) & 0x3]; }
}
static inline void pv_merge_flat_dN_avx2(uint8_t *out, int n, const uint8_t *bm,
                                         const uint8_t *c2s, int D,
                                         __m128i (*unpack)(const uint8_t *)) {
    uint32_t cmask = (1u << D) - 1u;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8_t cd[16];
        _mm_storeu_si128((__m128i *)cd, unpack(bm + (i >> 3) * D));
        for (int k = 0; k < 8; k++) out[i + k] = c2s[cd[k]];
    }
    for (; i < n; i++) { int bo = i * D; uint64_t acc; memcpy(&acc, bm + (bo>>3), 8);
                         out[i] = c2s[(acc >> (bo&7)) & cmask]; }
}
static void prim_merge_flat_asof_d2(const ctx_t *c){
    pv_merge_flat_d2_avx2(c->out, c->n, c->bm, c->c2s);
}
static void prim_merge_flat_asof_d3(const ctx_t *c){
    pv_merge_flat_dN_avx2(c->out, c->n, c->bm, c->c2s, 3, pv_flat_d3_unpack_avx2);
}
static void prim_merge_flat_asof_d5(const ctx_t *c){
    pv_merge_flat_dN_avx2(c->out, c->n, c->bm, c->c2s, 5, pv_flat_d5_unpack_avx2);
}
static void prim_merge_flat_asof_d6(const ctx_t *c){
    pv_merge_flat_dN_avx2(c->out, c->n, c->bm, c->c2s, 6, pv_flat_d6_unpack_avx2);
}
#endif /* __AVX2__ */


#if defined(USE_NEON_KERNELS)
/* ============================================================================
 * asof-e96529e merge_flat_dN — the production NEON flat decode before the
 * issue-#5 (dougallj, gist cf33841) kernels replaced it: separate flat_dN
 * unpack + vqtblN c2s scatter per 8/16 codes (d8: 256-entry vqtbl4+3x vqtbx4).
 * ========================================================================== */
static void pv_mf_e96529e_d2(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t codes = flat_d2_unpack(bm + (i >> 2));
        uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
        vst1q_u8(symbols + i, syms);
    }
    for (; i + 4 <= n; i += 4) {
        uint8_t b = bm[i >> 2];
        symbols[i    ] = c2s[(b     ) & 3];
        symbols[i + 1] = c2s[(b >> 2) & 3];
        symbols[i + 2] = c2s[(b >> 4) & 3];
        symbols[i + 3] = c2s[(b >> 6) & 3];
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 2, 2);
        symbols[i] = c2s[code];
    }
}

static void pv_mf_e96529e_d3(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    int i = 0;
    int fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        uint8x8_t codes_lo = flat_d3_unpack_fast(bm + ((i      * 3) >> 3));
        uint8x8_t codes_hi = flat_d3_unpack_fast(bm + (((i + 8) * 3) >> 3));
        uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
        uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
        vst1q_u8(symbols + i, syms);
    }
    for (; i + 8 <= fast_end; i += 8) {
        uint8x8_t codes = flat_d3_unpack_fast(bm + ((i * 3) >> 3));
        uint8x8_t syms  = vqtbl1_u8(c2s_vec, codes);
        vst1_u8(symbols + i, syms);
    }
    for (; i + 8 <= n; i += 8) {
        uint8x8_t codes = flat_d3_unpack_safe(bm + ((i * 3) >> 3));
        uint8x8_t syms  = vqtbl1_u8(c2s_vec, codes);
        vst1_u8(symbols + i, syms);
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 3, 3);
        symbols[i] = c2s[code];
    }
}

static void pv_mf_e96529e_d4(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t codes = flat_d4_unpack(bm + (i >> 1));
        uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
        vst1q_u8(symbols + i, syms);
    }
    for (; i + 2 <= n; i += 2) {
        uint8_t b = bm[i >> 1];
        symbols[i    ] = c2s[b & 0x0F];
        symbols[i + 1] = c2s[b >> 4];
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 4, 4);
        symbols[i] = c2s[code];
    }
}

static void pv_mf_e96529e_d5(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16x2_t c2s_vec;
    c2s_vec.val[0] = vld1q_u8(c2s);
    c2s_vec.val[1] = vld1q_u8(c2s + 16);
    int i = 0;
    int fast_end = n >= 24 ? n - 24 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        uint8x8_t codes_lo = flat_d5_unpack_fast(bm + ((i      * 5) >> 3));
        uint8x8_t codes_hi = flat_d5_unpack_fast(bm + (((i + 8) * 5) >> 3));
        uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
        uint8x16_t syms  = vqtbl2q_u8(c2s_vec, codes);
        vst1q_u8(symbols + i, syms);
    }
    for (; i + 8 <= fast_end; i += 8) {
        uint8x8_t codes = flat_d5_unpack_fast(bm + ((i * 5) >> 3));
        uint8x8_t syms  = vqtbl2_u8(c2s_vec, codes);
        vst1_u8(symbols + i, syms);
    }
    for (; i + 8 <= n; i += 8) {
        uint8x8_t codes = flat_d5_unpack_safe(bm + ((i * 5) >> 3));
        uint8x8_t syms  = vqtbl2_u8(c2s_vec, codes);
        vst1_u8(symbols + i, syms);
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 5, 5);
        symbols[i] = c2s[code];
    }
}

static void pv_mf_e96529e_d6(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16x4_t c2s_vec;
    c2s_vec.val[0] = vld1q_u8(c2s);
    c2s_vec.val[1] = vld1q_u8(c2s + 16);
    c2s_vec.val[2] = vld1q_u8(c2s + 32);
    c2s_vec.val[3] = vld1q_u8(c2s + 48);
    int i = 0;
    int fast_end = n >= 24 ? n - 24 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        uint8x8_t codes_lo = flat_d6_unpack_fast(bm + ((i      * 6) >> 3));
        uint8x8_t codes_hi = flat_d6_unpack_fast(bm + (((i + 8) * 6) >> 3));
        uint8x16_t codes = vcombine_u8(codes_lo, codes_hi);
        uint8x16_t syms  = vqtbl4q_u8(c2s_vec, codes);
        vst1q_u8(symbols + i, syms);
    }
    for (; i + 8 <= fast_end; i += 8) {
        uint8x8_t codes = flat_d6_unpack_fast(bm + ((i * 6) >> 3));
        uint8x8_t syms  = vqtbl4_u8(c2s_vec, codes);
        vst1_u8(symbols + i, syms);
    }
    for (; i + 8 <= n; i += 8) {
        uint8x8_t codes = flat_d6_unpack_safe(bm + ((i * 6) >> 3));
        uint8x8_t syms  = vqtbl4_u8(c2s_vec, codes);
        vst1_u8(symbols + i, syms);
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_neon(bm, i * 6, 6);
        symbols[i] = c2s[code];
    }
}

static void pv_mf_e96529e_d8(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    uint8x16x4_t t0, t1, t2, t3;
    t0.val[0]=vld1q_u8(c2s     ); t0.val[1]=vld1q_u8(c2s + 16);
    t0.val[2]=vld1q_u8(c2s + 32); t0.val[3]=vld1q_u8(c2s + 48);
    t1.val[0]=vld1q_u8(c2s + 64); t1.val[1]=vld1q_u8(c2s + 80);
    t1.val[2]=vld1q_u8(c2s + 96); t1.val[3]=vld1q_u8(c2s +112);
    t2.val[0]=vld1q_u8(c2s +128); t2.val[1]=vld1q_u8(c2s +144);
    t2.val[2]=vld1q_u8(c2s +160); t2.val[3]=vld1q_u8(c2s +176);
    t3.val[0]=vld1q_u8(c2s +192); t3.val[1]=vld1q_u8(c2s +208);
    t3.val[2]=vld1q_u8(c2s +224); t3.val[3]=vld1q_u8(c2s +240);
    uint8x16_t s64  = vdupq_n_u8(64);
    uint8x16_t s128 = vdupq_n_u8(128);
    uint8x16_t s192 = vdupq_n_u8(192);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t codes = vld1q_u8(bm + i);
        uint8x16_t s = vqtbl4q_u8(t0, codes);
        s = vqtbx4q_u8(s, t1, vsubq_u8(codes, s64));
        s = vqtbx4q_u8(s, t2, vsubq_u8(codes, s128));
        s = vqtbx4q_u8(s, t3, vsubq_u8(codes, s192));
        vst1q_u8(symbols + i, s);
    }
    for (; i < n; i++) symbols[i] = c2s[bm[i]];
}

static void prim_mf_e96529e_d2(const ctx_t *c){ pv_mf_e96529e_d2(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d3(const ctx_t *c){ pv_mf_e96529e_d3(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d4(const ctx_t *c){ pv_mf_e96529e_d4(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d5(const ctx_t *c){ pv_mf_e96529e_d5(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d6(const ctx_t *c){ pv_mf_e96529e_d6(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d8(const ctx_t *c){ pv_mf_e96529e_d8(c->out, c->n, c->bm, c->c2s); }
#endif /* USE_NEON_KERNELS */

/* ============================================================================
 * Registry — flat family (no-op where the ISA is unavailable)
 * ========================================================================== */
static void pv_register_flat(void) {
    PV_VARIANT_D(ST_UNPACK,     "fl_natural", 2, PV_ISA_NEON, "bench_unpack_fl_layout.c",
                 "row-major shift+mask + vst4q deinterleave (D|8 only)", 0, PV_FN_NEON(prim_flat_unpack_fl_natural_d2));
    PV_VARIANT_D(ST_UNPACK,     "fl_natural", 4, PV_ISA_NEON, "bench_unpack_fl_layout.c",
                 "row-major shift+mask + vst2q deinterleave (D|8 only)", 0, PV_FN_NEON(prim_flat_unpack_fl_natural_d4));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-6dc5632", 5, PV_ISA_NEON, "6dc5632",
                 "first-shipped D=5 flat decode: memcpy(5)+vsetq_lane_u64 unpack + vqtbl2 c2s", 0, PV_FN_NEON(prim_merge_flat_asof_6dc5632_d5));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-e96529e", 2, PV_ISA_NEON, "e96529e (prior production)",
                 "unpack+vqtbl1, 16/iter", 0, PV_FN_NEON(prim_mf_e96529e_d2));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-e96529e", 3, PV_ISA_NEON, "e96529e (prior production)",
                 "2x flat_d3_unpack_fast + vqtbl1q, 16/iter", 0, PV_FN_NEON(prim_mf_e96529e_d3));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-e96529e", 4, PV_ISA_NEON, "e96529e (prior production)",
                 "flat_d4_unpack + vqtbl1q, 16/iter", 0, PV_FN_NEON(prim_mf_e96529e_d4));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-e96529e", 5, PV_ISA_NEON, "e96529e (prior production)",
                 "2x flat_d5_unpack_fast + vqtbl2q, 16/iter", 0, PV_FN_NEON(prim_mf_e96529e_d5));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-e96529e", 6, PV_ISA_NEON, "e96529e (prior production)",
                 "2x flat_d6_unpack_fast + vqtbl4q, 16/iter", 0, PV_FN_NEON(prim_mf_e96529e_d6));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-e96529e", 8, PV_ISA_NEON, "e96529e (prior production)",
                 "256-entry c2s: vqtbl4 + 3x vqtbx4 per 16 (new prod is memcpy: d8 flat = full alphabet = identity c2s)", 0, PV_FN_NEON(prim_mf_e96529e_d8));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-d580b16", 2, PV_ISA_AVX2, "d580b16~1:pivco_huffman_x86_flat.h",
                 "pre-ryg vpsrlvd AVX2 flat unpack + scalar c2s gather", 0, PV_FN_AVX2(prim_merge_flat_asof_d2));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-d580b16", 3, PV_ISA_AVX2, "d580b16~1:pivco_huffman_x86_flat.h",
                 "pre-ryg vpsrlvd AVX2 flat unpack + scalar c2s gather", 0, PV_FN_AVX2(prim_merge_flat_asof_d3));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-d580b16", 5, PV_ISA_AVX2, "d580b16~1:pivco_huffman_x86_flat.h",
                 "pre-ryg vpsrlvd AVX2 flat unpack + scalar c2s gather", 0, PV_FN_AVX2(prim_merge_flat_asof_d5));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-d580b16", 6, PV_ISA_AVX2, "d580b16~1:pivco_huffman_x86_flat.h",
                 "pre-ryg vpsrlvd AVX2 flat unpack + scalar c2s gather", 0, PV_FN_AVX2(prim_merge_flat_asof_d6));
}

#endif /* PIVCO_PRIM_VARIANTS_FLAT_H */
