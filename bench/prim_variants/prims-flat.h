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

/* d7 pair32: u32-lane pair-gather probe (x86's winning d7 form, ported back
 * to NEON for comparison).  One 16-byte load feeds two vqtbl1 gathers (a
 * 14-bit pair per u32 lane), vshlq_u32 normalizes, shift+mask split
 * code0/code1 into the low bytes, vuzp1 compacts; the vqtbl4+vqtbx4 scatter
 * matches production.  vs production: 1 load instead of 2, but +3 ops of
 * two-field extraction over the one-code-per-u16 vmovn form. */
static void pv_mf_pair32_d7_neonk(uint8_t *symbols, int n, const uint8_t *bm, const uint8_t *c2s)
{
    int i = 0;
    if (n >= 19) {
        uint8x16x4_t lo, hi;
        lo.val[0] = vld1q_u8(c2s);       lo.val[1] = vld1q_u8(c2s + 16);
        lo.val[2] = vld1q_u8(c2s + 32);  lo.val[3] = vld1q_u8(c2s + 48);
        hi.val[0] = vld1q_u8(c2s + 64);  hi.val[1] = vld1q_u8(c2s + 80);
        hi.val[2] = vld1q_u8(c2s + 96);  hi.val[3] = vld1q_u8(c2s + 112);
        static const uint8_t g_lo_t[16] = {0,1,2,3, 1,2,3,4, 3,4,5,6, 5,6,7,8};
        static const uint8_t g_hi_t[16] = {7,8,9,10, 8,9,10,11, 10,11,12,13, 12,13,14,15};
        static const int32_t sh_t[4]    = {0,-6,-4,-2};   /* >>o, o={0,6,4,2} */
        const uint8x16_t g_lo = vld1q_u8(g_lo_t), g_hi = vld1q_u8(g_hi_t);
        const int32x4_t  sh   = vld1q_s32(sh_t);
        const uint32x4_t m7f = vdupq_n_u32(0x7F), m7f00 = vdupq_n_u32(0x7F00);
        const uint8x16_t sub64q = vdupq_n_u8(64);
        int blocks = (n - 3) >> 4;
        for (int b = 0; b < blocks; ++b) {
            uint8x16_t packed = vld1q_u8(bm + b * 14);
            uint32x4_t xl = vshlq_u32(vreinterpretq_u32_u8(vqtbl1q_u8(packed, g_lo)), sh);
            uint32x4_t xh = vshlq_u32(vreinterpretq_u32_u8(vqtbl1q_u8(packed, g_hi)), sh);
            /* code0 at bits 0..6, code1 at bits 7..13 of each u32 */
            uint32x4_t cl = vorrq_u32(vandq_u32(xl, m7f),
                                      vandq_u32(vshlq_n_u32(xl, 1), m7f00));
            uint32x4_t ch = vorrq_u32(vandq_u32(xh, m7f),
                                      vandq_u32(vshlq_n_u32(xh, 1), m7f00));
            uint8x16_t codes = vreinterpretq_u8_u16(
                vuzp1q_u16(vreinterpretq_u16_u32(cl), vreinterpretq_u16_u32(ch)));
            uint8x16_t s = vqtbl4q_u8(lo, codes);
            s = vqtbx4q_u8(s, hi, vsubq_u8(codes, sub64q));
            vst1q_u8(symbols + (b << 4), s);
        }
        i = blocks << 4;
    }
    merge_flat_d7_neon(symbols + i, n - i, bm + ((i * 7) >> 3), c2s);
}
static void prim_mf_pair32_d7_neon(const ctx_t *c){ pv_mf_pair32_d7_neonk(c->out, c->n, c->bm, c->c2s); }

static void prim_mf_e96529e_d2(const ctx_t *c){ pv_mf_e96529e_d2(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d3(const ctx_t *c){ pv_mf_e96529e_d3(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d4(const ctx_t *c){ pv_mf_e96529e_d4(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d5(const ctx_t *c){ pv_mf_e96529e_d5(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d6(const ctx_t *c){ pv_mf_e96529e_d6(c->out, c->n, c->bm, c->c2s); }
static void prim_mf_e96529e_d8(const ctx_t *c){ pv_mf_e96529e_d8(c->out, c->n, c->bm, c->c2s); }
#endif /* USE_NEON_KERNELS */

/* ============================================================================
 * dougallj merge_flat d2/d3 — x86 ports (issue #5; NEON originals promoted in
 * 917614a).  Mechanical translation: vqtbl1 -> pshufb; the bidirectional
 * per-lane u16 shifts -> pmullw by 2^(s-min) + one uniform psrlw; per-byte
 * shifts -> psrlw + byte-mask cleanup; vst4q/vst2q -> punpck interleave trees.
 * Tails delegate to the production merge_flat_dN_x86.
 * ========================================================================== */
#if defined(__SSE4_1__) && !defined(__AVX512VBMI2__)

/* d2: two prepped nibble tables (TL[nib]=c2s[nib&3], TH[nib]=c2s[(nib>>2)&3])
 * map input nibbles straight to symbol pairs; 64 codes/iter, the 4-way
 * interleave is a 2-level punpck tree (vst4q equivalent). */
static void pv_mf_dj_d2_x86k(uint8_t *symbols, int n, const uint8_t *bm, const uint8_t *c2s)
{
    int i = 0;
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
    merge_flat_d2_x86(symbols + i, n - i, bm + (i >> 2), c2s);
}
static void prim_mf_dj_d2_x86(const ctx_t *c){ pv_mf_dj_d2_x86k(c->out, c->n, c->bm, c->c2s); }

/* asof-149ecb0: the pre-pair-gather production d3 (ryg unpack + 8-entry
 * pshufb, 8 codes/iter), kept benchable as the baseline. */
/* D=3: 8 codes/iter, ryg unpack + 8-entry pshufb scatter. */
static void pv_mf_149ecb0_d3_x86k(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    /* the unpack reads a 16-byte window — keep a generous scalar tail. */
    __m128i c2s_vec = _mm_loadl_epi64((const __m128i *)c2s);  /* 8 entries */
    int i = 0;
    int fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 8 <= fast_end; i += 8) {
        __m128i codes = flat_d3_unpack_x86(bm + ((i * 3) >> 3));
        __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
        _mm_storel_epi64((__m128i *)(symbols + i), syms);
    }
    merge_flat_tail_x86(symbols, i, n, bm, 3, c2s);
}
static void prim_mf_149ecb0_d3_x86(const ctx_t *c){ pv_mf_149ecb0_d3_x86k(c->out, c->n, c->bm, c->c2s); }

/* asof-149ecb0: the pre-pair-gather production d5 (ryg unpack + 2-pshufb/
 * blendv scatter, 8 codes/iter). */
static void pv_mf_149ecb0_d5_x86k(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    /* pshufb on either table uses code&15; blend by bit 4. */
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
    merge_flat_tail_x86(symbols, i, n, bm, 5, c2s);
}
static void prim_mf_149ecb0_d5_x86(const ctx_t *c){ pv_mf_149ecb0_d5_x86k(c->out, c->n, c->bm, c->c2s); }

/* asof-149ecb0: the pre-pair-gather production d6 (ryg unpack + 4-pshufb/
 * 2-level-blend scatter, 8 codes/iter). */
static void pv_mf_149ecb0_d6_x86k(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    /* four pshufb (code&15 into each quarter) then a 2-level blend by
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
    merge_flat_tail_x86(symbols, i, n, bm, 6, c2s);
}
static void prim_mf_149ecb0_d6_x86(const ctx_t *c){ pv_mf_149ecb0_d6_x86k(c->out, c->n, c->bm, c->c2s); }

/* asof-695c36e: the pre-pair-gather production d7 (scalar-unrolled u64
 * gather, 8 codes/iter; production kept it as the remainder path). */
static void pv_mf_695c36e_d7_x86k(uint8_t *symbols, int n,
                                                const uint8_t *bm,
                                                const uint8_t *c2s)
{
    int i = 0;
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
static void prim_mf_695c36e_d7_x86(const ctx_t *c){ pv_mf_695c36e_d7_x86k(c->out, c->n, c->bm, c->c2s); }

#endif /* __SSE4_1__ && !__AVX512VBMI2__ */

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
    PV_VARIANT_D(ST_MERGE_FLAT, "dougallj", 2, PV_ISA_SSE4, "issue #5 gist cf33841 (x86 port)",
                 "TL/TH prepped nibble tables + 2-level punpck interleave, 64 codes/iter; production on SSE-only builds -- on AVX2 builds prod keeps the vpsrlvd unpack (Intel: dougallj +7% c4 / +31% c5; AMD: -32% c5a / -17% c6a -- vendor-dispatch material)", 0, PV_FN_SSE(prim_mf_dj_d2_x86));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-149ecb0", 3, PV_ISA_SSE4, "149ecb0 (prior production)",
                 "ryg unpack + 8-entry pshufb, 8 codes/iter; the pair-gather that replaced it wins -38..-54% across c3/c4/c5/c5a/c6a", 0, PV_FN_SSE(prim_mf_149ecb0_d3_x86));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-149ecb0", 5, PV_ISA_SSE4, "149ecb0 (prior production)",
                 "ryg unpack + 2-pshufb/blendv, 8 codes/iter; the pair-gather that replaced it wins -42..-51% across c3/c4/c5/c5a/c6a", 0, PV_FN_SSE(prim_mf_149ecb0_d5_x86));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-149ecb0", 6, PV_ISA_SSE4, "149ecb0 (prior production)",
                 "ryg unpack + 4-pshufb/2-level-blend, 8 codes/iter; the pair-gather that replaced it wins -39..-52% across c3/c4/c5/c5a/c6a", 0, PV_FN_SSE(prim_mf_149ecb0_d6_x86));
    PV_VARIANT_D(ST_MERGE_FLAT, "asof-695c36e", 7, PV_ISA_SSE4, "695c36e (prior production)",
                 "scalar-unrolled u64 gather, 8 codes/iter; the u32-lane pair-gather that replaced it wins -17% c3 / -36% c4/c5 / -42% c5a / -47% c6a", 0, PV_FN_SSE(prim_mf_695c36e_d7_x86));
    PV_VARIANT_D(ST_MERGE_FLAT, "pair32", 7, PV_ISA_NEON, "x86 d7 pair-gather ported back for comparison",
                 "2 vqtbl1 u32-lane gathers + vshlq_u32 + vuzp1 compact; production vqtbl4/vqtbx4 scatter; LOSES +46% M4 / +28% c8g -- prod's one-code-per-u16 unpack + vmovn is cheaper than the pair split; x86 won only because it escaped a scalar loop", 0, PV_FN_NEON(prim_mf_pair32_d7_neon));
}

#endif /* PIVCO_PRIM_VARIANTS_FLAT_H */
