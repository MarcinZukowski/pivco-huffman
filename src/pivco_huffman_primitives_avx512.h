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
 * pivco_avx512.c bodies they replace):
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
 *   - BU merge_vec_vec_avx512 family: 64-byte vpexpandb main loop (one
 *     ZMM load+expand per side, OR'd together), SSE stride-16 tail
 *     using expand_tab from pivco_huffman_x86_tables (the only x86
 *     table this backend depends on; codec_init_avx512 inits it).
 *
 *   - merge_flat_avx512: D=2/3/4/5/6 vector unpacks via the
 *     shared pivco_huffman_avx512_flat.h helpers (D=5/6 use
 *     vpmultishiftqb / vpermb -- the AVX-512-only fast paths).
 *
 *   - pack_dN_avx512: D=2..7 via 64-codes-per-zmm byte-laid +
 *     vpmultishiftqb (D=3,5,6,7) or vpermb-stride + shift (D=2,4).  See
 *     pivco_huffman_avx512_pack.h.  D=8 via vpmovwb byte narrow.
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
#include "pivco_huffman_avx512_pack.h"     /* pack_d{2..7}_avx512 — vpmultishiftqb */
#include "pivco_prof.h"

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/* Backend lifecycle.  Only the BU merge SSE-stride tail needs a
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

/* merge_vec_vec_avx512 — VBMI2 64-byte main loop via vpexpandb (two
 * masked expand-loads OR'd together), SSE stride-16/-8 tails using
 * expand_tab from x86_tables.  ~0.023 ns/byte on Xeon Ice Lake+. */
static inline void merge_vec_vec_avx512(const uint8_t *bm, int K,
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
        /* Merge-masked expands, not maskz: Zen 4/5 have a false dependency on
         * the output register of zero-masked compress/expand (issue #11,
         * aadaa-fgtaa), which serializes iterations at expand latency.  The
         * asm barrier keeps the compiler from folding the zero back into a
         * maskz form.  Expanding R into L also replaces the OR. */
        __m512i zero = _mm512_setzero_si512(); asm("":"+v"(zero));
        __m512i L = _mm512_mask_expandloadu_epi8(zero, nm, left + lc);
        __m512i o = _mm512_mask_expandloadu_epi8(L, m,  right + rc);
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
    PROF_TOC(PROF_BU_MERGE_VEC_VEC, K);
}

/* merge_cst_vec_avx512 — left input is a broadcast constant. */
static inline void merge_cst_vec_avx512(const uint8_t *bm, int K,
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
        /* expand straight into the broadcast (issue #11: avoids the Zen 4/5
         * maskz false dep and drops the blend) */
        __m512i o = _mm512_mask_expandloadu_epi8(Lbcast64, m, right + rc);
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
    PROF_TOC(PROF_BU_MERGE_CST_VEC, K);
}

/* merge_cst_cst_avx512 — both inputs are constants.  Native AVX-512
 * stride-64: read 64 bm bits as a kmask, single mask_blend_epi8 of two
 * broadcast registers, one 64-byte store.  SSE 16-byte and scalar tails. */
static inline void merge_cst_cst_avx512(const uint8_t *bm, int K,
                                             uint8_t left_sym, uint8_t right_sym,
                                             uint8_t *out)
{
    PROF_TIC();
    __m512i vL_64 = _mm512_set1_epi8((char)left_sym);
    __m512i vR_64 = _mm512_set1_epi8((char)right_sym);
    int j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __m512i o = _mm512_mask_blend_epi8((__mmask64)mask, vL_64, vR_64);
        _mm512_storeu_si512((__m512i *)(out + j), o);
    }
    __m128i vL = _mm_set1_epi8((char)left_sym);
    __m128i vR = _mm_set1_epi8((char)right_sym);
    __m128i bits = _mm_setr_epi8(1,2,4,8,16,32,64,(char)128,
                                  1,2,4,8,16,32,64,(char)128);
    __m128i shuf = _mm_setr_epi8(0,0,0,0,0,0,0,0,
                                  1,1,1,1,1,1,1,1);
    for (; j + 16 <= K; j += 16) {
        __m128i bm_pair = _mm_cvtsi32_si128(*(const uint16_t *)(bm + (j >> 3)));
        __m128i bm_dup  = _mm_shuffle_epi8(bm_pair, shuf);
        __m128i masked  = _mm_and_si128(bm_dup, bits);
        __m128i mask8   = _mm_cmpeq_epi8(masked, bits);
        __m128i o       = _mm_blendv_epi8(vL, vR, mask8);
        _mm_storeu_si128((__m128i *)(out + j), o);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left_sym;
    }
    PROF_TOC(PROF_BU_MERGE_CST_CST, K);
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

/* Per-D direct decode (D-bit packed bm -> symbols).  Reads n*D packed
 * bits, looks up each D-bit code in c2s, writes the resulting bytes
 * to symbols[0..n).  merge_flat_avx512 is a switch
 * dispatcher to the per-D specialisation; structure mirrors the NEON
 * file at pivco_huffman_primitives_neon.h.
 *
 * Per-D unpack helpers (flat_d{2,3,4,5,6,7}_unpack_avx512*) come from
 * pivco_huffman_avx512_flat.h. */

/* D=2: c2s = 4 entries; codes < 4 use only low 2 bits so vpermb on a
 * 64-byte register whose first 4 bytes are c2s works.  Wide 64-at-a-time
 * path first, then existing 16-wide tail + scalar nibble unpack. */
static inline void merge_flat_d2_avx512(uint8_t *symbols, int n,
                                                  const uint8_t *bm,
                                                  const uint8_t *c2s)
{
    uint32_t c2s_lo;
    memcpy(&c2s_lo, c2s, 4);
    __m128i c2s_xmm = _mm_set1_epi32((int32_t)c2s_lo);
    __m512i c2s_zmm = _mm512_castsi128_si512(c2s_xmm);
    int i = 0;
    /* Slack: strict bound is ceil(32*8/2)=128 codes — happens to match the
     * "i + 128" symmetry of the other widths exactly. */
    for (; i + 128 <= n; i += 64) {
        __m512i codes = flat_d2_unpack64_avx512_fast(bm + (i >> 2));
        __m512i syms  = _mm512_permutexvar_epi8(codes, c2s_zmm);
        _mm512_storeu_si512((__m512i *)(symbols + i), syms);
    }
    for (; i + 16 <= n; i += 16) {
        __m128i codes = flat_d2_unpack_avx512(bm + (i >> 2));
        __m128i syms  = _mm_shuffle_epi8(c2s_xmm, codes);
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
        uint32_t code = extract_D_bits_avx512(bm, i * 2, 2);
        symbols[i] = c2s[code];
    }
}

/* D=3: c2s = 8 entries fits in low 8 bytes of an xmm; codes < 8 use only
 * the low 3 bits so vpermb against a 64-byte register whose first 8 bytes
 * are c2s (the rest don't-care) lands on the right entry.  Wide 64-at-a
 * -time path first (using flat_d3_unpack64), then existing 16-wide tail. */
static inline void merge_flat_d3_avx512(uint8_t *symbols, int n,
                                                  const uint8_t *bm,
                                                  const uint8_t *c2s)
{
    uint64_t c2s_lo;
    memcpy(&c2s_lo, c2s, 8);
    __m128i c2s_xmm = _mm_cvtsi64_si128((int64_t)c2s_lo);
    __m512i c2s_zmm = _mm512_castsi128_si512(c2s_xmm);
    int i = 0;
    /* Slack: strict bound is ceil(32*8/3)=86 codes; use 128 to drop cleanly
     * into the 16-wide tail (same pattern as merge_flat_d5_avx512). */
    for (; i + 128 <= n; i += 64) {
        __m512i codes = flat_d3_unpack64_avx512_fast(bm + ((i * 3) >> 3));
        __m512i syms  = _mm512_permutexvar_epi8(codes, c2s_zmm);
        _mm512_storeu_si512((__m512i *)(symbols + i), syms);
    }
    int fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        __m128i codes = flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3));
        __m128i syms  = _mm_shuffle_epi8(c2s_xmm, codes);
        _mm_storeu_si128((__m128i *)(symbols + i), syms);
    }
    if (i + 16 <= n) {
        __m128i codes = flat_d3_unpack_avx512_safe(bm + ((i * 3) >> 3));
        __m128i syms  = _mm_shuffle_epi8(c2s_xmm, codes);
        _mm_storeu_si128((__m128i *)(symbols + i), syms);
        i += 16;
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_avx512(bm, i * 3, 3);
        symbols[i] = c2s[code];
    }
}

/* D=4: c2s = 16 entries; codes < 16 use only low 4 bits so vpermb on a
 * 64-byte register whose first 16 bytes are c2s (rest don't-care) works.
 * Wide 64-at-a-time path first, then existing 16-wide tail.  No
 * over-read concern for the wide path since flat_d4_unpack64's 32-byte
 * load only consumes bytes 0..31 = 32 valid bytes, but ymm read of bm
 * still over-reads past the bm region. */
static inline void merge_flat_d4_avx512(uint8_t *symbols, int n,
                                                  const uint8_t *bm,
                                                  const uint8_t *c2s)
{
    __m128i c2s_xmm = _mm_loadu_si128((const __m128i *)c2s);
    __m512i c2s_zmm = _mm512_castsi128_si512(c2s_xmm);
    int i = 0;
    /* Slack: strict bound is ceil(32*8/4)=64 codes; use 128 to drop into
     * the 16-wide tail and stay symmetric with the other widths. */
    for (; i + 128 <= n; i += 64) {
        __m512i codes = flat_d4_unpack64_avx512_fast(bm + ((i * 4) >> 3));
        __m512i syms  = _mm512_permutexvar_epi8(codes, c2s_zmm);
        _mm512_storeu_si512((__m512i *)(symbols + i), syms);
    }
    for (; i + 16 <= n; i += 16) {
        __m128i codes = flat_d4_unpack_avx512(bm + (i >> 1));
        __m128i syms  = _mm_shuffle_epi8(c2s_xmm, codes);
        _mm_storeu_si128((__m128i *)(symbols + i), syms);
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_avx512(bm, i * 4, 4);
        symbols[i] = c2s[code];
    }
}

/* D=5: c2s = 32 entries.  Fast path uses the 64-at-a-time zmm unpack
 * (flat_d5_unpack64_avx512_fast) + vpermb over a zmm broadcast of the
 * 32-byte c2s; codes < 32 keep the high half don't-care so the cast is
 * safe.  Tail of 16..63 codes drops to the 16-at-a-time vpermb-ymm
 * form, and the last <16 codes go scalar. */
static inline void merge_flat_d5_avx512(uint8_t *symbols, int n,
                                                  const uint8_t *bm,
                                                  const uint8_t *c2s)
{
    __m256i c2s_ymm = _mm256_loadu_si256((const __m256i *)c2s);
    __m512i c2s_zmm = _mm512_castsi256_si512(c2s_ymm);
    int i = 0;
    /* 64-wide fast loop.  Leave a 64-element safety margin so the
     * 64-byte zmm load past bm[i*5/8] never reads into uninitialised
     * pages — the tail handles the remainder. */
    int fast64_end = n >= 64 ? n - 64 : 0;
    for (; i + 64 <= fast64_end; i += 64) {
        __m512i codes = flat_d5_unpack64_avx512_fast(bm + ((i * 5) >> 3));
        __m512i syms  = _mm512_permutexvar_epi8(codes, c2s_zmm);
        _mm512_storeu_si512((__m512i *)(symbols + i), syms);
    }
    /* 16-wide fast loop drains down to ≤16 remaining. */
    int fast16_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast16_end; i += 16) {
        __m128i codes = flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3));
        __m256i codes_ext = _mm256_zextsi128_si256(codes);
        __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_ymm);
        _mm_storeu_si128((__m128i *)(symbols + i),
                         _mm256_castsi256_si128(syms_full));
    }
    if (i + 16 <= n) {
        __m128i codes = flat_d5_unpack_avx512_safe(bm + ((i * 5) >> 3));
        __m256i codes_ext = _mm256_zextsi128_si256(codes);
        __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_ymm);
        _mm_storeu_si128((__m128i *)(symbols + i),
                         _mm256_castsi256_si128(syms_full));
        i += 16;
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_avx512(bm, i * 5, 5);
        symbols[i] = c2s[code];
    }
}

/* D=6: c2s = 64 entries fits in a zmm.  Wide 64-at-a-time path first
 * (flat_d6_unpack64 + vpermb-zmm), then existing 16-wide tail. */
static inline void merge_flat_d6_avx512(uint8_t *symbols, int n,
                                                  const uint8_t *bm,
                                                  const uint8_t *c2s)
{
    __m512i c2s_zmm = _mm512_loadu_si512((const __m512i *)c2s);
    int i = 0;
    /* Slack: strict bound is ceil(64*8/6)=86 codes; use 128 to match the
     * other widths and drop into the 16-wide tail. */
    for (; i + 128 <= n; i += 64) {
        __m512i codes = flat_d6_unpack64_avx512_fast(bm + ((i * 6) >> 3));
        __m512i syms  = _mm512_permutexvar_epi8(codes, c2s_zmm);
        _mm512_storeu_si512((__m512i *)(symbols + i), syms);
    }
    int fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        __m128i codes = flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3));
        __m512i codes_ext = _mm512_castsi128_si512(codes);
        __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_zmm);
        _mm_storeu_si128((__m128i *)(symbols + i),
                         _mm512_castsi512_si128(syms_full));
    }
    if (i + 16 <= n) {
        __m128i codes = flat_d6_unpack_avx512_safe(bm + ((i * 6) >> 3));
        __m512i codes_ext = _mm512_castsi128_si512(codes);
        __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_zmm);
        _mm_storeu_si128((__m128i *)(symbols + i),
                         _mm512_castsi512_si128(syms_full));
        i += 16;
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_avx512(bm, i * 6, 6);
        symbols[i] = c2s[code];
    }
}

/* D=7: c2s = 128 entries spans two zmm tables.  One vpermi2b looks up
 * the full 128-byte table in a single op.  Wide 64-at-a-time path first
 * (flat_d7_unpack64 + vpermi2b), then existing 16-wide tail. */
static inline void merge_flat_d7_avx512(uint8_t *symbols, int n,
                                                  const uint8_t *bm,
                                                  const uint8_t *c2s)
{
    __m512i c2s_lo = _mm512_loadu_si512((const __m512i *)c2s);
    __m512i c2s_hi = _mm512_loadu_si512((const __m512i *)(c2s + 64));
    int i = 0;
    /* Slack: strict bound is ceil(64*8/7)=74 codes; use 128 for symmetry
     * with the other widths. */
    for (; i + 128 <= n; i += 64) {
        __m512i codes = flat_d7_unpack64_avx512_fast(bm + ((i * 7) >> 3));
        __m512i syms  = _mm512_permutex2var_epi8(c2s_lo, codes, c2s_hi);
        _mm512_storeu_si512((__m512i *)(symbols + i), syms);
    }
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
        uint32_t code = extract_D_bits_avx512(bm, i * 7, 7);
        symbols[i] = c2s[code];
    }
}

/* D=8: a depth-8 flat region is the full 256-symbol alphabet at equal code
 * length, whose canonical c2s is the identity permutation -- the byte-aligned
 * codes ARE the symbols, so the whole decode is a memcpy.  See the derivation
 * at merge_flat_d8_neon in pivco_huffman_primitives_neon.h. */
static inline void merge_flat_d8_avx512(uint8_t *symbols, int n,
                                                  const uint8_t *bm,
                                                  const uint8_t *c2s)
{
    (void)c2s;
    memcpy(symbols, bm, (size_t)n);
}

/* merge_flat_avx512 — D-bit flat-subtree decode into a
 * contiguous output buffer.  Dispatches to the per-D specialisation. */
static inline void merge_flat_avx512(uint8_t *out, int n,
                                                  const uint8_t *bm, int D,
                                                  const uint8_t *c2s)
{
    PROF_TIC();
    switch (D) {
    case 2: merge_flat_d2_avx512(out, n, bm, c2s); break;
    case 3: merge_flat_d3_avx512(out, n, bm, c2s); break;
    case 4: merge_flat_d4_avx512(out, n, bm, c2s); break;
    case 5: merge_flat_d5_avx512(out, n, bm, c2s); break;
    case 6: merge_flat_d6_avx512(out, n, bm, c2s); break;
    case 7: merge_flat_d7_avx512(out, n, bm, c2s); break;
    case 8: merge_flat_d8_avx512(out, n, bm, c2s); break;
    default:
        for (int i = 0; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            out[i] = c2s[code];
        }
        break;
    }
    PROF_TOC(PROF_BU_MERGE_FLAT, n);
}

/* ---------- Encode primitives: rank-based encoding (8-bit in-order ranks) ----------
 * Partition 8-bit leaf ranks against split_rank via vpcompressb (64/iter).
 * Flat pack subtracts flat_base_rank then reuses pack_dN_avx512. */
#include <stdlib.h>

/* init_avx512 — gather ranks[i] = sym_to_rank[symbols[i]] via two vpermi2b
 * over the 256-byte LUT.  Simpler than the code_la enc_init (output is a single
 * rank byte, no lo/hi code split): each vpermi2b covers 128 entries indexed by
 * the symbol's low 7 bits; blend the two halves by bit 7.  64 ranks/iter.
 *
 * This is the AVX-512 encode bottleneck if left scalar: a scalar 1 MB byte
 * gather runs several x longer than the whole vectorized partition tree, so on
 * AVX-512 it (not the partition) was what made the rank encode trail code_la. */
static inline void init_avx512(uint8_t *ranks, int n,
                                  const uint8_t *sym, const uint8_t *s2r)
{
    __m512i t_lo0 = _mm512_loadu_si512((const __m512i *)(s2r +   0));  /* entries [  0: 64) */
    __m512i t_lo1 = _mm512_loadu_si512((const __m512i *)(s2r +  64));  /* entries [ 64:128) */
    __m512i t_hi0 = _mm512_loadu_si512((const __m512i *)(s2r + 128));  /* entries [128:192) */
    __m512i t_hi1 = _mm512_loadu_si512((const __m512i *)(s2r + 192));  /* entries [192:256) */

    PROF_TIC();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i c = _mm512_loadu_si512((const __m512i *)(sym + i));
        __mmask64 hib = _mm512_movepi8_mask(c);                       /* bit 7 of each symbol */
        __m512i lo = _mm512_permutex2var_epi8(t_lo0, c, t_lo1);       /* LUT[ c & 127]        */
        __m512i hi = _mm512_permutex2var_epi8(t_hi0, c, t_hi1);       /* LUT[128 + (c & 127)] */
        _mm512_storeu_si512((__m512i *)(ranks + i),
                            _mm512_mask_blend_epi8(hib, lo, hi));
    }
    for (; i < n; i++) ranks[i] = s2r[sym[i]];
    PROF_TOC(PROF_ENC_INIT, n);
}

/* full: both sides compacted (right -> tmp, left in place into ranks). */
static inline int part_full_avx512(uint8_t *ranks, int n, uint8_t thr,
                                      uint8_t *bm, uint8_t *tmp)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    __m512i vt = _mm512_set1_epi8((char)thr);
    for (; j + 64 <= n; j += 64) {
        __m512i v = _mm512_loadu_si512((const void *)(ranks + j));
        __mmask64 k = _mm512_cmpgt_epu8_mask(v, vt);
        int p = __builtin_popcountll(k);
        memcpy(bm + (j >> 3), &k, 8);
        /* mask_compress with v as pass-through, not maskz: Zen 4/5 false-dep
         * on the maskz destination (issue #11, as in the merges); the lanes
         * past popcount are dead either way -- the next store overwrites. */
        _mm512_storeu_si512((void *)(tmp + n_right),   _mm512_mask_compress_epi8(v, k, v));
        _mm512_storeu_si512((void *)(ranks + n_left), _mm512_mask_compress_epi8(v, ~k, v));
        n_right += p;
        n_left += 64 - p;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}

/* right (LEAF_LEFT): compact the right side only, to tmp. */
static inline int part_right_avx512(uint8_t *ranks, int n, uint8_t thr,
                                       uint8_t *bm, uint8_t *tmp)
{
    int n_right = 0, j = 0;
    __m512i vt = _mm512_set1_epi8((char)thr);
    for (; j + 64 <= n; j += 64) {
        __m512i v = _mm512_loadu_si512((const void *)(ranks + j));
        __mmask64 k = _mm512_cmpgt_epu8_mask(v, vt);
        memcpy(bm + (j >> 3), &k, 8);
        /* mask_compress into v: see part_full (issue #11 false dep) */
        _mm512_storeu_si512((void *)(tmp + n_right), _mm512_mask_compress_epi8(v, k, v));
        n_right += __builtin_popcountll(k);
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
    }
    return n_right;
}

/* none (BOTH_LEAVES): bitmap + right count only, no compaction. */
static inline int part_none_avx512(uint8_t *ranks, int n, uint8_t thr, uint8_t *bm)
{
    int n_right = 0, j = 0;
    __m512i vt = _mm512_set1_epi8((char)thr);
    for (; j + 64 <= n; j += 64) {
        __m512i v = _mm512_loadu_si512((const void *)(ranks + j));
        __mmask64 k = _mm512_cmpgt_epu8_mask(v, vt);
        memcpy(bm + (j >> 3), &k, 8);
        n_right += __builtin_popcountll(k);
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); n_right++; }
    }
    return n_right;
}

/* Flat pack, native u8: the local code (rank - base) is already a D-bit byte,
 * so we pack straight from u8 (no u16 widen + cvtepi16 narrow round-trip).
 * Per-D kernels in pivco_huffman_avx512_pack.h; scalar tail for the residual. */
static inline void pack_dN_avx512(uint8_t *out, const uint8_t *ranks,
                                     int n, int D, uint8_t base)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;

    int i = 0;
    switch (D) {
    case 2: i = pack_d2_avx512(out, ranks, n, base); break;
    case 3: i = pack_d3_avx512(out, ranks, n, base); break;
    case 4: i = pack_d4_avx512(out, ranks, n, base); break;
    case 5: i = pack_d5_avx512(out, ranks, n, base); break;
    case 6: i = pack_d6_avx512(out, ranks, n, base); break;
    case 7: i = pack_d7_avx512(out, ranks, n, base); break;
    case 8: i = pack_d8_avx512(out, ranks, n, base); break;
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
        while (bits_in_buf >= 8) { out[byte_idx++] = (uint8_t)buf; buf >>= 8; bits_in_buf -= 8; }
    }
    if (bits_in_buf > 0) out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
}

/* ---------- Aliases consumed by codec.c ---------- */

#define PIVCO_PRIM_ALWAYS_INLINE __attribute__((always_inline)) static inline

/* Widest load a merge kernel issues at a child-buffer cursor (8B expandloadu tail window);
 * the cursor can rest AT `size` on the exhausted side, so buffers a
 * merge reads need this much trailing slack.  Consumed by the decode
 * placement logic (scratch_carve / place_tail). */
#define PIVCO_PRIM_MERGE_OVERREAD 8

PIVCO_PRIM_ALWAYS_INLINE void prim_codec_init(void)
{ codec_init_avx512(); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_init(uint8_t *ranks, int n,
                                              const uint8_t *symbols,
                                              const uint8_t *sym_to_rank,
                                              const pivco_enc_init_aux_t *aux)
{ (void)aux; init_avx512(ranks, n, symbols, sym_to_rank); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_full(uint8_t *ranks,
                                                      int n, uint8_t thr,
                                                      uint8_t *bm,
                                                      uint8_t *right_out)
{ return part_full_avx512(ranks, n, thr, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_right(uint8_t *ranks,
                                                      int n, uint8_t thr,
                                                      uint8_t *bm,
                                                      uint8_t *right_out)
{ return part_right_avx512(ranks, n, thr, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int prim_enc_partition_none(uint8_t *ranks,
                                                     int n, uint8_t thr,
                                                     uint8_t *bm)
{ return part_none_avx512(ranks, n, thr, bm); }

PIVCO_PRIM_ALWAYS_INLINE void prim_enc_pack_dN(const uint8_t *ranks,
                                             int n, int D, uint8_t base,
                                             uint8_t *out_packed)
{ pack_dN_avx512(out_packed, ranks, n, D, base); }


PIVCO_PRIM_ALWAYS_INLINE void prim_merge_flat(uint8_t *out, int n,
                                                          const uint8_t *bm, int D,
                                                          const uint8_t *c2s)
{ merge_flat_avx512(out, n, bm, D, c2s); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_cst(const uint8_t *bm, int K,
                                                      uint8_t left_sym,
                                                      uint8_t right_sym,
                                                      uint8_t *out)
{ merge_cst_cst_avx512(bm, K, left_sym, right_sym, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_cst_vec(const uint8_t *bm, int K,
                                                          uint8_t left_sym,
                                                          const uint8_t *right_buf,
                                                          uint8_t *out)
{ merge_cst_vec_avx512(bm, K, left_sym, right_buf, out); }

PIVCO_PRIM_ALWAYS_INLINE void prim_merge_vec_vec(const uint8_t *bm, int K,
                                               const uint8_t *left_buf,
                                               const uint8_t *right_buf,
                                               uint8_t *out)
{ merge_vec_vec_avx512(bm, K, left_buf, right_buf, out); }


/* ---------------------------------------------------------------------------
 * prim_histogram_chunk — positional (bit-plane) byte histogram.
 *
 * Port of Harold Aptroot's AVX-512 histogram, used under the MIT
 * License (Copyright (c) 2026 Harold Aptroot):
 *   https://gitlab.com/-/snippets/3745720  (hist.cpp + license.txt)
 * Dissected in Jorn Engel's write-up:
 *   https://github.com/JoernEngel/joernblog/blob/master/histogram.md
 *
 * Bytes are binned by their top 2 bits into four 16 KB buffers
 * (masked vpcompressb), the remaining 6 bits rotated via one GF2P8
 * affine so 1-bit "positional" counters (one bit of each of 512
 * counters per zmm) can absorb them; vpternlog full-adder trees
 * promote 1-bit -> 3-bit -> 8-bit -> 16-bit -> 32-bit counters.
 * Input-shape independent by construction (~7.5 GB/s GNR, ~12-14 GB/s
 * Zen 5 with clang; gcc trails ~10-35%% on the FA chains).
 *
 * Needs GFNI + BITALG + VBMI2 on top of the backend baseline; without
 * the compile flags the shared scalar core is used instead.
 * ------------------------------------------------------------------------- */
#include "pivco_huffman_hist_scalar.h"

#if defined(__GFNI__) && defined(__AVX512BITALG__)

#define HIST_FA(hh, ll, a, b, c) do {                              \
    __m512i _l = _mm512_ternarylogic_epi32((c), (b), (a), 0x96);   \
    (hh) = _mm512_ternarylogic_epi32(_l, (b), (a), 0x8E);          \
    (ll) = _l;                                                     \
} while (0)

/* consume one bin: N bytes of 6-bit morsels -> 64 u16 counters */
static void hist_consume_bin_avx512(uint8_t *data, size_t N,
                                    uint16_t *hist16)
{
    size_t tail = N & 63;
    if (tail) {
        __m512i *where = (__m512i *)(data + N - tail);
        _mm512_store_epi64(where,
            _mm512_or_epi64(_mm512_load_epi64(where),
                            _mm512_movm_epi8(~0ull << tail)));
        N = N + 64 - tail;
    }
    N /= 64;

    __m512i h0 = _mm512_setzero_si512();
    __m512i h1 = _mm512_setzero_si512();
    __m512i w0_0 = _mm512_setzero_si512();
    __m512i w1_0 = _mm512_setzero_si512();
    __m512i w2_0 = _mm512_setzero_si512();

    static const uint8_t tp_bytes[64] __attribute__((aligned(64))) = {
        0, 8, 16, 24, 32, 40, 48, 56,
        1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,
        5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,
        7, 15, 23, 31, 39, 47, 55, 63};
    const __m512i tp = _mm512_load_si512((const void *)tp_bytes);

    do {
        size_t M = N > 31 ? 31 : N;
        N -= M;
        __m512i w = _mm512_setzero_si512();
        do {
            __m512i x0 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 0)));
            __m512i x1 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 8)));
            __m512i x2 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 16)));
            __m512i x3 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 24)));
            __m512i x4 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 32)));
            __m512i x5 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 40)));
            __m512i x6 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 48)));
            __m512i x7 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 56)));
            data += 64;

            HIST_FA(x1, x2, x0, x1, x2);
            HIST_FA(x4, x5, x3, x4, x5);
            HIST_FA(x7, w0_0, x6, x7, w0_0);
            HIST_FA(x5, w0_0, x2, x5, w0_0);
            HIST_FA(x4, x7, x1, x4, x7);
            HIST_FA(x5, w1_0, x7, x5, w1_0);
            __m512i w3_0;
            HIST_FA(w3_0, w2_0, x5, x4, w2_0);

            w3_0 = _mm512_permutexvar_epi8(tp, w3_0);
            w3_0 = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(0x8040201008040201), w3_0, 0);
            w3_0 = _mm512_popcnt_epi8(w3_0);
            w = _mm512_add_epi8(w, w3_0);
        } while (--M);

        h0 = _mm512_add_epi16(h0, _mm512_and_epi64(w, _mm512_set1_epi16(0xFF)));
        h1 = _mm512_add_epi16(h1, _mm512_srli_epi16(w, 8));
    } while (N);

    w0_0 = _mm512_permutexvar_epi8(tp, w0_0);
    w1_0 = _mm512_permutexvar_epi8(tp, w1_0);
    w2_0 = _mm512_permutexvar_epi8(tp, w2_0);
    w0_0 = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(0x8040201008040201), w0_0, 0);
    w1_0 = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(0x8040201008040201), w1_0, 0);
    w2_0 = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(0x8040201008040201), w2_0, 0);
    w0_0 = _mm512_popcnt_epi8(w0_0);
    w1_0 = _mm512_popcnt_epi8(w1_0);
    w2_0 = _mm512_popcnt_epi8(w2_0);

    __m512i w = _mm512_add_epi8(_mm512_add_epi8(w0_0, _mm512_add_epi8(w1_0, w1_0)),
                                _mm512_slli_epi64(w2_0, 2));
    h0 = _mm512_add_epi16(_mm512_slli_epi16(h0, 3), _mm512_and_epi64(w, _mm512_set1_epi16(0xFF)));
    h1 = _mm512_add_epi16(_mm512_slli_epi16(h1, 3), _mm512_srli_epi16(w, 8));

    _mm512_storeu_epi16(hist16, _mm512_add_epi16(h0, _mm512_loadu_epi16(hist16)));
    _mm512_storeu_epi16(hist16 + 32, _mm512_add_epi16(h1, _mm512_loadu_epi16(hist16 + 32)));
}

static inline void histogram_chunk_avx512(const uint8_t *in, size_t n,
                                          uint32_t hist[256], uint8_t *scratch)
{
    const uint8_t *ptr = in;
    size_t N = n;
    if (N >= 64) {
        const uint8_t *end = ptr + N;
        while ((uintptr_t)ptr & 63) hist[*ptr++] += 1;
        N = (size_t)(end - ptr);
    }

    const size_t bufsize = 16 * 1024;
    uint8_t *buffer0 = (uint8_t *)(((uintptr_t)scratch + 63) & ~(uintptr_t)63);
    uint8_t *buffer1 = buffer0 + bufsize;
    uint8_t *buffer2 = buffer1 + bufsize;
    uint8_t *buffer3 = buffer2 + bufsize;

    while (N >= 64) {
        uint16_t hist16[256] = {0};
        size_t count0 = 0, count1 = 0, count2 = 0, count3 = 0;
        /* consume up to 2^16-64 bytes per round: a u16 counter cannot
         * overflow within a round by construction */
        size_t M = N >= 65472 ? 65472 : (N & (size_t)-64);
        N -= M;
        for (size_t i = 0; i < M; i += 64) {
            __m512i data = _mm512_load_si512(ptr + i);
            __mmask64 bit7 = _mm512_movepi8_mask(data);
            __mmask64 bit6 = _mm512_movepi8_mask(_mm512_add_epi8(data, data));
            __mmask64 b00 = _knot_mask64(_kor_mask64(bit6, bit7));
            __mmask64 b01 = _kandn_mask64(bit7, bit6);
            __mmask64 b10 = _kandn_mask64(bit6, bit7);
            __mmask64 b11 = _kand_mask64(bit7, bit6);
            data = _mm512_gf2p8affine_epi64_epi8(data, _mm512_set1_epi64(0x2010010204080000), 0);
            _mm512_storeu_epi8(buffer0 + count0, _mm512_maskz_compress_epi8(b00, data));
            _mm512_storeu_epi8(buffer1 + count1, _mm512_maskz_compress_epi8(b01, data));
            _mm512_storeu_epi8(buffer2 + count2, _mm512_maskz_compress_epi8(b10, data));
            _mm512_storeu_epi8(buffer3 + count3, _mm512_maskz_compress_epi8(b11, data));
            count0 += (size_t)_mm_popcnt_u64(b00);
            count1 += (size_t)_mm_popcnt_u64(b01);
            count2 += (size_t)_mm_popcnt_u64(b10);
            count3 += (size_t)_mm_popcnt_u64(b11);

            if (count0 >= bufsize - 64) { hist_consume_bin_avx512(buffer0, count0, &hist16[0]);   count0 = 0; }
            if (count1 >= bufsize - 64) { hist_consume_bin_avx512(buffer1, count1, &hist16[64]);  count1 = 0; }
            if (count2 >= bufsize - 64) { hist_consume_bin_avx512(buffer2, count2, &hist16[128]); count2 = 0; }
            if (count3 >= bufsize - 64) { hist_consume_bin_avx512(buffer3, count3, &hist16[192]); count3 = 0; }
        }
        ptr += M;

        if (count0) hist_consume_bin_avx512(buffer0, count0, &hist16[0]);
        if (count1) hist_consume_bin_avx512(buffer1, count1, &hist16[64]);
        if (count2) hist_consume_bin_avx512(buffer2, count2, &hist16[128]);
        if (count3) hist_consume_bin_avx512(buffer3, count3, &hist16[192]);

        for (size_t i = 0; i < 256; i += 64) {
            __m512i h0 = _mm512_loadu_epi16(hist16 + i);
            __m512i h1 = _mm512_loadu_epi16(hist16 + i + 32);
            __m512i w0 = _mm512_and_epi32(h0, _mm512_set1_epi32(0xFFFF));
            __m512i w1 = _mm512_srli_epi32(h0, 16);
            __m512i w2 = _mm512_and_epi32(h1, _mm512_set1_epi32(0xFFFF));
            __m512i w3 = _mm512_srli_epi32(h1, 16);
            _mm512_storeu_epi32(hist + i,      _mm512_add_epi32(w0, _mm512_loadu_epi32(hist + i)));
            _mm512_storeu_epi32(hist + i + 16, _mm512_add_epi32(w1, _mm512_loadu_epi32(hist + i + 16)));
            _mm512_storeu_epi32(hist + i + 32, _mm512_add_epi32(w2, _mm512_loadu_epi32(hist + i + 32)));
            _mm512_storeu_epi32(hist + i + 48, _mm512_add_epi32(w3, _mm512_loadu_epi32(hist + i + 48)));
        }
    }

    while (N) hist[ptr[--N]] += 1;
}
PIVCO_PRIM_ALWAYS_INLINE void prim_histogram_chunk(const uint8_t *in, size_t n,
                                                   uint32_t hist[256],
                                                   uint8_t *scratch)
{ histogram_chunk_avx512(in, n, hist, scratch); }

#else  /* no GFNI/BITALG compile support: fall back to the scalar core */
PIVCO_PRIM_ALWAYS_INLINE void prim_histogram_chunk(const uint8_t *in, size_t n,
                                                   uint32_t hist[256],
                                                   uint8_t *scratch)
{ histogram_chunk_scalar(in, n, hist, scratch); }
#endif /* __GFNI__ && __AVX512BITALG__ */

#endif  /* PIVCO_HUFFMAN_PRIMITIVES_AVX512_H */
