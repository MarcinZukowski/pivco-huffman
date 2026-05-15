#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
#include <stdlib.h>
#include <string.h>

#ifdef PIVCO_HAS_SSE4
#include <smmintrin.h>  /* SSE4.1 */
#ifdef PIVCO_HAS_AVX2
#include <immintrin.h>  /* AVX2 */
#endif
#include "pivco_huffman_x86_flat.h"

/* ---------- SSE4.1 Compress Shuffle Table ----------
 *
 * Identical to the NEON version: for each 8-bit mask, a 16-byte
 * shuffle that packs selected uint16_t elements to the front.
 * pshufb (_mm_shuffle_epi8) is the x86 equivalent of NEON TBL.
 */
/* SIMD compress shuffle table + init function: storage and constructor
 * live in pivco_huffman_x86_tables.{c,h} so codec.c (compiled per-
 * backend) can share the same runtime tables.  See the header for the
 * (mask -> shuf, popcount) layout. */
#include "pivco_huffman_x86_tables.h"

/* Local alias to keep the existing call sites short. */
static inline void init_compress_table(void) { init_compress_table_x86(); }

/* Partition 8 uint16_t by an 8-bit mask using SSE4.1 pshufb.
   bit=1 → right_out, bit=0 → left_out.
   Source is loaded first, so left_out may overlap src (n_left <= j).
   Returns count of right (bit=1) elements. */
static inline int partition_8_sse(const uint16_t *src,
                                   uint8_t mask,
                                   uint16_t *left_out,
                                   uint16_t *right_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);

    /* Load both shuffle patterns from combined table (contiguous) */
    const uint8_t *tab = compress_tab[mask];
    __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));

    __m128i right = _mm_shuffle_epi8(data, shuf_r);
    __m128i left  = _mm_shuffle_epi8(data, shuf_l);

    int n_right = compress_popcnt[mask];

    _mm_storeu_si128((__m128i *)right_out, right);
    _mm_storeu_si128((__m128i *)left_out, left);

    return n_right;
}

/* ---------- Leaf scatter-write (SSE4.1) ---------- */

static inline void scatter_write_sse(uint8_t *symbols,
                                      const uint16_t *indices, int n,
                                      uint8_t sym)
{
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m128i idx = _mm_loadu_si128((const __m128i *)(indices + j));
        symbols[_mm_extract_epi16(idx, 0)] = sym;
        symbols[_mm_extract_epi16(idx, 1)] = sym;
        symbols[_mm_extract_epi16(idx, 2)] = sym;
        symbols[_mm_extract_epi16(idx, 3)] = sym;
        symbols[_mm_extract_epi16(idx, 4)] = sym;
        symbols[_mm_extract_epi16(idx, 5)] = sym;
        symbols[_mm_extract_epi16(idx, 6)] = sym;
        symbols[_mm_extract_epi16(idx, 7)] = sym;
    }
    for (; j < n; j++) {
        symbols[indices[j]] = sym;
    }
}

/* ---------- x86 Encode (Tree-Walk) ---------- */

/* Pack n values of D bits (D<=8 typical, up to 15) into out, LSB-first.
 * Each element's local code = codes[indices[i]] & ((1<<D)-1).
 * Used for the flat-subtree fast path.  Writes ceil(n*D/8) bytes. */
static inline void pack_D_bits_x86(uint8_t *out, int n, int D,
                                    const uint16_t *indices,
                                    const uint16_t *codes)
{
    uint32_t mask = (1u << D) - 1;
    uint64_t buf = 0;
    int bits_in_buf = 0;
    int byte_idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t local = (uint32_t)codes[indices[i]] & mask;
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

/* Extract D bits at bit position `bit_pos`.  D <= 16. */
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

/* Only D=4 gets a SIMD fast path on pure SSE4.1.
 *
 * - D=2, D=3, D=5, D=6, D=7 all require either per-byte variable shifts
 *   (AVX2's _mm_srlv_*) or vpmultishiftqb (AVX-512 VBMI2) to build the
 *   per-byte code values efficiently.  Without those, the unpack would
 *   need ~4-8 separate pshufb + immediate shifts + blends, which
 *   benchmarked slower than the scalar FLAT_UNPACK_SWITCH_IDX on AVX-512
 *   (where even vpmultishiftqb wasn't enough for D=3/5/6), so the
 *   SSE4.1 variant is definitely not viable.
 * - D=4 is the special case where the unpack is simple: duplicate +
 *   mask + single-immediate-shift + blend gives (b_i & 0x0F, b_i >> 4)
 *   per input byte without any variable-shift primitive.
 * - D=8 has 256-entry c2s, too big for pshufb; scalar LDR wins.
 *
 * For real-world Zen-3-style hosts stuck on SSE4.1, the IDEAS.md
 * "Zen-3 hybrid block decoder" fallback (per-table selection between
 * PIVCO and trad_huffman_decode_4s) is the right escape for
 * bell_* / proba02 / english / zipfian. */

/* flat_d4_unpack_x86 lives in pivco_huffman_x86_flat.h (shared with
 * bench/bench_micro.c). */

/* Decode n elements through a D-bit packed region + code_to_sym table,
 * scattering to symbols[indices[i]].  Same per-D specialised unpackers
 * as NEON; scalar-fast on x86. */
static inline void flat_decode_scatter_x86(uint8_t *symbols,
                                            const uint16_t *indices, int n,
                                            const uint8_t *bm, int D,
                                            const uint8_t *c2s)
{
    if (D == 4) {
        /* c2s has 16 entries — exactly fills a pshufb register. */
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_x86(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            symbols[indices[i     ]] = (uint8_t)_mm_extract_epi8(syms, 0);
            symbols[indices[i +  1]] = (uint8_t)_mm_extract_epi8(syms, 1);
            symbols[indices[i +  2]] = (uint8_t)_mm_extract_epi8(syms, 2);
            symbols[indices[i +  3]] = (uint8_t)_mm_extract_epi8(syms, 3);
            symbols[indices[i +  4]] = (uint8_t)_mm_extract_epi8(syms, 4);
            symbols[indices[i +  5]] = (uint8_t)_mm_extract_epi8(syms, 5);
            symbols[indices[i +  6]] = (uint8_t)_mm_extract_epi8(syms, 6);
            symbols[indices[i +  7]] = (uint8_t)_mm_extract_epi8(syms, 7);
            symbols[indices[i +  8]] = (uint8_t)_mm_extract_epi8(syms, 8);
            symbols[indices[i +  9]] = (uint8_t)_mm_extract_epi8(syms, 9);
            symbols[indices[i + 10]] = (uint8_t)_mm_extract_epi8(syms, 10);
            symbols[indices[i + 11]] = (uint8_t)_mm_extract_epi8(syms, 11);
            symbols[indices[i + 12]] = (uint8_t)_mm_extract_epi8(syms, 12);
            symbols[indices[i + 13]] = (uint8_t)_mm_extract_epi8(syms, 13);
            symbols[indices[i + 14]] = (uint8_t)_mm_extract_epi8(syms, 14);
            symbols[indices[i + 15]] = (uint8_t)_mm_extract_epi8(syms, 15);
        }
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[indices[i    ]] = c2s[b & 0x0F];
            symbols[indices[i + 1]] = c2s[b >> 4];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_x86(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
    int i = 0;
    switch (D) {
    case 2:
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            symbols[indices[i    ]] = c2s[(b     ) & 3];
            symbols[indices[i + 1]] = c2s[(b >> 2) & 3];
            symbols[indices[i + 2]] = c2s[(b >> 4) & 3];
            symbols[indices[i + 3]] = c2s[(b >> 6) & 3];
        }
        break;
    case 3:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 3) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            symbols[indices[i    ]] = c2s[(w      ) & 7];
            symbols[indices[i + 1]] = c2s[(w >>  3) & 7];
            symbols[indices[i + 2]] = c2s[(w >>  6) & 7];
            symbols[indices[i + 3]] = c2s[(w >>  9) & 7];
            symbols[indices[i + 4]] = c2s[(w >> 12) & 7];
            symbols[indices[i + 5]] = c2s[(w >> 15) & 7];
            symbols[indices[i + 6]] = c2s[(w >> 18) & 7];
            symbols[indices[i + 7]] = c2s[(w >> 21) & 7];
        }
        break;
    case 4:
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[indices[i    ]] = c2s[b & 0x0F];
            symbols[indices[i + 1]] = c2s[b >> 4];
        }
        break;
    case 5:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 5) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32);
            symbols[indices[i    ]] = c2s[(w      ) & 0x1F];
            symbols[indices[i + 1]] = c2s[(w >>  5) & 0x1F];
            symbols[indices[i + 2]] = c2s[(w >> 10) & 0x1F];
            symbols[indices[i + 3]] = c2s[(w >> 15) & 0x1F];
            symbols[indices[i + 4]] = c2s[(w >> 20) & 0x1F];
            symbols[indices[i + 5]] = c2s[(w >> 25) & 0x1F];
            symbols[indices[i + 6]] = c2s[(w >> 30) & 0x1F];
            symbols[indices[i + 7]] = c2s[(w >> 35) & 0x1F];
        }
        break;
    case 6:
        for (; i + 4 <= n; i += 4) {
            const uint8_t *p = bm + ((i * 6) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            symbols[indices[i    ]] = c2s[(w      ) & 0x3F];
            symbols[indices[i + 1]] = c2s[(w >>  6) & 0x3F];
            symbols[indices[i + 2]] = c2s[(w >> 12) & 0x3F];
            symbols[indices[i + 3]] = c2s[(w >> 18) & 0x3F];
        }
        break;
    case 7:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 7) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)
                       | ((uint64_t)p[6] << 48);
            symbols[indices[i    ]] = c2s[(w      ) & 0x7F];
            symbols[indices[i + 1]] = c2s[(w >>  7) & 0x7F];
            symbols[indices[i + 2]] = c2s[(w >> 14) & 0x7F];
            symbols[indices[i + 3]] = c2s[(w >> 21) & 0x7F];
            symbols[indices[i + 4]] = c2s[(w >> 28) & 0x7F];
            symbols[indices[i + 5]] = c2s[(w >> 35) & 0x7F];
            symbols[indices[i + 6]] = c2s[(w >> 42) & 0x7F];
            symbols[indices[i + 7]] = c2s[(w >> 49) & 0x7F];
        }
        break;
    case 8:
        for (; i < n; i++) symbols[indices[i]] = c2s[bm[i]];
        break;
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_x86(bm, i * D, D);
        symbols[indices[i]] = c2s[code];
    }
}

/* Same as flat_decode_scatter_x86 but writes to symbols[i] directly
 * (used for root-flat where indices are identity). */
static inline void flat_decode_direct_x86(uint8_t *symbols, int n,
                                           const uint8_t *bm, int D,
                                           const uint8_t *c2s)
{
    if (D == 4) {
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_x86(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
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
    int i = 0;
    switch (D) {
    case 2:
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            symbols[i    ] = c2s[(b     ) & 3];
            symbols[i + 1] = c2s[(b >> 2) & 3];
            symbols[i + 2] = c2s[(b >> 4) & 3];
            symbols[i + 3] = c2s[(b >> 6) & 3];
        }
        break;
    case 3:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 3) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            symbols[i    ] = c2s[(w      ) & 7];
            symbols[i + 1] = c2s[(w >>  3) & 7];
            symbols[i + 2] = c2s[(w >>  6) & 7];
            symbols[i + 3] = c2s[(w >>  9) & 7];
            symbols[i + 4] = c2s[(w >> 12) & 7];
            symbols[i + 5] = c2s[(w >> 15) & 7];
            symbols[i + 6] = c2s[(w >> 18) & 7];
            symbols[i + 7] = c2s[(w >> 21) & 7];
        }
        break;
    case 4:
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[i    ] = c2s[b & 0x0F];
            symbols[i + 1] = c2s[b >> 4];
        }
        break;
    case 5:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 5) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32);
            symbols[i    ] = c2s[(w      ) & 0x1F];
            symbols[i + 1] = c2s[(w >>  5) & 0x1F];
            symbols[i + 2] = c2s[(w >> 10) & 0x1F];
            symbols[i + 3] = c2s[(w >> 15) & 0x1F];
            symbols[i + 4] = c2s[(w >> 20) & 0x1F];
            symbols[i + 5] = c2s[(w >> 25) & 0x1F];
            symbols[i + 6] = c2s[(w >> 30) & 0x1F];
            symbols[i + 7] = c2s[(w >> 35) & 0x1F];
        }
        break;
    case 6:
        for (; i + 4 <= n; i += 4) {
            const uint8_t *p = bm + ((i * 6) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            symbols[i    ] = c2s[(w      ) & 0x3F];
            symbols[i + 1] = c2s[(w >>  6) & 0x3F];
            symbols[i + 2] = c2s[(w >> 12) & 0x3F];
            symbols[i + 3] = c2s[(w >> 18) & 0x3F];
        }
        break;
    case 7:
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
        break;
    case 8:
        for (; i < n; i++) symbols[i] = c2s[bm[i]];
        break;
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_x86(bm, i * D, D);
        symbols[i] = c2s[code];
    }
}

/* ============ Per-D SIMD bit-pack helpers ============
 *
 * Symmetric to the NEON pack_dN in pivco_huffman_neon.c and the
 * AVX-512 variants in pivco_huffman_avx512.c.  Each one extracts the
 * D-bit local code from codes_la and packs LSB-first into the byte
 * stream.  Overpacks to a stride boundary; caller must zero-pad
 * codes_la past n by 16+ entries.
 *
 * Two implementation tiers gated by PIVCO_HAS_AVX2:
 *   AVX2:  uses _mm256_sllv_epi64 for per-lane shifts (D=3,5,6,7)
 *          and AVX2-widened SSE primitives for D=2,4,8.
 *   SSE4.1: hand-rolled tricks (_mm_maddubs_epi16 weighted pair-add
 *          for D=2,4, _mm_mullo_epi32 multiply-as-shift for D=3, and
 *          scalar for D=5,6,7 since SSE has no uint64 per-lane shift).
 */

#ifdef PIVCO_HAS_AVX2
/* AVX2 D=3,5,6,7: 8 codes per iter via __m256i uint64 lanes (8 lanes
 * × max shift 7*7 = 49 < 64).  Same shape as the AVX-512 macro just
 * narrower.  Wait — __m256i has 4 uint64 lanes; AVX2's _mm256_sllv_
 * epi64 takes 4 lanes.  Use 2× __m256i to cover 8 codes per iter. */
#define PACK_DN_AVX2_UNIFIED(NAME, D_VAL, BITS_OUT)                            \
static inline int NAME(uint8_t *out, const uint16_t *codes_la,                 \
                       int n, int right_shift)                                  \
{                                                                              \
    static const int64_t shifts_lo[4] = {0,         D_VAL,   2*D_VAL, 3*D_VAL};\
    static const int64_t shifts_hi[4] = {4*D_VAL, 5*D_VAL, 6*D_VAL, 7*D_VAL};  \
    __m256i sl = _mm256_loadu_si256((const __m256i *)shifts_lo);               \
    __m256i sh = _mm256_loadu_si256((const __m256i *)shifts_hi);               \
    __m256i mask_vec = _mm256_set1_epi64x((1LL << D_VAL) - 1);                 \
    int i = 0;                                                                 \
    for (; i + 8 <= n; i += 8) {                                               \
        __m128i v16 = _mm_loadu_si128((const __m128i *)(codes_la + i));        \
        /* Lo half: codes 0..3 widened to uint64x4. */                         \
        __m256i lo = _mm256_cvtepu16_epi64(v16);                               \
        /* Hi half: codes 4..7 widened to uint64x4. */                         \
        __m256i hi = _mm256_cvtepu16_epi64(_mm_unpackhi_epi64(v16, v16));      \
        lo = _mm256_and_si256(_mm256_srli_epi64(lo, right_shift), mask_vec);   \
        hi = _mm256_and_si256(_mm256_srli_epi64(hi, right_shift), mask_vec);   \
        lo = _mm256_sllv_epi64(lo, sl);                                        \
        hi = _mm256_sllv_epi64(hi, sh);                                        \
        __m256i sum = _mm256_add_epi64(lo, hi);                                \
        /* Horizontal-add 4 uint64 lanes into one. */                          \
        __m128i s128 = _mm_add_epi64(_mm256_castsi256_si128(sum),              \
                                      _mm256_extracti128_si256(sum, 1));       \
        uint64_t packed = _mm_cvtsi128_si64(s128)                              \
                        + _mm_cvtsi128_si64(_mm_unpackhi_epi64(s128, s128));   \
        int bi = i * D_VAL / 8;                                                \
        memcpy(out + bi, &packed, (BITS_OUT + 7) / 8);                         \
    }                                                                          \
    return i;                                                                  \
}
PACK_DN_AVX2_UNIFIED(pack_d3_avx2, 3, 24)
PACK_DN_AVX2_UNIFIED(pack_d5_avx2, 5, 40)
PACK_DN_AVX2_UNIFIED(pack_d6_avx2, 6, 48)
PACK_DN_AVX2_UNIFIED(pack_d7_avx2, 7, 56)
#undef PACK_DN_AVX2_UNIFIED
#endif

/* SSE4.1 D=2: 16 codes -> 4 bytes.  Pack 4 codes per byte via
 * _mm_maddubs_epi16 weighted pair-add (weights {1,4,16,64}); narrow to
 * bytes; store 4 bytes. */
static inline int pack_d2_sse(uint8_t *out, const uint16_t *codes_la,
                              int n, int right_shift)
{
    /* Weights: [1, 4, 16, 64] repeated 4× as int8.  Note int8 max is
     * 127, so 64 fits (just). */
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
        __m128i bytes = _mm_packus_epi16(v0, v1);  /* 16 bytes, each 0..3 */
        /* step1[k] = bytes[2k]*w[2k] + bytes[2k+1]*w[2k+1].
         * Pairs: (b0*1 + b1*4), (b2*16 + b3*64), (b4*1 + b5*4), ... */
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        /* Sum adjacent int16 pairs to get final bytes. */
        __m128i step2 = _mm_hadd_epi16(step1, _mm_setzero_si128());
        /* step2 lanes 0..3 hold the 4 packed bytes (low 8 bits). */
        __m128i out_bytes = _mm_packus_epi16(step2, _mm_setzero_si128());
        uint32_t packed4 = (uint32_t)_mm_cvtsi128_si32(out_bytes);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

/* SSE4.1 D=4: 16 codes -> 8 bytes.  Pair (c[2k], c[2k+1]) into one
 * byte each via _mm_maddubs_epi16 with weights {1, 16}. */
static inline int pack_d4_sse(uint8_t *out, const uint16_t *codes_la,
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
        __m128i bytes = _mm_packus_epi16(v0, v1);  /* 16 bytes, each 0..15 */
        /* Pair-add: step1[k] (int16) = bytes[2k]*1 + bytes[2k+1]*16. */
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        /* Narrow to bytes (low 8 bits of each lane). */
        __m128i out_bytes = _mm_packus_epi16(step1, _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)(out + (i * 4 / 8)), out_bytes);
    }
    return i;
}

/* SSE4.1 D=8: 16 codes -> 16 bytes, byte-aligned.  Trivial. */
static inline int pack_d8_sse(uint8_t *out, const uint16_t *codes_la,
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
 * same per-lane left shift.  Same FastPFor horizontal-bitpacking
 * trick, applied to encode. */
static inline int pack_d3_sse(uint8_t *out, const uint16_t *codes_la,
                              int n, int right_shift)
{
    const __m128i mlo = _mm_setr_epi32(1, 8, 64, 512);
    const __m128i mhi = _mm_setr_epi32(4096, 32768, 262144, 2097152);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128((const __m128i *)(codes_la + i));
        v = _mm_srli_epi16(v, right_shift);
        v = _mm_and_si128(v, _mm_set1_epi16(0x7));
        /* Widen 8 uint16 lanes to 4+4 uint32 lanes. */
        __m128i vlo = _mm_unpacklo_epi16(v, _mm_setzero_si128());
        __m128i vhi = _mm_unpackhi_epi16(v, _mm_setzero_si128());
        /* Per-lane shift via multiply (no _mm_sllv_epi32 in SSE4.1). */
        vlo = _mm_mullo_epi32(vlo, mlo);
        vhi = _mm_mullo_epi32(vhi, mhi);
        /* Sum the two halves then horizontally reduce. */
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

/* Dense-codes pack with per-D SIMD dispatch.  D=2/4/8 use SSE4.1
 * directly; D=3/5/6/7 use AVX2 sllv where available, falling back to
 * scalar on SSE4.1-only hosts (D=3 has an SSE multiply-as-shift
 * version, D=5/6/7 stay scalar since SSE has no uint64 per-lane
 * shift). */
static inline void pack_D_bits_dense_x86(uint8_t *out, int n, int D,
                                          int depth,
                                          const uint16_t *codes_la)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int right_shift = 16 - depth - D;

    int i = 0;
    switch (D) {
    case 2: i = pack_d2_sse(out, codes_la, n, right_shift); break;
    case 4: i = pack_d4_sse(out, codes_la, n, right_shift); break;
    case 8: i = pack_d8_sse(out, codes_la, n, right_shift); break;
#ifdef PIVCO_HAS_AVX2
    case 3: i = pack_d3_avx2(out, codes_la, n, right_shift); break;
    case 5: i = pack_d5_avx2(out, codes_la, n, right_shift); break;
    case 6: i = pack_d6_avx2(out, codes_la, n, right_shift); break;
    case 7: i = pack_d7_avx2(out, codes_la, n, right_shift); break;
#else
    case 3: i = pack_d3_sse(out, codes_la, n, right_shift); break;
    /* D=5,6,7 fall through to scalar tail on SSE4.1-only hosts. */
#endif
    default: break;
    }
    if (i >= n) return;

    /* Scalar tail (D=5,6,7 on SSE-only, or D >= 9). */
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

/* Dense-codes mask build via the classic SSE movemask trick.
 *
 * code_vec holds 8 left-aligned 16-bit Huffman codes.  At tree depth d,
 * bit d of the original code is at position (15 - d) of code_la; we
 * shift LEFT by d to move that bit to position 15 (= sign bit of each
 * int16 lane).  _mm_packs_epi16 with signed saturation then collapses
 * each int16 lane to an int8 byte where bit 7 is the sign bit, and
 * _mm_movemask_epi8 reads bit 7 of each byte into an 8-bit (well,
 * 16-bit but we mask to low 8) bitmask -- the per-element bit slice
 * we want.
 *
 * Cost: vsll (1) + vpacksw (1) + vpmovmskb (1) = 3 SSE ops + 1 mask. */
static inline uint8_t enc_mask8_codes_la_sse(__m128i code_vec, __m128i shift_count)
{
    __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
    __m128i bytes   = _mm_packs_epi16(shifted, _mm_setzero_si128());
    return (uint8_t)_mm_movemask_epi8(bytes);
}

static void encode_node_x86(const pivco_huffman_table_t *table,
                              int16_t node_id,
                              uint16_t *codes_la, int n,
                              int depth,
                              uint8_t **out_ptr,
                              uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return; /* leaf */

    PROF_COUNT_ONLY(PROF_ENC_NODE_VISIT, n);

    /* Flat-subtree fast path: emit n*D packed bits instead of D levels
       of bitmaps.  Detected at build_table time. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        uint8_t *out = *out_ptr;
        PROF_TIC();
        pack_D_bits_dense_x86(out, n, D, depth, codes_la);
        PROF_TOC(PROF_ENC_FLAT, n);
        *out_ptr += total_bytes;
        return;
    }

    /* K_right header (2026-05-12 wire format). */
    int need_kr = kr_header_needed(table, node_id);
    uint8_t *kr_hdr = NULL;
    if (need_kr) {
        kr_hdr = *out_ptr;
        *out_ptr += KR_HEADER_BYTES;
    }

    /* Bitmap + partition.  Each iter loads 8 left-aligned codes, builds
     * the mask byte via the movemask trick, partitions the SAME register
     * through compress_tab[mask] into left/right halves, writes left
     * in-place over codes_la and right into tmp. */
    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    *out_ptr += nbytes;

    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);

    PROF_TIC();
    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        uint8_t mask = enc_mask8_codes_la_sse(code_vec, shift_count);
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
    /* Scalar tail.  Read codes into a temp first since the in-place
     * left write can overlap the read once we're below a full group. */
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
    PROF_TOC(PROF_ENC_NODE_FULL, n);

    if (need_kr) {
        kr_hdr[0] = (uint8_t)(n_right & 0xFF);
        kr_hdr[1] = (uint8_t)((n_right >> 8) & 0xFF);
    }

    encode_node_x86(table, node->left, codes_la, n_left,
                     depth + 1, out_ptr, tmp + n_right);
    encode_node_x86(table, node->right, tmp,      n_right,
                     depth + 1, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_x86(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    init_compress_table();
    PROF_COUNT_ONLY(PROF_ENC_ENTRY, PIVCO_BLOCK_SIZE);

    const int N = PIVCO_BLOCK_SIZE;

    /* Dense left-aligned codes; +16 slack matches the NEON encoder:
     * partition_8_sse's 16-byte store can write at n_left + 8. */
    uint16_t codes_la[PIVCO_BLOCK_SIZE + 16];
    PROF_TIC();
    for (int i = 0; i < N; i++) codes_la[i] = table->code_la[symbols[i]];
    PROF_TOC(PROF_ENC_INIT, N);

    /* See pivco_huffman_neon.c for tmp sizing rationale. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)malloc(tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;
    uint8_t *ptr = out;

    encode_node_x86(table, table->tree_root, codes_la, N,
                     0, &ptr, tmp);

    free(tmp);
    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- x86 Decode (Tree-Walk with SSE Partition) ---------- */

/* Half-partition helpers: extract only one side */
static inline int partition_8_sse_right(const uint16_t *src,
                                         uint8_t mask,
                                         uint16_t *right_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_r = _mm_load_si128((const __m128i *)compress_tab[mask]);
    _mm_storeu_si128((__m128i *)right_out, _mm_shuffle_epi8(data, shuf_r));
    return compress_popcnt[mask];
}

static inline int partition_8_sse_left(const uint16_t *src,
                                        uint8_t mask,
                                        uint16_t *left_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(compress_tab[mask] + 16));
    _mm_storeu_si128((__m128i *)left_out, _mm_shuffle_epi8(data, shuf_l));
    return 8 - compress_popcnt[mask];
}

static void decode_node_x86(const pivco_huffman_table_t *table,
                              int16_t node_id,
                              uint16_t *indices, int n,
                              uint8_t *symbols,
                              const uint8_t **in_ptr,
                              uint16_t *tmp,
                              int16_t skip_node)
{
    if (n == 0) return;
    if (node_id == skip_node) return;  /* prefilled by memset */

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        scatter_write_sse(symbols, indices, n, (uint8_t)node->symbol);
        return;
    }

    /* Flat-subtree fast path. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[node_id]];
        flat_decode_scatter_x86(symbols, indices, n, bm, D, c2s);
        return;
    }

    /* K_right header (2026-05-12 wire format): skip 2 bytes when this
     * node has any non-leaf child.  TD decoder doesn't use K_right. */
    if (kr_header_needed(table, node_id)) {
        *in_ptr += KR_HEADER_BYTES;
    }

    /* Read n code bits */
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

    /* Check children for stage fusion */
    const pivco_tree_node_t *left_child  = &table->tree[node->left];
    const pivco_tree_node_t *right_child = &table->tree[node->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf
        && node->left != skip_node && node->right != skip_node) {
        /* Both children are leaves (neither prefilled) — scatter directly */
        uint8_t syms[2] = {(uint8_t)left_child->symbol,
                           (uint8_t)right_child->symbol};
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            uint8_t mask = bm[j >> 3];
            __m128i idx = _mm_loadu_si128((const __m128i *)(indices + j));
            symbols[_mm_extract_epi16(idx, 0)] = syms[(mask >> 0) & 1];
            symbols[_mm_extract_epi16(idx, 1)] = syms[(mask >> 1) & 1];
            symbols[_mm_extract_epi16(idx, 2)] = syms[(mask >> 2) & 1];
            symbols[_mm_extract_epi16(idx, 3)] = syms[(mask >> 3) & 1];
            symbols[_mm_extract_epi16(idx, 4)] = syms[(mask >> 4) & 1];
            symbols[_mm_extract_epi16(idx, 5)] = syms[(mask >> 5) & 1];
            symbols[_mm_extract_epi16(idx, 6)] = syms[(mask >> 6) & 1];
            symbols[_mm_extract_epi16(idx, 7)] = syms[(mask >> 7) & 1];
        }
        for (; j < n; j++) {
            symbols[indices[j]] = syms[bitmap_get(bm, j)];
        }
        return;
    }

    if (left_leaf && node->left == skip_node) {
        /* Left is prefilled leaf — half-partition right only */
        int n_right = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8)
            n_right += partition_8_sse_right(indices + j, bm[j >> 3],
                                              tmp + n_right);
        /* Masked vector tail (1..7 elements): tmp is a separate buffer
         * (no in-place aliasing like node_full has), so masking out
         * invalid bm bits is safe.  Verified by test_roundtrip on all
         * NEON+SSE+AVX-512 hosts after re-enabling. */
        if (j < n) {
            int rem = n - j;
            uint8_t mask = bm[j >> 3] & (uint8_t)((1u << rem) - 1);
            n_right += partition_8_sse_right(indices + j, mask,
                                              tmp + n_right);
        }
        decode_node_x86(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right, skip_node);
    } else if (right_leaf && node->right == skip_node) {
        /* Right is prefilled leaf — half-partition left only */
        int n_left = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8)
            n_left += partition_8_sse_left(indices + j, bm[j >> 3],
                                            indices + n_left);
        /* Masked vector tail.  In-place to indices+n_left, but n_left
         * <= j and the SSE pshufb sequence loads source before storing
         * (load-then-shuffle-then-store), so no read-after-write hazard.
         * Trailing filler bytes from the 16B store land past the valid
         * left-side range. */
        if (j < n) {
            int rem = n - j;
            uint8_t mask = bm[j >> 3] | (uint8_t)~((1u << rem) - 1);
            n_left += partition_8_sse_left(indices + j, mask,
                                            indices + n_left);
        }
        decode_node_x86(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp, skip_node);
    } else {
        /* Full partition */
        int n_left = 0, n_right = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            uint8_t mask = bm[j >> 3];
            int nr = partition_8_sse(indices + j, mask,
                                      indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (8 - nr);
        }
        /* Masked vector tail.  partition_8_sse_left's filler bytes
         * land in 8-element padding gap; right-child tmp passed at
         * tmp+n_right+8 below.  See pivco_huffman_neon.c for the
         * full rationale. */
        if (j < n) {
            int rem = n - j;
            uint8_t valid = (uint8_t)((1u << rem) - 1);
            uint8_t mask_r = bm[j >> 3] & valid;
            uint8_t mask_l = bm[j >> 3] | (uint8_t)~valid;
            n_right += partition_8_sse_right(indices + j, mask_r, tmp + n_right);
            n_left  += partition_8_sse_left (indices + j, mask_l, indices + n_left);
        }

        /* +8 padding before right child's tmp - see decode_node_neon. */
        decode_node_x86(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp + n_right + 8, skip_node);
        decode_node_x86(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right + 8, skip_node);
    }
}

int pivco_huffman_decode_x86(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root is leaf — fill everything */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Root is a flat subtree (whole tree flat, D>=2) — write symbols[i]
       directly, no prefill, no indices[]. */
    if (table->flat_depth[table->tree_root] >= 2) {
        int D = table->flat_depth[table->tree_root];
        int total_bytes = (N * D + 7) >> 3;
        const uint8_t *bm = ptr;
        ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[table->tree_root]];
        flat_decode_direct_x86(symbols, N, bm, D, c2s);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* K_right header for root. */
    if (kr_header_needed(table, table->tree_root)) ptr += KR_HEADER_BYTES;
    /* Read root bitmap */
    int nbytes = bitmap_bytes(N);
    const uint8_t *bm = ptr;
    ptr += nbytes;

    const pivco_tree_node_t *left_child  = &table->tree[root->left];
    const pivco_tree_node_t *right_child = &table->tree[root->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf) {
        /* Both-leaves at root — sequential stores, no scatter.
         * Vectorized: 2 bitmap bytes -> 16 output bytes per iter.
         * Each bitmap byte broadcast to 8 lanes via pshufb, ANDed with
         * the bit-position table, then cmpeq to produce a 0xFF/0x00
         * byte-mask, then pblendvb selects sym0 / sym1. */
        uint8_t sym0 = (uint8_t)left_child->symbol;
        uint8_t sym1 = (uint8_t)right_child->symbol;
        __m128i vsym0 = _mm_set1_epi8((char)sym0);
        __m128i vsym1 = _mm_set1_epi8((char)sym1);
        __m128i bits  = _mm_setr_epi8(1,2,4,8,16,32,64,(char)128,
                                       1,2,4,8,16,32,64,(char)128);
        __m128i shuf  = _mm_setr_epi8(0,0,0,0,0,0,0,0,
                                       1,1,1,1,1,1,1,1);
        int j = 0;
        for (; j + 16 <= N; j += 16) {
            __m128i bm_pair = _mm_cvtsi32_si128(*(const uint16_t *)(bm + (j >> 3)));
            __m128i bm_dup  = _mm_shuffle_epi8(bm_pair, shuf);
            __m128i masked  = _mm_and_si128(bm_dup, bits);
            __m128i mask8   = _mm_cmpeq_epi8(masked, bits);
            __m128i out     = _mm_blendv_epi8(vsym0, vsym1, mask8);
            _mm_storeu_si128((__m128i *)(symbols + j), out);
        }
        for (; j < N; j++) {
            uint8_t syms[2] = {sym0, sym1};
            symbols[j] = syms[(bm[j >> 3] >> (j & 7)) & 1];
        }
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Prefill output with most frequent symbol */
    uint8_t prefill_sym = table->prefill_sym;
    int16_t skip_node = table->prefill_node;
    memset(symbols, prefill_sym, (size_t)N);

    /* Partition at root — generate identity indices in-place.
     * +8 padding on indices to absorb partition_8_sse_left's 16-byte
     * filler; 64B-aligned to keep cache-set layout deterministic.
     * See decode_node_neon comment. */
    uint16_t indices[PIVCO_BLOCK_SIZE + 8] __attribute__((aligned(64)));
    /* See pivco_huffman_neon.c comment -- heap-alloc for worst-case
     * skewed-partition offset accumulation. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)aligned_alloc(64, tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;

    if (left_leaf && root->left == skip_node) {
        /* Left is prefilled — half-partition right only at root */
        int n_right = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            /* Generate identity indices [j..j+7] and partition right */
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            n_right += partition_8_sse_right(id, bm[j >> 3], tmp + n_right);
        }
        decode_node_x86(table, root->right, tmp, n_right,
                         symbols, &ptr, tmp + n_right, skip_node);
    } else if (right_leaf && root->right == skip_node) {
        /* Right is prefilled — half-partition left only at root */
        int n_left = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            n_left += partition_8_sse_left(id, bm[j >> 3], indices + n_left);
        }
        decode_node_x86(table, root->left, indices, n_left,
                         symbols, &ptr, tmp, skip_node);
    } else {
        /* Full partition at root */
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            uint8_t mask = bm[j >> 3];
            int nr = partition_8_sse(id, mask,
                                      indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (8 - nr);
        }

        /* Recurse into both; child's entry handles leaf/skip_node.
         * +8 padding before right child's tmp - see decode_node_neon. */
        decode_node_x86(table, root->left, indices, n_left,
                         symbols, &ptr, tmp + n_right + 8, skip_node);
        decode_node_x86(table, root->right, tmp, n_right,
                         symbols, &ptr, tmp + n_right + 8, skip_node);
    }

    free(tmp);
    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

/* Non-static wrapper exposed for the bottom-up decoder
 * (src/pivco_huffman_bu_x86.c) so it can reuse the vectorised D=4
 * flat-decode without duplicating the unpacker. */
void pivco_huffman_flat_decode_direct_x86_(uint8_t *symbols, int n,
                                            const uint8_t *bm, int D,
                                            const uint8_t *c2s) {
    flat_decode_direct_x86(symbols, n, bm, D, c2s);
}

#endif /* PIVCO_HAS_SSE4 */
