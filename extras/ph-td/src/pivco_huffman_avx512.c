#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
#include <stdlib.h>
#include <string.h>

#ifdef PIVCO_HAS_AVX512
#include <immintrin.h>
#include "pivco_huffman_avx512_flat.h"

/* ---------- AVX-512 VBMI2 Partition ----------
 *
 * vpcompressw: compress selected uint16_t elements to the front
 * of a 512-bit register in ONE instruction. No shuffle table needed.
 * Processes 32 × uint16_t per iteration (4x the SSE/NEON path).
 */

/* Partition up to 32 uint16_t by a 32-bit mask.
   bit=1 → right_out, bit=0 → left_out.
   Returns count of right (bit=1) elements. */
static inline int partition_32(const uint16_t *src, int n,
                                __mmask32 mask,
                                uint16_t *left_out,
                                uint16_t *right_out)
{
    __m512i data = _mm512_loadu_si512((const __m512i *)src);

    /* Right (bit=1): compress selected elements to front */
    __m512i right = _mm512_maskz_compress_epi16(mask, data);
    int n_right = _mm_popcnt_u32((uint32_t)mask & ((1u << n) - 1));
    _mm512_storeu_si512((__m512i *)right_out, right);

    /* Left (bit=0): compress complement */
    __mmask32 inv = ~mask & (((__mmask32)1 << n) - 1);
    __m512i left = _mm512_maskz_compress_epi16(inv, data);
    _mm512_storeu_si512((__m512i *)left_out, left);

    return n_right;
}

/* Partition exactly 32 elements (fast path, no n masking needed) */
static inline int partition_32_full(const uint16_t *src,
                                     uint32_t mask,
                                     uint16_t *left_out,
                                     uint16_t *right_out)
{
    __m512i data = _mm512_loadu_si512((const __m512i *)src);

    __m512i right = _mm512_maskz_compress_epi16((__mmask32)mask, data);
    int n_right = _mm_popcnt_u32(mask);
    _mm512_storeu_si512((__m512i *)right_out, right);

    __m512i left = _mm512_maskz_compress_epi16((__mmask32)~mask, data);
    _mm512_storeu_si512((__m512i *)left_out, left);

    return n_right;
}

/* ---------- Leaf scatter-write (AVX-512) ---------- */

static inline void scatter_write_avx512(uint8_t *symbols,
                                         const uint16_t *indices, int n,
                                         uint8_t sym)
{
    /* AVX-512 has no byte-granularity scatter — `vpscatterdd` and friends
     * only operate on 32/64-bit elements.  An earlier SIMD-wrapper
     * (`_mm_loadu_si128 + 8× _mm_extract_epi16`) bottlenecked on port 5
     * (pextrw is single-issue) and ran 2-3× slower than the same shape
     * used by scatter_both_leaves_avx512 — which reads its indices as
     * a regular uint16_t array, letting the compiler issue independent
     * movzwl loads across multiple load ports and feed the store buffer
     * at 1 store/cycle.  Match that shape here.  Unroll-16 to amortize
     * loop overhead. */
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        symbols[indices[j +  0]] = sym;
        symbols[indices[j +  1]] = sym;
        symbols[indices[j +  2]] = sym;
        symbols[indices[j +  3]] = sym;
        symbols[indices[j +  4]] = sym;
        symbols[indices[j +  5]] = sym;
        symbols[indices[j +  6]] = sym;
        symbols[indices[j +  7]] = sym;
        symbols[indices[j +  8]] = sym;
        symbols[indices[j +  9]] = sym;
        symbols[indices[j + 10]] = sym;
        symbols[indices[j + 11]] = sym;
        symbols[indices[j + 12]] = sym;
        symbols[indices[j + 13]] = sym;
        symbols[indices[j + 14]] = sym;
        symbols[indices[j + 15]] = sym;
    }
    for (; j < n; j++) symbols[indices[j]] = sym;
}

/* ---------- AVX-512 Encode (Tree-Walk) ---------- */

/* ---------- Flat-subtree helpers (scalar; AVX-512 VBMI2 vectorisation
 * via vpmultishiftqb would be a follow-up). */

static inline void pack_D_bits_avx512(uint8_t *out, int n, int D,
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

/* flat_d{2..6}_unpack_avx512* helpers + tables live in
 * pivco_huffman_avx512_flat.h (shared with bench/bench_micro.c). */

#define FLAT_UNPACK_SWITCH_IDX(dst_expr)                                 \
    int i = 0;                                                            \
    switch (D) {                                                          \
    case 2:                                                               \
        for (; i + 4 <= n; i += 4) {                                      \
            uint8_t b = bm[i >> 2];                                       \
            dst_expr(i    ) = c2s[(b     ) & 3];                          \
            dst_expr(i + 1) = c2s[(b >> 2) & 3];                          \
            dst_expr(i + 2) = c2s[(b >> 4) & 3];                          \
            dst_expr(i + 3) = c2s[(b >> 6) & 3];                          \
        } break;                                                          \
    case 3:                                                               \
        for (; i + 8 <= n; i += 8) {                                      \
            const uint8_t *p = bm + ((i * 3) >> 3);                       \
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16); \
            dst_expr(i    ) = c2s[(w      ) & 7];                         \
            dst_expr(i + 1) = c2s[(w >>  3) & 7];                         \
            dst_expr(i + 2) = c2s[(w >>  6) & 7];                         \
            dst_expr(i + 3) = c2s[(w >>  9) & 7];                         \
            dst_expr(i + 4) = c2s[(w >> 12) & 7];                         \
            dst_expr(i + 5) = c2s[(w >> 15) & 7];                         \
            dst_expr(i + 6) = c2s[(w >> 18) & 7];                         \
            dst_expr(i + 7) = c2s[(w >> 21) & 7];                         \
        } break;                                                          \
    case 4:                                                               \
        for (; i + 2 <= n; i += 2) {                                      \
            uint8_t b = bm[i >> 1];                                       \
            dst_expr(i    ) = c2s[b & 0x0F];                              \
            dst_expr(i + 1) = c2s[b >> 4];                                \
        } break;                                                          \
    case 5:                                                               \
        for (; i + 8 <= n; i += 8) {                                      \
            const uint8_t *p = bm + ((i * 5) >> 3);                       \
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)           \
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)  \
                       | ((uint64_t)p[4] << 32);                          \
            dst_expr(i    ) = c2s[(w      ) & 0x1F];                      \
            dst_expr(i + 1) = c2s[(w >>  5) & 0x1F];                      \
            dst_expr(i + 2) = c2s[(w >> 10) & 0x1F];                      \
            dst_expr(i + 3) = c2s[(w >> 15) & 0x1F];                      \
            dst_expr(i + 4) = c2s[(w >> 20) & 0x1F];                      \
            dst_expr(i + 5) = c2s[(w >> 25) & 0x1F];                      \
            dst_expr(i + 6) = c2s[(w >> 30) & 0x1F];                      \
            dst_expr(i + 7) = c2s[(w >> 35) & 0x1F];                      \
        } break;                                                          \
    case 6:                                                               \
        for (; i + 4 <= n; i += 4) {                                      \
            const uint8_t *p = bm + ((i * 6) >> 3);                       \
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16); \
            dst_expr(i    ) = c2s[(w      ) & 0x3F];                      \
            dst_expr(i + 1) = c2s[(w >>  6) & 0x3F];                      \
            dst_expr(i + 2) = c2s[(w >> 12) & 0x3F];                      \
            dst_expr(i + 3) = c2s[(w >> 18) & 0x3F];                      \
        } break;                                                          \
    case 7:                                                               \
        for (; i + 8 <= n; i += 8) {                                      \
            const uint8_t *p = bm + ((i * 7) >> 3);                       \
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)           \
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)  \
                       | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)  \
                       | ((uint64_t)p[6] << 48);                          \
            dst_expr(i    ) = c2s[(w      ) & 0x7F];                      \
            dst_expr(i + 1) = c2s[(w >>  7) & 0x7F];                      \
            dst_expr(i + 2) = c2s[(w >> 14) & 0x7F];                      \
            dst_expr(i + 3) = c2s[(w >> 21) & 0x7F];                      \
            dst_expr(i + 4) = c2s[(w >> 28) & 0x7F];                      \
            dst_expr(i + 5) = c2s[(w >> 35) & 0x7F];                      \
            dst_expr(i + 6) = c2s[(w >> 42) & 0x7F];                      \
            dst_expr(i + 7) = c2s[(w >> 49) & 0x7F];                      \
        } break;                                                          \
    case 8:                                                               \
        for (; i < n; i++) dst_expr(i) = c2s[bm[i]];                      \
        break;                                                            \
    }                                                                      \
    for (; i < n; i++) {                                                   \
        uint32_t code = extract_D_bits_avx512(bm, i * D, D);               \
        dst_expr(i) = c2s[code];                                           \
    }

static inline void flat_decode_scatter_avx512(uint8_t *symbols,
                                               const uint16_t *indices, int n,
                                               const uint8_t *bm, int D,
                                               const uint8_t *c2s)
{
    if (D == 6) {
        /* c2s has 64 entries — fits in zmm, use vpermb. */
        __m512i c2s_vec = _mm512_loadu_si512((const __m512i *)c2s);
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3));
            __m512i codes_ext = _mm512_castsi128_si512(codes);
            __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_vec);
            __m128i syms = _mm512_castsi512_si128(syms_full);
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
        if (i + 16 <= n) {
            __m128i codes = flat_d6_unpack_avx512_safe(bm + ((i * 6) >> 3));
            __m512i codes_ext = _mm512_castsi128_si512(codes);
            __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_vec);
            __m128i syms = _mm512_castsi512_si128(syms_full);
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
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
    if (D == 5) {
        /* c2s has 32 entries — needs vpermb over ymm. */
        __m256i c2s_vec = _mm256_loadu_si256((const __m256i *)c2s);
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3));
            __m256i codes_ext = _mm256_zextsi128_si256(codes);
            __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_vec);
            __m128i syms = _mm256_castsi256_si128(syms_full);
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
        if (i + 16 <= n) {
            __m128i codes = flat_d5_unpack_avx512_safe(bm + ((i * 5) >> 3));
            __m256i codes_ext = _mm256_zextsi128_si256(codes);
            __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_vec);
            __m128i syms = _mm256_castsi256_si128(syms_full);
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
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
    if (D == 3) {
        /* c2s has 8 entries — fits in low 8 bytes of pshufb register.
         * Codes are masked 0..7, so only low 8 bytes are indexed. */
        uint64_t c2s_lo;
        memcpy(&c2s_lo, c2s, 8);
        __m128i c2s_vec = _mm_cvtsi64_si128((int64_t)c2s_lo);
        int i = 0;
        /* All but the last 16-code chunk: unsafe fast path (8-byte load
         * overreads into the NEXT chunk's valid bytes). */
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3));
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
        /* Final 16-code chunk (if any): safe 6-byte-memcpy variant. */
        if (i + 16 <= n) {
            __m128i codes = flat_d3_unpack_avx512_safe(bm + ((i * 3) >> 3));
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
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
    if (D == 4) {
        /* c2s has 16 entries — exactly fills a pshufb register. */
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_avx512(bm + (i >> 1));
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
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
    if (D == 2) {
        /* c2s has 4 entries; broadcast to all 128-bit lanes for pshufb. */
        uint32_t c2s_lo;
        memcpy(&c2s_lo, c2s, 4);
        __m128i c2s_vec = _mm_set1_epi32((int32_t)c2s_lo);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d2_unpack_avx512(bm + (i >> 2));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            /* 16 lane-extract + strbs.  Using _mm_extract_epi8 for
             * compile-time lanes. */
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
        /* Tail: scalar 4-wide, then 1-wide. */
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            symbols[indices[i    ]] = c2s[(b     ) & 3];
            symbols[indices[i + 1]] = c2s[(b >> 2) & 3];
            symbols[indices[i + 2]] = c2s[(b >> 4) & 3];
            symbols[indices[i + 3]] = c2s[(b >> 6) & 3];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
#define DST_SCATTER(k) symbols[indices[k]]
    FLAT_UNPACK_SWITCH_IDX(DST_SCATTER)
#undef DST_SCATTER
}

static inline void flat_decode_direct_avx512(uint8_t *symbols, int n,
                                              const uint8_t *bm, int D,
                                              const uint8_t *c2s)
{
    if (D == 6) {
        __m512i c2s_vec = _mm512_loadu_si512((const __m512i *)c2s);
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3));
            __m512i codes_ext = _mm512_castsi128_si512(codes);
            __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm512_castsi512_si128(syms_full));
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d6_unpack_avx512_safe(bm + ((i * 6) >> 3));
            __m512i codes_ext = _mm512_castsi128_si512(codes);
            __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm512_castsi512_si128(syms_full));
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 5) {
        __m256i c2s_vec = _mm256_loadu_si256((const __m256i *)c2s);
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3));
            __m256i codes_ext = _mm256_zextsi128_si256(codes);
            __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm256_castsi256_si128(syms_full));
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d5_unpack_avx512_safe(bm + ((i * 5) >> 3));
            __m256i codes_ext = _mm256_zextsi128_si256(codes);
            __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_vec);
            _mm_storeu_si128((__m128i *)(symbols + i),
                             _mm256_castsi256_si128(syms_full));
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 3) {
        uint64_t c2s_lo;
        memcpy(&c2s_lo, c2s, 8);
        __m128i c2s_vec = _mm_cvtsi64_si128((int64_t)c2s_lo);
        int i = 0;
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d3_unpack_avx512_safe(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
            i += 16;
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 4) {
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_avx512(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    if (D == 2) {
        /* Same unpack/lookup as scatter, but block-store 16 bytes. */
        uint32_t c2s_lo;
        memcpy(&c2s_lo, c2s, 4);
        __m128i c2s_vec = _mm_set1_epi32((int32_t)c2s_lo);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d2_unpack_avx512(bm + (i >> 2));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
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
            uint32_t code = extract_D_bits_avx512(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
#define DST_DIRECT(k) symbols[k]
    FLAT_UNPACK_SWITCH_IDX(DST_DIRECT)
#undef DST_DIRECT
}

/* Per-D SIMD bit-pack helpers -- symmetric to the NEON pack_dN in
 * pivco_huffman_neon.c.  Each one extracts the D-bit local code from
 * codes_la, shifts each lane k by k*D, and horizontally ORs the
 * results.  Overpacks to ceil(n / 8) * 8 elements (D=8 uses stride
 * 16); caller must zero-pad codes_la past n by 16+ entries. */

/* D=2..7: 8 codes per iter via uint64 lanes (max shift 7*7=49 < 64). */
#define PACK_DN_AVX512_UNIFIED(NAME, D_VAL, BITS_OUT)                          \
static inline int NAME(uint8_t *out, const uint16_t *codes_la,                 \
                       int n, int right_shift)                                  \
{                                                                              \
    static const int64_t shifts[8] = {                                         \
        0, D_VAL, 2*D_VAL, 3*D_VAL, 4*D_VAL, 5*D_VAL, 6*D_VAL, 7*D_VAL         \
    };                                                                         \
    __m512i shift_vec = _mm512_loadu_si512((const __m512i *)shifts);           \
    __m512i mask_vec  = _mm512_set1_epi64((1ULL << D_VAL) - 1);                \
    int i = 0;                                                                 \
    for (; i + 8 <= n; i += 8) {                                               \
        __m128i v16 = _mm_loadu_si128((const __m128i *)(codes_la + i));        \
        __m512i v64 = _mm512_cvtepu16_epi64(v16);                              \
        v64 = _mm512_srli_epi64(v64, right_shift);                             \
        v64 = _mm512_and_si512(v64, mask_vec);                                 \
        v64 = _mm512_sllv_epi64(v64, shift_vec);                               \
        uint64_t packed = _mm512_reduce_add_epi64(v64);                        \
        int bi = i * D_VAL / 8;                                                \
        memcpy(out + bi, &packed, (BITS_OUT + 7) / 8);                         \
    }                                                                          \
    return i;                                                                  \
}
PACK_DN_AVX512_UNIFIED(pack_d2_avx512, 2, 16)
PACK_DN_AVX512_UNIFIED(pack_d3_avx512, 3, 24)
PACK_DN_AVX512_UNIFIED(pack_d4_avx512, 4, 32)
PACK_DN_AVX512_UNIFIED(pack_d5_avx512, 5, 40)
PACK_DN_AVX512_UNIFIED(pack_d6_avx512, 6, 48)
PACK_DN_AVX512_UNIFIED(pack_d7_avx512, 7, 56)
#undef PACK_DN_AVX512_UNIFIED

/* D=8: byte-aligned.  32 codes / iter via vpmovqb-style narrow + store. */
static inline int pack_d8_avx512(uint8_t *out, const uint16_t *codes_la,
                                  int n, int right_shift)
{
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512i v = _mm512_loadu_si512((const __m512i *)(codes_la + i));
        v = _mm512_srli_epi16(v, right_shift);
        /* Narrow uint16 lanes to uint8 (drop high byte): vpmovwb. */
        __m256i bytes = _mm512_cvtepi16_epi8(v);
        _mm256_storeu_si256((__m256i *)(out + i), bytes);
    }
    return i;
}

/* Dense-codes pack with per-D SIMD dispatch.  Symmetric to the NEON
 * pack_D_bits_dense in pivco_huffman_neon.c. */
static inline void pack_D_bits_dense_avx512(uint8_t *out, int n, int D,
                                             int depth,
                                             const uint16_t *codes_la)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int right_shift = 16 - depth - D;

    int i = 0;
    switch (D) {
    case 2: i = pack_d2_avx512(out, codes_la, n, right_shift); break;
    case 3: i = pack_d3_avx512(out, codes_la, n, right_shift); break;
    case 4: i = pack_d4_avx512(out, codes_la, n, right_shift); break;
    case 5: i = pack_d5_avx512(out, codes_la, n, right_shift); break;
    case 6: i = pack_d6_avx512(out, codes_la, n, right_shift); break;
    case 7: i = pack_d7_avx512(out, codes_la, n, right_shift); break;
    case 8: i = pack_d8_avx512(out, codes_la, n, right_shift); break;
    default: break;
    }
    if (i >= n) return;

    /* Scalar tail (only fires for D >= 9, which shouldn't happen with
     * PIVCO_MAX_CODE_LEN = 11 and a leaf D = depth bound). */
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

/* AVX-512 dense-codes mask build for 32 codes per call.
 *
 * Load 32 uint16 codes into a __m512i (= 64 bytes).  Shift left by
 * depth so bit-d lands at int16 position 15 (= sign bit).
 * `_mm512_movepi16_mask` reads the sign bit of each int16 lane into a
 * 32-bit mask register — exact analog of the SSE `_mm_packs_epi16 +
 * _mm_movemask_epi8` trick, but native and a single instruction. */
static inline uint32_t enc_mask32_codes_la_avx512(__m512i code_vec, int depth)
{
    __m512i shifted = _mm512_slli_epi16(code_vec, depth);
    return (uint32_t)_mm512_movepi16_mask(shifted);
}

static void encode_node_avx512(const pivco_huffman_table_t *table,
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

    /* Flat-subtree fast path: emit n*D packed bits. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        uint8_t *out = *out_ptr;
        PROF_TIC();
        pack_D_bits_dense_avx512(out, n, D, depth, codes_la);
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

    /* FSE marker byte (v0.2 wire format).  Always 0 here since this
     * legacy avx512 encoder never attempts FSE coding; codec.c's
     * BU decoder always reads one marker byte per non-flat internal
     * node regardless.  This brings AVX-512 wire format into line
     * with scalar / NEON / x86 (all routed through codec.c after
     * Phase 4.3).  Phase 5 retires this encoder entirely.
     *
     * Sidebar: this byte was missing from x86 + AVX-512 encoders
     * relative to scalar+NEON until the codec cutover (see
     * pivco_huffman_wire.h comment).  Cross-backend decode would
     * miscount the stream by one byte per node and segfault on the
     * flat-decode path -- which is precisely what surfaced when
     * codec_x86 BU decode was wired in on c8i but encode still
     * came through this legacy file. */
    /* Marker byte deleted in the ph-td slice: the legacy AVX-512
     * decoder below does NOT read a marker byte, and the slice is
     * built without FSE, so the slot would be 0-padding only.
     * Keeping it would desync against this file's own decoder. */

    /* Bitmap + partition.  Stride 32 codes / iter:
     * - vpsllw(code_vec, depth) + vpmovw2m   -- 32-bit mask in one shot
     * - write the 32-bit mask to bm[j >> 3..j>>3 + 4)
     * - vpcompressw on the SAME register: left half (mask=0) in place over
     *   codes_la, right half (mask=1) into tmp. */
    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    *out_ptr += nbytes;

    int n_left = 0, n_right = 0;
    int j = 0;

    PROF_TIC();
    for (; j + 32 <= n; j += 32) {
        __m512i code_vec = _mm512_loadu_si512((const __m512i *)(codes_la + j));
        uint32_t mask = enc_mask32_codes_la_avx512(code_vec, depth);
        memcpy(bm + (j >> 3), &mask, 4);

        __m512i right_v = _mm512_maskz_compress_epi16((__mmask32)mask,  code_vec);
        __m512i left_v  = _mm512_maskz_compress_epi16((__mmask32)~mask, code_vec);
        _mm512_storeu_si512((__m512i *)(tmp      + n_right), right_v);
        _mm512_storeu_si512((__m512i *)(codes_la + n_left ), left_v);
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += (32 - nr);
    }
    /* SSE-stride remainder: 8 codes / iter via the same movemask trick. */
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted  = _mm_sll_epi16(code_vec, shift_count);
        __m128i bytes    = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask     = (uint8_t)_mm_movemask_epi8(bytes);
        bm[j >> 3] = mask;

        __m128i right_v = _mm_maskz_compress_epi16((__mmask8)mask,  code_vec);
        __m128i left_v  = _mm_maskz_compress_epi16((__mmask8)~mask, code_vec);
        _mm_storeu_si128((__m128i *)(tmp      + n_right), right_v);
        _mm_storeu_si128((__m128i *)(codes_la + n_left ), left_v);
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += (8 - nr);
    }
    /* Scalar tail. */
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

    encode_node_avx512(table, node->left,  codes_la, n_left,
                        depth + 1, out_ptr, tmp + n_right);
    encode_node_avx512(table, node->right, tmp,      n_right,
                        depth + 1, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_avx512(const uint8_t *symbols,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;
    PROF_COUNT_ONLY(PROF_ENC_ENTRY, PIVCO_BLOCK_SIZE);

    const int N = PIVCO_BLOCK_SIZE;

    /* Dense left-aligned codes; +32 slack covers the AVX-512 stride-32
     * partition's 64-byte vpcompressw store at n_left + 32 worst case. */
    uint16_t codes_la[PIVCO_BLOCK_SIZE + 32];

    /* ===== AVX-512 enc_init via byte-split vpermex2var_epi8 =====
     *
     * Refinement of the original vpermi2w (uint16) version.  Same idea
     * -- 4 parallel chunked table lookups + blend by char's top bits --
     * but split each uint16 entry into its low byte and high byte,
     * doing the lookups on BYTE permutes instead of uint16 permutes.
     *
     * Why this wins on AVX-512 VBMI:
     *
     *  1. vpermex2var_epi8 covers 128 entries per call (64 indices ×
     *     2-source × 64 bytes) vs vpermex2var_epi16's 64 entries.  So
     *     the 256-byte lo_table needs only 2 chunks (1 mask blend)
     *     instead of the uint16 path's 4 chunks (3 mask blends).
     *
     *  2. The chunk-selector mask is char >> 7 -- exactly bit 7 of each
     *     char byte -- so `_mm512_movepi8_mask(chars)` produces it in
     *     one instruction.  The uint16 path needs vpsrlw + 3× vpcmpeqw
     *     to compute three comparisons against {1, 2, 3}.
     *
     *  3. 64 chars / iter (vs 32 in the uint16 path) -- one ZMM load of
     *     symbols, one ZMM produced per byte half.
     *
     * Per 64 input chars:
     *    1× vmovdqu64               (load 64 chars)
     *    1× vpmovb2m                (top bit -> 64-bit chunk-selector mask)
     *    4× vpermex2var_epi8        (2× lo lookup + 2× hi lookup, 1 per chunk)
     *    2× vpblendmb               (1 for lo, 1 for hi)
     *    2× vpermex2var_epi8        (interleave lo+hi into 2× uint16 streams)
     *    2× vmovdqu64               (store 64 uint16 codes_la)
     *  = 12 ops / 64 chars = 0.19 ops/char
     *
     * vs the old uint16 path: 13 ops / 32 chars = 0.41 ops/char.
     * Theoretical 2.2x throughput.
     *
     * Lookup geometry (lo byte table; same for hi):
     *   chunk 0 (chars [  0, 128)):  (lo_c0p1, lo_c0p2) hold lo bytes
     *                                for entries [0, 63] and [64, 127].
     *   chunk 1 (chars [128, 256)):  (lo_c1p1, lo_c1p2) hold lo bytes
     *                                for entries [128, 191] and [192, 255].
     *   vpermex2var_epi8 uses low 7 bits of each index, so a char value
     *   c naturally selects entry (c & 0x7F) within whichever chunk it
     *   belongs to.  For c < 128 the chunk-0 lookup is correct; for
     *   c >= 128 the chunk-1 lookup is.  vpmovb2m(chars) gives the
     *   per-lane chunk selector directly.
     *
     * The two byte halves are then interleaved into the sequential
     * uint16 stream by two more vpermex2var_epi8 calls with constant
     * selector tables (inter_sel0 produces uint16 codes for chars
     * 0..31, inter_sel1 for chars 32..63).  This avoids the in-lane
     * vpunpcklbw / vpunpckhbw limitation (those don't span 128-bit
     * lanes, so they'd put codes out of order across the ZMM).
     *
     * Register pressure: 4 lo + 4 hi byte-table chunks + 2 interleave
     * selectors + ~5 working = ~15 ZMM, fine in the 32-ZMM AVX-512 file.
     *
     * Requires AVX-512 VBMI for vpermex2var_epi8 (gated by build tier:
     * c8a Zen 5, c8i Granite Rapids).  Falls back to the uint16 path
     * for the (currently impossible) n < 64 tail. */

    /* --- Build lo/hi byte tables from the uint16 code_la table. --- *
     * Each byte-table chunk holds 64 sequential entries' lo (or hi)
     * bytes; two chunks per byte-half pair the chars [0,128) and
     * [128,256) regions for vpermex2var_epi8. */
    __m512i u0 = _mm512_loadu_si512((const __m512i *)&table->code_la[  0]);
    __m512i u1 = _mm512_loadu_si512((const __m512i *)&table->code_la[ 32]);
    __m512i u2 = _mm512_loadu_si512((const __m512i *)&table->code_la[ 64]);
    __m512i u3 = _mm512_loadu_si512((const __m512i *)&table->code_la[ 96]);
    __m512i u4 = _mm512_loadu_si512((const __m512i *)&table->code_la[128]);
    __m512i u5 = _mm512_loadu_si512((const __m512i *)&table->code_la[160]);
    __m512i u6 = _mm512_loadu_si512((const __m512i *)&table->code_la[192]);
    __m512i u7 = _mm512_loadu_si512((const __m512i *)&table->code_la[224]);

    /* vpmovwb: narrow each __m512i of 32 uint16 to a __m256i of 32
     * uint8 (low byte only). */
    __m512i lo_c0p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u0)),
        _mm512_cvtepi16_epi8(u1), 1);     /* entries 0..63 lo bytes */
    __m512i lo_c0p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u2)),
        _mm512_cvtepi16_epi8(u3), 1);     /* entries 64..127 lo bytes */
    __m512i lo_c1p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u4)),
        _mm512_cvtepi16_epi8(u5), 1);     /* entries 128..191 lo bytes */
    __m512i lo_c1p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u6)),
        _mm512_cvtepi16_epi8(u7), 1);     /* entries 192..255 lo bytes */

    /* Hi bytes: shift right by 8 first to put the hi byte at low. */
    __m512i hi_c0p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u0, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u1, 8)), 1);
    __m512i hi_c0p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u2, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u3, 8)), 1);
    __m512i hi_c1p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u4, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u5, 8)), 1);
    __m512i hi_c1p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u6, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u7, 8)), 1);

    /* Interleave selectors.  vpermex2var_epi8 takes 64 byte indices and
     * picks from two 64-byte sources; bit 6 of the index selects which
     * source (0 = first / lo, 0x40 = second / hi).  We want the output
     * to be the sequential uint16 stream for chars 0..31 (resp. 32..63):
     *   out_byte[2k]   = lo[k_offset + k]       (low byte of uint16 k)
     *   out_byte[2k+1] = hi[k_offset + k]       (high byte of uint16 k)
     */
    static const uint8_t inter_sel0_tab[64] __attribute__((aligned(64))) = {
         0, 64,  1, 65,  2, 66,  3, 67,  4, 68,  5, 69,  6, 70,  7, 71,
         8, 72,  9, 73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79,
        16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87,
        24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95
    };
    static const uint8_t inter_sel1_tab[64] __attribute__((aligned(64))) = {
        32, 96, 33, 97, 34, 98, 35, 99, 36,100, 37,101, 38,102, 39,103,
        40,104, 41,105, 42,106, 43,107, 44,108, 45,109, 46,110, 47,111,
        48,112, 49,113, 50,114, 51,115, 52,116, 53,117, 54,118, 55,119,
        56,120, 57,121, 58,122, 59,123, 60,124, 61,125, 62,126, 63,127
    };
    __m512i sel0 = _mm512_load_si512((const __m512i *)inter_sel0_tab);
    __m512i sel1 = _mm512_load_si512((const __m512i *)inter_sel1_tab);

    PROF_TIC();
    int i = 0;
    for (; i + 64 <= N; i += 64) {
        /* Load 64 chars in one shot. */
        __m512i chars = _mm512_loadu_si512((const __m512i *)(symbols + i));
        /* Top bit = chunk selector (0 if char in [0,128), 1 if [128,256)). */
        __mmask64 hi_chunk = _mm512_movepi8_mask(chars);

        /* Lo-byte lookup against both chunks; blend by chunk selector. */
        __m512i lo0 = _mm512_permutex2var_epi8(lo_c0p1, chars, lo_c0p2);
        __m512i lo1 = _mm512_permutex2var_epi8(lo_c1p1, chars, lo_c1p2);
        __m512i lo  = _mm512_mask_blend_epi8(hi_chunk, lo0, lo1);

        /* Hi-byte lookup symmetric. */
        __m512i hi0 = _mm512_permutex2var_epi8(hi_c0p1, chars, hi_c0p2);
        __m512i hi1 = _mm512_permutex2var_epi8(hi_c1p1, chars, hi_c1p2);
        __m512i hi  = _mm512_mask_blend_epi8(hi_chunk, hi0, hi1);

        /* Interleave lo + hi into sequential uint16 stream. */
        __m512i out0 = _mm512_permutex2var_epi8(lo, sel0, hi);  /* chars 0..31 */
        __m512i out1 = _mm512_permutex2var_epi8(lo, sel1, hi);  /* chars 32..63 */

        _mm512_storeu_si512((__m512i *)(codes_la + i     ), out0);
        _mm512_storeu_si512((__m512i *)(codes_la + i + 32), out1);
    }
    /* Scalar tail.  PIVCO_BLOCK_SIZE is always a multiple of 64 on
     * AVX-512 hosts, so this is currently dead code -- kept defensively. */
    for (; i < N; i++) codes_la[i] = table->code_la[symbols[i]];
    PROF_TOC(PROF_ENC_INIT, N);

    /* See pivco_huffman_neon.c for tmp sizing rationale -- skewed
     * partitions can accumulate offset up to max_tree_depth × N. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)malloc(tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;
    uint8_t *ptr = out;

    encode_node_avx512(table, table->tree_root, codes_la, N,
                        0, &ptr, tmp);

    free(tmp);
    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- AVX-512 Decode (Tree-Walk) ---------- */

/* Half-partition: only right (bit=1) side */
static inline int partition_32_right(const uint16_t *src,
                                      uint32_t mask,
                                      uint16_t *right_out)
{
    __m512i data = _mm512_loadu_si512((const __m512i *)src);
    __m512i right = _mm512_maskz_compress_epi16((__mmask32)mask, data);
    _mm512_storeu_si512((__m512i *)right_out, right);
    return _mm_popcnt_u32(mask);
}

/* Half-partition: only left (bit=0) side */
static inline int partition_32_left(const uint16_t *src,
                                     uint32_t mask,
                                     uint16_t *left_out)
{
    __m512i data = _mm512_loadu_si512((const __m512i *)src);
    __m512i left = _mm512_maskz_compress_epi16((__mmask32)~mask, data);
    _mm512_storeu_si512((__m512i *)left_out, left);
    return 32 - _mm_popcnt_u32(mask);
}

/* Both children are leaves: scatter sym0 (bit=0) or sym1 (bit=1) to each
   index position, selecting via byte-blend from the bitmap.
   AVX-512 has no byte scatter, so the actual stores are scalar; the SIMD
   blend at least lets the compiler keep symbol selection in registers. */
static inline void scatter_both_leaves_avx512(uint8_t *symbols,
                                               const uint16_t *indices, int n,
                                               const uint8_t *bm,
                                               uint8_t sym0, uint8_t sym1)
{
    /* SIMD 16 at a time: expand 16 bitmap bits to a 0x00/delta byte
     * vector via vpmovm2b-style maskz_set1, XOR with sym0 broadcast.
     * Scattered writes still go through scalar stores (AVX-512 has
     * no byte-granularity vector scatter), so 16 wide is the sweet
     * spot -- enough to amortize the broadcast+xor over a single
     * 16-byte spill, but no wider since the stores dominate. */
    const uint8_t delta = (uint8_t)(sym0 ^ sym1);
    const __m128i vsym0  = _mm_set1_epi8((char)sym0);
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        uint16_t bm16 = (uint16_t)bm[j >> 3]
                      | ((uint16_t)bm[(j >> 3) + 1] << 8);
        __mmask16 m = (__mmask16)bm16;
        __m128i delta_or_zero = _mm_maskz_set1_epi8(m, (char)delta);
        __m128i vals = _mm_xor_si128(vsym0, delta_or_zero);
        uint8_t vals_arr[16] __attribute__((aligned(16)));
        _mm_store_si128((__m128i *)vals_arr, vals);
        symbols[indices[j     ]] = vals_arr[ 0];
        symbols[indices[j +  1]] = vals_arr[ 1];
        symbols[indices[j +  2]] = vals_arr[ 2];
        symbols[indices[j +  3]] = vals_arr[ 3];
        symbols[indices[j +  4]] = vals_arr[ 4];
        symbols[indices[j +  5]] = vals_arr[ 5];
        symbols[indices[j +  6]] = vals_arr[ 6];
        symbols[indices[j +  7]] = vals_arr[ 7];
        symbols[indices[j +  8]] = vals_arr[ 8];
        symbols[indices[j +  9]] = vals_arr[ 9];
        symbols[indices[j + 10]] = vals_arr[10];
        symbols[indices[j + 11]] = vals_arr[11];
        symbols[indices[j + 12]] = vals_arr[12];
        symbols[indices[j + 13]] = vals_arr[13];
        symbols[indices[j + 14]] = vals_arr[14];
        symbols[indices[j + 15]] = vals_arr[15];
    }
    for (; j < n; j++) {
        uint8_t bit = (uint8_t)((bm[j >> 3] >> (j & 7)) & 1);
        symbols[indices[j]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-(int8_t)bit));
    }
}

/* ---------- Per-call-site partition loops (AVX-512 interior) ----------
 * Each loop is its own static function with a PROF_TIC/TOC pair so the
 * profiler can attribute time precisely.  Mirrors the NEON layout. */

static inline void node_full_avx512(uint16_t *indices, int n,
                                     const uint8_t *bm,
                                     uint16_t *tmp,
                                     int *n_left_out, int *n_right_out)
{
    PROF_TIC();
    int n_left = 0, n_right = 0;
    int j = 0;
    for (; j + 32 <= n; j += 32) {
        uint32_t mask;
        memcpy(&mask, bm + (j >> 3), 4);
        int nr = partition_32_full(indices + j, mask,
                                    indices + n_left, tmp + n_right);
        n_right += nr;
        n_left  += (32 - nr);
    }
    /* Masked vector tail (1..31 leftover elements).  Bug from b136b96
     * was the same indices/tmp aliasing pattern as the NEON case (see
     * pivco_huffman_neon.c for full diagnosis).  Fix: caller passes
     * right child's tmp at tmp+n_right+32 (= 1 vector wide of padding)
     * so partition_32's filler bytes harmlessly land in the gap. */
    if (j < n) {
        int rem = n - j;
        uint32_t mask = 0;
        memcpy(&mask, bm + (j >> 3), (size_t)bitmap_bytes(rem));
        mask &= (1u << rem) - 1;
        int nr = partition_32(indices + j, rem, (__mmask32)mask,
                              indices + n_left, tmp + n_right);
        n_right += nr;
        n_left  += (rem - nr);
    }
    *n_left_out  = n_left;
    *n_right_out = n_right;
    PROF_TOC(PROF_NODE_FULL, n);
}

static inline int node_half_right_avx512(uint16_t *indices, int n,
                                          const uint8_t *bm,
                                          uint16_t *tmp_right_out)
{
    PROF_TIC();
    int n_right = 0;
    int j = 0;
    for (; j + 32 <= n; j += 32) {
        uint32_t mask;
        memcpy(&mask, bm + (j >> 3), 4);
        n_right += partition_32_right(indices + j, mask,
                                       tmp_right_out + n_right);
    }
    /* Masked vector tail (1..31 elements): tmp_right_out is a separate
     * buffer (no in-place aliasing), so masking out invalid bm bits is
     * safe.  See node_half_right in pivco_huffman_neon.c for the full
     * argument; same logic applies. */
    if (j < n) {
        int rem = n - j;
        uint32_t mask = 0;
        memcpy(&mask, bm + (j >> 3), (size_t)bitmap_bytes(rem));
        mask &= (1u << rem) - 1;
        n_right += partition_32_right(indices + j, mask,
                                       tmp_right_out + n_right);
    }
    PROF_TOC(PROF_NODE_HALF_RIGHT, n);
    return n_right;
}

/* Root-level half partitions: skip the indices[] init and generate
 * identity in-loop, mirroring root_half_{right,left} in
 * pivco_huffman_neon.c.  Used when the root is HALF_RIGHT / HALF_LEFT
 * (e.g. proba80: dominant symbol is at depth 1 -> root.left is the
 * SKIP leaf, so root only emits right-side indices for recursion). */

static inline int root_half_right_avx512(int N, const uint8_t *bm,
                                           uint16_t *tmp_right_out)
{
    PROF_TIC();
    int n_right = 0;
    int j = 0;
    uint16_t id_buf[32];
    for (; j + 32 <= N; j += 32) {
        uint32_t mask;
        memcpy(&mask, bm + (j >> 3), 4);
        for (int k = 0; k < 32; k++) id_buf[k] = (uint16_t)(j + k);
        n_right += partition_32_right(id_buf, mask,
                                        tmp_right_out + n_right);
    }
    for (; j < N; j++) {
        if (bitmap_get(bm, j))
            tmp_right_out[n_right++] = (uint16_t)j;
    }
    PROF_TOC(PROF_ROOT_HALF_RIGHT, N);
    return n_right;
}

static inline int root_half_left_avx512(int N, const uint8_t *bm,
                                          uint16_t *indices_left_out)
{
    PROF_TIC();
    int n_left = 0;
    int j = 0;
    uint16_t id_buf[32];
    for (; j + 32 <= N; j += 32) {
        uint32_t mask;
        memcpy(&mask, bm + (j >> 3), 4);
        for (int k = 0; k < 32; k++) id_buf[k] = (uint16_t)(j + k);
        n_left += partition_32_left(id_buf, mask,
                                      indices_left_out + n_left);
    }
    for (; j < N; j++) {
        if (!bitmap_get(bm, j))
            indices_left_out[n_left++] = (uint16_t)j;
    }
    PROF_TOC(PROF_ROOT_HALF_LEFT, N);
    return n_left;
}

static inline int node_half_left_avx512(uint16_t *indices, int n,
                                         const uint8_t *bm)
{
    PROF_TIC();
    int n_left = 0;
    int j = 0;
    for (; j + 32 <= n; j += 32) {
        uint32_t mask;
        memcpy(&mask, bm + (j >> 3), 4);
        n_left += partition_32_left(indices + j, mask,
                                     indices + n_left);
    }
    /* Masked vector tail.  In-place to indices+n_left, but n_left <= j
     * and partition_32_left loads source before the store, so no RAW
     * hazard.  See node_half_left in pivco_huffman_neon.c for full
     * argument. */
    if (j < n) {
        int rem = n - j;
        uint32_t mask = 0;
        memcpy(&mask, bm + (j >> 3), (size_t)bitmap_bytes(rem));
        mask |= ~((1u << rem) - 1);
        n_left += partition_32_left(indices + j, mask,
                                     indices + n_left);
    }
    PROF_TOC(PROF_NODE_HALF_LEFT, n);
    return n_left;
}

static void decode_node_avx512(const pivco_huffman_table_t *table,
                                int16_t node_id,
                                uint16_t *indices, int n,
                                uint8_t *symbols,
                                const uint8_t **in_ptr,
                                uint16_t *tmp,
                                int16_t skip_node)
{
    if (n == 0) return;
    PROF_COUNT_ONLY(PROF_DECODE_NODE, n);

    const pivco_tree_node_t *node = &table->tree[node_id];

    /* Single dispatch on pre-classified node type — see pivco_node_type_t.
     * Replaces the chain of skip_node/leaf/flat/both-leaves/half-prefilled
     * checks that used to be re-evaluated per call. */
    (void)skip_node;
    switch ((pivco_node_type_t)table->node_type[node_id]) {
    case PIVCO_NODE_SKIP:
        return;

    case PIVCO_NODE_LEAF: {
        PROF_TIC();
        scatter_write_avx512(symbols, indices, n, (uint8_t)node->symbol);
        PROF_TOC(PROF_SCATTER_SYM, n);
        return;
    }

    case PIVCO_NODE_INTERNAL_FLAT: {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[node_id]];
        PROF_TIC();
        flat_decode_scatter_avx512(symbols, indices, n, bm, D, c2s);
        PROF_TOC(PROF_FLAT_DECODE_SCATTER, n);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        int nbytes = bitmap_bytes(n);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        const pivco_tree_node_t *left_child  = &table->tree[node->left];
        const pivco_tree_node_t *right_child = &table->tree[node->right];
        PROF_TIC();
        scatter_both_leaves_avx512(symbols, indices, n, bm,
                                    (uint8_t)left_child->symbol,
                                    (uint8_t)right_child->symbol);
        PROF_TOC(PROF_SCATTER_BOTH_LEAVES, n);
        return;
    }

    case PIVCO_NODE_HALF_RIGHT: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        int nbytes = bitmap_bytes(n);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        int n_right = node_half_right_avx512(indices, n, bm, tmp);
        decode_node_avx512(table, node->right, tmp, n_right,
                            symbols, in_ptr, tmp + n_right, skip_node);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        int nbytes = bitmap_bytes(n);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        int n_left = node_half_left_avx512(indices, n, bm);
        decode_node_avx512(table, node->left, indices, n_left,
                            symbols, in_ptr, tmp, skip_node);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        if (kr_header_needed(table, node_id)) *in_ptr += KR_HEADER_BYTES;
        int nbytes = bitmap_bytes(n);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        int n_left, n_right;
        node_full_avx512(indices, n, bm, tmp, &n_left, &n_right);
        /* +32 padding before right child's tmp - one full vector
         * stride so partition_32's filler harmlessly lands in the gap.
         * See decode_node_neon for full rationale. */
        decode_node_avx512(table, node->left, indices, n_left,
                            symbols, in_ptr, tmp + n_right + 32, skip_node);
        decode_node_avx512(table, node->right, tmp, n_right,
                            symbols, in_ptr, tmp + n_right + 32, skip_node);
        return;
    }
    }
}

int pivco_huffman_decode_avx512(const uint8_t *in, size_t in_len,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    PROF_COUNT_ONLY(PROF_DECODE_ENTRY, PIVCO_BLOCK_SIZE);

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Root is a flat subtree (whole tree flat, D>=2). */
    if (table->flat_depth[table->tree_root] >= 2) {
        int D = table->flat_depth[table->tree_root];
        int total_bytes = (N * D + 7) >> 3;
        const uint8_t *bm = ptr;
        ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[table->tree_root]];
        PROF_TIC();
        flat_decode_direct_avx512(symbols, N, bm, D, c2s);
        PROF_TOC(PROF_FLAT_DECODE_DIRECT, N);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* K_right header for root (TD-skips; encoder wrote it iff root
     * has any non-leaf child). */
    if (kr_header_needed(table, table->tree_root)) ptr += KR_HEADER_BYTES;
    int nbytes = bitmap_bytes(N);
    const uint8_t *bm = ptr;
    ptr += nbytes;

    /* Prefill with most frequent symbol */
    int16_t skip_node = table->prefill_node;
    memset(symbols, table->prefill_sym, (size_t)N);

    /* Partition at root — skip identity array init.
     * +32 padding on indices to absorb partition_32's 64-byte filler;
     * 64B-aligned to keep cache-set layout deterministic.
     * See decode_node_neon comment. */
    uint16_t indices[PIVCO_BLOCK_SIZE + 32] __attribute__((aligned(64)));
    /* See pivco_huffman_neon.c encode comment -- skewed partitions
     * accumulate up to max_tree_depth × N of offset.  Heap-alloc. */
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)aligned_alloc(64, tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;
    /* Root partition: dispatch on root node_type for HALF_RIGHT /
     * HALF_LEFT specialization, mirroring NEON.  Saves writing the
     * skipped side (which the prefill memset already covered) and
     * lets the per-primitive profile attribute the root cost to
     * the correct PROF slot. */
    const pivco_node_type_t root_nt =
        (pivco_node_type_t)table->node_type[table->tree_root];

    if (root_nt == PIVCO_NODE_HALF_RIGHT) {
        int n_right = root_half_right_avx512(N, bm, tmp);
        decode_node_avx512(table, root->right, tmp, n_right,
                            symbols, &ptr, tmp + n_right + 32, skip_node);
    } else if (root_nt == PIVCO_NODE_HALF_LEFT) {
        int n_left = root_half_left_avx512(N, bm, indices);
        decode_node_avx512(table, root->left, indices, n_left,
                            symbols, &ptr, tmp, skip_node);
    } else {
        int n_left = 0, n_right = 0;
        PROF_TIC();
        for (int j = 0; j + 32 <= N; j += 32) {
            uint32_t mask;
            memcpy(&mask, bm + (j >> 3), 4);
            uint16_t id[32];
            for (int k = 0; k < 32; k++) id[k] = (uint16_t)(j + k);
            int nr = partition_32_full(id, mask,
                                        indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (32 - nr);
        }
        PROF_TOC(PROF_ROOT_FULL, N);
        /* +32 padding before right child's tmp - see decode_node_neon. */
        decode_node_avx512(table, root->left, indices, n_left,
                            symbols, &ptr, tmp + n_right + 32, skip_node);
        decode_node_avx512(table, root->right, tmp, n_right,
                            symbols, &ptr, tmp + n_right + 32, skip_node);
    }

    free(tmp);
    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

/* ============================================================
 * "Naive-tree / SIMD-primitives" decoder for grid completeness.
 * Decodes the slim wire format produced by pivco_huffman_encode_naive
 * (raw bitmap per internal in DFS preorder, no FSE marker, no
 * K_right) using AVX-512 SIMD primitives. */

static void decode_node_naive_simd_avx512(
        const pivco_huffman_table_t *table, int16_t node_id,
        uint16_t *indices, int n, uint8_t *symbols,
        const uint8_t **in_ptr, uint16_t *tmp)
{
    if (n == 0) return;
    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        PROF_TIC();
        scatter_write_avx512(symbols, indices, n, (uint8_t)node->symbol);
        PROF_TOC(PROF_SCATTER_SYM, n);
        return;
    }
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;
    int n_left, n_right;
    node_full_avx512(indices, n, bm, tmp, &n_left, &n_right);
    decode_node_naive_simd_avx512(table, node->left,  indices, n_left,
                                    symbols, in_ptr, tmp + n_right + 32);
    decode_node_naive_simd_avx512(table, node->right, tmp, n_right,
                                    symbols, in_ptr, tmp + n_right + 32);
}

int pivco_huffman_decode_naive_simd_avx512(
        const uint8_t *in, size_t in_len,
        const pivco_huffman_table_t *table,
        uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    (void)in_len;
    const int N = PIVCO_BLOCK_SIZE;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    uint16_t indices[PIVCO_BLOCK_SIZE + 32] __attribute__((aligned(64)));
    const size_t tmp_capacity =
        (size_t)PIVCO_BLOCK_SIZE * (PIVCO_MAX_CODE_LEN + 2);
    uint16_t *tmp = (uint16_t *)aligned_alloc(64, tmp_capacity * sizeof(uint16_t));
    if (!tmp) return PIVCO_ERR_NULL;

    for (int k = 0; k < N; k++) indices[k] = (uint16_t)k;

    decode_node_naive_simd_avx512(table, table->tree_root, indices, N,
                                    symbols, &ptr, tmp);

    free(tmp);
    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

/* Non-static wrapper exposed for the bottom-up decoder
 * (src/pivco_huffman_bu_x86.c) so it can route through the AVX-512
 * flat unpack instead of the slower SSE flat_decode_direct_x86. */
void pivco_huffman_flat_decode_direct_avx512_(uint8_t *symbols, int n,
                                               const uint8_t *bm, int D,
                                               const uint8_t *c2s) {
    flat_decode_direct_avx512(symbols, n, bm, D, c2s);
}

#endif /* PIVCO_HAS_AVX512 */
