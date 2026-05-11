#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
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
    /* AVX-512 doesn't have byte scatter, but we can use vpscatterd
       with 32-bit indices for dword scatter. For byte writes, we
       fall back to SSE-style extract or scalar.
       Use SSE extract for now (same as x86 backend). */
    int j = 0;
    /* Process 8 at a time using SSE extract */
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

/* Dense-codes pack: extract D bits per element from codes_la, pack
 * LSB-first.  Stays scalar for x86 -- per-byte / per-lane variable
 * shifts available on AVX2+ but the dominant slice here is the node
 * partition, not enc_flat.  Symmetric to the NEON helper. */
static inline void pack_D_bits_dense_avx512(uint8_t *out, int n, int D,
                                             int depth,
                                             const uint16_t *codes_la)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    uint32_t mask = (1u << D) - 1;
    int right_shift = 16 - depth - D;
    uint64_t buf = 0;
    int bits_in_buf = 0;
    int byte_idx = 0;
    for (int i = 0; i < n; i++) {
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

    /* Flat-subtree fast path: emit n*D packed bits. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        uint8_t *out = *out_ptr;
        pack_D_bits_dense_avx512(out, n, D, depth, codes_la);
        *out_ptr += total_bytes;
        return;
    }

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

    const int N = PIVCO_BLOCK_SIZE;

    /* Dense left-aligned codes; +32 slack covers the AVX-512 stride-32
     * partition's 64-byte vpcompressw store at n_left + 32 worst case. */
    uint16_t codes_la[PIVCO_BLOCK_SIZE + 32];
    for (int i = 0; i < N; i++) codes_la[i] = table->code_la[symbols[i]];

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    uint8_t *ptr = out;

    encode_node_avx512(table, table->tree_root, codes_la, N,
                        0, &ptr, tmp);

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
    uint8_t delta = (uint8_t)(sym0 ^ sym1);
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        symbols[indices[j    ]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-((mask >> 0) & 1)));
        symbols[indices[j + 1]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-((mask >> 1) & 1)));
        symbols[indices[j + 2]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-((mask >> 2) & 1)));
        symbols[indices[j + 3]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-((mask >> 3) & 1)));
        symbols[indices[j + 4]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-((mask >> 4) & 1)));
        symbols[indices[j + 5]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-((mask >> 5) & 1)));
        symbols[indices[j + 6]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-((mask >> 6) & 1)));
        symbols[indices[j + 7]] = (uint8_t)(sym0 ^ (delta & (uint8_t)-((mask >> 7) & 1)));
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
        int nbytes = bitmap_bytes(n);
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        int n_right = node_half_right_avx512(indices, n, bm, tmp);
        decode_node_avx512(table, node->right, tmp, n_right,
                            symbols, in_ptr, tmp + n_right, skip_node);
        return;
    }

    case PIVCO_NODE_HALF_LEFT: {
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
    uint16_t tmp[PIVCO_BLOCK_SIZE * 2]       __attribute__((aligned(64)));
    int n_left = 0, n_right = 0;

    {
        PROF_TIC();
        for (int j = 0; j + 32 <= N; j += 32) {
            uint32_t mask;
            memcpy(&mask, bm + (j >> 3), 4);
            /* Generate identity [j..j+31] */
            uint16_t id[32];
            for (int k = 0; k < 32; k++) id[k] = (uint16_t)(j + k);
            int nr = partition_32_full(id, mask,
                                        indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (32 - nr);
        }
        PROF_TOC(PROF_ROOT_FULL, N);
    }

    /* +32 padding before right child's tmp - see decode_node_neon. */
    decode_node_avx512(table, root->left, indices, n_left,
                        symbols, &ptr, tmp + n_right + 32, skip_node);
    decode_node_avx512(table, root->right, tmp, n_right,
                        symbols, &ptr, tmp + n_right + 32, skip_node);

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
