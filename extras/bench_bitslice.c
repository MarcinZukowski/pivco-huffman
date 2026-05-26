/* bench_bitslice.c -- microbench: classic LSB bit-pack vs bit-sliced
 * (transposed) packing on AVX-512.  Measures encode + decode + c2s
 * throughput for D in {3, 5, 7}.
 *
 * Layouts compared:
 *   classic   -- LSB-first packing into a flat byte stream.  Decode
 *                via the scalar AVX512_FLAT_UNPACK_SWITCH path that
 *                ph currently uses for D=7/8 and the per-D code blocks
 *                for D=3/5.  (For D=3/5, ph also has a SIMD vpermb
 *                path -- not benched here; this is only the scalar
 *                baseline.)
 *
 *   bitslice  -- one byte per bit-plane per group of 8 symbols.  For
 *                64-symbol groups (one mask reg's worth), D bit-planes
 *                of 8 bytes each.  Encode = vptestmb per plane; decode
 *                = vpmovm2b-style mask -> byte + OR + final vpermb
 *                (or 2-table blend for D=7).
 *
 * Builds only on x86_64 with AVX-512 VBMI2.
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if !(defined(__x86_64__) && defined(__AVX512VBMI2__))
int main(void) {
    fprintf(stderr,
            "bench_bitslice: requires x86_64 with AVX-512 VBMI2 "
            "(build with -mavx512f -mavx512bw -mavx512vbmi -mavx512vbmi2)\n");
    return 1;
}
#else

#include <immintrin.h>

/* Pull in ph's actual D=2..6 flat-decode unpack helpers.  Self-contained
 * header (only needs BW + VBMI + VBMI2). */
#include "pivco_huffman_avx512_flat.h"

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* ===========================================================
 * Classic LSB bit-pack -- encode (scalar) + decode (scalar switch
 * matching AVX512_FLAT_UNPACK_SWITCH for D=3/5/7).
 * =========================================================== */

static void enc_classic(int D, const uint8_t *syms, int n, uint8_t *out) {
    uint64_t buf = 0;
    int bits = 0;
    uint8_t *p = out;
    for (int i = 0; i < n; i++) {
        buf |= ((uint64_t)syms[i] & ((1u << D) - 1)) << bits;
        bits += D;
        while (bits >= 8) {
            *p++ = (uint8_t)buf;
            buf >>= 8;
            bits -= 8;
        }
    }
    if (bits) *p++ = (uint8_t)buf;
}

static void dec_classic_scalar(int D, const uint8_t *bm, int n,
                                const uint8_t *c2s, uint8_t *out)
{
    int i = 0;
    if (D == 2) {
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            out[i    ] = c2s[(b     ) & 3];
            out[i + 1] = c2s[(b >> 2) & 3];
            out[i + 2] = c2s[(b >> 4) & 3];
            out[i + 3] = c2s[(b >> 6) & 3];
        }
    } else if (D == 4) {
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            out[i    ] = c2s[b & 0x0F];
            out[i + 1] = c2s[b >> 4];
        }
    } else if (D == 3) {
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 3) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
                       | ((uint32_t)p[2] << 16);
            out[i    ] = c2s[(w      ) & 7];
            out[i + 1] = c2s[(w >>  3) & 7];
            out[i + 2] = c2s[(w >>  6) & 7];
            out[i + 3] = c2s[(w >>  9) & 7];
            out[i + 4] = c2s[(w >> 12) & 7];
            out[i + 5] = c2s[(w >> 15) & 7];
            out[i + 6] = c2s[(w >> 18) & 7];
            out[i + 7] = c2s[(w >> 21) & 7];
        }
    } else if (D == 5) {
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 5) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32);
            out[i    ] = c2s[(w      ) & 0x1F];
            out[i + 1] = c2s[(w >>  5) & 0x1F];
            out[i + 2] = c2s[(w >> 10) & 0x1F];
            out[i + 3] = c2s[(w >> 15) & 0x1F];
            out[i + 4] = c2s[(w >> 20) & 0x1F];
            out[i + 5] = c2s[(w >> 25) & 0x1F];
            out[i + 6] = c2s[(w >> 30) & 0x1F];
            out[i + 7] = c2s[(w >> 35) & 0x1F];
        }
    } else if (D == 7) {
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 7) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)
                       | ((uint64_t)p[6] << 48);
            out[i    ] = c2s[(w      ) & 0x7F];
            out[i + 1] = c2s[(w >>  7) & 0x7F];
            out[i + 2] = c2s[(w >> 14) & 0x7F];
            out[i + 3] = c2s[(w >> 21) & 0x7F];
            out[i + 4] = c2s[(w >> 28) & 0x7F];
            out[i + 5] = c2s[(w >> 35) & 0x7F];
            out[i + 6] = c2s[(w >> 42) & 0x7F];
            out[i + 7] = c2s[(w >> 49) & 0x7F];
        }
    }
    /* tail */
    for (; i < n; i++) {
        uint64_t off  = (uint64_t)i * D;
        uint64_t byte = off >> 3;
        uint32_t shft = off & 7;
        uint64_t w    = (uint64_t)bm[byte]
                      | ((uint64_t)bm[byte + 1] <<  8)
                      | ((uint64_t)bm[byte + 2] << 16);
        out[i] = c2s[(w >> shft) & ((1u << D) - 1)];
    }
}

/* ===========================================================
 * Bit-sliced layout -- groups of 64 symbols.
 * Storage per group of 64 symbols: D × 8 bytes (D bit-planes).
 * Plane p byte k holds bit-p of symbols 8k..8k+7 (little-endian-bit).
 *
 * For N symbols (N must be a multiple of 64 here), total storage is
 * (N / 64) * D * 8 = N * D / 8 bytes -- same as classic.
 * =========================================================== */

static __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vbmi2")))
void enc_bitslice(int D, const uint8_t *syms, int n, uint8_t *out)
{
    /* n must be a multiple of 64 -- caller ensures. */
    for (int i = 0; i < n; i += 64) {
        __m512i v = _mm512_loadu_si512((const __m512i *)(syms + i));
        uint8_t *grp = out + (i / 64) * D * 8;
        for (int p = 0; p < D; p++) {
            __mmask64 m = _mm512_test_epi8_mask(
                v, _mm512_set1_epi8((char)(1u << p)));
            uint64_t mw = (uint64_t)m;
            memcpy(grp + p * 8, &mw, 8);
        }
    }
}

/* Decode helper: given D mask-planes for a group of 64 symbols,
 * reconstruct the codes zmm (each byte = D-bit code).  Then a single
 * vpermb (or 2-table blend for D=7) folds in the c2s table. */

static __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vbmi2")))
__m512i bitslice_recover_codes(const uint8_t *grp, int D)
{
    __m512i codes = _mm512_setzero_si512();
    for (int p = 0; p < D; p++) {
        uint64_t mw;
        memcpy(&mw, grp + p * 8, 8);
        __mmask64 m = (__mmask64)mw;
        __m512i plane = _mm512_maskz_set1_epi8(m, (char)(1u << p));
        codes = _mm512_or_si512(codes, plane);
    }
    return codes;
}

static __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vbmi2")))
void dec_bitslice(int D, const uint8_t *bm, int n,
                   const uint8_t *c2s, uint8_t *out)
{
    /* n must be a multiple of 64. */
    if (D <= 6) {
        /* c2s fits in one zmm (≤64 entries). */
        __m512i c2s_v = _mm512_loadu_si512((const __m512i *)c2s);
        for (int i = 0; i < n; i += 64) {
            const uint8_t *grp = bm + (i / 64) * D * 8;
            __m512i codes = bitslice_recover_codes(grp, D);
            __m512i syms  = _mm512_permutexvar_epi8(codes, c2s_v);
            _mm512_storeu_si512((__m512i *)(out + i), syms);
        }
    } else {
        /* D=7: 128-entry c2s -- two 64-byte halves + select on bit 6. */
        __m512i c2s_lo = _mm512_loadu_si512((const __m512i *)(c2s +  0));
        __m512i c2s_hi = _mm512_loadu_si512((const __m512i *)(c2s + 64));
        __m512i bit6   = _mm512_set1_epi8(0x40);
        for (int i = 0; i < n; i += 64) {
            const uint8_t *grp = bm + (i / 64) * D * 8;
            __m512i codes = bitslice_recover_codes(grp, D);
            __mmask64 sel = _mm512_test_epi8_mask(codes, bit6);
            __m512i lo = _mm512_permutexvar_epi8(codes, c2s_lo);
            __m512i hi = _mm512_permutexvar_epi8(codes, c2s_hi);
            __m512i syms = _mm512_mask_blend_epi8(sel, lo, hi);
            _mm512_storeu_si512((__m512i *)(out + i), syms);
        }
    }
}

/* ===========================================================
 * Blend-based decode -- alternative to bit-sliced.
 *
 * Same storage layout (bit-sliced).  Decode pre-broadcasts the c2s
 * table into 2^D zmm registers and walks the D bit-planes top-down
 * as a binary-tree of vpblendmb operations.  The bit-plane mask is
 * used DIRECTLY as a blend mask -- no maskz_set1 + OR + vpermb
 * dance.  Only works for D <= 4 (D=5 needs 32 broadcast regs -- max).
 *
 * The c2s lookup is FOLDED into the broadcasts (no separate vpermb).
 *
 * Ops per 64-symbol group:
 *   D=2: 2 kmov + 3 blend + 1 store = 6
 *   D=3: 3 kmov + 7 blend + 1 store = 11
 *   D=4: 4 kmov + 15 blend + 1 store = 20
 * =========================================================== */

static __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vbmi2")))
void dec_blend_D2(const uint8_t *bm, int n, const uint8_t *c2s, uint8_t *out)
{
    __m512i c0 = _mm512_set1_epi8((char)c2s[0]);
    __m512i c1 = _mm512_set1_epi8((char)c2s[1]);
    __m512i c2 = _mm512_set1_epi8((char)c2s[2]);
    __m512i c3 = _mm512_set1_epi8((char)c2s[3]);
    for (int i = 0; i < n; i += 64) {
        const uint8_t *grp = bm + (i / 64) * 2 * 8;
        uint64_t m0w, m1w;
        memcpy(&m0w, grp + 0, 8);
        memcpy(&m1w, grp + 8, 8);
        __mmask64 m0 = (__mmask64)m0w;
        __mmask64 m1 = (__mmask64)m1w;
        __m512i c01 = _mm512_mask_blend_epi8(m0, c0, c1);
        __m512i c23 = _mm512_mask_blend_epi8(m0, c2, c3);
        __m512i res = _mm512_mask_blend_epi8(m1, c01, c23);
        _mm512_storeu_si512((__m512i *)(out + i), res);
    }
}

static __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vbmi2")))
void dec_blend_D3(const uint8_t *bm, int n, const uint8_t *c2s, uint8_t *out)
{
    __m512i c[8];
    for (int k = 0; k < 8; k++) c[k] = _mm512_set1_epi8((char)c2s[k]);
    for (int i = 0; i < n; i += 64) {
        const uint8_t *grp = bm + (i / 64) * 3 * 8;
        uint64_t m0w, m1w, m2w;
        memcpy(&m0w, grp +  0, 8);
        memcpy(&m1w, grp +  8, 8);
        memcpy(&m2w, grp + 16, 8);
        __mmask64 m0 = (__mmask64)m0w;
        __mmask64 m1 = (__mmask64)m1w;
        __mmask64 m2 = (__mmask64)m2w;
        /* plane 0: pair up neighbours */
        __m512i p01 = _mm512_mask_blend_epi8(m0, c[0], c[1]);
        __m512i p23 = _mm512_mask_blend_epi8(m0, c[2], c[3]);
        __m512i p45 = _mm512_mask_blend_epi8(m0, c[4], c[5]);
        __m512i p67 = _mm512_mask_blend_epi8(m0, c[6], c[7]);
        /* plane 1 */
        __m512i p0123 = _mm512_mask_blend_epi8(m1, p01, p23);
        __m512i p4567 = _mm512_mask_blend_epi8(m1, p45, p67);
        /* plane 2 */
        __m512i res = _mm512_mask_blend_epi8(m2, p0123, p4567);
        _mm512_storeu_si512((__m512i *)(out + i), res);
    }
}

static __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vbmi2")))
void dec_blend_D4(const uint8_t *bm, int n, const uint8_t *c2s, uint8_t *out)
{
    __m512i c[16];
    for (int k = 0; k < 16; k++) c[k] = _mm512_set1_epi8((char)c2s[k]);
    for (int i = 0; i < n; i += 64) {
        const uint8_t *grp = bm + (i / 64) * 4 * 8;
        uint64_t mw[4];
        for (int p = 0; p < 4; p++) memcpy(&mw[p], grp + p * 8, 8);
        __mmask64 m0 = (__mmask64)mw[0];
        __mmask64 m1 = (__mmask64)mw[1];
        __mmask64 m2 = (__mmask64)mw[2];
        __mmask64 m3 = (__mmask64)mw[3];
        /* plane 0: 8 pairs */
        __m512i a[8];
        for (int k = 0; k < 8; k++)
            a[k] = _mm512_mask_blend_epi8(m0, c[2*k], c[2*k + 1]);
        /* plane 1: 4 quads */
        __m512i b[4];
        for (int k = 0; k < 4; k++)
            b[k] = _mm512_mask_blend_epi8(m1, a[2*k], a[2*k + 1]);
        /* plane 2: 2 octets */
        __m512i d[2];
        d[0] = _mm512_mask_blend_epi8(m2, b[0], b[1]);
        d[1] = _mm512_mask_blend_epi8(m2, b[2], b[3]);
        /* plane 3: final */
        __m512i res = _mm512_mask_blend_epi8(m3, d[0], d[1]);
        _mm512_storeu_si512((__m512i *)(out + i), res);
    }
}

static __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vbmi2")))
void dec_blend(int D, const uint8_t *bm, int n,
                const uint8_t *c2s, uint8_t *out)
{
    if      (D == 2) dec_blend_D2(bm, n, c2s, out);
    else if (D == 3) dec_blend_D3(bm, n, c2s, out);
    else if (D == 4) dec_blend_D4(bm, n, c2s, out);
}

/* ===========================================================
 * ph's actual SIMD flat-decode path (mirror of
 * flat_decode_direct_avx512_inner in pivco_huffman_primitives_avx512.h).
 * Uses the shared flat_d{2..6}_unpack_avx512 helpers from
 * pivco_huffman_avx512_flat.h, then pshufb / vpermb / vpermw for c2s.
 * Reads classic LSB-packed bitmaps.
 * =========================================================== */

static __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vbmi2")))
void dec_ph_simd(int D, const uint8_t *bm, int n,
                  const uint8_t *c2s, uint8_t *out)
{
    int i = 0;
    if (D == 2) {
        uint32_t lo; memcpy(&lo, c2s, 4);
        __m128i c2s_v = _mm_set1_epi32((int32_t)lo);
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d2_unpack_avx512(bm + (i >> 2));
            __m128i syms  = _mm_shuffle_epi8(c2s_v, codes);
            _mm_storeu_si128((__m128i *)(out + i), syms);
        }
    } else if (D == 3) {
        uint64_t lo; memcpy(&lo, c2s, 8);
        __m128i c2s_v = _mm_cvtsi64_si128((int64_t)lo);
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_v, codes);
            _mm_storeu_si128((__m128i *)(out + i), syms);
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d3_unpack_avx512_safe(bm + ((i * 3) >> 3));
            __m128i syms  = _mm_shuffle_epi8(c2s_v, codes);
            _mm_storeu_si128((__m128i *)(out + i), syms);
            i += 16;
        }
    } else if (D == 4) {
        __m128i c2s_v = _mm_loadu_si128((const __m128i *)c2s);
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_avx512(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_v, codes);
            _mm_storeu_si128((__m128i *)(out + i), syms);
        }
    } else if (D == 5) {
        __m256i c2s_v = _mm256_loadu_si256((const __m256i *)c2s);
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3));
            __m256i codes_ext = _mm256_zextsi128_si256(codes);
            __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_v);
            _mm_storeu_si128((__m128i *)(out + i),
                              _mm256_castsi256_si128(syms_full));
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d5_unpack_avx512_safe(bm + ((i * 5) >> 3));
            __m256i codes_ext = _mm256_zextsi128_si256(codes);
            __m256i syms_full = _mm256_permutexvar_epi8(codes_ext, c2s_v);
            _mm_storeu_si128((__m128i *)(out + i),
                              _mm256_castsi256_si128(syms_full));
            i += 16;
        }
    } else if (D == 6) {
        __m512i c2s_v = _mm512_loadu_si512((const __m512i *)c2s);
        int fast_end = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fast_end; i += 16) {
            __m128i codes = flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3));
            __m512i codes_ext = _mm512_castsi128_si512(codes);
            __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_v);
            _mm_storeu_si128((__m128i *)(out + i),
                              _mm512_castsi512_si128(syms_full));
        }
        if (i + 16 <= n) {
            __m128i codes = flat_d6_unpack_avx512_safe(bm + ((i * 6) >> 3));
            __m512i codes_ext = _mm512_castsi128_si512(codes);
            __m512i syms_full = _mm512_permutexvar_epi8(codes_ext, c2s_v);
            _mm_storeu_si128((__m128i *)(out + i),
                              _mm512_castsi512_si128(syms_full));
            i += 16;
        }
    }
    /* scalar tail */
    for (; i < n; i++) {
        uint64_t off  = (uint64_t)i * D;
        uint64_t byte = off >> 3;
        uint32_t shft = off & 7;
        uint64_t w    = (uint64_t)bm[byte]
                      | ((uint64_t)bm[byte + 1] <<  8)
                      | ((uint64_t)bm[byte + 2] << 16);
        out[i] = c2s[(w >> shft) & ((1u << D) - 1)];
    }
}

/* ===========================================================
 * Harness
 * =========================================================== */

static int verify_roundtrip(int D, int n,
                             void (*enc)(int, const uint8_t *, int, uint8_t *),
                             void (*dec)(int, const uint8_t *, int,
                                          const uint8_t *, uint8_t *))
{
    uint8_t *syms = aligned_alloc(64, (size_t)n);
    uint8_t *bm   = aligned_alloc(64, (size_t)n * D / 8 + 64);
    uint8_t *out  = aligned_alloc(64, (size_t)n);
    uint8_t c2s[256];
    for (int i = 0; i < 256; i++) c2s[i] = (uint8_t)i;  /* identity */
    for (int i = 0; i < n; i++) syms[i] = (uint8_t)(rand() & ((1u << D) - 1));
    enc(D, syms, n, bm);
    dec(D, bm, n, c2s, out);
    int ok = (memcmp(syms, out, (size_t)n) == 0);
    free(syms); free(bm); free(out);
    return ok;
}

typedef void (*enc_fn)(int, const uint8_t *, int, uint8_t *);
typedef void (*dec_fn)(int, const uint8_t *, int, const uint8_t *, uint8_t *);

static double bench_enc(enc_fn enc, int D, int n, int iters,
                         const uint8_t *syms, uint8_t *bm)
{
    /* warmup */
    for (int k = 0; k < 4; k++) enc(D, syms, n, bm);
    uint64_t t0 = now_ns();
    for (int k = 0; k < iters; k++) enc(D, syms, n, bm);
    uint64_t t1 = now_ns();
    return (double)(t1 - t0) / ((double)iters * (double)n);   /* ns/sym */
}

static double bench_dec(dec_fn dec, int D, int n, int iters,
                         const uint8_t *bm, const uint8_t *c2s, uint8_t *out)
{
    for (int k = 0; k < 4; k++) dec(D, bm, n, c2s, out);
    uint64_t t0 = now_ns();
    for (int k = 0; k < iters; k++) dec(D, bm, n, c2s, out);
    uint64_t t1 = now_ns();
    return (double)(t1 - t0) / ((double)iters * (double)n);
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 32768;
    int iters = (argc > 2) ? atoi(argv[2]) : 4000;
    if (n % 64) {
        n -= (n % 64);
        fprintf(stderr, "(rounded N down to %d, multiple of 64)\n", n);
    }
    fprintf(stderr, "N = %d, iters = %d\n", n, iters);

    /* Verify roundtrip for each D and layout. */
    srand(1);
    static const int DS[] = {2, 3, 4, 5, 6, 7};
    static const int NDS = sizeof(DS) / sizeof(DS[0]);
    for (int di = 0; di < NDS; di++) {
        int D = DS[di];
        if (!verify_roundtrip(D, n, enc_classic,  dec_classic_scalar))
            { fprintf(stderr, "classic D=%d FAILED\n", D);  return 1; }
        if (!verify_roundtrip(D, n, enc_bitslice, dec_bitslice))
            { fprintf(stderr, "bitslice D=%d FAILED\n", D); return 1; }
        if (D <= 4 &&
            !verify_roundtrip(D, n, enc_bitslice, dec_blend))
            { fprintf(stderr, "blend D=%d FAILED\n", D); return 1; }
        if (D <= 6 &&
            !verify_roundtrip(D, n, enc_classic, dec_ph_simd))
            { fprintf(stderr, "ph-simd D=%d FAILED\n", D); return 1; }
    }
    fprintf(stderr,
            "roundtrip OK for D=2..7 "
            "(classic + bitslice; ph-simd D<=6; blend D<=4)\n\n");

    uint8_t c2s[256];
    for (int i = 0; i < 256; i++) c2s[i] = (uint8_t)i;
    uint8_t *syms = aligned_alloc(64, (size_t)n);
    uint8_t *bm   = aligned_alloc(64, (size_t)n + 64);
    uint8_t *out  = aligned_alloc(64, (size_t)n);

    /* DECODE table (ns/sym) -- baseline column is dec-ph (ph's actual
     * SIMD path; falls back to scalar for D=7). */
    printf("decode (ns/sym; baseline = ph-simd for D<=6, scalar for D>=7):\n");
    printf("%-3s | %10s | %10s | %10s | %10s | %s\n",
           "D", "dec-cls", "dec-ph", "dec-bs", "dec-blend",
           "bs-vs-ph  blend-vs-ph");
    printf("----+------------+------------+------------+------------+----------------------\n");

    for (int di = 0; di < NDS; di++) {
        int D = DS[di];
        for (int i = 0; i < n; i++) syms[i] = (uint8_t)(rand() & ((1u << D) - 1));

        enc_classic (D, syms, n, bm);
        double dc = bench_dec(dec_classic_scalar, D, n, iters, bm, c2s, out);
        double dp = (D <= 6) ? bench_dec(dec_ph_simd, D, n, iters, bm, c2s, out) : dc;

        enc_bitslice(D, syms, n, bm);
        double db = bench_dec(dec_bitslice, D, n, iters, bm, c2s, out);
        double dbl = (D <= 4) ? bench_dec(dec_blend, D, n, iters, bm, c2s, out) : -1.0;

        double baseline = (D <= 6) ? dp : dc;
        if (dbl > 0) {
            printf("%-3d | %10.3f | %10.3f | %10.3f | %10.3f | %5.2fx     %5.2fx\n",
                   D, dc, dp, db, dbl, baseline / db, baseline / dbl);
        } else {
            printf("%-3d | %10.3f | %10.3f | %10.3f | %10s | %5.2fx       %5s\n",
                   D, dc, dp, db, "-", baseline / db, "-");
        }
    }

    /* ENCODE table -- caveat: enc-cls is naive scalar bit-pack, NOT
     * ph's actual prim_enc_pack_dN. */
    printf("\nencode (ns/sym; enc-cls is NAIVE scalar -- not ph's prim_enc_pack_dN):\n");
    printf("%-3s | %10s | %10s | %s\n", "D", "enc-cls", "enc-bs", "bs-vs-cls");
    printf("----+------------+------------+-----------\n");
    for (int di = 0; di < NDS; di++) {
        int D = DS[di];
        for (int i = 0; i < n; i++) syms[i] = (uint8_t)(rand() & ((1u << D) - 1));
        double ec = bench_enc(enc_classic,  D, n, iters, syms, bm);
        double eb = bench_enc(enc_bitslice, D, n, iters, syms, bm);
        printf("%-3d | %10.3f | %10.3f | %5.2fx\n", D, ec, eb, ec / eb);
    }

    free(syms); free(bm); free(out);
    return 0;
}

#endif  /* x86_64 + AVX512VBMI2 */
