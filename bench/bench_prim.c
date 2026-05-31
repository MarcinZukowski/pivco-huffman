/* bench_prim — isolated decode/encode-primitive microbench.
 *
 * Benches the flat-subtree + tree-walk primitives on SYNTHETIC random input,
 * with no Huffman tree and no data distribution to reason about: these kernels
 * are branchless / data-independent, so random packed bytes + a random
 * code_to_sym table fully characterize throughput.
 *
 * Stages, benched SEPARATELY:
 *   unpack    — read N D-bit codes from the packed stream -> codes[]  (flat_dN_unpack)
 *   scatter   — codes[] + c2s[2^D] -> out[]                           (NEON TBL)
 *   merge     — packed stream + c2s -> out[]  (production prim_merge_flat)
 *   pack      — codes_la[] -> packed N*D-bit stream  (production prim_enc_pack_dN)
 *   partition — codes_la[] + depth -> bitmap + left/right split  (prim_enc_partition)
 *
 * unpack/scatter/merge/pack are per-depth-D; partition is a 1-bit split (one
 * representative depth).  Every SIMD variant is checked against a scalar
 * reference before timing.  Metric: ns/elem (3 sig figs).
 *
 * Build: CMake target pivco_bench_prim (links pivco_huffman for the NEON
 * tables + lazy init).   Run: ./pivco_bench_prim [--n=] [--reps=] [--D=2,3,..]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Backend select (matches the codec's per-tier OBJECT libs).  Compile with
 * -march=native: aarch64 -> NEON; x86 with AVX-512 VBMI2 -> AVX512; else SSE/
 * AVX2.  pack/merge/partition are benched via the production prim_* (exist on
 * every backend); the standalone unpack/scatter kernels are NEON-only. */
#if defined(__aarch64__)
#  define PIVCO_BACKEND_NEON 1
#  define HAVE_SIMD 1
#  define HAVE_NEON_KERNELS 1
#  define BK "neon"
#elif defined(__x86_64__)
#  if defined(__SSE4_1__)
#    define PIVCO_HAS_SSE4 1
#  endif
#  if defined(__AVX2__)
#    define PIVCO_HAS_AVX2 1
#  endif
#  if defined(__AVX512VBMI2__)
#    define PIVCO_HAS_AVX512 1
#    define PIVCO_BACKEND_AVX512 1
#    define BK "avx512"
#    define HAVE_SIMD 1
#  elif defined(__SSE4_1__)
#    define PIVCO_BACKEND_X86 1
#    define BK "sse/avx2"
#    define HAVE_SIMD 1
#  endif
#endif
#include "pivco_huffman.h"
#if defined(HAVE_SIMD)
#  include "pivco_huffman_primitives.h"   /* prim_enc_pack_dN / prim_enc_partition /
                                             prim_merge_flat (+ NEON flat_dN) */
#endif
#ifndef BK
#  define BK "scalar-only"
#endif

#define MAXD 8         /* pack/merge/partition use prim_* (SIMD where the
                          backend supports D, scalar fallback otherwise). The
                          NEON standalone unpack/scatter kernels go to D7. */
#define PART_DEPTH 3   /* representative split depth for the partition bench */

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

/* ---------- scalar references (any D, LSB-first packed stream) ---------- */
static void scalar_unpack(uint8_t *codes, const uint8_t *bm, int n, int D) {
    uint64_t acc = 0; int bits = 0; const uint8_t *p = bm;
    const uint32_t mask = (1u << D) - 1u;
    for (int i = 0; i < n; i++) {
        while (bits < D) { acc |= (uint64_t)(*p++) << bits; bits += 8; }
        codes[i] = (uint8_t)(acc & mask); acc >>= D; bits -= D;
    }
}
static void scalar_pack(uint8_t *bm, const uint16_t *codes_la, int n, int D, int depth) {
    int rsh = 16 - depth - D; const uint32_t mask = (1u << D) - 1u;
    int nbytes = (n * D + 7) >> 3;
    memset(bm, 0, nbytes);
    uint64_t acc = 0; int bits = 0; uint8_t *o = bm;
    for (int i = 0; i < n; i++) {
        uint32_t v = ((uint32_t)codes_la[i] >> rsh) & mask;
        acc |= (uint64_t)v << bits; bits += D;
        while (bits >= 8) { *o++ = (uint8_t)acc; acc >>= 8; bits -= 8; }
    }
    if (bits) *o = (uint8_t)acc;
}
static void scalar_scatter(uint8_t *out, const uint8_t *codes,
                           const uint8_t *c2s, int n) {
    for (int i = 0; i < n; i++) out[i] = c2s[codes[i]];
}
/* writes bitmap (LSB=first code) + compacts left(bit0)/right(bit1); returns n_right */
static int scalar_partition(const uint16_t *codes_la, int n, int depth,
                            uint8_t *bm, uint16_t *left, uint16_t *right) {
    int nl = 0, nr = 0, sh = 15 - depth;
    memset(bm, 0, (n + 7) >> 3);
    for (int i = 0; i < n; i++) {
        int b = (codes_la[i] >> sh) & 1;
        if (b) { bm[i >> 3] |= (uint8_t)(1u << (i & 7)); right[nr++] = codes_la[i]; }
        else   { left[nl++] = codes_la[i]; }
    }
    return nr;
}
/* scalar references for the binary-merge family (bottom-up).  bm bit==0
   -> take next left, bm bit==1 -> take next right (or constants). */
static void scalar_merge_gen(uint8_t *out, const uint8_t *bm, int n,
                              const uint8_t *L, const uint8_t *R) {
    int lc=0, rc=0;
    for (int i=0; i<n; i++) {
        int b = (bm[i>>3] >> (i&7)) & 1;
        out[i] = b ? R[rc++] : L[lc++];
    }
}
static void scalar_merge_two(uint8_t *out, const uint8_t *bm, int n,
                              uint8_t l, uint8_t r) {
    for (int i=0; i<n; i++) {
        int b = (bm[i>>3] >> (i&7)) & 1;
        out[i] = b ? r : l;
    }
}
static void scalar_merge_const_l(uint8_t *out, const uint8_t *bm, int n,
                                  uint8_t l, const uint8_t *R) {
    int rc=0;
    for (int i=0; i<n; i++) {
        int b = (bm[i>>3] >> (i&7)) & 1;
        out[i] = b ? R[rc++] : l;
    }
}
static void scalar_merge_const_r(uint8_t *out, const uint8_t *bm, int n,
                                  const uint8_t *L, uint8_t r) {
    int lc=0;
    for (int i=0; i<n; i++) {
        int b = (bm[i>>3] >> (i&7)) & 1;
        out[i] = b ? r : L[lc++];
    }
}

/* MERGE_CONST_SYMBOLS are baked-in constants used by every merge_two /
 * merge_constant_* variant (scalar refs and SIMD wrappers) so the
 * correctness check across implementations agrees on which two symbols
 * to emit per bm bit. */
#define MERGE_LEFT_SYM   0x42
#define MERGE_RIGHT_SYM  0xA5

typedef struct {
    uint8_t  *bm, *codes, *c2s, *out, *pack_out;
    uint8_t  *merge_left, *merge_right;   /* dense byte sources for prim_merge */
    uint16_t *la_work, *tmp16;
    int n, D, depth;
} ctx_t;

static void p_unpack_scalar (const ctx_t *c){ scalar_unpack(c->codes,c->bm,c->n,c->D); }
static void p_scatter_scalar(const ctx_t *c){ scalar_scatter(c->out,c->codes,c->c2s,c->n); }
static void p_pack_scalar   (const ctx_t *c){ scalar_pack(c->pack_out,c->la_work,c->n,c->D,c->depth); }
static void p_merge_scalar  (const ctx_t *c){ scalar_unpack(c->codes,c->bm,c->n,c->D);
                                              scalar_scatter(c->out,c->codes,c->c2s,c->n); }
static void p_part_scalar   (const ctx_t *c){
    /* scratch left = la_work, right = tmp16 (la_work pre-filled from pristine) */
    static uint16_t lbuf[1<<16];
    scalar_partition(c->la_work, c->n, c->depth, c->bm, lbuf, c->tmp16);
    memcpy(c->la_work, lbuf, (size_t)(c->n - 0) * 2); /* keep left in place like the prim */
}
/* scalar refs for the new binary-merge stages */
static void p_merge_gen_scalar     (const ctx_t *c){ scalar_merge_gen(c->out, c->bm, c->n, c->merge_left, c->merge_right); }
static void p_merge_two_scalar     (const ctx_t *c){ scalar_merge_two(c->out, c->bm, c->n, MERGE_LEFT_SYM, MERGE_RIGHT_SYM); }
static void p_merge_const_l_scalar (const ctx_t *c){ scalar_merge_const_l(c->out, c->bm, c->n, MERGE_LEFT_SYM, c->merge_right); }
static void p_merge_const_r_scalar (const ctx_t *c){ scalar_merge_const_r(c->out, c->bm, c->n, c->merge_left, MERGE_RIGHT_SYM); }
/* Synthetic memory-bandwidth comparison points. */
static void scalar_xor(uint8_t *out, const uint8_t *a, const uint8_t *b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] ^ b[i];
}
/* xor_accum: read input only, write a single byte at the end (max-in-bw probe). */
static void scalar_xor_accum(uint8_t *out, const uint8_t *a, int n) {
    uint8_t acc = 0;
    for (int i = 0; i < n; i++) acc ^= a[i];
    out[0] = acc;
}
/* plus_one: write only, value out[i] = (uint8_t)i (max-out-bw probe).  */
static void scalar_plus_one(uint8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (uint8_t)i;
}
static void p_xor_scalar       (const ctx_t *c){ scalar_xor(c->out, c->merge_left, c->merge_right, c->n); }
static void p_xor_accum_scalar (const ctx_t *c){ scalar_xor_accum(c->out, c->merge_left, c->n); }
static void p_plus_one_scalar  (const ctx_t *c){ scalar_plus_one(c->out, c->n); }

#if defined(HAVE_NEON_KERNELS)   /* standalone unpack/scatter: NEON intrinsics */
static void neon_unpack(const ctx_t *c) {
    int n = c->n; uint8_t *cd = c->codes; const uint8_t *bm = c->bm;
    switch (c->D) {
    case 2: for (int i=0;i<n;i+=16) vst1q_u8(cd+i, flat_d2_unpack(bm + (i>>4)*4)); break;
    case 3: for (int i=0;i<n;i+= 8) vst1_u8 (cd+i, flat_d3_unpack(bm + (i>>3)*3)); break;
    case 4: for (int i=0;i<n;i+=16) vst1q_u8(cd+i, flat_d4_unpack(bm + (i>>4)*8)); break;
    case 5: for (int i=0;i<n;i+= 8) vst1_u8 (cd+i, flat_d5_unpack(bm + (i>>3)*5)); break;
    case 6: for (int i=0;i<n;i+= 8) vst1_u8 (cd+i, flat_d6_unpack(bm + (i>>3)*6)); break;
    case 7: for (int i=0;i<n;i+= 8) vst1_u8 (cd+i, flat_d7_unpack(bm + (i>>3)*7)); break;
    }
}
static void neon_scatter(const ctx_t *c) {
    int n = c->n, D = c->D; uint8_t *out = c->out; const uint8_t *cd = c->codes;
    if (D <= 4) {
        uint8x16_t t = vld1q_u8(c->c2s);
        for (int i=0;i<n;i+=16) vst1q_u8(out+i, vqtbl1q_u8(t, vld1q_u8(cd+i)));
    } else if (D == 5) {
        uint8x16x2_t t = { { vld1q_u8(c->c2s), vld1q_u8(c->c2s+16) } };
        for (int i=0;i<n;i+=16) vst1q_u8(out+i, vqtbl2q_u8(t, vld1q_u8(cd+i)));
    } else if (D == 6) {
        uint8x16x4_t t = { { vld1q_u8(c->c2s),    vld1q_u8(c->c2s+16),
                             vld1q_u8(c->c2s+32), vld1q_u8(c->c2s+48) } };
        for (int i=0;i<n;i+=16) vst1q_u8(out+i, vqtbl4q_u8(t, vld1q_u8(cd+i)));
    } else { /* D == 7: 128-entry table -> two vqtbl4 (lo/hi) + OR */
        uint8x16x4_t lo = { { vld1q_u8(c->c2s),    vld1q_u8(c->c2s+16),
                              vld1q_u8(c->c2s+32), vld1q_u8(c->c2s+48) } };
        uint8x16x4_t hi = { { vld1q_u8(c->c2s+64), vld1q_u8(c->c2s+80),
                              vld1q_u8(c->c2s+96), vld1q_u8(c->c2s+112) } };
        uint8x16_t s64 = vdupq_n_u8(64);
        for (int i=0;i<n;i+=16) {
            uint8x16_t k = vld1q_u8(cd+i);
            vst1q_u8(out+i, vorrq_u8(vqtbl4q_u8(lo,k), vqtbl4q_u8(hi, vsubq_u8(k,s64))));
        }
    }
}
#endif /* HAVE_NEON_KERNELS */

#if defined(HAVE_SIMD)   /* pack/merge/partition: production prim_*, all backends */
static void simd_pack (const ctx_t *c){ prim_enc_pack_dN(c->la_work, c->n, c->D, c->depth, c->pack_out); }
static void simd_merge_flat(const ctx_t *c){ prim_merge_flat(c->out, c->n, c->bm, c->D, c->c2s); }
static void simd_part (const ctx_t *c){ prim_enc_partition_full(c->la_work, c->n, c->depth, c->bm, c->tmp16); }
#endif
#if defined(HAVE_SIMD)   /* binary-merge production primitives — all backends */
static void simd_merge_gen     (const ctx_t *c){ prim_merge(c->bm, c->n, c->merge_left, c->merge_right, c->out); }
static void simd_merge_two     (const ctx_t *c){ prim_merge_two(c->bm, c->n, MERGE_LEFT_SYM, MERGE_RIGHT_SYM, c->out); }
static void simd_merge_const_l (const ctx_t *c){ prim_merge_constant_left(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out); }
static void simd_merge_const_r (const ctx_t *c){ prim_merge_constant_right(c->bm, c->n, c->merge_left, MERGE_RIGHT_SYM, c->out); }
/* Synthetic byte-XOR: explicit SIMD reference for memory-bandwidth comparison.
   16 bytes per iter via NEON veorq_u8 / 64 bytes via AVX-512 _mm512_xor_si512
   / 16 bytes via SSE _mm_xor_si128.  Scalar tail covers the unaligned end. */
static void simd_xor (const ctx_t *c){
    const uint8_t *a = c->merge_left, *b = c->merge_right; uint8_t *o = c->out;
    int n = c->n, i = 0;
#if defined(__aarch64__)
    for (; i + 16 <= n; i += 16) vst1q_u8(o + i, veorq_u8(vld1q_u8(a + i), vld1q_u8(b + i)));
#elif defined(__AVX512F__)
    for (; i + 64 <= n; i += 64) _mm512_storeu_si512((void*)(o + i),
        _mm512_xor_si512(_mm512_loadu_si512((const void*)(a + i)),
                         _mm512_loadu_si512((const void*)(b + i))));
#elif defined(__AVX2__)
    for (; i + 32 <= n; i += 32) _mm256_storeu_si256((__m256i*)(o + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(a + i)),
                         _mm256_loadu_si256((const __m256i*)(b + i))));
#elif defined(__SSE2__)
    for (; i + 16 <= n; i += 16) _mm_storeu_si128((__m128i*)(o + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i*)(a + i)),
                      _mm_loadu_si128((const __m128i*)(b + i))));
#endif
    for (; i < n; i++) o[i] = a[i] ^ b[i];
}

/* max-in-bw probe: vector XOR-fold of input, single-byte write at end. */
static void simd_xor_accum (const ctx_t *c){
    const uint8_t *a = c->merge_left; uint8_t *o = c->out; int n = c->n, i = 0;
#if defined(__aarch64__)
    uint8x16_t acc = vdupq_n_u8(0);
    for (; i + 16 <= n; i += 16) acc = veorq_u8(acc, vld1q_u8(a + i));
    uint8_t buf[16]; vst1q_u8(buf, acc); uint8_t r = 0;
    for (int j = 0; j < 16; j++) r ^= buf[j];
#elif defined(__AVX512F__)
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= n; i += 64) acc = _mm512_xor_si512(acc, _mm512_loadu_si512((const void*)(a + i)));
    uint8_t buf[64]; _mm512_storeu_si512((void*)buf, acc); uint8_t r = 0;
    for (int j = 0; j < 64; j++) r ^= buf[j];
#elif defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    for (; i + 32 <= n; i += 32) acc = _mm256_xor_si256(acc, _mm256_loadu_si256((const __m256i*)(a + i)));
    uint8_t buf[32]; _mm256_storeu_si256((__m256i*)buf, acc); uint8_t r = 0;
    for (int j = 0; j < 32; j++) r ^= buf[j];
#elif defined(__SSE2__)
    __m128i acc = _mm_setzero_si128();
    for (; i + 16 <= n; i += 16) acc = _mm_xor_si128(acc, _mm_loadu_si128((const __m128i*)(a + i)));
    uint8_t buf[16]; _mm_storeu_si128((__m128i*)buf, acc); uint8_t r = 0;
    for (int j = 0; j < 16; j++) r ^= buf[j];
#else
    uint8_t r = 0;
#endif
    for (; i < n; i++) r ^= a[i];
    o[0] = r;
}

/* max-out-bw probe: write only, value out[i] = (uint8_t)i. */
static void simd_plus_one (const ctx_t *c){
    uint8_t *o = c->out; int n = c->n, i = 0;
#if defined(__aarch64__)
    static const uint8_t IOTA16[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    uint8x16_t iota = vld1q_u8(IOTA16);
    for (; i + 16 <= n; i += 16) vst1q_u8(o + i, vaddq_u8(vdupq_n_u8((uint8_t)i), iota));
#elif defined(__AVX512F__)
    static const uint8_t IOTA64[64] __attribute__((aligned(64))) = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
        32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, 48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};
    __m512i iota = _mm512_load_si512((const void*)IOTA64);
    for (; i + 64 <= n; i += 64) _mm512_storeu_si512((void*)(o + i),
        _mm512_add_epi8(_mm512_set1_epi8((char)i), iota));
#elif defined(__AVX2__)
    static const uint8_t IOTA32[32] __attribute__((aligned(32))) = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    __m256i iota = _mm256_load_si256((const __m256i*)IOTA32);
    for (; i + 32 <= n; i += 32) _mm256_storeu_si256((__m256i*)(o + i),
        _mm256_add_epi8(_mm256_set1_epi8((char)i), iota));
#elif defined(__SSE2__)
    static const uint8_t IOTA16[16] __attribute__((aligned(16))) = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    __m128i iota = _mm_load_si128((const __m128i*)IOTA16);
    for (; i + 16 <= n; i += 16) _mm_storeu_si128((__m128i*)(o + i),
        _mm_add_epi8(_mm_set1_epi8((char)i), iota));
#endif
    for (; i < n; i++) o[i] = (uint8_t)i;
}
#endif

#if defined(HAVE_NEON_KERNELS)
/* ---------- experimental: precomputed-popcount merge ----------
 *
 * Replace the per-iter `expand_popcnt[bm[j>>3]]` table lookup with a
 * sequential read from a precomputed bm_popcnt[] array.  The hypothesis
 * is that the production lookup is on the critical path (mask compute ->
 * popcnt-table load -> cursor advance), and a statically-known sequential
 * address frees the load to start at iter top.
 *
 * Two variants:
 *   neon_pcpc_pure   — bm_popcnt[] is pre-filled (treat the populating
 *                       pass as amortized cost; bench the merge alone).
 *   neon_pcpc_full   — populate bm_popcnt[] inside the timed function
 *                       (the realistic cost a caller would pay).
 *
 * The shuffle vectors (expand_tab[m] / expand_tab_pre[nr0][m]) STILL
 * depend on the mask byte itself — only the popcount load is replaced.
 */
static uint8_t g_bm_popcnt[16384] __attribute__((aligned(64)));

static inline void fill_bm_popcnt(const uint8_t *bm, int K) {
    int bm_bytes = (K + 7) >> 3;
    int i = 0;
    for (; i + 16 <= bm_bytes; i += 16) {
        vst1q_u8(g_bm_popcnt + i, vcntq_u8(vld1q_u8(bm + i)));
    }
    for (; i < bm_bytes; i++) g_bm_popcnt[i] = (uint8_t)__builtin_popcount(bm[i]);
}

static inline void merge_neon_pcpc(const uint8_t *bm, int K,
                                     const uint8_t *left, const uint8_t *right,
                                     uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x16_t L_full = vld1q_u8(left + lc);
        uint8x16_t R_full = vld1q_u8(right + rc);

        uint8_t m0  = bm[j >> 3];
        int     nr0 = g_bm_popcnt[j >> 3];        /* sequential, address known early */
        uint8x16_t both0 = vcombine_u8(vget_low_u8(L_full), vget_low_u8(R_full));
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);

        uint8_t m1  = bm[(j >> 3) + 1];
        int     nr1 = g_bm_popcnt[(j >> 3) + 1];
        uint8x16x2_t src = {{ L_full, R_full }};
        uint8x8_t shuf1  = vld1_u8(expand_tab_pre[nr0][m1]);
        uint8x8_t o1     = vqtbl2_u8(src, shuf1);
        vst1_u8(out + j + 8, o1);

        rc += nr0 + nr1;
        lc += (16 - nr0 - nr1);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}

static void simd_merge_pcpc_pure (const ctx_t *c){
    /* Caller responsibility: bm_popcnt must already be filled.  Each
       outer rep keeps the same bm, so we let the bench's inner timer
       see only the merge work; fill_bm_popcnt is called once outside
       the timing loop via the correctness path. */
    merge_neon_pcpc(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}
static void simd_merge_pcpc_full (const ctx_t *c){
    fill_bm_popcnt(c->bm, c->n);
    merge_neon_pcpc(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* ---- 8-way unrolled, in-register popcount ----
 *
 * Per outer iter:
 *   - vld1_u8 8 bitmap bytes into bm_vec  (1 load, 8 masks)
 *   - vcnt_u8 produces 8 popcounts in pc_vec (1 op)
 *   - 8 unrolled sub-iters: lane K extracts mask[K] + popcount[K] via
 *     vget_lane_u8 (compile-time index, ~1c), then runs the standard
 *     merge body with 1-mask-per-sub-iter shape.
 *
 * Trade vs neon_pcpc: no temp store, no second-pass load — popcount
 * values stay in vector lanes and get pulled via vget_lane.  Cost: each
 * sub-iter is the simpler 1-mask version, so we load L + R every 8
 * outputs instead of every 16 (DOUBLES L/R load count).  If those loads
 * are not on the critical path the freed popcount dependency may still
 * net out positive. */
static inline void merge_neon_unroll8(const uint8_t *bm, int K,
                                        const uint8_t *left, const uint8_t *right,
                                        uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint8x8_t bm_vec = vld1_u8(bm + (j >> 3));
        uint8x8_t pc_vec = vcnt_u8(bm_vec);
#define UNROLL_STEP(K_) do {                                          \
            uint8_t  m  = vget_lane_u8(bm_vec, K_);                   \
            int      nr = vget_lane_u8(pc_vec, K_);                   \
            uint8x8_t  L = vld1_u8(left  + lc);                       \
            uint8x8_t  R = vld1_u8(right + rc);                       \
            uint8x16_t both = vcombine_u8(L, R);                      \
            uint8x8_t  shuf = vld1_u8(expand_tab[m]);                 \
            uint8x8_t  o    = vqtbl1_u8(both, shuf);                  \
            vst1_u8(out + j + 8 * K_, o);                             \
            rc += nr;                                                  \
            lc += (8 - nr);                                            \
        } while (0)
        UNROLL_STEP(0); UNROLL_STEP(1); UNROLL_STEP(2); UNROLL_STEP(3);
        UNROLL_STEP(4); UNROLL_STEP(5); UNROLL_STEP(6); UNROLL_STEP(7);
#undef UNROLL_STEP
    }
    /* scalar tail for elements past the last 64-aligned boundary */
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}
static void simd_merge_unroll8 (const ctx_t *c){
    merge_neon_unroll8(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}
#endif

#if defined(HAVE_NEON_KERNELS)   /* partition family + unfused comparison (NEON only) */
/* production fused variants */
static void simd_bmbuild (const ctx_t *c){ prim_enc_partition_none(c->la_work, c->n, c->depth, c->bm); }
static void simd_fusedhalf(const ctx_t *c){ prim_enc_partition_right(c->la_work, c->n, c->depth, c->bm, c->tmp16); }
/* non-fused (from prebuilt bm) via the same shared core, BUILD=0 — for the
   unfusing-cost comparison only; not used in production. */
static void simd_partbm  (const ctx_t *c){ part_core_neon(c->la_work, c->n, c->depth, NULL, c->bm, c->tmp16, 0, 1, 1); }
static void simd_parthalf(const ctx_t *c){ part_core_neon(c->la_work, c->n, c->depth, NULL, c->bm, c->tmp16, 0, 1, 0); }
#endif

typedef enum { ST_UNPACK, ST_SCATTER, ST_PACK, ST_MERGE_FLAT, ST_PART,
               ST_BMBUILD, ST_PARTBM, ST_PARTHALF, ST_FUSEDHALF,
               ST_MERGE_GEN, ST_MERGE_TWO, ST_MERGE_CL, ST_MERGE_CR,
               ST_XOR, ST_XOR_ACCUM, ST_PLUS_ONE } stage_t;
typedef struct {
    const char *variant; stage_t stage; int D; int inplace; void (*run)(const ctx_t *);
} prim_t;
static prim_t PRIMS[256]; static int NPRIMS = 0;
static void reg(const char *v, stage_t s, int D, int ip, void (*fn)(const ctx_t *)) {
    PRIMS[NPRIMS++] = (prim_t){ v, s, D, ip, fn };
}
/* Stage label = the .h primitive name (without the prim_ prefix), or the
   internal name for the few stages that have no production analogue
   (part_bm / part_half exist only as instrumentation for the unfused
   partition cost decomposition). */
static const char *stage_name(stage_t s){
    switch(s){
    case ST_UNPACK:     return "flat_dN_unpack";
    case ST_SCATTER:    return "flat_scatter";
    case ST_PACK:       return "enc_pack_dN";
    case ST_MERGE_FLAT: return "merge_flat";
    case ST_PART:       return "enc_partition_full";
    case ST_BMBUILD:    return "enc_partition_none";
    case ST_FUSEDHALF:  return "enc_partition_right";
    case ST_PARTBM:     return "part_bm";        /* no prod primitive */
    case ST_PARTHALF:   return "part_half";      /* no prod primitive */
    case ST_MERGE_GEN:  return "merge";
    case ST_MERGE_TWO:  return "merge_two";
    case ST_MERGE_CL:   return "merge_constant_left";
    case ST_MERGE_CR:   return "merge_constant_right";
    case ST_XOR:        return "xor";
    case ST_XOR_ACCUM:  return "xor_accum";
    case ST_PLUS_ONE:   return "plus_one";
    }
    return "?";
}
/* Per-element accounting (bits): data input, data output, lookup-table
 * load (control inputs like the bitmap or the mask vector treated as
 * "data input").  For partition stages the unit element is one input
 * code (n inputs = n outputs across the L/R split).  For merge / unpack /
 * scatter / merge_flat the unit element is one output byte.  Fractional
 * costs (e.g. 1 byte of expand_popcnt amortized over 8 output bytes =
 * 0.125 bit/elem) are kept; the bandwidth column then reports the
 * effective data-path pressure.
 *
 * Sources for the numbers:
 *   merge / merge_constant_*  -- tree_merge_neon / tree_merge_bcast_*_neon
 *     load expand_tab[mask] (1 byte/output) + expand_popcnt[mask]
 *     (0.125 byte/output) + 1 source byte/output + 1 bm bit/output.
 *   merge_two -- merge_both_const_neon: only consumes 1 bm bit/output;
 *     the bit_pos_tab is a compile-time vector held in a register.
 *   merge_flat (D) -- packed bm stream is D bits/output, the c2s lookup
 *     yields one byte/output (the symbol); the c2s table is held in
 *     registers (D<=6) or via vqtbl2/4 splits but each output still costs
 *     one looked-up byte = 8 bits.
 *   enc_partition_full -- compress_tab[mask] is 32 bytes/8 inputs = 4
 *     byte/elem = 32 bits/elem; +1 byte/8 inputs from compress_popcnt =
 *     1 bit/elem.  Outputs are 1 bm bit + 16 code bits per input (the
 *     code goes to exactly one of L or R) = 17 bits/elem.
 *   enc_partition_none -- builds bm only (no compress_tab load);
 *     out = 1 bit/elem.
 *   enc_partition_right (FUSEDHALF) -- like full but only loads the
 *     right half of compress_tab (16 bytes/8 inputs) and writes only
 *     right + bm.  Average emit is half the input codes worth of bytes;
 *     we report 1 + 16 = 17 (the LOADED right shuffle still uses 16/8 =
 *     2 byte/elem = 16 lut bits).
 *   part_bm / part_half -- unfused: same as enc_partition_full /
 *     _right but ADDITIONALLY load the prebuilt bm (1 bit/elem extra
 *     on the data-input side; lut unchanged).
 *   flat_dN_unpack -- packed input D bits/output; out is the 8-bit code
 *     (0..2^D-1).  No mask-indexed table load.
 *   flat_scatter -- input 8-bit code, output 8-bit symbol; c2s held in
 *     registers (one byte of effective lookup per output = 8 lut bits).
 *   enc_pack_dN -- read u16 code (16 bits/elem), emit D bits/elem;
 *     no table load.
 *
 * For partition stages the "out" count includes both the per-element
 * bitmap bit AND the dense code-byte traffic, so the out_MB/s column
 * reflects total write-port pressure. */
/* Accounting convention: each number is the LOAD/STORE bandwidth from the
 * named stream amortized per element, including the "over-read" /
 * "over-store" that comes from the SIMD primitives always operating on
 * full 16-byte vectors regardless of how much of the vector ends up in
 * the cursor advance.
 *
 * Concretely for merge (tree_merge_neon, 16 outputs per inner iter):
 *   - vld1q_u8(left + lc):  16 bytes loaded / 16 outputs = 1 byte/elem = 8 bits/elem
 *   - vld1q_u8(right + rc): 16 bytes loaded / 16 outputs = 1 byte/elem = 8 bits/elem
 *   - bm[j>>3]:             2 bytes loaded / 16 outputs = 1 bit/elem
 * The cursors `lc` and `rc` only advance by (16 - nr) and nr respectively
 * each iter, totaling 16 bytes of cursor advance per iter -- but we LOAD
 * 32 bytes, so 16 bytes per iter are over-read (the same source bytes get
 * loaded twice across consecutive iters when the partition is skewed).
 *
 * For partition_full (8 codes per inner iter, in-place left write):
 *   - vld1q_u16(codes_la + j):       16 bytes / 8 inputs = 16 bits/elem
 *   - compress_tab[mask] vld1q_u8 x2: 32 bytes / 8 inputs = 32 bits/elem (lut)
 *   - compress_popcnt[mask]:          1 byte  / 8 inputs =  1 bit/elem  (lut)
 *   - vst1q_u8 to right_out:         16 bytes / 8 inputs = 16 bits/elem
 *   - vst1q_u8 to codes_la (left):   16 bytes / 8 inputs = 16 bits/elem
 *   - bm[j>>3] write:                 1 byte  / 8 inputs =  1 bit/elem
 * The cursors n_right and n_left advance by nr*2 and (8-nr)*2 bytes per
 * iter, summing to exactly 16 bytes per iter -- but we WRITE 32 bytes per
 * iter, so 16 bytes/iter are "over-store" (garbage-tail bytes overwritten
 * by the next iter's valid+garbage write).  This is safe by construction:
 * see comment in main() for the n_left <= j invariant proof. */
static void stage_bits(stage_t s, int D, double *in, double *out, double *lut) {
    switch (s) {
    case ST_UNPACK:     *in=D;        *out=8;        *lut=0;       break;
    case ST_SCATTER:    *in=8;        *out=8;        *lut=8;       break;
    case ST_PACK:       *in=16;       *out=D;        *lut=0;       break;
    case ST_MERGE_FLAT: *in=D;        *out=8;        *lut=8;       break;
    case ST_PART:       *in=16;       *out=1+32;     *lut=32+1;    break;
    case ST_BMBUILD:    *in=16;       *out=1;        *lut=0;       break;
    case ST_PARTBM:     *in=16+1;     *out=32;       *lut=32+1;    break;
    case ST_PARTHALF:   *in=16+1;     *out=16;       *lut=16+1;    break;
    case ST_FUSEDHALF:  *in=16;       *out=1+16;     *lut=16+1;    break;
    case ST_MERGE_GEN:  *in=8+8+1;    *out=8;        *lut=8+1;     break;
    case ST_MERGE_TWO:  *in=1;        *out=8;        *lut=0;       break;
    case ST_MERGE_CL:   *in=8+1;      *out=8;        *lut=8+1;     break;
    case ST_MERGE_CR:   *in=8+1;      *out=8;        *lut=8+1;     break;
    case ST_XOR:        *in=16;       *out=8;        *lut=0;       break;
    case ST_XOR_ACCUM:  *in=8;        *out=0;        *lut=0;       break;
    case ST_PLUS_ONE:   *in=0;        *out=8;        *lut=0;       break;
    }
}

int main(int argc, char **argv) {
    int n = 8192, reps = 2000, want[MAXD+1] = {0}, any = 0;
    for (int i=1;i<argc;i++) {
        if      (!strncmp(argv[i],"--n=",4))    n = atoi(argv[i]+4);
        else if (!strncmp(argv[i],"--reps=",7)) reps = atoi(argv[i]+7);
        else if (!strncmp(argv[i],"--D=",4))
            for (char *t=strtok(argv[i]+4,","); t; t=strtok(NULL,",")) {
                int d=atoi(t); if (d>=2&&d<=MAXD){want[d]=1;any=1;} }
    }
    n &= ~15;
    if (!any) for (int d=2;d<=MAXD;d++) want[d]=1;

#if defined(HAVE_SIMD)
    prim_codec_init();   /* build the backend's partition/merge tables */
#endif
    uint8_t  *bm = malloc(n+16), *codes = malloc(n+16), *out = malloc(n+16);
    uint8_t  *ref = malloc(n+16), *pack_out = malloc(n+16);
    uint8_t  *merge_left = malloc(n+16), *merge_right = malloc(n+16);
    uint16_t *la_pristine = malloc((n+16)*2), *la_work = malloc((n+16)*2),
             *tmp16 = malloc((n+16)*2), *ref16l = malloc((n+16)*2), *ref16r = malloc((n+16)*2);
    uint8_t   c2s[256], ref_bm[ (8192/8) + 64 ];   /* 2^8 entries (D up to 8) */
    srand(0xC0FFEE);
    for (int i=0;i<n+16;i++){ bm[i]=(uint8_t)rand(); la_pristine[i]=(uint16_t)rand(); }
    for (int i=0;i<n+16;i++){ merge_left[i]=(uint8_t)rand(); merge_right[i]=(uint8_t)rand(); }
    for (int i=0;i<256;i++) c2s[i]=(uint8_t)rand();

    for (int d=2; d<=MAXD; d++) {
        if (!want[d]) continue;
        /* scalar + its SIMD peer adjacent per (D,stage).  NEON unpack now has
           a D7 kernel (D<=7); scatter/pack/merge flat kernels still cap at D6
           (D7 scatter needs a 128-entry table beyond TBL's reach). */
        /* unpack/scatter: standalone NEON kernels (D2..7).  pack/merge: the
           production prim_* on every backend (SIMD where the backend handles
           D, scalar fallback otherwise) -- so registered for all D. */
        reg("scalar",ST_UNPACK, d,0,p_unpack_scalar);
#if defined(HAVE_NEON_KERNELS)
        if (d <= 7) reg(BK,ST_UNPACK, d,0,neon_unpack);
#endif
        reg("scalar",ST_SCATTER,d,0,p_scatter_scalar);
#if defined(HAVE_NEON_KERNELS)
        if (d <= 7) reg(BK,ST_SCATTER,d,0,neon_scatter);
#endif
        reg("scalar",ST_PACK,   d,0,p_pack_scalar);
#if defined(HAVE_SIMD)
        reg(BK,ST_PACK,   d,0,simd_pack);
#endif
        reg("scalar",ST_MERGE_FLAT,  d,0,p_merge_scalar);
#if defined(HAVE_SIMD)
        reg(BK,ST_MERGE_FLAT,  d,0,simd_merge_flat);
#endif
    }
    reg("scalar",ST_PART,0,1,p_part_scalar);
#if defined(HAVE_SIMD)
    reg(BK,      ST_PART,0,1,simd_part);
#endif
#if defined(HAVE_NEON_KERNELS)
    /* Unfused decomposition: fused part == bm_build + part_bm (re-read cost).
       part_half == HALF-node saving (one-sided scatter). */
    reg(BK, ST_BMBUILD, 0,0, simd_bmbuild);
    reg(BK, ST_PARTBM,  0,1, simd_partbm);
    reg(BK, ST_PARTHALF,0,0, simd_parthalf);
    reg(BK, ST_FUSEDHALF,0,0, simd_fusedhalf);
#endif
    /* Binary-merge family: production prims used at every internal node in
       the bottom-up codec.  Same bm consumption pattern as merge_flat D=1
       but the symbol(s) come from buffer reads or scalar constants
       instead of a c2s lookup.  Scalar refs + production SIMD on all
       backends; the pcpc/unroll8 experimental variants are NEON-only. */
    reg("scalar", ST_MERGE_GEN, 0, 0, p_merge_gen_scalar);
    reg("scalar", ST_MERGE_TWO, 0, 0, p_merge_two_scalar);
    reg("scalar", ST_MERGE_CL,  0, 0, p_merge_const_l_scalar);
    reg("scalar", ST_MERGE_CR,  0, 0, p_merge_const_r_scalar);
#if defined(HAVE_SIMD)
    reg(BK,       ST_MERGE_GEN, 0, 0, simd_merge_gen);
    reg(BK,       ST_MERGE_TWO, 0, 0, simd_merge_two);
    reg(BK,       ST_MERGE_CL,  0, 0, simd_merge_const_l);
    reg(BK,       ST_MERGE_CR,  0, 0, simd_merge_const_r);
#endif
#if defined(HAVE_NEON_KERNELS)
    reg("neon_pcpc",     ST_MERGE_GEN, 0, 0, simd_merge_pcpc_pure);
    reg("neon_pcpc_full",ST_MERGE_GEN, 0, 0, simd_merge_pcpc_full);
    reg("neon_unroll8",  ST_MERGE_GEN, 0, 0, simd_merge_unroll8);
#endif
    /* Synthetic byte-XOR — memory-bandwidth comparison point. */
    reg("scalar", ST_XOR, 0, 0, p_xor_scalar);
#if defined(HAVE_SIMD)
    reg(BK,       ST_XOR, 0, 0, simd_xor);
#endif
    /* Max-in-bw probe: read-heavy XOR fold, single-byte write at end. */
    reg("scalar", ST_XOR_ACCUM, 0, 0, p_xor_accum_scalar);
#if defined(HAVE_SIMD)
    reg(BK,       ST_XOR_ACCUM, 0, 0, simd_xor_accum);
#endif
    /* Max-out-bw probe: write-only iota pattern. */
    reg("scalar", ST_PLUS_ONE, 0, 0, p_plus_one_scalar);
#if defined(HAVE_SIMD)
    reg(BK,       ST_PLUS_ONE, 0, 0, simd_plus_one);
#endif

    printf("bench_prim: n=%d elems, best-of-9 x %d reps, partition depth=%d\n",
           n, reps, PART_DEPTH);
    printf("%-22s %-3s %-9s %10s  %5s %5s %5s   %6s %6s %6s  %s\n",
           "stage","D","variant","ns/elem",
           "in_b","out_b","lut_b", "in_MB/s","out_MB/s","lut_MB/s","check");
    volatile uint8_t sink = 0; int prevD=-99; stage_t prevS=-1;

    for (int k=0;k<NPRIMS;k++) {
        prim_t *p = &PRIMS[k];
        ctx_t cx = { bm, codes, c2s, out, pack_out, merge_left, merge_right,
                     la_work, tmp16, n, p->D, PART_DEPTH };
        const char *chk = "ok";

        /* per-stage input prep + correctness vs scalar reference */
        if (p->stage == ST_SCATTER) {
            uint32_t m=(1u<<p->D)-1u; for (int i=0;i<n;i++) codes[i]=(uint8_t)(rand()&m);
            scalar_scatter(ref,codes,c2s,n); memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_UNPACK) {
            scalar_unpack(ref,bm,n,p->D); memset(codes,0,n); p->run(&cx);
            if (memcmp(codes,ref,n)) chk="FAIL";
        } else if (p->stage == ST_PACK) {
            memcpy(la_work,la_pristine,(size_t)n*2);
            scalar_pack(ref,la_pristine,n,p->D,PART_DEPTH); memset(pack_out,0,n); p->run(&cx);
            if (memcmp(pack_out,ref,(n*p->D+7)>>3)) chk="FAIL";
        } else if (p->stage == ST_MERGE_FLAT) {
            scalar_unpack(codes,bm,n,p->D); scalar_scatter(ref,codes,c2s,n);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_MERGE_GEN) {
#if defined(HAVE_NEON_KERNELS)
            /* Populate g_bm_popcnt once per row so the _pcpc_pure variant
               sees an up-to-date table; harmless for other variants. */
            fill_bm_popcnt(bm, n);
#endif
            scalar_merge_gen(ref, bm, n, merge_left, merge_right);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_MERGE_TWO) {
            scalar_merge_two(ref, bm, n, MERGE_LEFT_SYM, MERGE_RIGHT_SYM);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_MERGE_CL) {
            scalar_merge_const_l(ref, bm, n, MERGE_LEFT_SYM, merge_right);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_MERGE_CR) {
            scalar_merge_const_r(ref, bm, n, merge_left, MERGE_RIGHT_SYM);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_XOR) {
            scalar_xor(ref, merge_left, merge_right, n);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_XOR_ACCUM) {
            scalar_xor_accum(ref, merge_left, n);
            memset(out,0,1); p->run(&cx);
            if (out[0] != ref[0]) chk="FAIL";
        } else if (p->stage == ST_PLUS_ONE) {
            scalar_plus_one(ref, n);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_BMBUILD) {
            scalar_partition(la_pristine,n,PART_DEPTH,ref_bm,ref16l,ref16r);
            memcpy(la_work,la_pristine,(size_t)n*2); p->run(&cx);
            if (memcmp(bm,ref_bm,(n+7)>>3)) chk="FAIL";
        } else if (p->stage == ST_PARTBM) {
            int nr_ref = scalar_partition(la_pristine,n,PART_DEPTH,ref_bm,ref16l,ref16r);
            int nl_ref = n - nr_ref;
            memcpy(bm,ref_bm,(size_t)((n+7)>>3));        /* prebuilt bitmap */
            memcpy(la_work,la_pristine,(size_t)n*2); p->run(&cx);
            if (memcmp(la_work,ref16l,(size_t)nl_ref*2)
                || memcmp(tmp16,ref16r,(size_t)nr_ref*2)) chk="FAIL";
        } else if (p->stage == ST_PARTHALF) {
            int nr_ref = scalar_partition(la_pristine,n,PART_DEPTH,ref_bm,ref16l,ref16r);
            memcpy(bm,ref_bm,(size_t)((n+7)>>3));        /* prebuilt bitmap */
            memcpy(la_work,la_pristine,(size_t)n*2); p->run(&cx);
            if (memcmp(tmp16,ref16r,(size_t)nr_ref*2)) chk="FAIL";
        } else if (p->stage == ST_FUSEDHALF) {
            int nr_ref = scalar_partition(la_pristine,n,PART_DEPTH,ref_bm,ref16l,ref16r);
            memcpy(la_work,la_pristine,(size_t)n*2); p->run(&cx);
            if (memcmp(bm,ref_bm,(n+7)>>3) || memcmp(tmp16,ref16r,(size_t)nr_ref*2)) chk="FAIL";
        } else { /* partition (fused) */
            int nr_ref = scalar_partition(la_pristine,n,PART_DEPTH,ref_bm,ref16l,ref16r);
            memcpy(la_work,la_pristine,(size_t)n*2); p->run(&cx);
            /* prim writes left->la_work, right->tmp16, bm->bm; scalar variant
               also leaves left in la_work + right in tmp16. */
            int nl_ref = n - nr_ref;
            if (memcmp(bm,ref_bm,(n+7)>>3) || memcmp(la_work,ref16l,(size_t)nl_ref*2)
                || memcmp(tmp16,ref16r,(size_t)nr_ref*2)) chk="FAIL";
        }

        double best = 1e30;
        for (int s=0;s<9;s++) {
            double t0 = now_ns();
            if (p->inplace) for (int r=0;r<reps;r++){ memcpy(la_work,la_pristine,(size_t)n*2); p->run(&cx); }
            else            for (int r=0;r<reps;r++) p->run(&cx);
            double e = now_ns()-t0;
            if (p->inplace) {  /* subtract the per-rep memcpy baseline */
                double b0=now_ns();
                for (int r=0;r<reps;r++){ memcpy(la_work,la_pristine,(size_t)n*2); sink^=la_work[0]; }
                e -= (now_ns()-b0);
            }
            e /= (double)reps * n;
            if (e>0 && e<best) best = e;
        }
        sink ^= out[0]^codes[0]^pack_out[0]^bm[0];

        if (p->D!=prevD || p->stage!=prevS) printf("\n");
        prevD=p->D; prevS=p->stage;
        char dbuf[8]; if (p->D==0) strcpy(dbuf,"-"); else snprintf(dbuf,8,"%d",p->D);
        double in_b=0, out_b=0, lut_b=0;
        stage_bits(p->stage, p->D, &in_b, &out_b, &lut_b);
        /* MB/s = bits/elem * 125 / ns/elem.  Derivation: bytes/sec =
           (bits/8) / (ns*1e-9); divide by 1e6 -> bits * 125 / ns. */
        double in_bw  = best > 0 ? in_b  * 125.0 / best : 0;
        double out_bw = best > 0 ? out_b * 125.0 / best : 0;
        double lut_bw = best > 0 ? lut_b * 125.0 / best : 0;
        printf("%-22s %-3s %-9s %10.3g  %5.2f %5.2f %5.2f   %6.0f %6.0f %6.0f  %s\n",
               stage_name(p->stage), dbuf, p->variant, best,
               in_b, out_b, lut_b, in_bw, out_bw, lut_bw, chk);
    }
    (void)sink;
    free(merge_left); free(merge_right);
    return 0;
}
