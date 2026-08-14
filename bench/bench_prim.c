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
 *   pack      — ranks[] -> packed N*D-bit stream  (production prim_enc_pack_dN)
 *   partition — ranks[] + split_rank -> bitmap + left/right split (prim_enc_partition)
 *
 * The retired u16 (code_la) encode is benched alongside under u16enc_* rows.
 *
 * unpack/scatter/merge/pack are per-depth-D; partition is a 1-bit split (one
 * representative depth).  Every SIMD variant is checked against a scalar
 * reference before timing.  Metric: ns/elem (4 decimal places).
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
/* The primitives router self-selects the backend from the build arch
 * (see pivco_huffman_primitives.h); this block only derives the local
 * HAVE_SIMD / BK conveniences and the PIVCO_HAS_* tier defines the
 * included research headers expect when built standalone. */
#if defined(__aarch64__)
#  define HAVE_SIMD 1
#  define USE_NEON_KERNELS 1
#elif defined(__x86_64__)
#  if defined(__SSE4_1__)
#    define PIVCO_HAS_SSE4 1
#    define HAVE_SIMD 1
#  endif
#  if defined(__AVX2__)
#    define PIVCO_HAS_AVX2 1
#  endif
#  if defined(__AVX512VBMI2__)
#    define PIVCO_HAS_AVX512 1
#  endif
#endif
#include "pivco_huffman.h"
#include "bench_canary.h"
#if defined(HAVE_SIMD)
#  include "pivco_huffman_primitives.h"   /* backend self-selected; BK below */
#  define BK PIVCO_PRIM_BACKEND_NAME
   /* prim_enc_* (u8 rank) / prim_merge_flat
                                             (+ NEON flat_dN) */
   /* Retired u16 (code_la) encode kernels (owned by ph-td; production encodes on
      u8 ranks).  Provides the u16enc_* rows + enc_mask8_codes_la_* used by
      the adapters below and by prims-partition.h.  Needs the shared header above
      first (compress_tab / *_pack.h helpers / PIVCO_PRIM_ALWAYS_INLINE). */
#  include "pivco_huffman_u16enc.h"
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
static void scalar_merge_vec_vec(uint8_t *out, const uint8_t *bm, int n,
                              const uint8_t *L, const uint8_t *R) {
    int lc=0, rc=0;
    for (int i=0; i<n; i++) {
        int b = (bm[i>>3] >> (i&7)) & 1;
        out[i] = b ? R[rc++] : L[lc++];
    }
}
static void scalar_merge_cst_cst(uint8_t *out, const uint8_t *bm, int n,
                              uint8_t l, uint8_t r) {
    for (int i=0; i<n; i++) {
        int b = (bm[i>>3] >> (i&7)) & 1;
        out[i] = b ? r : l;
    }
}
static void scalar_merge_cst_vec(uint8_t *out, const uint8_t *bm, int n,
                                  uint8_t l, const uint8_t *R) {
    int rc=0;
    for (int i=0; i<n; i++) {
        int b = (bm[i>>3] >> (i&7)) & 1;
        out[i] = b ? R[rc++] : l;
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
    uint8_t  *merge_left, *merge_right;   /* dense byte sources for prim_merge_vec_vec */
    uint16_t *la_work, *tmp16;            /* u16 (code_la) encode: work buf + right scratch */
    /* u8 (rank) encode buffers. */
    uint8_t  *symbuf;                     /* input symbols (gather source)               */
    uint8_t  *ranks;                      /* gathered ranks, kept pristine for re-runs    */
    uint8_t  *ranks_work;                 /* partitioned in place (reset from ranks/rep)  */
    uint8_t  *tmp8;                       /* right-side scatter target                     */
    const uint16_t *code_la_lut;          /* sym -> u16 left-aligned code (u16enc gather) */
    const uint8_t  *sym_to_rank;          /* sym -> u8 rank              (prim_enc gather) */
    pivco_enc_init_aux_t enc_init_aux; /* pre-shifted rank tables (x86 prim_enc 4tab) */
    uint8_t  rank_thr, rank_base;
    uint32_t *hist;                       /* prim_histogram_chunk output bins   */
    uint8_t  *hist_scratch;               /* PIVCO_PRIM_HIST_SCRATCH bytes      */
    int n, D, depth;
} ctx_t;

#ifdef HAVE_SIMD
/* "scalar" row = the PRODUCTION shared core (histogram_chunk_scalar,
 * reachable from every backend build via the primitives chain) -- not a
 * naive reimplementation; the independent naive loop lives only in the
 * correctness check.  The BK row is the backend's prim alias (differs
 * from the core only on AVX-512). */
static void p_hist_scalar(const ctx_t *c){
    histogram_chunk_scalar(c->symbuf, (size_t)c->n, c->hist, c->hist_scratch);
}
static void p_hist_prim(const ctx_t *c){
    prim_histogram_chunk(c->symbuf, (size_t)c->n, c->hist, c->hist_scratch);
}
#endif
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
/* rank-based scalar reference implementations, used to verify the SIMD output */
static void scalar_rank_init(uint8_t *ranks, int n, const uint8_t *sym, const uint8_t *s2r) {
    for (int i = 0; i < n; i++) ranks[i] = s2r[sym[i]];
}
static int scalar_rank_partition(const uint8_t *ranks, int n, uint8_t thr,
                                 uint8_t *bm, uint8_t *left, uint8_t *right) {
    int nl = 0, nr = 0;
    memset(bm, 0, (size_t)((n + 7) >> 3));
    for (int j = 0; j < n; j++) {
        uint8_t v = ranks[j];
        if (v > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); right[nr++] = v; }
        else         { left[nl++] = v; }
    }
    return nr;
}
static void scalar_rank_pack(uint8_t *out, const uint8_t *ranks, int n, int D, uint8_t base) {
    uint32_t mask = (1u << D) - 1; uint64_t buf = 0; int bib = 0, bi = 0;
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)(uint8_t)(ranks[i] - base) & mask;
        buf |= (uint64_t)v << bib; bib += D;
        while (bib >= 8) { out[bi++] = (uint8_t)buf; buf >>= 8; bib -= 8; }
    }
    if (bib > 0) out[bi] = (uint8_t)(buf & ((1u << bib) - 1));
}

/* scalar refs for the new binary-merge stages */
static void p_merge_vec_vec_scalar     (const ctx_t *c){ scalar_merge_vec_vec(c->out, c->bm, c->n, c->merge_left, c->merge_right); }
static void p_merge_cst_cst_scalar     (const ctx_t *c){ scalar_merge_cst_cst(c->out, c->bm, c->n, MERGE_LEFT_SYM, MERGE_RIGHT_SYM); }
static void p_merge_cst_vec_scalar (const ctx_t *c){ scalar_merge_cst_vec(c->out, c->bm, c->n, MERGE_LEFT_SYM, c->merge_right); }
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

#if defined(USE_NEON_KERNELS)   /* standalone unpack/scatter: NEON intrinsics */
static void neon_unpack(const ctx_t *c) {
    int n = c->n; uint8_t *cd = c->codes; const uint8_t *bm = c->bm;
    switch (c->D) {
    case 2: for (int i=0;i<n;i+=16) vst1q_u8(cd+i, flat_d2_unpack(bm + (i>>4)*4)); break;
    case 3: for (int i=0;i<n;i+= 8) vst1_u8 (cd+i, flat_d3_unpack_safe(bm + (i>>3)*3)); break;
    case 4: for (int i=0;i<n;i+=16) vst1q_u8(cd+i, flat_d4_unpack(bm + (i>>4)*8)); break;
    case 5: for (int i=0;i<n;i+= 8) vst1_u8 (cd+i, flat_d5_unpack_safe(bm + (i>>3)*5)); break;
    case 6: for (int i=0;i<n;i+= 8) vst1_u8 (cd+i, flat_d6_unpack_safe(bm + (i>>3)*6)); break;
    case 7: for (int i=0;i<n;i+= 8) vst1_u8 (cd+i, flat_d7_unpack_safe(bm + (i>>3)*7)); break;
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
#endif /* USE_NEON_KERNELS */

#if defined(PIVCO_HAS_AVX512)   /* standalone unpack: production avx512_flat helpers */
/* Pure-unpack rows: each case is the matching merge_flat_dN_avx512 loop with
   the c2s vpermb removed -- codes are stored, not mapped.  Loop bounds (the
   128-slack wide loop, fast/safe 16-wide tails) are copied verbatim from
   pivco_huffman_primitives_avx512.h so the bm over-read guarantees are
   identical; the helpers themselves come from pivco_huffman_avx512_flat.h. */
static void avx512_unpack(const ctx_t *c) {
    int n = c->n, i = 0; uint8_t *cd = c->codes; const uint8_t *bm = c->bm;
    switch (c->D) {
    case 2:
        for (; i + 128 <= n; i += 64)
            _mm512_storeu_si512((__m512i *)(cd + i), flat_d2_unpack64_avx512_fast(bm + (i >> 2)));
        for (; i + 16 <= n; i += 16)
            _mm_storeu_si128((__m128i *)(cd + i), flat_d2_unpack_avx512(bm + (i >> 2)));
        break;
    case 3: {
        for (; i + 128 <= n; i += 64)
            _mm512_storeu_si512((__m512i *)(cd + i), flat_d3_unpack64_avx512_fast(bm + ((i * 3) >> 3)));
        int fe = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fe; i += 16)
            _mm_storeu_si128((__m128i *)(cd + i), flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3)));
        if (i + 16 <= n) {
            _mm_storeu_si128((__m128i *)(cd + i), flat_d3_unpack_avx512_safe(bm + ((i * 3) >> 3)));
            i += 16;
        }
        break;
    }
    case 4:
        for (; i + 128 <= n; i += 64)
            _mm512_storeu_si512((__m512i *)(cd + i), flat_d4_unpack64_avx512_fast(bm + ((i * 4) >> 3)));
        for (; i + 16 <= n; i += 16)
            _mm_storeu_si128((__m128i *)(cd + i), flat_d4_unpack_avx512(bm + (i >> 1)));
        break;
    case 5: {
        int f64 = n >= 64 ? n - 64 : 0;
        for (; i + 64 <= f64; i += 64)
            _mm512_storeu_si512((__m512i *)(cd + i), flat_d5_unpack64_avx512_fast(bm + ((i * 5) >> 3)));
        int fe = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fe; i += 16)
            _mm_storeu_si128((__m128i *)(cd + i), flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3)));
        if (i + 16 <= n) {
            _mm_storeu_si128((__m128i *)(cd + i), flat_d5_unpack_avx512_safe(bm + ((i * 5) >> 3)));
            i += 16;
        }
        break;
    }
    case 6: {
        for (; i + 128 <= n; i += 64)
            _mm512_storeu_si512((__m512i *)(cd + i), flat_d6_unpack64_avx512_fast(bm + ((i * 6) >> 3)));
        int fe = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fe; i += 16)
            _mm_storeu_si128((__m128i *)(cd + i), flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3)));
        if (i + 16 <= n) {
            _mm_storeu_si128((__m128i *)(cd + i), flat_d6_unpack_avx512_safe(bm + ((i * 6) >> 3)));
            i += 16;
        }
        break;
    }
    case 7: {
        for (; i + 128 <= n; i += 64)
            _mm512_storeu_si512((__m512i *)(cd + i), flat_d7_unpack64_avx512_fast(bm + ((i * 7) >> 3)));
        int fe = n >= 16 ? n - 16 : 0;
        for (; i + 16 <= fe; i += 16)
            _mm_storeu_si128((__m128i *)(cd + i), flat_d7_unpack_avx512_fast(bm + ((i * 7) >> 3)));
        if (i + 16 <= n) {
            _mm_storeu_si128((__m128i *)(cd + i), flat_d7_unpack_avx512_safe(bm + ((i * 7) >> 3)));
            i += 16;
        }
        break;
    }
    }
    for (; i < n; i++) cd[i] = (uint8_t)extract_D_bits_avx512(bm, i * c->D, c->D);
}
#endif /* PIVCO_HAS_AVX512 */

#if defined(HAVE_SIMD)   /* pack/merge/partition: production prim_*, all backends */
static void simd_merge_flat(const ctx_t *c){ prim_merge_flat(c->out, c->n, c->bm, c->D, c->c2s); }
/* rank-based production primitives */
static void simd_enc_init(const ctx_t *c){ prim_enc_init(c->ranks_work, c->n, c->symbuf, c->sym_to_rank, &c->enc_init_aux); }
static void simd_part    (const ctx_t *c){ prim_enc_partition_full(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }
static void simd_part_right(const ctx_t *c){ prim_enc_partition_right(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }
static void simd_pack    (const ctx_t *c){ prim_enc_pack_dN(c->ranks, c->n, c->D, c->rank_base, c->pack_out); }
/* retired u16 encode (benched alongside for A/B) */
static void simd_u16pack(const ctx_t *c){ u16enc_pack_dN(c->la_work, c->n, c->D, c->depth, c->pack_out); }
static void simd_u16part(const ctx_t *c){ u16enc_partition_full(c->la_work, c->n, c->depth, c->bm, c->tmp16); }
static void simd_u16enc_init(const ctx_t *c){ u16enc_init(c->la_work, c->n, c->symbuf, c->code_la_lut); }
#endif
/* scalar refs for enc_init (overwrite output each call; not inplace) */
static void p_enc_init_scalar      (const ctx_t *c){ for(int i=0;i<c->n;i++) c->la_work[i]    = c->code_la_lut[c->symbuf[i]]; }
static void p_enc_init_rank_scalar (const ctx_t *c){ scalar_rank_init(c->ranks_work, c->n, c->symbuf, c->sym_to_rank); }
static void p_pack_rank_scalar     (const ctx_t *c){ scalar_rank_pack(c->pack_out, c->ranks, c->n, c->D, c->rank_base); }
static void p_part_rank_scalar     (const ctx_t *c){ scalar_rank_partition(c->ranks_work, c->n, c->rank_thr, c->bm, c->ranks_work, c->tmp8); }
static void p_part_right_rank_scalar(const ctx_t *c){
    const uint8_t *r = c->ranks_work; uint8_t *bm = c->bm, *tmp = c->tmp8;
    int n = c->n, nr = 0; uint8_t thr = c->rank_thr;
    memset(bm, 0, (size_t)((n + 7) >> 3));
    for (int j = 0; j < n; j++) { uint8_t v = r[j]; if (v > thr) { bm[j>>3] |= (uint8_t)(1u<<(j&7)); tmp[nr++] = v; } }
}
#if defined(HAVE_SIMD)   /* binary-merge production primitives — all backends */
static void simd_merge_vec_vec     (const ctx_t *c){ prim_merge_vec_vec(c->bm, c->n, c->merge_left, c->merge_right, c->out); }
static void simd_merge_cst_cst     (const ctx_t *c){ prim_merge_cst_cst(c->bm, c->n, MERGE_LEFT_SYM, MERGE_RIGHT_SYM, c->out); }
static void simd_merge_cst_vec (const ctx_t *c){ prim_merge_cst_vec(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out); }
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


#if defined(USE_NEON_KERNELS)   /* partition family + unfused comparison (NEON only) */
/* u16 partition variants for the fused/unfused cost decomposition: bmbuild
 * builds the bitmap only (no scatter); fusedhalf builds + scatters one side. */
static void simd_u16bmbuild (const ctx_t *c){ u16enc_partition_none(c->la_work, c->n, c->depth, c->bm); }
static void simd_u16fusedhalf(const ctx_t *c){ u16enc_partition_right(c->la_work, c->n, c->depth, c->bm, c->tmp16); }
/* non-fused (from prebuilt bm) via the same shared core, BUILD=0 — for the
   unfusing-cost comparison only; not used in production. */
static void simd_u16partbm  (const ctx_t *c){ u16part_core_neon(c->la_work, c->n, c->depth, NULL, c->bm, c->tmp16, 0, 1, 1); }
static void simd_u16parthalf(const ctx_t *c){ u16part_core_neon(c->la_work, c->n, c->depth, NULL, c->bm, c->tmp16, 0, 1, 0); }
#endif

typedef enum {
    /* ---- flat-subtree decode (per depth-D) ---- */
    ST_UNPACK,        /* flat_dN_unpack: packed D-bit stream -> codes[]            */
    ST_SCATTER,       /* flat_scatter:   codes[] + c2s[2^D]  -> out[]             */
    ST_MERGE_FLAT,    /* merge_flat:     packed stream + c2s -> out[] (fused)     */

    /* ---- production encode (u8 in-order ranks, partbyrank) ---- */
    ST_ENC_INIT,      /* prim_enc_init:           symbols -> ranks[] (gather)      */
    ST_PART,          /* prim_enc_partition_full: ranks -> bitmap + L/R split      */
    ST_PART_RIGHT,    /* prim_enc_partition_right: ranks -> bitmap + right scatter  */
    ST_PACK,          /* prim_enc_pack_dN:        ranks -> packed D-bit stream     */

    /* ---- retired u16 (code_la) encode — bench-only, alongside the u8 rows ---- */
    ST_U16_ENC_INIT,  /* u16enc_init           (vs ST_ENC_INIT)                    */
    ST_U16_PART,      /* u16enc_partition_full (vs ST_PART)                        */
    ST_U16_PACK,      /* u16enc_pack_dN        (vs ST_PACK)                        */

    /* ---- u16 partition cost-decomposition (NEON instrumentation) ----
     * Splits the fused partition (build bitmap + scatter) into parts to size
     * the bitmap re-read cost; no standalone production primitive. */
    ST_BMBUILD,       /* u16enc_partition_none:  build bitmap only, no scatter     */
    ST_FUSEDHALF,     /* u16enc_partition_right: build + one-sided scatter         */
    ST_PARTBM,        /* scatter both sides from a prebuilt bitmap                 */
    ST_PARTHALF,      /* scatter one side from a prebuilt bitmap                   */

    /* ---- binary merge (decode, one per internal node) ---- */
    ST_MERGE_VEC_VEC, ST_MERGE_CST_CST, ST_MERGE_CST_VEC,

    /* ---- encode-side histogram (prim_histogram_chunk) ---- */
    ST_HISTOGRAM,

    /* ---- misc primitives ---- */
    ST_XOR, ST_XOR_ACCUM, ST_PLUS_ONE,
} stage_t;
#include "prim_variants/prims.h"   /* pv_isa_t + PV_VARIANT (needs stage_t) */
typedef struct {
    const char       *variant;
    stage_t           stage;
    int               D;
    int               inplace;
    void            (*run)(const ctx_t *);
    /* prim_variants metadata (NULL/0 for production + scalar-reference rows) */
    pv_isa_t          isa;
    const char       *origin;
    const char       *note;
} prim_t;
static prim_t PRIMS[1024]; static int NPRIMS = 0;
static void reg(const char *v, stage_t s, int D, int ip, void (*fn)(const ctx_t *)) {
    if (NPRIMS >= (int)(sizeof PRIMS / sizeof PRIMS[0])) {
        fprintf(stderr, "bench_prim: PRIMS[] overflow at '%s' — raise the cap\n", v);
        exit(1);
    }
    PRIMS[NPRIMS++] = (prim_t){ .variant=v, .stage=s, .D=D, .inplace=ip, .run=fn };
}
/* Graveyard variant families (bench-only; ctx_t / compress_tab /
   enc_mask8_codes_la_neon (from pivco_huffman_u16enc.h above) all in scope). */
#include "prim_variants/prims-partition.h"
#include "prim_variants/prims-merge.h"
#include "prim_variants/prims-flat.h"
#include "prim_variants/prims-pack.h"
#include "prim_variants/prims-init.h"
/* Stage label = the .h primitive name (without the prim_ prefix), or the
   internal name for the few stages that have no production analogue
   (part_bm / part_half exist only as instrumentation for the unfused
   partition cost decomposition). */
static const char *stage_name(stage_t s){
    switch(s){
    case ST_UNPACK:     return "flat_dN_unpack";
    case ST_SCATTER:    return "flat_scatter";
    case ST_U16_PACK:       return "u16enc_pack_dN";
    case ST_MERGE_FLAT: return "merge_flat";
    case ST_U16_PART:       return "u16enc_partition_full";
    case ST_BMBUILD:    return "u16enc_partition_none";
    case ST_FUSEDHALF:  return "u16enc_partition_right";
    case ST_PARTBM:     return "part_bm";        /* no prod primitive */
    case ST_PARTHALF:   return "part_half";      /* no prod primitive */
    case ST_MERGE_VEC_VEC:  return "merge_vec_vec";
    case ST_MERGE_CST_CST:  return "merge_cst_cst";
    case ST_MERGE_CST_VEC:   return "merge_cst_vec";
    case ST_HISTOGRAM:  return "histogram_chunk";
    case ST_XOR:        return "xor";
    case ST_XOR_ACCUM:  return "xor_accum";
    case ST_PLUS_ONE:   return "plus_one";
    case ST_U16_ENC_INIT:      return "u16enc_init";
    case ST_ENC_INIT: return "enc_init";
    case ST_PART:     return "enc_partition_full";
    case ST_PART_RIGHT: return "enc_partition_right";
    case ST_PACK:     return "enc_pack_dN";
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
 *   merge / merge_constant_*  -- merge_vec_vec_neon / merge_cst_vec_neon
 *     load expand_tab[mask] (1 byte/output) + expand_popcnt[mask]
 *     (0.125 byte/output) + 1 source byte/output + 1 bm bit/output.
 *   merge_two -- merge_cst_cst_neon: only consumes 1 bm bit/output;
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
 * Concretely for merge_vec_vec (merge_vec_vec_neon, 16 outputs per inner iter):
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
    case ST_U16_PACK:       *in=16;       *out=D;        *lut=0;       break;
    case ST_MERGE_FLAT: *in=D;        *out=8;        *lut=8;       break;
    case ST_U16_PART:       *in=16;       *out=1+32;     *lut=32+1;    break;
    case ST_BMBUILD:    *in=16;       *out=1;        *lut=0;       break;
    case ST_PARTBM:     *in=16+1;     *out=32;       *lut=32+1;    break;
    case ST_PARTHALF:   *in=16+1;     *out=16;       *lut=16+1;    break;
    case ST_FUSEDHALF:  *in=16;       *out=1+16;     *lut=16+1;    break;
    case ST_MERGE_VEC_VEC:  *in=8+8+1;    *out=8;        *lut=8+1;     break;
    case ST_MERGE_CST_CST:  *in=1;        *out=8;        *lut=0;       break;
    case ST_MERGE_CST_VEC:   *in=8+1;      *out=8;        *lut=8+1;     break;
    case ST_HISTOGRAM:       *in=8;        *out=0;        *lut=0;       break;
    case ST_XOR:        *in=16;       *out=8;        *lut=0;       break;
    case ST_XOR_ACCUM:  *in=8;        *out=0;        *lut=0;       break;
    case ST_PLUS_ONE:   *in=0;        *out=8;        *lut=0;       break;
    /* rank-based encoding (counts are bits/elem, like the stages above: a byte
     * symbol/rank is 8, a u16 code is 16, a bitmap bit is 1). */
    case ST_U16_ENC_INIT:      *in=8;        *out=16;       *lut=16;      break;  /* sym->u16 code_la */
    case ST_ENC_INIT: *in=8;        *out=8;        *lut=8;       break;  /* sym->u8 rank */
    case ST_PART:     *in=8;        *out=8+1;      *lut=16+1;    break;  /* u8 in/out + bitmap */
    case ST_PART_RIGHT: *in=8;      *out=8+1;      *lut=16+1;    break;  /* one-sided: bitmap + right */
    case ST_PACK:     *in=8;        *out=D;        *lut=0;       break;
    }
}

static void usage(FILE *f) {
    fprintf(f,
        "usage: pivco_bench_prim [--n=N] [--reps=N] [--D=d,d,...] [--variants[=<logical>]]\n"
        "\n"
        "  --n=N            elements per run (default 32768; rounded down to mult of 16)\n"
        "  --reps=N         inner reps per timed run (default 2000)\n"
        "  --canary=X       env-stability probe (compute/bw/lat); 0=off,\n"
        "                   1=start+end (default), 2=+between families, 3=+between each primitive\n"
        "  --bm=MODE        bitmap content for the merge stages: random (default),\n"
        "                   runs[:W,B] -- alternating 0-runs (mean W, default 400) and\n"
        "                   1-runs (mean B, default 16), calgary-pic-like run structure,\n"
        "                   mix[:P] -- P%% of u64 words all-zero in random order\n"
        "                   (default 50 = max entropy for a per-word easy/hard branch),\n"
        "                   or p:P -- i.i.d. bits, P%% ones density (proba80 ~ p:20)\n"
        "                   (default 50 = max entropy for a per-word easy/hard branch)\n"
        "  --D=d,d,...      restrict to these flat depths D (2..%d; default all)\n"
        "  --variants       also run the prim_variants/ graveyard (frozen non-shipping\n"
        "                   kernels) next to production + scalar reference\n"
        "  --variants=NAME  ...limited to one logical primitive, e.g.\n"
        "                   enc_partition_full, merge_vec_vec, flat_dN_unpack\n"
        "  --list           list logical primitives + their variants (* = production),\n"
        "                   then exit (combine with --variants for the graveyard)\n"
        "  --listv          verbose --list: one line per variant with what it is\n"
        "  -h, --help       show this help and exit\n",
        MAXD);
}

int main(int argc, char **argv) {
    int n = 32768, reps = 2000, want[MAXD+1] = {0}, any = 0;
    int variants = 0, do_list = 0; const char *vfilter = NULL;
    int canary = 1;
    int bm_runs_w = 0, bm_runs_b = 0;   /* --bm=runs: run-structured bitmap */
    int bm_mix_p = -1;                  /* --bm=mix: P% of u64 words all-zero, random order */
    int bm_dens_p = -1;                 /* --bm=p:P: i.i.d. bits at P% ones density */
    for (int i=1;i<argc;i++) {
        if      (!strncmp(argv[i],"--n=",4))    n = atoi(argv[i]+4);
        else if (!strncmp(argv[i],"--reps=",7)) reps = atoi(argv[i]+7);
        else if (!strncmp(argv[i],"--canary=",9)) canary = atoi(argv[i]+9);
        else if (!strncmp(argv[i],"--bm=",5)) {
            if (!strncmp(argv[i]+5,"runs",4)) {
                bm_runs_w = 400; bm_runs_b = 16;             /* calgary-pic-ish defaults */
                if (argv[i][9] == ':') sscanf(argv[i]+10, "%d,%d", &bm_runs_w, &bm_runs_b);
            } else if (!strncmp(argv[i]+5,"mix",3)) {
                bm_mix_p = 50;                               /* max branch entropy default */
                if (argv[i][8] == ':') bm_mix_p = atoi(argv[i]+9);
            } else if (!strncmp(argv[i]+5,"p:",2)) {
                bm_dens_p = atoi(argv[i]+7);
            } else if (strcmp(argv[i]+5,"random")) {
                fprintf(stderr, "bench_prim: --bm= expects random, runs[:W,B], mix[:P] or p:P\n"); return 2;
            }
        }
        else if (!strcmp(argv[i],"--variants") || !strncmp(argv[i],"--variants=",11)) {
            variants = 1;
            if (argv[i][10] == '=') vfilter = argv[i] + 11;  /* e.g. enc_partition_full */
        }
        else if (!strcmp(argv[i],"--list"))  do_list = 1;
        else if (!strcmp(argv[i],"--listv")) do_list = 2;
        else if (!strncmp(argv[i],"--D=",4))
            for (char *t=strtok(argv[i]+4,","); t; t=strtok(NULL,",")) {
                int d=atoi(t); if (d>=2&&d<=MAXD){want[d]=1;any=1;} }
        else if (!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")) {
            usage(stdout); return 0;
        }
        else {
            fprintf(stderr, "bench_prim: unknown arg '%s'\n\n", argv[i]);
            usage(stderr); return 2;
        }
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
    uint8_t   c2s[256], *ref_bm = malloc((size_t)(n/8) + 64);   /* 2^8 entries (D up to 8) */
    /* rank-based buffers: ranks (pristine), ranks_work (mutated), tmp8 (right),
     * symbuf (enc_init input), ref8l/r (partition ref), and the two gather LUTs. */
    uint8_t  *ranks = malloc(n+64), *ranks_work = malloc(n+64), *tmp8 = malloc(n+64);
    uint8_t  *symbuf = malloc(n+64), *ref8l = malloc(n+64), *ref8r = malloc(n+64);
    uint16_t code_la_lut[256]; uint8_t sym_to_rank[256];
    uint8_t c2s_id[256];                                     /* identity c2s (merge_flat D=8 invariant) */
    uint16_t s2r_hi[256];                                    /* enc_init_aux backing (x86 2tab) */
    uint32_t *hist = malloc(256 * sizeof(uint32_t));
    uint8_t  *hist_scratch = malloc(PIVCO_PRIM_HIST_SCRATCH);
    srand(0xC0FFEE);
    for (int i=0;i<n+16;i++){ bm[i]=(uint8_t)rand(); la_pristine[i]=(uint16_t)rand(); }
    for (int i=0;i<n+16;i++){ merge_left[i]=(uint8_t)rand(); merge_right[i]=(uint8_t)rand(); }
    for (int i=0;i<256;i++) c2s[i]=(uint8_t)rand();
    for (int i=0;i<n+64;i++){ ranks[i]=(uint8_t)rand(); symbuf[i]=(uint8_t)rand(); }
    for (int i=0;i<256;i++){ code_la_lut[i]=(uint16_t)rand(); sym_to_rank[i]=(uint8_t)rand(); }
    for (int i=0;i<256;i++) s2r_hi[i]=(uint16_t)((unsigned)sym_to_rank[i]<<8);
    if (bm_runs_w > 0) {
        /* --bm=runs: overwrite the random bitmap with alternating 0-runs (mean W)
           and 1-runs (mean B), uniform lengths in [1, 2*mean) -- a calgary-pic-like
           run structure so the merge stages see realistic long-run bitmaps. */
        long bits = (long)(n + 16) * 8, b = 0; int bit = 0;
        memset(bm, 0, (size_t)(n + 16));
        while (b < bits) {
            int mean = bit ? bm_runs_b : bm_runs_w;
            long len = 1 + rand() % (2 * mean);
            if (bit)
                for (long t = b; t < b + len && t < bits; t++)
                    bm[t >> 3] |= (uint8_t)(1u << (t & 7));
            b += len; bit ^= 1;
        }
        long ones = 0, zw = 0, ow = 0, nw = n >> 6;
        for (long t = 0; t < n; t++) ones += (bm[t >> 3] >> (t & 7)) & 1;
        for (long w = 0; w < nw; w++) {
            uint64_t v; memcpy(&v, bm + w * 8, 8);
            zw += (v == 0); ow += (v == ~0ull);
        }
        printf("bm=runs W=%d B=%d: ones=%.1f%%  zero-words=%.1f%%  one-words=%.1f%%  (of %ld u64 words)\n",
               bm_runs_w, bm_runs_b, 100.0 * ones / n, 100.0 * zw / nw, 100.0 * ow / nw, nw);
    }
    if (bm_mix_p >= 0) {
        /* --bm=mix: each u64 word independently all-zero with probability P%,
           else random -- word-granularity randomness, i.e. maximum entropy for a
           per-word easy/hard branch (P=50 is the adversarial case for fused). */
        long nwb = (long)(n + 16) >> 3, zw = 0;
        for (long w = 0; w < nwb; w++) {
            if (rand() % 100 < bm_mix_p) { memset(bm + w * 8, 0, 8); zw += (w < (n >> 6)); }
            else for (int t = 0; t < 8; t++) bm[w * 8 + t] = (uint8_t)rand();
        }
        printf("bm=mix P=%d: zero-words=%.1f%% in random order (of %d u64 words)\n",
               bm_mix_p, 100.0 * zw / (n >> 6), n >> 6);
    }
    if (bm_dens_p >= 0) {
        /* --bm=p:P: every bit independently one with probability P% -- the
           i.i.d. density knob (proba80-shaped masks at p:20). */
        long ones = 0;
        for (long i = 0; i < (long)n + 16; i++) {
            uint8_t b = 0;
            for (int t = 0; t < 8; t++)
                if (rand() % 100 < bm_dens_p) b |= (uint8_t)(1u << t);
            bm[i] = b;
            if (i < (long)(n >> 3)) ones += __builtin_popcount(b);
        }
        printf("bm=p P=%d: ones=%.1f%%\n", bm_dens_p, 100.0 * ones / n);
    }
    uint8_t rank_thr = 127, rank_base = 0;

    for (int d=2; d<=MAXD; d++) {
        if (!want[d]) continue;
        /* scalar + its SIMD peer adjacent per (D,stage).  NEON unpack now has
           a D7 kernel (D<=7); scatter/pack/merge flat kernels still cap at D6
           (D7 scatter needs a 128-entry table beyond TBL's reach). */
        /* unpack/scatter: standalone NEON kernels (D2..7).  pack/merge: the
           production prim_* on every backend (SIMD where the backend handles
           D, scalar fallback otherwise) -- so registered for all D. */
        reg("scalar",ST_UNPACK, d,0,p_unpack_scalar);
#if defined(USE_NEON_KERNELS)
        if (d <= 7) reg(BK,ST_UNPACK, d,0,neon_unpack);
#endif
#if defined(PIVCO_HAS_AVX512)
        if (d <= 7) reg(BK,ST_UNPACK, d,0,avx512_unpack);
#endif
        reg("scalar",ST_SCATTER,d,0,p_scatter_scalar);
#if defined(USE_NEON_KERNELS)
        if (d <= 7) reg(BK,ST_SCATTER,d,0,neon_scatter);
#endif
        reg("scalar",ST_U16_PACK,   d,0,p_pack_scalar);
#if defined(HAVE_SIMD)
        reg(BK,ST_U16_PACK,   d,0,simd_u16pack);
#endif
        reg("scalar",ST_PACK, d,0,p_pack_rank_scalar);
#if defined(HAVE_SIMD)
        reg(BK,ST_PACK, d,0,simd_pack);
#endif
        reg("scalar",ST_MERGE_FLAT,  d,0,p_merge_scalar);
#if defined(HAVE_SIMD)
        reg(BK,ST_MERGE_FLAT,  d,0,simd_merge_flat);
#endif
    }
    reg("scalar",ST_U16_PART,0,1,p_part_scalar);
#if defined(HAVE_SIMD)
    reg(BK,      ST_U16_PART,0,1,simd_u16part);
#endif
    /* rank-based encode: enc_init (u16 code_la vs u8 rank gather), partition full. */
    reg("scalar",ST_U16_ENC_INIT,      0,0,p_enc_init_scalar);
    reg("scalar",ST_ENC_INIT, 0,0,p_enc_init_rank_scalar);
    reg("scalar",ST_PART,     0,0,p_part_rank_scalar);
    reg("scalar",ST_PART_RIGHT,0,0,p_part_right_rank_scalar);
#if defined(HAVE_SIMD)
    reg(BK,      ST_U16_ENC_INIT,      0,0,simd_u16enc_init);
    reg(BK,      ST_ENC_INIT, 0,0,simd_enc_init);
    reg(BK,      ST_PART,     0,0,simd_part);
    reg(BK,      ST_PART_RIGHT,0,0,simd_part_right);
#endif
#if defined(USE_NEON_KERNELS)
    /* Unfused decomposition: fused part == bm_build + part_bm (re-read cost).
       part_half == HALF-node saving (one-sided scatter). */
    reg(BK, ST_BMBUILD, 0,0, simd_u16bmbuild);
    reg(BK, ST_PARTBM,  0,1, simd_u16partbm);
    reg(BK, ST_PARTHALF,0,0, simd_u16parthalf);
    reg(BK, ST_FUSEDHALF,0,0, simd_u16fusedhalf);
#endif
    /* Binary-merge family: production prims used at every internal node in
       the bottom-up codec.  Same bm consumption pattern as merge_flat D=1
       but the symbol(s) come from buffer reads or scalar constants
       instead of a c2s lookup.  Scalar refs + production SIMD on all
       backends; the pcpc/unroll8 experimental variants are NEON-only. */
    reg("scalar", ST_MERGE_VEC_VEC, 0, 0, p_merge_vec_vec_scalar);
    reg("scalar", ST_MERGE_CST_CST, 0, 0, p_merge_cst_cst_scalar);
    reg("scalar", ST_MERGE_CST_VEC,  0, 0, p_merge_cst_vec_scalar);
#if defined(HAVE_SIMD)
    reg(BK,       ST_MERGE_VEC_VEC, 0, 0, simd_merge_vec_vec);
    reg(BK,       ST_MERGE_CST_CST, 0, 0, simd_merge_cst_cst);
    reg(BK,       ST_MERGE_CST_VEC,  0, 0, simd_merge_cst_vec);
#endif
    /* merge_vec_vec / merge_cst_cst experimental variants (neon_pcpc,
       unroll8, tbl/blendtab/vtbl/vtblq/d1flat) now live in
       prim_variants/prims-merge.h — run with --variants. */
#if defined(HAVE_SIMD)
    /* Encode-side histogram primitive (symbuf as input). */
    reg("scalar", ST_HISTOGRAM, 0, 0, p_hist_scalar);
    reg(BK,       ST_HISTOGRAM, 0, 0, p_hist_prim);
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

    if (do_list) {
        /* --list always shows the full set (production + variants); running
           the variants still requires --variants. */
        pv_register_partition();
        pv_register_merge();
        pv_register_flat();
        pv_register_pack();
        pv_register_init();
        int verbose = (do_list == 2);
        printf("logical primitives  (* = production, + = variant, - = other-arch, not built here):\n");
        int shown[64] = {0};
        for (int k=0;k<NPRIMS;k++) {
            stage_t s = PRIMS[k].stage;
            if ((int)s < 64 && shown[s]) continue;
            if ((int)s < 64) shown[s] = 1;
            char seen[64][24]; int ns = 0;
            if (verbose) {
                printf("\n%s:\n", stage_name(s));
                for (int m=0;m<NPRIMS;m++) if (PRIMS[m].stage==s) {
                    prim_t *p = &PRIMS[m];
                    int dup=0; for (int t=0;t<ns;t++) if(!strcmp(seen[t],p->variant)) {dup=1;break;}
                    if (dup) continue;
                    if (ns<64) snprintf(seen[ns++],24,"%s",p->variant);
                    int isprod = !strcmp(p->variant, BK), issca = !strcmp(p->variant,"scalar");
                    int isstub = (p->run == NULL);   /* compiled out on this host */
                    char mk = isstub ? '-' : (isprod ? '*' : (issca ? ' ' : '+'));
                    printf("  %-15s %c  ", p->variant, mk);
                    if (issca)       printf("scalar reference");
                    else if (isprod) printf("production (real prim_*)");
                    else if (p->origin)
                        printf("[%s] %s%s%s%s", pv_isa_name(p->isa), p->origin,
                               p->note ? " · " : "", p->note ? p->note : "",
                               isstub ? "  (not built here)" : "");
                    else printf("experimental (bench_prim, unannotated)");
                    printf("\n");
                }
            } else {
                printf("  %-22s", stage_name(s));
                char vbuf[400]; int vo = 0; vbuf[0] = 0;   /* runnable variants */
                char sbuf[400]; int so = 0; sbuf[0] = 0;   /* other-arch stubs  */
                for (int m=0;m<NPRIMS;m++) if (PRIMS[m].stage==s) {
                    prim_t *p = &PRIMS[m];
                    int dup=0; for (int t=0;t<ns;t++) if(!strcmp(seen[t],p->variant)) {dup=1;break;}
                    if (dup) continue;
                    if (ns<64) snprintf(seen[ns++],24,"%s",p->variant);
                    if (!strcmp(p->variant, BK))           printf(" %s*", p->variant);
                    else if (!strcmp(p->variant,"scalar")) printf(" scalar");
                    else if (p->run == NULL) so += snprintf(sbuf+so, sizeof sbuf - so, " %s", p->variant);
                    else                     vo += snprintf(vbuf+vo, sizeof vbuf - vo, " %s", p->variant);
                }
                if (vo) printf(" [+%s]", vbuf);
                if (so) printf(" [-%s]", sbuf);
                printf("\n");
            }
        }
        return 0;
    }

    if (variants) {
        pv_register_partition();
        pv_register_merge();
        pv_register_flat();
        pv_register_pack();
        pv_register_init();
        /* --D also filters the graveyard: pv_register_* is unconditional, so
           drop per-D variants whose D wasn't requested. */
        int nk = 0;
        for (int k=0;k<NPRIMS;k++)
            if (!PRIMS[k].D || want[PRIMS[k].D]) PRIMS[nk++] = PRIMS[k];
        NPRIMS = nk;
        /* Regroup so each logical family (stage) runs as one contiguous block,
           sub-grouped by D — scalar, then production backend, then its variants
           — instead of all variants trailing at the very end.  Stable within
           each (stage, D): original registration order is preserved, so
           scalar/backend stay ahead of the later-registered variants, and
           per-D variants (e.g. flat fl_natural) slot under their own D rather
           than after the last D. */
        static prim_t grouped[sizeof PRIMS / sizeof PRIMS[0]]; int ng = 0;
        for (int k=0;k<NPRIMS;k++) {
            stage_t s = PRIMS[k].stage;
            int sseen=0; for (int j=0;j<k;j++) if (PRIMS[j].stage==s){sseen=1;break;}
            if (sseen) continue;
            /* flush this stage, sub-grouped by D in first-occurrence order */
            for (int m=0;m<NPRIMS;m++) {
                if (PRIMS[m].stage!=s) continue;
                int d = PRIMS[m].D;
                int dseen=0; for (int q=0;q<m;q++) if (PRIMS[q].stage==s && PRIMS[q].D==d){dseen=1;break;}
                if (dseen) continue;
                for (int t=0;t<NPRIMS;t++) if (PRIMS[t].stage==s && PRIMS[t].D==d) grouped[ng++]=PRIMS[t];
            }
        }
        memcpy(PRIMS, grouped, (size_t)ng*sizeof(prim_t)); NPRIMS = ng;
    }

    /* Line-buffer stdout: a crashing variant must not eat the rows already
       measured (block buffering hid the l7r8 SIGSEGV as clean truncation). */
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("bench_prim: n=%d elems, best-of-9 x %d reps, partition depth=%d%s\n",
           n, reps, PART_DEPTH, variants ? "  [* production, + variant]" : "");
    printf("%-22s %-3s %-15s %10s  %5s %5s %5s   %6s %6s %6s  %s\n",
           "stage","D","variant","ns/elem",
           "in_b","out_b","lut_b", "in_MB/s","out_MB/s","lut_MB/s","check");
    volatile uint8_t sink = 0; int prevD=-99; stage_t prevS=-1;

    /* Warm the core to max DVFS clock before any timing.  At the default
       best-of-9 x reps each variant's timed window is only ~5 ms — shorter
       than the turbo ramp — so the FIRST variant measured otherwise reads
       sub-turbo, and best-of-9 can't rescue it when all 9 samples are cold
       (measured: neon 0.0367 cold vs 0.035 warm on M4).  Busy-spin ~200 ms. */
    {
        double tw = now_ns(); volatile uint64_t w = 0; uint64_t a = 1;
        while (now_ns() - tw < 200e6)
            for (int i=0;i<200000;i++){ a = a*6364136223846793005ULL + 1442695040888963407ULL; w += a; }
        sink ^= (uint8_t)w;
    }

    if (canary >= 1) bench_canary("start");
    stage_t canary_prevS = (stage_t)-1;

    for (int k=0;k<NPRIMS;k++) {
        prim_t *p = &PRIMS[k];
        if (!p->run) continue;   /* other-arch stub (not built here); see --list */
        if (vfilter && strcmp(stage_name(p->stage), vfilter)) continue;
        /* --canary=2: one probe at each family (stage) boundary */
        if (canary == 2 && p->stage != canary_prevS && canary_prevS != (stage_t)-1)
            bench_canary(stage_name(p->stage));
        canary_prevS = p->stage;
        ctx_t cx = { .bm=bm, .codes=codes, .c2s=c2s, .out=out, .pack_out=pack_out,
                     .merge_left=merge_left, .merge_right=merge_right,
                     .la_work=la_work, .tmp16=tmp16,
                     .ranks=ranks, .ranks_work=ranks_work, .tmp8=tmp8, .symbuf=symbuf,
                     .code_la_lut=code_la_lut, .sym_to_rank=sym_to_rank,
                     .enc_init_aux={ .s2r_hi=s2r_hi },
                     .rank_thr=rank_thr, .rank_base=rank_base,
                     .hist=hist, .hist_scratch=hist_scratch,
                     .n=n, .D=p->D, .depth=PART_DEPTH };
        const char *chk = "ok";

        /* per-stage input prep + correctness vs scalar reference */
        if (p->stage == ST_SCATTER) {
            uint32_t m=(1u<<p->D)-1u; for (int i=0;i<n;i++) codes[i]=(uint8_t)(rand()&m);
            scalar_scatter(ref,codes,c2s,n); memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_UNPACK) {
            scalar_unpack(ref,bm,n,p->D); memset(codes,0,n); p->run(&cx);
            if (memcmp(codes,ref,n)) chk="FAIL";
        } else if (p->stage == ST_U16_PACK) {
            memcpy(la_work,la_pristine,(size_t)n*2);
            scalar_pack(ref,la_pristine,n,p->D,PART_DEPTH); memset(pack_out,0,n); p->run(&cx);
            if (memcmp(pack_out,ref,(n*p->D+7)>>3)) chk="FAIL";
        } else if (p->stage == ST_MERGE_FLAT) {
            /* D=8 flat = the full 256-symbol alphabet at equal length, whose
               canonical c2s is the identity permutation (production relies on
               this: merge_flat d8 is a memcpy) -- feed the invariant here. */
            if (p->D == 8) { for (int k=0;k<256;k++) c2s_id[k]=(uint8_t)k; cx.c2s = c2s_id; }
            scalar_unpack(codes,bm,n,p->D); scalar_scatter(ref,codes,cx.c2s,n);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_MERGE_VEC_VEC) {
#if defined(USE_NEON_KERNELS)
            /* Populate the graveyard pcpc popcount table once per row so its
               pre-filled "pcpc" variant sees an up-to-date table; harmless
               otherwise.  (Defined in prim_variants/prims-merge.h.) */
            pv_fill_bm_popcnt(bm, n);
#endif
            scalar_merge_vec_vec(ref, bm, n, merge_left, merge_right);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_MERGE_CST_CST) {
            scalar_merge_cst_cst(ref, bm, n, MERGE_LEFT_SYM, MERGE_RIGHT_SYM);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_MERGE_CST_VEC) {
            scalar_merge_cst_vec(ref, bm, n, MERGE_LEFT_SYM, merge_right);
            memset(out,0,n); p->run(&cx);
            if (memcmp(out,ref,n)) chk="FAIL";
        } else if (p->stage == ST_HISTOGRAM) {
            uint32_t refh[256] = {0};
            for (int i = 0; i < n; i++) refh[symbuf[i]]++;
            memset(hist, 0, 256 * sizeof(uint32_t));
            p->run(&cx);
            if (memcmp(hist, refh, sizeof(refh))) chk="FAIL";
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
        } else if (p->stage == ST_U16_ENC_INIT) {
            for (int i=0;i<n;i++) ((uint16_t*)ref16l)[i] = code_la_lut[symbuf[i]];
            p->run(&cx);
            if (memcmp(la_work,ref16l,(size_t)n*2)) chk="FAIL";
        } else if (p->stage == ST_ENC_INIT) {
            scalar_rank_init(ref8l,n,symbuf,sym_to_rank); p->run(&cx);
            if (memcmp(ranks_work,ref8l,(size_t)n)) chk="FAIL";
        } else if (p->stage == ST_PART) {
            int nr_ref = scalar_rank_partition(ranks,n,rank_thr,ref_bm,ref8l,ref8r);
            int nl_ref = n - nr_ref;
            memcpy(ranks_work,ranks,(size_t)n); p->run(&cx);
            if (memcmp(bm,ref_bm,(n+7)>>3) || memcmp(ranks_work,ref8l,(size_t)nl_ref)
                || memcmp(tmp8,ref8r,(size_t)nr_ref)) chk="FAIL";
        } else if (p->stage == ST_PART_RIGHT) {
            int nr_ref = scalar_rank_partition(ranks,n,rank_thr,ref_bm,ref8l,ref8r);
            memcpy(ranks_work,ranks,(size_t)n); p->run(&cx);   /* reads ranks_work, no left scatter */
            if (memcmp(bm,ref_bm,(n+7)>>3) || memcmp(tmp8,ref8r,(size_t)nr_ref)) chk="FAIL";
        } else if (p->stage == ST_PACK) {
            /* Production feeds in-range local codes (rank - flat_base_rank in
               [0,2^D)); mirror that so the pack sees realistic input (the SIMD
               packs no longer mask to D bits — they rely on this guarantee). */
            uint8_t cm = (uint8_t)((1u << p->D) - 1);
            for (int k = 0; k < n; k++) tmp8[k] = (uint8_t)(ranks[k] & cm);
            cx.ranks = tmp8;
            scalar_rank_pack(ref,tmp8,n,p->D,rank_base); memset(pack_out,0,n); p->run(&cx);
            if (memcmp(pack_out,ref,(n*p->D+7)>>3)) chk="FAIL";
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

        /* rank partition mutates ranks_work in place -> needs a per-rep u8
           reset (n bytes), distinct from the u16 `inplace` path (n*2). */
        int inplace8 = (p->stage == ST_PART);
        double best = 1e30;
        for (int s=0;s<9;s++) {
            double t0 = now_ns();
            if (inplace8)        for (int r=0;r<reps;r++){ memcpy(ranks_work,ranks,(size_t)n);     p->run(&cx); }
            else if (p->inplace) for (int r=0;r<reps;r++){ memcpy(la_work,la_pristine,(size_t)n*2); p->run(&cx); }
            else                 for (int r=0;r<reps;r++) p->run(&cx);
            double e = now_ns()-t0;
            if (inplace8) {        /* subtract the per-rep u8 memcpy baseline */
                double b0=now_ns();
                for (int r=0;r<reps;r++){ memcpy(ranks_work,ranks,(size_t)n); sink^=ranks_work[0]; }
                e -= (now_ns()-b0);
            } else if (p->inplace) {  /* subtract the per-rep u16 memcpy baseline */
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
        /* mark like --list: '*' production backend, '+' variant, ' ' scalar */
        char vmark = !strcmp(p->variant, BK) ? '*'
                   : !strcmp(p->variant, "scalar") ? ' ' : '+';
        char vbuf[24]; snprintf(vbuf, sizeof vbuf, "%c %s", vmark, p->variant);
        printf("%-22s %-3s %-15s %10.4f  %5.2f %5.2f %5.2f   %6.0f %6.0f %6.0f  %s",
               stage_name(p->stage), dbuf, vbuf, best,
               in_b, out_b, lut_b, in_bw, out_bw, lut_bw, chk);
        if (p->origin)     /* prim_variants provenance: [isa] origin · note */
            printf("  [%s] %s%s%s", pv_isa_name(p->isa), p->origin,
                   p->note ? " · " : "", p->note ? p->note : "");
        printf("\n");
        /* --canary=3: one probe after every primitive */
        if (canary >= 3) bench_canary(p->variant);
    }
    if (canary >= 1) { bench_canary("end"); bench_canary_summary(); }
    (void)sink;
    free(merge_left); free(merge_right);
    return 0;
}
