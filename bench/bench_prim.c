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

typedef struct {
    uint8_t  *bm, *codes, *c2s, *out, *pack_out;
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
static void simd_merge(const ctx_t *c){ prim_merge_flat(c->out, c->n, c->bm, c->D, c->c2s); }
static void simd_part (const ctx_t *c){ prim_enc_partition_full(c->la_work, c->n, c->depth, c->bm, c->tmp16); }
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

typedef enum { ST_UNPACK, ST_SCATTER, ST_PACK, ST_MERGE, ST_PART,
               ST_BMBUILD, ST_PARTBM, ST_PARTHALF, ST_FUSEDHALF } stage_t;
typedef struct {
    const char *variant; stage_t stage; int D; int inplace; void (*run)(const ctx_t *);
} prim_t;
static prim_t PRIMS[256]; static int NPRIMS = 0;
static void reg(const char *v, stage_t s, int D, int ip, void (*fn)(const ctx_t *)) {
    PRIMS[NPRIMS++] = (prim_t){ v, s, D, ip, fn };
}
static const char *stage_name(stage_t s){
    switch(s){case ST_UNPACK:return"unpack";case ST_SCATTER:return"scatter";
              case ST_PACK:return"pack";case ST_MERGE:return"merge";
              case ST_BMBUILD:return"bm_build";case ST_PARTBM:return"part_bm";
              case ST_PARTHALF:return"part_half";case ST_FUSEDHALF:return"fused_half";
              default:return"partition";}
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
    uint16_t *la_pristine = malloc((n+16)*2), *la_work = malloc((n+16)*2),
             *tmp16 = malloc((n+16)*2), *ref16l = malloc((n+16)*2), *ref16r = malloc((n+16)*2);
    uint8_t   c2s[256], ref_bm[ (8192/8) + 64 ];   /* 2^8 entries (D up to 8) */
    srand(0xC0FFEE);
    for (int i=0;i<n+16;i++){ bm[i]=(uint8_t)rand(); la_pristine[i]=(uint16_t)rand(); }
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
        reg("scalar",ST_MERGE,  d,0,p_merge_scalar);
#if defined(HAVE_SIMD)
        reg(BK,ST_MERGE,  d,0,simd_merge);
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

    printf("bench_prim: n=%d elems, best-of-9 x %d reps, partition depth=%d\n",
           n, reps, PART_DEPTH);
    printf("%-10s %-4s %-7s %10s  %s\n","stage","D","variant","ns/elem","check");
    volatile uint8_t sink = 0; int prevD=-99; stage_t prevS=-1;

    for (int k=0;k<NPRIMS;k++) {
        prim_t *p = &PRIMS[k];
        ctx_t cx = { bm, codes, c2s, out, pack_out, la_work, tmp16, n, p->D, PART_DEPTH };
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
        } else if (p->stage == ST_MERGE) {
            scalar_unpack(codes,bm,n,p->D); scalar_scatter(ref,codes,c2s,n);
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
        printf("%-10s %-4s %-7s %10.3g  %s\n", stage_name(p->stage), dbuf, p->variant, best, chk);
    }
    (void)sink;
    return 0;
}
