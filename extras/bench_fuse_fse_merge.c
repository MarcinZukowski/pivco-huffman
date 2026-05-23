/* bench_fuse_fse_merge — does interleaving the (scalar) multi-cursor FSE
 * bitmap decode with the (SIMD) bcast_left merge hide one behind the
 * other?  proba80-root shape: K=8192 elements, ~80% bit=0 (left=const),
 * ~20% bit=1 (right=right_buf[rc++]).
 *
 * Cursor count via -DXVAL=2 or 8.  Hypothesis: fusion helps the
 * latency-bound x2 (stall bubbles to fill) more than the throughput-bound
 * x8 (8-way ILP already keeps the pipeline full).
 *
 * Serial : decode_x{X}_y1(whole bitmap) ; tree_merge_bcast_left(whole)
 * Fused  : per CH-byte chunk { decode CH bytes ; merge CH*8 elements }
 *
 * NOTE (docs/FUSION.md): microbench overlap over-promises vs the real
 * recursive decoder by 2-5x.  Go/no-go probe, not a final number.
 */
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#include "bitstream.h"
#include <arm_neon.h>
#include "pivco_huffman_neon_tables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "fse_xy_codec.h"
#pragma GCC diagnostic pop

#ifndef XVAL
#define XVAL 8
#endif
#if XVAL == 2
#define SERIAL_DECODE decode_x2_y1
#define ROUND_BYTES   2
#define STEPR(OP) do { \
    (OP)[0]=FSE_decodeSymbolFast(&s[0],&bitD); (OP)[1]=FSE_decodeSymbolFast(&s[1],&bitD); \
} while(0)
#elif XVAL == 8
#define SERIAL_DECODE decode_x8_y1
#define ROUND_BYTES   8
#define STEPR(OP) do { \
    (OP)[0]=FSE_decodeSymbolFast(&s[0],&bitD); (OP)[1]=FSE_decodeSymbolFast(&s[1],&bitD); \
    (OP)[2]=FSE_decodeSymbolFast(&s[2],&bitD); (OP)[3]=FSE_decodeSymbolFast(&s[3],&bitD); \
    BIT_reloadDStream(&bitD); \
    (OP)[4]=FSE_decodeSymbolFast(&s[4],&bitD); (OP)[5]=FSE_decodeSymbolFast(&s[5],&bitD); \
    (OP)[6]=FSE_decodeSymbolFast(&s[6],&bitD); (OP)[7]=FSE_decodeSymbolFast(&s[7],&bitD); \
} while(0)
#else
#error "XVAL must be 2 or 8"
#endif

#define K        8192
#define NBYTES   (K / 8)
#ifndef CH
#define CH       32
#endif

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }

static inline void merge_chunk(const uint8_t *bm, int Kc, uint8_t left_sym,
                               const uint8_t *right, int *rcp, uint8_t *out)
{
    int rc = *rcp, j = 0;
    uint8x8_t  Lbcast   = vdup_n_u8(left_sym);
    uint8x16_t Lbcast_q = vdupq_n_u8(left_sym);
    for (; j + 16 <= Kc; j += 16) {
        uint8x16_t R_full = vld1q_u8(right + rc);
        uint8_t m0 = bm[j >> 3];
        uint8x16_t both0 = vcombine_u8(Lbcast, vget_low_u8(R_full));
        uint8x8_t  o0    = vqtbl1_u8(both0, vld1_u8(expand_tab[m0]));
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];
        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x16x2_t src = {{ Lbcast_q, R_full }};
        uint8x8_t o1 = vqtbl2_u8(src, vld1_u8(expand_tab_pre[nr0][m1]));
        vst1_u8(out + j + 8, o1);
        rc += nr0 + expand_popcnt[m1];
    }
    *rcp = rc;
}

static void serial(const uint8_t *cmp, size_t clen, const FSE_DTable *dt,
                   uint8_t *bm_scratch, uint8_t left_sym,
                   const uint8_t *right, uint8_t *out)
{
    SERIAL_DECODE(cmp, clen, bm_scratch, NBYTES, dt);
    int rc = 0;
    merge_chunk(bm_scratch, K, left_sym, right, &rc, out);
}

static void fused(const uint8_t *cmp, size_t clen, const FSE_DTable *dt,
                  uint8_t left_sym, const uint8_t *right, uint8_t *out)
{
    BIT_DStream_t bitD; BIT_initDStream(&bitD, cmp, clen);
    FSE_DState_t s[XVAL]; for (int k=0;k<XVAL;k++) FSE_initDState(&s[k],&bitD,dt);
    uint8_t chunk[CH];
    int j = 0, rc = 0;
    const int MARGIN = 128;
    int bulk_bytes = NBYTES - MARGIN;
    while (j/8 + CH <= bulk_bytes) {
        uint8_t *op = chunk;
        for (int r = 0; r < CH/ROUND_BYTES; r++) {
            BIT_reloadDStream(&bitD);
            STEPR(op); op += ROUND_BYTES;
        }
        merge_chunk(chunk, CH*8, left_sym, right, &rc, out + j);
        j += CH*8;
    }
    if (j < K) {
        int rem_bytes = NBYTES - j/8;
        uint8_t tail[NBYTES];
        uint8_t *op = tail; uint8_t * const olim = tail + rem_bytes;
        while (op + XVAL <= olim) {
            int of = 0;
            for (int k = 0; k < XVAL; k++) {
                *op++ = FSE_decodeSymbol(&s[k], &bitD);
                if (BIT_reloadDStream(&bitD) == BIT_DStream_overflow) {
                    for (int kk=k+1; kk<XVAL && op<olim; kk++) *op++ = FSE_decodeSymbol(&s[kk],&bitD);
                    of=1; break;
                }
            }
            if (of) break;
        }
        for (int k=0; k<XVAL && op<olim; k++) *op++ = FSE_decodeSymbol(&s[k],&bitD);
        merge_chunk(tail, rem_bytes*8, left_sym, right, &rc, out + j);
    }
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 300000;
    init_expand_table();

    uint8_t *bm = malloc(NBYTES);
    uint64_t rng = 0x1234; int ones = 0;
    for (int i = 0; i < NBYTES; i++) {
        uint8_t b = 0;
        for (int k=0;k<8;k++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
            double u=(rng>>11)*(1.0/9007199254740992.0); if(u<0.20){b|=(1u<<k);ones++;} }
        bm[i]=b;
    }
    uint8_t *right = malloc(ones + 64);
    for (int i=0;i<ones+64;i++) right[i] = (uint8_t)(0x30 + (i&0x3f));
    uint8_t left_sym = 0x41;

    unsigned hist[256]={0}; for(int i=0;i<NBYTES;i++) hist[bm[i]]++;
    short norm[256];
    FSE_normalizeCount(norm, 12, hist, NBYTES, 255);
    int safe=1; for(int s=0;s<=255;s++) if(norm[s]>=2048) safe=0;
    FSE_CTable *ct=FSE_createCTable(255,12); FSE_DTable *dt=FSE_createDTable(12);
    FSE_buildCTable(ct,norm,255,12); FSE_buildDTable(dt,norm,255,12);

    uint8_t *cmp=malloc(NBYTES+256);
    size_t clen=encode_x(XVAL, bm, NBYTES, cmp, NBYTES+256, ct);
    printf("X=%d K=%d nbytes=%d ones=%d clen=%zu fast_safe=%d CH=%d\n",
           XVAL,K,NBYTES,ones,clen,safe,CH);

    uint8_t *out_s=malloc(K), *out_f=malloc(K), *scratch=malloc(NBYTES+16);
    serial(cmp,clen,dt,scratch,left_sym,right,out_s);
    fused (cmp,clen,dt,        left_sym,right,out_f);
    int match = memcmp(out_s,out_f,K)==0;
    printf("fused==serial: %d\n", match);
    if(!match){ for(int i=0;i<K;i++) if(out_s[i]!=out_f[i]){printf(" diff@%d s=%02x f=%02x\n",i,out_s[i],out_f[i]);break;} }

    double t0=now_sec();
    for(int r=0;r<reps;r++) serial(cmp,clen,dt,scratch,left_sym,right,out_s);
    double ts=now_sec()-t0;
    t0=now_sec();
    for(int r=0;r<reps;r++) fused(cmp,clen,dt,left_sym,right,out_f);
    double tf=now_sec()-t0;

    double mb=(double)reps*K/1e6;
    printf("serial: %.3f s  %.0f M/s\n", ts, mb/ts);
    printf("fused : %.3f s  %.0f M/s   speedup %.2fx\n", tf, mb/tf, ts/tf);
    return match?0:1;
}
