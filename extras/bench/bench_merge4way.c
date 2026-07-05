/* bench_merge4way — cross-arch 4-way root merge (top 2 levels in one pass).
 *
 * The root's 4 grandchild streams A,B,C,D are interleaved per a 2-bit-per-output
 * routing (two bitplanes hi/lo), producing N outputs in ONE pass instead of the
 * 3 binary merges (L=merge(A,B), R=merge(C,D), out=merge(L,R)) the BU decoder
 * does today.  Naming = the 4 grandchild slots: 'v' = vector stream (internal
 * grandchild), 'c' = constant (leaf grandchild).  A leaf slot uses a broadcast
 * (set1 on AVX-512) instead of an expand-from-memory — no load, no cursor.
 *
 *   merge-vvvv : 4 streams                 merge-vvcv : C is a leaf
 *   merge-cvcv : A,C leaves                merge-cccv : only D a stream
 *
 * AVX-512 VBMI2 runs all four (the bitplane->kmask->vpexpandb / set1 form).
 * NEON runs only vvvv (vqtbl4 + computed 4-way rank — no byte-expand on ARM).
 *
 * Build: cc -O3 -march=native -o m4 bench_merge4way.c
 * Run:   ./m4 [n_elems] [reps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                            return t.tv_sec*1e9 + t.tv_nsec; }
#define CV(slot) ((uint8_t)(0xA0 + (slot)))   /* constant value for leaf slot */

/* scalar reference: isc[v]=1 -> slot v is a constant CV(v); else stream S[v]. */
static void ref4(const uint8_t *q,int N,const uint8_t *const S[4],const int isc[4],uint8_t *out){
    int cur[4]={0,0,0,0};
    for(int i=0;i<N;i++){ int v=q[i]; out[i]= isc[v]?CV(v):S[v][cur[v]++]; }
}

/* ====================================================================== */
#if defined(__AVX512VBMI2__)
#include <immintrin.h>

/* binary merge_vec_vec (production AVX-512 vpexpandb form) */
static void merge2(const uint64_t *bm,int K,const uint8_t *L,const uint8_t *R,uint8_t *out){
    int lc=0,rc=0,j=0;
    for(; j+64<=K; j+=64){ __mmask64 m=(__mmask64)bm[j>>6];
        __m512i l=_mm512_maskz_expandloadu_epi8(~m,L+lc), r=_mm512_maskz_expandloadu_epi8(m,R+rc);
        _mm512_storeu_si512((void*)(out+j), _mm512_or_si512(l,r));
        int pr=(int)__builtin_popcountll((uint64_t)m); rc+=pr; lc+=64-pr; }
    for(; j<K; j++){ int b=(int)((bm[j>>6]>>(j&63))&1); out[j]=b?R[rc++]:L[lc++]; }
}

/* the four 4-way variants.  m0=~H&~Lo (A), m1=~H&Lo (B), m2=H&~Lo (C), m3=H&Lo (D). */
#define MASKS  __mmask64 H=(__mmask64)hi[j>>6],Lo=(__mmask64)lo[j>>6],nH=~H,nLo=~Lo; \
               __mmask64 m0=nH&nLo,m1=nH&Lo,m2=H&nLo,m3=H&Lo;
#define EXP(reg,m,S,c) __m512i reg=_mm512_maskz_expandloadu_epi8(m,(S)+c)
#define CST(reg,m,v)   __m512i reg=_mm512_maskz_set1_epi8(m,(char)CV(v))
#define ST  _mm512_storeu_si512((void*)(out+j), _mm512_or_si512(_mm512_or_si512(a,b),_mm512_or_si512(c,d)))
#define PC(m) (int)__builtin_popcountll((uint64_t)(m))

static void merge_vvvv(const uint64_t*hi,const uint64_t*lo,int N,
        const uint8_t*A,const uint8_t*B,const uint8_t*C,const uint8_t*D,uint8_t*out){
    int ca=0,cb=0,cc=0,cd=0,j=0;
    for(;j+64<=N;j+=64){ MASKS; EXP(a,m0,A,ca);EXP(b,m1,B,cb);EXP(c,m2,C,cc);EXP(d,m3,D,cd); ST;
        ca+=PC(m0);cb+=PC(m1);cc+=PC(m2);cd+=PC(m3); }
    for(;j<N;j++){int h=(int)((hi[j>>6]>>(j&63))&1),l=(int)((lo[j>>6]>>(j&63))&1),v=(h<<1)|l;
        out[j]= v==0?A[ca++]:v==1?B[cb++]:v==2?C[cc++]:D[cd++];}
}
static void merge_vvcv(const uint64_t*hi,const uint64_t*lo,int N,
        const uint8_t*A,const uint8_t*B,const uint8_t*C,const uint8_t*D,uint8_t*out){
    (void)C; int ca=0,cb=0,cd=0,j=0;
    for(;j+64<=N;j+=64){ MASKS; EXP(a,m0,A,ca);EXP(b,m1,B,cb);CST(c,m2,2);EXP(d,m3,D,cd); ST;
        ca+=PC(m0);cb+=PC(m1);cd+=PC(m3); }
    for(;j<N;j++){int h=(int)((hi[j>>6]>>(j&63))&1),l=(int)((lo[j>>6]>>(j&63))&1),v=(h<<1)|l;
        out[j]= v==0?A[ca++]:v==1?B[cb++]:v==2?CV(2):D[cd++];}
}
static void merge_cvcv(const uint64_t*hi,const uint64_t*lo,int N,
        const uint8_t*A,const uint8_t*B,const uint8_t*C,const uint8_t*D,uint8_t*out){
    (void)A;(void)C; int cb=0,cd=0,j=0;
    for(;j+64<=N;j+=64){ MASKS; CST(a,m0,0);EXP(b,m1,B,cb);CST(c,m2,2);EXP(d,m3,D,cd); ST;
        cb+=PC(m1);cd+=PC(m3); }
    for(;j<N;j++){int h=(int)((hi[j>>6]>>(j&63))&1),l=(int)((lo[j>>6]>>(j&63))&1),v=(h<<1)|l;
        out[j]= v==0?CV(0):v==1?B[cb++]:v==2?CV(2):D[cd++];}
}
static void merge_cccv(const uint64_t*hi,const uint64_t*lo,int N,
        const uint8_t*A,const uint8_t*B,const uint8_t*C,const uint8_t*D,uint8_t*out){
    (void)A;(void)B;(void)C; int cd=0,j=0;
    for(;j+64<=N;j+=64){ MASKS; CST(a,m0,0);CST(b,m1,1);CST(c,m2,2);EXP(d,m3,D,cd); ST;
        cd+=PC(m3); }
    for(;j<N;j++){int h=(int)((hi[j>>6]>>(j&63))&1),l=(int)((lo[j>>6]>>(j&63))&1),v=(h<<1)|l;
        out[j]= v==0?CV(0):v==1?CV(1):v==2?CV(2):D[cd++];}
}

/* _e variants: leaf slot uses vpexpandb from a 64-byte constant buffer (cursor
 * pinned at 0 — every byte is the constant, so no advance) instead of set1.
 * One uniform expand code-path; should beat set1 where expand is cheap (Zen5). */
static uint8_t gconst[4][64] __attribute__((aligned(64)));
#define EXPC(reg,m,v) __m512i reg=_mm512_maskz_expandloadu_epi8(m, gconst[v])
static void merge_vvcv_e(const uint64_t*hi,const uint64_t*lo,int N,
        const uint8_t*A,const uint8_t*B,const uint8_t*C,const uint8_t*D,uint8_t*out){
    (void)C; int ca=0,cb=0,cd=0,j=0;
    for(;j+64<=N;j+=64){ MASKS; EXP(a,m0,A,ca);EXP(b,m1,B,cb);EXPC(c,m2,2);EXP(d,m3,D,cd); ST;
        ca+=PC(m0);cb+=PC(m1);cd+=PC(m3); }
    for(;j<N;j++){int h=(int)((hi[j>>6]>>(j&63))&1),l=(int)((lo[j>>6]>>(j&63))&1),v=(h<<1)|l;
        out[j]= v==0?A[ca++]:v==1?B[cb++]:v==2?CV(2):D[cd++];}
}
static void merge_cvcv_e(const uint64_t*hi,const uint64_t*lo,int N,
        const uint8_t*A,const uint8_t*B,const uint8_t*C,const uint8_t*D,uint8_t*out){
    (void)A;(void)C; int cb=0,cd=0,j=0;
    for(;j+64<=N;j+=64){ MASKS; EXPC(a,m0,0);EXP(b,m1,B,cb);EXPC(c,m2,2);EXP(d,m3,D,cd); ST;
        cb+=PC(m1);cd+=PC(m3); }
    for(;j<N;j++){int h=(int)((hi[j>>6]>>(j&63))&1),l=(int)((lo[j>>6]>>(j&63))&1),v=(h<<1)|l;
        out[j]= v==0?CV(0):v==1?B[cb++]:v==2?CV(2):D[cd++];}
}
static void merge_cccv_e(const uint64_t*hi,const uint64_t*lo,int N,
        const uint8_t*A,const uint8_t*B,const uint8_t*C,const uint8_t*D,uint8_t*out){
    (void)A;(void)B;(void)C; int cd=0,j=0;
    for(;j+64<=N;j+=64){ MASKS; EXPC(a,m0,0);EXPC(b,m1,1);EXPC(c,m2,2);EXP(d,m3,D,cd); ST;
        cd+=PC(m3); }
    for(;j<N;j++){int h=(int)((hi[j>>6]>>(j&63))&1),l=(int)((lo[j>>6]>>(j&63))&1),v=(h<<1)|l;
        out[j]= v==0?CV(0):v==1?CV(1):v==2?CV(2):D[cd++];}
}
#define HAVE_AVX512 1

/* ====================================================================== */
#elif defined(__aarch64__)
#include <arm_neon.h>

static uint8_t expand_tab[256][8] __attribute__((aligned(32)));
static uint8_t expand_popcnt[256];
static uint8_t expand_tab_pre[9][256][8] __attribute__((aligned(32)));
static void init_neon_tabs(void){
    for(int m=0;m<256;m++){int nz=0,no=0;
        for(int k=0;k<8;k++){if(m&(1<<k)){expand_tab[m][k]=(uint8_t)(8+no);no++;}else{expand_tab[m][k]=(uint8_t)nz;nz++;}}
        expand_popcnt[m]=(uint8_t)no;}
    for(int nr0=0;nr0<=8;nr0++)for(int m=0;m<256;m++)for(int k=0;k<8;k++){uint8_t r=expand_tab[m][k];
        expand_tab_pre[nr0][m][k]=(r<8)?(uint8_t)(r+(8-nr0)):(uint8_t)(r+8+nr0);}
}
static void merge2(const uint8_t *bm,int K,const uint8_t *L,const uint8_t *R,uint8_t *out){
    int lc=0,rc=0,j=0;
    for(;j+64<=K;j+=64){ uint64_t mk;memcpy(&mk,bm+(j>>3),8);
        uint8x8_t pcv=vcnt_u8(vcreate_u8(mk)); uint64_t pc=vget_lane_u64(vreinterpret_u64_u8(pcv),0);
        uint64_t pfx=pc*0x0101010101010101ull;
        uint8_t cr0=0,cr1=(uint8_t)(pfx>>8),cr2=(uint8_t)(pfx>>24),cr3=(uint8_t)(pfx>>40);
        uint8_t in0=(uint8_t)pc,in1=(uint8_t)(pc>>16),in2=(uint8_t)(pc>>32),in3=(uint8_t)(pc>>48);
        uint8_t m0=(uint8_t)mk,m1=(uint8_t)(mk>>8),m2=(uint8_t)(mk>>16),m3=(uint8_t)(mk>>24);
        uint8_t m4=(uint8_t)(mk>>32),m5=(uint8_t)(mk>>40),m6=(uint8_t)(mk>>48),m7=(uint8_t)(mk>>56);
#define CH(i,cr,in,ma,mb) do{uint8_t cl=(uint8_t)((i)*16-(cr)); uint8x16_t Lf=vld1q_u8(L+lc+cl);\
        uint8x16_t both=vcombine_u8(vget_low_u8(Lf),vld1_u8(R+rc+(cr)));\
        vst1_u8(out+j+(i)*16,vqtbl1_u8(both,vld1_u8(expand_tab[ma])));\
        uint8x16x2_t s={{Lf,vld1q_u8(R+rc+(cr))}}; vst1_u8(out+j+(i)*16+8,vqtbl2_u8(s,vld1_u8(expand_tab_pre[in][mb])));}while(0)
        CH(0,cr0,in0,m0,m1);CH(1,cr1,in1,m2,m3);CH(2,cr2,in2,m4,m5);CH(3,cr3,in3,m6,m7);
#undef CH
        uint32_t tr=(uint32_t)(pfx>>56); rc+=tr; lc+=64-tr; }
    for(;j<K;j++){int b=(bm[j>>3]>>(j&7))&1; out[j]=b?R[rc++]:L[lc++];}
}
static inline uint8x16_t psum16(uint8x16_t v){ uint8x16_t z=vdupq_n_u8(0);
    v=vaddq_u8(v,vextq_u8(z,v,15)); v=vaddq_u8(v,vextq_u8(z,v,14));
    v=vaddq_u8(v,vextq_u8(z,v,12)); v=vaddq_u8(v,vextq_u8(z,v,8)); return v; }
/* NEON 4-way: gather 16 outputs via vqtbl4 from {A,B,C,D} windows, index =
 * quadrant*16 + 4-way intra-chunk rank (the expensive part — no byte-expand). */
static void merge_vvvv(const uint8_t *quad,int N,
        const uint8_t*A,const uint8_t*B,const uint8_t*C,const uint8_t*D,uint8_t*out){
    int ca=0,cb=0,cc=0,cd=0,j=0; const uint8x16_t one=vdupq_n_u8(1);
    for(;j+16<=N;j+=16){ uint8x16_t q=vld1q_u8(quad+j);
        uint8x16_t q0=vceqq_u8(q,vdupq_n_u8(0)),q1=vceqq_u8(q,vdupq_n_u8(1)),
                   q2=vceqq_u8(q,vdupq_n_u8(2)),q3=vceqq_u8(q,vdupq_n_u8(3));
        uint8x16_t p0=psum16(vandq_u8(q0,one)),p1=psum16(vandq_u8(q1,one)),
                   p2=psum16(vandq_u8(q2,one)),p3=psum16(vandq_u8(q3,one));
        uint8x16_t rank=vorrq_u8(vorrq_u8(vandq_u8(vsubq_u8(p0,vandq_u8(q0,one)),q0),
                                          vandq_u8(vsubq_u8(p1,vandq_u8(q1,one)),q1)),
                                 vorrq_u8(vandq_u8(vsubq_u8(p2,vandq_u8(q2,one)),q2),
                                          vandq_u8(vsubq_u8(p3,vandq_u8(q3,one)),q3)));
        uint8x16_t idx=vaddq_u8(vshlq_n_u8(q,4),rank);
        uint8x16x4_t src={{vld1q_u8(A+ca),vld1q_u8(B+cb),vld1q_u8(C+cc),vld1q_u8(D+cd)}};
        vst1q_u8(out+j,vqtbl4q_u8(src,idx));
        ca+=vgetq_lane_u8(p0,15);cb+=vgetq_lane_u8(p1,15);cc+=vgetq_lane_u8(p2,15);cd+=vgetq_lane_u8(p3,15);}
    for(;j<N;j++){int v=quad[j]; out[j]=v==0?A[ca++]:v==1?B[cb++]:v==2?C[cc++]:D[cd++];}
}
#define HAVE_NEON 1
#endif

/* The whole bench is ISA-specific (merge2 and all 4-way kernels exist
 * only for AVX-512 / NEON); on anything else there is nothing to bench. */
#if !defined(HAVE_AVX512) && !defined(HAVE_NEON)
int main(void){
    printf("bench_merge4way: needs AVX-512 or NEON; nothing to bench on this ISA.\n");
    return 0;
}
#else

static double bench(void(*run)(void),int reps){
    double best=1e30;
    for(int s=0;s<9;s++){ double t0=now_ns(); for(int r=0;r<reps;r++) run(); double e=now_ns()-t0; if(e<best)best=e; }
    return best;
}

/* globals so the timed thunks are nullary */
static int gN; static uint8_t *gA,*gB,*gC,*gD,*gout;
#if defined(HAVE_AVX512)
static uint64_t *ghi,*glo,*grootbm,*gleftbm,*grightbm; static int gKl,gKr; static uint8_t *gLb,*gRb;
static void t_vvvv(void){ merge_vvvv(ghi,glo,gN,gA,gB,gC,gD,gout); }
static void t_vvcv(void){ merge_vvcv(ghi,glo,gN,gA,gB,gC,gD,gout); }
static void t_cvcv(void){ merge_cvcv(ghi,glo,gN,gA,gB,gC,gD,gout); }
static void t_cccv(void){ merge_cccv(ghi,glo,gN,gA,gB,gC,gD,gout); }
static void t_vvcv_e(void){ merge_vvcv_e(ghi,glo,gN,gA,gB,gC,gD,gout); }
static void t_cvcv_e(void){ merge_cvcv_e(ghi,glo,gN,gA,gB,gC,gD,gout); }
static void t_cccv_e(void){ merge_cccv_e(ghi,glo,gN,gA,gB,gC,gD,gout); }
static void t_bin (void){ merge2(gleftbm,gKl,gA,gB,gLb); merge2(grightbm,gKr,gC,gD,gRb); merge2(grootbm,gN,gLb,gRb,gout); }
#elif defined(HAVE_NEON)
static uint8_t *gquad,*grootbm,*gleftbm,*grightbm,*gLb,*gRb; static int gKl,gKr;
static void t_vvvv(void){ merge_vvvv(gquad,gN,gA,gB,gC,gD,gout); }
static void t_bin (void){ merge2(gleftbm,gKl,gA,gB,gLb); merge2(grightbm,gKr,gC,gD,gRb); merge2(grootbm,gN,gLb,gRb,gout); }
#endif

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):(1<<22); int reps=argc>2?atoi(argv[2]):200; N&=~63; gN=N;
    uint8_t *quad=malloc(N);
    gA=malloc(N+64);gB=malloc(N+64);gC=malloc(N+64);gD=malloc(N+64);gout=malloc(N+64);
    uint8_t *ref=malloc(N+64); gLb=malloc(N+64);gRb=malloc(N+64);
    srand(0xC0FFEE);
    for(int i=0;i<N;i++){int r=rand();int v=(r&7)<5?(r>>3)&3:(r>>5)&1;quad[i]=(uint8_t)v;}
    for(int i=0;i<N+64;i++){gA[i]=(uint8_t)rand();gB[i]=(uint8_t)rand();gC[i]=(uint8_t)rand();gD[i]=(uint8_t)rand();}
    const uint8_t *S[4]={gA,gB,gC,gD};
    int Kl=0,Kr=0;
    uint8_t *rootbm=calloc((N>>3)+8,1),*leftbm=calloc((N>>3)+8,1),*rightbm=calloc((N>>3)+8,1);
    for(int i=0;i<N;i++){int h=quad[i]>>1,l=quad[i]&1; if(h)rootbm[i>>3]|=1<<(i&7);
        if(h==0){if(l)leftbm[Kl>>3]|=1<<(Kl&7);Kl++;}else{if(l)rightbm[Kr>>3]|=1<<(Kr&7);Kr++;}}
    gKl=Kl;gKr=Kr;
    { volatile uint64_t w=0;double t=now_ns();uint64_t a=1;
      while(now_ns()-t<200e6){for(int i=0;i<200000;i++){a=a*6364136223846793005ull+1;w+=a;}}(void)w; }

    printf("N=%d (Kl=%d Kr=%d) reps=%d\n",N,Kl,Kr,reps);
#if defined(HAVE_AVX512)
    ghi=calloc((N>>6)+1,8); glo=calloc((N>>6)+1,8);
    for(int i=0;i<N;i++){if(quad[i]>>1)ghi[i>>6]|=1ull<<(i&63); if(quad[i]&1)glo[i>>6]|=1ull<<(i&63);}
    grootbm=(uint64_t*)rootbm;gleftbm=(uint64_t*)leftbm;grightbm=(uint64_t*)rightbm; /* reuse byte arrays as u64 */
    for(int v=0;v<4;v++) memset(gconst[v],CV(v),64);
    /* name, set1-fn, expand-const-fn (NULL for vvvv), isc */
    struct{const char*name;void(*fn)(void);void(*fn_e)(void);int isc[4];}V[]={
        {"merge-vvvv",t_vvvv,  NULL,    {0,0,0,0}},
        {"merge-vvcv",t_vvcv,  t_vvcv_e,{0,0,1,0}},
        {"merge-cvcv",t_cvcv,  t_cvcv_e,{1,0,1,0}},
        {"merge-cccv",t_cccv,  t_cccv_e,{1,1,1,0}} };
    double bb=bench(t_bin,reps)/((double)reps*N);
    printf("  %-12s %10s %10s\n","variant","set1","exp-const");
    printf("  binary-2lvl  %10.4f %10s   (reference)\n",bb,"-");
    for(int k=0;k<4;k++){
        ref4(quad,N,S,V[k].isc,ref);
        double e=bench(V[k].fn,reps)/((double)reps*N); int ok=(memcmp(gout,ref,N)==0);
        double ee=-1; int oke=1;
        if(V[k].fn_e){ ref4(quad,N,S,V[k].isc,ref); ee=bench(V[k].fn_e,reps)/((double)reps*N); oke=(memcmp(gout,ref,N)==0); }
        if(ee<0) printf("  %-12s %10.4f %10s   %s\n",V[k].name,e,"-",ok?"ok":"FAIL");
        else     printf("  %-12s %10.4f %10.4f   %s/%s\n",V[k].name,e,ee,ok?"ok":"FAIL",oke?"ok":"FAIL");
    }
#elif defined(HAVE_NEON)
    gquad=quad; grootbm=rootbm;gleftbm=leftbm;grightbm=rightbm;
    init_neon_tabs();
    double bb=bench(t_bin,reps)/((double)reps*N);
    int isc[4]={0,0,0,0}; ref4(quad,N,S,isc,ref);
    double e=bench(t_vvvv,reps)/((double)reps*N); int ok=(memcmp(gout,ref,N)==0);
    printf("  binary 2-level (vvvv, 3 pass)     : %.4f ns/elem  (reference)\n",bb);
    printf("  merge-vvvv (vqtbl4 + 4-way rank)  : %.4f ns/elem  %s\n",e,ok?"ok":"FAIL");
    printf("  (NEON: vvcv/cvcv/cccv not built — no byte-expand; 4-way rank dominates)\n");
#endif
    return 0;
}

#endif /* HAVE_AVX512 || HAVE_NEON */
