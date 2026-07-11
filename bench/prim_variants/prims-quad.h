/* bench/prim_variants/prims-quad.h — quad-node bench variants.
 *
 * The production quad merge is prim_merge_quad (16-wide on NEON + x86, 64-wide
 * on AVX-512, scalar fallback elsewhere), defined in the backend headers and
 * used by the codec.  The losing/alternate forms live here, bench-only:
 *   - prim_merge_quad8      8-wide (the original; superseded by 16-wide)
 *   - prim_merge_quad8_mt   NEON max-trick (single 2 KB base table; lost)
 *   - scalar_merge_quad / scalar_part_quad   leaf-aware references
 *
 * The SIMD kernels reference the shared quad tables (pv_q_lm / pv_r4_lx etc.)
 * and pv_q_build / pv_r4_build in the backend header, so this MUST be included
 * after pivco_huffman_primitives.h in the same TU (bench_prim.c).  No ctx_t
 * dependency here -- the bench runners/registration that wrap these stay in
 * bench_prim.c.
 */
#ifndef PIVCO_PRIM_VARIANTS_QUAD_H
#define PIVCO_PRIM_VARIANTS_QUAD_H

/* ---- leaf-aware scalar references (any backend) ----
 * For a leaf slot the cursor is pinned (the stream is a constant buffer),
 * matching the SIMD kernels' contract. */
static void scalar_merge_quad(uint8_t *out, const uint8_t *p0, const uint8_t *p1,
    int N, const uint8_t *A, const uint8_t *B, const uint8_t *C, const uint8_t *D, unsigned lm)
{
    int ca=0,cb=0,cc=0,cd=0;
    int lfA=(lm>>3)&1,lfB=(lm>>2)&1,lfC=(lm>>1)&1,lfD=lm&1;
    for (int j=0;j<N;j++){
        int h=(p0[j>>3]>>(j&7))&1,l=(p1[j>>3]>>(j&7))&1;
        switch((h<<1)|l){
        case 0: out[j]=A[ca]; if(!lfA)ca++; break;
        case 1: out[j]=B[cb]; if(!lfB)cb++; break;
        case 2: out[j]=C[cc]; if(!lfC)cc++; break;
        default:out[j]=D[cd]; if(!lfD)cd++; break; }
    }
}
static void scalar_part_quad(const uint8_t *ranks, int n, uint8_t thr0, uint8_t thrL, uint8_t thrR,
    uint8_t *p0, uint8_t *p1, uint8_t *const q[4], int cnt[4], unsigned lm)
{
    int c0=0,c1=0,c2=0,c3=0;
    int lf0=(lm>>3)&1,lf1=(lm>>2)&1,lf2=(lm>>1)&1,lf3=lm&1;
    memset(p0,0,(size_t)((n+7)>>3)); memset(p1,0,(size_t)((n+7)>>3));
    for (int j=0;j<n;j++){
        uint8_t r=ranks[j];
        int b0=r>thr0, b1=r>(b0?thrR:thrL);
        p0[j>>3]|=(uint8_t)(b0<<(j&7)); p1[j>>3]|=(uint8_t)(b1<<(j&7));
        switch((b0<<1)|b1){
        case 0: if(!lf0){q[0][c0]=r;c0++;} break;
        case 1: if(!lf1){q[1][c1]=r;c1++;} break;
        case 2: if(!lf2){q[2][c2]=r;c2++;} break;
        default:if(!lf3){q[3][c3]=r;c3++;} break; }
    }
    cnt[0]=c0;cnt[1]=c1;cnt[2]=c2;cnt[3]=c3;
}

#ifdef PIVCO_BACKEND_NEON
/* ---- NEON 8-wide (vqtbl4_u8, 8 outputs/iter) ---- */
PIVCO_PRIM_ALWAYS_INLINE void prim_merge_quad8_impl(const uint8_t *restrict p0,
    const uint8_t *restrict p1, int N,
    const uint8_t *restrict A, const uint8_t *restrict B, const uint8_t *restrict C, const uint8_t *restrict D,
    const int lfA, const int lfB, const int lfC, const int lfD, uint8_t *restrict out)
{
    int ca=0,cb=0,cc=0,cd=0,j=0;
    for(; j+8<=N; j+=8){
        uint8_t x0=p0[j>>3], x1=p1[j>>3];
        uint8_t mA=(uint8_t)(~x0&~x1),mB=(uint8_t)(~x0&x1),mC=(uint8_t)(x0&~x1),mD=(uint8_t)(x0&x1);
        uint8x8_t idx=vorr_u8(vorr_u8(vld1_u8(pv_q_lm[0][mA]),vld1_u8(pv_q_lm[1][mB])),
                              vorr_u8(vld1_u8(pv_q_lm[2][mC]),vld1_u8(pv_q_lm[3][mD])));
        uint8x16x4_t src={{vld1q_u8(A+ca),vld1q_u8(B+cb),vld1q_u8(C+cc),vld1q_u8(D+cd)}};
        vst1_u8(out+j, vqtbl4_u8(src, idx));
        if(!lfA) ca+=__builtin_popcount(mA);
        if(!lfB) cb+=__builtin_popcount(mB);
        if(!lfC) cc+=__builtin_popcount(mC);
        if(!lfD) cd+=__builtin_popcount(mD);
    }
    for(; j<N; j++){
        int h=(p0[j>>3]>>(j&7))&1,l=(p1[j>>3]>>(j&7))&1;
        switch((h<<1)|l){
        case 0: out[j]=A[ca]; if(!lfA)ca++; break;
        case 1: out[j]=B[cb]; if(!lfB)cb++; break;
        case 2: out[j]=C[cc]; if(!lfC)cc++; break;
        default:out[j]=D[cd]; if(!lfD)cd++; break; }
    }
}
PIVCO_PRIM_ALWAYS_INLINE void prim_merge_quad8(const uint8_t *restrict p0, const uint8_t *restrict p1,
    int N, const uint8_t *restrict A, const uint8_t *restrict B, const uint8_t *restrict C, const uint8_t *restrict D,
    unsigned leaf_mask, uint8_t *restrict out)
{
    pv_q_build();
    switch(leaf_mask & 15u){
#define PIVCO_Q(S) case (S): prim_merge_quad8_impl(p0,p1,N,A,B,C,D,\
        ((S)>>3)&1,((S)>>2)&1,((S)>>1)&1,(S)&1,out); break;
    PIVCO_Q(0) PIVCO_Q(1) PIVCO_Q(2) PIVCO_Q(3) PIVCO_Q(4) PIVCO_Q(5) PIVCO_Q(6) PIVCO_Q(7)
    PIVCO_Q(8) PIVCO_Q(9) PIVCO_Q(10) PIVCO_Q(11) PIVCO_Q(12) PIVCO_Q(13) PIVCO_Q(14) PIVCO_Q(15)
#undef PIVCO_Q
    }
}

/* ---- NEON max-trick 16-wide: single 2 KB base table (pv_q_base), the g*16
 * window offset + carry folded into one vadd, then vmax_s8 with 0 to clamp the
 * 0x80-sentinel non-owned lanes to 0.  Compute-for-cache; lost the A/B. ---- */
#define PV_MT_HALF(mask, off) vreinterpret_u8_s8(vmax_s8( \
    vreinterpret_s8_u8(vadd_u8(vld1_u8(pv_q_base[(mask)]), vdup_n_u8((uint8_t)(off)))), vdup_n_s8(0)))
PIVCO_PRIM_ALWAYS_INLINE void prim_merge_quad8_mt_impl(const uint8_t *restrict p0,
    const uint8_t *restrict p1, int N,
    const uint8_t *restrict A, const uint8_t *restrict B, const uint8_t *restrict C, const uint8_t *restrict D,
    const int lfA, const int lfB, const int lfC, const int lfD, uint8_t *restrict out)
{
    int ca=0,cb=0,cc=0,cd=0,j=0;
    for(; j+16<=N; j+=16){
        int b=j>>3;
        uint8_t x0=p0[b],y0=p0[b+1],x1=p1[b],y1=p1[b+1];
        uint8_t mA=(uint8_t)(~x0&~x1),mB=(uint8_t)(~x0&x1),mC=(uint8_t)(x0&~x1),mD=(uint8_t)(x0&x1);
        uint8_t nA=(uint8_t)(~y0&~y1),nB=(uint8_t)(~y0&y1),nC=(uint8_t)(y0&~y1),nD=(uint8_t)(y0&y1);
        int pa=__builtin_popcount(mA),pb=__builtin_popcount(mB),pc=__builtin_popcount(mC),pd=__builtin_popcount(mD);
        int ka=lfA?0:pa,kb=lfB?0:pb,kc=lfC?0:pc,kd=lfD?0:pd;  /* leaf: high rank restarts at 0 */
        uint8x16_t iA=vcombine_u8(PV_MT_HALF(mA,0),      PV_MT_HALF(nA,ka));
        uint8x16_t iB=vcombine_u8(PV_MT_HALF(mB,16),     PV_MT_HALF(nB,16+kb));
        uint8x16_t iC=vcombine_u8(PV_MT_HALF(mC,32),     PV_MT_HALF(nC,32+kc));
        uint8x16_t iD=vcombine_u8(PV_MT_HALF(mD,48),     PV_MT_HALF(nD,48+kd));
        uint8x16_t idx=vorrq_u8(vorrq_u8(iA,iB),vorrq_u8(iC,iD));
        uint8x16x4_t src={{vld1q_u8(A+ca),vld1q_u8(B+cb),vld1q_u8(C+cc),vld1q_u8(D+cd)}};
        vst1q_u8(out+j, vqtbl4q_u8(src, idx));
        if(!lfA) ca+=pa+__builtin_popcount(nA);
        if(!lfB) cb+=pb+__builtin_popcount(nB);
        if(!lfC) cc+=pc+__builtin_popcount(nC);
        if(!lfD) cd+=pd+__builtin_popcount(nD);
    }
    for(; j<N; j++){
        int h=(p0[j>>3]>>(j&7))&1,l=(p1[j>>3]>>(j&7))&1;
        switch((h<<1)|l){
        case 0: out[j]=A[ca]; if(!lfA)ca++; break;
        case 1: out[j]=B[cb]; if(!lfB)cb++; break;
        case 2: out[j]=C[cc]; if(!lfC)cc++; break;
        default:out[j]=D[cd]; if(!lfD)cd++; break; }
    }
}
PIVCO_PRIM_ALWAYS_INLINE void prim_merge_quad8_mt(const uint8_t *restrict p0, const uint8_t *restrict p1,
    int N, const uint8_t *restrict A, const uint8_t *restrict B, const uint8_t *restrict C, const uint8_t *restrict D,
    unsigned leaf_mask, uint8_t *restrict out)
{
    pv_q_build();
    switch(leaf_mask & 15u){
#define PIVCO_Q(S) case (S): prim_merge_quad8_mt_impl(p0,p1,N,A,B,C,D,\
        ((S)>>3)&1,((S)>>2)&1,((S)>>1)&1,(S)&1,out); break;
    PIVCO_Q(0) PIVCO_Q(1) PIVCO_Q(2) PIVCO_Q(3) PIVCO_Q(4) PIVCO_Q(5) PIVCO_Q(6) PIVCO_Q(7)
    PIVCO_Q(8) PIVCO_Q(9) PIVCO_Q(10) PIVCO_Q(11) PIVCO_Q(12) PIVCO_Q(13) PIVCO_Q(14) PIVCO_Q(15)
#undef PIVCO_Q
    }
}
#undef PV_MT_HALF
#endif /* PIVCO_BACKEND_NEON */

#ifdef PIVCO_BACKEND_X86
/* ---- x86 SSE 8-wide (4 pshufb + OR, storel_epi64, 8 outputs/iter) ---- */
PIVCO_PRIM_ALWAYS_INLINE void prim_merge_quad8_impl(const uint8_t *restrict p0,
    const uint8_t *restrict p1, int N,
    const uint8_t *restrict A, const uint8_t *restrict B, const uint8_t *restrict C, const uint8_t *restrict D,
    const int lfA, const int lfB, const int lfC, const int lfD, uint8_t *restrict out)
{
    int ca = 0, cb = 0, cc = 0, cd = 0, j = 0;
    for (; j + 8 <= N; j += 8) {
        uint8_t x0 = p0[j >> 3], x1 = p1[j >> 3];
        uint8_t mA = (uint8_t)(~x0 & ~x1), mB = (uint8_t)(~x0 & x1),
                mC = (uint8_t)(x0 & ~x1), mD = (uint8_t)(x0 & x1);
        __m128i r = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(A+ca)), _mm_load_si128((const __m128i*)pv_r4_lx[mA])),
            _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(B+cb)), _mm_load_si128((const __m128i*)pv_r4_lx[mB]))),
            _mm_or_si128(
            _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(C+cc)), _mm_load_si128((const __m128i*)pv_r4_lx[mC])),
            _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(D+cd)), _mm_load_si128((const __m128i*)pv_r4_lx[mD]))));
        _mm_storel_epi64((__m128i*)(out+j), r);
        if (!lfA) ca += __builtin_popcount(mA);
        if (!lfB) cb += __builtin_popcount(mB);
        if (!lfC) cc += __builtin_popcount(mC);
        if (!lfD) cd += __builtin_popcount(mD);
    }
    for (; j < N; j++) {
        int h = (p0[j>>3]>>(j&7))&1, l = (p1[j>>3]>>(j&7))&1;
        switch ((h<<1)|l){
        case 0: out[j]=A[ca]; if(!lfA)ca++; break;
        case 1: out[j]=B[cb]; if(!lfB)cb++; break;
        case 2: out[j]=C[cc]; if(!lfC)cc++; break;
        default:out[j]=D[cd]; if(!lfD)cd++; break; }
    }
}
PIVCO_PRIM_ALWAYS_INLINE void prim_merge_quad8(const uint8_t *restrict p0, const uint8_t *restrict p1,
    int N, const uint8_t *restrict A, const uint8_t *restrict B, const uint8_t *restrict C, const uint8_t *restrict D,
    unsigned leaf_mask, uint8_t *restrict out)
{
    pv_r4_build();
    switch (leaf_mask & 15u) {
#define PVQ(S) case (S): prim_merge_quad8_impl(p0,p1,N,A,B,C,D,((S)>>3)&1,((S)>>2)&1,((S)>>1)&1,(S)&1,out); break;
    PVQ(0)PVQ(1)PVQ(2)PVQ(3)PVQ(4)PVQ(5)PVQ(6)PVQ(7)PVQ(8)PVQ(9)PVQ(10)PVQ(11)PVQ(12)PVQ(13)PVQ(14)PVQ(15)
#undef PVQ
    }
}
#endif /* PIVCO_BACKEND_X86 */

#endif /* PIVCO_PRIM_VARIANTS_QUAD_H */
