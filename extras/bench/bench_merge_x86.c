/* bench_merge_x86.c -- x86 merge_vec_vec: current SSE 2x-unroll (serial
 * cursor) vs the NEON prefix-sum cursor-decoupling idea ported to x86.
 *
 * Variants (all verified against a scalar reference):
 *   old            : production SSE 2x-unrolled stride-16 (serial lc/rc)
 *   sse_com        : 64 codes/iter, 8 independent 8-code pshufb merges,
 *                    cursors from a SWAR bytewise popcount * 0x0101..
 *   sse_com_pshufb : sse_com but Mula pshufb nibble-LUT popcount
 *   sse_com128     : 128 codes/iter, 16 chunks, full-width pshufb popcount
 *   avx2_com       : 16 codes per _mm256_shuffle_epi8 (two lanes), COM cursors
 *   avx512         : production vpexpandb merge (reference ceiling; only the
 *                    VBMI2 hosts run this in production)
 *
 * Finding (see IDEAS.md "x86 COM merge"): the prefix-sum decoupling that
 * wins on NEON does NOT transfer to x86.  The x86 baseline merge has no
 * cursor-dependent table load to remove (flat pshufb + 1-cycle cursor
 * adds), so COM only adds code + a prefix-sum dependency.  In production
 * fair_bench on the SSE/AVX2 tier (clang-20): regresses on all Intel
 * (c3/c4/c5), wins only on Zen 2 (c5a), wash on Zen 3 (c6a).  Microbench
 * is compiler-fragile (gcc vs clang flip sign on Intel) and overpromises
 * vs production.  Kept for the record / future retry.  vpexpandb is ~3x
 * faster than any COM form, so VBMI2 hosts are unaffected regardless.
 *
 * perf_event_open CPU_CYCLES (Linux x86; wall-ns fallback if locked down).
 * Build:
 *   clang-20 -O3 -march=native -o bm extras/bench/bench_merge_x86.c
 */
#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <immintrin.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>

/* expand_tab[m][k]: shuffle index into unpacklo(L8,R8) (bytes 0..7=L, 8..15=R).
 * bit k set -> take R[rank of set bits before k] = 8 + r; else L[rank] = l. */
static uint8_t expand_tab[256][8];
static uint8_t expand_popcnt[256];

/* Per-byte popcount of a u64: each output byte = popcount of the
 * corresponding input byte.  ~5 ALU ops, no table, no loop. */
static inline uint64_t popcnt_bytes_u64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return x;
}

/* Muła pshufb nibble-LUT bytewise popcount.  Returns per-byte popcounts
 * of the 16 input bytes in an xmm (each lane 0..8). */
static inline __m128i popcnt_bytes_xmm(__m128i v) {
    const __m128i lut  = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m128i mask = _mm_set1_epi8(0x0f);
    __m128i lo = _mm_and_si128(v, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
    return _mm_add_epi8(_mm_shuffle_epi8(lut, lo), _mm_shuffle_epi8(lut, hi));
}
/* 8-byte version: per-byte popcounts of the low 8 bytes, returned as a u64. */
static inline uint64_t popcnt_bytes_u64_pshufb(uint64_t x) {
    __m128i v = _mm_cvtsi64_si128((long long)x);
    return (uint64_t)_mm_cvtsi128_si64(popcnt_bytes_xmm(v));
}
static void build_tables(void) {
    for (int m = 0; m < 256; m++) {
        int r = 0, l = 0;
        for (int k = 0; k < 8; k++) {
            if ((m >> k) & 1) { expand_tab[m][k] = (uint8_t)(8 + r); r++; }
            else              { expand_tab[m][k] = (uint8_t)l;       l++; }
        }
        expand_popcnt[m] = (uint8_t)__builtin_popcount(m);
    }
}

/* ============ OLD: current production SSE 2x-unroll (serial cursor) */
__attribute__((always_inline)) static inline void old_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i L0 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(L0, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        _mm_storel_epi64((__m128i *)(out + j), _mm_shuffle_epi8(both0, shuf0));
        int nr0 = expand_popcnt[m0];
        rc += nr0; lc += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i L1 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(L1, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        _mm_storel_epi64((__m128i *)(out + j + 8), _mm_shuffle_epi8(both1, shuf1));
        int nr1 = expand_popcnt[m1];
        rc += nr1; lc += (8 - nr1);
    }
}

/* ============ SSE_COM: 64 codes/iter = 8 independent 8-code pshufb merges,
 * cursors precomputed from a byte-granular prefix sum (pc * 0x0101..).
 * Each chunk = exactly 1 bm byte = 8 codes, so cr[k] = prefix of pc bytes. */
__attribute__((always_inline)) static inline void sse_com_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64;
        memcpy(&mask_u64, bm + (j >> 3), 8);
        uint64_t pc_u64 = popcnt_bytes_u64(mask_u64);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;  /* inclusive prefix sums */

        #define CH(K_) do {                                                     \
            uint8_t m  = (uint8_t)(mask_u64 >> (8*(K_)));                        \
            uint32_t cr = (K_)==0 ? 0 : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
            uint32_t cl = 8*(K_) - cr;                                          \
            __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc + cl));      \
            __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc + cr));     \
            __m128i both = _mm_unpacklo_epi64(L, R);                            \
            __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);      \
            _mm_storel_epi64((__m128i *)(out + j + 8*(K_)),                     \
                             _mm_shuffle_epi8(both, shuf));                      \
        } while (0)
        CH(0); CH(1); CH(2); CH(3); CH(4); CH(5); CH(6); CH(7);
        #undef CH
        uint32_t total_r = (uint32_t)(pfx >> 56);
        rc += total_r; lc += 64 - total_r;
    }
    (void)j;
}

/* ============ sse_com_pshufb: identical to sse_com but pshufb popcount. */
__attribute__((always_inline)) static inline void sse_com_pshufb_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64;
        memcpy(&mask_u64, bm + (j >> 3), 8);
        uint64_t pc_u64 = popcnt_bytes_u64_pshufb(mask_u64);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;
        #define CHP(K_) do {                                                    \
            uint8_t m  = (uint8_t)(mask_u64 >> (8*(K_)));                        \
            uint32_t cr = (K_)==0 ? 0 : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
            uint32_t cl = 8*(K_) - cr;                                          \
            __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc + cl));      \
            __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc + cr));     \
            __m128i both = _mm_unpacklo_epi64(L, R);                            \
            __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);      \
            _mm_storel_epi64((__m128i *)(out + j + 8*(K_)),                     \
                             _mm_shuffle_epi8(both, shuf));                      \
        } while (0)
        CHP(0); CHP(1); CHP(2); CHP(3); CHP(4); CHP(5); CHP(6); CHP(7);
        #undef CHP
        uint32_t total_r = (uint32_t)(pfx >> 56);
        rc += total_r; lc += 64 - total_r;
    }
    (void)j;
}

/* ============ sse_com128: 128 codes/iter = 16 chunks; pshufb popcount of all
 * 16 bm bytes in one shot, two u64 prefix sums (hi biased by lo total). */
__attribute__((always_inline)) static inline void sse_com128_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 128 <= K; j += 128) {
        __m128i bmv = _mm_loadu_si128((const __m128i *)(bm + (j >> 3)));
        __m128i pcv = popcnt_bytes_xmm(bmv);
        uint64_t mask_lo = (uint64_t)_mm_cvtsi128_si64(bmv);
        uint64_t mask_hi = (uint64_t)_mm_extract_epi64(bmv, 1);
        uint64_t pc_lo   = (uint64_t)_mm_cvtsi128_si64(pcv);
        uint64_t pc_hi   = (uint64_t)_mm_extract_epi64(pcv, 1);
        uint64_t pfx_lo = pc_lo * 0x0101010101010101ULL;
        uint64_t pfx_hi = pc_hi * 0x0101010101010101ULL;
        pfx_hi += (pfx_lo >> 56) * 0x0101010101010101ULL;
        #define CH16(K_, BMW, PFX, EBASE) do {                                  \
            uint8_t m  = (uint8_t)((BMW) >> (8*(K_)));                           \
            uint32_t cr = ((K_)==0 && (EBASE)==0) ? 0                            \
                        : ((K_)==0 ? (uint32_t)((pfx_lo>>56)&0xFF)              \
                        : (uint32_t)(((PFX) >> (8*((K_)-1))) & 0xFF));          \
            uint32_t cl = ((EBASE)+8*(K_)) - cr;                               \
            __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc + cl));      \
            __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc + cr));     \
            __m128i both = _mm_unpacklo_epi64(L, R);                            \
            __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);      \
            _mm_storel_epi64((__m128i *)(out + j + (EBASE) + 8*(K_)),           \
                             _mm_shuffle_epi8(both, shuf));                      \
        } while (0)
        CH16(0,mask_lo,pfx_lo,0); CH16(1,mask_lo,pfx_lo,0); CH16(2,mask_lo,pfx_lo,0); CH16(3,mask_lo,pfx_lo,0);
        CH16(4,mask_lo,pfx_lo,0); CH16(5,mask_lo,pfx_lo,0); CH16(6,mask_lo,pfx_lo,0); CH16(7,mask_lo,pfx_lo,0);
        CH16(0,mask_hi,pfx_hi,64); CH16(1,mask_hi,pfx_hi,64); CH16(2,mask_hi,pfx_hi,64); CH16(3,mask_hi,pfx_hi,64);
        CH16(4,mask_hi,pfx_hi,64); CH16(5,mask_hi,pfx_hi,64); CH16(6,mask_hi,pfx_hi,64); CH16(7,mask_hi,pfx_hi,64);
        #undef CH16
        uint32_t total_r = (uint32_t)(pfx_hi >> 56);
        rc += total_r; lc += 128 - total_r;
    }
    (void)j;
}

#ifdef __AVX2__
/* ============ AVX2_COM: 16 codes per _mm256_shuffle_epi8.  Low 128-bit lane
 * merges chunk 2k (L,R at its cursors), high lane merges chunk 2k+1.  The
 * 32-byte shuffle holds expand_tab[m_even] in low lane, expand_tab[m_odd]+
 * (lane-local) in high lane.  4 such 256-bit ops = 64 codes/iter. */
__attribute__((always_inline)) static inline void avx2_com_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64;
        memcpy(&mask_u64, bm + (j >> 3), 8);
        uint64_t pc_u64 = popcnt_bytes_u64(mask_u64);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;

        /* Pair (2k, 2k+1) -> one 256-bit op.  Each 128-bit lane is an
         * independent (unpacklo(L8,R8), pshufb(expand_tab[m])).  vpshufb is
         * per-lane so indices 0..15 address within each lane's own 16 bytes. */
        #define PAIR(P) do {                                                    \
            int k0 = 2*(P), k1 = 2*(P)+1;                                       \
            uint8_t ma = (uint8_t)(mask_u64 >> (8*k0));                         \
            uint8_t mb = (uint8_t)(mask_u64 >> (8*k1));                         \
            uint32_t cra = k0==0 ? 0 : (uint32_t)((pfx >> (8*(k0-1))) & 0xFF);  \
            uint32_t crb =          (uint32_t)((pfx >> (8*(k1-1))) & 0xFF);     \
            uint32_t cla = 8*k0 - cra, clb = 8*k1 - crb;                        \
            __m128i La = _mm_loadl_epi64((const __m128i *)(left + lc + cla));    \
            __m128i Ra = _mm_loadl_epi64((const __m128i *)(right + rc + cra));   \
            __m128i Lb = _mm_loadl_epi64((const __m128i *)(left + lc + clb));    \
            __m128i Rb = _mm_loadl_epi64((const __m128i *)(right + rc + crb));   \
            __m256i both = _mm256_set_m128i(_mm_unpacklo_epi64(Lb, Rb),         \
                                            _mm_unpacklo_epi64(La, Ra));        \
            __m128i sa = _mm_loadl_epi64((const __m128i *)expand_tab[ma]);       \
            __m128i sb = _mm_loadl_epi64((const __m128i *)expand_tab[mb]);       \
            __m256i shuf = _mm256_set_m128i(sb, sa);                            \
            __m256i o = _mm256_shuffle_epi8(both, shuf);                        \
            _mm_storel_epi64((__m128i *)(out + j + 8*k0), _mm256_castsi256_si128(o)); \
            _mm_storel_epi64((__m128i *)(out + j + 8*k1), _mm256_extracti128_si256(o,1)); \
        } while (0)
        PAIR(0); PAIR(1); PAIR(2); PAIR(3);
        #undef PAIR
        uint32_t total_r = (uint32_t)(pfx >> 56);
        rc += total_r; lc += 64 - total_r;
    }
    (void)j;
}
#endif

#ifdef __AVX512VBMI2__
/* ============ AVX512: production vpexpandb merge (64 codes/iter, two
 * masked expand-loads OR'd).  Verbatim from primitives_avx512.h main loop. */
__attribute__((always_inline)) static inline void avx512_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __mmask64 nm = ~m;
        __m512i L = _mm512_maskz_expandloadu_epi8(nm, left + lc);
        __m512i R = _mm512_maskz_expandloadu_epi8(m,  right + rc);
        _mm512_storeu_si512((__m512i *)(out + j), _mm512_or_si512(L, R));
        int nr = __builtin_popcountll(mask);
        rc += nr; lc += (64 - nr);
    }
    (void)j;
}
#endif

/* ============ scalar reference */
static void sca_merge(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0;
    for (int jj = 0; jj < K; jj++) {
        int b = (bm[jj >> 3] >> (jj & 7)) & 1;
        out[jj] = b ? right[rc++] : left[lc++];
    }
}

/* ============ perf */
static int g_perf_fd = -1, g_use_cyc = 0;
static void perf_init(void) {
    struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_HARDWARE; pe.size = sizeof pe;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
    g_perf_fd = (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    if (g_perf_fd < 0) { fprintf(stderr,"perf: %s\n", strerror(errno)); g_use_cyc=0; return; }
    g_use_cyc = 1;
}
static inline void perf_start(void){ if(g_use_cyc){ioctl(g_perf_fd,PERF_EVENT_IOC_RESET,0);ioctl(g_perf_fd,PERF_EVENT_IOC_ENABLE,0);} }
static inline uint64_t perf_stop(void){ if(!g_use_cyc)return 0; ioctl(g_perf_fd,PERF_EVENT_IOC_DISABLE,0); uint64_t c=0; if(read(g_perf_fd,&c,sizeof c)!=sizeof c)c=0; return c; }
static double ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }

#define K       (8 * 1024)
#define N_REPS  5000
#define N_ROUNDS 20
static uint8_t g_bm[K/8];
static uint8_t g_left[K + 128];
static uint8_t g_right[K + 128];
static uint8_t g_out[K + 128];
static uint8_t g_ref[K + 128];
static volatile uint64_t g_sink;

#define BENCH_ONE(NAME) do {                                                  \
    NAME##_merge(g_bm, K, g_left, g_right, g_out);                            \
    uint64_t best_cyc = UINT64_MAX; double best_ns = 1e18;                    \
    for (int _r=0;_r<N_ROUNDS;_r++){                                          \
        perf_start(); double _t0=ns();                                        \
        for(int _i=0;_i<N_REPS;_i++) NAME##_merge(g_bm,K,g_left,g_right,g_out);\
        double _t1=ns(); uint64_t _c=perf_stop();                            \
        uint64_t _s=0; for(int _b=0;_b<K;_b++)_s+=g_out[_b]; g_sink^=_s;       \
        if(_c<best_cyc)best_cyc=_c; if(_t1-_t0<best_ns)best_ns=_t1-_t0;        \
    }                                                                         \
    uint64_t _e=(uint64_t)N_REPS*K;                                          \
    if(g_use_cyc) printf("  %-9s %7.4f cyc/elem  %7.4f ns/elem\n",#NAME,(double)best_cyc/_e,best_ns/_e); \
    else          printf("  %-9s %7.4f ns/elem\n",#NAME,best_ns/_e);          \
} while(0)

static int verify(const char *nm, void(*f)(const uint8_t*,int,const uint8_t*,const uint8_t*,uint8_t*)) {
    memset(g_out,0,sizeof g_out);
    f(g_bm,K,g_left,g_right,g_out);
    for(int i=0;i<K;i++) if(g_out[i]!=g_ref[i]){ printf("%s MISMATCH @%d ref=%02x got=%02x\n",nm,i,g_ref[i],g_out[i]); return 1; }
    return 0;
}

int main(void) {
    build_tables();
    uint32_t x=0xC0FFEE13;
    for(size_t i=0;i<sizeof g_bm;i++){x^=x<<13;x^=x>>17;x^=x<<5;g_bm[i]=(uint8_t)x;}
    for(size_t i=0;i<sizeof g_left;i++){x^=x<<13;x^=x>>17;x^=x<<5;g_left[i]=(uint8_t)x;g_right[i]=(uint8_t)(x^0xA5);}
    perf_init();
    printf("# x86 merge K=%d reps=%d best-of-%d\n", K, N_REPS, N_ROUNDS);
    printf("# Counter: %s\n", g_use_cyc?"CPU_CYCLES":"wall ns");
    sca_merge(g_bm,K,g_left,g_right,g_ref);
    int fail=0;
    fail+=verify("old", old_merge);
    fail+=verify("sse_com", sse_com_merge);
    fail+=verify("sse_com_pshufb", sse_com_pshufb_merge);
    fail+=verify("sse_com128", sse_com128_merge);
#ifdef __AVX2__
    fail+=verify("avx2_com", avx2_com_merge);
#endif
#ifdef __AVX512VBMI2__
    fail+=verify("avx512", avx512_merge);
#endif
    if(fail) return 1;
    printf("# verify OK\n");
    BENCH_ONE(old);
    BENCH_ONE(sse_com);
    BENCH_ONE(sse_com_pshufb);
    BENCH_ONE(sse_com128);
#ifdef __AVX2__
    BENCH_ONE(avx2_com);
#endif
#ifdef __AVX512VBMI2__
    BENCH_ONE(avx512);
#endif
    return 0;
}
