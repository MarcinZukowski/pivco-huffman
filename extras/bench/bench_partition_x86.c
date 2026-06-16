/* bench_partition_x86.c -- x86 encode partition: production 2x-unroll
 * (serial cursor) vs the NEON prefix-sum cursor-decoupling COM ported to x86.
 *
 *   old : production build_bitmap_partition_x86 (16 codes/iter, 2x-unroll,
 *         per-chunk compress_popcnt[mask] load, per-byte bm store, chunk-1
 *         store address depends on chunk-0 popcount)
 *   com : 64 codes/iter, 8 chunks.  Load all 8 code vecs first (in-place
 *         left writes have no RAW hazard), 8x pmovmskb -> mask_word, one
 *         8-byte bm store, SWAR bytewise popcount + 0x0101.. prefix sum to
 *         precompute every chunk's left/right cursor, then 8 independent
 *         compact+scatter.  No compress_popcnt[] load; no serial cursor.
 *
 * Finding (see IDEAS.md "x86 COM merge / partition"): the COM transform that
 * wins big on NEON does NOT cleanly transfer to x86 -- same AMD-wins /
 * Intel-regresses split as the x86 merge COM.  On x86 the movemask is
 * already a single cheap pmovmskb (no addv to eliminate -- NEON's biggest
 * lever), so COM only offers bm-store-batching + popcnt-load-elimination +
 * cursor-decouple; AMD's wide OoO turns that into a win, Intel's narrower
 * frontend can't absorb the bigger 8-chunk body + SWAR popcount.  Microbench
 * (clang-20, ns/elem, partition cost with memcpy-restore subtracted):
 *   c6a (Zen 3)        0.207 -> 0.197  (-5%)
 *   c5a (Zen 2)        0.244 -> 0.212  (-13%)
 *   c4  (Haswell)      0.218 -> 0.256  (+17%, regress)
 *   c5  (Cascade Lake) 0.216 -> 0.254  (+18%, regress)
 *   c3  (Ivy Bridge)   0.243 -> 0.270  (+11%, regress)
 * Not shipped (Intel regression shows already in the microbench).  Kept for
 * the record / a possible AMD-gated retry.
 *
 * Destructive (left compacts in place); each timed call memcpy-restores a
 * pristine input, measured separately and subtracted.
 *
 * perf_event_open CPU_CYCLES (Linux x86; wall-ns fallback).
 * Build: clang-20 -O3 -march=native -o bp extras/bench/bench_partition_x86.c
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

static uint8_t compress_tab[256][32] __attribute__((aligned(32)));
static uint8_t compress_popcnt[256];
static void build_tables(void) {
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0;
        for (int i = 0; i < 8; i++) if (mask & (1<<i)) {
            compress_tab[mask][out_r*2]=(uint8_t)(i*2); compress_tab[mask][out_r*2+1]=(uint8_t)(i*2+1); out_r++; }
        for (; out_r < 8; out_r++){ compress_tab[mask][out_r*2]=0xFF; compress_tab[mask][out_r*2+1]=0xFF; }
        int out_l = 0;
        for (int i = 0; i < 8; i++) if (!(mask & (1<<i))) {
            compress_tab[mask][16+out_l*2]=(uint8_t)(i*2); compress_tab[mask][16+out_l*2+1]=(uint8_t)(i*2+1); out_l++; }
        for (; out_l < 8; out_l++){ compress_tab[mask][16+out_l*2]=0xFF; compress_tab[mask][16+out_l*2+1]=0xFF; }
        compress_popcnt[mask] = (uint8_t)__builtin_popcount(mask);
    }
}

static inline uint8_t enc_mask8_x86(__m128i code_vec, __m128i shift_count) {
    __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
    __m128i bytes   = _mm_packs_epi16(shifted, _mm_setzero_si128());
    return (uint8_t)_mm_movemask_epi8(bytes);
}
static inline uint64_t popcnt_bytes_u64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return x;
}

/* ============ OLD: production 2x-unroll. */
__attribute__((always_inline)) static inline int old_partition(uint16_t *codes_la, int n,
                       int depth, uint8_t *bm, uint16_t *right_out) {
    uint16_t *lp = codes_la, *rp = right_out;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 16 <= n; j += 16) {
        __m128i code0 = _mm_loadu_si128((const __m128i *)(codes_la+j));
        __m128i code1 = _mm_loadu_si128((const __m128i *)(codes_la+j+8));
        uint8_t m0 = enc_mask8_x86(code0, shift_count), m1 = enc_mask8_x86(code1, shift_count);
        bm[j>>3]=m0; bm[(j>>3)+1]=m1;
        const uint8_t *t0=compress_tab[m0], *t1=compress_tab[m1];
        __m128i r0=_mm_shuffle_epi8(code0,_mm_load_si128((const __m128i*)t0));
        __m128i l0=_mm_shuffle_epi8(code0,_mm_load_si128((const __m128i*)(t0+16)));
        __m128i r1=_mm_shuffle_epi8(code1,_mm_load_si128((const __m128i*)t1));
        __m128i l1=_mm_shuffle_epi8(code1,_mm_load_si128((const __m128i*)(t1+16)));
        int nr0=compress_popcnt[m0], nr1=compress_popcnt[m1];
        _mm_storeu_si128((__m128i*)rp, r0);
        _mm_storeu_si128((__m128i*)(rp+nr0), r1);
        _mm_storeu_si128((__m128i*)lp, l0);
        _mm_storeu_si128((__m128i*)(lp+(8-nr0)), l1);
        rp += nr0+nr1; lp += (8-nr0)+(8-nr1);
    }
    int n_right = (int)(rp - right_out), n_left = (int)(lp - codes_la);
    int shift_d = 15 - depth;
    for (; j < n; j++){ uint16_t c=codes_la[j]; if((c>>shift_d)&1) right_out[n_right++]=c; else codes_la[n_left++]=c; }
    return n_right;
}

/* ============ COM: 64 codes/iter, 8 chunks, prefix-sum cursors. */
__attribute__((always_inline)) static inline int com_partition(uint16_t *codes_la, int n,
                       int depth, uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 64 <= n; j += 64) {
        __m128i c0=_mm_loadu_si128((const __m128i*)(codes_la+j)),    c1=_mm_loadu_si128((const __m128i*)(codes_la+j+8)),
                c2=_mm_loadu_si128((const __m128i*)(codes_la+j+16)), c3=_mm_loadu_si128((const __m128i*)(codes_la+j+24)),
                c4=_mm_loadu_si128((const __m128i*)(codes_la+j+32)), c5=_mm_loadu_si128((const __m128i*)(codes_la+j+40)),
                c6=_mm_loadu_si128((const __m128i*)(codes_la+j+48)), c7=_mm_loadu_si128((const __m128i*)(codes_la+j+56));
        uint64_t mask_word = (uint64_t)enc_mask8_x86(c0,shift_count)
            | ((uint64_t)enc_mask8_x86(c1,shift_count)<<8)  | ((uint64_t)enc_mask8_x86(c2,shift_count)<<16)
            | ((uint64_t)enc_mask8_x86(c3,shift_count)<<24) | ((uint64_t)enc_mask8_x86(c4,shift_count)<<32)
            | ((uint64_t)enc_mask8_x86(c5,shift_count)<<40) | ((uint64_t)enc_mask8_x86(c6,shift_count)<<48)
            | ((uint64_t)enc_mask8_x86(c7,shift_count)<<56);
        memcpy(bm + (j>>3), &mask_word, 8);
        uint64_t pfx = popcnt_bytes_u64(mask_word) * 0x0101010101010101ULL;
#define PC(K_, CV) do {                                                     \
        uint8_t M = (uint8_t)(mask_word >> (8*(K_)));                        \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
        uint32_t cl = 8u*(K_) - cr;                                         \
        const uint8_t *t = compress_tab[M];                                \
        _mm_storeu_si128((__m128i*)(right_out + n_right + cr),             \
                         _mm_shuffle_epi8(CV,_mm_load_si128((const __m128i*)t)));\
        _mm_storeu_si128((__m128i*)(codes_la + n_left + cl),               \
                         _mm_shuffle_epi8(CV,_mm_load_si128((const __m128i*)(t+16))));\
    } while (0)
        PC(0,c0); PC(1,c1); PC(2,c2); PC(3,c3); PC(4,c4); PC(5,c5); PC(6,c6); PC(7,c7);
#undef PC
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += total_r; n_left += 64 - total_r;
    }
    int shift_d = 15 - depth;
    for (; j < n; j++){ uint16_t c=codes_la[j]; if((c>>shift_d)&1) right_out[n_right++]=c; else codes_la[n_left++]=c; }
    return n_right;
}

static int sca_partition(uint16_t *codes_la, int n, int depth, uint8_t *bm, uint16_t *right_out) {
    int n_left=0,n_right=0,shift_d=15-depth;
    for (int j=0;j<n;j++){ if((j&7)==0) bm[j>>3]=0; uint16_t c=codes_la[j];
        if((c>>shift_d)&1){ bm[j>>3]|=(uint8_t)(1<<(j&7)); right_out[n_right++]=c; } else codes_la[n_left++]=c; }
    return n_right;
}

static int g_perf_fd=-1, g_use_cyc=0;
static void perf_init(void){ struct perf_event_attr pe; memset(&pe,0,sizeof pe);
    pe.type=PERF_TYPE_HARDWARE; pe.size=sizeof pe; pe.config=PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled=1; pe.exclude_kernel=1; pe.exclude_hv=1;
    g_perf_fd=(int)syscall(SYS_perf_event_open,&pe,0,-1,-1,0);
    if(g_perf_fd<0){fprintf(stderr,"perf: %s\n",strerror(errno)); g_use_cyc=0; return;} g_use_cyc=1; }
static inline void perf_start(void){ if(g_use_cyc){ioctl(g_perf_fd,PERF_EVENT_IOC_RESET,0);ioctl(g_perf_fd,PERF_EVENT_IOC_ENABLE,0);} }
static inline uint64_t perf_stop(void){ if(!g_use_cyc)return 0; ioctl(g_perf_fd,PERF_EVENT_IOC_DISABLE,0); uint64_t c=0; if(read(g_perf_fd,&c,sizeof c)!=sizeof c)c=0; return c; }
static double ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }

#define N 8192
#define N_REPS 4000
#define N_ROUNDS 12
#define DEPTH 3
static uint16_t g_pristine[N];
static uint16_t g_work[N+16];
static uint16_t g_right[N+16];
static uint8_t  g_bm[N/8+8];
static volatile uint64_t g_sink;

#define BENCH_ONE(NAME) do {                                                  \
    memcpy(g_work,g_pristine,sizeof g_pristine); NAME##_partition(g_work,N,DEPTH,g_bm,g_right); \
    uint64_t bc=UINT64_MAX; double bn=1e18;                                   \
    for(int _r=0;_r<N_ROUNDS;_r++){ perf_start(); double t0=ns();             \
        for(int _i=0;_i<N_REPS;_i++){ memcpy(g_work,g_pristine,sizeof g_pristine); g_sink^=(uint64_t)NAME##_partition(g_work,N,DEPTH,g_bm,g_right);} \
        double t1=ns(); uint64_t c=perf_stop(); if(c<bc)bc=c; if(t1-t0<bn)bn=t1-t0; } \
    uint64_t mc=UINT64_MAX; double mn=1e18;                                   \
    for(int _r=0;_r<N_ROUNDS;_r++){ perf_start(); double t0=ns();             \
        for(int _i=0;_i<N_REPS;_i++){ memcpy(g_work,g_pristine,sizeof g_pristine); g_sink^=g_work[0]; } \
        double t1=ns(); uint64_t c=perf_stop(); if(c<mc)mc=c; if(t1-t0<mn)mn=t1-t0; } \
    uint64_t e=(uint64_t)N_REPS*N; double cyc=(double)(bc-mc)/e, nsec=(bn-mn)/e; \
    if(g_use_cyc) printf("  %-7s %7.4f cyc/elem  %7.4f ns/elem\n",#NAME,cyc,nsec); \
    else          printf("  %-7s %7.4f ns/elem\n",#NAME,nsec);                \
} while(0)

static int verify(const char *nm, int(*f)(uint16_t*,int,int,uint8_t*,uint16_t*)){
    uint16_t rw[N],rr[N]; uint8_t rb[N/8+8];
    memcpy(rw,g_pristine,sizeof g_pristine); int rnr=sca_partition(rw,N,DEPTH,rb,rr);
    memcpy(g_work,g_pristine,sizeof g_pristine); int nr=f(g_work,N,DEPTH,g_bm,g_right);
    if(nr!=rnr){printf("%s n_right %d!=%d\n",nm,nr,rnr); return 1;}
    if(memcmp(g_bm,rb,N/8)){printf("%s bm mismatch\n",nm); return 1;}
    if(memcmp(g_work,rw,(size_t)(N-rnr)*2)){printf("%s left mismatch\n",nm); return 1;}
    if(memcmp(g_right,rr,(size_t)rnr*2)){printf("%s right mismatch\n",nm); return 1;}
    return 0;
}

int main(void){
    build_tables();
    uint32_t x=0xC0FFEE13;
    for(int i=0;i<N;i++){x^=x<<13;x^=x>>17;x^=x<<5; g_pristine[i]=(uint16_t)(x&0xFFFF);}
    perf_init();
    printf("# x86 partition N=%d depth=%d reps=%d best-of-%d\n",N,DEPTH,N_REPS,N_ROUNDS);
    printf("# Counter: %s\n", g_use_cyc?"CPU_CYCLES":"wall ns");
    int fail=0; fail+=verify("old",old_partition); fail+=verify("com",com_partition);
    if(fail) return 1; printf("# verify OK\n");
    BENCH_ONE(old); BENCH_ONE(com);
    return 0;
}
