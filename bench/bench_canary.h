/* bench_canary.h — deterministic fixed-work probes for detecting whether the
 * measurement environment shifted mid-run (core co-scheduling, clock, memory
 * contention).  Each probe is fixed work; its wall time only moves if the box
 * moves under us.  Print one before/after (or between) timed sections and
 * compare: a stable canary means the surrounding numbers are trustworthy; a
 * canary that jumps (esp. compute ~2x) means that section shared the core and
 * should be discarded.  See the Skylake floor-lottery investigation.
 *
 *   compute — dependent LCG chain (mul+add) -> core frequency / core sharing
 *   bw      — streaming read of a >L3 buffer -> DRAM bandwidth
 *   lat     — pointer-chase over a Sattolo cycle -> DRAM latency
 *
 * Header-only; one static buffer per TU, allocated lazily and reused. */
#ifndef BENCH_CANARY_H
#define BENCH_CANARY_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifndef BENCH_CANARY_MB
#define BENCH_CANARY_MB     48u          /* > typical 32-36 MB L3 -> real DRAM */
#endif
#ifndef BENCH_CANARY_ITERS
#define BENCH_CANARY_ITERS  100000000ULL /* ~0.1 s at ~1.1 ns/iter */
#endif
#define BENCH_CANARY_ELEMS      ((BENCH_CANARY_MB*1024u*1024u)/8u)
#define BENCH_CANARY_BW_PASSES  4
#define BENCH_CANARY_LAT_STEPS  500000u
#ifndef BENCH_CANARY_WARN_PCT
#define BENCH_CANARY_WARN_PCT   5.0   /* warn if compute peak-to-peak exceeds this */
#endif

static double bench_canary_now_ns(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec*1e9 + (double)t.tv_nsec;
}

static uint64_t *bench_canary_buf = NULL;

/* running min/max/avg per metric, accumulated across every bench_canary() call */
static struct bench_canary_stats_s {
    long   cn;  double csum, cmin, cmax;   /* compute ns/it  (every call) */
    long   mn;  double bsum, bmin, bmax;   /* bw GB/s        (calls with buffer) */
                double lsum, lmin, lmax;   /* lat ns/acc     (calls with buffer) */
} bench_canary_stats = { 0, 0, 1e300, -1e300, 0, 0, 1e300, -1e300, 0, 1e300, -1e300 };

static void bench_canary_track(double c, int have_mem, double b, double l){
    struct bench_canary_stats_s *S = &bench_canary_stats;
    S->cn++; S->csum += c; if (c < S->cmin) S->cmin = c; if (c > S->cmax) S->cmax = c;
    if (have_mem) {
        S->mn++;
        S->bsum += b; if (b < S->bmin) S->bmin = b; if (b > S->bmax) S->bmax = b;
        S->lsum += l; if (l < S->lmin) S->lmin = l; if (l > S->lmax) S->lmax = l;
    }
}

/* Run all three probes and print one labelled line.  Non-fatal on OOM (skips
 * the memory probes). */
static void bench_canary(const char *label){
    if (!bench_canary_buf) {
        bench_canary_buf = (uint64_t*)malloc((size_t)BENCH_CANARY_ELEMS*8);
        if (bench_canary_buf)
            for (uint32_t i=0;i<BENCH_CANARY_ELEMS;i++) bench_canary_buf[i]=i*2654435761u+1;
    }

    /* compute: dependent LCG chain (mul+add) with a per-iteration asm barrier.
       Without the barrier the optimizer defeats this probe two ways: a bare
       multiply reduction (f*=i) is associative and auto-vectorises into
       VPMULLQ, and even an LCG gets closed-form / lane-split under some
       flags — both read a bogus sub-cycle ns/iter.  The empty asm forces x
       live in a register each step, so it stays a true latency chain
       (~1.1-1.5 ns/iter) on every compiler and flag set. */
    volatile uint64_t seed=1; uint64_t x=seed;
    double t0=bench_canary_now_ns();
    for (uint64_t i=0;i<BENCH_CANARY_ITERS;i++) {
        x = x*6364136223846793005ULL + 1442695040888963407ULL;
        __asm__ volatile("" : "+r"(x));
    }
    double cpu_ms=(bench_canary_now_ns()-t0)/1e6;
    double ns_it=cpu_ms*1e6/(double)BENCH_CANARY_ITERS;

    double gbps=0, lat_ns=0; uint64_t sink=x;
    if (bench_canary_buf) {
        uint64_t *b=bench_canary_buf;
        for (uint32_t i=0;i<BENCH_CANARY_ELEMS;i++) b[i]=i*2654435761u+1;   /* refill */
        /* bandwidth: streaming read */
        t0=bench_canary_now_ns();
        uint64_t s0=0,s1=0,s2=0,s3=0;
        for (int p=0;p<BENCH_CANARY_BW_PASSES;p++)
            for (uint32_t i=0;i+4<=BENCH_CANARY_ELEMS;i+=4){ s0+=b[i]; s1+=b[i+1]; s2+=b[i+2]; s3+=b[i+3]; }
        double bw_ms=(bench_canary_now_ns()-t0)/1e6;
        gbps=((double)BENCH_CANARY_ELEMS*8.0*BENCH_CANARY_BW_PASSES)/(bw_ms/1e3)/1e9;
        sink^=s0^s1^s2^s3;
        /* latency: Sattolo pointer-chase in the same buffer */
        for (uint32_t i=0;i<BENCH_CANARY_ELEMS;i++) b[i]=i;
        uint64_t r=0x9e3779b97f4a7c15ULL;
        for (uint32_t i=BENCH_CANARY_ELEMS-1;i>0;i--){ r^=r<<13; r^=r>>7; r^=r<<17; uint32_t j=(uint32_t)(r%i); uint64_t t=b[i]; b[i]=b[j]; b[j]=t; }
        uint64_t p=0; t0=bench_canary_now_ns();
        for (uint32_t s=0;s<BENCH_CANARY_LAT_STEPS;s++) p=b[p];
        lat_ns=(bench_canary_now_ns()-t0)/BENCH_CANARY_LAT_STEPS;
        sink^=p;
    }

    bench_canary_track(ns_it, bench_canary_buf != NULL, gbps, lat_ns);

    /* print the checksum so the compiler can't DCE the fixed-work loops */
    printf("CANARY %-12s compute=%.1f ms (%.4f ns/it) | bw=%.1f GB/s | lat=%.1f ns/acc  chk=%llx\n",
           label, cpu_ms, ns_it, gbps, lat_ns, (unsigned long long)sink);
    fflush(stdout);
}

/* Print min/max/avg per metric over all bench_canary() calls, and warn if the
 * compute probe's peak-to-peak exceeds BENCH_CANARY_WARN_PCT (a run that shared
 * the core / got throttled shows up here).  No-op if no canaries ran. */
static void bench_canary_summary(void){
    struct bench_canary_stats_s *S = &bench_canary_stats;
    if (S->cn == 0) return;
    double cspread = 100.0 * (S->cmax - S->cmin) / S->cmin;
    printf("\nCANARY summary (%ld probe%s):\n", S->cn, S->cn==1 ? "" : "s");
    printf("  compute  min=%.4f  max=%.4f  avg=%.4f ns/it   spread=%.1f%%%s\n",
           S->cmin, S->cmax, S->csum / (double)S->cn, cspread,
           cspread > BENCH_CANARY_WARN_PCT ? "   *** WARNING: compute varied "
           "beyond threshold — a run likely shared the core / was throttled;"
           " treat this session's numbers with suspicion ***" : "");
    if (S->mn) {
        double bspread = 100.0 * (S->bmax - S->bmin) / S->bmin;
        double lspread = 100.0 * (S->lmax - S->lmin) / S->lmin;
        printf("  bw       min=%.1f  max=%.1f  avg=%.1f GB/s   spread=%.1f%%\n",
               S->bmin, S->bmax, S->bsum / (double)S->mn, bspread);
        printf("  lat      min=%.1f  max=%.1f  avg=%.1f ns/acc  spread=%.1f%%\n",
               S->lmin, S->lmax, S->lsum / (double)S->mn, lspread);
    }
    fflush(stdout);
}

#endif /* BENCH_CANARY_H */
