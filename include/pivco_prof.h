/* pivco_prof.h — lightweight per-primitive instrumentation.
 *
 * Counts calls and elements unconditionally; times a subset of larger
 * primitives via a userspace cycle counter (cntvct_el0 on aarch64,
 * rdtsc on x86).  On Apple Silicon cntvct_el0 runs at 24 MHz nominal
 * but Apple's userspace timer reports as if 1 GHz (1 ns/tick); on
 * Linux aarch64 cntvct_el0 is typically 1 GHz; on x86 rdtsc runs at
 * the base CPU frequency.  Use pivco_prof_probe_tick_freq() to convert
 * ticks to ns.
 *
 * Disabled by default (zero cost).  Enable with -DPIVCO_PROF=1.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /* Bottom-up decoder per-primitive timings (the SIMD backends'
     * merge kernels).  Elements = bytes processed at this call. */
    PROF_BU_MERGE_VEC_VEC = 0,      /* general 2-buffer merge */
    PROF_BU_MERGE_CST_VEC,          /* left side broadcast constant */
    PROF_BU_MERGE_CST_CST,          /* BOTH_LEAVES / both-leaf collapse */
    PROF_BU_MERGE_FLAT,             /* INTERNAL_FLAT direct-to-buffer */
    PROF_BU_POPCOUNT_K,             /* compute K_right from bitmap */

    /* Wire-format decode reads (charged per node). */
    PROF_WIRE_KR,                   /* read K_right:u16 header */
    PROF_WIRE_BITMAP_RAW,           /* marker==0: raw bitmap, pointer + advance */
    PROF_WIRE_BITMAP_FSE,           /* marker!=0: FSE-decompress bitmap body */

    /* Encoder (pivco_encode / codec_encode_node).  Mirrors
     * the decode side: per-primitive timing of the work done inside a
     * node body (not the recursion itself), plus the per-block setup. */
    PROF_ENC_INIT,                  /* prim_enc_init: ranks[] setup, per block */
    PROF_ENC_NODE_FULL,             /* non-flat internal node: prim_enc_partition_* */
    PROF_ENC_FLAT,                  /* flat-subtree node: prim_enc_pack_dN */
    PROF_ENC_FLAT_SIMD_ELEMS,       /* count-only: elems handled by SIMD path (u16enc) */
    PROF_ENC_FLAT_TAIL_ELEMS,       /* count-only: elems handled by scalar tail (u16enc) */

    /* Recursion + entry call counts (count-only; recursive timing
     * would double-count). */
    PROF_ENC_NODE_VISIT,            /* count-only: calls to codec_encode_node */
    PROF_ENC_ENTRY,                 /* count-only: calls to pivco_codec_encode */

    /* File-codec layer (pivcohuf_file.c).  Wraps the entire file-level
     * pipeline so the CLI can show where time goes outside the
     * block-codec inner loops. */
    PROF_FILE_BUILD_TABLE_REAL,     /* first build_table from real freqs (compress) */
    PROF_FILE_BUILD_TABLE_SYN,      /* second build_table from synth freqs (both) */
    PROF_FILE_BLOCK_ENCODE,         /* per-block pivco_encode call */
    PROF_FILE_BLOCK_DECODE,         /* per-block pivco_decode call */
    PROF_FILE_BLOCK_PROLOGUE,       /* per-block length prefix + offset math */

    PROF_COUNT
} pivco_prof_id_t;

typedef struct {
    uint64_t calls;
    uint64_t elements;
    uint64_t ticks;       /* 0 if counter is counted-only. */
} pivco_prof_counter_t;

extern pivco_prof_counter_t pivco_prof_counters[PROF_COUNT];

const char *pivco_prof_name(pivco_prof_id_t id);
void pivco_prof_reset(void);

/* Dump a per-counter table.  `wall_seconds` is the wall time of the
 * measured region; `tick_freq_hz` is the cycle-counter frequency
 * (~1 GHz on Apple Silicon / Graviton, ~CPU base GHz on x86; pass 0 to
 * skip ns conversion).  `n_blocks` is shown for per-BLK averages. */
void pivco_prof_dump(const char *label,
                     double wall_seconds,
                     double tick_freq_hz,
                     uint64_t n_blocks);

/* Probe the cycle-counter frequency by reading it twice across a
 * known wall-time interval (~100 ms). */
double pivco_prof_probe_tick_freq(void);

/* Try to pin this thread to a high-perf core.
 *   macOS: bumps QoS to USER_INTERACTIVE (P-core preference).
 *   Linux: sched_setaffinity to the requested CPU id (default 0).
 * Returns 0 on success, -1 on failure. */
int pivco_prof_pin_cpu(int cpu_id);

#ifdef PIVCO_PROF

/* Read userspace cycle counter. */
static inline uint64_t pivco_prof_tick(void) {
#if defined(__aarch64__)
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned hi, lo;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#else
    return __builtin_readcyclecounter();
#endif
}

/* Counted-only.  No timing — single instr per call. */
#define PROF_COUNT_ONLY(id, n_elem) do { \
    pivco_prof_counters[(id)].calls++; \
    pivco_prof_counters[(id)].elements += (uint64_t)(n_elem); \
} while (0)

/* Begin a timed region.  Pairs with PROF_TOC.  PROF_TIC declares the
 * start time, so a second region in the same scope re-arms it with
 * PROF_RETIC instead. */
#define PROF_TIC() uint64_t _prof_t0 = pivco_prof_tick()
#define PROF_RETIC() ((void)(_prof_t0 = pivco_prof_tick()))

/* End a timed region. */
#define PROF_TOC(id, n_elem) do { \
    uint64_t _prof_t1 = pivco_prof_tick(); \
    pivco_prof_counters[(id)].calls++; \
    pivco_prof_counters[(id)].elements += (uint64_t)(n_elem); \
    pivco_prof_counters[(id)].ticks += _prof_t1 - _prof_t0; \
} while (0)

#else  /* !PIVCO_PROF */

#define PROF_COUNT_ONLY(id, n_elem)  ((void)0)
#define PROF_TIC()                   ((void)0)
#define PROF_RETIC()                 ((void)0)
#define PROF_TOC(id, n_elem)         ((void)0)

#endif

#ifdef __cplusplus
}
#endif
