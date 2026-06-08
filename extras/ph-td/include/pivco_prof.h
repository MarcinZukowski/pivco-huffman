/* pivco_prof.h — tic/toc instrumentation for the ph-td slice.
 *
 * Modelled on include/pivco_prof.h in the main project, but slimmed
 * to the per-primitive call sites this slice exercises.  Replaces
 * the earlier no-op stub when PIVCO_PROF is set.
 *
 * On aarch64 reads cntvct_el0 (Apple Silicon / Graviton: 1 ns/tick
 * effective resolution; nominal ~24 MHz hardware counter but the
 * userspace clock virtualises to 1 GHz on Apple Silicon).  On x86
 * reads rdtsc which runs at the base CPU frequency -- multiply ticks
 * by base_freq^-1 to get ns.
 *
 * Disabled by default: compile with -DPIVCO_PROF=1 to enable.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /* Per-primitive timed slots (naive + scalar-opt). */
    PROF_P_PARTITION = 0,    /* p_partition                  */
    PROF_P_HALF_RIGHT,       /* p_half_right                 */
    PROF_P_HALF_LEFT,        /* p_half_left                  */
    PROF_S1_SCATTER,         /* s1_scatter                   */
    PROF_S2_SCATTER_BOTH,    /* s2_scatter_both              */
    PROF_SFX_SCATTER_FLAT,   /* sfx_scatter_flat             */

    /* Count-only slots. */
    PROF_DECODE_ENTRY_NAIVE,
    PROF_DECODE_ENTRY_OPT,
    PROF_DECODE_NODE_NAIVE,
    PROF_DECODE_NODE_OPT,

    /* Legacy slots (kept for compatibility with the upstream codec.c
     * comments referenced from this slice's source files). */
    PROF_NODE_FULL,
    PROF_NODE_HALF_RIGHT,
    PROF_NODE_HALF_LEFT,
    PROF_ROOT_FULL,
    PROF_ROOT_HALF_RIGHT,
    PROF_ROOT_HALF_LEFT,
    PROF_SCATTER_SYM,
    PROF_SCATTER_BOTH_LEAVES,
    PROF_FLAT_DECODE_SCATTER,
    PROF_FLAT_DECODE_DIRECT,
    PROF_FSE_ENC,
    PROF_FSE_DEC,
    PROF_FSE_HIT_COUNT,
    PROF_FSE_RAW_COUNT,
    PROF_FSE_FALLBACK_COUNT,
    PROF_DECODE_ENTRY,
    PROF_DECODE_NODE,
    PROF_ENC_ENTRY,
    PROF_ENC_NODE_VISIT,
    PROF_ENC_NODE_FULL,
    PROF_ENC_FLAT,
    PROF_ENC_INIT,
    PROF_ENC_FLAT_SIMD_ELEMS,
    PROF_ENC_FLAT_TAIL_ELEMS,

    /* BU primitive slots — referenced by the shared (main-repo)
     * pivco_huffman_primitives_neon.h that TD now includes.  TD never calls
     * the BU primitives, but the slot identifiers must exist to compile. */
    PROF_BU_POPCOUNT_K,
    PROF_BU_TREE_MERGE,
    PROF_BU_TREE_MERGE_BCAST_LEFT,
    PROF_BU_TREE_MERGE_BCAST_RIGHT,
    PROF_BU_MERGE_BOTH_CONST,
    PROF_BU_FLAT_DECODE,
    /* New BU names introduced upstream; kept here as no-op slot identifiers
     * so the shared headers compile inside the TD slice. */
    PROF_BU_MERGE_VEC_VEC,
    PROF_BU_MERGE_CST_VEC,
    PROF_BU_MERGE_VEC_CST,
    PROF_BU_MERGE_CST_CST,
    PROF_BU_MERGE_FLAT,

    PROF_NR_SLOTS,
    PROF_COUNT = PROF_NR_SLOTS
} pivco_prof_slot_t;

typedef struct {
    uint64_t calls;
    uint64_t elements;
    uint64_t ticks;
} pivco_prof_counter_t;

extern pivco_prof_counter_t pivco_prof_counters[PROF_NR_SLOTS];

const char *pivco_prof_name(pivco_prof_slot_t slot);
void        pivco_prof_reset(void);
void        pivco_prof_dump(const char *label,
                             double wall_seconds,
                             double tick_freq_hz);
double      pivco_prof_probe_tick_freq(void);

#ifdef PIVCO_PROF

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
    return 0;
#endif
}

#define PROF_TIC() uint64_t _prof_t0 = pivco_prof_tick()

#define PROF_TOC(slot, n_elem) do {                                       \
    uint64_t _prof_t1 = pivco_prof_tick();                                \
    pivco_prof_counters[(slot)].calls++;                                  \
    pivco_prof_counters[(slot)].elements += (uint64_t)(n_elem);           \
    pivco_prof_counters[(slot)].ticks += _prof_t1 - _prof_t0;             \
} while (0)

#define PROF_COUNT_ONLY(slot, n_elem) do {                                \
    pivco_prof_counters[(slot)].calls++;                                  \
    pivco_prof_counters[(slot)].elements += (uint64_t)(n_elem);           \
} while (0)

#else  /* !PIVCO_PROF */

#define PROF_TIC()                            do {} while (0)
#define PROF_TOC(slot, n)                     do { (void)(slot); (void)(n); } while (0)
#define PROF_COUNT_ONLY(slot, n)              do { (void)(slot); (void)(n); } while (0)

#endif

#ifdef __cplusplus
}
#endif
