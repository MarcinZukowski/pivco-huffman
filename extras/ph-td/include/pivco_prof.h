/* pivco_prof.h — no-op stub for the standalone TD slice.
 *
 * Upstream pivco_prof.h is a sizeable per-primitive instrumentation
 * harness with timers, counters, and a per-thread cycle clock.  None
 * of that is needed for this historical reference build, so every
 * macro the TD code expects is a no-op here. */
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PROF_NODE_FULL = 0,
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
    PROF_NR_SLOTS
} pivco_prof_slot_t;

#define PROF_TIC()                            do {} while (0)
#define PROF_TOC(slot, n)                     do { (void)(slot); (void)(n); } while (0)
#define PROF_COUNT_ONLY(slot, n)              do { (void)(slot); (void)(n); } while (0)

#ifdef __cplusplus
}
#endif
