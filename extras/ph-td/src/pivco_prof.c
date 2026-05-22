/* pivco_prof.c — counter array + dump helpers for ph-td.
 *
 * Linked into the bench when -DPIVCO_PROF=1 is passed.  See
 * include/pivco_prof.h.
 */
#include "pivco_prof.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

pivco_prof_counter_t pivco_prof_counters[PROF_NR_SLOTS] = {0};

static const char *NAMES[PROF_NR_SLOTS] = {
    /* Scalar (naive + scalar-opt) primitives. */
    [PROF_P_PARTITION]        = "p_partition",
    [PROF_P_HALF_RIGHT]       = "p_half_right",
    [PROF_P_HALF_LEFT]        = "p_half_left",
    [PROF_S1_SCATTER]         = "s1_scatter",
    [PROF_S2_SCATTER_BOTH]    = "s2_scatter_both",
    [PROF_SFX_SCATTER_FLAT]   = "sfx_scatter_flat",
    [PROF_DECODE_ENTRY_NAIVE] = "decode_entry_naive",
    [PROF_DECODE_ENTRY_OPT]   = "decode_entry_opt",
    [PROF_DECODE_NODE_NAIVE]  = "decode_node_naive",
    [PROF_DECODE_NODE_OPT]    = "decode_node_opt",
    /* SIMD-optimised TD primitives (NEON in pivco_huffman_neon.c,
     * AVX-512 in pivco_huffman_avx512.c).  Same semantic operations
     * as the scalar ones above but per-platform-vectorised. */
    [PROF_NODE_FULL]          = "simd_partition",
    [PROF_NODE_HALF_RIGHT]    = "simd_partition_half_right",
    [PROF_NODE_HALF_LEFT]     = "simd_partition_half_left",
    [PROF_ROOT_FULL]          = "simd_partition_root",
    [PROF_ROOT_HALF_RIGHT]    = "simd_partition_root_half_right",
    [PROF_ROOT_HALF_LEFT]     = "simd_partition_root_half_left",
    [PROF_SCATTER_SYM]        = "simd_s1_scatter",
    [PROF_SCATTER_BOTH_LEAVES]= "simd_s2_scatter_both",
    [PROF_FLAT_DECODE_SCATTER]= "simd_sfx_scatter_flat",
    [PROF_FLAT_DECODE_DIRECT] = "simd_sfx_decode_direct",
};

const char *pivco_prof_name(pivco_prof_slot_t slot) {
    if ((unsigned)slot >= PROF_NR_SLOTS) return "?";
    return NAMES[slot] ? NAMES[slot] : "?";
}

void pivco_prof_reset(void) {
    memset(pivco_prof_counters, 0, sizeof(pivco_prof_counters));
}

double pivco_prof_probe_tick_freq(void) {
#ifdef PIVCO_PROF
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t c0 = pivco_prof_tick();
    /* Busy-wait ~100 ms. */
    do {
        clock_gettime(CLOCK_MONOTONIC, &t1);
    } while ((t1.tv_sec - t0.tv_sec) * 1e9 +
             (t1.tv_nsec - t0.tv_nsec) < 1e8);
    uint64_t c1 = pivco_prof_tick();
    double elapsed = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    return (double)(c1 - c0) / elapsed;
#else
    return 1e9;
#endif
}

void pivco_prof_dump(const char *label,
                     double wall_seconds,
                     double tick_freq_hz) {
    (void)wall_seconds;
    double ns_per_tick = (tick_freq_hz > 0) ? 1e9 / tick_freq_hz : 1.0;

    printf("=== profile: %s (tick=%.3f GHz, %.3f ns/tick) ===\n",
           label, tick_freq_hz * 1e-9, ns_per_tick);
    printf("%-22s %12s %14s %12s %10s %10s\n",
           "primitive", "calls", "elements", "ticks", "ns/elem", "ns/call");
    for (int i = 0; i < PROF_NR_SLOTS; i++) {
        pivco_prof_counter_t *c = &pivco_prof_counters[i];
        if (c->calls == 0) continue;
        const char *name = pivco_prof_name((pivco_prof_slot_t)i);
        if (!name || name[0] == '?') continue;
        double ns_total = (double)c->ticks * ns_per_tick;
        double ns_elem  = c->elements ? ns_total / (double)c->elements : 0.0;
        double ns_call  = (double)c->calls ? ns_total / (double)c->calls : 0.0;
        printf("%-22s %12llu %14llu %12llu %10.3f %10.1f\n",
               name,
               (unsigned long long)c->calls,
               (unsigned long long)c->elements,
               (unsigned long long)c->ticks,
               ns_elem, ns_call);
    }
    printf("\n");
}
