#define _GNU_SOURCE  /* for sched_setaffinity / CPU_SET on Linux */
#include "pivco_prof.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#elif defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

pivco_prof_counter_t pivco_prof_counters[PROF_COUNT];

static const char *prof_names[PROF_COUNT] = {
    [PROF_NODE_FULL]           = "node full       (decode_node)",
    [PROF_NODE_HALF_RIGHT]     = "node half-right (decode_node)",
    [PROF_NODE_HALF_LEFT]      = "node half-left  (decode_node)",
    [PROF_ROOT_FULL]           = "root full       (entry)",
    [PROF_ROOT_HALF_RIGHT]     = "root half-right (entry)",
    [PROF_ROOT_HALF_LEFT]      = "root half-left  (entry)",
    [PROF_SCATTER_SYM]         = "scatter_sym",
    [PROF_SCATTER_BOTH_LEAVES] = "scatter_both_leaves",
    [PROF_FLAT_DECODE_SCATTER] = "flat_decode_scatter",
    [PROF_FLAT_DECODE_DIRECT]  = "flat_decode_direct",
    [PROF_DECODE_NODE]         = "decode_node (call count)",
    [PROF_DECODE_ENTRY]        = "pivco_huffman_decode (entry)",
};

const char *pivco_prof_name(pivco_prof_id_t id) {
    return (id < PROF_COUNT && prof_names[id]) ? prof_names[id] : "?";
}

void pivco_prof_reset(void) {
    memset(pivco_prof_counters, 0, sizeof(pivco_prof_counters));
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double pivco_prof_probe_tick_freq(void) {
#ifdef PIVCO_PROF
    double t0 = now_sec();
    uint64_t c0 = pivco_prof_tick();
    while (now_sec() - t0 < 0.1) { /* spin */ }
    double t1 = now_sec();
    uint64_t c1 = pivco_prof_tick();
    return (double)(c1 - c0) / (t1 - t0);
#else
    return 0;
#endif
}

void pivco_prof_dump(const char *label,
                     double wall_seconds,
                     double tick_freq_hz,
                     uint64_t n_blocks)
{
    printf("\n=== pivco_prof: %s ===\n", label);
    printf("  wall: %.3f s   blocks: %llu",
           wall_seconds, (unsigned long long)n_blocks);
    if (tick_freq_hz > 0)
        printf("   counter freq: %.2f MHz", tick_freq_hz / 1e6);
    printf("\n");

    printf("\n  %-32s %12s %14s %10s %12s %10s %10s\n",
           "primitive", "calls", "elements", "elem/call", "ticks",
           "ns/call", "ns/elem");
    printf("  ---------------------------------------------------------"
           "----------------------------------------\n");

    for (int i = 0; i < PROF_COUNT; i++) {
        pivco_prof_counter_t *c = &pivco_prof_counters[i];
        if (c->calls == 0) continue;

        double ns_per_call = 0, ns_per_elem = 0;
        if (c->ticks > 0 && tick_freq_hz > 0) {
            double ns = (double)c->ticks * 1e9 / tick_freq_hz;
            ns_per_call = ns / (double)c->calls;
            ns_per_elem = c->elements > 0 ? ns / (double)c->elements : 0;
        }

        char ticks_str[32];
        if (c->ticks > 0) {
            snprintf(ticks_str, sizeof(ticks_str), "%llu",
                     (unsigned long long)c->ticks);
        } else {
            snprintf(ticks_str, sizeof(ticks_str), "(count-only)");
        }

        double elem_per_call = (double)c->elements / (double)c->calls;
        printf("  %-32s %12llu %14llu %10.1f %12s %10.1f %10.2f\n",
               pivco_prof_name((pivco_prof_id_t)i),
               (unsigned long long)c->calls,
               (unsigned long long)c->elements,
               elem_per_call,
               ticks_str,
               ns_per_call, ns_per_elem);
    }

    if (n_blocks > 0) {
        printf("\n  Per-BLK averages:\n");
        for (int i = 0; i < PROF_COUNT; i++) {
            pivco_prof_counter_t *c = &pivco_prof_counters[i];
            if (c->calls == 0) continue;
            printf("    %-32s %8.1f calls/BLK %12.1f elems/BLK\n",
                   pivco_prof_name((pivco_prof_id_t)i),
                   (double)c->calls / (double)n_blocks,
                   (double)c->elements / (double)n_blocks);
        }
    }
    printf("\n");
}

int pivco_prof_pin_cpu(int cpu_id) {
    (void)cpu_id;
#ifdef __APPLE__
    /* No fine-grained pinning in user space.  USER_INTERACTIVE QoS is
     * the highest non-entitlement-restricted class; it strongly
     * prefers P-cores on Apple Silicon. */
    int rc = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    return rc == 0 ? 0 : -1;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    return sched_setaffinity(0, sizeof(set), &set);
#else
    return -1;
#endif
}
