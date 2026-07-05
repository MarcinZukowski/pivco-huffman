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
    [PROF_NODE_FULL]           = "node_full",
    [PROF_NODE_HALF_RIGHT]     = "node_half_right",
    [PROF_NODE_HALF_LEFT]      = "node_half_left",
    [PROF_ROOT_FULL]           = "root_full",
    [PROF_ROOT_HALF_RIGHT]     = "root_half_right",
    [PROF_ROOT_HALF_LEFT]      = "root_half_left",
    [PROF_SCATTER_SYM]         = "scatter_sym",
    [PROF_SCATTER_BOTH_LEAVES] = "scatter_both_leaves",
    [PROF_FLAT_DECODE_SCATTER] = "flat_decode_scatter",
    [PROF_FLAT_DECODE_DIRECT]  = "flat_decode_direct",
    [PROF_BU_MERGE_VEC_VEC]            = "bu_merge_vec_vec",
    [PROF_BU_MERGE_CST_VEC] = "bu_merge_cst_vec",
    [PROF_BU_MERGE_CST_CST]      = "bu_merge_cst_cst",
    [PROF_BU_MERGE_FLAT]           = "bu_merge_flat",
    [PROF_BU_POPCOUNT_K]            = "bu_popcount_K",
    [PROF_BU_LEAF_MEMSET]           = "bu_leaf_memset",
    [PROF_WIRE_KR]                  = "wire_kr",
    [PROF_WIRE_BITMAP_RAW]          = "wire_bitmap_raw",
    [PROF_WIRE_BITMAP_FSE]          = "wire_bitmap_fse",
    [PROF_ENC_INIT]                 = "enc_init",
    [PROF_ENC_NODE_FULL]            = "enc_node_full",
    [PROF_ENC_FLAT]                 = "enc_flat",
    [PROF_ENC_FLAT_SIMD_ELEMS]      = "  enc_flat_simd",
    [PROF_ENC_FLAT_TAIL_ELEMS]      = "  enc_flat_tail",
    [PROF_ENC_REPACK_U8]            = "enc_repack_u8",
    [PROF_ENC_NODE_FULL_U8]         = "enc_node_full_u8",
    [PROF_ENC_FLAT_U8]              = "enc_flat_u8",
    [PROF_DECODE_NODE]         = "decode_node_calls",
    [PROF_DECODE_ENTRY]        = "decode_entry",
    [PROF_ENC_NODE_VISIT]      = "enc_node_calls",
    [PROF_ENC_ENTRY]           = "enc_entry",
    [PROF_FILE_HISTOGRAM]      = "file_histogram",
    [PROF_FILE_BUILD_TABLE_REAL] = "file_build_table_real",
    [PROF_FILE_BUILD_TABLE_SYN]  = "file_build_table_syn",
    [PROF_FILE_BODY_CSUM]      = "file_body_csum",
    [PROF_FILE_HDR]            = "file_hdr",
    [PROF_FILE_PAD]            = "file_pad",
    [PROF_FILE_BLOCK_ENCODE]   = "file_block_encode",
    [PROF_FILE_BLOCK_DECODE]   = "file_block_decode",
    [PROF_FILE_BLOCK_PROLOGUE] = "file_block_prologue",
    [PROF_FSE_ENC]             = "fse_enc",
    [PROF_FSE_DEC]             = "fse_dec",
    [PROF_FSE_HIT_COUNT]       = "fse_hit_count",
    [PROF_FSE_RAW_COUNT]       = "fse_raw_count",
    [PROF_FSE_FALLBACK_COUNT]  = "fse_fallback_count",
};

const char *pivco_prof_name(pivco_prof_id_t id) {
    return (id < PROF_COUNT && prof_names[id]) ? prof_names[id] : "?";
}

void pivco_prof_reset(void) {
    memset(pivco_prof_counters, 0, sizeof(pivco_prof_counters));
}

#ifdef PIVCO_PROF
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

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

    printf("\n  %-32s %12s %14s %10s %10s %10s %7s\n",
           "primitive", "calls", "elements", "elem/call",
           "ns/call", "ns/elem", "% wall");
    printf("  ---------------------------------------------------------"
           "-----------------------------------------\n");

    /* Sum of timed (non-count-only) primitive ns to compute "unaccounted". */
    double total_timed_ns = 0.0;
    for (int i = 0; i < PROF_COUNT; i++) {
        pivco_prof_counter_t *c = &pivco_prof_counters[i];
        if (c->calls == 0 || c->ticks == 0 || tick_freq_hz <= 0) continue;
        total_timed_ns += (double)c->ticks * 1e9 / tick_freq_hz;
    }
    double wall_ns = wall_seconds * 1e9;

    for (int i = 0; i < PROF_COUNT; i++) {
        pivco_prof_counter_t *c = &pivco_prof_counters[i];
        if (c->calls == 0) continue;

        double ns_per_call = 0, ns_per_elem = 0, pct_wall = 0;
        if (c->ticks > 0 && tick_freq_hz > 0) {
            double ns = (double)c->ticks * 1e9 / tick_freq_hz;
            ns_per_call = ns / (double)c->calls;
            ns_per_elem = c->elements > 0 ? ns / (double)c->elements : 0;
            pct_wall    = wall_ns > 0 ? 100.0 * ns / wall_ns : 0;
        }

        double elem_per_call = (double)c->elements / (double)c->calls;
        if (c->ticks > 0) {
            printf("  %-32s %12llu %14llu %10.1f %10.1f %10.2f %6.2f%%\n",
                   pivco_prof_name((pivco_prof_id_t)i),
                   (unsigned long long)c->calls,
                   (unsigned long long)c->elements,
                   elem_per_call,
                   ns_per_call, ns_per_elem, pct_wall);
        } else {
            printf("  %-32s %12llu %14llu %10.1f %10s %10s %7s\n",
                   pivco_prof_name((pivco_prof_id_t)i),
                   (unsigned long long)c->calls,
                   (unsigned long long)c->elements,
                   elem_per_call,
                   "(count)", "(count)", "—");
        }
    }

    /* Unaccounted line: wall - sum of timed primitives.
     * Includes recursion / dispatch / cache effects / anything not
     * inside an explicit PROF_TIC/TOC region. */
    if (tick_freq_hz > 0 && wall_ns > 0) {
        double unaccounted_ns = wall_ns - total_timed_ns;
        double pct = 100.0 * unaccounted_ns / wall_ns;
        printf("  %-32s %12s %14s %10s %10.1f %10s %6.2f%%\n",
               "(unaccounted)", "", "", "",
               n_blocks > 0 ? unaccounted_ns / (double)n_blocks : 0.0,
               "—", pct);
        printf("  %-32s %12s %14s %10s %10.1f %10s %6.2f%%\n",
               "TOTAL (wall)", "", "", "",
               n_blocks > 0 ? wall_ns / (double)n_blocks : 0.0,
               "—", 100.0);
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
