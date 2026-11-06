/* pivco_cpu.h — which CPU we run on: vendor and microarchitecture.
 *
 * x86 reads cpuid (vendor, family, model) on any OS; Linux arm64
 * reads the CPU implementer and part from /proc/cpuinfo; Apple Silicon
 * macOS reads the brand string.  Unknown parts report UNKNOWN plus the
 * raw ids, so a new CPU shows up as unknown rather than as a wrong
 * name.
 *
 * Internal header.  Not part of the public API. */

#ifndef PIVCO_CPU_H
#define PIVCO_CPU_H

typedef enum {
    PIVCO_VENDOR_UNKNOWN,
    PIVCO_VENDOR_INTEL,
    PIVCO_VENDOR_AMD,
    PIVCO_VENDOR_ARM,
    PIVCO_VENDOR_APPLE,
    PIVCO_VENDOR_COUNT
} pivco_vendor_t;

typedef enum {
    PIVCO_UARCH_UNKNOWN,
    /* Intel */
    PIVCO_UARCH_IVYBRIDGE,
    PIVCO_UARCH_HASWELL,
    PIVCO_UARCH_BROADWELL,
    PIVCO_UARCH_SKYLAKE,
    PIVCO_UARCH_SKYLAKE_SP,
    PIVCO_UARCH_CASCADELAKE,
    PIVCO_UARCH_COOPERLAKE,
    PIVCO_UARCH_ICELAKE,
    PIVCO_UARCH_SAPPHIRERAPIDS,
    PIVCO_UARCH_EMERALDRAPIDS,
    PIVCO_UARCH_GRANITERAPIDS,
    PIVCO_UARCH_SIERRAFOREST,
    /* AMD */
    PIVCO_UARCH_ZEN1,
    PIVCO_UARCH_ZEN2,
    PIVCO_UARCH_ZEN3,
    PIVCO_UARCH_ZEN4,
    PIVCO_UARCH_ZEN5,
    PIVCO_UARCH_ZEN6,
    /* Arm Ltd. cores */
    PIVCO_UARCH_CORTEX_A72,
    PIVCO_UARCH_NEOVERSE_N1,
    PIVCO_UARCH_NEOVERSE_N2,
    PIVCO_UARCH_NEOVERSE_N3,
    PIVCO_UARCH_NEOVERSE_V1,
    PIVCO_UARCH_NEOVERSE_V2,
    PIVCO_UARCH_NEOVERSE_V3,
    /* Apple, by chip generation (Pro / Max / Ultra included) */
    PIVCO_UARCH_APPLE_M1,
    PIVCO_UARCH_APPLE_M2,
    PIVCO_UARCH_APPLE_M3,
    PIVCO_UARCH_APPLE_M4,
    PIVCO_UARCH_APPLE_M5,
    PIVCO_UARCH_APPLE_M6,
    PIVCO_UARCH_COUNT
} pivco_uarch_t;

typedef struct {
    pivco_vendor_t vendor;
    pivco_uarch_t uarch;
    unsigned family;      /* x86: display family; arm: implementer */
    unsigned model;       /* x86: display model; arm: part number;
                             Apple Silicon: hw.cpufamily */
    char brand[64];       /* the CPU's own name string, when it has one */
} pivco_cpu_t;

/* Detected once, on first call. */
const pivco_cpu_t *pivco_cpu(void);

/* Lowercase names for printing: "amd", "zen5", "neoverse-v2", ... */
const char *pivco_vendor_name(pivco_vendor_t v);
const char *pivco_uarch_name(pivco_uarch_t u);

#endif /* PIVCO_CPU_H */
