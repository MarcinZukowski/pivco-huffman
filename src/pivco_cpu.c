/* pivco_cpu.c — CPU vendor and microarchitecture detection (see
 * pivco_cpu.h). */

#include "pivco_cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

static pivco_cpu_t g_cpu;
static int g_cpu_done;

static const char *const vendor_names[PIVCO_VENDOR_COUNT] = {
    [PIVCO_VENDOR_UNKNOWN] = "unknown",
    [PIVCO_VENDOR_INTEL]   = "intel",
    [PIVCO_VENDOR_AMD]     = "amd",
    [PIVCO_VENDOR_ARM]     = "arm",
    [PIVCO_VENDOR_APPLE]   = "apple",
};

static const char *const uarch_names[PIVCO_UARCH_COUNT] = {
    [PIVCO_UARCH_UNKNOWN]        = "unknown",
    [PIVCO_UARCH_IVYBRIDGE]      = "ivybridge",
    [PIVCO_UARCH_HASWELL]        = "haswell",
    [PIVCO_UARCH_BROADWELL]      = "broadwell",
    [PIVCO_UARCH_SKYLAKE]        = "skylake",
    [PIVCO_UARCH_SKYLAKE_SP]     = "skylake-sp",
    [PIVCO_UARCH_CASCADELAKE]    = "cascadelake",
    [PIVCO_UARCH_COOPERLAKE]     = "cooperlake",
    [PIVCO_UARCH_ICELAKE]        = "icelake",
    [PIVCO_UARCH_SAPPHIRERAPIDS] = "sapphirerapids",
    [PIVCO_UARCH_EMERALDRAPIDS]  = "emeraldrapids",
    [PIVCO_UARCH_GRANITERAPIDS]  = "graniterapids",
    [PIVCO_UARCH_SIERRAFOREST]   = "sierraforest",
    [PIVCO_UARCH_ZEN1]           = "zen1",
    [PIVCO_UARCH_ZEN2]           = "zen2",
    [PIVCO_UARCH_ZEN3]           = "zen3",
    [PIVCO_UARCH_ZEN4]           = "zen4",
    [PIVCO_UARCH_ZEN5]           = "zen5",
    [PIVCO_UARCH_ZEN6]           = "zen6",
    [PIVCO_UARCH_CORTEX_A72]     = "cortex-a72",
    [PIVCO_UARCH_NEOVERSE_N1]    = "neoverse-n1",
    [PIVCO_UARCH_NEOVERSE_N2]    = "neoverse-n2",
    [PIVCO_UARCH_NEOVERSE_N3]    = "neoverse-n3",
    [PIVCO_UARCH_NEOVERSE_V1]    = "neoverse-v1",
    [PIVCO_UARCH_NEOVERSE_V2]    = "neoverse-v2",
    [PIVCO_UARCH_NEOVERSE_V3]    = "neoverse-v3",
    [PIVCO_UARCH_APPLE_M1]       = "apple-m1",
    [PIVCO_UARCH_APPLE_M2]       = "apple-m2",
    [PIVCO_UARCH_APPLE_M3]       = "apple-m3",
    [PIVCO_UARCH_APPLE_M4]       = "apple-m4",
    [PIVCO_UARCH_APPLE_M5]       = "apple-m5",
    [PIVCO_UARCH_APPLE_M6]       = "apple-m6",
};

const char *pivco_vendor_name(pivco_vendor_t v)
{
    return (unsigned)v < PIVCO_VENDOR_COUNT ? vendor_names[v] : "unknown";
}

const char *pivco_uarch_name(pivco_uarch_t u)
{
    return (unsigned)u < PIVCO_UARCH_COUNT ? uarch_names[u] : "unknown";
}

#if defined(__x86_64__) || defined(__i386__)
/* m in [lo, hi]. */
static int in_range(unsigned m, unsigned lo, unsigned hi)
{
    return m >= lo && m <= hi;
}

/* AMD by display family and model, the ranges the Linux kernel uses for
 * its X86_FEATURE_ZEN* bits:
 * https://github.com/torvalds/linux/blob/master/arch/x86/kernel/cpu/amd.c */
static pivco_uarch_t amd_uarch(unsigned fam, unsigned m)
{
    if (fam == 0x17) {
        if (in_range(m, 0x00, 0x2f) || in_range(m, 0x50, 0x5f))
            return PIVCO_UARCH_ZEN1;
        if (in_range(m, 0x30, 0x4f) || in_range(m, 0x60, 0x7f)
            || in_range(m, 0x90, 0x91) || in_range(m, 0xa0, 0xaf))
            return PIVCO_UARCH_ZEN2;
    } else if (fam == 0x19) {
        if (in_range(m, 0x00, 0x0f) || in_range(m, 0x20, 0x5f))
            return PIVCO_UARCH_ZEN3;
        if (in_range(m, 0x10, 0x1f) || in_range(m, 0x60, 0xaf))
            return PIVCO_UARCH_ZEN4;
    } else if (fam == 0x1a) {
        if (in_range(m, 0x00, 0x2f) || in_range(m, 0x40, 0x4f)
            || in_range(m, 0x60, 0x7f) || in_range(m, 0xd0, 0xd7))
            return PIVCO_UARCH_ZEN5;
        if (in_range(m, 0x50, 0x5f) || in_range(m, 0x80, 0xaf)
            || in_range(m, 0xc0, 0xcf) || in_range(m, 0xd8, 0xef))
            return PIVCO_UARCH_ZEN6;
    }
    return PIVCO_UARCH_UNKNOWN;
}

/* Intel family 6 by display model (server parts and their client
 * siblings that matter here), as named in the Linux kernel:
 * https://github.com/torvalds/linux/blob/master/arch/x86/include/asm/intel-family.h */
static pivco_uarch_t intel_uarch(unsigned fam, unsigned model)
{
    if (fam != 6)
        return PIVCO_UARCH_UNKNOWN;
    switch (model) {
    case 0x3a:
    case 0x3e:
        return PIVCO_UARCH_IVYBRIDGE;
    case 0x3c:
    case 0x3f:
    case 0x45:
    case 0x46:
        return PIVCO_UARCH_HASWELL;
    case 0x3d:
    case 0x47:
    case 0x4f:
    case 0x56:
        return PIVCO_UARCH_BROADWELL;
    case 0x4e:
    case 0x5e:
    case 0x8e:
    case 0x9e:
        return PIVCO_UARCH_SKYLAKE;
    case 0x55:
        return PIVCO_UARCH_SKYLAKE_SP;  /* see the stepping check in detect_x86 */
    case 0x6a:
    case 0x6c:
    case 0x7d:
    case 0x7e:
        return PIVCO_UARCH_ICELAKE;
    case 0x8f:
        return PIVCO_UARCH_SAPPHIRERAPIDS;
    case 0xcf:
        return PIVCO_UARCH_EMERALDRAPIDS;
    case 0xad:
    case 0xae:
        return PIVCO_UARCH_GRANITERAPIDS;
    case 0xaf:
        return PIVCO_UARCH_SIERRAFOREST;
    default:
        return PIVCO_UARCH_UNKNOWN;
    }
}

static void detect_x86(pivco_cpu_t *c)
{
    unsigned a, b, cx, d;
    char vendor[13] = {0};
    if (!__get_cpuid(0, &a, &b, &cx, &d))
        return;
    memcpy(vendor, &b, 4);
    memcpy(vendor + 4, &d, 4);
    memcpy(vendor + 8, &cx, 4);
    if (!__get_cpuid(1, &a, &b, &cx, &d))
        return;
    unsigned fam = (a >> 8) & 0xf;
    unsigned model = (a >> 4) & 0xf;
    if (fam == 0xf)
        fam += (a >> 20) & 0xff;
    if (fam == 0x6 || fam >= 0xf)
        model |= ((a >> 16) & 0xf) << 4;
    c->family = fam;
    c->model = model;
    if (!strcmp(vendor, "AuthenticAMD")) {
        c->vendor = PIVCO_VENDOR_AMD;
        c->uarch = amd_uarch(fam, model);
    } else if (!strcmp(vendor, "GenuineIntel")) {
        c->vendor = PIVCO_VENDOR_INTEL;
        c->uarch = intel_uarch(fam, model);
        /* Cascade Lake (stepping 7) and Cooper Lake (stepping 11) share
         * Skylake-SP's model (intel-family.h above). */
        if (model == 0x55 && (a & 0xf) == 7)
            c->uarch = PIVCO_UARCH_CASCADELAKE;
        if (model == 0x55 && (a & 0xf) == 11)
            c->uarch = PIVCO_UARCH_COOPERLAKE;
    }
    unsigned max_ext = 0;
    __get_cpuid(0x80000000, &max_ext, &b, &cx, &d);
    if (max_ext >= 0x80000004) {
        unsigned *w = (unsigned *)c->brand;
        for (unsigned i = 0; i < 3; i++)
            __get_cpuid(0x80000002 + i, &w[4 * i], &w[4 * i + 1],
                        &w[4 * i + 2], &w[4 * i + 3]);
        c->brand[48] = 0;
        size_t lead = strspn(c->brand, " ");
        memmove(c->brand, c->brand + lead, strlen(c->brand + lead) + 1);
    }
}
#endif

#if defined(__aarch64__) && defined(__linux__)
/* Arm Ltd. parts by MIDR part number, as in the Linux kernel:
 * https://github.com/torvalds/linux/blob/master/arch/arm64/include/asm/cputype.h */
static pivco_uarch_t arm_uarch(unsigned part)
{
    switch (part) {
    case 0xd08:
        return PIVCO_UARCH_CORTEX_A72;  /* Graviton 1 */
    case 0xd0c:
        return PIVCO_UARCH_NEOVERSE_N1;  /* Graviton 2 */
    case 0xd40:
        return PIVCO_UARCH_NEOVERSE_V1;  /* Graviton 3 */
    case 0xd49:
        return PIVCO_UARCH_NEOVERSE_N2;
    case 0xd4f:
        return PIVCO_UARCH_NEOVERSE_V2;  /* Graviton 4, Grace */
    case 0xd84:
        return PIVCO_UARCH_NEOVERSE_V3;
    case 0xd8e:
        return PIVCO_UARCH_NEOVERSE_N3;
    default:
        return PIVCO_UARCH_UNKNOWN;
    }
}

static void detect_linux_arm(pivco_cpu_t *c)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f)
        return;
    char line[256];
    unsigned impl = 0;
    unsigned part = 0;
    int have_impl = 0;
    int have_part = 0;
    while (fgets(line, sizeof line, f) && !(have_impl && have_part)) {
        const char *v = strchr(line, ':');
        if (!v)
            continue;
        if (!strncmp(line, "CPU implementer", strlen("CPU implementer"))) {
            impl = (unsigned)strtoul(v + 1, NULL, 0);
            have_impl = 1;
        } else if (!strncmp(line, "CPU part", strlen("CPU part"))) {
            part = (unsigned)strtoul(v + 1, NULL, 0);
            have_part = 1;
        }
    }
    fclose(f);
    c->family = impl;
    c->model = part;
    if (impl == 0x41) {
        c->vendor = PIVCO_VENDOR_ARM;
        c->uarch = arm_uarch(part);
    } else if (impl == 0x61) {
        c->vendor = PIVCO_VENDOR_APPLE;
    }
}
#endif

#if defined(__APPLE__) && defined(__aarch64__)
/* "Apple M4 Pro" -> PIVCO_UARCH_APPLE_M4.  hw.cpufamily is kept as the
 * raw id; Apple's values are the CPUFAMILY_ARM_* constants in
 * https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/machine.h */
static void detect_apple(pivco_cpu_t *c)
{
    size_t len = sizeof c->brand;
    if (sysctlbyname("machdep.cpu.brand_string", c->brand, &len,
                     NULL, 0) != 0)
        return;
    c->brand[sizeof c->brand - 1] = 0;
    if (strncmp(c->brand, "Apple M", strlen("Apple M")))
        return;
    c->vendor = PIVCO_VENDOR_APPLE;
    unsigned fam = 0;
    len = sizeof fam;
    if (sysctlbyname("hw.cpufamily", &fam, &len, NULL, 0) == 0)
        c->model = fam;
    /* The generation is the number after "Apple M": 4 in "Apple M4 Pro".
     * The variant ("Pro", "Max") stays in brand; hw.cpufamily (model)
     * tells those dies apart.  A generation past the table is UNKNOWN. */
    int gen = atoi(c->brand + strlen("Apple M"));
    if (gen >= 1 && gen <= PIVCO_UARCH_APPLE_M6 - PIVCO_UARCH_APPLE_M1 + 1)
        c->uarch = (pivco_uarch_t)(PIVCO_UARCH_APPLE_M1 + gen - 1);
}
#endif

const pivco_cpu_t *pivco_cpu(void)
{
    if (g_cpu_done)
        return &g_cpu;
    g_cpu.vendor = PIVCO_VENDOR_UNKNOWN;
    g_cpu.uarch = PIVCO_UARCH_UNKNOWN;
#if defined(__x86_64__) || defined(__i386__)
    detect_x86(&g_cpu);
#elif defined(__APPLE__) && defined(__aarch64__)
    detect_apple(&g_cpu);
#elif defined(__aarch64__) && defined(__linux__)
    detect_linux_arm(&g_cpu);
#endif
    g_cpu_done = 1;
    return &g_cpu;
}
