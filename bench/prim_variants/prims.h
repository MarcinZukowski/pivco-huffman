/* bench/prim_variants/prims.h — primitive-variant registry ("graveyard").
 *
 * A home for primitive kernels we tried but don't ship, so they aren't lost to
 * git history and can be re-evaluated on new hardware/compilers.  What earns an
 * entry (see README.md): every external contribution, every kernel that ever
 * shipped, and a few notable parked experiments.  Consumed ONLY by
 * bench/bench_prim.c via `--variants`; zero dependency on src/ — nothing here
 * is built into the codec.
 *
 * Naming convention:
 *   PV_*    macros, enum values, true constants  (PV_VARIANT, PV_ISA_NEON,
 *           PV_WLO ...)
 *   pv_*    this module's types + plumbing  (pv_isa_t, pv_register_*, the
 *           internal lookup tables)
 *   prim_*  the benchmarked kernels and their ctx_t adapters
 *
 * Contract: a variant is  static void prim_fn(const ctx_t *c);  with the SAME
 * semantics as the logical primitive — bench_prim verifies it byte-for-byte
 * against the scalar reference before timing.  ISA-gated at the kernel via
 * `#if defined(USE_NEON_KERNELS)` etc.; all variants are static (TU-local).
 *
 * Include order in bench_prim.c: this header before the prim_t/stage_t
 * definitions (it only adds the enums + the PV_VARIANT macro); the per-family
 * prims-*.h after reg()/ctx_t/the production kernels.
 */
#ifndef PIVCO_PRIM_VARIANTS_H
#define PIVCO_PRIM_VARIANTS_H

typedef enum { PV_ISA_NONE, PV_ISA_SCALAR, PV_ISA_NEON,
               PV_ISA_SSE4, PV_ISA_AVX2, PV_ISA_AVX512 } pv_isa_t;

static inline const char *pv_isa_name(pv_isa_t i) {
    switch (i) {
    case PV_ISA_SCALAR: return "scalar";
    case PV_ISA_NEON:   return "neon";
    case PV_ISA_SSE4:   return "sse4";
    case PV_ISA_AVX2:   return "avx2";
    case PV_ISA_AVX512: return "avx512";
    default:            return "-";
    }
}

/* Append a variant to bench_prim's PRIMS[] table (same translation unit;
 * prim_t / PRIMS / NPRIMS / stage_t / ctx_t are provided by bench_prim.c).
 *   STAGE   logical primitive, e.g. ST_PART / ST_MERGE_VEC_VEC
 *   NAME    variant name string (shown in --list, used by --variants=)
 *   ISA     pv_isa_t (display/doc; runtime gating is the #if at the kernel)
 *   ORIGIN  commit / PR / author provenance      NOTE  one line: why parked
 *   IP      inplace flag (1 if it mutates codes_la in place)
 *   FN      static void fn(const ctx_t *)
 */
#define PV_VARIANT(STAGE, NAME, ISA, ORIGIN, NOTE, IP, FN)                      \
    (PRIMS[NPRIMS++] = (prim_t){ .variant = (NAME), .stage = (STAGE), .D = 0,   \
        .inplace = (IP), .run = (FN), .isa = (ISA),                            \
        .origin = (ORIGIN), .note = (NOTE) })

/* Per-D variant: same as PV_VARIANT but sets .D = DVAL.  For the per-depth
 * flat stages (ST_UNPACK / ST_MERGE_FLAT), whose verify keys on prim->D. */
#define PV_VARIANT_D(STAGE, NAME, DVAL, ISA, ORIGIN, NOTE, IP, FN)               \
    (PRIMS[NPRIMS++] = (prim_t){ .variant = (NAME), .stage = (STAGE), .D = (DVAL),\
        .inplace = (IP), .run = (FN), .isa = (ISA),                            \
        .origin = (ORIGIN), .note = (NOTE) })

#endif /* PIVCO_PRIM_VARIANTS_H */
