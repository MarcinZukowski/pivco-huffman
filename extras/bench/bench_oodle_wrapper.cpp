/* C wrapper around Oodle's newlz_arrays_huff so the FSE x*y
 * microbench can time Oodle's shipping Huffman path next to FSE
 * and huff0.  Compiles only when PIVCO_HAS_OODLE is defined (i.e.
 * when ext/oodle is a symlink to a built OodleUE clone).
 *
 * Oodle source lives in ext/oodle/Engine/Source/.../Sdks/2.9.16/src/
 * oodle2/{base,core}/.  The decoder symbol is `oo2::newlz_get_
 * array_huff(comp, comp_len, dst, dst_cap, is_huff6)`; the
 * encoder is the tuner `oo2::newLZ_put_array_huff(...)` which
 * picks NEWLZ_ARRAY_TYPE_HUFF (3-stream) or NEWLZ_ARRAY_TYPE_HUFF6
 * (6-stream) based on size and J-cost.
 *
 * EULA note (per oodle.md and the UE EULA reviewed 2026-05-15):
 * Oodle source is under the Unreal Engine EULA + RAD's separate
 * Oodle license.  Private use including benchmarking is fine
 * under "private use however you want" in the UE EULA.  We do
 * NOT redistribute any Oodle source — `ext/oodle` is a user-
 * provided symlink, not a vendored copy.
 */

#include <cstdint>
#include <cstring>

#include "newlz_arrays_huff.h"
#include "newlz_arrays_tans.h"
#include "newlz_arrays.h"
#include "newlz_speedfit.h"
#include "rrarenaallocator.h"
#include "histogram.h"

extern "C" {

/* Encode `src` (n bytes) into `dst` using Oodle's Huffman tuner.
 * `*huff_type_out` is set to 2 (HUFF, 3-stream) or 4 (HUFF6, 6-
 * stream) based on what the tuner picked.  Returns compressed
 * bytes on success, < 0 on failure-without-touching-dst, or >n
 * on failure-with-dst-clobbered. */
int oodle_huff_encode(const unsigned char *src, size_t n,
                       unsigned char *dst, size_t dst_cap,
                       int *huff_type_out)
{
    using namespace oo2;

    /* Histogram (1 KB stack). */
    uint32_t histo[256];
    CountHistoArrayU8(src, (SINTa)n, histo, 256);

    /* Speedfit + scratch arena (128 KB stack should cover our cells). */
    const OodleSpeedFit *speedfit = speedfit_get_default();
    unsigned char arena_buf[131072];
    rrArenaAllocator arena(arena_buf, sizeof(arena_buf), /*allowFallback=*/false);

    /* lambda biases the tuner's J-cost between size (smaller =
     * win-on-size) and decode time (bigger = win-on-speed).  At
     * lambda=0 the tuner always picks huff3 because its table
     * header is smaller than huff6's; with lambda=1000 the
     * faster huff6 decode wins the J-cost comparison and fires
     * for cells where huff6 actually applies (from_len >= 256). */
    /* NEWLZ_ARRAY_FLAG_ALLOW_HUFF6 (bit 0) lets the tuner consider
     * huff6.  Without this flag huff6 is permanently disabled and
     * the tuner always picks huff3. */
    float    pJ           = 1e30f;
    uint32_t huff_type    = 0;
    int      level        = 8;             /* high compression level */
    float    lambda       = 1.0f;
    float    deadline_t   = 1e30f;
    uint32_t entropy_flags = 1;  /* NEWLZ_ARRAY_FLAG_ALLOW_HUFF6 */

    SINTa comp_len = newLZ_put_array_huff(
        dst, dst + dst_cap,
        src, (SINTa)n,
        histo, lambda, speedfit, &pJ, deadline_t,
        &huff_type, entropy_flags, &arena, level);

    if (huff_type_out) *huff_type_out = (int)huff_type;
    return (int)comp_len;
}

/* Decode `comp` (comp_len bytes) into `dst` using Oodle's Huffman
 * decoder.  `huff_type` is 2 (HUFF, 3-stream) or 4 (HUFF6, 6-stream).
 * Returns decoded byte count on success, < 0 on failure. */
int oodle_huff_decode(const unsigned char *comp, size_t comp_len,
                       unsigned char *dst, size_t dst_cap, int huff_type)
{
    using namespace oo2;
    bool is_huff6 = (huff_type == NEWLZ_ARRAY_TYPE_HUFF6);
    return (int)newlz_get_array_huff(comp, (SINTa)comp_len,
                                      dst, (SINTa)dst_cap,
                                      is_huff6);
}

/* ---- tANS (newlz_arrays_tans) ---- */

int oodle_tans_encode(const unsigned char *src, size_t n,
                      unsigned char *dst, size_t dst_cap)
{
    using namespace oo2;

    uint32_t histo[256];
    CountHistoArrayU8(src, (SINTa)n, histo, 256);

    const OodleSpeedFit *speedfit = speedfit_get_default();
    unsigned char arena_buf[131072];
    rrArenaAllocator arena(arena_buf, sizeof(arena_buf), /*allowFallback=*/false);

    /* pJ is the J-cost budget. Oodle requires it <= from_len+3 (a real size
     * bound) and internally does (SINTa)(pJ - cost). Passing 1e30 overflows
     * the float->int64 cast: ARM saturates to INT64_MAX so TANS proceeds, but
     * x86 yields INT64_MIN (< 4) so TANS bails with -1. Pass a valid bound,
     * with lambda=0 so TANS is accepted on size alone (force the measurement
     * regardless of Oodle's per-arch speed model -- which only gates accept/
     * reject here; the encoded stream itself does not depend on lambda). */
    float pJ     = (float)(n + 3);
    float lambda = 0.0f;

    SINTa comp_len = newLZ_put_array_tans(
        dst, dst + dst_cap,
        src, (SINTa)n,
        histo, lambda, speedfit, &pJ, &arena);

    return (int)comp_len;
}

int oodle_tans_decode(const unsigned char *comp, size_t comp_len,
                      unsigned char *dst, size_t dst_cap)
{
    using namespace oo2;
    /* tANS decode needs scratch for the decode tables + interleave
     * state.  64 KB is comfortably above the 2^L table footprint. */
    unsigned char scratch[65536];
    return (int)newlz_get_array_tans(comp, (SINTa)comp_len,
                                      dst, (SINTa)dst_cap,
                                      scratch, scratch + sizeof(scratch));
}

}  /* extern "C" */
