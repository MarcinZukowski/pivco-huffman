/* C interface to Oodle's newlz_arrays_huff for the FSE x*y
 * microbench.  Compiled only when ext/oodle/ exists as a symlink
 * to a built OodleUE clone (CMake auto-detects).
 *
 * No Oodle source is vendored in this repo — see oodle.md and
 * the 2026-05-15 EULA review notes.  The user must:
 *   1. Clone OodleUE somewhere (e.g. ~/src/OodleUE)
 *   2. Build it: cmake -S OodleUE/build -B OodleUE/build-out -DCMAKE_BUILD_TYPE=Release && cmake --build OodleUE/build-out
 *   3. Symlink: ln -s ~/src/OodleUE ext/oodle
 * Then cmake-reconfigure pivco-huffman and Oodle columns appear.
 */

#ifndef PIVCO_BENCH_OODLE_WRAPPER_H
#define PIVCO_BENCH_OODLE_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Encode `src` (n bytes) into `dst`.  *huff_type_out is set to 2
 * (HUFF, 3-stream) or 4 (HUFF6, 6-stream) — the tuner picks based
 * on size and J-cost.  Returns compressed bytes on success;
 * < 0 = failure without touching dst; > n = failure with dst clobbered. */
int oodle_huff_encode(const unsigned char *src, size_t n,
                       unsigned char *dst, size_t dst_cap,
                       int *huff_type_out);

/* Decode `comp` (comp_len bytes) into `dst`.  huff_type must be the
 * value returned by the encoder.  Returns bytes consumed from
 * `comp` on success; < 0 on failure. */
int oodle_huff_decode(const unsigned char *comp, size_t comp_len,
                       unsigned char *dst, size_t dst_cap,
                       int huff_type);

/* NEWLZ_ARRAY_TYPE values exposed for the bench so it can label
 * the variant (3-stream vs 6-stream) chosen by the tuner. */
#define OODLE_HUFF_TYPE_HUFF3  2
#define OODLE_HUFF_TYPE_HUFF6  4

/* Oodle tANS (newlz_arrays_tans): 2 physical bitstreams, 5-way
 * interleave.  Encoder is the J-cost tuner; it may decline (return
 * < 0) on data that doesn't benefit. */

/* Encode `src` (n bytes) into `dst`.  Returns compressed bytes on
 * success; < 0 = decline / failure without touching dst; > n =
 * failure with dst clobbered. */
int oodle_tans_encode(const unsigned char *src, size_t n,
                      unsigned char *dst, size_t dst_cap);

/* Decode `comp` (comp_len bytes) into `dst`.  Returns decoded byte
 * count on success; < 0 on failure. */
int oodle_tans_decode(const unsigned char *comp, size_t comp_len,
                      unsigned char *dst, size_t dst_cap);

#ifdef __cplusplus
}
#endif

#endif
