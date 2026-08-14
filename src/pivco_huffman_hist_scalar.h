/* pivco_hist_scalar.h — the shared scalar histogram core.
 *
 * Included by every backend primitives header; backends without their
 * own implementation alias prim_histogram_chunk to this (currently
 * scalar, NEON and SSE4.1 — an 8-cursor u64-load variant measured on
 * par on M4/Graviton4/Zen3, so one portable core suffices; AVX-512 has
 * its own, see primitives_avx512.h).  Not a backend header itself: no
 * prim_* aliases here, specialized name only.
 *
 * Contract: see prim_histogram_chunk in pivco_primitives.h.
 */
#ifndef PIVCO_HUFFMAN_HIST_SCALAR_H
#define PIVCO_HUFFMAN_HIST_SCALAR_H

#include <stdint.h>
#include <stddef.h>

#define PIVCO_PRIM_HIST_CHUNK   ((size_t)1 << 30)
#define PIVCO_PRIM_HIST_SCRATCH (4 * 16 * 1024 + 64)

/* 4 interleaved u32 sub-histograms break the same-bucket
 * read-modify-write dependence that caps the naive loop at L1
 * store-to-load latency; the 4 KB working set stays L1-resident. */
static inline void histogram_chunk_scalar(const uint8_t *in, size_t n,
                                          uint32_t hist[256],
                                          uint8_t *scratch)
{
    (void)scratch;
    uint32_t h0[256] = {0}, h1[256] = {0}, h2[256] = {0}, h3[256] = {0};
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        h0[in[i + 0]]++;
        h1[in[i + 1]]++;
        h2[in[i + 2]]++;
        h3[in[i + 3]]++;
    }
    for (; i < n; i++) h0[in[i]]++;
    for (int s = 0; s < 256; s++)
        hist[s] += h0[s] + h1[s] + h2[s] + h3[s];
}

#endif /* PIVCO_HUFFMAN_HIST_SCALAR_H */
