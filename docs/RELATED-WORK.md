# Related Work

> **Last content review:** _NEVER_

Survey of the literature and open-source landscape for fast Huffman
decoders, with PIVCO-Huffman's positioning.  Full prior-art notes on
the wavelet-tree connection are in
[`WAVELET_TREES.md`](WAVELET_TREES.md).

## Prior art on Huffman decoding

- **huff0/zstd is the CPU SotA.** Recent zstd PRs (#3826, #3827)
  focus on compiler-level fixes (manual unrolling, bit masking for
  optimizer hints), not algorithmic changes.  The 4-stream table
  lookup architecture hasn't changed fundamentally.

- **Dougall Johnson's sync-point parallel decode** (2022): split the
  bitstream at arbitrary points, find synchronization by running
  parallel decoders at n consecutive offsets (n = max code length).
  ~25% speedup on M1 for DEFLATE.  Orthogonal to PIVCO — parallelizes
  the same serial bitstream rather than using a different format.

- **Fabian Giesen's alias Huffman** (2014): uses the alias method
  with rANS for a unified decode table sized by symbol count.
  Branch-free, but the multiply-heavy decode step is slower than
  table lookup on ARM.  Tested: 200–440 M/s on M4, significantly
  slower than huff0.

- **GPU massively parallel** (Weissenberger et al., 2018): uses
  Huffman self-synchronization for thousands of GPU threads.
  10×+ over CPU.  Not relevant for single-core CPU comparison.

- **512-bit SIMD Huffman encoding** (IEEE TCE, 2023): 2.66× speedup
  for *encoding* on NEON.  Decoding remains the harder problem.

- **No known CPU decoder beats huff0 on general Huffman decode.**
  PIVCO's headline ratios (see [BENCHMARKS.md](BENCHMARKS.md))
  appear to be a novel result for a single-core CPU Huffman decoder.

## Relationship to wavelet trees

The bitmap-per-Huffman-internal-node wire format PIVCO uses is
structurally identical to a **Huffman-shaped wavelet tree**:
Grossi–Gupta–Vitter (SODA 2003) and Mäkinen–Navarro (the Huffman
shape).  The SIMD partition primitives (TBL/pshufb/vpcompress) are
also published — Kaneta (SPIRE 2018) and Dinklage–Fischer–Kurpicz–
Tarnowski (DCC 2023, AVX-512) — for wavelet-tree *construction*.
Both prior-art families are strictly top-down and frame the
representation as a rank/select **index** for substring queries
(FM-index / r-index), not as a stream codec.

PIVCO's contributions, after this survey, are:

1. Framing the representation as a bulk stream codec.
2. SIMD bulk *decode* — in particular the bottom-up `tree_merge`
   direction not seen in any WT paper.
3. The flat-subtree fast path (maximal `D ≥ 2` flat subtrees → one
   `N·D`-bit packed region + `code_to_sym` lookup).
4. Per-node FSE on the bitmap.
5. Empirical positioning against huf0 / zstd / brotli / FSE / Oodle
   on real distributions across Apple M4 / Graviton 4 / Xeon Granite
   Rapids / Zen 3 / Zen 5.

Dinklage et al. report wavelet-tree *construction* throughput at
~100 MB/s of input on i9-11900KF AVX-512 for the Huffman-shaped
variant (their headline "1.4 Gbit/s tops" applies to the binary
fixed-⌈lg σ⌉-code variant only, 2–3× faster than the Huffman-shaped
shape because they don't have to filter just-ended codes per level).
Decode is not measured anywhere in either paper.  No published
bulk-decode bytes/sec exists for this representation — that regime
appears to be open.

Baruch, Klein & Shapira (DAM 2020) is the closest prior work on the
decode side: a strictly top-down, scalar, per-node `rnk(v)` cache
that exploits "rank on consecutive positions differs by ≤ 1" to
avoid recomputing rank during a range query.  Reports ~50% full /
~30% partial decode speedup vs the SDSL succinct-DS library; no
SIMD, no comparison to huf0 / FSE / zstd.  Same "shared upper tree"
insight pivco uses, but as scalar rank caching across t independent
root-to-leaf walks rather than bulk SIMD over the whole bitmap.
