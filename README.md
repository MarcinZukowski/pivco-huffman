# PIVCO Huffman: Implementation & Benchmark Results

## Contents

- [TL;DR](#tldr)
- [What is PIVCO-Huffman?](#what-is-pivco-huffman)
  - [Encoded Format](#encoded-format)
- [Implementation](#implementation)
  - [Key Design Decisions](#key-design-decisions)
- [Baselines](#baselines)
  - [Traditional 1-stream (trad_1s)](#traditional-1-stream-trad_1s)
  - [Our 4-stream (trad_4s)](#our-4-stream-trad_4s)
  - [huff0 (cyan4973/FiniteStateEntropy)](#huff0-cyan4973finitestateentropy)
  - [ryg_rans alias (rygorous/ryg_rans)](#ryg_rans-alias-rygorousryg_rans)
- [Prior Art Survey](#prior-art-survey)
- [Test Datasets](#test-datasets)
  - [Synthetic distributions](#synthetic-distributions)
  - [Real-world byte distributions](#real-world-byte-distributions)
- [Benchmark Results](#benchmark-results)
  - [Apple M4 Max (NEON)](#apple-m4-max-neon-128kb-l1d-block-8192)
  - [Intel Xeon 6975P-C (AVX-512 VBMI2)](#intel-xeon-6975p-c-avx-512-vbmi2-48kb-l1d-block-8192)
  - [AWS Graviton 4 (NEON)](#aws-graviton-4-neoverse-v2-neon-64kb-l1d-block-8192)
  - [AMD EPYC 7R13 (Zen 3 SSE4.1)](#amd-epyc-7r13-zen-3-sse41-32kb-l1d-block-4096)
  - [Cross-Platform Summary](#cross-platform-summary-pivco-simd-vs-best-other-decoder)
  - [Compression Ratio](#compression-ratio)
  - [Block Size Sweep](#block-size-sweep)
- [Key Compute Primitives](#key-compute-primitives)
- [Profiling](#profiling)
- [Analysis](#analysis)
  - [Where PIVCO Wins](#where-pivco-wins)
  - [Where PIVCO Still Loses](#where-pivco-still-loses)
  - [The Core Tradeoff](#the-core-tradeoff)
  - [SIMD Width Scaling](#simd-width-scaling)
- [Ideas That Can Make Things Faster](#ideas-that-can-make-things-faster)
  - [Tested and adopted](#tested-and-adopted)
  - [Tested and discarded](#tested-and-discarded)
  - [Worth exploring](#worth-exploring)
- [Building & Running](#building--running)

---

## TL;DR

SIMD Huffman decoder that walks the tree top-down and partitions the
whole block at each internal node, instead of decoding one symbol at
a time.  Plus a **flat-subtree fast path**: every maximal flat subtree
of depth D ≥ 2 in the Huffman tree emits a single N·D-bit packed
region instead of D levels of bitmaps, decoded via direct lookup.

Concrete on Apple M4, `pivco_bu` decode vs `huf0_x2`
(default-recommended `--no-fse` configuration):
- `proba80` heavily skewed: **15.3 GB/s, 5.9× huf0_x2**.
- `proba50` / `proba14`: **9.2 / 5.2 GB/s, 3.6× / 2.1×**.
- `flat_M*` fully flat: **20–24 GB/s, 4.1–4.8×**.
- `uniform` / `bell_*` near-uniform: **4–6 GB/s, 2.8–3.2×** (`huf0`
  fails on uniform; PIVCO doesn't).
- `english` / `prose_pride` / `html_wiki` / `chinese_text` real text:
  **4.3–4.8 GB/s, 2.0–2.5× huf0_x2**.
- `gzip_random` / `image_jpeg` high-entropy: **4.1–4.9 GB/s,
  2.7–3.2×**.

Cross-ISA, the SIMD primitive landscape spreads ratios across an
order of magnitude even at the same algorithm:
- **Xeon AVX-512 VBMI2** (`vpcompressw` partition + `vpexpandb`
  tree_merge): 1.43–13.8× across the 30-distribution grid.
- **Apple M4 NEON** (TBL partition + `vqtbl1` tree_merge): 1.43–10.7×.
- **Graviton 4 NEON** (same NEON ISA, slower `vqtbl{2,4}` at small n):
  1.29–8.59×.
- **Zen 3 SSE/AVX2** (`pshufb` partition, no `vpcompressw`): 0.94×
  on three deep-real-text rows, 1.06–22.5× on everything else.

The Zen-3 0.94× rows are interesting on their own merits (real-text
trees of Dmax 15 stress the per-cycle partition cost; the
`vpcompressw` primitive that Xeon and Zen 4+ enjoy is the obvious
gap-closer).  They're not framed as "things to fix" — ph is a
research vehicle for what SIMD Huffman decoding can do at the
algorithmic edge, not a tool people will compress with.  Losses are
data, wins are observations.

The bench grid intentionally includes 10 real-world byte
distributions (Wikipedia HTML, Project Gutenberg prose, JPEG image,
GitHub JSON, C source, Apache log, E. coli FASTA, OWID CSV, gzip
output, classical Chinese text — sources in
[`extras/datasets/`](extras/datasets/)).  The synthetic `english`
distribution overstates PIVCO's real-prose win by ~25% on M4 and
~33% on Xeon — quote real-world numbers when comparing.

Encoded size within 1–4% of traditional Huffman.  The flat-subtree
format actually *improves* packing slightly — one tail padding per
flat region instead of D per-level byte-alignment paddings.

## What is PIVCO-Huffman?

PIVCO-Huffman applies the PIVoted COding approach to Huffman coding.
Instead of decoding symbols one at a time via table lookup (traditional),
PIVCO processes an entire block of N symbols simultaneously, using
whichever of two complementary strategies fits the shape of each
Huffman subtree best:

- a **SIMD tree-walk partition** for mixed-depth subtrees, which
  splits the block's index set by the bitmap at each internal node
  and recurses;
- a **flat-subtree fast path** for subtrees whose leaves all sit at
  the same relative depth, which replaces a sequence of per-level
  bitmaps with a single packed D-bit code per element and one direct
  `code_to_sym[code]` lookup at the bottom.

Detection and dispatch happen once at `pivco_huffman_build_table`
time — the encoder walks the tree and flags every maximal flat subtree
(local_min_depth == local_max_depth ≥ 2), pre-computes the
`code_to_sym` lookup per flat subtree, and both encoder and decoder
consult the flags to pick the right path at each node.

**Traditional Huffman decode** processes one symbol at a time:
peek bits → table lookup → emit symbol → consume bits → repeat.
The serial dependency chain (can't start the next symbol until you know
how many bits the current one consumed) limits throughput to ~1 symbol
per 3-4 cycles, even with 4-stream ILP tricks (huff0/zstd).

**PIVCO-Huffman decode** processes all K symbols of the subtree at
once, with per-node dispatch.  The production path is bottom-up: each
call writes a contiguous K-byte output buffer for its subtree, then
the parent merges its two child buffers per the bitmap.

```
decode_subtree(node, K, out_buf):
  if leaf:
    memset(out_buf, node->symbol, K)
    return

  if node is a flat-subtree root (D >= 2):
    read K*D packed bits from stream
    for k in [0..K): out_buf[k] = code_to_sym[D-bit code k]
    return                                # no recursion below this

  read K-bit bitmap (or FSE-coded equivalent) from stream
  K_right = read_kr_header()              # 2 bytes if non-trivial node

  if both children are leaves:            # BOTH_LEAVES fast path
    merge_both_const(out_buf, K, bitmap, left->sym, right->sym)
    return

  decode_subtree(node->left,  K - K_right, left_buf)
  decode_subtree(node->right, K_right,     right_buf)
  tree_merge(out_buf, K, bitmap, left_buf, right_buf)
```

At each mixed-depth internal node, the two child output buffers get
gather-merged into the parent output using a TBL-based 8-wide merge
on NEON / SSE4.1 (`expand_tab[bitmap_byte]`-driven `vqtbl1` or
`pshufb`), or `vpexpandb` 64-byte chunks on AVX-512 VBMI2.  At each
leaf, `memset` fills the buffer.  At each flat-subtree root, the
whole subtree is collapsed into a single packed-bit region — the
decoder performs K D-bit extracts + K table lookups + K sequential
byte stores, with no recursion below that node.

**Both-leaves and flat-subtree are the same mechanism at different
depths.**  `merge_both_const` is exactly the D=1 case of the
flat-subtree path (one bit per element, 2-entry inline `syms[]`
lookup).  Flat-subtree at D ≥ 2 generalises it to deeper regions of
the tree using a per-subtree `code_to_sym` table.  Both replace D
levels of partition-and-recurse with a single packed-bit blend.

**Coverage of the two paths** (measured on the standard bench
distributions, see `extras/bench_flat_subtree_stats.c`):
flat-subtree fires on 55–100% of elements for bell_* / proba02 /
zipfian / english, and 100% of elements for the whole-tree flat cases
(uniform / sparse_* / flat_M*).  Tree-walk partition handles the rest
(proba14, stick-tree shapes like proba80/50/geometric).

No accumulation of code bits across symbols, no sequential table-lookup
dependency chain.  The tree position tells you which path a symbol
takes; the inner loop of each path is purely:
load indices → load bits → shuffle/lookup → store.

### Encoded Format

The encoded data is a DFS-ordered bitstream matching the tree walk.
See `src/pivco_huffman_wire.h` for the authoritative format spec.
Per-node record (v0.3, since 2026-05-13):

```
[optional K_right_header : uint16 LE, 2 bytes]   if kr_header_needed()
[FSE marker             : uint8,     1 byte]     always
[bitmap body]                                    marker == 0: raw n-bit
                                                  bitmap, ceil(n/8) bytes
                                                 marker != 0: [fse_len:u16
                                                  LE][fse_payload]
                                                  (FSE-compressed bitmap)
```

The K_right header (2 bytes) is emitted at every internal node whose
right child is non-leaf, letting the bottom-up decoder size each
child's output buffer ahead of time instead of computing it from a
bitmap popcount.  Adds 2 bytes per qualifying node; saves a popcount
pass per node at decode time.

The FSE marker byte gates per-node entropy coding of the partition
bitmap: when the bitmap is heavily skewed (one bit value dominates
≥ 62.5%) and the per-codeword cost ratio passes the commit gate, the
encoder ships an FSE-compressed payload instead of the raw bitmap.
Decoder dispatches generically based on the marker byte.  Wire
overhead when FSE doesn't fire: 1 byte per non-flat internal node
(~0.06% on proba80, ~1.6% on incompressible image data).
**FSE coding is experimental and disabled in the headline bench
numbers below** (`--no-fse`); see Implementation section item 5 for
the rationale.  The marker byte is still emitted unconditionally so
the wire format is stable when the runtime gate flips.

At each flat-subtree root with `n` active symbols and depth `D`, a
single `ceil(n·D/8)`-byte packed region is stored — one `D`-bit
code per element, no per-level framing, no marker byte (the flat
path doesn't FSE-code).

The decoder has the Huffman tree, so it knows which path each node
uses and exactly how many bytes to consume.  No continuation bitmaps
or stream-level metadata are needed — the Huffman tree structure
plus the per-node K_right / FSE markers are sufficient.

Encoded size equals traditional Huffman (sum of code lengths) plus
byte-alignment rounding, which is typically 1-4% overhead — minus
the FSE win on skewed bitmaps (~25% on proba80, ~24% on
calgary_pic).  The flat-subtree format is marginally tighter than
bitmap-per-level on flat-heavy regions (one tail padding for the
whole packed region vs `D` per-level paddings).

## Implementation

Written in C11.  After the unify-framework refactor landed 2026-05-14,
the entire encode + bottom-up decode tree walk + wire-format I/O live
in **one source file**, `src/pivco_huffman_codec.c`, compiled four
times — once per backend — into separate OBJECT libraries.  Each
compile pulls in the matching `src/pivco_huffman_primitives_<backend>.h`
via the router header `pivco_huffman_primitives.h`.  Adding a fifth
backend is a primitives header plus a CMake OBJECT lib entry; the
tree walk and wire format are inherited automatically.  The runtime
dispatcher in `src/pivco_huffman.c::resolve_impl` picks the best
backend per host.

Tree-shape detection (which nodes get the flat-subtree fast path,
which get tree-walk partition, which collapse to BOTH_LEAVES /
HALF_RIGHT / HALF_LEFT / SKIP) happens once at
`pivco_huffman_build_table` time and is shared across backends in
`src/huffman_table.c`.

Backends:

- **Scalar**: portable C bitmap partition and scalar D-bit unpack.
  Specialisations for `D ∈ {2,3,4,5,6,7,8}` in the unpack switch.
  No SIMD — used on RISC-V and as a fallback.

- **NEON** (AArch64): TBL-based 8-wide partition (`vqtbl1q_u8` over a
  combined 8KB shuffle table loaded with a single `ldp q0, q1`,
  12 instructions per 8 indices).  Bottom-up `tree_merge` via
  `vqtbl1q_u8` over `expand_tab[mask]` (V4 strategy: 16-byte loads +
  precomputed `(nr0, m1)` shuf, ~13-15% gain on M4).  Flat-subtree
  SIMD unpack for `D ∈ {2,3,4,5,6}`; D=7/8 scalar.

- **x86 SSE4.1 / AVX2** (x86-64 non-AVX-512): `pshufb`-based 8-wide
  partition mirroring NEON.  Bottom-up `tree_merge` uses 2x-unrolled
  stride-16 pshufb over `expand_tab[mask]`.  AVX2 widens
  `merge_both_const` to 32-byte `pblendvb` and gives D=4 flat decode
  a 32-byte vpshufb fast path.

- **AVX-512 VBMI2** (x86-64): `vpcompressw` partition — 32 × uint16_t
  per iteration in a single instruction, no shuffle table.  BU
  `tree_merge` via `vpexpandb` (64-byte chunks, ~0.023 ns/byte on
  Xeon Granite Rapids).  Flat-subtree D=5/D=6 SIMD via
  `vpermexvar_epi8` over zmm/ymm c2s registers (the AVX-512-only
  wins).  Available on Ice Lake+ / Zen 4+.

An **SVE** backend exists but is disabled: at 128-bit SVE (Graviton 4),
`svcompact` handles only 4 × uint32 (requiring widen/narrow), which
is slower than NEON TBL's 8 × uint16. SVE would help at 256-bit+
vector lengths (e.g. Fujitsu A64FX).

### Key Design Decisions

1. **Tree walk, not layer-by-layer**: Earlier prototypes used flat
   BFS layers (like PIVCO-varint). This stored 1 code bit per symbol
   per layer — too little work per layer to amortize overhead. The
   tree walk naturally groups work by subtree, and leaf nodes terminate
   entire groups at once.

2. **No continuation bitmaps**: With canonical Huffman codes, the
   tree structure is known to both encoder and decoder. The decoder
   derives which symbols terminate at each level from the tree itself.
   This eliminates the 2x size overhead that explicit bitmaps caused.

3. **Bottom-up tree_merge decode**: Production decode is bottom-up
   since 2026-05-12 (`5828ddb` K_right wire format).  Each call
   decodes a subtree into a contiguous K-byte output buffer; internal
   nodes recurse into children (which write to scratch), then merge
   via the bitmap.  Beats the prior top-down scatter approach on
   every benched distribution.  The K_right header (2 bytes per
   non-leaf-child internal node) lets the decoder size each child's
   buffer ahead of time instead of computing it from a popcount.

4. **DFS-ordered encoding**, v0.3 wire format (2026-05-13).  The
   encoder walks the tree in the same DFS order as the decoder.
   Per non-flat internal node, the on-the-wire record is
   `[optional K_right:u16 LE][FSE marker:u8][bitmap or FSE
   payload]`.  See `src/pivco_huffman_wire.h` for the authoritative
   format spec.

5. **Per-node FSE coding of partition bitmaps** is implemented
   but **experimental — not enabled in the headline bench
   numbers below**.  When toggled on at runtime
   (`pivco_huffman_set_fse_enabled(1)`, default on, or via the
   bench's lack of `--no-fse`), heavily-skewed bitmaps get
   FSE-compressed at encode time (gated on partition skew +
   per-codeword cost) for ~25% compression-ratio win on heavy-skew
   distributions (proba80, calgary_pic) at the cost of ~3-4× decode
   speed on the FSE-firing nodes specifically.  The decode-speed
   tradeoff isn't right yet for "default" status; see
   [`IDEAS.md`](IDEAS.md) "FSE wide-cursor decoder" (2026-05-15) for
   the current research direction — the microbench in
   `extras/bench_fse_xy_micro.c` shows that FSE's stock x=2-state
   shape is 2.5-3× off the actual primitive limit; swapping to
   x=8..16 cursors closes most of the per-node decode-speed gap to
   huf0_x2 on wide-OoO cores (M4, Granite Rapids) and roughly halves
   it on Zen and Graviton.  The benchmark tables and per-platform
   commentary below use `--no-fse` to reflect the default-recommended
   path on the *current* (x=2) per-node FSE shape.

For step-by-step traces of the hot SIMD kernels (NEON `partition_8`,
`tree_merge`, `flat_dN_unpack`) with worked examples and register-
state-per-instruction walkthroughs, see [`KERNELS.md`](KERNELS.md).

## Baselines

Four decode implementations for comparison:

### Traditional 1-stream (trad_1s)

Flat 2^max_code_len lookup table. For each symbol: peek bits, table
lookup, consume bits, refill. Simple loop-based byte-at-a-time refill.

### Our 4-stream (trad_4s)

huff0-style 4-stream decode with adaptive table size (2^max_code_len
entries, packed uint16_t). Branchless 64-bit refill. Interleaved
decode: 2 rounds of 4 streams (8 decodes) per refill cycle.

### huff0 (cyan4973/FiniteStateEntropy)

The actual huff0 library from https://github.com/cyan4973/FiniteStateEntropy.
State-of-the-art Huffman decoder used in zstd. Features:
- X1 decoder: 11-bit primary table, single-symbol per lookup
- 4-stream interleaved decode for ILP
- Branchless bit buffer with byte-granularity refill
- Aggressive unrolling

Tested in both 1-stream (`HUF_decompress1X`) and 4-stream
(`HUF_decompress4X`) modes.

### ryg_rans alias (rygorous/ryg_rans)

Fabian Giesen's rANS decoder with alias method from
https://github.com/rygorous/ryg_rans. Uses the alias method to build
a decode table sized by symbol count (256 entries), not code length.
Single-pass, branch-free per symbol. Tested in 1-stream and
interleaved 2-stream modes.

## Prior Art Survey

We surveyed the literature and open-source landscape for fast Huffman
decoders. Key findings:

- **huff0/zstd is the CPU SotA.** Recent zstd PRs (#3826, #3827)
  focus on compiler-level fixes (manual unrolling, bit masking for
  optimizer hints), not algorithmic changes. The 4-stream table lookup
  architecture hasn't changed fundamentally.

- **Dougall Johnson's sync-point parallel decode** (2022): Split the
  bitstream at arbitrary points, find synchronization by running
  parallel decoders at n consecutive offsets (n = max code length).
  ~25% speedup on M1 for DEFLATE. Orthogonal to PIVCO — parallelizes
  the same serial bitstream rather than using a different format.

- **Fabian Giesen's alias Huffman** (2014): Uses the alias method with
  rANS for a unified decode table sized by symbol count. Branch-free,
  but the multiply-heavy decode step is slower than table lookup on
  ARM. Tested: 200-440 M/s on M4, significantly slower than huff0.

- **GPU massively parallel** (Weissenberger et al., 2018): Uses
  Huffman self-synchronization for thousands of GPU threads. 10x+ over
  CPU. Not relevant for single-core CPU comparison.

- **512-bit SIMD Huffman encoding** (IEEE TCE, 2023): 2.66x speedup
  for encoding on NEON. Decoding remains the harder problem.

- **No known CPU decoder beats huff0 on general Huffman decode.**
  PIVCO's 1.4x on skewed data with a realistic workload appears
  to be a novel result for a single-core CPU Huffman decoder.

### Relationship to wavelet trees

The bitmap-per-Huffman-internal-node wire format PIVCO uses is
structurally identical to a **Huffman-shaped wavelet tree**:
Grossi-Gupta-Vitter (SODA 2003) and Mäkinen-Navarro (the Huffman
shape).  The SIMD partition primitives (TBL/pshufb/vpcompress) are
also published — Kaneta (SPIRE 2018) and Dinklage-Fischer-Kurpicz-
Tarnowski (DCC 2023, AVX-512) — for wavelet-tree *construction*.
Both prior-art families are strictly top-down and frame the
representation as a rank/select **index** for substring queries
(FM-index / r-index), not as a stream codec.

PIVCO's contributions, after this survey, are: (a) framing the
representation as a bulk stream codec, (b) SIMD bulk *decode* —
in particular the bottom-up `tree_merge` direction not seen in any
WT paper, (c) the flat-subtree fast path (maximal `D ≥ 2` flat
subtrees → one `N·D`-bit packed region + `code_to_sym` lookup),
(d) per-node FSE on the bitmap, (e) empirical positioning against
huf0 / zstd / brotli / FSE on real distributions across Apple M4 /
Graviton 4 / Xeon Granite Rapids / Zen 3 / Zen 5.

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

Full prior-art notes in [`docs/WAVELET_TREES.md`](docs/WAVELET_TREES.md).

## Test Datasets

29 distributions, split into 19 synthetic (parametric/historical) and
10 real-world (byte frequencies of actual files).  Per-distribution
metadata below is generated by [`extras/bench_dist_stats.c`](extras/bench_dist_stats.c)
(`./build/pivco_dist_stats`):

- **distinct** — number of byte values with non-zero frequency.
- **entropy** — Shannon entropy in bits/symbol (bits/byte for the
  real-world set).  Lower bound on bits/symbol any prefix code can
  achieve.
- **min / max code len** — the resulting Huffman tree's shallowest
  and deepest leaf depths.  Determines partition recursion depth.

Tree-shape SVGs for every distribution are pre-rendered in
[`extras/figures/`](extras/figures/).

### Synthetic distributions

Generated by `bench_distributions.c`.  The `proba*` family is
identical to FiniteStateEntropy's bench: a 2048-entry table where
each symbol gets p% of remaining entries, then symbols are sampled
uniformly.  The `flat_M*` and `sparse_*` families exercise the
flat-tree fast path.

| Name           | Description                                    | distinct | entropy | min / max code len |
|----------------|------------------------------------------------|---------:|--------:|-------------------:|
| `proba80`      | each symbol gets 80% of remaining mass         |        6 |   0.90  | 1 / 5              |
| `proba50`      | 50% of remaining mass                          |       12 |   2.00  | 1 / 11             |
| `proba14`      | 14% of remaining mass                          |       48 |   4.18  | 3 / 11             |
| `proba02`      | 2% of remaining mass (near-uniform)            |      255 |   7.08  | 6 / 11             |
| `bell_s10`     | discretised Gaussian, σ=10                     |      256 |   5.38  | 5 / 15             |
| `bell_s30`     | discretised Gaussian, σ=30                     |      256 |   6.95  | 6 / 15             |
| `bell_s80`     | discretised Gaussian, σ=80                     |      256 |   7.91  | 7 / 9              |
| `uniform`      | all 256 symbols equally likely                 |      256 |   8.00  | 8 / 8              |
| `english`      | English character frequencies (lowercase)      |       30 |   4.23  | 3 / 10             |
| `zipfian`      | Zipf(s=1) over 256 symbols                     |      256 |   6.22  | 3 / 10             |
| `sparse_4`     | 4 symbols, equal frequency                     |        4 |   2.00  | 2 / 2              |
| `sparse_16`    | 16 symbols, equal frequency                    |       16 |   4.00  | 4 / 4              |
| `geometric`    | freq[i] = 2^(30-i), steep skew                 |      256 |   2.00  | 1 / 15             |
| `two_sym_eq`   | 2 symbols, 50/50                               |        2 |   1.00  | 1 / 1              |
| `two_sym_90/10`| 2 symbols, 90 / 10                             |        2 |   0.47  | 1 / 1              |
| `flat_M3`      | 8 equal-frequency symbols (flat tree)          |        8 |   3.00  | 3 / 3              |
| `flat_M5`      | 32 equal-frequency symbols                     |       32 |   5.00  | 5 / 5              |
| `flat_M6`      | 64 equal-frequency symbols                     |       64 |   6.00  | 6 / 6              |
| `flat_M7`      | 128 equal-frequency symbols                    |      128 |   7.00  | 7 / 7              |

### Real-world byte distributions

Byte frequencies of actual files in [`extras/datasets/`](extras/datasets/) —
provenance / regeneration commands documented there.  These are the
realistic stress tests; the synthetic `english` overstates real prose
performance because real text has a wider alphabet and a deeper Huffman
tree (see the `prose_pride` row).

| Name           | Source                                                 | distinct | entropy (b/B) | min / max code len |
|----------------|--------------------------------------------------------|---------:|--------------:|-------------------:|
| `html_wiki`    | en.wikipedia.org/wiki/Cat HTML                         |      197 |     5.48      | 4 / 15             |
| `prose_pride`  | Project Gutenberg *Pride and Prejudice*                |       96 |     4.53      | 3 / 15             |
| `image_jpeg`   | Wikimedia Commons cat photo (JPEG)                     |      256 |     7.89      | 6 / 10             |
| `json_api`     | GitHub API commits feed JSON                           |       98 |     5.20      | 3 / 15             |
| `source_c`     | zstd `lib/compress/zstd_compress.c`                    |       97 |     4.96      | 2 / 15             |
| `log_apache`   | Apache HTTP access log sample                          |       85 |     5.50      | 4 / 15             |
| `dna_fasta`    | NCBI E. coli K-12 genome FASTA                         |       38 |     2.08      | 2 / 10             |
| `csv_numeric`  | OWID CO2 dataset CSV                                   |       54 |     3.30      | 2 / 15             |
| `gzip_random`  | gzip(`cat-wiki.html`) — near-random bytes              |      256 |     8.00      | 8 / 8              |
| `chinese_text` | Project Gutenberg 紅樓夢 (Chinese, UTF-8)              |      151 |     5.81      | 4 / 15             |
| `calgary_pic`  | Calgary Corpus 1bpp CCITT scanned page (mostly white)  |      159 |     1.21      | 1 / 17             |

`pivco_file_to_dist FILE` regenerates a freq table for any input.

## Benchmark Results

**Methodology**: Decode a 4M-symbol sequence, repeated 100 times per
timed run (400M symbols/run). 5 runs, drop 2 slowest, report median
of 3 best. Each codec uses its natural block size: PIVCO/trad use
4096-8192 symbol blocks (auto-detected per backend), huf0 uses 128KB
chunks, rANS decodes full 4M.

Baselines include huf0 X1 (single-symbol lookup) and X2 (double-symbol
lookup). "vs best" = best PIVCO / best of all other decoders.

Per-host raw sweep files in [`results/`](results/) — most recent on
the current code at the time of this README are the post-unify-framework
sweeps: [`SUMMARY-20260515-unify-all-nofse.md`](results/SUMMARY-20260515-unify-all-nofse.md)
(the default-recommended `--no-fse` configuration used in the tables
below) and [`SUMMARY-20260515-unify-all-fseon.md`](results/SUMMARY-20260515-unify-all-fseon.md)
(FSE-on for ratio/speed comparison).  See also
[`SUMMARY-20260514-unify-framework.md`](results/SUMMARY-20260514-unify-framework.md)
for the MAIN-set sweep that validated the refactor.

### Apple M4 Max (NEON, 128KB L1D, block 8192)

*(post unify-framework refactor, 2026-05-15; 100 reps × 4M symbols,
median of 3 of 5 runs.  Full sweep file:
[`results/sweep_m4-20260515-unify-all-nofse.txt`](results/sweep_m4-20260515-unify-all-nofse.txt).
Real-world distributions (`html_wiki` … `calgary_pic`) source files in
[`extras/datasets/`](extras/datasets/).)*

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |     15339 |    1360 |    2617 |    1605 | **5.91x** |
| proba50       |      9151 |    1344 |    2550 |    1466 | **3.62x** |
| proba14       |      5204 |    1338 |    2482 |    1461 | **2.10x** |
| proba02       |      4516 |    1315 |    1472 |    1464 | **3.08x** |
| bell_s10      |      6398 |    1314 |    2261 |    1457 | **2.83x** |
| bell_s30      |      4329 |    1326 |    1402 |    1472 | **2.94x** |
| bell_s80      |      4335 |       0 |       0 |    1567 | **2.77x** |
| uniform       |      5017 |       0 |       0 |    1577 | **3.18x** |
| english       |      6142 |    1314 |    2414 |    1508 | **2.55x** |
| zipfian       |      4159 |    1305 |    1763 |    1508 | **2.36x** |
| sparse_4      |     47619 |    3408 |    5034 |    1575 | **9.46x** |
| sparse_16     |     46162 |    3120 |    4435 |    1579 | **10.68x** |
| geometric     |      7360 |    1310 |    2500 |    1453 | **2.94x** |
| two_sym_eq    |     24863 |    3411 |    5168 |    1583 | **4.82x** |
| two_sym_90/10 |     24860 |    3397 |    4938 |    1580 | **5.03x** |
| flat_M3       |     21478 |    3402 |    5244 |    1577 | **4.10x** |
| flat_M5       |     24006 |    3407 |    5005 |    1572 | **4.80x** |
| flat_M6       |     19963 |    3338 |    4363 |    1563 | **4.58x** |
| flat_M7       |      4829 |    3420 |    2673 |    1562 | **1.43x** |
| html_wiki     |      4316 |    1304 |    2110 |    1453 | **2.05x** |
| prose_pride   |      4713 |    1299 |    2312 |    1458 | **2.04x** |
| image_jpeg    |      4092 |    1302 |    1278 |    1509 | **2.71x** |
| json_api      |      4267 |    1290 |    2199 |    1451 | **1.94x** |
| source_c      |      4664 |    1305 |    2158 |    1472 | **2.18x** |
| log_apache    |      4490 |    1305 |    2135 |    1457 | **2.10x** |
| dna_fasta     |      8323 |    1320 |    2558 |    1515 | **3.25x** |
| csv_numeric   |      6359 |    1305 |    2459 |    1461 | **2.59x** |
| gzip_random   |      4931 |       0 |       0 |    1559 | **3.19x** |
| chinese_text  |      4826 |    1311 |    1945 |    1452 | **2.50x** |
| calgary_pic   |     11274 |    1292 |    2369 |    1460 | **4.76x** |

(`--no-fse` numbers — FSE coding of partition bitmaps is a separate
ratio/speed knob, see Implementation notes.)

### Intel Xeon 6975P-C (AVX-512 VBMI2 + VBMI, 48KB L1D, block 8192)

*(post unify-framework refactor, 2026-05-15; AWS `test-c8i`,
2 vCPU, GCC 11.5.0, Amazon Linux 2023; 100 reps × 4M symbols.
Full sweep file:
[`results/sweep_c8i-20260515-unify-all-nofse.txt`](results/sweep_c8i-20260515-unify-all-nofse.txt).)*

| Distribution  | PIVCO AVX512 | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |     22581 |    1139 |    1930 |     798 | **11.70x** |
| proba50       |     11091 |    1143 |    1927 |     723 | **5.76x** |
| proba14       |      5849 |    1149 |    1862 |     722 | **3.14x** |
| proba02       |      4388 |    1131 |    1108 |     721 | **3.88x** |
| bell_s10      |      7240 |    1133 |    1722 |     721 | **4.20x** |
| bell_s30      |      4693 |    1134 |    1051 |     721 | **4.14x** |
| bell_s80      |      4202 |       0 |       0 |     775 | **5.44x** |
| uniform       |      4415 |       0 |       0 |     786 | **5.63x** |
| english       |      7909 |    1143 |    1875 |     757 | **4.22x** |
| zipfian       |      4593 |    1132 |    1340 |     757 | **3.43x** |
| sparse_4      |     24021 |    1138 |    1947 |     799 | **12.34x** |
| sparse_16     |     20008 |    1146 |    1928 |     798 | **10.39x** |
| geometric     |     10503 |    1146 |    1925 |     722 | **5.46x** |
| two_sym_eq    |     26598 |    1128 |    1943 |     799 | **13.69x** |
| two_sym_90/10 |     26551 |    1124 |    1922 |     799 | **13.81x** |
| flat_M3       |     21834 |    1145 |    1937 |     799 | **11.28x** |
| flat_M5       |     18481 |    1149 |    1917 |     796 | **9.64x** |
| flat_M6       |     17137 |    1147 |    1770 |     793 | **9.69x** |
| flat_M7       |      3727 |    1144 |     985 |     791 | **3.27x** |
| html_wiki     |      4765 |    1136 |    1608 |     721 | **2.96x** |
| prose_pride   |      5689 |    1135 |    1768 |     721 | **3.22x** |
| image_jpeg    |      3891 |    1132 |     975 |     755 | **3.44x** |
| json_api      |      5023 |    1138 |    1687 |     721 | **2.98x** |
| source_c      |      5159 |    1139 |    1647 |     722 | **3.13x** |
| log_apache    |      4899 |    1143 |    1628 |     722 | **3.01x** |
| dna_fasta     |     13615 |    1144 |    1920 |     758 | **7.09x** |
| csv_numeric   |      7156 |    1145 |    1861 |     722 | **3.84x** |
| gzip_random   |      4416 |       0 |       0 |     786 | **5.63x** |
| chinese_text  |      5623 |    1137 |    1483 |     720 | **3.79x** |
| calgary_pic   |      9764 |    1128 |    1858 |     723 | **5.26x** |

### AWS Graviton 4 Neoverse V2 (NEON, 64KB L1D, block 8192)

*(post unify-framework refactor, 2026-05-15; AWS `test-c8g`,
2 vCPU c8g.large pinned `taskset -c 0`, GCC 11.5.0, Amazon Linux 2023;
100 reps × 4M symbols.  Full sweep file:
[`results/sweep_c8g-20260515-unify-all-nofse.txt`](results/sweep_c8g-20260515-unify-all-nofse.txt).
D=5/D=6 NEON paths re-enabled on the BU direct path since 2026-05-15
— see `IDEAS.md` and the lifted `PIVCO_NEON_FAST_MULTI_TBL` gate.)*

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      8523 |    1049 |    1927 |    1029 | **4.42x** |
| proba50       |      5095 |    1038 |    1935 |     900 | **2.63x** |
| proba14       |      2630 |    1042 |    1869 |     897 | **1.41x** |
| proba02       |      2271 |    1031 |    1117 |     894 | **2.03x** |
| bell_s10      |      3180 |    1032 |    1709 |     897 | **1.86x** |
| bell_s30      |      2192 |    1031 |    1060 |     894 | **2.07x** |
| bell_s80      |      2165 |       0 |       0 |     992 | **2.18x** |
| uniform       |      2398 |       0 |       0 |    1000 | **2.40x** |
| english       |      3076 |    1037 |    1869 |     954 | **1.65x** |
| zipfian       |      2116 |    1030 |    1350 |     955 | **1.57x** |
| sparse_4      |     16714 |    1050 |    1945 |    1026 | **8.59x** |
| sparse_16     |     15165 |    1052 |    1888 |    1020 | **8.04x** |
| geometric     |      4155 |    1038 |    1935 |     899 | **2.15x** |
| two_sym_eq    |     12860 |    1047 |    1939 |    1026 | **6.63x** |
| two_sym_90/10 |     12856 |    1046 |    1917 |    1026 | **6.71x** |
| flat_M3       |      9860 |    1052 |    1944 |    1024 | **5.10x** |
| flat_M5       |      9195 |    1051 |    1868 |    1022 | **4.94x** |
| flat_M6       |      9299 |    1044 |    1712 |    1019 | **5.43x** |
| flat_M7       |      2519 |    1045 |     984 |    1011 | **2.41x** |
| html_wiki     |      2184 |    1032 |    1610 |     906 | **1.36x** |
| prose_pride   |      2417 |    1032 |    1784 |     895 | **1.36x** |
| image_jpeg    |      2011 |    1034 |     970 |     948 | **1.98x** |
| json_api      |      2179 |    1034 |    1689 |     898 | **1.29x** |
| source_c      |      2428 |    1035 |    1655 |     898 | **1.47x** |
| log_apache    |      2259 |    1036 |    1628 |     897 | **1.39x** |
| dna_fasta     |      4512 |    1043 |    1914 |     959 | **2.36x** |
| csv_numeric   |      3264 |    1037 |    1872 |     900 | **1.74x** |
| gzip_random   |      2398 |       0 |       0 |     999 | **2.40x** |
| chinese_text  |      2442 |    1032 |    1487 |     897 | **1.64x** |
| calgary_pic   |      5926 |    1031 |    1839 |     901 | **3.22x** |

### AMD EPYC 7R13 Zen 3 (AVX2 + SSE4.1, 32KB L1D, block 4096)

*(post unify-framework refactor, 2026-05-15; AWS `test-c6a`,
2 vCPU, clang-20, Amazon Linux 2023; 100 reps × 4M symbols.
Full sweep file:
[`results/sweep_c6a-20260515-unify-all-nofse.txt`](results/sweep_c6a-20260515-unify-all-nofse.txt).
Note: Zen 3 lacks AVX-512, so codec_x86 (SSE/AVX2 paths) is dispatched
— `vpcompressw`-class partition is unavailable, expressed via pshufb
+ compress_tab instead.)*

| Distribution  | PIVCO SSE/AVX2 | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      8087 |    1081 |    1631 |     929 | **4.96x** |
| proba50       |      4235 |    1083 |    1615 |     806 | **2.63x** |
| proba14       |      1624 |     999 |    1530 |     802 | **1.06x** |
| proba02       |      1245 |     992 |     912 |     802 | **1.26x** |
| bell_s10      |      2213 |     992 |    1402 |     803 | **1.58x** |
| bell_s30      |      1340 |     992 |     866 |     802 | **1.35x** |
| bell_s80      |      1505 |       0 |       0 |     891 | **1.69x** |
| uniform       |      2963 |       0 |       0 |     907 | **3.27x** |
| english       |      1753 |     993 |    1533 |     861 | **1.14x** |
| zipfian       |      1400 |     986 |    1104 |     860 | **1.27x** |
| sparse_4      |      2945 |    1003 |    1619 |     931 | **1.82x** |
| sparse_16     |     24519 |    1002 |    1606 |     928 | **15.27x** |
| geometric     |      3751 |     999 |    1576 |     807 | **2.38x** |
| two_sym_eq    |     36417 |     998 |    1622 |     931 | **22.45x** |
| two_sym_90/10 |     36245 |    1001 |    1635 |     931 | **22.17x** |
| flat_M3       |      2650 |    1004 |    1625 |     929 | **1.63x** |
| flat_M5       |      2306 |    1005 |    1571 |     924 | **1.47x** |
| flat_M6       |      2197 |    1002 |    1498 |     921 | **1.47x** |
| flat_M7       |      2163 |     998 |     833 |     916 | **2.17x** |
| html_wiki     |      1246 |     993 |    1312 |     802 |   0.95x |
| prose_pride   |      1568 |     990 |    1452 |     803 | **1.08x** |
| image_jpeg    |      1412 |     991 |     798 |     859 | **1.42x** |
| json_api      |      1312 |     992 |    1391 |     802 |   0.94x |
| source_c      |      1514 |     996 |    1340 |     803 | **1.13x** |
| log_apache    |      1247 |     995 |    1323 |     802 |   0.94x |
| dna_fasta     |      4535 |    1007 |    1596 |     864 | **2.84x** |
| csv_numeric   |      2210 |     993 |    1533 |     805 | **1.44x** |
| gzip_random   |      2967 |       0 |       0 |     908 | **3.27x** |
| chinese_text  |      1503 |     995 |    1204 |     803 | **1.25x** |
| calgary_pic   |      3827 |     991 |    1527 |     807 | **2.51x** |

### Cross-Platform Summary

*(post unify-framework refactor, 2026-05-15; see per-platform
sections above for raw M/s.  `--no-fse` configuration — FSE
parameter tuning trades decode speed for compression ratio on
heavy-skew nodes, see Implementation notes.  Real-world byte
distributions (`html_wiki` … `calgary_pic`) sourced from the files
in [`extras/datasets/`](extras/datasets/).)*

`pivco_bu` vs `huf0_x2` (or `trad_4s` where `huf0` fails), one
column per host:

| Distribution | M4 NEON | Xeon AVX-512 | Graviton4 NEON | Zen3 SSE/AVX2 |
|---|---:|---:|---:|---:|
| proba80         | **5.91x** | **11.70x** | **4.42x** | **4.96x** |
| proba50         | **3.62x** | **5.76x** | **2.63x** | **2.63x** |
| proba14         | **2.10x** | **3.14x** | **1.41x** | **1.06x** |
| proba02         | **3.08x** | **3.88x** | **2.03x** | **1.26x** |
| bell_s10        | **2.83x** | **4.20x** | **1.86x** | **1.58x** |
| bell_s30        | **2.94x** | **4.14x** | **2.07x** | **1.35x** |
| bell_s80        | **2.77x** | **5.44x** | **2.18x** | **1.69x** |
| uniform         | **3.18x** | **5.63x** | **2.40x** | **3.27x** |
| english         | **2.55x** | **4.22x** | **1.65x** | **1.14x** |
| zipfian         | **2.36x** | **3.43x** | **1.57x** | **1.27x** |
| sparse_4        | **9.46x** | **12.34x** | **8.59x** | **1.82x** |
| sparse_16       | **10.68x** | **10.39x** | **8.04x** | **15.27x** |
| geometric       | **2.94x** | **5.46x** | **2.15x** | **2.38x** |
| two_sym_eq      | **4.82x** | **13.69x** | **6.63x** | **22.45x** |
| two_sym_90/10   | **5.03x** | **13.81x** | **6.71x** | **22.17x** |
| flat_M3         | **4.10x** | **11.28x** | **5.10x** | **1.63x** |
| flat_M5         | **4.80x** | **9.64x** | **4.94x** | **1.47x** |
| flat_M6         | **4.58x** | **9.69x** | **5.43x** | **1.47x** |
| flat_M7         | **1.43x** | **3.27x** | **2.41x** | **2.17x** |
| `html_wiki`   ‡ | **2.05x** | **2.96x** | **1.36x** | 0.95x |
| `prose_pride` ‡ | **2.04x** | **3.22x** | **1.36x** | **1.08x** |
| `image_jpeg`  ‡ | **2.71x** | **3.44x** | **1.98x** | **1.42x** |
| `json_api`    ‡ | **1.94x** | **2.98x** | **1.29x** | 0.94x |
| `source_c`    ‡ | **2.18x** | **3.13x** | **1.47x** | **1.13x** |
| `log_apache`  ‡ | **2.10x** | **3.01x** | **1.39x** | 0.94x |
| `dna_fasta`   ‡ | **3.25x** | **7.09x** | **2.36x** | **2.84x** |
| `csv_numeric` ‡ | **2.59x** | **3.84x** | **1.74x** | **1.44x** |
| `gzip_random` ‡ | **3.19x** | **5.63x** | **2.40x** | **3.27x** |
| `chinese_text`‡ | **2.50x** | **3.79x** | **1.64x** | **1.25x** |
| `calgary_pic` ‡ | **4.76x** | **5.26x** | **3.22x** | **2.51x** |

‡ Real-world byte-frequency distributions.  Source files in
[`extras/datasets/`](extras/datasets/), regeneration via
`pivco_file_to_dist`.  `calgary_pic` is the Calgary Corpus 1bpp
CCITT scanned page (real-world proba80-shaped: 1.21 b/B entropy).

Observations across the grid:

- **Cost asymmetry between platforms is the most striking part of
  this data.**  Same algorithm, same C source, four backends —
  ratios span 0.94× (Zen 3 deep-real-text) to 23× (Zen 3
  `two_sym_eq`).  Xeon AVX-512 has the lowest minimum ratio (2.96×)
  and the highest dynamic range; Graviton 4 sits between M4 and
  Zen 3 on every dimension.
- **The K_right wire format (`5828ddb`, 2026-05-12) is the big
  recent landing.**  Real-text BU decode wins jumped from
  0.44-1.08× in late April to 0.94-3.22× now.  The `vpcompressw`
  partition + K_right-sized child buffers together amortise the
  per-node overhead that real-text trees (many internal nodes,
  Dmax 15) used to lose to.
- **`vpcompressw` matters on the partition path, but the BU
  tree_merge bridge made it less critical.**  Zen 3 has no
  `vpcompressw` and now wins 27/30 distributions — up from 8/29
  in April.  The partition cost is still real (the deepest-tree
  real-text distributions are the closest-to-parity losses) but
  the structural advantage of AVX-512 has narrowed.
- **Graviton 4 D=5/D=6 SIMD flat-decode was briefly disabled** by
  a too-broad uarch gate in the unify-framework refactor.
  Restored 2026-05-15; before the fix, `flat_M5` was 1.93× on c8g
  vs the 4.94× shown above.  The `vqtbl{2,4}q_u8`-over-32/64-byte-
  source pattern remains slow on Neoverse-V2 at small n, which is
  why the gate exists in the first place; the BU direct path keeps
  n large enough to amortise.
- **`two_sym_*` ratios spike on Zen 3 (22-23×)** because those are
  the only synthetic distributions where BOTH_LEAVES-at-root
  fires, hitting the per-block fast-path that bypasses
  `codec_decode_subtree` entirely.  The recent `8be22e7` restore
  of that fast path was the difference between 1.03× and 22.5×
  on these rows.

FSE-coded bitmaps are a separate ratio/speed knob, see
Implementation notes; enabling FSE moves several of the
proba80-shaped distributions toward lower decode speed and ~25%
smaller encoded size.

### Compression Ratio

PIVCO encoded size matches traditional Huffman within 1-4%, the only
overhead being byte-alignment rounding at each tree node.

### Block Size Sweep

*(Numbers below are pre-flat-subtree (early 2026-04) and stale.
They pre-date flat-subtree, leaf-fusion, the Graviton 4 D=5/D=6
gate, the K_right wire format, FSE-coded bitmaps, and the
unify-framework codec refactor.  N-dependence for stick-tree-shaped
distributions (proba80/50) was always closest to flat across N once
the prefill memset landed; the table below is consistent with that.
A fresh block-size sweep on the current code is still TODO.)*

PIVCO NEON decode throughput (M/s) by block size. Measured with the
4M realistic workload (each block size is recompiled and re-benchmarked):

| N     | proba80 | proba50 | proba14 | proba02 | english | geometric |
|------:|--------:|--------:|--------:|--------:|--------:|----------:|
|  4096 |    9122 |    4845 |    2118 |     845 |    2251 |      4502 |
|  8192 |    9401 |    5020 |    2306 |    1119 |    2418 |      4872 |
| 16384 |    9410 |    5034 |    2402 |    1333 |    2454 |      5124 |
| 65536 |    9883 |    4677 |    2245 |    1509 |    2238 |      4732 |

- **proba80 is now ~flat across block sizes** (9.1-9.9 GB/s): the
  prefill memset + skip_node optimization dominates — the tree walk
  only processes ~20% of indices regardless of block size.
- **N=8192 remains the default**: good balance across distributions.
  16384 is slightly better on moderate (proba14, geometric) but
  65536 regresses on proba50/english as index arrays spill L1.
- **Compared to pre-optimization**: all block sizes roughly doubled
  (e.g. proba80 4.1-4.3 GB/s → 9.1-9.9 GB/s).

## Key Compute Primitives

Bottoms-up per-element cost of every SIMD primitive the decoder uses,
isolated from the surrounding control flow.  Useful for reasoning about
which inner-loop pieces are the bottleneck at a given D / table shape.

The primitives below are the same bodies that landed in the
`primitives_<backend>.h` headers as part of the 2026-05-14
unify-framework refactor — the microbench cost is functionally
identical, only the file location changed.  `flat_scatter_*` were
the TD-era scatter variants (since retired); production decode uses
the contiguous `flat_direct_*` rows.

Measured by [`bench/bench_micro.c`](bench/bench_micro.c) on all four
test platforms (block N = 8192, 100k repeats per row, ~820M elements
total per row).  Numbers are **ns / elem** (lower is better).  Build:

```sh
# NEON (M4 / Graviton 4):
cc -O2 -o bench_micro bench/bench_micro.c -I include -I src
# x86_64 (Xeon AVX-512 / Zen 3 SSE4.1+AVX2):
cc -O3 -march=native -o bench_micro bench/bench_micro.c -I include -I src
```

Raw outputs in [`results/bench_micro-*-20260426-0625.txt`](results/).

### Cross-platform primitive costs

Each cell shows **`ns/elem (GB/s)`**.  The table treats one element as one
output byte (the symbol decode), so GB/s is throughput in output bytes
per second.

| Primitive                                | M4 (NEON)        | Graviton 4 (NEON) | Xeon (AVX-512 VBMI2) | Zen 3 (SSE4.1+AVX2) |
|------------------------------------------|------------------|-------------------|----------------------|---------------------|
| **Reference floors**                     |                  |                   |                      |                     |
| `memset`                                 | 0.01 (119)       | 0.01 (81)         | **0.00 (226)**       | 0.01 (112)          |
| `scatter_scalar`                         | 0.23 (4.4)       | 0.36 (2.8)        | **0.17 (5.8)**       | 0.28 (3.6)          |
| **Partition (2-way decoder core)**       |                  |                   |                      |                     |
| `partition` (load + TBL/compress)        | 0.06 (15.6)      | 0.16 (6.3)        | **0.04 (24.2)**      | 0.20 (5.1)          |
| `partition_root` (identity + TBL)        | 0.07 (14.6)      | 0.16 (6.3)        | **0.05 (20.7)**      | 0.19 (5.2)          |
| `partition_half` (load + 1 TBL)          | 0.05 (21.9)      | 0.11 (9.2)        | **0.03 (36.1)**      | 0.13 (7.5)          |
| `partition_root_half`                    | 0.05 (19.8)      | 0.11 (8.9)        | **0.03 (30.9)**      | 0.12 (8.4)          |
| **Indexed scatter (leaf write)**         |                  |                   |                      |                     |
| `scatter_simd` (const sym)               | **0.13 (7.5)**   | 0.36 (2.8)        | *(= scatter_scalar)* | 0.35 (2.8)          |
| `both_leaves_vst1` (root flat 2-sym)     | 0.03 (33.7)      | 0.07 (13.9)       | **0.01 (67.7)**      | 0.07 (14.2)         |
| `both_leaves_scatter` (idx 2-sym)        | **0.15 (6.6)**   | 0.40 (2.5)        | 0.17 (5.7)           | 0.33 (3.0)          |
| **Flat-subtree direct** (sequential out) |                  |                   |                      |                     |
| `flat_direct_d2`                         | **0.02 (52.1)**  | 0.05 (18.4)       | 0.03 (30.4)          | *(scalar)*          |
| `flat_direct_d3`                         | 0.04 (23.4)      | 0.14 (7.1)        | **0.03 (30.3)**      | *(scalar)*          |
| `flat_direct_d4`                         | **0.02 (51.7)**  | 0.06 (17.5)       | 0.03 (31.7)          | 0.04 (28.1)         |
| `flat_direct_d5`                         | **0.04 (25.4)**  | 0.78 (1.3)        | 0.04 (27.3)          | *(scalar)*          |
| `flat_direct_d6`                         | **0.04 (22.8)**  | 0.84 (1.2)        | 0.05 (20.6)          | *(scalar)*          |
| **Flat-subtree scatter** (indexed out)   |                  |                   |                      |                     |
| `flat_scatter_d2`                        | **0.14 (7.1)**   | 0.66 (1.5)        | 0.26 (3.8)           | *(scalar)*          |
| `flat_scatter_d3`                        | **0.16 (6.1)**   | 0.67 (1.5)        | 0.27 (3.8)           | *(scalar)*          |
| `flat_scatter_d4`                        | **0.14 (7.1)**   | 0.66 (1.5)        | 0.27 (3.7)           | 0.64 (1.6)          |
| `flat_scatter_d5`                        | **0.17 (5.8)**   | 1.41 (0.7)        | 0.28 (3.6)           | *(scalar)*          |
| `flat_scatter_d6`                        | **0.18 (5.5)**   | 1.50 (0.7)        | 0.32 (3.1)           | *(scalar)*          |

TBL primitive used per platform / D:

| D | NEON          | AVX-512        | SSE4.1                              |
|---|---------------|----------------|-------------------------------------|
| 2 | `vqtbl1q_u8`  | `pshufb`       | scalar (no per-byte var-shift)      |
| 3 | `vqtbl1`      | `pshufb`       | scalar                              |
| 4 | `vqtbl1q_u8`  | `pshufb`       | `pshufb` (only D with SIMD unpack)  |
| 5 | `vqtbl2q_u8`  | `vpermb` (ymm) | scalar                              |
| 6 | `vqtbl4q_u8`  | `vpermb` (zmm) | scalar                              |

Reading the table:

- **The indexed scatter floor varies hugely across platforms.**  M4
  hits ~0.14–0.18 ns/elem (5–7 GB/s) and the SIMD unpack upstream of it
  is essentially free.  Xeon AVX-512 sits at ~0.26–0.32 (3–4 GB/s, 2×
  M4) — `_mm_extract_epi8` is a 1-cycle uop but emits ~16 of them per
  16-element chunk.  Graviton 4 (0.66 / 1.5 GB/s) and Zen 3 (0.66 even
  on its D=4 SIMD path) are 4–5× M4 — per-element scalar-store throughput
  is the bottleneck, not the TBL.

- **The flat-subtree direct path is fastest on M4** at 0.02 ns/elem
  (~52 GB/s) for D=2 / D=4 — single `vqtbl1q_u8` per 16 codes.  Xeon is
  within a couple-percent at 0.03 (~30 GB/s) for D ≤ 4 (single
  `pshufb`); D=6 `vpermb-zmm` lands at 0.05 (~21 GB/s) — broadly the
  same ballpark.  Surprisingly, **Xeon `both_leaves_vst1` is 67.7 GB/s
  (0.01 ns/elem) — the fastest single row in the table**.  AVX-512's
  `mask_blend_epi8` over a 32-byte register is essentially free on
  Sapphire/Granite Rapids, beating M4's `vbslq_u8` blend by 2×.

- **Graviton 4's `vqtbl{2,4}q_u8` regression is real and visible at
  the primitive level.**  D=5 / D=6 are 0.78 / 0.84 ns/elem (1.2 GB/s)
  — **20× the M4 cost**, even at the same NEON ISA.  Empirical
  motivation for the production `PIVCO_NEON_FAST_MULTI_TBL=0` gate (see
  IDEAS.md "Graviton 4 NEON D=5/D=6 regression").

- **Zen 3 has the slowest partition** at 0.20 ns/elem (5.1 GB/s) — 3×
  M4's NEON partition (0.06 / 15.6) and 5× Xeon's `vpcompressw` (0.04
  / 24.2).  Combined with the 0.64–0.66-ns indexed scatter, Zen 3 has
  the highest absolute floor on the 2-way decoder hot path of any
  tested platform — primitive-level evidence behind the IDEAS.md
  "Zen 3 hybrid block decoder" recommendation.

- **The half-tree partition saves materially on every platform.**
  partition_half / partition_root_half (one-side store, used when one
  child is a leaf) drops cost by 30–40% vs full partition: M4 21.9 vs
  15.6 GB/s, Xeon 36.1 vs 24.2, Zen 3 7.5 vs 5.1, Graviton 4 9.2 vs
  6.3 — validating the production "leaf-child fusion" optimisation
  uniformly.

- **The flat-subtree fast path has a real edge over partition-and-
  scatter when the output is sequential** (root-flat or covered
  subtree).  M4 0.02 vs partition's 0.06 — 3× cheaper.  Once stores
  are indexed (non-root flat subtree), the gap collapses to the
  per-platform scatter floor and most of the SIMD unpack savings are
  absorbed — visible in the `flat_scatter_dN` rows being tightly
  bunched within each platform regardless of D.

These primitives explain the per-distribution numbers in the
[per-platform decode tables above](#apple-m4-max-neon-128kb-l1d-block-8192):
flat-heavy distributions (`uniform`, `flat_M*`, `sparse_*`) cash in
the cheap direct path; deep-tree distributions (`prose_pride`,
`html_wiki`) pay the partition cost per level repeatedly.

## Profiling

> **Historical snapshot.**  The profile below was taken on the
> top-down decoder (`decode_node_neon`, `partition_8`,
> `scatter_both_leaves`, `flat_decode_scatter_neon`) on 2026-04-26.
> The production decoder has been bottom-up since 2026-05-12
> (`5828ddb` K_right wire format) and the source files referenced
> here (`pivco_huffman_neon.c`) have been folded into
> `pivco_huffman_codec.c` + `pivco_huffman_primitives_neon.h` as of
> the 2026-05-14 unify-framework refactor.  The per-function names
> below no longer exist verbatim.  The section is retained because
> the qualitative breakdown — partition body 41%, flat-subtree 24%,
> leaf scatter 18%, recursion glue 12% — and especially the
> conclusion that **NEON store-port throughput, not TBL latency, is
> the partition bottleneck** still describe the bottom-up decoder
> faithfully (the BU `tree_merge` is store-port bound for the same
> reason).  A BU re-profile is planned.

**Last refreshed:** 2026-04-26 07:30 UTC, commit
[`0a99f6c`](../) (post AVX-512 / SSE4.1 bench port, leaf-child fusion +
flat-subtree fast path both shipped).  Workload: **`prose_pride`** —
real Project Gutenberg prose, 96 distinct bytes, max code length 11
(length-limited Huffman, matching `huf0`'s default cap),
~47% flat-subtree coverage — the real-world deep-tree distribution
that PIVCO most closely contests against `huf0_x2`.

### Methodology

Earlier revisions of this section used macOS `sample` (1 ms IP
sampling, no source-line attribution) and **manual** instruction-
offset → source-region mapping by hand-disassembling
`decode_node_neon`.  That approach produced numbers that turned out
to be off by 10–15 percentage points — the collapsed-offset display
in `sample` plus M4's deep OoO retirement attribution made the
hand-mapping noisier than it looked.

This refresh uses **Instruments / `xctrace` Time Profiler with
DWARF inlined-frame attribution** instead.  The profile binary is
built `RelWithDebInfo` and `dsymutil`'d so DWARF debug info covers
every inlined helper (`partition_8`, `scatter_sym`,
`scatter_both_leaves`, `flat_decode_scatter_neon`, `flat_dN_unpack`,
…), letting the trace attribute each sample directly to a source
function and line.  Decode-loop samples are isolated by filtering
backtraces that contain `decode_node_neon` or
`pivco_huffman_decode_neon` (excludes the encode-phase setup the
harness runs before the decode loop).

All five steps are wrapped in
[`extras/profile_m4.sh`](extras/profile_m4.sh) for a one-line
re-run:

```sh
./extras/profile_m4.sh prose_pride 12   # dist, duration_s
```

The script (1) configures+builds RelWithDebInfo, (2) generates the
`.dSYM`, (3) records an xctrace Time Profiler trace, (4) exports
the `time-profile` table to XML, (5) runs
[`extras/profile_xctrace_parse.py`](extras/profile_xctrace_parse.py)
to filter to decode-loop samples and aggregate by leaf frame.
Output goes to `results/profile-${HOST}-${DIST}-xctrace-${TS}.txt`.

10 s wall window → 9996 decode-loop samples × 1 ms.  Parsed
summary:
[`results/profile-m4_max-prose_pride-xctrace-20260426-0625.txt`](results/profile-m4_max-prose_pride-xctrace-20260426-0625.txt).

### Per-function self-time (decode loop only, % of 9996 samples)

xctrace's DWARF inlined-frame attribution is what makes this
breakdown trustworthy: the leaf frame is the source-level innermost
function the IP belongs to, even when that function was inlined
into `decode_node_neon` at `-O2`.

| Function                   | %       | Source location        | Description                                       |
|----------------------------|--------:|------------------------|---------------------------------------------------|
| `partition_8`              | **37.9%** | `pivco_huffman_neon.c:88+`  | 2-way partition core (TBL + store)              |
| `flat_decode_scatter_neon` | **16.2%** | `pivco_huffman_neon.c:230+` | flat-subtree TBL + indexed store                |
| `decode_node_neon`         | **11.8%** | `pivco_huffman_neon.c:754+` | recursion glue + leaf checks + recurse setup    |
| `scatter_both_leaves`      |    9.9% | `pivco_huffman_neon.c:704+` | both-leaves stage fusion (sequential write)     |
| `scatter_sym`              |    8.6% | `pivco_huffman_neon.c:660+` | leaf scatter (one child = leaf)                 |
| `flat_d3_unpack`           |    4.1% | `pivco_huffman_neon_flat.h:71` | D=3 bit-unpack inside flat path                 |
| `flat_d2_unpack`           |    3.9% | `pivco_huffman_neon_flat.h:44` | D=2 bit-unpack                                   |
| `partition_8_right`        |    3.7% | `pivco_huffman_neon.c:638+` | half-partition (one side store, leaf-fusion)    |
| `_platform_memset`         |    3.2% | (libsystem)            | phase-0 `prefill_sym` of most-frequent leaf       |
| `pivco_huffman_decode_neon`|    0.5% | `pivco_huffman_neon.c:1061` | per-block wrapper (root partition setup)        |
| `bitmap_get` / `extract_D_bits` / `bitmap_bytes` | 0.3% | `pivco_huffman_common.h` | scalar tail / fallback paths     |

Top single source line: `partition_8` at `pivco_huffman_neon.c:98`
(the second `vst1q_u8` storing the left-partition output) — **28.5%
of all CPU time** alone.  The first `vst1q_u8` (line 95, popcnt
load + first store) takes another 9.4%.  Together those two stores
in `partition_8` account for **38% of total** — the actual TBL
shuffle and bitmap loads scarcely show up.  Consistent with M4's
partition microbench cost (0.06 ns/elem at ~15.5 GB/s) being store-
port bound, not TBL-throughput bound.

### Aggregated by source region (% of total CPU)

| Region                         | %        | Comprises                                                |
|--------------------------------|---------:|----------------------------------------------------------|
| **Partition body**             | **41.6%** | `partition_8` + `partition_8_right`                      |
| **Flat-subtree path**          | **24.2%** | `flat_decode_scatter_neon` + `flat_d2_unpack` + `flat_d3_unpack` |
| **Leaf scatter**               | **18.4%** | `scatter_sym` + `scatter_both_leaves`                    |
| **Recursion glue**             | **11.8%** | `decode_node_neon` (non-inlined: leaf checks, recurse)   |
| **Phase-0 prefill**            |     3.2% | `_platform_memset`                                        |
| **Per-block frame + scalar tail** |  0.8% | `pivco_huffman_decode_neon`, `bitmap_get`, …             |

(Total 100.0%.)

### Comparison to the old profile

The previous profile (zipfian on the pre-flat-subtree code, hand-
mapped) reported the breakdown below.  Side-by-side with the
xctrace numbers on prose_pride:

| Region                  | Old (zipfian, hand-mapped) | New (prose_pride, xctrace) | Note                                                  |
|-------------------------|---------------------------:|---------------------------:|-------------------------------------------------------|
| Partition body          |                     44.4% |                  **41.6%** | Algorithm unchanged; close match validates new tooling |
| Flat-subtree path       |                       0%  |                  **24.2%** | New region — fast path didn't exist before            |
| Leaf scatter            |                     12.3% |                  **18.4%** | Real cost was higher than previously credited; the old hand-mapping under-attributed because `sample` collapsed leaf-scatter offsets with adjacent partition offsets |
| Recursion glue          |                     14.1% (function prologue) |        **11.8%** | OoO still hides the prologue; the rest is leaf-checks and recurse-setup |
| Frame entry/epilogue    |                       —   |                     <1%   | Confirmed negligible                                   |
| Phase-0 prefill         |                       —   |                     3.2%  | Wasn't called out in old profile                      |

Two things to read here: (1) the partition-body share is essentially
unchanged from the old profile — that algorithm hasn't changed —
which is a sanity check that the new tooling is producing
believable numbers, and (2) the "regional breakdown" I did one
commit ago by hand-mapping `sample` offsets was directionally right
but quantitatively off (over-attributing partition body by ~15 pp,
under-attributing leaf scatter and recursion glue).  The xctrace
numbers above replace it.

### Profiling lesson (preserved from earlier)

"Occupies X% of execution slots" is not the same as "removing it
would be X% faster."  On an OoO core, non-critical-path work is
essentially free if it doesn't compete for the bottleneck resource
— which on the partition path is **NEON store-port throughput**
(per the line-level data: 38% of total time is the two `vst1q_u8`
stores in `partition_8`), not TBL latency or instruction issue.

## Analysis

*(As of [ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../).  Pre-flat-
subtree commentary on the losing-moderate-entropy case is preserved
in git history — commit `984dad3` and earlier.)*

### Where PIVCO Wins

**Skewed / stick-tree distributions on all platforms: 1.1-13× over
huf0.**  Prefill memset + skip_node + half-partition benefit all
backends.  On M4: 3.3× proba80 (9.6 GB/s), 5.0× two_sym_eq (27 GB/s).
On AVX-512: 3.1× proba80, 4.7× two_sym_90/10, **13.0×** sparse_4.
On Graviton 4: **9.6×** two_sym_90/10, 9.5× two_sym_eq.  On Zen 3:
1.15× proba80, 0.86× two_sym_eq (Zen 3 SSE4.1 is the floor).  proba80
now wins on all four platforms.

The tree's early-exit property means most symbols are decoded in
the first 2-3 tree levels via large scatter-writes — the per-symbol
cost drops well below a single table lookup.

**Moderate-entropy distributions on M4 / AVX-512: 1.06-1.78×.**  The
flat-subtree fast path (April 2026, commits `a275d05`+) replaces D
levels of per-level bitmaps with a single N·D-bit packed region at
every maximal flat subtree it detects, and the flat-aware tree
restructurer ([ffbfeac](../)) lays out symbols so the largest
possible flat-D≥2 subtrees form (analyzer:
`extras/bench_flat_optimal.c` predicted 16-27% partition-step
savings on the loss cluster, achieved 11-23% throughput gain).
Post-restructure: M4 wins the full moderate-entropy set
(english 1.28×, bell_s10 1.31×, bell_s30 1.60×, bell_s80 1.78×,
proba02 1.62×, proba14 1.06×, zipfian 1.40×).  Xeon AVX-512 wins
the same set with bigger margins where partition is relatively more
expensive (english 1.23×, proba14 1.07×, bell_s80 3.28×, bell_s30
1.33×, bell_s10 1.20×, proba02 1.49×, zipfian 1.38×).

**Flat-tree distributions (uniform, sparse_*, flat_M*) on all
platforms: 1.1-6.5×.**  Root-flat tables route through the same
flat-subtree packed-bit path as their non-root cousins.  Direct
scatter to `symbols[i]` with no indices materialisation or prefill
memset.

### Where PIVCO Still Loses

After the K_right wire format (2026-05-12), the flat-aware tree
restructurer, and the Graviton-4 D=5/D=6 gate, the historical loss
clusters on M4 / Xeon / Graviton 4 have all crossed to wins.  The
remaining sub-1× rows live on **Zen 3 SSE/AVX2** for three
deep-real-text distributions: `html_wiki` 0.95×, `json_api` 0.94×,
`log_apache` 0.94×.

These three share the same shape: max_len 15, alphabet 85–200,
entropy ~5.3–5.5 b/B — i.e. exactly the case where huf0_x2's
two-symbol-per-table-lookup is most valuable.  On Zen 3 the SSE/AVX2
partition primitive (pshufb + compress_tab) costs ~0.20 ns/elem,
~3× M4's NEON partition and ~5× Xeon's `vpcompressw`; combined with
no D=5/D=6 SIMD unpack and a 32 KB L1D that the compress_tab eats
~25% of, the per-partition tax is what huf0_x2 wins on.

These losses are within 6% of parity, framed as data rather than
"things to fix" — ph is a research vehicle for what SIMD Huffman
decoding can do at the algorithmic edge, not a tool people will
compress with.  The natural remediation, if it ever matters, is the
existing IDEAS.md "Zen 3 hybrid block decoder" entry: gate per
table on flat-subtree coverage and fall back to huf0_x2 below a
threshold (e.g. D=2 coverage < 30% AND max_len > 10).  Other
backends (NEON, AVX-512) would stay on PIVCO unconditionally.

### The Core Tradeoff

PIVCO's throughput is **distribution-dependent** and **hardware-
dependent**.  It wins on distributions where either the tree
terminates early (stick-tree) or the tree contains large flat
subtrees (most real data in the 4-9 bit entropy range).  It needs
wide shuffle throughput on the host to stay competitive on the
remaining moderate-depth shapes.

huf0's throughput is **mostly distribution-independent** and
**consistently fast** across platforms. Its per-symbol cost is
constant (one table lookup), and 128KB blocks amortize overhead well.

PIVCO uses small blocks (4096-8192 symbols) because larger blocks
spill L1D. huf0 uses large blocks (128KB) because that's optimal
for its 4-stream architecture. Each codec operates at its natural
block size.

### SIMD Width Scaling

PIVCO's inner loop scales directly with SIMD primitive width:

- **NEON 128-bit** (M4, Graviton 4): 8 × uint16_t per `vqtbl1q_u8`
  partition.  Shipped.
- **SSE4.1+AVX2 128/256-bit** (Zen 3): same 8-wide partition via
  pshufb + compress_tab; `pblendvb` widens `merge_both_const` to
  32-byte.  Shipped.
- **AVX-512 VBMI2 512-bit** (Xeon Granite Rapids / Zen 4+):
  `vpcompressw` partitions 32 × uint16_t in ONE instruction, no
  shuffle table.  `vpexpandb` does the BU `tree_merge` 64 bytes per
  iteration.  Shipped — peak `partition` cost is 0.04 ns/elem
  (24 GB/s), 4-5× faster than Zen 3's SSE path.
- **SVE/SVE2** (Graviton 4 at 128-bit): `svcompact` only handles
  4 × uint32 at this width, requiring widen/narrow — slower than
  NEON TBL.  Disabled.  Would help at 256-bit+ vector lengths (e.g.
  Fujitsu A64FX 512-bit).

Cross-platform numbers are now in the per-host tables above; the
qualitative picture: peak `vs huf0_x2` ratios scale Xeon > M4 ≈
Graviton 4 > Zen 3, tracking primitive width (and `vpcompressw`
availability).  Traditional Huffman gains nothing from wider SIMD —
it's 4 independent scalar dependency chains regardless of vector
width.

## Ideas That Can Make Things Faster

The canonical, full-detail log of optimization ideas (shipped, in
flight, discarded, deferred, with cycle-level analysis) lives in
[`IDEAS.md`](IDEAS.md).  The store-coalescing investigation
(prototyped on M4 / Graviton 4 / Xeon AVX-512, all losing) has its
own write-up in [`docs/COALESCE.md`](docs/COALESCE.md).

This section is a summary.

### Tested and adopted

- **AVX-512 port**: Implemented and tested on Xeon 6975P. `vpcompressw`
  does the partition in ONE instruction (no shuffle table), 32 elements
  per iteration. Beats huf0 on 6 of 12 distributions. See results above.
- **Flat-subtree fast path** (April 2026, commits `a275d05` /
  `0d9ed64` / `7c3238b`): every internal Huffman tree node whose
  subtree is *maximally flat* with depth D ≥ 2 (all 2^D leaves at
  the same relative depth) is detected at `build_table` time.  The
  encoder emits a single N·D-bit packed region for those elements
  instead of D bitmap-per-level.  The decoder reads N·D bits and
  looks up a per-subtree `code_to_sym[local_code]` table — same
  mechanism that already powered the full-tree flat path.  The
  root-flat special case was unified into this same mechanism
  (`0d9ed64`), making the old
  `pivco_huffman_decode_neon_prefix` redundant for flat-tree cases
  (still present as a research backend for non-flat prefix-radix).
  Ported to scalar, NEON, x86 SSE4.1, and AVX-512 backends.
  **Headlines on M4 (pivco_n, M/s):**
  bell_s80 1102 → 2508 (+127%),
  zipfian 1231 → 2405 (+95%),
  bell_s30 1194 → 2007 (+68%),
  proba02 1135 → 1953 (+72%),
  bell_s10 1746 → 2544 (+46%),
  english 2506 → 2649 (+6%),
  uniform 1148 → 4053 (+253%),
  flat_M7 1480 → 5046 (+241%).
  See `results/20260424-204720-0a92fe3-flat-subtree-sweep.md` for the
  full 4-platform sweep; `extras/bench_flat_subtree_stats.c` is the
  analyzer used to predict coverage per distribution.  Format-wise,
  this replaces the older "prefix-radix backend for flat trees" bullet
  below — the flat-subtree path subsumes it.

- **TBL-accelerated flat-subtree decode, per-D**
  (April 2026, commits `b0639ff`..`a77c589` for NEON, `210211d` /
  `ad17cdc` / `fa1134b` for AVX-512, `0e037ab` / `ae14323` for SSE4.1):
  After the format-change above, the decoder's `flat_decode_*`
  helpers were specialised per `D` using per-ISA table-lookup
  primitives (`vqtbl{1,2,4}q_u8` on NEON, `vpshufb` /
  `vpmultishiftqb` / `vpermb` on AVX-512, `pshufb` on SSE4.1).
  *(headline per-ISA gains as of `ae14323`, 2026-04-24; measured on
  M4 Max / c8i / c6a, 20-30 reps × 4M symbols; full sweep pending)*:

  | backend | D's shipped | headline wins |
  |---|---|---|
  | NEON (M4) | D=2..6 | english +10%, bell_s80 +16%, flat_M5/6 +300% (root-flat direct-write) |
  | NEON (Graviton 4) | D=2..4 (D=5/6 fall through to scalar — see below) | english +10%, sparse_4 +619%, sparse_16 +848% |
  | AVX-512 (Xeon) | D=2..6 | english 0.93×→1.03× (first Xeon win), bell_s10 0.86×→1.03× (parity-cross), bell_s80 2.72×→3.03×, flat_M3/M5/M6 +472/322/492% |
  | SSE4.1 (Zen 3) | D=4 | sparse_16 +842%, bell_s10 +7% |

  AVX-512 D=3/5/6 initially regressed and were reverted (commit
  [fa1134b17d951f385270c20b656cdbf6c0f45beb](../) — kept as historical
  record), then re-landed in [3f27e812ba7ab2c9e0998b501475ede86a89637d](../)
  / [7b2fb8d882e6570f1b1e4e226e9c84bf055dbb04](../) after identifying the
  real cause: `memcpy(ptr, src, N)` for non-power-of-2 N compiles to
  split loads that add 2-3 cycles of serial latency.  Loading the next
  natural size (uint64 or __m128i) unconditionally, with a safe-memcpy
  variant for the final chunk to avoid page-boundary overreads, fixed
  all three.

  **Graviton 4 D=5/D=6 fix** (discovered in the
  [20260424-2327 sweep](results/20260424-2327-6762c55-full-sweep.md),
  fixed in [cee2366bb2372cd173e1900db0b5ea99f4c0c65b](../) — see
  [`results/20260425-0126-cee2366-graviton-d56-fix.md`](results/20260425-0126-cee2366-graviton-d56-fix.md)):
  the NEON paths use `vqtbl2q_u8` (D=5, 32-byte table) and `vqtbl4q_u8`
  (D=6, 64-byte table).  On Apple M4 these retire at ~1/cycle; on
  Neoverse-V2 they were measurably slower than the scalar fallback,
  and a 2× `vqtbl1` + blend emulation was slower still.  Resolution:
  build-time gate `PIVCO_NEON_FAST_MULTI_TBL`, defaulting to 1 on
  `__APPLE__` and 0 elsewhere.  When 0 the D=5/D=6 cases fall through
  to `NEON_FLAT_UNPACK_SWITCH`, the same scalar handling already used
  for D=7 / D≥8.  bell_s80 0.56× → 1.24×, flat_M5 0.77× → 1.93×,
  flat_M6 0.91× → 1.59× on c8g.large.

  SSE4.1 D=2/3/5/6 remain skipped because pure SSE4.1 lacks
  `vpmultishiftqb` and variable per-lane byte shifts; enabling AVX2
  (`_mm_srlv_epi16`) on Zen-3-class hosts may unlock D=2/3, logged in
  IDEAS.md §"Revisit SSE4.1".

- **Flat-aware Huffman tree restructurer** (April 2026, commit
  [ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../) — see
  [`results/20260425-0329-ffbfeac-flat-aware-tree.md`](results/20260425-0329-ffbfeac-flat-aware-tree.md)):
  `pivco_huffman_build_table` no longer produces canonical Huffman
  trees; instead it lays out symbols of each length L into greedy
  power-of-2 chunks (largest D first) and assigns codes so that
  same-length leaves are bunched into the largest possible flat-D≥2
  subtrees.  Same code-length multiset = identical compression; tree
  shape minimises the partition-step count along leaf paths
  (provably optimal for D≥2 leaf coverage — see analyzer
  [`extras/bench_flat_optimal.c`](extras/bench_flat_optimal.c)).
  Mechanism: a flat-D≥2 root absorbs the entire partition path
  through its 2^D-leaf subtree, while D=1 stage-fusion only saves the
  partition at the immediate parent; consolidating multiple D=1 sib
  pairs into one D≥2 flat compounds the savings.
  Results: `english` and `proba14` cross from loss to win on M4 and
  Xeon; `proba02` flips on Graviton 4; `bell_s80` +12-19% on the
  three non-M4 platforms; throughout +11-23% on the historical
  loss cluster.

- **Prefix-radix backend (non-flat research path)**
  (`pivco_huffman_neon_prefix.c`): For Huffman tables with
  `min_len < max_len`, a separate prefix-radix partition approach.
  Non-flat case remained slower than `pivco_n` (~1.2–2.0× on M4);
  superseded by the flat-subtree path above for practical purposes.
  Retired 2026-05-14 — moved to `extras/pivco_huffman_neon_prefix.c`.
  Full writeup: [`docs/PREFIX_RADIX.md`](docs/PREFIX_RADIX.md).
- **Combined shuffle table [256][32]**: Stores both right (mask) and
  left (~mask) shuffle patterns contiguously, enabling `ldp q0, q1` on
  ARM (one load-pair vs two separate loads). 5-9% improvement across
  platforms.
- **Stage fusion (leaf detection at parent)**: Before partitioning,
  check if children are leaves. Both-leaves: scatter both symbols
  directly from bitmap bits using branchless `syms[(mask >> k) & 1]`,
  eliminating partition entirely. One-leaf: partition + scatter leaf
  side inline, avoiding recursive call. +10-38% on NEON (M4, Graviton4),
  neutral on AVX-512 (vpcompressw partition is already ~free, so fusion
  was reverted there). Zen3 SSE sees +6-10% on deep trees but ~5%
  regression on shallow trees from extra child-checking overhead in the
  smaller icache.
  Key finding: fusing scatter INTO the partition loop (conditional
  stores interleaved with sequential partition writes) causes massive
  regression from branch misprediction (NEON) and store-buffer
  interference (scalar). Keeping partition and scatter as separate
  phases is essential.
- **Prefill memset**: Before decoding, memset the entire output to the
  most frequent symbol (precomputed in table as `prefill_sym` /
  `prefill_node`). The tree walk skips that leaf via a single
  `node_id == skip_node` check — no scatter, no index loads. +70% on
  proba80 (4.9→8.3 GB/s), +25% on geometric/proba50, neutral on
  moderate/uniform distributions. Skipped for root both-leaves case
  where sequential vst1 stores are already optimal.
- **Half-partition for prefill parent**: At the parent node of the
  prefilled leaf, only partition the non-prefilled side (one TBL + one
  store instead of two). Uses `partition_8_right` or `partition_8_left`
  which skip the unused shuffle. +10% on proba80 (8.3→9.2 GB/s), +12%
  on geometric. Applies at both root and non-root levels.
  Microbenchmark: half-partition = 0.05 ns/elem vs full = 0.07.
- **Root identity partition**: At the root level, indices are [0..N-1],
  so `partition_root_8` generates them in-register via
  `vaddq_u16(base, {0..7})` instead of loading from a pre-filled array.
  Eliminates 16KB identity array initialization. +7% across all
  distributions. For root both-leaves, the sequential vst1 path writes
  `symbols[j]` directly (no scatter) — 25+ GB/s on two-symbol data.

### Tested and discarded

- **Iterative DFS with explicit stack**: Showed ~1% improvement on M4
  despite the function prologue occupying 14% of execution slots. The
  stp/ldp register saves pipeline perfectly with partition work — they
  don't stall the critical path. The OoO engine hides the latency. Not
  worth the code complexity. May matter on in-order cores.
- **Wider NEON partition**: Processed 16 indices via two TBL
  instructions per iteration. 34% regression on M4 — the 10 memory ops
  (6 loads + 4 stores) per iteration saturate the load/store units.
  The OoO engine already overlaps consecutive 8-wide iterations
  effectively. May help on architectures with wider load/store
  throughput.
- **Replace `compress_popcnt` table with `__builtin_popcount`**: Showed
  no improvement (~1% slower consistently on M4). The popcnt table
  shares a cache line with the shuffle table access.
- **SVE backend at 128-bit (Graviton 4)**: `svcompact_u32` only handles
  4 elements per instruction at this width, requiring widen/narrow for
  uint16 — slower than NEON TBL. Disabled by default. Would help at
  256-bit+ SVE (see Worth Exploring).
- **4-way fused partition (neon2)**: At double-internal nodes, read 2
  bits per symbol and partition into 4 groups in one pass, skipping one
  tree level. `partition_8_4way` works correctly, but the DFS
  encode/decode order and scratch management have bugs when multiple
  4-way levels nest. Code archived at `extras/pivco_huffman_neon2.c`.
- **4-way fused partition, reworked (neon2b)**: Rewrote with clean
  scratch management — LL in-place in `indices`, LR/RL/RR packed in
  `tmp` with 8-uint16 safety gaps between groups to absorb `vst1q_u8`
  trailing-zero overflow, two-pass layout (popcount-based count →
  partition). All 20 roundtrips pass. **Slower than neon on every
  distribution on M4 Max**: proba80 9.4→7.6 GB/s (−19%), english
  2.5→1.75 GB/s (−30%), uniform 1.17→0.77 GB/s (−35%), two_sym_eq
  25.6→6.7 GB/s (−74% — no root both-leaves fast path in neon2b). Root
  cause: one 4-way partition of 8 elements costs 4 TBLs (one per output
  group), identical to 2× 2-way. The theoretical wins are only 1 shared
  index `vld` + 1 skipped recursion frame, and the pass-1 popcount scan
  to compute packed offsets burns more than those savings on NEON's
  TBL-bound hot path. The concept pays off only when a single
  instruction compresses wider than the 8-element TBL (AVX-512
  `vpcompressw` → 32). Code archived at `extras/pivco_huffman_neon2b.c`.
- **Fused one-leaf partition+scatter (`fused_1leaf`)**: At non-prefill
  one-leaf nodes, kept the TBL-compacted leaf-side indices in register
  and drained via lane-extract + strb directly, skipping the memory
  round-trip of the separate scatter pass. Tried three dispatch flavours:
  switch/fallthrough (clang emitted a branch tree — proba80 −50%),
  computed-goto real jump table (proba80 −60% from indirect-branch
  mispredicts), and Trick-2 bucketed (chunks sorted by n_left, eight
  straight-line inner loops with compile-time-constant scatter counts).
  Bucketed recovered most of the regression (proba80 −36%, proba50
  −53%) but still slower than baseline because the two-pass design
  re-loads `indices[]` per chunk, whereas the baseline's `scatter_sym`
  amortizes its `vld` over 8 leaf elements. Net: the "save 1 vld+vst
  per chunk" premise was wrong — baseline only pays a fractional vld
  per chunk for scatter. Code archived at
  `extras/pivco_huffman_neon_fused_1leaf.c`; full writeup with cycle
  models and disassembly evidence in
  [`extras/README_FUSED_1LEAF.md`](extras/README_FUSED_1LEAF.md).
- **uint8 level-0 partition**: At level 0, indices are contiguous
  [0..N-1], so within 256-element windows we can partition uint8_t
  positions (16 per TBL) instead of uint16_t indices (8 per TBL), then
  widen back cheaply. Tested two variants:
  - *Split lo/hi with 256-entry uint8 table + add-8*: 25% regression.
    4 TBL per 16 elements (same as 2 TBL per 8 uint16) plus combine
    overhead. No TBL throughput gain.
  - *64K-entry full 16-bit mask table (2MB)*: 1 TBL per 16 elements
    (genuine 2x partition speedup) but massive cache thrashing. Net
    25% regression on proba80 (3333 vs 4488 M/s).
  The widen-convert step and window management overhead outweigh the
  partition speedup. Might revisit if a way to avoid the gather is
  found (e.g. carry uint8 positions through multiple tree levels).

### Worth exploring

- **Single-stage prefix-radix for non-flat trees**: Extended the
  prefix backend to tables where `min_len < max_len` (english, proba14,
  bell, zipfian, …).  Implemented and all 20 roundtrips pass, but
  **currently slower than baseline on every non-flat distribution**
  (−23% to −97%).  Per-phase profiling via `bench_prefix_profile`
  showed phases 2 (histogram) + 4 (bucket) together cost ~3.84 c/elem,
  dominating the whole decode.  Both are serial-data-dependency-limited
  scalar loops — classic radix-sort bottlenecks that need SIMD
  treatment (parallel histograms for phase 2, TBL-based K-way partition
  for phase 4) to break even.  Detailed profile data, diagnosis, and
  optimisation path in [`docs/PREFIX_RADIX.md`](docs/PREFIX_RADIX.md) §5+§6.
- **Nested (multi-stage) prefix-radix**: At each internal node during
  decode, use `M_local = local_min` of that subtree.  An analysis in
  `bench/bench_multi_stage_stats.c` shows this fires on a meaningful
  fraction of elements in several distributions — notably zipfian
  (70% of elements land in subtree bins with local_min ≥ 2 after a
  top-level M=3 radix).  Would stack on top of single-stage once
  that's in.  See [`docs/PREFIX_RADIX.md`](docs/PREFIX_RADIX.md) §4.
- **SVE at 256-bit+**: Untested. `svcompact` on wider SVE (e.g. A64FX
  at 512-bit) would handle 16+ uint32 per instruction, potentially
  matching AVX-512 performance.
- **Leaf cutoff**: For subtrees with < 16 indices, switch to scalar
  table-based decode. Avoids partition overhead on tiny groups deep in
  the tree. The 9.5% scalar scatter remainder suggests many leaf nodes
  have n < 8.
- **Hybrid decoder**: Select PIVCO or traditional per-block based on
  the Huffman table shape (e.g., max code length or entropy). Uniform
  blocks get table lookup, skewed blocks get tree walk.
- **Encode optimization**: The encoder's bitmap construction is still
  scalar. NEON comparison + movemask could help, as could encoding
  multiple blocks in parallel.
- **Computed shuffle (eliminate compress_tab)**: The 8KB compress_tab
  takes 128 cache lines — 6% of M4's L1D, 12% of Graviton4's, 25% of
  Zen3's. Could compute the shuffle pattern on the fly using a NEON
  prefix sum: expand mask bits to 0/1, prefix-add via 3 shift-and-add
  steps (vext + vadd), then convert to byte-pair indices. ~10 NEON
  instructions vs 1 table load (~4 cycles when hot). Per-partition
  ~2.5x slower, but freeing 8KB of L1D could improve overall working
  set pressure, especially on Zen3 where it's 25% of L1D. Worth A/B
  testing on small-cache platforms.

## Building & Running

```sh
git clone --depth 1 https://github.com/cyan4973/FiniteStateEntropy.git ext/fse
git clone --depth 1 https://github.com/rygorous/ryg_rans.git ext/ryg_rans
git clone --depth 1 https://github.com/google/brotli.git ext/brotli  # optional, extras/bench_brotli
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build
./build/pivco_huffman_tests        # run tests
./build/pivco_huffman_bench        # run benchmarks (default 100 repeats)
./build/pivco_huffman_bench 10     # quick run (10 repeats)
./build/pivco_huffman_bench 200    # thorough run (200 repeats)
./build/pivco_huffman_bench --no-fse    # default-recommended config
./build/pivco_huffman_bench --tdbu      # only pivco_n + pivco_bu (TD/BU compare)
```

### Try it on your own data

PIVCO-Huffman is usable as a library — you don't have to adopt our file format
to measure it.  Three ways, easiest first:

**CLI** — `pivcohuf` compresses a file and prints size / ratio / time / bandwidth:
```sh
./build/pivcohuf c  yourfile         # PH   -> yourfile.ph
./build/pivcohuf c -a yourfile       # PHA  (ANS-coded bitmaps; better ratio on skewed data)
./build/pivcohuf d  yourfile.ph      # decompress (auto-detects PH vs PHA)
```

**Example** — `examples/try.c` (CMake target `pivco_try`) compresses one file
with both PH and PHA and reports ratio + encode/decode throughput:
```sh
./build/pivco_try yourfile
#   yourfile (2000000 bytes)   [ratio = in/out, higher = better]
#     ph    6.28x  (2000000 -> 318379)   enc 704 MB/s   dec 5405 MB/s   roundtrip ok
#     pha   8.44x  (2000000 -> 236833)   enc 495 MB/s   dec 3578 MB/s   roundtrip ok
```

**Library** — link `libpivco_huffman.a` and call the buffer API in
[`include/pivcohuf_file.h`](include/pivcohuf_file.h) (no wire-format knowledge
needed):
```c
#include "pivcohuf_file.h"
size_t cap = pivcohuf_compress_bound(in_len);
uint8_t *out = malloc(cap); size_t out_len = cap;
pivcohuf_compress_ex(in, in_len, out, &out_len, /*use_ans=*/1);   // PHA; 0 = PH

size_t usz; pivcohuf_peek_uncompressed_size(out, out_len, &usz);
uint8_t *dec = malloc(usz); size_t dlen = usz;
pivcohuf_decompress(out, out_len, dec, &dlen);                    // auto-detects PH/PHA
```
To embed the codec in *your own* container/framing, use the block primitives in
[`include/pivco_huffman.h`](include/pivco_huffman.h)
(`pivco_huffman_build_table` then `pivco_huffman_encode`/`pivco_huffman_decode`
over `PIVCO_BLOCK_SIZE`-symbol blocks; call `pivco_huffman_set_fse_enabled(1)`
for PHA).

Custom block size:
```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DPIVCO_BLOCK_SIZE=16384"
```

### Interactive tree visualization

[`figures/tree_viz.html`](figures/tree_viz.html) is a self-contained
HTML/JS explorer for Huffman trees with the flat-subtree fast path
overlaid.  Loads the 29 bench distributions from
[`figures/tree_viz_data.js`](figures/tree_viz_data.js) (regenerated
by `./build/pivco_dump_distributions > figures/tree_viz_data.js`),
accepts file/text uploads, and lets you toggle flat-subtree
detection, click flat-roots to (un)flatten for what-if analysis on
ops/leaf and chain-rule entropy totals, and scrub a max-code-length
slider.  Open the file directly in a browser — no build server
required.
