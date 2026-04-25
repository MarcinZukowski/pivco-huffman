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
- [Benchmark Results](#benchmark-results)
  - [Apple M4 Max (NEON)](#apple-m4-max-neon-128kb-l1d-block-8192)
  - [Intel Xeon 6975P-C (AVX-512 VBMI2)](#intel-xeon-6975p-c-avx-512-vbmi2-48kb-l1d-block-8192)
  - [AWS Graviton 4 (NEON)](#aws-graviton-4-neoverse-v2-neon-64kb-l1d-block-8192)
  - [AMD EPYC 7R13 (Zen 3 SSE4.1)](#amd-epyc-7r13-zen-3-sse41-32kb-l1d-block-4096)
  - [Cross-Platform Summary](#cross-platform-summary-pivco-simd-vs-best-other-decoder)
  - [Data Distributions](#data-distributions)
  - [Compression Ratio](#compression-ratio)
  - [Block Size Sweep](#block-size-sweep)
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

**Wins everywhere measured on Apple M4** — 1.06× to 10.1× huf0/trad_4s
across the full bench grid.  Concrete:
- `two_sym_eq` single-bit codes: **27 GB/s**, 5× huf0.
- `proba80` strongly skewed: **9.6 GB/s**, 3.3× huf0.
- `uniform` fully flat: **4.2 GB/s**, 2.6× huf0 (via flat-tree path).
- `bell_s80` / `bell_s30` / `zipfian` / `proba02` / `proba14` /
  `english` — the **historically losing** moderate-entropy distributions
  — now all win 1.06× to 1.78× huf0 thanks to the flat-subtree path
  plus the flat-aware tree restructurer.

**Portable to other ISAs**: AVX-512 VBMI2 (1.07–13× on Xeon),
Graviton 4 NEON (1.17–9.60× on 14/19 distributions), SSE4.1
(1.15–12.66× on 8/19 distributions).

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

**PIVCO-Huffman decode** processes all N symbols of the block in
parallel, with per-node dispatch:

```
decode_node(indices[], n, tree_node):
  if leaf:
    write tree_node->symbol to all n indices
    return

  if tree_node is a flat-subtree root (D >= 2):
    read n*D packed bits from stream
    for each element: symbols[indices[i]] = code_to_sym[D-bit code]
    return                              # no recursion below this

  read n code bits (bitmap) from stream

  if both children are leaves:           # stage fusion (D=1 case)
    scatter sym_left/sym_right based on code bits — no partition
    return
  if prefill leaf is one side:           # half-partition
    compact only the other side, recurse
    return
  SIMD-partition indices into left[] (bit=0) and right[] (bit=1)
  decode_node(left,  n_left,  tree_node->left)    # leaf child handled
  decode_node(right, n_right, tree_node->right)   # at entry, not here
```

At each mixed-depth internal node, the block of indices is split into
two dense sub-arrays using a TBL-based SIMD compress (precomputed
256-entry shuffle table, one `vqtbl1q_u8` per 8 uint16_t indices on
NEON).  At each leaf, the symbol is scatter-written to all indices in
the list.  At each flat-subtree root, the whole subtree is collapsed
into a single packed-bit region — the decoder performs `n` D-bit
extracts + `n` table lookups + `n` scalar byte stores, with no
recursion below that node.

**Stage fusion and flat-subtree are the same mechanism at different
depths.**  `scatter_both_leaves` is exactly the D=1 case of the
flat-subtree path (one bit per element, 2-entry inline `syms[]`
lookup).  Flat-subtree at D ≥ 2 generalises it to deeper regions of
the tree using a per-subtree `code_to_sym` table.  Both replace D
levels of partition-and-recurse with a single packed-bit scatter.

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
At each internal tree-walk node with `n` active symbols, `ceil(n/8)`
bytes of bitmap (one bit per element) are stored.  At each flat-subtree
root with `n` active symbols and depth `D`, a single
`ceil(n·D/8)`-byte packed region is stored — one `D`-bit code per
element, no per-level framing.

The decoder has the Huffman tree, so it knows which path each node
uses and exactly how many bytes to consume.  No continuation bitmaps
or stream-level metadata are needed — the Huffman tree structure is
sufficient.

Encoded size equals traditional Huffman (sum of code lengths) plus
byte-alignment rounding, which is typically 1-4% overhead.  The
flat-subtree format is marginally tighter than bitmap-per-level on
flat-heavy regions (one tail padding for the whole packed region vs
`D` per-level paddings).

## Implementation

Written in C11 with four backends.  Each backend implements both the
SIMD tree-walk partition and the flat-subtree packed-bit fast path;
detection of which path applies at which node is shared in
`src/huffman_table.c` (`pivco_huffman_build_table`).

- **Scalar**: bitmap-based partition. For each of n indices, extract
  the code bit, write to left or right output. O(n) per tree node.
  Flat-subtree decode unpacks `D`-bit codes via a scalar shift chain
  specialised for `D ∈ {2,3,4,5,6,7,8}`.

- **NEON** (AArch64): TBL-based SIMD partition. Processes 8 × uint16_t
  indices per iteration using `vqtbl1q_u8` with a combined 8KB shuffle
  table (256 entries × 32 bytes: right + left patterns contiguous,
  loaded with a single `ldp q0, q1`). 12 instructions per 8 indices.
  Flat-subtree decode uses the same scalar D-bit unpacker as the
  reference path; fast paths for `D ∈ {2,3,4,5,6,7,8}` one byte at a
  time.

- **SSE4.1** (x86-64): `pshufb`-based partition, same 8-wide approach
  as NEON with the combined shuffle table.  Flat-subtree decode
  mirrors the NEON scalar unpacker.

- **AVX-512 VBMI2** (x86-64): `vpcompressw` partition — 32 × uint16_t
  per iteration in a single instruction. No shuffle table needed.
  Available on Intel Granite Rapids (Xeon 6000P) and later.  The
  flat-subtree D-bit unpack is currently scalar; a VBMI2
  `vpmultishiftqb` implementation is on the shortlist (see IDEAS.md).

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

3. **In-place left partition**: The left (bit=0) partition is written
   back into the input indices array. Safe because output position <=
   input position (n_left <= j). The right (bit=1) partition goes to
   a scratch buffer. Scratch space usage is O(N) total.

4. **DFS-ordered encoding**: The encoder walks the tree in the same
   DFS order as the decoder, emitting code bits at each internal node.
   This makes the format self-describing given the Huffman tree — no
   per-node headers or metadata.

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

## Benchmark Results

**Methodology**: Decode a 4M-symbol sequence, repeated 25 times per
timed run (100M symbols/run). 5 runs, drop 2 slowest, report median
of 3 best. Each codec uses its natural block size: PIVCO/trad use
4096-8192 symbol blocks, huf0 uses 128KB chunks, rANS decodes full 4M.

Baselines include huf0 X1 (single-symbol lookup) and X2 (double-symbol
lookup). "vs best" = best PIVCO / best of all other decoders.

Full detailed results with system info in `results/` directory.

### Apple M4 Max (NEON, 128KB L1D, block 8192)

*(as of [ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../) commit, 2026-04-25; 30 reps × 4M
symbols, median of 3 of 5 runs.  Full sweep file:
[`results/20260425-0329-ffbfeac-flat-aware-tree.md`](results/20260425-0329-ffbfeac-flat-aware-tree.md))*

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      9556 |    1485 |    2867 |    1706 | **3.33x** |
| proba50       |      5119 |    1472 |    2815 |    1543 | **1.82x** |
| proba14       |      2866 |    1478 |    2705 |    1529 | **1.06x** |
| english       |      3333 |    1449 |    2610 |    1594 | **1.28x** |
| zipfian       |      2666 |    1430 |    1907 |    1580 | **1.40x** |
| geometric     |      5049 |    1439 |    2696 |     696 | **1.87x** |
| bell_s10      |      3204 |    1417 |    2455 |     689 | **1.31x** |
| bell_s30      |      2396 |    1427 |    1497 |     687 | **1.60x** |
| bell_s80      |      2872 |       0 |       0 |    1612 | **1.78x** |
| proba02       |      2588 |    1446 |    1602 |    1526 | **1.62x** |
| uniform       |      4169 |       0 |       0 |    1633 | **2.55x** |
| sparse_4      |     48377 |    3547 |    5281 |    1657 | **9.16x** |
| sparse_16     |     46210 |    3248 |    4581 |    1655 | **10.09x** |
| flat_M3       |     23033 |    3572 |    5445 |    1654 | **4.23x** |
| flat_M5       |     24848 |    3577 |    5211 |    1647 | **4.77x** |
| flat_M6       |     22083 |    3468 |    4509 |    1641 | **4.90x** |
| flat_M7       |      5071 |    3554 |    2781 |    1623 | **1.43x** |
| two_sym_eq    |     26795 |    3548 |    5384 |    1657 | **4.98x** |
| two_sym_90/10 |     26363 |    3549 |    5183 |    1650 | **5.09x** |

### Intel Xeon 6975P-C (AVX-512 VBMI2 + VBMI, 48KB L1D, block 8192)

*(as of [ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../) commit, 2026-04-25; AWS `test-c8i`,
2 vCPU, GCC 11.5.0, Amazon Linux 2023; 30 reps × 4M symbols)*

| Distribution  | PIVCO AVX512 | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|------------:|--------:|--------:|--------:|--------:|
| proba80       |       5684 |    1062 |    1817 |     722 | **3.13x** |
| proba50       |       2712 |    1063 |    1816 |     652 | **1.49x** |
| proba14       |       1876 |    1065 |    1748 |     651 | **1.07x** |
| english       |       2171 |    1059 |    1760 |     681 | **1.23x** |
| zipfian       |       1743 |    1052 |    1263 |     680 | **1.38x** |
| geometric     |       2830 |    1063 |    1816 |     274 | **1.56x** |
| bell_s10      |       1941 |    1051 |    1613 |     273 | **1.20x** |
| bell_s30      |       1396 |    1051 |     985 |     272 | **1.33x** |
| bell_s80      |       2277 |       0 |       0 |     694 | **3.28x** |
| proba02       |       1564 |    1050 |    1040 |     649 | **1.49x** |
| uniform       |       4585 |       0 |       0 |     701 | **6.54x** |
| sparse_4      |      23742 |    1067 |    1831 |     721 | **12.97x** |
| sparse_16     |      19903 |    1068 |    1816 |     720 | **10.96x** |
| flat_M3       |      21665 |    1069 |    1821 |     722 | **11.89x** |
| flat_M5       |      18303 |    1067 |    1806 |     719 | **10.13x** |
| flat_M6       |      16998 |    1063 |    1681 |     714 | **10.11x** |
| flat_M7       |       3779 |    1064 |     921 |     707 | **3.55x** |
| two_sym_eq    |       4590 |    1063 |    1826 |     724 | **2.51x** |
| two_sym_90/10 |       8474 |    1063 |    1809 |     723 | **4.68x** |

### AWS Graviton 4 Neoverse V2 (NEON, 64KB L1D, block 8192)

*(as of [ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../) commit, 2026-04-25; AWS `test-c8g`,
2 vCPU c8g.large pinned `taskset -c 0`, GCC 11.5.0, Amazon Linux 2023; 30 reps × 4M symbols.
D=5/D=6 NEON paths gated off via `PIVCO_NEON_FAST_MULTI_TBL=0`.)*

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      4031 |     939 |    1680 |    1020 | **2.40x** |
| proba50       |      2015 |     932 |    1686 |     837 | **1.20x** |
| proba14       |      1133 |     935 |    1636 |     826 |   0.69x |
| english       |      1200 |     931 |    1640 |     893 |   0.73x |
| zipfian       |       972 |     925 |    1188 |     872 |   0.82x |
| geometric     |      1962 |     933 |    1682 |     229 | **1.17x** |
| bell_s10      |      1279 |     926 |    1518 |     228 |   0.84x |
| bell_s30      |       914 |     925 |     925 |     228 |   0.99x |
| bell_s80      |      1310 |       0 |       0 |     906 | **1.45x** |
| proba02       |       997 |     926 |     979 |     813 | **1.02x** |
| uniform       |      2552 |       0 |       0 |     955 | **2.67x** |
| sparse_4      |     15389 |     939 |    1689 |    1021 | **9.11x** |
| sparse_16     |     15319 |     941 |    1652 |    1009 | **9.27x** |
| flat_M3       |      7012 |     939 |    1686 |    1018 | **4.16x** |
| flat_M5       |      3180 |     939 |    1656 |    1007 | **1.92x** |
| flat_M6       |      2459 |     935 |    1541 |     979 | **1.60x** |
| flat_M7       |      2786 |     935 |     859 |     932 | **2.98x** |
| two_sym_eq    |     16007 |     937 |    1684 |    1022 | **9.50x** |
| two_sym_90/10 |     16017 |     937 |    1668 |    1022 | **9.60x** |

### AMD EPYC 7R13 Zen 3 (SSE4.1, 32KB L1D, block 4096)

*(as of [ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../) commit, 2026-04-25; AWS `test-c6a`,
2 vCPU, GCC 11.5.0, Amazon Linux 2023; 30 reps × 4M symbols)*

| Distribution  | PIVCO SSE | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      2006 |     959 |    1751 |     863 | **1.15x** |
| proba50       |      1277 |     956 |    1714 |     691 |   0.75x |
| proba14       |       773 |     956 |    1639 |     689 |   0.47x |
| english       |       887 |     951 |    1654 |     759 |   0.54x |
| zipfian       |       741 |     945 |    1188 |     757 |   0.62x |
| geometric     |      1282 |     954 |    1708 |     168 |   0.75x |
| bell_s10      |       871 |     947 |    1514 |     168 |   0.58x |
| bell_s30      |       647 |     943 |     932 |     168 |   0.69x |
| bell_s80      |       923 |       0 |       0 |     773 | **1.20x** |
| proba02       |       710 |     945 |     987 |     688 |   0.72x |
| uniform       |      1768 |       0 |       0 |     818 | **2.16x** |
| sparse_4      |      2695 |     961 |    1741 |     873 | **1.55x** |
| sparse_16     |     21932 |     963 |    1732 |     864 | **12.66x** |
| flat_M3       |      2444 |     962 |    1739 |     867 | **1.41x** |
| flat_M5       |      2760 |     961 |    1713 |     859 | **1.61x** |
| flat_M6       |      2157 |     959 |    1638 |     846 | **1.32x** |
| flat_M7       |      2274 |     958 |     869 |     844 | **2.38x** |
| two_sym_eq    |      1493 |     959 |    1740 |     874 |   0.86x |
| two_sym_90/10 |      1492 |     959 |    1753 |     872 |   0.85x |

### Cross-Platform Summary (PIVCO SIMD vs best other decoder)

*(as of [ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../) commit, 2026-04-25; see per-platform
sections above for raw M/s)*

| Distribution | M4 NEON | Xeon AVX-512 | Graviton4 NEON | Zen3 SSE |
|---|---:|---:|---:|---:|
| **two_sym_90/10** | **5.09x** | **4.68x** | **9.60x** | 0.85x |
| **two_sym_eq** | **4.98x** | **2.51x** | **9.50x** | 0.86x |
| **proba80** | **3.33x** | **3.13x** | **2.40x** | **1.15x** |
| **proba50** | **1.82x** | **1.49x** | **1.20x** | 0.75x |
| **proba14** † | **1.06x** | **1.07x** | 0.69x | 0.47x |
| **english** † | **1.28x** | **1.23x** | 0.73x | 0.54x |
| **zipfian** | **1.40x** | **1.38x** | 0.82x | 0.62x |
| **geometric** | **1.87x** | **1.56x** | **1.17x** | 0.75x |
| **bell_s10** | **1.31x** | **1.20x** | 0.84x | 0.58x |
| **bell_s30** | **1.60x** | **1.33x** | 0.99x | 0.69x |
| **bell_s80** | **1.78x** | **3.28x** | **1.45x** | **1.20x** |
| **proba02** † | **1.62x** | **1.49x** | **1.02x** | 0.72x |
| **uniform** | **2.55x** | **6.54x** | **2.67x** | **2.16x** |
| **sparse_4** | **9.16x** | **12.97x** | **9.11x** | **1.55x** |
| **sparse_16** | **10.09x** | **10.96x** | **9.27x** | **12.66x** |
| **flat_M3** | **4.23x** | **11.89x** | **4.16x** | **1.41x** |
| **flat_M5** | **4.77x** | **10.13x** | **1.92x** | **1.61x** |
| **flat_M6** | **4.90x** | **10.11x** | **1.60x** | **1.32x** |
| **flat_M7** | **1.43x** | **3.55x** | **2.98x** | **2.38x** |

† New parity-cross from the flat-aware tree restructurer
([ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../)): `proba14` flips on
M4 (0.91× → 1.06×) and Xeon (0.67× → 1.07×); `proba02` flips on
Graviton 4 (0.92× → 1.02×); `english` lifts from 1.10× to 1.28× on M4
and from 1.00× to 1.23× on Xeon.

**Post-flat-subtree (April 2026):**  The flat-subtree fast path
(flat regions of the tree emit one N·D-bit packed region instead of
D bitmap levels) flipped the historical loss cluster — `bell_*`,
`proba02`, `zipfian`, `english` — from 0.44–0.98× into 0.48–2.76×,
winning against huf0/trad_4s on M4 and Xeon AVX-512 for the full set,
and on Graviton 4 for several.  Flat-tree synthetics (`uniform` / 
`sparse_*` / `flat_M*`) also moved sharply up because the unified
flat-subtree path bypasses the `indices[]` materialisation and prefill
memset that the old dedicated prefix backend still paid.

Platform coverage of wins after flat-subtree + flat-aware tree
restructurer
([ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../)):
- **Apple M4 Max (NEON)**: 18/19 wins (proba14 only at 0.91× before;
  the flat-aware tree pulled it to 1.06× as the second-to-last
  win).  All other distributions ≥ 1.06×.
- **Intel Xeon 6975P-C (AVX-512)**: 17/19 wins.  Remaining losses are
  proba14 (0.67× → 1.07×, parity-flipped) joined by english (1.00× →
  1.23×, parity-flipped); only proba14-derivative cases stay below
  parity.
- **AWS Graviton 4 (NEON)**: 14/19 wins.  proba02 newly flips to
  1.02× from the flat-aware tree; bell_s30 lifts to 0.99× (parity).
  Moderate-entropy losses on `english` / `zipfian` / `bell_s10` /
  `proba14` linger at 0.69–0.84× — Graviton 4's `tbl` per-cycle
  throughput is below M4's, so partition cost outside flat subtrees
  dominates.
- **AMD EPYC 7R13 (Zen 3 SSE4.1)**: 8/19 wins.  flat_* / sparse_* /
  uniform / proba80 / bell_s80 all win cleanly; moderate-entropy
  cluster lifted 12-16% by the flat-aware tree but still below parity
  (Zen 3 has fewer shuffle ports — per-cycle partition cost already
  smaller, so flat-subtree's absolute savings are smaller too).

### Data Distributions

| Name          | Description                                       | Max code len |
|---------------|---------------------------------------------------|-------------|
| proba80       | FSE bench: each symbol gets 80% of remaining mass | ~3          |
| proba50       | FSE bench: 50% of remaining mass                  | ~6          |
| proba14       | FSE bench: 14% of remaining mass                  | ~10         |
| proba02       | FSE bench: 2% of remaining mass (near-uniform)    | ~12         |
| uniform       | All 256 symbols equally likely                     | 8           |
| english       | English character frequencies                      | ~9          |
| zipfian       | Zipf(s=1) over 256 symbols                         | ~12         |
| sparse_4      | Only 4 symbols, equal frequency                    | 2           |
| sparse_16     | Only 16 symbols, equal frequency                   | 4           |
| geometric     | freq[i] = 2^(30-i), steep skew                     | 15          |
| two_sym_eq    | 2 symbols, 50/50                                   | 1           |
| two_sym_90/10 | 2 symbols, 90%/10%                                 | ~2          |

The proba distributions are generated identically to FiniteStateEntropy's
benchmark: a 2048-entry table where each symbol gets p% of remaining
entries, then symbols are sampled uniformly from the table.

### Compression Ratio

PIVCO encoded size matches traditional Huffman within 1-4%, the only
overhead being byte-alignment rounding at each tree node.

### Block Size Sweep

*(as of a pre-flat-subtree commit; numbers are stale for flat-heavy
distributions — the N-dependence for stick-tree-shaped distributions
(proba80/50) is still a good reference.  TODO: re-sweep post-`7c3238b`.)*

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

## Profiling

*(as of a pre-flat-subtree commit; measures the 2-way partition path only
— on moderate-entropy / flat-heavy distributions the flat-subtree fast
path now bypasses most of this.  Still representative for stick-tree
shapes where flat-subtree doesn't fire.)*

Profiled with macOS `sample` on zipfian decode (10M iterations, N=4096).
Self-time extracted from 25K weighted leaf samples in the call tree.

**After NEON scatter optimization:**

| Region                          | % of self-time | Description |
|---------------------------------|---------------:|-------------|
| SIMD partition (TBL core)       |          44.4% | `partition_8` — the actual useful work |
| Function prologue (reg saves)   |          14.1% | 6 stp instructions per recursive call |
| Leaf scatter NEON (vld+lane+strb) |        12.3% | Bulk-load 8 indices, 8 scalar stores |
| Leaf scatter scalar remainder   |           9.5% | Tail loop for n % 8 != 0 |
| n==0 + leaf check               |           6.5% | Per-node early-exit tests |
| Partition loop bookkeeping      |           4.1% | Loop control, n_left/n_right updates |
| Partition scalar remainder      |           3.5% | Leftover < 8 indices per partition |
| Recursive calls                 |           2.2% | Argument setup (mov x0..x6, bl) |
| Stream advance                  |           2.0% | Bitmap pointer read |
| Other                           |           1.4% | |

The SIMD partition dominates at 44% — this is the core work and
expected to be the largest cost. The function prologue at 14% looks
like a target, but **an iterative DFS with explicit stack showed no
measurable improvement** (~1% within noise on M4). The stp/ldp
instructions pipeline perfectly with the partition work — they occupy
execution slots but don't stall the critical path. The M4's deep OoO
window hides the latency completely.

**Profiling lesson**: "occupies 14% of execution slots" is not the same
as "removing it would be 14% faster." On an OoO core, non-critical-path
work is essentially free if it doesn't compete for the bottleneck
resource (in this case, the NEON execution units doing TBL shuffles).

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

**proba14 still loses on Graviton 4 / Zen 3** (0.69× / 0.47×).  The
flat-aware tree restructurer pulled `proba14` from 0.91× to 1.06× on
M4 and from 0.67× to 1.07× on Xeon (commit
[ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../)) — the rule "max-
chunk per length" lifted leaf coverage 33%→83% and freq coverage
0.9%→54.7%.  The remaining gap on Graviton / Zen 3 is residual
partition-cost: even with optimal tree shape, those uarchs partition
slower than M4/Xeon per element.

**Graviton 4 moderate-entropy cluster** (english 0.73×, zipfian
0.82×, bell_s10 0.84×).  Graviton 4 `tbl` throughput is measurably
below M4's; the partition cost outside flat subtrees dominates.
D=5/D=6 NEON paths fall through to scalar on this uarch
([cee2366bb2372cd173e1900db0b5ea99f4c0c65b](../)) since the multi-
register `vqtbl{2,4}q_u8` instructions retire too slowly to beat the
scalar switch — see the "Graviton 4 D=5/D=6 fix" callout above.

**Zen 3 SSE4.1 moderate-entropy cluster** (0.47-0.75× across
proba14/proba02/english/zipfian/bell_*).  Zen 3 has fewer shuffle
execution ports than Xeon / M4 / Graviton, so the per-cycle
partition cost is already smaller — making flat-subtree's absolute
savings smaller in turn.  The flat-aware tree restructurer
([ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../)) moved Zen 3
moderate-entropy numbers up another 12-16% but didn't flip them yet.

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

PIVCO's inner loop (TBL-based partition) processes 8 × uint16_t indices
per NEON instruction. Wider SIMD would directly improve throughput:

- **AVX2** (256-bit): 16 indices per partition → ~2x current
- **AVX-512**: `vpcompressw` does the partition in ONE instruction —
  no shuffle table needed. Plus `vpscatterd` for leaf writes.
- **SVE/SVE2** (256-512 bit): similar gains with scalable vectors

Traditional Huffman gains nothing from wider SIMD — it's 4 independent
scalar dependency chains regardless of vector width. PIVCO's advantage
scales directly with SIMD capabilities, suggesting that on AVX-512
hardware the crossover point would shift further toward uniform
distributions.

## Ideas That Can Make Things Faster

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
  Still compiled on NEON as the `pivco_p` benchmark column for
  research comparison.  Full writeup:
  [`PREFIX_RADIX.md`](PREFIX_RADIX.md).
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
  optimisation path in [`PREFIX_RADIX.md`](PREFIX_RADIX.md) §5+§6.
- **Nested (multi-stage) prefix-radix**: At each internal node during
  decode, use `M_local = local_min` of that subtree.  An analysis in
  `bench/bench_multi_stage_stats.c` shows this fires on a meaningful
  fraction of elements in several distributions — notably zipfian
  (70% of elements land in subtree bins with local_min ≥ 2 after a
  top-level M=3 radix).  Would stack on top of single-stage once
  that's in.  See [`PREFIX_RADIX.md`](PREFIX_RADIX.md) §4.
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
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build
./build/pivco_huffman_tests        # run tests
./build/pivco_huffman_bench        # run benchmarks (default 100 repeats)
./build/pivco_huffman_bench 10     # quick run (10 repeats)
./build/pivco_huffman_bench 200    # thorough run (200 repeats)
```

Custom block size:
```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DPIVCO_BLOCK_SIZE=16384"
```
