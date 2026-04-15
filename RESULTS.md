# PIVCO Huffman: Implementation & Benchmark Results

## What is PIVCO-Huffman?

PIVCO-Huffman applies the PIVoted COlumnar approach to Huffman coding.
Instead of decoding symbols one at a time via table lookup (traditional),
PIVCO processes an entire block of N symbols simultaneously by walking
the Huffman tree in bulk — partitioning the block's index set at each
tree node using SIMD.

**Traditional Huffman decode** processes one symbol at a time:
peek bits → table lookup → emit symbol → consume bits → repeat.
The serial dependency chain (can't start the next symbol until you know
how many bits the current one consumed) limits throughput to ~1 symbol
per 3-4 cycles, even with 4-stream ILP tricks (huff0/zstd).

**PIVCO-Huffman decode** processes all N symbols in parallel:

```
decode_node(indices[], n, tree_node):
  if leaf:
    write tree_node->symbol to all n indices
    return

  read n code bits from stream

  if both children are leaves:           # stage fusion
    scatter sym_left/sym_right based on code bits — no partition
    return
  SIMD-partition indices into left[] (bit=0) and right[] (bit=1)
  if one child is leaf:                  # stage fusion
    scatter leaf symbol inline, recurse non-leaf child only
  else:
    decode_node(left, n_left, tree_node->left)
    decode_node(right, n_right, tree_node->right)
```

At each internal node, the block of indices is split into two dense
sub-arrays using a TBL-based SIMD compress (precomputed 256-entry
shuffle table, one `vqtbl1q_u8` per 8 uint16_t indices on NEON).
At each leaf, the symbol is scatter-written to all indices in the list.

**Stage fusion**: When children are leaves, the partition and scatter
stages are fused. Both-leaves nodes scatter directly from the bitmap
(branchless `syms[(mask >> k) & 1]`), eliminating the TBL shuffle
entirely. One-leaf nodes partition only the non-leaf side and scatter
the leaf symbol inline. This gives +10-38% on NEON platforms.

No accumulation of code bits, no table lookup, no range comparisons.
The tree position tells you the symbol. The inner loop is purely:
load indices → load code bits → shuffle-partition → store.

### Encoded Format

The encoded data is a DFS-ordered bitstream matching the tree walk.
At each internal node with n active symbols, ceil(n/8) bytes of code
bits are stored. The decoder knows the Huffman tree, so it knows
exactly how many bits to read at each node. No continuation bitmaps
or metadata are needed — the Huffman tree structure is sufficient.

Encoded size equals traditional Huffman (sum of code lengths) plus
byte-alignment rounding at each tree node (typically 1-4% overhead).

## Implementation

Written in C11 with four backends:

- **Scalar**: bitmap-based partition. For each of n indices, extract
  the code bit, write to left or right output. O(n) per tree node.

- **NEON** (AArch64): TBL-based SIMD partition. Processes 8 × uint16_t
  indices per iteration using `vqtbl1q_u8` with a combined 8KB shuffle
  table (256 entries × 32 bytes: right + left patterns contiguous,
  loaded with a single `ldp q0, q1`). 12 instructions per 8 indices.

- **SSE4.1** (x86-64): `pshufb`-based partition, same 8-wide approach
  as NEON with the combined shuffle table.

- **AVX-512 VBMI2** (x86-64): `vpcompressw` partition — 32 × uint16_t
  per iteration in a single instruction. No shuffle table needed.
  Available on Intel Granite Rapids (Xeon 6000P) and later.

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

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      4423 |    1448 |    2843 |    1684 | **1.56x** |
| proba50       |      3571 |    1449 |    2740 |    1556 | **1.30x** |
| proba14       |      2248 |    1441 |    2687 |    1513 |   0.84x |
| english       |      2323 |    1428 |    2529 |    1578 |   0.92x |
| geometric     |      3373 |    1413 |    2679 |     681 | **1.26x** |
| bell_s10      |      1729 |    1429 |    2406 |     636 |   0.72x |
| uniform       |      1104 |       0 |       0 |    1622 |   0.68x |
| two_sym_eq    |      5394 |    3329 |    5064 |    1629 | **1.07x** |

### Intel Xeon 6975P Granite Rapids (AVX-512 VBMI2, 48KB L1D, block 8192)

| Distribution  | PIVCO AVX512 | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|------------:|--------:|--------:|--------:|--------:|
| proba80       |        2686 |    1061 |    1815 |     722 | **1.48x** |
| proba50       |        2054 |    1064 |    1812 |     652 | **1.13x** |
| proba14       |         995 |    1064 |    1743 |     651 |   0.57x |
| english       |        1227 |    1059 |    1754 |     682 |   0.70x |
| geometric     |        1955 |    1064 |    1813 |     274 | **1.08x** |
| bell_s10      |         708 |    1051 |    1612 |     274 |   0.44x |
| uniform       |         288 |       0 |       0 |     702 |   0.41x |
| two_sym_eq    |        2899 |    1062 |    1825 |     725 | **1.59x** |

### AWS Graviton 4 Neoverse V2 (NEON, 64KB L1D, block 8192)

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      1441 |     939 |    1679 |    1016 |   0.86x |
| proba50       |      1165 |     933 |    1683 |     841 |   0.69x |
| proba14       |       824 |     936 |    1638 |     837 |   0.50x |
| english       |       874 |     932 |    1641 |     910 |   0.53x |
| geometric     |      1156 |     934 |    1682 |     229 |   0.69x |
| bell_s10      |       641 |     927 |    1522 |     230 |   0.42x |
| uniform       |       445 |       0 |       0 |     977 |   0.46x |
| two_sym_eq    |      2117 |     939 |    1684 |    1021 | **1.26x** |

### AMD EPYC 7R13 Zen 3 (SSE4.1, 32KB L1D, block 4096)

| Distribution  | PIVCO SSE | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      1463 |     959 |    1741 |     864 |   0.84x |
| proba50       |      1122 |     955 |    1703 |     690 |   0.66x |
| proba14       |       677 |     956 |    1633 |     688 |   0.41x |
| english       |       728 |     951 |    1648 |     758 |   0.44x |
| geometric     |      1100 |     955 |    1697 |     168 |   0.65x |
| bell_s10      |       518 |     947 |    1512 |     167 |   0.34x |
| uniform       |       322 |       0 |       0 |     811 |   0.40x |
| two_sym_eq    |      1743 |     959 |    1728 |     873 | **1.01x** |

### Cross-Platform Summary (PIVCO SIMD vs best other decoder)

| Distribution | M4 NEON | Xeon AVX-512 | Graviton4 NEON | Zen3 SSE |
|---|---:|---:|---:|---:|
| **proba80** | **1.56x** | **1.48x** | 0.86x | 0.84x |
| **proba50** | **1.30x** | **1.13x** | 0.69x | 0.66x |
| proba14 | 0.84x | 0.57x | 0.50x | 0.41x |
| english | 0.92x | 0.70x | 0.53x | 0.44x |
| **geometric** | **1.26x** | **1.08x** | 0.69x | 0.65x |
| bell_s10 | 0.72x | 0.44x | 0.42x | 0.34x |
| **two_sym_eq** | **1.07x** | **1.59x** | **1.26x** | **1.01x** |

PIVCO beats huf0 X2 on Apple M4 (large L1D) and Intel AVX-512 (wide
partition) for skewed distributions. With stage fusion, two_sym_eq now
wins on all four platforms. Loses on Graviton 4 (64KB L1D) and AMD
Zen 3 (32KB L1D) for moderate/deep distributions. Bell curves are
the hardest case across all platforms.

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

PIVCO NEON decode throughput (M/s) by block size. Measured with the
4M realistic workload (each block size is recompiled and re-benchmarked):

| N     | proba80 | proba50 | proba14 | proba02 | english | geometric |
|------:|--------:|--------:|--------:|--------:|--------:|----------:|
|  4096 |    4149 |    3158 |    1583 |     645 |    1787 |      3048 |
|  8192 |    4271 |    3277 |    1777 |     851 |    1938 |      3259 |
| 16384 |    4120 |    3345 |    1948 |    1034 |    1953 |      3194 |
| 65536 |    3525 |    2958 |    1818 |    1215 |    1812 |      2895 |

- **N=8192 is the default**: best on highly skewed distributions
  (proba80, geometric) where PIVCO's advantage is largest. 16KB
  index array fits comfortably in M4's 128KB L1D.
- 16384 is slightly better on moderate distributions (proba14,
  proba02) but loses on the most skewed ones.
- 65536 regresses — index arrays start spilling L1.

## Profiling

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

### Where PIVCO Wins

**Skewed distributions on capable hardware: 1.1-1.6x over huf0.**
PIVCO beats huf0 on proba80 and proba50 on both Apple M4 (1.6x)
and Intel AVX-512 (1.5x). On AVX-512, PIVCO also wins on two-symbol
and sparse_4 distributions (1.6x) thanks to the 32-wide `vpcompressw`
partition. With stage fusion, two_sym_eq now wins on all four platforms
(1.01-1.59x) by eliminating the partition entirely for both-leaves nodes.

The tree's early-exit property means most symbols are decoded in
the first 2-3 tree levels via large scatter-writes — the per-symbol
cost drops well below a single table lookup.

### Where PIVCO Loses

**Most distributions on small-L1D / narrow-SIMD platforms.** On
Graviton 4 (64KB L1D) and Zen 3 (32KB L1D), PIVCO loses on moderate
and deep distributions. The 8KB shuffle table + 32KB index arrays
pressure L1D, and 8-wide SIMD doesn't provide enough throughput
advantage over huf0's 4-stream ILP. Exception: two_sym_eq now wins
everywhere thanks to stage fusion eliminating the partition.

**Moderate and uniform distributions everywhere.** On non-skewed data,
the tree is deep with few early terminations, so PIVCO does full-depth
traversal without the benefit of large early leaf writes.

### The Core Tradeoff

PIVCO's throughput is **distribution-dependent** and **hardware-
dependent**: it needs either large L1D (128KB on M4) or wide SIMD
(AVX-512 vpcompressw) to overcome huf0's advantages.

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

Tested and discarded:

- **Iterative DFS with explicit stack**: Tested, showed ~1% improvement
  on M4 despite the function prologue occupying 14% of execution slots.
  The stp/ldp register saves pipeline perfectly with partition work —
  they don't stall the critical path. The OoO engine hides the latency.
  Not worth the code complexity. May matter on in-order cores.

Worth exploring:

- **AVX-512 port**: Implemented and tested on Xeon 6975P. `vpcompressw`
  does the partition in ONE instruction (no shuffle table), 32 elements
  per iteration. Beats huf0 on 6 of 12 distributions. See results above.
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
- **Wider NEON partition**: Tested: process 16 indices via two
  TBL instructions per iteration. 34% regression on M4 — the 10
  memory ops (6 loads + 4 stores) per iteration saturate the
  load/store units. The OoO engine already overlaps consecutive
  8-wide iterations effectively. May help on architectures with
  wider load/store throughput.
- **Replace `compress_popcnt` table with `__builtin_popcount`**: Tested,
  showed no improvement (~1% slower consistently on M4). The popcnt
  table shares a cache line with the shuffle table access.
- **Combined shuffle table [256][32]**: Tested and adopted. Stores both
  right (mask) and left (~mask) shuffle patterns contiguously, enabling
  `ldp q0, q1` on ARM (one load-pair vs two separate loads). 5-9%
  improvement across platforms.
- **Stage fusion (leaf detection at parent)**: Tested and adopted.
  Before partitioning, check if children are leaves. Both-leaves:
  scatter both symbols directly from bitmap bits using branchless
  `syms[(mask >> k) & 1]`, eliminating partition entirely. One-leaf:
  partition + scatter leaf side inline, avoiding recursive call.
  +10-38% on NEON (M4, Graviton4), neutral on AVX-512 (vpcompressw
  partition is already ~free, so fusion was reverted there). Zen3 SSE
  sees +6-10% on deep trees but ~5% regression on shallow trees from
  extra child-checking overhead in the smaller icache.
  Key finding: fusing scatter INTO the partition loop (conditional
  stores interleaved with sequential partition writes) causes massive
  regression from branch misprediction (NEON) and store-buffer
  interference (scalar). Keeping partition and scatter as separate
  phases is essential.
- **SVE backend**: Tested on Graviton 4 (128-bit SVE). `svcompact_u32`
  only handles 4 elements per instruction at this width, requiring
  widen/narrow for uint16 — slower than NEON TBL. Disabled by default.
  Would help at 256-bit+ SVE.
- **4-way fused partition (neon2)**: At double-internal nodes, read 2
  bits per symbol and partition into 4 groups in one pass, skipping one
  tree level. Partition_8_4way works correctly, but the DFS
  encode/decode order and scratch management have bugs when multiple
  4-way levels nest. Code exists as WIP (`pivco_huffman_neon2.c`).
- **uint8 level-0 partition**: At level 0, indices are contiguous
  [0..N-1], so within 256-element windows we can partition uint8_t
  positions (16 per TBL) instead of uint16_t indices (8 per TBL),
  then widen back cheaply. Tested two variants:
  - *Split lo/hi with 256-entry uint8 table + add-8*: 25% regression.
    4 TBL per 16 elements (same as 2 TBL per 8 uint16) plus combine
    overhead. No TBL throughput gain.
  - *64K-entry full 16-bit mask table (2MB)*: 1 TBL per 16 elements
    (genuine 2x partition speedup) but massive cache thrashing. Net
    25% regression on proba80 (3333 vs 4488 M/s).
  The widen-convert step and window management overhead outweigh the
  partition speedup. Might revisit if a way to avoid the gather is
  found (e.g. carry uint8 positions through multiple tree levels).

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
