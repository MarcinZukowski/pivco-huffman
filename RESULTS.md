# PIVCO Huffman: Implementation & Benchmark Results

## TL;DR

SIMD Huffman decoder that walks the tree top-down and partitions the
whole block at each internal node, instead of decoding one symbol at
a time.

**Wins on skewed distributions** where most symbols resolve in 1–3
tree levels. Peak throughput on M4 (proba80): **9.9 GB/s**, 3.5× huf0
X2 and 5.6× huf0 X1. On `two_sym_eq` (single-bit codes): **27 GB/s**,
5× huf0. Ports to AVX-512 VBMI2 (3.1× on Xeon), Graviton 4 NEON
(2.1–7.9×), and SSE4.1 (1.2× on Zen 3, the only platform where PIVCO
just barely beats huf0).

**Loses on high-entropy distributions** — english text, bell curves,
near-uniform — where every symbol traverses 8–12 levels and the
per-node partition cost stacks up. 0.4–0.9× huf0 X2 there.

Encoded size within 1–4% of traditional Huffman (byte-alignment
overhead at each tree node).

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
| proba80       |      9906 |    1466 |    2853 |    1762 | **3.47x** |
| proba50       |      5270 |    1503 |    2826 |    1603 | **1.86x** |
| proba14       |      2399 |    1496 |    2760 |    1605 |   0.87x |
| english       |      2427 |    1420 |    2567 |    1597 |   0.95x |
| geometric     |      4823 |    1450 |    2690 |     686 | **1.79x** |
| bell_s10      |      1744 |    1447 |    2464 |     690 |   0.71x |
| uniform       |      1176 |       0 |       0 |    1639 |   0.72x |
| two_sym_eq    |     26878 |    3474 |    5320 |    1656 | **5.05x** |

### Intel Xeon 6975P Granite Rapids (AVX-512 VBMI2, 48KB L1D, block 8192)

| Distribution  | PIVCO AVX512 | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|------------:|--------:|--------:|--------:|--------:|
| proba80       |        5540 |    1062 |    1819 |     722 | **3.05x** |
| proba50       |        2703 |    1065 |    1811 |     652 | **1.49x** |
| proba14       |        1017 |    1066 |    1746 |     651 |   0.58x |
| english       |        1267 |    1060 |    1754 |     681 |   0.72x |
| geometric     |        2550 |    1064 |    1815 |     273 | **1.40x** |
| bell_s10      |         714 |    1052 |    1614 |     273 |   0.44x |
| uniform       |         282 |       0 |       0 |     701 |   0.40x |
| two_sym_eq    |        4522 |    1064 |    1829 |     724 | **2.47x** |

### AWS Graviton 4 Neoverse V2 (NEON, 64KB L1D, block 8192)

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      3647 |     938 |    1680 |    1020 | **2.17x** |
| proba50       |      1845 |     933 |    1683 |     837 | **1.10x** |
| proba14       |       875 |     934 |    1635 |     808 |   0.54x |
| english       |       925 |     931 |    1639 |     889 |   0.56x |
| geometric     |      1830 |     933 |    1680 |     231 | **1.09x** |
| bell_s10      |       682 |     925 |    1517 |     231 |   0.45x |
| uniform       |       460 |       0 |       0 |     953 |   0.48x |
| two_sym_eq    |     14161 |     936 |    1685 |    1021 | **8.41x** |

### AMD EPYC 7R13 Zen 3 (SSE4.1, 32KB L1D, block 4096)

| Distribution  | PIVCO SSE | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      1985 |     905 |    1669 |     864 | **1.19x** |
| proba50       |      1241 |     959 |    1741 |     695 |   0.71x |
| proba14       |       631 |     961 |    1674 |     692 |   0.38x |
| english       |       675 |     956 |    1684 |     759 |   0.40x |
| geometric     |      1220 |     959 |    1737 |     171 |   0.70x |
| bell_s10      |       489 |     950 |    1547 |     171 |   0.32x |
| uniform       |       310 |       0 |       0 |     814 |   0.38x |
| two_sym_eq    |      1495 |     963 |    1769 |     873 |   0.85x |

### Cross-Platform Summary (PIVCO SIMD vs best other decoder)

| Distribution | M4 NEON | Xeon AVX-512 | Graviton4 NEON | Zen3 SSE |
|---|---:|---:|---:|---:|
| **proba80** | **3.47x** | **3.05x** | **2.14x** | **1.19x** |
| **proba50** | **1.86x** | **1.49x** | **1.11x** | 0.71x |
| proba14 | 0.87x | 0.58x | 0.53x | 0.38x |
| english | 0.95x | 0.72x | 0.56x | 0.40x |
| **geometric** | **1.79x** | **1.40x** | **1.09x** | 0.70x |
| bell_s10 | 0.71x | 0.44x | 0.45x | 0.32x |
| **two_sym_eq** | **5.05x** | **2.47x** | **7.94x** | 0.85x |

PIVCO now beats huf0 X2 on all four platforms for proba80. On M4:
3.5x proba80, 5x two_sym_eq. On AVX-512: 3.1x proba80, 2.5x
two_sym_eq (prefill + skip_node doubled throughput). On Graviton 4:
2.1x proba80, 7.9x two_sym_eq. On Zen 3: 1.2x proba80 (first win
on this platform). Bell curves remain the hardest case across all
platforms.

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

**Skewed distributions on all platforms: 1.2-5x over huf0.**
Prefill memset + skip_node + half-partition benefit all backends.
On M4: 3.5x proba80 (9.9 GB/s), 5x two_sym_eq (27 GB/s). On
AVX-512: 3.1x proba80 (5.5 GB/s) — prefill doubled throughput by
skipping the scatter on vpcompressw-partitioned data. On Graviton 4:
2.1x proba80, 7.9x two_sym_eq. On Zen 3: 1.2x proba80 — first win
on this platform. proba80 now wins on all four platforms.

The tree's early-exit property means most symbols are decoded in
the first 2-3 tree levels via large scatter-writes — the per-symbol
cost drops well below a single table lookup.

### Where PIVCO Loses

**Moderate/deep distributions on small-L1D platforms.** On Zen 3
(32KB L1D), PIVCO wins on proba80 (1.2x) but loses on everything
else. On Graviton 4 (64KB L1D), PIVCO wins on skewed data (2.1x
proba80, 7.9x two_sym_eq) but loses on moderate and deep
distributions where the 8KB shuffle table + index arrays pressure
L1D.

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

### Tested and adopted

- **AVX-512 port**: Implemented and tested on Xeon 6975P. `vpcompressw`
  does the partition in ONE instruction (no shuffle table), 32 elements
  per iteration. Beats huf0 on 6 of 12 distributions. See results above.
- **Prefix-radix backend for flat trees** (`pivco_huffman_neon_prefix.c`):
  For Huffman tables with `min_len == max_len` (flat trees: uniform,
  sparse_*, two_sym_*), bypass the tree walk entirely and emit the first
  `M = min_len` bits of every element's code as a packed per-element
  stream.  Decoder unpacks each element's `M`-bit code and does a
  single `code_to_sym[code]` table lookup — no partition, no scatter.
  **Uniform: 1.17 → 3.97 GB/s (+243%)** — now PIVCO's best distribution
  vs. all four alternatives (was PIVCO's worst).  sparse_16: +118%.
  sparse_4: +37%.  two_sym_* regresses (−75%) because neon's root-
  both-leaves fast path at 26 GB/s is hard to beat; a runtime
  `max(pivco_n, pivco_p)` gate handles this cleanly.  Full writeup:
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
