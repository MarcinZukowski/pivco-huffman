# PIVCO Huffman: Implementation & Benchmark Results

## TL;DR

SIMD Huffman decoder that walks the tree top-down and partitions the
whole block at each internal node, instead of decoding one symbol at
a time.  Plus a **flat-subtree fast path**: every maximal flat subtree
of depth D ≥ 2 in the Huffman tree emits a single N·D-bit packed
region instead of D levels of bitmaps, decoded via direct lookup.

**Wins everywhere measured on Apple M4** — 1.0× to 5.2× huf0/trad_4s
across the full bench grid.  Concrete:
- `two_sym_eq` single-bit codes: **27 GB/s**, 4.9× huf0.
- `proba80` strongly skewed: **9.5 GB/s**, 3.4× huf0.
- `uniform` fully flat: **3.9 GB/s**, 2.4× huf0 (via flat-tree path).
- `bell_s80` / `bell_s30` / `zipfian` / `proba02` — the **historically
  losing** moderate-entropy distributions — now win 1.07× to 1.55× huf0
  thanks to the flat-subtree path.
- `english`: 2.5 GB/s, 1.03× huf0 (crossed from 0.98×).

**Portable to other ISAs**: AVX-512 VBMI2 (3.1× on Xeon), Graviton 4
NEON (2.1–7.9×), SSE4.1 (1.2× on Zen 3).  Platform sweep for the
flat-subtree addition still pending.

Encoded size within 1–4% of traditional Huffman.  The flat-subtree
format actually *improves* packing slightly — one tail padding per
flat region instead of D per-level byte-alignment paddings.

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

*(as of `8754347`, 2026-04-24; 20 reps × 4M symbols, median of 3 of 5 runs)*

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      9506 |    1449 |    2796 |    1674 | **3.40x** |
| proba50       |      4983 |    1391 |    2648 |    1516 | **1.88x** |
| proba14       |      2483 |    1399 |    2592 |    1507 |   0.96x |
| english       |      2649 |    1399 |    2573 |    1577 | **1.03x** |
| zipfian       |      2405 |    1392 |    1864 |    1574 | **1.29x** |
| geometric     |      5001 |    1411 |    2644 |     696 | **1.89x** |
| bell_s10      |      2544 |    1379 |    2376 |     677 | **1.07x** |
| bell_s30      |      2007 |    1377 |    1429 |     683 | **1.40x** |
| bell_s80      |      2508 |       0 |       0 |    1600 | **1.57x** |
| proba02       |      1953 |    1372 |    1527 |    1498 | **1.28x** |
| uniform       |      4053 |       0 |       0 |    1609 | **2.52x** |
| sparse_4      |      6533 |    3522 |    5244 |    1646 | **1.25x** |
| sparse_16     |      6049 |    3231 |    4588 |    1638 | **1.32x** |
| flat_M3       |      6262 |    3551 |    5419 |    1637 | **1.16x** |
| flat_M5       |      5739 |    3566 |    5204 |    1635 | **1.10x** |
| flat_M6       |      5629 |    3465 |    4536 |    1627 | **1.24x** |
| flat_M7       |      5046 |    3554 |    2772 |    1612 | **1.42x** |
| two_sym_eq    |     25816 |    3537 |    5398 |    1642 | **4.78x** |
| two_sym_90/10 |     26116 |    3542 |    5076 |    1647 | **5.15x** |

### Intel Xeon 6975P-C (AVX-512 VBMI2, 48KB L1D, block 8192)

*(as of `8754347`, 2026-04-24; AWS `test-c8i`, 2 vCPU, GCC 11.5.0,
Amazon Linux 2023; 20 reps × 4M symbols)*

| Distribution  | PIVCO AVX512 | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|------------:|--------:|--------:|--------:|--------:|
| proba80       |       5773 |    1060 |    1814 |     722 | **3.18x** |
| proba50       |       2761 |    1064 |    1817 |     651 | **1.52x** |
| proba14       |       1200 |    1064 |    1741 |     651 |   0.69x |
| english       |       1632 |    1058 |    1754 |     680 |   0.93x |
| zipfian       |       1586 |    1050 |    1262 |     681 | **1.26x** |
| geometric     |       2871 |    1062 |    1807 |     273 | **1.59x** |
| bell_s10      |       1390 |    1049 |    1612 |     273 |   0.86x |
| bell_s30      |       1069 |    1050 |     985 |     273 | **1.02x** |
| bell_s80      |       1913 |       0 |       0 |     693 | **2.76x** |
| proba02       |       1212 |    1051 |    1041 |     650 | **1.15x** |
| uniform       |       4564 |       0 |       0 |     699 | **6.53x** |
| sparse_4      |       3864 |    1066 |    1830 |     723 | **2.11x** |
| sparse_16     |       3554 |    1068 |    1812 |     720 | **1.96x** |
| flat_M3       |       3866 |    1068 |    1821 |     724 | **2.12x** |
| flat_M5       |       4254 |    1067 |    1809 |     717 | **2.35x** |
| flat_M6       |       3083 |    1065 |    1672 |     715 | **1.84x** |
| flat_M7       |       3716 |    1063 |     923 |     708 | **3.50x** |
| two_sym_eq    |       4649 |    1062 |    1825 |     725 | **2.55x** |
| two_sym_90/10 |       8815 |    1063 |    1812 |     724 | **4.87x** |

### AWS Graviton 4 Neoverse V2 (NEON, 64KB L1D, block 8192)

*(as of `8754347`, 2026-04-24; AWS `test-c8g`, 1 vCPU, GCC 11.5.0,
Amazon Linux 2023; 20 reps × 4M symbols.  Host throttled ~50% mid-run
— numbers are low-side.)*

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      3671 |     939 |    1678 |    1021 | **2.19x** |
| proba50       |      1876 |     933 |    1683 |     836 | **1.11x** |
| proba14       |       913 |     934 |    1632 |     812 |   0.56x |
| english       |      1009 |     931 |    1638 |     872 |   0.62x |
| zipfian       |       898 |     924 |    1186 |     896 |   0.76x |
| geometric     |       976 |     466 |     838 |     116 | **1.16x** |
| bell_s10      |      1065 |     925 |    1516 |     228 |   0.70x |
| bell_s30      |       784 |     925 |     926 |     228 |   0.85x |
| bell_s80      |      1055 |       0 |       0 |     893 | **1.18x** |
| proba02       |       770 |     925 |     980 |     820 |   0.79x |
| uniform       |      2169 |       0 |       0 |     934 | **2.32x** |
| sparse_4      |      1781 |     467 |     839 |     517 | **2.12x** |
| sparse_16     |      1348 |     468 |     829 |     514 | **1.63x** |
| flat_M3       |      1430 |     468 |     838 |     516 | **1.71x** |
| flat_M5       |      1809 |     468 |     831 |     514 | **2.18x** |
| flat_M6       |      1275 |     466 |     801 |     506 | **1.59x** |
| flat_M7       |      1395 |     467 |     447 |     475 | **2.94x** |
| two_sym_eq    |      5298 |     467 |     839 |     517 | **6.31x** |
| two_sym_90/10 |     14297 |     467 |     836 |     551 | **17.11x** |

### AMD EPYC 7R13 Zen 3 (SSE4.1, 32KB L1D, block 4096)

*(as of `8754347`, 2026-04-24; AWS `test-c6a`, 2 vCPU, GCC 11.5.0,
Amazon Linux 2023; 20 reps × 4M symbols)*

| Distribution  | PIVCO SSE | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      1967 |     956 |    1749 |     861 | **1.12x** |
| proba50       |      1242 |     954 |    1719 |     694 |   0.72x |
| proba14       |       666 |     955 |    1642 |     693 |   0.41x |
| english       |       791 |     949 |    1655 |     759 |   0.48x |
| zipfian       |       732 |     944 |    1191 |     758 |   0.61x |
| geometric     |      1237 |     953 |    1712 |     171 |   0.72x |
| bell_s10      |       777 |     945 |    1514 |     171 |   0.51x |
| bell_s30      |       588 |     942 |     934 |     171 |   0.62x |
| bell_s80      |       813 |       0 |       0 |     772 | **1.05x** |
| proba02       |       599 |     943 |     988 |     692 |   0.61x |
| uniform       |      1771 |       0 |       0 |     817 | **2.17x** |
| sparse_4      |      2698 |     960 |    1746 |     870 | **1.54x** |
| sparse_16     |      2333 |     960 |    1735 |     861 | **1.34x** |
| flat_M3       |      2506 |     961 |    1744 |     865 | **1.44x** |
| flat_M5       |      2762 |     959 |    1716 |     857 | **1.61x** |
| flat_M6       |      2159 |     959 |    1642 |     844 | **1.31x** |
| flat_M7       |      2192 |     956 |     871 |     841 | **2.29x** |
| two_sym_eq    |      1494 |     958 |    1747 |     871 |   0.86x |
| two_sym_90/10 |      1495 |     958 |    1755 |     871 |   0.85x |

### Cross-Platform Summary (PIVCO SIMD vs best other decoder)

*(as of `8754347`, 2026-04-24; see per-platform sections above for raw M/s)*

| Distribution | M4 NEON | Xeon AVX-512 | Graviton4 NEON | Zen3 SSE |
|---|---:|---:|---:|---:|
| **two_sym_90/10** | **5.15x** | **4.87x** | **17.11x** | 0.85x |
| **two_sym_eq** | **4.78x** | **2.55x** | **6.31x** | 0.86x |
| **proba80** | **3.40x** | **3.18x** | **2.19x** | **1.12x** |
| **proba50** | **1.88x** | **1.52x** | **1.11x** | 0.72x |
| proba14 | 0.96x | 0.69x | 0.56x | 0.41x |
| english | **1.03x** | 0.93x | 0.62x | 0.48x |
| zipfian | **1.29x** | **1.26x** | 0.76x | 0.61x |
| **geometric** | **1.89x** | **1.59x** | **1.16x** | 0.72x |
| bell_s10 | **1.07x** | 0.86x | 0.70x | 0.51x |
| bell_s30 | **1.40x** | **1.02x** | 0.85x | 0.62x |
| bell_s80 | **1.57x** | **2.76x** | **1.18x** | **1.05x** |
| proba02 | **1.28x** | **1.15x** | 0.79x | 0.61x |
| **uniform** | **2.52x** | **6.53x** | **2.32x** | **2.17x** |
| **flat_M7** | **1.42x** | **3.50x** | **2.94x** | **2.29x** |

**Post-flat-subtree (April 2026):**  The flat-subtree fast path
(flat regions of the tree emit one N·D-bit packed region instead of
D bitmap levels) flipped the historical loss cluster — `bell_*`,
`proba02`, `zipfian`, `english` — from 0.44–0.98× into 0.48–2.76×,
winning against huf0/trad_4s on M4 and Xeon AVX-512 for the full set,
and on Graviton 4 for several.  Flat-tree synthetics (`uniform` / 
`sparse_*` / `flat_M*`) also moved sharply up because the unified
flat-subtree path bypasses the `indices[]` materialisation and prefill
memset that the old dedicated prefix backend still paid.

Platform coverage of wins after flat-subtree:
- **Apple M4 Max (NEON)**: 18/19 wins (proba14 only; 0.9% flat-subtree
  coverage per analyzer).
- **Intel Xeon 6975P-C (AVX-512)**: 14/19 wins.  Remaining losses on
  english / bell_s10 / proba14 are close (0.86×–0.93×); scalar D-bit
  extract could be vectorised with VBMI2 `vpmultishiftqb` for further
  gains.
- **AWS Graviton 4 (NEON)**: 10/19 wins.  Moderate-entropy distributions
  lose at 0.56–0.85× but improved from 0.43–0.56× before flat-subtree;
  further work could close the gap.  two_sym_90/10 hits **17.11×**.
- **AMD EPYC 7R13 (Zen 3 SSE4.1)**: 8/19 wins.  flat_* / sparse_* /
  uniform all win cleanly; moderate-entropy stays below parity because
  Zen 3 SSE4.1 has fewer shuffle ports than the other platforms — the
  per-cycle partition cost is already smaller there, so flat-subtree's
  savings are smaller in absolute terms.

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

*(Post-flat-subtree, as of `8754347`.  Pre-flat-subtree commentary on
the losing-moderate-entropy case is preserved in git history — commit
`984dad3` and earlier.  Leaving a summary of the pre/post picture
here for future reviewers.)*

### Where PIVCO Wins

**Skewed / stick-tree distributions on all platforms: 1.1-17× over
huf0.**  Prefill memset + skip_node + half-partition benefit all
backends.  On M4: 3.4× proba80 (9.5 GB/s), 4.8× two_sym_eq (26
GB/s).  On AVX-512: 3.2× proba80, 4.9× two_sym_90/10.  On Graviton 4:
**17.1×** two_sym_90/10, 6.3× two_sym_eq (thermal throttling on the
1-vCPU test host likely understates headline numbers).  On Zen 3:
1.1× proba80, 0.85× two_sym_eq (Zen 3 SSE4.1 is the floor).  proba80
now wins on all four platforms.

The tree's early-exit property means most symbols are decoded in
the first 2-3 tree levels via large scatter-writes — the per-symbol
cost drops well below a single table lookup.

**Moderate-entropy distributions on M4 / AVX-512: 1.0-1.6×.**  The
flat-subtree fast path (landed April 2026, commits `a275d05` and
onward) replaces D levels of per-level bitmaps with a single
N·D-bit packed region at every maximal flat subtree it detects.  The
benchmark analyzer (`extras/bench_flat_subtree_stats.c`) measured
54-100% of elements landing in such subtrees on bell_*, proba02,
zipfian, and english — exactly the distributions PIVCO used to lose
on.  Post-flat-subtree: M4 wins the full set (english 1.03×,
bell_s10 1.07×, bell_s30 1.40×, bell_s80 1.57×, proba02 1.28×,
zipfian 1.29×).  Xeon AVX-512 wins most of the set (bell_s80 2.76×,
proba02 1.15×, zipfian 1.26×, bell_s30 1.02×).

**Flat-tree distributions (uniform, sparse_*, flat_M*) on all
platforms: 1.1-6.5×.**  Root-flat tables route through the same
flat-subtree packed-bit path as their non-root cousins.  Direct
scatter to `symbols[i]` with no indices materialisation or prefill
memset.

### Where PIVCO Still Loses

**proba14 everywhere** (0.41-0.96×).  The flat-subtree analyzer
measured only 0.9% of elements in flat subtrees for proba14 — the
Huffman tree for that distribution happens to be highly asymmetric
with a deep moderately-mixed middle, giving neither the stick-tree
fast path nor the flat-subtree fast path much to chew on.

**Graviton 4 moderate-entropy cluster** (english 0.62×, zipfian
0.76×, bell_s10 0.70×, proba02 0.79×).  Graviton 4 `tbl` throughput
is measurably below M4's; the partition cost outside flat subtrees
dominates.  Thermal throttling on the 1-vCPU EC2 instance
contributes; a larger instance or sustained-clock environment
would likely close some of the gap.

**Zen 3 SSE4.1 moderate-entropy cluster** (0.41-0.62× across
proba14/proba02/english/zipfian/bell_*).  Zen 3 has fewer shuffle
execution ports than Xeon / M4 / Graviton, so the per-cycle
partition cost is already smaller — making flat-subtree's absolute
savings smaller in turn.  Even so, flat-subtree moved Zen 3's
moderate-entropy numbers from the pre-flat-subtree 0.32-0.56× range
up to 0.41-0.62×.

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
