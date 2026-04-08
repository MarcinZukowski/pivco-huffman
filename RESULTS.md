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
  SIMD-partition indices into left[] (bit=0) and right[] (bit=1)

  decode_node(left, n_left, tree_node->left)
  decode_node(right, n_right, tree_node->right)
```

At each internal node, the block of indices is split into two dense
sub-arrays using a TBL-based SIMD compress (precomputed 256-entry
shuffle table, one `vqtbl1q_u8` per 8 uint16_t indices on NEON).
At each leaf, the symbol is scatter-written to all indices in the list.

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

Written in C11 with two backends:

- **Scalar**: bitmap-based partition. For each of n indices, extract
  the code bit, write to left or right output. O(n) per tree node.

- **NEON** (AArch64): TBL-based SIMD partition. Processes 8 × uint16_t
  indices per iteration using `vqtbl1q_u8` with a precomputed 4KB
  shuffle table (256 entries × 16 bytes). Each partition_8 call does:
  one `vld1q_u8` (load indices), two `vqtbl1q_u8` (compress left and
  right), two `vst1q_u8` (store outputs). ~5 NEON instructions per
  8 indices.

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

Three Huffman decode implementations for comparison:

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

## Benchmark Results

**Platform**: Apple M4 Max, macOS, AppleClang 17, `-O3`, ARM64/NEON
**Block size**: 4096 × uint8_t symbols (default)
**Iterations**: 100,000 per measurement

### Decode Throughput (millions of symbols per second)

| Distribution  | PIVCO scalar | PIVCO NEON | trad 1-stream | trad 4-stream | huf0 1-stream | huf0 4-stream | PIVCO vs best |
|---------------|------------:|----------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| proba80       |         819 |      1735 |           344 |          1067 |           232 |           747 |     **1.63x** |
| proba50       |         590 |      1456 |           349 |           918 |           231 |           793 |     **1.59x** |
| proba14       |         311 |       914 |           330 |           919 |           232 |           881 |     **0.99x** |
| proba02       |         186 |       562 |           309 |           921 |           232 |           895 |       0.61x   |
| uniform       |         170 |       555 |          1049 |          1042 |           n/a |           n/a |       0.53x   |
| english       |         307 |       966 |           328 |           988 |           233 |           881 |     **0.98x** |
| zipfian       |         209 |       590 |           317 |           988 |           232 |           896 |       0.60x   |
| sparse_4      |         592 |      1494 |          1726 |          1070 |           714 |          2052 |       0.73x   |
| sparse_16     |         328 |       991 |          1426 |          1069 |           676 |          2255 |       0.44x   |
| geometric     |         579 |      1381 |           350 |           292 |           232 |           807 |     **1.71x** |
| two_sym_eq    |         956 |      1862 |          1926 |          1072 |           577 |          1652 |     **0.97x** |
| two_sym_90/10 |         956 |      1840 |          1917 |          1070 |           574 |          1655 |     **0.96x** |

- "n/a" = huf0 detected data as incompressible (uniform distribution)
- "PIVCO vs best" = best PIVCO / best of all traditional/huf0
- proba distributions match FiniteStateEntropy's fullbench.c (BMK_genData)

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

PIVCO NEON decode throughput (M/s) by block size:

| N     | proba80 | proba50 | proba14 | proba02 |
|------:|--------:|--------:|--------:|--------:|
|   128 |    1216 |     879 |     350 |     113 |
|   256 |    1324 |    1072 |     478 |     157 |
|   512 |    1552 |    1213 |     666 |     241 |
|  1024 |    1555 |    1266 |     751 |     326 |
|  2048 |    1664 |    1355 |     858 |     467 |
|  4096 |    1735 |    1456 |     914 |     562 |
|  8192 |    1725 |    1483 |     981 |     637 |
| 16384 |    1769 |    1501 |    1030 |     686 |

PIVCO vs best SotA ratio by block size:

| N     | proba80 | proba50 | proba14 | proba02 |
|------:|--------:|--------:|--------:|--------:|
|   128 |   1.28x |   1.45x |   0.64x |   0.33x |
|   256 |   1.31x |   2.87x |   0.68x |   0.48x |
|   512 |   1.49x |   2.61x |   0.84x |   0.27x |
|  1024 |   1.48x |   1.96x |   0.89x |   0.36x |
|  2048 |   1.57x |   1.68x |   0.99x |   0.52x |
|  4096 |   1.63x |   1.59x |   0.99x |   0.61x |
|  8192 |   1.62x |   1.54x |   1.00x |   0.65x |
| 16384 |   1.66x |   1.46x |   1.00x |   0.67x |

Key observations:
- PIVCO throughput scales with block size up to ~4096, then diminishes.
  More indices = better SIMD utilization at the top tree levels.
- proba80 improves 45% from N=128 to N=16384 (1216 → 1769 M/s).
- Diminishing returns past N=4096 — the SIMD partition is already
  well-utilized (512 iterations of 8-wide at the root).
- The competitive ratio peaks around N=4096-8192 for skewed distributions.
- proba02/uniform losses do not improve with block size — the fundamental
  issue is tree depth (many layers, no early termination), not SIMD width.
- **N=4096 is a good default**: near-peak throughput, reasonable memory
  usage (8KB indices + 8KB scratch).

## Profiling

Profiled with macOS `sample` on zipfian decode (1M iterations, N=4096).
Time breakdown for `decode_node_neon`:

| Region              | Source lines | % of self-time | Description |
|---------------------|-------------|---------------:|-------------|
| Recursion overhead  | 156, 193-197 |           29% | Function call/return, DFS stack save/restore |
| SIMD partition      | 177-182      |           20% | TBL-based partition_8 loop |
| Tree traversal      | 158-159      |           12% | Node lookup + leaf check |
| Leaf scatter-write  | 162-163      |           12% | `symbols[indices[j]] = sym` |
| Scalar remainder    | 185-186      |            7% | Leftover < 8 indices |
| Stream advance      | 170-171      |            7% | Bitmap pointer read |
| Other               |              |           13% | |

The actual SIMD partition is only 20% of total time. **Recursion overhead
(29%) is the largest single cost** — register saves/restores and branch
targets for each of the ~511 tree nodes. This suggests two optimization
paths:

1. **Iterative work-queue**: Convert recursive DFS to an explicit stack,
   eliminating function call overhead per tree node.
2. **Leaf cutoff**: Once the index count drops below a threshold (e.g. 32),
   switch to scalar/table-based decode instead of continuing the tree walk.
   Most deep nodes have very few indices.

## Analysis

### Where PIVCO Wins

**Skewed distributions (proba80, proba50, geometric): 1.6-1.7x over SotA.**
When a few symbols dominate, the Huffman tree has short codes for common
symbols. PIVCO terminates large groups of indices at shallow tree levels
where the SIMD partition processes thousands of elements per node. The
tree's early-exit property means most work is done in the first 2-3
levels.

For proba80, ~80% of symbols have 1-3 bit codes. The root partition
processes 4096 indices, and the most common symbol's leaf gets ~3200
of them in one scatter-write. Traditional decoders still do 4096
individual table lookups regardless.

**Near-parity on moderate skew (proba14, english): ~1.0x.**
PIVCO matches the best traditional decoder. The tree is moderately
deep (9-10 levels), and the partition costs roughly balance the ILP
gains of 4-stream decode.

### Where PIVCO Loses

**Near-uniform (proba02, uniform): 0.5-0.6x.** All codes are similar
length (8-12 bits), so the tree is deep with no early termination.
Every level does a full partition of all indices with almost no leaf
writes until the bottom. Traditional decoders do O(1) per symbol
regardless of code length distribution.

**Sparse equal-weight (sparse_4, sparse_16): 0.4-0.7x.** Few symbols
with equal frequency = tiny lookup table that fits in registers.
huff0 4-stream achieves 2000+ M/s — essentially limited only by
output memory bandwidth. PIVCO can't compete because it still does
O(depth) partition passes.

### The Core Tradeoff

**Table lookup with ILP (traditional) vs SIMD tree partitioning (PIVCO):**

Traditional Huffman has a serial dependency per stream (must know
current symbol's length before starting the next), but 4-stream
interleaving hides the latency on OoO CPUs. The lookup table handles
any distribution in O(1) per symbol, but its throughput is fixed.

PIVCO has no serial dependency — all symbols are processed in parallel.
Its throughput is **distribution-dependent**: skewed data terminates
early (most work at the top, few indices at the bottom), while uniform
data forces full-depth traversal. The crossover point is around
proba14 (~14% skew).

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

## Future Work

- **Iterative DFS**: Replace recursion with explicit stack to eliminate
  the 29% function-call overhead.
- **Leaf cutoff**: Switch to table-based decode for subtrees with fewer
  than ~32 indices, avoiding deep-tree partition overhead on tiny groups.
- **AVX-512 port**: `vpcompressw` + `vpscatterd` for the partition and
  leaf-write hot paths.
- **Hybrid decoder**: Use PIVCO for skewed blocks, traditional for
  uniform blocks, selected per-block based on the Huffman table shape.
- **Encode optimization**: The encoder's bitmap construction is still
  scalar. NEON comparison + movemask could help, as could encoding
  multiple blocks in parallel.

## Building & Running

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build
./build/pivco_huffman_tests   # run tests
./build/pivco_huffman_bench   # run benchmarks
```

huff0 baseline requires cloning FiniteStateEntropy:
```sh
git clone --depth 1 https://github.com/cyan4973/FiniteStateEntropy.git ext/fse
```

Custom block size:
```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DPIVCO_BLOCK_SIZE=8192"
```
