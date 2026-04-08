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

Flat 2^15 lookup table (64KB). For each symbol: peek 15 bits, table
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
**Block size**: 4096 × uint8_t symbols
**Iterations**: 100,000 per measurement

### Decode Throughput (millions of symbols per second)

| Distribution  | PIVCO scalar | PIVCO NEON | trad 1-stream | trad 4-stream | huf0 1-stream | huf0 4-stream | PIVCO vs best |
|---------------|------------:|----------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| uniform       |         169 |       549 |          1046 |          1038 |           n/a |           n/a |       **0.53x** |
| english       |         306 |       970 |           326 |           986 |           232 |           880 |       **0.98x** |
| zipfian       |         207 |       592 |           316 |           985 |           232 |           894 |       **0.60x** |
| sparse_4      |         587 |      1491 |          1720 |          1066 |           713 |          1959 |       **0.76x** |
| sparse_16     |         324 |       985 |          1421 |          1066 |           709 |          2254 |       **0.44x** |
| geometric     |         573 |      1399 |           348 |           289 |           231 |           806 |       **1.74x** |
| two_sym_eq    |         947 |      1853 |          1922 |          1066 |           577 |          1636 |       **0.96x** |
| two_sym_90/10 |         947 |      1849 |          1912 |          1066 |           575 |          1639 |       **0.97x** |

- "n/a" = huf0 detected data as incompressible (uniform distribution)
- "PIVCO vs best" = best PIVCO / best of all traditional/huf0

### Data Distributions

| Name          | Description                          | Max code len | Encoded size |
|---------------|--------------------------------------|-------------|-------------|
| uniform       | All 256 symbols equally likely       | 8           | 4096 B      |
| english       | English character frequencies        | ~9          | ~2190 B     |
| zipfian       | Zipf(s=1) over 256 symbols           | ~12         | ~3290 B     |
| sparse_4      | Only 4 symbols, equal frequency      | 2           | 1024 B      |
| sparse_16     | Only 16 symbols, equal frequency     | 4           | 2048 B      |
| geometric     | freq[i] = 2^(30-i), steep skew       | 15          | ~1050 B     |
| two_sym_eq    | 2 symbols, 50/50                     | 1           | 512 B       |
| two_sym_90/10 | 2 symbols, 90%/10%                   | ~2          | 512 B       |

### Compression Ratio

PIVCO encoded size matches traditional Huffman within 1-4%, the only
overhead being byte-alignment rounding at each tree node.

## Analysis

### Where PIVCO Wins

**Deep trees (geometric): 1.74x over huf0 4-stream.** The geometric
distribution produces codes up to 15 bits long, creating a deep Huffman
tree with a large lookup table (32K entries for 15-bit codes). This
thrashes L1 cache for table-based decoders. PIVCO doesn't use a table —
it walks the tree, and most symbols terminate at shallow leaves where
the index groups are large and SIMD-friendly.

**Near-parity on skewed distributions (english, two-symbol): 0.96-0.98x.**
PIVCO is within 2-4% of the best traditional decoder on English text
and two-symbol distributions. These have short average code lengths,
meaning most symbols are decoded in the first few tree levels where
the index groups are still large.

### Where PIVCO Loses

**Uniform distribution: 0.53x.** All codes are ~8 bits, so the tree is
8 levels deep with no early termination. Every level does a full partition
of all 4096 indices, with no leaf writes until depth 8. Traditional
decoders do a single table lookup per symbol regardless of distribution.

**Sparse distributions (sparse_16): 0.44x.** Few symbols with equal
frequency = small lookup table (16 entries) that fits entirely in
registers. huff0 4-stream achieves 2254 M/s — essentially memory-bound
on the output write. PIVCO can't compete because it still does O(depth)
partition passes.

**Zipfian: 0.60x.** The 4-stream decoders (both trad and huf0) achieve
~985-894 M/s through ILP. PIVCO gets 592 M/s — faster than any
single-stream decoder (316/232 M/s), but the 4-stream technique is
a stronger optimization here than SIMD partitioning.

### The Core Tradeoff

**Table lookup with ILP (traditional) vs SIMD tree partitioning (PIVCO):**

Traditional Huffman has a serial dependency per stream (must know
current symbol's length before starting the next), but 4-stream
interleaving hides the latency on OoO CPUs. The lookup table handles
any distribution in O(1) per symbol, but cache pressure grows with
table size.

PIVCO has no serial dependency — all symbols are processed in parallel.
But it does O(depth) partition passes, each touching all remaining
indices. The tree structure means early termination of common symbols
(short codes), but uniform distributions get no benefit.

### Key Insight: SIMD Width Matters

PIVCO's inner loop (TBL-based partition) processes 8 × uint16_t indices
per NEON instruction. Wider SIMD would directly improve throughput:

- **AVX2** (256-bit): 16 indices per partition → 2x current
- **AVX-512**: `vpcompressw` does partition in ONE instruction, no
  shuffle table needed. Plus scatter stores for leaf writes.
- **SVE/SVE2** (256-512 bit): similar gains with scalable vectors

Traditional Huffman gains nothing from wider SIMD — it's 4 independent
scalar chains regardless of vector width. PIVCO's advantage scales
with SIMD capabilities.

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
