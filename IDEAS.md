# PIVCO-Huffman Decode Ideas

## Best candidate: fuse leaf handling into partition

The current decode path in `src/pivco_huffman_neon.c`,
`src/pivco_huffman_x86.c`, and `src/pivco_huffman_avx512.c` always:

1. reads the bitmap,
2. partitions indices into left/right,
3. recurses,
4. eventually scatter-writes at leaves.

This creates avoidable work when one or both children are already leaves.

### Idea

Add decode fast paths for these common cases:

- **left leaf, right internal**
  - scan bitmap once,
  - scatter the left symbol immediately for bit=0 entries,
  - compact only bit=1 indices into `tmp`,
  - recurse only into the right child.

- **left internal, right leaf**
  - symmetric version of the above.

- **both children are leaves**
  - do not partition into scratch at all,
  - scan bitmap once,
  - directly scatter one of two symbols per index,
  - no recursive calls.

### Why this looks promising

From `RESULTS.md` profiling on M4:

- SIMD partition: 44.4%
- Leaf scatter NEON: 12.3%
- Leaf scatter scalar remainder: 9.5%
- Function prologue: 14.1%

This suggests a worthwhile opportunity to remove:

- one full leaf pass,
- scratch traffic for leaf-side partitions,
- recursive calls/checks on shallow subtrees.

This should help most on **skewed distributions**, where many symbols terminate
near the top of the tree and PIVCO already performs best.

## ~~Next best idea: finish `neon2` 4-way fused decode~~ — attempted, didn't pay off

Implemented in `src/pivco_huffman_neon2b.c` with clean scratch management
(LL in-place, LR/RL/RR packed into `tmp` with 8-uint16 gaps to absorb
`vst1q_u8` trailing-zero overflow, two-pass popcount+partition). All 20
roundtrips pass. **Slower than neon on every distribution on M4**
(proba80 −19%, english −30%, uniform −35%, two_sym_eq −74%).

Root cause: on NEON a 4-way partition of 8 elements costs 4 TBLs (one per
output group), identical to 2× 2-way. The only theoretical wins are 1
shared index `vld` and 1 skipped recursion frame — the pass-1 popcount scan
to compute packed offsets costs more than that on the TBL-bound hot path.

Fusion only pays off when one instruction can compress wider than the 8-
element TBL (AVX-512 `vpcompressw` → 32). Not worth further NEON work on
this track. See RESULTS.md for full numbers.

## AVX-512 improvement: better small-node tail

`src/pivco_huffman_avx512.c` does a strong 32-wide partition using
`vpcompressw`, but deeper in the tree it falls back to relatively simple scalar
handling for smaller groups.

### Idea

For the `< 32` remainder, especially 8-wide chunks:

- reuse the existing SSE `pshufb` partition helper, or
- add an explicit 8-wide vector tail instead of scalar loops.

### Why

Deep nodes are small, but they are also common on moderate distributions.
Reducing scalar fallback overhead should improve the AVX-512 backend's worst
cases without disturbing the strong 32-wide fast path.

## Product-level idea: hybrid block decoder

The results strongly suggest that PIVCO wins on skewed distributions and loses
on moderate/uniform ones.

### Idea

Choose decode strategy per block / per table:

- use PIVCO tree-walk for shallow, skewed Huffman tables,
- use `trad_huffman_decode_4s()` for flatter, more uniform tables.

### Why

This is likely a bigger overall throughput win than micro-optimizing PIVCO on
tables where huff0-style decode is fundamentally a better fit.

Possible heuristics:

- `max_len`,
- `min_len`,
- symbol count,
- entropy estimate,
- expected shallow-leaf fraction.

## Ideas probably not worth more time

These already appear explored or unlikely to pay off:

- **Iterative DFS instead of recursion**
  - tested already, essentially noise on M4.

- **Wider NEON partition via two TBLs per 16 indices**
  - already tested, regressed due to load/store pressure.

- **Replacing `compress_popcnt` table with builtin popcount**
  - already tested, slightly worse.

- **SVE at 128-bit width**
  - already slower than NEON on Graviton4.

## Suggested implementation order

1. Add **leaf-child fusion** to `decode_node_neon`.
2. Mirror the same optimization in `decode_node_x86`.
3. Add the same fast paths to `decode_node_avx512`.
4. Benchmark on skewed distributions first (`proba80`, `proba50`, `sparse_2`).
5. Only after that, revisit `neon2` and/or hybrid decode selection.
