# Codex Notes: Performance Ideas

> **Last content review:** _NEVER_

These are follow-up ideas after reviewing the current docs and source.  This
file supersedes the older NEON-only note: several items there were written
before the flat-subtree fast path, flat-aware tree assignment, and AVX-512
flat-unpack work landed.

Current baseline:

- M4 NEON is already in good shape and wins the full benchmark grid.
- AVX-512 is close, but still leaves code-shape optimizations on the table.
- Graviton 4 and Zen 3 still lose much of the real-text cluster.
- Store coalescing and 4-way NEON fusion have enough negative evidence; do not
  reopen those unless the target architecture changes materially.

## 1. Port NEON/SSE leaf fusion to AVX-512

This is the most concrete source-level gap I found.

`decode_node_neon()` and `decode_node_x86()` both handle:

- both children leaves: scatter/select symbols directly from the bitmap
- prefilled left child: half-partition only the right side
- prefilled right child: half-partition only the left side

`decode_node_avx512()` currently reads the bitmap, full-partitions every
non-flat internal node, and recurses unconditionally.  That misses the same
stage-fusion and half-partition wins that are already present in the NEON/SSE
paths.

Why it matters:

- one-leaf/prefill nodes avoid one `vpcompressw` side and one store
- both-leaf nodes avoid partitioning entirely
- it should help skewed trees and deep text trees with many leaf-parent nodes

Implementation shape:

- add AVX-512 `scatter_both_leaves` equivalent, initially scalar/extract based
  like `scatter_write_avx512`
- add `partition_32_right` / `partition_32_left` use in `decode_node_avx512`
- mirror NEON's branch structure before the full-partition fallback
- benchmark `proba80`, `proba50`, `geometric`, `english`, `prose_pride`,
  `html_wiki`, and `json_api`

Docs note: README says leaf-child fusion is shipped and neutral on AVX-512, but
the current AVX-512 source does not appear to contain the internal-node fusion
logic.  Either the docs are ahead of the code or the implementation regressed.

## 2. AVX-512 root iota generation

The AVX-512 root path still builds a stack `uint16_t id[32]` in a scalar loop
for every 32-symbol chunk before calling `partition_32_full`.

NEON already avoids this with an in-register identity generator.  AVX-512 can do
the same:

- load a constant zmm `uint16_t iota = [0..31]`
- add broadcast `j`
- compress the generated vector directly

This is cheap, local, and affects every non-flat AVX-512 block.  It should be
done before smaller AVX-512 tail work.

## 3. Hybrid decode selection

This remains high-value, but the old rationale was too broad.

Outdated premise: "PIVCO loses on flatter / high-entropy tables."  After
flat-subtree, PIVCO now wins uniform, sparse, JPEG/gzip-random, and most
high-entropy flat cases.  The current problem is narrower:

- Graviton 4 loses much of the real-text cluster.
- Zen 3 SSE loses most moderate/deep real-text cases.
- AVX-512 has a few deep real-text losses near parity.

So hybrid should be table-shape and backend aware, not a generic "flat tables
use traditional" rule.

Possible heuristic inputs:

- expected partition operations per element
- flat-subtree element coverage, by depth
- fraction of weight under `local_min >= 2`
- `max_len`, `min_len`, and symbol count
- backend family: M4 NEON, Graviton NEON, AVX-512, SSE/AVX2

Practical route:

- keep PIVCO on M4 unless a new dataset shows a loss
- fallback to `trad_huffman_decode_4s()` on Zen/Graviton real-text-like shapes
- test a simple threshold first: low flat coverage plus `max_len >= 10`
- if this becomes a real product format, the block/container needs a codec flag

## 4. Enable AVX2 on non-AVX-512 x86

The current CMake x86 fallback enables only `-msse4.1`, even on Zen 3-class
machines that have AVX2.

AVX2 would not give `vpcompressw`, but it does unlock per-lane variable shifts
for better D-bit flat-subtree unpackers.  The existing SSE path only has a D=4
SIMD flat-unpack fast path; D=2, D=3, D=5, and D=6 fall mostly to scalar.

Suggested scope:

- add compile/runtime detection for AVX2 in the non-AVX-512 path
- keep the existing SSE backend as fallback
- prototype AVX2 unpack for D=2/D=3 first
- benchmark Zen 3 `bell_*`, `proba02`, `english`, `prose_pride`, `flat_M*`

This may not fix Zen 3 real-text alone, but it is a real backend capability
currently unused by the build.

## 5. SSE root both-leaves vectorization

The SSE root both-leaves case is scalar byte-by-byte.  That path dominates
`two_sym_eq` and `two_sym_90/10`, where the root is the whole decode.

Try a small vector path:

- process 16 output symbols from two bitmap bytes
- expand bits to byte masks with `pshufb` or a small lookup table
- blend/b xor two symbols
- store 16 contiguous bytes

This is low-risk and does not affect general tree-walk logic.

## 6. Tiny-node fast paths inside `decode_node_neon`

Still worth testing, but lower priority than the cross-backend gaps above.

The recursive decoder still sends very small groups through general node
machinery:

- bitmap handling
- TBL partition helpers
- scratch management
- recursive calls

Potential paths:

- `n <= 8` scalar partition
- `n <= 16` scalar or simplified vector partition
- tiny both-leaves path with direct scalar scatter
- tiny one-leaf path that scalar-partitions only the non-leaf side

Important clarification:

- `scatter_sym()` already avoids a heavy SIMD strategy for tiny scatter sizes.
- The opportunity is to avoid the whole partition/recurse setup for tiny nodes.

## 7. Precompute per-node decode shape metadata

`decode_node_neon()` and sibling backends repeatedly derive:

- whether left child is leaf
- whether right child is leaf
- whether either child is the prefill leaf
- whether the node is flat

This is cheap individually but happens at every recursive node.  Consider table
metadata such as:

- node kind: flat / both-internal / left-leaf / right-leaf / both-leaves
- prefill side: none / left / right
- local min/max depth or expected subtree weight for heuristics

This pairs naturally with shape-specific helper splitting.

## 8. Split hot decode helpers by shape

The main decode functions handle many cases in one large branchy function.  A
split may help register allocation and inlining:

- internal/internal full partition
- both leaves
- prefill-left
- prefill-right
- flat-subtree

This is more likely to matter on NEON than AVX-512, but keep the structure
parallel across backends if possible.

## 9. Scalar root-flat direct path

The scalar decoder does not have the SIMD backends' root-flat direct path.  It
prefills, initializes identity indices, and then reaches flat decode through the
recursive path.

This is not a headline optimization, but it is easy cleanup:

- if `flat_depth[root] >= 2`, read the packed root-flat region directly
- decode to `symbols[i]` without `indices[]`, `tmp[]`, or prefill

Useful for cleaner scalar baselines and ASAN/debug runs.

## 10. Keep these paths closed unless new evidence appears

Do not spend more time on:

- NEON partition store coalescing
- `str d` / `str q` mixed-store tricks
- 4-way NEON fused decode
- per-chunk dispatched leaf scatter variants
- SVE 128-bit on Graviton 4

The existing docs have good negative-result coverage for these.

## Suggested order

1. AVX-512 leaf fusion / half-partition.
2. AVX-512 root iota generation.
3. AVX2 enablement and one or two flat-unpack prototypes.
4. Hybrid decode heuristic for Graviton/Zen real-text losses.
5. SSE root both-leaves vectorization.
6. NEON tiny-node paths and shape metadata/helper splitting.
7. Scalar root-flat cleanup.
