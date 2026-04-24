# PIVCO-Huffman Decode Ideas

## Flat-subtree fast path — format-change variant

The full-tree flat case (`min_len == max_len`) has a shipped fast path
in `pivco_huffman_decode_neon_prefix` — the format stores `M` packed
bits per element and decode is a single `code_to_sym[prefix]` lookup.
Wins 2.47× on uniform and 1.18-1.24× on sparse_* (§3 of PREFIX_RADIX.md).

**The generalisation is a *format change*, not a decode-only
optimisation**: when the encoder detects that a specific internal node's
subtree is flat with depth `D >= 2` (all leaves reach the same relative
depth, 2^D codes total), it stops emitting `D` levels of bitmaps for
the elements entering that subtree and instead emits a single
**N × D-bit contiguous stream** — one `D`-bit code per element, packed
MSB-first, in the order the elements reach the subtree root.  The
decoder, working from the same table, knows this node is flat and
switches to the packed-bit reader + direct `code_to_sym[D_bits]` lookup.

This is exactly the mechanism that makes the full-tree flat path fast.
We are just applying it locally to every maximal flat subtree, not only
the whole tree.  Compression-wise it is slightly better than the
bitmap-per-level format because per-level byte-alignment padding is
replaced by one tail padding for the whole packed region; same number
of bits otherwise.

### Measured applicability

`extras/bench_flat_subtree_stats.c` counts elements that would land in
*maximal* flat subtrees (flat AND parent not flat, so each element is
counted once).  Results on the standard bench distributions:

| Distribution | coverage | breakdown by D | pivco_n vs best |
|---|--:|---|--:|
| **bell_s80** | **100.0%** | D=5: 28.6%, D=6: 71.4% | 0.69× |
| **proba02** | **98.4%** | D=3: 35%, D=4: 52%, D=2: 8% | 0.73× |
| **bell_s30** | **95.9%** | D=5: 32%, D=4: 30%, D=3: 23%, D=2: 11% | 0.83× |
| **bell_s10** | **94.4%** | D=4: 55%, D=3: 34%, D=2: 5% | 0.74× |
| **zipfian** | **69.4%** | spread D=2..7+ | 0.66× |
| **english** | **54.8%** | D=2: 54.8% (all at one depth) | 0.98× |
| proba14 | 0.9% | negligible | 0.93× |
| proba50 / proba80 / geometric | 0 | stick-tree shape | — |
| uniform / sparse_* / two_sym_* / flat_M* | — | root is flat, existing path handles | — |

The flat-subtree fast path would fire on *exactly* the distributions
where `pivco_n` currently loses to huf0 / trad_4s, with 70-100% of
elements addressable on the bell_* / proba02 set.

### Implementation sketch

Build-time (in `pivco_huffman_build_table`):

1. Walk tree post-order, computing `(subtree_min, subtree_max)` per
   internal node.
2. If `subtree_min == subtree_max == D` and `D >= 2`, mark node as
   `flat_subtree`, store `D`, and precompute `code_to_sym[1 << D]` for
   the subtree (indexed by the local `D`-bit code relative to the
   subtree root).
3. Also mark parent-chain so we only treat *maximal* flat subtrees as
   flat (descendants of a flat subtree are not separately flagged).

Encoder (`encode_node_neon`): when entering a flat-subtree node with
`n` elements, instead of `D` recursive partition steps, emit `n * D`
bits packed in the output stream.  For each element, pack the `D` low
bits of its Huffman code (offset by the subtree prefix) into the
output.  Advance the bitstream pointer by `ceil(n*D/8)` bytes.

Decoder (`decode_node_neon`): when entering a flat-subtree node with
`n` elements, read `n * D` packed bits, extract one `D`-bit code per
element using the existing bit-unpack routines from
`pivco_huffman_neon_prefix.c` (they already handle D ∈ {2,3,4,5,6,7,8}
with specialised paths), look up `code_to_sym[code]`, scatter to
`symbols[indices[i]]`.

Stream-format compatibility: this changes the bitstream format, so
existing encoded blobs wouldn't decode with a new decoder.  Two
mitigation options:
  - **Version bump on the block header** — gate the format change on a
    new block flag; backward-compatible.
  - **Just change the format** — cleanest, no version logic, but flag
    day for consumers.  Given this is still a research codebase, the
    clean option is probably right.

### Why this design is clean

- Decoder hot path for flat subtrees = the existing packed-bit path
  that already runs at the fast full-tree-flat speed.  No new
  bit-accumulate-per-level machinery required.
- Same total bit budget: `N × D` bits either way.  Slightly better
  packing (one tail padding vs `D` per-level paddings).
- Today's `scatter_both_leaves` is *already* the `D = 1` case of this
  scheme — one packed bit per element, direct
  `syms[(mask >> lane) & 1]` lookup.  The generalisation to `D >= 2`
  uses the same mechanism with wider codes and a `code_to_sym` table
  instead of a 2-entry inline array.
- No composition conflict with existing leaf-child fusion.  By
  construction every internal node inside a flat subtree of depth `D`
  has both children at relative depth `>= 1` with matching depths —
  so `partition_8_right`/`_left` half-partitions (which only fire when
  the leaf child is the `skip_node`, i.e., prefill, per
  `pivco_huffman_neon.c:338,353`) cannot apply inside the flat region.
  One-leaf-parent nodes always sit *outside* any flat subtree
  (`subtree_min = 1, subtree_max > 1` → not flat).
- No small-n threshold needed.  The packed-bit decode path's
  per-element cost is lower than the current per-level recursive
  tree walk even for single-digit `n`, because tree-walk carries
  function-call / scalar-remainder overhead per level.  `D = 1` with
  `scatter_both_leaves` already runs at all sizes without a threshold;
  `D >= 2` inherits that.

## Scatter fusion into partition — tried, reverted (recorded here for future reference)

See RESULTS.md §"Stage fusion" and §"Tested and discarded": the
*leaf-child* fusion (check child-type before partitioning, run half-
partition + scatter when one side is a leaf, sequential blend when both
are leaves) is *already shipped* and gives +10-38% on NEON.

The variant that *wasn't* kept is **conditional stores interleaved
inside the partition loop body** — letting each lane's output either
go to `tmp` (compaction) or to `symbols[scattered_position]` (leaf
scatter) based on the bitmap bit.  That regressed massively from branch
misprediction (NEON) and store-buffer interference (scalar).  The
current code keeps partition and scatter as separate phases
deliberately.

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

## ~~Interleave the 16-elem partition pair in decode~~ — attempted, null result

In `decode_node_neon`, the 16-elem-unrolled inner loop calls
`partition_8` twice per iteration.  Inspecting `objdump -d` on the
original code showed the compiler serialized the two chains: `m1`,
`popcnt[m1]`, `data1`, `compress_tab[m1]`, shuffles, TBLs and stores
for the second partition were all scheduled after the first partition's
stores completed — partly because `n_left`/`n_right` update through
`compress_popcnt[m0]` (a load with ~4-5 cycle latency) and partly
because pointer aliasing on `partition_8`'s `src`/`left_out`/`right_out`
forces store-before-load ordering.

Tried: inline both partitions manually — hoist both `m0`+`m1` loads,
both `compress_popcnt[m*]` loads, both data loads, all four store
addresses above any stores; then do the four TBLs and four stores.

Codegen transform was dramatic — clang merged both data vectors into a
single `ldp q0, q1`, issued both popcount loads consecutively, and
grouped all four TBLs.  Runtime delta on `pivco_n` across all 19
benchmarked distributions: ±2%, symmetric around zero, within
thermal-drift noise (CPU freq drift alone covered the spread).

Interpretation: M4's OoO engine was already renaming through the
textual serialization.  The bottleneck of `pivco_n` at ~0.25 c/elem is
elsewhere — likely TBL/store-port throughput or instruction fetch
width, not this dep chain.  Change doubled the loop body line count
for no measurable win, so reverted.  Recorded here so the next person
reading the assembly output doesn't repeat the experiment.

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
  - Tested twice, always worse on M4.  Retried 2026-04-24 on current
    clang: `pivco_n` drops 3–11% across distributions (proba80 −10%,
    bell_s10 −11%, english −4%, zipfian −3%).  Clang lowers
    `__builtin_popcount((int)mask)` to a 4-uop vector chain
    (`fmov d, x; cnt.8b; uaddlv.8b; fmov w, s`) with ~8-10c serial
    latency, routed through the same vector pipe as TBL.  The load
    from `compress_popcnt[mask]` is 1 LSU uop at ~5c on a separate
    pipe.  Replacing the load both adds uops to the already-TBL-heavy
    vector pipe AND lengthens the latency — strictly worse on every
    distribution.

- **SVE at 128-bit width**
  - already slower than NEON on Graviton4.

## Suggested implementation order

Leaf-child fusion is *shipped*; the remaining outstanding work, roughly
in increasing cost / decreasing certainty:

1. **Runtime flat-tree gate** in `pivco_huffman_decode` — route tables
   with `min_len == max_len` to `decode_neon_prefix`.  Pure win on
   uniform / sparse_* / flat_M*; no regression on non-flat tables.
   Smallest and most certain.
2. **Flat-subtree fast path** (§"Flat-subtree fast path — format-change
   variant" above) — measurement is done
   (`./build/pivco_flat_subtree_stats`) and shows 70-100% coverage on
   bell_* / proba02 / english / zipfian, exactly the distributions
   `pivco_n` currently loses on.  Format-change approach reuses the
   packed-bit decode path already shipped for full-tree flat.  Highest-
   impact remaining item.
3. **TBL-based K-way bucket** for `decode_neon_prefix` phase 4 — the
   main remaining phase-4 lever in the non-flat prefix-radix path.
   Lower priority if flat-subtree fast path displaces the prefix-radix
   approach for most real-world tables.
4. **Nested (multi-stage) prefix-radix** — only if we've already
   committed to the prefix-radix path winning on zipfian-shaped
   staircases (§4 of PREFIX_RADIX.md).  Likely superseded by the
   flat-subtree fast path on similar distributions.
