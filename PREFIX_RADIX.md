# Prefix-radix decoder

An alternative PIVCO decode path for Huffman tables whose **minimum code
length is ≥ 2**: replace several levels of 2-way SIMD partition with a
single radix pass over the first `M` bits of every element's code,
where `M = table->min_len`.

This document covers:

1. The core idea and why it should help.
2. The v1 prototype — works for flat trees only.
3. Bench results on M4 Max.
4. Multi-stage analysis — when nested radix would unlock additional gains.
5. Next steps.

---

## 1. The core idea

### What PIVCO does today

The current `pivco_huffman_decode_neon` performs one 2-way SIMD partition
per tree level during a DFS tree walk.  Each level splits the active
indices into "bit=0" (left child) and "bit=1" (right child) groups,
recurses, and eventually scatter-writes each leaf's symbol.

For a Huffman table with `min_code_len = M`:

- Every element descends through **at least** `M` levels before it can
  possibly terminate at a leaf.
- That means the first `M` levels do `M × N / 8` TBL shuffles (plus stores,
  recursion, etc.) to do what is effectively a bulk radix partition.

### What prefix-radix does instead

Encode the first `M` bits of every element's code as a contiguous
per-element bit-packed stream, then decode by:

1. **Extract** each element's `M`-bit code prefix → a "bin index" in
   `[0, 2^M)`.
2. **Radix-partition** the `N` elements into `2^M` bins by their prefix.
3. For each bin:
   - If the `M`-bit prefix is a **complete code** (leaf at depth `M`):
     scatter that leaf's symbol to every element in the bin — done.
   - Otherwise (internal subtree at depth `M`): recurse on the bin's
     elements using the standard 2-way PIVCO decoder for the remaining
     levels of the tree.

For flat trees (`min_len == max_len`) all bins are leaves, so step 3
reduces to a simple `symbols[k] = code_to_sym[prefix[k]]` permutation —
no partition, no recursion.

### Why it should win on deep trees

For `M = 1` (stick tree like `proba80`), one radix pass ≡ one 2-way
partition pass — no change.  For `M ≥ 2` the single radix pass
replaces `M` levels of 2-way partition.

On M4 NEON each level of 2-way partition costs roughly 0.25 c per
element.  The radix pass costs roughly 1–1.5 c per element regardless
of `M`.  So the crossover is theoretically near `M = 4–6`.  In
practice (see Section 3), even `M = 2` wins because of the memset +
multi-phase overhead of the current path.

---

## 2. v1 prototype scope

`src/pivco_huffman_neon_prefix.c` implements only the **flat-tree case**
(`min_len == max_len`).  Behaviour:

- **Encoder** (`pivco_huffman_encode_neon_prefix`): packs `M` bits per
  element into the output stream.  Returns `PIVCO_ERR_CORRUPT` when the
  table is not flat.
- **Decoder** (`pivco_huffman_decode_neon_prefix`): unpacks each element's
  `M`-bit prefix and does a single `code_to_sym[prefix]` lookup.  No
  radix partitioning, no subtree recursion, because every bin is a leaf.
  Specialised fast paths for `M ∈ {1, 2, 4, 8}`; generic bit-unpacker
  for other `M`.

In-scope distributions:

| Distribution  | M | Why flat |
|---------------|--:|----------|
| `two_sym_eq`  | 1 | 2 symbols, 1-bit codes |
| `two_sym_90/10` | 1 | 2 symbols, 1-bit codes |
| `sparse_4`    | 2 | 4 equal symbols, 2-bit codes |
| `sparse_16`   | 4 | 16 equal symbols, 4-bit codes |
| `uniform`     | 8 | 256 equal symbols, 8-bit codes |

For every other distribution the v1 path returns an error and the
benchmark reports 0.

---

## 3. v1 bench results (Apple M4 Max, 20 reps × 4M symbols)

| Distribution | pivco_n | pivco_p | Δ vs pivco_n | best other | PIVCO-vs-best ratio |
|--------------|--------:|--------:|-------------:|-----------:|--------------------:|
| uniform      | 1155    | **3965** | **+243%**   | trad_4s 1604 | **2.47×** (was 0.73×) |
| sparse_16    | 2581    | **5622** | **+118%**   | huf0_x2 4611 | **1.22×** (was 0.57×) |
| sparse_4     | 4535    | **6199** | **+37%**    | huf0_x2 5261 | **1.18×** (was 0.87×) |
| two_sym_eq   | 26027   | 6222    | −76%         | huf0_x2 5243 | 4.96× (via pivco_n path) |
| two_sym_90/10| 25716   | 6314    | −75%         | huf0_x2 4967 | 5.18× (via pivco_n path) |

### Headline: uniform goes from PIVCO's worst to PIVCO's winner

Before: 1155 M/s, `0.73×` of the best non-PIVCO decoder — PIVCO *lost* to
trad_4s.

After: 3965 M/s, `2.47×` the best non-PIVCO decoder — PIVCO **beats all
four** alternatives (huf0_x1, huf0_x2, trad_4s, rans_x2) on uniform.

### Why two_sym regresses

The existing `decode_neon` has a specialised **root both-leaves** fast
path (`scatter_both_leaves` at root: sequential `vst1` blending two
symbols directly, no scatter write).  For `M = 1` flat trees it runs
at ~26 GB/s on M4 — the generic bit-unpack loop of the prefix backend
can't touch that.

In production, a runtime gate inside `pivco_huffman_decode` would pick
`max(pivco_n, pivco_p)` per block; the prefix backend would only run
where it wins.

### Surprise: M=2 already wins (sparse_4, +37%)

I expected the crossover to be around M=4–6.  The actual crossover is
around M=2.  The cycle-level reason (see cycle analysis in the
conversation log): the current path pays real overhead per block for
the prefill `memset`, a separate partition phase, a separate scatter
phase, and NEON→GPR transfers for scattered stores.  The prefix path
is a single tight loop that writes each output byte exactly once.
Those overheads accumulate enough that by `M = 2` the prefix path
already wins.

This suggests the **single-stage generalisation (Section 5) will
likely help even at `M = 3`**, which covers english and zipfian —
exactly the distributions PIVCO currently loses on.

---

## 4. Multi-stage analysis

`bench/bench_multi_stage_stats.c` measures: after a single-stage radix
at `M_top = table->min_len`, what fraction of elements would land in
non-leaf subtree bins whose **local minimum code length ≥ 2** (i.e.,
where applying another radix at the subtree root would save more work)?

### Methodology

For each distribution:

1. Build the Huffman table.
2. Walk the tree following each `M_top`-bit prefix `v ∈ [0, 2^M_top)`:
   - If we land on a leaf → this is a leaf bin; the `M_top` bits are a
     complete code.
   - If we land on an internal node at depth `M_top` → this is a subtree
     bin; compute `local_min` = shortest leaf depth relative to that node.
3. Weight each bin by the sum of frequencies of symbols whose codes
   start with that prefix.
4. Report the element-weighted fraction landing in subtree bins with
   `local_min ≥ 2` (multi-stage addressable) and `≥ 3` (multi-stage
   saves more than one extra level).

### Findings

| Distribution | M_top | % elems in subtree bins | **% elems where multi-stage fires (local_min ≥ 2)** | % elems where multi-stage saves > 1 level |
|--------------|:-----:|:-----------------------:|:---------------------------------------------------:|:-----------------------------------------:|
| **zipfian**  | 3     | 84%                     | **70.0%**                                           | **57.6%**                                 |
| proba02      | 6     | 57%                     | 28.0%                                               | 13.8%                                     |
| **english**  | 3     | 74%                     | 26.9%                                               | 14.1%                                     |
| proba14      | 3     | 64%                     | 22.2%                                               | 10.6%                                     |
| bell_s30     | 6     | 63%                     | 21.1%                                               | 7.2%                                      |
| bell_s80     | 7     | 82%                     | 13.5%                                               | 0.0%                                      |
| bell_s10     | 5     | 23%                     | 8.8%                                                | 2.6%                                      |
| proba80      | 1     | 20%                     | 0%                                                  | 0%                                        |
| proba50      | 1     | 50%                     | 0%                                                  | 0%                                        |
| geometric    | 1     | 50%                     | 0%                                                  | 0%                                        |
| uniform      | 8     | 0% (all leaf)           | N/A                                                 | N/A                                       |
| sparse_4/16  | 2/4   | 0% (all leaf)           | N/A                                                 | N/A                                       |
| two_sym_*    | 1     | 0% (all leaf)           | N/A                                                 | N/A                                       |

### Where multi-stage fires

- **zipfian** is the standout: 70% of elements would land in subtree
  bins with `local_min ≥ 2`, and 58% where `local_min ≥ 3`.  zipfian's
  code-length histogram `{3:1, 4:2, 5:4, 6:8, 7:16, 8:32, 9:63, 10:130}`
  creates a "staircase" of subtree bins with progressively deeper
  local minima (2, 3, 4, 5, 6, 7).  Nested radix could cascade down
  this staircase, collapsing multiple 2-way partition levels at each
  recursive step.
- **english, proba02, proba14, bell_s30**: all 20–30% multi-stage
  addressable — meaningful but not dominant.
- **bell_s10, bell_s80**: <15% — modest.

### Where multi-stage does *not* apply

- **Stick trees** (proba80, proba50, geometric): `local_min = 1` at
  every internal node all the way down.  Nested radix never finds
  `M_local ≥ 2`.  Equivalent to 2-way PIVCO.
- **Flat trees** (uniform, sparse_*, two_sym_*): single-stage already
  handles everything, no subtree bins remain.

### Corrections to the earlier "no gap in Huffman code lengths" heuristic

I initially claimed that multi-stage only fires when the Huffman table
has a gap right after `min_len` (e.g. 3-bit codes but no 4-bit codes).
That predictor turned out to be wrong: zipfian has no gap in its
length distribution and still has 70% multi-stage-addressable weight.

The correct predictor is "fraction of subtree bins with `local_min ≥ 2`",
which depends on the *shape* of the tree — specifically, whether some
subtree roots happen to have two internal children (rather than a leaf
child + an internal child).  This is a second-order property of the
frequency distribution, not a first-order property of the length
histogram.

---

## 5. Single-stage for non-flat trees — implemented, correct, slow

`src/pivco_huffman_neon_prefix.c` now handles the non-flat case as well
as the flat case.  The decoder flow is:

1. **Phase 0** — `memset` output with `prefill_sym`.
2. **Phase 1** — extract each element's `M`-bit prefix from the stream
   into a per-element `prefix[N]` buffer (specialised fast paths for
   `M ∈ {1, 2, 4, 8}`, scalar-unrolled for `M ∈ {3, 5, 6, 7}`).
3. **Phase 2** — histogram: for each `k`, increment `bin_count[prefix[k]]`.
4. **Phase 3** — prefix-sum for `bin_offset[K+1]`.
5. **Phase 4** — bucket element ids by bin: `bin_elements[place[prefix[k]]++] = k`.
6. **Phase 5** — per-bin dispatch: leaf bin → `scatter_sym`, subtree bin
   → `pivco_neon_decode_subtree_` on the bin's elements (with a scratch
   copy to avoid the existing subtree encoder writing past the bin's
   segment via its 16-byte partition stores).

All 20 roundtrip tests pass.  But it's **slower than baseline across
every non-flat distribution**:

| Distribution | pivco_n | pivco_p | Δ      |
|--------------|--------:|--------:|-------:|
| proba80 (M=1)  | 9478    | 318     | −97%   |
| proba50 (M=1)  | 5085    | 342     | −93%   |
| geometric (M=1)| 4816    | 342     | −93%   |
| english (M=3)  | 2468    | 697     | −72%   |
| proba14 (M=3)  | 2366    | 700     | −70%   |
| zipfian (M=3)  | 1252    | 541     | −57%   |
| bell_s10 (M=5) | 1754    | 844     | −52%   |
| bell_s30 (M=6) | 1195    | 871     | −27%   |
| proba02 (M=6)  | 1135    | 837     | −26%   |
| bell_s80 (M=7) | 1091    | 842     | −23%   |

### Per-phase profile (`bench_prefix_profile`)

`bench/bench_prefix_profile.c` times each phase over 50k iterations.
Results on M4 Max (c/elem assumes 3.5 GHz):

| Phase | english (M=3) | zipfian (M=3) | proba02 (M=6) |
|-------|:-------------:|:-------------:|:-------------:|
| 0: memset               | 0.05 | 0.04 | 0.04 |
| 1: extract M-bit prefix | 0.28 | 0.23 | 0.22 |
| **2: histogram**        | **1.70** | **1.73** | **1.67** |
| 3: prefix-sum           | 0.00 | 0.00 | 0.01 |
| **4: bucket**           | **2.14** | **2.16** | **2.14** |
| 5: per-bin dispatch     | 0.67 | 1.52 | ~0   |
| **TOTAL**               | **4.84** | **5.67** | **3.71** |
| baseline `pivco_n`      | 1.32 | 2.05 | 2.41 |
| ratio                   | 3.7× | 2.8× | 1.5× |

### Diagnosis

**Phases 2 and 4 together cost ~3.84 c/elem regardless of `M` or
distribution.**  They're the expensive part and the reason the approach
loses.  Both are serial-data-dependency-limited scalar loops:

- Phase 2: `bin_count[prefix[k]]++` — when prefix values cluster (e.g.
  english's 25% on one bin), adjacent increments of the same counter
  serialise on the load-add-store dependency.
- Phase 4: `bin_elements[place[prefix[k]]++] = k` — same serial dep
  on `place[v]`, plus a store with v-dependent address (poor write
  combining when bins are interleaved).

**Phase 5 scales favourably with `M`** — it's just the existing neon
subtree work on smaller, pre-partitioned groups.  For english (M=3),
phase 5 is 0.67 c/elem vs baseline's full 1.32; for proba02 (M=6),
phase 5 is ≈0 because most bins are leaves.

**Phases 2+4 DON'T scale with `M`** — they're O(N) regardless, and
that's the problem.  Baseline's 3 levels of 2-way partition (for M=3)
cost ~0.75 c/elem on M4 NEON (three `partition_root_8` passes × 0.25
c/elem each) with all SIMD.  We're spending 3.84 c/elem to do the same
amount of conceptual work in scalar + serial.

### Why `init_compress_table()` was the root cause of an earlier crash

The first non-flat implementation segfaulted on english.  Root cause:
`pivco_huffman_encode_neon_prefix` didn't call `init_compress_table()`
before delegating to `pivco_neon_encode_subtree_`.  Since `compress_tab`
and `compress_popcnt` are zero-initialised globals, `partition_8` would
use an all-zero shuffle (collapsing all 8 outputs to the first byte
of the input) and return `popcount = 0` for every mask — producing
garbage "valid" entries that propagated until one was dereferenced as
a `codes[idx]` lookup with an out-of-range idx.  Fixed by calling
`init_compress_table()` at the top of both public entry points.

## 6. Next steps

The profile makes the direction clear: phases 2 and 4 have to drop by
roughly 3× combined to break even with baseline, or 5× to have a
meaningful win.  Both optimisations are classic radix-sort techniques:

### Parallel histogram (phase 2)

Use 4 independent counter arrays indexed by `k % 4`, sum at end.  OoO
pipelines 4 independent `counter_p[prefix[k]]++` dependency chains.
Expected: 1.70 → ~0.45 c/elem (near-linear 4× speedup) on proba80-ish
clustered distributions.

### SIMD K-way bucket partition (phase 4)

Replace the serial `bin_elements[place[v]++] = k` with a TBL-based
per-chunk radix partition: load 8 prefix values + the corresponding
element identities, for each bin `v ∈ [0..K)` compute the 8-lane mask
where `prefix == v`, TBL-compact the matching identities into that
bin's output offset, advance `place[v]`.  Per 8 elements: K TBLs +
K vsts.  For K=8 (M=3): 8 TBLs per 8 elements = 1 TBL/elem — same order
as a 2-way partition's 2 TBLs per 8 elements but doing the work of 3
tree levels in one sweep.  Expected: 2.14 → ~0.8 c/elem.

### Combined projection

If both optimisations hit their targets:

| Phase | Current | After SIMD |
|-------|--------:|-----------:|
| 0     | 0.05    | 0.05       |
| 1     | 0.28    | 0.28       |
| 2     | 1.70    | 0.45       |
| 3     | 0.00    | 0.00       |
| 4     | 2.14    | 0.80       |
| 5     | 0.67    | 0.67       |
| total | 4.84    | **2.25**   |

That's within striking distance of baseline's 1.32 c/elem on english —
still probably a modest loss, but flipping to a win on zipfian
(baseline 2.05) and comfortably beating baseline on proba02+ (baseline
2.41).  With nested radix (the zipfian staircase from §4), zipfian's
phase 5 could drop further as well.

### Nested (multi-stage) prefix-radix

At each internal node during decode, use `M_local = local_min`
(precomputed at `build_table` time).  If `M_local ≥ 2`, another radix
pass; else fall back to a bitmap byte (== 2-way partition).  The
zipfian case suggests this could unlock a further 30–50% on top of
single-stage, specifically for distributions with staircase code length
profiles.

### Runtime gate in the public decoder

`pivco_huffman_decode` should pick between `decode_neon` and
`decode_neon_prefix` based on the table shape (flat-tree → prefix;
stick tree → neon; otherwise → whatever is faster at that `M` and
distribution shape).  Trivial once both backends are competitive.

---

## Files

- `src/pivco_huffman_neon_prefix.c` — encoder + decoder.  Flat case is
  a fast direct permutation; non-flat case is correct but slower than
  baseline pending the phase-2/phase-4 SIMD work described in §6.
- `bench/bench_multi_stage_stats.c` — per-distribution applicability
  analyser used in §4.
- `bench/bench_prefix_profile.c` — per-phase profiler used in §5
  (`./build/pivco_prefix_profile [distribution]`).  Times each of the
  six phases across 50k iterations and reports ns/block and c/elem.
