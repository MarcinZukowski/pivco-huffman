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

## 5. Next steps

In order of expected payoff / implementation cost:

### Single-stage prefix-radix for non-flat trees (**next**)

Generalise v1: at a non-flat table, emit `M = min_len` bits per element
as a prefix stream, then emit standard 2-way PIVCO bitmaps for each
non-leaf subtree's remaining levels.  Decoder does the radix partition
at the top, then falls through to the existing `decode_node_neon` for
each non-leaf bin.

Expected wins on the distributions PIVCO currently loses on:

- english (M=3): plausibly +20–40%, enough to beat huf0_x2.
- proba14 (M=3): similar.
- bell_s10 (M=5): larger win because M is bigger.
- zipfian (M=3): modest initial win, biggest upside if we then also add
  nested radix for its staircase subtrees.
- proba02 (M=6): already close; should close the gap.

Complexity: ~2–3 days.  Main work is the K-way radix partition (can
reuse the bit-pack/unpack helpers already in v1).  Bit-extraction for
`M ∈ {3, 5, 6, 7}` needs slightly more care than the nice power-of-2
cases.

### Nested (multi-stage) prefix-radix (**after that**)

At each internal node during decode, use `M_local = local_min`
(precomputed at `build_table` time).  If `M_local ≥ 2`, another radix
pass; else fall back to a bitmap byte (== 2-way partition).

The zipfian case suggests this could unlock a further 30–50%+ on
top of single-stage.

Complexity: +1 day on top of single-stage.  Mostly bitstream-format
work — the decoder dispatch is a simple per-node check.

### Runtime gate in the public decoder

`pivco_huffman_decode` should pick between `decode_neon` and
`decode_neon_prefix` based on the table shape (flat-tree → prefix;
stick tree → neon; otherwise → whatever the benchmark says is faster
at that `M` and distribution shape).  Trivial once multiple backends
exist.

---

## Files

- `src/pivco_huffman_neon_prefix.c` — v1 encoder + decoder, flat trees
  only.
- `bench/bench_multi_stage_stats.c` — applicability analyser used in
  Section 4.
