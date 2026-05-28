# Prefix-radix decoder

> **Last content review:** _NEVER_

> **Status (as of `8754347`, 2026-04-24):** the flat-tree special case
> described in §2-3 of this document has been **superseded** by the
> *flat-subtree fast path* landed in April 2026 (commit `a275d05` and
> follow-ups).  The flat-subtree path detects every maximal flat region
> in the Huffman tree — including the whole-tree flat case — and uses
> the same N·D-bit packed-stream mechanism described here, but from
> inside `decode_node_neon` directly, without a separate backend.
> See README.md §"Tested and adopted" for the current state.
>
> The **non-flat prefix-radix path** described in §5-6 remains an
> unretired research track — slower than `pivco_n` on M4 (1.2–2.0×) but
> kept as the `pivco_p` benchmark column for comparison against the
> main decoder's flat-subtree path.  Nested / multi-stage radix (§4, §6)
> would target the same distributions the flat-subtree path already
> wins on, so that work is also effectively displaced.
>
> This document is preserved as historical record of the design
> exploration.  Its proofs about *where* flat-region exploitation
> pays off (the "single-stage generalisation wins at M≥2" result in
> §3 and the multi-stage staircase analysis in §4) were directly
> validated by the flat-subtree sweep results.

An alternative PIVCO decode path for Huffman tables whose **minimum code
length is ≥ 2**: replace several levels of 2-way SIMD partition with a
single radix pass over the first `M` bits of every element's code,
where `M = table->min_len`.

This document covers:

1. The core idea and why it should help.
2. The v1 prototype — works for flat trees only.
3. Bench results on M4 Max for the flat case.
4. Multi-stage analysis — when nested radix would unlock additional gains.
5. Single-stage for non-flat trees — current state and per-phase profile.
6. Experiments tried and remaining candidates.

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

`extras/bench/bench_multi_stage_stats.c` measures: after a single-stage radix
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

## 5. Single-stage for non-flat trees — implemented, correct, still losing

`src/pivco_huffman_neon_prefix.c` handles the non-flat case as well as
the flat case.  The decoder flow is:

1. **Phase 0** — `memset` output with `prefill_sym`.
2. **Phase 1** — extract each element's `M`-bit prefix from the stream
   into a per-element `prefix[N]` buffer (specialised fast paths for
   `M ∈ {1, 2, 4, 8}`, scalar-unrolled for `M ∈ {3, 5, 6, 7}`).
3. **Phase 2** — histogram: 8 independent counter arrays `bc[8][K]`
   indexed by `k % 8`, summed at the end.
4. **Phase 3** — prefix-sum for `bin_offset[K+1]`.
5. **Phase 4** — bucket: 8 independent `place[8][K]` arrays
   pre-computed so lane `s` writes `bin_elements[place[s][prefix[k+s]]++] = k+s`.
6. **Phase 5** — per-bin dispatch: leaf bin → `scatter_sym`, subtree bin
   → `pivco_neon_decode_subtree_` on the bin's elements (with a scratch
   copy to avoid the existing subtree encoder writing past the bin's
   segment via its 16-byte partition stores).

All 20 roundtrip tests pass.  After the 4-way and then 8-way parallel
rewrites of phases 2 + 4 it is **roughly 2× faster than the initial
scalar version** but still **slower than the `pivco_n` 2-way decoder on
every non-flat distribution**:

| Distribution | pivco_n | pivco_p | Δ    |
|--------------|--------:|--------:|-----:|
| proba80 (M=1)   | 9625 | 1167 | −88% |
| geometric (M=1) | 4956 | 1054 | −79% |
| proba50 (M=1)   | 5137 | 1075 | −79% |
| english (M=3)   | 2514 | 1242 | −51% |
| proba14 (M=3)   | 2423 | 1232 | −49% |
| bell_s10 (M=5)  | 1785 | 1115 | −38% |
| zipfian (M=3)   | 1261 |  822 | −35% |
| bell_s80 (M=7)  | 1124 |  958 | −15% |
| bell_s30 (M=6)  | 1210 | 1036 | −14% |
| proba02 (M=6)   | 1144 |  998 | −13% |

(M/s, `./build/pivco_huffman_bench 20`, M4 Max, 4M × 20 reps.)

The gap narrows as `M` grows because phase 5 shrinks toward zero
(deeper prefixes → more bins are leaves at depth `M`).  At `M = 6–7`
we are within 15% of `pivco_n` without having touched phase 4 SIMD yet.

### Per-phase profile (`bench_prefix_profile`, current 8-way tree)

c/elem assuming 3.5 GHz:

| Phase                 | english | zipfian | proba14 | proba02 | bell_s30 | bell_s80 |
|-----------------------|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| 0: memset             | 0.04    | 0.03    | 0.03    | 0.03    | 0.03     | 0.03     |
| 1: extract M-bit      | 0.24    | 0.20    | 0.21    | 0.20    | 0.20     | 0.20     |
| **2: histogram**      | **0.72**| **0.72**| **0.73**| **0.73**| **0.74** | **0.73** |
| 3: prefix-sum         | 0.00    | 0.00    | 0.00    | 0.01    | 0.01     | 0.02     |
| **4: bucket**         | **1.74**| **1.80**| **1.78**| **1.79**| **1.79** | **1.82** |
| 5: per-bin dispatch   | 0.00    | 0.66    | 0.00    | 0.18    | 0.11     | 0.55     |
| **TOTAL**             | **2.66**| **3.41**| **2.70**| **2.94**| **2.89** | **3.34** |
| `pivco_n`             | 1.30    | 1.98    | 1.32    | 2.38    | 2.26     | 2.41     |
| ratio                 | 2.05×   | 1.72×   | 2.05×   | 1.24×   | 1.28×    | 1.37×    |

### Diagnosis

**Phase 2 (histogram) is close to port-bound and hard to speed up
further.**  8 independent streams give OoO enough to sustain ~1 load +
1 store per cycle on the counter RMW.  At 0.72 c/elem the load/store
ports are well-saturated per 8 elements — going wider would add
counter-summing overhead that eats the gains.

**Phase 4 (bucket) is dep-chain-latency-limited on `place[s][v]`, not
memory-port-limited** — my earlier hypothesis was wrong.  The 4→8-way
split empirically delivered another +12–19% on top of the 4-way version
(english, zipfian, proba14 across the benches), which ruled out the
"we're already at the load/store-port cap" story.  What's actually
happening: each lane's `place[s][prefix[k+s]]++` is a load → add →
store RMW chain of ~4 cycles on the same address when prefixes cluster;
4 independent lanes weren't enough for the OoO window to hide the
latency on inputs where two prefixes within a group of 8 collide
(english has a 25% prefix, so collisions are common).  8 lanes fit,
and the next doubling (16) would need 16 × K × 4 B ≥ 8 KB of L1 at
K = 128 — enough to start thrashing.

**Phase 5 scales favourably with `M`.**  Its cost is distribution-
dependent: zipfian (M=3) pays 0.66 c/elem because four deep subtrees
carry most of the weight, while proba02/bell_s30 (M=6) pay 0.11–0.18
because most bins are leaves at depth 6.  The profiler computes phase
5 as `full_decode − Σ(phase 0..4)` clamped at 0, so the 0.00 reported
for english / proba14 is an overlap artifact (the isolated-phase
measurements overestimate slightly), not literal zero subtree work —
but it is small enough to vanish in the jitter.

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

## 6. Experiments tried

Summary of what has been explored against the profile above.

### Parallel histogram + bucket (phase 2 + 4) — *shipped*

4-way counter arrays indexed by `k % 4`, summed at end.  Delivered
~30–40% vs the initial scalar version (commit `96b578a`).  Extended to
8-way for another +12–19% (commit `5557966`).  Current 8-way phase 2
at 0.72 c/elem, phase 4 at 1.78 c/elem.  Widening further to 16 lanes
would cost 16 × K × 4 B of working set (8 KB at K = 128) — cache
pressure + summing overhead outweigh the marginal dep-chain slack.

### Skip-histogram, pre-allocated per-bin buffers — *tried, reverted*

Replace phases 2–4 with: allocate one `uint16_t[N]` scratch per bin, in
one pass append `k` to `scratch[prefix[k]]`, then copy each scratch
back out to `bin_elements` respecting the prefix-sum.  Eliminates the
separate histogram pass, but `K × N × 2 B` (e.g. 8 × 8192 × 2 = 128 KB
at M=3 block) blows L1d on M4; copy-out phase dominates and the total
came in slower than the 4-way baseline.  Reverted.

### Skip-histogram, wide-stride N+K buffer — *tried, reverted*

Single `uint16_t[N + padding_per_bin × K]` buffer, each bin owns a
stride of up to `N` slots inside it.  Avoided the per-bin allocation
but regressed on M ≥ 6 because the stride overcommit grew with K
(e.g. 128 × N at M=7) and the copy-out still did bin-ordered
gather-like reads.  Mixed wins on M=3/5, losses elsewhere — net neutral.
Reverted.

### Software write-combining buffer (SWCB) for phase 4 — *tried, reverted*

Per-bin small buffer (16 entries × K), flush to `bin_elements` when
full.  The flush amortises the cache-line ping-pong of interleaved
bucket writes, which helps on much larger N where cache lines for
different bins conflict.  At our N = 8192 and K ≤ 128 block, the flush
bookkeeping (per-bin count check each loop iter, or branch-on-full)
added more latency than the sparse-write penalty it was trying to
avoid.  Reverted.

### Remaining candidates (not yet tried)

- **TBL-based SIMD K-way bucket partition.**  Per chunk of 8 elements,
  for each bin `v` compute a 8-lane mask where `prefix == v`, TBL-
  compact the matching indices into `bin_elements[place[v]]`, advance
  `place[v]` by popcount.  Cost per 8 elements: K mask-compares + K
  TBLs + K popcounts + K stores — cheap at K = 8 (M=3), expensive at
  K = 128 (M=7).  This is the main phase-4 lever still on the table
  for small-M distributions.
- **Nested (multi-stage) prefix-radix.**  At each internal node, use
  `M_local = local_min` (precomputed).  §4 shows zipfian has 70% of
  its weight in subtree bins with `local_min ≥ 2`; nested radix could
  collapse phase 5's subtree cost for exactly that shape.
- **Runtime gate in `pivco_huffman_decode`.**  The prefix backend
  already wins on flat trees (§3) by 37–243%.  A trivial gate picking
  the prefix path when `min_len == max_len` would be a pure win; for
  non-flat tables it would keep using `pivco_n` until the TBL bucket
  lands.  This is the concrete next incremental step.

---

## Files

- `src/pivco_huffman_neon_prefix.c` — encoder + decoder.  Flat case is
  a fast direct permutation; non-flat case is 8-way-parallel in phases
  2 + 4 but still slower than `pivco_n` pending the TBL bucket work
  described in §6.
- `extras/bench/bench_multi_stage_stats.c` — per-distribution applicability
  analyser used in §4.
- `bench/bench_prefix_profile.c` — per-phase profiler used in §5
  (`./build/pivco_prefix_profile [distribution]`).  Times each of the
  six phases across 50k iterations and reports ns/block and c/elem.
