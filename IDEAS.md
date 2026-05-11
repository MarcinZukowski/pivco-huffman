# PIVCO-Huffman Decode Ideas

## ~~Flat-aware Huffman tree restructurer~~ — SHIPPED

**Status (2026-04-25):** SHIPPED.  `pivco_huffman_build_table` now
produces flat-aware tree shapes; same code-length multiset (=
identical compression).  Analyzer:
[`extras/bench_flat_optimal.c`](extras/bench_flat_optimal.c).

**Headline wins (pivco_n M/s, before → after):**

| Distribution | Apple M4 | Xeon AVX-512 | Graviton 4 | Zen 3 |
|---|--:|--:|--:|--:|
| `english`  | 2908 → 3303 (**+14%**) | 1758 → 2171 (**+23%**) | 1177 → 1246 (+6%)  | 794 → 887 (**+12%**) |
| `proba14`  | 2510 → 2829 (**+13%**) | 1176 → 1876 (**+60%**) | 973 → 1147 (**+18%**) | 669 → 773 (**+16%**) |
| `proba02`  | 2304 → 2569 (**+12%**) | 1405 → 1564 (**+11%**) | 899 → 1015 (**+13%**) | 626 → 710 (**+13%**) |
| `bell_s80` | 2890 (no change)        | 2041 → 2277 (**+12%**) | 1105 → 1312 (**+19%**) | 818 → 923 (**+13%**) |
| `bell_s10` | 3114 → 3179 (+2%)      | 1639 → 1941 (**+18%**) | 1189 → 1304 (**+10%**) | 830 → 871 (+5%)  |
| `bell_s30` | 2303 → 2370 (+3%)      | 1175 → 1396 (**+19%**) | 882 → 920 (+4%)   | 610 → 647 (+6%)  |

Parity-cross flips: **`proba14` on M4 (0.91× → 1.10×)**, **`proba14`
on Xeon (0.67× → 1.07×)**, **`proba02` on Graviton (0.92× → 1.04×)**.

The Xeon wins are larger than M4 because Xeon's `vpcompressw` partition
has higher relative cost than M4's NEON `tbl`, so saving partitions
translates to a bigger throughput gain there.

**The opportunity.**  `pivco_huffman_build_table` currently produces the
canonical Huffman tree (sort symbols by `(length, value)`, assign codes
sequentially, walk codes MSB-first to materialise the tree).  This
preserves a property useful for many Huffman implementations — codes
within a length class are contiguous — but it does **not** maximise
flat-subtree coverage.  A different arrangement of the same code-length
multiset (= identical compression) can produce more / larger flat-D≥2
subtrees and fewer partition steps in the tree-walk decode.

**Algorithm (greedy, optimal):**  For each length L, decompose `c_L` by
its binary representation:
- Bits at positions ≥ 2 → flat-D≥2 chunks of size 2^D.  Highest-freq
  length-L symbols go to largest-D chunks (deepest savings).
- Bit 1 → D=1 sibling pair (handled by stage fusion in production).
- Bit 0 → singleton (paired with whatever residual structure exists).

This is provably optimal for D≥2 leaf coverage:
`max_per_length = c_L & ~3` because any flat-D≥2 subtree must contain
2^D ≥ 4 same-length leaves.  The union of all per-length greedy chunks
is realisable in a valid tree because each flat-D root contributes the
same Kraft (`2^-(L-D)`) as one residual internal node would; the
leftovers (`c_L mod 4` per length, 0..3 each) fill the remaining mass.

**Why it matters — partition steps, not coverage.**  The first-pass
analysis (D≥2 leaf coverage) was misleading: canonical Huffman already
handles "leaf-coverage equivalents" via stage-fusion D=1 sibling pairs.
The real win is in **partition-step count**: a flat-D=2 root absorbs
the partition that would otherwise happen at depths between root and
leaves, while two D=1 stage-fusion pairs only avoid the partition at
their immediate parent.  Flatness *compounds*: a D=3 flat eliminates
3 levels of partition path, vs 3 separate D=1 pairs which only save
the bottom level.

**Measured gap (canonical vs optimal, partition operations per element):**

| Distribution | canon | opt  | savings |
|---|--:|--:|--:|
| `bell_s80`   | 2.24 | 1.64 | **−26.7%** |
| `english`    | 2.73 | 2.04 | **−25.3%** |
| `proba02`    | 3.60 | 2.90 | **−19.4%** |
| `proba14`    | 3.39 | 2.84 | **−16.0%** |
| `bell_s10`   | 2.04 | 1.93 | −5.8% |
| `bell_s30`   | 3.26 | 3.17 | −2.8% |
| `zipfian`    | 3.21 | 3.20 | −0.2% |
| `geometric`  | 2.02 | 2.01 | −0.8% |
| `proba80`, `proba50`, fully-flat dists | (no change) | | |

`bell_s80` notably already has 100% flat coverage in canonical, but
optimal still saves 26.7% partitions — by consolidating multiple small
flat-roots into fewer, deeper ones.

**Expected runtime gain.**  Partition_8 costs ~1 cycle / 8 elements.
For an 8192-block of english: canonical ≈ 2800 partition cycles,
optimal ≈ 2090, saving ~700 cycles.  Out of ~25k cycles total decode
cost (M4 at 2900 M/s on 3 GHz core), that's roughly 3% throughput.
Across the four big-savings distributions, expect 3–8% absolute
throughput gain — small but real, and on top of the existing flat-
subtree work.

**Implementation sketch.**
1. In `pivco_huffman_build_table`, replace the "walk canonical bits to
   build tree" pass (lines ~435–470 today) with a constructive layout:
   - Per length L (descending): emit chunks in greedy order (largest D
     first), then D=1 pairs, then singleton.
   - For each chunk, allocate one residual-internal slot at depth L−D
     and instantiate the perfect-binary subtree of 2^D leaves below it.
   - Place these residual leaves (= flat roots + leftover leaves) into
     a binary tree by concatenating Kraft mass — equivalent to
     canonical assignment over the *residual* multiset.
2. Re-derive `table->code[sym]` from the resulting tree shape (walk DFS,
   emit the bit string for each leaf).  `decode_sym` / `decode_len` /
   `first_code` / `sorted_symbols` fall out the same way.
3. `flat_mark_subtrees` (existing) detects the new flat roots.
4. Encoded format is self-describing — no on-the-wire change.

The decoder side requires no changes (it already routes via
`flat_depth[node]` and stage fusion).

## ~~Flat-subtree fast path — format-change variant~~ — SHIPPED

> **Status (as of `8754347`, 2026-04-24):** SHIPPED.  Commits
> `a275d05` (initial, flat-subtree at non-root nodes on scalar + NEON),
> `0d9ed64` (root-flat unified into the same mechanism), `7c3238b`
> (ported to x86 SSE4.1 and AVX-512), `0a92fe3` (libm link fix for
> Linux).  Measurement analyzer:
> [`extras/bench_flat_subtree_stats.c`](extras/bench_flat_subtree_stats.c).
> Four-platform sweep:
> [`results/20260424-204720-0a92fe3-flat-subtree-sweep.md`](results/20260424-204720-0a92fe3-flat-subtree-sweep.md).
> Full-grid numbers in README.md §"Tested and adopted" /
> §"Benchmark Results".  Replaces (subsumes) the old root-flat
> `decode_neon_prefix` fast path; the old backend stays as the
> `pivco_p` column for research comparison.
>
> The proposal text below is preserved as historical record —
> everything here matches what was implemented, and the §"Measured
> applicability" and §"Implementation sketch" sections are a
> faithful description of how the shipped code behaves.  The
> predicted gains matched reality — see README.md for the landed
> numbers.

The full-tree flat case (`min_len == max_len`) had a shipped fast path
in `pivco_huffman_decode_neon_prefix` before this work — the format
stored `M` packed bits per element and decode was a single
`code_to_sym[prefix]` lookup.  Won 2.47× on uniform and 1.18-1.24× on
sparse_* (§3 of PREFIX_RADIX.md).

**The generalisation was a *format change*, not a decode-only
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
| **zipfian** | **69.4%** | unpack D=2..7+ | 0.66× |
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

## ~~PIVCO-Huffman X2 (two-cursor)~~ — tried, dropped (2026-05-07)

Hypothesis: split the BLK batch into two halves with independent
bitstreams, walk the tree in lockstep, issue paired primitive calls
(`partition_8_x2(cursor_A, cursor_B)`) so the OoO core overlaps two
independent dependency chains.  Same ILP idea as huf0's 4-stream, but
applied at the SIMD primitive level.

Implemented across all four backends (scalar, NEON, AVX-512, SSE4.1)
with a canonical wire format: `[len_A:u16] || stream_A || stream_B`,
ceil/floor index rebalancing at every internal node.  Decoder dispatches
via the new `node_type` switch the same as X1; all roundtrip tests pass
across 17 distributions.

**Measured negative result on every uarch tested.**  Decode-throughput
ratio (X2 / X1) on the real-text cluster (`prose_pride`, `english`,
`html_wiki`, `source_c`, etc.) at default BLK:

| host          | uarch                    | x2/x1 avg | range       |
|---------------|--------------------------|----------:|-------------|
| M4            | Apple Avalanche NEON     | 0.85      | 0.81–0.89   |
| Graviton 4    | Neoverse V2 NEON         | 0.76      | 0.73–0.84   |
| Zen 3         | x86 SSE4.1               | 0.78      | 0.67–1.00   |
| Xeon SR       | Sapphire Rapids AVX-512  | 0.79      | 0.73–0.87   |
| Zen 5 (EPYC)  | AVX-512                  | 0.59      | 0.50–0.71   |

### Why it didn't pay off

A focused microbench (`extras/bench_partition_micro.c`, removed with
this commit but recoverable from git) tested `partition_8` /
`partition_32` in three loop shapes — stride-8, stride-16 (2-way
unroll, serial counter chain), paired (independent counter chains) —
sweeping inner loop length from 8 to 1024.

Result: at **decoder-realistic INNER ≈ 91 iters** per node-call:
- M4 NEON paired = 0.24 ns/call vs **stride-16 = 0.16 ns/call** — paired is
  **1.48× SLOWER** than serial-counter 2-way unroll
- Graviton 4 NEON: paired = 0.43 vs stride-16 = 0.32 — 1.33× slower
- Xeon AVX-512: paired = 1.29 vs single = 1.55 — paired *wins* 0.83×, but
  only at L1-fitting sizes; cliff at INNER=256 once 4 streams exceed L1d.

The bottleneck **isn't the counter dependency chain** as the original
hypothesis assumed.  It's the **store-stream count**: stride-16 keeps
writes confined to 2 cache lines (`indices`, `tmp`); paired-X2 spreads
them across 4 (`indices_A`, `tmp_A`, `indices_B`, `tmp_B`).  The 2× store-
buffer / cache-line allocation pressure costs more than the ILP gain.
On AVX-512, `vpcompressw` saturates a single throughput pipe so paired
calls can't even overlap effectively.

### Things to consider if revisiting

- The **architectural pattern** (paired primitives + index rebalance)
  is fundamentally write-stream-bound.  Any future implementation
  needs to reduce write streams, not increase them.
- Possible angle: alternating-slot output where cursor A and B SHARE
  one output buffer at predictable interleaved positions.  Cuts back
  to 2 streams.  Encoder has to know the slotting; downstream consumers
  have to merge or process interleaved.
- AVX-512 had a real bug in the masked-tail `partition_32` call inside
  X2 context — same code worked in X1 single-cursor.  Worked around
  with a scalar tail; root cause not identified.  Worth understanding
  before any X2 retry.

### Reference data

Per-primitive ns/call breakdown was captured for prose_pride at BLK=16384
on M4, Graviton 4, and Xeon AVX-512 via `__builtin_readcyclecounter` /
rdtsc instrumentation.  rebalance memcpy was the single biggest contributor
to the X2-X1 gap on M4 (32% of 4 s of overhead on prose_pride).  The
recursion-edge function-call overhead (which an earlier xctrace profile
had blamed) was actually only ~8% of the gap — xctrace was misleading.

---

## Scatter fusion into partition — tried, reverted (recorded here for future reference)

See README.md §"Stage fusion" and §"Tested and discarded": the
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

## Relaxed / near-flat subtree detection — discarded (2026-05-09)

Considered adding "relaxed" flat-subtree detection where a subtree is
treated as flat-at-depth-D even when a few leaves are at depth D-1
or D+1.  Two flavours, both bad:

A) **Tree-shape modification.** Force more subtrees to be flat by
   choosing non-Huffman code lengths.  Directly hurts compression
   (Σ freq×len no longer optimal).  Existing
   `pivco_huffman_build_table` already does the *only* compression-
   neutral version: rearrange the tree shape while preserving the
   code-length multiset.  Going further means worse compression.

B) **Almost-flat detection on optimal tree.**  Codes preserved but
   decoder needs to escape from the flat fast path on outlier codes
   (codes that "should have ended" at a shallower depth).  Requires:
   - Per-code outlier marker bit in the bitstream → compression loss
     in format overhead
   - Per-element conditional in the flat decode hot loop → kills
     scatter throughput (we measured this is store-port-saturated;
     adding a branch per elem regresses badly)
   - Format change touching encoder + decoder

The well-optimised recursive path already handles non-flat subtrees
just fine; this would add complexity for unclear win at best, real
loss otherwise.  Discarded.

## AVX-512 byte-scatter (vpscatterdd / vpermb-based) — open question (2026-05-09)

Currently scatter on AVX-512 falls back to per-lane SSE-style
`_mm_extract_epi16` + byte stores (see `scatter_write_avx512`).
AVX-512 has true scatter instructions (`vpscatterdd` etc.) but they
operate on 32-bit lanes, not bytes.  Possible alternatives:

- Pack 4 bytes per 32-bit lane, use vpscatterdd to scatter 16 dwords
  at once.  But the destinations are scattered byte-positions in
  symbols[], not aligned dword-positions.  Doesn't fit naturally.
- vpermb-based gather-then-store: not really a scatter pattern.

Verdict: probably no easy AVX-512 byte-scatter primitive that beats
the current SSE-style extract.  But worth a focused microbench to
confirm before committing more time.  Park as low-priority probe.

## ~~Full-tail masked partition~~ — SHIPPED (2026-05-08, third attempt)

After `771c3ce` re-enabled the half-tail masked variants, the full-tail
remained scalar with two failed retry attempts (scratch+memcpy, hybrid).
Then a one-line user observation — *"can't we just use slightly
overallocated buffers?"* — pointed at the actual fix.

**The bug** (rediscovered): in RIGHT recursion `decode_node` passes
`tmp = indices + parent_n` so indices and tmp share `buf2` with no gap.
`partition_8_left`'s 16-byte vector store includes `8 - nl` filler
zeros past the valid left side; if `n_left + 8 > parent_n` the filler
lands at `tmp[0..)` and overwrites the right-side data
`partition_8_right` just wrote.

**The fix:** caller passes the right child's tmp at `tmp + n_right + 8`
(NEON/SSE) or `+ 32` (AVX-512) — one full vector wide of padding —
between right child's indices and tmp.  Filler now harmlessly lands in
the gap.  Indices buffer also bumped to `BLK + 8` (or `+ 32`) at the
entry point to absorb the top-level partition_8_left's filler.

Padding propagates additively per-RIGHT-recursion-level, so worst-case
total padding is `max_depth × 8 = 16 × 8 = 128` elements (NEON/SSE) or
`128` elements (AVX-512 with × 32 ≈ 512).  `tmp` is sized `2 × BLK`
which has plenty of headroom.

**End-to-end pivco_n change** (10-rep bench, vs `771c3ce`-state baseline):

| host                | real-text avg | all-dist avg | best                    | worst                  |
|---------------------|--------------:|-------------:|-------------------------|------------------------|
| M4 NEON             |        +1.68% |       +2.01% | +4.7% (sparse_16)       | −1.7% (sparse_4)       |
| Graviton 4 NEON     |        +3.32% |       +1.56% | +9.2% (sparse_16)       | −14.5% (uniform)       |
| Zen 3 SSE           |        +7.31% |       +4.12% | +9.6% (bell_s30)        | −0.6% (two_sym_90/10)  |
| Xeon SR AVX-512     |       **+23.17%** |   **+12.37%** | +38.3% (bell_s30)       | −7.7% (two_sym_90/10)  |
| Zen 5 AVX-512       |       **+24.39%** |   **+15.66%** | +52.1% (sparse_16)      | −1.6% (proba80)        |

The AVX-512 wins are huge because `partition_32` processes 32 elements
at once, so masking 1..31 leftover-element tails saves much more wall
than scalar conditional stores.  Validates the original b136b96
"+20-40% AVX-512" claim that turned out to be on incorrect output —
the gain was real, the kernel just needed the right calling convention.

A few negative outliers (Graviton uniform −14.5%, Xeon SR
two_sym_90/10 −7.7%) probably reflect cache-line layout shifts from
the buffer offsets and are workload-specific.  Net is overwhelmingly
positive.

Verified by:
- `pivco_huffman_tests` (4f2fd5c added per-backend coverage)
- bench scalar-reference cksum (da3c9fd) — all 5 hosts report 0
  errors after the fix.

The earlier "parked" entry below is preserved as the diagnostic
trail; would-be re-experimenters can read it to understand why the
naive masked tail was wrong without padding.

### Diagnostic trail (preserved for reference)

**The bug.** In `decode_node_neon`'s RIGHT recursion, the caller passes
`tmp = indices + parent_n` so that `indices` and `tmp` are *adjacent
inside the same buf2 scratch*.  When the right recursion's own
`node_full` reaches its tail, the masked `partition_8_left` stores 16
bytes (= 8 uint16) to `indices + n_left`, including `8 - nl` zero filler
bytes past the valid left-side range.  If `n_left + 8 > parent_n`, the
filler ZEROS land at `indices + parent_n` onwards, which is `tmp[0..]` —
exactly where `partition_8_right` just wrote the valid right-side data.
So the right side gets clobbered.  Algebraically the overlap fires when
`n_right_main < (8 - rem) / 2`, i.e. when the main-loop produced few
right elements before the tail.  Symmetric analysis on the AVX-512 path
explains why proba50 / bell_* / prose_pride / html_wiki failed on every
backend.

**Fixes evaluated, none shipped.**

1. **All-vector with LEFT through scratch + sized memcpy** (writes only
   the `nl` valid uint16_t to `indices+n_left`, never the filler):
   correct on every host.  Bench delta on 10-rep:
       M4         real-text avg  −0.48%   all-dist  −0.35%
       Graviton 4 real-text avg  +1.13%   all-dist  +1.13%
   Memcpy overhead absorbs most of the M4 win.

2. **Hybrid: vector RIGHT + scalar LEFT** (`partition_8_right` is safe
   because its filler lands in the right child's own tmp area which gets
   overwritten or never read; LEFT stays scalar to dodge the indices-
   vs-tmp overlap):
       M4         real-text avg  −4.18%   all-dist  −1.78%
       Graviton 4 real-text avg  +1.65%   all-dist  +0.41%
   Mixing scalar and vector in the tail blows up M4 register allocation.

A few-percent Graviton win costing M4 isn't worth shipping behind a
single code path.  Per-uarch dispatch could fix it but at code-cost
disproportionate to the absolute throughput gain (the half-tail is
already only ~5% of decode wall on real text, and the full-tail is even
smaller).

If we ever:
- Add a tiered backend with explicit per-arch fast paths (pivco_n_apple
  vs pivco_n_neoverse), OR
- Find a way to do the LEFT vector store WITHOUT writing filler bytes
  past `indices+n_left+nl` (NEON has no native masked store; could use
  load-blend-store or per-lane `vst1q_lane` but those benchmark slower
  on the few-elem hot path) —

then this is back on the table.  Reproducer logic is simply re-applying
the e9a668f-style masked tail to `node_full`; correctness is verified
by `pivco_huffman_tests` (4f2fd5c added the missing AVX-512/SSE
coverage).  The bench's scalar-reference cksum (da3c9fd) catches the
overlap bug immediately.

## Cross-platform partition+scatter fusion — back on the table (2026-05-09)

**UPDATE 2026-05-09:** the original "parked" verdict below was based on
two methodology errors in the microbench:
  1. Wrong P:S element ratio (1:2 instead of the real-decoder's 4:1)
  2. Random scatter destinations instead of sorted-ascending
     (real-decoder leaves get sorted indices because partition
     preserves order)

A corrected microbench (`extras/bench_fusion_v3_*_cnt.cpp`) sweeping
PpS (partition calls per scatter chunk) ∈ {1,2,4,8} with sorted
ascending scatter indices shows fusion is a substantial win on every
platform — biggest at the realistic PpS=4 or PpS=8:

| host                 | best PpS | PSPS-vs-PP_SS | end-to-end est. |
|----------------------|:--------:|--------------:|----------------:|
| M4 NEON              |    4     |        −24%   |    ~12-15%      |
| Graviton 4 NEON      |    4     |    **−37%**   |    ~22-25%      |
| Zen 3 SSE            |    4     |        −31%   |    ~18-22%      |
| Xeon SR AVX-512      |    8     |    **−37%**   |    ~22-25%      |
| Zen 5 AVX-512        |    4     |        −32%   |    ~20-23%      |

The platforms that show the **biggest** fusion gains are exactly the
ones where pivco currently loses on real text vs huf0_x2 (Graviton 4,
Xeon SR AVX-512 prose, Zen 5).  Well-aligned opportunity.

Mechanics: M4 scatter is store-port-saturated at 1.86/cyc; partition
has 13% utilization with the rest free.  Adding partition's stores to
scatter's store stream saturates the ports without contention up to
~PpS=4-8 (above which front-end / IPC pressure starts dominating).
Graviton/Zen 5 have even more headroom (S_only at 0.99/cyc) so wins
are bigger.

Reproducer:
  - `extras/bench_fusion_v3_cnt.cpp`        (NEON)
  - `extras/bench_fusion_v3_sse_cnt.cpp`    (SSE4.1)
  - `extras/bench_fusion_v3_avx512_cnt.cpp` (AVX-512)
  - sweep saved at
    `results/microbench-20260509-fusion-v3-all-platforms.txt`

### Implementation paths (open after MINIMAL TEST 2026-05-09)

**Update: minimal cross-block lookahead test (2026-05-09)**

Implemented the smallest-viable fusion test: thread-local lookahead
state pointing at next block's root partition, hook in 3 NEON scatter
inner loops (`scatter_sym`, `scatter_both_leaves`, `flat_decode_scatter`
D=2 / D=4) calling `partition_root_8(K)` per 16-elem store chunk.
Driver: `pivco_huffman_decode_dual_neon` decodes blocks A then B with
lookahead during A's scatters covering B's root partition.

Source: `src/pivco_huffman_neon.c`, public API in `pivco_huffman.h`.
Test/bench: `extras/bench_dual_decode_test.c`.

End-to-end results on M4 (200K iters per distribution, K swept 2/4/8/16):

| dist          | serial M/s | dual M/s | delta  |
|---------------|-----------:|---------:|-------:|
| english       |     3471   |   3505   | +0.99% |
| prose_pride   |     2849   |   2865   | +0.57% |
| source_c      |     3128   |   3085   | −1.38% |
| html_wiki     |     2668   |   2677   | +0.35% |
| json_api      |     2822   |   2800   | −0.80% |
| log_apache    |     2708   |   2733   | +0.92% |
| csv_numeric   |     3920   |   3953   | +0.81% |
| chinese_text  |     2843   |   2858   | +0.55% |
| dna_fasta     |     4214   |   4244   | +0.71% |
| bell_s30      |     2659   |   2645   | −0.55% |
| zipfian       |     2816   |   2847   | +1.10% |

Mean: ~+0.3%.  Range: −1.4% to +1.1%.  **Essentially noise.**

The 12-15% microbench prediction does not translate.  Reasons:

1. The microbench measures kernels in **one tight inner loop** —
   `partition_8` and `scatter16` with all instructions visible to OOO
   simultaneously.  Real decoder has these in **different inline
   functions** at different recursion depths, separated by call-frame
   boundaries that the OOO window can't span efficiently.
2. The hook adds a branch + thread-local load per scatter chunk; even
   when inactive, this costs ~1 cyc per chunk, eating into baseline.
3. `partition_root_8` work for B is fundamentally still being done —
   shifted from "after A" to "during A's scatter" — but the OOO
   doesn't actually pipeline it; the calls execute serially within
   each scatter chunk.

**Conclusion:** the microbench-style fusion gain requires kernels to
be **co-resident in a single inner loop body**, not just both called
from the same function.  Realising this in the real decoder needs
much deeper restructuring — a true *fused kernel* per (parent, leaf-
type-of-child) shape that interleaves partition stores and scatter
stores in one tight loop.  That is option B below (within-block
sibling fusion), which is high engineering for combinatorial kernel
zoo.

Recommendation: keep the dual decode infrastructure as a parked
experiment.  Don't ship it as a perf optimization.  Don't pursue
the within-block fused-kernel option without first sketching the
specific (parent, leaf-pair) shapes most worth fusing — and even
then expect realised wins to be a fraction of the microbench
prediction.

Microbench prediction held for the kernel itself; the GAP between
microbench and realised is the cost of the recursive control flow.

The recursive structure means partition at depth N produces indices that
scatter at depth N+1 reads — same data, different time, OOO can't
pipeline across function calls.  Two paths:

**A) Cross-batch (block-level) fusion** — decode 2 blocks in lockstep
through a state machine, with block A's scatter stages overlapping
block B's partition stages.  Independent buffers; no producer-consumer
chain across the fusion boundary.  Predicted realisable win: ~half of
the microbench gain (some of the headroom is already captured by OOO
within a single block's recursion).  Engineering: medium — requires
restructuring `decode_node_neon/x86/avx512` from recursion to an
explicit work queue.

**B) Within-block sibling fusion** — at each `node_full` parent whose
children are both terminal (one partition + one leaf-scatter, or one
flat-scatter + one half-partition), inline both into a single fused
kernel.  Engineering: high — combinatorial number of sibling-pair
shapes × per-uarch tuning.  Per-pair captures more theoretical win
but only fires on specific tree topologies.

Recommendation: **prototype A first on Graviton 4** (where the +20-25%
estimate would close most of the prose-text gap vs huf0_x2).  Measure
end-to-end before generalising.

### Original parked microbench notes (kept for diagnostic trail)

Built a 6-platform microbench sweep (`extras/bench_micro_{sse,avx512}_cnt.cpp`
and the existing NEON ones) measuring whether running `partition_8`/`p32`
in the same inner loop body as a 16-elem `scatter_sym` saves cycles vs.
running them serially.  Hypothesis: partition's store-port slack (it's
not store-saturated on any uarch) should overlap with scatter's
store-bound work.

Results — fusion saving (sum-of-parts minus serial), positive = win:

| platform              | partition cyc/call | scatter stores/cyc | fusion saving |
|-----------------------|:--:|:--:|:--:|
| Apple M4 NEON         | 2.58 (8e)          | 1.86 (peak)        | **−10%** (hurts)  |
| Graviton 4 NEON       | 5.95 (8e)          | 0.57               | **+6%**           |
| Zen 3 SSE4.1 (c6a)    | ~3.8 ns/call       | 0.51 ns/store      | ≈ 0% (TBD with sudo) |
| Xeon SR SSE4.1 (c8i)  | 3.79 (8e)          | 0.56               | **−18%** (hurts!) |
| Xeon SR AVX-512 (c8i) | 6.09 (32e)         | 0.56               | **−18%** (hurts)  |
| Zen 5 SSE4.1 (c8a)    | 3.88 (8e)          | 0.97               | **+10%**          |
| Zen 5 AVX-512 (c8a)   | 10.01 (32e)        | 0.98               | **+14%**          |

**Wins on Zen 5 + Graviton 4. Hurts on M4 + Xeon SR.**  Apple M4
already runs scatter at the 2-stores/cyc port ceiling so there's no
slack to absorb partition; Sapphire Rapids interferes destructively
(probably store-buffer / µop-cache pressure on the larger combined
kernel, evident also in `dual_indep` running 4.09× single rather than
2× — Xeon SR has a specific scheduling hazard around `vpcompressw`
stores).

Decision: **not shipping.**  An optimization that helps two uarches by
+10/+14% but hurts two others by −10/−18% can't go behind a single code
path, and the engineering cost of per-uarch dispatch (we'd need
runtime detection plus separate `decode_node_fused` variants for at
least the AVX-512/SSE4.1/NEON×{Apple,Neoverse-V2,Zen}-class branches)
is too high relative to the absolute win.  The sibling fusion would
also need careful tree-shape pre-classification (the win only applies
when one child is a leaf-scatter and the other is a partition — see
node_type already in `pivco_huffman_table_t`), further multiplying the
case matrix.

If the platform mix changes — e.g. Zen 5 becomes the dominant target,
or we add a "tiered" backend with explicit per-arch fast paths — this
is back on the table.  Reproducer is `extras/bench_micro_sse_cnt.cpp`
and `extras/bench_micro_avx512_cnt.cpp`; raw sweep at
`results/microbench-20260508-x86.txt` and
`results/microbench-20260508-c8g.txt`.

Drive-by findings from the same sweep, recorded here so we don't redo
the experiments:
- **Multi-cursor partition (2-cursor / 4-cursor / X2 family) is dead
  on every uarch tested.**  `dual_indep` cost ≥ 2× single everywhere.
  No spare back-end parallelism to extract; OOO already finds it all.
  Independently confirms the prior X2 drop (commit 20f9b33) at the
  microbench level.
- **Scatter store-throughput tier list** (stores/cyc on byte-scattered
  16-store loop): Apple M4 1.86 (port-saturated), Zen 5 0.97, everyone
  else ~0.56.  Apple Silicon and Zen 5 have meaningfully better
  scatter throughput than Sapphire Rapids / Neoverse V2 / Zen 3.
- **Lane-extracts are free everywhere we measured** — halving them
  (variant D of bench_scatter_split_cnt) changed nothing.  The
  bottleneck is purely the byte stores.

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
this track. See README.md for full numbers.

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
thermal-drift noise (CPU freq drift alone covered the unpack).

Interpretation: M4's OoO engine was already renaming through the
textual serialization.  The bottleneck of `pivco_n` at ~0.25 c/elem is
elsewhere — likely TBL/store-port throughput or instruction fetch
width, not this dep chain.  Change doubled the loop body line count
for no measurable win, so reverted.  Recorded here so the next person
reading the assembly output doesn't repeat the experiment.

## ~~Mix scalar `str d` with `str q` to use a 2nd store AGU on M-series~~ — tried, slower

`partition_8`'s two `vst1q_u8` stores account for ~38 % of decode CPU
on M4 (per the `prose_pride` xctrace profile).  Hypothesis worth
testing: M-series chips reportedly have separate SIMD-store and
scalar-store dispatch (1 SIMD store/cycle + extra scalar AGU
capacity), so replacing one or both `str q`s with paired `str d`s
might hit a 2nd dispatch port and run faster.

Variants benchmarked (`bench/bench_micro.c`, "store-port topology probe"):

  - `simd_only`    — 2× `str q` per iter (baseline, current code)
  - `mixed`        — 1× `str q` + 2× `str d` per iter (forced via
                     inline asm; without `volatile` the compiler
                     fuses `str d, str d` back to `str q`)
  - `scalar_only`  — 4× `str d` per iter (also forced via asm)

Apple M4 Max @ 4 GHz, partition_neon-shape inner loop, 4 stable runs:

| variant      | stores/iter | bench GB/s | ns/iter | cycles | stores/cyc |
|--------------|------------:|-----------:|--------:|-------:|-----------:|
| simd_only    |           2 |       15.6 |    0.51 |  ~2.05 |       0.97 |
| mixed        |           3 |       14.6 |    0.55 |  ~2.19 |       1.37 |
| scalar_only  |           4 |       13.0 |    0.61 |  ~2.46 |       1.63 |

Two readings:

1. M4 *can* dispatch >1 store per cycle (`scalar_only` issues 1.6
   stores/cycle on average), so the topology probably *is* something
   like "1 SIMD port + extra scalar AGU bandwidth".

2. **But you can't usefully exploit it from partition_8.**  The data
   lives in NEON registers (output of `vqtbl1q_u8`); reaching it via
   `str d` requires `ext.16b v_, v_, v_, #8` to extract the high
   half, which adds a register-rename op and a critical-path cycle
   between `tbl` and the second store.  The "saved" port-issue
   bandwidth is exactly absorbed by the longer dep chain — both sides
   together cost ~0.4 extra cycles per iter, which matches the
   measured slowdown (2.46 − 2.05 = 0.41 cycles).

Implication: there's no clever store-port trick that rescues
partition_8 on M4.  The 0.06 ns/elem (= 1 vst1q/cycle) floor is the
true ceiling for this kernel shape; the only way to go faster is to
issue *fewer* stores per element — exactly what `partition_8_right`
(half-tree) and `scatter_both_leaves` (no partition at all) already
do.

Recorded here so the next person profiling the partition store cost
doesn't repeat the experiment.

## ~~Coalesce small-side partition stores into a register accumulator~~ — tried on 3 platforms, lost on all

`partition_8` writes a full 16-byte `vst1q_u8` to both sides every
iter even when the popcount fills only `cnt` of the 8 lanes.  The
natural follow-up to the store-port saturation finding was:
accumulate variable-sized contributions into a register and flush
only when a full 16-byte chunk is ready, halving the store rate.

Six NEON variants + three AVX-512 variants tested
([extras/bench_coalesce.c](extras/bench_coalesce.c) /
[extras/bench_coalesce_avx512.c](extras/bench_coalesce_avx512.c),
full investigation in [COALESCE.md](COALESCE.md)).  Best result per
platform:

| Platform | Best variant | Ratio vs baseline |
|---|---|---:|
| Apple M4 (NEON) | 1-sided macro coalesce | 0.88× |
| Graviton 4 (NEON) | 1-sided macro coalesce | 0.96× (closest) |
| Xeon AVX-512 (Sapphire Rapids) | 2-iter macro coalesce | 0.75× |

Each variant ruled out one suspected failure mode and exposed the
next: indirect-branch mispredict (switch) → cross-iter dep chain
(per-iter TBL) → SIMD-throughput-bound place-shift cost (4-iter
macro) → ditto with OR-tree balancing (no help — it's throughput,
not latency).  The cleanest negative result is half-partition
coalesce on Graviton 4 with balanced OR-tree (1 store/iter
baseline, branchless macro, depth-2 OR-tree, on the platform with
the more favorable SIMD-to-store ratio): still loses by 23%.

**Bonus AVX-512 finding:** `_mm512_mask_compressstoreu_epi16` (the
single-instruction compress-and-store) is **2× SLOWER** than the
production `vpcompressw` + `vmovdqu64` — Sapphire Rapids decomposes
it to ~6 µops vs production's 2 µops.  Don't migrate the AVX-512
backend to compressstoreu.

This isn't a clever-coding problem — it's resource balance on every
modern uarch tested.  The kernel is at a Pareto-optimal point: any
non-trivial SIMD work added to enable store reduction immediately
re-balances the bottleneck.  Untested: Zen 3 SSE4.1 (would need a
new port; expected to lose given the cross-platform pattern).

See [COALESCE.md](COALESCE.md) for the full investigation.

## ~~SSE root both-leaves vectorisation~~ — SHIPPED

**Status (2026-04-27): SHIPPED.**

`pivco_huffman_decode_x86` had a scalar byte-by-byte loop for the
"root both children leaves" case (`two_sym_eq` / `two_sym_90/10`).
Replaced with a 16-output-bytes-per-iter SSE4.1 path using
`pshufb` (broadcast each bitmap byte to 8 lanes), `pcmpeqb`
(bit-to-byte-mask), and `pblendvb` (sym0/sym1 select).

A/B headline (Zen 3, paired-t over 5 alternated pairs, see
[`results/SSE_BOTH_LEAVES-AB-20260427.md`](results/SSE_BOTH_LEAVES-AB-20260427.md)):

- two_sym_eq: 1507 → 22677 M/s (**+1405%**, t=357)
- two_sym_90/10: 1502 → 22660 M/s (**+1409%**, t=197)

Bonus (codegen, not direct from this patch): uniform +71.8%,
gzip_random +73.1% — both restored to morning's pre-rename level.
The larger parent function changes how the compiler inlines/schedules
the inlined `flat_decode_direct_x86` path; net positive.

Other distributions: ±1.5% noise, no significant losses.

## ~~AVX-512 leaf-fusion port~~ — SHIPPED

**Status (2026-04-27): SHIPPED.**

`decode_node_neon` and `decode_node_x86` had stage-fusion logic for
shallow internal nodes (both-leaves dispatch + half-partition for
prefilled-leaf side).  `decode_node_avx512` was missing all three —
flagged by the Codex review.  The AVX-512 helpers `partition_32_right`
/ `partition_32_left` already existed but weren't called from the
dispatcher.

This port adds `scatter_both_leaves_avx512` (scalar STRBs since AVX-512
has no byte scatter) and the three early-return branches before the
standard full-partition path.

A/B headline (Xeon 6975P-C, paired-t over 7 alternated pairs, full
write-up in [`results/LEAF_FUSION_AVX512-AB-20260427.md`](results/LEAF_FUSION_AVX512-AB-20260427.md)):

| Distribution | Δ | t | sig |
|---|---:|---:|:--:|
| source_c    | +7.5% | 8.0 | ! |
| english     | +6.1% | 6.2 | ! |
| html_wiki   | +4.1% | 3.7 | ! |
| log_apache  | +2.2% | 3.0 | ! |
| proba80     | +4.2% | 3.9 | ! |
| proba02     | +4.3% | 8.2 | ! |
| bell_s10    | +3.5% | 3.2 | ! |
| bell_s30    | +3.2% | 6.3 | ! |
| zipfian     | +3.2% | 5.4 | ! |

9 wins reach p<0.05, 0 losses reach p<0.05.  Real-text and
moderately-skewed distributions (the hypothesis target) all gain
+3-7%; flat-only distributions (`sparse_*`, `two_sym_*`, `flat_M*`,
`uniform`) unchanged within noise as expected (those don't trigger
the new paths).

**`PIVCO_BENCH_QUICK`** was added in the same session ([commit
`7bbfc8e`](../)) to make this kind of A/B fast — 5 min for the full
verification on Xeon, vs ~70 min if we'd had to run full sweeps.

## Tree-walk node-size histogram (prerequisite for tiny-node fast paths)

**Status: idea, not yet measured.  Codex review item #6.**

The recursive decoder calls `decode_node_*` at every internal node.
For each call: read bitmap, run the 32/16/8-wide partition loop (with
scalar tail for `n % 8`), do scratch management, recurse into both
children.  At small `n` (deep flat-corner nodes) the fixed per-call
overhead can dwarf the actual partition work.  Codex suggests adding
tiny-node fast paths:

- `n ≤ 8` direct scalar partition (no TBL setup, no scratch)
- `n ≤ 16` simplified vector partition
- tiny-both-leaves variant — *already covered by the leaf-fusion paths
  shipped today (commits `a36546c`, `1a2dadd`)*
- tiny one-leaf-prefilled variant — *also covered*

**The data we need first.**  An n-distribution histogram of recursive
`decode_node_*` calls per workload, weighted both by call count and
by elements processed (= time proxy):

```
prose_pride NEON tree-walk node sizes (illustrative, not measured):
  bucket    calls   %       n×calls  time%
  [1..3]    48000  24.0%      96000   1.2%
  [4..7]    52000  26.0%     260000   3.3%
  [8..15]   32000  16.0%     320000   4.0%
  [16..31]  24000  12.0%     528000   6.7%
  [32..63]  18000   9.0%     864000  11.0%
  [64+]     26000  13.0%    3360000  73.8%
```

If the smallest buckets dominate by call count but contribute <5% of
elements processed, tiny-node paths reduce call overhead but won't
move element-throughput much.  If [4..15] is a big slice of *time*,
sub-32 fast paths (this section + the AVX-512 small-node-tail entry
below) are worth the source complexity.

**Implementation sketch:** ~30-line instrumentation behind
`-DPIVCO_INSTRUMENT_NODE_SIZES` in `decode_node_neon` /
`decode_node_avx512` / `decode_node_x86`, increment
`static uint64_t node_size_hist[BLOCK_SIZE+1]` at function entry, dump
at exit.  Run with `PIVCO_BENCH_QUICK=1` on prose_pride / english /
source_c / bell_s30.  ~1 hour to measure, then decide whether to
proceed with tiny-node paths or with the AVX-512 8-wide-tail idea
below.

**Risk of skipping the histogram and shipping tiny-node paths blind:**
50–100 lines of branchy code that adds an `if (n ≤ K)` check at every
internal node call.  If the call frequency is low, the dispatcher
overhead wipes out the per-call savings — could be net-negative on
real-text where most time is in larger-n nodes.

## ~~AVX-512 small-node tail — masked vector partition~~ — SHIPPED (2026-05-07)

Replaced the three scalar `for (; j < n; j++) { bitmap_get(...); ... }`
tails in `decode_node_avx512` (left-only, right-only, full split) with a
single masked `partition_32_*` call: load remaining `bm` bytes into a
`uint32_t`, mask off bits beyond `rem = n - j`, run one
vpcompressw-based partition.

A/B on test-c8i (Xeon) and test-c8a (Zen 5 EPYC 9R45) — gcc-14, BLK=8192,
5 alternated rounds × repeats=20 (results in
[`results/avx512-masked-tail-20260507/`](results/avx512-masked-tail-20260507/SUMMARY.md)):

| host  | typical real-text delta | notable                                      |
|-------|-------------------------|----------------------------------------------|
| c8i   | +20 to +40%             | flat/sparse/two_sym ~0% (correctly untouched) |
| c8a   | +17 to +40%             | two_sym_eq -5.9% (Zen-5-only, vpcompress overhead at depth-1 trees) |

The win is much bigger than "fix a tail" suggests because the partition is
*recursive*: with stride=32 and BLK=8192, the deepest 30-40% of internal-node
calls have `n < 32` and therefore run **purely** through what we called the
"tail".  At those depths the bm bits are essentially random on real text, so
the old scalar loop was eating ~15-20 cycles/element on branch mispredict.
Masked vector partition collapses that to ~1 cycle/element regardless of `rem`.

### Follow-ups

- **Re-test with larger BLK.**  At BLK=16384 or 32768 the recursion still
  fans out to small-n internal nodes (nature of Huffman trees), but the
  fraction of work spent at those depths drops because the upper levels
  process more elements.  Expected: smaller real-text delta, but the
  patch should still be a free win or break-even.  Worth measuring to
  inform "should we raise BLK on AVX-512?"
- **Port to NEON / SSE.**  Both backends have the same shape:
  `src/pivco_huffman_neon.c` `partition_8` stride-8 + scalar tail (search
  for `for (; j < n; j++)` near `n_right += partition_8_*`), and
  `src/pivco_huffman_x86.c` `partition_8_sse` ditto.  Stride-8 means the
  tail is at most 7 elements per call (vs 31 on AVX-512), so the per-call
  speedup is smaller — but the *number* of small-n recursive calls
  (those with `n < 8`) is similarly large for real text.  Mechanism:
  load remaining `bm` byte, mask to `(1u << rem) - 1`, run one
  `partition_8_*`.  Probable shape of win: smaller percentages than
  AVX-512 (because stride-8 already captures most of the partition work)
  but still nonzero on the M4/Graviton/Zen 3 real-text cluster.
- **two_sym_eq Zen-5 regression.**  Single-deep tree, masked partition
  adds vpcompress dependency for negligible work.  If we care, add a
  `if (n < 8) scalar` micro-fallback — but two_sym_eq is already 5.9 GS/s
  post-patch so probably not worth a branch.

## Hybrid block decoder — recognised escape hatch, intentionally not pursued

> **Update (2026-05-09):** the hybrid is the *easy way out* — pick the
> faster of pivco / trad_4s / huf0_x2 per block and you trivially never
> lose on real text outside Apple silicon.  We deliberately don't want
> to do this:
>
> - It doesn't move the science forward — the interesting question is
>   whether tree-walk / pivot-style decode can be made competitive on
>   real text on every uarch on its own merits.
> - Once you start picking-the-winner you stop debugging the loss.
>   The Graviton/Zen 3/Zen 5 prose gap remains the open problem we
>   should keep banging on (fusion, structural changes, etc.).
> - Worth mentioning in the paper as a deployment-engineering
>   fallback ("if you ship this in production, do hybrid"), not as
>   the research result.
>
> Keep the idea recorded for completeness; do not work on it.

> **Status (as of [a1e742cded9f059d4e165177deca1ac8e26ba49e](../),
> 2026-04-25):** the 2026-04-25 mega sweep added 10 real-world byte
> distributions (Wikipedia HTML, Project Gutenberg prose, JPEG, JSON,
> source code, Apache log, FASTA, CSV, gzip output, classical Chinese
> text — see [`extras/datasets/`](extras/datasets/)).  PIVCO loses
> badly on the real-text cluster outside Apple silicon:
>
> - Xeon AVX-512: `prose_pride` 0.90×, `html_wiki` 0.92×,
>   `json_api` 0.97× (3 losses).
> - Graviton 4: real-text cluster at 0.63–0.78× (8 losses).
> - Zen 3 SSE4.1: real-text cluster at 0.44–0.55× (9 losses).
>
> Apple M4 still wins all 29 distributions (1.08–10×).
>
> The synthetic `english` distribution (1.33× M4 / 1.24× Xeon) was
> overstating the win by ~25-33% vs real prose.
>
> ~~Hybrid is now the highest-EV outstanding optimisation~~ - see
> 2026-05-09 update at the top of this section.

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

## PDEP/expand-based per-position code reconstruction — researched 2026-05-10, parked

**Status:** documented, not implemented.  Conditionally promising on
**Xeon (AVX-512 VBMI2)** but requires non-trivial encoder+decoder rewrite.
NEON cannot benefit (no bit-PDEP, no `vpexpandw` analogue on Apple/G4).

### The idea

Instead of decoding via tree-walk partition + indexed scatter
(`symbols[indices[i]] = sym`), reconstruct the **full Huffman code per
output position** as a contiguous `uint16[N]` array, then do a
sequential c2s lookup with **scalar-stride sequential stores**:

```c
for (i = 0; i < N; i++) symbols[i] = c2s[codes[i]];
```

Sequential output writes are dramatically cheaper than the scrambled
scatter (the partition-reordered indices) on x86, where the L1 store
queue can't coalesce scattered byte stores.

### How reconstruction works (bottom-up)

Walk the Huffman tree **bottom-up**.  Each subtree maintains a 16-bit
per-element code-so-far representation.

At an internal node combining left (n_l elements, D_l-bit codes) and
right (n_r elements, D_r-bit codes), with bm selecting between them:

```c
out_left  = _mm512_maskz_expand_epi16(~bm_chunk, left_codes)    // place at bm=0 positions
out_right = _mm512_maskz_expand_epi16( bm_chunk, right_codes)   // place at bm=1 positions
prefix    = bm_chunk-broadcast-to-16-bit << D_max;              // add the new prefix bit
out       = out_left | out_right | prefix
```

`vpexpandw` (AVX-512 VBMI2) is the key primitive: for each set bit in
the mask, take the next 16-bit word from source and place it at that
output position; for each unset bit, write 0.  Exactly what we need.

At leaves the code is empty (0 bits).  After D combines we have N
codes of D bits each, contiguous in memory.

### Cost analysis (Xeon Ice Lake+ / Zen 4+)

Work per `vpexpandw` combine = (subtree size in words).  Total work
across all combines = sum over symbols of `count[s] * depth[s]` = N ×
entropy.

For `prose_pride` N=8192 with average code length ~4.5 bits:
```
   reconstruction ≈ 8192 × 4.5 = 37000 word-ops
   vpexpandw throughput ≈ 32 words/cycle
   → ~1150 cycles ≈ 380 ns/block
```

Then the contiguous c2s lookup cost depends on table size:

| max code D | c2s size | Lookup | Cost on Xeon |
|---|---:|---|---:|
| ≤ 6 | 64 B | `vpermb` (single reg) | ~50 ns |
| ≤ 8 | 256 B | 4× `vpermb` + blend | ~150 ns |
| ≤ 11 (default after this commit) | 2 KB | `vpgatherdd` from L1 | ~1370 ns |
| ≤ 15 | 32 KB | gather + L1 misses | ~3000+ ns |

### Headline projection vs current

| approach (Xeon prose_pride) | reconstruct | c2s | total | vs current ~2500 ns |
|---|---:|---:|---:|---:|
| current (partition + scatter) | ~1000 | ~1500 | 2500 | 1.0× |
| **BU reconstruct + D≤11 gather** | 380 | 1370 | **~1750** | **1.4×** |
| BU reconstruct + D≤8 vpermb | 380 | 150 | **~530** | **5×** |

At D=11 (now our default): ~1.4× decode speedup, **ABI-compatible
with huf0's 11-bit canonical tables** — could be marketed as
"a faster huf0".  At D=8: 5× speedup at a small ratio cost.

### Architectural fit

- **Xeon Ice Lake+** (`vpexpandw` available): primary target.  ~1.4-5×
  speedup depending on D cap.
- **Zen 4+**: same — has VBMI2.
- **Zen 3**: no `vpexpandw` — would need byte-wise emulation; unclear
  whether positive net.
- **Apple M4 / Graviton 4 (NEON)**: no equivalent instruction.  Would
  need scalar `pdep` emulation per-byte from a 256-entry table, which
  benchmarks at ~0.27 ns/A-bit (12× slower than BMI2).  Stay with the
  existing partition+scatter decoder.
- **Graviton 3+ / Neoverse V1/V2**: SVE2 has `bdep` (bit deposit), which
  matches BMI2 PDEP.  Untested.

Runtime backend selection per architecture (already in place).

### Why the D=11 default (this commit) helps even without implementing

Independent of whether we ever build the BU-reconstruct path, capping at
D=11 means `decode_sym/decode_len` shrink from 32KB to 2KB and fit L1
cleanly.  The existing scatter decoder gets a small win on text-like
data (+2-10% on prose/html/source across hosts), and the c2s table for
a *future* BU-reconstruct decoder fits in L1 as a 2KB gather target.

### Variable-length codes wrinkle

PIVCO trees have variable depth.  At each combine, normalize to the
deeper child's bit-width by padding the shorter child's codes with
zeros at the top bit positions.  In transposed (per-bit-bitmap)
representation this is free; in packed (per-element-code) representation
we already work in fixed 16-bit cells so it's also free as long as
unused high bits stay zero.

The leftover question is: for elements whose code length L_i < D_max,
the bits beyond L_i are "don't care" — they must be masked or
handled in the c2s table by replicating the symbol across all 2^(D_max-L_i)
table entries that share the relevant prefix.  Standard canonical-Huffman
trick.

### Why parked

- Cost-benefit unclear without a full implementation: the +1.4× at
  D=11 is real but not dramatic.  Real wins kick in only at D≤8,
  which costs ~1-3% compression on text — a different trade-off.
- AVX-512-VBMI2-only.  Half the deployed Xeon fleet is older.
- Format change: need to design the bitmap layout for bottom-up
  traversal, build a new encoder pathway.
- Existing decoder already beats huf0 by 1.0-5× depending on
  distribution; no urgent bottleneck.

If revisiting: prototype on test-c8i first (Sapphire Rapids), 1-2 hrs
to validate the 1.4× claim.  If real, the design pivots from
partition+scatter to BU-reconstruct + sequential c2s with runtime
backend selection per architecture.

### Related microbench primitives (`extras/bench_primitives_*_cnt.cpp`)

Standalone benches for the four bit-manipulation primitives that came
up during this exploration:

| primitive | Input | Output | M4 NEON | G4 NEON | Zen3 | Xeon |
|---|---|---|---:|---:|---:|---:|
| **P1** select_vec | V[N] uint8 + char C | uint16 indices where V[i]==C | 0.058 | 0.130 | 0.154 (SSE) | **0.033** (AVX-512 vpcompressw) |
| **P2** pdep_bm | bitmap A (N b), B (popcount(A) b) | C[i] = A[i] ? B[rank(A,i)] : 0 | 0.265 | 0.392 | **0.022** (BMI2) | **0.017** (BMI2) |
| **P3** pdep_idx | A, B as P2 | uint16 indices where C[i]==1 | 0.309 | 0.455 | **0.189** (BMI2+tzcnt) | **0.099** (BMI2+tzcnt) |
| **P4** interleave | A, B as P2 | C[2i]=A[i], C[2i+1]=A[i]?B[rank]:0 | 0.288 | 0.431 | **0.053** (2× PDEP) | **0.034** (2× PDEP) |

(all in ns per input unit — P1 per V-byte, P2-P4 per A-bit)

Key asymmetry: **BMI2 PDEP is 12-15× faster than NEON's table-driven
emulation**.  Any future algorithm that wants to phrase a hot loop as
a deposit operation should expect huge x86-vs-ARM asymmetry.

## ~~Bottom-up tree_merge decoder: 2× unroll + SIMD popcount_K_right~~ — SHIPPED (2026-05-10)

**Status (2026-05-10): SHIPPED.**  Commits `8f03fac`, `462db14`, `a962789`.

Two NEON optimisations to the brand-new bottom-up decoder (`b2e6c38`,
`0e7a41e`), informed by the first round of `bu_*` prof instrumentation
(`8032c0e`).  Both followed straight from a prof breakdown showing
`bu_tree_merge` at 53% of wall and `bu_popcount_K` at 29% with a
scalar `__builtin_popcount` byte loop.

### 1) 2× unroll on `tree_merge` and `tree_merge_bcast_{left,right}`

The TD `node_full` has had a stride-16 unrolled main loop for ages —
adjacent 8-elem groups have independent loads / TBLs / stores so OOO
overlaps iteration N's TBL with iteration N+1's load.  The BU merge
primitives were stride-8 only.  Applied the same unroll pattern:
stride-16 with two independent merges per iter, stride-8 mop-up,
scalar 1..7 tail.  (`merge_both_const` already had it.)

`bu_tree_merge`: 184.8 → 176.2 ns/call, 0.066 → 0.063 ns/elem.

### 2) SIMD `popcount_K_right`, 4-wide ILP

The helper counts "1" bits in the first K bits of a bitmap — called
at every HALF / INTERNAL_FULL node to find K_right before recursing.
At 14 calls/blk × ~218 bytes/call on prose_pride the scalar loop
cost **28.76% of BU wall** (802 ns/blk).

64-byte main loop with 4-wide ILP: four independent `vcntq_u8`,
three-level lane-wise add tree via `vaddq_u8` (~4/cycle on M-series
vs ~2/cycle for `vpaddq_u8` — same data-reduction shape, simpler op
and equivalent or faster across ARM cores), one `vpaddlq_u8` widen
to u16 (max 64 per lane), one `vaddq_u16` into the u16x8
accumulator.  16-byte mop-up for the trailing 1..3 vectors, scalar
tail for the final 0..15 full bytes plus the optional partial last
byte (`K & 7` valid bits).

`bu_popcount_K`: 57.1 → 7.5 ns/call (7.6×), 0.033 → 0.005 ns/elem,
**28.76% → 5.26% of wall**.

### Headline (M4, prose_pride, ns per block)

| primitive                    | before | after  |     Δ |
|------------------------------|-------:|-------:|------:|
| bu_tree_merge                |   1478 |   1415 |   −63 |
| bu_tree_merge_bcast_left     |    157 |    132 |   −25 |
| bu_merge_both_const          |    154 |    151 |    −3 |
| bu_flat_decode               |    136 |    132 |    −4 |
| bu_popcount_K                |    802 |    105 |  −697 |
| (unaccounted)                |     67 |     62 |    −5 |
| **BU TOTAL**                 | **2802** | **1999** | **−803** |
| TD TOTAL (reference)         |   3002 |   2993 |     — |

BU is now **1.50× TD** on prose_pride.

### bench MAIN-set on M4 (pivco_bu M sym/s vs pivco_n TD)

| dataset       |    TD  | BU before | BU after | BU/TD |
|---------------|-------:|----------:|---------:|------:|
| proba80       |   9764 |      9159 |    13958 | 1.43× |
| english       |   3356 |      4095 |     5487 | 1.63× |
| flat_M5       |  25056 |     25497 |    25033 | 1.00× |
| html_wiki     |   2629 |      2681 |     3754 | 1.43× |
| prose_pride   |   2759 |      2894 |     4063 | 1.47× |
| image_jpeg    |   2844 |      3057 |     3787 | 1.33× |
| json_api      |   2666 |      2672 |     3718 | 1.39× |
| gzip_random   |   5168 |      5181 |     5160 | 1.00× |
| chinese_text  |   2713 |      2961 |     4247 | 1.57× |

(`flat_M5` and `gzip_random` are flat-path and uniform respectively —
neither exercises `tree_merge` or `popcount_K_right`.)

## Bottom-up decoder: store K_right per internal node to skip popcount

**Status:** open follow-up to the SHIPPED entries above.  Estimated
**5–15% additional BU win on real-text distributions, depending on
platform** — biggest on the AVX-512 hosts.

`bu_popcount_K` is fundamental to the current format: at each internal
node we have to count the "1" bits in the bitmap to learn K_right
before we can recurse into the right child.  Both NEON SIMD popcount
(`vcntq_u8` + `vaddvq`, 2026-05-10) and AVX-512 VPOPCNTQ
(`_mm512_popcnt_epi64`, 2026-05-11) have already shrunk this primitive
substantially, but it's still a measurable wall slice — and on AVX-512
hosts where everything else is fast, it's actually a *bigger* relative
share than on slower NEON cores.

The cheap shortcut: at encode time, prepend a 1-byte (or short varint)
`K_right` header immediately before each internal node's bitmap.
Decoder reads one byte instead of running the popcount loop — a single
cache-line touch already paid for the bitmap fetch.

**Cost:** ~1 byte per visible internal node.  prose_pride has ~95
internal nodes per 8192-byte block → ~95 B/block = **~1.2% encoded-
size bloat**.  Lower for skewed distributions (fewer internals),
higher for flat ones (which mostly skip this path anyway via
`bu_flat_decode`).

**Estimated win, per platform** (prose_pride; bu_popcount_K share of
post-optimisation BU wall):

| platform                          | bu_popcount_K | BU TOTAL  | popcnt % wall |  est. BU gain |
|-----------------------------------|--------------:|----------:|--------------:|--------------:|
| M4 (NEON)                         |    105 ns/blk |  1999 ns/blk |   **5.3%**    |  ~5%          |
| Zen 3 (AVX2, scalar 4× POPCNT)    |    235 ns/blk |  3303 ns/blk |   **7.1%**    |  ~7%          |
| Zen 5 Turin (AVX-512 VPOPCNTQ)    |    143 ns/blk |  1387 ns/blk |  **10.3%**    | ~10%          |
| **Xeon Granite Rapids (AVX-512)** |    371 ns/blk |  2557 ns/blk |  **14.5%**    | **~15%**      |

The Xeon win is largest because: (a) everything else on Xeon BU is
already AVX-512–accelerated, so popcount is a bigger relative share;
(b) Xeon ICX/Granite Rapids has a single POPCNT execution port even
after VPOPCNTQ widening, while Zen 3+ have 4 scalar POPCNT ports.
Removing the primitive entirely is the path to making the AVX-512
backend really shine.

**Open design questions:**
- Width: 1 byte covers K up to 255 (block size 256).  For K > 255 fall
  back to a 2-byte short varint.  Most internal nodes already operate
  on K ≪ block_size.
- Header placement: easiest is `[K_right_byte][bitmap_bytes]`.  Slight
  alignment cost — bitmap no longer starts on a byte boundary that's
  a multiple of the bitmap length.  Probably fine.
- Encoder change: trivial, popcount once at encode time per internal
  node.
- Worth a prototype on prose_pride / english before propagating to
  x86 backends.

## ~~NEON `bu_tree_merge`: 16-byte loads + precomputed (nr0, m1) shuf~~ — SHIPPED (2026-05-11)

**Status (2026-05-11): SHIPPED.**  Stride-16 main loop in
`tree_merge` (`src/pivco_huffman_bu_neon.c`) now loads 16-byte
L_full / R_full once per iter, does iter 0 with baseline
`vqtbl1`, and uses `vqtbl2_u8` over the full 32-byte source for
iter 1 with a **precomputed adjusted shuf** indexed by `(nr0, m1)`.

Motivating intuition: `bu_tree_merge` was the dominant BU primitive
on every platform (70% of M4 wall, 64% G4, 44% Zen 3, 34% Xeon).
The previous 2× unrolled path did 4× 8-byte L/R loads per iter to
serve two separate 8-byte merges.  Replacing them with 2× 16-byte
loads + a clever iter-1 shuf cuts load count in half AND eliminates
the per-iter runtime adjustment work.

### The precomputed (nr0, m1) table

`expand_tab_pre[9][256][8]` (18 KB, fits L1d) — for each possible
iter-0 popcount `nr0 ∈ [0..8]` and each iter-1 mask byte `m1`,
the 8-byte adjusted shuf vector for `vqtbl2` over the 32-byte
(L_full, R_full) source.

  - L-lane idx = `expand_tab[m1][k] + (8 - nr0)`  ∈ [(8-nr0)..15]
  - R-lane idx = `expand_tab[m1][k] + 8 + nr0`    ∈ [(16+nr0)..(23+nr0)]

One indexed `vld_8` per iter 1 replaces what would otherwise be ~4
vector ALU ops on the dep chain from `nr0`.

### Numbers (prose_pride, M sym/s; pivco_bu column):

| host | before | after | gain    |
|------|-------:|------:|--------:|
| M4   |  4165  | **4727** | **+13.5%** |
| G4   |  2065  | **2274** | **+10.1%** |

Across the MAIN dist set on M4: english +13%, html_wiki +14%,
prose_pride +14%, json_api +13%, chinese_text +15%, image_jpeg +9%.
proba80 / flat_M5 / gzip_random unchanged (those paths don't hit
`tree_merge`).  G4 wins are similar but slightly smaller.

### Per-primitive profile delta on M4 prose_pride:

  bu_tree_merge ns/call:  176.2 → 145.2  (−18%)
  bu_tree_merge ns/elem:  0.063 → 0.052
  bu_tree_merge % wall:   70.8% → 66.4%
  BU TOTAL ns/blk:        1991 → 1748     (−12.1%)

The win comes from collapsing the iter-1 critical path from
~10 cycles (4 ALU + vqtbl2) to ~5 cycles (load + vqtbl2).

### Alternatives tried and dropped

Two earlier attempts at the same problem (V2: runtime ALU
adjustment of the shuf; V3: `vextq_u8` rotate via 9-case switch on
`nr0`) — both kept here as historical reference for *why* the
precomputed-table approach is the right answer:

- **V2 (vqtbl2 + 4 ALU ops on dep chain)**:
  +0.4–1.8% on M4 (marginal), +4–6% on G4 (Neoverse-V2 has lower
  load throughput so load consolidation helps more there).  Strictly
  worse than V4 (V4 keeps V2's load consolidation and eliminates
  the ALU chain on top).
- **V3 (`vextq` rotate + switch on `nr0`)**:
  DEAD on real text — `−77%` on prose_pride / english / html_wiki /
  chinese_text.  `vextq_u8` requires a compile-time immediate,
  forcing a 9-way switch which the compiler emits as an indirect
  branch.  For near-uniform bitmaps `nr0` is binomial(8, 0.5) →
  basically random → ~85% misprediction.  At ~175 iters/call ×
  8 calls/blk that's 20–30 µs of pure misprediction overhead per
  block.  Only `proba80` survived (its mask bytes cluster around
  `nr0=6..7`, so the branch IS predictable).

Branchless V3 variants we considered and rejected before settling
on the precomputed table:

- `vqtbl1q_u8(L_full, iota + (8-nr0))` for the shift → adds 1 TBL
  per side, 3 TBLs total per iter (worse than V2's 1 vqtbl2).
- `vqtbl2q_u8` over `(L_full, R_full)` with a runtime-built index +
  baseline `vqtbl1` for the merge → 2 TBLs per iter on the dep
  chain, same critical path as V2.
- `vbslq` blending all 9 precomputed `vextq` outputs → 8 blends,
  far too expensive.

The shipped version (precomputed-table) is strictly better than
all of these.  The remaining open angle is whether the same trick
applies to the bcast_left/bcast_right merges (one side is a
constant, so the L or R "shift" is degenerate) — probably yes for
similar but smaller wins on a smaller wall slice.

## Profiling overhead is HUGE on x86 and varies 60× across platforms — never quote prof numbers as headline (2026-05-11)

**Status: measured & documented.**  Treat `pivco_huffman_profile_english`
output as RELATIVE only.  All headline performance numbers must
come from the bench harness built WITHOUT `-DPIVCO_PROF=1`.

### Why this matters

Our `PROF_TIC` / `PROF_TOC` macros wrap each primitive with a pair of
cycle-counter reads (`mrs cntvct_el0` on ARM, `rdtsc` on x86) plus a
3-store accounting block.  As the decoder got faster this year (BU
+50% on Graviton 4, +67% on M4, BU/TD ratio 2.94× over huf0 on
Granite Rapids on prose_pride), the per-primitive work got close
enough to the prof overhead that the overhead became visible — and
on some platforms it became LARGER than the work.

### Microbenched cycle-counter cost (empty loop, isolated)

  M4   cntvct_el0       0.0 ns/read    (Apple userspace — free)
  c3   rdtsc Ivy Bridge 7.1 ns         (oldest x86)
  c4   rdtsc Haswell    6.5 ns
  c5   rdtsc CascadeL   6.3 ns
  c8a  rdtsc Zen 5      8.3 ns
  c6a  rdtsc Zen 3     10.0 ns
  c8g  cntvct_el0 G4   10.0 ns         (Linux ARM, NOT free like Apple)
  c8i  rdtsc GraniteR  10.2 ns         (worst Intel — added serialisation)

A PROF_TIC+TOC pair is 2 of these reads + 3 stores: roughly
**14–22 ns on every non-Apple platform**.

### Microbench vs reality — OOO hides RDTSC differently

Real prof overhead in the decoder vs predicted from microbench × call count:

  c8i Granite Rapids   actual ≈ predicted   (RDTSC mostly serial here)
  c3/c4/c5 older Intel actual ≈ predicted
  c6a Zen 3            actual ≈ 0.5× pred  (OOO hides half)
  c8a Zen 5            actual ≈ 0.2× pred  (excellent OOO)
  c8g Graviton 4       actual ≈ 0.25× pred (aggressive ARM OOO)
  M4                   actual ≈ 0          (CNTVCT is free in renamer)

Granite Rapids is the worst case — its RDTSC is more serialising
than any other modern core.

### Measured prof overhead on pivco_bu wall (prose_pride, MAIN sweep avg over text dists)

  host           overhead    BLK   noprof bu (M sym/s)    prof bu (M sym/s)
  M4             +0.3%       8192    4756 prose_pride        4754
  c8g G4         +5.5%       8192    2236                    2126
  c8a Zen 5      +8.1%       8192    6217                    5846
  c6a Zen 3     +11.4%       4096    1456                    1299
  c3  Ivy Bridge +14.1%      4096     952                     832
  c4  Haswell   +15.7%       4096    1149                     984
  c5  Cascade   +15.9%       4096    1191                    1024
  c8i Granite R **+61.6%**   8192    5204                    3230

  flat_M5 and gzip_random show <1% overhead on every platform —
  those distributions skip tree_merge entirely (flat path takes
  over), so very few PROF_TIC/TOC calls fire.

### Why c8i blows up

prose_pride at BLK=8192 fires ~36 PROF_TIC/TOC pairs per block.
On Granite Rapids that's 36 × 22.3 ns = 800 ns/blk of overhead
(measured ~1000 ns including indirect costs like i-cache pressure).
Real bu wall on c8i prose_pride is 1574 ns/blk, so prof DOUBLES the
wall.  Per-primitive ns/call output is similarly inflated by
~25-30 ns per call site.

### Rules going forward

1. **Bench numbers (`pivco_huffman_bench` from `build/`, NO
   `-DPIVCO_PROF=1`) are the only authoritative headline numbers.**
   All cross-platform performance tables, IDEAS entries, commit
   message numbers, and READMEs must cite these.
2. `pivco_huffman_profile_english` output is RELATIVE.  Use it for
   "primitive A is X% of wall on platform P" — but don't quote its
   absolute ns/call as production performance.
3. The new `--tdbu` flag on the bench runs pivco_n + pivco_bu only
   (skips trad / huf0 / rans benches) — use for fast prof-on/off
   A/B verification of any kernel work.
4. **On small-K primitives (`bu_tree_merge_bcast_*`, etc.) the
   per-call overhead skews the ns/elem comparison.** Subtract
   ~25-30 ns/call on c8i, ~15 ns/call on older Intel, ~10 ns/call
   on Zen 3, ~negligible elsewhere.

### Open: reduce prof overhead

If we ever want trustable Granite Rapids profiles:
- Switch to `RDPMC` (perf-counter, ~5 ns vs RDTSC ~10) — needs CAP_SYS_RAWIO or perf_event_open setup
- Aggregate prof updates batch-wise (one TIC at loop entry, one TOC at exit per outer iteration instead of per primitive)
- Cycle-counter calibration constant subtracted from each ns/call

For now: just don't trust the c8i absolute ns/call.  The relative
breakdown ("bu_tree_merge is 35% of BU wall on c8i") is still valid
as long as all primitives are equally inflated by the constant
~25-30 ns/call overhead.

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

## D=7 NEON TBL flat-subtree path — not done, not worth much

D=2..6 all got TBL-accelerated flat-subtree paths in
`src/pivco_huffman_neon.c` (commits `b0639ff` through `a77c589`).  D=7
was skipped because:

- **No single TBL covers 128 entries.**  Would need 2× `vqtbl4q_u8` and
  a mask-based merge on the code's high bit, or a blend via
  `vbslq_u8`.  Extra NEON ops on the inner loop.
- **Coverage is small.**  From `extras/bench_flat_subtree_stats.c`:
  - bell_s80 (M=7–9): D≥7 = 0.0% — top distribution unaffected.
  - zipfian: D≥7 = 11.2% — only meaningful real-world share.
  - flat_M7: D=7 root-flat — currently 5086 M/s at 1.43× vs huf0
    (i.e., already winning).
- **bench/bench_flat_subtree_stats.c** reports the D-coverage breakdown
  per distribution; re-run if you want fresh numbers before attempting
  this.

Sketch if someone does attempt it:
```c
/* Split 7-bit code by high bit, do two vqtbl4q_u8 and blend. */
uint8x16_t codes = flat_d7_unpack(...);         /* 16 × (0..127) */
uint8x16_t hi    = vshrq_n_u8(codes, 6);        /* 0 or 1 (top bit) */
uint8x16_t lo    = vandq_u8(codes, vdupq_n_u8(0x3F));
uint8x16_t lo_syms = vqtbl4q_u8(c2s_lo64, lo);
uint8x16_t hi_syms = vqtbl4q_u8(c2s_hi64, lo);
uint8x16_t mask   = vceqq_u8(hi, vdupq_n_u8(1));
uint8x16_t syms  = vbslq_u8(mask, hi_syms, lo_syms);
```
(~4 extra NEON ops per 16 codes vs D=6's path.)

Expected win: <2% on zipfian; 3-4× on flat_M7 direct root-flat (via
the same vst1q_u8 pattern as D=2..6).  flat_M7 is already winning so
the marginal EV is low.

## FastLanes-style bitpacking — investigated, not shipped, still on the table

**Status (2026-04-27): investigated.  Full write-up in
[`BITPACKING.md`](BITPACKING.md).**

Three layouts compared in `extras/bench_unpack_dN.c` and
`extras/bench_unpack_fl_layout.c`:
1. **current** (`flat_dN_unpack`, dup-tbl + var-shift + and + vst1q)
2. **FL-natural** (same wire format, shift-imm + and + `vstKq`,
   only D ∈ {2,4})
3. **FL-layout** (FastLanes transposed bitstream, shift-imm + and,
   any D ∈ {2..7}, **wire-format change**)

Pure-unpack microbench (M4 Max, output GB/s):

| D | flat | FL-natural | FL-layout |
|---|------|------------|-----------|
| 2 | 46.5 | 63.9       | 109.0     |
| 3 | 26.0 | —          | 105.9     |
| 4 | 65.1 | 117.8      | 141.2     |
| 5 | 25.6 | —          | 112.9     |
| 6 | 25.9 | —          | 112.1     |

Graviton 4 numbers are larger relative gains, especially D=5/D=6
(22× over the current path — the existing `flat_d{5,6}_unpack` has a
known pathology there, fixed in production by the
`PIVCO_NEON_FAST_MULTI_TBL` gate but still a blocking factor on
that platform's flat-subtree throughput).

**End-to-end picture (M4):**
- FL-natural D=2 in `flat_decode_direct_neon` shipped briefly as an
  A/B; sparse_4 +12.3%, all other distributions inside ±2% noise.
  Reverted (synthetic-only win).
- Profile (`extras/profile_m4.sh`) shows prose_pride spends only
  ~7% of decode in `flat_dX_unpack` and ~14.5% in the surrounding
  scatter loop body (which is forced scalar by the indexed scatter
  — NEON has no vector scatter).  Realistic FL-layout end-to-end
  ceiling on real text is ~+5–6%.

**Open question.**  Whether the wire-format flip is worth ~5–6% on
real text.  Costs: encoder rewrite (transpose bit-packing), 4 backends
× 2 unpack styles (FL bulk + natural-layout tail for inner subtrees
of size not a multiple of FL block), version bookkeeping.  See
BITPACKING.md "Suggestions" for cost-ordered alternatives, including
the bigger fish — partition kernel (40% of prose_pride) and per-leaf
scatter (18%) — that FL-layout doesn't touch.

## AVX-512 root iota — investigated, end-to-end wash, kept in extras

**Status (2026-04-27): tested, ~0% net end-to-end on Xeon, not shipped.**

`pivco_huffman_decode_avx512`'s root partition built `uint16_t id[32]`
in a scalar `for k in 0..31: id[k] = j+k` loop on every iteration before
calling `partition_32_full`.  Codex item #2 suggested replacing that
with a precomputed iota-table read — the same idea as the NEON
iota_root experiment.

A/B on Xeon (5 alternated pairs, paired-t):

```
Significant wins:               Significant losses:
  bell_s80    +2.6% (t=3.1)       uniform     -3.9% (t=-7.0)
  english     +2.1% (t=2.2)       proba02     -1.7% (t=-2.8)
  image_jpeg  +1.2% (t=19.3)      geometric   -1.4% (t=-2.2)
```

3 wins vs 3 losses at p<0.05 across 29 distributions — right at the
noise floor (29 × 0.05 ≈ 1.5 false positives expected per direction).
Codex's hypothesis (vpcompressw being more expensive than NEON TBL
would make the iota gain bigger on Xeon than M4) didn't hold up: same
end-to-end-wash shape as the M4 NEON case.

The patch is preserved as a drop-in diff at
[`extras/avx512_root_iota.diff`](extras/avx512_root_iota.diff) along
with the A/B raw files in
[`results/xeon_quick_lf_{baseline,iota}_p{1..5}-20260427.txt`](results/).

## iota-table for `partition_root_8` — investigated, microbench only

**Status (2026-04-26): tested, ~0% end-to-end, kept in extras for
posterity.**

The non-root `partition_8` reads its 8 source uint16 indices from
memory.  The root partition has identity indices and synthesises them
via `vdupq_n_u16(base) + vaddq_u16(off)` (2 SIMD ops).  Variant:
replace synthesis with a `vld1q_u8` from a precomputed
`uint16_t static_iota_tab[N]` (1 SIMD op).

Standalone bench: [`extras/bench_partition_root_iota.c`](extras/bench_partition_root_iota.c).
M4 Max:
- `partition_root`            (vdup+vadd): 14.5 GB/s
- `partition_root_iota`       (vld1q_u8) : 15.6 GB/s  (**+8%**)
- `partition_root_half`       (vdup+vadd): 19.9 GB/s
- `partition_root_half_iota`  (vld1q_u8) : 21.9 GB/s  (**+9%**)

Productionised briefly in `pivco_huffman_decode_neon`; full sweep on
M4 showed **no end-to-end win** (deltas inside ±3% noise, mostly
trending slightly negative).  Reason: `partition_root_8` fires once
per block (1024×) but the decoder spends most of its time in 7
deeper levels of `partition_8`, which are unaffected by the change.
Reverted.

Might still be worth on a hypothetical future ARM uarch where
`vld1q_u8` is materially cheaper than `vdup+vadd` and the gap is
load-bearing — Graviton 4 didn't show the M4 microbench gap at all
(both at ~6.3 GB/s), so today only Apple silicon would benefit, and
even there end-to-end is a wash.

## ~~Graviton 4 NEON D=5/D=6 unpack — SIMD path still broken~~ — SHIPPED

**Status (2026-04-27): SHIPPED.**

Two-part fix:

1. **Byte-wise vector lane loads in `flat_d5/d6_unpack`.**  The old
   `memcpy(&packed, bm_ptr, 5/6) + vsetq_lane_u64(packed, ...)` form
   compiled to a stack round-trip on G4 (int-load → stack-store →
   vector-load-from-stack), causing a store-forward stall on every
   iteration that cost ~5x throughput.  Replaced with the same byte-wise
   pattern `flat_d3_unpack` already uses
   (`vdupq_n_u8(0) + vsetq_lane_u8(bm_ptr[k], ., k)` × N).  Compiler
   emits direct `ldr b` / `ld1 {v.b}[k]` / `ins` — no stack, no stall.
   Microbench: G4 D=5 1.3 → 5.8 GB/s (+345%), D=6 1.3 → 5.2 GB/s
   (+300%); M4 D=5/D=6 +12-17%.

2. **Re-enabled SIMD in flat_decode_direct path (still gated in
   scatter path).**  With the unpack 5x faster, the direct-path SIMD
   beats scalar by 60% on flat_M5/M6 (n=8192, plenty of work to
   amortise vqtbl{2,4}q_u8 setup).  Scatter path stays gated to
   scalar on non-Apple because the smaller n there means the TBL
   setup overhead still dominates.

Final A/B on G4 (5 paired pairs, full write-up in
[`results/G4_D5D6_FIX-AB-20260427.md`](results/G4_D5D6_FIX-AB-20260427.md)):

| Distribution | Δ | t |
|---|---:|---:|
| flat_M5      | +59.8% | 102 |
| flat_M6      | +64.1% | 307 |
| flat_M7      | +0.3%  | 2.5 |

No significant losses on any other distribution; bell_s30/s80/zipfian
unchanged because the scatter gate keeps them scalar.

The store-forward-stall pathology on Neoverse-V2 (stale `flat_d5/6`
text below in this file) is fully resolved.  The D=3 throughput
on G4 is *also* uint16x8-bound at ~7-8 GB/s, which is the natural
ceiling for that layout — improving past it requires the FL-layout
work tracked in BITPACKING.md (~30 GB/s on G4 in microbench).

## ~~Graviton 4 NEON D=5/D=6 unpack — SIMD path still broken~~ (historical)

**Status (2026-04-27): production gates around the pathology with
`PIVCO_NEON_FAST_MULTI_TBL=0`, but the SIMD unpack itself is still
broken on Neoverse-V2.**

`extras/bench_unpack_fl_layout.c` (today's microbench) shows the
current `flat_d{5,6}_unpack` running at **1.3 GB/s on Graviton 4**
vs **25 GB/s on M4** — a ~20× cliff with the same source.  M4 and G4
microbenches saved as
[`results/unpack_fl_layout-m4_max-20260426.txt`](results/unpack_fl_layout-m4_max-20260426.txt)
and [`results/unpack_fl_layout-graviton4-20260426.txt`](results/unpack_fl_layout-graviton4-20260426.txt).

The `PIVCO_NEON_FAST_MULTI_TBL` gate (commit `cee2366`, see the
already-shipped section above) hides this in production by
falling through to scalar, so distributions don't see the worst
case.  But:
- The fast SIMD path on Graviton 4 still doesn't exist.  Anything
  that *would* speed up the unpack (FL-natural, FL-layout, a
  vqtbl-free rewrite) gets a 22× microbench multiplier on D=5/D=6
  on this platform.
- The D=5/D=6 part of bell_s30, zipfian, and similar distributions
  decodes through the scalar fallback today — fast, but leaves real
  ceiling on the table.

Likely root cause is the `vqtbl{2,4}q_u8` + uint16-lane shift
pattern in `flat_d{5,6}_unpack` mapping poorly to Neoverse-V2's
SIMD pipes.  An FL-layout (uint8x16 only, shift+mask, no TBL)
sidesteps this entirely and hits ~30 GB/s on the same chip.

Same pattern but milder for D=3 (G4: 7.7 GB/s vs M4 26.0 GB/s).

## ~~Graviton 4 NEON D=5/D=6 regression — gate or replace vqtbl{2,4}q_u8~~ — SHIPPED

> **Status (as of [cee2366bb2372cd173e1900db0b5ea99f4c0c65b](../), 2026-04-25):** SHIPPED via build-time gate.
> Add `PIVCO_NEON_FAST_MULTI_TBL` macro defaulting to 1 on `__APPLE__`
> and 0 elsewhere.  When 0, the D=5 / D=6 cases in
> `flat_decode_scatter_neon` and `flat_decode_direct_neon` fall through
> to `NEON_FLAT_UNPACK_SWITCH` — the same scalar handling already used
> for D=7 / D≥8.  Override with `-DPIVCO_NEON_FAST_MULTI_TBL=0/1` if
> a future ARM uarch flips the trade.
>
> Sweep file: [`results/20260425-0126-cee2366-graviton-d56-fix.md`](results/20260425-0126-cee2366-graviton-d56-fix.md).
>
> Graviton 4 (c8g.large pinned, 30 reps × 4M) headlines:
> - bell_s80: 537 → **1105** M/s (+106%, 0.60× → **1.24×** — now wins)
> - flat_M5: 1282 → **3187** M/s (+148%, 0.77× → **1.93×** — now wins)
> - flat_M6: 1194 → **2458** M/s (+106%, 0.91× → **1.59×** — now wins)
> - bell_s30 +27%, zipfian +19% (subtree D=5/6 contributions)
> - sparse_4/16: small −5%/−11% nick (D=5/6 not dominant; vqtbl was
>   competitive there but the scalar path is still well ahead of huf0)
>
> Win count on c8g: 10/19 → 13/19 distributions.
>
> **Tried-and-rejected variant:** replacing `vqtbl{2,4}q_u8` with
> `2× vqtbl1` (or `4× vqtbl1`) + `vsub` + `vorrq` blends.  bell_s80
> went 537 → **458** M/s — *slower than the regressed multi-register
> path*.  Conclusion: on Neoverse-V2, any 2-byte-output-per-iter
> vector lookup over a 32/64-byte table loses to scalar OoO-extracted
> parallelism for the iteration counts these flat subtrees produce.
> The right fix is to fall through to scalar entirely.
>
> **Methodology note:** the original sweep was on `c8g.medium`
> (1 vCPU, burstable) with visible CPU-steal variance on individual
> rows.  Re-baselined on `c8g.large` (2 vCPU, dedicated) before
> diagnosing — confirmed the regression was real and not host noise.

The proposal text below is preserved as historical record.

---

The NEON D=5 path uses `vqtbl2q_u8` (32-byte table across 2 reg) and
D=6 uses `vqtbl4q_u8` (64-byte table across 4 reg).  On Apple M4 both
are ~1/cycle throughput; on AWS Graviton 4 (Neoverse-V2) they run at
noticeably lower throughput, making the SIMD path slower than the
scalar-vectorised fallback for these D values.

Concrete numbers (test-c8g, 30 reps × 4M):
- bell_s80 (71% D=6, 29% D=5): 1055 → 526 M/s (**−50%**)
- flat_M5 (D=5 root-flat): 1809 → 1282 M/s (−29%)
- flat_M6 (D=6 root-flat): 1275 → 1194 M/s (−6%)
- bell_s30 (32% D=5): smaller hit, ~−10%

D=2/D=3/D=4 (single-register `vqtbl1q_u8`) are fine on c8g.

## ~~Revisit AVX-512 D=3, D=5, D=6 flat-subtree TBL~~ — SHIPPED

> **Status (as of `7b2fb8d`, 2026-04-24):** landed.  The earlier
> "tried and reverted" attempts (committed in `fa1134b`) blamed
> `vpermb` and the scalar compiler.  Wrong diagnosis: the real cause
> was `memcpy(&packed, bm_ptr, N)` for **non-power-of-2 N** (5, 6,
> 10, 12).  GCC lowers those to split loads + OR + store-forwarding
> chains — adding 2-3 cycles of serial latency that killed the fast
> path at 16-codes-per-iter.  Fix: load the next natural size
> (`uint64` / `__m128i`) unconditionally, relying on the fact that
> the unpack control only references the valid-byte range; add a
> `_safe` variant with the original `N`-byte memcpy for the final
> chunk so we don't overread past a page boundary.  Fast variant is
> 1 instruction, safe variant is 1 stack memcpy + 1 xmm load.
>
> Headlines (Xeon 6975P-C):
> - flat_M3: 3820 → 21860 (+472%, 2.10× → 12.00×)
> - flat_M5: 4370 → 18446 (+322%, 2.42× → 10.21×)
> - flat_M6: 2890 → 17118 (+492%, 1.72× → 10.18×)
> - bell_s80: 1891 → 2107 (+11%, 2.72× → 3.03×)
> - bell_s10 crosses parity 0.95× → 1.03× (D=3 34% contribution)
> - bell_s30 +2%, zipfian +2%, proba02 +5%, english unchanged
>
> Committed in `3f27e81` (D=3) and `7b2fb8d` (D=5, D=6).  AVX-512 now
> has D=2..6 fast paths, same coverage as NEON.
>
> **Lesson for future work:** always check whether `memcpy(ptr, src,
> N)` for an awkward N actually compiles to a single load before
> blaming the SIMD primitives.

## Revisit SSE4.1 D=2, D=3, D=5, D=6 flat-subtree TBL

**Status (as of `ae14323`, 2026-04-24):** skipped.  Pure SSE4.1 has no
per-byte variable shift (no `vpsrlv*` until AVX2) and no
`vpmultishiftqb` (VBMI2), so the unpack would need 4–8 separate
`pshufb` + immediate shifts + blends — which benchmarked slower than
scalar even on AVX-512 (§"Revisit AVX-512 D=3, D=5, D=6").

Approaches to try:

- **AVX2 when available**.  `_mm_srlv_epi16` (AVX2) would give the
  per-uint16-lane variable shift we need.  Zen 3 (the current SSE4.1
  test host) has AVX2 but CMakeLists doesn't enable `-mavx2` on the
  non-AVX-512 path.  Adding it would unlock D=2/3 on Zen 3.
- **GFNI fallback**.  Same primitive as the AVX-512 revisit above;
  Zen 3 has GFNI (check for SSSE3 + GFNI at build time).
- **Hybrid with scalar**.  Skip TBL for odd D and keep the scalar
  scatter (compiler will vectorise it as on AVX-512).  Only add TBL
  for D=4 (already done).  Already the current state.

Real-world impact on Zen 3 moderate-entropy distributions is limited
even with perfect TBL — the IDEAS.md "Zen 3 hybrid block decoder"
fallback (route per-table between PIVCO and trad_huffman_decode_4s
when flat-subtree coverage is low) is probably the right escape
there regardless.

## SSE4.1 D=5..8 flat unpack — gcc unroll pragma

**Status: open, easy win for SSE-only fallback path.**

The D=5..8 cases in `flat_decode_direct_x86` and `flat_decode_scatter_x86`
are scalar dependent-load loops:

```c
for (; i < n; i++) symbols[i] = c2s[bm[i]];   /* D=8 */
```

Disassembly on test-c6a (Zen 3, gcc-11) shows a 1-byte / iter serial dep
chain.  clang-20 manually unrolls 4× and exposes 4 independent
`(load bm → load c2s → store)` chains, hitting Zen 3's 3-load/cycle L1d
limit.  Measured throughput delta on uniform/gzip_random distributions:
**+~120% (gcc 825 MB/s → clang 1859 MB/s)** end-to-end.

Compiler-recommendation work (`results/compiler-sweep-20260507/`)
already steers Zen 3 to clang-20 by default, so this is moot for the
recommended path.  But adding `#pragma GCC unroll 8` (or `#pragma clang
loop unroll(enable)`) on the D=5/6/7/8 cases would:

- Make the SSE-only fallback (when the user pins an older gcc) match the
  clang numbers on the dominant hot path for near-uniform distributions.
- Cost: trivial — just a pragma above each `for` loop.  No correctness
  risk.

Files: `src/pivco_huffman_x86.c:285-292` (scatter D=8) and `:395-397`
(direct D=8); same idea for D=5/6/7 in the same file.

## RISC-V (RVV 1.0) backend — speculative, watch-and-wait

**Status: open, not yet attempted.  Watch-and-wait until hardware
becomes competitive or a cloud provider offers it.**

RVV 1.0 (RISC-V Vector, ratified 2021) maps surprisingly well onto
this decoder's hot-path primitives.  Specifically:

| Op our hot path needs | NEON | AVX-512 | RVV 1.0 |
|---|---|---|---|
| compress (= partition kernel) | TBL via shuffle | `vpcompressw` | `vcompress` |
| TBL (= shuffle) | `vqtbl1q` | `vpermb` | `vrgather` |
| **Indexed byte scatter** (= leaf scatter, today's bottleneck on real-text) | **none** — scalar STRBs | **none** at byte granularity (`vpscatterdd` is dword-only — overlap clobbering) | **`vsuxei8` — native** |

`vsuxei8` is the only ISA-level asymmetry that matters here: it
addresses the per-leaf scatter cost that dominates real-text decode
(~18% of prose_pride decode time per today's profile, all in
`scatter_sym` / `scatter_both_leaves` doing 8 scalar STRBs per
8 codes).  Combined with vector-length-agnostic loop bodies (one
source for VLEN=128/256/512/...), the RVV port should look a lot like
the disabled SVE backend (sister ISA).

**Why hold off in early 2026:**

- **No cloud option.**  AWS / GCP / Azure / Akamai don't offer RISC-V
  SKUs.  Scaleway has SOPHGO SG2042 (RVV 0.7.1, the *pre-ratification*
  draft — `vsuxei` semantics differ; not directly comparable to RVV
  1.0).  Niche Asian providers may have RVV 1.0 boards but cross-border
  access is friction.
- **No competitive hardware yet.**  SpacemiT K1 (8× X60 @ 1.6 GHz,
  256-bit VLEN) is the cheap accessible RVV 1.0 SoC.  Tenstorrent
  Ascalon and SiFive Performance P870 are coming but not in volume.
  All are 3-10× slower per clock than M4 / Xeon / Graviton 4 today, so
  benchmark numbers wouldn't be apples-to-apples against the rest of
  our cross-platform table.

**When this becomes a yes:**

- AWS or GCP launches a RISC-V instance class.
- A SiFive/Tenstorrent/Andes chip lands at competitive clocks (3+ GHz
  with 256-bit VLEN minimum to be in the ballpark).
- We want the README's cross-platform "works everywhere" story to
  include RISC-V regardless of perf parity.

**Cheap dev path if any of those triggers fire:**

| Option | Cost | Convenience |
|---|---|---|
| QEMU RVV emulation (correctness + asm only) | $0 | Runs on x86 dev box; not for benchmarking |
| Banana Pi BPI-F3 / Milk-V Jupiter (SpacemiT K1) | ~$80-120 | SBC; bench-quality numbers; Ubuntu-able |
| DC-Roma II laptop (same K1) | ~$700 | "It's just a laptop"; max friction-free |

**Cross-asymmetric data point that would be interesting:**

If we did port and run on a SpacemiT K1, the per-leaf scatter would
collapse from `8× STRB` per 8 codes (today on every non-RVV backend)
to a single `vsuxei8`.  On real-text distributions where the scatter
is ~18% of decode time, a 5-10× speedup of that fraction could yield
+10-15% end-to-end on real-text — comparable to today's leaf-fusion
or G4-D5/D6 wins, but as a pure architectural advantage.  That's the
single most interesting data point for whether the algorithm shape is
actually as ISA-portable as we claim.

Until cloud RISC-V or a competitive chip lands, the work-to-data ratio
is poor (~1-2 weeks of porting + intrinsics + build setup, for numbers
that won't translate to anyone's production hardware).

## Suggested implementation order

Leaf-child fusion and the flat-subtree fast path (both the scatter
loop and the TBL-accelerated variants) are *shipped*:

- NEON D=2..6 (commits `b0639ff` through `a77c589`)
- AVX-512 D=2..6 (`210211d`, `ad17cdc`, `3f27e81`, `7b2fb8d`)
- SSE4.1 D=4 (`0e037ab`)

Remaining outstanding work, roughly in increasing cost / decreasing
certainty:

1. **Check `flat_dX_unpack` against FastLanes' unpack kernels**
   (see §"Check flat_dX_unpack against FastLanes unpackers" above).
   Maybe a few percent per D.
2. **Revisit SSE4.1 D=2, D=3, D=5, D=6** with AVX2 `_mm_srlv_epi16`
   or GFNI — see section above.
3. **Zen 3 SSE4.1 hybrid block decoder** — on Zen 3 the
   moderate-entropy distributions stay at 0.41–0.62× vs huf0_x2 even
   after flat-subtree.  A per-table fallback to `trad_huffman_decode_4s`
   (§"Product-level idea: hybrid block decoder") would plausibly
   recover most of the gap.  Gate heuristic: when flat-subtree coverage
   estimate is low (e.g., `≤ 20%`) AND `max_len > 6`, fall back to huf0.
4. **TBL-based K-way bucket** for `decode_neon_prefix` phase 4 — only
   relevant to the non-flat prefix-radix research path, which is
   effectively unused now that flat-subtree wins on the same
   distributions.  Can drop.

   *Speculative-fill alternative tried* (2026-04-25): generalise the
   phase-0 `memset(prefill_sym)` (which covers only the SINGLE most-
   frequent leaf bin) into a NEON TBL pass writing
   `bin_to_sym[prefix[k]]` for every k.  After this pass, all
   leaf-bin elements are already correctly placed; phase 5 only
   handles non-leaf bins.  Gated to K ≤ 16 (NEON `vqtbl1q_u8` covers
   exactly that — for K > 16 the scalar fallback regresses
   noticeably).  Result on M4: +2-3% on K=8 distributions
   (`proba14` 1305 → 1340, `english` 1275 → 1306), neutral
   elsewhere.  Diagnosis: phase-5 leaf-scatter on K=8 is already
   essentially zero (sequential byte writes, bandwidth-free), so the
   savings come from eliminating the `scatter_sym` branch in phase 5
   only — small.  Phase 4 still buckets leaf-bin elements
   unconditionally; making it skip them needs a hot-loop branch
   (mispredict-risky) or a separate compaction pass.  Reverted —
   gain too small to justify complexity given pivco_p retirement.
5. **Nested (multi-stage) prefix-radix** — same as (4); subsumed by
   flat-subtree.  Can drop.
6. **Retire `pivco_huffman_decode_neon_prefix`** — the remaining
   research backend.  All of its flat-tree wins are now handled by
   the flat-subtree path inside `decode_node_neon`; the non-flat
   prefix-radix never became competitive.  Straight deletion clears
   ~600 lines of code + the `pivco_p` benchmark column.

   **Confirming-the-retirement experiment** (canasort microbench,
   2026-04-25):  before retiring, we tested whether a "FastLanes-
   like" lane-interleaved phase 4 could push pivco_p past pivco_n.
   Extended canasort's `bench_scatter` to add `stream8` (8 parallel
   streams + private cursors + included histogram = exact pivco_p
   phase 2+4 shape) and `stream8_pre` (placement loop only, fair
   apples-to-apples vs `direct`).  Canasort commit `55c0adc`,
   discussion in `extras/scatter/BENCH-SCATTER.md` Round 5.

   Apple M, n=10M, elem=4B (ns/elem):

   | P  | direct | stream8 | stream8_pre |
   |----|-------:|--------:|------------:|
   |  8 | 0.68   | 0.55    | **0.35**    |
   | 16 | 0.62   | 0.53    | **0.36**    |
   | 32 | 0.54   | 0.57    | **0.40**    |
   | 64 | 0.48   | 0.64    | **0.44**    |

   `stream8_pre` wins at P ≤ 64, validating that pivco_p's existing
   8-way SWCB code in `pivco_huffman_neon_prefix.c` is the right
   algorithm for K=8.  The implementation is already at the
   algorithmic ceiling for the problem shape — TBL-compaction would
   be ~5-6× slower for K=8 (wrong tool), and any further phase-4
   optimisation cannot meaningfully improve on the measured
   ~0.35 ns/elem floor.

   **But pivco_n on `english` runs at 0.30 ns/elem total decode**
   (3333 M/s on M4) — faster than pivco_p's *phase 4 alone*, never
   mind phases 1, 2, 5.  pivco_p cannot catch pivco_n by phase-4
   optimisation; the flat-subtree mechanism is fundamentally faster
   for these distributions.

   Conclusion: retire as planned.  Phase 4 is fine; the algorithm
   above it is the loser.

   Concrete scope (TODO):
   - Delete `src/pivco_huffman_neon_prefix.c` (~600 lines).
   - Remove `pivco_huffman_{encode,decode}_neon_prefix` from
     `include/pivco_huffman.h`.
   - Remove the prefix-backend test block from `test/test_roundtrip.c`
     (the primary scalar↔NEON roundtrip coverage stays).
   - Drop the `pivco_p` column from `bench/bench_main.c`.
   - Delete `bench/bench_prefix_profile.c` and its `CMakeLists.txt`
     entry.
   - Keep `bench/bench_multi_stage_stats.c` and
     `extras/bench_flat_subtree_stats.c` (they read tree structure, not
     the retired backend).
   - Keep `PREFIX_RADIX.md` as historical record (already banner'd as
     superseded).
   - Update README.md / CLAUDE.md to reflect deletion.
