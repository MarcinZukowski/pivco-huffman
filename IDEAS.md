# PIVCO-Huffman Decode Ideas

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

## ~~Product-level idea: hybrid block decoder~~ — largely obsoleted

> **Status (as of `8754347`, 2026-04-24):** largely obsoleted by the
> flat-subtree fast path.  On Apple M4 the PIVCO decoder now beats
> `huf0`/`trad_4s` on 18/19 distributions.  The only remaining
> consistent loss is proba14 (0.96×) where the flat-subtree analyzer
> predicted 0.9% coverage — i.e., the tree shape doesn't permit
> flat-subtree savings.  A per-table fallback to huf0 would help there
> but the absolute gap is small.
>
> A hybrid block decoder still makes sense on **Zen 3 SSE4.1** where
> PIVCO loses on several moderate-entropy distributions (bell_*,
> proba02, english, zipfian at 0.41–0.62×) because the per-cycle
> partition cost is already cheap.  On that platform falling back to
> huf0_x2 for those tables would meaningfully boost the geometric mean.

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
uint8x16_t codes = flat_d7_spread(...);         /* 16 × (0..127) */
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

## Check `flat_dX_spread` helpers against FastLanes unpackers

The `flat_d2_spread` / `flat_d3_spread` / ... routines in
`src/pivco_huffman_neon.c` turn a packed D-bit stream into a
byte-per-code NEON vector via a small shuffle + shift + mask sequence.
This is exactly the same primitive as **FastLanes**'
[bit-unpacking](https://github.com/cwida/FastLanes) inner loop (the
"unpack N-bit values" operation), which the FL authors have tuned
heavily on NEON and AVX-512.

**Worth comparing**: our spread sequence vs FastLanes's for each D.
FL may use slightly different permute constants or a smarter
shift-mask combo that we're missing.  Even a few percent per spread
compounds across bell/zipfian/proba02 (which have the largest
flat-subtree TBL windows).

Sketch of the check:
- Clone https://github.com/cwida/FastLanes; find the NEON unpack
  kernel for D ∈ {2,3,4,5,6}.
- Compare instruction-level vs our `flat_dX_spread` in
  `src/pivco_huffman_neon.c` (offsets ~150-320 after the
  `pivco_huffman_common.h` include).
- If FL is faster, port the trick.  Benchmark with
  `./build/pivco_huffman_bench 30` and compare english / bell_* / zipfian.

Easy (lookup-only) win candidate.  5-15 minute first-look exercise.

## Revisit AVX-512 D=3, D=5, D=6 flat-subtree TBL

**Status (as of `ae14323`, 2026-04-24):** tried and reverted.  All
three regressed vs the scalar `FLAT_UNPACK_SWITCH_IDX` macro on
Xeon 6975P-C (Granite Rapids):
- D=3 (attempted with `vpmultishiftqb` 16-code-per-iter and with
  `pshufb` + `_mm_srlv_epi16` 8-code-per-iter): flat_M3 −38%.
- D=5 (`vpermb` over ymm): flat_M5 −50%, bell_s80 −17%, bell_s30 −13%.
- D=6 (`vpermb` over zmm): flat_M6 −27%, bell_s80 −32%.

GCC auto-vectorises the scalar 8-wide scatter (`symbols[i+k] = c2s[...]`
for k=0..7) into wide moves, so the bar is higher than expected.  The
winning D values were D=2 and D=4 (simpler spread, smaller table, no
byte-crossings).  See `src/pivco_huffman_avx512.c` for the recorded
constraints — if you want to reattempt, approaches worth trying:

- **Wider iteration**.  Our attempts all used 128-bit registers to
  match NEON; a 32-code-per-iter or 64-code-per-iter version using
  full ymm/zmm might amortise the spread cost enough to win.  Zmm
  usage may trigger Intel's clock throttling on some SKUs but
  Granite Rapids is said to be less affected.
- **Different spread primitive**.  GFNI (`vgf2p8affineqb_epi8`) can
  do byte-level shift + mask in a single instruction with the right
  matrix constant; might beat `vpmultishiftqb` for D=3/5.
- **Skip the spread entirely** for D=4-aligned cases: reorganise
  the packed format so codes for neighbouring elements land in
  consecutive bytes aligned for a direct pshufb.

## Revisit SSE4.1 D=2, D=3, D=5, D=6 flat-subtree TBL

**Status (as of `ae14323`, 2026-04-24):** skipped.  Pure SSE4.1 has no
per-byte variable shift (no `vpsrlv*` until AVX2) and no
`vpmultishiftqb` (VBMI2), so the spread would need 4–8 separate
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

## Suggested implementation order

Leaf-child fusion and the flat-subtree fast path (both the scatter
loop and the TBL-accelerated variants) are *shipped*:

- NEON D=2..6 (commits `b0639ff` through `a77c589`)
- AVX-512 D=2 (`210211d`) and D=4 (`ad17cdc`)
- SSE4.1 D=4 (`0e037ab`)

Remaining outstanding work, roughly in increasing cost / decreasing
certainty:

1. **Check `flat_dX_spread` against FastLanes' unpack kernels**
   (see §"Check flat_dX_spread against FastLanes unpackers" above).
   Maybe a few percent per D.
2. **Revisit AVX-512 D=3, D=5, D=6** — see section above.  Wider
   iteration, GFNI, or format tweaks may flip these from regression
   to win.
3. **Revisit SSE4.1 D=2, D=3, D=5, D=6** with AVX2 `_mm_srlv_epi16`
   or GFNI — see section above.
4. **Zen 3 SSE4.1 hybrid block decoder** — on Zen 3 the
   moderate-entropy distributions stay at 0.41–0.62× vs huf0_x2 even
   after flat-subtree.  A per-table fallback to `trad_huffman_decode_4s`
   (§"Product-level idea: hybrid block decoder") would plausibly
   recover most of the gap.  Gate heuristic: when flat-subtree coverage
   estimate is low (e.g., `≤ 20%`) AND `max_len > 6`, fall back to huf0.
5. **TBL-based K-way bucket** for `decode_neon_prefix` phase 4 — only
   relevant to the non-flat prefix-radix research path, which is
   effectively unused now that flat-subtree wins on the same
   distributions.  Can drop.
6. **Nested (multi-stage) prefix-radix** — same as (5); subsumed by
   flat-subtree.  Can drop.
7. **Retire `pivco_huffman_decode_neon_prefix`** — the remaining
   research backend.  All of its flat-tree wins are now handled by
   the flat-subtree path inside `decode_node_neon`; the non-flat
   prefix-radix never became competitive.  Straight deletion clears
   ~600 lines of code + the `pivco_p` benchmark column.

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
