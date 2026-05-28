# `pivco_huffman_neon_fused_1leaf.c` — failed experiment, kept for reference

This is a writeup of an attempt to speed up the PIVCO-Huffman NEON decoder
by fusing the partition and the leaf-side scatter at **non-prefill one-leaf
nodes** (internal nodes whose one child is a leaf that isn't the prefilled
most-frequent symbol).  Three variants were implemented and measured on
Apple M4 Max; all regressed vs the baseline `pivco_huffman_neon.c`.

The file in this directory contains the final (best-performing) variant —
Trick 2 / bucketed scatter — for archival purposes.

---

## Motivation

### The one-leaf non-prefill case today

At a node where one child is a leaf and the other is an internal subtree,
the current `decode_node_neon` (in `src/pivco_huffman_neon.c`) does:

1. **Full two-sided partition** of the N elements: write compacted
   bit=0 positions to `indices[]` (in-place), compacted bit=1 positions
   to `tmp[]`.  Cost per 8 parent elements: 1 vld + 2 TBL + 2 vst.
2. **Separate scatter pass** over the compacted leaf-side indices:
   load 8 at a time with `vld1q_u16`, 8 `umov` + 8 `strb` per 8 leaf
   indices.
3. Recurse into the non-leaf child.

### Why it looked worth fusing

The "separate scatter pass" reads indices that were just written in step 1
— a clean memory round-trip.  Speculation: if we kept the leaf-side
compacted indices **in-register** after the TBL, we'd eliminate that
round-trip entirely.  Rough counting per 8 parent elements:

- **Save:** 1 vst (partition's leaf-side output) + 1 vld (scatter's
  reload of that output) ≈ 1 cycle.
- **Pay:** some form of dispatch (0–8 stores per chunk depending on
  popcount).

The one-leaf-stats analyzer (`extras/bench/bench_one_leaf_stats.c`) confirmed
the shape is common on the distributions where PIVCO already wins:
proba80 hits the non-prefill one-leaf path for **99.2%** of all non-root
partition work, proba50 for 99.9%, geometric for 95.8%.  English / zipfian
/ uniform almost never hit it (< 5%).

---

## Variant 1 — switch / fallthrough

```c
switch (n_left) {
case 8: symbols[vgetq_lane_u16(left_idx, 7)] = sym_L; /* FALLTHROUGH */
case 7: symbols[vgetq_lane_u16(left_idx, 6)] = sym_L;
...
case 1: symbols[vgetq_lane_u16(left_idx, 0)] = sym_L;
case 0: break;
}
```

Expected: clang would emit a 9-entry jump table indexed by `n_left`, then
fall through to do the right number of stores.

**What clang actually did** (from `otool -tV` on the compiled object):
lowered it to a **binary search tree of conditional branches** — a mix
of `b.eq`, `b.gt`, `b.le`.  Roughly 3–4 conditional branches per chunk.

### Bench

Apple M4 Max, 20 repeats × 4M symbols per run, 5 runs, median of best 3:

| Distribution | neon baseline | switch JT  | Δ      |
|--------------|--------------:|-----------:|-------:|
| proba80      | 9465          | 4752       | −50%   |
| proba50      | 5058          | 1330       | −74%   |
| geometric    | 4898          | 1301       | −73%   |
| proba14      | 2433          | 1422       | −42%   |
| english      | 2504          | 2326       | −7%    |
| uniform      | 1156          | 1154       | 0%     |
| two_sym_eq   | 26801         | 24974      | −7%    |

### Why it regressed

On proba80, `n_left` clusters in 5–7 (leaf side is the ~80% mass second-
most-common symbol at each stick-level below the prefill skip).  With
3–4 conditional branches per chunk, each potentially mispredicting at
~30% on shifting patterns, the flush cost is roughly:

```
  500K chunks × 3.5 branches × 30% mispredict × 14 cycle flush
≈ 7M cycles / 4M symbols ≈ 1.8 ms of pure branch-flush
```

Matches the observed ~1.7 ms regression per 4M decode.

---

## Variant 2 — computed goto (real jump table)

Hypothesis: if the compare-tree was the problem, a true indirect-branch
jump table would help.  Used GCC's `&&label` / `goto *ptr` extension to
force exactly one indirect branch per chunk followed by fallthrough
straight-line stores.

```c
static const void *const labels[] = {&&s0, &&s1, &&s2, ..., &&s8};
goto *labels[n_scatter];
s8: symbols[vgetq_lane_u16(idx, 7)] = sym;  /* fallthrough */
s7: symbols[vgetq_lane_u16(idx, 6)] = sym;
...
```

Confirmed via `otool -tV`: clean `ldr x10, [x11, x10, lsl #3]` + `br x10`
pattern.  Exactly one indirect branch, then straight-line code.

### Bench

| Distribution | neon | switch JT | computed-goto JT (inline) |
|--------------|-----:|----------:|--------------------------:|
| proba80      | 9325 | 4752      | 3627                      |
| proba50      | 5022 | 1330      | 1076                      |
| geometric    | 4883 | 1301      | 1058                      |

**Worse than the compare-tree.**  Two reasons:

1. M4's indirect-branch predictor handles 9 targets with a shifting
   distribution poorly — mispredict rate appears at least as high as
   the compare-tree's per-branch rate, and a mispredicted indirect
   branch on M4 is at least as expensive as a mispredicted conditional
   one.
2. Noinline helper added call overhead on top.  Inlined version didn't
   help meaningfully — the mispredict is still the dominant cost.

**Conclusion from (1) + (2):** the *dispatch itself* is the problem, not
the compiler's lowering mechanism.  Any per-chunk dispatch on a variable
target is too expensive for this hot path on M4.

---

## Variant 3 — Trick 2, bucketed scatter (no dispatch at all)

If per-chunk dispatch is fatal, eliminate dispatch entirely.  Structure:

- **Phase 1** — stream through chunks, partition **non-leaf side only**
  to `tmp[]`, record each chunk's `n_left` in a per-chunk byte array.
- **Phase 2** — bucket-sort chunk IDs by `n_left` (via count + prefix sum
  + single placement pass).  Result: 8 buckets, each containing the IDs
  of chunks with that exact leaf-side count.
- **Phase 3** — 8 *separate* inner loops, each with `v` as a compile-time
  constant; inside each loop, re-load the chunk's indices, redo the
  left-side TBL, and do straight-line `v` `umov`+`strb` pairs.  No
  dispatch, no mispredicting branch.

Implemented via an inlined helper whose `if (v >= k)` cascades collapse
at compile time:

```c
static inline __attribute__((always_inline))
void scatter_v_lanes(uint8_t *symbols, uint16x8_t idx, uint8_t sym, int v) {
    if (v >= 1) symbols[vgetq_lane_u16(idx, 0)] = sym;
    if (v >= 2) symbols[vgetq_lane_u16(idx, 1)] = sym;
    ...
    if (v >= 8) symbols[vgetq_lane_u16(idx, 7)] = sym;
}
```

Called 8 times from 8 distinct outer-loop sites with constants `V=1..8`.
Clang unrolls each to exactly `V` stores, branch-free.

### Bench

| Distribution | neon | switch JT | Trick 2 bucket | Δ vs neon |
|--------------|-----:|----------:|---------------:|----------:|
| proba80      | 9535 | 4752      | **6105**       | **−36%**  |
| proba50      | 5144 | 1330      | **2413**       | **−53%**  |
| geometric    | 4888 | 1301      | **2350**       | **−52%**  |
| proba14      | 2389 | 1422      | 1838           | −23%      |
| english      | 2482 | 2326      | 2334           | −6%       |
| uniform      | 1173 | 1154      | 1166           | 0%        |
| two_sym_eq   | 25482| 24974     | 24804          | −3%       |

**Recovered big vs the dispatched versions** (proba80 +28%, proba50
+81%, geometric +81%) — confirming that mispredict flushes were the
dominant cost of variants 1 and 2.

**Still slower than neon baseline.**

### Why it regressed (even without dispatch)

The original cycle model was sloppy.  Revisiting it precisely:

Baseline `neon` at a one-leaf non-prefill node, per 8-parent-element chunk:

```
Full partition:  1 vld (indices) + 2 vld (shuffles) + 2 TBL + 2 vst   ≈ 3.5 c
Scatter of leaf side, amortized over chunks with avg v = n_left/chunk:
   1 vld per 8 *leaf* indices (not per chunk) + 8 umov + 8 strb
   per 8 leaf indices ⇒ cost per parent-chunk ≈ 0.5 × v
Total: 3.5 + 0.5 × v
```

Trick 2 (bucketed), per 8-parent-element chunk:

```
Phase 1 (right-partition): 1 vld + 1 vld (shuffle) + 1 TBL + 1 vst ≈ 2 c
Phase 3 (scatter, re-read): 1 vld + 1 vld (shuffle) + 1 TBL + v × strbs
                              ≈ 2 + 0.5 × v
Plus phase 2 bookkeeping:  ≈ 1 c/chunk
Total: 2 + 2 + 1 + 0.5 × v = 5 + 0.5 × v
```

**Net +1.5 c/chunk vs baseline.**  For proba80 with ~488K chunks through
the fused path per 4M decode:

```
488K × 1.5 c × (1/3.5 GHz) ≈ 0.21 ms per 4M
```

Consistent with the observed ~36% regression on proba80.

### Where the model went wrong initially

- Thought: "save 1 vld + 1 vst per chunk."
- Reality: baseline's scatter vld is amortized over 8 *leaf* elements,
  not over 8 *parent* elements.  So the baseline only pays ~v/8 vlds per
  chunk for scatter — much less than one.
- And Trick 2 pays **two** loads of original `indices[]` per chunk
  (phase 1 + phase 3) plus redoes the left-side TBL in phase 3, so its
  indices-load cost is 2× the baseline's vld cost, not 1×.

The asymmetry between "amortized over 8 leaf elements" (baseline) and
"per chunk of 8 parent elements" (Trick 2) is the structural loss.

---

## Summary table

Single source of truth for the numbers; Apple M4 Max, `./build/pivco_huffman_bench 20`:

| Variant                       | proba80 | proba50 | geometric | english | uniform | two_sym_eq |
|-------------------------------|--------:|--------:|----------:|--------:|--------:|-----------:|
| neon (baseline)               | 9535    | 5144    | 4888      | 2482    | 1173    | 25482      |
| switch fallthrough            | 4752    | 1330    | 1301      | 2326    | 1154    | 24974      |
| computed-goto noinline        | 3771    | 1081    | 1069      | 2164    | 1147    | 25351      |
| computed-goto inline          | 3627    | 1076    | 1058      | 2175    | 1147    | 25033      |
| Trick 2 (bucketed, no dispatch) | 6105  | 2413    | 2350      | 2334    | 1166    | 24804      |

---

## Ideas not tried

- **Trick 2B (chunk merging):** pair chunks with complementary `n_left`
  (e.g. 3 + 5 = 8) into one 8-store batch, halving phase-3 loads.
  Best-case analytic savings ≈ 0.5 c/chunk, offset by per-chunk merge
  bookkeeping and possibly additional inter-register TBL work.  Would
  close the gap to ~−15% rather than −36% on proba80 at best, still a
  regression.  Not implemented.
- **Unconditional 8 strbs with sentinel:** always do 8 stores per chunk,
  with unused lanes pointing at a scratch byte outside the output.  Zero
  dispatch, zero mispredict — but wastes `8 − n_left` stores per chunk.
  On proba80 (n_left 5–7) that's 1–3 wasted stores per chunk — maybe
  viable.  On proba50 / geometric (n_left 3–5) it's 3–5 wasted stores
  per chunk — clear regression.  Not implemented; proba80-only win,
  which is the case neon is already fastest on.

---

## Takeaway

The baseline `pivco_huffman_neon` decoder is already tight enough on M4
that the "save a vld+vst per chunk" opportunity I estimated was both too
small and incorrectly counted.  Any form of per-chunk rework — switch
dispatch, jump table, or bucketed two-pass — loses more in extra loads
or branch flushes than it saves in avoided stores.

The concept *could* still pay off on architectures where:

- SIMD scatter is fast (reduces `scatter_sym` cost that the baseline
  pays) — but NEON doesn't have scatter.
- Indirect branch prediction is cheap even for shifting 9-target
  distributions — the M4 ITAC apparently doesn't handle this shape.
- `vld`/`vst` bandwidth is the binding constraint and store-side savings
  matter more than load-side duplication — not true on M4 at this
  working set size.

None of those is M4 NEON's profile, so this experiment is filed as a
negative result.  Main entry in `README.md` → "Tested and discarded".
