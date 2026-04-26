# Partition store coalescing — investigation log

A multi-day investigation into whether `partition_8`'s two `vst1q_u8`
stores per iteration could be reduced — and the conclusion that on
Apple M4 they cannot, with the data and reasoning to back it up.

Working code in [`extras/bench_coalesce.c`](extras/bench_coalesce.c).
Raw results in
[`results/coalesce-m4_max-20260426.txt`](results/coalesce-m4_max-20260426.txt).
Build via `cmake --build build --target pivco_bench_coalesce`, run
the standalone binary.

## TL;DR

Three increasingly clever variants tested.  All three lose to baseline
on M4, even after eliminating the obvious failure modes (branch
mispredict, cross-iter dep chains).  The store-port saturation in the
production kernel is a tight Pareto-optimal point on M4: any added
SIMD work to enable store reduction exceeds what coalescing saves.

| Variant | 50% random | skew popcount 2/6 | vs baseline |
|---|---:|---:|---:|
| **baseline** (2× vst1q per iter) | 12.5–15.5 GB/s | 15.5 GB/s | 1.00× |
| coalesce_vext (switch on so_far) | 3.8 GB/s | 4.6 GB/s | **0.30×** |
| coalesce_tbl (runtime shuf) | 7.6 GB/s | 7.6 GB/s | **0.55×** |
| coalesce_macro (4-iter lookahead) | 9.6 GB/s | 9.6 GB/s | **0.66×** |

GB/s figures vary across runs by ~5% (M4 thermal/boost variance);
ratios are stable.

## Motivation

The xctrace M4 profile of decoding `prose_pride` (real-world deep-tree
Huffman workload, see [README.md §Profiling](README.md#profiling))
showed `partition_8`'s two `vst1q_u8` stores accounting for **~38% of
all decode CPU time**.  Bench microbench data
([Key Compute Primitives](README.md#key-compute-primitives))
confirmed: the kernel issues stores at exactly **1 vst1q/cycle = 64
GB/s** on the M4 store port — the architectural ceiling.  Two stores
per iteration → 2 cycles/iter, with the SIMD compute fitting in
parallel.

The natural follow-up question: **can we issue fewer stores?**  Each
iteration writes 16 bytes to each side, but the compress operation
produces only `popcount(mask)` valid bytes on the right (and `8 -
popcount(mask)` valid on the left) — the rest are zero-padding that
the *next* iteration's write overwrites.  In the average case
(`popcount` ≈ 4 for random masks), each store wastes half its
bytes.  If we could buffer in a register and only flush when full,
we'd issue ~1 store per iter on average — a theoretical 50% saving.

## Pre-experiment data

Two pieces of data collected before writing any prototypes, both via
[`bench_micro.c`](bench/bench_micro.c) and
[`bench_partition_skew.c`](extras/bench_partition_skew.c):

### M4 SIMD throughput probes

Sixteen independent chains of one NEON op:

| Op | M4 throughput |
|---|---:|
| `vqtbl1q_u8` (16-byte 1-source TBL) | **3.8 ops/cycle** |
| `vqtbl2q_u8` (32-byte 2-source TBL) | 4.0 ops/cycle |
| `vqtbl4q_u8` (64-byte 4-source TBL) | 1.4 ops/cycle |
| `vextq_u8` (immediate-shift permute) | 3.6 ops/cycle |

Apple M4 P-core has effectively a 4-way SIMD pipe that runs `vqtbl1q`
and `vextq` at ~4/cycle.  This invalidates the "TBL is at 1/cycle, so
adding place-at-offset TBLs would saturate it" objection: there's
plenty of TBL bandwidth for the place-shift work.

### Partition skewness histogram

For each Huffman distribution, the smaller-side fraction
`min(L, R) / total` at every internal node, weighted by elements
flowing through that node:

| Distribution class | Mean smaller-side fraction |
|---|---:|
| Real text (prose, html, json, source, log, chinese) | **~46%** (near-balanced) |
| Heavy-skew (proba80) | 22.5% (highly predictable) |
| Two-symbol skew (two_sym_90/10) | 12.5% (very predictable) |
| Already root-flat | n/a — never hits `partition_8` |

For the workloads where partition stores actually matter (real text),
each iteration's popcount ≈ 4 with stdev ~1.4 — so the running
`so_far` counter walks roughly i.i.d. through `[0, 7]`, and any
"is the buffer full?" branch is essentially a coin flip.

This was the first warning sign — but the throughput data above
suggested coalescing might still pay off if implemented carefully.

## Variant 1: per-iter coalesce with switch + vextq

The cleanest design: maintain `(accum, so_far)` per side, where
`so_far ∈ [0, 7]`.  Each iter:

1. Compress (1 TBL).
2. Shift the compressed bytes left by `so_far * 2` bytes via
   `vextq_u8` with a compile-time-constant immediate, dispatched via
   `switch (so_far) { case K: vextq_u8(zero, compacted, 16 - K*2); }`.
3. OR into `accum`.
4. If `so_far + cnt >= 8`: store `accum`, set up new `accum` from
   the overflow bytes (right-shifted via another `vextq` immediate),
   advance the side's byte counter.

Pros: `vextq_u8` runs on a different SIMD pipe than `vqtbl1q`, so the
place-shift cost doesn't compete with the compress TBLs.

The implementation is ~80 lines per side because each `case K`
needs its own constant-immediate `vextq` — Apple Clang range-checks
NEON immediates during semantic analysis, and even
`__builtin_choose_expr` doesn't dodge the check.  Two macros
(`COALESCE_CASE_0` for K=0 with no shift, `COALESCE_CASE_K` for
K∈[1,7]) keep it manageable.

### Result: **0.27 ns/elem (3.8 GB/s) — 4× slower than baseline.**

Disassembly + measured ~9 cycles/iter on M4 = baseline + ~7 cycles
of overhead.  That's consistent with **frequent indirect-branch
mispredicts on the 8-way switch jump table** (M4 mispredict penalty
~10–15 cycles, two switches per iter).  The skewness histogram
predicted this: `so_far` walks pseudo-randomly, the predictor
can't track it.

## Variant 2: per-iter coalesce with runtime TBL

To remove the indirect branch, replace the switch with one runtime-
computed shuffle:

```c
shuf_left = vsubq_u8(iota, vdupq_n_u8(so_far * 2));   /* 1 vsub */
shifted   = vqtbl1q_u8(compacted, shuf_left);         /* 1 TBL */
merged    = vorrq_u8(accum, shifted);                 /* 1 OR */
```

`vqtbl1q_u8` maps any index ≥ 16 to 0, so the `(iota - so_far*2) mod
256` subtraction naturally produces out-of-range markers for lanes
that should be zero.  Same trick for the right-shift on overflow:
`vaddq_u8(iota, vdupq_n_u8((8 - so_far) * 2))`.

Per side per iter: 1 TBL + 1 OR + 1 vsub/vadd + 1 cond branch (the
flush check).  No indirect branch.

### Result: **0.13 ns/elem (7.6 GB/s) — 2× slower than baseline.**

Major improvement over the switch variant (eliminated the
mispredict storm), but still 2× behind baseline.  The new bottleneck:
**iteration-to-iteration dependency through `accum` and `so_far`**.
Each iter's shift amount depends on the previous iter's running
`so_far`; each iter's `accum` input depends on the previous iter's
output.  M4's OoO engine can't overlap consecutive iterations
anymore.

Per-iter critical path:
```
mask byte load → popcnt → vsub (build shuf) → vqtbl1q (shift) →
vorr (merge) → cmp/branch → so_far/accum update
```
~5 cycles serial latency.  Baseline has no cross-iter SIMD chain
(the integer `n_left`/`n_right` updates rename through the OoO
window), so iters pipeline freely at 2 cycles each.

## Variant 3: 4-iter macro-block with lookahead

Break the dep chain by **precomputing all 4 prefix sums upfront**
(scalar), so each iter's destination offset is determined *before*
any place-shift TBL fires:

```c
/* Read 4 mask bytes, popcount each, prefix-sum scalar */
int pop0 = popcnt[m0], pop1 = popcnt[m1], pop2 = popcnt[m2], pop3 = popcnt[m3];
int cum1 = pop0;        /* iter 1 starts at cum1 */
int cum2 = cum1 + pop1; /* iter 2 starts at cum2 */
int cum3 = cum2 + pop2; /* iter 3 starts at cum3 */
int total = cum3 + pop3;

/* Compress each iter, place into a 32-byte (lo, hi) accumulator
 * via 2 TBLs each (lo half + hi half).  All places are independent
 * — no cross-iter dep on accum state. */
lo |= compress(m0);                                    /* iter 0: cum=0 */
{ shuf_lo = iota - cum1*2;  shuf_hi = iota + 16 - cum1*2;
  lo |= vqtbl1q(compress(m1), shuf_lo);
  hi |= vqtbl1q(compress(m1), shuf_hi); }
{ ... iter 2 ... }
{ ... iter 3 ... }

/* Always store both halves; advance by total*2 bytes. */
vst1q_u8(out + n_out,      lo);
vst1q_u8(out + n_out + 16, hi);
n_out += total * 2;
```

Per macro-block: **always exactly 2 stores per side = 1 store/iter**,
half of baseline's 2/iter.  And the lookahead means iter N+1's place
TBLs don't depend on iter N's `accum` state — the OoO engine can
freely pipeline within the macro-block.

### Result: **0.10 ns/elem (9.5 GB/s) — 1.6× slower than baseline.**

The cleanest of the three — the lookahead trick worked, the cross-iter
dep chain is gone.  But it introduces a **new** bottleneck.  Per
macro-block (4 iters) the SIMD work-budget is:

| Op | Per macro | Per iter |
|---|---:|---:|
| Compress TBLs | 8 | 2.0 |
| Place TBLs (lo + hi per iter, both sides) | 12 | 3.0 |
| `vorrq` (merge into accumulators) | 12 | 3.0 |
| Shuf-vector ALU (vdupq + vsubq + vaddq) | 18 | 4.5 |
| **Total SIMD ops** | **50** | **12.5** |
| Stores | 4 | 1.0 |

At M4's ~4 SIMD ops/cycle: 12.5 / 4 = **~3.1 cycles/iter** —
SIMD-throughput-bound, matching the measured 3.3.

Baseline is store-port-bound at 2 cycles/iter (2 stores × 1 cycle).
The macro variant saves 1 store cycle/iter but adds ~2.5 cycles of
SIMD work to compute placements.  Net **−1.3 cycles/iter** —
slower.

## Why this is a *fundamental* result, not a coding problem

The three variants form a clean progression that rules out the
obvious failure modes:

1. **vext (switch)** ruled out indirect-branch mispredict as the
   sole problem.  Removing the switch does help (variant 2 is 2× the
   vext variant), but doesn't recover the gap to baseline.
2. **tbl (runtime shuf)** ruled out branch overhead.  Eliminating
   the mispredict-prone inner branch helps, but exposes the
   cross-iter dep chain.
3. **macro (4-iter lookahead)** ruled out the cross-iter dep chain.
   With the chain broken, iterations pipeline freely — but the
   added SIMD work to enable coalescing exceeds what's saved.

Each variant exposed a deeper bottleneck.  Reaching variant 3 and
*still* losing means the issue isn't a clever-coding problem; it's
the M4's resource balance.

## What it would take for this to win

The optimization assumes a uarch where:

- **SIMD throughput is much higher relative to store-port**.  M4 is
  4 SIMD ops/cycle and 1 store/cycle (4:1 ratio).  Even at this
  ratio, the place-shift work eats the savings.  A uarch with 8:1
  or 16:1 SIMD-to-store ratio could plausibly tip the math.
- **Store-port bandwidth is genuinely scarcer than SIMD**.  E.g., a
  uarch where unaligned stores are ½-rate, or where stores conflict
  with loads on a shared port.
- **A "compress-and-shift" instruction exists**, fusing the TBL
  shuffle with a lane offset — eliminating the place-shift TBLs.
  No NEON instruction does this; AVX-512 has `vpcompressb` which
  comes close.

None of these apply on M4.

## Could it win on another platform?

Open question.  Plausible candidates:

- **Graviton 4 (Neoverse-V2)**: store port may be similar to M4 (1
  SIMD store/cycle); SIMD is also ~4 ops/cycle but `vqtbl{2,4}q_u8`
  is much slower than M4 (per [Key Compute Primitives](README.md)).
  Would need a new throughput probe to estimate.
- **Xeon AVX-512 (Sapphire/Granite Rapids)**: `vpcompressw` already
  does the compress-and-place in one instruction; the analogue of
  this whole investigation is moot — production AVX-512 backend
  already uses `vpcompressw` and gets the optimal store pattern for
  free.
- **Zen 3 SSE4.1**: store port is the documented bottleneck on the
  Zen 3 partition path (see `bench_micro` data).  Worth running the
  same prototypes — but the absence of efficient cross-byte permute
  primitives in pure SSE4.1 makes the place-shift expensive on Zen 3.

If anyone re-runs these on a different platform, the standalone bench
(`./build/pivco_bench_coalesce`) reproduces the four numbers in ~30
seconds.

## Conclusion

The 0.06 ns/elem partition_8 baseline on M4 is a **genuine
Pareto-optimal point**.  It's not "stores are slow"; it's "the
kernel is correctly issuing stores at exactly the rate the M4 can
sustain, and the SIMD work to enable issuing fewer of them costs
more than the saved cycles."

The way to make `partition_8` faster on M4 is not to optimise the
kernel — it's to *avoid invoking it*.  That's exactly what the
production decoder already does for the cases where it matters:

- **Half-tree partition** (`partition_8_right`): when one child is a
  leaf-or-prefilled, only one side needs to be stored.
- **Both-leaves stage fusion** (`scatter_both_leaves`): when both
  children are leaves, skip partition entirely.
- **Flat-subtree fast path** (`flat_decode_*`): when the subtree is
  uniform-depth, replace the partition tree-walk with a direct
  D-bit lookup — 0.02 ns/elem at D=2/D=4.

Everything has already been done that can be done.

## Files

- [`extras/bench_coalesce.c`](extras/bench_coalesce.c) — the three
  prototypes + baseline, self-contained (~370 lines).
- [`results/coalesce-m4_max-20260426.txt`](results/coalesce-m4_max-20260426.txt)
  — raw measurement output.
- [`extras/bench_partition_skew.c`](extras/bench_partition_skew.c) —
  the per-distribution skewness histogram tool used in the
  pre-experiment analysis.
- [`bench/bench_micro.c`](bench/bench_micro.c) — TBL/vext throughput
  probes, store-port topology probe.
