# Cross-Block Fusion: Investigation Log

> **Last content review:** _NEVER_

Comprehensive record of the cross-block fusion experiment — what we tried,
what worked, what didn't, and the end-to-end numbers we landed on. Parked
2026-05-09 with a clear understanding that microbench predictions
over-promise vs. real-decoder gains by 2–5×.

**Status: in-source fusion code removed; preserved as `extras/fusion.diff`.**
Apply that patch on top of pre-fusion `bcc092b` to restore the working
fusion implementation (NEON: full coverage; SSE: LEAF-only). The
microbenches in `extras/bench/bench_fusion_v{3,4,5}_*_cnt.cpp` are kept in tree
as standalone research artifacts — they don't depend on the production
fusion API.

## Premise

The decoder has two heavy phases per block:

1. **`root_full`** — partition the root bitmap, producing `indices[]` (left
   side) and `tmp[]` (right side) arrays for the recursion. ~590 ns/block on
   M4 prose_pride (8192 elements at 0.07 ns/elem).
2. **Recursion** — walks the Huffman tree, scattering symbols via
   `symbols[indices[i]] = sym`. Multiple primitives (`scatter_sym`,
   `scatter_both_leaves`, `flat_decode_scatter`) called repeatedly.

These two phases are **temporally separated** but **structurally
independent**: `root_full` of block B+1 only depends on B+1's input bitmap,
not on anything block B is doing. So while block B is running its scatter
loops (store-port-bound on M4/G4), we could run B+1's `root_full` partition
in the OOO-overlap window — for "free" — and B+1 starts with its
partition already done.

Microbench predicted up to **24–37% saving** on the partition+scatter
phase. End-to-end we got **+0.6–1.7%** on wall time. The rest of this
document explains the gap.

## Final result

| Platform | end-to-end (prose_pride dual_decode_test) | v4 microbench (P+S phase) |
|---|---:|---:|
| M4 NEON | **+1.7%** | 8.4% |
| Graviton 4 NEON | **+0.6%** | 18.9% |
| Zen 3 SSE (LEAF-only port) | ~0% | 15.7% |
| Xeon AVX-512 | (not implemented) | 3.9% (was 35% noise) |
| Xeon SSE | (not implemented) | 22.7% |

**Decision**: parked. NEON fusion is shipped and works correctly, but the
gain is small. x86 ports not worth the engineering cost — see "Why x86
port was skipped" below.

## How fusion works (NEON, shipped)

State carried across blocks via two **ping-pong slots** (`g_la[2]`):

```c
typedef struct {
    int j;                  /* partition cursor (0..PIVCO_BLOCK_SIZE) */
    int n_left;
    int n_right;
    const uint8_t *bm;      /* next-block root bitmap */
    uint16_t *indices;      /* next-block left-output buffer */
    uint16_t *tmp;          /* next-block right-output buffer */
} pivco_la_neon_t;
```

Caller pattern (see `extras/bench/bench_dual_decode_test.c`):
```c
g_pivco_fusion_enabled = 1;
for (each block N) {
    pivco_huffman_set_next_neon(N+1 < last ? in[N+1] : NULL);
    pivco_huffman_decode_neon(in[N], ...);
}
```

Inside the decoder, three fused kernels exist (file: `src/pivco_huffman_neon.c`):
- `scatter_sym_fused_root_full` — LEAF case
- `scatter_both_leaves_fused_root_full` — BOTH_LEAVES case
- `flat_decode_scatter_neon_fused_root_full` — INTERNAL_FLAT case (D=2,4 only)

Each fused kernel has the shape:

```c
/* Per fused chunk: 16 scatter elements + K * partition_root_8 = 8K
 * partition elements.  K=4 by default (matches v4 microbench sweet spot). */
for (int j = 0; j < n_fused; j += 16) {
    /* === scatter side === */
    uint16x8_t i0 = vld1q_u16(indices + j);
    uint16x8_t i1 = vld1q_u16(indices + j + 8);
    symbols[vgetq_lane_u16(i0, 0)] = sym;  /* ... 16 byte stores ... */

    /* === partition side === */
    #pragma GCC unroll PIVCO_LA_K
    for (int k = 0; k < PIVCO_LA_K; k++) {
        /* partition_root_8 inlined: dup + tbl + tbl + str + str + popcnt */
        nxt_j += 8;
    }
}
```

`n_fused` is precomputed at function entry as
`min(n & ~15, max_p_iters * 16)` so the inner loop has **no per-iter
branch on partition budget**. Tail (`n - n_fused`) falls back to the
plain non-fused kernel.

`g_la_writes` is set once per block at the entry to the FULL-partition
branch only — half/both-leaves paths don't enable fusion since their
next-block's root will likewise skip fusion (table is shared across
blocks of a stream, so root-shape is fixed).

## Microbench evolution

We went through **five versions** of microbench. Each fixed a specific
flaw the previous one had.

### v1/v2 (`extras/bench/bench_fusion_micro_cnt.cpp`, `_v2_cnt.cpp`)
Earliest probes. Sketched the partition-vs-scatter overlap question
without representative element counts.

### v3 (`bench_fusion_v3*_cnt.cpp`)
"Realistic" — sorted-ascending scatter indices, varying P:S element
ratio (PpS = 1, 2, 4, 8). Three variants per platform: NEON, SSE,
AVX-512.

Result on M4:
- PpS=4: PP_SS 279 ns vs PSPS 211 ns → **24% saving**
- Predicted big wins.

**Flaw**: N=64 outer iters is too small. PP_SS does 256 partition calls
followed by 64 scatter calls; the OOO window can almost span both
loops, so PP_SS is artificially pessimistic vs. real code where
hundreds of cycles of recursion sit between root_full and the first
scatter.

### v4 (`bench_fusion_v4*_cnt.cpp`)
**Block-realistic sizes.** P_ELEM=8192 (full block partition),
S_ELEM=4096 (≈ prose_pride scatter coverage). Same kernel reimpl as v3
but at scale.

Result on M4 (S_ELEM=4096, K=4):
- serial_tight: 1051 ns
- fused: 963 ns → **8.4% saving**

Cross-arch v4 results (best K, S_ELEM=4096):
- M4 NEON:    8.4% (K=4)
- G4 NEON:   18.9% (K=4)
- Zen3 SSE:  15.7% (K=2)
- Xeon SSE:  22.7% (K=2)
- Xeon AVX-512: **3.9%** stable (initial 35.9% reading was timer noise on a single run)

This was much more honest than v3. But still over-predicted reality by
~5× on M4 and ~30× on G4.

### v5 (`bench_fusion_v5_real_kernel_cnt.cpp`)
**Uses the actual production kernel body verbatim**, not a re-implementation.
This was the user's challenge — "do we use our actual kernels in
microbenchmarks?" The answer for v3/v4 was **no** (they used `p_chunk`
and `s_chunk` reimplementations with by-reference parameters). v5
copies `scatter_sym_fused_root_full` byte-for-byte from
`pivco_huffman_neon.c`.

Result on M4 (S_ELEM=4096, K=4):
- serial_tight:           925 ns (12% faster than v4!)
- fused 1-call:           849 ns → **8.2% saving**
- fused 16-calls (S/16):  878 ns → **5.0% saving**
- fused 32-calls (S/32):  907 ns → 1.9% saving

Real decoder calls `scatter_sym_fused_root_full` ~18 times per block
on prose_pride. v5's "16 small calls" variant predicted **5.0%**.
End-to-end measured 1.7% wall = 3.4% on the P+S phase (since P+S is
~50% of total decode).

So v5 closed most of the gap: from "v4 says 8% on M4 vs reality 1.7%" to
"v5-many-calls says 5% on the kernel, P+S is 50% of decode → 2.5%
wall expected, 1.7% measured". Remaining ~50% erosion is attributable
to integration overhead inside `decode_node_neon`'s switch dispatch
and recursion.

## Where the gap goes (M4 prose_pride decomposition)

Per-block accounting from `pivco_huffman_profile_english`:

| primitive | NO FUSION ns/blk | FUSED ns/blk | Δ |
|---|---:|---:|---:|
| node_full (interior) | 1075 | 1097 | +22 |
| node_half_right | 107 | 107 | 0 |
| **root_full (entry)** | **594** | **2** | **−592** |
| scatter_sym | 224 | 369 | **+145** |
| scatter_both_leaves | 258 | 368 | **+110** |
| flat_decode_scatter | 634 | 868 | **+234** |
| Sum | 2892 | 2811 | −82 |
| **Wall** | **3092** | **3032** | **−59** |

The **scatter primitives all run ~+0.11 ns/elem more expensive in
fused mode** (e.g. scatter_sym 0.146 → 0.259 ns/elem). That's the
intrinsic cost of carrying 32 partition elements per 16 scatter
elements through the same loop body. The **plain-tail** runs at
~0.18-0.19 ns/elem (vs 0.15 plain, no fusion mode) because it's
processing small chunks (60-180 elem/call) that don't amortize per-
call setup.

So fusion buys 594 ns of partition for 488 ns of scatter overhead.
Net 100 ns/block, ~2% wall. That ratio holds on M4 but not always on
other archs (G4 has 30× erosion factor due to wider OOO already
draining the store-buffer naturally during recursion).

## Per-call overhead investigation

User pushed back on "kernel call overhead can't be that high." Block-
size sweep on M4 (S_ELEM = BLK/2, K=4) — saving% vs serial:

| BLK | 1-call | 4-c | 16-c | 32-c | call-size at 16-c |
|---:|---:|---:|---:|---:|---:|
| 2048 | 6.8% | 0.5% | -11.2% | -17.8% | 64 elem |
| 4096 | 8.6% | 6.4% | 3.6% | -2.9% | 128 elem |
| **8192** (default) | 8.0% | 7.4% | 5.5% | 2.3% | 256 elem |
| 16384 | (cache-thrash anomaly) | — | — | — | — |
| 32768 | 3.8% | 3.7% | 3.0% | 3.0% | 512 elem |

Below ~100-200 elements per call, per-call overhead overwhelms the
partition savings. Real decoder's prose_pride has ~226 elem/call on
average (4087 fused elements / 18 calls), which puts it in the
"barely positive" zone.

Per-call overhead breakdown (~1.5-2 ns/call on M4):
- Function entry/exit: ~3 cycles
- `g_pivco_fused_calls++`: 1-2 cycles (one global store)
- Read 6 fields from `nxt`: 2-3 cycles
- `n_fused = min(...)` calc: 3 cycles
- Write back 3 fields to `nxt`: 3 cycles
- Total: ~12-15 cycles ≈ 1.2-1.5 ns at M4's ~3 GHz

## Compiler/codegen verification

Compared assembly between v5 microbench (.cpp, clang++) and
production decoder (.c, clang) at the same point in the fused inner
loop. **Every instruction matches** — only register names differ
(different register allocation due to surrounding context).

| v5 (.cpp, offset a3d0+) | production (.c, offset 1c8c+) |
|---|---|
| `ldp q1, q2, [x14, #-0x10]` | `ldp q1, q2, [x16, #-0x10]` |
| `umov.h w6, v1[0]` | `umov.h w1, v1[0]` |
| `strb w15, [x0, x6]` | `strb w8, [x4, x1]` |
| (… 16 byte stores in identical pattern …) | |
| `dup.8h v1, w9` | `dup.8h v1, w11` |
| `tbl.16b v2, {v1}, v2` | `tbl.16b v2, {v1}, v2` |
| `str q2, [x13, x7]` | `str q2, [x15, x5]` |
| (… etc, identical pattern through K=4 partition …) | |

Same compiler family (Apple clang), same `-O3 -arch arm64`, same
backend (LLVM). Difference is C (`-std=gnu11`) vs C++ (`-std=gnu++17`)
frontend, but they emit the same code for the same body.

**Conclusion**: the kernel itself is fine. Microbench-vs-reality gap
comes entirely from surrounding context (cache state, branch
prediction, register pressure inside `decode_node_neon`'s switch +
recursion).

## SSE port attempt (Zen 3)

Did a minimal SSE fusion port (LEAF case only) on `pivco_huffman_x86.c`:
- Added `pivco_la_x86_t` ping-pong state
- `pivco_huffman_set_next_x86()` API
- Resumable root_full in entry function
- One fused kernel: `scatter_write_sse_fused_root_full`
- Wired only PIVCO_NODE_LEAF dispatch (skipped scatter_both_leaves
  and flat_decode_scatter)

**Result on Zen 3 (test-c6a)**: ~0% end-to-end on prose_pride.

LEAF is only ~22% of scatter on prose_pride. The other 78% (scatter_both_leaves
22%, flat_decode_scatter 56%) still uses plain code. Fused coverage is too
small. Correctness verified, but no measurable wall-time win.

To make SSE fusion meaningfully positive, would need:
- scatter_both_leaves fused (currently inlined at decode_node_x86 L559-581
  — needs factoring out first)
- flat_decode_scatter fused (D=4 only on pure SSE; D=2/3/5/6/7/8 are scalar)
- Theoretical ceiling per math: ~1.5% wall on Zen 3 prose_pride

Not worth the engineering cost given M4 NEON only delivers 1.7% and
Zen 3 has wider OOO (likely worse erosion factor).

## Why x86 port was skipped

| platform | v4 microbench (stable median) | expected end-to-end (5–30× erosion) |
|---|---:|---:|
| Xeon AVX-512 | 3.9% | 0.1–0.8% (likely wash) |
| Zen 3 SSE | 15.7% | 0.5–3% |
| Xeon SSE (rarely used) | 22.7% | 0.7–4.5% |

AVX-512's `vpcompressw` is so fast that there's little partition cost
to overlap (root_full at ~0.16 ns/elem on G4, much faster on Xeon
AVX-512). SSE on Zen 3 is the only remotely-meaningful target, and
the work is substantial:

- Add ping-pong + state plumbing (done in port)
- Fuse 3 primitives (only 1 done in port)
- Each requires correctness testing + perf measurement
- Total ~1-2 hours per primitive of focused work
- Theoretical ceiling: 1-3% wall-time gain
- Expected: <1% based on M4/G4 experience

Not a good ROI compared to other optimization paths in `IDEAS.md`.

## What's shipped

Files modified for fusion:

- `src/pivco_huffman_neon.c`:
  - `pivco_la_neon_t` struct, `g_la[2]` ping-pong, `g_la_writes`
  - `pivco_huffman_set_next_neon()` public API
  - 3 fused kernels (scatter_sym, scatter_both_leaves, flat D=2/D=4)
  - Dispatch wired in `decode_node_neon`'s LEAF, BOTH_LEAVES, INTERNAL_FLAT cases
  - Resumable `root_full` consuming pre-state from `cur` slot
  - `g_la_writes` enabled only inside the FULL-partition branch
- `src/pivco_huffman_x86.c`:
  - `pivco_la_x86_t` ping-pong state
  - `pivco_huffman_set_next_x86()` API
  - LEAF-only fused kernel (`scatter_write_sse_fused_root_full`)
  - Resumable root_full
  - **NB: incomplete coverage; 0% end-to-end on Zen 3 — keep but don't
    enable in production until full coverage is added**
- `src/pivco_huffman.c`:
  - `g_pivco_fusion_enabled` global (shared NEON+SSE)
- `include/pivco_huffman.h`:
  - Public API for `g_pivco_fusion_enabled`, `set_next_neon`, `set_next_x86`
- `include/pivco_prof.h` + `src/pivco_prof.c`:
  - PROF_*_FUSED counter IDs (separate timing for fused vs plain primitives)
  - PROF_ROOT_BOTH_LEAVES (for the dedicated root both-leaves path)
- `extras/bench/bench_profile_english.c`:
  - Two-pass profiler (NO FUSION + FUSED) so per-primitive deltas are visible
- `extras/bench/bench_dual_decode_test.c` and `_x86.c`:
  - Correctness + perf test for fusion (2-block decode, fusion on/off)
- `extras/bench/bench_fusion_v4_*.cpp`, `bench_fusion_v5_real_kernel_cnt.cpp`:
  - Block-realistic + real-kernel microbenches (NEON, SSE, AVX-512)

`g_pivco_fusion_enabled` defaults to **0**. Fusion is opt-in via the
caller pattern shown above; default builds and the existing
`pivco_huffman_decode()` dispatcher don't enable it.

## Engineering levers if revisiting

In rough order of estimated payoff vs. cost:

### 1. Hoist `g_la_writes` branch to once-per-block (medium effort, +0.3–0.5%?)
Currently every leaf dispatch has `if (g_la_writes != NULL) { fused }
else { plain }`. Could specialize `decode_node_neon` into two variants
(fused vs plain) and dispatch once at the entry. Eliminates ~18
mispredicted-branch checks per block.

### 2. Reduce per-call kernel overhead (low effort, +0.5%?)
Strip `g_pivco_fused_calls++`, hoist `n_fused` calc, force aggressive
inlining via `__attribute__((always_inline))`. Saves ~15 cycles per
fused call × 18 calls = 270 cycles ≈ 90 ns/block on M4.

### 3. Coalesce adjacent leaf scatters (high effort, variable payoff)
Some recursion paths visit several siblings consecutively; their scatters
could be fused into one larger call. But the recursion is a depth-first
walk, so adjacent-in-time leaves aren't always adjacent-in-output.

### 4. Different fusion target — partition into RECURSION instead of LEAVES
Currently fused kernel adds partition work to leaf scatters. Could
instead overlap NEXT block's partition with CURRENT block's interior
`node_full` partitions — bigger calls, less per-call overhead.
Substantial redesign though.

### 5. AVX-512: don't bother
Stable v4 prediction is 3.9%. Even at 0× erosion that's marginal. The
`vpcompressw` partition is so cheap there's nothing to fuse against.

### 6. SSE Zen 3: extend port to all primitives (medium effort, +1–2%?)
Predicted ceiling is ~1.5–3% on prose_pride. Worth doing only if the
extension itself is interesting (e.g. as a teaching exercise for the
SSE backend's structure) — not as a perf win.

## Cross-references

- `extras/bench/bench_fusion_v3_cnt.cpp` — original microbench (over-predicted)
- `extras/bench/bench_fusion_v4_cnt.cpp` — block-realistic microbench
- `extras/bench/bench_fusion_v5_real_kernel_cnt.cpp` — real-kernel microbench (definitive)
- `extras/bench/bench_dual_decode_test.c` — NEON correctness + perf
- `extras/bench/bench_dual_decode_test_x86.c` — SSE correctness + perf
- `extras/bench/bench_profile_english.c` — prof comparison (NO FUSION vs FUSED)
- `IDEAS.md` — broader optimization ideas list
- `docs/KERNELS.md` — NEON kernel walkthroughs

## Recovery

To resurrect the implementation:

```sh
# From the pivco-huffman directory:
git checkout bcc092b -- src/pivco_huffman_neon.c src/pivco_huffman_x86.c \
                        src/pivco_huffman.c include/pivco_huffman.h \
                        include/pivco_prof.h src/pivco_prof.c \
                        extras/bench/bench_profile_english.c
git apply extras/fusion.diff
```

The diff was generated as `git diff ad18740..HEAD` (committed work) plus
the uncommitted polish (n_fused refactor, scatter_sym/flat fused kernels,
SSE port, dual_decode tests, profile-english two-pass harness). Total
~1800 lines.

## Verdict

**Fusion works. The microbench was honest about kernel-level savings.
The end-to-end ceiling is just lower than the kernel-level ceiling**
because:

1. Adding partition to scatter raises per-elem scatter cost from ~0.15
   to ~0.27 ns/elem (M4). The saved partition cost (~0.07 ns/elem ×
   8192 elem) barely exceeds the added scatter cost (~0.11 ns/elem ×
   ~4087 fused elem). Net ~100 ns/block of true work saved, which is
   ~3% of total decode time.

2. Per-call overhead from many small fused kernel invocations
   (~18/block on prose_pride) costs another ~30 ns/block.

3. Surrounding context (cache state, branch prediction, register
   pressure inside `decode_node_neon`) costs another ~10-30 ns/block.

End result: ~60 ns/block out of 3092 ns/block ≈ **+2% on M4**.
That's the moon, and we got there. It just isn't a very big moon.
