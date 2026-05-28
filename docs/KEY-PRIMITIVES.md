# Key Compute Primitives

> **Last content review:** _NEVER_

Bottoms-up per-element cost of every SIMD primitive the decoder uses,
isolated from the surrounding control flow.  Useful for reasoning
about which inner-loop pieces are the bottleneck at a given D / table
shape.

The primitives below are the same bodies that landed in the
`primitives_<backend>.h` headers as part of the 2026-05-14
unify-framework refactor — the microbench cost is functionally
identical, only the file location changed.  `flat_scatter_*` were
the TD-era scatter variants (since retired); production decode uses
the contiguous `flat_direct_*` rows.

Measured by [`bench/bench_micro.c`](../bench/bench_micro.c) on all
four test platforms (block N = 8192, 100k repeats per row, ~820M
elements total per row).  Numbers are **ns / elem** (lower is
better).  Build:

```sh
# NEON (M4 / Graviton 4):
cc -O2 -o bench_micro bench/bench_micro.c -I include -I src
# x86_64 (Xeon AVX-512 / Zen 3 SSE4.1+AVX2):
cc -O3 -march=native -o bench_micro bench/bench_micro.c -I include -I src
```

Raw outputs in
[`results/bench_micro-*-20260426-0625.txt`](../results/).

## Cross-platform primitive costs

Each cell shows **`ns/elem (GB/s)`**.  The table treats one element
as one output byte (the symbol decode), so GB/s is throughput in
output bytes per second.

| Primitive                                | M4 (NEON)        | Graviton 4 (NEON) | Xeon (AVX-512 VBMI2) | Zen 3 (SSE4.1+AVX2) |
|------------------------------------------|------------------|-------------------|----------------------|---------------------|
| **Reference floors**                     |                  |                   |                      |                     |
| `memset`                                 | 0.01 (119)       | 0.01 (81)         | **0.00 (226)**       | 0.01 (112)          |
| `scatter_scalar`                         | 0.23 (4.4)       | 0.36 (2.8)        | **0.17 (5.8)**       | 0.28 (3.6)          |
| **Partition (2-way decoder core)**       |                  |                   |                      |                     |
| `partition` (load + TBL/compress)        | 0.06 (15.6)      | 0.16 (6.3)        | **0.04 (24.2)**      | 0.20 (5.1)          |
| `partition_root` (identity + TBL)        | 0.07 (14.6)      | 0.16 (6.3)        | **0.05 (20.7)**      | 0.19 (5.2)          |
| `partition_half` (load + 1 TBL)          | 0.05 (21.9)      | 0.11 (9.2)        | **0.03 (36.1)**      | 0.13 (7.5)          |
| `partition_root_half`                    | 0.05 (19.8)      | 0.11 (8.9)        | **0.03 (30.9)**      | 0.12 (8.4)          |
| **Indexed scatter (leaf write)**         |                  |                   |                      |                     |
| `scatter_simd` (const sym)               | **0.13 (7.5)**   | 0.36 (2.8)        | *(= scatter_scalar)* | 0.35 (2.8)          |
| `both_leaves_vst1` (root flat 2-sym)     | 0.03 (33.7)      | 0.07 (13.9)       | **0.01 (67.7)**      | 0.07 (14.2)         |
| `both_leaves_scatter` (idx 2-sym)        | **0.15 (6.6)**   | 0.40 (2.5)        | 0.17 (5.7)           | 0.33 (3.0)          |
| **Flat-subtree direct** (sequential out) |                  |                   |                      |                     |
| `flat_direct_d2`                         | **0.02 (52.1)**  | 0.05 (18.4)       | 0.03 (30.4)          | *(scalar)*          |
| `flat_direct_d3`                         | 0.04 (23.4)      | 0.14 (7.1)        | **0.03 (30.3)**      | *(scalar)*          |
| `flat_direct_d4`                         | **0.02 (51.7)**  | 0.06 (17.5)       | 0.03 (31.7)          | 0.04 (28.1)         |
| `flat_direct_d5`                         | **0.04 (25.4)**  | 0.78 (1.3)        | 0.04 (27.3)          | *(scalar)*          |
| `flat_direct_d6`                         | **0.04 (22.8)**  | 0.84 (1.2)        | 0.05 (20.6)          | *(scalar)*          |
| **Flat-subtree scatter** (indexed out)   |                  |                   |                      |                     |
| `flat_scatter_d2`                        | **0.14 (7.1)**   | 0.66 (1.5)        | 0.26 (3.8)           | *(scalar)*          |
| `flat_scatter_d3`                        | **0.16 (6.1)**   | 0.67 (1.5)        | 0.27 (3.8)           | *(scalar)*          |
| `flat_scatter_d4`                        | **0.14 (7.1)**   | 0.66 (1.5)        | 0.27 (3.7)           | 0.64 (1.6)          |
| `flat_scatter_d5`                        | **0.17 (5.8)**   | 1.41 (0.7)        | 0.28 (3.6)           | *(scalar)*          |
| `flat_scatter_d6`                        | **0.18 (5.5)**   | 1.50 (0.7)        | 0.32 (3.1)           | *(scalar)*          |

TBL primitive used per platform / D:

| D | NEON          | AVX-512        | SSE4.1                              |
|---|---------------|----------------|-------------------------------------|
| 2 | `vqtbl1q_u8`  | `pshufb`       | scalar (no per-byte var-shift)      |
| 3 | `vqtbl1`      | `pshufb`       | scalar                              |
| 4 | `vqtbl1q_u8`  | `pshufb`       | `pshufb` (only D with SIMD unpack)  |
| 5 | `vqtbl2q_u8`  | `vpermb` (ymm) | scalar                              |
| 6 | `vqtbl4q_u8`  | `vpermb` (zmm) | scalar                              |

Reading the table:

- **The indexed scatter floor varies hugely across platforms.**  M4
  hits ~0.14–0.18 ns/elem (5–7 GB/s) and the SIMD unpack upstream of
  it is essentially free.  Xeon AVX-512 sits at ~0.26–0.32 (3–4 GB/s,
  2× M4) — `_mm_extract_epi8` is a 1-cycle uop but emits ~16 of them
  per 16-element chunk.  Graviton 4 (0.66 / 1.5 GB/s) and Zen 3
  (0.66 even on its D=4 SIMD path) are 4–5× M4 — per-element
  scalar-store throughput is the bottleneck, not the TBL.

- **The flat-subtree direct path is fastest on M4** at 0.02 ns/elem
  (~52 GB/s) for D=2 / D=4 — single `vqtbl1q_u8` per 16 codes.  Xeon
  is within a couple-percent at 0.03 (~30 GB/s) for D ≤ 4 (single
  `pshufb`); D=6 `vpermb-zmm` lands at 0.05 (~21 GB/s) — broadly the
  same ballpark.  Surprisingly, **Xeon `both_leaves_vst1` is
  67.7 GB/s (0.01 ns/elem) — the fastest single row in the table**.
  AVX-512's `mask_blend_epi8` over a 32-byte register is essentially
  free on Sapphire/Granite Rapids, beating M4's `vbslq_u8` blend
  by 2×.

- **Graviton 4's `vqtbl{2,4}q_u8` regression is real and visible at
  the primitive level.**  D=5 / D=6 are 0.78 / 0.84 ns/elem
  (1.2 GB/s) — **20× the M4 cost**, even at the same NEON ISA.
  Empirical motivation for the production `PIVCO_NEON_FAST_MULTI_TBL=0`
  gate (see [`../IDEAS.md`](../IDEAS.md) "Graviton 4 NEON D=5/D=6
  regression").

- **Zen 3 has the slowest partition** at 0.20 ns/elem (5.1 GB/s) —
  3× M4's NEON partition (0.06 / 15.6) and 5× Xeon's `vpcompressw`
  (0.04 / 24.2).  Combined with the 0.64–0.66-ns indexed scatter,
  Zen 3 has the highest absolute floor on the 2-way decoder hot path
  of any tested platform — primitive-level evidence behind the
  IDEAS.md "Zen 3 hybrid block decoder" recommendation.

- **The half-tree partition saves materially on every platform.**
  `partition_half` / `partition_root_half` (one-side store, used when
  one child is a leaf) drops cost by 30–40% vs full partition:
  M4 21.9 vs 15.6 GB/s, Xeon 36.1 vs 24.2, Zen 3 7.5 vs 5.1,
  Graviton 4 9.2 vs 6.3 — validating the production "leaf-child
  fusion" optimisation uniformly.

- **The flat-subtree fast path has a real edge over partition-and-
  scatter when the output is sequential** (root-flat or covered
  subtree).  M4 0.02 vs partition's 0.06 — 3× cheaper.  Once stores
  are indexed (non-root flat subtree), the gap collapses to the
  per-platform scatter floor and most of the SIMD unpack savings are
  absorbed — visible in the `flat_scatter_dN` rows being tightly
  bunched within each platform regardless of D.

These primitives explain the per-distribution numbers in the
per-platform decode tables in
[`BENCHMARKS.md`](BENCHMARKS.md): flat-heavy distributions
(`uniform`, `flat_M*`, `sparse_*`) cash in the cheap direct path;
deep-tree distributions (`prose_pride`, `html_wiki`) pay the
partition cost per level repeatedly.

For step-by-step register-level traces of these primitives (NEON
`partition_8`, `tree_merge`, `flat_dN_unpack`) with worked examples,
see [`KERNELS.md`](KERNELS.md).
