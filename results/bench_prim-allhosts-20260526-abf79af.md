# bench_prim — all hosts (commit abf79af, 2026-05-26)

`bench_prim: n=8192 elems, best-of-9 x 2000 reps, partition depth=3`
SIMD-variant ns/elem (lower = better).  Hosts: M4 (local, NEON),
c6a (Zen3, SSE/AVX2), c8g (Graviton4, NEON), c8i (Xeon 6, AVX-512 VBMI2).

Captured right after landing the x86 AVX2 D2/D5/D6 flat-decode kernels.
`unpack` / `scatter` are exposed as standalone kernels only on the NEON
hosts; on x86/AVX-512 only the fused `merge` (+ pack, partition) is
SIMD-measurable.

## merge (flat decode = unpack + scatter)

| D | M4 NEON | c6a SSE/AVX2 | c8g NEON | c8i AVX-512 |
|---|---------|--------------|----------|-------------|
| 2 | 0.0203  | 0.0853       | 0.0458   | 0.0317      |
| 3 | 0.0457  | 0.0831       | 0.138    | 0.030       |
| 4 | 0.0179  | 0.0214       | 0.0491   | 0.0316      |
| 5 | 0.0388  | 0.146        | 0.191    | 0.0364      |
| 6 | 0.0433  | 0.226        | 0.242    | 0.0482      |
| 7 | 0.0723  | 0.376 *      | 0.329    | 0.0643      |
| 8 | 0.182   | 0.319 *      | 0.377    | 0.222       |

\* c6a D7/D8 run the scalar-unrolled inner — no x86 SIMD path (pshufb is
16-wide; the 128-entry D7 scatter is a wash, see abf79af commit msg).

## pack (encode)

| D | M4 NEON | c6a SSE/AVX2 | c8g NEON | c8i AVX-512 |
|---|---------|--------------|----------|-------------|
| 2 | 0.030   | 0.0706       | 0.0709   | 0.203       |
| 3 | 0.0674  | 0.225        | 0.209    | 0.211       |
| 4 | 0.024   | 0.0445       | 0.0621   | 0.203       |
| 5 | 0.0759  | 0.228        | 0.224    | 0.204       |
| 6 | 0.0722  | 0.222        | 0.224    | 0.204       |
| 7 | 0.0786  | 0.247        | 0.557    | 0.216       |
| 8 | 0.0159  | 0.0373       | 0.0426   | 0.0253      |

## partition

| M4 NEON | c6a SSE/AVX2 | c8g NEON | c8i AVX-512 |
|---------|--------------|----------|-------------|
| 0.0817  | 0.247        | 0.244    | 0.0554      |

## unpack / scatter (NEON hosts only)

| D | unpack M4 | unpack c8g | scatter M4 | scatter c8g |
|---|-----------|------------|------------|-------------|
| 2 | 0.015     | 0.0495     | 0.0168     | 0.0249      |
| 3 | 0.0424    | 0.143      | 0.0147     | 0.0235      |
| 4 | 0.0173    | 0.0487     | 0.0148     | 0.0237      |
| 5 | 0.0389    | 0.185      | 0.015      | 0.0251      |
| 6 | 0.0397    | 0.212      | 0.0151     | 0.0313      |
| 7 | 0.0427    | 0.236      | 0.0303     | 0.0601      |

## Observations

- **x86 D2/D5/D6 (this commit):** c6a merge D2 1.51→0.085, D5 1.70→0.146,
  D6 1.83→0.226 vs the prior scalar-inner.  Every practical depth (D2–D6)
  now has an x86 SIMD flat decode.
- **c8i (AVX-512) leads merge** for D2–D7 (all ≤0.077) — vpmultishift
  unpack + vpermb/vpermi2b scatter.
- **pack on c8i is flat ~0.20** for D2–D7, notably *worse* than NEON's
  ~0.07 and even c6a — an AVX-512 pack opportunity (uint-lane pack path
  not paying off there).
- **c8g (Graviton4) odd-D unpack/merge** (D3/5/6/7) lag M4 markedly —
  the known Neoverse V2 TBL/variable-shift weakness.
- **pack D7 on c8g = 0.557**, a clear outlier vs M4 0.0786 / c6a 0.247.
