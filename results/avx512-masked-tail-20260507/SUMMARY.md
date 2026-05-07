# AVX-512 partition tail — masked vector vs scalar — 2026-05-07

A/B for `src/pivco_huffman_avx512.c` change: replaces three scalar
tail loops in `decode_node_avx512` with a single masked
`partition_32` / `partition_32_left` / `partition_32_right` call
covering 1..31 remaining elements.  Same idea in all three branches:
load remaining `bm` bytes into a `uint32_t`, mask off bits beyond
`rem = n - j`, run one vector partition.

Working-tree diff against HEAD `c1f6cd3`:
`pivco-huffman/src/pivco_huffman_avx512.c`, 33 +/- 15 across three hunks
(see `git diff` at session start).

## Methodology

- Hosts: `test-c8i` (Xeon AVX-512 VBMI2), `test-c8a` (AMD Zen 5 EPYC 9R45, AVX-512 VBMI2)
- Compiler: gcc-14 (auto-selected by CMake on both hosts)
- Mode: PIVCO_BENCH_QUICK=1, repeats=20, runs=2 (drop 0)
- 5 alternated rounds of {HEAD, PATCH} per host; per-distribution avg of pivco_n
- BLK = 8192 on both hosts
- All other code identical (uses the AVX2-WIP CMakeLists, but AVX-512 path is unchanged outside `pivco_huffman_avx512.c`)

## test-c8i (Xeon AVX-512) — PATCH wins

Real-text cluster +20% to +40%.  Partition path the dominant hot path
on every Huffman-realistic distribution.  Flat-subtree paths unchanged
(as expected — the change doesn't touch `flat_decode_*`).

| distribution  |  HEAD |  PATCH | delta   |
|---------------|------:|-------:|--------:|
| source_c      |  1719 |   2397 | +39.4% |
| html_wiki     |  1469 |   2025 | +37.8% |
| bell_s30      |  1439 |   1950 | +35.5% |
| csv_numeric   |  2122 |   2791 | +31.6% |
| chinese_text  |  1631 |   2131 | +30.7% |
| prose_pride   |  1623 |   2096 | +29.1% |
| log_apache    |  1656 |   2130 | +28.7% |
| json_api      |  1706 |   2190 | +28.3% |
| proba02       |  1692 |   2170 | +28.2% |
| proba14       |  1885 |   2349 | +24.6% |
| bell_s10      |  2045 |   2525 | +23.5% |
| proba50       |  2785 |   3414 | +22.6% |
| zipfian       |  1906 |   2314 | +21.4% |
| english       |  2381 |   2852 | +19.8% |
| geometric     |  2836 |   3372 | +18.9% |
| image_jpeg    |  2010 |   2245 | +11.7% |
| uniform       |  4286 |   4530 |  +5.7% |
| proba80       |  5541 |   5830 |  +5.2% |
| dna_fasta     |  2738 |   2873 |  +4.9% |
| gzip_random   |  4317 |   4502 |  +4.3% |
| flat_M3..M7   |     ~ |      ~ |  ~0%   |
| sparse_4/16   |     ~ |      ~ |  ~0%   |
| two_sym_*     |     ~ |      ~ |  ~0%   |

## test-c8a (Zen 5 AVX-512) — same shape, plus a regression

Real-text deltas mirror c8i closely (+17% to +40%).  Two anomalies:

- **two_sym_eq -5.9%** (6306 → 5936 M/s) — single-deep tree, mask = bitmap directly,
  every recursive call hits a tail.  Zen 5 is the only host where this regresses;
  c8i shows ~0%.  Hypothesis: at this kernel density (1 element / 32 elements
  per `partition_32` invocation in shallow trees), the masked partition's
  `vpcompress` dependency chain costs more than the previous scalar loop's
  3-instruction inner.  Still fast in absolute terms (5.9 GS/s post-patch).
- **sparse_4 -2.9%** — almost certainly noise (flat-only path, change shouldn't touch it).

| distribution  |  HEAD |  PATCH | delta   |
|---------------|------:|-------:|--------:|
| source_c      |  1974 |   2754 | +39.5% |
| html_wiki     |  1692 |   2319 | +37.1% |
| bell_s30      |  1643 |   2224 | +35.3% |
| csv_numeric   |  2536 |   3379 | +33.2% |
| prose_pride   |  1855 |   2384 | +28.5% |
| log_apache    |  1916 |   2425 | +26.6% |
| bell_s10      |  2439 |   3086 | +26.5% |
| proba02       |  1949 |   2461 | +26.3% |
| json_api      |  1949 |   2459 | +26.1% |
| chinese_text  |  1946 |   2433 | +25.0% |
| proba14       |  2184 |   2696 | +23.4% |
| proba50       |  3360 |   4114 | +22.4% |
| zipfian       |  2132 |   2534 | +18.9% |
| geometric     |  3484 |   4079 | +17.1% |
| english       |  2840 |   3313 | +16.6% |
| image_jpeg    |  2407 |   2690 | +11.8% |
| gzip_random   |  4113 |   4422 |  +7.5% |
| two_sym_eq    |  6306 |   5936 | **-5.9%** |
| sparse_4      | 48975 |  47576 |  -2.9% |
| uniform       |  4227 |   4417 |  +4.5% |
| flat_*        |     ~ |      ~ |  ~0%   |

## Verdict

Land it.  The two_sym_eq regression on Zen 5 is real but two_sym_eq is
already 5.9 GS/s post-patch and the win on every real-text dist is
20–40%.  Net win on every workload anyone actually decodes.

Raw per-round logs in `ab-c8i/` and `ab-c8a/`; full table in
`aggregated.txt`.
