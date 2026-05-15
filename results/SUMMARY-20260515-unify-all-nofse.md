# Full --all sweep, FSE off — 2026-05-15

End-of-refactor full sweep across all 30 benched distributions
(19 synthetic + 11 real-world) on the four headline test hosts, in
the **default-recommended `--no-fse` configuration**.  FSE coding
of partition bitmaps is one of the ratio/speed tuning parameters
ph exposes; this sweep pegs it off to characterise the maximum-
throughput end of the tradeoff curve.

For the with-FSE comparison data (~25% smaller encoded bitmaps on
heavy-skew distributions, ~3-4× decode-speed cost on the FSE-firing
nodes) see [`SUMMARY-20260515-unify-all-fseon.md`](SUMMARY-20260515-unify-all-fseon.md).

Raw outputs in this directory:
  - `sweep_<host>-20260515-unify-all-nofse.txt`     (decode)
  - `enc_sweep_<host>-20260515-unify-all-nofse.txt` (encode)

Bench config: 100 reps × 4M-symbol stream = 400M sym/run, 5 runs,
drop 2 slowest.  Blocks of 8192 symbols on aarch64/AVX-512, 4096
on SSE/AVX2-only x86.

## Win counts (`pivco_bu` vs best of `huf0_x2` / `huf0_x1` / `trad_4s`)

|     host                |   wins | range          |
|---                      |   ---: | ---            |
| M4 (NEON)               |  30/30 | 1.43x – 10.68x |
| c8i (Xeon AVX-512)      |  30/30 | 2.96x – 13.81x |
| c8g (Graviton 4 NEON)   |  30/30 | 1.29x –  8.59x |
| c6a (Zen 3 SSE/AVX2)    |  27/30 | 0.94x – 22.45x |

Vs. the `-fseon` companion sweep, the no-FSE configuration shifts
the headline floor up on every platform (M4: 1.03x → 1.43x;
c8i: 1.51x → 2.96x — that's the FSE decode tax gone) at the cost
of giving up the ratio win on heavy-skew bitmaps.

## What's interesting in this dataset

- **Same algorithm, four backends, 0.94×–22.45× ratio spread.**
  The dynamic range is dominated by uarch and primitive
  availability: Xeon (`vpcompressw` + `vpexpandb` + `vpermb`) at
  the high end, Zen 3 (`pshufb` + `compress_tab`) at the low end.
  The same C source path is compiled four times into separate
  OBJECT libraries; the only thing that differs is the per-backend
  primitives header.

- **The Zen 3 ≤parity rows (3/30, all 0.94-0.95×)** are all deep-
  real-text distributions: `html_wiki`, `json_api`, `log_apache`,
  all Dmax 15.  Same family of inputs that lose on every host
  without `vpcompressw` — they're the partition-cost-dominated
  regime, where huf0_x2's table-driven decode still has a slight
  edge.  Not a problem to fix (ph is research, not a tool); a
  property of the SIMD primitive landscape.

- **Graviton 4 `two_sym_*` 6.6-6.7×** vs M4 `two_sym_eq` 4.82× —
  the BOTH_LEAVES-at-root fast path hits harder on Graviton 4 than
  on M4 in absolute throughput.  Worth a microbench to understand
  the per-platform `merge_both_const` cost more directly.

- **Zen 3 `two_sym_*` 22.17-22.45×** is the highest single ratio
  in the dataset.  Partly the BOTH_LEAVES fast path, partly huf0_x2
  on Zen 3 being unusually slow for the 2-symbol case (1622 M/s vs
  3438 on M4) — so the ratio is amplified by both ends.

- **calgary_pic 4.76-5.26×** validates the proba80-shaped real-
  world inclusion that landed 2026-05-12.  Synthetic proba80 was
  already a strong row; calgary_pic gives a real-data point with
  comparable behaviour.

## Methodology

Sweep was run on each host via:
```
./build/pivco_huffman_bench 100 --all --no-fse > /tmp/sweep.txt
./build/pivco_huffman_bench_encode 100 --all --no-fse > /tmp/enc.txt
```

Each row is the median of 3 of 5 runs (drop 2 slowest), 100 reps ×
4M-symbol stream per run.  All hosts ran the same source tree at
`HEAD = f281581` (NEON D=5/D=6 BU gate fix + bench harness fix).

Test validation: `pivco_huffman_tests` passed on all four hosts
with the current code (verified 2026-05-14 as part of the unify-
framework cutover; no test changes since).
