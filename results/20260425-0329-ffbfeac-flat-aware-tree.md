# Flat-aware Huffman tree restructurer

**Date:** 2026-04-25 03:29 UTC
**Commit:** [ffbfeac2ae9f56bf9f435574cb21dedbdef13ae5](../) (tip of main)
**Predecessor sweep:** [`20260425-0126-cee2366-graviton-d56-fix.md`](20260425-0126-cee2366-graviton-d56-fix.md) (commit cee2366)
**Methodology:** 30 reps × 4M symbols, 5 runs (drop 2), `taskset -c 0` on Linux hosts.  All 20 round-trip tests pass on every platform.

## Background

`pivco_huffman_build_table` previously produced canonical Huffman trees: sort by `(length, value)`, assign sequential codes, walk the codes MSB-first to build the tree.  This is the standard approach but doesn't maximise the flat-subtree fast path's coverage — same-length leaves can end up scattered across distant subtrees of the canonical tree, splitting what could be a single flat-D≥2 root into multiple smaller pieces (or D=1 stage-fusion sibling pairs).

Investigation: [`extras/bench_flat_optimal.c`](../extras/bench_flat_optimal.c) showed that on the historically losing distributions, the *partition-step count* differs significantly between canonical and the flat-optimal layout — even when D≥1 leaf coverage is identical.  Predicted partition-op savings ranged 16–27% on `english` / `proba14` / `proba02` / `bell_s80`.

This commit replaces the canonical assignment with a constructive flat-aware layout (provably optimal for D≥2 leaf coverage; see IDEAS.md "Flat-aware Huffman tree restructurer").  Per length L, decompose `c_L` by binary representation: bits at position ≥2 form D≥2 flat chunks of size 2^D, bit 1 forms a D=1 sibling pair, bit 0 is a singleton.  Highest-freq length-L symbols go to the largest-D chunk per length (deepest partition-path savings).  Chunks across lengths are sorted by tree-depth and canonical-coded.  Same code-length multiset = identical compression.

## Headline wins (pivco_n M/s, before → after)

| Distribution | Apple M4 | Xeon AVX-512 | Graviton 4 | Zen 3 SSE4.1 |
|---|--:|--:|--:|--:|
| `english`  | 2908 → **3333** (+15%) | 1758 → **2171** (+23%) | 1177 → 1200 (+2%)  | 794 → **887** (+12%) |
| `proba14`  | 2510 → **2866** (+14%) | 1176 → **1876** (+60%) | 973 → **1133** (+16%) | 669 → **773** (+16%) |
| `proba02`  | 2304 → **2588** (+12%) | 1405 → **1564** (+11%) | 899 → **997** (+11%) | 626 → **710** (+13%) |
| `bell_s80` | 2890 → 2872 (-1%)        | 2041 → **2277** (+12%) | 1105 → **1310** (+19%) | 818 → **923** (+13%) |
| `bell_s10` | 3114 → 3204 (+3%)      | 1639 → **1941** (+18%) | 1189 → **1279** (+8%) | 830 → 871 (+5%)  |
| `bell_s30` | 2303 → 2396 (+4%)      | 1175 → **1396** (+19%) | 882 → 914 (+4%)   | 610 → 647 (+6%)  |

## Parity-cross flips (loss → win)

| | Distribution | Before | After |
|---|---|--:|--:|
| **M4 NEON** | `proba14` | 0.91× | **1.06×** |
| **Xeon AVX-512** | `proba14` | 0.67× | **1.07×** |
| **Graviton 4** | `proba02` | 0.92× | **1.02×** |

`proba14` was the standing loss case across all 4 platforms.  It now wins on M4 and Xeon, much closer to parity on Graviton (0.60× → 0.69×) and Zen 3 (0.40× → 0.47×).

## Mechanism

Flat-D≥2 roots eliminate the entire partition path through their 2^D-leaf subtree (D levels of would-be partitions absorbed by a single flat decode).  D=1 stage-fusion only removes the partition at the immediate parent.  When the canonical tree splits same-length leaves into multiple D=1 sibling pairs separated by partition-doing internal nodes, consolidating them into a single flat-D≥2 subtree saves all the partition steps along the path.

Predicted partition-count savings (from analyzer) vs measured throughput gain:

| Distribution | Predicted partition-op saving | Measured throughput Δ (Xeon) |
|---|--:|--:|
| `bell_s80` | −26.7% | +12% |
| `english`  | −25.3% | +23% |
| `proba02`  | −19.4% | +11% |
| `proba14`  | −16.0% | +60% |

Translation factor isn't constant: partitions aren't the only decode cost (flat-decode + scatter overhead also matter), and gains depend on partition cost relative to the rest.  Xeon and Graviton see larger throughput gains than M4 because their partition primitive (`vpcompressw` / NEON `tbl` on Neoverse-V2) costs more per element than M4's `tbl`.  `proba14` Xeon is dramatic because canonical tree had almost zero D≥2 coverage there (33% leaf, 0.9% freq), and the restructure drops it onto the now-fast AVX-512 D=2/3/4 paths.

## Cross-platform Win Counts (PIVCO SIMD vs best other decoder)

| Platform | cee2366 | ffbfeac | Δ |
|---|--:|--:|--:|
| Apple M4 (NEON) | 18/19 | 18/19 | – |
| Xeon 6975P-C (AVX-512 VBMI2) | 15/19 | **17/19** | +2 (`english` parity-cross, `proba14` parity-cross) |
| Graviton 4 (NEON, Neoverse-V2) | 13/19 | **14/19** | +1 (`proba02` parity-cross) |
| Zen 3 SSE4.1 | 8/19 | 8/19 | – |

Total PIVCO wins across the 4-platform × 19-distribution grid: **57/76 → 60/76** (4 distinct parity flips, two on Xeon).

## Files

- [`m4_max-20260425-0329.txt`](m4_max-20260425-0329.txt)
- [`graviton4-20260425-0329.txt`](graviton4-20260425-0329.txt)
- [`xeon_6975p-20260425-0329.txt`](xeon_6975p-20260425-0329.txt)
- [`zen3-20260425-0329.txt`](zen3-20260425-0329.txt)
