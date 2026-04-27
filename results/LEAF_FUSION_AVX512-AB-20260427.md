# AVX-512 leaf-fusion port — A/B verification

**Date:** 2026-04-27
**Host:** AWS c8i (Xeon 6975P-C, AVX-512 VBMI2)
**Methodology:** 7 alternated baseline-vs-patched A/B pairs, each pair = 12s
quick-mode bench (`PIVCO_BENCH_QUICK=1 ./build-release/pivco_huffman_bench 5`,
RUNS=2 drop=0).  Paired-t per distribution (host drift cancels).

## Background

`decode_node_neon` and `decode_node_x86` both have stage-fusion logic
for shallow internal nodes:

- both children leaves → `scatter_both_leaves` (no partition, just
  bitmap-driven select between the two leaf symbols)
- left child = prefilled leaf → half-partition right side only via
  `partition_8_right`
- right child = prefilled leaf → half-partition left side only via
  `partition_8_left`

`decode_node_avx512` was missing all three.  The AVX-512 helpers
(`partition_32_right`, `partition_32_left`) already existed but were
never called from the dispatcher.

This patch ports the NEON/SSE shape: adds `scatter_both_leaves_avx512`
(scalar STRBs since AVX-512 has no byte scatter; the SIMD blend is
implicit at compile time) and the three early-return branches before
the standard full-partition path.

## Results

Paired-t deltas across 7 A/B pairs.  `t > 2` ⇒ p < 0.05 (`!`),
`t > 1` ⇒ borderline (`?`).

```
distribution      mean Δ%    sd%    t      sig
─────────────────────────────────────────────
source_c          +7.5%      2.5    +8.0   !
proba02           +4.3%      1.4    +8.2   !
bell_s30          +3.2%      1.3    +6.3   !
english           +6.1%      2.6    +6.2   !
zipfian           +3.2%      1.5    +5.4   !
proba80           +4.2%      2.9    +3.9   !
html_wiki         +4.1%      2.9    +3.7   !
bell_s10          +3.5%      3.0    +3.2   !
log_apache        +2.2%      1.9    +3.0   !
prose_pride       +3.2%      4.9    +1.8   ?
gzip_random       -4.0%      5.6    -1.9   ?
─── below |t|=1, no significant change: ─────
proba50, proba14, bell_s80, uniform, sparse_4, sparse_16, geometric,
two_sym_eq, two_sym_90/10, flat_M3..M7, json_api (+13.9% but high sd),
image_jpeg, dna_fasta, csv_numeric, chinese_text
```

**9 distributions reach p<0.05 wins, 0 reach p<0.05 losses.** Real-text
cluster (english, source_c, html_wiki, log_apache, prose_pride) gets
a clean +3-7% boost; skewed distributions (proba80, proba02, bell_s10,
bell_s30, zipfian) +3-4%.

Distributions that don't trigger the leaf-fusion paths (sparse_*, all
flat_M*, two_sym_*, uniform — these go through `flat_decode_*_avx512`
or have tree shapes without depth-2 leaf-leaf nodes) are unchanged
within noise as expected.

## Pattern matches the hypothesis

The AVX-512 `vpcompressw` (32-element compress) is relatively expensive
on Xeon vs the NEON TBL-based compress on M4.  Skipping `vpcompressw`
on shallow internal nodes is therefore proportionally a bigger win on
this backend.  The pattern (real-text wins, synthetic flat unchanged)
matches the same change shipped on NEON earlier.

## Files

Two A/B pairs from a noisier run (12s quick + full 280s comparators):
- `xeon_quick_baseline-20260427-0921.txt`
- `xeon_quick_leaffusion-20260427-0921.txt`
- `xeon_quick_baseline2-20260427-0925.txt`
- `xeon_quick_leaffusion2-20260427-0925.txt`

Five fresh alternated pairs:
- `xeon_quick_baseline_p{1..5}-20260427.txt`
- `xeon_quick_leaffusion_p{1..5}-20260427.txt`

The two earlier full-bench runs (before quick mode existed) are also
preserved for the record:
- `xeon_6975p-leaffusion-20260427-0903.txt`
- `xeon_6975p-leaffusion-20260427-0914.txt`

## Methodology note: paired-t with `PIVCO_BENCH_QUICK`

Within one quick run the bench loops through datasets sequentially, so
all decoders for one dist are measured back-to-back (tight grouping).
Across two consecutive quick runs (binary A then binary B) there's a
~12s window during which host load can drift, which is the dominant
noise term on this c8i instance.  Paired-t per distribution treats
each (baseline_i, leaffusion_i) pair as one observation — the shared
host-drift factor cancels.  CV-per-cell on a single quick run is
10–25%, but pair-level deltas have CV of 1–5% on the wins.

Net: 5-minute total wall to refute the noisy first impression and
verify the change is real.  Same A/B with full-bench methodology would
have taken ~70 min on the same host.
