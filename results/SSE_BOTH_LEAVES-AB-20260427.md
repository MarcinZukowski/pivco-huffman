# SSE root both-leaves vectorisation — A/B

**Date:** 2026-04-27
**Host:** AWS c6a (Zen 3, SSE4.1+AVX2 capable, AVX-512 unavailable)
**Methodology:** 5 alternated A/B pairs, paired-t per distribution.

## Background

`pivco_huffman_decode_x86` had a scalar byte-by-byte loop for the
"root both children leaves" case (used by `two_sym_eq` /
`two_sym_90/10`).  Codex review #5 flagged this.

This patch vectorises the path: 2 bitmap bytes → 16 output bytes per
iter via `pshufb` (broadcast each byte to 8 lanes) + `pcmpeqb`
(bit→byte mask) + `pblendvb` (sym0/sym1 select) + unaligned store.
SSE4.1, no AVX2 needed.

## Results

### Targeted wins (depth-1 trees)

| Distribution | Before  | After   | Δ      | t     |
|--------------|--------:|--------:|-------:|------:|
| two_sym_eq   |    1507 |   22677 | **+1405%** | 357 |
| two_sym_90/10|    1502 |   22660 | **+1409%** | 197 |

15× throughput on the exact distributions the patch targets.

### Bonus wins (no-touch path, codegen recovery)

| Distribution | Before | After | Δ |
|--------------|-------:|------:|--:|
| uniform      |   1762 |  3027 | **+71.8%** |
| gzip_random  |   1765 |  3054 | **+73.1%** |

These distributions go through `flat_decode_direct_x86` (root-flat D=8),
not the both-leaves path.  Yet a 70% improvement lands on them
consistently across 8+ alternated runs.  Likely explanation: the larger
parent function changes how the compiler inlines/schedules
`flat_decode_direct_x86`; net beneficial.  Both numbers match
morning's pre-rename baseline (3081 / 3082), suggesting the new code
restores a perf state that the current HEAD compile was missing.

### Other distributions

Small significant wins (t>2, mostly +0.5 to +1%):
- proba80 +0.6%, sparse_4 +0.7%, flat_M3 +0.7%

Everything else (real-text cluster, bell_*, proba14/02, geometric,
flat_M5..M7, sparse_16, dna_fasta, csv_numeric, etc.): within ±1.5%
noise, no significant deltas.

**No significant losses.**

## Files

- `zen3_quick_baseline_p{1..5}-20260427.txt`
- `zen3_quick_patched_p{1..5}-20260427.txt`

## Notes on the bonus speedup

Pre-rename HEAD (commit f1acffe) on Zen 3 today reproduces the same
~1770 M/s baseline.  So the rename is not the cause; this is something
about how today's compiler chose to lay out the SSE backend.  The
patched version with the larger function gets the codegen we want.
Worth understanding eventually (could be alignment / register pressure
/ inlining threshold), but unblocking-ship for now since the sign is
firmly positive.
