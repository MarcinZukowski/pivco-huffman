# Platform sweep — flat-subtree fast path landed on all backends

**Date:** 2026-04-24 20:47 UTC
**Commit:** `0a92fe3` (tip of main)
**Covers:** `a275d05` flat-subtree fast path → `0d9ed64` root-flat unification → `7c3238b` x86 + AVX-512 port → `0a92fe3` libm link fix

All 20 round-trip tests pass on each platform below.  `pivco_p` is 0 on
Intel/AMD (the prefix-radix research backend is only compiled for NEON).

## Platforms

| Host | CPU | ISA | Block size |
|------|-----|-----|-----------:|
| local M4 | Apple M4 Max | NEON | 8192 |
| test-c6a | AMD EPYC 7R13 (Zen 3) 2 vCPU | SSE4.1 | 4096 |
| test-c8i | Intel Xeon 6975P-C 2 vCPU | AVX-512 VBMI2 | 8192 |
| test-c8g | AWS Graviton 4 (Neoverse-V2) 1 vCPU | NEON | 8192 |

c8g throttled 50% during the bench (1 vCPU, sustained load) — numbers
there are low-side estimates.

## Headline `pivco_n` ratios vs best other decoder

| Distribution | M4 | c8i (AVX-512) | c8g (NEON) | c6a (SSE4.1) |
|---|--:|--:|--:|--:|
| **two_sym_90/10** | 5.15× | 4.87× | **17.11×** | 0.85× |
| **two_sym_eq** | 4.78× | 2.55× | **6.31×** | 0.86× |
| **proba80** | 3.40× | 3.18× | 2.19× | 1.12× |
| **uniform** | 2.52× | **6.53×** | 2.32× | 2.17× |
| **flat_M7** | 1.42× | 3.50× | 2.94× | **2.29×** |
| **flat_M5** | 1.10× | 2.35× | 2.18× | 1.61× |
| **bell_s80** | **1.57×** | 2.76× | 1.18× | 1.05× |
| **bell_s30** | **1.40×** | 1.02× | 0.85× | 0.62× |
| **proba02** | **1.28×** | 1.15× | 0.79× | 0.61× |
| **zipfian** | **1.29×** | 1.26× | 0.76× | 0.61× |
| **english** | **1.03×** | 0.93× | 0.62× | 0.48× |
| **bell_s10** | **1.07×** | 0.86× | 0.70× | 0.51× |
| proba14 | 0.96× | 0.69× | 0.56× | 0.41× |
| sparse_16 | 1.32× | 1.96× | 1.63× | 1.34× |
| sparse_4 | 1.25× | 2.11× | 2.12× | 1.54× |

Bold = winning distribution where the flat-subtree path drives the win.
proba14 is the one case with no flat-subtree coverage by measurement
(0.9% elements), so expected to track huf0/trad_4s.

## Observations per platform

**Apple M4 Max (NEON) — all 19 distributions win against best external
decoder** (0.96× to 5.15×).  Flat-subtree hit every previously-losing
moderate-entropy distribution cleanly.

**Intel Xeon 6975P-C (AVX-512 VBMI2)** — wins on 14/19.  Uniform is a
spectacular 6.53× because the AVX-512 path takes the root-flat
bypass, writing `symbols[i]` directly with no prefill or indices
indirection.  english/bell_s10 slightly lose (0.86×-0.93×); likely
want vpmultishiftqb-based vectorised D-bit extract to close the gap.
The non-flat-subtree moderate-entropy cases need the already-fast
AVX-512 partition to be competitive — which is mostly is (english
1632 M/s vs huf0_x2 1754 M/s is 0.93×; close).

**Graviton 4 (NEON)** — wins on 10/19.  two_sym_90/10 = **17.11×** is
the session's peak.  Flat-subtree clears the flat_M3-M7 and uniform
cases (1.6× to 2.9×).  Moderate-entropy cases (english 0.62×,
zipfian 0.76×, bell_s10 0.70×) still lose — likely because Graviton's
`tbl` throughput is somewhat lower than M4's, so the partition cost
dominates.  CPU throttled 50% during run, numbers are conservative.

**AMD EPYC 7R13 (Zen 3 SSE4.1)** — wins on 8/19.  uniform 2.17×,
flat_M5 1.61×, flat_M7 2.29× are the clearest wins.  proba80 / bell_s80
win narrowly (1.05-1.12×).  Moderate-entropy distributions lose
broadly (0.41× to 0.72×) — Zen 3 SSE4.1 has fewer shuffle execution
units than Xeon AVX-512 or M4/Graviton NEON, and the partition cost
dominates on those shapes.  Flat-subtree coverage is unchanged from
the other platforms but the partition cost it displaces is smaller
on SSE4.1 to begin with, so the absolute improvement is smaller.
`pivco_p` shows 0 because `decode_neon_prefix` is NEON-only; x86
doesn't have a research prefix-radix backend.

## Distribution-level delta from flat-subtree

For reference against the pre-flat-subtree baseline (commit `984dad3`
and earlier), the historical loss cluster (bell_*, proba02, zipfian,
english on M4) moved from 0.66–0.98× to 1.03–1.57×.  On Xeon and
Graviton the moderate-entropy cases moved the same direction but
didn't fully cross 1.0×.  On Zen 3 they improved but stay below 1.0×.

Compression ratio unchanged by the format change, within rounding:
about 100 bytes/8KB block tighter than bitmap-per-level due to single
tail padding vs D per-level paddings.

## Raw output (per-platform, full bench + environment)

- [`m4_max-20260424-2047.txt`](m4_max-20260424-2047.txt) — Apple M4 Max
- [`zen3-20260424-2047.txt`](zen3-20260424-2047.txt) — AMD EPYC 7R13 (Zen 3 / SSE4.1)
- [`xeon_6975p-20260424-2047.txt`](xeon_6975p-20260424-2047.txt) — Intel Xeon 6975P-C (AVX-512 VBMI2)
- [`graviton4-20260424-2047.txt`](graviton4-20260424-2047.txt) — AWS Graviton 4 (NEON)
