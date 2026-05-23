# pha stock vs pha+ (x8y1 wide-cursor FSE) — M4 (2026-05-23)

Fair-bench, 1 MB buffer, best-of-5×10, ph table-G=128 KB, BLK=8192.
Engine `pha` only.  Stock = `PIVCO_FSE_WIDE=0` (FSE reference 2-state
decode of the per-node bitmaps); pha+ = default (x8y1 8-state wide
decode).  Numbers are decode prebuilt-table MB/s, best of two full runs
(single-run noise was ±3-5%; best-of-2 removes the scatter).

Raw: `pha_wide_fse_stock-m4-20260523.txt`, `pha_wide_fse_opt-m4-20260523.txt`.

## Decode MB/s (prebuilt table)

| distribution   | stock | pha+  | speedup |
|----------------|------:|------:|--------:|
| **proba80**        |  4143 |  6394 | **1.54×** |
| **two_sym_90/10**  |  5456 |  9939 | **1.82×** |
| **calgary_pic**    |  4198 |  7104 | **1.69×** |
| sparse_16          | 48999 | 53499 | 1.09× |
| sparse_4           | 48771 | 52168 | 1.07× |
| proba14            |  5504 |  5490 | 1.00× |
| english            |  6562 |  6570 | 1.00× |
| zipfian            |  4753 |  4753 | 1.00× |
| ... (all others)   |   —   |   —   | ~1.00× |
| flat_M6            | 21845 | 20972 | 0.96× (noise; flat path, no FSE) |
| **GEOMEAN (30 dists)** | | | **1.05×** |

## Reading it

- The win is **concentrated on heavily-skewed, FSE-bitmap-dominated
  distributions** — proba80, two_sym_90/10, calgary_pic — where the
  per-node FSE bitmap decode is the decode bottleneck.  There it's
  **1.5–1.8×**, matching the standalone FSE-slice profile numbers.
- Everywhere else it's **neutral**: either FSE-on-bitmaps barely fires,
  the table isn't fast-mode-safe (falls back to stock), or the
  tree-merge / flat path dominates the decode so there's nothing for the
  wide FSE to accelerate.  The flat_* cases don't use FSE at all.
- Net **+5% geomean**, but that headline undersells the targeted gain:
  pha+ is a clear win exactly on the distributions where pha leans on the
  FSE bitmap path, and free (neutral) everywhere else.

## Correctness

Default-on; full test suite passes (two_sym all ratios, json/csv/calgary
files, skew80, adversarial, zipf).  Two bugs fixed while enabling:
encode_x flush-after-5 → 4 (64-bit container overflow on high-nbBits
runs), and the fast-mode gate `<=` → `>=` (matches FSE_buildDTable;
avoids BIT_readBitsFast zero-bit UB).
