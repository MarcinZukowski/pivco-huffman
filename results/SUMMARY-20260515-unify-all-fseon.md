# Full --all sweep, post unify-framework — 2026-05-15

End-of-refactor full sweep across all 30 benched distributions
(19 synthetic + 11 real-world) on the four headline test hosts.
Same code as `SUMMARY-20260514-unify-framework.md` plus three small
fixes from the cross-host validation: BOTH_LEAVES root fast path
(`8be22e7`), NEON D=5/D=6 BU gate re-enabled (`f281581`), bench
harness routes to codec_avx512 on AVX-512 hosts (`f281581`).

Raw outputs in this directory:
  - `sweep_<host>-20260515-unify-all.txt`     (decode, --all)
  - `enc_sweep_<host>-20260515-unify-all.txt` (encode, --all)

All numbers in **M symbols / second**.  Bench config: 100 reps × 4M-
symbol stream = 400M sym/run, 5 runs, drop 2 slowest.  Blocks of
8192 symbols on aarch64/AVX-512, 4096 on SSE/AVX2-only x86.

## Win counts (PIVCO `pivco_bu` vs best of huf0_x2 / huf0_x1 / trad_4s)

|     host                |   wins | range          |
|---                      |   ---: | ---            |
| M4 (NEON)               |  30/30 | 1.03x – 11.06x |
| c8i (Xeon AVX-512)      |  30/30 | 1.51x – 13.69x |
| c8g (Graviton 4 NEON)   |  30/30 | 1.30x –  8.57x |
| c6a (Zen 3 SSE/AVX2)    |  27/30 | 0.94x – 23.11x |

Big movers vs the prior 2026-04-25 baseline (28 / 25 / 16 / 8 of 29):

- **c8g** went from 16/29 to 30/30 — the K_right wire format plus
  the BU-direct `PIVCO_NEON_FAST_MULTI_TBL` lift (re-enabled in
  `f281581`, this session) flipped the real-text and D=5/D=6
  flat-subtree clusters from 0.63-0.78x into 1.30-1.69x.
- **c6a** went from 8/29 to 27/30 — the K_right BU decode is a
  much bigger win on Zen 3 than the legacy TD path was; deep-tree
  real-text now wins 1.08-1.43x with three deepest holdouts
  (`html_wiki`, `json_api`, `log_apache`) at 0.94-0.95x.
- **c8i** went from 25/29 to 30/30 — the K_right + BU + AVX-512
  `vpexpandb` tree_merge combo turned the April-era 0.90-0.97x
  real-text losses into 2.95-3.78x wins.
- **M4** went from 28/29 to 30/30 — gain is smaller because M4 was
  already winning broadly, but `proba14` jumped 1.11x to 2.05x and
  the real-text cluster lifted from 1.08-1.33x to 1.88-2.55x.

## Headline takeaways

1. **Real-text decode is no longer the loss cluster.**  In April
   the deepest-tree real-world inputs (`html_wiki`, `prose_pride`,
   `json_api`, `chinese_text`) lost on every platform except M4.
   Today they win on every platform except Zen 3, and on Zen 3 they
   sit at 0.94-1.08x — within striking distance.  The K_right
   header was the structural change.

2. **FSE is the only "loss" surface.**  Two distributions ratio
   ≤ 1.7x on a single host: `proba80` (1.34-1.65x) and
   `two_sym_90/10` (1.03-2.76x).  Both are heavy-skew, and the
   ratio dip is the v0.3 FSE wire-format trade-off:
   ~25-90% smaller encoded bitmap at the cost of FSE-decompress
   overhead.  Toggle off via `--no-fse` to see the raw-bitmap
   numbers — `proba80` on M4 returns from 4.2 GB/s to 15.6 GB/s.

3. **`vpcompressw` matters less than it used to.**  Zen 3 (no
   AVX-512) wins 27/30 today vs 8/29 in April.  The K_right + BU +
   FSE combo amortises the partition cost over fewer / cheaper
   operations, so the structural advantage of `vpcompressw` on
   Xeon AVX-512 / Zen 4+ has narrowed.  Zen 3 still pays for
   `pshufb + compress_tab` on the partition hot path, but only
   the deepest-tree real-text distributions still fall short of
   parity.

## Validation

Per-platform `pivco_huffman_tests` (cross-decode + edge cases +
FSE dispatch): 0 failures on M4 / c6a / c8g / c8i (verified
2026-05-14 as part of the unify-framework cutover; no test changes
since).
