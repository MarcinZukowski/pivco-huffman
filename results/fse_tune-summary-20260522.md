# FSE wide-cursor shape sweep — picking one "tuned" shape (2026-05-22)

Goal: among multi-cursor (x interleaved `FSE_DState_t`) × unroll (y) FSE
decode shapes, pick **one** that is *decent but almost always faster than
stock `FSE_decompress`*, robust across microarchitectures and data.

Codec: hand-rolled x-cursor / y-unroll FSE over the stock FSE primitives
(`FSE_initDState` / `FSE_decodeSymbolFast` / `BIT_reloadDStream`), tableLog
12, 256-symbol byte alphabet, table built from the data.  Stock = the
reference single-state `FSE_decompress`.  Source: `extras/fse_xy_codec.h`,
driver `bench/bench_fse_tune.c`.  Methodology: best of 5 runs × 10 passes
over a 983040-byte (≈0.94 MB) buffer (divisible by every x in
{2,4,6,8,10,12,16}).

## Byte-data decode MB/s (english + prose average)

| host | uarch                    | stock | x4y4 | **x8y1** | x8y2 | x10y4* | x16y2 | x16y4 |
|------|--------------------------|------:|-----:|------:|-----:|------:|------:|------:|
| m4   | Apple M4 (Avalanche)     |  870  | 1490 | 2106  | 1946 | 2233  | 2322  | 2028  |
| c8i  | Xeon 6 (Granite Rapids)  |  530  |  954 | 1396  | 1424 | 1582  | 1548  | 1572  |
| c8a  | EPYC (Zen 5 / Turin)     | 1044  | 1656 | 1934  | 1960 | 1939  | 1758  | 1765  |
| c8g  | Graviton 4 (Neoverse V2) |  579  |  942 | 1268  | 1224 | 1068  | 1060  | 1030  |
| c6a  | EPYC (Zen 3 / Milan)     |  692  | 1026 | 1078  |  841 |  804  |  699  |  713  |

\* x10y4 / x12y1 don't divide a 128 KB block, so they aren't deployable
in a 128 KB-chunked container; deployable shapes are x∈{2,4,8,16}.

## Choice: **x8y1**

- Beats stock on **every host**, 1.56×–2.65× (floor: c6a, 1.56×).
- **Outright best on the two weakest hosts** (c8g, c6a) — where peak
  shapes collapse: x16y2 drops to ~1.0× stock on c6a (Zen 3), x8y2 to 1.2×.
- On wide OOO cores (M4, c8i) it's not the peak (x16y2 / x10y4 are) but
  still 2.4–2.6× stock.
- x=8 divides a 128 KB block; y=1 = no unroll → smallest decoder.

x16y2 was the M4 peak but fails "almost always better" (≈1.0× on c6a).

## Cross-check on bitmap-byte data (per-node partition bitmaps)

DECODE at pmaj=0.80, 8160 B (from the 2026-05-15 `fse_xy` sweep):

| host | x8y1 | per-host best | x8y1 % of best |
|------|-----:|--------------:|---------------:|
| m4   | 2199 | x10y4 = 2401  | 92% |
| c8i  | 1394 | x10y4 = 1586  | 88% |
| c8a  | 1940 | x8y4  = 1999  | 97% |
| c8g  | 1270 | **x8y1 = 1270** | 100% (best) |
| c6a  | 1089 | x4y1  = 1182  | 92% |

x8y1 is 88–100% of the per-host peak and best on c8g — so the same shape
works well for both byte streams and ph's per-node bitmaps.

## Caveat: `FSE_decodeSymbolFast` is unsafe when P(max symbol) > 50%

A single byte at 80% (the `proba80` distribution → 6-symbol alphabet)
makes the dominant symbol exceed half the table; `FSE_decodeSymbolFast`
then diverges (invalid state → OOB read).  All wide-cursor shapes share
this limit, so such extreme single-symbol skew falls back to stock FSE
(or, in ph, to its flat/RLE path).  The bitmap study never tripped this
because packed-bitmap *bytes* stay < 50% even at pmaj 0.9 (0.9^8 ≈ 0.43).

Raw per-host captures: `results/fse_tune-{m4,c8i,c8a,c8g,c6a}-20260522-4cf46d0wip.txt`.
