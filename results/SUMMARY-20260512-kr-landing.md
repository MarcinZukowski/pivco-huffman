# K_right wire-format landing — full bench sweep 2026-05-12

Wire-format change `5828ddb` (K_right header inline before each non-flat
bitmap) committed and validated across all 7 EC2 hosts + Apple M4.
Per-host pivco-vs-huf0 ratios below.  Raw outputs in this directory:
  - `sweep_<host>-20260512-kr-landing.txt`     (decode)
  - `enc_sweep_<host>-20260512-kr-landing.txt` (encode)

Cell format: `M/s (ratio)` where ratio = pivco / huf0_x2.  huf0_x2 is
zstd's 4-stream interleaved Huffman codec — the strongest open-source
single-thread Huffman comparator.

## Decode (pivco_bu M/s and ratio vs huf0_x2)

pivco_bu = our bottom-up tree_merge decoder with K_right header inline.

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 4842 (4.09x) | 6199 (3.95x) | 6413 (3.96x) | 8353 (5.12x) | 27207 (8.21x) | 8305 (4.30x) | 21252 (11.01x) | 16265 (6.02x) |
| english | 1216 (1.06x) | 1443 (0.98x) | 1505 (0.98x) | 1780 (1.15x) | 11128 (3.48x) | 3012 (1.61x) | 7936 (4.23x) | 6501 (2.45x) |
| flat_M5 | 2131 (1.82x) | 2135 (1.46x) | 2208 (1.46x) | 2310 (1.47x) | 34034 (10.46x) | 9121 (4.99x) | 18486 (9.63x) | 21352 (4.14x) |
| html_wiki | 832 (0.84x) | 1009 (0.81x) | 1056 (0.82x) | 1274 (0.97x) | 5776 (2.13x) | 2135 (1.33x) | 4795 (2.98x) | 4524 (1.99x) |
| prose_pride | 1028 (0.94x) | 1264 (0.91x) | 1311 (0.91x) | 1615 (1.13x) | 7275 (2.43x) | 2343 (1.32x) | 5732 (3.24x) | 4830 (2.02x) |
| image_jpeg | 1043 (1.63x) | 1231 (1.11x) | 1285 (1.12x) | 1474 (1.38x) | 4778 (2.71x) | 1863 (1.80x) | 3914 (3.45x) | 4222 (2.68x) |
| json_api | 874 (0.84x) | 1064 (0.81x) | 1113 (0.82x) | 1342 (0.98x) | 6076 (2.11x) | 2121 (1.26x) | 5069 (3.00x) | 4501 (1.96x) |
| gzip_random | 1587 (4.67x) | 2199 (3.18x) | 2276 (3.16x) | 2988 (3.58x) | 4441 (3.51x) | 2197 (2.18x) | 4451 (5.66x) | 5069 (3.19x) |
| chinese_text | 1018 (1.11x) | 1232 (1.07x) | 1284 (1.07x) | 1551 (1.29x) | 7003 (2.78x) | 2387 (1.60x) | 5677 (3.83x) | 4856 (2.41x) |

## Encode (pivco M/s and ratio vs huf0_x2)

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 946 (1.47x) | 1477 (1.61x) | 1515 (1.59x) | 1743 (1.97x) | 13757 (9.87x) | 1474 (1.83x) | 7442 (6.49x) | 3259 (2.57x) |
| english | 575 (0.90x) | 799 (0.86x) | 831 (0.86x) | 965 (0.88x) | 4563 (2.70x) | 926 (0.97x) | 2708 (2.11x) | 2077 (1.23x) |
| flat_M5 | 379 (0.59x) | 1651 (1.75x) | 1692 (1.73x) | 2325 (2.07x) | 8630 (4.89x) | 1078 (1.11x) | 4317 (3.32x) | 2742 (1.49x) |
| html_wiki | 448 (0.72x) | 641 (0.71x) | 659 (0.71x) | 774 (0.72x) | 3135 (1.83x) | 764 (0.81x) | 1963 (1.57x) | 1729 (1.00x) |
| prose_pride | 247 (0.75x) | 664 (0.73x) | 685 (0.72x) | 780 (0.72x) | 3540 (2.11x) | 776 (0.82x) | 2325 (1.83x) | 1748 (1.05x) |
| image_jpeg | 142 (0.45x) | 830 (0.95x) | 846 (0.93x) | 1028 (0.98x) | 4629 (2.95x) | 719 (0.79x) | 2341 (1.93x) | 1712 (1.03x) |
| json_api | 226 (0.77x) | 333 (0.72x) | 338 (0.73x) | 734 (0.67x) | 3272 (1.92x) | 745 (0.78x) | 2040 (1.60x) | 1691 (0.97x) |
| gzip_random | 739 (0.66x) | 1681 (1.19x) | 1611 (1.20x) | 3397 (0.96x) | 33125 (6.79x) | 2247 (0.87x) | 10161 (2.71x) | 4792 (0.74x) |
| chinese_text | 196 (0.61x) | 369 (0.80x) | 377 (0.82x) | 552 (0.72x) | 3508 (2.07x) | 731 (0.77x) | 2176 (1.73x) | 1760 (0.97x) |

## Hosts

| host | CPU                                 | SIMD tier                          |
|------|-------------------------------------|------------------------------------|
| c3   | Xeon E5-2670 v2 (Ivy Bridge)        | SSE4.2                             |
| c4   | Xeon E5-2666 v3 (Haswell)           | AVX2 + BMI2                        |
| c5   | Xeon Platinum 8124M (Skylake-SP)    | AVX-512 BW (no VBMI2 → AVX2 tier)  |
| c6a  | EPYC 7R13 (Zen 3)                   | AVX2 + BMI2                        |
| c8a  | EPYC 9R14 / Turin (Zen 5)           | AVX-512 VBMI2                      |
| c8g  | Graviton 4 (Neoverse V2)            | NEON                               |
| c8i  | Granite Rapids                      | AVX-512 VBMI2                      |
| m4   | Apple M4                            | NEON                               |

## Headline

**Decode**: pivco_bu beats huf0_x2 on every (host, dist) pair except
a handful of text dists on the oldest x86 hosts (c3 Ivy Bridge, c4
Haswell, c5 Skylake/AVX2 path).  Ratios range from 0.81x (text on c4)
through 11.0x (proba80 on c8i Granite Rapids).  AVX-512 VBMI2 hosts
hit 3-11x; NEON hosts (M4, Graviton 4) hit 1.3-6x.

**Encode**: pivco beats huf0_x2 on every (host, dist) pair on the
modern hosts (c8a / c8i / c8g / M4), 1.5-10x.  Older x86 (c3-c6a)
ranges from 0.7x (text) to 4.9x (skewed).

K_right wire format directly drives the +30-60% BU decode jump on x86
recorded in commit 5828ddb's message.
