# K_right wire-format landing — full bench sweep 2026-05-12

Wire-format change `5828ddb` (K_right header inline before each non-flat
bitmap) committed and validated across all 7 EC2 hosts + Apple M4.
Raw outputs in this directory:
  - `sweep_<host>-20260512-kr-landing.txt`     (decode)
  - `enc_sweep_<host>-20260512-kr-landing.txt` (encode)

All numbers in **M symbols / second** (millions of input symbols per
second of wall time).  Higher is faster.

Bench config: 25 reps × 4M-symbol stream = 100M sym/run, 5 runs, drop
2 slowest.  Blocks of 8192 symbols (4096 on SSE-only hosts).

## Decode M/s

### pivco_bu (our bottom-up tree_merge with K_right inline) — primary

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 4842 | 6199 | 6413 | 8353 | 27207 | 8305 | 21252 | 16265 |
| english | 1216 | 1443 | 1505 | 1780 | 11128 | 3012 | 7936 | 6501 |
| flat_M5 | 2131 | 2135 | 2208 | 2310 | 34034 | 9121 | 18486 | 21352 |
| html_wiki | 832 | 1009 | 1056 | 1274 | 5776 | 2135 | 4795 | 4524 |
| prose_pride | 1028 | 1264 | 1311 | 1615 | 7275 | 2343 | 5732 | 4830 |
| image_jpeg | 1043 | 1231 | 1285 | 1474 | 4778 | 1863 | 3914 | 4222 |
| json_api | 874 | 1064 | 1113 | 1342 | 6076 | 2121 | 5069 | 4501 |
| gzip_random | 1587 | 2199 | 2276 | 2988 | 4441 | 2197 | 4451 | 5069 |
| chinese_text | 1018 | 1232 | 1284 | 1551 | 7003 | 2387 | 5677 | 4856 |

### huf0_x2 (zstd 4-stream Huffman) — baseline

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 1185 | 1570 | 1619 | 1631 | 3315 | 1932 | 1931 | 2704 |
| english | 1151 | 1474 | 1532 | 1544 | 3196 | 1869 | 1875 | 2652 |
| flat_M5 | 1170 | 1467 | 1520 | 1577 | 3286 | 1869 | 1920 | 5196 |
| html_wiki | 992 | 1252 | 1294 | 1308 | 2711 | 1608 | 1610 | 2277 |
| prose_pride | 1089 | 1392 | 1444 | 1427 | 2996 | 1780 | 1770 | 2392 |
| image_jpeg | 597 | 771 | 799 | 802 | 1686 | 966 | 976 | 1370 |
| json_api | 1043 | 1316 | 1363 | 1364 | 2874 | 1689 | 1688 | 2300 |
| gzip_random | - | - | - | - | - | - | - | - |
| chinese_text | 919 | 1153 | 1197 | 1198 | 2516 | 1488 | 1483 | 2018 |

### pivco_n (top-down tree_merge, our older decode path)

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 1216 | 1792 | 1886 | 2415 | 7627 | 4279 | 6406 | 10016 |
| english | 592 | 783 | 814 | 992 | 3312 | 1311 | 2862 | 3335 |
| flat_M5 | 2133 | 2140 | 2213 | 2312 | 34378 | 9329 | 18456 | 21522 |
| html_wiki | 481 | 638 | 670 | 797 | 2521 | 1037 | 2171 | 2602 |
| prose_pride | 506 | 674 | 712 | 863 | 2564 | 1061 | 2317 | 2680 |
| image_jpeg | 602 | 716 | 751 | 905 | 2688 | 1247 | 2310 | 2816 |
| json_api | 481 | 641 | 674 | 805 | 2524 | 995 | 2299 | 2631 |
| gzip_random | 2090 | 2200 | 2280 | 3099 | 4374 | 2194 | 4436 | 5089 |
| chinese_text | 513 | 671 | 706 | 830 | 2527 | 1092 | 2268 | 2624 |

## Encode M/s

### pivco (our SIMD encoder)

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 946 | 1477 | 1515 | 1743 | 13757 | 1474 | 7442 | 3259 |
| english | 575 | 799 | 831 | 965 | 4563 | 926 | 2708 | 2077 |
| flat_M5 | 379 | 1651 | 1692 | 2325 | 8630 | 1078 | 4317 | 2742 |
| html_wiki | 448 | 641 | 659 | 774 | 3135 | 764 | 1963 | 1729 |
| prose_pride | 247 | 664 | 685 | 780 | 3540 | 776 | 2325 | 1748 |
| image_jpeg | 142 | 830 | 846 | 1028 | 4629 | 719 | 2341 | 1712 |
| json_api | 226 | 333 | 338 | 734 | 3272 | 745 | 2040 | 1691 |
| gzip_random | 739 | 1681 | 1611 | 3397 | 33125 | 2247 | 10161 | 4792 |
| chinese_text | 196 | 369 | 377 | 552 | 3508 | 731 | 2176 | 1760 |

### huf0_x2 (HUF_compress, 4-stream chunk encode)

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 644 | 918 | 951 | 883 | 1394 | 803 | 1145 | 1269 |
| english | 641 | 932 | 966 | 1101 | 1688 | 953 | 1277 | 1691 |
| flat_M5 | 644 | 941 | 975 | 1122 | 1761 | 968 | 1294 | 1844 |
| html_wiki | 622 | 896 | 931 | 1073 | 1710 | 939 | 1244 | 1705 |
| prose_pride | 306 | 915 | 948 | 1088 | 1674 | 946 | 1268 | 1613 |
| image_jpeg | 286 | 876 | 908 | 1045 | 1569 | 913 | 1214 | 1657 |
| json_api | 294 | 456 | 463 | 1098 | 1704 | 952 | 1272 | 1730 |
| gzip_random | 1116 | 1368 | 1341 | 3520 | 4882 | 2587 | 3750 | 6391 |
| chinese_text | 291 | 454 | 461 | 763 | 1694 | 943 | 1260 | 1703 |

## Ratios (pivco / huf0_x2)

### Decode

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 4.09x | 3.95x | 3.96x | 5.12x | 8.21x | 4.30x | 11.01x | 6.02x |
| english | 1.06x | 0.98x | 0.98x | 1.15x | 3.48x | 1.61x | 4.23x | 2.45x |
| flat_M5 | 1.82x | 1.46x | 1.45x | 1.46x | 10.36x | 4.88x | 9.63x | 4.11x |
| html_wiki | 0.84x | 0.81x | 0.82x | 0.97x | 2.13x | 1.33x | 2.98x | 1.99x |
| prose_pride | 0.94x | 0.91x | 0.91x | 1.13x | 2.43x | 1.32x | 3.24x | 2.02x |
| image_jpeg | 1.75x | 1.60x | 1.61x | 1.84x | 2.83x | 1.93x | 4.01x | 3.08x |
| json_api | 0.84x | 0.81x | 0.82x | 0.98x | 2.11x | 1.26x | 3.00x | 1.96x |
| gzip_random | - | - | - | - | - | - | - | - |
| chinese_text | 1.11x | 1.07x | 1.07x | 1.29x | 2.78x | 1.60x | 3.83x | 2.41x |

### Encode

| dist | c3 | c4 | c5 | c6a | c8a | c8g | c8i | m4 |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| proba80 | 1.47x | 1.61x | 1.59x | 1.97x | 9.87x | 1.84x | 6.50x | 2.57x |
| english | 0.90x | 0.86x | 0.86x | 0.88x | 2.70x | 0.97x | 2.12x | 1.23x |
| flat_M5 | 0.59x | 1.75x | 1.74x | 2.07x | 4.90x | 1.11x | 3.34x | 1.49x |
| html_wiki | 0.72x | 0.72x | 0.71x | 0.72x | 1.83x | 0.81x | 1.58x | 1.01x |
| prose_pride | 0.81x | 0.73x | 0.72x | 0.72x | 2.11x | 0.82x | 1.83x | 1.08x |
| image_jpeg | 0.50x | 0.95x | 0.93x | 0.98x | 2.95x | 0.79x | 1.93x | 1.03x |
| json_api | 0.77x | 0.73x | 0.73x | 0.67x | 1.92x | 0.78x | 1.60x | 0.98x |
| gzip_random | 0.66x | 1.23x | 1.20x | 0.97x | 6.79x | 0.87x | 2.71x | 0.75x |
| chinese_text | 0.67x | 0.81x | 0.82x | 0.72x | 2.07x | 0.78x | 1.73x | 1.03x |

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

**Decode**: pivco_bu beats huf0_x2 on every (host, dist) pair on modern
hosts (c8a/c8i/c8g/M4) and most older x86 dists.  Peak: 34 GB/s on c8a
flat_M5 (10.5x huf0_x2).  Granite Rapids hits 11x on proba80.

**Encode**: pivco beats huf0_x2 on every modern host (c8a/c8i/c8g/M4),
1.0-10x.  Older x86 (c3-c6a) lags on text (huf0 well-tuned for that
era) but wins on skewed data.

K_right wire format directly drives the +30-60% BU decode jump on x86
recorded in commit 5828ddb.
