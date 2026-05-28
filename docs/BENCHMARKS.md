# Benchmark Results

> **Last content review:** _NEVER_

**Methodology**: Decode a 4M-symbol sequence, repeated 100 times per
timed run (400M symbols/run). 5 runs, drop 2 slowest, report median
of 3 best.  Each codec uses its natural block size: PIVCO/trad use
4096–8192 symbol blocks (auto-detected per backend), huf0 uses
128 KB chunks, rANS decodes full 4M.

Baselines include huf0 X1 (single-symbol lookup) and X2 (double-symbol
lookup). "vs best" = best PIVCO / best of all other decoders.

Per-host raw sweep files in [`results/`](../results/) — most recent on
the current code at the time of this snapshot are the post-unify-
framework sweeps:
[`SUMMARY-20260515-unify-all-nofse.md`](../results/SUMMARY-20260515-unify-all-nofse.md)
(the default-recommended `--no-fse` configuration used in the tables
below) and
[`SUMMARY-20260515-unify-all-fseon.md`](../results/SUMMARY-20260515-unify-all-fseon.md)
(FSE-on for ratio/speed comparison).  See also
[`SUMMARY-20260514-unify-framework.md`](../results/SUMMARY-20260514-unify-framework.md)
for the MAIN-set sweep that validated the refactor.

## Apple M4 Max (NEON, 128 KB L1D, block 8192)

*(post unify-framework refactor, 2026-05-15; 100 reps × 4M symbols,
median of 3 of 5 runs.  Full sweep file:
[`results/sweep_m4-20260515-unify-all-nofse.txt`](../results/sweep_m4-20260515-unify-all-nofse.txt).
Real-world distributions (`html_wiki` … `calgary_pic`) source files in
[`extras/datasets/`](../extras/datasets/).)*

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |     15339 |    1360 |    2617 |    1605 | **5.91x** |
| proba50       |      9151 |    1344 |    2550 |    1466 | **3.62x** |
| proba14       |      5204 |    1338 |    2482 |    1461 | **2.10x** |
| proba02       |      4516 |    1315 |    1472 |    1464 | **3.08x** |
| bell_s10      |      6398 |    1314 |    2261 |    1457 | **2.83x** |
| bell_s30      |      4329 |    1326 |    1402 |    1472 | **2.94x** |
| bell_s80      |      4335 |       0 |       0 |    1567 | **2.77x** |
| uniform       |      5017 |       0 |       0 |    1577 | **3.18x** |
| english       |      6142 |    1314 |    2414 |    1508 | **2.55x** |
| zipfian       |      4159 |    1305 |    1763 |    1508 | **2.36x** |
| sparse_4      |     47619 |    3408 |    5034 |    1575 | **9.46x** |
| sparse_16     |     46162 |    3120 |    4435 |    1579 | **10.68x** |
| geometric     |      7360 |    1310 |    2500 |    1453 | **2.94x** |
| two_sym_eq    |     24863 |    3411 |    5168 |    1583 | **4.82x** |
| two_sym_90/10 |     24860 |    3397 |    4938 |    1580 | **5.03x** |
| flat_M3       |     21478 |    3402 |    5244 |    1577 | **4.10x** |
| flat_M5       |     24006 |    3407 |    5005 |    1572 | **4.80x** |
| flat_M6       |     19963 |    3338 |    4363 |    1563 | **4.58x** |
| flat_M7       |      4829 |    3420 |    2673 |    1562 | **1.43x** |
| html_wiki     |      4316 |    1304 |    2110 |    1453 | **2.05x** |
| prose_pride   |      4713 |    1299 |    2312 |    1458 | **2.04x** |
| image_jpeg    |      4092 |    1302 |    1278 |    1509 | **2.71x** |
| json_api      |      4267 |    1290 |    2199 |    1451 | **1.94x** |
| source_c      |      4664 |    1305 |    2158 |    1472 | **2.18x** |
| log_apache    |      4490 |    1305 |    2135 |    1457 | **2.10x** |
| dna_fasta     |      8323 |    1320 |    2558 |    1515 | **3.25x** |
| csv_numeric   |      6359 |    1305 |    2459 |    1461 | **2.59x** |
| gzip_random   |      4931 |       0 |       0 |    1559 | **3.19x** |
| chinese_text  |      4826 |    1311 |    1945 |    1452 | **2.50x** |
| calgary_pic   |     11274 |    1292 |    2369 |    1460 | **4.76x** |

(`--no-fse` numbers — FSE coding of partition bitmaps is a separate
ratio/speed knob, see [`DATA_FORMAT.md`](DATA_FORMAT.md).)

## Intel Xeon 6975P-C (AVX-512 VBMI2 + VBMI, 48 KB L1D, block 8192)

*(post unify-framework refactor, 2026-05-15; AWS `test-c8i`,
2 vCPU, GCC 11.5.0, Amazon Linux 2023; 100 reps × 4M symbols.
Full sweep file:
[`results/sweep_c8i-20260515-unify-all-nofse.txt`](../results/sweep_c8i-20260515-unify-all-nofse.txt).)*

| Distribution  | PIVCO AVX512 | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |     22581 |    1139 |    1930 |     798 | **11.70x** |
| proba50       |     11091 |    1143 |    1927 |     723 | **5.76x** |
| proba14       |      5849 |    1149 |    1862 |     722 | **3.14x** |
| proba02       |      4388 |    1131 |    1108 |     721 | **3.88x** |
| bell_s10      |      7240 |    1133 |    1722 |     721 | **4.20x** |
| bell_s30      |      4693 |    1134 |    1051 |     721 | **4.14x** |
| bell_s80      |      4202 |       0 |       0 |     775 | **5.44x** |
| uniform       |      4415 |       0 |       0 |     786 | **5.63x** |
| english       |      7909 |    1143 |    1875 |     757 | **4.22x** |
| zipfian       |      4593 |    1132 |    1340 |     757 | **3.43x** |
| sparse_4      |     24021 |    1138 |    1947 |     799 | **12.34x** |
| sparse_16     |     20008 |    1146 |    1928 |     798 | **10.39x** |
| geometric     |     10503 |    1146 |    1925 |     722 | **5.46x** |
| two_sym_eq    |     26598 |    1128 |    1943 |     799 | **13.69x** |
| two_sym_90/10 |     26551 |    1124 |    1922 |     799 | **13.81x** |
| flat_M3       |     21834 |    1145 |    1937 |     799 | **11.28x** |
| flat_M5       |     18481 |    1149 |    1917 |     796 | **9.64x** |
| flat_M6       |     17137 |    1147 |    1770 |     793 | **9.69x** |
| flat_M7       |      3727 |    1144 |     985 |     791 | **3.27x** |
| html_wiki     |      4765 |    1136 |    1608 |     721 | **2.96x** |
| prose_pride   |      5689 |    1135 |    1768 |     721 | **3.22x** |
| image_jpeg    |      3891 |    1132 |     975 |     755 | **3.44x** |
| json_api      |      5023 |    1138 |    1687 |     721 | **2.98x** |
| source_c      |      5159 |    1139 |    1647 |     722 | **3.13x** |
| log_apache    |      4899 |    1143 |    1628 |     722 | **3.01x** |
| dna_fasta     |     13615 |    1144 |    1920 |     758 | **7.09x** |
| csv_numeric   |      7156 |    1145 |    1861 |     722 | **3.84x** |
| gzip_random   |      4416 |       0 |       0 |     786 | **5.63x** |
| chinese_text  |      5623 |    1137 |    1483 |     720 | **3.79x** |
| calgary_pic   |      9764 |    1128 |    1858 |     723 | **5.26x** |

## AWS Graviton 4 Neoverse V2 (NEON, 64 KB L1D, block 8192)

*(post unify-framework refactor, 2026-05-15; AWS `test-c8g`,
2 vCPU c8g.large pinned `taskset -c 0`, GCC 11.5.0, Amazon Linux 2023;
100 reps × 4M symbols.  Full sweep file:
[`results/sweep_c8g-20260515-unify-all-nofse.txt`](../results/sweep_c8g-20260515-unify-all-nofse.txt).
D=5/D=6 NEON paths re-enabled on the BU direct path since 2026-05-15
— see [`IDEAS.md`](../IDEAS.md) and the lifted `PIVCO_NEON_FAST_MULTI_TBL` gate.)*

| Distribution  | PIVCO NEON | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      8523 |    1049 |    1927 |    1029 | **4.42x** |
| proba50       |      5095 |    1038 |    1935 |     900 | **2.63x** |
| proba14       |      2630 |    1042 |    1869 |     897 | **1.41x** |
| proba02       |      2271 |    1031 |    1117 |     894 | **2.03x** |
| bell_s10      |      3180 |    1032 |    1709 |     897 | **1.86x** |
| bell_s30      |      2192 |    1031 |    1060 |     894 | **2.07x** |
| bell_s80      |      2165 |       0 |       0 |     992 | **2.18x** |
| uniform       |      2398 |       0 |       0 |    1000 | **2.40x** |
| english       |      3076 |    1037 |    1869 |     954 | **1.65x** |
| zipfian       |      2116 |    1030 |    1350 |     955 | **1.57x** |
| sparse_4      |     16714 |    1050 |    1945 |    1026 | **8.59x** |
| sparse_16     |     15165 |    1052 |    1888 |    1020 | **8.04x** |
| geometric     |      4155 |    1038 |    1935 |     899 | **2.15x** |
| two_sym_eq    |     12860 |    1047 |    1939 |    1026 | **6.63x** |
| two_sym_90/10 |     12856 |    1046 |    1917 |    1026 | **6.71x** |
| flat_M3       |      9860 |    1052 |    1944 |    1024 | **5.10x** |
| flat_M5       |      9195 |    1051 |    1868 |    1022 | **4.94x** |
| flat_M6       |      9299 |    1044 |    1712 |    1019 | **5.43x** |
| flat_M7       |      2519 |    1045 |     984 |    1011 | **2.41x** |
| html_wiki     |      2184 |    1032 |    1610 |     906 | **1.36x** |
| prose_pride   |      2417 |    1032 |    1784 |     895 | **1.36x** |
| image_jpeg    |      2011 |    1034 |     970 |     948 | **1.98x** |
| json_api      |      2179 |    1034 |    1689 |     898 | **1.29x** |
| source_c      |      2428 |    1035 |    1655 |     898 | **1.47x** |
| log_apache    |      2259 |    1036 |    1628 |     897 | **1.39x** |
| dna_fasta     |      4512 |    1043 |    1914 |     959 | **2.36x** |
| csv_numeric   |      3264 |    1037 |    1872 |     900 | **1.74x** |
| gzip_random   |      2398 |       0 |       0 |     999 | **2.40x** |
| chinese_text  |      2442 |    1032 |    1487 |     897 | **1.64x** |
| calgary_pic   |      5926 |    1031 |    1839 |     901 | **3.22x** |

## AMD EPYC 7R13 Zen 3 (AVX2 + SSE4.1, 32 KB L1D, block 4096)

*(post unify-framework refactor, 2026-05-15; AWS `test-c6a`,
2 vCPU, clang-20, Amazon Linux 2023; 100 reps × 4M symbols.
Full sweep file:
[`results/sweep_c6a-20260515-unify-all-nofse.txt`](../results/sweep_c6a-20260515-unify-all-nofse.txt).
Note: Zen 3 lacks AVX-512, so codec_x86 (SSE/AVX2 paths) is dispatched
— `vpcompressw`-class partition is unavailable, expressed via pshufb
+ compress_tab instead.)*

| Distribution  | PIVCO SSE/AVX2 | huf0 X1 | huf0 X2 | trad 4s | vs best |
|---------------|----------:|--------:|--------:|--------:|--------:|
| proba80       |      8087 |    1081 |    1631 |     929 | **4.96x** |
| proba50       |      4235 |    1083 |    1615 |     806 | **2.63x** |
| proba14       |      1624 |     999 |    1530 |     802 | **1.06x** |
| proba02       |      1245 |     992 |     912 |     802 | **1.26x** |
| bell_s10      |      2213 |     992 |    1402 |     803 | **1.58x** |
| bell_s30      |      1340 |     992 |     866 |     802 | **1.35x** |
| bell_s80      |      1505 |       0 |       0 |     891 | **1.69x** |
| uniform       |      2963 |       0 |       0 |     907 | **3.27x** |
| english       |      1753 |     993 |    1533 |     861 | **1.14x** |
| zipfian       |      1400 |     986 |    1104 |     860 | **1.27x** |
| sparse_4      |      2945 |    1003 |    1619 |     931 | **1.82x** |
| sparse_16     |     24519 |    1002 |    1606 |     928 | **15.27x** |
| geometric     |      3751 |     999 |    1576 |     807 | **2.38x** |
| two_sym_eq    |     36417 |     998 |    1622 |     931 | **22.45x** |
| two_sym_90/10 |     36245 |    1001 |    1635 |     931 | **22.17x** |
| flat_M3       |      2650 |    1004 |    1625 |     929 | **1.63x** |
| flat_M5       |      2306 |    1005 |    1571 |     924 | **1.47x** |
| flat_M6       |      2197 |    1002 |    1498 |     921 | **1.47x** |
| flat_M7       |      2163 |     998 |     833 |     916 | **2.17x** |
| html_wiki     |      1246 |     993 |    1312 |     802 |   0.95x |
| prose_pride   |      1568 |     990 |    1452 |     803 | **1.08x** |
| image_jpeg    |      1412 |     991 |     798 |     859 | **1.42x** |
| json_api      |      1312 |     992 |    1391 |     802 |   0.94x |
| source_c      |      1514 |     996 |    1340 |     803 | **1.13x** |
| log_apache    |      1247 |     995 |    1323 |     802 |   0.94x |
| dna_fasta     |      4535 |    1007 |    1596 |     864 | **2.84x** |
| csv_numeric   |      2210 |     993 |    1533 |     805 | **1.44x** |
| gzip_random   |      2967 |       0 |       0 |     908 | **3.27x** |
| chinese_text  |      1503 |     995 |    1204 |     803 | **1.25x** |
| calgary_pic   |      3827 |     991 |    1527 |     807 | **2.51x** |

## Cross-Platform Summary

*(`pivco_bu` vs `huf0_x2` — or `trad_4s` where `huf0` fails — one
column per host; `--no-fse` configuration; real-world byte
distributions (`html_wiki` … `calgary_pic`) sourced from
[`extras/datasets/`](../extras/datasets/).)*

| Distribution | M4 NEON | Xeon AVX-512 | Graviton4 NEON | Zen3 SSE/AVX2 |
|---|---:|---:|---:|---:|
| proba80         | **5.91x** | **11.70x** | **4.42x** | **4.96x** |
| proba50         | **3.62x** | **5.76x** | **2.63x** | **2.63x** |
| proba14         | **2.10x** | **3.14x** | **1.41x** | **1.06x** |
| proba02         | **3.08x** | **3.88x** | **2.03x** | **1.26x** |
| bell_s10        | **2.83x** | **4.20x** | **1.86x** | **1.58x** |
| bell_s30        | **2.94x** | **4.14x** | **2.07x** | **1.35x** |
| bell_s80        | **2.77x** | **5.44x** | **2.18x** | **1.69x** |
| uniform         | **3.18x** | **5.63x** | **2.40x** | **3.27x** |
| english         | **2.55x** | **4.22x** | **1.65x** | **1.14x** |
| zipfian         | **2.36x** | **3.43x** | **1.57x** | **1.27x** |
| sparse_4        | **9.46x** | **12.34x** | **8.59x** | **1.82x** |
| sparse_16       | **10.68x** | **10.39x** | **8.04x** | **15.27x** |
| geometric       | **2.94x** | **5.46x** | **2.15x** | **2.38x** |
| two_sym_eq      | **4.82x** | **13.69x** | **6.63x** | **22.45x** |
| two_sym_90/10   | **5.03x** | **13.81x** | **6.71x** | **22.17x** |
| flat_M3         | **4.10x** | **11.28x** | **5.10x** | **1.63x** |
| flat_M5         | **4.80x** | **9.64x** | **4.94x** | **1.47x** |
| flat_M6         | **4.58x** | **9.69x** | **5.43x** | **1.47x** |
| flat_M7         | **1.43x** | **3.27x** | **2.41x** | **2.17x** |
| `html_wiki`   ‡ | **2.05x** | **2.96x** | **1.36x** | 0.95x |
| `prose_pride` ‡ | **2.04x** | **3.22x** | **1.36x** | **1.08x** |
| `image_jpeg`  ‡ | **2.71x** | **3.44x** | **1.98x** | **1.42x** |
| `json_api`    ‡ | **1.94x** | **2.98x** | **1.29x** | 0.94x |
| `source_c`    ‡ | **2.18x** | **3.13x** | **1.47x** | **1.13x** |
| `log_apache`  ‡ | **2.10x** | **3.01x** | **1.39x** | 0.94x |
| `dna_fasta`   ‡ | **3.25x** | **7.09x** | **2.36x** | **2.84x** |
| `csv_numeric` ‡ | **2.59x** | **3.84x** | **1.74x** | **1.44x** |
| `gzip_random` ‡ | **3.19x** | **5.63x** | **2.40x** | **3.27x** |
| `chinese_text`‡ | **2.50x** | **3.79x** | **1.64x** | **1.25x** |
| `calgary_pic` ‡ | **4.76x** | **5.26x** | **3.22x** | **2.51x** |

‡ Real-world byte-frequency distributions.  Source files in
[`extras/datasets/`](../extras/datasets/), regeneration via
`pivco_file_to_dist`.  `calgary_pic` is the Calgary Corpus 1bpp
CCITT scanned page (real-world proba80-shaped: 1.21 b/B entropy).

Observations across the grid:

- **Cost asymmetry between platforms is the most striking part of
  this data.**  Same algorithm, same C source, four backends —
  ratios span 0.94× (Zen 3 deep-real-text) to 23× (Zen 3
  `two_sym_eq`).  Xeon AVX-512 has the lowest minimum ratio (2.96×)
  and the highest dynamic range; Graviton 4 sits between M4 and
  Zen 3 on every dimension.
- **The K_right wire format (`5828ddb`, 2026-05-12) is the big
  recent landing.**  Real-text BU decode wins jumped from
  0.44-1.08× in late April to 0.94-3.22× now.  The `vpcompressw`
  partition + K_right-sized child buffers together amortise the
  per-node overhead that real-text trees (many internal nodes,
  Dmax 15) used to lose to.
- **`vpcompressw` matters on the partition path, but the BU
  tree_merge bridge made it less critical.**  Zen 3 has no
  `vpcompressw` and now wins 27/30 distributions — up from 8/29
  in April.  The partition cost is still real (the deepest-tree
  real-text distributions are the closest-to-parity losses) but
  the structural advantage of AVX-512 has narrowed.
- **Graviton 4 D=5/D=6 SIMD flat-decode was briefly disabled** by
  a too-broad uarch gate in the unify-framework refactor.
  Restored 2026-05-15; before the fix, `flat_M5` was 1.93× on c8g
  vs the 4.94× shown above.  The `vqtbl{2,4}q_u8`-over-32/64-byte-
  source pattern remains slow on Neoverse-V2 at small n, which is
  why the gate exists in the first place; the BU direct path keeps
  n large enough to amortise.
- **`two_sym_*` ratios spike on Zen 3 (22-23×)** because those are
  the only synthetic distributions where BOTH_LEAVES-at-root
  fires, hitting the per-block fast-path that bypasses the
  recursive subtree decoder entirely.

FSE-coded bitmaps are a separate ratio/speed knob — see
[`DATA_FORMAT.md`](DATA_FORMAT.md); enabling FSE moves several of
the proba80-shaped distributions toward lower decode speed and ~25%
smaller encoded size.

## Compression Ratio

PIVCO encoded size matches traditional Huffman within 1–4%, the only
overhead being byte-alignment rounding at each tree node.

For the block-size sensitivity sweep, see
[`BLOCK_SIZE.md`](BLOCK_SIZE.md).
