# Block Size Sweep

> **Last content review:** _NEVER_

*(Numbers below are pre-flat-subtree (early 2026-04) and stale.
They pre-date flat-subtree, leaf-fusion, the Graviton 4 D=5/D=6
gate, the K_right wire format, FSE-coded bitmaps, and the
unify-framework codec refactor.  N-dependence for stick-tree-shaped
distributions (proba80/50) was always closest to flat across N once
the prefill memset landed; the table below is consistent with that.
A fresh block-size sweep on the current code is still TODO.)*

PIVCO NEON decode throughput (M/s) by block size.  Measured with the
4M realistic workload (each block size is recompiled and
re-benchmarked):

| N     | proba80 | proba50 | proba14 | proba02 | english | geometric |
|------:|--------:|--------:|--------:|--------:|--------:|----------:|
|  4096 |    9122 |    4845 |    2118 |     845 |    2251 |      4502 |
|  8192 |    9401 |    5020 |    2306 |    1119 |    2418 |      4872 |
| 16384 |    9410 |    5034 |    2402 |    1333 |    2454 |      5124 |
| 65536 |    9883 |    4677 |    2245 |    1509 |    2238 |      4732 |

- **proba80 is now ~flat across block sizes** (9.1–9.9 GB/s): the
  prefill memset + skip_node optimization dominates — the tree walk
  only processes ~20% of indices regardless of block size.
- **N=8192 remains the default**: good balance across distributions.
  16384 is slightly better on moderate (proba14, geometric) but
  65536 regresses on proba50/english as index arrays spill L1.
- **Compared to pre-optimization**: all block sizes roughly doubled
  (e.g. proba80 4.1–4.3 GB/s → 9.1–9.9 GB/s).

To rebuild the codec at a custom block size:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-DPIVCO_BLOCK_SIZE=16384"
```
