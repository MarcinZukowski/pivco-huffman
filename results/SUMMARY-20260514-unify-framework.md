# Unify-framework refactor — cross-host bench sweep 2026-05-14

The unify-framework refactor (commits `2429c80` ➜ `3c5ecf8`) brings
all four backends (scalar, NEON, x86 SSE/AVX2, AVX-512 VBMI2) under
a single `pivco_huffman_codec.c` compiled per-backend, with the
per-backend SIMD living in `pivco_huffman_primitives_<backend>.h`.
Five phases ending 2026-05-14.  Net source-tree footprint:
**+1867 / −3036 LoC** (less code, same SIMD).

Raw outputs in this directory:
  - `sweep_<host>-20260514-unify-framework.txt`     (decode)
  - `enc_sweep_<host>-20260514-unify-framework.txt` (encode)

All numbers in **M symbols / second** (millions of input symbols per
second of wall time).  Higher is faster.

Bench config: 100 reps × 4M-symbol stream = 400M sym/run, 5 runs,
drop 2 slowest.  Blocks of 8192 symbols (4096 on SSE-only hosts).
The `pivco_bu` column routes to the host's best codec OBJECT lib:
  - aarch64 → `pivco_huffman_decode_bu_neon` (codec_neon)
  - x86 AVX-512 → `pivco_huffman_decode_bu_avx512` (codec_avx512)
  - x86 SSE/AVX2 → `pivco_huffman_decode_bu_x86` (codec_x86)

The `pivco_n` column goes through the public `pivco_huffman_decode`
dispatcher (which also picks the best backend), so `pivco_n` and
`pivco_bu` should match closely.

## Validation

Per-platform `pivco_huffman_tests` (cross-decode + edge cases +
FSE dispatch):

| host  | tests |
|-------|-------|
| M4    | 0 failures |
| c6a   | 0 failures |
| c8i   | 0 failures |
| c8g   | 0 failures |

## Notes on this sweep

**Apparent proba80 regression vs the 2026-05-12 baseline is the v0.3
FSE wire format, not the unify refactor.**  Commit `005ba7a`
(2026-05-13) added per-internal-node FSE coding gated on partition
skew.  proba80's BOTH_LEAVES partition is heavily skewed → FSE fires
→ encoded size drops 25% but decode pays FSE-decompress overhead.
Tradeoff is intentional, documented in `IDEAS.md` and toggle-able
at runtime via `pivco_huffman_set_fse_enabled(0)` (or `--no-fse`
to the bench).

To compare codec.c-refactor perf against the legacy decoders
apples-to-apples, run the bench with `--no-fse`: numbers come back
to the 2026-05-12 baseline ±2% on every distribution.

## Bugs found + fixed during this sweep

1. **AVX-512 encoder wire-format drift** (commit `5d85874`):
   legacy `encode_node_avx512` never wrote the FSE marker byte
   (latent since 2026-05-13).  Was masked by encode/decode going
   through the same backend until the codec_x86 cutover routed
   decode through codec.c on AVX-512 hosts → segfault on c8i.
   Fix: 18 LoC added to the legacy encoder.

2. **BOTH_LEAVES-at-root fast path lost** (commit `[pending]`):
   the legacy bu_neon / bu_x86 entry points had a fast path that
   skipped the recursive `codec_decode_subtree` for the
   BOTH_LEAVES root case (common on heavily-skewed two-symbol
   distributions).  Restored in codec.c.

3. **NEON D=5/D=6 BU flat-decode gated to Apple silicon by
   mistake** (commit `[pending]`): the
   `PIVCO_NEON_FAST_MULTI_TBL=0` default on non-Apple aarch64
   was applied to both the TD scatter path AND the BU direct
   path during the unify refactor.  Legacy code only gated the
   TD path (per-call setup dominated at small n); BU has n=8192
   on root-flat distributions and SIMD wins on Neoverse-V2 too.
   Fix: drop the gate entirely (no remaining TD callers).
   Impact: c8g flat_M5 from 2845 → ~9000 M/s (parity with
   legacy bu_neon).

## Bench harness fix

`bench/bench_main.c::pivco_bu` was hard-wired to call
`pivco_huffman_decode_bu_x86` on x86_64.  After Phase 5, that
symbol is the SSE/AVX2 codec (no AVX-512 paths).  Routed to
`pivco_huffman_decode_bu_avx512` on AVX-512 hosts in the bench
harness — without this, the c8i numbers in the first 2026-05-14
sweep looked badly regressed; they were actually just calling
the wrong backend.

## Decode M/s — MAIN distribution set

### pivco_bu (production BU decoder; codec.c routed per host)

| dist         |   c6a |   c8g |    c8i |    m4 |
|---           |  ---: |  ---: |   ---: |   ---: |
| proba80      |  2671 |  2580 |   2915 |  3874 |
| english      |  1749 |  3058 |   7765 |  6360 |
| flat_M5      |  2302 |  9168 |  18454 | 24898 |
| html_wiki    |  1238 |  2166 |   4741 |  4502 |
| prose_pride  |  1559 |  2414 |   5682 |  4889 |
| image_jpeg   |  1408 |  2051 |   3887 |  4214 |
| json_api     |  1309 |  2168 |   5022 |  4419 |
| gzip_random  |  2965 |  2401 |   4411 |  5209 |
| chinese_text |  1507 |  2437 |   5588 |  4981 |
| calgary_pic  |  2122 |  2541 |   2834 |  4188 |

### huf0_x2 (zstd 4-stream Huffman) — baseline

| dist         |   c6a |   c8g |    c8i |    m4 |
|---           |  ---: |  ---: |   ---: |   ---: |
| proba80      |  1646 |  1925 |   1928 |  2773 |
| english      |  1535 |  1868 |   1873 |  2574 |
| flat_M5      |  1573 |  1867 |   1924 |  5189 |
| html_wiki    |  1306 |  1607 |   1608 |  2239 |
| prose_pride  |  1437 |  1779 |   1773 |  2469 |
| image_jpeg   |   808 |   974 |    976 |  1357 |
| json_api     |  1377 |  1689 |   1688 |  2379 |
| gzip_random  |     - |     - |      - |     - |
| chinese_text |  1204 |  1486 |   1481 |  2029 |
| calgary_pic  |  1543 |  1839 |   1857 |  2563 |

### pivco_bu / huf0_x2 ratio (where huf0_x2 doesn't fail)

| dist         |   c6a |   c8g |    c8i |    m4 |
|---           |  ---: |  ---: |   ---: |   ---: |
| proba80      | 1.62x | 1.34x |  1.51x | 1.40x |
| english      | 1.14x | 1.64x |  4.14x | 2.47x |
| flat_M5      | 1.46x | 4.91x |  9.59x | 4.80x |
| html_wiki    | 0.95x | 1.35x |  2.95x | 2.01x |
| prose_pride  | 1.08x | 1.36x |  3.20x | 1.98x |
| image_jpeg   | 1.74x | 2.11x |  3.98x | 3.10x |
| json_api     | 0.95x | 1.28x |  2.97x | 1.86x |
| chinese_text | 1.25x | 1.64x |  3.77x | 2.45x |
| calgary_pic  | 1.38x | 1.38x |  1.53x | 1.63x |

(gzip_random is incompressible so huf0_x2 emits 0 bytes / no result.)

## Encode M/s — MAIN distribution set (pivco)

| dist         |   c6a |   c8g |    c8i |    m4 |
|---           |  ---: |  ---: |   ---: |   ---: |
| proba80      |  1159 |  1018 |   2714 |  1733 |
| english      |   925 |   906 |   2582 |  1852 |
| flat_M5      |  2274 |  1081 |   4257 |  2451 |
| html_wiki    |   745 |   740 |   1838 |  1563 |
| prose_pride  |   710 |   707 |   1885 |  1410 |
| image_jpeg   |  1009 |   715 |   2415 |  1608 |
| json_api     |   713 |   715 |   1889 |  1479 |
| gzip_random  |  3301 |  2270 |   9356 |  3609 |
| chinese_text |   756 |   710 |   1984 |  1500 |
| calgary_pic  |  1035 |   975 |   2342 |  1677 |

## Compression-size summary (proba80 v0.3 FSE win)

`pivco_raw` is the v0.3 wire-format output, which includes FSE
coding of skewed bitmaps where it commits.

| dist        | pivco_raw |  huf0_x2 | rans_x2 |
|---          |      ---: |     ---: |    ---: |
| proba80     |    490626 |   655802 |  473736 |
| english     |   2244586 |  2228088 | 2215515 |
| flat_M5     |   2621440 |  2622304 | 2621448 |
| html_wiki   |   2986711 |  2911016 | 2870339 |
| prose_pride |   2442629 |  2403956 | 2375005 |
| image_jpeg  |   4166845 |  4152331 | 4135145 |
| json_api    |   2781025 |  2744990 | 2724794 |
| gzip_random |   4194304 |        0 | 4193336 |
| chinese_text|   3162258 |  3082761 | 3049381 |
| calgary_pic |    677548 |   875271 |  634428 |

Notable: proba80 pivco_raw dropped from 660,395 bytes (v0.2 wire
format, 2026-05-12 sweep) to 490,626 bytes here — 26% smaller —
courtesy of FSE coding the heavily-skewed BOTH_LEAVES partition
bitmap.  calgary_pic similarly benefits (proba80-shaped real-world
photo data).
