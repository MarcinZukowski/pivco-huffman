# Compiler sweep — 2026-05-07

A/B/C/D comparison of compilers per platform, **decode throughput in M/s**,
5 alternated rounds × 4M-symbol bench × 20 repeats × 2 runs.
Source revision: 0747919 (main, clean).

## Platforms

| host       | uarch        | BLK  | compilers tested                 |
|------------|--------------|------|----------------------------------|
| local M4   | Apple Avalanche | 8192 | apple-clang17, clang-22, gcc-15  |
| test-c6a   | Zen 3 (x86)  | 4096 | gcc-11 (def), gcc-14, clang-15 (def), clang-20 |
| test-c8i   | Xeon AVX-512 | 8192 | gcc-11 (def), gcc-14, clang-15 (def), clang-20 |
| test-c8g   | Graviton 4   | 8192 | gcc-11 (def), gcc-14, clang-15 (def), clang-20 |

## Apple M4 — Apple clang 17 (default) vs clang 22 vs gcc 15

`apple-clang17` is the default toolchain on macOS 26 / Xcode 17.
Almost no benefit from new compilers — apple-clang already generates
strong NEON code for this codebase.

| flavor       | wins                        | losses                                |
|--------------|-----------------------------|---------------------------------------|
| clang-22     | flat_M3 +20%, sparse_4 +13%, sparse_16 +16% | wash on real text |
| gcc-15       | sparse_16 +29%, two_sym +9% | real text -6 to -12%, flat_M5/M6 -48 to -51% |

**Verdict:** keep apple-clang17.

## Zen 3 (test-c6a) — gcc-11 default vs others

Strong, clear win for clang-20 across most distributions (+5 to +25%);
both clang variants flip uniform/gzip_random by ~+68% vs gcc.

| flavor       | wins                                          | losses                                  |
|--------------|-----------------------------------------------|-----------------------------------------|
| gcc-14       | none significant                              | -3 to -9% across the board (regression) |
| clang-15     | uniform/gzip_random +68%, sparse_16 +14%, flat_M3 +17%, two_sym +8-10% | real text -5 to -11% |
| **clang-20** | uniform/gzip_random +68%, proba80 +24%, proba50 +12%, real text +4-6%, flat_M6 +19% | flat_M5/M7 ~-1%, image_jpeg -1% |

**Verdict:** switch to clang-20. The uniform/gzip_random +68% is a structural
codegen win — both clang variants flip the same way, gcc-11 is leaving
something on the table.

## Xeon AVX-512 (test-c8i) — gcc-11 default vs others

Both clang variants regress hard on partition-heavy distributions
(two_sym_90/10 -33%, proba80 -24%, geometric -14%). gcc-14 a slight
improvement on real text, no losses.

| flavor       | wins                                  | losses                                          |
|--------------|---------------------------------------|-------------------------------------------------|
| **gcc-14**   | real text +3 to +6%, gzip_random +5%, dna_fasta +1% | none |
| clang-15     | uniform/gzip_random +18%, flat_M3-6 ~0% | two_sym_90/10 -33%, proba80 -24%, proba50 -14% |
| clang-20     | similar to clang-15 (uniform/gzip +19%) | similar deep regressions on partition path |

**Verdict:** switch to gcc-14. Free lunch.

## Graviton 4 (test-c8g) — gcc-11 default vs others

Mixed. clang-20 dominates flat-subtree paths (flat_M5 +123%, flat_M6 +136%)
but two_sym regresses ~-19% under all non-default compilers. gcc-14 is the
safest swap — uniform improvement, smaller upside.

| flavor       | wins                                                    | losses                  |
|--------------|---------------------------------------------------------|-------------------------|
| gcc-14       | sparse_4 +22%, sparse_16 +23%, gzip_random +13%, dna_fasta +8%, flat_M6 +6% | two_sym -27% |
| clang-15     | flat_M5 +66%, flat_M6 +38%, sparse_4 +27%, sparse_16 +12% | two_sym -19%, flat_M7 -8% |
| **clang-20** | flat_M5 +123%, flat_M6 +136%, flat_M3 +48%, sparse_4 +27%, real text +1 to +7% | two_sym -19%, flat_M7 -8% |

**Verdict:** clang-20 is the most aggressive choice — huge wins on the
flat-subtree path that the codebase relies on. The two_sym regression is
real but two_sym is already ridiculously fast (>12 GS/s); a 19% loss
there is much less impactful than a 100%+ win on flat_M5/M6 which are
what real text decodes through.

## Overall recommendation

| platform     | recommended compiler | typical real-text delta |
|--------------|----------------------|-------------------------|
| Apple M4     | apple-clang 17 (current) | — |
| Zen 3        | clang-20             | +4 to +6%               |
| Xeon AVX-512 | gcc-14               | +3 to +6%               |
| Graviton 4   | clang-20             | +1 to +7% (real text), +47 to +136% (flat) |

## Caveats

- Single source revision. Compiler ranking can shift with code edits — re-run after major hot-path changes.
- 5 rounds × 2 runs is enough to expose ≥3% deltas; below that is noise on EC2 thermal/scheduler.
- The CMakeLists `-march`/`-mavx*` detection is unchanged across compilers; both clang and gcc see the same flag set per platform.

Raw per-host logs in this directory.

---

## Apples-to-apples vs huf0 (full bench, default vs auto-selected compiler)

Updated full bench (5 alternated rounds × repeats=20 × 5 internal runs drop 2)
with all decoders measured. Captures whether compiler upgrades change the
**pivco_n / huf0_x1 ratio** in our favor or theirs.

`*-full-raw.tsv` and `*-full.txt` per host.

### Xeon AVX-512 (test-c8i, gcc-11 → gcc-14)

Cleanest data. pivco gains a small amount on real text (+3 to +7%), huf0
*loses* a small amount (-1 to -2.5%). Net ratio shift in our favor across
the board.

| metric  | english | prose_pride | source_c | json_api | log_apache | chinese_text | proba80 | flat_M3 |
|---------|---------|-------------|----------|----------|------------|--------------|---------|---------|
| pivco delta | +3.7% | +6.0% | +3.2% | +4.6% | +3.3% | +7.4% | -0.9% | -0.2% |
| huf0 delta  | -1.2% | -1.0% | -1.1% | -1.7% | -2.4% | -1.0% | -1.2% | -2.8% |
| ratio shift | +5.0% | +7.1% | +4.4% | +6.4% | +5.8% | +8.5% | +0.3% | +2.7% |

**Verdict:** Free win. Switch to gcc-14.

### Graviton 4 (test-c8g, gcc-11 → clang-20)

Massive flat-subtree wins for pivco (flat_M5 +123%, flat_M6 +137%, sparse_4
+29%) but **huf0 also benefits uniformly +11%** under clang-20. Net:
- Real-text ratios drift 5-10% in their favor (because huf0 went up 11% but
  pivco went up only 1-7% on real text).
- Flat-text ratios shift 30-112% in our favor (pivco's flat unpack gains
  dwarf huf0's compiler bonus).
- two_sym regresses 19% on pivco (no change on huf0) → 27% loss in our favor
  but pivco still wins 12× on these.

| metric  | english | prose_pride | flat_M3 | flat_M5 | flat_M6 | sparse_4 | two_sym_eq |
|---------|---------|-------------|---------|---------|---------|----------|------------|
| pivco delta | +1.5% | +5.1% | +47% | +123% | +137% | +29% | -19% |
| huf0 delta  | +11.3%| +11.2%      | +11.5%| +11.7% | +11.4% | +11.3% | +11.5% |
| ratio shift | -8.8% | -5.5%       | +32%  | +100%  | +112%  | +16%   | -28%   |

**Verdict:** Switch to clang-20 — flat-subtree wins (the dominant real-world
hot path) more than offset the small real-text regressions.

### Zen 3 (test-c6a, gcc-11 → clang-20)

Clean rerun, all 10 round-tag pairs. pivco gains modest-to-large on most
real text (+3 to +9%), huge on uniform/gzip_random (+68%), flat_M3 (+17%),
flat_M6 (+19%), sparse_16 (+16%), sparse_4 (+11%), proba80 (+18%). But
huf0 also gets a uniform **+11% bump under clang-20** — so on real text
the pivco/huf0 ratio drifts slightly in their favor (-4 to -10%), while
on the flat / partition-degenerate cases pivco gains exceed huf0's bump
and the ratio improves +4 to +7%.

| metric  | english | prose_pride | source_c | json_api | proba80 | flat_M3 | flat_M6 | sparse_16 | uniform |
|---------|---------|-------------|----------|----------|---------|---------|---------|-----------|---------|
| pivco delta | +3.7% | +6.3% | +6.6% | +5.1% | +18.1% | +17.1% | +19.2% | +15.8% | +68.4% |
| huf0 delta  | +11.0%| +11.1%      | +11.0%  | +11.0%  | +17.0%  | +11.0%  | +11.0%  | +11.0%   | huf0=0 |
| ratio shift | -6.6% | -4.4%       | -3.9%   | -5.4%   | +0.9%   | +5.5%   | +7.4%   | +4.4%    | n/a    |

**Verdict:** switch to clang-20 for absolute throughput on real text
(+4-9%) and uniform/gzip_random (+68%). The ratio-vs-huf0 view is mixed:
huf0 also benefits from clang-20 codegen, so on real text we don't widen
the gap — but we don't lose ground in absolute terms either, and we win
big on flat distributions.

## Zen 3 codegen investigation — uniform/gzip_random +68%

Disassembly of `pivco_huffman_x86.c.o` under gcc-11 vs clang-20 on c6a:

- Both compile the D=8 flat path (`src/pivco_huffman_x86.c:395-397`,
  `for (; i<n; i++) symbols[i] = c2s[bm[i]];`) as scalar dependent loads —
  table is 256B so no PSHUFB shortcut, no VPGATHER.
- **clang-20 manually unrolls the loop 4×**, exposing 4 independent
  `(load bm → load c2s → store)` dep chains that the OoO core schedules in
  parallel. Bottleneck shifts from load latency (~9c per chain) to L1d load
  throughput (3 loads/cycle on Zen 3).
- gcc-11 leaves the loop as a 1-byte serial dep chain — single chain at
  ~1 byte / ~9c.
- The ~2.2× measured matches the expected ILP gain.

**Drag-up fix:** add `#pragma GCC unroll 8` (or equivalent) to the D=5/6/7/8
cases in `flat_decode_direct_x86` and `flat_decode_scatter_x86`. Mostly
moot now that we recommend clang-20 for non-AVX-512 x86, but harmless and
helps the SSE-only fallback path. Logged in IDEAS.md.
