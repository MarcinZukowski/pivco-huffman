# Bit-packing layouts for flat-region decode — the vertical investigation

> **Last content review:** 2026-08-17 (LANDED on main: THREE selectable
> layouts — natural / vertical-128 / hybrid-512+128 — via
> cfg.flat_layout + pivcohuf v0.9 FLAGS bits0-1, hybrid default, plus a
> kernel diet dropping the wide 128-block x86 forms.  An 11-host x
> 30-dist x dec+enc matrix (results/2026-08-17-flat-layout-3arm-*)
> found the 512 span REGRESSES Graviton E2E (G4 english −10% vs
> vertical-128) — the "ARM carries over" claim below is wrong for the
> hybrid — while Zen/M4 want the hybrid and Intel AVX-512 is
> natural-vs-hybrid ±1% decode / hybrid +2-3% encode.  See IDEAS.md
> "Vertical flat regions" for follow-ups.  Supersedes the 2026-05
> ARM-only investigation, summarized at the bottom.)

How flat-region bytes should be laid out on the wire, and what each layout
costs to decode (unpack + c2s map) and encode (pack) per platform.  Covers
the shipped hybrid vertical format (512-value + 128-value blocks + natural
tail, branch `vertical-pack`), the cross-check against simdcomp and
FastLanes that motivated it, and the region-size analysis behind the
gate choice.

## TL;DR

- **The hybrid vertical format is shipped** (branch `vertical-pack`):
  a flat region with D ∈ {2..7} stores its bulk in **512-value/64-lane
  blocks**, the 128..511 remainder in **128-value/16-lane blocks**, and
  the tail naturally (`pivco_vert_n512` / `pivco_vert_n` gates, both
  pure functions of (n, D)).  With it, decode and pack beat or match
  the natural row-major layout essentially everywhere — including the
  AVX-512 VBMI2 quartet that resisted vertical-128.
- The path here: vertical-128 won everywhere *except* the quartet
  (`vpmultishiftqb` is hardware vertical extraction, so their natural
  path was already near-optimal).  A cross-check against **FastLanes'
  generated AVX-512 kernels** revealed the residual cost was the
  **lane→step reorder**, not extraction: with 64 lanes one step fills a
  whole zmm row and extraction is a uniform immediate shift — no srlv,
  no multishift, no reorder.  512-value blocks capture that; and the
  64-lane block decomposes into four interleaved 16-lane quarters, so
  NEON/SSE process it with the existing 128-block kernels at different
  strides — the same cost as before, from one shared block form.
- Tree analysis over the benchmark distributions: with 16K blocks,
  **~90–100% of vertical-eligible elements sit in flat regions with
  expected n ≥ 512**; the 128..511 band holds ≤ 8% (usually ≤ 4%) —
  that band is what the 128-block middle span preserves.
- **simdcomp is not competitive** for this use case (×3–7 slower on the
  hosts tested): it unpacks to `uint32_t` (4× store traffic) and its
  kernels predate the transposed-layout idea.

## The hybrid wire format (shipped)

Canonical definition and scalar reference kernels:
`src/pivco_huffman_vertical.h`.  A block at lane count LN is 8·LN
values in LN·D bytes, byte-column major: code v lives in lane
L = v mod LN at step s = v div LN; lane substream = D bytes
little-endian with its 8 codes at bit offsets {0, D, ..., 7D}.  A flat
region stores `pivco_vert_n512(n)` codes in LN = 64 blocks, then
`pivco_vert_n(remainder)` codes in LN = 16 blocks, then a natural
tail.  D = 8 and n < 128 stay natural.  Same compressed size; only
`prim_enc_pack_dN` / `prim_merge_flat` touch flat bytes, so the codec
is unchanged.

The 64-lane block is four interleaved 16-lane sub-blocks: quarter q is
a 16-lane vertical stream at column stride 64 and output stride 64, so
128-bit engines reuse the 16-lane kernels body-for-body (macro-shared,
literal strides).  256-bit engines process 32-lane half-rows, 512-bit
engines full rows — uniform immediate shifts at every width.

Kernels:

- **NEON** (`primitives_neon.h`): merge = per-D column loads + uniform
  `vshl` + tbl map; pack = fully unrolled per D with `VSLI`/`VSRI`
  (one-op shift+insert; first touch per column is a plain move, no
  zero-init pass — safe because per-column contributions arrive at
  monotonically increasing offsets).
- **x86** (`pivco_huffman_x86_vertical.h`, shared by the x86 and avx512
  backends): xmm group cores walk 16-lane sub-blocks (both the 128
  blocks and the 512 quarters); AVX2 forms — two-128-blocks-per-ymm
  for the middle span, 32-lane half-rows for 512 blocks (contiguous
  rank loads, no `inserti128` gather); AVX-512 512-block kernels are
  the uniform-shift row form with a fused `vpermb`/`vpermi2b` map.
  The 128-block zmm merges keep the srlv (D ≤ 4) / multishift (D ≥ 5)
  forms with the D = 3 **vendor split** (`PIVCO_X86_INTEL`, from
  `/proc/cpuinfo` at configure time): Intel prefers srlv, AMD
  multishift.  The AVX2-tier 512-block D = 7 *pack* is also
  vendor-split: Intel keeps ymm half-rows, Zen 3 (which splits 256-bit
  ops) takes the xmm quarter walk.
- Two forms were tried and dropped: a 4-block zmm pack for the 128
  layout (4-load + 3-insert gather per step chokes the shuffle port)
  and, on SPR only, the zmm 512 D = 4 merge dips (see below).

### Fleet map, hybrid format (nat/vert at n = 32768, > 1 = vertical wins)

bench_prim's merge_vflat / enc_vpack rows at n = 32768 exercise the
512-block span end to end (clean rebuilds, tests pass on every host):

| host | decode D2..D7 | pack D2..D7 |
|---|---|---|
| c3 (IVB, SSE4.1) | 1.64 1.99 1.65 1.52 1.36 1.82 | 3.30 2.74 1.72 2.33 1.61 2.00 |
| c5 (SKX-era, AVX2) | 2.02 2.59 1.66 2.17 1.55 1.48 | 2.67 2.72 2.45 1.98 1.46 1.51 |
| c6a (Zen 3, AVX2) | 3.16 3.84 2.78 1.78 1.60 1.68 | 1.93 2.42 2.63 1.93 1.40 1.14 |
| c6i (ICL) | 1.19 1.07 0.97 1.19 1.09 1.08 | 2.52 2.24 1.39 1.55 1.34 1.35 |
| c7i (SPR) | 1.37 0.98 0.73 0.93 0.92 0.88 | 3.35 2.85 1.23 1.44 1.29 1.21 |
| c7a (Zen 4) | 1.06 1.00 1.01 1.03 1.01 1.01 | 2.51 2.49 1.09 1.30 1.12 1.09 |
| c8a (Zen 5) | 1.38 1.43 1.48 1.10 0.95 0.94 | 2.86 3.08 1.64 1.17 0.95 0.96 |
| c8g (Graviton 4) | 1.70 1.35 1.18 1.55 1.57 1.48 | 1.29 2.09 1.51 2.33 3.73 1.91 |
| c8i (GNR) | 1.25 1.16 1.02 0.86 0.95 0.99 | 3.14 2.85 1.73 1.33 1.15 1.22 |
| M4 | all > 1 | all > 1 |

Versus vertical-128, the quartet transformed: c7a decode is all ≥ 1.00
(was 0.76–1.01), c6i all ≥ 0.97 (D=3 0.61 → 1.07), c8i D=3 0.79 → 1.16,
and pack closed its worst holes (c8a D=2/3 1.40/1.27 → 2.86/3.08, D≥4
0.77–0.95 → 0.95–1.64).  Non-VBMI2 hosts kept or grew their wins.

Residual sub-1.0 cells: c7i decode D=4 at 0.73 is an SPR-specific dip
in the zmm 512 D=4 merge (the same kernel measures 0.97/1.02 on
ICL/GNR; the ymm half-row form measures 0.89 on SPR and would need
model-level dispatch, not just a vendor split); the rest sit at
0.86–0.99 — noise-adjacent, against a natural path that is hardware
multishift on those hosts.

E2E (ARM, vertical-128 era): G4 sparse_4 +83%, flat_M* +45–65%, jpeg
+15%, text +2–5%; M4 text +5–10%.  **The "carries over" assumption
this paragraph originally made was wrong**: re-measured E2E
(2026-08-17 matrix), the hybrid's 512 span costs Graviton real speed
(G4 english −10% vs vertical-128, hybrid even below natural on G2..G5
text) even though the n=32768 primitive ratios above look fine — the
quarter walk's strided access hurts at real region sizes.  M4 is
hybrid ≈ v128.  This is why the layout is runtime-selectable
(cfg.flat_layout) with VERTICAL_128 as the ARM-server choice.

## Cross-check: simdcomp and FastLanes on AVX-512 hosts

Harness: n = 32768, best-of-9 × 2000 reps, same binary, taskset,
ns/elem.  FastLanes rows are their generated
`x86_64_avx512bw_intrinsic_1024_uf1` 8-bit-output kernels (1024-value
transposed blocks, **no c2s map**, output in FLS order); simdcomp rows
are stock `simdunpack`/`avxunpack`/`avx512unpack` (u32 output).

c8i (Granite Rapids):

| ns/elem | D2 | D3 | D4 | D5 | D6 | D7 |
|---|---|---|---|---|---|---|
| pivco natural unpack (no map) | 0.0093 | 0.0088 | 0.0095 | 0.0142 | 0.0170 | 0.0217 |
| pivco vert-128 merge (+map) | 0.0096 | 0.0158 | 0.0087 | 0.0241 | 0.0241 | 0.0281 |
| pivco vert-512 prototype (+map) | 0.0061 | 0.0074 | 0.0066 | 0.0122 | 0.0139 | 0.0169 |
| FastLanes-1024 (no map) | 0.0044 | 0.0055 | 0.0063 | 0.0125 | 0.0141 | 0.0165 |
| simdcomp (best flavor) | 0.064–0.070 | | | | | |

c8a (Zen 5):

| ns/elem | D2 | D3 | D4 | D5 | D6 | D7 |
|---|---|---|---|---|---|---|
| pivco natural unpack (no map) | 0.0054 | 0.0053 | 0.0073 | 0.0067 | 0.0076 | 0.0088 |
| pivco vert-128 merge (+map) | 0.0045 | 0.0060 | 0.0041 | 0.0077 | 0.0078 | 0.0087 |
| pivco vert-512 prototype (+map) | 0.0035 | 0.0037 | 0.0036 | 0.0066 | 0.0082 | 0.0095 |
| FastLanes-1024 (no map) | 0.0035 | 0.0035 | 0.0036 | 0.0069 | 0.0091 | 0.0103 |
| simdcomp (best flavor) | 0.020–0.035 | | | | | |

Readings:

- simdcomp: ×3–7 slower everywhere — u32 output (4× store traffic),
  pre-transposed-layout design.  Not a contender for u8 codes.
- FastLanes beats our **natural** unpack on GNR at every D and on
  Zen 5 at D=2..4.  Our vert-128 merge (map included) already beats FL
  at D=6/7 on Zen 5.
- The vert-512 prototype ≈ FastLanes everywhere (ahead at D=5..7 on
  both hosts once FL would add a map), and beats our natural path at
  **every D on both hosts**.

## The lane-count insight

Extracting bit-field s from a lane substream needs a shift by
(s·D) mod 8 — different per step.  If one register row holds **one
step across all lanes**, the shift is uniform (immediate) and the
output is contiguous: no srlv, no multishift, no reorder, no vendor
split.  That requires lane count ≥ register byte width:

- 16 lanes (vertical-128) fill NEON/SSE registers — which is why ARM
  and old x86 are all-green already.
- zmm needs **64 lanes → 64 × 8 = 512-value blocks**.  FastLanes' 1024
  granularity is more than needed; 512 suffices (their kernels just
  process two such halves).
- The 512-block decode kernel is: keep the D byte-rows (64 B each) in
  registers, and per step one `srli_epi16`+mask (two-row OR when the
  field straddles a byte), fused `vpermb`/`vpermi2b` c2s map,
  contiguous 64 B store.
- The remaining FL edge at D=2..4 on GNR (~0.006 vs 0.0044) is their
  two-zmm unroll plus our map cost.

## Region-size analysis: is 512 too coarse a gate?

`extras/bench/pivco_flat_subtree_stats` now classifies each maximal
flat region by its **expected per-block element count** n = p·B (p =
region weight, B = block elems; multinomial concentration makes the
expectation a good proxy).  With B = 16384 (Apple default; 32K default
is strictly more favorable):

| distribution | vert-eligible ≥512 | 128..511 | <128 |
|---|---|---|---|
| proba02 | 92.0% | 5.5% | 1.2% |
| bell_s10 / s30 / s80 | 98.3 / 93.0 / 100% | 0 / 4.2 / 0% | ~0 |
| english | 61.6% | 0.0% | 0.4% |
| zipfian | 65.3% | 3.7% | 0.5% |
| html_wiki | 78.3% | 1.4% | 0.2% |
| image_jpeg | 94.4% | 2.9% | 0.5% |
| json_api | 51.0% | 8.1% | 0.1% |
| source_c | 55.2% | 4.2% | 1.2% |
| log_apache | 83.9% | 1.7% | 0.0% |
| chinese_text | 61.5% | 6.0% | 0.5% |
| root-flat (uniform, sparse_*, flat_M*) | 100% | — | — |

The 128..511 band never exceeds ~8% of elements and is usually ≤ 4%;
at B = 32768 it roughly halves again.  So a 512-value block format
sacrifices almost none of vertical's coverage.

## Decision: hybrid shipped (2026-08-02)

Hybrid won over wholesale-512 because the 128-block kernels already
existed and the quarter decomposition makes them the same code as the
512 path on narrow engines — the marginal cost of keeping the 128..511
band vertical is the wire spec listing two lane counts, not new
kernels.  The 64-lane *pack* turned out even cheaper than the 128 form
on wide engines (contiguous rank loads replace the strided gather),
which is where the big pack gains in the fleet map come from.

Open: bulk-select (`bench_select`) interaction with vertical regions;
the SPR D=4 merge dip; whether the quartet's D=2..4 wants the FL-style
two-zmm unroll.

## Historical note — the 2026-05 ARM investigation

The original version of this file investigated three layouts on
M4/Graviton 4 only: production row-major unpack, "FL-natural" (same
wire, `vst4q` interleaved stores, D ∈ {2,4} root-flat only), and a
FastLanes-transposed layout at 1024 granularity.  Findings that
remain relevant: unpack-throughput headroom was large in microbench
(up to ×4 M4, ×22 G4 at D=5/6) but E2E on real-text gained only ~5–6%
because the per-leaf scatter dominated the inner-flat path (the
scatter path has since been retired); FL-natural gave +12% E2E on
sparse_4 with no encoder change but nothing on real text.  The
vertical-128 format above is the productionized descendant of that
investigation's FL-layout branch, at 128-value granularity and with
the map fused.  See git history of this file for the full worked
examples.
