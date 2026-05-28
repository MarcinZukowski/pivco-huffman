# Bit-packing layout investigation — flat-subtree decode

> **Last content review:** _NEVER_

Investigation into faster D-bit unpacking for the flat-subtree decode path.
Three candidate layouts, with worked examples, microbenches, end-to-end
A/B numbers, and a profile-driven assessment of how much it actually
moves on real-text decode.

## TL;DR

Microbench shows large unpack-throughput wins are available — up to
**4×** on Apple M4 and **22×** on Graviton 4 for D=5/D=6 — but the
end-to-end gain on real-text distributions (the open frontier) is
modest (~5–6%) because the per-leaf scatter dominates the inner-flat
path, not the unpack. Synthetic distributions (`sparse_*`, `flat_M*`)
gain more, but are already winning by 5–20× over huf0.

The wire-format-preserving variant ("FL-natural") gives a clean +12%
end-to-end on `sparse_4` with no encoder changes, but does nothing for
real-text. The full FastLanes-transposed layout would need a wire-format
break and dual-format streams (bulk + tail) for inner subtrees.

## The three layouts

| Layout       | Bitstream     | Unpack ops/iter | Output store/iter | Where it works |
|--------------|---------------|-----------------|-------------------|----------------|
| **current**  | row-major     | dup-tbl + var-shift + and | `vst1q` × 1 (16 codes) | anywhere — production today |
| **FL-natural** | row-major (same) | shift-imm + and × K | `vstKq` × 1 (16K codes) | only D ∈ {2,4} (D divides 8); only when output is sequential (root-flat) |
| **FL-layout** | transposed (column-major into 16 byte-lanes per FL block) | shift-imm + and per group | `vst1q` × 8 per outer iter (128 codes) | every D ∈ {2..32}; works for any output (root-flat *or* scatter); **wire format change** |

## Worked example 1 — current `flat_d2_unpack` (production)

Stream layout: 4 D=2 codes per byte, low bits first.

```
byte  k:  bit  7  6  5  4  3  2  1  0
              [c3   ][c2   ][c1   ][c0  ]
```

To unpack 16 consecutive D=2 codes, read 4 input bytes and produce one
`uint8x16_t` of codes via:

```c
uint32_t packed; memcpy(&packed, bm + k, 4);          // load 4 bytes (codes 0..15)
uint8x16_t bm_lo = vsetq_lane_u32(packed, ..., 0);    // 4 bytes in lanes 0..3, rest = 0
uint8x16_t dup   = vqtbl1q_u8(bm_lo, dup_tab);        // duplicate each byte 4×:
                                                       // [b0 b0 b0 b0  b1 b1 b1 b1  b2 b2 b2 b2  b3 b3 b3 b3]
uint8x16_t shf   = vshlq_u8(dup, shift_tab);          // per-lane variable shift:
                                                       // shift_tab = {0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6}
                                                       // → each lane now has the right code in bits 0..1
uint8x16_t out   = vandq_u8(shf, vdupq_n_u8(0x03));   // mask off the high bits → 16 codes
```

Then `vst1q_u8(symbols + i, vqtbl1q_u8(c2s_vec, out))` to apply the
code-to-symbol mapping and store 16 bytes. Per 16 codes:
**5 SIMD ops + 1 store**.

## Worked example 2 — FL-natural (same wire format, vst4q)

Same input bitstream. Process **16 input bytes = 64 codes** per
iteration. Build 4 phase-interleaved code groups via shift+mask:

```
input register reg = vld1q_u8(bm + k):
  reg[0]: [c3   c2   c1   c0  ]
  reg[1]: [c7   c6   c5   c4  ]
  reg[2]: [c11  c10  c9   c8  ]
  ...
  reg[15]:[c63  c62  c61  c60 ]

g0 = (reg >> 0) & 3:  [c0,  c4,  c8,  c12, ..., c60]   ← every 4th code, phase 0
g1 = (reg >> 2) & 3:  [c1,  c5,  c9,  c13, ..., c61]   ← every 4th code, phase 1
g2 = (reg >> 4) & 3:  [c2,  c6,  c10, c14, ..., c62]   ← every 4th code, phase 2
g3 = (reg >> 6) & 3:  [c3,  c7,  c11, c15, ..., c63]   ← every 4th code, phase 3
```

Apply `c2s` lookup to each group, then `vst4q_u8`:

```c
uint8x16_t reg = vld1q_u8(bm + (i >> 2));
uint8x16_t mask3 = vdupq_n_u8(0x03);
uint8x16_t g0 = vandq_u8(reg, mask3);
uint8x16_t g1 = vandq_u8(vshrq_n_u8(reg, 2), mask3);
uint8x16_t g2 = vandq_u8(vshrq_n_u8(reg, 4), mask3);
uint8x16_t g3 = vandq_u8(vshrq_n_u8(reg, 6), mask3);
uint8x16x4_t syms = {{
    vqtbl1q_u8(c2s_vec, g0), vqtbl1q_u8(c2s_vec, g1),
    vqtbl1q_u8(c2s_vec, g2), vqtbl1q_u8(c2s_vec, g3)
}};
vst4q_u8(symbols + i, syms);   // writes 64 bytes interleaved → stream order
```

`vst4q_u8` interleaves lane k of each vector at memory position `4*k + j`,
so the 4 phase-interleaved groups land as `c0, c1, c2, c3, c4, ..., c63`
in memory. Per 64 codes: **9 SIMD ops + 1 store**.

### Why FL-natural only works for D=2 and D=4

- D divides 8 → every byte holds K = 8/D complete codes at fixed bit
  positions → shift+mask alone produces phase-interleaved groups → vstKq.
- D=2: K=4, vst4q.
- D=4: K=2, mask 0x0F, shifts {0, 4}, vst2q.
- D=3, 5, 6, 7: codes cross byte boundaries, need TBL realignment first
  → no clean shift+mask production of phase-interleaved groups.

### Why FL-natural only works at the root

`vstKq` writes to consecutive memory. That requires output positions to
be sequential — true only for the root-flat case (`flat_decode_direct`).
Inner-flat subtrees scatter to `symbols[indices[i]]` (arbitrary order),
which forces 16 scalar STRBs and breaks the vstKq amortization.

## Worked example 3 — FL-layout (FastLanes transposed, full bitstream change)

Layout principle: split the 1024 codes of an FL block across **16
parallel byte-lanes**. Lane `k` holds codes `[k, k+16, k+32, ..., k+1008]`
(64 codes per lane). Each lane's 64*D bits live in `8*D` consecutive
bytes that interleave across lanes at byte granularity.

For D=2, FL-1024 block layout in memory (256 bytes):

```
offset    byte 0 of lane 0..15        byte 1 of lane 0..15        ...  byte 15 of lane 0..15
0         L0_b0 L1_b0 ... L15_b0      L0_b1 L1_b1 ... L15_b1      ...  L0_b15 L1_b15 ... L15_b15
                                                                       ^ offset 240..255
```

Each lane's `b0` byte holds that lane's first 4 D=2 codes (codes 0..3
within that lane's sub-stream). One `vld1q_u8(packed)` loads "byte 0
from every lane" simultaneously into a `uint8x16_t reg`. Shift+mask
extracts one code from every lane in lock-step:

```c
// reg = "byte j across all 16 lanes"
uint8x16_t c_phase0 = vandq_u8(reg, vdupq_n_u8(3));               // 16 codes (one per lane)
uint8x16_t c_phase1 = vandq_u8(vshrq_n_u8(reg, 2), vdupq_n_u8(3));
uint8x16_t c_phase2 = vandq_u8(vshrq_n_u8(reg, 4), vdupq_n_u8(3));
uint8x16_t c_phase3 = vandq_u8(vshrq_n_u8(reg, 6), vdupq_n_u8(3));
// Each c_phaseN is 16 codes from across all 16 lanes at the same intra-lane bit position.
```

The four output vectors get stored as **separate 16-byte blobs** — each
goes to its own region of the output buffer (the four "code positions"
within this byte chunk). After 8 outer iterations covering 16 input
bytes per lane, you've produced 1024 output codes.

The **critical property**: this scheme works identically for any D ∈ {2..32},
because every code lives entirely within its own lane (no byte
boundary crossings ever — bytes only cross within a single lane's
sub-stream, which can be handled by a "carry" register).

For D=3 (where natural-layout codes cross byte boundaries):

```c
// per-lane sub-stream looks like:
// lane k bytes: [b0 b1 b2 b3 ... b23]  (24 bytes per lane for D=3)
// codes 0..7 of lane k occupy bits 0..23 of lane k = b0,b1,b2

reg0 = vld1q_u8("byte 0 across lanes");
c0 = (reg0 >> 0) & 7;                // 16 codes (each lane's code 0)
c1 = (reg0 >> 3) & 7;                // each lane's code 1
partial = (reg0 >> 6) & 3;           // low 2 bits of each lane's code 2
reg1 = vld1q_u8("byte 1 across lanes");
c2 = (partial | ((reg1 & 1) << 2)) ; // complete each lane's code 2 from reg1's low bit
// ...continue
```

Same shift+mask pattern, with a `vorrq` to splice across the byte
boundary. Throughput stays comparable to D=2/D=4 because the
cross-byte handling is per-lane (uniform across all 16 lanes), not
per-code.

### Why FL-layout would need a wire-format change

Today's bitstream is row-major: code `i` at bit position `i*D`. FL
layout requires the encoder to **transpose** codes into 16 column-major
sub-streams per FL block. Old streams won't decode with new code and
vice versa. All 4 backends (scalar, NEON, SSE4.1, AVX-512) must agree.

For inner-flat subtrees of size N where N is not a multiple of 1024,
the encoder also has to emit a tail in natural layout (since shrinking
FL block size only shrinks the tail, not eliminates it). Decoder
dispatches per subtree:

```
[ FL-1024 × ⌊N/1024⌋ ][ tail of (N mod 1024) codes in natural row-major layout ]
```

Same compression ratio (both layouts pack at exactly N×D bits), but
**8 codepaths** to keep correct (4 backends × 2 unpack styles).

## Microbench results — pure D-bit unpack

`extras/bench/bench_unpack_fl_layout.c` — N=8192 codes, output GB/s
(one byte per code, no c2s lookup so the unpack itself is the
work measured).

### Apple M4 Max

| D | flat (current) | FL-natural | FL-layout | layout vs flat |
|---|----------------|------------|-----------|----------------|
| 2 | 46.5 GB/s      | 63.9       | **109.0** | **2.34×**      |
| 3 | 26.0           | —          | **105.9** | **4.07×**      |
| 4 | 65.1           | 117.8      | **141.2** | **2.17×**      |
| 5 | 25.6           | —          | **112.9** | **4.40×**      |
| 6 | 25.9           | —          | **112.1** | **4.32×**      |
| 7 | —              | —          | **101.8** | (no flat baseline; production has no flat_d7) |

### AWS Graviton 4 (Neoverse-V2)

| D | flat (current) | FL-natural | FL-layout | layout vs flat |
|---|----------------|------------|-----------|----------------|
| 2 | 18.0           | 22.1       | 35.1      | 1.95×          |
| 3 | 7.7            | —          | 33.7      | 4.40×          |
| 4 | 20.2           | 37.3       | 34.6      | 1.71×          |
| 5 | **1.3**        | —          | 30.3      | **22.5×**      |
| 6 | **1.3**        | —          | 30.5      | **22.7×**      |
| 7 | —              | —          | 29.8      | (no baseline)  |

The G4 D=5/D=6 numbers expose a pathology in the current `flat_d5/d6_unpack`
implementation: they crash to 1.3 GB/s on Graviton (vs 25 GB/s on M4).
FL-layout fixes this by avoiding the uint16-lane shift pattern those
unpackers rely on.

## End-to-end A/B — `flat_decode_direct_neon` D=2 path

Tested today on M4 Max (`./build-release/pivco_huffman_bench 20`):

| Distribution | Baseline (M sym/s) | +iota-root | +iota+FL-natural | iota Δ | FL-natural Δ |
|--------------|--------------------|------------|------------------|--------|--------------|
| **sparse_4** | 48155              | 46526      | **52265**        | -3.4%  | **+12.3%**   |
| sparse_16    | 46733              | 46346      | 46193            | -0.8%  | -0.3%        |
| flat_M3      | 23199              | 22611      | 23211            | -2.5%  | +2.7%        |
| two_sym_eq   | 26605              | 27183      | 26462            | +2.2%  | -2.7%        |
| proba80      | 9733               | 9356       | 9497             | -3.9%  | +1.5%        |
| prose_pride  | 2657               | 2655       | 2661             | -0.1%  | +0.2%        |
| (others)     | —                  | —          | —                | ±2-3% noise | ±2-3% noise |

- Iota-root (replacing `vdup+vadd` in `partition_root_8` with a static iota table load): **no end-to-end win** despite +8% microbench. partition_root is amortized across 7 deeper partition_8 calls per block.
- FL-natural in `flat_decode_direct` D=2: **+12.3% on sparse_4**, ≈0% elsewhere. Sparse_4 is the only D=2 root-flat distribution in the bench set; sparse_16 is D=4 (would need an FL-natural-D4 variant, untested but predicted by microbench to be larger gain than D=2 since vst2q amortization is bigger relative to flat_d4_unpack).

## Profile data — where decode time actually goes (M4)

`extras/profile_m4.sh` xctrace Time Profiler captures, leaf-frame
attribution with DWARF inlined frames.

**`sparse_4`** (whole tree is flat D=2, 100% via `flat_decode_direct`):
- 80.6% pivco_huffman_decode_neon@line 958 — the inlined `flat_decode_direct_neon` call site
- ≈ 100% of useful decode time is in the flat path

**`proba80`** (skewed but tree-walk path, no flat triggers):
- 47.7% pivco_huffman_decode_neon (driver + inlined partition_root_8)
- 25.8% scatter_sym
- 13.3% partition_8 (3 lines summed)
- 8.5% memset
- ≈ **0% in flat_***

**`prose_pride`** (real text, mixed depths, inner-flat subtrees fire):
- 39.6% partition_8 + partition_8_right (3 lines summed)
- 14.5% flat_decode_scatter_neon@282 — the c2s TBL + 8-element scatter loop body for inner D=3 subtree decode
- 11.9% decode_node_neon (recursive driver)
- 9.3% scatter_both_leaves (depth-1 scatter, both children leaves)
- 8.6% scatter_sym (singleton-leaf scatter)
- 5.0% pivco_huffman_decode_neon (top driver)
- 3.8% flat_d3_unpack — the D=3 unpack inside flat_decode_scatter (inlined; reported separately because xctrace recognizes it as an inlined frame)
- 3.4% flat_d2_unpack — likewise for D=2 inner subtrees
- 2.9% memset
- = **~7.2% in the unpack (unpack) functions**, **~14.5% in the flat scatter loop body** that the unpack feeds, **~21.7% in flat_* total**

The 14.5% on `flat_decode_scatter_neon@282` is **not** unpack — it's the
indexed-store scatter (16 scalar STRBs through `indices[]`) which
NEON cannot vectorize. Only the 7.2% in the unpack functions is
addressable by FL-layout's faster unpack.

## Realistic FL-layout payoff

For prose_pride the unpack accounts for **7.2% of decode time**. FL-layout
microbench predicts ~4× faster unpack on D=3 (the dominant inner-flat
case here), so realistic saving:

```
ΔT ≈ 7.2% × (1 − 1/4)  ≈ 5.4%  of decode time
end-to-end speedup ≈ 1 / (1 − 0.054) − 1 ≈ +5.7%
```

Round to **+5–6% on real-text decode**. Bigger on Graviton 4 because
of the 22× D=5/D=6 microbench multiplier, but only on subtrees that
actually use D=5/D=6 (need to check distribution histogram —
`extras/bench/bench_flat_subtree_stats` should have it).

## Findings

1. **Pure-unpack wins are large.** FL-layout cleanly delivers 2–4× on M4
   and (for D=3/5/6) up to 22× on Graviton 4 vs the current
   `flat_dX_unpack` family. The prediction that "shift+mask on
   transposed layout matches store-port bandwidth uniformly across
   D" is confirmed.

2. **End-to-end the unpack is a small fraction of real-text decode.**
   On prose_pride, only ~7% of decode time is the unpack itself; the
   bulk is partition kernels (40%), per-leaf scatter (18%), and the
   indexed-store body of inner-flat decode (15%). FL-layout doesn't
   touch any of those.

3. **Iota-root (a separate micro-optimization that closes the
   partition_root_8 vs partition_8 throughput gap on M4 microbench)
   doesn't translate end-to-end.** The root partition is amortized
   across deeper-tree partitions and contributes <1% to total decode.

4. **FL-natural D=2 in production is a clean +12% on sparse_4** and
   ≈0% elsewhere. D=4 variant (untested end-to-end) would help
   sparse_16 / flat_M3 by an even larger fraction (microbench: D=4
   FL-natural is 1.81× the flat_d4 baseline vs 1.37× for D=2, and
   `flat_decode_direct` is the only thing those distributions do).

5. **vst4q_u8 retires as ~1 store-port slot on M4 P-cores** (empirical
   from microbench, not Apple-confirmed). That's the entire reason
   FL-natural and FL-layout work — without store-port amplification,
   the trade of more shifts for "more codes per iter" wouldn't be a
   net win.

6. **Graviton 4 has a flat_d5/d6 pathology** — current 1.3 GB/s vs
   M4 25 GB/s. Worth investigating *as a separate bug* even without
   any layout change. The uint16-lane shift pattern in those
   unpackers may map poorly to Neoverse-V2's SIMD pipes.

## Suggestions

In rough order of cost-effectiveness:

### Cheap, no wire-format change
- **Investigate Graviton 4 flat_d5/d6 pathology.** 1.3 GB/s is broken,
  not just slow. Likely a fix that doesn't touch wire format and helps
  flat_M5/M6 specifically on that platform.
- **Ship FL-natural D=2 in `flat_decode_direct_neon`.** +12% on sparse_4,
  no-op everywhere else, ~30 lines of code per backend, no encoder
  change. Low risk. Concentrates win on synthetic distributions
  (low priority for the maintainer per today's discussion).
- **Ship FL-natural D=4 in `flat_decode_direct_neon`.** Untested
  end-to-end, but microbench predicts a larger relative gain than D=2
  (sparse_16 / flat_M3 / flat_M5 / flat_M6 / flat_M7 candidates).
  Same low risk; same "synthetic only" caveat.

### Expensive, wire-format change
- **Full FL-layout for inner-flat regions** is the only change that
  helps real-text. Predicted +5–6% on prose_pride, less on the smaller
  flat fractions. Cost: encoder rewrite + dual-format dispatch (FL
  bulk + natural-layout tail) for any subtree whose size isn't a
  multiple of FL block size; 4 backends × 2 unpack styles. Probably
  not worth the complexity for the predicted gain.

### Different fish — likely bigger
- Real-text decode is dominated by the **partition kernels (40%)** and
  **per-leaf scatter (18%)**. Optimizing either of those would dwarf
  any FL-layout work. Candidates (untested):
  - Vector scatter via TBL-then-store on M4 (NEON has no native
    vector scatter; AVX-512 has `vpscatter` natively, untried in this
    project).
  - Wider partition kernel (partition_16/32) with bigger TBL tables
    to amortize loads. Lookup table size grows quickly though.
  - Iterative tree walk with explicit stack to remove `decode_node_neon`
    recursive overhead (11.9% on prose_pride).

## Reference: vst4q_u8

NEON intrinsic, ARM `ST4 {V0.16B, V1.16B, V2.16B, V3.16B}, [Xn]`. Stores
4 vectors of 16 bytes each (64 bytes total) **interleaved**: lane `k`
of vector `j` lands at `out[4*k + j]`.

```
out[0]  = v0[0]    out[1]  = v1[0]    out[2]  = v2[0]    out[3]  = v3[0]
out[4]  = v0[1]    out[5]  = v1[1]    out[6]  = v2[1]    out[7]  = v3[1]
...
out[60] = v0[15]   out[61] = v1[15]   out[62] = v2[15]   out[63] = v3[15]
```

Family: `vst1q` (1 vec, 16 B), `vst2q` (2 vecs, 32 B), `vst3q` (3 vecs,
48 B, RGB triples), `vst4q` (4 vecs, 64 B, RGBA quads). The de-interleave
load is `vld4q_u8`. On M4 P-core, `vst4q_u8` empirically retires as
~1 store-port slot (microbench sustained > 1 vst1q-equivalent
throughput).

## Files

- `extras/bench/bench_unpack_dN.c` — pure-unpack microbench, current `flat_dX`
  vs FL-natural for D=2..6.
- `extras/bench/bench_unpack_fl_layout.c` — adds FL-layout (FastLanes
  transposed) for D=2..7. Lift of FastLanes' published NEON unpack
  reference impls.
- `extras/bench/bench_fl_unpack.c` — the original D=2-only experiment that
  kicked this off.
- `results/unpack_dN-m4_max-20260426.txt`
- `results/unpack_fl_layout-m4_max-20260426.txt`
- `results/unpack_fl_layout-graviton4-20260426.txt`
- `results/m4_max-iota-fl-20260426-2222.txt` — end-to-end A/B with
  FL-natural D=2 productionized in `flat_decode_direct`.
- `results/profile-mbp14-m4-{sparse_4,proba80,prose_pride}-xctrace-20260427-*.txt`
  — xctrace Time Profiler captures used for the time-share breakdown.
