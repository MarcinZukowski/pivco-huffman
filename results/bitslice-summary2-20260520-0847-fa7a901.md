# bitslice microbench v2 — vs ph's actual SIMD path

Updated baseline.  Now compares against ph's real `flat_decode_direct_avx512_inner`
SIMD path (via the shared `flat_d{2..6}_unpack_avx512` helpers), not just the
scalar `AVX512_FLAT_UNPACK_SWITCH` fallback.  `dec-ph` IS what ph runs today
for D ≤ 6; for D ≥ 7 ph falls back to scalar (same as `dec-cls`).

## Granite Rapids (test-c8i, Xeon 6975P-C)

```
decode (ns/sym; baseline = ph-simd for D<=6, scalar for D>=7):
D   |    dec-cls |     dec-ph |     dec-bs |  dec-blend | bs-vs-ph  blend-vs-ph
----+------------+------------+------------+------------+----------------------
2   |      0.259 |      0.030 |      0.020 |      0.014 |  1.50x      2.20x
3   |      0.246 |      0.029 |      0.029 |      0.022 |  1.00x      1.31x
4   |      0.279 |      0.031 |      0.036 |      0.040 |  0.84x      0.76x
5   |      0.229 |      0.036 |      0.044 |          - |  0.81x           -
6   |      0.772 |      0.048 |      0.052 |          - |  0.92x           -
7   |      0.273 |      0.273 |      0.069 |          - |  3.97x           -
```

## Zen 5 (test-c8a, EPYC 9R45)

```
D   |    dec-cls |     dec-ph |     dec-bs |  dec-blend | bs-vs-ph  blend-vs-ph
----+------------+------------+------------+------------+----------------------
2   |      0.221 |      0.017 |      0.012 |      0.011 |  1.38x      1.50x
3   |      0.223 |      0.018 |      0.015 |      0.011 |  1.23x      1.66x
4   |      0.335 |      0.017 |      0.020 |      0.018 |  0.86x      0.93x
5   |      0.212 |      0.028 |      0.024 |          - |  1.16x           -
6   |      0.587 |      0.022 |      0.028 |          - |  0.77x           -
7   |      0.245 |      0.245 |      0.040 |          - |  6.17x           -
```

## Takeaways

| D    | Winner             | vs ph today  |
| ---  | ------------------ | ------------ |
| 2    | **blend**          | 1.5x (SPR), 1.5x (Zen5)  |
| 3    | **blend**          | 1.3x (SPR), 1.7x (Zen5)  |
| 4    | ph (current)       | bit-sliced/blend lose 7-24%  |
| 5    | ph (current; close) | bs slightly wins on Zen5, loses on SPR  |
| 6    | ph (current)       | bs loses 8-23%  |
| 7    | **bit-sliced**     | 4.0x (SPR), 6.2x (Zen5)  |

### Two real wins surface

1. **D = 2, 3:** blend gives 1.3-2.2x over ph's existing SIMD path on
   both hosts.  The c2s lookup folded into the broadcast vectors is
   the source of the win -- ph's path still pays for a separate pshufb.
2. **D = 7 (and presumably D=8):** bit-sliced gives 4-6x because ph is
   scalar there today.

### No-go zone

- **D = 4, 5, 6:** ph's `flat_d{4,5,6}_unpack_avx512` + pshufb/vpermb
  pipeline is faster than our bit-sliced variant.  The 16-symbols-per-iter
  with cheap code extraction beats the 64-symbols-per-iter with 4-6 ops
  of OR/shift bookkeeping.

### Encode (unchanged)

Still 26-40x vs naive scalar -- but the baseline is naive scalar, not
ph's actual `prim_pack_dN`.  Need a proper comparison before claiming
anything for encode.

## Honest scope of the win

If we shipped the obvious changes:

- D ≤ 3 flat decode -> blend     (covers ~most low-entropy data)
- D = 7,8 flat decode -> bit-sliced  (covers the long-tail high-D case)
- D = 4,5,6 -> leave alone

Need: estimate how often each D shows up in real data.  D=2,3 are
common in low-entropy distributions; D=7,8 mostly when leaves don't
flatten well.  Worth profiling distributions to weight the upside.
