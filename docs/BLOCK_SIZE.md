# Block Size

> **Last content review:** 2026-06-16 (full-fleet sweep, dynamic-block codec)

`PIVCO_BLOCK_SIZE` is the per-block symbol count. As of the dynamic-block
work it is a **runtime** knob, not a compile-time limit:

- The codec sizes all of its scratch off the runtime `N` (the per-block
  count carried in the 2-byte wire header), so any block size in
  `[1, PIVCO_WIRE_MAX_N]` (`= 65535`, the uint16 wire-N cap) works with no
  recompile.
- `PIVCO_BLOCK_SIZE` is now only the *default* chosen by the file codec,
  CLI, and benchmarks. It defaults to **32768** on every architecture
  except **Apple Silicon (macOS/arm64), which defaults to 16384** — see the
  M4 exception below. An explicit `-DPIVCO_BLOCK_SIZE` overrides either.
- Streams are self-describing: the block size is written into the
  `.ph` header, and `pivcohuf_decompress` reads it back. A file made at one
  block size decodes on any build.

Select a block size at runtime:

```sh
pivcohuf c -b 32768 in out.ph          # CLI: symbols/block, 1..65535
./pivco_fair_bench --engines=ph --blk=16384   # benchmark a block size
```

```c
pivcohuf_compress_blk(in, n, out, &out_len, /*use_ans=*/0,
                      /*block_size=*/32768, /*timing=*/NULL);
size_t bound = pivcohuf_compress_bound_blk(n, 32768);  /* size the out buffer */
```

To override the compile-time default for a whole build:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-DPIVCO_BLOCK_SIZE=16384" \
  -DCMAKE_CXX_FLAGS="-DPIVCO_BLOCK_SIZE=16384"
```

> Note: `cmake -DPIVCO_BLOCK_SIZE=N` (a bare cache variable) does **not**
> work — nothing forwards it to the compiler. It must go through
> `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS` as above, or (preferably) just use the
> runtime `--blk` / `-b` flags.

## Why 32K

Bigger blocks amortize the fixed per-block cost (Huffman table reload + tree
walk setup + per-node wire headers) over more symbols. That fixed cost is
small relative to the per-symbol decode work on wide-L1 cores but dominates
on the smaller-L1 x86 parts — so the win is large there and modest on
Apple/Graviton. This was first reported by terrelln (issue #2) on
Zen5/Skylake; the table below maps it across the whole fleet.

### Full-fleet decode sweep (2026-06-16)

Geomean decode speedup vs an 8K block, across the `main` distribution set
(`pivco_fair_bench --engines=ph --blk=N`, dec_pb). Raw numbers in
`results/blocksize_sweep-allhosts-2026-06-16.txt`.

| host | arch                     |  4K  |  8K  | 16K  | 32K  | 64K  | peak |
|------|--------------------------|-----:|-----:|-----:|-----:|-----:|:----:|
| c3   | Ivy Bridge (SSE)         | −13% |   —  |  +8% | +12% | +13% | 64K  |
| c4   | Haswell (AVX2)           | −14% |   —  |  +9% | +15% | +16% | 64K  |
| c5   | Cascade Lake (AVX-512)   | −14% |   —  |  +9% | +15% | +16% | 64K  |
| c5a  | Zen2 (AVX2)              | −13% |   —  |  +8% | +12% | +12% | 64K  |
| c6a  | Zen3 (AVX2)              | −11% |   —  |  +5% |  +7% |  +7% | 32K  |
| c7i  | Sapphire Rapids (AVX-512)| −32% |   —  | +24% | +24% | +21% | 32K  |
| c7a  | Zen4 (AVX-512)          | −19% |   —  |  +8% | +12% | +12% | 32K  |
| c8a  | Zen5 (AVX-512)          | −25% |   —  |  +8% | +11% |  +9% | 32K  |
| c8i  | Granite Rapids (AVX-512) | −35% |   —  | +29% | +34% | +22% | 32K  |
| c7g  | Graviton3 (NEON)         | −13% |   —  |  +8% | +12% | +13% | 64K  |
| c8g  | Graviton4 (NEON)         | −12% |   —  |  +6% |  +8% |  +6% | 32K  |
| m9g  | Graviton4+ (NEON)        | −10% |   —  |  +4% |  +5% |  +4% | 32K  |
| M4   | Apple M4 (NEON)          |   *  |   —  | +3..16% per-dist | mixed (text regresses) | — | 16K |

Findings:

- **Every part benefits above 8K; none regress.** 4K is universally worse
  (−10…−35%): per-block overhead is real everywhere.
- **Magnitude tracks how overhead-bound the core is.** Modern Intel AVX-512
  gains most (Granite Rapids +34%, Sapphire Rapids +24% at 32K). AMD Zen and
  Graviton gain a moderate +5…+12%.
- **32K is the robust optimum.** Fast parts peak at 32K and *regress at 64K*
  (working set spills L2 — e.g. c8i +34% → +22%). Overhead-bound older parts
  keep inching to 64K, but 32K→64K is ≤4% there.
- **Ratio is uarch-independent: +1.1% (8K→64K) on every host.** Bigger blocks
  improve compression slightly too (fewer per-block headers), so up to 32K
  it is not a speed/size tradeoff.

### Encode

Encode tracks decode but more gently. Geomean `enc_pb` vs 8K (32K column):
Granite Rapids +14%, Zen5 +22%, Zen4 +14%; the older x86 and AVX2 parts
+4…+6%; Graviton +1…+5%; **M4 flat (−0%) — 32K does not hurt M4 encode.**
The lone regression is Sapphire Rapids (−3% at 32K; it peaks at 16K). 4K is
worse for encode too (−5…−21%). So 32K is the right call on both axes: a win
or wash everywhere for encode, with one −3% outlier.

### The Apple Silicon exception (default 16K)

M4 is the one part that prefers **16K**: 32K regresses its text/medium-entropy
distributions (and occasionally drops below 8K) because its very wide L1/L2
already absorbs the per-block cost at 16K, after which the larger working set
only hurts. So the default is gated: **macOS/arm64 → 16K, everything else →
32K** (the cloud targets, including Graviton, want 32K).

The gate is compile-time (`#if defined(__APPLE__) && defined(__aarch64__)` in
`include/pivco_huffman.h`). That is exact, not a heuristic: a macOS arm64
binary only ever runs on Apple Silicon, and a macOS binary's ISA is fixed at
build time — so there is nothing a runtime probe could learn that the build
doesn't already know. (CPU-ID asm doesn't help either: `MIDR_EL1` is EL1-only
and faults at userspace on macOS; `sysctl hw.optional.arm64` is the only
supported probe, and it still has to be `#ifdef __APPLE__`-guarded, collapsing
back to the compile-time check.)

Caveats / future work:

- Only **M4** was measured; M1–M3 are assumed to share the wide-L1 behaviour
  and inherit the 16K default.
- The *real* cause is cache size, not vendor. A principled **runtime gate
  keyed on L1/L2 size** (`sysctl hw.l1dcachesize` on macOS,
  `sysconf(_SC_LEVEL1_DCACHE_SIZE)` / sysfs on Linux) would generalise — e.g.
  a future wide-L1 Graviton could also opt into 16K — and could supersede the
  `__APPLE__` gate.
- An explicit `-DPIVCO_BLOCK_SIZE=N` overrides the gate entirely.
