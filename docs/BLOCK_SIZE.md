# Block Size

> **Last content review:** 2026-07-02 (Apple exception retired — 32K everywhere)

`PIVCO_BLOCK_SIZE` is the per-block symbol count. As of the dynamic-block
work it is a **runtime** knob, not a compile-time limit:

- The codec sizes all of its scratch off the runtime `N` (the per-block
  count carried in the 2-byte wire header), so any block size in
  `[1, PIVCO_WIRE_MAX_N]` (`= 65535`, the uint16 wire-N cap) works with no
  recompile.
- `PIVCO_BLOCK_SIZE` is now only the *default* chosen by the file codec,
  CLI, and benchmarks. It defaults to **32768** on every architecture.
  (Apple Silicon defaulted to 16384 from 2026-06-16 to 2026-07-02 — see the
  retired exception below.) An explicit `-DPIVCO_BLOCK_SIZE` overrides it.
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
| M4   | Apple M4 (NEON)          |   *  |   —  | +3..16% per-dist | mixed pre-2026-07-02, now a win (see below) | — | 32K |

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

### The Apple Silicon exception (16K, 2026-06-16 → 2026-07-02, retired)

M4 used to prefer **16K**: 32K regressed its text/medium-entropy
distributions because the per-block *scratch working set* of the BU decode
walk grew with the block and spilled what its L1 absorbed at 16K.  The
default was gated `#if defined(__APPLE__) && defined(__aarch64__)` → 16K.

The 2026-07-02 decode changes removed the cause rather than the symptom:

- **In-place merge** (one child per node decodes into `out_buf`'s tail)
  roughly halves the per-level scratch working set, and
- **page-hazard-aware carving** keeps slow-moving merge cursors off
  repeated 16 KB page-boundary splits (28-40 cycle penalty per load on
  Apple Silicon; measured ~40% worst-case on calgary_pic from placement
  alone).

With those in, 32K beats the old 16K default on Apple Silicon and the
gate was dropped — 32K everywhere.  Measured MAIN-set decode, 32K+in-place
vs the prior 16K default (5 interleaved 20-rep rounds):

- **M1 Max**: +2..11% on 8/9 dists; the slow dists gain most
  (calgary_pic +8.4%, html_wiki +6.3%, chinese_text +7.4%, json_api
  +7.5%, image_jpeg +11.4%); dna_fasta −1.4%.
- **M4**: floor-raising — calgary_pic +8.8%, image_jpeg +8.1%, proba80
  +4.1%, prose_pride/json_api +2.7%; english/html_wiki/dna −1..−3%
  (M4 run-to-run spread was 3-9%, so the small negatives are at the
  noise edge; the big positives are consistent across every run).
- Ratio improves slightly at 32K (fewer per-block headers), as on the
  rest of the fleet.

Caveat: the in-place + carving numbers above are Apple-only measurements
(M1 Max + M4).  The mechanism (fewer touched lines, hazard avoidance)
should be neutral-to-positive on the smaller-L1 x86 parts, but the EC2
fleet hasn't been re-swept with them — worth a re-run of this sweep next
time the fleet is up.
