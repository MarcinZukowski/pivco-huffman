# PIVCO-Huffman

> # 🚧 WIP — work in progress 🚧
>
> PivCo-Huffman is a research project optimizing Huffman coding performance. While it provides a library and a lot of code, it is not production-ready by any means.
>
> The paper ([HTML](https://marcinzukowski.github.io/pivco-huffman/paper-1.0/ph.html), [PDF](https://marcinzukowski.github.io/pivco-huffman/paper-1.0/ph.pdf)) is the canonical write-up; this
README is a short summary.

## TL;DR

Concrete on Apple M4, `pivco_bu` decode vs `huf0_x2`:

- `proba80` heavily skewed: **15.3 GB/s, 5.9× huf0_x2**.
- `proba50` / `proba14`: **9.2 / 5.2 GB/s, 3.6× / 2.1×**.
- `flat_M*` fully flat: **20–24 GB/s, 4.1–4.8×**.
- `english` / `prose_pride` / `html_wiki` / `chinese_text` real text:
  **4.3–4.8 GB/s, 2.0–2.5× huf0_x2**.
- `gzip_random` / `image_jpeg` high-entropy: **4.1–4.9 GB/s,
  2.7–3.2×**.

**PHA** (PH + per-node FSE/ANS-coded partition bitmaps) trades some
decode bandwidth for a better compression ratio on skewed data
(M4 numbers; huf0 / oo-huff produce identical Huffman ratios):

- `proba80`: ratio **8.45× vs 6.40×** for huf0 / oo-huff (+32%);
  decode **5.9 GB/s, still 2.2× huf0_x2**.
- `calgary_pic` (real proba80-shaped 1bpp scanned page): ratio
  **6.13× vs 4.79×** (+28%); decode **6.4 GB/s, 2.6× huf0_x2**.
- Moderate-entropy / real-text distributions (`english`,
  `prose_pride`, `html_wiki`, `image_jpeg`): the FSE gate doesn't
  fire when partition bitmaps aren't skewed, so PHA's ratio is
  within ±1% of plain Huffman.  PHA is the safe-default for ratio,
  PH for peak decode bandwidth.

Cross-ISA peak ratios scale with SIMD primitive width:
**Xeon AVX-512** 1.43–13.8× · **Apple M4 NEON** 1.43–10.7× ·
**Graviton 4 NEON** 1.29–8.59× · **Zen 3 SSE/AVX2** 0.94–22.5×
(three deep-real-text rows lose on Zen 3 by ~6%).

Encoded size within 1–4% of traditional Huffman.

Per-host tables, methodology, and observations across the bench grid
are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## What is PIVCO-Huffman?

PIVCO-Huffman applies the PIVoted COding approach to Huffman.
Instead of decoding symbols one at a time via table lookup, PIVCO
processes an entire block of N symbols simultaneously, using
whichever of two complementary strategies fits the shape of each
Huffman subtree best:

- a **SIMD tree-walk partition** for mixed-depth subtrees, which
  splits the block's index set by the bitmap at each internal node
  and recurses;
- a **flat-subtree fast path** for subtrees whose leaves all sit at
  the same relative depth, which replaces a sequence of per-level
  bitmaps with a single packed D-bit code per element and one direct
  `code_to_sym[code]` lookup at the bottom.

Detection and dispatch happen once at `pivco_huffman_build_table`
time — the encoder walks the tree and flags every maximal flat
subtree (local_min_depth == local_max_depth ≥ 2), pre-computes
`code_to_sym` per flat subtree, and both encoder and decoder consult
the flags to pick the right path at each node.

The full algorithm description, motivation, and analysis are in the
paper.  Pointers for the curious reader:

- **Wire format** — [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) and
  [`src/pivco_huffman_wire.h`](src/pivco_huffman_wire.h).
- **SIMD kernel walkthroughs** — [`docs/KERNELS.md`](docs/KERNELS.md)
  (NEON `partition_8`, `tree_merge`, `flat_dN_unpack` with worked
  examples).
- **Per-primitive microbench costs** —
  [`docs/KEY-PRIMITIVES.md`](docs/KEY-PRIMITIVES.md).
- **Profiling notes (historical)** —
  [`docs/PROFILING.md`](docs/PROFILING.md).
- **Block-size sweep** —
  [`docs/BLOCK_SIZE.md`](docs/BLOCK_SIZE.md).
- **Related work + wavelet-tree connection** —
  [`docs/RELATED-WORK.md`](docs/RELATED-WORK.md) and
  [`docs/WAVELET_TREES.md`](docs/WAVELET_TREES.md).
- **Test datasets** —
  [`extras/datasets/`](extras/datasets/) (synthetic +
  real-world distributions).
- **Optimization-ideas log** — [`IDEAS.md`](IDEAS.md) (shipped /
  discarded / open, with cycle-level analysis).

## Baselines

The bench grid compares PIVCO-Huffman decode against the two
production-grade Huffman decoders we consider state of the art:

- **`huf0`** — [`cyan4973/FiniteStateEntropy`](https://github.com/cyan4973/FiniteStateEntropy),
  the Huffman decoder in zstd.  4-stream interleaved, 11-bit primary
  table (X1) or 11+5-bit double-lookup (X2).  Stock auto-dispatch is
  the default headline baseline.
- **`oo-huff`** — Oodle's `newlz_arrays_huff`
  ([RAD's published OodleUE source](https://github.com/WorkingRobot/OodleUE)),
  6-stream hand-tuned ASM.  Considered the absolute SotA on Huffman
  decode.  Linked into [`bench/bench_fair.c`](bench/bench_fair.c) when
  an Oodle SDK is symlinked at `ext/oodle`.

Older legacy baselines (in-tree `trad_1s` / `trad_4s` 4-stream
reference decoders) have been retired from the headline tables.  ph is
positioned against the two codecs people actually ship.

## Build & Test

```sh
# Prerequisites (first time only)
git submodule update --init ext/fse   # FSE entropy coder (required, for PHA)

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Test
./build/pivco_huffman_tests

# Benchmark (arg = repeats per run, default 100)
./build/pivco_huffman_bench 20      # quick
./build/pivco_huffman_bench 100     # thorough
```

## Try it on your own data

PIVCO-Huffman is usable as a library — you don't have to adopt our
file format to measure it.  Three ways, easiest first:

**CLI** — `pivcohuf` compresses a file and prints size / ratio /
time / bandwidth:

```sh
./build/pivcohuf c  yourfile         # PH   -> yourfile.ph
./build/pivcohuf c -a yourfile       # PHA  (ANS-coded bitmaps; better ratio on skewed data)
./build/pivcohuf d  yourfile.ph      # decompress (auto-detects PH vs PHA)
```

**Example** — [`examples/try.c`](examples/try.c) (CMake target
`pivco_try`) compresses one file with both PH and PHA and reports
ratio + encode/decode throughput:

```sh
./build/pivco_try yourfile
#   yourfile (2000000 bytes)   [ratio = in/out, higher = better]
#     ph    6.28x  (2000000 -> 318379)   enc 704 MB/s   dec 5405 MB/s   roundtrip ok
#     pha   8.44x  (2000000 -> 236833)   enc 495 MB/s   dec 3578 MB/s   roundtrip ok
```

**Library** — link `libpivco_huffman.a` and call the buffer API in
[`include/pivcohuf_file.h`](include/pivcohuf_file.h) (no wire-format
knowledge needed):

```c
#include "pivcohuf_file.h"
size_t cap = pivcohuf_compress_bound(in_len);
uint8_t *out = malloc(cap); size_t out_len = cap;
pivcohuf_compress_ex(in, in_len, out, &out_len, /*use_ans=*/1);   // PHA; 0 = PH

size_t usz; pivcohuf_peek_uncompressed_size(out, out_len, &usz);
uint8_t *dec = malloc(usz); size_t dlen = usz;
pivcohuf_decompress(out, out_len, dec, &dlen);                    // auto-detects PH/PHA
```

To embed the codec in *your own* container/framing, use the block
primitives in [`include/pivco_huffman.h`](include/pivco_huffman.h)
(`pivco_huffman_build_table` then `pivco_huffman_encode` /
`pivco_huffman_decode` over `PIVCO_BLOCK_SIZE`-symbol blocks; call
`pivco_huffman_set_fse_enabled(1)` for PHA).

Custom block size at compile time:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-DPIVCO_BLOCK_SIZE=16384"
```

**Linking alongside zstd / FSE** — `libpivco_huffman.a` vendors
FiniteStateEntropy (`FSE_*`/`HUF_*`/`HIST_*`/`g_debuglevel`), so linking it
next to anything that *also* vendors FSE (zstd, lz4's entropy layer, …) hits
duplicate-symbol errors.  For that case the build emits a drop-in relocatable
object, `build/libpivco_huffman_local.o`, with those symbols localized and
pivco's public `pivco_*`/`pivcohuf_*` API kept global — link it instead of the
`.a` and the clash is gone:

```sh
cmake --build build --target pivco_huffman_local   # built by default too
cc your_app.c build/libpivco_huffman_local.o -Iinclude -o your_app
```

This is what e.g. `extras/phaz` links.

## Interactive tree visualization

[`figures/tree_viz.html`](figures/tree_viz.html) is a self-contained
HTML/JS explorer for Huffman trees with the flat-subtree fast path
overlaid.  Loads the 29 bench distributions from
[`figures/tree_viz_data.js`](figures/tree_viz_data.js) (regenerated
by `./build/pivco_dump_distributions > figures/tree_viz_data.js`),
accepts file/text uploads, and lets you toggle flat-subtree
detection, click flat-roots to (un)flatten for what-if analysis on
ops/leaf and chain-rule entropy totals, and scrub a max-code-length
slider.  Open the file directly in a browser — no build server
required.

---

> **Last content review:** _NEVER_
