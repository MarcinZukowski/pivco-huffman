# phaz — pivco-Huffman-ANS + zstd

(*p*ivco-*h*uffman-*a*ns-*z*std: zstd's LZ parse + copy engine, our branchless
SIMD entropy layer. Formerly "zph".)

Experiment: keep zstd's *parse, copy engine, repcodes, and window* unchanged,
but replace the **entropy layer** — FSE on the sequence code streams and HUF on
literals — with **pivco-Huffman (PH / PHA)** over *pivoted* (separated) streams.

Motivation (measured, see `results/`): ~60–68% of zstd-19 decode time is
entropy (FSE seq-decode 38–64%, literal-Huffman 4–21%); the copy engine is only
24–36%. PH decodes pivoted streams branchlessly. Goal: zstd ratio, faster decode.

## Minimal fork — one file + an 11-line patch

The entire in-tree footprint of the fork is **`phaz.h`** plus **`phaz.patch`**.
zstd is *not* vendored: it's the pinned `ext/zstd` submodule. The build copies it
to `build/zstd`, patches the copy (submodule stays pristine), and compiles.

- **`phaz.h`** — the whole codec, `#include`d into two zstd translation units so
  it can reuse their static internals (`SeqStore_t`, `seq_t`, `ZSTD_execSequence`):
  - *compress side* (`PHAZ_COMPRESS_SIDE`): `phaz_capture()` + `g_phaz_*` globals.
    Fires after `ZSTD_seqToCodes`; emits pivoted ll/ml/of code streams, extra
    bits, and literals. (No public zstd API runs one sequence, hence the include.)
  - *decode side* (`PHAZ_DECODE_SIDE`): `ZSTD_phazDecode()` — reconstruct each
    sequence (replaying the compressor's own `ZSTD_updateRep`, so repcodes are
    byte-exact) and call zstd's `ZSTD_execSequence`. Offset/overlap/window all
    just work — no caps, no MINOFF, no branchless-copy constraints.
- **`phaz.patch`** — 11 lines across 2 files: two `#include "phaz.h"` + one
  guarded `phaz_capture(...)` call. Pinned to the `ext/zstd` commit.

## Build

phaz lives under `pivco-huffman/extras/phaz` and reuses the enclosing repo's
pinned `ext/zstd` checkout (no submodule).  A local Makefile drives the build —
it is *not* part of pivco-huffman's CMake.

```sh
# from pivco-huffman/: make sure the entropy lib is built (phaz links it)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target pivco_huffman -j

cd extras/phaz
make                               # = tools/build.sh: copy → patch → libzstd → phaz
```

`PH=<dir>` points at pivco-huffman (default `../..`, the enclosing repo) for the
PH/PHA stream codec. Bumping the enclosing repo's zstd off the pinned commit may
break `phaz.patch` (line-pinned); re-run, re-verify byte-exact (`phaz d` checks
the decoded length; `phaz stats` round-trips in memory).

Both libzstd and libpivco_huffman vendor FiniteStateEntropy, so `build.sh` merges
pivco-huffman's lib into one object with `FSE_*/HUF_*/HIST_*/XXH*` localized
before linking (avoids the duplicate-symbol clash with zstd's copy).

## The `phaz` CLI

One binary, pivcohuf-style subcommands (`phaz -h` for the full list):

- `phaz c IN [OUT]` / `phaz d IN [OUT]` — compress / decompress a `.phaz`
  container (zstd's parse + PH/PHA-coded pivoted streams). Round-trips byte-exact.
- `phaz stats IN` — compress in memory: phaz size vs stock zstd + fused decode
  timing (PH-decode streams + reconstruct/execSequence) vs stock `ZSTD_decompress`.
- `phaz dump IN OUTDIR` — debug: write the raw pivoted streams + `meta.txt`.
- `phaz profile parse|litcost IN` — stock-zstd profiling via its public API
  (parse characterisation / literal-Huffman decode cost).
- `-l N` sets the zstd level (default 19).
- `silesia_bench.sh` (local), `remote_silesia.sh` (per-host) — corpus runners over `phaz stats`.

Dependencies: the enclosing pivco-huffman's `ext/zstd` checkout (reused, not a
submodule) and its built `build/libpivco_huffman.a`.
