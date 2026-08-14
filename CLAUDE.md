# PIVCO-Huffman

Novel Huffman decoder using SIMD tree-walk partitioning plus a flat-subtree
fast path.  On Apple M4 it beats huf0 (zstd's Huffman) on every tested
distribution by 1.0–5×, including the moderate-entropy bell / zipfian /
english cases that previously lost against huf0 / trad_4s.  Historical
strong wins on skewed distributions (proba80 3.4×, two_sym_eq 4.9×,
uniform 2.4×) are preserved.

The flat-subtree path detects at `build_table` time every maximal
internal node whose subtree is flat with depth D ≥ 2 (all 2^D leaves at
the same relative depth), replaces D levels of bitmap-per-level with a
single N·D-bit packed region in the stream, and decodes via direct
`code_to_sym[local_code]` lookup + scatter — the same mechanism that
already powered the full-tree flat path.

## Build & Test

```sh
# Prerequisites (first time only)
git submodule update --init ext/fse  # FSE entropy coder (required, for PHA)

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Test
./build/pivco_huffman_tests

# Benchmark (arg = repeats per run, default 100)
./build/pivco_huffman_bench 20      # quick
./build/pivco_huffman_bench 100     # thorough
```

## Architecture

- **Backends**: scalar, NEON (ARM), x86 (SSE4.1 / AVX2), AVX-512 VBMI2 (Intel).  SVE is disabled (svcompact at 128-bit isn't competitive with NEON TBL).
- **Codec framework**: one `pivco_huffman_codec.c` compiled per backend as an OBJECT library, each pulling in `primitives_<backend>.h` (the only file with SIMD intrinsics).  `pivco_encode`/`pivco_decode` in `src/pivco_huffman.c` compile-time-dispatch to the best backend the build enabled (CMake detects the host tier).
- **Block size**: 32K default, 16K on Apple Silicon (see `PIVCO_BLOCK_SIZE` in `include/pivco_huffman.h`)
- **Wire format**: see `src/pivco_huffman_wire.h` for the canonical doc.  Post-order records: `[optional K_right:u16 LE at node entry][children's regions, larger-K child first][FSE marker:u8][bitmap or FSE payload]` — each bitmap sits where its merge consumes it.  Flat subtrees (D ≥ 2) skip the header and emit one N·D-bit packed region.
- **Key data structures**:
  - `compress_tab[256][32]` combined shuffle table (TBL/pshufb partition; per-arch in `pivco_huffman_{neon,x86}_tables.c`)
  - `expand_tab[256][8]` BU tree_merge shuffle table (same files)
  - `table->flat_depth[node]`, `table->flat_offset[node]`,
    `table->flat_code_to_sym[pool]` — per-table flat-subtree dispatch

## Test Hosts (AWS EC2)

```sh
# Sync to remote (cloud code is assumed stale — rsync before every run)
rsync -avz --delete --exclude='build/' --exclude='build-asan/' \
  --exclude='build-release/' --exclude='.git/' --exclude='.claude/' \
  --exclude='.vscode/' --exclude='*.dSYM' --exclude='.venv/' \
  --exclude='ext/oodle' \
  . test-XXX:pivco-huffman/
# NB: --exclude='ext/oodle' is required.  ext/oodle is a LOCAL symlink to a
# built OodleUE clone; without the exclude, rsync --delete pushes the dangling
# symlink and wipes the real Oodle SDK staged on the remote (disabling Oodle).
# To (re)enable Oodle on a remote, stage the SDK once: copy src/, include/, and
# lib/<platform>/ under ext/oodle/Engine/.../Sdks/2.9.16/, then configure with
# -DOODLE_LIB_VARIANT=shipped.

# SSH aliases: test-c6a (Zen 3 SSE4.1), test-c8i (Xeon AVX-512 VBMI2),
#              test-c8g (Graviton 4 NEON)
```

After every full sweep, save the per-platform raw output and a
headline-level `.md` summary to `results/` so we can diff across
revisions and cite prior numbers.  Then regenerate the HTML
figures with `cmake --build build --target figures` (or run
`python3 extras/figures/build.py` directly) — outputs go to
`figures/` at the project root, read by `extras/figures/build.py`
from the most recent sweep tag.

## Key Files

The codec is a single tree-walk + wire-format engine in
`pivco_huffman_codec.c`, compiled once per backend (`PIVCO_BACKEND_*`)
into an OBJECT library.  Each compile pulls in the matching
`pivco_huffman_primitives_<backend>.h` via the router header
`pivco_huffman_primitives.h`.  The unify-framework refactor landed in
five phases ending 2026-05-14; before that, each backend had its own
.c file with a duplicated tree walk -- now all four share one.

- `include/pivco_huffman.h` — public API + table struct
- `src/huffman_table.c` — `pivco_build_table` + flat-subtree detection
- `src/joint_lengths.c` — joint code-length/flat-shape optimization: bends
  Huffman lengths toward fewer, larger flat subtrees under a guard
  (`pivco_effort_t` modes; encoder-side only, wire carries plain lengths)
- `src/pivco_huffman_codec.c` — unified codec.  Compiled once per backend
  with `-DPIVCO_BACKEND_{SCALAR,NEON,X86,AVX512}`.  Owns: tree walk
  (encode + BU decode), wire-format I/O via `pivco_huffman_wire.h`,
  optional FSE attempt on the raw bitmap.  Does not include any SIMD.
- `src/pivco_huffman_primitives.h` — router; selects the backend header
  based on the `PIVCO_BACKEND_*` define
- `src/pivco_huffman_primitives_scalar.h` — scalar primitive implementations
- `src/pivco_huffman_primitives_neon.h` — NEON primitive implementations
- `src/pivco_huffman_primitives_x86.h` — SSE4.1 + AVX2 primitive implementations
- `src/pivco_huffman_primitives_avx512.h` — AVX-512 VBMI2 primitive implementations
- `src/pivco_huffman_wire.h` — single source of truth for the per-node wire record (K_right + FSE marker + bitmap)
- `src/pivco_huffman_neon_tables.{c,h}` — shared NEON compress_tab + expand_tab
- `src/pivco_huffman_x86_tables.{c,h}` — shared x86 compress_tab + expand_tab (used by codec_x86 + codec_avx512 BU SSE-tail)
- `src/pivco_huffman_neon_flat.h` — D=2..6 NEON unpack helpers (shared with bench_micro)
- `src/pivco_huffman_x86_flat.h` — D=4 SSE unpack helper
- `src/pivco_huffman_avx512_flat.h` — D=2..6 AVX-512 VBMI2 unpack helpers
- `extras/pivco_huffman_neon_prefix.c` — retired research prefix-radix backend (moved to extras 2026-05-14; BU on the standard 2-way wire format beat it on every dist/host)
- `extras/legacy_td/README.md` — git-archaeology pointer for the retired top-down decoders
- `bench/bench_main.c` — benchmark harness (4M × repeats methodology)
- `bench/bench_micro.c` — per-primitive microbench (scatter, partition, flat decode, TBL/vext throughput probes, store-port topology)
- `extras/bench/bench_flat_subtree_stats.c` — flat-subtree applicability analyzer
- `extras/bench/bench_partition_skew.c` — per-distribution partition-skewness histogram
- `extras/bench/bench_multicore.c` — multi-threaded decode scaling vs huf0_x2
- `extras/bench/bench_coalesce.c` + `bench_coalesce_avx512.c` — store-coalescing experiments (all losers)
- `extras/profile_m4.sh` + `profile_xctrace_parse.py` — one-line xctrace Time Profiler capture + per-source-line aggregator
- `README.md` — benchmark results, analysis, primary project doc
- `docs/KERNELS.md` — step-by-step NEON kernel walkthroughs (worked examples per intrinsic)
- `IDEAS.md` — full optimization-ideas log (shipped / discarded / open)
- `docs/COALESCE.md` — partition store-coalescing investigation log
- `docs/PREFIX_RADIX.md` — historical design record of the prefix-radix path
- `docs/WAVELET_TREES.md` — prior-art notes on Huffman-shaped wavelet trees
- `docs/FSE-V0.md` — per-node FSE design + the quarter-powers-of-two table schedule
- `docs/TANS-INVESTIGATION.md` — TANS routing-cost bound + flat-carve-out tax analysis
- `docs/BITPACKING.md` — FastLanes / per-D layout investigation
- `docs/FUSION.md` — leaf-fusion design and per-platform write-up
- `docs/IDEAS-codex.md`, `docs/IDEAS-gemini.md`, `docs/PIVCO-HUFFMAN-PROGRAM.md` — side notes / external-LLM idea dumps
- `results/` — timestamped + sha'd full-sweep captures
