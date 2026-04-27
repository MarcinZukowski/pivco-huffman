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
git clone --depth 1 https://github.com/cyan4973/FiniteStateEntropy.git ext/fse
git clone --depth 1 https://github.com/rygorous/ryg_rans.git ext/ryg_rans

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

- **Backends**: scalar, NEON (ARM), SSE4.1 (x86), AVX-512 VBMI2 (Intel), SVE (disabled)
- **Block size**: 8192 (ARM/AVX-512), 4096 (x86 SSE) — auto-detected
- **Format**: DFS-ordered code bits; flat subtrees with depth D ≥ 2 emit one N·D-bit packed region instead of D bitmap levels
- **Key data structures**:
  - `compress_tab[256][32]` combined shuffle table (TBL/pshufb partition)
  - `table->flat_depth[node]`, `table->flat_offset[node]`,
    `table->flat_code_to_sym[pool]` — per-table flat-subtree dispatch

## Test Hosts (AWS EC2)

```sh
# Sync to remote (cloud code is assumed stale — rsync before every run)
rsync -avz --delete --exclude='build/' --exclude='build-asan/' \
  --exclude='build-release/' --exclude='.git/' --exclude='.claude/' \
  --exclude='.vscode/' --exclude='*.dSYM' --exclude='.venv/' \
  . test-XXX:pivco-huffman/

# SSH aliases: test-c6a (Zen 3 SSE4.1), test-c8i (Xeon AVX-512 VBMI2),
#              test-c8g (Graviton 4 NEON)
```

After every full sweep, save the per-platform raw output and a
headline-level `.md` summary to `results/` so we can diff across
revisions and cite prior numbers.

## Key Files

- `include/pivco_huffman.h` — public API + table struct
- `src/huffman_table.c` — `pivco_huffman_build_table` + flat-subtree detection
- `src/pivco_huffman_neon.c` — main NEON decode/encode (hot path)
- `src/pivco_huffman_neon_flat.h` — D=2..6 spread helpers (shared with bench_micro)
- `src/pivco_huffman_avx512.c` + `_flat.h` — AVX-512 VBMI2 backend
- `src/pivco_huffman_x86.c` + `_flat.h` — SSE4.1 backend
- `src/pivco_huffman_scalar.c` — reference scalar backend
- `src/pivco_huffman_neon_prefix.c` — research prefix-radix backend (`pivco_p` bench column)
- `bench/bench_main.c` — benchmark harness (4M × repeats methodology)
- `bench/bench_micro.c` — per-primitive microbench (scatter, partition, flat decode, TBL/vext throughput probes, store-port topology)
- `extras/bench_flat_subtree_stats.c` — flat-subtree applicability analyzer
- `extras/bench_partition_skew.c` — per-distribution partition-skewness histogram
- `extras/bench_multicore.c` — multi-threaded decode scaling vs huf0_x2
- `extras/bench_coalesce.c` + `bench_coalesce_avx512.c` — store-coalescing experiments (all losers)
- `extras/profile_m4.sh` + `profile_xctrace_parse.py` — one-line xctrace Time Profiler capture + per-source-line aggregator
- `README.md` — benchmark results, analysis, primary project doc
- `KERNELS.md` — step-by-step NEON kernel walkthroughs (worked examples per intrinsic)
- `IDEAS.md` — full optimization-ideas log (shipped / discarded / open)
- `COALESCE.md` — partition store-coalescing investigation log
- `PREFIX_RADIX.md` — historical design record of the prefix-radix path
- `results/` — timestamped + sha'd full-sweep captures
