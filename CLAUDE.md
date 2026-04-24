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
- **Format**: DFS-ordered code bits, no continuation bitmaps
- **Key data structure**: `compress_tab[256][32]` combined shuffle table (shared between NEON and neon2)

## EC2 Test Hosts

```sh
# Sync to remote
rsync -avz --delete --exclude='build/' --exclude='.git/' --exclude='.claude/' --exclude='*.dSYM' --exclude='.venv/' . ec2test-XXX:pivco-huffman/

# Hosts: ec2test-c6a (Zen3), ec2test-c8i (Xeon AVX-512), ec2test-c8g (Graviton4)
```

## Key Files

- `src/pivco_huffman_neon.c` — main NEON decode/encode (hot path)
- `src/pivco_huffman_avx512.c` — AVX-512 vpcompressw backend
- `src/pivco_huffman_neon2.c` — WIP 4-way fused partition (has bugs)
- `bench/bench_main.c` — benchmark harness (4M × repeats methodology)
- `RESULTS.md` — all benchmark results and analysis
