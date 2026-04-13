# PIVCO-Huffman

Novel Huffman decoder using SIMD tree-walk partitioning. Beats huf0 (zstd's Huffman) by 1.3-1.6x on skewed distributions on Apple M4 and Intel AVX-512.

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
