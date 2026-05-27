# ph-td — standalone NEON top-down decoder slice

This directory is a self-contained, buildable resurrection of the
historical top-down (TD) NEON decoder that the project shipped before
the bottom-up (BU) `tree_merge` decoder superseded it on 2026-05-12
and the TD entry points were retired from the public API on 2026-05-14.

Provenance: the TD sources originate from upstream SHA **`31cbf75`**
(the last commit where TD was a working production decode path).

**Consolidation (2026-05-26):** the slice no longer forks the primitive /
table / flat headers.  Those duplicated copies were deleted; the TD sources
now `#include` the parent repo's shared headers (via `../../src` + `../../
include` on the include path), so TD uses the **current** primitives —
`pack_dN`, `prim_enc_partition_full`, `compress_tab`, the flat unpack helpers —
rather than a stale 31cbf75 snapshot.  This kills the divergence that had the
forked `pack_d7_neon` etc. lagging the main repo.  Only the TD-unique sources
remain local:

- `pivco_huffman.c` — TD dispatcher (slim).
- `pivco_huffman_neon.c` — encoder + NEON TD decoder (the unique TD walk +
  index-scatter kernels).  Encode partition calls the shared
  `prim_enc_partition_full`.
- `pivco_huffman_avx512.c` — self-contained AVX-512 TD decoder (uses the
  shared `pivco_huffman_avx512_flat.h`).
- `pivco_huffman_naive.c` + `huffman_table.c` — the naive-tree build/decode
  baseline (no main-repo equivalent).
- `include/pivco_huffman.h` — TD/naive public API + the table struct
  (compatible with main's).
- `include/pivco_prof.h` — prof header (a superset: TD slots + the `PROF_BU_*`
  slots the shared primitives reference).
- `include/phtd_names.h` — symbol namespacing so the lib links beside the
  main (BU) lib in one binary.
- `test/` driver + this `CMakeLists.txt` (now depends on the parent tree).

FSE is **not** compiled in for this lib (`-UPIVCO_HAS_FSE`), so the encoder
always emits raw-bitmap nodes and the decoder ignores the FSE marker slot.

## What it builds

```
extras/ph-td/
├── CMakeLists.txt                 — depends on parent (../../src, ../../include, ../../ext/fse)
├── include/
│   ├── pivco_huffman.h            — TD/naive API + table struct
│   ├── pivco_prof.h               — prof header (TD + PROF_BU_* slots)
│   └── phtd_names.h               — phtd_* symbol namespacing
├── src/
│   ├── pivco_huffman.c            — TD dispatcher (slim)
│   ├── pivco_huffman_naive.c      — naive-tree build + decode baseline
│   ├── pivco_huffman_neon.c       — encoder + NEON TD decoder (shared prims)
│   ├── pivco_huffman_avx512.c     — AVX-512 TD decoder (shared flat header)
│   ├── huffman_table.c            — opt + naive table builders
│   └── phtd_shim.c                — bench-facing adapter
│   (primitive / table / flat headers now come from ../../src + ../../include)
└── test/
    └── ph_td_test.c               — roundtrip on 4 distributions × 3 block counts
```

## Build & run

```sh
cd extras/ph-td
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/ph_td_test
```

Output should end with `12 tests, 0 failed`.

## Why keep this around

The TD walk is the more pedagogically natural way to describe a
prefix-code decoder ("start with the whole input, partition by the
root bit, recurse"), and the implementation is structured around
that descent.  The BU walk is a duality that's faster on every
microarchitecture we've measured but harder to derive from first
principles.  Keeping a buildable TD reference makes A/B comparisons
trivial and gives us a clean target for future experiments (e.g. the
GPU decoder bench in `extras/gpu/` started from the TD-style
partition primitive).

## What's NOT in here

- BU `tree_merge_neon` and friends (use the parent project for those).
- x86 / AVX-512 / SVE backends (the TD code existed there too — see
  `extras/legacy_td/README.md` for a pointer; only the NEON path is
  resurrected here).
- The unified `codec.c` framework from the post-2026-05-14 refactor.
- FSE per-node entropy coding.
