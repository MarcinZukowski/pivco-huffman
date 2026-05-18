# ph-td — standalone NEON top-down decoder slice

This directory is a self-contained, buildable resurrection of the
historical top-down (TD) NEON decoder that the project shipped before
the bottom-up (BU) `tree_merge` decoder superseded it on 2026-05-12
and the TD entry points were retired from the public API on 2026-05-14.

Provenance: every source file is extracted verbatim from upstream SHA
**`31cbf75`** (the last commit where TD was a working production decode
path).  Local edits are limited to:

- A stripped, no-op `pivco_prof.h` replacement.
- A trimmed `pivco_huffman.c` that dispatches to `pivco_huffman_decode_neon`
  (the TD entry) instead of the BU variant.
- `pivco_huffman_primitives_neon.h` stripped of all BU primitives
  (`tree_merge_neon`, `popcount_K_right_neon`, `expand_tab` callers).
- `pivco_huffman_neon_tables.{c,h}` stripped of the BU `expand_tab`
  family (saving ~18 KB of L1d-resident table data).
- A minimal roundtrip test driver under `test/`.
- A local `CMakeLists.txt`.

FSE is **not** compiled in (no `PIVCO_HAS_FSE` define), so the
encoder always emits raw-bitmap nodes and the decoder ignores the FSE
marker byte slot.

## What it builds

```
extras/ph-td/
├── CMakeLists.txt                 — standalone (no parent dependency)
├── include/
│   ├── pivco_huffman.h            — public API (unchanged from 31cbf75)
│   └── pivco_prof.h               — no-op stub
├── src/
│   ├── pivco_huffman.c            — TD dispatcher (rewritten, slim)
│   ├── pivco_huffman_neon.c       — encoder + TD decoder (1417 LoC, verbatim)
│   ├── pivco_huffman_neon_tables.{c,h}   — compress_tab only
│   ├── pivco_huffman_neon_flat.h         — flat-subtree NEON helpers
│   ├── pivco_huffman_primitives_neon.h   — encoder partition + init + pack_dN
│   ├── pivco_huffman_common.h            — shared defs
│   └── huffman_table.c                   — table builder
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
