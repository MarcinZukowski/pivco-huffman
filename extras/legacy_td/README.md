# Legacy top-down decoder — retired 2026-05-14

The original PIVCO-Huffman decoder used a top-down (TD) tree walk:
partition indices left/right at each internal node via a SIMD shuffle,
recurse, scatter symbols at the leaves via `indices[]`.  The bottom-up
(BU) `tree_merge` decoder superseded it in production on every
platform (NEON: 24/29 distributions faster; AVX-512: 35-57% on
proba80/text; SSE: 14-57% on text — see commit `5828ddb` for the
K_right header landing that made BU dominant).

This directory is a pointer to where the TD implementations used to
live, not a buildable archive — git history preserves the actual
code perfectly.

## What was deleted

The TD entry points and their helpers were exposed as public API:

  - `pivco_huffman_decode_neon`     — NEON TD entry
  - `pivco_huffman_decode_x86`      — SSE4.1 / AVX2 TD entry
  - `pivco_huffman_decode_avx512`   — AVX-512 VBMI2 TD entry

Plus the supporting per-backend TD-only helpers (`decode_node_*`,
`flat_decode_scatter_*`, `scatter_sym`, `scatter_both_leaves`,
`node_half_right`, `node_half_left`, `node_full`, `partition_root_8`,
`root_full`, `root_half_*`, `read_bitmap_td`).

Two bench programs that A/B-compared TD against BU also retired:

  - `extras/bench/bench_bu_decoder.c`     (NEON A/B)
  - `extras/bench/bench_bu_decoder_x86.c` (x86 A/B)

## Where to find them

NEON TD lived in `src/pivco_huffman_neon.c` until step 3.7c.  To
resurrect the implementation:

    git show e3216a0^:src/pivco_huffman_neon.c > /tmp/neon_with_td.c

(Replace `e3216a0` with the SHA of the commit BEFORE the deletion --
check `git log -- src/pivco_huffman_neon.c` to find it.)

x86 TD lived in `src/pivco_huffman_x86.c`; AVX-512 TD in
`src/pivco_huffman_avx512.c`.  Both x86 backends' TD had a known
wire-format-drift bug (FSE marker byte not handled — see the
"sse cross mismatch" failures in pre-2026-05-14 test runs).  The NEON
TD was correct and is the cleanest reference for the design.

## Why this isn't a buildable archive

Each TD decoder shared dozens of helper functions with its file's
encoder (e.g. `partition_8_sse` on x86).  Extracting a buildable
"legacy TD library" would have required duplicating the helpers or
elevating them to extern-visible symbols just for archive code.  Not
worth the maintenance for code that isn't on the critical path of any
question we care about.

If the TD-vs-BU comparison ever needs revisiting, check out the
parent commit, run the bench, capture the numbers, then return to HEAD.
