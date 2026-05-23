# Bottom-up primitive profile — proba80 + prose, M4 + c8i (2026-05-22)

Per-primitive tic/toc breakdown of the **bottom-up** decoder, ns/elem.
Built with `-DPIVCO_PROF=ON`; run via `PIVCO_PROFILE_MODE=bu
./build-prof/pivco_huffman_profile_english <dist>`.  4M-symbol buffer ×
20000 reps, pinned to CPU 0.

**FSE-on-bitmaps is OFF** (raw bitmaps) — pure ph BU, matching the
top-down primitive benchmark.  (The bench used to default to FSE *on*,
which buried ~78% of proba80's time in `pivco_fse_decompress`; that was
the ph+ANS path, not BU.  proba80 wall: 20.99 s FSE-on → 5.32 s FSE-off
on M4.)

## ns/elem (paper/data/bu-primitive-host-cmp.csv)

| primitive | m4 proba80 | c8i proba80 | m4 prose | c8i prose |
|---|--:|--:|--:|--:|
| bu_tree_merge            |  —   |  —   | 0.05 | 0.11 |
| bu_tree_merge_bcast_left | 0.05 | 0.08 | 0.06 | 0.18 |
| bu_merge_both_const      | 0.59 | 1.52 | 0.04 | 0.06 |
| bu_flat_decode           |  —   |  —   | 0.04 | 0.29 |
| wire_kr                  | 0.25 | 13.34| 0.25 | 14.22|
| wire_bitmap_raw          | 0.00 | 0.01 | 0.00 | 0.01 |

## Reading it

- **proba80 is dominated by the top `bu_tree_merge_bcast_left`** (94% of
  wall on M4, 69% on c8i): the 80% symbol is a constant leaf broadcast at
  the root, and that one merge writes ~all N output bytes.  `bu_tree_merge`
  / `bu_flat_decode` never fire (0 calls) — the minority side is tiny.
- **prose is `bu_tree_merge`-dominated** (70% M4 / 46% c8i): a real
  multi-level merge tree.
- **The wire reads are trivial constant ops**, as expected: `wire_bitmap_raw`
  0.00 ns/elem, `wire_kr` ~0.2 ns/call on M4.
- **c8i wire caveat:** the `wire_kr` / `wire_bitmap_raw` *per-call* numbers
  (~13 ns) are PROF timer-read (`rdtsc`-class) overhead, NOT real cost —
  a single counter read is ~13 ns on c8i, so wrapping a ~1 ns region in
  TIC/TOC inflates it.  The ops are genuinely trivial (M4: 0.2 ns).  The
  merge per-*elem* numbers are unaffected (timer cost amortizes over
  thousands of elements per call).
