# FSE for pivco-huffman — v0 plan

> **Last content review:** _NEVER_

Goal: get FSE coding of skewed partition bitmaps into the codec for
real, end-to-end, with measurable speed cost and ratio gain.  Targets
the per-internal-node bitmap stream; flat-subtree packing stays as-is.

The v0 is deliberately scoped to "minimum that lets us measure
something real and supports follow-ups."  Compression decisions are
per-node and based purely on the (n_left, n_right) cardinalities we
already have at encode time — no per-block tuning, no oracle search.

## Why this matters

The investigation so far (`TANS-INVESTIGATION.md`) is *entirely*
analytical bounds.  No real FSE has touched our bitmaps.  The
estimates suggest 0.3–0.8% prose-class upside under exact IID; but
real numbers depend on:

- Actual FSE compressed size on our specific bitmaps, with all the
  small-block / table-overhead realities a real encoder hits.
- Decode-side speed cost in a SIMD-heavy decode loop (we go FROM
  ~5 GB/s pure tree-walk decode TOWARD FSE-level latency on the
  FSE-coded nodes; how much does it drag the average down?).
- Skewed-data ratio gains where FSE genuinely should pay
  (proba80-style, dna_fasta, two_sym).

Also: most credible follow-ups (per-node opt-in, predefined-table
dispatch, entropy-length flat grouping, fast bit-level TANS) need a
working FSE-in-pivco substrate as their starting point.

## High-level design

1. **Pre-built FSE tables** for 25 fixed bit-probability buckets
   (quarter-powers-of-two between 0.5 and 0.992188).  Each table is
   tuned for the *byte* distribution that arises from packing IID
   Bernoulli(p) bits 8-at-a-time, so FSE operates on its native byte
   alphabet.  Tables are static; the decoder uses the same tables.

2. **Per-node dispatch.**  At encode time, for each non-flat internal
   node:
   - Read `p = max(n_left, n_right) / n` ("frequent-bit probability").
   - If `p < threshold X` (default 0.579552, table 2's bucket lower
     edge): emit raw bitmap as today.  No FSE.
   - Else: pick the highest-numbered table whose frequency ≤ `p`,
     FSE-compress the bitmap.  If FSE output is bigger than raw
     (small-block, sample-noise loss), fall back to raw.
   - If the *right* bit (rather than left) is the frequent one,
     XOR-flip the bitmap before FSE-coding so the table's expected
     distribution matches.  Mark XOR in the per-node header.

3. **Per-node header byte** indicates the path:
   - `0x00`: no FSE — raw bitmap follows (existing wire format).
   - `0x01..0x19` (1..25): FSE with that table id, no XOR.
   - `0x81..0x99` (1..25, bit 7 set): FSE with that table id, XOR'd.
   - Other values: reserved / corrupt-stream signal.
   Followed (when FSE is used) by 2-byte uint16 FSE payload length.

4. **Decoder mirror:** reads the marker byte, dispatches to FSE-decode
   with the chosen static table or to the raw-bitmap path.  XOR is
   applied to decoded bits before the existing partition logic
   consumes them.

5. **Instrumentation:** PROF_TIC/PROF_TOC pairs around FSE
   compress/decompress to surface per-node costs in the existing
   `--prof` table.

That's it for v0.  Everything else — flat-subtree splitting,
per-block table optimization, faster table selection — is v1+.

## FSE library integration

The `ext/fse` submodule is already cloned per `CLAUDE.md`'s build
prerequisites.  We need:

- `lib/fse.h`, `lib/fse_compress.c`, `lib/fse_decompress.c`,
  `lib/entropy_common.c`, `lib/error_private.h`, `lib/mem.h`.
- API surface used by v0:
  - `FSE_buildCTable_raw(...)` — once at startup per table, from our
    pre-normalized counts.
  - `FSE_compress_usingCTable(...)` — per-node, per encode.
  - `FSE_buildDTable_raw(...)` — once at startup per table.
  - `FSE_decompress_usingDTable(...)` — per-node, per decode.

Add a thin wrapper `src/pivco_fse.{c,h}` that:
- Owns the 25 CTable + 25 DTable globals (or a single struct holding
  both arrays).
- Provides `pivco_fse_init(void)` — lazy idempotent init via a
  `pthread_once`-style flag (or simple `static int initialized`).
- Provides `pivco_fse_compress(table_id, src, src_len, dst, dst_cap,
  *out_len)` and a matching decompress, both returning a small
  status enum (OK / FALLBACK_TO_RAW / ERROR).
- Keeps the FSE includes contained — no FSE types leak into the
  rest of the codec.

CMakeLists.txt: add the six FSE .c files to the pivco_huffman static
library, gated by an option (`PIVCO_FSE` default ON, OFF for
"don't link FSE" if we ever want to compare).

## Pre-computed table layout

Quarter-powers-of-two spaced "frequent-bit" probabilities, 25 entries.
We index 1..25; index 0 is reserved for "no FSE".

```
table  freq       1-in-N
-----  --------   --------
   1   0.500000     2.000
   2   0.579552     2.378
   3   0.646447     2.828
   4   0.702698     3.364
   5   0.750000     4.000
   6   0.789776     4.757
   7   0.823223     5.657
   8   0.851349     6.727
   9   0.875000     8.000
  10   0.894888     9.514
  11   0.911612    11.314
  12   0.925675    13.454
  13   0.937500    16.000
  14   0.947444    19.027
  15   0.955806    22.627
  16   0.962837    26.909
  17   0.968750    32.000
  18   0.973722    38.055
  19   0.977903    45.255
  20   0.981419    53.817
  21   0.984375    64.000
  22   0.986861    76.109
  23   0.988951    90.510
  24   0.990709   107.635
  25   0.992188   128.000
```

Generation formula (Perl one-liner, captured in `extras/gen_fse_tables.pl`):
```
for ($i=1; $i<26; $i++) { $v = 0.5**(($i+3)/4); printf("%d %f %f\n", $i, 1-$v, 1/$v); }
```

For each table `i`, the byte distribution `byte_dist[i][256]` under
IID Bernoulli(p_i):
```
P(byte = v) = p_i^popcount(v) * (1 - p_i)^(8 - popcount(v))
```
Quantise to FSE-normalized counts (sum to a fixed `tableLog`; FSE
default `tableLog = 12` → counts sum to 4096).  We're predefining,
so just bake the normalized counts into a header file
(`src/pivco_fse_tables.h`) generated by a small script — never
recomputed at runtime.

Memory: 25 CTables + 25 DTables.  FSE table size at `tableLog=12`:
~16 KB per table → ~400 KB CTables + ~400 KB DTables = 800 KB total
in BSS.  Acceptable.  (Standalone process; not a library-of-libraries.)

Asymmetry choice: only one direction is tabulated.  When the empirical
majority bit is the unexpected one, XOR the bitmap and flag it.
Cheaper than maintaining 50 tables.

## Threshold and table-selection logic

At encode time for a non-flat internal node with `n` codes and
`n_left, n_right` partition counts:

```c
double p_major = (n_left >= n_right) ? (double)n_left / n
                                      : (double)n_right / n;
int xor_flag  = (n_right > n_left);

if (p_major < THRESHOLD_X) {
    /* raw bitmap path */
    emit_marker(0x00);
    emit_raw_bitmap();
} else {
    int t_id = select_table(p_major);   /* 1..25 */
    if (xor_flag) flip_bits(bitmap, bm_bytes);
    uint8_t fse_out[FSE_MAX_OUT];
    size_t fse_len;
    fse_status_t s = pivco_fse_compress(t_id, bitmap, bm_bytes,
                                        fse_out, sizeof(fse_out),
                                        &fse_len);
    if (s != FSE_OK || fse_len + 3 >= bm_bytes + 1) {
        /* FSE failed or didn't beat raw -- fall back */
        if (xor_flag) flip_bits(bitmap, bm_bytes);  /* un-XOR */
        emit_marker(0x00);
        emit_raw_bitmap();
    } else {
        emit_marker(xor_flag ? (0x80 | t_id) : t_id);
        emit_u16(fse_len);
        emit_bytes(fse_out, fse_len);
    }
}
```

`select_table(p)`: largest `i` such that `freq[i] <= p`.  Binary
search over the 25-entry table, ~5 comparisons, cheap.

`THRESHOLD_X` is a `#define` for v0 (default `2` = table 2's
frequency 0.579552).  Make it a runtime knob in the file codec API
later if useful.

Why prefer "largest i with freq[i] <= p" over "nearest i": the
"larger" table assumes more skew than we have, which overpays for
the less-frequent bits.  The "smaller" table assumes less skew,
which slightly underpays — closer to break-even.  Marginal call;
revisit after seeing real numbers.

## Wire-format changes

Existing per-node layout (non-flat, with optional K_right header):

```
[ K_right header: 2 bytes, sometimes ]
[ raw bitmap: ceil(n/8) bytes        ]
```

New layout:

```
[ K_right header: 2 bytes, sometimes ]
[ FSE marker: 1 byte                 ]   <-- NEW
[ if marker != 0:                     ]
    [ FSE compressed length: 2 bytes ]
    [ FSE payload: <len> bytes       ]
[ else:                              ]
    [ raw bitmap: ceil(n/8) bytes    ]
```

Marker byte format:
```
bit 7    : XOR flag (1 = bitmap XOR'd before FSE)
bits 0-6 : 0 = raw bitmap, 1..25 = FSE table id
```

This bumps wire-format version (PIVCOHUF v0.1 → v0.2).  Decoder
must reject v0.1 streams if we want a clean break, OR keep a v0.1
read path for one release.  Default v0: clean break.  `include/
pivcohuf_file.h` bumps `PIVCOHUF_VERSION_MINOR`.

## Decoder changes

For each non-flat internal node in `decode_node_*` / `decode_subtree_bu`:

```c
uint8_t marker = read_u8(in);
const uint8_t *bitmap;
uint8_t bitmap_scratch[BM_MAX];

if (marker == 0) {
    bitmap = in;
    in += bm_bytes;
} else {
    int t_id = marker & 0x7F;
    int xor_flag = marker >> 7;
    uint16_t fse_len = read_u16(in);
    in += 2;
    if (pivco_fse_decompress(t_id, in, fse_len,
                             bitmap_scratch, bm_bytes) != FSE_OK)
        return PIVCO_ERR_CORRUPT;
    in += fse_len;
    if (xor_flag) flip_bits(bitmap_scratch, bm_bytes);
    bitmap = bitmap_scratch;
}
/* existing partition path consumes bitmap[] as today */
```

Per-node FSE-decode time enters the hot path proportionally to the
fraction of nodes that took the FSE branch.  The instrumentation
will tell us how much.

## Instrumentation

Add three PROF buckets:
- `PROF_FSE_ENC` — wall around `pivco_fse_compress`, by-node.
- `PROF_FSE_DEC` — wall around `pivco_fse_decompress`, by-node.
- `PROF_FSE_DECISION` — wall around `select_table` + threshold
  check, to confirm dispatch is cheap.

Plus counters:
- `PROF_FSE_HIT_COUNT[1..25]` — how often each table got picked.
- `PROF_FSE_RAW_COUNT` — how often we stayed raw.
- `PROF_FSE_FALLBACK_COUNT` — FSE attempted but didn't beat raw.

Surfaces in the existing `pivcohuf --prof` output and per-block
benchmark prints.

## Testing

`test/test_edge_cases.c` already covers:
- All 10 real datasets (`extras/datasets/*`)
- Size edge cases (0, 1, 7, 8, block-boundary, etc.)
- Uniform random + skewed + adversarial distributions

These should ALL still pass after FSE integration — same public
file-codec API, same roundtrip guarantees.  Add 2 more cases
specifically targeting FSE:
- A "high-skew" input where most internal nodes will hit the FSE
  path (e.g. byte stream with 95% one byte, 5% spread over 20).
- A "low-skew" input where almost no nodes hit FSE (uniform random
  byte stream).  Confirms the fallback / threshold logic.

Optionally add a property test that sets `THRESHOLD_X = 0.0`
(always-FSE) and confirms roundtrip still works on all datasets.
Catches "FSE path works" independently of "threshold dispatch
works."

## Postponed for v1+

- **D-flat tree splitting by entropy length.**  Per-quarter-bit (or
  half-bit) grouping of equal-Huffman-length leaves so the flat-
  subtree packing better matches real per-element entropy.
  Addresses the html_wiki / chinese_text "flat carve-out tax"
  observed in `TANS-INVESTIGATION.md`.  Separate change, doesn't
  interact with the FSE wire-format work.
- **Static-baked tables.**  v0 runs `FSE_buildCTable_raw` at startup.
  Once the normalized counts are stable, hand-bake the CTables /
  DTables into a `.rodata` header.  Saves ~ms of startup; lets us
  ship a one-binary distribution.
- **Per-table speed micro-bench.**  Once compressed/decompressed
  output is real, build `bench/bench_fse_micro.c` to measure
  per-table FSE encode/decode throughput in isolation, and the
  break-even bitmap size where FSE wins vs raw.
- **AVX-512 / NEON FSE intrinsics.**  FSE has SIMD'd variants
  (FSE_decompress_usingDTable with multi-state interleave).  Decide
  per-platform once v0 numbers are in.
- **Threshold autotuning.**  Per-block or per-file `THRESHOLD_X`
  selection based on observed dispatch hit rates.

## Out of scope for v0

- Touching the flat-subtree path at all (it stays raw N·D bits).
- Touching `pivco_huffman_table_t` shape — FSE state lives next to
  the wire bitstream, not inside the Huffman table.
- Multi-threading the FSE init.  One-time, cheap, single-threaded.
- Per-block adaptive table selection (every node uses the same 25
  prebuilt tables; the choice is per-node).

## Acceptance criteria

v0 ships when:
1. All `test/test_edge_cases.c` cases pass (including FSE-stressed
   variants).
2. `pivcohuf c` + `pivcohuf d` round-trip every dataset in
   `extras/datasets/` byte-for-byte.
3. The `--prof` output shows non-zero `PROF_FSE_ENC` / `PROF_FSE_DEC`
   and a meaningful `PROF_FSE_HIT_COUNT` distribution on a skewed
   dataset (e.g. dna_fasta or a constructed proba80 file).
4. The cross-codec sweep in `TANS-INVESTIGATION.md` re-run with
   FSE-enabled `pivcohuf`: report new ratio numbers vs the
   pre-FSE row.  Document deltas, especially on dna_fasta,
   chinese_text, and the synthetic proba80.

---

## Adaptivity in production — measured on Calgary `pic` (2026-05-13)

After v0 shipped we went looking for a **realistic** proba80-like
dataset to stress the FSE path on real, structured bytes (as opposed
to the synthetic IID stream that bench_distributions generates from a
fixed frequency histogram).  Two findings.

### LZ4-compressed data is NOT proba80-like

Tested LZ4 outputs of every file in `extras/datasets/` plus
prose/JSON variants.  Byte-level Huffman ratios land at 86–94% of
raw — same band as `english` / `html_wiki`, **not** the 25%-of-raw
extreme of proba80.  Reason: LZ4 is pure LZ77 with no entropy
coding, so the residual byte distribution after match elimination
converges toward the source's "what LZ77 couldn't predict" floor —
mildly skewed, not heavy-tailed.

On the 4-way bench (`pivco_bench_4way --file …`):
- `ph` ≡ `phe` on every LZ4 file (FSE dispatch never fires
  meaningfully — partition skew at every node is too close to 50/50).
- `huf0 − fse` gap is 0.05–0.1% on LZ4 outputs vs 27% on proba80 —
  there's almost no fractional-bit entropy left for TANS to capture.

LZ4 *flattens* byte histograms — that's what makes it fast.  Don't
look there for proba80 stand-ins.

### Calgary Corpus `pic` is a perfect realistic proba80 stand-in

The 1989 Calgary Corpus `pic` file (1bpp 1728×2376 scanned CCITT
test page of a French signal-processing textbook) has been a
standard compression benchmark for 40 years.  Byte distribution:

```
0x00 = 447139 (87.12%)   ← 8 consecutive white pixels
0xff =  10692  (2.08%)   ← 8 consecutive black pixels
0x0f =   4088  (0.80%)   ← edge bytes (white→black transition)
0x1f =   2901  (0.57%)
... long geometric tail across 159 used byte values
```

That's textbook proba80 shape — one dominant symbol at ~87%,
geometric falloff, sparse tail.  And it's *real bytes from a real
scanned document*, not a synthetic stream.

Headline numbers on `pic` (real bytes, file mode):

| codec           | ratio   | decode M/s |
|-----------------|---------|-----------:|
| `ph` (FSE off)  | 21.7%   | 11602      |
| `phe` (FSE on)  | 14.6%   |  3210      |
| `huf0`          | 20.9%   |  2339      |
| `fse`           | 15.4%   |   649      |

`phe` beats `fse` by 0.8% on ratio while decoding 5× faster.  Why?

### The structural win: per-internal-node-per-block table dispatch

Yann's FSE codes a 128 KB chunk with one normalized-count table.
`phe` codes each non-flat internal node's partition bitmap with its
own choice from 25 pre-built quarter-power-of-two tables.  At 62
blocks × ~1.9 non-flat root-attempt nodes/block, that's 118 FSE
table-selection events per file, each adapting to local bitmap
skew.

Concrete demonstration: dump the **root node**'s table-id choice
across all 62 blocks (`pivco_bench_fse_table_use extras/datasets/calgary_pic`):

```
Root-node committed-table histogram (blocks per table_id):
tid  p_nom    blocks
  0   no         2   ##                          ← p < 0.625, no FSE
  3  0.646       4   ####
  4  0.703       1   #
  5  0.750       8   ########
  6  0.790       6   ######
  7  0.823       4   ####
  8  0.851       3   ###
  9  0.875       2   ##
 10  0.895       6   ######
 11  0.912       3   ###
 12  0.926       1   #
 13  0.938       4   ####
 14  0.947       2   ##
 15  0.956       1   #
 16  0.963       1   #
 17  0.969       2   ##
 18  0.974       2   ##
 25  0.992      10   ##########                  ← blank-page blocks

Block-by-block root timeline (1-9 = tbl 1-9, a-p = tbl 10-25,
. = below MIN_THRESHOLD):
[   0] pppppp6558p8ad66754576.333.99dab
[  32] bdfi536baaeciaghe765585h7adppp
```

The same tree position, in 62 blocks of the same file, picks **17
different FSE tables**.  You can read the document's structure off
that string:
- Blocks 0–5: `pppppp` — table 25 (p=0.992): blank top margin.
- Blocks 6–22: mid-tables 5/6/8/10: textbook body.
- Block 22/26: `.` — partition too even (dense figure region).
- Blocks 60–61: `ppp` — blank bottom margin.

That whole dynamic range is what Yann's FSE averages out into a
single 256-symbol probability table per chunk.  The ~0.8% ratio
gap to FSE comes specifically from the 10 blocks where pivco picks
table 25 (compresses 1024 bytes → 24 bytes — 98% reduction) plus
the table-3 sparse-bitmap regions, neither of which a global
per-chunk distribution can express.

### Synthetic-bench note

`pivco_huffman_bench`'s sweep generates IID 4M-symbol streams from
the registered frequency histogram, so on `calgary_pic` the
**synthetic** numbers (`phe` ≈ 16.97%, `fse` ≈ 16.35%) understate
the real-data advantage — synthetic IID has no per-block structure
for the dispatch to adapt to.  Real-data `phe` beats `fse`; synthetic
`fse` slightly beats `phe`.  Use `pivco_bench_4way --file
extras/datasets/calgary_pic` to see the real-data behavior.

### Tooling

- `pivco_huffman_fse_stats_reset/get` — per-table-id commit/attempt
  counters (public API in `include/pivco_huffman.h`).
- `pivco_huffman_fse_root_count/get` — per-block root-event log
  (table_id chosen, observed p_major, committed flag).
- `pivco_bench_fse_table_use FILE` — dumps the histogram + per-block
  root timeline as shown above.

All instrumentation lives in `src/pivco_huffman_neon.c`
(`g_pivco_fse_commit[26]`, `g_pivco_fse_root_log[]`) and is
single-threaded debug-only — not on the hot path.

## 2026-05-30 — pha vs stock FSE on real data: the prebuilt-table mechanism

After fixing `bench/bench_distributions.c` to read real source files
verbatim (instead of IID-resampling from each file's byte histogram),
`pha` shows much stronger wins on locally-clustered data than the
IID benchmark suggested:

| dataset | `fse_stk` ratio | `pha` ratio |
|---|---|---|
| calgary_pic (real) | 6.63 | **7.09** |
| calgary_pic (IID resampled) | 6.42 | 6.13 |

The on-disk methodology preserves spatial clustering (long runs of
0x00 in white-page bitmaps) which IID resampling destroyed.  On
text-y datasets the change is small (most ratios within ±2%); on
calgary_pic it flips the leaderboard — pha now beats fse_stk on
ratio AND decodes ~6× faster (4231 vs 666 MB/s).

### Why pha wins on calgary_pic

Probed by adding a temporary `fse_8k` engine (same FSE codec, force
8 KB chunks instead of 128 KB) and timing FSE_buildDTable
separately.  Numbers on M4, real calgary_pic:

| engine | ratio_op | ratio_pb | dec_op MB/s | build / chunk | build share |
|---|---|---|---|---|---|
| `fse_stk` (12 × 128 KB) | 6.63 | 6.61 | 666 | 4.8 μs | 2.6% |
| `fse_8k` (192 × 8 KB) | **6.95** | 6.60 | 650 | 2.2 μs | **18.9%** |
| `pha` | **7.09** | 7.10 | **4231** | ~0 (table prebuilt) | ~0% |

Findings:

1. **Per-chunk adaptation is intrinsically valuable for clustered
   data.**  fse_8k → fse_stk swap alone gains +0.32 ratio
   (6.63 → 6.95) on calgary_pic — about 70% of the way to pha.
   The gain comes from each 8 KB chunk's FSE table being a better
   fit to its local byte distribution than the 128 KB-averaged
   table.

2. **The cost is asymmetric, and it's about the table header.**
   fse_stk pays ~256-500 bytes of NCount header per 128 KB chunk
   (< 0.4% overhead).  fse_8k pays the same ~300 bytes per
   chunk — but 16× more chunks → 3-6% header overhead.  That's why
   fse_8k's ratio doesn't reach pha's 7.09 despite per-chunk
   adaptation.  pha sidesteps this entirely: per-block table
   selection costs **1 byte** (the marker for one of 50 prebuilt
   tables), so per-block adaptation is essentially free.

3. **Build time is also asymmetric, and it's significant on small
   chunks.**  FSE_buildDTable takes ~2-5 μs per chunk depending on
   alphabet size and count magnitudes.  At 128 KB chunks this is
   negligible (~2-3% of decode time).  At 8 KB chunks it becomes
   **15-19%** of total decode time across all measured datasets.
   pha pays zero build cost — its 50 DTables are constructed once
   at library init.  This is why pha decodes 6× faster than
   fse_stk on calgary even with stricter per-block adaptation.

4. **The pha mechanism in plain English:** instead of fitting a
   bespoke FSE table per chunk and shipping it (the natural FSE
   path), pha picks the closest of 50 prebuilt Bernoulli(p) tables
   and tags it with one byte.  This trades a small fit penalty
   (prebuilt is ~0.5 bit/elem worse than fitted on a perfect-fit
   chunk) for two enormous savings: (a) **table header overhead
   drops from ~300 B/chunk to 1 B/chunk**, and (b) **decoder build
   cost drops from ~2-5 μs/chunk to 0**.  The math works out
   strongly in pha's favor on heavily clustered data; on text-y
   data the per-chunk fit penalty is more visible but the decoder
   speedup remains decisive.

5. **The prebuilt-table approach has a ceiling.**  If a hypothetical
   FSE variant could ship a fitted-per-chunk table with the same
   1-byte selection overhead, calgary_pic ratio could likely
   reach ~9+ (estimated from `fse_8k`'s output minus its header
   overhead).  That's the natural next research direction — but
   it requires a different wire format (table-quantization
   strategy or learned-vocab tables), out of scope for v0.

### Source of truth

- bench: `bench/bench_fair.c` (`fse_stk`, `pha`, `ph` rows).
- real-data path: `bench/bench_distributions.c` (`load_file` +
  `bench_dist_size`).  CMake passes the dataset dir at compile
  time (`PIVCO_DATASET_DIR`).
- raw sweep outputs: `results/fair/<host>-<date>.txt`.
- aggregated table: `paper/data/fair.csv` (driven by the same
  bench output via `extras/fair_csv.py`).

## 2026-05-30 — ph/pha on zstd's entropy streams (parked)

A side investigation: extract every byte stream zstd hands to its
entropy coders (literal residuals → HUF, plus LL/OF/ML codes → three
FSE streams) and benchmark ph/pha against huf0/fse_stk on each.
**Not part of the official benchmark set** — these distributions
were temporarily added to `bench/bench_distributions.c` to produce
the numbers below and then removed.

### How to reproduce

- Patched libzstd lives at `ext/zstd/` (cloned from upstream
  5233c58 = v1.6.0).  Patches in `lib/decompress/zstd_prof.{h,c}`
  add accumulators for HUF / FSE-build / sequence-loop on decode
  and LZ-match-find / HUF / FSE-build / FSE-emit on encode, plus
  optional dump pointers (`g_zstd_prof_lit_dump_fp`,
  `g_zstd_prof_{ll,of,ml}_dump_fp`) that snapshot the four
  streams during compression.
- Extractor: `extras/bench/extract_zstd_lits.c` — dumps the four
  streams + Shannon H per stream to `extras/datasets/lits/`.
- Decode/encode time-breakdown driver:
  `extras/bench/bench_zstd_breakdown.c`.
- Rebuild + run:
  ```
  (cd ext/zstd/lib && make clean && CFLAGS='-O3 -DNDEBUG' make libzstd.a -j)
  cc -O3 -I ext/zstd/lib -o build/extract_zstd_lits \
      extras/bench/extract_zstd_lits.c ext/zstd/lib/libzstd.a
  cc -O3 -I ext/zstd/lib -o build/bench_zstd_breakdown \
      extras/bench/bench_zstd_breakdown.c ext/zstd/lib/libzstd.a
  ```

### zstd decode time-budget (M4, L3, real data, hot cache)

| dataset | huff% | fseB% | fseD% | lzX% | other% | MB/s |
|---|---|---|---|---|---|---|
| calgary_pic | 21.5% | 6.9% | 57.1% | 14.3% | 0.2% | 4458 |
| cat-wiki.html | 14.1% | 5.2% | 78.1% | 2.5% | 0.1% | 2732 |
| pride.txt | 5.5% | 3.0% | 87.8% | 3.6% | 0.1% | 1512 |
| dna_fasta | 1.7% | 3.0% | 95.8% | 0.0% | 0.0% | 1449 |
| cat-image.jpg | 0.0% | 11.5% | 24.9% | 6.0% | 57.6% | 50096 |

LZ match-execution is ~free on hot M4 caches (cache-resident
source reads + tight wildcopy).  The FSE sequence-decode loop
dominates on text-y data (78–96%).  Huffman of literals is
at most 22% even on the most literal-heavy real dataset.

### Encode time-budget (M4, L3)

| dataset | lz% | huff% | fseB% | fseE% | MB/s |
|---|---|---|---|---|---|
| calgary_pic | 62.8% | 10.4% | 9.8% | 12.2% | 1078 |
| cat-wiki.html | 74.5% | 4.4% | 8.2% | 10.9% | 683 |
| pride.txt | 81.9% | 1.5% | 6.3% | 9.5% | 301 |
| dna_fasta | 79.5% | 0.8% | 8.0% | 11.0% | 316 |
| cat-image.jpg | 13.7% | 82.9% | 0.9% | 0.5% | 1604 |

LZ match-find dominates encode (63–82% on compressible data).
Symmetric inverse of the decode picture: matches are expensive
to *discover* and cheap to *redeem*.

### Stream sizes + Shannon H (bytes in)

| dataset | raw | lit | LL | OF | ML | H_raw | H_lit | H_LL | H_OF | H_ML |
|---|---|---|---|---|---|---|---|---|---|---|
| calgary_pic | 501 | 39 | 12 | 12 | 12 | 1.21 | 5.37 | 2.80 | 2.89 | 4.74 |
| cat-wiki.html | 985 | 65 | 49 | 49 | 49 | 5.48 | 5.96 | 2.03 | 4.04 | 4.54 |
| pride.txt | 721 | 43 | 84 | 84 | 84 | 4.53 | 4.99 | 1.24 | 3.47 | 3.15 |
| dna_fasta | 488 | 11 | 63 | 63 | 63 | 2.08 | 2.35 | **0.68** | 3.41 | 2.06 |
| chinese_text | 483 | 112 | 51 | 51 | 51 | 5.79 | 5.95 | 2.54 | 3.56 | 2.78 |
| json_api | 515 | 37 | 12 | 12 | 12 | 5.20 | 5.82 | 2.37 | 3.79 | 4.33 |
| cat-image.jpg | 273 | 263 | 0 | 0 | 0 | 7.89 | 7.89 | — | — | — |

(All sizes KB.  Bits/byte for H.)

### Relative share of compressed bits (Shannon estimate)

| dataset | lit% | LL% | OF% | ML% |
|---|---|---|---|---|
| calgary_pic | **63%** | 10% | 10% | 17% |
| cat-wiki.html | 41% | 11% | 21% | 23% |
| chinese_text | **66%** | 14% | 19% | 16% |
| pride.txt | 25% | 12% | **33%** | 30% |
| dna_fasta | 6% | 10% | **52%** | 31% |
| cat-image.jpg | **100%** | 0% | 0% | 0% |

The FSE streams (especially OF) own most of the compressed
bitstream on text-y data, not the literals.  Image-like data
inverts that.

### ph/pha vs huf0/fse_stk on these streams (M4, ratio_op)

| stream | H | huf0 | ph | pha | fse_stk |
|---|---|---|---|---|---|
| L_calgary | 5.37 | 1.47 | 1.45 | 1.45 | 1.48 |
| L_pride | 4.99 | 1.59 | 1.57 | 1.57 | 1.60 |
| L_dna | 2.35 | 3.22 | 3.18 | 3.30 | 3.40 |
| LL_pride | 1.24 | 5.33 | 5.17 | **6.16** | **6.39** |
| LL_dna | **0.68** | 6.82 | 6.71 | **11.29** | **11.72** |
| OF_dna | 3.41 | 2.31 | 2.28 | 2.35 | 2.34 |
| ML_pride | 3.15 | 2.51 | 2.47 | 2.48 | 2.54 |

### Conclusions (then parked)

1. **Literals are flat-ish; ph ≈ pha ≈ huf0 within ~2% ratio.**
   pha's prebuilt Bernoulli tables have nothing to fit on
   post-LZ residuals.

2. **LL is where Huffman fails and pha shines.**  When the
   alphabet is small and one symbol is very common (LL_dna
   H=0.68), the integer-bit Huffman floor caps ratio at 6.82
   — pha's prebuilt-Bernoulli mechanism nearly doubles that
   to 11.29, almost matching fse_stk's 11.72.  This is the
   archetypal "where does ANS beat Huffman" case made
   concrete on zstd's actual workload.

3. **pha closes the huf0→fse_stk gap on FSE streams much
   better than ph does** (LL_dna gap to fse_stk: huf0=4.90,
   ph=5.01, pha=0.43).  pha's per-block adaptation matters
   most when the underlying distribution is far from uniform.

4. **A hypothetical ph-in-zstd swap would buy ~10% decode
   time on Huffman replacement and 10-40% on FSE-stream
   replacement** (driven by the FSE share in the decode
   profile above).  Not pursued — wire-format change rules
   it out as a real zstd contribution; logged as research.

### Reproducing the per-stream A/B

If you re-add the stream distributions, the entries (now
removed from `bench/bench_distributions.c`) were:

```
{ .name = "L_<dataset>",  .source = "lits/<file>.lits" }
{ .name = "LL_<dataset>", .source = "lits/<file>.ll" }
{ .name = "OF_<dataset>", .source = "lits/<file>.of" }
{ .name = "ML_<dataset>", .source = "lits/<file>.ml" }
```

The `.lits/.ll/.of/.ml` files in `extras/datasets/lits/` are
generated by `extract_zstd_lits` and are deliberately not
committed.
