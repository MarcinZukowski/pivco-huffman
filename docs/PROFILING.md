# Profiling

> **Last content review:** _NEVER_

> **Historical snapshot.**  The profile below was taken on the
> top-down decoder (`decode_node_neon`, `partition_8`,
> `scatter_both_leaves`, `flat_decode_scatter_neon`) on 2026-04-26.
> The production decoder has been bottom-up since 2026-05-12
> (`5828ddb` K_right wire format) and the source files referenced
> here (`pivco_huffman_neon.c`) have been folded into
> `pivco_huffman_codec.c` + `pivco_huffman_primitives_neon.h` as of
> the 2026-05-14 unify-framework refactor.  The per-function names
> below no longer exist verbatim.  The section is retained because
> the qualitative breakdown — partition body 41%, flat-subtree 24%,
> leaf scatter 18%, recursion glue 12% — and especially the
> conclusion that **NEON store-port throughput, not TBL latency, is
> the partition bottleneck** still describe the bottom-up decoder
> faithfully (the BU `tree_merge` is store-port bound for the same
> reason).  A BU re-profile is planned.

**Last refreshed:** 2026-04-26 07:30 UTC, commit
[`0a99f6c`](../) (post AVX-512 / SSE4.1 bench port, leaf-child fusion +
flat-subtree fast path both shipped).  Workload: **`prose_pride`** —
real Project Gutenberg prose, 96 distinct bytes, max code length 11
(length-limited Huffman, matching `huf0`'s default cap),
~47% flat-subtree coverage — the real-world deep-tree distribution
that PIVCO most closely contests against `huf0_x2`.

## Methodology

Earlier revisions of this section used macOS `sample` (1 ms IP
sampling, no source-line attribution) and **manual** instruction-
offset → source-region mapping by hand-disassembling
`decode_node_neon`.  That approach produced numbers that turned out
to be off by 10–15 percentage points — the collapsed-offset display
in `sample` plus M4's deep OoO retirement attribution made the
hand-mapping noisier than it looked.

This refresh uses **Instruments / `xctrace` Time Profiler with
DWARF inlined-frame attribution** instead.  The profile binary is
built `RelWithDebInfo` and `dsymutil`'d so DWARF debug info covers
every inlined helper (`partition_8`, `scatter_sym`,
`scatter_both_leaves`, `flat_decode_scatter_neon`, `flat_dN_unpack`,
…), letting the trace attribute each sample directly to a source
function and line.  Decode-loop samples are isolated by filtering
backtraces that contain `decode_node_neon` or
`pivco_huffman_decode_neon` (excludes the encode-phase setup the
harness runs before the decode loop).

All five steps are wrapped in
[`../extras/profile_m4.sh`](../extras/profile_m4.sh) for a one-line
re-run:

```sh
./extras/profile_m4.sh prose_pride 12   # dist, duration_s
```

The script (1) configures+builds RelWithDebInfo, (2) generates the
`.dSYM`, (3) records an xctrace Time Profiler trace, (4) exports
the `time-profile` table to XML, (5) runs
[`../extras/profile_xctrace_parse.py`](../extras/profile_xctrace_parse.py)
to filter to decode-loop samples and aggregate by leaf frame.
Output goes to `results/profile-${HOST}-${DIST}-xctrace-${TS}.txt`.

10 s wall window → 9996 decode-loop samples × 1 ms.  Parsed
summary:
[`../results/profile-m4_max-prose_pride-xctrace-20260426-0625.txt`](../results/profile-m4_max-prose_pride-xctrace-20260426-0625.txt).

## Per-function self-time (decode loop only, % of 9996 samples)

xctrace's DWARF inlined-frame attribution is what makes this
breakdown trustworthy: the leaf frame is the source-level innermost
function the IP belongs to, even when that function was inlined
into `decode_node_neon` at `-O2`.

| Function                   | %       | Source location        | Description                                       |
|----------------------------|--------:|------------------------|---------------------------------------------------|
| `partition_8`              | **37.9%** | `pivco_huffman_neon.c:88+`  | 2-way partition core (TBL + store)              |
| `flat_decode_scatter_neon` | **16.2%** | `pivco_huffman_neon.c:230+` | flat-subtree TBL + indexed store                |
| `decode_node_neon`         | **11.8%** | `pivco_huffman_neon.c:754+` | recursion glue + leaf checks + recurse setup    |
| `scatter_both_leaves`      |    9.9% | `pivco_huffman_neon.c:704+` | both-leaves stage fusion (sequential write)     |
| `scatter_sym`              |    8.6% | `pivco_huffman_neon.c:660+` | leaf scatter (one child = leaf)                 |
| `flat_d3_unpack`           |    4.1% | `pivco_huffman_neon_flat.h:71` | D=3 bit-unpack inside flat path                 |
| `flat_d2_unpack`           |    3.9% | `pivco_huffman_neon_flat.h:44` | D=2 bit-unpack                                   |
| `partition_8_right`        |    3.7% | `pivco_huffman_neon.c:638+` | half-partition (one side store, leaf-fusion)    |
| `_platform_memset`         |    3.2% | (libsystem)            | phase-0 `prefill_sym` of most-frequent leaf       |
| `pivco_huffman_decode_neon`|    0.5% | `pivco_huffman_neon.c:1061` | per-block wrapper (root partition setup)        |
| `bitmap_get` / `extract_D_bits` / `bitmap_bytes` | 0.3% | `pivco_huffman_common.h` | scalar tail / fallback paths     |

Top single source line: `partition_8` at `pivco_huffman_neon.c:98`
(the second `vst1q_u8` storing the left-partition output) — **28.5%
of all CPU time** alone.  The first `vst1q_u8` (line 95, popcnt
load + first store) takes another 9.4%.  Together those two stores
in `partition_8` account for **38% of total** — the actual TBL
shuffle and bitmap loads scarcely show up.  Consistent with M4's
partition microbench cost (0.06 ns/elem at ~15.5 GB/s) being store-
port bound, not TBL-throughput bound.

## Aggregated by source region (% of total CPU)

| Region                         | %        | Comprises                                                |
|--------------------------------|---------:|----------------------------------------------------------|
| **Partition body**             | **41.6%** | `partition_8` + `partition_8_right`                      |
| **Flat-subtree path**          | **24.2%** | `flat_decode_scatter_neon` + `flat_d2_unpack` + `flat_d3_unpack` |
| **Leaf scatter**               | **18.4%** | `scatter_sym` + `scatter_both_leaves`                    |
| **Recursion glue**             | **11.8%** | `decode_node_neon` (non-inlined: leaf checks, recurse)   |
| **Phase-0 prefill**            |     3.2% | `_platform_memset`                                        |
| **Per-block frame + scalar tail** |  0.8% | `pivco_huffman_decode_neon`, `bitmap_get`, …             |

(Total 100.0%.)

## Comparison to the old profile

The previous profile (zipfian on the pre-flat-subtree code, hand-
mapped) reported the breakdown below.  Side-by-side with the
xctrace numbers on prose_pride:

| Region                  | Old (zipfian, hand-mapped) | New (prose_pride, xctrace) | Note                                                  |
|-------------------------|---------------------------:|---------------------------:|-------------------------------------------------------|
| Partition body          |                     44.4% |                  **41.6%** | Algorithm unchanged; close match validates new tooling |
| Flat-subtree path       |                       0%  |                  **24.2%** | New region — fast path didn't exist before            |
| Leaf scatter            |                     12.3% |                  **18.4%** | Real cost was higher than previously credited; the old hand-mapping under-attributed because `sample` collapsed leaf-scatter offsets with adjacent partition offsets |
| Recursion glue          |                     14.1% (function prologue) |        **11.8%** | OoO still hides the prologue; the rest is leaf-checks and recurse-setup |
| Frame entry/epilogue    |                       —   |                     <1%   | Confirmed negligible                                   |
| Phase-0 prefill         |                       —   |                     3.2%  | Wasn't called out in old profile                      |

Two things to read here: (1) the partition-body share is essentially
unchanged from the old profile — that algorithm hasn't changed —
which is a sanity check that the new tooling is producing
believable numbers, and (2) the "regional breakdown" I did one
commit ago by hand-mapping `sample` offsets was directionally right
but quantitatively off (over-attributing partition body by ~15 pp,
under-attributing leaf scatter and recursion glue).  The xctrace
numbers above replace it.

## Profiling lesson (preserved from earlier)

"Occupies X% of execution slots" is not the same as "removing it
would be X% faster."  On an OoO core, non-critical-path work is
essentially free if it doesn't compete for the bottleneck resource
— which on the partition path is **NEON store-port throughput**
(per the line-level data: 38% of total time is the two `vst1q_u8`
stores in `partition_8`), not TBL latency or instruction issue.
