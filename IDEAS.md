# PIVCO-Huffman Ideas Log

> Curated 2026-06-13.  Each entry: status, date, one-paragraph essence.
> Detailed write-ups live in `docs/`, `results/`, and the cited commits; pre-curation history in git.

## Table of Contents

### CONSIDERED — open / parked / research direction

**General**
- [Cross-port post-June-5 x86 optimizations to NEON](#cross-port-post-june-5-x86-optimizations-to-neon-2026-06-13)
- [SIMD-ify scalar tails (overwrite or masked stores)](#simd-ify-scalar-tails-overwrite-or-masked-stores-2026-06-15)
- [Golomb/Rice tier for very-high-skew nodes (p > 0.95)](#golombrice-tier-for-very-high-skew-nodes-p--095-2026-05-16-research)
- [pivcohuf block-structured file format + tiny-input header](#pivcohuf-block-structured-file-format--tiny-input-header-2026-05-17-parked)
- [v0.2 FSE marker byte unconditional overhead](#v02-fse-marker-byte-unconditional-overhead-2026-05-13)
- [Tree-walk node-size histogram → tiny-node fast paths](#tree-walk-node-size-histogram--tiny-node-fast-paths)
- [FastLanes-style transposed bitpacking](#fastlanes-style-transposed-bitpacking-2026-04-27)
- [LSB-first canonical codes](#lsb-first-canonical-codes-2026-05-12)

**NEON**
(no open entries)

**SSE/AVX2**
- [SSE/AVX2 enc_node_full gap vs huf0 on text](#sseavx2-enc_node_full-gap-vs-huf0-on-text-2026-05-12-partial)
- [SSE/AVX2 D=7 flat-subtree path](#sseavx2-d7-flat-subtree-path)
- [x86 COM merge (prefix-sum cursor decoupling)](#x86-com-merge-prefix-sum-cursor-decoupling-2026-06-16)

**AVX-512**
- [AVX-512 BW tier for Cascade Lake (c5)](#avx-512-bw-tier-for-cascade-lake-c5-2026-05-12)

**Other ISA**
- [RISC-V RVV 1.0 backend](#risc-v-rvv-10-backend-speculative)

### DONE — shipped

**General**
- [Unify-framework refactor (5 phases)](#unify-framework-refactor-2026-05-14)
- [encode_node infinite recursion (stack OOB write) fix](#encode_node-infinite-recursion-stack-oob-write-fix-2026-05-13)
- [Flat-aware Huffman tree restructurer](#flat-aware-huffman-tree-restructurer-2026-04-25)
- [Drop within-tier freq sort](#drop-within-tier-freq-sort)
- [Flat-subtree fast path (format change)](#flat-subtree-fast-path-format-change-2026-04-24)
- [Full-tail masked partition](#full-tail-masked-partition-2026-05-08)
- [BU stored K_right in wire format](#bu-stored-k_right-in-wire-format-2026-05-12)
- [FSE wide-cursor decoder (x8y1) productionized](#fse-wide-cursor-decoder-x8y1-productionized-2026-05-15)
- [LZ4 + ph decode-speed probe](#lz4--ph-decode-speed-probe-2026-05-17-probe-done)
- [Oodle huff6 reference + ARM ASM kernels integrated](#oodle-huff6-reference--arm-asm-kernels-integrated-2026-05-15)
- [Arbitrary-sized blocks (variable-N codec)](#arbitrary-sized-blocks-variable-n-codec-2026-06-15)
- [Profiling overhead caveat (documented)](#profiling-overhead-caveat-2026-05-11)

**NEON**
- [BU tree_merge 2× unroll + SIMD popcount_K_right](#bu-tree_merge-2-unroll--simd-popcount_k_right-2026-05-10)
- [NEON bu_tree_merge 16B loads + precomputed (nr0,m1) shuf](#neon-bu_tree_merge-16b-loads--precomputed-nr0m1-shuf-2026-05-11)
- [Graviton4 NEON D=5/6 unpack store-forward stall fix](#graviton4-neon-d56-unpack-store-forward-stall-fix-2026-04-27)
- [Graviton4 NEON D=5/6 vqtbl{2,4} regression gate](#graviton4-neon-d56-vqtbl24-regression-gate-2026-04-25)
- [NEON flat_dN_unpack single vld + overread guarantee](#neon-flat_dn_unpack-single-vld--overread-guarantee-2026-06-16)
- [Graviton4 (Neoverse V2) flat-unpack weakness, resolved](#graviton4-neoverse-v2-flat-unpack-weakness-2026-05-26-resolved-2026-06-16)

**SSE/AVX2**
- [SSE root both-leaves vectorisation](#sse-root-both-leaves-vectorisation-2026-04-27)
- [x86 AVX2 pack_d{2,3,5,6,7} via ryg multiply-as-shift](#x86-avx2-pack_d23567-via-ryg-multiply-as-shift-2026-06)
- [x86 partition 2x-unrolled stride-16](#x86-partition-2x-unrolled-stride-16-2026-06)
- [SSE/AVX2 flat unpack via ryg multiply-as-shift](#sseavx2-flat-unpack-via-ryg-multiply-as-shift-2026-06-15)

**AVX-512**
- [AVX-512 leaf-fusion port](#avx-512-leaf-fusion-port-2026-04-27)
- [AVX-512 small-node tail (masked partition)](#avx-512-small-node-tail-masked-partition-2026-05-07)
- [AVX-512 D=3/5/6 flat-subtree TBL](#avx-512-d356-flat-subtree-tbl-2026-04-24)
- [AVX-512 enc_init via vpermi2w + vpermex2var_epi8](#avx-512-enc_init-via-vpermi2w--vpermex2var_epi8-2026-05-12)
- [AVX-512 pack_dN multishift, 64 codes/iter](#avx-512-pack_dn-multishift-64-codesiter-2026-06)

**NEON + AVX-512**
- [NEON + AVX-512 D=7 flat decode](#neon--avx-512-d7-flat-decode-2026-05)

### REJECTED — tried, lost or discarded

**General**
- [Hybrid block decoder (deliberately not pursued)](#hybrid-block-decoder-deliberately-not-pursued)
- [Entropy-skew flat-subtree split (per-block real-freq)](#entropy-skew-flat-subtree-split-per-block-real-freq-2026-05-13)
- [FSE-decode ↔ merge fusion (microbench win, integration regression)](#fse-decode--merge-fusion-2026-05-23)
- [Bitpacking libraries survey (simdcomp / FastPFor)](#bitpacking-libraries-survey-simdcomp--fastpfor-2026-05-11)
- [PIVCO-Huffman X2 (two-cursor)](#pivco-huffman-x2-two-cursor-2026-05-07)
- [Scatter fusion into partition loop body (conditional stores)](#scatter-fusion-into-partition-loop-body-conditional-stores)
- [Relaxed / near-flat subtree detection](#relaxed--near-flat-subtree-detection-2026-05-09)
- [Iterative DFS instead of recursion](#iterative-dfs-instead-of-recursion)
- [Cross-platform partition+scatter block-level fusion](#cross-platform-partitionscatter-block-level-fusion-2026-05-09)
- [SIMD/GPU Huffman literature survey](#simdgpu-huffman-literature-survey-2026-05-12)
- [EWAH word-aligned hybrid bitmap](#ewah-word-aligned-hybrid-bitmap-2026-05-16)

**NEON**
- [Root fusion (init + root encode)](#root-fusion-init--root-encode-2026-05-11)
- [neon2 4-way fused decode](#neon2-4-way-fused-decode)
- [Interleave 16-elem partition pair in decode](#interleave-16-elem-partition-pair-in-decode)
- [Mix str d + str q to use 2nd store AGU on M-series](#mix-str-d--str-q-to-use-2nd-store-agu-on-m-series)
- [Coalesce small-side partition stores](#coalesce-small-side-partition-stores)
- [iota-table for partition_root_8](#iota-table-for-partition_root_8-2026-04-26)
- [Wider NEON partition (2 TBLs per 16 indices)](#wider-neon-partition-2-tbls-per-16-indices)
- [builtin popcount instead of compress_popcnt table](#builtin-popcount-instead-of-compress_popcnt-table)
- [prim_merge popcount via pre-pass / in-register vcnt](#prim_merge-popcount-via-pre-pass--in-register-vcnt-2026-05-29)

**SSE/AVX2**
- [AVX2 partition_16 stride for enc_node_full](#avx2-partition_16-stride-for-enc_node_full-2026-05-12)
- [SSE flat_decode_direct mostly scalar for D ∈ {2,3,5,6} (TD-only)](#sse-flat_decode_direct-mostly-scalar-for-d--2356-td-only)
- [Revisit SSE4.1 D=2,3,5,6 flat-subtree TBL (TD-only)](#revisit-sse41-d2356-flat-subtree-tbl-td-only)
- [SSE D=5..8 unpack — gcc unroll pragma (TD-only)](#sse-d58-unpack--gcc-unroll-pragma-td-only)

**AVX-512**
- [AVX-512 root iota](#avx-512-root-iota-2026-04-27)
- [AVX-512 byte-scatter (vpscatterdd / vpermb) (TD-only)](#avx-512-byte-scatter-vpscatterdd--vpermb-td-only-2026-05-09)
- [PDEP / vpexpandw per-position code reconstruction (TD-only)](#pdep--vpexpandw-per-position-code-reconstruction-td-only-2026-05-10)

**Other ISA**
- [SVE 128-bit and SVE 256-bit](#sve-128-bit-and-sve-256-bit-2026-05-28)

---

## CONSIDERED

**General**

### Cross-port post-June-5 x86 optimizations to NEON, 2026-06-13
Recent x86 wins targeted SSE/AVX2 + AVX-512 only: AVX-512 pack_d{2..7} multishift 64 codes/iter (`0c80e3a`), AVX-512 merge_flat_d{2..7} widened to 64 codes/iter (`e5a199a`), x86 AVX2 pack_d{2,3,5,6,7} via ryg multiply-as-shift (`a1aa6b9`), x86 partition 2x-unrolled stride-16 (`83e23a0`).  Open question on NEON (M4 + Graviton 4): is the analogous primitive already at width ceiling, or is there headroom?  Multishift has no NEON analogue (x86-only), and the multiply-as-shift trick is irrelevant on NEON (`vshlq_u32` is native).  But the merge_flat widening pattern and the 2x partition unroll may transfer.  Action: per-primitive bench (`bench_prim`) before/after a NEON 64-codes/iter sketch + fair_bench A/B on c8g.

### SIMD-ify scalar tails (overwrite or masked stores), 2026-06-15
Most primitives (partition, merge, pack, unpack) drop to scalar for the final 1..K-1 elements where K is the SIMD stride (8/16/32 depending on primitive).  At small block sizes (deep recursion or short final block) those tails dominate.  Two general approaches: (a) **overwrite tails** — round n up to the next stride boundary, write valid+garbage past the real n, then truncate via cursor adjustment (already done in some AVX-512 paths; works when downstream readers know n).  (b) **masked SIMD** — `vmaskmovps`/AVX-512 `k` masks / NEON's `vbslq_u8` on a tail-mask vector to do exactly n elements per SIMD op.  AVX-512 full-tail masked partition shipped 2026-05-08 (+23% on c8i); same pattern hasn't been tried on NEON/SSE/AVX2 tails of merge / pack / unpack / scatter primitives.

### Golomb/Rice tier for very-high-skew nodes (p > 0.95), 2026-05-16, research
Bench (`bench_golomb.c`): Rice loses to FSE at p ≤ 0.95 but wins at p ≥ 0.97 — **p=0.99: Rice 6.3 GB/s vs FSE 2.0 GB/s, AND 25% smaller** (74 B vs 99 B, since static FSE tables don't extrapolate past ~0.94).  Crossover ~0.97.  Verdict: ship Rice as a third tier only for the extreme-skew tail (FSE marker `0=raw / 1=Rice / 2=FSE`), not as a general FSE replacement.  Encode is bit-by-bit scalar (not pivco's hot path).

### pivcohuf block-structured file format + tiny-input header, 2026-05-17, parked
Two design pieces worked out in detail: (a) replace the monolithic file with independent 4 MB blocks (own header + Huffman table) — enables streaming, parallel decode, range queries, crash recovery; per-block overhead is in the noise (≤ 0.05% at 4 MB+); (b) sparse-alphabet + narrow-code-length encoding shrinks tiny-input expansion from ~29× to ~3× on 7-byte inputs.  Park until LZ4+ph prototype starts — block boundaries naturally line up with the LZ window then.  pivcohuf is an internal experiment harness, not a real-world compressor.

### v0.2 FSE marker byte unconditional overhead, 2026-05-13
Every non-flat internal node emits a 1 B marker (`0x00 raw / 0x01..0x19 FSE table id / 0x80|id XOR flip`) even with FSE off.  Overhead: proba80 0.06% / prose_pride 0.6% / chinese_text 0.9% / cat-image 1.6%.  Three options to recover, all wire-format-affecting (file-flag gate, conditional by bitmap size, or 1-bit/node packed vector at block start).  Sub-1% overall, parked.

### Tree-walk node-size histogram → tiny-node fast paths
Codex item.  Recursive decoder calls `decode_node_*` per internal node; small-n nodes may have per-call overhead > partition work.  Instrumentation helper landed (`e54e2d5`), histogram analysis + tiny-node fast paths (`n ≤ 8 / ≤ 16`) still open.  Risk of skipping the analysis: dispatcher overhead wipes per-call savings, net negative on real text.

### FastLanes-style transposed bitpacking, 2026-04-27
Microbench (`bench_unpack_fl_layout.c`) shows FL-layout 2–22× faster than current `flat_dN_unpack` (G4 D=5/6 the extreme case).  End-to-end ceiling on real text ~5–6% (flat unpack is only ~7% of wall on prose_pride).  Costs: encoder rewrite (transpose bit-packing), 4 backends × 2 unpack styles, wire-format bump.  Bigger fish (partition 40% / scatter 18%) untouched by FL-layout.  See `docs/BITPACKING.md`.

### LSB-first canonical codes, 2026-05-12
Bit-reversing canonical codes (root at bit 0) would let `code` replace `code_la` (-512 B table) but does NOT improve any hot operation: u8 subtree repack and partition kernels are symmetric.  Multi-day rewrite of every SIMD partition kernel for a marginal cleanup — not justified.  Logged so the question doesn't get re-derived.

**SSE/AVX2**

### SSE/AVX2 enc_node_full gap vs huf0 on text, 2026-05-12, partial
Older Intel (c3/c4/c5) and Zen 3 (c6a) trailed huf0_x2 by ~25–30% on text encode.  Partial progress: AVX2 pack_d{2,3,5,6,7} shipped (`a1aa6b9`, ryg multiply-as-shift) + 2x partition unroll (`83e23a0`).  Remaining angle: AVX2 partition_16 via two pshufb + concat, or 4-stream FastLanes pack (format change).  See rejected "AVX2 partition_16 stride" for the failed 2x unroll attempt.

### SSE/AVX2 D=7 flat-subtree path
NEON and AVX-512 D=7 BU `merge_flat_d7_*` shipped (`7e4bc44`, `dcfaecc`); SSE/AVX2 still has no D=7 SIMD because pshufb is only 16-wide and the 128-entry c2s doesn't fit (see comment in `pivco_huffman_primitives_x86.h:536`).  Coverage: bell_s80 0%, zipfian 11.2%, flat_M7 already 1.43× vs huf0 on the SSE host.  Expected end-to-end win <2% on zipfian.  Low marginal EV — parked.

### x86 COM merge (prefix-sum cursor decoupling), 2026-06-16
The NEON COM64 merge (`5cccccc`) decouples per-chunk cursors via a popcount
prefix sum, breaking the `expand_tab_pre[nr0][m1]` cursor-dependent table
load.  Tried porting the same to the x86 SSE/AVX2 merge (8 independent
8-code pshufb chunks per 64 codes; SWAR or pshufb bytewise popcount; also
a 128-wide and an AVX2 two-lane variant).  Bench harness lives in
`extras/bench/bench_merge_x86.c`.  Microbench looked like a -9..-18% win,
but **production fair_bench (SSE/AVX2 tier, clang-20) regresses on every
Intel host (c3/c4/c5, -2..-11% dec_pb), wins only on Zen 2 (c5a), wash on
Zen 3 (c6a)**; the microbench is also compiler-fragile (gcc vs clang flip
the Intel sign).  Root cause: unlike NEON, the x86 baseline merge has no
cursor-dependent table load — it's flat pshufb + 1-cycle cursor adds — so
COM only adds code + a prefix-sum dependency the older Intel frontends
don't absorb.  vpexpandb (VBMI2 hosts) is ~3× faster than any COM form, so
those hosts are unaffected regardless.  The production merge logic is the
standalone `sse_com*` functions in the bench (drop-in for the three
`merge_*_x86` main loops).  Open: a vendor-gated `sse_com128` for the AMD
non-VBMI2 tier (Zen 2/3) is the only angle that might pay off; needs a
production A/B on c5a/c6a before it's worth a dispatch split.

**AVX-512**

### AVX-512 BW tier for Cascade Lake (c5), 2026-05-12
c5 (Xeon 8275CL) has F/BW/CD/DQ/VL but no VBMI2; today drops to AVX2 tier.  Could shim a new tier that uses vpermi2w for enc_init (BW) while keeping AVX2 partition + SSE pack.  Estimated ~10–12% wall on c5, prose_pride 691 → ~775 M/s, crosses huf0_x2 parity.  Defer unless c5 specifically becomes important.

**Other ISA**

### RISC-V RVV 1.0 backend, speculative
RVV maps well: `vcompress` = partition, `vrgather` = TBL, **`vsuxei8` = native indexed byte scatter** (the only ISA-level asymmetry — collapses today's per-leaf scatter from 8 STRBs to 1 instruction).  Hold off until AWS/GCP offers RISC-V or a competitive chip (3+ GHz, 256-bit VLEN) lands.  Cheap dev: QEMU emulation ($0 correctness), Banana Pi BPI-F3 (~$100 bench).  Cross-asymmetric data point: +10–15% on real-text from `vsuxei8` alone would be the strongest evidence the algorithm is ISA-portable.

---

## DONE

**General**

### Unify-framework refactor, 2026-05-14
All 4 backends now share `src/pivco_huffman_codec.c` (compiled once per backend as an OBJECT library); SIMD lives only in `primitives_<backend>.h`.  Net -3036/+1867 LoC across the codec source.  Surfaced + fixed the wire-format drift bug (FSE marker byte missing from legacy AVX-512 encoder).  Adding a 5th backend is now a primitives header + a CMake entry.  Commits: `2429c80`, `91bea73`, `5d85874`, `3c5ecf8`.  Sweep: `results/SUMMARY-20260514-unify-framework.md`.

### encode_node infinite recursion (stack OOB write) fix, 2026-05-13
Looked like infinite recursion on near-uniform random; actual cause was `tmp[PIVCO_BLOCK_SIZE * 2]` scratch sized for balanced trees, overflowing on skewed-tree paths (depth × N elements vs 2N).  Stack OOB write clobbered the caller's table struct.  Fix: scratch buffers sized for worst-case `(PIVCO_MAX_CODE_LEN + 2) × BLOCK_SIZE` in every backend.  Test coverage in `test/test_edge_cases.c`.

### Flat-aware Huffman tree restructurer, 2026-04-25
At `build_table` time, rearrange leaves within each code-length tier to maximise flat-D≥2 subtree coverage — same code-length multiset, identical compression, no wire change.  Algorithm: per length L, decompose `c_L` by binary representation, place highest-freq symbols in largest-D chunks.  Parity-cross flips: proba14 on M4 0.91×→1.10×, proba14 on Xeon 0.67×→1.07×, proba02 on Graviton 0.92×→1.04×.  Cross-platform gains 6–60% on real text.  Analyzer: `extras/bench/bench_flat_optimal.c`.

### Flat-subtree fast path (format change), 2026-04-24
Encoder detects every maximal internal node whose subtree is flat with depth D ≥ 2; emits one N×D-bit packed region instead of D levels of bitmaps.  Decoder uses direct `code_to_sym[local_code]` lookup + scatter (same mechanism as full-tree flat path).  Coverage: bell_s80 100% / proba02 98.4% / bell_s30 95.9% / bell_s10 94.4% / zipfian 69.4% / english 54.8%.  Sweep: `results/20260424-204720-0a92fe3-flat-subtree-sweep.md`.  Commits `a275d05` → `7c3238b`.

### Drop within-tier freq sort
Encoder used to sort within each code-length tier by real freq descending so the heaviest symbols landed in the largest flat chunk.  That was the only thing making the tree depend on within-tier freq order, which forced rank info onto the wire (v0.3 ORDERING + `rank_within_tier`) so the decoder could reproduce it.  Dropped: now sorts by symbol value asc — deterministic from code lengths alone, no rank info transmitted, no-op on ratio.  See `src/huffman_table.c:399-409`.

### Full-tail masked partition, 2026-05-08
Replaced scalar tails in `partition_*` with a masked vector partition.  Bug rediscovery: RIGHT recursion shares `buf2` between `indices` and `tmp` with no gap; vector tail's filler zeros clobbered just-written right-side data.  Fix: caller passes right child's tmp at `tmp + n_right + 8` (NEON/SSE) or `+ 32` (AVX-512) — one full vector-width of padding.  Wins: c5 +7.3%, c8i **+23.2%**, c8a **+24.4%** real-text avg; AVX-512 gains huge because partition_32 masks 1..31-element tails.  Two earlier fix attempts (scratch+memcpy, hybrid vector/scalar) lost on M4 and were dropped.  Verified by `pivco_huffman_tests` (4f2fd5c) + scalar-ref cksum (da3c9fd).

### BU stored K_right in wire format, 2026-05-12
2 B little-endian K_right uint16 written immediately before the bitmap at every non-flat internal node feeding a non-leaf child.  BU decoder reads it directly instead of popcount_K_right.  Cross-platform decode win: **c8i +57% / c8a +41% / c4 +53% / c8g +22% / M4 +0%** (popcount was already cheap on M4).  Storage cost 0.32–0.53% on text (vs 1.2% upper bound; `kr_header_needed` gating skips leaves / both-leaves / flat terminals).  Inline placement beat sidecar by 10–20% on x86 (L1 prefetch pulls K_right with the bitmap).  Commits `5828ddb`, `231bcac`, `bf68feb`.

### FSE wide-cursor decoder (x8y1) productionized, 2026-05-15
Stock FSE x=2 was 2.5–3× too slow vs huf4X2 on per-node bitmaps.  Microbench (`bench_fse_xy_micro.c`) showed x=8..16 cursors close the gap to within 0–28% of huf4X2 across hosts.  Shipped `decode_x8_y1` in `src/pivco_fse.c` (commit `2a5cfff`, +1.5–1.8× per memory).  ph's 1-byte-table-id wire format is decisive at small bitmap sizes (2–5× vs huf4X2 / Oodle huff6); Oodle wins past ~16 KB.  See `docs/FSE-V0.md`, `results/fse_xy_micro-allhosts-20260515-*`, and memory entry [project_fse_wide_cursor_2026_05_15].

### LZ4 + ph decode-speed probe, 2026-05-17, probe done
zstd is structurally LZ4 + huf0 + FSE; substituting ph for huf0 was the hypothesis.  Compression-side: LZ4 flattens literal histograms (86–94% Huffman ratio band, uninteresting).  Decode-speed probe ran (`bb55760 results: ph vs huf0_x2 on LZ4 residuals`).  Further exploration: `dfe5a46 bench_lz4_split`, `35be64c LZ4-split RAW`, `e0c1b22 LZ4-split 1.5–2× decode`, `ad9b037 bench_lz4_ph encode-speed`.  Probe complete; full LZ4 + ph integration not pursued.

### Oodle huff6 reference + ARM ASM kernels integrated, 2026-05-15
ph optionally links OodleUE via `ext/oodle` symlink (don't rsync — see CLAUDE.md).  Cross-platform sweep at 1440 B / pmaj=0.80: Oodle huff6 ASM 750–1458 MB/s vs FSE x*y 1172–2300 vs huf4X2 full 227–439 (with per-call setup).  Crossover at ~16 KB: above this, Oodle's amortized ASM wins.  Validates ph's framing: the 1-byte-table-id wire format (static FSE tables) is the architectural feature that wins the small-bitmap regime by 2–5×.  EULA reviewed 2026-05-15: bench + publish OK; vendoring not (hence the symlink-as-clone pattern).  Results in `results/fse_xy_full-*-20260515-*`.

### Arbitrary-sized blocks (variable-N codec), 2026-06-15
The codec was hardcoded to BLK (8192/4096 on ARM-AVX-512/x86), and `pivcohuf_file.c:233` padded short final blocks to full BLK with `prefill_sym` — a 7-byte input did ~4089 bytes of extra encoder work.  Shipped a small wire-format addition: a uint16 `N` prefix at the start of each encoded block (helpers in `pivco_huffman_wire.h`).  `pivco_huffman_encode` gains an explicit `size_t n` parameter; decode reads N from the wire.  Codec entries (`pivco_huffman_codec.c:243`/`:440`) take N at runtime; primitives untouched (they already accept any K with tails internal).  File codec drops the padding path entirely.  Bloat: 2 bytes per block (0.05% at x86 BLK, 0.024% at ARM/AVX-512 BLK).  7-byte input now encodes to 177 B (was ~206 B); more importantly, encoder work is proportional to N, not BLK.  Enables future LZ4+ph residual-stream handling.

### Profiling overhead caveat, 2026-05-11
Documented, not a code change.  `PROF_TIC`/`TOC` overhead varies 60× across platforms: M4 ~0%, Zen 3 +11%, **c8i Granite Rapids +61.6%** on prose_pride (Granite Rapids RDTSC is the most serializing of any modern core).  Microbench predicts well except on aggressive-OOO cores (c8a 0.2× pred, c8g 0.25×, M4 0×).  Rules going forward: bench numbers from `pivco_huffman_bench` (no `-DPIVCO_PROF=1`) are the ONLY authoritative headline; prof output is RELATIVE only.  Subtract ~25–30 ns/call on c8i, ~15 ns on older Intel, ~10 ns on Zen 3 when comparing primitive ns/call.

**NEON**

### BU tree_merge 2× unroll + SIMD popcount_K_right, 2026-05-10
First round of BU optimisation after the new BU decoder landed.  (1) Stride-16 unroll on `tree_merge` and `tree_merge_bcast_{left,right}` (TD had it for ages, BU was stride-8): -7% per call.  (2) SIMD `popcount_K_right` with 4-wide ILP — `bu_popcount_K` from 57 → 7.5 ns/call (7.6×), 29% → 5% of wall.  BU/TD ratio went from ~0.94× to **1.50×** on M4 prose_pride.  Commits `8f03fac`, `462db14`, `a962789`.

### NEON bu_tree_merge 16B loads + precomputed (nr0,m1) shuf, 2026-05-11
Stride-16 main loop: 2× 16-byte L_full/R_full loads (vs 4× 8-byte before), iter 1 uses `vqtbl2_u8` over the full 32-byte source with a precomputed adjusted shuf indexed by `(nr0, m1)` from `expand_tab_pre[9][256][8]` (18 KB, fits L1d).  M4 prose_pride +13.5%, G4 +10.1%.  Bu_tree_merge ns/call 176 → 145 (-18%), BU total -12%.  Earlier V2 (runtime ALU adjust) and V3 (`vextq` switch — DEAD, -77% on real text, branch misprediction) tried and dropped.

### Graviton4 NEON D=5/6 unpack store-forward stall fix, 2026-04-27
The old `memcpy(&packed, bm_ptr, 5/6) + vsetq_lane_u64` compiled to a stack round-trip on G4 → store-forward stall every iter, ~5× throughput loss.  Replaced with byte-wise `vsetq_lane_u8 × N` (same as D=3): G4 D=5 1.3 → 5.8 GB/s, D=6 1.3 → 5.2 GB/s.  With unpack 5× faster, re-enabled SIMD in the direct path (scatter path stays gated to scalar — smaller n).  flat_M5 +59.8%, flat_M6 +64.1%.  Write-up: `results/G4_D5D6_FIX-AB-20260427.md`.

### Graviton4 NEON D=5/6 vqtbl{2,4} regression gate, 2026-04-25
Build-time gate `PIVCO_NEON_FAST_MULTI_TBL` (default 1 on `__APPLE__`, 0 elsewhere) falls D=5/6 through to scalar in `flat_decode_{scatter,direct}_neon`.  Bell_s80 537 → 1105 M/s (+106%), flat_M5 1282 → 3187 (+148%), flat_M6 1194 → 2458 (+106%).  Win count on c8g: 10/19 → 13/19.  Rejected variant (`2× vqtbl1` + sub + blend): even slower than the original regressed path.  Methodology note: re-baselined on c8g.large (dedicated) before diagnosing — c8g.medium burstable had real CPU-steal variance.  Commit `cee2366`; sweep `results/20260425-0126-cee2366-graviton-d56-fix.md`.

### NEON flat_dN_unpack single vld + overread guarantee, 2026-06-16
Replaced the byte-wise `vsetq_lane_u8 × N` chain in `flat_d{3,5,6,7}_unpack` with a single 16-byte `vld1q_u8` register load.  Three-tier merge loop in callers bounds the fast region (`fast_end = n - K`) and falls through to a `_safe` byte-wise variant for the ≤8-code tail.  Same fast/safe pattern already used by the AVX-512 flat unpacks.  The original byte-wise load (`cee2366`) was the workaround for a V2 store-forward stall on the prior `memcpy + vsetq_lane_u64` form; a direct vld register load doesn't go through the stack at all.  Per-primitive isolation (perf_event_open CPU_CYCLES on G4, CLOCK_MONOTONIC ns on M4, 3-round median):
- G4 cyc/elem: D=3 0.428→0.246 (−43%), D=5 0.367→0.246 (−33%), D=6 0.422→0.247 (−42%), D=7 0.477→0.245 (−49%).
- M4 ns/elem: D=3 0.048→0.039 (−19%), D=5 0.048→0.039 (−19%), D=6 0.047→0.039 (−17%), D=7 0.054→0.038 (−30%).

ryg multiply-as-shift (`vmulq_u16 + vshrq_n_u16`) was tested on top of the single-load form and measured neutral on both hosts (G4 cycles within ±1%, M4 ns within ±5%), so the variable-shift + mask form was kept.  Commit `d856bcc`.

### Graviton4 (Neoverse V2) flat-unpack weakness, 2026-05-26, resolved 2026-06-16
Original observation: G4 flat-decode unpack 3–5.5× slower than M4 for odd/high D (D=3 0.143 vs 0.042 ns/elem; D=7 0.236 vs 0.043).  Diagnosis ascribed it to NEON variable shifts + `vqtbl` cross-byte field repositioning being V2-narrow-pipe-bound, suggesting a V2-tuned unpack with more immediate `ushr`/`sli`.  That fix was never tried; the actual dominant cost turned out to be the byte-wise `vsetq_lane_u8 × N` load chain (originally added in `cee2366` to dodge a different V2 pathology — store-forward stall on the prior `memcpy + vsetq_lane_u64`).  Replacing that with a single `vld1q_u8` register load (`d856bcc`) brought G4 unpack to ~0.087 ns/elem uniformly across D=3/5/6/7 — D=3 −39%, D=7 −63%.  Residual gap to M4 (~0.039 ns/elem) is now ~2.2×, uniform across D, attributable to V2's narrower vector pipe width; no further uarch-specific tuning planned.

**SSE/AVX2**

### SSE root both-leaves vectorisation, 2026-04-27
Scalar byte-by-byte loop replaced with 16-output-bytes-per-iter SSE4.1 (`pshufb` broadcast + `pcmpeqb` mask + `pblendvb` select).  two_sym_eq +1405% / two_sym_90/10 +1409% on Zen 3.  Bonus codegen wins on uniform/gzip_random (+71/73%, parent function inlining changes).  Write-up: `results/SSE_BOTH_LEAVES-AB-20260427.md`.

### x86 AVX2 pack_d{2,3,5,6,7} via ryg multiply-as-shift, 2026-06
AVX2 pack helpers shipped in commit `a1aa6b9` using the multiply-as-shift trick from ryg / FastPFor.  Plus follow-on bugfix `532cb74` (pack_d8 saturating-pack mask).

### x86 partition 2x-unrolled stride-16, 2026-06
2x unroll of the SSE/AVX2 partition inner loop in `enc_node_full`.  Commit `83e23a0`.  Small encoder gain on text dists on c5 (Cascade Lake) + c6a (Zen 3).

### SSE/AVX2 flat unpack via ryg multiply-as-shift, 2026-06-15
Replaced the AVX2 `vpsrlvd`-based `flat_d{2,3,5,6}_unpack` with ryg's PSHUFB + PMULLO_EPI16 + PSRLI + PAND pattern, plus a 3-op SSE2 trick for D=4 (`srli_epi16` + `unpacklo_epi8` + `and`).  Works on SSE4.1+ — drops the `#ifdef PIVCO_HAS_AVX2` gate that previously left c3 (Ivy Bridge) on a scalar fallback for D=2/3/5/6.  Idea via email from Fabian Giesen (ryg), pattern adapted from his Oodle BC7 `simd_multigetbits` extractor.  Primitive-level wins (bench_prim `merge_flat sse/avx2` median across 3 rounds): c3 D=2 -72% / D=3 -65% / D=5 -59% / D=6 -40%; c5a Zen 2 D=2 -39% / D=3 -29% / D=5 -40%; c6a Zen 3 D=5 -29% / D=6 -19%.  End-to-end fair_bench A/B on c3 dec_pb (3-round medians): english **+29%**, html_wiki **+23%**, chinese_text **+23%**, json_api +16%, prose_pride +14%, image_jpeg +11%.  Other hosts smaller (the pshufb-based c2s scatter dominates merge_flat there).

**AVX-512**

### AVX-512 leaf-fusion port, 2026-04-27
`decode_node_avx512` was missing the stage-fusion logic (both-leaves dispatch + half-partition for prefilled-leaf side) that NEON and SSE had.  Helpers existed but weren't called.  Port added `scatter_both_leaves_avx512` + three early-return branches.  9 wins p<0.05 (source_c +7.5%, english +6.1%, html_wiki +4.1%, etc.), 0 losses p<0.05.  Same session added `PIVCO_BENCH_QUICK` for fast A/B (`7bbfc8e`).  Write-up: `results/LEAF_FUSION_AVX512-AB-20260427.md`.

### AVX-512 small-node tail (masked partition), 2026-05-07
Three scalar `for (; j < n; j++)` tails in `decode_node_avx512` replaced with masked `partition_32_*`.  Win is bigger than "fix a tail" because partition is recursive: with stride=32 at BLK=8192, the deepest 30–40% of internal-node calls have `n < 32` and run entirely through the tail.  c8i +20–40%, c8a +17–40%.  Lone regression: two_sym_eq Zen 5 −5.9% (vpcompress overhead at depth-1 trees).  Results in `results/avx512-masked-tail-20260507/`.

### AVX-512 D=3/5/6 flat-subtree TBL, 2026-04-24
Earlier "tried and reverted" attempt (`fa1134b`) blamed vpermb + scalar compiler — wrong.  Real cause: `memcpy(&packed, bm_ptr, N)` for non-power-of-2 N (5, 6, 10, 12) lowered to split loads + OR + store-forwarding chains.  Fix: load the next natural size (uint64 / __m128i) unconditionally, with a `_safe` variant for the final chunk to avoid page over-read.  flat_M3 +472%, flat_M5 +322%, flat_M6 +492%; bell_s10 crosses parity 0.95× → 1.03×.  Commits `3f27e81` (D=3), `7b2fb8d` (D=5,6).  Lesson: always check whether `memcpy(ptr, src, N)` for awkward N compiles to a single load before blaming SIMD primitives.

### AVX-512 enc_init via vpermi2w + vpermex2var_epi8, 2026-05-12
`enc_init` was 34% (c8i) / 46% (c8a Zen 5) of encode wall.  Shipped via two commits: `7c08c19` (vpermi2w hierarchical gather over 4 chunks + blend) then `a282bb3` (byte-split via vpermex2var_epi8, +13% c8i, +6% c8a).

### AVX-512 pack_dN multishift, 64 codes/iter, 2026-06
Widened AVX-512 pack_dN from 8 codes/iter (uint64 lanes) to 64 codes/iter via vpmultishiftqb.  Commit `0c80e3a`.  Same iteration also widened `merge_flat_d{2..7}` to 64 codes/iter (`e5a199a`).

**NEON + AVX-512**

### NEON + AVX-512 D=7 flat decode, 2026-05
NEON (`7e4bc44`, `flat_d7_unpack` + two-level TBL scatter) and AVX-512 (`dcfaecc`, vpmultishift unpack + vpermi2b scatter) both shipped.  SSE/AVX2 still has no D=7 SIMD (pshufb is 16-wide, 128-entry c2s overflows) — see CONSIDERED entry.

---

## REJECTED

**General**

### Hybrid block decoder (deliberately not pursued)
Pick faster of pivco / trad_4s / huf0_x2 per block — trivially never loses on real text outside Apple silicon.  **Intentionally not done**: doesn't move the science; once you pick-the-winner you stop debugging the loss; the Graviton/Zen 3/Zen 5 prose gap is the open research problem.  Worth mentioning in the paper as a deployment-engineering fallback.

### Entropy-skew flat-subtree split (per-block real-freq), 2026-05-13
At `build_table`, if a flat-candidate's top-half/bottom-half real-freq is skewed past 0.625, un-flatten and let FSE compress the routing bitmap.  Expected ratio gain: chinese_text ~1.7 pp / html_wiki ~2.2 pp / bell_s10 ~4 pp.  First attempt (synth-freq-based, v0.3 within-tier ordering wire format) made ratio WORSE on every dataset because synth-freq can't see real-freq skew at all.  Real path would be per-BLOCK 1-bit-per-flat-root unflatten markers driven by per-block real data — not pursuing.

### FSE-decode ↔ merge fusion, 2026-05-23
Hypothesis: interleave scalar FSE-decode with SIMD merge to hide one behind the other.  Microbench: x8-fused +13% over x8-serial, x2-fused +24%, BUT x8-serial >> x2-fused (wide cursors and fusion are SUBSTITUTES — both compete for the same ILP).  "Fused" gain decomposed into chunked-locality (working-set effect, monotonic decay as chunk grows past L1) NOT overlap.  Real-decoder integration (NEON, bcast_left only): proba80 −2.2%, calgary −1.4%.  Microbench `bench_fuse_fse_merge.c` kept.  Lesson: wide-cursor FSE already spent the ILP fusion wanted; locality benefit doesn't survive integration overhead.

### Bitpacking libraries survey (simdcomp / FastPFor), 2026-05-11
simdcomp: no NEON, uint32-only, fixed block size (128 inputs SSE / 512 AVX-512), wire-format incompatible with our LSB-first horizontal layout.  FastPFor: same SIMDBP128 layout; horizontal-pack helper exists but is explicitly "for technical comparisons" of an older paper.  Nothing directly portable.  FastPFor's multiply-as-shift trick is irrelevant on NEON (vshlq_u32 is one insn) and on AVX2+ (has vpsllvd).  For a future v2 wire format, simdcomp + FastPFor are drop-in implementations; the format-change cost is the bottleneck.

### PIVCO-Huffman X2 (two-cursor), 2026-05-07
Split BLK into two halves with independent bitstreams + paired primitive calls.  All four backends implemented, all roundtrips pass.  **Measured negative on every uarch**: M4 0.85×, G4 0.76×, Zen 3 0.78×, Xeon GR 0.79×, Zen 5 **0.59×**.  Root cause: write-stream count, not counter chain.  Stride-16 confines writes to 2 cache lines; X2 spreads them across 4.  Microbench `bench_partition_micro.c` (removed, recoverable from git) showed stride-16 1.48× faster than paired-cursor at decoder-realistic INNER ≈ 91 iters.  Lesson: any X2 retry needs to REDUCE write streams, not increase them.

### Scatter fusion into partition loop body (conditional stores)
Already shipped: *leaf-child* fusion (half-partition + scatter when one side is a leaf).  Variant that was NOT kept: conditional stores in the partition loop body that go to either `tmp` or `symbols[scattered_position]` per lane.  Branch misprediction on NEON, store-buffer interference on scalar — massive regression.

### Relaxed / near-flat subtree detection, 2026-05-09
Two flavours, both bad: (A) force non-Huffman code lengths to create more flat subtrees — directly hurts compression; (B) almost-flat detection with per-element outlier handling — requires per-code outlier-marker bit (compression loss) AND per-element conditional in the scatter-bound hot loop (kills throughput).  Existing recursive path handles non-flat subtrees fine.

### Iterative DFS instead of recursion
Tested.  Essentially noise on M4.

### Cross-platform partition+scatter block-level fusion, 2026-05-09
Corrected microbench (P:S=4:1, sorted ascending dests): fusion saves 24–37% per-call across all 5 platforms, best at PpS=4.  Minimal-viable integration via NEON `decode_dual_neon` (block-level lookahead) measured **±1% — noise**.  The microbench-style win requires kernels in a single inner loop body, not just same function.  Realising it needs true fused per-(parent, leaf-type) kernels — combinatorial.  Not shipping.

### SIMD/GPU Huffman literature survey, 2026-05-12
Surveyed.  Best-of-list: (a) Oodle 6-stream BMI2 (ryg, Oct 2023) — SOTA branchless reference; (b) alias-Huffman (ryg, 2014) — closest conceptual relative to our flat-subtree path; (c) `vpmultishiftqb` (AVX-512 VBMI) — untested bitfield-gather primitive that could replace flat-subtree packed-region reads on Intel; (d) dougallj's "decoder convergence" parallel decode (Jul 2022) — REVIEWED, not applicable to our DFS format; (e) Intel IAA hardware Huffman (Sapphire/Granite Rapids) — non-portable but the c8i ceiling.  Frontier still open: vpmultishiftqb-based bitfield gather + decoder-convergence on ARM.  Tested via Brotli (libjxl's Huffman is a port): consistently 5–10× slower than ph on text decode.  Full annotated list in pre-curation log.  No survey item adopted; logged here so the option space doesn't get re-derived.

### EWAH word-aligned hybrid bitmap, 2026-05-16
Measured (`ext/ewah` via Lemire's EWAHBoolArray, uword=uint64_t) on ph's per-node bitmaps: EWAH **expands** the bitmap by 1.6% across p ∈ [0.5, 0.95] and only manages ratio 0.617 at p=0.99 (vs Rice 0.072 and FSE 0.097).  Decode 140–310 MB/s at moderate skew (slower than both Rice and FSE everywhere).  Root cause: EWAH targets clustered minorities (posting lists); ph's per-node bitmaps are IID Bernoulli — every 64-bit word at p ∈ [0.5, 0.9] contains both 0s and 1s, gets classified "literal" + pays a per-word header on top.  Wrong tool for ph's regime.  Results: `results/bench_golomb_ewah-m4-20260516-*.txt`.

**NEON**

### Root fusion (init + root encode), 2026-05-11
Hypothesis: inline the scalar `codes_la[i] = table->code_la[symbols[i]]` gather inside the root encode loop to avoid the L1 round-trip.  M4 prose_pride 1812 → 1725 (−4.8%).  Lesson: the separate path used spare LSU bandwidth (3 loads/2 stores per cycle); fusing onto the vector pipe makes the vector pipe the bottleneck.  Variant 16-stride unrolled with vld1q_lane recovers some ILP but still −2.9%.  Might revisit on wider vector pipes (AVX-512) or with true vector gather (SVE2 LD1H).

### neon2 4-way fused decode
Implemented in `src/pivco_huffman_neon2b.c` with clean scratch management.  Slower than neon on every distribution on M4 (proba80 −19%, english −30%, uniform −35%, two_sym_eq −74%).  Cause: 4-way partition of 8 elements on NEON costs 4 TBLs (identical to 2× 2-way); the pass-1 popcount scan to compute packed offsets costs more than the saved load/recursion-frame.  Fusion only pays when one instruction compresses wider than 8 (AVX-512 vpcompressw → 32).

### Interleave 16-elem partition pair in decode
Manually inlined both partitions in the stride-16 inner loop of `decode_node_neon` to hoist loads above stores and group TBLs.  Codegen transform dramatic (single ldp q0,q1, grouped popcounts, grouped TBLs) — runtime delta **±2%**, symmetric around zero.  M4 OoO was already renaming through the textual serialization.  Reverted to avoid doubling the loop body for no measurable win.

### Mix str d + str q to use 2nd store AGU on M-series
Hypothesis: M-series has separate SIMD-store + scalar-store dispatch.  Tested `simd_only` (2× str q), `mixed` (1× q + 2× d), `scalar_only` (4× str d).  Scalar can dispatch 1.6 stores/cyc (vs 0.97 for SIMD) — topology hypothesis confirmed.  BUT reaching the data via `str d` requires `ext.16b ...` to extract the high half, adding a rename op and a critical-path cycle.  Net wash to slowdown.  The 1 vst1q/cycle floor for partition_8 is real.  Path forward is FEWER stores per element (already done via half-tree + scatter fusion).

### Coalesce small-side partition stores
Accumulate variable-sized partition contributions in a register, flush only when 16 bytes ready.  6 NEON + 3 AVX-512 variants in `bench_coalesce*`.  Best result per platform: M4 0.88×, G4 0.96×, Xeon SR 0.75× — lost on every platform.  Each variant exposed the next failure mode: indirect-branch mispredict → cross-iter dep chain → SIMD-throughput-bound place-shift cost → ditto with OR-tree balancing (no help, it's throughput).  Bonus finding: AVX-512 `vpcompressstoreu_epi16` is **2× SLOWER** than `vpcompressw` + `vmovdqu64` on Granite Rapids (6 µops vs 2).  Full investigation in `docs/COALESCE.md`.

### iota-table for partition_root_8, 2026-04-26
Replace `vdupq_n_u16 + vaddq_u16` with `vld1q_u8` from a precomputed iota table.  Microbench +8–9% on M4, none on G4.  Full sweep: **end-to-end wash** — `partition_root_8` fires once per block; the decoder spends most time in 7 deeper partition levels unaffected by the change.

### Wider NEON partition (2 TBLs per 16 indices)
Tested.  Regressed due to load/store pressure.

### builtin popcount instead of compress_popcnt table
Tested twice, always worse on M4.  Re-retried 2026-04-24 on current clang: pivco_n drops 3–11%.  Clang lowers `__builtin_popcount((int)mask)` to a 4-uop vector chain (~8–10 c serial latency) routed through the same vector pipe as TBL.  The `compress_popcnt[mask]` load is 1 LSU uop at ~5 c on a separate pipe.  Replacing it adds vector-pipe uops AND lengthens latency.

### prim_merge popcount via pre-pass / in-register vcnt, 2026-05-29
Decode-side variants registered in `bench_prim.c` as `neon_pcpc`, `neon_pcpc_full`, `neon_unroll8`.  M4: `neon_unroll8` **−8%** vs production (in-lane popcount + freed dep chain wins despite doubled L/R loads).  c8g Neoverse V2: `neon_unroll8` **+15% slower** (narrower LSU + smaller OoO window can't absorb extra loads).  No single best variant across the ARM lineup — parked.  Confirms production merge is dependency-bound on `expand_popcnt[mask]` on Apple cores — relevant for any future Apple-specific kernel selection.

**SSE/AVX2**

### AVX2 partition_16 stride for enc_node_full, 2026-05-12
NEON V4-style 2× unroll for x86 partition.  Hot pipe is pshufb + store (per-element-bound, not per-iter); cursor dependency was already overlapped by Zen 3 OOO.  ±1–2% on every SSE/AVX2 host.  Lesson: count the busiest pipe, not the op count.

### SSE flat_decode_direct mostly scalar for D ∈ {2,3,5,6} (TD-only)
`flat_decode_direct_x86` is the legacy top-down flat-decode path; production BU uses `merge_flat_x86_impl` which has AVX2 D=2..6 paths and the D=4 SSE path.  Not pursuing TD optimization.

### Revisit SSE4.1 D=2,3,5,6 flat-subtree TBL (TD-only)
Was about pure SSE4.1 (no AVX2) flat-subtree TBL for hosts like c3 Ivy Bridge.  TD-only; BU production already has AVX2 D=2..6 paths.  Not pursuing.

### SSE D=5..8 unpack — gcc unroll pragma (TD-only)
Was about adding `#pragma GCC unroll` to the scalar `for (; i < n; i++) symbols[i] = c2s[bm[i]];` loops in `flat_decode_direct_x86` and `flat_decode_scatter_x86`.  TD-only; production BU has no such scalar loops.  Not pursuing.

**AVX-512**

### AVX-512 root iota, 2026-04-27
Replace scalar `id[k] = j+k` loop with precomputed iota-table read.  3 wins vs 3 losses at p<0.05 across 29 distributions — at the noise floor.  Patch preserved at `extras/avx512_root_iota.diff`.

### AVX-512 byte-scatter (vpscatterdd / vpermb) (TD-only), 2026-05-09
Originally a microbench probe for the top-down scatter path (`_mm_extract_epi16` + byte stores in `extras/bench/bench_fusion_v4_sse_cnt.cpp` / `extras/fusion.diff`).  AVX-512 has no byte scatter (vpscatterdd is dword-only) and no obvious dword-packing trick fits scattered destinations.  Moot: production BU writes contiguously (`code_to_sym[code]` lookups stored sequentially), so the byte-scatter primitive is not on any production path.  Not pursuing.

### PDEP / vpexpandw per-position code reconstruction (TD-only), 2026-05-10
Proposed an alternative decoder: build per-output-position code as contiguous uint16[N] via vpexpandw, then sequential c2s lookup with scalar stores.  Stated motivation: "eliminates scattered byte stores", projected 1.4× at D≤11 / 5× at D≤8 on Xeon prose_pride.  The projection was against the old top-down decoder's `_mm_extract_epi16` scattered stores, which were retired when TD went away.  Production BU already writes contiguously, so the original motivation is moot, and re-justifying vs current BU (which already beats huf0 1.0–5×) would require a fresh projection that nobody has done.  AVX-512 VBMI2-only, significant format change.  Not pursuing without new evidence.

**Other ISA**

### SVE 128-bit and SVE 256-bit, 2026-05-28
128-bit: already slower than NEON on G4.  256-bit on c7g (Neoverse V1): partition = **0.26× NEON-128 (~4× slower)**, D=5 flat-decode = 1.05× NEON (wash).  Structural reasons (paper-worthy): (1) `svcompact` only works on 32/64-bit elements; widen uint16 → uint32 costs more than the compress saves; (2) no 8-bit-mask → 8-lane-predicate primitive; (3) NEON's vqtbl1q_u8 throughput unbeatable; (4) V1 has 4 NEON pipes vs 2 SVE pipes — SVE-256 has the same bytes/cyc as 2× NEON-128.  SVE2 `bdep_u8` isn't the same primitive as AVX-512 `vpcompressb`.  **Implication**: the "Ice Lake VBMI2 moment" has no ARM analogue until ARM ships byte-granular compress.
