# Order-1 entropy coding for PH / phaz

Findings from the 2026-06 → 2026-08 order-1 research thread (MZ + Claude,
originally run in the pivco-lz session; merged here 2026-08-23).  "Order-1"
throughout = conditioning the entropy code on the previous symbol via a small
number of **context classes** (context quantization: 256 prev-bytes → K
classes; cf. LZMA `lc`, brotli context map, Oodle Leviathan O1).  Everything
below is PH-compatible: contexts select *tables or streams*, never adaptive
coder state.  All kernel numbers are byte-exact-verified; all four platforms
= Apple M4, AMD Zen5 (EPYC 9R45, test-c8a), Intel GNR (Xeon 6975P, test-c8i),
Graviton4 (test-c8g).

The full narrative with per-experiment tables lives in `RESULTS.html`
("The Order-1 Ledger", three parts, §1–§25), published as a live artifact at
<https://claude.ai/code/artifact/b84c2dd6-a991-4067-a87e-ed12f19401c8>.
This README is the digest: what to build on, what is settled, where the
numbers came from.

## 2026-08-23: merged with the PH-ctx session

The pivco-huffman session's parallel thread (PR #30 → the ctx coder
family → phaz e2e → xbits) is digested in `PH-CTX-SESSION.md` here.
Headlines of the merge, measured on identical dumps (details there):

- **Convergent findings** (independently measured twice, now settled):
  literal-tree bitmaps carry no bias (`phbias.py` ≡ the p_major
  histogram); Huffman ≈ Shannon on literals (`litfse.py` ≡ the #29
  study); contexts must select static tables, never adaptive state;
  per-set table builds dominate decode unless bounded (o5's canonical
  p-levels ≡ the ctx parameter dictionary); x-ray wants lag-2.
- **Complementarity law**: their within-stream carry-context coder and
  o6's cross-field contexts capture different information.  ctx wins
  ml/ll-class streams (its per-block bitmaps add free nonstationarity
  adaptation that global tables lack); o6 wins where cross-field
  correlation dominates (x-ray ml|ll at L3: 16.8 KB vs ctx's 29.8 KB;
  x-ray of|ml at L19: −6.9%).  Which pair carries the correlation
  MOVES with the parse level → per-stream argmax over the full menu,
  never a fixed recipe.  New L19 o6 numbers (open-work item 4, now
  partially done) are in `PH-CTX-SESSION.md`.
- **The fused-phaz × xbits composition** is the standout joint design:
  FUSED-PHAZ's per-code leaf runs ARE the per-(stream,width) classes
  the xbits analysis needs; coding each leaf's top byte inside that
  structure adds ~0.4pp of corpus ratio to the extract-free decode.
- Revised open-work ranking: (1) cross-field × per-block (o6-style
  route-conditioned table select inside the per-node marker machinery
  — the largest untried cell); (2) fused-phaz rebase + top-byte
  leaves; (3) kernel-law transfer to the ctx decode loop; (4) grouped-w
  roundtrip only if its xml-class gains survive L19 re-dump.

## The two money pots

**Sequence codes (the general win).**  zstd's parse whitens LL/ML/OF on text,
but structure survives as *same-sequence cross-field* correlation (ml|ll,
of|ml — not lag-1) plus *temporal/state-chain* structure.  Both are decodable
with zero serial-reconstruction cost because the routing stream is pre-decoded
(contexts come off the chain for free).  At sequence volume (~3 code symbols
per ~10.5-byte sequence) even 1 ns/sym ≈ 0.1 ns/output-B.

**Literals (per-file opt-in).**  Ideal H(X|prev) saves 8–17.5% of post-LZ
literal streams; realizable flat K=8 nets 3–9%; x-ray with lag-2 contexts
nets ~−13% of container.  But byte-granular reconstruction has a measured
~0.5–0.7 ns/B floor at meaningful K (M4) — multi-GB/s decode and meaningful
literal gains do not coexist.  Exception: the K=2 window kernel (6+ GB/s),
whose ratio is thin (70–80% of K=8's gain on binary files only).  Verdict:
contextual literals ship as encoder-selected opt-in for literal-heavy
binaries (x-ray, sao, mozilla) — exactly Oodle Leviathan's O1 positioning.

## Deployable results, ranked

1. **o6ctans — order-1 context-switched tANS for sequence codes**
   (`o6ctans.c`, roundtrip-verified; the thread's most deployable result).
   ANS state is table-agnostic ⇒ per-context 512-entry FSE tables, decode
   `e = dpk[(ctx<<TLOG)|x]` with ctx from the already-decoded route stream —
   context never touches the serial chain.  x-ray ml|ll 0.385 → 0.205 b/sym
   (at cond-Shannon) at identical speed (~1.0 ns/sym).  Containers (phaz l3,
   per-stream argmax PHA vs o1-tans): mozilla **−1.55%** (closes ~40% of the
   gap to zstd-19), x-ray −0.31%, dickens −0.33%, xml −0.27%.  Full cost
   audit passed: decoder builds 10–40 µs for all tables, Elias-gamma headers
   net-positive on every pair (low-mass contexts merge into a shared
   fallback).  Recipe: ll (order-0) → ml|llc → of|ml, each a plain FSE pass
   with table select.
2. **Grouped w-values, secondary-coded (MZ's scheme)** (`o4tans.c` extended;
   ledger §14).  One tANS table; the raw w-bit streams are partitioned by the
   *state* that consumes them and each group gets a secondary coder chosen
   per group from {raw, Huffman, canonical binary tANS} (2-bit flag + 4-bit
   p-level; o5bans prebuilt tables for nb=1 Bernoulli groups).  w|state
   captures 3–8% of stream bits = the full lag-1 Markov structure (the
   memory lives in the slot, not the symbol).  Unique property: a
   single-table walk that reads no bits — all entropy decode happens in
   parallel pre-passes.  Containers (realizable, 3-way secondary): xml
   **≈−2.1%** (xml's redundancy is temporal; beats o6 there), mozilla ≈−1.5%
   (ties o6), dickens −0.5%, x-ray −0.26%.  **Complementary** to o6
   (temporal vs cross-stream): per-stream argmax over both; stacked mozilla/
   xml plausibly −2.5–3% sequence-side.  *No roundtrip coder yet* — Shannon/
   realizable accounting only.
3. **K=2 demux on code streams** — the minimal-diff shipping path: remap +
   split codes by prev-code class, per-class order-0 FSE sub-streams, k2
   kernel re-interleave.  Zero new entropy coder (reuses phaz_pack_stream +
   the built kernel).  xml −0.87% container (beats measured o6 there — self-
   lag structure), mozilla −0.58%, samba ~−0.36%.  ~1.03 ns/sym.
4. **K=2 literal demux kernels** (`o2demux.c`) — (6,6)-window table walk,
   13-bit index (6 L bits | 6 R bits | class), meta+pattern split tables,
   ~9.1 B per dependent lookup.  Final dispatch (2026-08-18, mozilla ns/B):
   | arch | kernel | mozilla | range (6 files) |
   |---|---|---|---|
   | M4 | `k2g_u4` (2×16 B double-step) | 0.150 | 0.139–0.171 |
   | Zen5 | `k2qc_u4` (clang) ≈ `k2p_u4` (gcc) | 0.206 | 0.186–0.241 |
   | GNR | `k2q_u6` | 0.229 | 0.207–0.287 |
   | Graviton4 | `k2ntc_u4` | 0.328 | 0.319–0.432 |
   Records: M4 webster 0.130 ns/B = 7.7 GB/s (`k2gc`), Zen5 webster 0.186 =
   5.4 GB/s.  Ingredients that got here, each measured on all four archs:
   chain count swept to 4 (not the inherited 8), MZ's shared-header
   double-step (`k2g`), cursor compaction (dense live slots, FCFS segment
   counter, CMOVE-shrink loops — −7.9 instr/pair exactly as predicted), and
   pext + byte-packed meta (one `pext` replaces the shift/and index
   assembly; 40.8 → 32.2 instr/step, single-step retakes the x86 crown).
5. **K=4 demux**: trivial scalar array walk (`*P[c]++`) at 6 cursors on
   M4 (0.50–0.56) and Graviton4 (0.99); `k4x_u16` on AVX-512 — cross-chain
   lane-parallel arithmetic walk, bit-plane headers, vpermb-512 emit,
   vpternlog class fold — Zen5 0.485–0.488, GNR 0.582–0.587 (−27%/−35% vs
   scalar).  Tables cannot vectorize across chains; arithmetic can.
6. **tANS window walk** (`o4tans.c`): demux-table pattern on entropy —
   idx = (state<<W)|next-W-bits, emit all determined symbols.  4.4× classic
   interleaved tANS on skewed streams, but the 64K-entry eager build costs
   more than it saves below ~3M symbols.  Key reframe: *the ratio never
   needed the window* — classic 8-way FSE hits the same 0.385 b/sym on
   x-ray-ml.  phaz fix for degenerate streams = a plain FSE method in
   pack_stream; window walk is an optional length-gated accelerator.
7. **Binary window tANS for Bernoulli groups** (`o5bans.c`): at-Shannon,
   32-bit packed groups per lookup, hybrid crossover vs byte-FSE at
   p≈0.85–0.88 on all four platforms; ~16 canonical p-levels with prebuilt
   shared tables (2–4 MB) kill per-node build cost.  Its customer is the
   nb=1 groups of result 2 (the PHA-bitmap niche measured empty — see
   negatives).
8. **Context-map storage** (`o1seq3.py`): freeform entropy-coded grid
   (150–900 B, keeps all), product f(ll)×g(pml) (28 B, keeps 72–99%),
   spec-predefined pooled map (0 B, keeps 75–97% — context structure is
   semantic and transfers across files).  Ship as a per-stream mode flag,
   exactly zstd's FSE table-mode structure.

## Laws (each measured, not argued)

- **Conservation law**: order-1 information flows through serial
  reconstruction, or transmitted bits, or is surrendered — no fourth
  option.  Priced at every corner (exact class-plane = flat ratio but same
  serial step count; parallel plane forfeits I(c;c_prev) = 30–75% of gain).
- **Kernel metric = yield / effective dependent-lookup cost**; locality
  sets the cost (L1 160 KB @ 8.35 B/lookup beats L2 512 KB @ ~11).  The
  dependent-lookup constant is ~1.4 ns on M4, ~3.0–3.7 ns on cloud cores;
  yield is the only lever.
- **The IPC wall**: at optimal chain count, cycles/pair = instructions/pair
  ÷ an arch-specific effective-IPC wall (Zen5 ~3.7–3.9, GNR 3.45–4.9,
  Graviton4 ~4.2–4.4, M4 ~5.5–6).  Not load ports, not dispatch width, not
  chain latency at u4.  Serialization matters only through its effect on
  the wall (GNR dropped k2g's wall to 3.45).
- **Count the scaffolding**: ~40% of kernel instructions were field
  extraction + guards + spills, all removable (pext, byte-packed meta,
  cursor compaction) — the wins were pure instruction count.
- ~4 lockstep chains hide the recurrence; more chains spill registers
  instead of raising IPC.  u2 is the latency floor.
- Huffman is the zero-state member of the ANS family — no free order-1
  channel exists in the code itself; table/stream switching on the
  just-emitted byte *is* the Huffman equivalent of FSE's state-carried
  context, with the freeness relocated to the decode-side switch.
- Compiler hazards: gcc compiles a two-sided x86 ternary to a
  data-dependent branch (kills chain interleaving; write the ALU mux
  explicitly); 4K-aliased chain pointers collapse 6× (jitter segment
  bases); goto-chained compacted loops are compiler roulette (gcc and
  clang each mangle a different subset — production pins per compiler or
  freezes asm).

## Settled negatives (do not re-explore)

- **Nested K-leaf-stream scheme** cannot be exact below the root
  (value/residence decoupling); keeps ~half of flat's gain.  Exactness
  requires transmitting per-level routing bits — see conservation law.
- **Window-walk lazy build** dead: 52–97% of entries touched in one decode.
- **PHA bitmaps have no biased mass on literal trees** (`phbias.py`): all
  literal trees sit at mean bias 0.52–0.53 — Huffman is the de-skewing
  transform; bias exists only on dominant-symbol trees, where symbol-level
  tANS supersedes PH+bitmap entirely.
- **K=4 SIMD visibility tables** (heads4/win4d3) lose to the trivial scalar
  walk everywhere off-M4; K=4 big-table lands on scalar; scalar arithmetic
  re-serialization (k4r) loses everywhere — redeemed only when vectorized
  *across chains* (k4v/k4x, AVX-512 only).
- **Gather-based headers** (k4w): −45–52% — 32 scalar loads + movemask +
  pdep beat vpgatherqq by ~2×.
- **k2d/k2e/k2f double-step family** retired (chained second meta load is
  layout-invariant); k2g survived by sharing the header gather, then was
  superseded on x86 by pext singles.
- **Table shrinking** (small TLOG/W, packed entries): loses everywhere —
  the active subset of a big table self-shrinks into L1; W is the binding
  resource.
- **Routed walk** (pre-decoded class array) does not speed byte demux —
  the kernel is µop-bound, the chain was already hidden.
- **Literal-side coder tech**: Huffman is within 0.4–0.6% of Shannon on
  literal streams and K=2 contexts don't sharpen into the sub-bit regime
  (`litfse.py`) — literal-side money is structural, not coder-tech.
- **K=2 architecture on codes via table-switch** is dominated by full-
  context o6 at equal cost (but see deployable result 3 for the
  effort-adjusted demux variant).

## Prior art (ledger §21)

Oodle Leviathan (2017) is the closest shipped system: O1 literal mode = 16
streams split by last_byte>>4, per-stream entropy menu, encoder-selected —
validating both the stream-split architecture and the opt-in verdict; its
SUB (match-predicted residuals) and SUBAND3 (pos%4) modes are unexplored in
phaz.  LZMA `lc` = K by top bits in the serial-adaptive regime; brotli
context map = clustered contexts with transmitted map.  OpenZL's Dispatch
transmits routing tags — derived-control-stream routing (our o6) is exactly
what it lacks.  Ours: same-sequence cross-field contexts, the w-bits/
state-chain decomposition, the demux kernel-geometry theory (yield/lookup
law, visibility economics), capacitated small-K remap (`mapgen2.py`).

## File inventory

| file | what it is |
|---|---|
| `RESULTS.html` | The Order-1 Ledger — three parts, §1–§25 (live artifact, URL above) |
| `o2demux.c` | demux kernel shootout: all K=2/K=4 kernels incl. champions (k2g/k2q/k2p, GEN_K2C compaction, k4x/k4v, scalar walks), selftests + byte-exact verify, `ONLY=` filter |
| `o3huf.c` | fused order-1 Huffman (per-class bitstreams, ctx-switched decode) — first roundtrip order-1 coder; fused-K4 ≥ two-pass on all platforms |
| `o4tans.c` | tANS window walk + MZ's grouped-w extensions (prepacked width arrays, w\|state measurement) |
| `o5bans.c` | binary tANS window walk for Bernoulli/bitmap groups, canonical p-levels |
| `o6ctans.c` | order-1 context-switched tANS, roundtrip + full cost audit — the deployable sequence-side coder |
| `mapgen.py`, `mapgen2.py` | bijective remap so class = top bits; mapgen2 = capacity-constrained clustering done right (cap inside the optimizer, ≥ raw by construction) |
| `k8ratio.py` | order-1 gain curve over K (2…256), ideal + constrained |
| `k4yield.py`, `k4cluster.py` | yield-per-dependent-lookup simulation; ratio-vs-yield Pareto sweep |
| `o1seq.py`, `o1seq2.py`, `o1seq3.py` | sequence-code contexts: single, joint-bucketed (K=32 pair contexts), context-map storage options |
| `litfse.py` | literals Shannon vs Huffman, order-0 and K=2 (decisive negative) |
| `seqk2.py`, `seqk2b.py` | K=2 on code streams; seqk2b's FSE accounting exposed ll self-structure hidden by the 1-bit floor |
| `phbias.py` | traffic-weighted PH-tree bitmap bias (closed the o5-on-PHA niche) |

## Data & bench recipes

Inputs are Silesia stream dumps + class maps (not committed; ~25 MB bundle
in `data/o1bundle.tgz`, gitignored, also on test-c8a:/tmp/o1bundle.tgz):

```sh
# regenerate dumps (extras/phaz tool; level 3 used throughout this thread)
./phaz dump <silesia/file> /tmp/phd_<name> -l 3     # writes lit/ll/ml/of + meta.txt
python3 mapgen2.py                                   # -> /tmp/o1maps/<f>.map2 (+ .map4/8/16)
# or just restore the exact bundle
tar xzf data/o1bundle.tgz -C /tmp
```

Files used everywhere: dickens mozilla samba webster x-ray xml.

```sh
cc -O3 -march=native -o o2demux o2demux.c && ./o2demux    # x86 + Graviton (-mcpu=native there)
cc -O3 -Wall -o o2demux o2demux.c && ./o2demux            # M4
ONLY=k2q ./o2demux                                        # substring filter (strstr, no regex)
# perf delta method (cloud boxes): measure ONLY=zzzz (setup baseline), subtract
perf stat -x, -e cycles:u,instructions:u env ONLY=zzzz ./o2demux
```

Machines: `test-c8a` = Zen5 EPYC 9R45, `test-c8i` = GNR Xeon 6975P,
`test-c8g` = Graviton4 (all 2 vCPU, gcc11; clang-15 also present — the
Zen5 champion `k2qc` needs clang).  Bench harness: 8–16 KB striding
segments, per-block context restart, jittered bases.
