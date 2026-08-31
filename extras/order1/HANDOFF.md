# Handoff: order-1 thread → PIVCO-HUFFMAN session

Written 2026-08-23 by the pivco-lz Claude session, where the order-1
research thread (2026-06 zph-era measurement → 2026-08 kernel endgame) ran.
This directory is the merge point: everything useful from that thread now
lives here.  Read `README.md` first (curated digest), then `RESULTS.html`
(The Order-1 Ledger — the full experiment record, three parts, §1–§25).

## State in one paragraph

Order-1 conditioning is worth ~1.5–3% of container on binaries via two
complementary sequence-side mechanisms — o6 context-switched tANS
(cross-field: roundtrip-verified, mozilla −1.55%) and MZ's grouped-w
scheme (temporal: xml ≈−2.1%, accounting only) — plus an encoder-selected
literal-side mode for literal-heavy files (x-ray lag-2 ≈ −13% container).
The reconstruction kernels are done and byte-exact on all four platforms
(K=2 at 0.13–0.33 ns/B per arch-specific champion; K=4 at 0.49–0.99).
What remains is coder engineering, not research: wire it into
phaz_pack_stream, build the two missing roundtrip coders, validate on
holdout + level-19 parses.

## Trust levels — what is proven vs accounted

- **Roundtrip-verified coders** (real bits, decoded back): `o3huf.c`
  (fused order-1 Huffman), `o4tans.c` (window walk), `o5bans.c` (binary
  window), `o6ctans.c` (context-switched tANS, incl. header + build audit).
- **Byte-exact kernels** (bit-identical reconstruction, no entropy stage):
  everything in `o2demux.c`.
- **Accounting only** (Shannon / realizable-Huffman arithmetic on real
  streams, no coder): all *literal-side ratio* numbers (routing simulated
  via maps), the grouped-w containers, the K=2-on-codes containers.
  In-sample bias estimated +0.1–0.8 pp — holdout check pending.
- All ratio work used **level-3** zstd parses (`phaz dump … -l 3`);
  level-19 re-dump is untested territory.

## Open work, in rough priority order

1. **Wire into phaz_pack_stream** (extras/phaz): (a) plain order-0 FSE
   method for degenerate streams — immediate, fixes the x-ray-ml 1-bit-floor
   56 KB loss with zero new machinery; (b) o6 context-switched FSE as
   ll → ml|llc → of|ml with per-stream argmax {PHA, o0-tans, o1-tans};
   (c) context-map mode flag {predefined, product, freeform, off}.
2. **Roundtrip literal coder** (K=2 first): mapgen2 remap → split by prev
   class → per-class PH/Huffman streams → k2 kernel re-interleave.
   Validates the simulated-routing literal numbers end-to-end.
3. **Grouped-w roundtrip coder** (the xml −2.1% prize): one tANS table,
   per-state w-groups, 3-way secondary {raw, Huffman, canonical binary
   tANS}; clustered-state grouping (~32–64 groups) to fix the 8 KB/stream
   header on small streams (unmeasured sketch).
4. **Holdout + level-19 validation** of all accounting-tier numbers.
5. **Production hardening of the kernels**: per-arch dispatch table is in
   README §"Deployable results"; gcc/clang each mangle a different champion
   (GEN_K2C goto-chained loops) — pin per compiler or freeze asm; jitter
   segment bases (4K aliasing); arch-gate the terminal trick (M4-only).
6. **Unmeasured but sketched**: K=2 lag-2 literal ratio (cheapest kernel,
   x-ray's best model); threshold classes (byte≥T after cluster-ordered
   remap — lifts the 64/128-codes-per-class cap that costs x-ray/samba real
   ratio); pair/record-granularity demux (u16 units, ~2× projected);
   ll|(pll,pof) seq-at-a-time contexts (mozilla ~+0.5% container);
   Leviathan-style SUB (match-predicted residual) literals.

## Environment

- Remote boxes (ssh aliases): `test-c8a` Zen5 EPYC 9R45, `test-c8i` GNR
  Xeon 6975P, `test-c8g` Graviton4 — all 2 vCPU, gcc11 + clang-15, `perf`
  installed.  Recipe: `scp -q o2demux.c $h:/tmp/ && ssh $h 'cc -O3
  -march=native -o /tmp/o2demux /tmp/o2demux.c && cd /tmp && ./o2demux'`
  (Graviton: `-mcpu=native`).  M4 = this laptop.
- Data: `tar xzf data/o1bundle.tgz -C /tmp` → `/tmp/phd_<file>/{lit,ll,ml,of}`
  + `/tmp/o1maps/<file>.map{2,4,8,16}` for dickens mozilla samba webster
  x-ray xml.  The bundle also sits on test-c8a:/tmp/o1bundle.tgz.  macOS
  purges /tmp periodically — the copy in `data/` is the durable one.
  Regeneration: `phaz dump` + `mapgen2.py` (see README).
- Bench conventions: `ONLY=<substr>` filters kernels (strstr, no regex);
  perf delta method = measure `ONLY=zzzz` (setup-only baseline) and
  subtract; mozilla lit = 12,560,280 B, harness does 49 decodes, K=2 yield
  9.09 B/step ⇒ 18.2 B per double-step pair.

## Cross-references

- **The Order-1 Ledger** (live artifact):
  <https://claude.ai/code/artifact/b84c2dd6-a991-4067-a87e-ed12f19401c8>.
  Source is `RESULTS.html` here — a Claude session updates it by editing
  the file and republishing with that url.
- `extras/phaz/` — the codec this feeds (zstd parse + PH entropy;
  README there).  `docs/FUSED-PHAZ.md` on the `fused-phaz` branch — the
  fused PHA decoder (extra-bits folded into the PH tree), the natural
  literal-side entropy stage.
- Origin session's memory (dense chronological log incl. every retracted
  measurement and who-caught-what):
  `~/.claude/projects/-Users-mzukowski-src-pivco-lz/memory/order1-literals.md`
  (plus `phaz-status.md`, `fused-pha-decoder.md` there).
- Working scratch dir this material came from (may be deleted):
  `notes/2026-08-15-o1demux-k4/` — contents are identical to this
  directory as of 2026-08-23.
