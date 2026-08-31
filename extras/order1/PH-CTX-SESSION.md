# PH-ctx session digest (2026-08-17 → 08-23)

The pivco-huffman session's side of the merge (counterpart of `README.md`,
which digests the pivco-lz order-1 thread).  Origin: PR #30 (Nick's
dynamic nibble-FSE demo) → "how much more can the per-node coders
squeeze" → a full prototype.  Everything below is roundtrip-verified
unless marked; M4 unless stated.

## Where the code lives (IMPORTANT — ephemeral!)

The whole ctx implementation exists only as UNCOMMITTED changes in a
scratchpad worktree of the PR #30 branch
(`/private/tmp/claude-503/.../scratchpad/pr30tree`, base `e1bdce0`):
`src/pivco_ctx.{h,c}` (the coder), codec/wire/pivcohuf/bench edits, and
the phaz env knobs.  Raw sweep outputs sit next to it.  **This needs to
be committed to a branch before the scratchpad is purged.**

The purge already hit once (2026-08-24: `bench/`, `ext/fse` vanished;
restored from `git archive e1bdce0`).  `ph-ctx-session.patch` was
regenerated the same day as a plain `diff -urN` (apply with `patch -p1`
from a pristine e1bdce0 tree; 11 files) and now also carries the
per-level census instrumentation: `PIVCO_LVL_STATS=1` dumps a per-depth
winner/savings census at every bitmap commit decision;
`PIVCO_NOFUSE=1` disables quad fusion for clean attribution.
Census dumps: `data/lvlstats-2026-08-24.tgz` (24 inputs × 2 arms).

## The modeling arc (probes, KT-priced, before any coder was built)

- Symbol-width sweep over PH node bitmaps (W ∈ 1..16): wider beats
  nibble (PR #30's W=4 leaves 4–15% on the table), W≈8–10 optimal,
  per-bitmap W adds ≤0.7% over one good fixed W and is predictable
  from (K, depth) at zero wire cost.
- **Bit-context models dominate the whole width family**: order-k
  P(bit | last k bits) beats the best fixed-W histogram by 7–10% (mr,
  x-ray) to 26% (nci, k=8 runs) — width is a speed knob, context depth
  is the modeling knob.  Post-LZ the optimal order drops to k=2..6 and
  the gains to 1–5% (structured literals/streams), ~0 on text.
- Quad-node fusion (parent+children bitmaps as one 4-ary route stream):
  −8..−14% on stride data, ~75–95% of it at the top tree level;
  order-0 on pairs is provably identical to separate bitmaps — all the
  gain is context.  Under plain byte-Huffman (no context) fusion still
  gets −2..−7% via within-byte joint structure + saved headers, but
  can lose where windows shrink (gate per node).
- Parametric transmit-and-expand loses only 0.1–0.25% vs the adaptive
  (KT) bound at matched order; Huffman-expanded tables instead of tANS
  cost +0.2–0.5% post-LZ (+4% at raw-file skew) = eats 6–13% of the
  incremental benefit for the faster decode loop.

## The coder (wire ids 52–56 on the PR #30 branch)

One mechanism: **order-c carry-indexed tANS over bytes** (c = k or 2q
context bits; per-carry 2^L-entry tables, L=11/10; alphabet 257 with an
ESC symbol for model-impossible bytes; FSE-style spread; encode
backward / read backward).  Params = per-context 4-bit quantized probs
(2–32 B on the wire), expanded to tables on both sides.  Bit-context on
partition bitmaps + natural flats (ids 52/53), pair-context on fused
quad records (54–56; fusion flagged in K_right bit 15, children encode
into a temp buffer, commit-if-smaller against the complete unfused
form).  Decode hot loop: `e = T[carry][state]; emit e.sym (8 bits);
carry = cmap[e.sym >> (8-c)]; state = e.base + read(e.nbBits)`.

Two bugs found by *pricing*, not roundtrips (both sides consistent →
correct output, silently bad ratio): carry bit-order mismatch (rolling
context is newest-bit-first; a byte's top bits are stream-order —
tables consulted under permuted labels), and the min-freq-1 floor
(taxes skewed tables up to 256/2048 of the mass → ESC symbol).

**Table-build economics is the whole game** (fse_8k lesson at scale):
free per-node params → build per node → 97% of decode time.  Fixes:
split enc/dec builds, O(256) normalize, snap-to-cache (±1/nibble
adoption), and a **capped per-mode parameter dictionary**
(`PIVCO_CTX_DICT`, default 48): after the cap, nodes must adopt the
nearest existing set; bad fits fall back via commit-if-smaller.  The
cap is the ratio↔decode dial: free = −1.66% vs zstd-3 at 0.02–0.65
GB/s; cap 128 = −0.89% at 0.3–2.4 GB/s.

## Results ladder (the same codec, four questions)

| question | baseline | result |
|---|---|---|
| vs order-0 floor, raw files (Nick's sets) | entropy floor | src −14.7% (snow −81.7%), extras −4.9% |
| vs floor, post-LZ streams | entropy floor | literals −5.7%, tokens −2.7% (no file regresses vs dANS) |
| whole zstd-3 files (phaz, silesia) | real zstd-3 | −1.66% free / −0.89% dict-128 |
| whole zstd-19 files | real zstd-19 | +0.75..0.92% total; mr/sao/x-ray −0.8..−1.0% each |

phaz e2e decode (level 19, M4, corrected 16K build, clean baselines):
PH-static 1.2–1.9× faster than zstd-19 at +2.5% size; PH+ctx(128)
+0.9% size at 0.6–0.9× zstd decode.  Two contaminations to never
repeat: the PR #30 branch defaults `fse_dynamic=1` (dANS silently in
"plain" arms, −30..50% decode); Apple builds need PIVCO_BLOCK_SIZE
16384 (the forced-32K harness workaround cost ~10% decode; Nick's
bench_chunk_sizes overflows its 16K dec buffer with 32K chunks on
Apple = the old mystery SIGTRAP).

## xbits (extra bits) findings — feeds FUSED-PHAZ directly

Packed xbits ≈ 7.95–7.99 bits/byte (looks random), but per
(stream, width) classes hold 5–17% slack, OF-dominated; at level 19
xbits are 32% of corpus output (61% of nci's).  Real coding of
per-width top-8-bit streams through PH+ctx = **−0.4pp of corpus vs
zstd-19** (x-ray −1.2→−2.8, sao −1.1→−1.7), costing ~one extra small
stream's decode where classes commit, nothing where they don't.
Decomposition: x-ray/sao/mr-class slack is genuine top-byte bias
(H(top8) ≈ 6.9–7.4); mozilla-class slack is exact-value repetition
(H(top8) ≈ 7.95, structure only in the joint) — that part needs
full-value/rep-code treatment (LZ-side), not entropy tricks.  Mixing
widths in one stream uniformizes everything — width-conditioning is
mandatory (three broken measurement schemes proved it).
**The offset dictionary (2026-08-23)**: per-file top-N full offsets
transmitted (~3 B each), of-code alphabet extended by N symbols; a hit
is one symbol — no code, no extras; dictionary sits before the width
split.  Corrected savings on L19 of-streams (2026-08-24 audit fixed a
rep-extras leak in the first numbers): mozilla 241 KB (1.6% of output),
x-ray 105 KB, nci 35 KB, mr 12 KB; samba opts out (−0.7 KB).
Not parse-sensitive (L3 keeps 2/3 of mozilla's save); ll/ml dicts dead;
gap-coded transmission saves a further 2–8× on the entry bytes.  Matches the regime split: dictionary wins
exact-value files, ties top-byte on bias files — per-file argmax.
Lessons: separate hit/miss stream accounting silently gifts the decoder
the flag (n·H2(p) ≈ 480 KB on mozilla — first version overstated 2.7×);
frequency ranking ≈ exact marginals; individually-positive picks can
jointly lose (samba N=4096) — guard = two-pass exact eval incl. N=0.

**Composition with FUSED-PHAZ**: its leaves already group residuals by
code (= by width); "code each leaf run's top byte, low bits raw" gets
this ratio inside the structure that also kills the serial extract.

## Per-level census (2026-08-24, ledger §1.1/§2.1)

Where in the tree the bitmap coders win, by node depth: corpus
byte-weighted savings split lv0 30% / lv1 25% / lv2 24% / lv3 13% /
lv4+ 8%.  Top-level-only holds ONLY for dominant-symbol trees
(ml-x-ray 99% at lv0); literal/text trees are mid-heavy (dna 99.6% at
lv2, chinese peaks lv7).  ctx's increment (+33%) has the SAME level
shape as id 51 — the remembered top-level concentration is quad
fusion's (75–95%), a different sub-mechanism.  Don't depth-gate.

## OpenZL joint-token measurement (Nick, 2026-08-22)

token = min(ll,15) | min(ml−3,15)<<4 (+2 rep bits variant): for
order-0 stages a real win; against our per-stream ctx it measured
**ratio-neutral to −0.3pp worse** (correlation gained ≈ within-stream
context lost) — but it halves sequence-symbol count = a decode-speed
lever.  o6ctans (below) is the superior form of the same idea:
condition instead of fuse.

## o6 cross-field vs our ctx — the merge A/B (identical dumps)

Level 3 (their bundle) and level 19 (our dumps), absolute coded bytes
(o6 = merged-K variant, b/sym × n/8 + header):

| stream | our ctx | o6 o1 | winner |
|---|---|---|---|
| L3 x-ray ml | 29841 | **16845** | o6 by 43% (!) |
| L3 mozilla ml | 1485415 | **1421057** | o6 by 4.3% |
| L3 mozilla of | **1243607** | 1288943 | ctx |
| L3 xml ml/of | **121262 / 94251** | 125639 / 94704 | ctx |
| L19 x-ray of | 686936 | **~639950** | o6 by 6.9% |
| L19 mozilla ml | **1788347** | ~1844400 | ctx |
| L19 dickens of | 479969 | **~472500** | o6 |

Laws that fall out: (1) the mechanisms are complementary, not
competing — ctx wins where serial/nonstationary structure dominates
(our per-16K-block bitmaps give free per-block adaptation; o6's global
tables start 0.1–0.3 b/sym behind on mozilla-class data), o6 wins
where the correlation is cross-field; (2) **which field pair carries
the correlation depends on the parse** (x-ray: ml|ll at L3 →
evaporates at L19 while of|ml opens to −0.44 b/sym) → per-stream
argmax over the menu is mandatory, fixed recipes are wrong; (3) the
untried cell with the largest expected value is **cross-field ×
per-block**: o6-style route-conditioned table selection wired into the
per-node marker machinery (context class × carry selecting dictionary
param sets).

## Settled on our side (do not re-explore)

- Bias-only coding of PH bitmaps: dead (p_major histogram: 99% of
  bitmap bytes < 0.625 on silesia-class data; matches `phbias.py`).
- Static-vs-dynamic FSE history: docs/FSE-V0.md compared PHA vs
  transmitted-table FSE and won *under an order-0 model*; the iid
  assumption hid #30's sequence-structure win (nibble alphabet, cheap
  NCount) — recorded so the foot-gun isn't repeated.
- Tree shaping for per-node coders: hill-climb (placement + Kraft-exact
  reshape) tops out at −1..−5% at absurd encoder cost; dstride
  preprocessing subsumes it on record data; fusion-aware shaping
  ("make the root full two levels") is the only shaping idea left.
- Per-bitmap free parameters at decode: dead (build-cost law).
- W>8 histogram tANS: unnecessary — context depth k replaces width W.

## Pointers

- PR #30 (github): the harness (`pivco_bench_chunk_sizes`) both
  sessions' container numbers run through; review findings pending.
- `docs/FUSED-PHAZ.md` on `fused-phaz` (base bda0ac1): the fused
  extra-bits decoder.  Diff is 2.0K lines but entirely under
  extras/phaz/ (self-contained kernels, no production prims) — landing
  cost is a rebase, not a prim project.
- This session's chat archive holds the full experiment log (wscan /
  mkv / fusion probes, the phaz ladder, xbits decomposition).
