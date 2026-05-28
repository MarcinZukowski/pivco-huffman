# TANS Investigation — 2026-05-13

> **Last content review:** _NEVER_

Investigation log for the "PIVoted-COding tANS" idea sketched in
`pivco-tans.md`: per-internal-node TANS-coding of partition bitmaps,
keeping the flat-subtree fast path.  Goal of this session was to bound
the achievable upside (ratio-wise) before committing to a prototype.

Tool: `extras/bench/bench_tans_bound.c`, builds `build/pivco_tans_bound`.
Modes added during the session:

```
build/pivco_tans_bound                  # MAIN dists, single-shot per-histogram
build/pivco_tans_bound --dist-all       # all 29 dists
build/pivco_tans_bound <file>           # per-block on real file
build/pivco_tans_bound --verify-bytes   # MAIN dists, 100 blocks × 8K, empirical byte hist per depth
build/pivco_tans_bound --exact-tier     # MAIN dists, exact (no sampling), D≤K cumulative savings
```

## Bottom line

For most real-world prose/json/source data, the achievable ratio
upside from per-node TANS is **0.3–0.8% of huff bits under exact IID,
or 3–5% if byte-correlation can be exploited**.  Not enough to justify
the speed hit on a fast-decode codec.  The only regime where TANS
clearly wins is highly skewed alphabets (proba80, two_sym_90/10,
dna_fasta, geometric — 8–53% smaller) which are not pivco-huffman's
target use case.

**Parked. Worth revisiting if:**
- We have a *fast bit-level TANS* (the speed bottleneck disappears).
- A specific deployment targets skewed-alphabet data (genomics, LZ
  output streams, telemetry, etc.).
- Predefined ratio-optimal tables dispatched by per-node `(n_left,
  n_right)` make the table-cost problem disappear (see ideas below).

## The math

For a non-flat internal node with `n` codes routed through it and
left-fraction `p = n_left / n`:

| quantity | formula | what it represents |
|---|---|---|
| bit_H₂ entropy lower bound | `n · H₂(p)` | what bit-level arithmetic coding achieves if it knows `p` |
| byte-level entropy under *true* binomial(8, p) | `n · H₂(p)` | identical to bit-level under IID source |
| byte-level entropy from *empirical* sample histogram | depends | what FSE-on-bytes sees from this specific bitmap |

The first two are provably equal: each full byte under IID bits is
binomial(8, p) with entropy `8·H₂(p)`; a partial r-bit trailing byte
has entropy `r·H₂(p)`; the total always sums to `n_bits·H₂(p)`.  See
derivation: for byte value `v` with popcount `k`,
`P(v) = p^k · (1-p)^(8-k)`, and the entropy sum collapses to
`-log₂(p)·E[K] - log₂(1-p)·E[8-K]` = `-log₂(p)·8p - log₂(1-p)·8(1-p)`
= `8·H₂(p)`.

The third quantity is what matters in *practice*.  FSE doesn't have an
oracle for the true distribution — it estimates from data.  For small
bitmaps, the empirical histogram is noisy and the table description
cost can exceed the savings.

### Worked example (your 64-bit case)

64 bits with 48 zeros and 16 ones, p = 0.25:

| approach | encoded bits | table cost | total |
|---|---:|---:|---:|
| raw | 64 | 0 | **64** |
| bit-level FSE with known p | 64·H₂(0.25) = 51.9 | ~10 (one parameter) | **~62** |
| byte-level FSE with TRUE binomial(8, 0.25) | 8·6.49 = 51.9 | ~10 | **~62** |
| byte-level FSE with EMPIRICAL histogram (8 unique bytes seen) | 8·log₂(8) = 24 | ~53 (which 8 of 256 bytes) | **~77 (WORSE THAN RAW)** |

So byte-level FSE *cannot* compress 64 bits with p = 0.25 in practice
— exactly the user's intuition.  The empirical-vs-true gap is huge at
small sample sizes, and the table cost dominates.

## Per-distribution results (under exact IID, MAIN bench dists)

From `build/pivco_tans_bound`:

| distribution | huff (bps) | huff_flat (bps) | shannon (bps) | huff_flat saves | shannon saves |
|---|---:|---:|---:|---:|---:|
| `proba80` | 1.2505 | 0.9044 | 0.9044 | **27.67%** | 27.67% |
| `english` | 4.2475 | 4.2355 | 4.2253 | 0.28% | 0.52% |
| `flat_M5` | 5.0000 | 5.0000 | 5.0000 | 0% | 0% |
| `html_wiki` | 5.6340 | 5.5996 | 5.4759 | 0.61% | 2.81% |
| `prose_pride` | 4.6052 | 4.5678 | 4.5295 | 0.81% | 1.64% |
| `image_jpeg` | 7.9173 | 7.9016 | 7.8867 | 0.20% | 0.39% |
| `json_api` | 5.2398 | 5.2108 | 5.1984 | 0.55% | 0.79% |
| `gzip_random` | 8.0000 | 8.0000 | 7.9981 | 0% | 0.02% |
| `chinese_text` | 5.9778 | 5.9410 | 5.8143 | 0.62% | 2.74% |

From `--dist-all` (the full 29-dist sweep), three additional standout
results:

| distribution | huff_flat saves | shannon saves |
|---|---:|---:|
| `two_sym_90/10` | **53.10%** | 53.10% |
| `geometric` | 11.90% | 23.81% |
| `dna_fasta` | 8.19% | 8.19% |
| `bell_s10` | 1.73% | 6.02% |

### Three regimes

1. **Heavily skewed (`proba80`, `two_sym_90/10`, `dna_fasta`,
   `geometric`):** big upside (8–53%), flat carve-out captures it all
   (huff_flat == shannon).  These distributions have fully imbalanced
   Huffman trees with no flat subtrees, so the "keep the flat fast
   path" variant loses nothing.

2. **Normal text/json (`english`, `prose_pride`, `json_api`,
   `source_c`, `log_apache`, `image_jpeg`):** tiny upside (0.2–0.8%),
   huff_flat ≈ shannon.  Both upper bounds are noise; not worth
   pursuing.

3. **Wide-alphabet text (`html_wiki`, `chinese_text`, `bell_s10`,
   `bell_s30`):** small huff_flat upside (~0.6%), but real shannon
   upside (1.6–6%) — the flat carve-out is **costing 1.5–4 pp** by
   treating not-actually-flat subtrees as flat.  This is a sharper
   finding: on wide-alphabet inputs the flat-subtree fast path
   measurably hurts ratio.

## Per-depth (--exact-tier mode)

For each MAIN dist, computed exact expected `n_bits` and `H₂(p)` at
every non-flat internal node in an 8K block, bucketed by depth.  Then
the D≤K cumulative: what % of huff bits we save if we TANS-code only
nodes at depth ≤ K.

**Tier-K TANS savings as % of pivco-huff bits (exact-IID):**

| distribution | D≤0 | D≤1 | D≤2 | D≤3 | D≤4 | D≤∞ |
|---|---:|---:|---:|---:|---:|---:|
| `proba80` | 22.21% | 26.66% | 27.50% | 27.64% | 27.67% | 27.67% |
| `english` | 0.05% | 0.05% | 0.21% | 0.24% | 0.27% | 0.28% |
| `flat_M5` | 0% | 0% | 0% | 0% | 0% | 0% |
| `html_wiki` | 0.01% | 0.04% | 0.31% | 0.32% | 0.33% | 0.61% |
| `prose_pride` | 0.00% | 0.04% | 0.48% | 0.60% | 0.67% | 0.81% |
| `image_jpeg` | 0.10% | 0.12% | 0.19% | 0.19% | 0.19% | 0.20% |
| `json_api` | 0.04% | 0.12% | 0.32% | 0.39% | 0.39% | 0.55% |
| `gzip_random` | 0% | 0% | 0% | 0% | 0% | 0% |
| `chinese_text` | 0.01% | 0.09% | 0.12% | 0.31% | 0.37% | 0.62% |

Observations:

- For `proba80`, **D≤0 alone captures 80% of the total upside** (22.21
  of 27.67).  This is the canonical case where TANS at the root
  trivially recovers most of the Huffman slack.
- For most other dists, D≤2–4 captures 80–100% of the (already tiny)
  total upside.  The deep-tier savings are largely unreachable in
  practice anyway (table cost).
- `html_wiki` and `chinese_text` are exceptions: D≤4 captures only
  ~55% of the total achievable savings.  The rest sits at depth ≥5
  where bitmaps are 1–5 bytes long — empirically unreachable by FSE.

## Empirical byte-vs-bit gap (--verify-bytes mode)

Sampled 100 blocks × 8192 symbols per MAIN distribution, simulated the
actual tree-walk partition, measured **empirical byte-histogram
entropy** vs bit-level `n·H₂(p)` at each depth bucket.

| dist | aggregate byte_H − bit_H delta |
|---|---:|
| `proba80` | −4.87% |
| `english` | −8.03% |
| `html_wiki` | −8.95% |
| `prose_pride` | −8.72% |
| `image_jpeg` | −7.12% |
| `json_api` | −9.30% |
| `chinese_text` | −8.41% |

Per-depth breakdown (typical, e.g. `prose_pride`):

| depth | bytes/bitmap (typical) | byte_H − bit_H |
|---|---:|---:|
| 0 | ~1024 | −2.4% |
| 1 | ~512 | −5.0% |
| 2 | ~250 | −9.7% |
| 3 | ~120 | −17.8% |
| 4 | ~60 | −24.8% |
| 5 | ~25 | −33.4% |
| 6+ | <15 | ≥ −50% |

The shallow-depth delta (−2 to −10%) is **real byte-correlation gain**
from source non-IID structure (bigrams in prose).  The deep-depth
"wins" are **measurement artifact** — too few samples to estimate the
byte distribution.  Empirical-entropy bias goes both directions; one
spectacular outlier was `html_wiki` depth 8 showing **+393%** (one
unusually-popcounted byte spiking the histogram).

### Reconciling exact vs empirical

| measurement | prose_pride upside |
|---|---:|
| Exact-IID upper bound | 0.81% |
| Empirical-byte upper bound (depth 0–3) | ~5% |
| Realistic after FSE table costs (rough) | ~3% |

The exact-IID model can't see byte-correlation gain because it assumes
IID bits.  The empirical model captures it but inflates deep-tier
"savings" that aren't actually reachable.  The truth is probably
somewhere around 2–4% on prose-class data, exclusively from the top
2–3 tree levels.

## Where does FSE actually pay off? (zstd reference)

A useful sanity check: where in zstd does Yann Collet actually use
FSE, given that zstd is the gold-standard production deployment?

**zstd uses FSE only for specific small-alphabet, heavily-skewed
streams**, not for compressing literal bytes:

- **Sequence headers** — Literal Length codes (~36 symbols), Match
  Length codes (~53 symbols), Offset codes (~32 symbols).  These are
  small alphabets with strongly geometric/skewed distributions (most
  matches are short, most offsets small).  Huffman would quantize
  these poorly; FSE/TANS recovers significant fractional-bit slack.

- **Huffman code-length stream** — the alphabet that describes a
  Huffman table (lengths 0..MAX_LEN, ~12 values), itself skewed.

- **Literal bytes (the actual content)**: **Huffman, not FSE.**  The
  byte-frequency distribution in real text/source/json is already
  close enough to Shannon that FSE's gain over Huffman is tiny — same
  finding we hit independently above.

### LZFSE counterpoint

Apple's LZFSE (open-sourced 2016, used in iOS/macOS libcompression)
makes the *opposite* architectural choice from zstd: **all four
streams — literal lengths, match lengths, distances, and the literal
bytes themselves — are FSE-coded.**  No Huffman anywhere.

This isn't evidence that FSE beats Huffman on literals — more likely
an engineering trade-off:

- Single entropy stage = simpler implementation, smaller binary,
  important when shipping in the OS compression path.
- FSE was Yann's main public deliverable at the time LZFSE was
  designed; zstd was not yet the obvious winner.
- The post-LZ77 literal stream is *more uniform* than raw input
  (LZ77 extracts the highly-correlated bytes as matches; what's left
  is residual/novel bytes closer to uniform).  That's the *worst*
  case for FSE-over-Huffman gain — less skew = less slack to recover.

LZFSE benchmarks consistent with this: it lands a few % behind zstd
at comparable speed levels.  Some of that gap is match-finder
differences, but some is plausibly the FSE-vs-Huffman-on-literals
trade.

### The codec zoo

| codec | literal entropy | metadata entropy |
|---|---|---|
| zstd | Huffman | FSE |
| LZFSE | FSE | FSE |
| Brotli | static + context-mixed Huffman variants | (same) |
| LZ4 | none (raw literals) | none (raw escape-coded fields) |
| gzip / zlib | Huffman | Huffman |

The split that *zstd* — the more recent, more aggressively ratio-tuned
design — chose (Huffman where the alphabet is large/near-uniform, FSE
where the alphabet is small/skewed) is empirical confirmation of the
regime split we hit on our distribution sweep.

### Other "FSE genuinely helps" datasets (outside zstd-internal metadata)

- **FASTQ quality scores** (Phred values, ~40-symbol alphabet,
  strongly skewed) — CRAM uses range coding for this.
- **Run-length-encoded bitmap container metadata** (e.g. Roaring).
- **Time-series delta residuals** (Gorilla / Facebook TSDB).
- **DNS query rate-limit counters / telemetry quantiles** —
  geometric-ish.
- **LZ77 output streams** specifically — see experiment below.

All share the same shape: small alphabets with substantial
fractional-bit slack.  Same regime as `proba80` / `dna_fasta` /
`two_sym_90/10` on our sweep.

### Experiment: LZ4 output as "FSE food"

LZ4 deliberately does **no entropy coding** — its output is LZ77
matches/literals packed with a tiny token-byte format and raw 16-bit
offsets.  All the byte-level redundancy is left on the table.  This
makes LZ4 output an interesting "what does post-LZ77 data look like
for entropy coding?" probe.

Compressed `/tmp/prose_4mb.dat` (4,194,304 B of Pride and Prejudice)
four ways and ran `pivco_tans_bound` on each output:

| input | size (B) | ratio | byte-level huff | byte-level shannon | huff−shannon gap |
|---|---:|---:|---:|---:|---:|
| raw prose | 4,194,304 | 100.0% | 57.0% | 56.5% | 0.5pp |
| prose.lz4 (LZ77 only) | 3,487,450 | **83.2%** | **85.8%** | **85.4%** | **0.4pp** |
| prose.lzfse (LZ77+FSE) | 2,652,062 | 63.3% | 99.9% | 99.7% | 0.2pp |
| prose.zlib (LZ77+Huffman) | 2,675,734 | 63.8% | 99.9% | 99.6% | 0.3pp |
| prose.ph (pivco-huffman) | 2,435,390 | 58.1% | — | — | — |

**What jumps out:**

1. **LZ4 output has substantial byte-level redundancy left (85.4%
   Shannon, 14.6% byte-level compressible).**  This *is* what
   makes it "FSE food" in the broad sense: LZFSE and zlib both squeeze
   that 14% out via their entropy stage.  LZFSE's final 63.3% matches
   zlib's 63.8% within rounding — the post-LZ77 entropy gap is small,
   and *both* coders saturate.

2. **The huff−shannon gap on LZ4 output is only 0.4pp** at the
   whole-stream byte level.  That seems to argue against FSE-over-
   Huffman on LZ77 output — but it's misleading because **LZ4's
   output is a multiplexed stream of four different distributions**
   (literals, LL, ML, offsets), and the 0.4pp gap is averaged across
   them.  Byte-level Huffman sees a single mixed alphabet and gets
   close to its Shannon bound *for that mixture*.

3. **The actual FSE upside lives in stream separation, not in
   "FSE-vs-Huffman on raw bytes".**  zstd's win over LZFSE in
   practice comes from (a) splitting literals from sequences and
   coding each with the right entropy stage, plus (b) the fact that
   inside *separated* sequence streams (LL, ML, offset, 32–53 symbol
   alphabets with geometric decay), Huffman *does* waste 5–15% vs
   FSE.  When you re-mix all those streams into a single 256-byte
   alphabet stream, the per-stream gains average out to ~0.4pp.

4. **LZ4 byte histogram (top 8 of 256):**
   ```
   0x20 (' ')   4.94%   ← literals (prose space)
   0x00         4.14%   ← match-offset high-byte
   0x65 ('e')   3.15%   ← literal
   0x61 ('a')   2.85%   ← literal
   0x74 ('t')   2.78%   ← literal
   0x6f ('o')   2.64%   ← literal
   0x6e ('n')   2.57%   ← literal
   0x10         2.17%   ← LZ4 token byte (0:0 nibbles)
   ```
   The distribution is **prose-with-LZ77-flavoring**: literal-byte
   frequencies dominate (prose still drives the histogram), but
   structural bytes — match offsets clustering near 0x00 and the
   common LZ4 token bytes — add a secondary skew.  In our distribution
   zoo this lands somewhere between `bell_s30` (huff 7.09 bps) and
   `zipfian` (6.25 bps): moderately compressible at the byte level, but
   not heavily skewed in the way FSE's natural sweet spot demands.

5. **⚠ Important correction — our `prose_4mb.dat` is NOT real prose.**
   The file we've been calling "prose" throughout the session is
   actually random bytes drawn from a prose byte-frequency
   distribution: no words, no sentences, no string-level structure.
   First 100 bytes read like "wytpeatsl [Motleo ohftmntnhurnahl
   wsaeMdl tr ggelueto.gretdano recye…".  This makes any LZ77 stage
   useless (nothing to match), so on this synthetic:

   - pivco-huffman: 58.1% (byte-level Shannon = 56.5%, near optimal)
   - LZFSE: 63.3% (LZ77 overhead exceeds zero savings)
   - zlib: 63.8% (same)
   - zstd-1: 57.3% / zstd-9: **61.3%** (U-shape! middle levels are
     *worse* because their LZ77 strategy spends more metadata bytes
     than -1's lighter LZ77 does) / zstd-19: 57.4% (back to mostly
     entropy mode)

   On **real text** (a 277 KB concatenation of our markdown docs):
   - pivco-huffman: 66.9% (byte-Shannon at 63.5%, near optimal)
   - zstd-19: **33.5%** — about half of pivco-huffman, because
     real prose has enormous *string-level* redundancy ("the", "and",
     whole phrases) that LZ77 captures and *no* byte-only entropy
     coder can ever see.

   The bigger compression story (30+ pp on real text) is unreachable
   from inside the byte alphabet.  pivco-huffman, like any byte-only
   entropy coder, can only approach byte-level Shannon — not the much
   lower string-level entropy that LZ77+entropy combinations achieve.
   The current investigation lives entirely on the byte-level side of
   that divide.

### Cross-codec ratio sweep on MAIN datasets

Compared LZ4 (-9), zstd (-1 and -19), pivco-huffman alone, and
**LZ4+pivco-huffman as a composition** (LZ4 first, pivco-huffman as
the entropy stage over LZ4's output) across all 9 MAIN distributions.
Six are the real source files in `extras/datasets/`; three (proba80,
english, flat_M5) are synthetic byte streams sampled from their
distributions via `bench_generate_symbols`.

| dist | raw | lz4 | zstd-1 | zstd-19 | ph | lz4+ph |
|---|---:|---:|---:|---:|---:|---:|
| proba80 (syn) | 1.0 MB | 22.3% | 20.0% | **13.7%** | 15.8% | 18.7% |
| english (syn) | 1.0 MB | 79.6% | 53.1% | 53.2% | 53.4% | 69.5% |
| flat_M5 (syn) | 1.0 MB | 96.9% | 62.5% | 62.6% | 62.6% | 73.4% |
| html_wiki     | 1.0 MB | 21.4% | 19.5% | **14.7%** | 71.4% | 19.6% |
| prose_pride   | 738 KB | 41.1% | 39.9% | **29.6%** | 58.5% | **38.7%** |
| image_jpeg    | 280 KB | 96.5% | 96.5% | 94.5% | *segfault* | 96.3% |
| json_api      | 527 KB | 15.2% | 12.3% | **10.7%** | 66.5% | 13.8% |
| gzip_random   | 181 KB | 100.0% | 100.0% | 100.0% | 104.4% | 104.4% |
| chinese_text  | 494 KB | 53.1% | 56.2% | **38.9%** | 75.5% | **50.8%** |

**Three findings from this sweep:**

1. **`lz4 + pivco-huffman` is a competitive composition on real text.**
   Lands at zstd-1 level on most inputs, sometimes a touch better:
   `prose_pride` 38.7% vs zstd-1's 39.9%; `chinese_text` 50.8% vs
   56.2%; within 1 pp on html_wiki and json_api.  The combination
   "fast LZ77 + tight byte-Huffman" is a clean point in the codec
   design space.

2. **zstd-19 beats lz4+ph by 5–12 pp on real text** — but the gap is
   mostly attributable to **zstd-19's heavier LZ77** (longer matches,
   repcodes, larger window, contextual literal models), not its
   entropy stage.  zstd-19 uses the same Huffman that zstd-1 does for
   literals; pivco-huffman is already at byte-Shannon, so the entropy
   coder isn't the differentiator.

3. **The byte-sampled "english" stays revealing.**  Both LZ4 and
   zstd-1 land near 53% on `english`-sampled bytes — same as pivco-
   huffman alone — because there's no string redundancy to exploit on
   per-byte-sampled streams.  This is the same effect that made
   `prose_4mb.dat` (also byte-sampled) misleading earlier in this
   investigation.  The 6 real-source datasets are the trustworthy
   numbers.

**Two side-notes from the run:**

- `image_jpeg` pivcohuf compress **segfaults** — the encoder hits the
  near-uniform-input infinite-recursion bug logged in `IDEAS.md`
  (image_jpeg has byte-entropy 7.92 bps, essentially uniform).  Fix
  before pivco-huffman could be paired with LZ77 in production.
- `gzip_random` shows pivco-huffman's pure header/table overhead on
  truly incompressible data: 104.39% of raw = 0.04% overhead per byte
  spread across the 8K-block tables.

**Implication for the TANS investigation:** the composition `lz4+ph`
demonstrates that pivco-huffman is a viable entropy stage *above* an
LZ77 layer, achieving zstd-1-class ratio at potentially much higher
decode throughput (LZ4 + pivco-huffman are both designed for SIMD-
friendly fast decode).  Whether to ever pursue TANS over Huffman
inside pivco-huffman remains the same parked-for-now decision; but
the *bigger* product opportunity might actually be packaging pivco-
huffman as the entropy stage of a fast LZ77+entropy codec —
positioned as a faster, ratio-competitive alternative to zstd-1.

**Net read on "LZ4 as FSE food":**

LZ4 output *is* compressible at the byte level (15%-ish for prose),
but the headline upside isn't FSE-vs-Huffman — it's "*any* entropy
coder vs no entropy coder".  Both zlib (Huffman) and LZFSE (FSE) close
the gap to within 0.3pp of each other.  The remaining FSE-only upside
that zstd captures is from **separating the streams**, not from
better coding of the mixed byte alphabet.

For pivco-huffman's investigation, the takeaway is: **LZ77 output as a
mixed byte stream is roughly a `zipfian`/`bell_s30` lookalike, where
huff_flat saves ~0.3-0.6%** — same regime as natural text, not the
`proba80` regime.

This is exactly consistent with our results:

- `proba80` / `proba50` / `proba14` / `proba02` are **constructed
  precisely to stress-test FSE's natural use case** — small-alphabet
  geometric-style distributions like LL/ML/Offset.  Yann tests on
  these because they mirror the *kind* of data FSE handles inside
  zstd, not because they appear in raw input streams.

- "english" and "prose_pride" exist as bench distributions but **FSE
  shows poorly on them** because real-text byte distributions don't
  have enough fractional-bit slack for the gain to matter.  This is
  why zstd uses Huffman for literals.

So **the datasets where FSE/TANS actually wins over Huffman are**:

1. **Compressed-stream metadata** (LZ77-style match descriptors).
   This is zstd's actual use case.
2. **Skewed integer streams** — delta-encoded sensor data, genomics
   quality scores (FASTQ Phred scores, Illumina-style distributions),
   delta-residuals in time-series.
3. **Small alphabets with geometric/exponential decay** — proba80,
   geometric, two_sym_90/10, etc.  Real instances: run lengths in
   bitmaps, queue/buffer depth distributions, exponential service
   times in trace data.
4. **DNA-style 4–8-symbol alphabets** — `dna_fasta` shows 8% upside,
   real enough for genomics compressors (CRAM uses range coding for
   exactly this).

**What does NOT benefit from FSE over Huffman**: anything where the
alphabet is large (≥ ~64 symbols) and the distribution is "natural
text-like" — prose, source code, JSON, HTML, image data (DCT
coefficients have huge alphabets and the slack is in the *coding
model*, not the symbol distribution).

**For pivco-huffman specifically**: the codec operates on 256-byte
symbol alphabets with realistic real-world distributions.  This is
exactly the case where Huffman is within 1–2% of Shannon and
FSE/TANS-style coding cannot meaningfully outperform it.  Our
measurements confirm this directly.

## Open ideas

### 1. Specialized "N-bits-at-once" bitmap TANS

Standard byte-level TANS produces one byte (8 implied bits) per state
transition.  For very skewed bitmaps (p near 0 or 1), most bytes are
all-zeros or all-ones, and a more natural representation is
**run-length pairs `(run_length, value)`**.  TANS can be specialized
to emit runs directly instead of bytes.

This is closely related to Golomb-Rice coding (which is RLE over
bit-streams with parameter chosen to match `p`).  A TANS variant of
RLE could amortize the table cost across many runs.  Probably slow on
the decode side (each run is a variable-length emission), but worth a
prototype before dismissing.

### 2. Per-node "is it worth coding?" detection

The critical observation: at build_table time, we already have
`(n_left, n_right)` for every internal node.  We can **compute the
exact entropy savings per node before any coding**:

```
savings_per_node = n_node · (1 − H₂(p_node))   [bits]
```

A node is only worth TANS-coding if `savings_per_node > table_cost`.
The decoder can be told per-node whether the partition is raw bitmap
or TANS-coded via 1 dispatch bit (or a node-type byte we already
have).  This makes the design **adaptive per block, per node**:

- Skewed-data blocks → most internal nodes use TANS, big savings.
- Text-like blocks → most nodes stay raw bitmap, small/no savings.
- The encoder picks optimally; the decoder dispatches.

This eliminates the "what if the block has no skew?" downside.  Cost
of the per-node opt-in flag: ~1 bit per non-flat internal node, ≤ 255
bits/block.

### 3. Predefined, ratio-optimal table dispatch

Building a full FSE table per node is expensive (table cost +
construction).  But we don't need per-node tables if we **dispatch to
one of ~10–20 predefined tables** parameterized by `(n_left, n_right)`
quantized into buckets.

Buckets in p-space, e.g.: `[0, 0.025), [0.025, 0.075), [0.075, 0.15),
..., [0.45, 0.5]` (and symmetric for `p > 0.5`).  ~16 buckets.  Each
bucket has a fixed TANS table built once at startup.  Per-node cost:
just 4 bits to indicate which bucket.

This sidesteps the table-cost problem entirely.  Loss vs optimal
per-node table is small (a few % within each bucket) but you gain a
massive amortization advantage.  This is the design most likely to be
*both* fast and compress well.

### 4. Entropy-length grouping for flat subtrees

`pivco-tans.md` suggests grouping codes for flat-subtree treatment not
by their Huffman length (current behavior) but by their **entropy
length** `−log₂(freq[s])`.  E.g. group all symbols with entropy length
∈ [3.0, 3.5) into one "flat" pool, [3.5, 4.0) into another, etc.

Two pools with different effective lengths → emit a 1-bit selector
between them at the boundary node, then use N·D packed bits for each
pool with the pool's local D.

This could fix the **html_wiki / chinese_text "flat carve-out tax"**
we measured: those distributions have wide alphabets where current
flat subtrees group symbols with substantially different real
probabilities.  Entropy-length grouping puts the boundaries in the
right place.

Cost: 1 extra dispatch bit per flat-group boundary.  Encoder must
search for optimal grouping (small enumeration).  Probably worth a
prototype just to chase that 2 pp loss on Chinese/HTML.

### 5. Fast bit-level TANS (holy grail)

The bit-level entropy bound is achievable in principle by bit-level
arithmetic coding (Golomb-Rice, ELS, ABS, etc.).  In practice these
are slow because they're serial per state.

A bit-level TANS variant would need:
- 2-state (left/right) tiny tables — could be `2N` entries for `N`
  states, fits in L1.
- Multi-state interleaving (FSE-style 2× or 4×) to expose ILP.
- LUT-based decode (one indexed load per bit).

The end-to-end decode rate might land in the 500–1000 MB/s range —
slower than pivco-huffman's current ~5 GB/s, but not catastrophic.

If we had this, the per-node TANS idea becomes viable across the
board (no table cost problem), and the per-block opt-in is just a
recompress-or-not decision per node.

### 6. FSE pass-count observation

For more skewed data, the codes are shorter on average, so the FSE
state needs fewer renormalization passes per encoded byte.  This
means the *speed cost* of FSE is lower on the inputs where the *ratio
benefit* is highest.  These align favorably — if we target skewed
data, FSE is both better-compressing *and* relatively cheaper to
decode than on uniform-ish data.

## Tooling additions in this session

- `extras/bench/bench_tans_bound.c` — three modes:
  - default / `--dist` / `--dist-all`: per-histogram huff vs
    huff_flat vs shannon (entropy-only, no sampling).
  - `<file> [<file>...]`: per-block on real files (same metrics).
  - `--verify-bytes` / `--verify-bytes-all`: 100 blocks × 8K sampled,
    simulates real partition and measures empirical byte-histogram
    entropy vs bit_H per depth bucket.
  - `--exact-tier` / `--exact-tier-all`: no sampling, exact expected
    sizes per non-flat internal node, D≤K cumulative savings table.

- CMake target: `pivco_tans_bound`.

## Status

Investigation **paused**.  Not pursuing per-node TANS in current
pivco-huffman without one of:

- A fast bit-level TANS substrate (open idea #5 above).
- A target deployment that benefits from skewed-data ratio (idea
  #2's adaptive design makes this safe — costs nothing on text).
- A prototype for predefined-table dispatch (idea #3) that
  demonstrates the table-cost problem is genuinely solved.

The flat-subtree carve-out tax on `html_wiki` / `chinese_text` (~2 pp)
is the most actionable independent finding — addressable via entropy-
length grouping (idea #4) without any TANS at all.  That might be a
better next step than TANS proper.
