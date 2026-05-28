# Prior Art: Huffman-Shaped Wavelet Trees

> **Last content review:** _NEVER_

The pivco-huffman bitmap-per-Huffman-internal-node wire format is structurally
identical to a **Huffman-shaped wavelet tree** — an established data structure
from the succinct-data-structures literature (2003+). This document records
what overlaps with prior work and what does not, so the paper / talk / README
framing stays accurate.

## The representation

At each internal node of the Huffman tree, store a bitmap of length =
number of symbols passing through that node, with one bit per symbol indicating
which child it descends to. This is exactly:

- **Grossi, Gupta, Vitter — "High-order entropy-compressed text indexes"**
  (SODA 2003) — the original wavelet tree.
- **Mäkinen, Navarro** — gave the wavelet tree the *Huffman shape*, so the
  tree IS the Huffman tree and total stored bits = n·H₀ (zero-order entropy).
- Survey: Ferragina, Giancarlo, Manzini, "The Myriad Virtues of Wavelet Trees"
  (ICALP 2006); Navarro, "Wavelet trees for all" (JDA 2014).
- Alternative layout: Claude, Navarro, Ordóñez, "The Wavelet Matrix" (IS 2015).

In the WT literature, this representation is used as an **index** supporting
rank/select/access queries on text in O(log σ) per query — the core building
block of the FM-index and r-index for substring search. Not as a stream codec.

## Index vs. codec: the core distinction

The sharpest way to separate pivco from the entire wavelet-tree literature:
**wavelet trees are studied almost exclusively as a queryable *index*
(access/rank/select in compressed space); pivco repurposes the same
Huffman-shaped bitmap layout as a bulk *decompression codec*.** That change of
contract flips every design axis:

| axis | WT as **index** | pivco as **codec** |
|------|-----------------|--------------------|
| access pattern | random per-symbol rank/select queries | one linear bulk pass |
| metric | µs / query | GB/s throughput |
| bitmap augmentation | O(1) rank/select dict (RRR, `rrr2007`), +o(n) | none (never ranks) |
| bitmap compression | must stay queryable → RRR | free → FSE/tANS (pha) |
| decode direction | top-down, per symbol (`access`) | bottom-up, bulk merge |
| hot path | queries (build is offline) | decode (encode is offline) |
| K_right split | derivable by rank when needed | transmitted (no rank in the loop) |

Every pivco design choice is a *consequence* of "codec, not index": no
succinct rank/select structures, SIMD partition/merge, per-node FSE (only
possible because queryability is dropped), transmitted K_right, the
flat-subtree fast path.

**Numbers (Claude/Navarro/Ordóñez IS 2015 experiments, `claude2015matrix`).**
WT `access`/`rank`/`select` run at **~1–10 µs per query** (their Figs 7–8,
Xeon E5620), random access, on large alphabets (σ ≈ 2¹⁵–2²³ → 15–23 rank
ops/query). pivco bulk-decodes byte data at **~0.05–0.25 ns/byte** (GB/s). The
~10⁴–10⁵× per-symbol gap is the **index-vs-codec model**, NOT "RRR is slow":
RRR vs a plain (uncompressed) succinct bitmap is only ~10–40% in that same
paper (`WT-RRR` vs `WT-CM`). Frame the speed story as *random-query index vs
bulk-scan codec*, never as "RRR is the bottleneck."

**Rebuttal to pre-empt.** A reviewer will say "decoding a WT is just n
`access` queries — known." True, but doing it as n independent top-down random
queries IS the µs-per-symbol path; pivco's contribution is the **bulk
bottom-up SIMD reconstruction** that is ~10⁴× faster per symbol, plus the codec
wire format around it. The novelty is the bulk-decode algorithm and dropping
the index contract to unlock FSE — *not* the layout.

**pivco encode *is* WT construction.** Stripped of the query concern, the two
halves are symmetric: pivco *encode* = construct the Huffman-shaped WT bitmaps
via top-down SIMD partition (the Kaneta 2018 / Dinklage 2023 primitive — prior
art, no encode-side novelty); pivco *decode* = reconstruct the text via
bottom-up SIMD merge (the inverse; absent from the WT literature). They meet at
the partition step and diverge: the index line adds rank/select (RRR); pivco
adds the bulk decoder + per-node FSE. pivco's partition encoder could even
serve as a fast bitmap-build front-end for a WT *index* (then bolt on the
rank/select layer pivco omits) — a plausible bridge, but the construction
primitive itself is not pivco's contribution.

## SIMD construction primitives

The partition kernel pivco uses (TBL / pshufb / vpcompress to split a per-
position vector into left/right child ranges based on one bit per element) is
also published, applied to building wavelet trees:

- **Kaneta — "Fast wavelet tree construction in practice"** (SPIRE 2018) —
  first practical use of pshufb/pext for list-splitting at each node.
- **Dinklage, Fischer, Kurpicz, Tarnowski — "Bit-Parallel (Compressed)
  Wavelet Tree Construction"** (DCC 2023) —
  https://www.kurpicz.org/assets/publications/dcc_2023.pdf — full PDF
  read locally (`~/Downloads/dcc_2023.pdf`).  Extends to AVX-512 with
  vpcompress and vpshufbit; their algorithm is **construction only**
  (T → levelwise bitmaps {B_ℓ}; decode is not mentioned anywhere in the
  paper).  Includes the **Huffman-shaped** variant via inverse canonical
  codes + a parallel packed list L of remaining code-lengths filtered
  with vpcmp ≤ t before each list-split.  They explicitly document the
  same trick pivco's TD encoder uses:
  *"for large words and τ := 8, we use vpcompress with a side benefit:
  the selection mask for the right child is given directly the bits written
  ... and does not need to be computed separately. For the left child, it
  is simply inverted."*
  → **PIVCO's TD encode primitive is identical to theirs.**  No encode-side
  novelty against this work; the novelty stays on the BU / bulk-decode /
  flat-subtree side.

- **Dinklage, Ellert, Fischer, Kurpicz, Löbel — "Practical Wavelet Tree
  Construction"** (ACM JEA 26(1), Art. 1.8, 2021; `dinklage2021jea`) — the
  journal consolidation of this group's construction line (sequential +
  shared/distributed/external-memory parallel; Huffman-shaped variant in §9).
  ⚠ **Name collision to pre-empt:** it introduces a "**bottom-up**" technique,
  but theirs is *construction* (data-level): compute the leaf (full-character)
  histogram once, then derive every coarser level's histogram + interval
  positions by aggregating bit-prefix pairs (leaf → root), avoiding per-level
  text scans, and fill the bitmaps (their Alg. 1, output "a bit vector BVℓ per
  level"). This is the **inverse** of pivco's bottom-up, which is *decode*
  (bitmaps → text by merging child streams up to the root). Construction only;
  never measures decode.

  Three distinct "bottom-up"s to keep straight:
    1. classic Huffman tree build — **metadata** level (merge least-frequent
       pairs → code lengths / tree shape).
    2. Dinklage JEA 2021 — **data** level, bitmap **construction** (text →
       bit vectors via leaf-histogram aggregation).
    3. pivco — **data** level, bitmap **decode** (bit vectors → text via
       child-stream merge).
  pivco's table build (code lengths → canonical tree) is the metadata side and
  is standard/separate from all three.

Kaneta and Dinklage et al. (DCC 2023) are strictly top-down: start at the root
level, partition by the most-significant bit, descend. That's the same
direction as pivco's TD-encode and TD-decode. (Dinklage JEA 2021's bottom-up is
construction, not decode — see the name-collision note above.)

## Closest prior decode work

- **Baruch, Klein, Shapira — "Accelerated partial decoding in wavelet trees"**
  (PSC 2016 → Discrete Applied Mathematics 274 (2020) 2-10) —
  https://www.sciencedirect.com/science/article/pii/S0166218X18303974
  The full text has now been read locally
  (`~/Downloads/Accelerated-partial-decoding-in-wavelet-tree_2020_Discrete-Applied-Mathemati.pdf`).
  Their algorithm is **strictly top-down, scalar, one source index at a
  time**.  Traditional `range_decoding(i,j)` calls `extract(root,k)` for
  each k=i..j, each walking root-to-leaf with a fresh `rank_0(B_v,i)`
  query at every internal node.  Their contribution is a per-node scalar
  cache `rnk(v)`: on the first visit at v during a range query, pay a real
  rank query; on every subsequent visit, just increment by 1 (justified
  by the fact that rank on consecutive positions differs by at most 1, and
  the if-branch only fires when the bit is 0).  Two extra lines of code
  per direction; that's the entire algorithmic contribution (Fig. 4 of
  the PDF).
  Reported gains: ~50% full-decode speedup, ~30% partial-decode speedup,
  over SDSL's `wt_huff` on an i7-6700 (Skylake).  Scalar only — no SIMD.
  Compared only against the SDSL baseline (a scalar succinct-DS library);
  no head-to-head against huf0/FSE/zstd/brotli or any other fast Huffman
  codec.  No absolute MB/s reported.

  This is fundamentally a different regime than what pivco does:
    - They cache scalar rank state across t independent root-to-leaf
      walks, where t is the range length.  Each walk still touches one
      bitmap position per node.
    - pivco processes each node's *entire bitmap* with a single SIMD
      primitive (partition / tree_merge / scatter), traversing the tree
      once across all output symbols.  No per-symbol walk at all.
    - Their worst-case (all-distinct characters) additive savings is
      t·log t − 2t rank operations in the upper tree levels — i.e. the
      same "shared upper tree" insight pivco exploits, but their fix is
      a scalar cache while pivco's is bulk SIMD over the whole bitmap.

I have not found any WT paper that:

- Decodes bottom-up (leaves → root) via a tree_merge primitive.
- Uses SIMD scatter/expand for bulk decode (i.e. the partition primitive
  applied in reverse direction).
- Detects maximal flat subtrees and replaces D≥2 levels of bitmap with one
  packed N·D-bit region.
- Applies entropy coding per node (FSE on the raw bitmap).
- Benchmarks bulk decode bytes/sec against huf0 / FSE / zstd / brotli.

## Performance numbers in the literature

Dinklage et al. (DCC 2023) is the only paper with concrete SIMD numbers on this
representation. They report **construction** throughput in MiBit/s (output bits
produced per second). Their `shuf512` variant on an i9-11900KF (Rocket Lake
AVX-512, 3.5 GHz, turbo off).

**Two regimes — Binary WT vs Huffman-shaped WT (HWT):**

The paper measures BOTH a "binary WT" (fixed ⌈lg σ⌉-bit codes for every
character — i.e. a plain bitmap-per-level over a fixed-width encoding) AND
the Huffman-shaped variant (variable-length codes, which is the directly
comparable structure to pivco).  Binary WT is **2-3× faster** than HWT
because HWT adds a per-level filtering pass (vpcmp ≤ t on a parallel
remaining-code-length list to drop just-ended codes before list-splitting).

Binary WT shuf512 (Table 2 in the paper) — peaks at 1643 MiBit/s on english
(≈ 215 MB/s input).  This is the "1.4 Gbit/s tops" headline.

HWT shuf512 (Table 3) — the comparable shape:

| File         | H₀    | shuf512 HWT MiBit/s | ≈ MB/s input |
|--------------|-------|---------------------|--------------|
| dblp.xml     | 5.26  | 467                 | ~93          |
| english      | 4.53  | 428                 | ~99          |
| dna          | 1.98  | 243                 | ~129         |
| pitches      | 5.63  | 584                 | ~104         |
| proteins     | 4.21  | 435                 | ~108         |
| dna.16gib    | 2.00  | 259                 | ~136         |
| wiki.16gib   | 5.38  | 415                 | ~76          |

So the apples-to-apples (HWT, same structure as pivco) construction
throughput is **~100 MB/s** on Rocket Lake AVX-512 — *construction only*,
producing both wire data and a rank/select-ready bit vector.  Decode
throughput is not measured.  The "1.4 Gbit/s tops" headline applies to
the binary variant only.

**Multi-thread cap.**  §5 notes 16 threads → only 6× speedup, attributed
to Rocket Lake AVX-512 downclocking under heavy multicore use.  Same
constraint would apply to pivco encode on Skylake/Ice Lake/Rocket Lake
AVX-512 hosts.

Klein/Shapira report only relative speedups (~50% full, ~30% partial); no
absolute MB/s. Weissenberger/Schmidt (ICPP 2018) report >10× speedups for GPU
Huffman decode vs CPU Zstandard, but that's canonical-Huffman bitstreams on a
GPU, not bitmap-per-node on a CPU.

**There is no published bulk-decode-bytes-per-second number for the
Huffman-shaped wavelet tree representation that competes with fast CPU Huffman
codecs.** pivco appears to be the first to measure this regime.

## What stays novel about pivco

After this survey, the contributions that are NOT in the wavelet-tree
literature:

1. Framing the per-node-bitmap representation as a **stream codec** (encode
   → bitstream → bulk decompress on a CPU) rather than as a rank/select index.
2. **SIMD bulk-DECODE via partition/scatter** — particularly the BU
   (tree_merge / expand_tab) decoder, which doesn't appear in WT papers at all.
3. The **flat-subtree fast path** (maximal D ≥ 2 flat subtrees → N·D packed
   region + code_to_sym lookup).
4. **Per-node FSE** on the raw bitmap (the FSE marker / FSE payload
   alternative).
5. **Empirical positioning** against huf0 / zstd / brotli / FSE on real
   distributions, on Apple M4 / Graviton / AVX-512 hosts.

## How this should affect pivco docs

- README.md / IDEAS.md / future paper drafts must cite the WT lineage and
  NOT claim the bitmap-per-node layout is new.
- Reframe contribution as: "use of the (Huffman-shaped) wavelet-tree
  representation as a bulk stream codec, with SIMD partition-based decode
  (including a novel BU direction), a flat-subtree fast path, and per-node
  FSE — competitive with huf0/zstd."
- Lead with the BU decoder + flat-subtree + per-node FSE when arguing
  novelty; the TD path has more overlap with Dinklage 2023's construction
  primitives applied in reverse.

## Still worth verifying

- Whether any GPU-wavelet-tree paper does bulk decode with partition-style
  SIMD-equivalent (warp-level) primitives.

Resolved (2026-05-16): Baruch/Klein/Shapira 2018 read in full; strictly TD,
scalar, one-source-index-at-a-time with a per-node rank cache.  pivco's
BU/tree_merge and bulk-SIMD partition stay clearly orthogonal.
