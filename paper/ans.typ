#import "conf.typ": _html, _pdf, _fmt, anote, setup, PH, he, sym, PHA, fair-cell, h0

= Breaking the bit-barrier <ans>

Huffman encoding, while ubiquitous, has one well known, key limitation.
The code lengths are limited to full bits.
That means, for some distributions, it is further from entropy-optimal
than desired.

A well known solution to this problem is arithmetic coding (AC, @arithmetic), however,
it has not been popular due to its performance and patent controversies.
Luckily, Jarek Duda proposed _ANS-based encoding_ (@dudaans),
which solves both problems.
It led to two main algorithms: tabled-ANS (*tANS*) and range-ANS (*rANS* @encodesu1920).
All these algorithms allow codes to have average lengths close to entropy-optimal.
We will focus on tANS.

The typical tANS decoding is actually very similar to an optimized
Huffman decoding routine from @sota.

```c
t = decoding_table[state];
state = t.newX + read_bits(t.numBits);  //state transition
emit_symbol(t.symbol);                  //decoded symbol
```

While similar, the critical difference is that we see a `state` variable that
is carried through the iterations and mutated depending on the data.
Also, the `decoding_table` is constructed such, that for the same code
a different number of bits may be consumed.

As a result, the same approach to tANS decoding as we did to Huffman in @sideways
is not directly applicable.
Still, we will demonstrate how #PH opens a unique opportunity to apply ANS.


== Skew analysis in Huffman trees

#let bold-rows = ("proba80", "dna_fasta", "calgary_pic")
#let rows = csv("data/dist-stats.node-benefit.csv")
#let data = rows.slice(1).map(r => {
let b = r.at(0) in bold-rows
r.map(c => if b { strong[#c] } else { [#c] })   // bold the whole row
})
#figure(
table(
    columns: 6,
    align: (col, _) => if col == 0 { left } else { right },
    table.header(
    [*Distribution*], [_H_ (bits)], [Huffman (bits)],
    [_H / Huff_], [_Huff - H_], [max node benefit],
    ),
    ..data.flatten(),
),
caption: [Per-dataset entropy gap and peak single-node bitmap benefit (bits/byte).
          Bold rows are the heavily-skewed datasets.
          In all of these, a single partition node captures most of the Huffman redundancy.]
)<tab-node-benefit>

@tab-node-benefit shows additional analysis for datasets from @datasets. We can see that for most of them,
Huffman encoding actually achieves almost perfect code length, reaching typically 97-99% of entropy.
As a result, for most of them, applying a more expensive compression method is probably not useful.
Three datasets stand out:

- *proba80* - artificial dataset, skewed on purpose
- *calgary_pic* - a mostly-white bitmap
- *dna_fasta* - DNA dataset, mostly #sym("A C G T") letters plus some extras

#he("myfig")[
  #table(
    columns: (50%, 50%),
    stroke: 0pt,
    align: center,
    [#figure(
      image("figures/skew-calgary.svg"),
      caption: [Skew visualization for *calgary_pic* (top part of the tree only)]
    )<fig-skew-calgary-pic>
    ],
    [#figure(
      image("figures/skew-dna-fasta.svg"),
      caption: [Skew visualization for *dna_fasta* (top part of the tree only)]
    )<fig-skew-dna-fasta>
    ],
  )
]

@fig-skew-calgary-pic and @fig-skew-dna-fasta show visualization of the
top parts of the tree for *calgary_pic* and *dna_fasta* datasets.
For each tree node, we report left/right skew, and the percentage of data
covered by a given subtree.

For *calgary_pic* we see how the root node has a skew of 87.1/12.9, with *H=0.554*
That means, if that *one node* was entropy-encoded, we would save *0.446* bits
per encoded element, almost reaching the Huffman encoding gap of *0.480*.

*dna_fasta* is slightly different, because the interesting node is not the root, but a node 2-levels deep.
With #sym("A C G T") symbols occupying the vast majority of the input, #sym("C G T") were assigned
2-bit codes, but #sym("A") had to be assigned a 3-bit code to make room for
the remaining, infrequent symbols.
As a result, 25% of all symbols get to the parent node of #sym("A") and 94.3% of those go to #sym("A").
That node has *H=0.315*, so entropy-encoding would save *0.685* bits for each symbol,
but with only 25% of the input reaching that node, it results in *0.171* average bits saved per code,
also very close to the *0.185* bits Huffman gap.
Note that the sibling node of #sym("A") has an even stronger skew, but with only 1.4% of data reaching it,
optimizing it is not worth it.

== Pivco-Huffman+ANS implementation

The analysis above suggests, that for most datasets applying ANS-based encoding is not worth additional complexity.
This is consistent with what e.g. @fse does - the literal stream is only Huffman compressed, but the significantly
skewed length/offset data is tANS-compressed.

Additionally, we see how for the dataset where ANS-encoding _would_ be useful, the vast majority of benefit
often comes from just a few (usually one, sometimes two) nodes in the Huffman tree.
To exploit that, #PH was extended with _applying ANS-based encoding only for the nodes where it matters_.
This means, that for most datasets, no ANS-overhead is paid, and if it's applied, it's only paid for a small subset of data.

A concrete implementation of which node should be FSE-selected is currently as follows:
- node needs to have at least `PIVCO_FSE_MIN_BITMAP_BYTES` (32 bytes/ 256 bits default)
- node skew needs to be higher than `PIVCO_FSE_MIN_THRESHOLD` (0.625 default)
- fse _benefit_ needs to be better than `PIVCO_FSE_MIN_RATIO` (0.95 default)  with benefit computed as
  `(depth + fse_H) / (depth + 1)`, where *fse_H* is the average bit-cost of fse-encoding (close to _H_, but usually a bit higher).
  The motivation here is that while saving e.g. 0.2 bits for a root-node makes sense, doing it for a
  node at depth 5 (so already 5-bits long) is probably not worth it.

Note, that we know all the above information purely from the symbol frequencies used to construct the Huffman tree,
we do not need to gather any additional data statistics.

When we decide to compress a particular bitmap with ANS, today we use FSE (@fse).
Note, that we compress the bitmap as _bytes_, not as bits.
This means, that for each symbol decoded with FSE, we cover _8 symbols_.
This, combined with applying FSE selectively, is critical to making #PHA efficient.

One non-trivial cost of FSE is creation of decode tables.
To avoid it, we use statically precomputed 50 decode tables for bitmap skew in range (50,51,..,98,99)%.
Then, we simply choose a table based on the symbol skew during encoding/decoding.

Note, we use a _tuned_ version of FSE (_x8y1_), as we found that the default implementation
can be significantly improved for our needs, see @tuning-fse.
See also @fuse-fse-merge for another possible optimization.

== Benefits

* FSE is slow, but we do "1 bit of 8 tokens" at a time.
* only applied for nodes where it actually matters (mostly highly skewed)

== Results

#let fair = csv("data/fair.csv")
#let _na(v) = if v == "na" { [—] } else { [#v] }
#let _dsets = ("proba80", "english", "html_wiki", "prose_pride", "image_jpeg",
               "json_api", "dna_fasta", "chinese_text", "calgary_pic")
#let _engs = ("ph", "pha", "huf0", "fse_x8y1")
#let _body = _dsets.map(d => {
  ([#d],) + _engs.map(e => (
    _na(fair-cell(fair, "m4", d, e, "ratio_op")),
    _na(fair-cell(fair, "m4", d, e, "dec_op")),
  )).flatten()
}).flatten()
#figure(
table(
    columns: 9,
    align: (col, _) => if col == 0 { left } else { right },
    table.header(
    table.cell(rowspan: 2)[*Dataset*],
    table.cell(colspan: 2)[*ph*],   table.cell(colspan: 2)[*pha*],
    table.cell(colspan: 2)[*#h0*], table.cell(colspan: 2)[*fse_x8y1*],
    [ratio], [MB/s], [ratio], [MB/s], [ratio], [MB/s], [ratio], [MB/s],
    ),
    .._body,
),
caption: [M4 fair-bench: compression ratio (higher = better) and decode
            throughput (MB/s, opaque / realistic per-call) per engine, across
            the MAIN distributions. #PH and #h0 are plain Huffman (≈equal
            ratio); *pha* gains ratio from ANS-coded partition bitmaps; the
            standalone *fse_x8y1* reaches the best ratio (full FSE) but is
            the slowest. #h0 is stock `HUF_decompress` (auto-dispatch).
            "—" = not available.],
)<tab-fair-m4>

