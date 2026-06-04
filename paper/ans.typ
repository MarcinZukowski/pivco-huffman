#import "conf.typ": _html, _pdf, _fmt, anote, setup, PH, he, sym, PHA, fair-cell, h0, mf
#import "style.typ": colors-tab
#import "tab.typ": tab

= Breaking the bit-barrier <ans>

Huffman encoding, while ubiquitous, has one key limitation:
its code lengths are constrained to whole bits.
That means, for some distributions, it is further from entropy-optimal
than desired.

A well-known solution to this problem is arithmetic coding (@arithmetic); however,
it has not been popular due to its performance and patent controversies.
Luckily, Jarek Duda proposed _ANS-based encoding_ (@dudaans),
which solves both problems.
It led to two main algorithms: tabled-ANS (*tANS*) and range-ANS (*rANS* @encodesu1920),
both allowing codes to have lengths close to entropy-optimal.

We will focus on tANS, which is used by FSE (@fse) and Oodle TANS library (@oodle),
 both related to the Huffman implementations we discussed in @sota.

The typical tANS decoding is actually quite similar to an optimized
Huffman decoding routine from @sota:

```c
  t = decoding_table[state];
  state = t.newX + read_bits(t.numBits);  //state transition
  emit_symbol(t.symbol);                  //decoded symbol
```

While similar, the critical difference is that we see a `state` variable which
is carried through the iterations and mutated depending on the data.
Also, the `decoding_table` is constructed such that, for the same symbol,
a different number of bits may be consumed depending on the current `state`.

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
  placement: top,
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

#he("gridtable")[
  #table(
    columns: (50%, 50%),
    stroke: 0pt,
    align: center,
    [#figure(
      mf("skew-calgary", width:70%),
      caption: [Skew visualization for *calgary_pic* (top part of the tree only)]
    )<fig-skew-calgary-pic>
    ],
    [#figure(
      mf("skew-dna-fasta"),
      caption: [Skew visualization for *dna_fasta* (top part of the tree only)]
    )<fig-skew-dna-fasta>
    ],
  )
]

@fig-skew-calgary-pic and @fig-skew-dna-fasta show visualization of the
top parts of the tree for *calgary_pic* and *dna_fasta* datasets.
For each tree node, we report left/right skew, and the percentage of data
covered by a given subtree.
Bar color reflects skew severity (red - highly skewed).

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

== #PHA implementation

The analysis above suggests that for most datasets applying ANS-based encoding is not worth the additional complexity.
This is consistent with what e.g. @fse does - the literal stream is only Huffman compressed, but the significantly
skewed length/offset data is tANS-compressed.

Additionally, we see how for datasets where ANS-encoding _would_ be useful, the vast majority of benefit
often comes from just a few (usually one, sometimes two) nodes in the Huffman tree.
To exploit that, #PH was extended with _selective ANS encoding — applied only to the nodes where it matters_.
This means that for most datasets no ANS overhead is paid, and when it is applied, it is only paid for a small subset of the data.

A concrete implementation of which node should be FSE-selected is currently as follows:
- node needs to have at least `PIVCO_FSE_MIN_BITMAP_BYTES` (32 bytes/ 256 bits default)
- node skew needs to be higher than `PIVCO_FSE_MIN_THRESHOLD` (0.625 default)
- fse _benefit_ needs to be better than `PIVCO_FSE_MIN_RATIO` (0.95 default) with benefit computed as
  `(depth + fse_H) / (depth + 1)`, where *fse_H* is the average bit-cost of fse-encoding (close to _H_, but usually a bit higher).
  The motivation here is that while saving e.g. 0.2 bits for a root-node makes sense, doing it for a
  node at depth 5 (so already 5-bits long) is probably not worth it.

Note that we know all of the above information purely from the symbol frequencies used to construct the Huffman tree;
we do not need to gather any additional data statistics. We _know_ when ANS will help.

When we decide to compress a particular bitmap with ANS, today we use FSE (@fse).
Note that we compress the bitmap as _bytes_, not as bits.
This means that for each symbol decoded with FSE, we cover _8 symbols_ of the original bitmap.
This, combined with applying FSE selectively, is critical to making #PHA efficient.

One non-trivial cost of FSE is creation of decode tables.
To avoid it, we use statically precomputed 50 decode tables for bitmap skew in range (50,51,..,98,99)%.
Then, we simply choose a table based on the symbol skew during encoding/decoding.

An interesting aspect of this decision is that the tables above are built for _bytes_
 constructed using a random _bit_ distribution.
Depending on the actual distribution of bits, this can result
 in compression efficiency lower than if we actually
 built the table for a specific dataset.
For example, let us take a collection of 300 `0xFF` and 100 `0x00` values.
If the decoding table were custom-built, these values
 would take 75% and 25%, respectively, of the frequencies.
However, since the pre-built partitions assume random bit distribution,
 these symbols' expected frequencies
 will be much lower, resulting in more bits assigned to them during encoding.
This shows that our approach might not be able to exploit some order-based compression ratio
 opportunities in the data.

Note, we use a _tuned_ version of FSE (_x8y1_), as we found that the default implementation
can be significantly improved for our needs, see @tuning-fse.
See also @fuse-fse-merge for another possible optimization.

== Benefits

This approach to FSE application in #PHA has the following benefits:
- FSE is slower than Huffman, but since each FSE-symbol we decode covers 8 Huffman-symbols from our main tree, we pay only 1/8th of the cost
  per bitmap.
- It can be applied _only_ to nodes where it actually matters (mostly highly skewed)
- Compression ratio vs performance can actually be _tuned_ (slightly) depending on the actual FSE-triggering
  strategy
- There is no FSE table construction
- The FSE table selection can be done for every decompression block separately (8KB).
  This allows exploiting locally-optimum distributions.
  Stock FSE decides on the decoding table every 128KB, and so it will not exploit these local properties.

== Results

#let fair = csv("data/fair.csv")
#let _na(v) = if v == "na" { [—] } else { [#v] }
#let _dsets = ("proba80", "english", "html_wiki", "prose_pride", "image_jpeg",
               "json_api", "dna_fasta", "chinese_text", "calgary_pic")
#let _engs = ("ph", "pha", "huf0", "oo_tans")
#let _body = _dsets.map(d => {
  ([#d],) + _engs.map(e => (
    _na(fair-cell(fair, "m4", d, e, "ratio_op")),
    _na(fair-cell(fair, "m4", d, e, "dec_op")),
    _na(fair-cell(fair, "c8i", d, e, "dec_op")),
  )).flatten()
}).flatten()

#tab(
  name:    "tab-fair-m4",
  columns: 13,
  align: (col, row) => if row < 2 { center }
                       else if col == 0 { left }
                       else { right },  header_rows: 2,
  placement: top,
  inset: (x: 3.5pt),
  header:  (
    table.cell(rowspan: 2)[*Dataset*],
    table.cell(colspan: 3)[*PH*],
    table.cell(colspan: 3)[*PH+ANS*],
    table.cell(colspan: 3)[*#h0*],
    table.cell(colspan: 3)[*oo-tans*],
    [ratio], [M4 \ MB/s], [c8i \ MB/s],
    [ratio], [M4 \ MB/s], [c8i \ MB/s],
    [ratio], [M4 \ MB/s], [c8i \ MB/s],
    [ratio], [M4 \ MB/s], [c8i \ MB/s],
  ),
  body:    _body,
  rules: (
    ((x: 0),                (align: left)),
    ((y: "h"),              (align: center)),
    ((x: (4,5,6)),           (weight: "bold")),     // pha ratio + MB/s columns
    ((y: 0),                (style: "italic")),    // proba80 — skew-heavy
    ((y: 6),                (style: "italic")),    // dna_fasta — skew-heavy
    ((y: 8),                (style: "italic")),    // calgary_pic — skew-heavy
    ((y: 8, x: (4, 10, 13)),  (color: red)),         // calgary ratio cells
    ((x: (1,2,3)),            (fill: colors-tab.ph)),
    ((x: (4,5,6)),            (fill: colors-tab.pha)),
    ((x: (7,8,9)),            (fill: colors-tab.huf0)),
    ((x: (10,11,12)),           (fill: colors-tab.oo_tans)),
  ),
  caption: [#PHA benchmark: compression ratio (higher = better) and decode
            throughput (MB/s) on M4 and c8i. *PH* and #h0 are plain Huffman (≈equal
            ratio); *PHA* improves ratio from ANS-coded partition bitmaps.
            Skew-heavy datasets in _italic_.
            "Calgary" compression ratio in #text(red)[red].
            Huff0 refuses to compress the _image_jpeg_ dataset.],
)

#figure(
  placement: top,
  [
    #image("plots/dec-bw-m4.svg", width: 100%)

    #image("plots/dec-bw-c8i.svg", width: 100%)
  ],
  caption: [Decode throughput per engine, M4 (top) and c8i (bottom) — the bandwidth
    columns of @tab-fair-m4, with the addition of *FSE x8y1* ],
)<fig-dec-bw>

@tab-fair-m4 compares the #PH (*PH*) and #PHA (*PHA*) performance with Huff0 and Oodle's TANS library.
@fig-dec-bw visualizes these results, additionally including FSE.
We see that for non-skewed datasets, *PH* and *PHA* achieve the same performance, but for skewed datasets
*PHA* detects an opportunity to _selectively_ apply FSE - this brings the compression ratio close to full FSE,
while still achieving significantly higher decode performance.
Interestingly, for _dna_fasta_ we see that the FSE performance impact is relatively low,
 as FSE is applied to a non-root node covering only 25% of the symbols.
Finally, the compression ratio of the _calgary_ dataset (a scanned text-on-white image)
 showcases the impact of #PHA utilizing locally-optimal decompression tables.
#footnote[The author by no means suggests #PHA is better than FSE - it just occasionally has this slightly unexpected property]
