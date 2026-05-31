#import "conf.typ": PH, mf, he, fair-cell, h0

= Encoding <encoding>

Data encoding for #PH is relatively straightforward - it, naturally, uses the same tree shape that we build for decoding,
and follows the same pattern of tree traversal with high-performance SIMD primitives.

#he("gridtable")[
  #table(
    columns: (50%, 50%),
    stroke: 0pt,
    align: center,
    [
      #figure(
        mf("enc-ops"),
        caption: [Encoding tree operations (_optimized flat trees_ off)]
      )<fig-enc-ops>
    ],
    [
      #he("enc-symbols")[
      #figure(
        table(
          columns: 2,
          align: (center, left),
          table.header([*Symbol*], [*Explanation*]),
          [`EPF`], [`enc_partition_full` - create a bitmap, split codes into left/right outputs],
          [`EPL`], [`enc_partition_left` - right child is a leaf, only produce bitmap+left codes],
          [`EPR`], [`enc_partition_right` - left child is a leaf, only produce bitmap+right codes],
          [`EPN`], [`enc_partition_none` - both children are leaves, only produce bitmap],
          [`PKN`], [`packN` - pack codes into N-bits sequence, used for flat subtrees],
        ),
        caption: [Primitives used in encoding]
      )<enc-symbols>
      ]
    ]
  )
]

@fig-enc-ops shows operations for an example encoding tree, and @enc-symbols lists operations
used in that phase.
Note extreme similarity to @treeopt-symbols from @sideways.
In fact, `enc_partition_*` operations are functionally equivalent to first building a bitmap
from symbols, and then applying the `partition` operations used in top-down decoding.
A small difference is that the elements we partition are not 16-bit _indices in the output_,
 but 16-bit _codes_. Still, the primitives are the same.

#let fair = csv("data/fair.csv")
#let _na(v) = if v == "na" { [—] } else { [#v] }
#let _dsets = ("proba80", "english", "html_wiki", "prose_pride", "image_jpeg",
               "json_api", "dna_fasta", "chinese_text", "calgary_pic")
#let _engs = ("huf0", "oo_huff")
#let _body = _dsets.map(d => {
  ([#d],) + (
      fair-cell(fair, "m4", d, "ph", "enc_op"),
      fair-cell(fair, "c8i", d, "ph", "enc_op"),
      fair-cell(fair, "m4", d, "ph", "enc_pb"),
      fair-cell(fair, "c8i", d, "ph", "enc_pb"),
    ) + _engs.map(e => (
    fair-cell(fair, "m4", d, e, "enc_op"),
    fair-cell(fair, "c8i", d, e, "enc_op"),
    )).flatten()
//    + (
//    fair-cell(fair, "m4", d, "oo-huff", "enc_op"),
//    fair-cell(fair, "c8i", d, "oo-huff", "enc_op")
//    ).flatten()
}).flatten()

#he("tab-enc")[

#figure(
table(
    columns: 9,
    align: (col, _) => if col == 0 { left } else { right },
    table.header(

    table.cell(rowspan: 3)[*Dataset*],
    table.cell(colspan: 4)[*#PH*],
    table.cell(colspan: 2, rowspan: 2)[*#h0*],
    table.cell(colspan: 2, rowspan: 2)[*oo-huffman*],

    table.cell(colspan: 2)[*end-to-end*],
    table.cell(colspan: 2)[*prebuilt tree*],

    [M4], [c8i], [M4], [c8i], [M4], [c8i], [M4], [c8i],

    ),
    .._body,
),
caption: [Encoding performance on M4 and c8i (MB/s).
For #PH, we report both "end-to-end" and "prebuilt tree" results],
)<tab-enc>]

#figure(
  [
    #image("plots/enc-bars-m4.svg", width: 100%)

    #image("plots/enc-bars-c8i.svg", width: 100%)
  ],
  caption: [Encoding throughput, M4 (top) and c8i (bottom) — the data of @tab-enc.
    #PH end-to-end (`ph-op`) and prebuilt-tree (`ph-pb`) vs #h0 and Oodle-Huffman.
    On c8i the `ph-pb` proba80 bar exceeds the axis cap and is clipped (true value labelled).],
)<fig-enc-bars>

@tab-enc shows the encoding performance on various datasets and hosts.
End-to-end results show a fair comparison with other solutions.
#PH's performance is hindered here by the Huffman-code creation time,
dominated by symbol frequency counting.
The "prebuilt-tree" results show the actual encoding performance,
showing e.g. how highly-skewed datasets can achieve
very high "raw" encoding performance.

One compression ratio / performance tradeoff we have made
is that every bitmap (which needs it) stores additional information
about the length of its right child (equivalent to the number of 1s in the bitmap).
This is necessary during decoding, as we need to know where to locate the
 bitmaps of the children nodes to decode them.
We could slightly improve the compression ratio by not including this information
and recalculating it during decoding, at the cost of reduced decompression speed.
