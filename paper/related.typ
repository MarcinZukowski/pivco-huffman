#import "conf.typ": PH, he, anote

= Related work <related>

== Huffman encoding

== Wavelet trees <wt>

#anote()[I'm not smart enough for most of the wavelet trees papers.
They give me headaches. ]

Wavelet trees, introduced in @grossi2003wt, are a popular structure used in
many different applications, typically succint-indices, full-text indexing,
and even compression (see @ferragina2009myriad).

#PH reuses the idea of a "tree of bitmaps" from wavelet trees, but to author's knowledge,
most other aspects of the solutions are quite different, see @tab-wavelet for comparison.
Still, there is definitely some interesting overlap, especially around wavelet-tree creation, suggesting
that ideas from wavelet-trees research could be applied to #PH and the other way around.
For example, @dinklage2021jea proposes a _bottom-up building_ of wavelet trees,
and @dinklage2023wt apply SIMD instructions to this problem.

#he("tab-wavelet",
  style: "
  .tab-wavelet td { text-align: left; }
  .tab-wavelet td:nth-child(1) { font-weight: bold; }

  "
)[
#figure(
table(
  stroke: 0pt,
  align: center,
  columns: 3,
  table.header(
    [*Dimension*],
    [*Wavelet Trees*],
    [*#PH*],
  ),
  [Core representation],
    [Alphabet tree with node bitmaps],
    [Code tree with node bitmaps],
  [Primary purpose],
    [Indexed sequence representation: access, rank, select, range queries, etc.],
    [Sequential compression/decompression throughput],
  [Aux structures],
    [Usually add rank/select support per bitmap],
    [none (unless ANS-compressed)],
  [Operations],
    [Navigate query positions through levels],
    [Reconstruct whole dense output stream],
  [Node bitmap constraints],
    [Often must remain rank/select-friend, e.g. use RRR @rrr2007],
    [Can use decode-friendly encodings, including FSE/ANS],
  [Tree shape],
    [Fixed/balanced, Huffman-shaped, wavelet matrix variants, etc.],
    [Huffman-derived with flat subtrees],
  [Performance target],
    [Query latency/space tradeoff],
    [GB/s-scale sequential decode throughput],
  [Block model],
    [Often whole sequence/static text index],
    [Block codec, streaming possible]
),
caption: [Comparision of wavelet trees and  #PH]
)<tab-wavelet>
]

== ANS/FSE

== Bit-packing

== ?? bitmap compression

TODO: check golomb coding