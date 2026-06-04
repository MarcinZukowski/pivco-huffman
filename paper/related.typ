#import "conf.typ": PH, he, anote
#import "tab.typ": tab

= Related work <related>

== Entropy coding <entropy>

Entropy-based encoding systems have been intensely researched for many decades,
 with Huffman, arithmetic compression and (recently) ANS-family being the most popular,
 both in research and in applications.
Other well-known approaches include Golomb-Rice (@golomb1966, @rice1971), Elias-Fano (@ottaviano2014pef), Tunstall (@tunstall1967), lightweight integer compression (@for, @lemire2017streamvbyte), and others (@sayood2017).

In this context, #PH can be seen as a performance-focused variant of Huffman.
The ANS application, while useful, is more of an extension to this core idea.

Separately, it would be interesting to see if some of the techniques applied in this paper
 could be used to other methods in this space.
For example, @app-golomb discusses how the _pivoted coding_ approach can be applied
 to Golomb coding.

== Wavelet trees <wt>

#anote()[I'm not smart enough for most of the wavelet trees papers.
They give me headaches. ]

Wavelet trees, introduced in @grossi2003wt, are a popular structure used in
many different applications, typically succinct indexing, full-text indexing,
and even compression (see @ferragina2009myriad).

#PH reuses the idea of a "tree of bitmaps" from wavelet trees, but to the author's knowledge,
most other aspects of the solutions are quite different; see @tab-wavelet for comparison.
Still, there is definitely some interesting overlap, especially around wavelet-tree creation, suggesting
that ideas from wavelet-trees research could be applied to #PH and the other way around.
For example, @dinklage2021jea proposes a _bottom-up building_ of wavelet trees,
and @dinklage2023wt apply SIMD instructions to this problem.

#tab(
  name:    "tab-wavelet",
  columns: 3,
  align:   left,
  stroke:  1pt,
  header: (
    table.cell(align: center)[*Dimension*],
    table.cell(align: center)[*Wavelet Trees*],
    table.cell(align: center)[*#PH*],
  ),
  body: (
    [Core representation],
      [Alphabet tree with node bitmaps],
      [Code tree with node bitmaps],
    [Primary purpose],
      [Indexed sequence representation: access, rank, select, range queries, etc.],
      [Sequential compression/decompression throughput],
    [Aux structures],
      [Usually add rank/select support per bitmap],
      [none],
    [Operations],
      [Navigate query positions through levels],
      [Reconstruct whole dense output stream],
    [Node bitmap constraints],
      [Often must remain rank/select-friendly, e.g. use RRR @rrr2007],
      [Can use decode-friendly encodings, including FSE/ANS],
    [Tree shape],
      [Fixed/balanced, Huffman-shaped, wavelet matrix variants, etc.],
      [Huffman-derived with flat subtrees],
    [Performance target],
      [Query latency/space tradeoff],
      [GB/s-scale sequential decode throughput],
    [Block model],
      [Often whole sequence/static text index],
      [Block codec, streaming possible],
  ),
  rules: (
    ((x: 0), (weight: "bold")),
  ),
  caption: [Comparison of wavelet trees and #PH],
)

== Bit-packing <bitpack>

_Flat-subtrees_ are one of the key performance aspects of #PH.
Their implementation depends heavily on packing/unpacking D-bit integers,
 often called a _Frame-of-Reference_ coding (which additionally applies an offset).
This problem appears in many different areas, including databases and information retrieval.
A lot of work focuses on this problem in its original setting, where values are packed
 contiguously (e.g. @for, @zuk06).
However, other approaches use non-linear data organization allowing them to achieve
 much higher performance (@simdcomp, @fastlanes).

#PH currently uses a relatively straightforward, linear, SIMD-based bit-packing.
Applying techniques from other work in that space could possibly further improve #PH's performance.
