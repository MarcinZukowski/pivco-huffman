// import variables, functions
#import "conf.typ": _html, _pdf, _fmt, anote, setup, PH
// run the side effects (style injection, page setup, show rules)
#include "conf.typ"

#show: setup

#set document( title: [PivCo-Huffman] )

#title()
#set align(center)
#text(size: 12pt, weight: "bold")[
  Marcin Żukowski
    \
 ]
#text(size: 10pt)[
  _v.0.9, built on #datetime.today().display()_
]

#set align(left)
#set par(justify: true)

= Abstract

Huffman encoding has been an _enduring_ technique for 70+ years,
ubiquitous in compression algorithms since its invention.
In this paper we propose a new approach to Huffman
coding, based on a data structure from _wavelet trees_.
The resulting _pivot-coded Huffman_ (*#PH*) enables high-performance
SIMD-friendly encoding and decoding operations.
In our tests #PH consistently outperforms
 state-of-the-art Huffman codecs
 in decoding throughput.
Additionally, we show how ANS-coding can be _selectively_
 applied to _skewed_ nodes in this structure,
 yielding compression ratios approaching ANS codecs,
 while preserving very high decompression speeds.

#if _html { outline() }

#set heading(numbering: "1.")

#anote[
This document is mostly a verbatim copy of the arXiv paper.

However, in boxes like this you'll see some additional author's
thoughts that might not be fitting a scientific paper.

The HTML version has some nice _quality-of-life_ features:
- clicking the tree images takes the reader to a visualizer
- simd code has intrinsics tooltips via #link("https://simd.dev")[simd.dev]
- active ToC
]

#include "intro.typ"
#include "sideways.typ"
#include "bottom-up.typ"
#include "enc.typ"
#include "ans.typ"
#include "cao.typ"
#include "related.typ"


= Conclusions

In this paper we presented #PH, a novel approach to Huffman-compression, based on the structure from _wavelet trees_.
The main contributions are:
- adapting the bitmap format of _Huffman-shaped wavelet trees_ into a sequential block-compression wire format
- compression-focused tree-structure optimizations allowing better encoding/decoding performance
- novel "bottom-up" decoding approach, eliminating the _scatter_ problem of the more natural top-down approach
- highly performant SIMD primitives for both encoding and decoding operations
- the concept of _selective_ application of ANS-based encoding in the Huffman tree to further improve its compression ratio

The resulting approach, while slightly slower on the encoding side,
 in our tests consistently beats the decoding speeds of the state-of-the-art solutions by a large margin.
It also provides compression ratios approaching the ANS-based methods at significantly better speeds.

== AI disclosure

Anthropic Claude was used extensively during development of this project, especially in areas
like coding, testing, automation and research.
It also contributed many small ideas and improvements, especially around SIMD code.
At the same time, Author declares that the vast majority of the key ideas and concepts here are human-invented.
This document is purely human-written (except for spellcheck etc.).

== Acknowledgments

Author would like to thank Fabian "Ryg" Giesen for his help with Oodle and his comments on an early draft of the paper.

#bibliography("refs.bib")

#include "appendix.typ"
