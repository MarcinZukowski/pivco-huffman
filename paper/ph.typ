// import variables, functions
#import "conf.typ": _html, _pdf, _fmt, anote, setup, PH
// run the side effects (style injection, page setup, show rules)
#include "conf.typ"

#show: setup

#set document( title: [PivCo-Huffman] )

#title()
#set align(center)
#text(size: 12pt, weight: "bold")[
  Marcin Zukowski
    \
 ]
#text(size: 10pt)[
  _v.0.5, built on #datetime.today().display()_
]

#set align(left)

= Abstract
Huffman encoding has been an _enduring_ technique for 70+ years,
ubiquitous in compression algorithms since its invention.
In this paper we propose a new approach to Huffman
coding, based on a data structure from _wavelet trees_.
The resulting _pivot-coded Huffman_ (*#PH*) allows utilizing high-performance
encoding and decoding operations, providing
significant performance improvements.
Additionally, we show how ANS-coding can be _selectively_
applied to this structure when it is actually useful.
The end result is a novel algorithm that consistently beats
the decoding performance of state-of-the-art Huffman codecs,
while providing compression ratios close to ANS-based solutions.

#if _html { outline() }

#set heading(numbering: "1.")

#anote[
This document is mostly a verbatim copy of the arXiv paper.

However, in boxes like this you'll see some additional author's
thoughts that might not be fitting a scientific paper.

The HTML version has some nice QoL features:
- clicking the tree images takes the reader to a visualizer
- simd code has intrinsics tooltips via simd.dev
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

== Contributions
In this paper


== AI disclosure

Anthropic Claude was used extensively during development of this project, especially in areas
like coding, testing, automation and research.
It also contributed many small ideas and improvements, especially around SIMD code.
At the same time, Author declares that the vast majority of the key ideas and concepts here are human-invented.
This paper is purely human-written (except for spellcheck etc.).

== Acknowledgments

Author would like to thank Fabian Giesen for his help with Oodle.

#bibliography("refs.bib")

#include "appendix.typ"
