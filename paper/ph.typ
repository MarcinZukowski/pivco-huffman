// import variables, functions
#import "conf.typ": _html, _pdf, _fmt, htmlonly, setup, PH
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
  _v.0.1, built on #datetime.today().display()_
]

#set align(left)

= Abstract
Blablalbasd

#if _html { outline() }

#set heading(numbering: "1.")

#htmlonly[
This document is mostly a verbatim copy of the arXiv paper.

However, in boxes like this you'll see some additional thoughts that might not be fitting
a scientific paper.

Oh well, maybe one day reviewers will have a sense of humor...
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

== Open questions
While interesting and possibly promising, PivCo-Huffman still has some problems and exposes multiple improvement opportunities.

- getting a performant implementation in languages that do not expose SIMD is very hard
- due to its nature, it doesn't work well for very small datasets
- Huffman encoding is rarely applied by itself, so integrating it with an actual compression library (like @lz4) is an interesting option
- further improvements of compute primitives are surely possible
- porting  #PH to GPUs and possibly FPGAs is a natural next step

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
