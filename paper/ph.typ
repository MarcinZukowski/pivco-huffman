#import "conf.typ": _html, _pdf, _fmt   // brings the format flags in scope
#include "conf.typ"                      // runs the side effects (style
                                         // injection, page setup, show rules)

#set document(
  title: [PivCo Huffman]
)

#title()

#text(size: 12pt, weight: "bold")[
  Marcin Zukowski
    \
 ]
// ]

#set align(left)

= Abstract
Blablalbasd

#if _html { outline() }

#set heading(numbering: "1.")

#include "intro.typ"
#include "sideways.typ"
#include "updown.typ"
#include "enc.typ"
#include "ans.typ"

= Related work <related>

== Huffman encoding

== ANS/FSE

== Wavelet trees <wt>

= AI disclosure

Anthropic Claude was used extensively
during development of this project, in areas
like coding, testing, automation and research.
It also contributed many small ideas and improvements,
especially around SIMD code.
At the same time, Author declares that the vast majority of the ideas and concepts here are human-invented.

= Summary


