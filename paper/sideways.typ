#import "conf.typ": htmlonly, PH, he, mf

= Pivoting Huffman

#he("myfig")[
  #table(
    columns: (55%, 45%),
    stroke: 0pt,
    align: center,
    figure(
      image("figures/pivot-bitmaps.svg"),
      caption: [
        Example of pivoting a Huffman-encoded string.
        All bits sharing the same prefix (noted in quotes)
        are grouped together as a bitmap.
        Color-coded letters with subscript denote
        which-letter which-bit combinations.
        Asterisk denotes a terminal bit.
      ]
    ),
    figure(
      image("figures/pivot-tree.svg"),
      caption: [
      Example of a Huffman tree with "pivoted" data traversing
      it, reusing the previous example.
      ]
    ),
  )
]


#mf("pivot-bitmaps")

Example tree, show bits, group them

Show a tree with primitives

Mention Wavelet, @wt.

#htmlonly[
While working on #PH I did a lot of literature review, and for the longest time couldn't find
anything like it.

Close to the end of the research/experimental work, Claude found Wavelet trees.

Initially I was in panic (a few weeks of life lost?), but the deeper review showed that
it's really quite different.

So here we go. I will still call it #PH. Sue me :)
]

== Naive implementation

== Tree Optimizations

=== Merging primitives

=== Constant Key

=== Flat Subtrees

#he("myfig")[
  #figure(
    kind: image,
    table(
      columns: 3,
      stroke: 0pt,
      align: center,
      image("figures/flattree-canon.svg"),
      image("figures/flattree-flat.svg"),
      image("figures/flattree-opt.svg"),
      [(a)], [(b)], [(c)],
    ),
    caption: [
      Impact of using "flat trees" on the number of decoding operations.
      By flattening subtrees, and then optimizing the tree structure,
      the average ops/byte is reduced from 3.071 to 2.286.
    ]
  )
]
/*
#html.elem("div", attrs: (class: "myfig"))[
  #figure(
    kind: image,
    table(
      columns: 3,
      align: center,
      image("figures/flattree-canon.svg"),
      image("figures/flattree-flat.svg"),
      image("figures/flattree-opt.svg"),
      [(a)], [(b)], [(c)],
    ),
    caption: [
      Impact of using "flat trees" on the number of decoding operations.
      By flattening subtrees, and then optimizing the tree structure,
      the average ops/byte is reduced from 3.071 to 2.286.
    ]
  )
]
*/

=== Non-canonical Subtrees

Show a canonical tree vs optimized trees, number of ops

Also mention sorting by frequency to give performance benefit to
same-length more-frequent codes.

// Example to use for the D-flat tree reorg figure: "coconut papaya".
// 8 distinct chars, frequencies (c=2, o=2, n=1, u=1, t=1, p=2, a=3, y=1) with
// 13 symbols total — small enough to draw at full size, the canonical Huffman
// tree has a clear D=2 flat candidate after the flat-aware restructurer rolls
// same-length leaves into a power-of-2 chunk.  Pairs well with the
// tree_viz.html screenshot.

== Computing Primitives

== Results

