#import "conf.typ": htmlonly, PH, he, mf

= Pivoting Huffman

Following the example from @hj, it should be possible to create a
Huffman decoder using similar principles.
However, for each node in the Huffman tree to have access to relevant
bits from the stream, a different data layout is needed.

#he("myfig")[
  #table(
    columns: (55%, 45%),
    stroke: 0pt,
    align: center,
    [#figure(
      image("figures/pivot-bitmaps.svg"),
      caption: [
        Example of pivoting a Huffman-encoded string.
        All bits sharing the same prefix (noted in quotes)
        are grouped together as a bitmap.
        Color-coded letters with subscript denote
        which-letter which-bit combinations.
        *\** marks a code-terminal bit.
      ]
    )<fig-pivot>
    ],
    [#figure(
      image("figures/pivot-tree.svg"),
      caption: [
      Example of a Huffman tree with "pivoted" data traversing
      it, reusing the previous example.
      ]
    )<fig-pivot-tree>
    ],
  )
]

// #mf("pivot-bitmaps")

@fig-pivot presents how the encoded word "huffman" can be presented differently.
Instead of a code-after-code stream, we divide all the bits stream bits
by their (possibly empty) prefix.
@fig-pivot-tree shows how this layout maps onto the Huffman tree when decoding
data.
Each node receives all the bits of the codes that pass through it, and navigates
these codes to its children, where another bitmap is used for the next step.

While logically this representation contains the same information, since bitmaps
are typically stored byte-aligned, it might reduce in a marginally worse compression
ratio due to byte-rounding.
However, for non-trivial datasets this overhead acceptable if this approach provides other benefits.

Note, that this basic tree representation is equivalent to
_wavelet trees_ (see @wt), specifically _Huffman-shaped wavelet trees_ (e.g. @dinklage2023wt).
However, with a very different focus, use cases, and structural changes to the tree,
we do not use this term in this paper.

#htmlonly[
While working on #PH I did a lot of literature review, and for the longest time couldn't find
anything like it.

Close to the end of the research/experimental work, Claude found Wavelet trees.

Initially I was in panic (a few weeks of life lost?), but the deeper review showed that
it's really quite different.

So here we go. I will still call it #PH. Sue me :)
]

== Naive implementation

With a defined Huffman tree, and data stored in per-node bitmaps, we can traverse
the tree top-down, and apply two operations:

*`partition(bitmap, indices) => (indices_left, indices_right)`* -- applied for
all internal nodes.
Takes a list of indices (positions in the output stream), and divides it
based on the bitmap into left and right indices for its subtrees.
Note, a special `partition_root` version can be used in the root node,
as its list of indices is  the complete input.

The `partition` primitive can be expressed naively with:
```
for (i: 0..len(indices)):
  bit = bitmap[i];
  if bit:
    indices_right.append(indices[i])
  else:
    indices_left.append(indices[i])
```

*`scatter(output, indices, symbol)`* -- fills all positions in the output stream
with a given symbol.

```
for (i: 0..len(indices)):
  output[indices[i]] = symbol
```

We measured the decoding performance of such an implementation on Apple M4 CPU,
and, as expected, the performance is very sub-par.

#figure(
  table(
      columns: (auto, auto, auto, auto),
      inset: 6pt,
      align: (left, right, right, right, right, right),
      table.header(
        [*distribution*],
        [*ph-naive*\ (GB/s)],
        [*huf0_x2*\ (GB/s)],
        [*ph / huf0_x2*],
      ),
      [proba80],     [0.589], [1.457], [0.40×],
      [prose_pride], [0.436], [1.603], [0.27×],
  ),
  caption: [Naive #PH implementation performance.]
)

There are two main reasons for this:

- for each decoded symbol, we perform multiple operations: `len(symbol)` times
  `partition` nodes + 1 `scatter`
- the decoding partitions as written are not effficient

In the following two sections we'll discuss how to address both problems.

== Tree Optimizations

A naive Huffman tree discussed before suffers from a large number
of operations per byte.
In this section we'll use a test string `coconut-papaya` to demonstrate
a number of techniques that can bring that number down significantly.

In @treeopt-naive we see our starting point - a basic tree with
2 kinds of primitives (marked in orange boxes), and 4.071 operations per output byte
(weighted by symbol frequency).
@treeopt-symbols translates symbols used in figures in this section
to the actual compute primitives.

#table(
  columns:2,
  stroke: 0pt,
  [
    #figure(
      mf("treeopt-naive"),
      caption: [Decoding-strategy for a naive Huffman tree]
    )<treeopt-naive>
  ],
  [
    #figure(
      table(
        columns: 2,
        [`P`],  [`partition` - split indices into left/right based on bitmap],
        [`PR`], [`partition_root` - like `partition` but for the root node],
        [`S1`], [`scatter` - scatters a single symbol into output],
        [`S2`], [`scatter_two` - scatters two symbols into output],
        [`PH`], [`partitions_half` - like `partition`, but produces only one output],
        [`C`], [_not a primivite_ - marks the "constant", top-frequency key]
      ),
    caption: [Primitive symbols used in figures in this Section]
    )<treeopt-symbols>
  ],
)

=== Merging leaves

One simple approach of reducing the number of operations is
to avoid the last-level `partition`, and simply fill
all input indices with the symbol based on the bitmap.

This results in a *`scatter_two(output, indices, bitmap, symbol0, symbol1)`* primitive:
```
for (i: 0..len(indices)):
  output[indices[i]] = bitmap[i] ? symbol1 : symbol0
```
@treeopt-fuse shows the benefit in reduced operations per node.

=== Top symbol optimization

One of the problems of our decoding primitives is writing into non-contiguous
positions in the output, presenting challenges for modern CPUs and memory subsystems.

We can mitigate it by avoiding this completely for the _most frequent symbol_,
by simply prefilling the entire output with `memset` before decoding.
Then, that symbol never needs to be processed during the tree traversal.
Note that `memset` is 1-2 orders of magnitude faster than our primitives, so that
cost is negligible.

@treeopt-constant shows the tree with this optimization applied.
Note, that as a result we introduce a new operation called `PH` - `partition_half`,
similar to `partition`, but only producing one of the output index lists.

#table(
  columns:2,
  stroke: 0pt,
  [
    #figure(
      mf("treeopt-fuse"),
      caption: [Merging `partition` and `scatter` into `scatter-two`]
    )<treeopt-fuse>
  ],
  [
    #figure(
      mf("treeopt-constant"),
      caption: [Pre-filling the most frequent "constant" symbol]
    )<treeopt-constant>
  ],
)

=== Flat Subtrees

Huffman trees often contain entire subtrees where all the symbols
share the same length.
In our example, 4 right-most (`-ntu`) nodes form such a subtree.

We can decode an entire subtree like that with a single operation
*`scatter_flat_-D(output, indices, bitmap, symbols)`*, where `D` represents
the depth of the subtree.

Note, that for this, the input `bitmap` is not _binary_, but _D-ary_, with
bits packed contiguously.
Also, note that `scatter-two` is a special case of this approach, with _D=1_.

```
code_indices = unpack(bitmap, D);
for (i: 0..len(indices)):
  output[indices[i]] = symbols[code_indices[i]]
```

#table(
  columns:2,
  stroke: 0pt,

  [
    #figure(
      mf("treeopt-flat"),
      caption: [Detecting "flat" subtrees]
    )<treeopt-flat>
  ],
  [
    #figure(
      mf("treeopt-opt"),
      caption: [Optimizing "flat" subtrees]
    )<treeopt-opt>
  ],
)

=== Non-Canonical Subtrees

Looking at @treeopt-flat, we can see that while the `-ntu` symbols
benefit from the "flat subtrees" strategy,  we also have `copy` symbols,
which share the same code lenghts, but are not decoded together.
This is because the canonical Huffman trees produced a tree of this particular
shape.

We can reorganize the Huffman tree to make it more amenable to the "flat subtree"
optimization by making sure that codes with the same length are grouped as much as
possible.
To achieve that, after building an initial Huffman tree, we sort the codes
by their length.
Then, within each length-group, we combine the largest _power of two_ number of nodes
into a single node with a combined frequency.
We repeat the process, with one length-group possibly creating multiple such nodes (of different depth).

The result is a new, (usually) non-canonical Huffman tree, with the exact same average
code withs, but a different shape.
@treeopt-opt shows how applying this strategy allows the `copy` nodes to be processed
together, further reducing ops/byte.

XXX-maybe: Also mention sorting by frequency to give performance benefit to
same-length more-frequent codes.

/*

#he("myfig")[
  #figure(
    kind: image,
    table(
      columns: 3,
      stroke: 0pt,
      align: center,
      image("figures/treeopt-constant.svg"),
      image("figures/treeopt-flat.svg"),
      image("figures/treeopt-opt.svg"),
      [(a)], [(b)], [(c)],
    ),
    caption: [
      Impact of using "flat trees" on the number of decoding operations.
      By flattening subtrees, and then optimizing the tree structure,
      the average ops/byte is reduced from 3.071 to 2.286.
    ]
  )
]
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

// Example to use for the D-flat tree reorg figure: "coconut papaya".
// 8 distinct chars, frequencies (c=2, o=2, n=1, u=1, t=1, p=2, a=3, y=1) with
// 13 symbols total — small enough to draw at full size, the canonical Huffman
// tree has a clear D=2 flat candidate after the flat-aware restructurer rolls
// same-length leaves into a power-of-2 chunk.  Pairs well with the
// tree_viz.html screenshot.

== Computing Primitives

== Results

