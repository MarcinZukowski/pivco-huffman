= Pivoting Huffman

Example tree, show bits, group them

Show a tree with primitives

Mention Wavelet, @wt.

== Tree Optimizations

=== Merging primitives

=== Constant Key

=== Flat Subtrees

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

