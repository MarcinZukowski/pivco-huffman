#import "conf.typ": anote, PH, he, mf, sym, pick-cols, todo

= Going Bottom-Up<bottom-up>

In @sideways we process the tree *top-down*,
 which is a very natural way, directly translating to the "textbook" Huffman decoding.
Still, while achieving decent performance,
 it is heavily penalized by the high number of scattered writes.

When exploring solutions to this problem, we looked at
an idea of first traversing the tree to identify index-symbol positions,
then merging index positions back into a contiguous sequence,
and then using that for a scatter-free writing of symbols.
During the merging phase, we would need to carry per-position _symbols_
together with the _indices_ all the way back to the root.
That particular idea was quickly dropped due to a high cost of merging added to already significant
cost of tree traversal.

However, it led to another realization.
We can use the idea of *bottom-up* merging without the first partitioning stage at all.
This idea led to a new variant of #PH, which is the actual proposed solution.
The process is as follows:

Every tree node produces the values for all output positions with symbols
that traverse a given node - these were the indices in the top-down traversal.
For leaves, all these values are constants.
For non-leaf nodes, we can construct the output using children symbols, and the same
bitmaps we used for *bitmap-based partitioning*, but now using *bitmap-based merging*.
This process proceeds all the way to the top, resulting in the final sequence
of codes equal to the complete expected output.
#footnote[Note that a similar symmetry of _partitioning_ vs _merging_ can be found
in other places, e.g. sorting or joins in databases].

This approach has some very interesting properties:
- leaf nodes don't require any processing, as they just produce a constant value.
  This is different than in the top-down approach, where we had to apply a `scatter` primitive
- inputs and output of each node are _dense_, alleviating the scatter problem.
- most of the tree optimizations from @naive can be applied

#figure(
  mf("bu-tree"),
  caption: [Bottom-up traversed "huffman" tree (_flat trees_ off). See how each node produces a dense list of symbols.]
)<fig-bu-tree>

This results in the approach presented in @fig-bu-tree.
Note, that this tree is symmetrical to @fig-pivot-tree, with just data travelling in the opposite direction,
 and different data flowing with the bitmaps (symbols vs indices).

== Bottom-up tree operations

#figure(
  mf("bu-ops"),
  caption: [Upside-down tree operations (_optimized flat trees_ off)]
)<fig-bu-ops>

#figure(
  table(
    columns: 3,
    align: (center, center, left),
    table.header([*Top-down equivalent*], [*Bottom-up operation*], [*Explanation*]),

    [`P`],  [`M`], [`merge` is symmetrical to `partition`],
    [`PR`], [`M`], [`merge` for the root node is identical to other cases],
    [`PH`], [`MC`], [`merge_constant` - special `merge` variants where one input is constant],
    [`C`],  [--], [Note that in bottom-up multiple leaves can be "constant"],
    [`S1`], [--],  [No operation needed for leaves when going bottom up],
    [`S2`], [`M2`],   [`merge_two` - merges two constant symbols into output],
    [`SFD`], [`MFD`], [`merge_flat_D` - merges 2^D constant symbols into output],
  ),
caption: [Primitives used in bottom-up processing and their top-down equivalents]
)<bu-symbols>

@fig-bu-ops shows the example tree we used before, but this time with operations used for the bottom-up processing.
Again, it is straightforwardly symmetrical to @treeopt-flat.

Note, that the _most frequent symbol_ optimization from the top-down approach is not applicable
when going bottom-up.
This is because the final merge operation in root will always write the entire output sequence anyway.
The other tree optimizations apply directly.

== Bottom-up primitives

Unlike top-down processing, bottom-up processing results in only one family of operations: `merge`,
with 4 main variants listed in @bu-symbols.

A naive implementation of e.g. `merge` would be directly symmetrical to `partition` from @naive:
```c
  for (i = 0; i < n; i++) {
    bit = get_bit(bitmap, i);
    if (bit) output[i] = symbols_right[n_right++]
    else     output[i] = symbols_left[n_left++]
```

Naturally, we implement this logic with SIMD, using the following code on ARM NEON, for 8 entries

```c
  uint8_t mask  = bitmap[i >> 3];
  // Load eight left and right symbols
  uint8x8_t  lsyms = vld1_u8(left + n_left);
  uint8x8_t  rsyms = vld1_u8(right + n_right);
  // Combine them into a single vector
  uint8x16_t both = vcombine_u8(lsyms, rsyms);
  // Load the precomputed shuffle vector for this mask
  uint8x8_t  shuf = vld1_u8(expand_tab[mask]);
  // Gather values from either left or right input based on mask
  uint8x8_t  o    = vqtbl1_u8(both, shuf);
  // Save 8 bytes, always
  vst1_u8(out + i, o);
  // Update n_left and n_right for the next iteration
  int nr = expand_popcnt[mask];
  n_right += nr;
  n_left += (8 - nr);
```

The code for `merge_constant` is identical, except we use a precomputed
(outside the hot loop) vector of constant values, e.g.:

```c
  uint8x8_t  lsyms = vdup_n_u8(left_sym);
```

For `merge_two` we could also use the same trick for both sides.
But in our testing an `eor` based approach similar to `scatter_two`
turned out slightly more performant.

#todo[Verify]

`merge_flat_D` is an interesting case, where depending on D we might
need to use different approaches.
For D=2..6, so up to 64 symbols, we can use the family of `vqtbl*` operations
that for each symbol index simply fetch the proper symbol.
Here's an example for D=4 (16 symbols):
```c
// Before loop - load the code-to-symbol mapping into a vector
uint8x16_t c2s_vec = vld1q_u8(c2s);

// In a loop, for 16 (!) elements
// Unpack 16 nibbles (8 bytes) into 16 code-index bytes
uint8x16_t codes = flat_d4_unpack(bitmap + (i / 2));
// Fetch the symbols we need
uint8x16_t syms  = vqtbl1q_u8(c2s_vec, codes);
// Save 16 symbols at once
vst1q_u8(symbols + i, syms);
```
See how we can decode 16 symbols with just bit-unpacking and 3 extra instructions.

For D=7 and D=8 we currently use non-simd variants, but that could be further optimized.

#todo[consider optimizing]

== Bottom-up primitive performance


#let rows = csv("data/bu-primitive-host-cmp.csv")
#let rows = pick-cols(rows, ("primitive","m4_proba80","c8i_proba80","m4_prose","c8i_prose"))
#figure(
  table(
    columns: 5,
    table.header(
      table.cell(rowspan: 2)[*Primitive*],
      table.cell(colspan: 2)[*proba80*],
      table.cell(colspan: 2)[*prose_pride*],
      [M4], [c8i],
      [M4], [c8i],
    ),
    ..rows.slice(1).flatten(),
  ),
  caption: [Performance of bottom-up primitives (ns/code)]
)<prim-bu>

@prim-bu demonstrates bottom-up primitive-performance.
We see that all primitives (except for the degenerated `merge_two` in proba80 due to a very small input)
achieve performance comparable to the fast `partition` primitives from @prim-td-opt, and none
pay the memory-overload penalty that the slow `scatter` primitives suffered from.

== Bottom-up decoding performance

