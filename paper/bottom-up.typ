#import "conf.typ": anote, PH, he, mf, sym, pick-cols, todo, fair-cell

= Going Bottom-Up<bottom-up>

In @sideways we processed the tree *top-down*,
 which is a very natural approach, directly translating to "textbook" Huffman decoding.
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

In this approach, leaf nodes do not require any processing, as they just produce a constant value.
This is different from the top-down approach, where we had to apply a `scatter` primitive.
Additionally, inputs and output of each node are _dense_, alleviating the scatter problem of the top-down approach.

#figure(
  mf("bu-tree"),
  caption: [Bottom-up traversed "huffman" tree (_flat trees_ off). See how each node produces a dense list of symbols.]
)<fig-bu-tree>

This results in the approach presented in @fig-bu-tree.
Note that this tree is symmetrical to @fig-pivot-tree, with just data traveling in the opposite direction,
 and different data flowing with the Huffman-code bitmaps (symbols vs indices).

== Bottom-up tree optimizations

With a different processing model, let us see how tree optimizations from @ph-opt apply to the bottom-up approach:
- _fused-leaves_ - not applicable, as the leaves just produce constant values
- _frequent-symbol_ - not applicable, as the final merge in the root produces a dense full-output sequence
- _flat-subtrees_ - directly applicable, reduces the tree size
- _non-canonical flat-subtrees_ - directly applicable, further reduces the tree size

#let _bts = csv("data/bu-tree-stats.csv")
#figure(
  table(
    columns: 9,
    align: (left, right, right, right, right, right, right, right, right),
    table.header(
      table.cell(rowspan: 2)[*Dataset*],
      table.cell(rowspan: 2)[*H*],
      table.cell(rowspan: 2)[*L*],
      table.cell(colspan: 2)[*naive*],
      table.cell(colspan: 2)[*flat*],
      table.cell(colspan: 2)[*flat-opt*],
      [nodes], [ops/B],
      [nodes], [ops/B],
      [nodes], [ops/B],
    ),
    ..(_bts.slice(1).flatten()),
  ),
  caption: [The impact of tree optimizations on the decoding cost. *L* - the average (weighted) Huffman code symbol.]
)<bu-tree-stats>

Table @bu-tree-stats shows the impact of flat-subtrees and fully-optimized non-canonical trees.
The significant reduction in _ops/B_ results in a better encoding and decoding performance.
Additionally, the reduction of the number of nodes helps tree construction time and tree traversal overheads.
See also @tab-tree-modes for the actual decoding performance.

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

    [`P`],  [`MVV`], [`merge_vec_vec` is symmetrical to two-sided `partition`],
    [`PR`], [`MVV`], [`merge_vec_vec` for the root node is identical to other cases],
    [`PH`], [`MCV/MVC`], [`merge_cst_vec/merge_vec_cst` - _merge_ variant where one input is constant],
    [`S2`], [`MCC`],   [`merge_cst_cst` - merges two constant symbols into output],
    [`SFD`], [`MFD`], [`merge_flat_D` - merges 2^D constant symbols into output],
    [`C`],  [--], [Note that in bottom-up multiple leaves can be "constant"],
    [`S1`], [--],  [No operation needed for leaves when going bottom up],
  ),
caption: [Primitives used in bottom-up processing and their top-down equivalents]
)<bu-symbols>

@fig-bu-ops shows the example tree we used before, but this time with operations used for the bottom-up processing.
Again, it is directly symmetrical to @treeopt-flat.

== Bottom-up primitives

Bottom-up processing uses two main fami in only one family of operations: binary `merge_X_Y` family,
where both `X` and `Y` can be `vec` (a vector of symbols) or `cst` (a constant symbol),
and N-ary `merge_flat_D` primitives, specialized for `D` values.

A naive implementation of e.g. `merge_vec_vec` would be directly symmetrical to `partition` from @naive:
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

The code for `merge_cst_vec` is identical, except we use a precomputed
 (outside the hot loop) vector of constant values, e.g.:

```c
  uint8x8_t  lsyms = vdup_n_u8(left_sym);
```

`merge_vec_cst` is symmetrical.
Other _merge_ primitives do not have symbols as an input, but rather, logically,
a bit-packed index into a _code-to-symbol_ table.
As such, the process for all these primitives consists of
unpacking the packed-values, and then performing such a lookup.
We found this solution to be the fastest even for `merge_cst_cst`,
 which can be seen as `merge_flat_D` with `D=1`.

For the lookup, on M4 we use the family of `vqtbl*` operations
for D=1..6, chained with `vqtbx*` operations for D=7..8.
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

== Bottom-up primitive performance

/*
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
*/
#let _bp = csv("data/bu-prim-bits.csv")

#he("prim-bu", style:"
  .prim-bu td:nth-child(1) { text-align: left; }
  .prim-bu td,th {padding: 3pt; font-size: ;}
  .prim-bu td:nth-child(1) { font-weight: bold; border-right: solid 2px black;}
  .prim-bu td:nth-child(5) { font-weight: bold; border-left: solid 2px black;}
  .prim-bu td:nth-child(9) { font-weight: bold; border-left: solid 2px black;}
")[
#figure(
  table(
    columns: 12,
    align: (left, right, right, right, right, right, right, right, right, right, right, right),
    table.header(
      table.cell(colspan: 4)[*primitive*],
      table.cell(colspan: 4)[*M4*],
      table.cell(colspan: 4)[*c8i*],
      [name], [in_b], [out_b], [lut_b],
      [ns/el], [in_bw], [out_bw], [lut_bw],
      [ns/el], [in_bw], [out_bw], [lut_bw],
    ),
    ..(_bp.slice(1).flatten()),
  ),
  caption: [Bottom-up primitive performance on M4 and c8i. *in_b / out_b / lut_b* - input /output / lookup table *bits* used per element.
  *in_bw / out_bw / lut_bw* - respective memory bandwidths achieved in *GB/s*.]
)<prim-bu>
]

@prim-bu demonstrates bottom-up primitive-performance.
Looking at the *ns/elem* metric, we see that all primitives
achieve performance comparable or better to the fast `partition` primitives from @prim-td-opt, and none
pay the memory-overload penalty that the slow `scatter` primitives suffered from.


== Bottom-up decoding performance

#let fair = csv("data/fair.csv")
#let _dsets_tm = ("proba80", "english", "html_wiki", "prose_pride",
                  "json_api", "dna_fasta", "chinese_text", "calgary_pic")
#let _engs_tm = ("ph_naive", "ph_flat", "ph", "huf0", "oo_huff")
#let _body_tm = _dsets_tm.map(d => {
  ([#d],) + ("m4", "c8i").map(h =>
    _engs_tm.map(e => fair-cell(fair, h, d, e, "dec_op"))).flatten()
}).flatten()

#he("tab-tree-nodes", style:"
  .tab-tree-nodes td:nth-child(1) { text-align: left; }
  .tab-tree-nodes td,th {padding: 3pt; font-size: ;}
  .tab-tree-nodes td:nth-child(1) { font-weight: bold; }
  .tab-tree-nodes td:nth-child(4) { font-weight: bold; }
  .tab-tree-nodes td:nth-child(9) { font-weight: bold; }
")[

  #figure(
    table(
      columns: 11,
      align: (col, _) => if col == 0 { left } else { right },
      table.header(
        table.cell(rowspan: 3)[*Dataset*],
        table.cell(colspan: 5)[*M4*],
        table.cell(colspan: 5)[*c8i*],

        table.cell(colspan: 3)[*#PH tree opt.*],
        table.cell(rowspan: 2)[*Huff0*],
        table.cell(rowspan: 2)[*Oo-Huff*],
        table.cell(colspan: 3)[*#PH tree opt.*],
        table.cell(rowspan: 2)[*Huff0*],
        table.cell(rowspan: 2)[*Oo-Huff*],

        [*naive*],  [*flat*],  [*flat-opt*],
        [*naive*],  [*flat*],  [*flat-opt*],
      ),
      .._body_tm,
    ),
    caption: [#PH (bottom-up) decode bandwidth (MB/s) for different tree optimization levels.
    We compare naive, fused-leaves, flat-tables and optimized flat-tables against Huff0 and Oodle-Huffman.]
  )<tab-tree-modes>
]

#figure(
  [
    #image("plots/tree_modes_m4.svg")
    #image("plots/tree_modes_c8i.svg")
  ],
  caption:[#PH decoding performance with different tree optimization levels]
)<plot-tree-nodes>

To evaluate the performance of bottom-up #PH decoding we looked at our datasets on two machines.
We're also testing the impact of tree-complexity optimizations from @ph-opt.
@tab-tree-modes and @plot-tree-nodes show the results.
We can see that #PH decisively beats decoding performance of Huff0 and Oodle Huffman on all datasets and platforms.
The magnitude of #PH benefits depends on three factors:
- tree level optimization - we see that both flat subtrees and their optimized versions provide significant benefits.
  This makes sense, as with the reduction of the number of operations, the performance improves.
- dataset - skewed datasets benefit most, as on these #PH can reduce the number of operations for shorter codes / more frequent symbols.
  In particular, tree optimizations have no impact on _proba80_ and only marginal on _dna_fasta_.
- CPU - on c8i, with its worse OoO capabilities, Huff0 and Oodle Huffman perform worse.
  On the other hand, with its better SIMD capabilities of AVX-512, #PH actually performs better here, magnifying its win over other algorithms.
