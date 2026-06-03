#import "conf.typ": anote, PH, he, mf, sym, pick-cols, todo, h0, fair-cell

= Pivoting Huffman <sideways>

Following the example from @hj, it should be possible to create a
Huffman decoder using similar principles.
However, for each node in the Huffman tree to have access to relevant
bits from the stream, a different data layout is needed.
We found a great solution for this in the _wavelet tree_ structure @grossi2003wt.

#figure(
numbering: none,
placement: top,
he("gridtable")[
  #table(
    columns: (50%, 50%),
    stroke: 0pt,
    align: center,
    [#figure(
      mf("pivot-bitmaps"),
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
      mf("pivot-tree"),
      caption: [
      Example of a Huffman tree with "pivoted" data traversing
      it, reusing the previous example.
      Each node produces a list of indices for its symbols.
      ]
    )<fig-pivot-tree>
    ],
  )
]
)

// #mf("pivot-bitmaps")

@fig-pivot shows an alternative representation of the encoded word "huffman".
Instead of a code-after-code stream, we divide all the stream bits
by their (possibly empty) prefix.
@fig-pivot-tree shows how this layout maps onto the Huffman tree when decoding
data.
Each node receives all the bits of the codes that pass through it, and navigates
these codes to its children, where another bitmap is used for the next step.

Note that while logically this representation contains the same information as standard Huffman coding,
it typically stores bitmaps byte-aligned.
This might lead to a _marginally worse_ compression ratio due to byte-padding.
However, for non-trivial datasets this overhead is acceptable if this approach provides other benefits.

This data representation and tree traversal are the basis for *_pivot-coded Huffman_ (#PH)*, presented in this paper.
In this section we present the _initial_ implementation of this approach, which is actually
_not_ used in the final solution.
Still, since it is a more _natural_ approach, we describe it first, and use it to introduce
a set of optimizations and implementation techniques.
In @bottom-up, we will propose the final, more performant solution.

As mentioned, the tree representation used is equivalent to _wavelet trees_,
specifically _Huffman-shaped wavelet trees_ (e.g. @dinklage2023wt).
However, we treat #PH as a related, but separate solution - see @wt for more discussion.

#anote[
While working on #PH I did a lot of literature review, and for the longest time couldn't find
anything like it.

Close to the end of the research/experimental work, Claude found Wavelet trees.

Initially I was in panic (a few weeks of life lost?), but the deeper review showed that
it's really quite different.

So here we go. I will still call it #PH.
]

== Naive implementation <naive>

With a defined Huffman tree, and data stored in per-node bitmaps, we can traverse
the tree top-down.
Note, as we do it, we need to know which output elements we are decoding.
For that, we carry an additional `indices` list (the root node does not need it).
In our implementation we use 16-bit values for indices, as we decode data in small
blocks (e.g. 8KB).
With that, #PH tree traversal boils down to applying two operations:

*`partition(bitmap, indices) => (indices_left, indices_right)`* -- applied for
all internal nodes.
Takes a list of indices (positions in the output stream), and divides it
based on the bitmap into left and right indices for its subtrees.
Note, a special `partition_root` version can be used in the root node,
as its list of indices is the complete input.

The `partition` primitive can be expressed naively with:
```c
  for (i = 0; i < n; i++) {
    bit = get_bit(bitmap, i);
    if (bit) indices_right[n_right++] = indices[i];
    else     indices_left[n_left++] = indices[i];
  }
```

*`scatter(output, indices, symbol)`* -- fills all positions in the output stream
with a given symbol.

```c
  for (i = 0; i < n; i++) {
    output[indices[i]] = symbol;
  }
```

We measured the decoding performance of such a naive implementation on Apple M4 CPU,
and, as expected, the performance is very sub-par.

// ---- TD-vs-Huff0 tables, sourced from the consolidated fair.csv ----
// TD decoders use prebuilt-kernel throughput (dec_pb) -- these tables compare
// decoder KERNELS, so we exclude per-call table-build overhead.  Huff0 is
// stock HUF_decompress (dec_op); its table build is negligible at 128 KB
// chunks, so its opaque throughput equals its prebuilt kernel.  Values in GB/s.
#let fair = csv("data/fair.csv")
#let _tdm = (naive: "td_naive", naive_simd: "td_nv_simd",
             scalar: "td_scl_opt", simd: "td_simdopt")
#let tdg(host, ds, dec) = {            // TD kernel throughput, GB/s
  let v = fair-cell(fair, host, ds, _tdm.at(dec), "dec_pb")
  if v == "na" { [—] } else { [#calc.round(float(v) / 1000, digits: 2)] }
}
#let h0g(host, ds) = {                 // Huff0 throughput, GB/s
  let v = fair-cell(fair, host, ds, "huf0", "dec_op")
  if v == "na" { [—] } else { [#calc.round(float(v) / 1000, digits: 2)] }
}
#let tdr(host, ds, dec) = {            // ratio: TD kernel / Huff0
  let a = fair-cell(fair, host, ds, _tdm.at(dec), "dec_pb")
  let b = fair-cell(fair, host, ds, "huf0", "dec_op")
  if a == "na" or b == "na" { [—] } else { [#calc.round(float(a) / float(b), digits: 2)] }
}

#show table.cell.where(x: 6): strong
#figure(numbering: none)[
#table(
  columns: 7,
  align: (left, right,right,right,right,right,right),
  table.header(
    table.cell(rowspan: 2, align: center)[*Data*],
    table.cell(colspan: 3, align: center)[*M4 * (GB/s)],
    table.cell(colspan: 3, align: center)[*c8i* (GB/s)],
    [naive], [#h0], [naive/#h0],
    [naive], [#h0], [naive/#h0],
  ),
  [proba80],
  tdg("m4", "proba80", "naive"), h0g("m4", "proba80"), tdr("m4", "proba80", "naive"),
  tdg("c8i", "proba80", "naive"), h0g("c8i", "proba80"), tdr("c8i", "proba80", "naive"),
  [prose_pride],
  tdg("m4", "prose_pride", "naive"), h0g("m4", "prose_pride"), tdr("m4", "prose_pride", "naive"),
  tdg("c8i", "prose_pride", "naive"), h0g("c8i", "prose_pride"), tdr("c8i", "prose_pride", "naive"),
)
]

There are two main reasons for this:

- for each decoded symbol, we perform multiple operations: we run `partition` for each bit in the code,
  followed by a final `scatter` at each leaf (`len(code)+1` operations in total).
- the `partition` and `scatter` primitives, as written, are not efficient.

In the following two sections we will discuss how to address both problems.

== Tree Optimizations <ph-opt>

A naive Huffman tree discussed before suffers from a large number
of operations per byte.
In this section we demonstrate
a number of techniques that can bring that number down significantly,
using string `coconut-papaya` as a test case.

In @treeopt-naive we see our starting point - a basic tree with
2 kinds of primitives (marked in orange boxes), and 4.071 operations per output byte
(weighted by symbol frequency).
@treeopt-symbols translates symbols used in figures in this section
to the actual compute primitives.

#he("gridtable")[
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
        align: (center, left),
        table.header([*Code*], [*Explanation*]),
        [`P`],  [`partition` - split indices into left/right based on bitmap],
        [`PR`], [`partition_root` - like `partition` but for the root node],
        [`PH`], [`partition_half` - like `partition`, but produces only one output],
        [`C`], [_not a primitive_ - marks the "constant", top-frequency key],
        [`S1`], [`scatter` - scatters a single symbol into output],
        [`S2`], [`scatter_two` - scatters two symbols into output],
        [`SFD`], [`scatter_flat_D` - scatters 2^D symbols into output],
      ),
    caption: [Primitive symbols used in figures in this Section]
    )<treeopt-symbols>
  ],
)
]
=== Merging leaves

One simple approach of reducing the number of operations is
to avoid the last-level `partition`, and simply fill
all input indices with the symbol based on the bitmap.

This results in a *`scatter_two(output, indices, bitmap, symbol0, symbol1)`* primitive:

```c
  for (i = 0; i < n; i++) {
    output[indices[i]] = get_bit(bitmap, i) ? symbol1 : symbol0;
  }
```

@treeopt-fuse shows the benefit in reduced operations per decoded byte going from 4.071 to 3.286.

=== Frequent symbol optimization

One of the problems of our decoding primitives is writing into non-contiguous
positions in the output, presenting challenges for modern CPUs and memory subsystems (see @scatter).

We can mitigate it by avoiding this completely for the _most frequent symbol_,
by simply prefilling the entire output with `memset` before decoding.
Then, that symbol never needs to be processed during the tree traversal.
Note that `memset` is 1-2 orders of magnitude faster than our primitives, so that
cost is negligible.

@treeopt-constant shows the tree with this optimization applied to symbol #sym("a").
Note that as a result we introduce a new operation called `PH` - `partition_half`,
similar to `partition`, but only producing one of the output index lists.

#he("gridtable")[
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
]

=== Flat Subtrees

Huffman trees often contain subtrees where all the symbols
share the same length.
In our example, 4 right-most #sym("- n t u") nodes form such a subtree.

We can decode such a subtree with a single operation
*`scatter_flat_D(output, indices, bitmap, symbols)`*, where `D` represents
the depth of the subtree.

Note that for this, the input `bitmap` is not _binary_, but _(2^D)-ary_, with
bits packed contiguously.
Also, note that `scatter_two` is a special case of this approach, with _D=1_.

```c
  bit_unpack(bitmap, D, code_indices);
  for (i = 0; i < n; i++) {
    output[indices[i]] = code_to_symbols[code_indices[i]];
  }
```

#he("gridtable")[
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
]

=== Non-Canonical Subtrees

Looking at @treeopt-flat, we can see that while the #sym("- n t u") symbols
benefit from the "flat subtrees" strategy, we also have #sym("c o p y") symbols,
which share the same code lengths, but are not decoded together.

We can reorganize the canonical Huffman tree to make it more amenable to the "flat subtree"
optimization by making sure that codes with the same length are grouped as much as
possible.
To achieve that, after determining code lengths, within each length-group, we combine the largest _power of two_ number of nodes
into a single node with a combined frequency.
We repeat the process, with one length-group possibly creating multiple such nodes (of different depth).

The result is a new, (usually) non-canonical Huffman tree, with the exact same average
code lengths, but a different shape.
@treeopt-opt shows how applying this strategy allows the #sym("c o p y") nodes to be processed
together, further reducing ops/byte.

== Impact of tree optimizations

#figure(
  table(
    columns: 9,
    align: (left, right,right,right,right,right,right,right,right),
    table.header(
      table.cell(rowspan: 2, align:center)[*Data*],
      table.cell(colspan: 4, align:center)[*M4* (GB/s)],
      table.cell(colspan: 4, align:center)[*c8i* (GB/s)],
      [naive], [opt], [#h0], [opt/#h0],
      [naive], [opt], [#h0], [opt/#h0],
    ),
    [proba80],
    tdg("m4", "proba80", "naive"), tdg("m4", "proba80", "scalar"), h0g("m4", "proba80"), tdr("m4", "proba80", "scalar"),
    tdg("c8i", "proba80", "naive"), tdg("c8i", "proba80", "scalar"), h0g("c8i", "proba80"), tdr("c8i", "proba80", "scalar"),
    [prose_pride],
    tdg("m4", "prose_pride", "naive"), tdg("m4", "prose_pride", "scalar"), h0g("m4", "prose_pride"), tdr("m4", "prose_pride", "scalar"),
    tdg("c8i", "prose_pride", "naive"), tdg("c8i", "prose_pride", "scalar"), h0g("c8i", "prose_pride"), tdr("c8i", "prose_pride", "scalar"),
  ),
  caption: [Combined impact of tree optimizations on performance]
)<tab-tree-opt>

@tab-tree-opt shows that applying all optimizations above allows improving #PH performance
by up to factor two.
However, as we will demonstrate later, these optimizations are even more important
with faster compute primitives.

== Computing Primitives

@treeopt-symbols lists primitives used during decoding of #PH.
Most of them can be expressed with a simple "scalar" loop, but
since they work on multiple items, they also represent
opportunities for SIMD optimization.
And even with a scalar loop, coding methods can make a dramatic difference.

In this Section we'll discuss how some of them are implemented using SIMD instructions from the ARM NEON family.

#let rows = csv("data/td-naive-vs-opt.primitive-host-cmp.csv")
#let rows = pick-cols(rows, ("primitive","m4_proba80","c8i_proba80","m4_prose","c8i_prose"))
#let data = rows.slice(1)
#figure(
  table(
    columns: (20%,10%,10%,10%,10%),
    align: (left, right,right,right,right),
    table.header(
      table.cell(rowspan: 2, align:center)[*Primitive*],
      table.cell(colspan: 2, align:center)[*proba80*],
      table.cell(colspan: 2, align:center)[*prose_pride*],
      table.cell(align:center)[M4],   table.cell(align:center)[c8i],
      table.cell(align:center)[M4],   table.cell(align:center)[c8i],
    ),
    ..data.flatten()
  ),
  caption: [Performance of "naive" primitive implementations (ns/code)]
)<prim-td-naive>

=== `partition`

`partition` is the most important operation during #PH tree traversal,
as each code goes through it multiple
times before ending up in one of the versions of `scatter`.

A naive implementation of this primitive was presented in @naive.
The critical performance aspect there is this fragment:
```c
  if (bit) indices_right[n_right++] = indices[i];
  else     indices_left[n_left++] = indices[i];
}
```
This statement depends on the extracted bit value,
and may be hard to predict for the branch predictor.
If the distribution is skewed (like `proba80`), it makes the branch easier to guess.
For more uniform data, the branch predictor cannot guess properly,
as we can see in @prim-td-naive for `prose_pride`.
On `m4`, with its cheaper branch predictor misses, the difference is minimal, but on `c8i` it is significant.

One way to alleviate this is to replace branching with an unconditional assignment, here's this approach shown
inside the smaller `partition_half_right` primitive:

```c
    // ... same setup
    right[n_right] = indices[i];
    n_right += b;
```

In @prim-td-naive we see that this primitive doesn't suffer from the branch misprediction on `prose_pride` as much
as `partition`.

Partitioning performance can be further improved with SIMD. For example, here's an ARM NEON
implementation (just an 8-value kernel with a given 8-bit `mask` from the bitmap).

```c
  uint8x16_t data = vld1q_u8((const uint8_t *)indices);

  /* Load shuffle patterns for right/left side - they are stored together */
  const uint8_t *tab = compress_tab[mask];
  uint8x16_t shuf_r = vld1q_u8(tab);       /* bytes 0-15: right */
  uint8x16_t shuf_l = vld1q_u8(tab + 16);  /* bytes 16-31: left */

  /* Save input indices in either right or left */
  uint8x16_t right = vqtbl1q_u8(data, shuf_r);
  uint8x16_t left  = vqtbl1q_u8(data, shuf_l);

  /* Compute how many values went right */
  int n_right = compress_popcnt[mask];

  /* Store both results - always 16 bytes */
  vst1q_u8((uint8_t *)right_out, right);
  vst1q_u8((uint8_t *)left_out, left);
```

How it works:
- `compress_tab`, for each of the 256 possible `mask` values, stores two
    (for left and right) 16-byte arrays
    determining which bytes from the input should be written to a given output
- `vqtbl1q_8` operation creates _condensed_ subsets of input for left/right.
  Note, these vectors might have zeros in their tail, as on average they are
  half-full.
- result vectors are always written as 16 bytes - while it might sound wasteful,
  it is simpler and faster.
- the next iteration needs to adjust both output pointers based on `n_right`
  (`n_left` is simply `8 - n_right`)

Note that this code is fully branch-free, and with just a few instructions we process 8 16-bit input indices.
We can apply a similar strategy on other architectures.
For example, on AVX-512 one can partition even 32 16-bit index values in a few steps like that.
The results in @prim-td-opt show how the performance of the `partition` primitive
improved by a factor of *6x* on *m4* and *25-70x* (!) on *c8i*.

=== `scatter` primitives <scatter>

The other big part of tree traversal are `scatter` primitives. They come in a few forms:

- `S1` - scatter_one - puts a constant symbol at all input indices
- `S2` - scatter_two - puts one of two symbols based on the bitmap
- `SFD` - scatter_flat_D - puts one of the 2^D symbols in the indices based on a packed x-bit bitmap.

Each of these primitives can be decomposed into two stages:

1. Determine which symbol to write at a given index
2. Actually write the symbols

For `scatter_one`, part one is trivial. For `scatter_two`, it's a bit more interesting.
One could of course do code similar to
```c
  output[indices[i]] = bitmap[i] ? symbol0 : symbol1
```

For an efficient SIMD implementation we prefer another approach, based on SIMD
arithmetic and a precomputed delta between `symbol0` and `symbol1`.
```c
  // Performed once
  uint8x8_t vsym0  = vdup_n_u8(sym0);           // vector of sym0
  uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);    // vector of sym0^sym1
  static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
  uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);    // vector used to "expand a bitmap"

  // Performed for each bitmap byte
  // convert 8-bits into 8 00/FF bytes
  uint8x8_t bits = vtst_u8(vdup_n_u8(bitmap[i >> 3]), vbit_pos);
  // set 8 symbols to either sym0 or sym1 (=sym0^delta)
  uint8x8_t vals = veor_u8(vsym0, vand_u8(vdelta, bits));
  // write 8 symbols
  output[indices[i+0]] = vget_lane_u8(vals, 0);
  // ...
  output[indices[i+7]] = vget_lane_u8(vals, 7);
```

For `scatter_flat_D` we get a D-bit packed bitmap.
The first step is to unpack it, using an optimized unpacking kernel (see @bitpack).
Then each unpacked value can be used to lookup an actual symbol to write from a table.
On ARM this can be done with `vqtbl*` instructions.
For example, here's an implementation for D=5:

```c
  // Performed once.
  // Assumes c2s is an array mapping index 0..31 to a given symbol.
  // We load these values into two uint8x16 vectors.
  uint8x16x2_t c2s_vec;
  c2s_vec.val[0] = vld1q_u8(c2s);
  c2s_vec.val[1] = vld1q_u8(c2s + 16);

  // Performed for each group of 8 packed values
  uint8x8_t codes = flat_d5_unpack(bitmap + ((i * 5) >> 3));  // unpack
  uint8x8_t vals  = vqtbl2_u8(c2s_vec, codes);                // lookup
  // ... write 8 symbols - identical as in scatter_two
```

The second problem is the actual writing of the symbols. It boils down to the following problem:
```c
  output[indices[i+0]] = input_function(i+0)
  // ...
  output[indices[i+7]] = input_function(i+7)
```

Note that with indices being ordered, but not contiguous, this results in a lot of
individual writes.
All 3 versions of `scatter` use this approach.
This tends to saturate the CPU's load/store units, and limits further performance improvements.
The author does not know of an efficient solution to this problem on either x86 or ARM architectures.
Only AVX-512 provides _scatter_ instructions, but they do not seem applicable here,
as they only work with 32- and 64-bit values.

#let rows = csv("data/td-naive-vs-opt.simd-primitive-host-cmp.csv")
#let rows = pick-cols(rows, ("primitive","m4_proba80","c8i_proba80","m4_prose","c8i_prose"))
#figure(
  placement: top,
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header(
      table.cell(rowspan: 2, align:center+horizon)[*Primitive*],
      table.cell(colspan: 2, align:center)[*proba80*],
      table.cell(colspan: 2, align:center)[*prose_pride*],
      [M4], [c8i],
      [M4], [c8i],
    ),
    ..rows.slice(1).flatten(),
  ),
  caption: [Performance of SIMD optimized primitive implementations (ns/code)]
)<prim-td-opt>

@prim-td-opt demonstrates the memory-writes problem.
You can see even the seemingly trivial `simd_s1_scatter` taking 2-5x more time per element
than `simd_partition`.
We also see that SIMD optimizations only improved scatter performance by up to a factor of 2x
compared to @prim-td-naive.

Note that per-dataset numbers vary due to different cardinalities.
In particular, `proba80` has very few elements reaching `simd_s2_scatter_both`, causing a high
per/element cost.

// #todo[weird c8i scatter numbers, slower than naive]

// #todo[perhaps a pure primitive benchmark would be better]

== Results

@ph-td-final-bw and @ph-td-final-ratio show how
thanks to combining tree optimizations and high-performance SIMD primitives,
this version of #PH enters the performance territory of #h0.
We also see how with faster primitives the impact of the optimized tree shape provides
stronger and more consistent benefits compared to @tab-tree-opt.

Notably, the performance really depends on dataset - in `proba80`, with its lower
entropy / shorter codes, average number of operations per symbol is much smaller.
This behavior is unique to #PH, and allows it to decidedly beat #h0 on such
distributions.

Still, the performance is not consistently impressive - this is mostly
impacted by the bottleneck of writes in `scatter` primitives.

In the next Section we'll discuss a different approach to #PH that works around this problem.

#figure(
  table(
    columns: 6,
    table.header(
      table.cell(rowspan: 2, align:center+horizon)[*Dataset*],
      table.cell(rowspan: 2, align:center+horizon)[*Tree*],
      table.cell(colspan: 2, align:center+horizon)[*M4*],
      table.cell(colspan: 2, align:center+horizon)[*c8i*],
      [scalar],[simd],[scalar],[simd],
    ),
    table.cell(rowspan: 2)[proba80],
    [naive],
    tdg("m4", "proba80", "naive"), tdg("m4", "proba80", "naive_simd"),
    tdg("c8i", "proba80", "naive"), tdg("c8i", "proba80", "naive_simd"),
    [optimized],
    tdg("m4", "proba80", "scalar"), strong(tdg("m4", "proba80", "simd")),
    tdg("c8i", "proba80", "scalar"), strong(tdg("c8i", "proba80", "simd")),
    table.cell(rowspan: 2)[prose_pride],
    [naive],
    tdg("m4", "prose_pride", "naive"), tdg("m4", "prose_pride", "naive_simd"),
    tdg("c8i", "prose_pride", "naive"), tdg("c8i", "prose_pride", "naive_simd"),
    [optimized],
    tdg("m4", "prose_pride", "scalar"), strong(tdg("m4", "prose_pride", "simd")),
    tdg("c8i", "prose_pride", "scalar"), strong(tdg("c8i", "prose_pride", "simd")),
  ),
  caption: [Impact of tree and primitive optimizations (GB/s)]
)<ph-td-final-bw>

#figure(
  table(
    columns: 6,
    table.header(
      table.cell(rowspan: 2)[*Dataset*],
      table.cell(rowspan: 2)[*Tree*],
      table.cell(colspan: 2)[*M4*],
      table.cell(colspan: 2)[*c8i*],
      [scalar],[simd],[scalar],[simd],
    ),
    table.cell(rowspan: 2)[proba80],
    [naive],
    tdr("m4", "proba80", "naive"), tdr("m4", "proba80", "naive_simd"),
    tdr("c8i", "proba80", "naive"), tdr("c8i", "proba80", "naive_simd"),
    [optimized],
    tdr("m4", "proba80", "scalar"), strong(tdr("m4", "proba80", "simd")),
    tdr("c8i", "proba80", "scalar"), strong(tdr("c8i", "proba80", "simd")),
    table.cell(rowspan: 2)[prose_pride],
    [naive],
    tdr("m4", "prose_pride", "naive"), tdr("m4", "prose_pride", "naive_simd"),
    tdr("c8i", "prose_pride", "naive"), tdr("c8i", "prose_pride", "naive_simd"),
    [optimized],
    tdr("m4", "prose_pride", "scalar"), strong(tdr("m4", "prose_pride", "simd")),
    tdr("c8i", "prose_pride", "scalar"), strong(tdr("c8i", "prose_pride", "simd")),
  ),
  caption: [Impact of tree and primitive optimizations (ratio to #h0, higher is better)]
)<ph-td-final-ratio>

