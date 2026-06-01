#import "conf.typ": PH, he

#heading(numbering:none, level: 1)[Appendices]

#let appendix(body) = {
  set heading(numbering: "A.1", supplement: [Appendix])
  counter(heading).update(0)
  body
}
#set heading(numbering: "1")

#show: appendix

// #outline(target: heading.where(supplement: [Appendix]), title: [Appendix])

= Datasets <datasets>

#let rows = csv("data/dist-stats.overview.csv")
#let data = rows.slice(1)   // drop the CSV header row

#he("tab-datasets", style:"
.tab-datasets td:nth-child(1) { text-align: left; font-weight: bold;}
.tab-datasets td:nth-child(7) { text-align: left; }
")[
#figure(
  table(
    columns: 7,
    align: (col, _) => if col == 0 or col == 6 { left } else { right },
    table.header(
      [*Name*], [*\#syms*], [_H_ (bits)], [*Huffman* (bits)],
      [*min*], [*max*], [*Source / description*],
    ),
    ..data.flatten(),
  ),
  caption: [Used datasets: alphabet size, entropy _H_, mean Huffman
            code length, code-length min/max, and source. Available #link("https://github.com/MarcinZukowski/pivco-huffman/blob/main/extras/datasets/README.md")[in #PH repo] ],
)<tab-datasets>
]

= Machines Tested <machines>

// Test machines used for the benchmarks.  CPU strings verified from the
// hosts: M4 = `sysctl machdep.cpu.brand_string`; EC2 = `lscpu` model name.
// AWS instances are .large (2 vCPU).
#he("tab-machines", style:"
.tab-machines td {text-align: center}
.tab-machines td:nth-child(1) { font-weight: bold; }
.tab-machines td:nth-child(2) { text-align: left; }
.tab-machines td:nth-child(4) { text-align: left; }
")[
#table(
  columns: 5,
  inset: 5pt,
  align: (left, left, left, left, right),
  table.header(
    [*Symbol*], [*Machine type*], [*Arch*], [*CPU family*], [*CPU year*],
  ),
  [M4],  [Apple MacBook Pro (Mac16,6)], [aarch64], [Apple M4 Max],                  [2024],
//  [c8g], [AWS EC2 c8g.large],           [aarch64], [AWS Graviton4 (Neoverse V2)],   [2024],
  [c8i], [AWS EC2 c8i.large],           [x86-64],  [Intel Xeon 6 (Granite Rapids)], [2024],
//  [c6a], [AWS EC2 c6a.large],           [x86-64],  [AMD EPYC 7R13 (Milan, Zen 3)],  [2021],
)<tab-machines>
]

= Testing methodology <testing-method>

In our testing we use datasets from @datasets and machines from @machines.

Unless stated otherwise, we compute the time that _includes_ the setup time.

For datasets that are smaller than 1MB, we create copies to cross the 1MB size.

For bandwidth tests,
to determine the optimal performance of each algorithm (reduce noise etc),
we use the following setup:
- one _run_ performs 20 repetitions of the operation back-to-back
- we execute 20 _runs_ and collect the timings
- if the top-two _run_ timings are within 2%, we stop
- otherwise, two more _runs_, until the 2%
  difference goal is met, up to 40 times in total.
- the best _run_ time is used

= Failed optimizations

As we worked on this paper, we have tried many things that didn't pan out.
We list them briefly in here for the reader, either to save some time,
or perhaps inspire to try harder.

== Root-levels decoding

For top-down decoding, before we settled on flat-subtrees, we investigated the idea of
decoding the top D-deep part of the tree for situations where the shortest code was D-bits long.
The intuition was that in one operation we would cover a lot of the most frequent nodes.
However, such an operation would result in 2^D-way stream partitioning, which turned out
to be simply too slow.

== Fusing `scatter` and `partition`

When we tried to optimize top-down decoding, we realized that `scatter` and `partition` are limited
by different CPU resources - `scatter` by writes, and `partition` by table lookups and computations.
We tried to combine them, by having `scatter` for one decompressed block also perform a part of the
`partition` effort for the following block.
Alas, we couldn't achieve any significant benefits.

== Tree optimizations for FSE

Similarly to optimizing the Huffman tree to reduce the number of operations (see @ph-opt),
we tried optimizing the tree to maximize the benefit of FSE.

Two approaches have been attempted:

- allowing _splitting_ of the _flat trees_ if the root node had significant skew
- arranging a tree in a left-heavy (by frequency) way, to force more "skewed" nodes

While both optimizations provided occasional benefits, the impact was so small we decided
to park them, especially as both required transferring the actual frequencies (not only code lengths)
to the decompressor.

== Fusing FSE with `merge` <fuse-fse-merge>

To further optimize PHA performance, we tried to fuse the FSE decoding with the `merge` step
of the bottom-up processing.
While we achieved small improvements (a few percent), the complexity of this solution was not worth incorporating
into the code base.

= Tuning FSE <tuning-fse>

When working with FSE for @ans, we noticed that the FSE overhead had a more severe impact than we would like.
As a result, we performed a side experiment where we tuned FSE's main loop.
By default, it looks like this (slightly simplified):
```c
  while ((BIT_reloadDStream(&bitD) == BIT_DStream_unfinished)   // reload once at top
          & (op + 4 <= olim)) {
      op[0] = FSE_decodeSymbolFast(&states[0], &bitD);
      op[1] = FSE_decodeSymbolFast(&states[1], &bitD);
      op[2] = FSE_decodeSymbolFast(&states[0], &bitD);
      op[3] = FSE_decodeSymbolFast(&states[1], &bitD);
      op += 4;
  }
```
In that code, `states` refers to a table of two states in the FSE table - this is similar to using two independent cursors
 and provides more independent instructions to modern CPUs.
Still, the data for both states comes from a single, interleaved stream.
We also see that the loop is explicitly _2-unrolled_, reducing the loop overhead.
We call this particular implementation *x2y2* (x: 2 cursors, y: 2-unroll).

We performed a thorough testing of equivalent implementations of FSE with *x={2,4,6,8,10,12,16}* and *y={1,2,4}* on a number of machines.
The example results for M4 are in @tab-fse-xy-m4.
The interesting points are in bold.
We see how the peak performance for M4 is at *x10y4*, almost 3x the default *x2y2*.
Still, for our experiments we chose *x8y1* as it provided robust close-to-peak performance on all hosts we tested on.
Note, *x8y1* requires a _wire format change_, so is not directly applicable to _stock_ FSE-encoded data.

#let rows = csv("data/fse-xy-m4.csv")
#let body = rows.slice(1)
#let xs = body.map(r => r.at(0))            // 7 cursor counts
#let ys = ("y=1", "y=2", "y=4")             // 3 unrolls
// Bolded sweet spots (same as the non-transposed version):
//   x=8,  y=1   x=2,  y=2   x=10, y=4
#let cell(y_idx, x_idx) = {
  let x = xs.at(x_idx)
  let v = body.at(x_idx).at(y_idx + 1)
  let is_bold = ((x == "8"  and y_idx == 0)
              or (x == "2"  and y_idx == 1)
              or (x == "10" and y_idx == 2))
  if is_bold { strong[#v] } else { [#v] }
}
#let body_cells = ys.enumerate().map(((y_idx, y_lbl)) => {
  ([*#y_lbl*],) + xs.enumerate().map(((x_idx, _)) => cell(y_idx, x_idx))
}).flatten()
#figure(
  table(
    columns: 1 + xs.len(),
    align: (col, _) => if col == 0 { left } else { right },
    table.header(
      [*y \\ x*],
      ..xs.map(x => [*#x*])
    ),
    ..body_cells,
  ),
  caption: [FSE wide-cursor decode throughput on M4 (MB/s), per
            cursor count _x_ and unroll _y_, at _p_maj=0.80_, 2880 B.],
)<tab-fse-xy-m4>

= PivCo-Golomb <app-golomb>

As discussed in @entropy, applying _pivoted coding_ in other areas is an interesting research question.
Coincidentally, just days before publishing this paper, a blog post appeared on fast Golomb decoding (@ryg-golomb).
Its author proposed a Tunstall-style approach,
 using a precomputed decoding table to quickly generate multiple output symbols (up to 8)
 from one byte of encoded data.

To investigate if we can apply our ideas in this field as well, we implemented *PivCo-Golomb*.
It follows the approach very similar to #PH, except instead of a _tree_, it simply processes a _list_ of bitmaps,
 starting from the bitmap for the last code bit all the way to the first bit.
It uses a single primitive `merge_vec_cst_plus1` - which extends `merge_vec_cst` by adding `0x01` to produced values.
The `cst` value used is `0xFF`.
This way, when decoding a bitmap, values with `1` set will get an output symbol of `0x00`,
and bitmap values of `0` will get the input symbols increased by 1.

  #figure(
    table(
      columns: 7,
      align: (col, _) => if col == 0 { left } else { right },
      table.header(
        table.cell(rowspan: 2)[*avg code length*],
        table.cell(colspan: 3)[*M4* (ns/symbol)],
        table.cell(colspan: 3)[*c8i* (ns/symbol)],
        [naive], [tunstall-64], [PivCo-Golomb],
        [naive], [tunstall-64], [PivCo-Golomb],
      ),
      [1.5], [2.23], [0.09], [0.07], [2.28], [0.11], [0.03],
      [2],   [1.70], [0.10], [0.08], [2.26], [0.15], [0.05],
      [3],   [1.66], [0.22], [0.13], [2.27], [0.32], [0.08],
      [4],   [1.64], [0.43], [0.17], [2.26], [0.62], [0.11],
      [5],   [1.68], [0.74], [0.23], [2.27], [1.03], [0.13],
      [6],   [1.72], [1.08], [0.26], [2.27], [1.51], [0.17],
    ),
    caption: [Comparing Golomb-decoding implementation: naive and Tunstall-style 64-bit table from @ryg-golomb and PivCo-Golomb],
  )<tab-pivco-golomb>

In @tab-pivco-golomb we can see that also in this application our approach can provide excellent performance.
It is especially visible on longer average code lengths,
 where _tunstall-64_ from @ryg-golomb slows down, mostly because of branch mispredictions.
PivCo-Golomb's performance scales linearly with the length of the code, with the per-bit cost
 comparable to the `merge_vec_cst` performance we measured in @prim-bu.

This provides an interesting validation point that the _pivoted coding_ idea could apply efficiently in other areas.
