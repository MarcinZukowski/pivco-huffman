#import "conf.typ": _html, _pdf, _fmt, htmlonly, setup, PH

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
  caption: [MAIN test distributions: alphabet size, entropy _H_, mean Huffman
            code length, code-length min/max, and source. Available #link("https://github.com/MarcinZukowski/pivco-huffman/blob/main/extras/datasets/README.md")[in #PH repo] ],
)<tab-datasets>

= Machines Tested
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
      op[0] = FSE_decodeSymbolFast(&s[0], &bitD);
      op[1] = FSE_decodeSymbolFast(&s[1], &bitD);
      op[2] = FSE_decodeSymbolFast(&s[0], &bitD);
      op[3] = FSE_decodeSymbolFast(&s[1], &bitD);
      op += 4;
  }
```
In that code, `s` refers to a table of two states in the FSE table - this is similar to using two cursors.
Still, the data for both states comes from a single, interleaved stream.
We also see that the loop is explicitly _2-unrolled_ - this allows reducing the loop overhead.
We call this particular implementation *x2y2* (x: 2 cursors, y: 2-unroll).

We performed a thorough testing of equivalent implementations of FSE with *x={2,4,6,8,10,12,16}* and *y={1,2,4}* on a number of machines.
The example results for M4 are in @tab-fse-xy-m4.
The interesting points are in bold.
We see how the peak performance is at *x10y4*, almost 3x the default *x2y2*.
Still, for our experiments we chose *x8y1* as it provided robust close-to-peak performance on all hosts we tested on.
Note, *x8y1* requires a _wire format change_, so is not directly applicable for _stock_ FSE-encoded data.

#let rows = csv("data/fse-xy-m4.csv")
#let data = rows.slice(1).map(r => {
  // bold the x8y1 cell: x==8 (col 0) -> y1 (col 1)
  r.enumerate().map(((i, c)) =>
    if (r.at(0) == "8" and i == 1) or (r.at(0) == "2" and i == 2) or (r.at(0) == "10" and i == 3) { strong[#c] } else { [#c] })
})
#figure(
  table(
    columns: 4,
    align: (col, _) => if col == 0 { left } else { right },
    table.header(
      [*x*], [*y=1*], [*y=2*], [*y=4*],
    ),
    ..data.flatten(),
  ),
  caption: [FSE wide-cursor decode throughput on M4 (MB/s), per
            cursor count _x_ and unroll _y_, at _p_maj=0.80_, 2880 B.],
)<tab-fse-xy-m4>
