#import "conf.typ": _html, _pdf, _fmt, anote, setup, PH, he, sym, PHA, todo

= Challenges and opportunities

While #PH is an interesting approach, the author does not claim
it's a drop-in replacement to the existing solution.
In this Section we discuss challenges it faces, but also
further research and improvement opportunities.

== Data size

#PH performs a tree traversal, with the tree size
comparable to the dictionary size (minus the flat-subtree optimization).
Additionally, on each level, it requires a sizable amount of data
to achieve high performance (typically 64+ codes).
That means, that a minimum data size at which its application
makes sense is in the range of at least kilobytes.

== SIMD-dependency

To achieve high performance, #PH requires a careful
design of its primitives using SIMD.
As a result, it might be challenging or impossible
to port it to hardware platforms or programming
languages that do not provide such capabilities.

== Primitive optimization

While presented primitives already provide good
performance, optimizations are surely possible,
especially when targeting different hardware.

For example, we prototyped significantly
faster `merge-flat-D2/D3` primitives for AVX-512
that use bit-sliced packing to achieve
around 1.5x performance. But this version
required a wire format change, which resulted
in slower performance on other systems.
Still, optimizations like that are feasible
in specific scenarios.

== GPU porting / acceleration

Presented primitives map pretty well to GPU.
A naive reimplementation of `partition` primitive
for GPU on M4 resulted in ~40 GB/s throughput.

#todo[A simple experiment]

== Better sub-bit strategies

While FSE-per-level is a promising optimization,
other strategies are possible.
For example, specialized bitmap compressions methods
could provide an interesting design point
in a performance/compression ratio space.

== Impact on LZ-codecs

Huffman and other entropy-codecs are rarely applied standalone,
they are most commonly used as a step after LZ-family or similar compressors.
They often change the data distribution dramatically, resulting
in #PH's benefits possibly difeferent.

== CPU development trends

#figure(
  image("plots/sweep_uarch_dec_op.svg"),
  caption: [#PH performance over Huff0 on 3 CPU families on AWS (across datasets from @datasets)]
)<fig-trends>

While the presented #PH performance focuses on the recent CPUs,
it is interesting to see how it performes on older hardware.
@fig-trends shows that #PH provides consistent benefits on all these architectures,
even going back to 2012, but the benefit _increases_ with newer CPU types.
The main reason for that is the improvement in SIMD capabilities and performance.
Additionally, #PH presents a lot of simple, predictable code that utilizes
 modern wide out-of-order CPUs better.
If that trend continues, e.g. with the possible introduction of SVE-256 (@arm-neoverse-v1),
 the benefits of #PH _might_ be even more valuable in the future.
