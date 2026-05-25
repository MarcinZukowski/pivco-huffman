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
