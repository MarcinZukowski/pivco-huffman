= Breaking the bit-barrier <ans>

Mention FSE, tANS/rANS. Discuss performance problems etc.
Limited benefit on a lot of distributions.
Applied only to non-data in zstd etc.

A tempting idea would be to optimize FSE in a manner similar to ph.
Alas, the author could not come up with a way to do it, mostly do to
linear dependency between decoded symbols related to the table state change.

Instead, the idea came up to _apply_ existing FSE as part of Huffman.

Thesis - applying per-bit fse on every node level _should_ give a very similar
compression as total FSE.

Benchmarks confirmed.

== Implementation

precomputed tables

== Tree splitting

== Benefits

* FSE is slow, but we do "1 bit of 8 tokens" at a time.
* only applied for nodes where it actually matters (mostly highly skewed)

== Results

Discuss how applying bit-wise FSE on every

