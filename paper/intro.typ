#import "conf.typ": anote, PH, OOH, he, mf, sym, pick-cols, todo, fair-filter, h0

= Introduction

Huffman encoding @huff is one of the most important algorithms in the area of compression.
Moffat and Turpin nicely put it in @moffat - it is very _enduring_:
despite the introduction of better-compressing encodings (e.g. @arithmetic or @dudaans),
70+ years on, it's still ubiquitous.

Note: formally, most modern systems don't necessarily use the _exact_ encoding proposed in @huff,
but rather "canonical" coding from @schwartz1964canonical.

== Classical Huffman tree

#he("gridtable")[
#table(
  columns: (50%, 50%),
  stroke: 0pt,
  align: (center+horizon, left+horizon),
  [#figure(
    mf("huf-tree"),
    caption: [Classical Huffman tree for the word "huffman"]
   )<fig-huf-tree>],
  [#figure(
    [```js
  node = root

  while not is_leaf(node)
    if read_bit() == 1:
      node = node->right
    else:
      node = node->left
  return node.symbol
  ```
    ],
    caption: [Naive Huffman decoding for one symbol]
  )<fig-huf-decode>
  ]
)
]

In classical Huffman coding, each symbol is encoded using a code (a sequence of bits),
with more frequent symbols getting shorter codes.
@fig-huf-tree shows a Huffman tree for the word "huffman", and @fig-huf-decode
shows a naive decoding algorithm for decoding one symbol.
For example, to decode symbol #sym("h"), we traverse the tree using bits #sym("1 0 1") to get
to the proper leaf node representing that symbol.

== Modern Huffman solutions <sota>

Implementation from @fig-huf-decode is not very performant, as it uses a lot of operations
and is not friendly for modern CPUs.
Instead, modern Huffman decoding implementations use a _decoding table_, which allows decoding
an entire symbol without traversing its code bit by bit.
The size of supported code lengths is typically constrained, e.g. to _L=11_ bits.
Then a table of size _2^L_ is created, allowing the following implementation:
```c
  code_bits = peek_bits(L);
  emit_symbol(decoding_table[code_bits].symbol);
  skip_bits(decoding_table[code_bits].numBits);
```

Such code can be further accelerated by using multiple cursors (@giesen2014interleaved, @giesen2023oodle),
or by building a table that decodes two symbols in one iteration instead of one.

We measured various Huffman decoding implementations, and the most performant solutions we found were:

- *#h0* - part of the open-source FSE (@fse) library, which is also a building block of the popular zstd compression library (@zstd).
  Implemented in pure C, permissive license.
- *#OOH* - Huffman decoder from Oodle (@giesen2021oodle) - a proprietary compression library by RAD Game Tools.
  Implemented in C with a lot of assembly optimizations.
  Oodle requires a license for most uses.

Here are the measured bandwidths on two example datasets on two hosts (see @testing-method for more info):

#let fair = csv("data/fair.csv")
#let _na(v) = if v == "na" { [—] } else { [#v] }
// opaque enc/dec MB/s for huff0(stock) + oodle-huffman at (host, dataset)
#let _hp(host, ds) = {
  let f(method) = {
    let r = fair-filter(fair, (host: host, dataset: ds, method: method),
                        ("enc_op", "dec_op"))
    if r.len() == 0 { ("na", "na") } else { r.first() }
  }
  let hf = f("huf0")
  let oo = f("oo_huff")
  (_na(hf.at(0)), _na(hf.at(1)), _na(oo.at(0)), _na(oo.at(1)))
}

#figure(numbering: none)[
#table(
  columns: 6,
  inset: 5pt,
  align: (left, center, right, right, right, right, right),
  table.header(
    table.cell(rowspan: 2)[*Dataset*],
    table.cell(rowspan: 2)[*Host*],
    table.cell(colspan: 2, align: center)[*#h0*],
    table.cell(colspan: 2, align: center)[*#OOH*],
    [enc MB/s],[dec MB/s],
    [enc MB/s],[dec MB/s],
  ),
  table.cell(rowspan: 2)[proba80],     [M4], .._hp("m4", "proba80"),
                                       [c8i], .._hp("c8i", "proba80"),
  table.cell(rowspan: 2)[prose_pride], [M4], .._hp("m4", "prose_pride"),
                                       [c8i], .._hp("c8i", "prose_pride"),
)<tab-huffman-perf>
]

These are impressive results.
Still, in this paper we investigate if the performance could be further improved by using a completely different approach.

== Motivating Example: Hash Join in Databases <hj>

Hash table lookup is one of the most performance-intensive operations in many systems, including
databases.
Below, we can see the pseudocode of a simple linear-hashing lookup:

```js
  hash = compute_hash(key)
  pos = hash_table_first(hash)
  while not hash_table_empty(pos)
    val = hash_table_value(pos)
    if val == key:
      return true
    pos = hash_table_next(pos)
  return false
```

Just like Huffman decoding from @fig-huf-decode, this problem can be seen as a state-machine traversal.
In both cases, data dependencies in the loop and unpredictable branching prevent the CPU from achieving high performance.
Hash join additionally performs an expensive memory lookup causing additional stalls.

@zuk09 (Section 5.3.3.2) proposed an alternative hash table lookup approach based on the idea of going through
each node in the state machine not for one, but for a _vector_ of records,
presented in this pseudocode:

```js
  misses = []                          // miss input positions
  hits = []                            // hits input positions
  hash = compute_hash(keys)            // all input hash values
  active = hash_table_first(hash)      // input positions we're still looking up
  while not active.empty():            // if we still have work to do
    // move empty slots to misses, reduce active
    hash_table_split_empty(&active, &misses)
    // get all values from the hash table for active indices
    vals = hash_table_vals(active)
    // compute comparisons
    comp_results = compare(vals, keys, active)
    // split into hits if equal, active if not - those need more work
    split_on_equality(comp_results, &active, &hits)
    // get all the next positions for all still active records
    active = hash_table_next(active)
  // misses have all miss positions, hits have all hit positions
```

This approach, while more complex and seemingly labor-intensive (definitely issues more CPU instructions),
in each phase exposes to CPUs a lot of simple, independent operations, avoids any data or control dependencies,
and allows overlapping memory accesses.
As a result, it achieves a significant performance benefit (even >10x) over the _scalar_ approach.

#anote[
So I've been trying to apply this general approach to a few different problems, including compression, but also
stuff like regular expression processing.

Got some decent results on VarInt, but then I saw Daniel Lemire's Stream VByte @lemire2017streamvbyte and gave up - can't beat that.

Luckily, with Huffman, it seems to "click" reasonably well.
]
