#import "conf.typ": anote, PH, he, mf, sym, pick-cols, todo, fair-filter, h0

= Introduction

Huffman encoding @huff is one of the most important algorithms in the area of compression.
Moffat and Turpin nicely put it in @moffat - it is very _enduring_:
despite the introduction of better-compressing encodings (e.g. @arithmetic or @dudaans),
70+ years on, it's still ubiquitous.

Note: formally, most modern systems don't necessarily use the _exact_ encoding proposed in @huff,
but rather "canonical" coding from @schwartz1964canonical.
In this paper we use the term "Huffman" for all versions of it.

== Classical Huffman tree



== Current Huffman SotA <sota>

Most modern Huffman implementations
#h0, Oodle

Most modern Huffman decoding implementations use a _decoding table_ idea, allowing decoding
an entire code without traversing it bit by bit.
The size of supported code lengths is typically constrained, e.g. to L=11 bits.
Then a table of 2^L is created, allowing the following code:
```c
  code = peek_bits(L);
  emit_symbol(decoding_table[code].symbol);
  skip_bits(decoding_table[code].numBits);
```

This or similar code is used in fast Huffman decompressors like Huff0 (@fse).
Two typical approaches to accelerating it are using multiple cursors (@giesen2014interleaved, @giesen2023oodle)
or using a table that decodes 2 symbols instead of one.

We measured various Huffman decoding implementations, and the most performant ones we found
were #h0 (from @fse) and Oodle's Huffman decoder (@giesen2021oodle).

Here are the measured bandwidths on two example datasets on two hosts:

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
#table(
  columns: 6,
  inset: 5pt,
  align: (left, right, right, right, right, right, right),
  table.header(
    table.cell(rowspan: 2)[*Dataset*],
    table.cell(rowspan: 2)[*Host*],
    table.cell(colspan: 2)[*#h0*],
    table.cell(colspan: 2)[*oodle-huffman*],
    [enc MB/s],[dec MB/s],
    [enc MB/s],[dec MB/s],
  ),
  table.cell(rowspan: 2)[proba80],     [M4], .._hp("m4", "proba80"),
                                       [c8i], .._hp("c8i", "proba80"),
  table.cell(rowspan: 2)[prose_pride], [M4], .._hp("m4", "prose_pride"),
                                       [c8i], .._hp("c8i", "prose_pride"),
)<tab-huffman-perf>

Throughput is measured in realistic per-call mode (table rebuilt per block);
#h0 is stock `HUF_decompress`, oodle-huffman is Oodle's `OodleLZ` Huffman
backend.

This is a respectable performance.
Still, in this paper we investigate if it could be improved by using a completely different approach.

== Motivating Example: Hash Join in Databases <hj>

Hash table lookup is one of the most performance-intensive operations in many systems, including
databases.
Below, we can see the pseudocode of a simple linear-hashing lookup, and its relation to Huffman decoding.

#table(
  columns: (50%, 50%),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Huffman symbol decoding*], [*Hash table lookup*]
  ),
  [```js
    state = root

    while not is_leaf(state)
      if read_bit() == 1:
        state = state->right
      else:
        state = state->left
    return state
    ```
  ],
  [```js
    hash = compute_hash(key)
    pos = hash_table_first(hash)
    while not hash_table_empty(pos)
      val = hash_table_value(pos)
      if val == key:
        return true
      pos = hash_table_next(pos)
    return false
    ```
  ],
)

Both problems can be seen as a state-machine traversal, and in both, data dependencies in the loop and unpredictable branching prevent the CPU from achieving high performance.
Hash join additionally performs an expensive memory lookup causing additional stalls.

In @zuk09, Section 5.3.3.2, the author proposed an alternative hash table lookup approach based on the idea of going through
each node in the state machine not for one, but for a _vector_ of records,
presented in this simplified pseudocode:

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
  // split into hits if equal, active if not - those needs more work
  split_on_equality(comp_results, &active, &hits)
  // get all the next positions for all still active records
  active = hash_table_next(active)
// misses have all miss positions, hits have all hit positions
```

This approach, while more complex and seemingly labor-intensive (definitely issues more CPU instructions),
in each phase exposes to CPUs a lot of simple, independent operations and avoids any data or control dependencies.
As a result, it achieves a significant performance benefit (even >10x) over the _scalar_ approach.


#anote[
So I've been trying to apply this general approach to a few different problems, including compression, but also
stuff like regular expression processing.

Got some decent results on VarInt, but then I saw Daniel Lemire's Stream VByte @lemire2017streamvbyte and gave up - can't beat that.

Luckily, with Huffman, it seems to "click" reasonably well.
]
