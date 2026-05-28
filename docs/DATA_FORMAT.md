# Encoded Data Format

> **Last content review:** _NEVER_

The encoded data is a DFS-ordered bitstream matching the tree walk.
See [`src/pivco_huffman_wire.h`](../src/pivco_huffman_wire.h) for the
authoritative format spec.  Per-node record (v0.3, since 2026-05-13):

```
[optional K_right_header : uint16 LE, 2 bytes]   if kr_header_needed()
[FSE marker             : uint8,     1 byte]     always
[bitmap body]                                    marker == 0: raw n-bit
                                                  bitmap, ceil(n/8) bytes
                                                 marker != 0: [fse_len:u16
                                                  LE][fse_payload]
                                                  (FSE-compressed bitmap)
```

The K_right header (2 bytes) is emitted at every internal node whose
right child is non-leaf, letting the bottom-up decoder size each
child's output buffer ahead of time instead of computing it from a
bitmap popcount.  Adds 2 bytes per qualifying node; saves a popcount
pass per node at decode time.

The FSE marker byte gates per-node entropy coding of the partition
bitmap: when the bitmap is heavily skewed (one bit value dominates
≥ 62.5%) and the per-codeword cost ratio passes the commit gate, the
encoder ships an FSE-compressed payload instead of the raw bitmap.
Decoder dispatches generically based on the marker byte.  Wire
overhead when FSE doesn't fire: 1 byte per non-flat internal node
(~0.06% on proba80, ~1.6% on incompressible image data).
**FSE coding is experimental and disabled in the headline bench
numbers** (`--no-fse`); the marker byte is still emitted unconditionally
so the wire format is stable when the runtime gate flips.

At each flat-subtree root with `n` active symbols and depth `D`, a
single `ceil(n·D/8)`-byte packed region is stored — one `D`-bit
code per element, no per-level framing, no marker byte (the flat
path doesn't FSE-code).

The decoder has the Huffman tree, so it knows which path each node
uses and exactly how many bytes to consume.  No continuation bitmaps
or stream-level metadata are needed — the Huffman tree structure
plus the per-node K_right / FSE markers are sufficient.

Encoded size equals traditional Huffman (sum of code lengths) plus
byte-alignment rounding, which is typically 1-4% overhead — minus
the FSE win on skewed bitmaps (~25% on proba80, ~24% on
calgary_pic).  The flat-subtree format is marginally tighter than
bitmap-per-level on flat-heavy regions (one tail padding for the
whole packed region vs `D` per-level paddings).

For the higher-level pivcohuf file container that wraps this stream,
see [`include/pivcohuf_file.h`](../include/pivcohuf_file.h).
