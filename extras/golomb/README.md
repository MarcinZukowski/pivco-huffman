# Batch unary decoding — reproduction of fgiesen post

Local reproduction of the four unary decoders from Fabian Giesen's
2026-05-30 post, *"Simple batch decoding of unary codes"*:

  https://fgiesen.wordpress.com/2026/05/30/simple-batch-decoding-of-unary-codes/

The four decoders, in the post's order:

  1. `decode_serial`    — naive one-code-at-a-time, 56-bit refill
  2. `decode_pair`      — two codes per iteration via `bitbuf & (bitbuf-1)`
  3. `decode_tunstall`  — byte-at-a-time table lookup (struct-of-arrays table)
  4. `decode_tunstall64`— byte-at-a-time, 64-bit-packed table, single store

The blog only shows fragments — the surrounding bit-buffer refill, loop
control, encoder, and test harness in `main.c` are filled in here.
Each decoder body labels what is verbatim from the post vs. inferred.

The table generator (256-entry, 64-bit packed) is verbatim from the post.

## Build & run

    make            # builds ./golomb
    ./golomb        # encode random geometric data, decode 4 ways, verify equality + print throughput
    ./golomb 0.2    # set the geometric parameter p (default 0.5); lower p = more zeros = deeper unary

## pivco merge decoders

Beyond the four post decoders, the harness adds a bottom-up *merge* decode of
the same unary stream (`decode_pivco` scalar / `decode_pivco_neon` /
`decode_pivco_avx512`).  It treats the unary code as a tree of levels and walks
it with `merge_vec_cst_plus1` — the production `merge_vec_cst` primitive with a
constant right operand `0xFF` and a post-merge `+1` (so the bit==1 lanes wrap
`0xFF → 0`).  One merge per unary level, ping-ponging two buffers.

- NEON: ported from the production COM64 `merge_vec_cst_neon` (`5cccccc`) — 64
  codes/iter as 4 independent chunks whose left (vec) cursor comes from a
  `popcnt * 0x0101…` byte prefix-sum, so there's no loop-carried per-chunk
  cursor.  Replaced an earlier V4-style form that chained the cursor every 16
  codes.
- AVX-512: uses `vpexpandb` (`maskz_expandloadu_epi8`) — the production
  `merge_vec_cst_avx512` form.  A single-instruction byte-expand; COM is the
  NEON workaround for not having it, so there is no COM variant on VBMI2.

### Results (2026-06-18, best warm ns/code, lower = better)

`decode_pivco_neon` on Apple M4, NEON baseline (V4 cursor-chained) vs the
COM64 port; `decode_pivco_avx512` on c8i (Granite Rapids, clang-20):

| p     | M4 NEON V4 | M4 NEON COM64 | M4 speedup | c8i AVX-512 (vpexpandb) |
|-------|-----------:|--------------:|:----------:|------------------------:|
| 0.8   | 0.05       | 0.04          | ~1.25×     | 0.03                    |
| 0.666 | 0.06       | 0.05          | ~1.20×     | 0.04                    |
| 0.5   | 0.08       | 0.06          | ~1.33×     | 0.05                    |
| 0.333 | 0.12       | 0.10          | ~1.20×     | 0.08                    |
| 0.25  | 0.16       | 0.13          | ~1.23×     | 0.11                    |
| 0.20  | 0.20       | 0.16          | ~1.25×     | 0.15                    |

COM64 wins ~20–33% across the distribution on NEON (all decoders verify
byte-exact against the encoder).  Lower p → more zeros → deeper unary tree →
more merge levels → higher ns/code.
