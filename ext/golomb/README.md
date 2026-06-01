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
