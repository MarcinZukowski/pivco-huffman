# Notes for Daniel Lemire on counters/bench

> **Last content review:** _NEVER_

Findings from porting [lemire/counters](https://github.com/lemire/counters) v4.1.0 into pivco-huffman to time NEON `partition_8` on Apple M4. **The framework is correct;** these are user-side gotchas worth documenting in the README.

Reproducer: `extras/bench/bench_lemire_compare.cpp` (commit `a46b0e1`).

## Gotcha 1 — `[&]` capture reloads `vector::data()` every iteration

The natural pattern:

```cpp
auto agg = counters::bench([&]() {
    for (int i = 0; i < n; i++) acc += kernel(src.data() + i, ...);
});
```

…produces an inner loop with one `ldr xN, [stack_slot]` per captured vector per iteration, because the compiler cannot prove the lambda body doesn't mutate `src` through the captured reference. On a `partition_8` kernel with 4 captured vectors, that's 4 extra loads/iter.

**Fix:** capture pointers by value:

```cpp
const T *srcp = src.data(); /* etc. */
counters::bench([srcp, /* ... */]() {
    for (int i = 0; i < n; i++) acc += kernel(srcp + i, ...);
});
```

Inner-loop diff (Apple M4, `-O3`):

```
[&] capture (slow):                 [ptrs] capture (fast):
  ldr x10, [x23]  ; reload src       (no reload)
  ldr x12, [x19]  ; reload m         (no reload)
  ldr x13, [x22]  ; reload R         (no reload)
  ldr x14, [x21]  ; reload L         (no reload)
  ldr q0, [x10, x11]                 ldr q0, [x_const, x11]
  ldp q1, q2, [...]                  ldp q1, q2, [...]
  tbl/tbl/str/str                    tbl/tbl/str/str
```

Per-iter difference: ~4 cycles, observed throughput delta ~33% on M4.

## Gotcha 2 — Static arrays + unobserved writes → kernel deleted

Putting test buffers in static arrays so the addresses are compile-time constants is tempting, but if the *only* observed effect of the kernel is the volatile sink, the optimizer may dead-store-eliminate writes to other buffers.

In our reproducer, `lemire_static_arrays` reported `0.149 ns/call`. Disassembly showed the entire SIMD partition kernel was deleted — only `compress_popcnt[mask]` lookups + accumulate survived.

**Mitigations:**
- Use heap buffers (`std::vector`) — the compiler can't prove no-aliasing for runtime pointers.
- Or insert `asm volatile("" : : "r"(buf) : "memory")` on each output buffer inside the loop.
- Or reference all output buffers from another translation unit / through a function pointer.

A note on this in the README would help. The danger here is that the bench *succeeds*, the numbers are stable across runs, but they're fictional.

## Suggested API tweak (optional)

A `bench_per_iter` helper would steer users away from manual `agg.fastest_elapsed_ns() / n_groups`:

```cpp
auto agg = counters::bench_per_iter(n_iters, [srcp, dstp](size_t i) {
    return kernel(srcp + i, dstp + i);
});
// agg.cycles() now reports per-iter, not per-fn-call
```

Optional. Current API works once you know the gotchas above.

## What we measured (Apple M4 P-core, NEON `partition_8`, n_groups=1024)

| variant                                  | ns/call | what's happening |
|---|---|---|
| L4 lemire static arrays                  | 0.40    | dead-store elim deletes kernel |
| L5 lemire static + asm barrier           | 0.40    | same — barrier doesn't recover |
| **L6 lemire vec, value capture**         | **0.57**| kernel runs, pointers loop-invariant |
| L1/L2/L3 lemire vec, `[&]` capture       | 0.86    | kernel runs, pointers reloaded each iter |
| D3 direct timer, vector buffers          | 0.85    | matches L1 — confirms not lemire's fault |
| D1/D2 direct timer, static arrays        | 0.46    | partial dead-store elim |

The 2.5x gap between "lemire vec [&]" and "lemire static" that initially surprised us decomposed cleanly into the two gotchas — neither is a framework bug.
