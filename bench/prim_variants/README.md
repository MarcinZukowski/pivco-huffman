# Primitive-variant graveyard

A home for primitive kernels we've tried but don't ship, so they survive past
the commit that rejected them and can be **re-evaluated on new hardware /
compilers**. Consumed only by `bench_prim` (`--variants`); **zero dependency on
`src/`** — nothing here is built into the codec.

## Why this exists

Over PH's life we try many variants per primitive. Today a rejected one is lost
three ways: deleted in a commit (git-archaeology to recover), buried in a
one-off `extras/bench/bench_*.c` with its own duplicated harness (bit-rots,
never re-run), or as a suffixed inline in `bench_prim.c` (only the survivors).
And "re-run *every* merge_vec_vec we ever tried on this new core" is impossible
without rebuilding each by hand. The graveyard fixes that: one harness
(`bench_prim`), one registry, variants as data.

## What earns an entry (curation)

- **Every external contribution** (PRs, suggestions) — always.
- **Every kernel version that ever shipped** in production — a good baseline set.
- **A few notable parked experiments** — e.g. wins-on-AMD / loses-on-Intel, or
  a technique that may pay off on future ISAs.

Trivial losers don't earn an entry. The bar is "different *technique*, or a
uarch-dependent result worth re-checking."

## Layout

```
prims.h              registry: gv_status_t + GV_VARIANT macro + the contract
prims-merge.h        merge_vec_vec / cst_vec / cst_cst / flat variants
prims-partition.h    enc_partition_full / none / right variants
prims-pack.h         enc_pack_dN variants            (todo)
prims-flat.h         flat_dN_unpack variants         (todo)
```

Variants of a **family** share one file; each is **keyed at the fine grain**
(`merge_vec_vec`, not `merge`) so `--variants=merge_vec_vec` runs just that one.

## Contract (per variant)

- `static void fn(const ctx_t *c);` — same signature as bench_prim's kernels.
- Same logical semantics as the primitive: `bench_prim` verifies it
  byte-for-byte against the scalar reference before timing it.
- ISA-gated (`#if defined(USE_NEON_KERNELS)` / `USE_*_KERNELS`).
- Frozen: it's an independent copy. The "production" row in a comparison is
  bench_prim's wrapper calling the real `prim_*`, never copied here.

## Adding a variant

1. Drop the kernel into the right `prims-<family>.h`, ISA-gated, `static`.
2. Register it in that file's `gv_register_<family>()`:
   ```c
   GV_VARIANT(ST_PART, "neon_prefix", "neon", GV_GRAVEYARD,
              "Author / <commit>", "one-line why-parked / last result", /*inplace=*/1,
              gv_simd_part_full_prefix);
   ```
3. `bench_prim --variants` (or `--variants=enc_partition_full`) runs it next to
   the production kernel + scalar reference, verified and timed.

## Usage

```sh
bench_prim --variants                      # production + scalar + all graveyard
bench_prim --variants=enc_partition_full   # one logical primitive
```

## Seeded so far

- `prims-partition.h` — `neon_prefix` (full/none/right): Jeff Plaisance's
  64-codes/iter wide-mask + SWAR prefix-sum partition (`6d61760`), an
  independent rediscovery of the shipped COM idea. M4: FULL +15–18% slower than
  the shipped COM, RIGHT ~2–3% slower, NONE ~3–4% faster — never benched on
  Graviton (the gate Jeff flagged).
- `prims-merge.h` (merge_vec_vec):
  - `old` — the pre-COM64 shipped stride-16 merge (historical baseline).
  - `jeff` / `jeff64` — Jeff Plaisance's prefix128 / 64-stride merge (PR #4).
    M4: `jeff` ~10% *faster* than the shipped COM64, but it's the M1/M4-fast,
    Graviton-slow form we parked — ARM-untested, needs a c8g run.
