# tree_viz WASM: the real joint_lengths.c in the browser

`figures/tree_viz.html`'s `joint:` control runs the REAL library length
pipeline — `src/joint_lengths.c` + `src/huffman_table.c` compiled to
wasm32 — instead of a JS port, so the visualized shape is exactly what
`pivco_build_table` adopts, adoption-guard verdict included.  No port
to drift while #20's λ/guard tuning is active.

## Build

```sh
./extras/tree_viz_wasm/build.sh    # needs homebrew llvm (clang wasm32 + wasm-ld)
```

Produces `figures/tree_viz_wasm.js`: the wasm binary (25 KB)
base64-embedded in a plain `<script src>`-loadable file, so the viz
keeps working from `file://` (where `fetch()` of a .wasm is blocked).
No emscripten: the build is `-nostdlib` against the stub headers in
`include/` plus `shim.c` (bump allocator for the library's single
malloc/free pair, mem functions, `pivco_cfg_default`,
`pivco_check_fail` -> trap).  wasm32 defines neither `__ARM_NEON` nor
`__SSE4_1__`, so joint_lengths takes its documented complete scalar
path.  The only imports are `env.log2` / `env.ceil` = `Math.log2` /
`Math.ceil` (`-ffreestanding` implies `-fno-builtin`, so even ceil
doesn't lower to the f64.ceil instruction).

Exports (see `viz.c`, loader API in `loader.js`):

- `viz_plain_lengths()` — two-queue Huffman build + length-limit to
  `PIVCO_MAX_CODE_LEN`, via the real `pivco_build_table`.
- `viz_joint_lengths(effort, fse_enabled)` — joint shaping on top,
  exactly as the library runs it; returns adopted vs kept-plain.

## Fidelity (verified 2026-08-19)

`test_fidelity.js` (wasm via node) vs `native_dump.c` (same `viz.c`
entry points compiled natively), over all 30 bench distributions x
{plain, effort 1/2/3 x fse on/off}:

- **wasm == glibc-built native, bit-for-bit** (c6a, gcc): 0 differing
  lines out of 240.  V8's `Math.log2` and glibc's `log2` agreed on
  every input these runs hit — an empirical coincidence on the bench
  set, not an identity: over 2M random doubles in (0,1) the three
  libms disagree pairwise by 1 ulp (Apple-vs-V8 0.43%, glibc-vs-V8
  0.43%, glibc-vs-Apple 0.08% of inputs).  Safari's JSC is a fourth
  log2.  If per-browser determinism is ever wanted, vendor one fixed
  log2 (musl/fdlibm) into the module — it then has zero imports.
- **Apple-libm native differs**: 9 lines — 6 equal-cost effort-3 shape
  flips and **3 adoption-verdict flips** (guard-edge cases where
  dp_time/base_time sits within one ulp of the guard threshold).
  `-ffp-contract=off` changes nothing, so this is purely log2
  provenance: **production joint shapes are already libm-dependent
  (M4 vs fleet)** — a datum for the #20 guard tuning (the guard
  boundary needs margin, or the cost model a libm-independent log2,
  if cross-platform shape reproducibility ever matters).
- native scalar == native NEON on all inputs (the FMA/last-ulp tie
  caveat in joint_lengths.c does not fire on the bench set).

So the wasm module matches the library as the fleet builds it; on
macOS-built binaries, guard-edge ties can differ — same class as the
existing macOS/Linux difference.  Adoption at effort 3 (PH pricing)
for the record: adopted on proba14/02, bell_s30/s80, zipfian,
geometric, html_wiki, prose_pride, image_jpeg, json_api, source_c,
log_apache, chinese_text; guard kept plain lengths on proba80/50,
bell_s10, uniform, english, sparse_*, two_sym_*, flat_M*, dna, csv,
gzip_random, calgary_pic — matching the E2E effort-sweep movers
exactly.

## Viz usage

`joint:` select (off / 1 balanced / 2 faster-dec / 3 fastest-dec) +
`joint-fse` checkbox (PHA-tax pricing in the guard model).  Status
line shows the verdict.  URL params `joint=` / `joint-fse=`.  The
library length-limits at 11, overriding the max-len slider; max-skew
overrides joint.  If `tree_viz_wasm.js` is missing, the control falls
back to the JS plain-Huffman build with a status note.
