# paper/

The PIVCO-Huffman paper, plus the figure registry and rendering
pipeline.

## Building the paper

```sh
make          # PDF + HTML
make pdf      # PDF only
make html     # HTML only
make watch    # live-rebuild PDF on save
make clean
```

Output lands in `paper/out/`.  Typst's HTML backend is experimental;
`style.css` is inlined by `ph.typ` so the HTML is self-contained.

## Benchmarks

`paper/benches.yaml` is the registry of paper benchmarks (parallel
to `figures.json` but YAML since there's no in-browser consumer).
Each entry describes: source, build + run commands, host x compiler
matrix, and output directory.  Every bench CLI accepts
`--csv-out=PATH` and writes a long-form CSV that the paper consumes
via Typst `#csv()`.

`paper/bench.py` has four subcommands:

```sh
paper/bench.py list                        # what's registered
paper/bench.py show-how <bench-name>       # build + run recipe per host
paper/bench.py collate <bench-name>        # gather latest CSVs into paper/data/
paper/bench.py build-views <bench-name>    # run SQL views (DuckDB) over the collated CSV
```

Workflow:

1. `paper/bench.py show-how <name>` prints the exact reproduction
   commands for every host x compiler cell.  Run them manually
   (bench.py never SSHes -- cross-host automation gets brittle fast,
   and EC2 instances aren't always up).
2. Each command writes `results/<bench-name>/{host}-{compiler}-{date}-{sha}.{csv,txt}`.
3. `paper/bench.py collate <name>` finds the lexically-newest matching
   CSV per (host, compiler) cell, adds `host` and `compiler` columns,
   and writes `paper/data/<name>.csv`.
4. The paper reads `paper/data/<name>.csv` with `#csv()` and applies
   per-table filtering in Typst (see `paper/data/README.md` for the
   worked example).

The `compiler` field in the matrix uses aliases like `clang-latest`
or `gcc-latest`, resolved to per-host binary paths under
`hosts.<name>.compilers` in `benches.yaml`.  Same alias means
"newest compiler of this family available on this host" -- the
actual versions differ per host.

### Setup for bench.py

```sh
source .venv/bin/activate    # the figures venv suffices
uv pip install pyyaml duckdb
```

### CSV identifier contract

Any value that ends up as a column name in a wide view (via DuckDB
`PIVOT`) MUST be a SQL-safe identifier -- `[a-z0-9_]`, no dashes.
Bench tools enforce this for their own value columns at emit time;
`bench.py collate` rewrites `-` -> `_` for `host` and `compiler`
whose canonical aliases (e.g. `test-c8i`, `clang-latest`) naturally
contain dashes.  See `paper/benches.yaml` for the full contract
notes.

## Figures

`paper/figures/figures.json` is the registry.  Two layers:

- **tools** — viz programs.  Each names a `cli` (the SVG renderer) and
  a `web` page (the live-editing URL), plus tool-wide default params.
- **figures** — named instances of a tool with parameter overrides.

`paper/figures/fig.py` dispatches:

```sh
paper/figures/fig.py list                # list known figures
paper/figures/fig.py svg --all           # render every figure to .svg
paper/figures/fig.py svg <name> [<name>] # render named figures
paper/figures/fig.py svg --all --filter pivot   # subset by substring
paper/figures/fig.py web <name>          # print the browser URL
```

SVGs land next to `figures.json` by default; override with `--out-dir`.

`paper/figures/fig-web.html` is the browser redirector: open it with
`?name=<figure>` and it bounces to the tool's web viewer with the
merged params applied.

### Setup for SVG rendering (one-time)

The SVG capture path (currently only `extras/figures/capture_tree_viz.py`,
for the `tree_viz` tool) drives a headless Chromium via Playwright:

```sh
# from the project root
uv venv
source .venv/bin/activate
uv pip install playwright
playwright install chromium       # ~150 MB browser binary
```

After that, `paper/figures/fig.py svg --all` works because it shells
out to `sys.executable`, which is now the venv's Python.

No-activation alternative:

```sh
uv run --with playwright paper/figures/fig.py svg --all
```

(`playwright install chromium` still needs to run once.)
