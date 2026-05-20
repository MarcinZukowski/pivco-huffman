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
