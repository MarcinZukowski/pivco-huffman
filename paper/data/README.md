# paper/data/

Collated benchmark CSVs + SQL views, consumed by the paper's `.typ`
sources via Typst's `#csv()`.  Files here are **generated**, not
hand-written.

Two layers per bench:

```
paper/data/
├─ td-naive-vs-opt.csv                          # long-form, from collate
└─ td-naive-vs-opt.hosts-x-decoders.csv         # wide view, from build-views
```

To regenerate after fresh results drop into `results/<bench>/`:

```sh
paper/bench.py collate     <bench>     # gathers raw CSVs -> long form
paper/bench.py build-views <bench>     # runs SQL views (in benches.yaml) -> wide files
```

## Long-form CSV (from collate)

Columns are always: `host, compiler, <bench-specific columns...>`.
The bench tool writes the bench-specific columns; `bench.py collate`
prepends `host` and `compiler` and rewrites any `-` in those values
to `_` so they survive PIVOT.

## Wide view CSVs (from build-views)

Driven by SQL embedded in `paper/benches.yaml` under each bench's
`views:` block.  DuckDB dialect; views can `PIVOT`, add derived
columns (ratios, formatting), join across multiple bench outputs,
etc.

## Reading wide views from Typst

The wide CSV is shaped to land directly in `#table()`: column order
matches the layout you want, headers are hand-written in the `.typ`.

```typst
#let rows = csv("data/td-naive-vs-opt.hosts-x-decoders.csv")
#let data = rows.slice(1)   // drop CSV header; column order is fixed by the SQL view

// Worked example: the table from the discussion.  Two-row header,
// distribution rows, two-block body (M4 / c8i).
#table(
  columns: 9,
  align: (left, right, right, right, right, right, right, right, right),
  inset: 6pt,
  table.header(
    table.cell(rowspan: 2)[*Distribution*],
    table.cell(colspan: 4)[*M4 Max* (GB/s)],
    table.cell(colspan: 4)[*Granite Rapids c8i* (GB/s)],
    [naive], [opt], [huf0_x2], [opt/huf0],
    [naive], [opt], [huf0_x2], [opt/huf0],
  ),
  ..data.flatten(),
)
```

Filtering / formatting choices live in the SQL view + the `.typ`
table layout; the long-form CSV stays neutral and reusable.

## Contract on identifiers

Every value that ever ends up as a column name (via PIVOT or
otherwise) must be SQL-safe -- `[a-z0-9_]` only, no dashes.
Bench tools enforce this for their own columns at emit time;
`bench.py collate` enforces it for `host` and `compiler` whose
canonical aliases (e.g. `test-c8i`, `clang-latest`) naturally contain
dashes.
