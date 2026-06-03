// Shared helpers for paper plots (grouped column charts from data/fair.csv).
//
// Plots are NOT mf() figures: no HTML visualiser (unlike tree_viz).  Each is
// rendered to a standalone SVG here (Makefile, one per host) and included in
// the paper with a plain `image("plots/<name>-<host>.svg")`.
//
// Single source of truth: /data/fair.csv (root-relative) — the same
// long-format table the result tables read, so a re-sweep + rebuild updates
// both.  Bars are hand-drawn in cetz (not cetz-plot) so we can cap the y-axis
// and label clipped outliers (`cap:`), which cetz-plot's columnchart can't do.

#import "@preview/cetz:0.3.4"

#let fair = csv("/data/fair.csv")
#let _h = fair.first()
#let _ci(n) = _h.position(c => c == n)
#let cell(host, ds, m, metric) = {
  let rows = fair.slice(1).filter(r =>
    r.at(_ci("host")) == host and r.at(_ci("dataset")) == ds and r.at(_ci("method")) == m)
  if rows.len() == 0 { "na" } else { rows.first().at(_ci(metric)) }
}
#let num(s) = if s == "na" { 0.0 } else { float(s) }

// (canonical name, short x-axis label)
#let dsets = (
  ("proba80", "pb80"), ("english", "eng"), ("html_wiki", "html"),
  ("prose_pride", "prose"), ("image_jpeg", "jpeg"), ("json_api", "json"),
  ("dna_fasta", "dna"), ("chinese_text", "chin"), ("calgary_pic", "calg"),
)

// Series colors + patterns live in /style.typ — re-exported here so the
// plot files can `import "common.typ": ..., colors, patterns` without
// having to know about the upstream stylesheet.
#import "/style.typ": colors, patterns

// Pattern kinds, split by HOW they're drawn:
//   - centered marks (single column of dots / one vertical line in the
//     horizontal CENTER of the bar): drawn per-bar with cetz primitives
//     so they're always exactly centered regardless of bar width.
//   - area fills (continuous hatch / checkerboards): use a native Typst
//     `tiling(...)` pattern as the rect's fill.
//
// Kinds (passed as the optional 5th element of the series tuple):
//   "solid"     — flat color (default)
//   "dot"       — column of dots, centered horizontally
//   "hlines"    — horizontal-line hatch
//   "d1"        — diagonal / (positive-slope hatch)
//   "d2"        — diagonal \ (negative-slope hatch)
//   "checker"   — square (horizontal/vertical) checkerboard
//   "checkerd"  — diagonal checkerboard (diamonds)
#let _pat_color  = rgb(0, 0, 0, 70%)
#let _pat_stroke = 0.6pt + _pat_color
#let _pat_size   = 6pt

// "area fill" patterns: return a tiling() Typst fill value.  For "solid" and
// the centered-mark patterns, just return the plain color — the mark will be
// drawn separately on top.
#let _pat_fill(color, kind) = {
  if kind == none or kind == "solid" or kind == "dot" { return color }
  let s = _pat_size
  let half = s / 2
  let bg = place(rect(width: s, height: s, fill: color, stroke: none))

  if kind == "hlines" {
    tiling(size: (s, s), bg
      + place(line(start: (0pt, half), end: (s, half), stroke: _pat_stroke)))
  } else if kind == "d1" {
    tiling(size: (s, s), bg
      + place(line(start: (0pt, s), end: (s, 0pt), stroke: _pat_stroke)))
  } else if kind == "d2" {
    tiling(size: (s, s), bg
      + place(line(start: (0pt, 0pt), end: (s, s), stroke: _pat_stroke)))
  } else if kind == "checker" {
    tiling(size: (s, s), bg
      + place(dx: 0pt,  dy: 0pt,  rect(width: half, height: half, fill: _pat_color, stroke: none))
      + place(dx: half, dy: half, rect(width: half, height: half, fill: _pat_color, stroke: none)))
  } else if kind == "checkerd" {
    tiling(size: (s, s), bg
      + place(polygon(
          fill: _pat_color, stroke: none,
          (half, 0pt), (s, half), (half, s), (0pt, half))))
  } else {
    color
  }
}

// Cetz overlay for the centered-mark patterns.  Coordinates are in cetz
// units (cm).  Called inside a cetz.canvas after the rect is drawn.
// (Tiling patterns above handle the rest; this only does marks that need
// to be exactly centered on the bar regardless of bar width.)
#let _pat_overlay(kind, x0, y0, x1, y1) = {
  import cetz.draw: circle
  if kind == "dot" {
    let cx = (x0 + x1) / 2
    let spacing = 0.22  // cm — visually pleasing density
    let y = y0 + spacing / 2
    while y < y1 {
      circle((cx, y), radius: 0.045, fill: _pat_color, stroke: none)
      y = y + spacing
    }
  }
}
#let _pat_of(entry) = if entry.len() >= 5 { entry.at(4) } else { "solid" }

// Grouped vertical bars.  host: "m4"/"c8i".  series: array of
// (label, method, metric, color) — one clustered bar per series, per dataset.
// Optional 5th element specifies a B/W-distinguishable pattern overlay
// (see _draw_pattern above).
// fair.csv values are MB/s; we divide by 1000 here so the axis, ticks, and
// clipped-bar labels are all in GB/s.  cap is expressed in GB/s.
#let grouped(host, series, ylabel: "GB/s", cap: none, plot-w: 16, plot-h: 6.2) = {
  let vals = dsets.map(((d, short)) =>
    series.map(entry => num(cell(host, d, entry.at(1), entry.at(2))) / 1000.0))
  let rawmax = calc.max(..vals.flatten())
  let ymax = if cap == none { rawmax } else { cap }
  // "Nice" tick step: pick from {1, 2, 5} × 10^k so labels are clean.
  let raw_step = ymax / 5
  let mag = calc.pow(10, calc.floor(calc.log(raw_step)))
  let norm = raw_step / mag
  let nice = if norm <= 1.5 { 1 } else if norm <= 3 { 2 } else if norm <= 6 { 5 } else { 10 }
  let step = nice * mag
  let ys = plot-h / ymax
  let ng = dsets.len()
  let ns = series.len()
  let gp = plot-w / ng
  let inner = gp * 0.84
  let bw = inner / ns

  cetz.canvas(length: 1cm, {
    import cetz.draw: *

    // y gridlines + tick labels
    let t = 0.0
    while t <= ymax + step * 0.01 {
      let y = t * ys
      line((0, y), (plot-w, y), stroke: 0.3pt + luma(210))
      let lab = if step >= 1 { str(int(calc.round(t))) }
                else { str(calc.round(t, digits: 1)) }
      content((-0.18, y), text(11pt)[#lab], anchor: "east")
      t = t + step
    }
    // axes + y label
    line((0, 0), (plot-w, 0), stroke: 0.7pt + black)
    line((0, 0), (0, plot-h), stroke: 0.7pt + black)
    content((-1.95, plot-h / 2), text(12pt)[#ylabel], angle: 90deg, anchor: "center")

    // bars
    for (gi, (d, short)) in dsets.enumerate() {
      let gx = gi * gp + (gp - inner) / 2
      for (si, entry) in series.enumerate() {
        let col = entry.at(3)
        let pat = _pat_of(entry)
        let v = vals.at(gi).at(si)
        let clipped = v > ymax
        let h = (if clipped { ymax } else { v }) * ys
        let x0 = gx + si * bw
        let x1 = x0 + bw * 0.9
        rect((x0, 0), (x1, h), fill: _pat_fill(col, pat), stroke: 0.3pt + black)
        _pat_overlay(pat, x0, 0, x1, h)
        if clipped {
          // break mark (zigzag) near the top + true value rotated 90° above
          line((x0, h - 0.20), (x1, h - 0.07), stroke: 0.7pt + black)
          line((x0, h - 0.13), (x1, h), stroke: 0.7pt + black)
          content(((x0 + x1) / 2, h + 0.14),
                  text(9pt, weight: "bold")[#calc.round(v, digits: 1)],
                  anchor: "west", angle: 90deg)
        }
      }
      content((gi * gp + gp / 2, -0.18),
              text(11pt)[#short], anchor: "north-east", angle: 25deg)
    }

    // host name + vertical legend on the right (auto page width includes them)
    let host_disp = if host == "m4" { "M4" } else { host }
    content((plot-w + 0.35, plot-h + 0.40),
            text(15pt, weight: "extrabold")[#host_disp], anchor: "west")
    let ly = plot-h - 0.50
    for entry in series {
      let lab = entry.at(0)
      let col = entry.at(3)
      let pat = _pat_of(entry)
      let lx0 = plot-w + 0.35
      let lx1 = plot-w + 0.75
      rect((lx0, ly - 0.4), (lx1, ly), fill: _pat_fill(col, pat), stroke: 0.3pt + black)
      _pat_overlay(pat, lx0, ly - 0.4, lx1, ly)
      content((plot-w + 0.9, ly - 0.2), text(12pt)[#lab], anchor: "west")
      ly = ly - 0.68
    }
  })
}

#let host = sys.inputs.at("host", default: "m4")
// NB: `#set page` does NOT cross an #import boundary — each plot file sets its
// own auto page (otherwise it renders on the default A4 and clips).
