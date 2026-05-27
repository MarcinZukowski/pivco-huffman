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

#let C = (
  dgreen: rgb("#1b7837"),  // ph (op / plain)
  lgreen: rgb("#a6dba0"),  // ph prebuilt
  teal:   rgb("#2aa3a3"),  // pha
  orange: rgb("#ef8a3b"),  // huf0
  purple: rgb("#9467bd"),  // oodle huffman
  rose:   rgb("#c45b96"),  // oodle tans
  blue:   rgb("#3b78c3"),  // fse
)

// Grouped vertical bars.  host: "m4"/"c8i".  series: array of
// (label, method, metric, color) — one clustered bar per series, per dataset.
// cap (none|number): clip bars above `cap` to the axis top, draw a break mark,
// and print the true value above the bar.
#let grouped(host, series, ylabel: "MB/s", cap: none, plot-w: 16, plot-h: 6.2) = {
  let vals = dsets.map(((d, short)) =>
    series.map(((lab, m, metric, col)) => num(cell(host, d, m, metric))))
  let rawmax = calc.max(..vals.flatten())
  let ymax = if cap == none { rawmax } else { cap }
  // nice tick step ~ ymax/5
  let mag = calc.pow(10, calc.floor(calc.log(ymax / 5)))
  let step = calc.ceil((ymax / 5) / mag) * mag
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
      content((-0.18, y), text(11pt)[#int(calc.round(t))], anchor: "east")
      t = t + step
    }
    // axes + y label
    line((0, 0), (plot-w, 0), stroke: 0.7pt + black)
    line((0, 0), (0, plot-h), stroke: 0.7pt + black)
    content((-1.95, plot-h / 2), text(12pt)[#ylabel], angle: 90deg, anchor: "center")

    // bars
    for (gi, (d, short)) in dsets.enumerate() {
      let gx = gi * gp + (gp - inner) / 2
      for (si, (lab, m, metric, col)) in series.enumerate() {
        let v = vals.at(gi).at(si)
        let clipped = v > ymax
        let h = (if clipped { ymax } else { v }) * ys
        let x0 = gx + si * bw
        let x1 = x0 + bw * 0.9
        rect((x0, 0), (x1, h), fill: col, stroke: 0.3pt + black)
        if clipped {
          // break mark (zigzag) near the top + true value horizontally above
          line((x0, h - 0.20), (x1, h - 0.07), stroke: 0.7pt + black)
          line((x0, h - 0.13), (x1, h), stroke: 0.7pt + black)
          content(((x0 + x1) / 2, h + 0.14), text(10pt, weight: "bold")[#int(v)],
                  anchor: "south")
        }
      }
      content((gi * gp + gp / 2, -0.34), text(12pt)[#short], anchor: "north")
    }

    // vertical legend on the right (auto page width includes it)
    let ly = plot-h
    for (lab, m, metric, col) in series {
      rect((plot-w + 0.35, ly - 0.4), (plot-w + 0.75, ly), fill: col, stroke: 0.3pt + black)
      content((plot-w + 0.9, ly - 0.2), text(12pt)[#lab], anchor: "west")
      ly = ly - 0.68
    }
  })
}

#let host = sys.inputs.at("host", default: "m4")
// NB: `#set page` does NOT cross an #import boundary — each plot file sets its
// own auto page (otherwise it renders on the default A4 and clips).
