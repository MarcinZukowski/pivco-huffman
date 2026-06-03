// paper/style.typ — single source of truth for series colors + patterns.
//
// HSL-based so we can vary lightness per series for B/W distinguishability
// while keeping the SAME hue + saturation available for tinted table-cell
// fills.  Used by the plot files (paper/plots/*) and by body files that
// tint table columns by engine (e.g. ans.typ → tab-fair-m4).
//
// To add a series: append an entry to `_series` (hue, saturation, bar L)
// and assign a `patterns` kind; everything else flows from those.

#let _series = (
  //         (   hue,   saturation, bar lightness )
  ph:        ( 150deg,  90%,        40% ),
  ph_naive:  ( 110deg,  60%,        80% ),
  ph_flat:   ( 130deg,  70%,        60% ),
  ph_pb:     ( 120deg,  40%,        75% ),  // same hue as ph, brighter
  pha:       ( 180deg,  90%,        55% ),
  huf0:      (  20deg,  80%,        60% ),
  fse:       (  60deg,  80%,        70% ),
  oo_huff:   ( 270deg,  65%,        65% ),
  oo_tans:   ( 300deg,  85%,        75% ),
)

// Lightness used for tinted table-cell backgrounds (very light, so the
// cell text stays readable).  Shared across all series.
#let _tab_l = 94%

// Build a {name: color} dict by applying `to_l` to each series.  Saturation
// is preserved; lightness comes from the function so the same dict shape
// works for bars (per-series L) and table tints (fixed L).
// NOTE: we round-trip via .rgb() because Typst emits HSL colors with
// negative hues for values > 180° (e.g. 215° -> -145°) in its SVG output,
// and some PDF viewers render those as black.  RGB is unambiguous.
#let _build_colors(to_l) = {
  let out = (:)
  for (name, hsl) in _series.pairs() {
    let (h, s, bar_l) = hsl
    out.insert(name, color.hsl(h, s, to_l(bar_l)).rgb())
  }
  out
}

#let colors     = _build_colors(bar_l => bar_l)
#let colors-tab = _build_colors(_ => _tab_l)

// B/W-distinguishable pattern kind per series — used by paper/plots/*.typ
// to drive the `_pat_fill` / `_pat_overlay` helpers in plots/common.typ.
// Available kinds: "solid", "dot", "hlines", "d1", "d2", "checker", "checkerd".
#let patterns = (
  ph:      "solid",
  ph_pb:   "dot",
  pha:     "dot",
  huf0:    "hlines",
  oo_huff: "d1",
  oo_tans: "checker",
  fse:     "d2",
)
