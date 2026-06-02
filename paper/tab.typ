/* paper/tab.typ — unified styled-table DSL for PDF + HTML.
 *
 * One `tab(...)` call produces a styled table in both backends, driven from
 * one configuration so we don't maintain Typst align functions + CSS class
 * selectors in parallel.
 *
 * USAGE:
 *
 *   #import "tab.typ": tab
 *   #tab(
 *     name:    "tab-foo",                  // <tab-foo> label + ".tab-foo" HTML class
 *     columns: 7,                           // or (auto, "30%", ...)
 *     align:   right,                       // single or per-column tuple
 *     header:  ([*A*], [*B*], ..., [*G*]),  // header row content, verbatim
 *     body:    data,                        // flat list of cells, verbatim
 *     header_rows: 1,                       // for multi-row headers (e.g. 2)
 *     rules: (
 *       ((x: 0),            (align: left, weight: "bold")),
 *       ((y: "h"),          (style: "italic")),                  // header row(s)
 *       ((y: "odd"),        (fill: rgb("#f7f7f7"))),             // zebra rows
 *       ((y: 1, x: (2, 3)), (fill: rgb("#fcc"))),
 *       ((y: 8, x: 2),      (fill: rgb("#ff8"), weight: "bold")),
 *       ((x: 5),            (border-right: 1pt, inset: 8pt)),
 *     ),
 *     caption: [...],                       // optional — see below
 *   )
 *
 * The Typst path uses `show table.cell.where(...): set ...` rules around the
 * table.  Header cells (incl. ones with rowspan/colspan) and body cells are
 * passed VERBATIM into Typst's `table(...)`, never wrapped — so the user's
 * own table.cell(...) constructions are preserved exactly as written.
 *
 * COORDINATE CONVENTION (DSL-side):
 *   - x: column, 0-based.
 *   - y: body row, 0-based.  `"h"` = header.  `"odd"` / `"even"` = zebra rows
 *     (CSS convention: 1st body row is "odd", 2nd is "even").
 *
 *   The PDF backend internally bumps y by `header_rows` for body cells; the
 *   HTML backend translates to 1-based :nth-child selectors with the right
 *   offset for thead/tbody.  None of that surfaces in the DSL.
 *
 * SELECTOR FORMS:
 *   - x: N        single column
 *   - x: (N, M)   multiple columns
 *   - x: A..B     range (Typst-native — coerced to array)
 *   - same for y, plus the string sentinels "h" / "odd" / "even"
 *
 * RECOGNISED ACTION KEYS:
 *   align  → text-align              (CSS) / set align(...)          (Typst)
 *   weight → font-weight             (CSS) / set text(weight: ...)   (Typst)
 *   style  → font-style              (CSS) / set text(style: ...)    (Typst)
 *   size   → font-size               (CSS) / set text(size: ...)     (Typst)
 *   color  → color                   (CSS) / set text(fill: ...)     (Typst)
 *   fill   → background-color        (CSS) / set table.cell(fill: ...)
 *   border → border (all sides)      (CSS) / set table.cell(stroke: ...)
 *   border-top/bottom/left/right
 *          → border-side             (CSS) / per-side stroke
 *
 * PASSTHROUGH: any key NOT in the list above is forwarded verbatim as a
 * `set table.cell(...)` kwarg on the Typst side, and ignored by the CSS side.
 * Common candidates: `inset`, `colspan` (probably you mean to set on the cell
 * itself, but it works as a default too).
 *
 * CAPTION & FIGURE WRAPPING:
 *   - caption provided → wraps in `#figure(table(...), caption: ...) <name>`
 *     (numbered, captioned, references work via `@name`).
 *   - caption absent   → emits the bare table with the label attached
 *     (still referenceable; doesn't bump the figure counter).
 */

#import "conf.typ": _html, _pdf, he

/* ============================================================
 * shared helpers
 * ============================================================ */

#let _axis_list(spec) = {
  if spec == none { none }
  else if type(spec) == str { (spec,) }
  else if type(spec) == int { (spec,) }
  else if type(spec) == array { spec }
  else { (spec,) }
}

/* ============================================================
 * Typst (PDF) path: build a list of (where_kwargs, action) tuples,
 * then wrap the table with `show table.cell.where(...): set ...` rules.
 * ============================================================ */

/* Expand DSL rules into concrete Typst-where tuples.
 *   header_rows : how many rows the header spans (Typst y = 0 .. header_rows-1).
 *   body_n      : number of body rows (used for "odd" / "even"). */
#let _expand_typst(rules, header_rows, body_n) = {
  let out = ()
  for entry in rules {
    let (sel, action) = entry
    let xs = _axis_list(sel.at("x", default: none))
    let ys = _axis_list(sel.at("y", default: none))

    let typst_ys = if ys == none { (none,) } else {
      let result = ()
      for y in ys {
        if y == "h" {
          for ty in range(0, header_rows) { result.push(ty) }
        } else if y == "odd" {
          for bi in range(0, body_n) { if calc.even(bi) { result.push(bi + header_rows) } }
        } else if y == "even" {
          for bi in range(0, body_n) { if calc.odd(bi)  { result.push(bi + header_rows) } }
        } else if type(y) == int {
          result.push(y + header_rows)
        }
      }
      result
    }
    let typst_xs = if xs == none { (none,) } else { xs }

    for tx in typst_xs {
      for ty in typst_ys {
        let where_args = (:)
        if tx != none { where_args.insert("x", tx) }
        if ty != none { where_args.insert("y", ty) }
        out.push((where_args, action))
      }
    }
  }
  out
}

/* IMPORTANT: in Typst, `show` (and `set`) rules INSIDE an `if` block don't
 * propagate to the surrounding scope — they're confined to the if branch.
 * So we can't write `if "weight" in action { show ...: set ... }` directly,
 * because the show rule would be immediately scoped away.
 *
 * The trick: each show rule is the FIRST statement in a tiny helper
 * function's body, so it's at the top level of that function's block.
 * Calling `inner = _show_X(w, v, inner)` wraps `inner` with that show rule
 * in scope.  The `if "X" in action` then decides whether to make the call,
 * not whether to emit the show rule. */

#let _show_text_weight(w, v, body) = { show table.cell.where(..w): set text(weight: v); body }
#let _show_text_style(w, v, body) = { show table.cell.where(..w): set text(style: v); body }
#let _show_text_size(w, v, body) = { show table.cell.where(..w): set text(size: v); body }
#let _show_text_fill(w, v, body) = { show table.cell.where(..w): set text(fill: v); body }
#let _show_cell_align(w, v, body) = { show table.cell.where(..w): set table.cell(align: v); body }
#let _show_cell_fill(w, v, body) = { show table.cell.where(..w): set table.cell(fill: v); body }
#let _show_cell_stroke(w, v, body) = { show table.cell.where(..w): set table.cell(stroke: v); body }
#let _show_cell_kwargs(w, kw, body) = { show table.cell.where(..w): set table.cell(..kw); body }

#let _wrap_typst_entry(entry, body) = {
  let (where_args, action) = entry

  /* Per-side stroke dict from border-* keys */
  let per_side_stroke = (:)
  if "border-top"    in action { per_side_stroke.insert("top",    action.at("border-top")) }
  if "border-bottom" in action { per_side_stroke.insert("bottom", action.at("border-bottom")) }
  if "border-left"   in action { per_side_stroke.insert("left",   action.at("border-left")) }
  if "border-right"  in action { per_side_stroke.insert("right",  action.at("border-right")) }

  /* Passthrough kwargs (anything we don't recognise) */
  let handled = ("weight", "style", "size", "color", "align", "fill",
                 "border", "border-top", "border-bottom", "border-left", "border-right")
  let passthrough = (:)
  for (k, v) in action.pairs() {
    if not handled.contains(k) { passthrough.insert(k, v) }
  }

  /* Wrap `inner` with one show rule per present key.  The if-conditional
   * picks whether to wrap; the show rule is inside the helper, not the if. */
  let inner = body
  if "weight" in action { inner = _show_text_weight(where_args, action.weight, inner) }
  if "style"  in action { inner = _show_text_style(where_args, action.style, inner) }
  if "size"   in action { inner = _show_text_size(where_args, action.size, inner) }
  if "color"  in action { inner = _show_text_fill(where_args, action.color, inner) }
  if "align"  in action { inner = _show_cell_align(where_args, action.align, inner) }
  if "fill"   in action { inner = _show_cell_fill(where_args, action.fill, inner) }
  if "border" in action { inner = _show_cell_stroke(where_args, action.border, inner) }
  if per_side_stroke.len() > 0 { inner = _show_cell_stroke(where_args, per_side_stroke, inner) }
  if passthrough.len() > 0 { inner = _show_cell_kwargs(where_args, passthrough, inner) }
  inner
}

#let _wrap_typst(expanded, body) = {
  if expanded.len() == 0 { return body }
  let entry = expanded.first()
  let rest = expanded.slice(1)
  _wrap_typst_entry(entry, _wrap_typst(rest, body))
}

/* ============================================================
 * HTML path: emit CSS rules; the table itself is unmodified.
 * ============================================================ */

#let _css_val(v) = {
  if type(v) == int or type(v) == float { str(v) + "pt" }
  else if type(v) == length { repr(v) }    // e.g. "8pt"
  else if type(v) == color  { v.to-hex() } // "#rrggbb"
  else if type(v) == str    { v }
  else { repr(v) }
}

/* Typst alignment value → CSS text-align horizontal direction. */
#let _css_align(a) = {
  let s = repr(a)
  if s.contains("left")        { "left" }
  else if s.contains("right")  { "right" }
  else if s.contains("center") { "center" }
  else { s }
}

#let _css_props(action) = {
  let p = ()
  if "align"  in action { p.push("text-align: "       + _css_align(action.align) + ";") }
  if "weight" in action { p.push("font-weight: "      + str(action.weight) + ";") }
  if "style"  in action { p.push("font-style: "       + str(action.style)  + ";") }
  if "size"   in action { p.push("font-size: "        + _css_val(action.size)  + ";") }
  if "color"  in action { p.push("color: "            + _css_val(action.color) + ";") }
  if "fill"   in action { p.push("background-color: " + _css_val(action.fill)  + ";") }
  if "border" in action { p.push("border: " + _css_val(action.border) + " solid currentColor;") }
  if "border-top"    in action { p.push("border-top: "    + _css_val(action.at("border-top"))    + " solid currentColor;") }
  if "border-bottom" in action { p.push("border-bottom: " + _css_val(action.at("border-bottom")) + " solid currentColor;") }
  if "border-left"   in action { p.push("border-left: "   + _css_val(action.at("border-left"))   + " solid currentColor;") }
  if "border-right"  in action { p.push("border-right: "  + _css_val(action.at("border-right"))  + " solid currentColor;") }
  p.join(" ")
}

/* Typst HTML output wraps the header row in <thead> and body rows in
 * <tbody>, with header cells as <th> and body cells as <td>.  Selectors:
 *
 *   y = "h"        → thead tr
 *   y = N (int)    → tbody tr:nth-child(N+1)         (1-based body row)
 *   y = "odd"      → tbody tr:nth-child(odd)         (1st, 3rd, 5th, ...)
 *   y = "even"     → tbody tr:nth-child(even)        (2nd, 4th, 6th, ...)
 *
 * Cell selector uses `> *:nth-child(N+1)` so it matches both <td> and <th>
 * uniformly. */
#let _css_y(name, y) = {
  if y == "h" { "." + name + " thead tr" }
  else if y == "odd"  { "." + name + " tbody tr:nth-child(odd)" }
  else if y == "even" { "." + name + " tbody tr:nth-child(even)" }
  else if type(y) == int { "." + name + " tbody tr:nth-child(" + str(y + 1) + ")" }
  else { "." + name + " tr" }
}

#let _css_selectors(name, sel) = {
  let xs = _axis_list(sel.at("x", default: none))
  let ys = _axis_list(sel.at("y", default: none))

  let x_strs = if xs == none { ("",) } else {
    xs.map(x => " > *:nth-child(" + str(x + 1) + ")")
  }
  /* When the rule has no y constraint, default the y-part to `tbody tr`
   * (not just `tr`).  Reason: `:nth-child(N)` indexes DOM children of the
   * row, not the visual column.  In header rows with colspan/rowspan, the
   * DOM index drifts away from the column index, so a "column N" rule that
   * also fires on header rows lands on the wrong cell.  Scoping column-
   * only rules to `tbody` avoids that for the common case (body usually
   * has no spans).  Header styling should use `y: "h"` explicitly, and if
   * the header has colspan/rowspan too, style those cells directly in the
   * `header:` content — pure-CSS positional selectors can't handle it. */
  let y_strs = if ys == none {
    ("." + name + " tbody tr",)
  } else {
    ys.map(y => _css_y(name, y))
  }

  let combos = ()
  for y in y_strs {
    for x in x_strs {
      combos.push(y + x)
    }
  }
  combos.join(",\n")
}

/* ============================================================
 * align / fill overrides → table-level functions (PDF only)
 *
 * Typst's `set table.cell(align: ...)` inside a show rule doesn't actually
 * modify the matched cell — it only sets the default for future cell
 * constructions.  Same for `fill` and `stroke`.  So show+set is a no-op for
 * these properties in PDF.
 *
 * Workaround: translate rules with `align:` / `fill:` actions into the
 * table's own `align: (col, row) => ...` and `fill: (col, row) => ...`
 * parameters, which ARE applied at construction.  We pre-compute a map of
 * (col, row) → value and wrap the user's align/fill with a function that
 * consults the map, falling back to whatever the user passed.
 *
 * Column-only rules (no y constraint) scope to body rows only, matching
 * the CSS scoping — header rows with rowspan/colspan can't be addressed
 * positionally anyway.
 * ============================================================ */

#let _key(col, row) = str(col) + "," + str(row)

#let _build_cell_overrides(rules, header_rows, body_n, cols_n) = {
  let aligns = (:)
  let fills  = (:)

  for entry in rules {
    let (sel, action) = entry
    let want_align = "align" in action
    let want_fill  = "fill"  in action
    if not (want_align or want_fill) { continue }

    let xs = _axis_list(sel.at("x", default: none))
    let ys = _axis_list(sel.at("y", default: none))

    /* Body-only when no y constraint (matches CSS tbody scoping). */
    let rows = if ys == none {
      range(header_rows, header_rows + body_n)
    } else {
      let result = ()
      for y in ys {
        if y == "h" {
          for ty in range(0, header_rows) { result.push(ty) }
        } else if y == "odd"  {
          for bi in range(0, body_n) { if calc.even(bi) { result.push(bi + header_rows) } }
        } else if y == "even" {
          for bi in range(0, body_n) { if calc.odd(bi)  { result.push(bi + header_rows) } }
        } else if type(y) == int {
          result.push(y + header_rows)
        }
      }
      result
    }

    let cols = if xs == none { range(0, cols_n) } else { xs }

    for col in cols {
      for row in rows {
        let k = _key(col, row)
        if want_align { aligns.insert(k, action.align) }
        if want_fill  { fills.insert(k, action.fill)  }
      }
    }
  }

  (aligns: aligns, fills: fills)
}

#let _wrap_cell_fn(user_value, overrides) = {
  /* Return a (col, row) => value function.  Order of resolution:
   *   1. rule override for (col, row), if any
   *   2. user_value as a (col, row) function, if it's one
   *   3. user_value as a per-column tuple, indexed by col
   *   4. user_value as a scalar (or `auto` / `none`) */
  (col, row) => {
    let k = _key(col, row)
    if k in overrides { overrides.at(k) }
    else if user_value == auto { auto }
    else if type(user_value) == function { user_value(col, row) }
    else if type(user_value) == array { user_value.at(col, default: auto) }
    else { user_value }
  }
}

/* ============================================================
 * main entry point
 * ============================================================ */

/* `inset` / `stroke` default to a sentinel `auto`; if the user doesn't pass
 * them, we don't forward them to `table(...)` (Typst doesn't accept `auto`
 * for these — it uses its own internal defaults).  Forwarded only when
 * explicitly set. */
#let tab(
  name: none,
  columns: 1,
  align: auto,
  fill: none,
  inset: auto,
  stroke: auto,
  header: (),
  body: (),
  header_rows: 1,
  rules: (),
  caption: none,
  placement: auto,        // forwarded to `figure(..., placement: ...)`
) = {
  assert(name != none, message: "tab(): `name` is required")
  let cols_n = if type(columns) == int { columns } else { columns.len() }
  let body_n = int(body.len() / cols_n)

  /* Pre-compute align/fill overrides from rules — applied at construction. */
  let overrides = _build_cell_overrides(rules, header_rows, body_n, cols_n)
  let final_align = if overrides.aligns.len() == 0 { align } else {
    _wrap_cell_fn(align, overrides.aligns)
  }
  let final_fill = if overrides.fills.len() == 0 { fill } else {
    _wrap_cell_fn(fill, overrides.fills)
  }

  if _html {
    let css = rules.map(entry => {
      let (sel, action) = entry
      _css_selectors(name, sel) + " { " + _css_props(action) + " }"
    }).join("\n")

    let extra = (:)
    if inset  != auto { extra.insert("inset",  inset)  }
    if stroke != auto { extra.insert("stroke", stroke) }

    let tbl = table(
      columns: columns,
      align: align,         // CSS handles overrides; no need to wrap
      fill: fill,
      ..extra,
      table.header(..header),
      ..body,
    )
    let inner = if caption != none { figure(tbl, caption: caption, placement: placement) } else { tbl }
    he(name, style: css)[#inner #label(name)]
  } else {
    let expanded = _expand_typst(rules, header_rows, body_n)

    let extra = (:)
    if inset  != auto { extra.insert("inset",  inset)  }
    if stroke != auto { extra.insert("stroke", stroke) }

    let tbl = table(
      columns: columns,
      align: final_align,
      fill: final_fill,
      ..extra,
      table.header(..header),
      ..body,
    )

    let styled = _wrap_typst(expanded, tbl)

    if caption != none {
      [#figure(styled, caption: caption, placement: placement) #label(name)]
    } else {
      [#styled #label(name)]
    }
  }
}
