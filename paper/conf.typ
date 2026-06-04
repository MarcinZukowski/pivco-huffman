// Bind format flags at module scope.  Don't wrap these in [..] content
// blocks: `#let` inside a content block is scoped to that block, and
// the outer references later in the file would always see the initial
// `false`.  That was the bug that broke style.css injection.
#let _html = sys.inputs.at("target", default: none) == "html"
#let _pdf  = not _html
#let _fmt  = if _html { "html" } else { "pdf" }

// Inline the stylesheet into the HTML.  Typst places <style> in <body>
// rather than <head>, but browsers happily apply it globally — no
// post-processing needed.
#if _html {
  // simd.dev tooltips
//  html.elem("script", attrs: (src: "https://simd.dev/dist/simd-tooltips.js"))

  html.elem("style", read("style.css"))
  // web-code.js: runtime extras (author-notes toggle today; more
  // later).  Inlined the same way as style.css — keeps the HTML
  // build self-contained, PDF build never sees it.
  html.elem("script", read("web-code.js"))

// Hyvor
  html.elem("script", attrs: (
    async: "",
    src: "https://talk.hyvor.com/embed/embed.js",
    type: "module"
  ))
}

#if _pdf {
  set page(paper: "a4", margin: 2cm, columns: 1)
  set page(numbering: "1")
}

#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set figure(placement: auto)

#let anote(body) = {
  if _html {
    html.elem("div", attrs: (class: "anote"), body)
  }
}

#let todo(body) = {
  if _html {
    html.elem("div", attrs: (class: "todo"))[TODO: #body]
  } else {
    text(fill: red)[*TODO: #body*]
  }
}

#let setup(body) = {
  show title: set text(size: 17pt)
  show title: set align(center)
  show heading.where(level: 1): smallcaps
  show heading.where(level: 1): set text(
    size: 16pt
  )
  show heading.where(level: 2): set text(
    size: 12pt
  )
  show raw.where(block: true): it => block(
    fill: rgb("#e0f0ff"),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
    if _pdf { align(left, it) } else { it }
  )
  body
}

#let PH="PivCo-Huffman"
#let PHA="PivCo-Huffman+ANS"
#let OOH="Oodle Huffman"
#let h0="Huff0"   // display name for the huf0 baseline (stock HUF_decompress)
#let URLBASE="https://marcinzukowski.github.io/pivco-huffman/paper-1.0/"

// HTML element with a provided class name
#let he(cname, style:none, body) = {
  if _html {
    if (style != none) {
      html.elem("style", style);
    }
    html.elem("div", attrs: (class: cname), body)
  } else {
    body
  }
}

#let mf(figname, ..opts) = {
  let base = "./"
  if _pdf {
    base = URLBASE
  }
  he("myfig",
  link(
      base + "figures/fig-web.html?name=" + figname,
      image("figures/" + figname + ".svg", ..opts))
  )
}

#let sym(t) = { [*#raw("\"" + t + "\"")*] }

// (c) Keep only specified columns (often cleaner than repeated drops)
#let pick-cols(table, names) = {
  let h = table.first()
  let idxs = names.map(n => {
    let i = h.position(c => c == n)
    if i == none {
      panic("pick-cols: no column '" + n + "' (have: " + h.join(", ") + ")")
    }
    i
  })
  table.map(row => idxs.map(i => row.at(i)))
}

// (d) Long-format fair-bench CSV (data/fair.csv).  Columns are:
//   host,dataset,method,enc_op,enc_pb,dec_op,dec_pb,ratio_op,ratio_pb,builds
// "_op" = opaque (realistic per-call: table rebuilt per window), "_pb" =
// prebuilt (one table reused, isolates kernel throughput).  Tables select the
// cells they want from this single source of truth (one CSV, all hosts).

// fair-filter: keep rows matching ALL (column: value) constraints in `where`
// (a dictionary), then project each match to `metrics` (an array of column
// names).  Returns an array of rows, each an array of metric strings -- so it
// generalises from one cell to a whole sub-table.  Example:
//   fair-filter(fair, (host:"m4", dataset:"proba80", method:"huf0"),
//               ("enc_op","dec_op")).first()        =>  ("1332", "2902")
#let fair-filter(table, where, metrics) = {
  let h = table.first()
  let ci(name) = {
    let i = h.position(c => c == name)
    if i == none { panic("fair-filter: no column '" + name + "'") }
    i
  }
  table.slice(1)
    .filter(row => where.pairs().all(((k, v)) => row.at(ci(k)) == v))
    .map(row => metrics.map(m => row.at(ci(m))))
}

// fair-cell: convenience for a single metric of a unique (host,dataset,method);
// returns the cell string, or "na" if that row is absent.
#let fair-cell(table, host, dataset, method, metric) = {
  let r = fair-filter(table, (host: host, dataset: dataset, method: method), (metric,))
  if r.len() == 0 { "na" } else { r.first().first() }
}
