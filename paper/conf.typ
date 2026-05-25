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
  html.elem("script", attrs: (src: "https://simd.dev/dist/simd-tooltips.js"))
  html.elem("style", read("style.css"))
  // web-code.js: runtime extras (author-notes toggle today; more
  // later).  Inlined the same way as style.css — keeps the HTML
  // build self-contained, PDF build never sees it.
  html.elem("script", read("web-code.js"))
}

#if _pdf {
  set page(paper: "a4", margin: 2cm, columns: 1)
  set page(numbering: "1")
}

#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)

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
    fill: rgb("#ece6d3"),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
    it
  )
  body
}

#let PH="PivCo-Huffman"
#let PHA="PivCo-Huffman+ANS"
#let URLBASE="githubpages/blalba/paper"

// HTML element with a provided class name
#let he(cname, body) = {
  if _html {
    html.elem("div", attrs: (class: cname), body)
  } else {
    body
  }
}

#let mf(figname) = {
  let base = "../"
  if _pdf {
    base = URLBASE
  }
  link(
      base + "figures/fig-web.html?" + figname,
      image("figures/" + figname + ".svg"))
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
