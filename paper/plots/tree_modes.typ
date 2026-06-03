// Tree-mode decode bandwidth: ph_naive / ph_flat / ph (optimized) / huf0 /
// oo_huff, dec_op per dataset.  One SVG per host (m4, c8i).  Same shape as
// dec-bw.typ; series differs.
#import "common.typ": grouped, host, colors, patterns
#set page(width: auto, height: auto, margin: 8pt)
#set text(font: "DejaVu Sans")

// `ph_naive` and `ph_flat` aren't in the canonical _series dict (those are
// intermediate variants of `ph`).  Map them to the closest reasonable
// stand-ins: lighter PH tints + their own patterns for B/W distinguishability.
#grouped(host, (
  ("PH naive",       "ph_naive", "dec_op", colors.ph_naive, "dot"),
  ("PH flat",        "ph_flat",  "dec_op", colors.ph_flat,  "hlines"),
  ("PH flat opt.",   "ph",       "dec_op", colors.ph,      patterns.ph),
  ("Huff0",          "huf0",     "dec_op", colors.huf0,    "d1"),
  ("Oodle Huffman",  "oo_huff",  "dec_op", colors.oo_huff, "checker"),
), cap: 11)
