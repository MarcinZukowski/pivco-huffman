// Decode throughput (the bandwidth half of <tab-fair-m4>, dropping the ratio
// columns): ph / pha / huf0 / fse_x8y1 / oodle-tans, dec_op per dataset.
// One SVG per host (m4, c8i).
#import "common.typ": grouped, host, colors, patterns
#set page(width: auto, height: auto, margin: 8pt)
#set text(font: "DejaVu Sans")

#grouped(host, (
  ("Pivco-Huffman",     "ph",       "dec_op", colors.ph,      patterns.ph),
  ("Pivco-Huffman+ANS", "pha",      "dec_op", colors.pha,     patterns.pha),
  ("Huff0",             "huf0",     "dec_op", colors.huf0,    patterns.huf0),
  ("FSE x8y1",          "fse_x8y1", "dec_op", colors.fse,     patterns.fse),
  ("Oodle TANS",        "oo_tans",  "dec_op", colors.oo_tans, patterns.oo_tans),
), cap: 11)
