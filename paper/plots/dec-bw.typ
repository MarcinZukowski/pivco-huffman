// Decode throughput (the bandwidth half of <tab-fair-m4>, dropping the ratio
// columns): ph / pha / huf0 / fse_x8y1 / oodle-tans, dec_op per dataset.
// One SVG per host (m4, c8i).
#import "common.typ": grouped, host, C
#set page(width: auto, height: auto, margin: 8pt)
#set text(font: "DejaVu Sans")

#grouped(host, (
  ("Pivco-Huffman",     "ph",       "dec_op", C.dgreen),
  ("Pivco-Huffman+ANS", "pha",      "dec_op", C.teal),
  ("Huff0",             "huf0",     "dec_op", C.orange),
  ("FSE x8y1",          "fse_x8y1", "dec_op", C.blue),
  ("Oodle TANS",        "oo_tans",  "dec_op", C.rose),
), cap: 11)
