// Tree-mode decode bandwidth: ph_naive / ph_flat / ph (optimized) / huf0 /
// oo_huff, dec_op per dataset.  One SVG per host (m4, c8i).  Same shape as
// dec-bw.typ; series differs.
#import "common.typ": grouped, host, C
#set page(width: auto, height: auto, margin: 8pt)
#set text(font: "DejaVu Sans")

#grouped(host, (
  ("PH naive",       "ph_naive", "dec_op", C.lgreen),
  ("PH flat",        "ph_flat",  "dec_op", C.teal),
  ("PH flat opt.",   "ph",       "dec_op", C.dgreen),
  ("Huff0",          "huf0",     "dec_op", C.orange),
  ("Oodle Huffman",  "oo_huff",  "dec_op", C.purple),
), cap: 11)
