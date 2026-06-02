// Encoding throughput (data behind <tab-enc>): ph end-to-end (enc_op) +
// ph prebuilt-tree (enc_pb) + huf0 + oodle-huff, per dataset.  One SVG/host.
#import "common.typ": grouped, host, colors, patterns
#set page(width: auto, height: auto, margin: 8pt)
#set text(font: "DejaVu Sans")

#grouped(host, (
  ("Pivco-Huffman",          "ph",      "enc_op", colors.ph,      patterns.ph),
  ("Pivco-Huffman prebuilt", "ph",      "enc_pb", colors.ph_pb,   patterns.ph_pb),
  ("Huff0",                  "huf0",    "enc_op", colors.huf0,    patterns.huf0),
  ("Oodle Huffman",          "oo_huff", "enc_op", colors.oo_huff, patterns.oo_huff),
), cap: if host == "c8i" { 4 } else { none })
