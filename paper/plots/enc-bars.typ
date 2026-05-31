// Encoding throughput (data behind <tab-enc>): ph end-to-end (enc_op) +
// ph prebuilt-tree (enc_pb) + huf0 + oodle-huff, per dataset.  One SVG/host.
#import "common.typ": grouped, host, C
#set page(width: auto, height: auto, margin: 8pt)
#set text(font: "DejaVu Sans")

#grouped(host, (
  ("Pivco-Huffman",          "ph",      "enc_op", C.dgreen),
  ("Pivco-Huffman prebuilt", "ph",      "enc_pb", C.lgreen),
  ("Huff0",                  "huf0",    "enc_op", C.orange),
  ("Oodle Huffman",          "oo_huff", "enc_op", C.purple),
), cap: if host == "c8i" { 4 } else { none })
