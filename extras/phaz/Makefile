# phaz — pivco-Huffman entropy layer transplanted onto zstd.
#
# The fork lives in a single file (phaz.h) applied to the enclosing repo's
# pinned zstd checkout (../../ext/zstd) via phaz.patch. `make` copies zstd to
# build/, patches the copy, builds the patched libzstd, then builds the single
# `phaz` CLI (linking that libzstd + pivco-huffman's lib). See tools/build.sh.
#
#   make          build everything (copy -> patch -> libzstd -> phaz)
#   make clean    remove the phaz binary and the build/ tree

all:
	@bash tools/build.sh

clean:
	rm -f phaz
	rm -rf build/

.PHONY: all clean
