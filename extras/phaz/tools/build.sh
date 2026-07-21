#!/bin/bash
# phaz build: copy the enclosing pivco-huffman's pinned zstd checkout out, patch
# the copy (keeping the source pristine), build the patched libzstd, then build
# the single `phaz` CLI.
#   env: PH=<pivco-huffman dir>  (default = the enclosing repo, ../..; its
#        libpivco_huffman.a is the PH/PHA stream codec phaz links)
set -e
PHAZ="$(cd "$(dirname "$0")/.." && pwd)"
# zstd source to copy + patch (any checkout pinned at 5233c58e).  Default: the
# enclosing pivco-huffman's ext/zstd; override with ZSTD_SRC (e.g. an embedder
# like TurboBench points this at its own zstd/ submodule at the same SHA).
SRC="${ZSTD_SRC:-$PHAZ/../../ext/zstd}"
WORK="$PHAZ/build/zstd"
PH="${PH:-$PHAZ/../..}"             # pivco-huffman is two levels up
CC="${CC:-cc}"
J="${J:-4}"
MARCH="${MARCH:-}"   # e.g. -march=native / -mcpu=native for benchmarking

# 1. ensure pivco-huffman's zstd checkout is present (pinned 5233c58e)
[ -f "$SRC/lib/zstd.h" ] || { echo "missing $SRC — clone pivco-huffman's ext/zstd first"; exit 1; }

# 2. fresh disposable copy + apply the minimal patch.  Only lib/ is
# needed (phaz.patch touches lib/compress + lib/decompress; the build is
# `make -C lib libzstd.a`) -- copying the whole checkout dragged in
# tests/cli-tests/bin symlinks, which rsync cannot create on Windows
# hosts (TurboBench CI, MSYS: "symlink ... failed", error 23).
echo ">> syncing zstd/lib -> build/zstd/lib and applying phaz.patch"
rm -rf "$WORK"; mkdir -p "$WORK"
rsync -a --exclude='.git' "$SRC/lib/" "$WORK/lib/"
# apply with `patch` if present, else `git apply` (always available on dev boxes)
if command -v patch >/dev/null 2>&1; then
  ( cd "$WORK" && patch -p1 < "$PHAZ/phaz.patch" )
else
  ( cd "$WORK" && git apply -p1 "$PHAZ/phaz.patch" )
fi

# 3. patched libzstd (MOREFLAGS injects the include path to phaz.h + arch flags).
#    -Wno-declaration-after-statement: the pinned ext/zstd carries its own
#    zstd_prof_* instrumentation that trips zstd's strict C89 flags -- it's
#    zstd's code, not ours, so we silence it rather than -Werror it.
echo ">> building patched libzstd.a"
make -C "$WORK/lib" libzstd.a MOREFLAGS="-I$PHAZ $MARCH -Wno-declaration-after-statement" -j"$J" >/dev/null
LIB="$WORK/lib/libzstd.a"; ZINC="-I$WORK/lib"

# 4. the single phaz CLI — links the patched libzstd (capture hook +
#    ZSTD_phazDecode) and pivco-huffman's lib (PH/PHA stream codec).
# libpivco_huffman vendors FiniteStateEntropy (FSE_*/HUF_*), which would clash
# with zstd's at the final link.  pivco-huffman ships a pre-localized drop-in
# object (libpivco_huffman_local.o) with those symbols already localized and its
# public pivco_*/pivcohuf_* API kept global -- link that and the clash is gone.
PLOCAL="$PH/build/libpivco_huffman_local.o"
[ -f "$PLOCAL" ] || { echo "missing $PLOCAL — build pivco-huffman first:"; \
  echo "  (cd $PH && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target pivco_huffman_local -j)"; exit 1; }

echo ">> building phaz"
# -Werror for real issues; -Wno-deprecated-declarations (ZSTD_generateSequences)
# and -Wno-misleading-indentation (GCC style-nanny on the terse one-line style).
# -I"$PHAZ" so the CLI + codec find phaz_codec.h.
# -lm: libpivco's joint_lengths.c uses log2(); glibc keeps it in libm
# (no-op on macOS).
$CC -O3 -Wall -Werror -Wno-deprecated-declarations -Wno-misleading-indentation $MARCH $ZINC -I"$PH/include" -I"$PHAZ" \
    "$PHAZ/tools/phaz.c" "$PHAZ/phaz_codec.c" "$LIB" "$PLOCAL" -o "$PHAZ/phaz" -lm

# Relocatable, drop-in object for embedding (e.g. the TurboBench plugin):
# phaz_codec + the *private patched* libzstd + pivco, merged into one .o that
# exports ONLY phaz_compress / phaz_decompress / phaz_compress_bound.  Every
# other symbol (all of zstd, FSE/HUF, the g_phaz_* capture globals) is
# localized, so it coexists with a host that links its own (vanilla) zstd.
echo ">> building phaz_local.o (embeddable blob)"
PCODEC="$PHAZ/build/phaz_codec.o"
PBLOB="$PHAZ/build/phaz_local.o"
$CC -O3 $MARCH $ZINC -I"$PH/include" -I"$PHAZ" -c "$PHAZ/phaz_codec.c" -o "$PCODEC"
if [ "$(uname)" = "Darwin" ]; then
  $CC -nostdlib -Wl,-r -Wl,-all_load \
      -Wl,-exported_symbol,'_phaz_compress' \
      -Wl,-exported_symbol,'_phaz_decompress' \
      -Wl,-exported_symbol,'_phaz_compress_bound' \
      "$PCODEC" "$LIB" "$PLOCAL" -o "$PBLOB"
else
  ld -r -o "$PHAZ/build/phaz_all.o" "$PCODEC" --whole-archive "$LIB" --no-whole-archive "$PLOCAL"
  objcopy --keep-global-symbol phaz_compress --keep-global-symbol phaz_decompress \
          --keep-global-symbol phaz_compress_bound \
          "$PHAZ/build/phaz_all.o" "$PBLOB"
fi

echo ">> done. phaz in $PHAZ/  (embeddable blob: $PBLOB)"
