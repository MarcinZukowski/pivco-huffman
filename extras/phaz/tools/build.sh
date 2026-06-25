#!/bin/bash
# phaz build: copy the enclosing pivco-huffman's pinned zstd checkout out, patch
# the copy (keeping the source pristine), build the patched libzstd, then build
# the single `phaz` CLI.
#   env: PH=<pivco-huffman dir>  (default = the enclosing repo, ../..; its
#        libpivco_huffman.a is the PH/PHA stream codec phaz links)
set -e
PHAZ="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$PHAZ/../../ext/zstd"          # reuse the enclosing pivco-huffman's zstd checkout
WORK="$PHAZ/build/zstd"
PH="${PH:-$PHAZ/../..}"             # pivco-huffman is two levels up
CC="${CC:-cc}"
J="${J:-4}"
MARCH="${MARCH:-}"   # e.g. -march=native / -mcpu=native for benchmarking

# 1. ensure pivco-huffman's zstd checkout is present (pinned 5233c58e)
[ -f "$SRC/lib/zstd.h" ] || { echo "missing $SRC — clone pivco-huffman's ext/zstd first"; exit 1; }

# 2. fresh disposable copy + apply the minimal patch
echo ">> syncing zstd -> build/zstd and applying phaz.patch"
rm -rf "$WORK"; mkdir -p "$PHAZ/build"
rsync -a --exclude='.git' "$SRC/" "$WORK/"
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
PLIB="$PH/build/libpivco_huffman.a"
[ -f "$PLIB" ] || { echo "missing $PLIB — build pivco-huffman first:"; \
  echo "  (cd $PH && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target pivco_huffman -j)"; exit 1; }

# Both libzstd and libpivco_huffman vendor FiniteStateEntropy (FSE_*/HUF_*).
# Merge pivco's lib into one relocatable object with those symbols localized so
# they don't clash with zstd's at the final link (pivco's internal FSE calls
# bind to its own now-local copy).  Syntax differs ld64 (macOS) vs GNU.
PLOCAL="$PHAZ/build/pivco_local.o"
echo ">> merging pivco-huffman lib (localizing FSE_*/HUF_* to avoid zstd clash)"
if [ "$(uname)" = "Darwin" ]; then
  # drive the partial link through clang (injects -platform_version); ld64's
  # -unexported_symbol turns the matched globals into locals in the .o
  $CC -nostdlib -Wl,-r -Wl,-all_load \
      -Wl,-unexported_symbol,'_FSE_*' -Wl,-unexported_symbol,'_HUF_*' \
      -Wl,-unexported_symbol,'_HIST_*' -Wl,-unexported_symbol,'_XXH*' \
      -Wl,-unexported_symbol,'_ERR_*' -Wl,-unexported_symbol,'_BIT_*' \
      "$PLIB" -o "$PLOCAL"
else
  # GNU binutils: partial-link the whole archive, then localize via objcopy
  ld -r -o "$PHAZ/build/pivco_all.o" --whole-archive "$PLIB" --no-whole-archive
  objcopy -w --localize-symbol 'FSE_*' --localize-symbol 'HUF_*' \
            --localize-symbol 'HIST_*' --localize-symbol 'XXH*' \
            --localize-symbol 'ERR_*' --localize-symbol 'BIT_*' \
            "$PHAZ/build/pivco_all.o" "$PLOCAL"
fi

echo ">> building phaz"
# -Werror for real issues; -Wno-deprecated-declarations (ZSTD_generateSequences)
# and -Wno-misleading-indentation (GCC style-nanny on the terse one-line style).
$CC -O3 -Wall -Werror -Wno-deprecated-declarations -Wno-misleading-indentation $MARCH $ZINC -I"$PH/include" \
    "$PHAZ/tools/phaz.c" "$LIB" "$PLOCAL" -o "$PHAZ/phaz"

echo ">> done. phaz in $PHAZ/"
