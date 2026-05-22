#!/bin/sh
# pivco-authored.  Splice the prebuilt-table export shim (pha_export.inc)
# into the freshly-extracted Oodle clone's newlz_arrays_tans.cpp at build
# time, so pivco-huffman ships no Oodle source -- only our own shim + this
# script.  The shim is copied next to the .cpp and pulled in by an
# #include inserted immediately before the file's final OODLE_NS_END
# (which places it inside namespace oo2, same translation unit as
# KrakenTansState + the static tansx2_* kernels).
#
# Idempotent: a no-op if the include is already present (so a re-configure
# without a fresh extract doesn't double-inject).
#
# Args:
#   $1 = oodle_data source dir (contains core/newlz_arrays_tans.cpp)
#   $2 = absolute path to pha_export.inc
set -e

SRC_DIR="$1"
SHIM_INC="$2"
CPP="$SRC_DIR/core/newlz_arrays_tans.cpp"

[ -f "$CPP" ]      || { echo "inject_export_shim: $CPP not found" >&2; exit 1; }
[ -f "$SHIM_INC" ] || { echo "inject_export_shim: $SHIM_INC not found" >&2; exit 1; }

cp "$SHIM_INC" "$SRC_DIR/core/pha_export.inc"

if grep -q 'pha_export.inc' "$CPP"; then
    exit 0   # already injected
fi

last=$(grep -n 'OODLE_NS_END' "$CPP" | tail -1 | cut -d: -f1)
[ -n "$last" ] || { echo "inject_export_shim: no OODLE_NS_END anchor in $CPP" >&2; exit 1; }

awk -v n="$last" 'NR==n{print "#include \"pha_export.inc\""} {print}' \
    "$CPP" > "$CPP.pha_tmp" && mv "$CPP.pha_tmp" "$CPP"
