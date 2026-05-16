#!/bin/sh
# Helper for FetchContent_DeclareWithPatch: normalize CRLF line endings
# in the extracted source tree (Linux unzip preserves them) and then
# apply the data.patch with git apply.  Patched 2026-05-15 for
# pivco-huffman cross-platform builds.
#
# Args:
#   $1 = absolute path to the .patch file
set -e
PATCH_FILE="$1"
find . -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.inl' \
                   -o -name '*.inc' -o -name '*.S' -o -name '*.nas' \) \
    -exec perl -i -pe 's/\r$//' {} +
exec git apply --verbose --no-index --ignore-space-change --whitespace=fix "$PATCH_FILE"
