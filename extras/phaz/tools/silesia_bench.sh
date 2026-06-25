#!/bin/bash
# Run `phaz stats` over a corpus at levels 9 and 19: phaz size vs stock zstd +
# fused decode timing, one file at a time.
set -e
PHAZ=~/src/pivco-huffman/extras/phaz
CORP=${1:-/tmp/silesia}
cd "$PHAZ"
for f in $(ls "$CORP" | grep -vE '\.zip$|\.sh$'); do
  F="$CORP/$f"; [ -f "$F" ] || continue
  echo "==================== $f ===================="
  ./phaz stats "$F" -l 9
  ./phaz stats "$F" -l 19
done
