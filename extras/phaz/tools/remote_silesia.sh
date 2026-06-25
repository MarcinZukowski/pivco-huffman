#!/bin/bash
# Build pivco-huffman + the patched-zstd phaz CLI on this host, fetch Silesia,
# and run `phaz stats` (size vs stock zstd + fused decode timing) at levels 9/19.
# Assumes ~/src/pivco-huffman was synced here (phaz lives under
# pivco-huffman/extras/phaz and reuses the enclosing repo's ext/zstd checkout).
set -e
ARCH=$(uname -m); J=$(nproc)
if [ "$ARCH" = "x86_64" ]; then MARCH="-march=native"; else MARCH="-mcpu=native"; fi
H=$HOME/src; PH=$H/pivco-huffman
echo "### $(hostname) arch=$ARCH march=$MARCH cores=$J"

echo "[1/3] pivco-huffman lib"
cd $PH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="$MARCH" -DCMAKE_CXX_FLAGS="$MARCH" >/tmp/cm.log 2>&1 \
  || { echo PIVCO_CFG_FAIL; tail -25 /tmp/cm.log; exit 1; }
cmake --build build --target pivco_huffman -j$J >/tmp/cmb.log 2>&1 \
  || { echo PIVCO_BUILD_FAIL; tail -30 /tmp/cmb.log; exit 1; }

echo "[2/3] phaz (copy zstd -> patch -> libzstd -> phaz)"
cd $PH/extras/phaz
MARCH="$MARCH" CC=clang J=$J bash tools/build.sh >/tmp/phaz.log 2>&1 \
  || { echo PHAZ_BUILD_FAIL; tail -40 /tmp/phaz.log; exit 1; }

echo "[3/3] silesia corpus + bench"
if [ ! -f /tmp/silesia/dickens ]; then
  mkdir -p /tmp/silesia && cd /tmp/silesia
  curl -fsSL -o sil.zip http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip && unzip -oq sil.zip
fi
cd $PH/extras/phaz
for f in $(ls /tmp/silesia | grep -vE '\.zip$'); do
  F=/tmp/silesia/$f; [ -f "$F" ] || continue
  echo "==================== $f ===================="
  ./phaz stats "$F" -l 9
  ./phaz stats "$F" -l 19
done
echo "### done $(hostname)"
