#!/usr/bin/env bash
# profile_m4.sh — capture an Instruments / xctrace Time Profiler trace
# of pivco_huffman_profile_english on a chosen distribution and emit a
# per-source-region breakdown.  macOS only.
#
# Usage:
#   ./extras/profile_m4.sh                  # prose_pride, 12 s
#   ./extras/profile_m4.sh english          # english,     12 s
#   ./extras/profile_m4.sh prose_pride 20   # prose_pride, 20 s
#
# Steps performed:
#   1. cmake configure (RelWithDebInfo)        — for DWARF debug info
#   2. cmake build pivco_huffman_profile_english
#   3. dsymutil                                — link debug info into a .dSYM
#   4. xctrace record --template "Time Profiler" --launch
#   5. xctrace export --xpath time-profile     — XML dump of samples
#   6. python3 profile_xctrace_parse.py        — aggregate by leaf frame
#
# Output:
#   results/profile-${HOST}-${DIST}-xctrace-${TS}.txt    (parsed summary)
#   /tmp/profile-${HOST}-${DIST}-${TS}.trace             (raw .trace bundle)
#   /tmp/profile-${HOST}-${DIST}-${TS}.xml               (raw XML export)

set -euo pipefail

DIST="${1:-prose_pride}"
DURATION="${2:-12}"

# Locate repo root (this script lives in extras/).
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

BIN="build/pivco_huffman_profile_english"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: profile_m4.sh requires macOS (uses xctrace)." >&2
    exit 1
fi
command -v xctrace >/dev/null || {
    echo "error: xctrace not found.  Install Xcode (or 'xcode-select --install')." >&2
    exit 1
}

echo "==> Configuring (RelWithDebInfo)"
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo > /tmp/cmake_profile.log 2>&1

echo "==> Building $BIN"
cmake --build build --target pivco_huffman_profile_english 2>&1 | tail -3

echo "==> Generating dSYM (DWARF debug info bundle)"
dsymutil "$BIN"

# Output naming.
TS=$(date -u +%Y%m%d-%H%M)
HOST=$(hostname -s | tr '[:upper:]' '[:lower:]')
TRACE="/tmp/profile-${HOST}-${DIST}-${TS}.trace"
XML="/tmp/profile-${HOST}-${DIST}-${TS}.xml"
SUMMARY="results/profile-${HOST}-${DIST}-xctrace-${TS}.txt"

mkdir -p results

echo "==> Recording ${DURATION} s of ${DIST} decode with xctrace Time Profiler"
rm -rf "$TRACE"
# xctrace record exits with the launched process's exit status (it sends
# SIGKILL when the time limit hits, so the exit is non-zero on success).
# Capture to a log and ignore the exit code — verify success by checking
# that the .trace bundle was produced.
xctrace record --template "Time Profiler" --time-limit "${DURATION}s" \
    --output "$TRACE" \
    --launch -- "$BIN" "$DIST" > /tmp/xctrace_record.log 2>&1 || true
tail -3 /tmp/xctrace_record.log
if [[ ! -d "$TRACE" ]]; then
    echo "error: xctrace did not produce $TRACE — see /tmp/xctrace_record.log" >&2
    exit 1
fi

echo "==> Exporting trace to XML"
xctrace export --input "$TRACE" \
    --xpath '/trace-toc/run/data/table[@schema="time-profile"]' \
    --output "$XML" >/dev/null 2>&1

echo "==> Parsing"
python3 "$SCRIPT_DIR/profile_xctrace_parse.py" "$XML" | tee "$SUMMARY"

echo
echo "Summary: $SUMMARY"
echo "Trace:   $TRACE"
echo "XML:     $XML"
