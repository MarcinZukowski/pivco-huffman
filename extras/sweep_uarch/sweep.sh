#!/usr/bin/env bash
# Sweep: rsync source to every test-* host, build, run bench_fair, collect output.
# Per-host raw output -> results/sweep_uarch/<date>/<alias>.txt
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
DATE=$(date +%Y-%m-%d-%H%M)
OUT="$ROOT/results/sweep_uarch/$DATE"
mkdir -p "$OUT"
echo "[sweep] writing results to $OUT" >&2

# NB: current pivco_fair_bench takes no positional repeats arg (it self-paces
# via adaptive runs and rejects unknown args since f3e3ef3).  No-args run =
# default dist set, all engines (ph + huf0 both present), at the build's
# default block size.

run_one() {
    local alias=$1
    local log="$OUT/$alias.txt"
    local err="$OUT/$alias.err"
    : > "$log"; : > "$err"

    {
        echo "=== host: $alias  date: $(date -u +%FT%TZ) ==="

        # ssh probe (waits up to ~2 min for sshd)
        for i in $(seq 1 24); do
            if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
                   -o BatchMode=yes "$alias" true 2>/dev/null; then break; fi
            sleep 5
        done

        # wait for cloud-init bootstrap to finish (new hosts only).
        ssh "$alias" 'for i in $(seq 1 60); do [ -f ~/.bootstrap_done ] && break; \
                       command -v gcc >/dev/null && command -v cmake >/dev/null && break; \
                       sleep 5; done; \
                      gcc --version | head -1; cmake --version | head -1' \
            2>>"$err"

        # rsync source (mirrors the documented CLAUDE.md command).
        rsync -az --delete \
            --exclude='build/' --exclude='build-asan/' --exclude='build-release/' \
            --exclude='.git/' --exclude='.claude/' --exclude='.vscode/' \
            --exclude='*.dSYM' --exclude='.venv/' --exclude='ext/oodle' \
            --exclude='results/' --exclude='paper/out/' \
            "$ROOT/" "$alias:pivco-huffman/" 2>>"$err"

        # build + run
        ssh "$alias" "cd pivco-huffman && rm -rf build && \
                      cmake -B build -DCMAKE_BUILD_TYPE=Release >/tmp/cmake.log 2>&1 && \
                      cmake --build build --target pivco_fair_bench -j2 >/tmp/build.log 2>&1 && \
                      taskset -c 0 ./build/pivco_fair_bench" \
            2>>"$err"

        echo "=== done: $alias  date: $(date -u +%FT%TZ) ==="
    } >>"$log" 2>>"$err"
}

PIDS=()
NAMES=()
while IFS=$'\t ' read -r alias rest; do
    [[ -z "${alias:-}" || "${alias:0:1}" == "#" ]] && continue
    echo "[sweep] starting $alias..." >&2
    run_one "$alias" &
    PIDS+=($!)
    NAMES+=("$alias")
done < <(awk 'NF && $1 !~ /^#/' hosts.tsv)

# wait for all, report
fail=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[sweep] OK   ${NAMES[$i]}" >&2
    else
        echo "[sweep] FAIL ${NAMES[$i]}" >&2
        fail=$((fail+1))
    fi
done

echo "[sweep] all done; results in $OUT" >&2
[[ $fail -gt 0 ]] && echo "[sweep] $fail host(s) failed; check *.err in $OUT" >&2
exit 0
