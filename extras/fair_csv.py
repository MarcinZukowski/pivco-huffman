#!/usr/bin/env python3
"""Consolidate per-host `pivco_fair_bench` output into one long-format CSV.

Usage:
    fair_csv.py HOST=FILE [HOST=FILE ...] > paper/data/fair.csv

Each FILE is the raw stdout of `./build/pivco_fair_bench` (MAIN datasets,
all engines).  Output columns:

    host,dataset,method,enc_op,enc_pb,dec_op,dec_pb,ratio_op,ratio_pb,builds

Long format: one row per (host, dataset, method).  The paper selects the
cells it wants from this single table.  Metrics mirror the fair-bench
columns: enc/dec MB/s in opaque (op) and prebuilt (pb) modes, compression
ratio (op/pb), and table builds per 1 MB.  Missing cells (opaque-only
engines' prebuilt columns, failed engines) are emitted as `na`.

`huf0_4x2` (forced HUF_decompress4X2) is EXCLUDED from the reported set --
`huf0` is stock HUF_decompress (auto-dispatch).  The forced variant stays
in the raw results/ captures for ad-hoc eyeballing.
"""
import sys

# canonical reported set -- forced huf0_4x2 intentionally omitted.
# ph_naive / ph_flat are tree-mode ablation variants of `ph` (same codec,
# different chunk decomposition at build_table).
ENGINES = {
    "ph", "pha", "ph_naive", "ph_flat",
    "td_naive", "td_scl_opt", "td_nv_simd", "td_simdopt",
    "huf0", "fse_stk", "fse_x8y1", "oo_huff", "oo_tans",
}
EXCLUDE = {"huf0_4x2"}


def fix(tok):
    return "na" if tok.strip() in ("-", "") else tok.strip()


def parse(path):
    """yield (dataset, method, [enc_op,enc_pb,dec_op,dec_pb,r_op,r_pb,builds])"""
    cur = None
    for line in open(path):
        s = line.rstrip("\n")
        if s.startswith("== "):
            cur = s.split()[1]
            continue
        tok0 = s.split()[0] if s.split() else ""
        if tok0 in EXCLUDE:
            continue
        if tok0 not in ENGINES:
            continue
        if "(n/a)" in s or "n/a" in s:
            yield cur, tok0, ["na"] * 7
            continue
        parts = s.split("|")
        try:
            enc = parts[1].split()
            dec = parts[2].split()
            rat = parts[3].split()
            blds = parts[4].strip()
            row = [fix(enc[0]), fix(enc[1]), fix(dec[0]), fix(dec[1]),
                   fix(rat[0]), fix(rat[1]), fix(blds)]
            yield cur, tok0, row
        except (IndexError, ValueError):
            sys.stderr.write(f"warn: unparsed row in {path}: {s!r}\n")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("host,dataset,method,enc_op,enc_pb,dec_op,dec_pb,ratio_op,ratio_pb,builds")
    for arg in sys.argv[1:]:
        host, _, path = arg.partition("=")
        for dataset, method, row in parse(path):
            print(",".join([host, dataset, method] + row))


if __name__ == "__main__":
    main()
