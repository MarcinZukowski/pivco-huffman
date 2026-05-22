#!/usr/bin/env python3
"""bench.py -- paper-benchmark dispatcher.

Reads paper/benches.yaml and provides three subcommands:

    bench.py list
        Show every registered benchmark + its host x compiler matrix.

    bench.py show-how <name>
        Print the reproduction recipe (build + run command) for each
        cell of the matrix, with placeholders filled in.  Run these
        manually; bench.py never SSHes.

    bench.py collate <name>
        Glob the matching result files under the bench's outputs.dir,
        pick the lexically-newest .csv per (host, compiler), add
        host + compiler columns, and emit paper/data/<name>.csv.

Requires pyyaml.  TOML alternative was considered (Python 3.11+
stdlib) but YAML reads nicer for the nested hosts/benches structure
here.
"""

from __future__ import annotations

import argparse
import csv
import sys
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("missing pyyaml -- install with: pip install pyyaml\n")
    sys.exit(1)


ROOT       = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "paper" / "benches.yaml"
DATA_DIR    = ROOT / "paper" / "data"


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def find_bench(cfg: dict, name: str) -> dict:
    for b in cfg.get("benches", []):
        if b["name"] == name:
            return b
    sys.stderr.write(f"unknown benchmark: {name}\n")
    sys.stderr.write("known: " + ", ".join(b["name"] for b in cfg.get("benches", [])) + "\n")
    sys.exit(1)


def cmd_list(cfg: dict) -> int:
    for b in cfg.get("benches", []):
        hosts = b["matrix"]["hosts"]
        comps = b["matrix"]["compilers"]
        print(f"{b['name']}")
        print(f"    hosts:     {', '.join(hosts)}")
        print(f"    compilers: {', '.join(comps)}")
        print(f"    output:    {b['outputs']['dir']}")
        desc = (b.get("description") or "").strip().split("\n", 1)[0]
        if desc:
            print(f"    {desc}")
        print()
    return 0


# ---- show-how -------------------------------------------------------

def _ssh_alias_for(host: str) -> str | None:
    """test-* hosts use SSH aliases; m4 is local."""
    if host == "m4":
        return None
    return host   # test-c8i, test-c8a, ... use the bare name as ssh alias


def _chained(cwd_build: str, cc_path: str, build_cmd: str,
              cwd_run: str, run_cmd: str,
              csv_path: str, text_path: str) -> str:
    """A single shell pipeline that cd's, builds, then runs.  Designed
    so it can be wrapped in `ssh host '...'` without nested quoting."""
    run_cmd = run_cmd.replace("{csv}", csv_path).replace("{text}", text_path)
    return (
        f"cd {cwd_build} && "
        f"CC={cc_path} {build_cmd} && "
        + (f"cd {cwd_run} && " if cwd_run != cwd_build else "")
        + run_cmd
    )


def cmd_show_how(cfg: dict, name: str) -> int:
    bench = find_bench(cfg, name)
    hosts_cfg = cfg.get("hosts", {})

    print(f"# {bench['name']}")
    desc = (bench.get("description") or "").strip()
    if desc:
        print("# " + "\n# ".join(desc.splitlines()))
    print()
    print(f"# Output directory: {bench['outputs']['dir']}")
    print(f"# Naming: {{host}}-{{compiler}}-{{YYYYMMDD-HHMM}}-{{git-sha}}.{{csv,txt}}")
    print()

    out_dir = bench["outputs"]["dir"]

    for host in bench["matrix"]["hosts"]:
        host_cfg = hosts_cfg.get(host)
        if not host_cfg:
            print(f"# (no host config for {host} -- skipping)")
            continue
        for compiler in bench["matrix"]["compilers"]:
            cc_path = host_cfg["compilers"].get(compiler)
            if not cc_path:
                print(f"# {host}: no compiler '{compiler}' configured; skipping")
                continue

            stem = f"{host}-{compiler}-$(date +%Y%m%d-%H%M)-$(git rev-parse --short HEAD)"
            csv_path  = f"{ROOT}/{out_dir}{stem}.csv"
            text_path = f"{ROOT}/{out_dir}{stem}.txt"

            ssh = _ssh_alias_for(host)
            build = bench["build"]
            run   = bench["run"]
            print(f"## {host} x {compiler}  ({cc_path})")
            if ssh is None:
                print("# local (m4):")
                print(f"  mkdir -p {ROOT}/{out_dir}")
                print(f"  rm -rf {ROOT}/{build['cwd']}/build")
                chain = _chained(
                    f"{ROOT}/{build['cwd']}", cc_path, build["cmd"],
                    f"{ROOT}/{run['cwd']}",   run["cmd"],
                    csv_path, text_path)
                print(f"  {chain}")
            else:
                rcsv  = f"~/pivco-huffman/{out_dir}{stem}.csv"
                rtext = f"~/pivco-huffman/{out_dir}{stem}.txt"
                chain = _chained(
                    f"~/pivco-huffman/{build['cwd']}", cc_path, build["cmd"],
                    f"~/pivco-huffman/{run['cwd']}",   run["cmd"],
                    rcsv, rtext)
                print(f"# remote ({ssh}):  (rsync project tree first)")
                print(f"  rsync -avz --delete --exclude='build/' --exclude='.git/' \\")
                print(f"    {ROOT}/ {ssh}:pivco-huffman/")
                print(f"  ssh {ssh} \"mkdir -p ~/pivco-huffman/{out_dir} && {chain}\"")
                print(f"  rsync -avz {ssh}:~/pivco-huffman/{out_dir}{host}-{compiler}-* \\")
                print(f"    {ROOT}/{out_dir}")
            print()
    return 0


# ---- collate --------------------------------------------------------

# Filenames look like: m4-clang-latest-20260520-1438-fa7a901.csv
# Capture (host, compiler) from the prefix; date+sha trail.
_FN_RE = re.compile(
    r"^(?P<host>[^-]+(?:-[^-]+)*?)-(?P<compiler>(?:clang|gcc)-[a-z-]+)-"
    r"\d{8}-\d{4}-[0-9a-f]+\.csv$")


def _parse_filename(name: str, known_hosts: list[str]) -> tuple[str, str] | None:
    """Greedy host match against known hosts (since host names may
    contain `-`).  Returns (host, compiler) or None."""
    for host in sorted(known_hosts, key=len, reverse=True):
        prefix = host + "-"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        m = re.match(r"^((?:clang|gcc)-[a-z-]+)-\d{8}-\d{4}-[0-9a-f]+\.csv$", rest)
        if m:
            return host, m.group(1)
    return None


def _collate_stream(out_dir: Path, bench_name: str, stream_suffix: str,
                     known_hosts: list[str]) -> tuple[int, list[str]] | None:
    """Collate one CSV stream into paper/data/<bench_name><stream_suffix>.

    `stream_suffix`: extension(s) to match -- "" for the throughput
    stream (matches *.csv excluding *.profile.csv etc.), or
    e.g. ".profile" to match *.profile.csv.  Writes to
    paper/data/<bench>{stream_suffix}.csv.  Returns (n_cells, skipped)
    or None if no matches."""
    # Build a glob + an exclude predicate so the throughput stream
    # doesn't sweep up *.profile.csv (and vice versa).
    glob_pat   = f"*{stream_suffix}.csv"
    if stream_suffix == "":
        # Primary stream: match *.csv, exclude any *.<token>.csv with
        # an extra extension token (profile, etc.).
        def keep(p: Path) -> bool:
            # If the filename has more than one extension after the
            # date+sha stem, it's an auxiliary stream.
            stem = p.name[:-4]  # strip ".csv"
            # Stem must end with the SHA (hex chars after the last `-`).
            return not re.search(r"\.[a-z][a-z0-9_]*$", stem)
    else:
        def keep(p: Path) -> bool:
            return p.name.endswith(stream_suffix + ".csv")

    latest: dict[tuple[str, str], Path] = {}
    skipped: list[str] = []
    for p in sorted(out_dir.glob(glob_pat)):
        if not keep(p):
            continue
        # Strip stream suffix from name for filename parsing.
        base_name = p.name
        if stream_suffix:
            base_name = base_name.replace(stream_suffix + ".csv", ".csv")
        meta = _parse_filename(base_name, known_hosts)
        if not meta:
            skipped.append(p.name)
            continue
        latest[meta] = p
    if not latest:
        return None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{bench_name}{stream_suffix}.csv"

    rows: list[list[str]] = []
    schema_header: list[str] | None = None
    for (host, compiler), path in sorted(latest.items()):
        # Contract (see paper/benches.yaml): identifier columns in
        # the collated CSV must be SQL-safe -- no `-`.
        host_csv     = host.replace("-", "_")
        compiler_csv = compiler.replace("-", "_")
        with path.open() as f:
            r = csv.reader(f)
            header = next(r)
            if schema_header is None:
                schema_header = header
            elif schema_header != header:
                sys.stderr.write(
                    f"schema mismatch: {path.name} has {header}, "
                    f"expected {schema_header}\n")
                return None
            for row in r:
                rows.append([host_csv, compiler_csv] + row)

    with out_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["host", "compiler"] + (schema_header or []))
        w.writerows(rows)

    print(f"wrote {out_path}  ({len(latest)} cells, {len(rows)} rows)")
    for cell, path in sorted(latest.items()):
        print(f"    {cell[0]:<12} {cell[1]:<14} {path.name}")
    return (len(latest), skipped)


def cmd_collate(cfg: dict, name: str) -> int:
    bench = find_bench(cfg, name)
    known_hosts = list(cfg.get("hosts", {}).keys())
    out_dir = ROOT / bench["outputs"]["dir"]
    if not out_dir.exists():
        sys.stderr.write(f"no results directory: {out_dir}\n")
        return 1

    # Primary stream (.csv) is always collated.  Aux streams are
    # discovered by scanning for extra-extension patterns the
    # primary skipped.
    streams = [""]
    extra: set[str] = set()
    for p in out_dir.glob("*.csv"):
        stem = p.name[:-4]
        m = re.search(r"(\.[a-z][a-z0-9_]*)$", stem)
        if m:
            extra.add(m.group(1))
    for suffix in sorted(extra):
        streams.append(suffix)

    any_ok = False
    for suffix in streams:
        result = _collate_stream(out_dir, name, suffix, known_hosts)
        if result is not None:
            any_ok = True
    if not any_ok:
        sys.stderr.write(f"no matching CSVs in {out_dir}\n")
        return 1
    return 0


# ---- build-views ----------------------------------------------------

def cmd_build_views(cfg: dict, name: str) -> int:
    """Run every SQL view defined under benches[<name>].views and
    write the result to paper/data/<name>.<view>.csv.  Each view's
    SQL is embedded in the YAML as a multi-line string; it can
    reference the long-form CSV at `paper/data/<name>.csv` and any
    other already-collated data file in that directory."""
    try:
        import duckdb
    except ImportError:
        sys.stderr.write("missing duckdb -- install with: pip install duckdb\n")
        return 1

    bench = find_bench(cfg, name)
    views = bench.get("views", [])
    if not views:
        sys.stderr.write(f"{name}: no views: defined in benches.yaml\n")
        return 1

    long_csv = DATA_DIR / f"{name}.csv"
    if not long_csv.exists():
        sys.stderr.write(
            f"{name}: long-form CSV missing ({long_csv}).  "
            f"Run `bench.py collate {name}` first.\n")
        return 1

    conn = duckdb.connect(":memory:")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for v in views:
        vname = v["name"]
        sql   = v["sql"]
        out   = DATA_DIR / f"{name}.{vname}.csv"
        # Run the SQL inside CWD=ROOT so relative paths like
        # 'paper/data/<bench>.csv' resolve as the user wrote them.
        try:
            conn.execute(f"COPY ({sql}) TO '{out}' (HEADER, FORMAT CSV)")
        except Exception as e:
            sys.stderr.write(f"{name}.{vname}: SQL error:\n{e}\n")
            return 1
        # Sanity: peek at first 3 rows.
        with out.open() as f:
            lines = f.readlines()
        print(f"wrote {out}  ({len(lines) - 1} rows)")
        for line in lines[:3]:
            print(f"    {line.rstrip()}")
        if len(lines) > 3:
            print(f"    ...")
    return 0


# ---- main -----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG,
                    help="benches.yaml path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list registered benchmarks")

    p_how = sub.add_parser("show-how", help="print reproduction recipe")
    p_how.add_argument("name")

    p_col = sub.add_parser("collate", help="gather latest CSVs into paper/data/")
    p_col.add_argument("name")

    p_bv = sub.add_parser("build-views",
                            help="run SQL views (from benches.yaml) over the collated CSV")
    p_bv.add_argument("name")

    args = ap.parse_args()
    cfg = load_cfg(args.config)

    if args.cmd == "list":        return cmd_list(cfg)
    if args.cmd == "show-how":    return cmd_show_how(cfg, args.name)
    if args.cmd == "collate":     return cmd_collate(cfg, args.name)
    if args.cmd == "build-views": return cmd_build_views(cfg, args.name)
    return 1


if __name__ == "__main__":
    sys.exit(main())
