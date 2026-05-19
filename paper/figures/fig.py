#!/usr/bin/env python3
"""fig.py — paper-figure dispatcher.

Reads paper/figures/figures.json and renders named figures to SVG via
each figure's `tool.cli`.  Each tool is invoked once per batch of
figures it owns, with a temp `--config` JSON of the same shape its
own batch mode expects:

    {"figures": [{"name": "...", "params": {...}}, ...]}

Defaults are merged tool-wide → per-figure (figure overrides win).

USAGE
    fig.py list                          # list known figures + their tool
    fig.py svg <name> [<name>...]        # render the listed figures
    fig.py svg --all                     # render every figure
    fig.py web <name>                    # print the viewer URL (do not open)

Optional flags:
    --config PATH        non-default figures.json
    --out-dir PATH       where SVGs land (default: <figures.json dir>)
    --filter SUBSTR      with --all, only names containing SUBSTR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT / "paper" / "figures" / "figures.json"


def load_cfg(path: Path):
    cfg = json.loads(path.read_text())
    return cfg


def merged_params(cfg, fig):
    """Merge tool.defaults <- figure.params (figure wins)."""
    tool = cfg["tools"].get(fig["tool"], {})
    out = dict(tool.get("defaults") or {})
    out.update(fig.get("params") or {})
    return out


def figures_named(cfg, names, all_, filt):
    """Return the [(name, fig)] pairs to act on.  Errors on unknown names."""
    by_name = {f["name"]: f for f in cfg["figures"] if "name" in f}
    if all_:
        names = list(by_name.keys())
        if filt:
            names = [n for n in names if filt in n]
    selected = []
    for n in names:
        if n not in by_name:
            sys.stderr.write(f"unknown figure: {n}\n")
            sys.stderr.write(f"known: {', '.join(sorted(by_name))}\n")
            sys.exit(1)
        selected.append((n, by_name[n]))
    return selected


def cmd_list(cfg):
    by_tool = {}
    for fig in cfg["figures"]:
        if "name" not in fig:
            continue
        by_tool.setdefault(fig["tool"], []).append(fig["name"])
    for tool_name in sorted(by_tool):
        print(f"{tool_name}:")
        for n in by_tool[tool_name]:
            print(f"  {n}")
    return 0


def cmd_svg(cfg, cfg_path, names, all_, filt, out_dir):
    sel = figures_named(cfg, names, all_, filt)
    if not sel:
        sys.stderr.write("no figures selected (use --all or pass names)\n")
        return 1
    # Group by tool so each tool's CLI runs once.
    by_tool = {}
    for name, fig in sel:
        by_tool.setdefault(fig["tool"], []).append((name, fig))
    for tool_name, fig_list in by_tool.items():
        tool = cfg["tools"].get(tool_name)
        if not tool or "cli" not in tool:
            sys.stderr.write(f"tool {tool_name!r} has no cli — skipping\n")
            continue
        cli = ROOT / tool["cli"]
        if not cli.exists():
            sys.stderr.write(f"cli not found for {tool_name}: {cli}\n")
            return 1
        # Build a per-tool temp config.
        tool_cfg = {"figures": []}
        for name, fig in fig_list:
            tool_cfg["figures"].append({
                "name": name,
                "params": merged_params(cfg, fig),
            })
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(tool_cfg, tmp)
            tmp_path = tmp.name
        try:
            cmd = [sys.executable, str(cli),
                   "--config", tmp_path,
                   "--out-dir", str(out_dir)]
            print(f"→ {tool_name}: {len(fig_list)} figure(s)")
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                sys.stderr.write(f"{cli} exited {rc}\n")
                return rc
        finally:
            os.unlink(tmp_path)
    return 0


def cmd_web(cfg, cfg_path, name):
    by_name = {f["name"]: f for f in cfg["figures"] if "name" in f}
    if name not in by_name:
        sys.stderr.write(f"unknown figure: {name}\n")
        return 1
    fig = by_name[name]
    tool = cfg["tools"].get(fig["tool"], {})
    if "web" not in tool:
        sys.stderr.write(f"tool {fig['tool']!r} has no web viewer\n")
        return 1
    params = merged_params(cfg, fig)
    viewer = ROOT / tool["web"]
    print(f"file://{viewer}?{urlencode(params)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG,
                    help=f"figures.json path (default: {DEFAULT_CFG.relative_to(ROOT)})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list known figures grouped by tool")

    p_svg = sub.add_parser("svg", help="render figures to SVG")
    p_svg.add_argument("names", nargs="*",
                       help="figure names (omit with --all)")
    p_svg.add_argument("--all", action="store_true",
                       help="render every figure in the config")
    p_svg.add_argument("--filter", default=None,
                       help="with --all, only names containing this substring")
    p_svg.add_argument("--out-dir", type=Path, default=None,
                       help="output directory (default: <figures.json dir>)")

    p_web = sub.add_parser("web", help="print the viewer URL for a figure")
    p_web.add_argument("name", help="figure name")

    args = ap.parse_args()
    cfg = load_cfg(args.config)

    if args.cmd == "list":
        return cmd_list(cfg)
    if args.cmd == "svg":
        out_dir = args.out_dir or args.config.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return cmd_svg(cfg, args.config, args.names, args.all,
                        args.filter, out_dir)
    if args.cmd == "web":
        return cmd_web(cfg, args.config, args.name)
    return 1


if __name__ == "__main__":
    sys.exit(main())
