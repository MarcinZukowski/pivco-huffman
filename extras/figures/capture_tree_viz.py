#!/usr/bin/env python3
"""capture_tree_viz.py — render figures/tree_viz.html configurations
to standalone .svg files using a headless Chromium driven by Playwright.

Setup (one-time):
    pip install playwright
    playwright install chromium

Usage:
    python3 extras/figures/capture_tree_viz.py
    python3 extras/figures/capture_tree_viz.py --config path/to/figs.json
    python3 extras/figures/capture_tree_viz.py --out-dir paper/figs

Reads extras/figures/tree_figures.json by default and writes the
resulting SVGs into figures/ (or --out-dir).  Each config entry is:

    {
      "name":   "tree-foo",                  # filename stem; ".svg" appended
      "params": { "text": "...", "...": ... } # tree_viz URL params
    }

How it works:
    Each config entry is converted to a tree_viz.html URL with the
    given params plus download=<name>.svg.  The in-page JS catches
    the download param after the first render and triggers a Blob
    download — Playwright captures that download and saves it to
    --out-dir.  All CSS inlining + content-bbox sizing happens
    inside the page, so this script stays dumb.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlencode

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.stderr.write(
        "playwright not installed.  Run:\n"
        "    pip install playwright\n"
        "    playwright install chromium\n"
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
TREE_VIZ = ROOT / "figures" / "tree_viz.html"
DEFAULT_CFG = ROOT / "extras" / "figures" / "tree_figures.json"
DEFAULT_OUT = ROOT / "figures"


def build_url(params: dict) -> str:
    """tree_viz.html URL with the given query params.

    URL-encoding is via urllib.parse.urlencode — handles spaces, %,
    #, & etc. correctly so caller doesn't have to pre-escape."""
    return f"file://{TREE_VIZ}?{urlencode(params)}"


def capture_one(page, name: str, params: dict, out_dir: Path) -> Path:
    """Capture one figure.  Returns the path to the saved .svg."""
    # Force download= so the in-page auto-trigger fires.
    p = dict(params)
    p["download"] = f"{name}.svg"
    url = build_url(p)
    # expect_download must be registered BEFORE the navigation that
    # triggers the download.  Default timeout 30 s is fine; the page
    # waits ~150 ms after first render before triggering, so the
    # whole thing completes in well under a second on a warm browser.
    with page.expect_download(timeout=20_000) as dl_info:
        page.goto(url, wait_until="load")
    target = out_dir / f"{name}.svg"
    dl_info.value.save_as(target)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG,
                    help=f"figure config JSON (default: {DEFAULT_CFG.relative_to(ROOT)})")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"directory to write .svg into (default: {DEFAULT_OUT.relative_to(ROOT)})")
    ap.add_argument("--filter", type=str, default=None,
                    help="only capture figures whose name contains this substring")
    args = ap.parse_args()

    if not TREE_VIZ.exists():
        sys.stderr.write(f"tree_viz.html not found at {TREE_VIZ}\n")
        return 1
    if not args.config.exists():
        sys.stderr.write(f"config not found: {args.config}\n")
        return 1

    cfg = json.loads(args.config.read_text())
    figs = cfg.get("figures", [])
    if args.filter:
        figs = [f for f in figs if args.filter in f.get("name", "")]
        if not figs:
            sys.stderr.write(f"no figures match filter {args.filter!r}\n")
            return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"capturing {len(figs)} figure(s) to {args.out_dir.relative_to(ROOT)}/")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()
        for fig in figs:
            name = fig["name"]
            params = fig.get("params", {})
            print(f"  {name}.svg")
            target = capture_one(page, name, params, args.out_dir)
            size_kb = target.stat().st_size / 1024
            print(f"    → {target.relative_to(ROOT)} ({size_kb:.1f} KB)")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
