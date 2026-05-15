"""Shared HTML/Plotly + table renderer for pivco-huffman figures.

Used by `build.py` (entry point that emits all figures) and by per-
figure modules that want to share the CSS scaffold, tier-coloring
helpers, sweep-file parser, and color palette.

Output goes to `figures/<name>.html` at the project root — self-
contained pages with Plotly via CDN, data embedded as JSON.  No
build-time JS bundling; refreshing a figure is just re-running the
generator.

Design idioms borrowed from canasort's `extras/plot_html.py`
(tier-colored analysis pane, custom hover tooltip, monospace
tabular-nums for number columns) but the code here is fresh —
canasort's data shape is `(strategy × n × platform → throughput)`,
ph's is `(host × distribution × metric)`.
"""

import datetime
import json
import math
import os
import re
from pathlib import Path


# --- Layout ---------------------------------------------------------------

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR   = REPO_ROOT / "results"
FIGURES_DIR   = REPO_ROOT / "figures"


# --- Host metadata --------------------------------------------------------

# Display order across figures.  Canonical labels used in headers.
HOSTS = ["m4", "c8i", "c8g", "c6a"]

# Each host row: short label (used in column headers), full description
# (instance + CPU + clock, used in tooltip / subtitle).
HOST_LABEL = {
    "m4":  "Apple M4 (NEON)",
    "c8i": "Xeon AVX-512",
    "c8g": "Graviton 4 (NEON)",
    "c6a": "Zen 3 (SSE+AVX2)",
}
HOST_DETAIL = {
    "m4":  "Apple M4 Max, ~4.4 GHz P-core",
    "c8i": "AWS c8i.large — Xeon 6975P-C (Granite Rapids), ~3.9 GHz",
    "c8g": "AWS c8g.large — Graviton 4 Neoverse-V2, ~2.8 GHz",
    "c6a": "AWS c6a.large — EPYC 7R13 (Zen 3 / Milan), ~3.6 GHz",
}

# Color per host — paired so the same host gets the same color in
# every figure that ranks by host.  Pulled from a colorblind-safe
# palette (Wong 2011 / Okabe-Ito).
HOST_COLOR = {
    "m4":  "#0072B2",  # blue
    "c8i": "#D55E00",  # vermillion
    "c8g": "#009E73",  # bluish green
    "c6a": "#CC79A7",  # reddish purple
}


# --- Distribution metadata -----------------------------------------------

# Canonical order within figures.  Matches the bench output order.
DIST_ORDER = [
    # Skew/probaN family.
    "proba80", "proba50", "proba14", "proba02",
    # Bell.
    "bell_s10", "bell_s30", "bell_s80",
    # Generic.
    "uniform", "english", "zipfian", "geometric",
    # Sparse / two_sym.
    "sparse_4", "sparse_16",
    "two_sym_eq", "two_sym_90/10",
    # Flat tree.
    "flat_M3", "flat_M5", "flat_M6", "flat_M7",
    # Real-world.
    "html_wiki", "prose_pride", "image_jpeg", "json_api",
    "source_c", "log_apache", "dna_fasta", "csv_numeric",
    "gzip_random", "chinese_text", "calgary_pic",
]

# MAIN distribution set used by `pivco_huffman_bench` (no `--all`).
# Matches bench/bench_distributions.c::DISTS_MAIN.  10 entries.
DIST_MAIN = [
    "proba80", "english", "flat_M5",
    "html_wiki", "prose_pride", "image_jpeg",
    "json_api", "gzip_random", "chinese_text", "calgary_pic",
]
DIST_MAIN_SET = set(DIST_MAIN)


# Family grouping for filter chips.
DIST_FAMILY = {}
for d in ("proba80", "proba50", "proba14", "proba02"):     DIST_FAMILY[d] = "probaN"
for d in ("bell_s10", "bell_s30", "bell_s80"):              DIST_FAMILY[d] = "bell"
for d in ("uniform", "english", "zipfian", "geometric"):    DIST_FAMILY[d] = "generic"
for d in ("sparse_4", "sparse_16",
          "two_sym_eq", "two_sym_90/10"):                   DIST_FAMILY[d] = "sparse/two_sym"
for d in ("flat_M3", "flat_M5", "flat_M6", "flat_M7"):      DIST_FAMILY[d] = "flat"
for d in ("html_wiki", "prose_pride", "image_jpeg",
          "json_api", "source_c", "log_apache", "dna_fasta",
          "csv_numeric", "gzip_random", "chinese_text",
          "calgary_pic"):                                    DIST_FAMILY[d] = "real-world"


# --- Sweep file parser ---------------------------------------------------
#
# Reads `results/sweep_<host>-<tag>.txt` and the matching
# `enc_sweep_<host>-<tag>.txt`.  Returns one dict per (host, dist)
# with the metric columns we care about.

def parse_decode_sweep(path):
    """Parse a `pivco_huffman_bench` output file.

    Returns dict { dist: { 'pivco_s': M/s, 'pivco_n': ..., 'pivco_bu': ...,
                            'trad_4s': ..., 'huf0_x1': ..., 'huf0_x2': ...,
                            'rans_x2': ..., 'ratio': float } }

    Also captures the per-block "Compression sizes" table appended
    after the decode block: { dist: { 'pivco_raw': bytes, 'huf0_x2_bytes': ...,
                                        ... } }, returned as the second tuple.
    """
    decode = {}
    sizes = {}
    in_decode = False
    in_sizes = False
    with open(path) as f:
        for line in f:
            if line.startswith("DECODE M/s"):
                in_decode, in_sizes = True, False
                continue
            if line.startswith("=== Compression"):
                in_decode, in_sizes = False, True
                continue
            if line.startswith("===") or line.startswith("---"):
                continue
            if not line.strip():
                continue
            fields = line.split()
            # In both tables, fields[1:] is mostly pipe-separated.  The
            # bench prints `name | ... | ... | ... |` so '|' tokens fall
            # at predictable positions.  We strip them and read by
            # contiguous numeric position.
            if in_decode and len(fields) >= 17 and fields[1] == "|":
                name = fields[0]
                pivco_s  = int(fields[2])
                pivco_n  = int(fields[3])
                pivco_bu = int(fields[4])
                # fields[5] = trad_1s (often 0 = not run), fields[6] = '|'
                trad_4s  = int(fields[8])
                huf0_x1  = int(fields[11])
                huf0_x2  = int(fields[12])
                rans_x2  = int(fields[14])
                ratio    = float(fields[16].rstrip("x"))
                decode[name] = dict(pivco_s=pivco_s, pivco_n=pivco_n,
                                    pivco_bu=pivco_bu, trad_4s=trad_4s,
                                    huf0_x1=huf0_x1, huf0_x2=huf0_x2,
                                    rans_x2=rans_x2, ratio=ratio)
            elif in_sizes and len(fields) >= 14:
                # DIST | Dmax Lvs Ful Flt Hal B2L | vIN vLv | pivco_raw +hdr_est | trad_4s huf0_1s huf0_x2 rans_x2
                name = fields[0]
                if name in DIST_FAMILY:
                    # find pivco_raw: 12th numeric column after name.
                    # fields[1]='|', 2..7=Dmax..B2L, 8='|', 9..10=vIN/vLv,
                    # 11='|', 12=pivco_raw, 13=+hdr_est, 14='|', 15=trad_4s,
                    # 16=huf0_1s, 17=huf0_x2, 18=rans_x2 (last col may vary).
                    try:
                        pivco_raw_bytes = int(fields[12])
                        sizes[name] = dict(pivco_raw_bytes=pivco_raw_bytes)
                    except (ValueError, IndexError):
                        pass
    return decode, sizes


def parse_encode_sweep(path):
    """Parse a `pivco_huffman_bench_encode` output file.

    Returns dict { dist: { 'pivco_s': M/s, 'pivco': ..., 'trad_4s': ...,
                            'huf0_x1': ..., 'huf0_x2': ..., 'ratio': float } }
    """
    encode = {}
    in_encode = False
    with open(path) as f:
        for line in f:
            if line.startswith("ENCODE M/s"):
                in_encode = True
                continue
            if not line.strip() or line.startswith("---"):
                continue
            if in_encode and line.startswith("==="):
                break
            if in_encode:
                fields = line.split()
                if len(fields) >= 11 and fields[1] == "|":
                    name = fields[0]
                    pivco_s = int(fields[2])
                    pivco   = int(fields[3])
                    trad_4s = int(fields[5])
                    huf0_x1 = int(fields[7])
                    huf0_x2 = int(fields[8])
                    ratio   = float(fields[10].rstrip("x"))
                    encode[name] = dict(pivco_s=pivco_s, pivco=pivco,
                                        trad_4s=trad_4s,
                                        huf0_x1=huf0_x1, huf0_x2=huf0_x2,
                                        ratio=ratio)
    return encode


def load_sweep_set(tag):
    """Load decode + encode + sizes for all four hosts at `tag`.

    `tag` examples: '20260515-unify-all-nofse', '20260515-unify-all'.
    Returns nested dict { host: { 'decode': {...}, 'encode': {...},
                                    'sizes': {...} } }.

    Hosts with missing files are silently dropped.
    """
    out = {}
    for host in HOSTS:
        dec_path = RESULTS_DIR / f"sweep_{host}-{tag}.txt"
        enc_path = RESULTS_DIR / f"enc_sweep_{host}-{tag}.txt"
        if not dec_path.exists():
            continue
        decode, sizes = parse_decode_sweep(dec_path)
        encode = parse_encode_sweep(enc_path) if enc_path.exists() else {}
        out[host] = dict(decode=decode, encode=encode, sizes=sizes,
                         dec_path=str(dec_path.relative_to(REPO_ROOT)),
                         enc_path=str(enc_path.relative_to(REPO_ROOT))
                                  if enc_path.exists() else None)
    return out


# --- Tier coloring -------------------------------------------------------
#
# Map a ratio-vs-best number to a tier color.  Symmetric scale: anything
# below parity is a loss tier; anything above is a win tier.  Tiered so
# colors stay distinguishable even when the value range spans an order
# of magnitude.  Matches canasort's tier styling so the eye learns it
# once across projects.

def tier_class(ratio):
    """Return CSS class name for a ratio value (1.0 = parity)."""
    if ratio is None:
        return "empty"
    if ratio >= 8.0:    return "tier-win-a"
    if ratio >= 4.0:    return "tier-win-b"
    if ratio >= 2.0:    return "tier-win-c"
    if ratio >= 1.2:    return "tier-win-d"
    if ratio >= 0.95:   return "tier-parity"
    if ratio >= 0.7:    return "tier-loss-d"
    if ratio >= 0.5:    return "tier-loss-c"
    if ratio >= 0.3:    return "tier-loss-b"
    return "tier-loss-a"


def fmt_ratio(ratio):
    if ratio is None:
        return ""
    return f"{ratio:.2f}×"


# --- HTML scaffolding ----------------------------------------------------

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


SHARED_CSS = r"""
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0; padding: 0; color: #222; }
.page-header { padding: 12px 18px; background: #f7f7f7;
                border-bottom: 1px solid #ddd; position: relative; }
.page-header h1 { margin: 0 0 4px 0; font-size: 16px; font-weight: 600; }
.page-header .subtitle { font-size: 12px; color: #666; }
.page-header .gen-ts { position: absolute; top: 10px; right: 18px;
                        color: #555; font-size: 11px;
                        font-family: "SF Mono", Menlo, monospace;
                        background: white; padding: 2px 8px;
                        border: 1px solid #ddd; border-radius: 3px; }

.section { padding: 12px 18px; }
.section h2 { font-size: 14px; margin: 0 0 8px 0; font-weight: 600; }

.controls { padding: 8px 18px; background: #fafafa; border-bottom: 1px solid #eee;
            font-size: 12px; }
.controls .lbl { font-weight: 600; color: #555; margin-right: 8px; }
.controls .pill { display: inline-block; padding: 2px 8px;
                    margin: 0 3px 3px 0; border: 1px solid #bbb;
                    border-radius: 12px; cursor: pointer; background: white;
                    user-select: none; font-size: 11px; }
.controls .pill.on  { background: #2c5aa0; color: white; border-color: #2c5aa0; }
.controls .pill.off { background: #fff; color: #999; }

table.grid { border-collapse: collapse; font-size: 11px;
                font-variant-numeric: tabular-nums;
                font-family: "SF Mono", Menlo, monospace; }
table.grid th, table.grid td { padding: 3px 8px; border: 1px solid #ddd;
                                 text-align: right; white-space: nowrap; }
table.grid th.dist, table.grid td.dist { text-align: left; font-family: inherit;
                                            font-weight: 500; }
table.grid thead th { background: #ececec; }
table.grid tbody tr:hover td {
    box-shadow: inset 0 2px 0 0 #222, inset 0 -2px 0 0 #222;
}
table.grid tbody tr:hover td:first-child {
    box-shadow: inset 0 2px 0 0 #222, inset 0 -2px 0 0 #222,
                inset 2px 0 0 0 #222;
}
table.grid tbody tr:hover td:last-child {
    box-shadow: inset 0 2px 0 0 #222, inset 0 -2px 0 0 #222,
                inset -2px 0 0 0 #222;
}
/* Per-host group separators. */
table.grid th.group-end, table.grid td.group-end {
    border-right: 2px solid #777;
}

/* Tier coloring -- diverging palette centered on parity.
 * Same classes used by table cells AND legend swatches. */
.empty       { color: #ccc; text-align: center; }
.tier-win-a  { background: #006d2c; color: white; font-weight: 600; }
.tier-win-b  { background: #31a354; color: white; }
.tier-win-c  { background: #74c476; }
.tier-win-d  { background: #c7e9c0; }
.tier-parity { background: #fef8d1; }
.tier-loss-d { background: #fee0b6; }
.tier-loss-c { background: #fdb863; }
.tier-loss-b { background: #e08214; color: white; }
.tier-loss-a { background: #b35806; color: white; font-weight: 600; }

.legend { font-size: 11px; padding: 8px 18px; color: #555;
            border-bottom: 1px solid #eee; }
.legend .swatch { display: inline-block; width: 32px; height: 14px;
                   vertical-align: middle; margin: 0 4px 0 8px;
                   border: 1px solid #ccc; }

.footer { padding: 12px 18px; font-size: 11px; color: #888;
            border-top: 1px solid #eee; background: #fafafa; }
.footer a { color: #557; }
"""


def html_page(title, body_html, plotly=False, extra_css=""):
    """Wrap body content in a full HTML document with shared CSS.

    Set plotly=True to include the Plotly CDN script tag.
    `extra_css` is appended after SHARED_CSS for per-figure styling.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").strip()
    head_extras = ""
    if plotly:
        head_extras = f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
{head_extras}
<style>
{SHARED_CSS}
{extra_css}
</style>
</head>
<body>
<div class="page-header">
<span class="gen-ts">generated {ts}</span>
<h1>{title}</h1>
<div class="subtitle">pivco-huffman — research/exploration project</div>
</div>
{body_html}
<div class="footer">
Generated from <code>results/sweep_*.txt</code> by
<code>extras/figures/build.py</code>.
Re-run after a sweep: <code>cmake --build build --target figures</code>.
</div>
</body>
</html>"""


def tier_legend_html():
    """Reusable tier legend block for figures that use the tier scale."""
    return """<div class="legend">
  vs.&nbsp;huf0_x2 ratio:
  <span class="swatch tier-loss-a"></span>&lt;0.3
  <span class="swatch tier-loss-b"></span>&lt;0.5
  <span class="swatch tier-loss-c"></span>&lt;0.7
  <span class="swatch tier-loss-d"></span>&lt;0.95
  <span class="swatch tier-parity"></span>parity
  <span class="swatch tier-win-d"></span>&lt;2×
  <span class="swatch tier-win-c"></span>&lt;4×
  <span class="swatch tier-win-b"></span>&lt;8×
  <span class="swatch tier-win-a"></span>≥8×
</div>"""


def write_figure(filename, html):
    """Write to figures/<filename> at repo root.  Creates dir if needed."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename
    out.write_text(html, encoding="utf-8")
    return out
