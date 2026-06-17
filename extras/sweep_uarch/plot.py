#!/usr/bin/env python3
"""Plot ph-vs-huf0 dec MB/s ratio by CPU year, per family.

Reads results/sweep_uarch/<date>/<alias>.txt produced by sweep.sh and
extras/sweep_uarch/hosts.tsv for (alias -> family, year, uarch).

Per host: parse each `== <dist> ==` block, extract `ph` / `pha` / `huf0`
dec_op and dec_pb columns.  For each distribution where both ph and
huf0 succeeded, compute ratio = ph.dec_pb / huf0.dec_pb.

Per family-year point: min / mean / max ratio across distributions.

Output: PNG with one panel per family (intel / amd / graviton), x = year,
y = ratio (log), three series (min / mean / max), shaded min-max band.

Usage: plot.py <results_dir>
"""

import os, re, sys, glob, statistics, json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

HERE = os.path.dirname(os.path.abspath(__file__))

def load_hosts():
    out = {}
    with open(os.path.join(HERE, "hosts.tsv")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = re.split(r"[\t ]+", line)
            alias, itype, arch, family, year, uarch = parts[:6]
            out[alias] = {
                "instance": itype, "arch": arch, "family": family,
                "year": int(year), "uarch": uarch,
            }
    return out

# rows look like:  ph       |   1234    2345 |   3456    4567 |  1.20  1.30 | 12
ROW_RE = re.compile(
    r"^(\S+)\s*\|\s*"
    r"(\S+)\s+(\S+)\s*\|\s*"  # enc_op enc_pb
    r"(\S+)\s+(\S+)\s*\|\s*"  # dec_op dec_pb
    r"(\S+)\s+(\S+)\s*\|\s*"  # r_op r_pb
    r"(\d+)"
)
DIST_RE = re.compile(r"^==\s*(\S+)\s*==")

def parse_host_log(path):
    """Return {dist: {engine: (dec_op, dec_pb)}}"""
    out = defaultdict(dict)
    cur = None
    with open(path) as f:
        for line in f:
            m = DIST_RE.match(line)
            if m:
                cur = m.group(1); continue
            if cur is None: continue
            m = ROW_RE.match(line)
            if not m: continue
            name = m.group(1)
            try:
                dec_op = float(m.group(4)) if m.group(4) != "-" else None
                dec_pb = float(m.group(5)) if m.group(5) != "-" else None
            except ValueError:
                continue
            out[cur][name] = (dec_op, dec_pb)
    return out

def compute_ratios(host_data, num_engine="ph", den_engine="huf0", metric="dec_pb"):
    """Return list of ph/huf0 ratios across distributions (for one host)."""
    idx = 0 if metric == "dec_op" else 1
    rs = []
    for dist, eng in host_data.items():
        n = eng.get(num_engine, (None,None))[idx]
        d = eng.get(den_engine, (None,None))[idx]
        if n and d and d > 0:
            rs.append((dist, n / d))
    return rs

FAMILY_COLOR = {"intel": "#1f77b4", "amd": "#d62728", "graviton": "#2ca02c"}
FAMILY_TITLE = {"intel": "Intel", "amd": "AMD", "graviton": "AWS Graviton"}

def collect_series(results_dir, hosts, metric):
    """results_dir -> {family: [(year, alias, uarch, ratios), ...]}"""
    by_family = defaultdict(list)
    for alias, info in hosts.items():
        path = os.path.join(results_dir, alias + ".txt")
        if not os.path.exists(path):
            print(f"[warn] no result for {alias} at {path}", file=sys.stderr); continue
        data = parse_host_log(path)
        if not data:
            print(f"[warn] empty parse for {alias}", file=sys.stderr); continue
        ratios = compute_ratios(data, "ph", "huf0", metric)
        if not ratios:
            print(f"[warn] no ph/huf0 ratios for {alias}", file=sys.stderr); continue
        by_family[info["family"]].append((info["year"], alias, info["uarch"], ratios))
    return by_family


def _stats(points):
    """sorted points -> (xs, ys_min, ys_max, ys_mean, labels)"""
    points = sorted(points)
    xs      = [p[0] for p in points]
    ys_min  = [min(r for _, r in p[3]) for p in points]
    ys_max  = [max(r for _, r in p[3]) for p in points]
    ys_mean = [statistics.fmean(r for _, r in p[3]) for p in points]
    labels  = [p[1].replace("test-", "") for p in points]
    return xs, ys_min, ys_max, ys_mean, labels


def dump_summary(by_family, results_dir, metric):
    dump = {}
    for family, points in by_family.items():
        dump[family] = []
        for year, alias, uarch, ratios in sorted(points):
            rs = [r for (_, r) in ratios]
            dump[family].append({
                "year": year, "alias": alias, "uarch": uarch,
                "n_dists": len(rs), "min": min(rs), "max": max(rs),
                "mean": statistics.fmean(rs), "median": statistics.median(rs),
                "per_dist": dict(ratios),
            })
    with open(os.path.join(results_dir, f"summary_{metric}.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"wrote summary_{metric}.json")


def draw(by_family_new, metric, out_svg, mirror_svg, by_family_old=None):
    families = ["intel", "amd", "graviton"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, fam in zip(axes, families):
        # Old (paper v1.0) series, gray, drawn behind the new one.  Same shape:
        # dashed min/max, thick mean with dot markers.
        if by_family_old and by_family_old.get(fam):
            oxs, omn, omx, omean, _ = _stats(by_family_old[fam])
            ax.plot(oxs, omean, "-o", color="0.55", lw=2, ms=5, zorder=2,
                    label="paper v1.0 (mean)")
            ax.plot(oxs, omn, "--", color="0.55", lw=1, alpha=0.8, zorder=2)
            ax.plot(oxs, omx, "--", color="0.55", lw=1, alpha=0.8, zorder=2)
        points = by_family_new.get(fam, [])
        if not points:
            ax.set_title(f"{FAMILY_TITLE[fam]} (no data)")
            continue
        xs, ys_min, ys_max, ys_mean, labels = _stats(points)
        c = FAMILY_COLOR[fam]
        ax.fill_between(xs, ys_min, ys_max, color=c, alpha=0.18, label="min-max range")
        ax.plot(xs, ys_mean, "-o", color=c, lw=2, ms=7, zorder=3,
                label="mean across dists")
        ax.plot(xs, ys_min, "--", color=c, lw=1, alpha=0.7, zorder=3)
        ax.plot(xs, ys_max, "--", color=c, lw=1, alpha=0.7, zorder=3)
        for x, y, lbl in zip(xs, ys_mean, labels):
            ax.annotate(lbl, (x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        ax.text(0.98, 0.04, "1.0× (parity)", transform=ax.transAxes,
                ha="right", va="bottom", color="grey", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="grey", lw=0.5, alpha=0.85))
        ax.set_title(FAMILY_TITLE[fam])
        ax.set_xlabel("instance launch year")
        ax.set_ylabel(f"ph / Huff0 decode {metric} (×)")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0.0)          # always anchor Y at 0
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xticks(xs)
        tick_labels = []
        for i, x in enumerate(xs):
            if i > 0 and (x - xs[i-1]) <= 1:
                tick_labels.append(f"'{x % 100:02d}")
            else:
                tick_labels.append(str(x))
        ax.set_xticklabels(tick_labels)
    title = "PIVCO-Huffman decode speedup over Huff0 by CPU generation"
    if by_family_old:
        title += "  (gray = paper v1.0)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_svg);   print(f"wrote {out_svg}")
    fig.savefig(mirror_svg); print(f"wrote {mirror_svg}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("usage: plot.py <results_dir> [--overlay <old_results_dir>] [metric]",
              file=sys.stderr); sys.exit(1)
    results_dir = sys.argv[1]
    overlay_dir = None
    metric = "dec_op"   # stock huf0 has no prebuilt API; dec_op is apples-to-apples
    rest = sys.argv[2:]
    i = 0
    while i < len(rest):
        if rest[i] == "--overlay" and i + 1 < len(rest):
            overlay_dir = rest[i+1]; i += 2
        else:
            metric = rest[i]; i += 1

    hosts = load_hosts()
    new = collect_series(results_dir, hosts, metric)
    dump_summary(new, results_dir, metric)

    paper_plots = os.path.join(HERE, "..", "..", "paper", "plots")
    os.makedirs(paper_plots, exist_ok=True)

    # Plain current-only figure.
    draw(new, metric,
         os.path.join(results_dir, f"ph_vs_huf0_by_year_{metric}.svg"),
         os.path.join(paper_plots, f"sweep_uarch_{metric}.svg"))

    # Optional overlay of the old (paper v1.0) sweep in gray.
    if overlay_dir:
        old = collect_series(overlay_dir, hosts, metric)
        draw(new, metric,
             os.path.join(results_dir, f"ph_vs_huf0_by_year_{metric}_vs_paper.svg"),
             os.path.join(paper_plots, f"sweep_uarch_{metric}_vs_paper.svg"),
             by_family_old=old)

if __name__ == "__main__":
    main()
