#!/usr/bin/env python3
"""Generate the typst table + per-host bar plot for the tree-mode ablation.

Reads bench_tree_modes output files (one per host) and produces:
 - paper-ready typst snippet on stdout (12 data columns: 4 ph modes +
   huf0 + oo-huff for each of M4 and c8i)
 - paper/plots/tree_modes_<host>.svg per host (grouped bar chart)

Missing engines (e.g. oo_huff on hosts without Oodle staged) are filled
from paper/data/fair.csv when available; cell becomes "—" otherwise.

usage: tree_modes_table.py m4=<path> c8i=<path>
"""
import csv, os, re, sys
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.normpath(os.path.join(HERE, "..", ".."))
PLOTS  = os.path.join(ROOT, "paper", "plots")
FAIRCSV = os.path.join(ROOT, "paper", "data", "fair.csv")

ENGINES_PH    = ["NAIVE", "FUSED", "CANONICAL_FLAT", "OPTIMIZED"]
ENGINES_BASE  = ["huf0", "oo_huff"]
ENGINES_ALL   = ENGINES_PH + ENGINES_BASE
ENGINE_LABEL  = {"NAIVE": "PH naive", "FUSED": "PH fused leaves",
                 "CANONICAL_FLAT": "PH flat", "OPTIMIZED": "PH flat opt.",
                 "huf0": "Huff0", "oo_huff": "Oodle Huffman"}
# Short labels for the dense typst table header (kept brief).
TABLE_LABEL   = {"NAIVE": "naive", "FUSED": "fused leaves",
                 "CANONICAL_FLAT": "flat", "OPTIMIZED": "flat opt.",
                 "huf0": "Huff0", "oo_huff": "Oodle Huffman"}
HOST_LABEL    = {"m4": "M4", "c8i": "c8i"}

ROW_RE = re.compile(
    r"^\s*(\S+)\s+(NAIVE|FUSED|CANONICAL_FLAT|OPTIMIZED|huf0|oo_huff)\s+"
    r"(\d+|n/a)\b"
)

def parse_bench(path):
    """Returns {dataset: {engine: mbs or None}}"""
    out = {}
    with open(path) as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m: continue
            ds, eng, mbs = m.group(1), m.group(2), m.group(3)
            out.setdefault(ds, {})[eng] = None if mbs == "n/a" else float(mbs)
    return out

def parse_faircsv(host, engine, dataset):
    """Backfill from paper/data/fair.csv (column 6 = dec_op)."""
    if not os.path.exists(FAIRCSV): return None
    with open(FAIRCSV) as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) < 6: continue
            if row[0] == host and row[1] == dataset and row[2] == engine:
                try: return float(row[5])
                except ValueError: return None
    return None

def fill_missing(host, data):
    """In place: replace None cells with fair.csv values where available."""
    for ds, eng_dict in data.items():
        for eng in ENGINES_ALL:
            if eng_dict.get(eng) is None:
                v = parse_faircsv(host, eng, ds)
                if v is not None: eng_dict[eng] = v

# -------------------- typst emit --------------------

def emit_typst(host_data, datasets, hosts):
    # 13 columns: dataset, then 6 engines x len(hosts)
    n_eng = len(ENGINES_ALL)
    print("// Per-dataset decode bandwidth (MB/s) across the 4 ph tree-build")
    print("// modes + huf0 (stock HUF_decompress) + Oodle Huffman.  Same session")
    print("// per host; FSE disabled in all ph variants.")
    print(f"#figure(")
    print(f"  table(")
    print(f"    columns: {1 + n_eng * len(hosts)},")
    print(f"    align: (col, _) => if col == 0 {{ left }} else {{ right }},")
    print(f"    table.header(")
    print(f"      table.cell(rowspan: 2)[*Dataset*],")
    for h in hosts:
        print(f"      table.cell(colspan: {n_eng})[*{HOST_LABEL.get(h,h)}*],")
    print(f"      " + ",  ".join(f"[*{TABLE_LABEL[e]}*]" for h in hosts for e in ENGINES_ALL) + ",")
    print(f"    ),")
    for ds in datasets:
        cells = [f"[{ds}]"]
        for h in hosts:
            for e in ENGINES_ALL:
                v = host_data[h].get(ds, {}).get(e)
                cells.append(f"[{v:.0f}]" if v is not None else "[—]")
        print(f"    " + ",  ".join(cells) + ",")
    print(f"  ),")
    print(f"  caption: [Decode bandwidth (MB/s) per dataset and engine on")
    print(f"            M4 and c8i.  *ph* tree-build modes: *naive* (every")
    print(f"            symbol a singleton, pure canonical Huffman tree);")
    print(f"            *fused* (D=1 sibling-pair leaf fusion); *canon+flat*")
    print(f"            (detect flat subtrees in the canonical tree);")
    print(f"            *optimized* (reorganize to maximize flat coverage).")
    print(f"            *huf0* is stock @fse HUF_decompress.")
    print(f"            *oo-huff* is Oodle Huffman @giesen2021oodle.")
    print(f"            FSE disabled in all ph variants to isolate")
    print(f"            tree-shape effects.],")
    print(f")<tab-tree-modes>")

# -------------------- per-host bar plot --------------------

# PH gradient = colorbrewer Greens 4-step (light -> dark).  Endpoints
# (#a6dba0 light, #1b7837 dark) match the existing enc-bars-* SVG palette
# in paper/plots/common.typ so PH-flat-opt reads as the same engine across
# all paper plots.  Huff0 (#ef8a3b orange) and Oodle Huffman (#9467bd
# purple) also match common.typ.
ENG_COLORS = {
    "NAIVE":          "#d9f0d3",
    "FUSED":          "#a6dba0",
    "CANONICAL_FLAT": "#5aae61",
    "OPTIMIZED":      "#1b7837",
    "huf0":           "#ef8a3b",
    "oo_huff":        "#9467bd",
}

def plot_host(host, data, datasets):
    n_eng = len(ENGINES_ALL)
    n_ds  = len(datasets)
    # Y-axis caps match paper/plots/dec-bw.typ so all "decode bandwidth"
    # plots in the paper share the same vertical scale.  Bars exceeding the
    # cap are clipped at the axis top and the true value is labelled above
    # them in bold (matches the break-mark + label convention in common.typ).
    CAPS = {"m4": 9000, "c8i": 10000}
    cap = CAPS.get(host)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bar_w = 0.85 / n_eng
    xs = list(range(n_ds))
    for i, e in enumerate(ENGINES_ALL):
        ys_true  = [data.get(ds, {}).get(e) or 0 for ds in datasets]
        ys_drawn = [min(y, cap) if cap is not None else y for y in ys_true]
        offs     = [x + (i - (n_eng - 1) / 2) * bar_w for x in xs]
        ax.bar(offs, ys_drawn, width=bar_w, label=ENGINE_LABEL[e],
               color=ENG_COLORS.get(e, "#888"), edgecolor="black", lw=0.3)
        # Annotate clipped bars with their true value + draw a small "broken"
        # marker at the bar top.
        if cap is None: continue
        for x_off, yv in zip(offs, ys_true):
            if yv > cap:
                ax.text(x_off, cap * 0.985, f"{int(round(yv))}",
                        ha="center", va="top", fontsize=7, weight="bold",
                        rotation=90, color="white")
                # zigzag break-mark across the bar top
                bx0, bx1 = x_off - bar_w * 0.40, x_off + bar_w * 0.40
                ax.plot([bx0, bx1], [cap * 0.97, cap * 1.00],
                        color="black", lw=0.7, clip_on=False)
                ax.plot([bx0, bx1], [cap * 0.985, cap * 1.015],
                        color="black", lw=0.7, clip_on=False)
    if cap is not None:
        ax.set_ylim(0, cap)
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("decode MB/s")
    ax.set_title(HOST_LABEL.get(host, host), fontsize=14, weight="bold")
    # Vertical legend outside the plot area on the right — matches the
    # cetz-rendered legends in paper/plots/enc-bars-*.svg and dec-bw-*.svg.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10,
              frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    out = os.path.join(PLOTS, f"tree_modes_{host}.svg")
    fig.savefig(out)
    print(f"wrote {out}", file=sys.stderr)
    plt.close(fig)

# -------------------- main --------------------

def main():
    hosts = []
    host_data = {}
    for arg in sys.argv[1:]:
        h, _, p = arg.partition("=")
        if not h or not p: continue
        host_data[h] = parse_bench(p)
        hosts.append(h)
    if not hosts:
        print("usage: tree_modes_table.py m4=<m4.txt> c8i=<c8i.txt>", file=sys.stderr)
        sys.exit(1)
    # backfill missing engines from fair.csv (e.g. oo_huff on c8i where
    # Oodle isn't staged on the remote)
    for h in hosts:
        fill_missing(h, host_data[h])
    # collect dataset names (union, in the order they appear in the first host)
    datasets = []
    seen = set()
    for h in hosts:
        for ds in host_data[h].keys():
            if ds not in seen:
                seen.add(ds); datasets.append(ds)
    emit_typst(host_data, datasets, hosts)
    for h in hosts:
        plot_host(h, host_data[h], datasets)

if __name__ == "__main__":
    main()
