#!/usr/bin/env python3
"""Generate the typst table + per-host bar plot for the tree-mode ablation.

Reads paper/data/fair.csv directly (the long-format consolidated table that
all paper figures share -- so this comparison stays consistent with the
ANS / encoding plots).  Engines used: ph_naive, ph_flat, ph (= OPTIMIZED),
huf0, oo_huff.  FUSED is excluded -- in the bottom-up codec the merge_two
fast path fires automatically for any sibling-pair internal node, so
explicit FUSED tree-build mode is indistinguishable from NAIVE.

Outputs:
 - paper-ready typst snippet on stdout (10 data columns: 4 engines
   for each of M4 and c8i)
 - paper/plots/tree_modes_<host>.svg per host (grouped bar chart)

usage: tree_modes_table.py [hosts=m4,c8i]
"""
import csv, os, re, sys
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.normpath(os.path.join(HERE, "..", ".."))
PLOTS  = os.path.join(ROOT, "paper", "plots")
FAIRCSV = os.path.join(ROOT, "paper", "data", "fair.csv")

# fair.csv engine names, in plot left-to-right order.
ENGINES_ALL   = ["ph_naive", "ph_flat", "ph", "huf0", "oo_huff"]
ENGINE_LABEL  = {"ph_naive": "PH naive",
                 "ph_flat":  "PH flat",
                 "ph":       "PH flat opt.",
                 "huf0":     "Huff0",
                 "oo_huff":  "Oodle Huffman"}
# Short labels for the dense typst table header (kept brief).
TABLE_LABEL   = {"ph_naive": "naive",
                 "ph_flat":  "flat",
                 "ph":       "flat opt.",
                 "huf0":     "Huff0",
                 "oo_huff":  "Oodle Huffman"}
HOST_LABEL    = {"m4": "M4", "c8i": "c8i"}
# Datasets to plot, in left-to-right order on the x-axis.
DATASETS      = ["proba80", "english", "html_wiki", "prose_pride",
                 "json_api", "dna_fasta", "chinese_text", "calgary_pic"]

def load_faircsv():
    """Returns {(host, dataset, engine): dec_op}."""
    out = {}
    if not os.path.exists(FAIRCSV): return out
    with open(FAIRCSV) as f:
        rdr = csv.reader(f)
        next(rdr, None)        # skip header
        for row in rdr:
            if len(row) < 6: continue
            host, ds, eng = row[0], row[1], row[2]
            try: out[(host, ds, eng)] = float(row[5])  # dec_op column
            except ValueError: pass
    return out

# -------------------- typst emit --------------------

def emit_typst(fair, datasets, hosts):
    # 1 + 5*len(hosts) columns: dataset, then 5 engines x len(hosts)
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
                v = fair.get((h, ds, e))
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

# PH gradient = colorbrewer Greens 3-step (light -> dark).  The dark
# endpoint (#1b7837) matches the existing enc-bars-* / dec-bw-* SVG
# palette in paper/plots/common.typ so PH-flat-opt reads as the same
# engine across all paper plots.  Huff0 (#ef8a3b orange) and Oodle
# Huffman (#9467bd purple) also match common.typ.
ENG_COLORS = {
    "ph_naive": "#a6dba0",
    "ph_flat":  "#5aae61",
    "ph":       "#1b7837",
    "huf0":     "#ef8a3b",
    "oo_huff":  "#9467bd",
}

def plot_host(host, fair, datasets):
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
        ys_true  = [fair.get((host, ds, e)) or 0 for ds in datasets]
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
    hosts = ["m4", "c8i"]
    for arg in sys.argv[1:]:
        if arg.startswith("hosts="):
            hosts = arg.partition("=")[2].split(",")
    fair = load_faircsv()
    if not fair:
        print(f"no data found in {FAIRCSV}", file=sys.stderr); sys.exit(1)
    emit_typst(fair, DATASETS, hosts)
    for h in hosts:
        plot_host(h, fair, DATASETS)

if __name__ == "__main__":
    main()
