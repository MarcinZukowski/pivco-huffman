#!/usr/bin/env python3
"""Build all pivco-huffman HTML figures into `figures/`.

Run manually:
    python3 extras/figures/build.py

Or via CMake (regenerates from the latest results/ files):
    cmake --build build --target figures

Figures emitted:
    figures/grid.html         tier-colored ratio table, hosts × distributions
    figures/ratio_curve.html  decode-throughput vs compression-ratio scatter,
                               with the FSE-on/off pair connected per (host, dist)
    figures/primitives.html   bench_micro per-primitive cost table
                               (placeholder for now -- needs bench_micro parser)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_html import (
    HOSTS, HOST_LABEL, HOST_DETAIL, HOST_COLOR,
    DIST_ORDER, DIST_FAMILY, DIST_MAIN_SET,
    load_sweep_set,
    tier_class, fmt_ratio, tier_legend_html,
    html_page, write_figure,
    PLOTLY_CDN,
)


# Default sweep tags.  build.py picks the no-FSE sweep as the headline
# (matching the README pitch); the with-FSE companion is loaded for the
# ratio-curve figure so the parameter sweep has at least two points
# per (host, distribution).
NOFSE_TAG = "20260515-unify-all-nofse"
FSEON_TAG = "20260515-unify-all"

INPUT_BYTES = 4 * 1024 * 1024  # the bench encodes a 4M-symbol stream


# ============================================================================
# grid.html -- combined absolute-throughput table for ph vs huff0,
#               decode + encode, hosts × distributions, tier-colored
#               by ph/huff0 ratio.
# ============================================================================

GRID_CSS = """
/* Per-host group cells: 4 cols (ph_dec, h0_dec, ph_enc, h0_enc). */
table.grid th.host-group { background: #ddd; font-size: 11px;
                             padding: 4px 8px; border-bottom: 2px solid #888;
                             text-align: center; }
table.grid th.host-detail { font-size: 10px; color: #666;
                              font-weight: 400; padding: 1px 4px 6px;
                              text-align: center; background: #ddd;
                              border-bottom: 2px solid #888; }
table.grid th.sub { font-size: 10px; color: #333; padding: 2px 6px;
                     background: #ececec; }
table.grid td.ph, table.grid td.h0 { padding: 3px 8px; min-width: 60px; }
table.grid td.h0 { color: #888; background: white !important; }
/* Group separator: thick right border every 4 cols. */
table.grid th.sep, table.grid td.sep { border-right: 2px solid #777; }
table.grid tr.hidden { display: none; }

/* Toggle pills. */
.toggle-row { padding: 8px 18px; background: #fafafa;
                border-bottom: 1px solid #eee; font-size: 12px; }
.toggle-row .lbl { font-weight: 600; color: #555; margin-right: 8px; }
.toggle-row .pill { display: inline-block; padding: 3px 10px;
                     margin: 0 4px 0 0; border: 1px solid #bbb;
                     border-radius: 12px; cursor: pointer; background: white;
                     user-select: none; font-size: 11px; }
.toggle-row .pill.on  { background: #2c5aa0; color: white; border-color: #2c5aa0; }
.toggle-row .pill:hover { box-shadow: 0 1px 3px rgba(0,0,0,0.15); }
"""


def render_combined_grid(sweep, hosts):
    """Combined absolute-throughput table.

    Per host: 4 columns — ph_dec, h0_dec, ph_enc, h0_enc.  ph_* cells
    are tier-colored by ph/h0 ratio for that metric direction.  h0_*
    cells stay uncolored as reference numbers.  Rows tagged data-main
    so the MAIN/ALL toggle can hide non-MAIN rows.
    """
    n_hosts = len(hosts)
    rows = []
    rows.append('<table class="grid"><thead>')

    # Row 1: host instance label spanning 4 cols per host.
    rows.append('<tr><th rowspan="3" class="dist">distribution</th>')
    for i, h in enumerate(hosts):
        cls = "host-group" + (" sep" if i < n_hosts - 1 else "")
        rows.append(f'<th class="{cls}" colspan="4">{HOST_LABEL[h]}</th>')
    rows.append('</tr>')

    # Row 2: host detail (instance type + chip + clock).
    rows.append('<tr>')
    for i, h in enumerate(hosts):
        cls = "host-detail" + (" sep" if i < n_hosts - 1 else "")
        rows.append(f'<th class="{cls}" colspan="4">{HOST_DETAIL[h]}</th>')
    rows.append('</tr>')

    # Row 3: per-host sub-columns.
    rows.append('<tr>')
    for i, h in enumerate(hosts):
        sub_cls_last = " sep" if i < n_hosts - 1 else ""
        rows.append('<th class="sub" title="PIVCO decode M/s">ph dec</th>')
        rows.append('<th class="sub" title="huf0_x2 decode M/s">huf0 dec</th>')
        rows.append('<th class="sub" title="PIVCO encode M/s">ph enc</th>')
        rows.append(f'<th class="sub{sub_cls_last}" title="huf0_x2 encode M/s">huf0 enc</th>')
    rows.append('</tr></thead><tbody>')

    # Data rows.
    for d in DIST_ORDER:
        any_host = any(d in sweep[h]['decode'] for h in hosts)
        if not any_host:
            continue
        in_main = "true" if d in DIST_MAIN_SET else "false"
        rows.append(f'<tr data-main="{in_main}"><td class="dist">{d}</td>')
        for i, h in enumerate(hosts):
            dec = sweep[h]['decode'].get(d)
            enc = sweep[h]['encode'].get(d)
            sep_last = " sep" if i < n_hosts - 1 else ""

            # ph decode + huff0 decode.
            if dec:
                ph_dec, h0_dec = dec['pivco_bu'], dec['huf0_x2']
                ratio = dec.get('ratio')
                tier = tier_class(ratio)
                rows.append(f'<td class="ph {tier}" title="ratio {fmt_ratio(ratio)}">'
                            f'{ph_dec:,}</td>')
                rows.append(f'<td class="h0">'
                            f'{h0_dec:,}' if h0_dec else '<td class="h0 empty">—</td>')
                if h0_dec:
                    rows.append('</td>')
            else:
                rows.append('<td class="ph empty">—</td><td class="h0 empty">—</td>')

            # ph encode + huff0 encode.
            if enc:
                ph_enc, h0_enc = enc['pivco'], enc['huf0_x2']
                ratio_e = enc.get('ratio')
                tier_e = tier_class(ratio_e)
                rows.append(f'<td class="ph {tier_e}" title="ratio {fmt_ratio(ratio_e)}">'
                            f'{ph_enc:,}</td>')
                rows.append(f'<td class="h0{sep_last}">'
                            f'{h0_enc:,}' if h0_enc else f'<td class="h0 empty{sep_last}">—</td>')
                if h0_enc:
                    rows.append('</td>')
            else:
                rows.append(f'<td class="ph empty">—</td>'
                            f'<td class="h0 empty{sep_last}">—</td>')
        rows.append('</tr>')

    rows.append('</tbody></table>')
    return '\n'.join(rows)


GRID_TOGGLE_JS = """
<script>
(function() {
    var pills = document.querySelectorAll('.toggle-row .pill');
    pills.forEach(function(p) {
        p.addEventListener('click', function() {
            pills.forEach(function(x) { x.classList.remove('on'); });
            p.classList.add('on');
            var mode = p.dataset.mode;
            document.querySelectorAll('table.grid tbody tr').forEach(function(tr) {
                if (mode === 'all') {
                    tr.classList.remove('hidden');
                } else {
                    if (tr.dataset.main === 'true') tr.classList.remove('hidden');
                    else                            tr.classList.add('hidden');
                }
            });
        });
    });
})();
</script>
"""


def build_grid_html(sweep_nofse):
    hosts = [h for h in HOSTS if h in sweep_nofse]
    legend = tier_legend_html()

    toggle = """
<div class="toggle-row">
  <span class="lbl">Distributions:</span>
  <span class="pill on" data-mode="main">MAIN (10)</span>
  <span class="pill" data-mode="all">ALL (30)</span>
  <span style="color:#888; margin-left:12px; font-size:11px;">
    Tier coloring on PIVCO cells is the per-direction ratio vs huf0_x2
    (or trad_4s when huf0 fails).  huf0 cells are reference, uncolored.
  </span>
</div>
"""

    grid_section = (
        '<div class="section">'
        '<h2>Decode + encode throughput, M symbols / second '
        '(<code>--no-fse</code> default config)</h2>\n'
        + render_combined_grid(sweep_nofse, hosts)
        + '</div>'
    )

    # On initial render, hide non-MAIN rows so MAIN is the default view.
    initial_hide_script = """
<script>
document.querySelectorAll('table.grid tbody tr').forEach(function(tr) {
    if (tr.dataset.main !== 'true') tr.classList.add('hidden');
});
</script>
"""

    body = legend + toggle + grid_section + initial_hide_script + GRID_TOGGLE_JS
    extra_css = GRID_CSS
    return html_page("pivco-huffman — throughput grid",
                     body, extra_css=extra_css)


# ============================================================================
# ratio_curve.html -- decode-throughput vs compression-ratio scatter,
#                      with FSE-on/off pairs connected per (host, dist)
# ============================================================================

def build_ratio_curve_html(sweep_nofse, sweep_fseon):
    """One trace per host.  Each (host, dist) is two points (no-FSE +
    FSE-on) connected by a line.  X = decode M/s, Y = compression ratio.
    """
    traces = []
    for host in HOSTS:
        if host not in sweep_nofse:
            continue
        nofse = sweep_nofse[host]
        fseon = sweep_fseon.get(host, {'decode': {}, 'sizes': {}})

        xs, ys, txts = [], [], []
        for d in DIST_ORDER:
            n_dec = nofse['decode'].get(d)
            f_dec = fseon.get('decode', {}).get(d)
            n_sz  = nofse['sizes'].get(d)
            f_sz  = fseon.get('sizes', {}).get(d)

            # Plot two points + a connecting None to break the line at
            # the segment end.  Plotly accepts None as a gap marker.
            if n_dec and n_sz:
                xs.append(n_dec['pivco_bu'])
                ys.append(INPUT_BYTES / n_sz['pivco_raw_bytes'])
                txts.append(f"{d} (no-FSE)<br>"
                            f"decode {n_dec['pivco_bu']:,} M/s<br>"
                            f"ratio {INPUT_BYTES/n_sz['pivco_raw_bytes']:.3f}×")
            if f_dec and f_sz:
                xs.append(f_dec['pivco_bu'])
                ys.append(INPUT_BYTES / f_sz['pivco_raw_bytes'])
                txts.append(f"{d} (FSE on)<br>"
                            f"decode {f_dec['pivco_bu']:,} M/s<br>"
                            f"ratio {INPUT_BYTES/f_sz['pivco_raw_bytes']:.3f}×")
            # Segment break.
            xs.append(None); ys.append(None); txts.append(None)

        traces.append(dict(
            name=HOST_LABEL[host],
            x=xs, y=ys,
            text=txts,
            mode="lines+markers",
            marker=dict(size=7, color=HOST_COLOR[host]),
            line=dict(color=HOST_COLOR[host], width=1),
            hovertemplate="%{text}<extra></extra>",
        ))

    fig = dict(
        data=traces,
        layout=dict(
            title="Decode throughput vs compression ratio "
                  "(FSE off → FSE on per (host, distribution))",
            xaxis=dict(title="decode M symbols/sec  (log)", type="log"),
            yaxis=dict(title="compression ratio  (input_bytes / pivco_raw_bytes)"),
            hovermode="closest",
            template="plotly_white",
            height=620,
            legend=dict(orientation="h", y=-0.15),
        ),
    )

    body = f"""
<div class="section">
<p style="font-size:12px;color:#555;">
Each (host, distribution) appears as a 2-point line: <b>no-FSE</b>
and <b>FSE-on</b> endpoints connected.  The line slope is the
local ratio/speed tradeoff exposed by the FSE knob — steep
downward = decode pays heavily for a small ratio gain, gentle = the
FSE compression win is "free-ish" for that distribution.  Click a
host in the legend to toggle.
</p>
</div>

<div id="plot" style="width:100vw; height:660px;"></div>

<script>
var FIG = {json.dumps(fig)};
Plotly.newPlot('plot', FIG.data, FIG.layout, {{responsive: true}});
</script>
"""
    return html_page("pivco-huffman — decode/ratio curve", body, plotly=True)


# ============================================================================
# primitives.html -- placeholder; needs a parser for results/bench_micro-*.txt
# ============================================================================

def build_primitives_html():
    body = """
<div class="section">
<p>Per-primitive microbench cost (<code>bench_micro</code>) figure is
not yet implemented.  Parser for <code>results/bench_micro-*.txt</code>
is the missing piece; once landed this page will show ns/elem and
GB/s per primitive × platform with tier coloring.</p>

<p>Tracking: <a href="../IDEAS.md">IDEAS.md</a>.  Static reference
copy of the current numbers is in the project
<a href="../README.md#cross-platform-primitive-costs">README</a>
"Cross-platform primitive costs" section.</p>
</div>"""
    return html_page("pivco-huffman — primitive costs (placeholder)", body)


# ============================================================================

def main():
    sweep_nofse = load_sweep_set(NOFSE_TAG)
    sweep_fseon = load_sweep_set(FSEON_TAG)

    if not sweep_nofse:
        print(f"error: no-FSE sweep '{NOFSE_TAG}' not found in results/", file=sys.stderr)
        return 1

    grid_html = build_grid_html(sweep_nofse)
    out = write_figure("grid.html", grid_html)
    print(f"wrote {out.relative_to(out.parents[1])}")

    curve_html = build_ratio_curve_html(sweep_nofse, sweep_fseon)
    out = write_figure("ratio_curve.html", curve_html)
    print(f"wrote {out.relative_to(out.parents[1])}")

    prim_html = build_primitives_html()
    out = write_figure("primitives.html", prim_html)
    print(f"wrote {out.relative_to(out.parents[1])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
