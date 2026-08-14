/* Huffman tree visualizer.  Emits an SVG comparing the canonical-Huffman
 * tree shape against the flat-aware tree shape (current production) for
 * each benchmark distribution.  Maximal flat-D>=2 subtrees are
 * highlighted with colored rectangles so the difference between the two
 * shapes is visually obvious.
 *
 * No code labels or 0/1 markers — shape only, plus flat-subtree boxes.
 *
 * Usage:
 *   pivco_tree_viz [distribution] > tree.svg
 *   pivco_tree_viz --all > all.svg
 */

#include "pivco_huffman.h"
#include "bench_ctx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern void           bench_init(void);
extern int            bench_num_distributions(void);
extern const char    *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

/* ---------- Tree helpers ---------- */

static int subtree_leaves(const pivco_tree_node_t *tree, int16_t node)
{
    if (tree[node].symbol >= 0) return 1;
    return subtree_leaves(tree, tree[node].left) +
           subtree_leaves(tree, tree[node].right);
}

static int local_min(const pivco_tree_node_t *tree, int16_t node)
{
    if (tree[node].symbol >= 0) return 0;
    int l = local_min(tree, tree[node].left);
    int r = local_min(tree, tree[node].right);
    return 1 + (l < r ? l : r);
}

static int local_max(const pivco_tree_node_t *tree, int16_t node)
{
    if (tree[node].symbol >= 0) return 0;
    int l = local_max(tree, tree[node].left);
    int r = local_max(tree, tree[node].right);
    return 1 + (l > r ? l : r);
}

/* Maximal flat-subtree depth at `node`, or -1 if not a flat-D>=1 root.
 * D>=2 = flat-subtree fast path (production); D=1 = sibling pair handled
 * by stage fusion in decode_node_neon.  Both are "fast-path" subtrees;
 * mark them so we can colour each kind distinctly. */
static int flat_D_at(const pivco_tree_node_t *tree, int16_t node)
{
    if (tree[node].symbol >= 0) return -1;
    int lmin = local_min(tree, node);
    int lmax = local_max(tree, node);
    return (lmin == lmax && lmin >= 1) ? lmin : -1;
}

/* ---------- Build canonical tree from code lengths ----------
 * Replicates the OLD pivco_build_table behaviour (sort symbols
 * by (length, value), assign canonical codes, walk codes MSB-first).
 * `code_len[]` is taken from a freshly-built table — code lengths are
 * the same in both canonical and flat-aware variants. */
static void build_canonical(const uint8_t *code_len,
                             pivco_table_t *out)
{
    memset(out, 0, sizeof(*out));
    memcpy(out->code_len, code_len, sizeof(out->code_len));

    uint8_t max_len = 0, min_len = 255;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (code_len[s] > 0) {
            out->sym_count[code_len[s]]++;
            if (code_len[s] > max_len) max_len = code_len[s];
            if (code_len[s] < min_len) min_len = code_len[s];
        }
    }
    out->max_len = max_len;
    out->min_len = min_len;

    /* Canonical assignment: sort by (length, sym), assign sequential
     * codes, shift up between length classes. */
    uint16_t code = 0;
    for (int len = 1; len <= max_len; len++) {
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            if (code_len[s] == len) {
                out->code[s] = code;
                code++;
            }
        }
        code <<= 1;
    }

    /* Build tree from canonical codes (walk MSB-first). */
    int16_t nc = 0;
    out->tree[nc].symbol = -1;
    out->tree[nc].left   = -1;
    out->tree[nc].right  = -1;
    nc++;
    out->tree_root = 0;

    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (code_len[s] == 0) continue;
        uint16_t c   = out->code[s];
        int     len  = code_len[s];
        int16_t cur  = 0;
        for (int b = len - 1; b >= 0; b--) {
            int bit = (c >> b) & 1;
            int16_t *child = bit ? &out->tree[cur].right : &out->tree[cur].left;
            if (*child < 0) {
                *child = nc;
                out->tree[nc].symbol = -1;
                out->tree[nc].left   = -1;
                out->tree[nc].right  = -1;
                nc++;
            }
            cur = *child;
        }
        out->tree[cur].symbol = (int16_t)s;
    }
    out->tree_node_count = nc;
}

/* ---------- SVG rendering ---------- */

#define LEAF_W            6.0   /* horizontal slot per leaf, px */
#define LEVEL_H          22.0   /* vertical px per tree depth */
#define LEAF_R            1.5   /* leaf circle radius */
#define INTERN_R          1.0   /* internal-node radius */
#define BOX_PAD           3.0   /* flat-subtree box padding around contents */

/* Emit lines+nodes for `node`.  `in_box` is true if any ancestor is a
 * flat-D>=1 maximal subtree (= leaf is inside a colored flat box).
 * Solo leaves (in_box=false) get a red fill so they stand out as the
 * elements that take the slowest decode path. */
static double draw_tree(FILE *f, const pivco_tree_node_t *tree,
                         int16_t node, double x_min, double x_max,
                         int depth, double x_parent, int in_box)
{
    double cx = (x_min + x_max) * 0.5;
    double cy = depth * LEVEL_H + LEVEL_H * 0.5;

    if (x_parent >= 0) {
        double py = (depth - 1) * LEVEL_H + LEVEL_H * 0.5;
        fprintf(f, "  <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" "
                "stroke=\"#888\" stroke-width=\"0.5\"/>\n",
                x_parent, py, cx, cy);
    }

    if (tree[node].symbol >= 0) {
        const char *fill = in_box ? "#222" : "#d63a3a";  /* solo = red */
        fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"%s\"/>\n",
                cx, cy, LEAF_R, fill);
    } else {
        int next_in_box = in_box || (flat_D_at(tree, node) >= 1);
        int lleaves = subtree_leaves(tree, tree[node].left);
        int rleaves = subtree_leaves(tree, tree[node].right);
        double split = x_min + (x_max - x_min) *
                       (double)lleaves / (double)(lleaves + rleaves);
        draw_tree(f, tree, tree[node].left,  x_min, split, depth + 1, cx, next_in_box);
        draw_tree(f, tree, tree[node].right, split, x_max, depth + 1, cx, next_in_box);
        fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" "
                "fill=\"none\" stroke=\"#aaa\" stroke-width=\"0.4\"/>\n",
                cx, cy, INTERN_R);
    }
    return cx;
}

/* Walk each leaf, compute its op cost = (#partitioning ancestors + 1
 * terminal op).  Accumulate unweighted totals (per leaf) and freq-
 * weighted totals (per decoded element).  `flat_root_depth` is -1
 * outside a flat-D>=2 subtree, else the depth of the enclosing flat
 * root — leaves inside that subtree all share the same ops count
 * (flat_root_depth + 1).  Leaves outside (solo / D=1) charge `depth`. */
static void leaf_ops_walk(const pivco_tree_node_t *tree, int16_t node,
                           int depth, int flat_root_depth,
                           int *total_ops, int *n_leaves,
                           const uint64_t *freq,
                           uint64_t *total_freq, double *total_freq_ops)
{
    if (tree[node].symbol >= 0) {
        int ops = (flat_root_depth >= 0) ? (flat_root_depth + 1) : depth;
        *total_ops += ops;
        *n_leaves += 1;
        if (freq) {
            uint64_t fw = freq[tree[node].symbol];
            *total_freq     += fw;
            *total_freq_ops += (double)fw * (double)ops;
        }
        return;
    }
    int next = flat_root_depth;
    if (next < 0 && flat_D_at(tree, node) >= 2) next = depth;
    leaf_ops_walk(tree, tree[node].left,  depth + 1, next,
                   total_ops, n_leaves, freq, total_freq, total_freq_ops);
    leaf_ops_walk(tree, tree[node].right, depth + 1, next,
                   total_ops, n_leaves, freq, total_freq, total_freq_ops);
}

static void avg_ops_per_leaf(const pivco_table_t *t,
                              const uint64_t *freq,
                              double *out_unweighted,
                              double *out_weighted)
{
    int total_ops = 0, n_leaves = 0;
    uint64_t total_freq = 0;
    double total_freq_ops = 0.0;
    leaf_ops_walk(t->tree, t->tree_root, 0, -1,
                   &total_ops, &n_leaves,
                   freq, &total_freq, &total_freq_ops);
    *out_unweighted = n_leaves ? (double)total_ops / (double)n_leaves : 0.0;
    *out_weighted   = (freq && total_freq) ? total_freq_ops / (double)total_freq : 0.0;
}

/* Walk and emit a colored rectangle around every maximal flat-D>=2
 * subtree in this tree.  Done as a separate first pass so the boxes
 * draw BEHIND the tree edges/nodes. */
typedef struct {
    double x, y, w, h;
    int    D, n_leaves;
} flat_box_t;

static void collect_flat_boxes(const pivco_tree_node_t *tree, int16_t node,
                                double x_min, double x_max, int depth,
                                flat_box_t *boxes, int *nbox)
{
    if (tree[node].symbol >= 0) return;
    int D = flat_D_at(tree, node);
    if (D >= 1) {
        int n_leaves = 1 << D;
        boxes[*nbox].x = x_min - BOX_PAD;
        boxes[*nbox].y = depth * LEVEL_H + LEVEL_H * 0.5 - BOX_PAD - LEAF_R;
        boxes[*nbox].w = (x_max - x_min) + BOX_PAD * 2;
        boxes[*nbox].h = D * LEVEL_H + BOX_PAD * 2 + LEAF_R * 2;
        boxes[*nbox].D = D;
        boxes[*nbox].n_leaves = n_leaves;
        (*nbox)++;
        return;     /* maximal — don't descend */
    }
    int lleaves = subtree_leaves(tree, tree[node].left);
    int rleaves = subtree_leaves(tree, tree[node].right);
    double split = x_min + (x_max - x_min) *
                   (double)lleaves / (double)(lleaves + rleaves);
    collect_flat_boxes(tree, tree[node].left,  x_min, split, depth + 1,
                       boxes, nbox);
    collect_flat_boxes(tree, tree[node].right, split, x_max, depth + 1,
                       boxes, nbox);
}

/* Render one tree at canvas offset (ox, oy).  Width = leaves * LEAF_W,
 * height = max_len * LEVEL_H + LEAF_R*2.  Returns total width drawn. */
static double render_tree_panel(FILE *f, const pivco_table_t *t,
                                 const uint64_t *freq,
                                 const char *title, double ox, double oy,
                                 double *out_panel_h)
{
    int n_leaves = subtree_leaves(t->tree, t->tree_root);
    int max_len  = t->max_len;
    double w = n_leaves * LEAF_W;
    double h = (double)max_len * LEVEL_H + LEAF_R * 2 + LEVEL_H * 0.5;

    fprintf(f, "<g transform=\"translate(%.1f, %.1f)\">\n", ox, oy);
    fprintf(f, "  <text x=\"%.1f\" y=\"-6\" text-anchor=\"middle\" "
            "font-family=\"sans-serif\" font-size=\"11\" fill=\"#222\">%s</text>\n",
            w * 0.5, title);

    /* Flat boxes first (background). */
    flat_box_t boxes[PIVCO_MAX_SYMBOLS];
    int nbox = 0;
    collect_flat_boxes(t->tree, t->tree_root, 0.0, w, 0, boxes, &nbox);
    int total_flat_leaves = 0;
    int total_d2plus_leaves = 0;
    for (int i = 0; i < nbox; i++) {
        /* D=1 (sibling pair, stage-fusion path): muted blue.
         * D>=2 (flat-subtree path): green, deeper = darker. */
        int D = boxes[i].D;
        const char *fill;
        if      (D == 1) fill = "#a8c3e0";   /* D=1 sib pair */
        else if (D >= 6) fill = "#1f7a3a";
        else if (D == 5) fill = "#3aa65b";
        else if (D == 4) fill = "#62c282";
        else if (D == 3) fill = "#9ad9b1";
        else             fill = "#cdebd6";   /* D=2 */
        fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" "
                "fill=\"%s\" fill-opacity=\"0.55\" stroke=\"%s\" "
                "stroke-width=\"0.6\" rx=\"2\"/>\n",
                boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h, fill, fill);
        /* Compact label below the box: just "D=k", count is implied. */
        double label_y = boxes[i].y + boxes[i].h + 7.0;
        const char *color = (D == 1) ? "#1d3b5e" : "#114421";
        int        size  = (D == 1) ? 7         : 8;
        fprintf(f, "  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\" "
                "font-family=\"sans-serif\" font-size=\"%d\" fill=\"%s\">"
                "D=%d</text>\n",
                boxes[i].x + boxes[i].w * 0.5, label_y, size, color, D);
        total_flat_leaves += boxes[i].n_leaves;
        if (D >= 2) total_d2plus_leaves += boxes[i].n_leaves;
    }

    /* Tree on top. */
    draw_tree(f, t->tree, t->tree_root, 0.0, w, 0, -1.0, /*in_box=*/0);

    /* Footer: stats.  Avg ops/leaf = mean over all leaves of
     * (#partitioning ancestors + 1 terminal op).  "weighted" =
     * freq-weighted = mean over decoded ELEMENTS, the runtime metric.
     * Smaller is better. */
    double avg_ops_uw = 0, avg_ops_w = 0;
    avg_ops_per_leaf(t, freq, &avg_ops_uw, &avg_ops_w);
    /* Footer sits below the lowest possible D=k label (which can be
     * up to ~14px below the deepest tree row). */
    double stat_y = h + 24;
    fprintf(f, "  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\" "
            "font-family=\"sans-serif\" font-size=\"9\" fill=\"#444\">"
            "%d leaves, max_len %d, %d box%s | "
            "D&gt;=2 covers %d/%d (%.0f%%) | "
            "D&gt;=1 covers %d/%d (%.0f%%) | "
            "ops/leaf %.2f (freq-weighted %.2f)</text>\n",
            w * 0.5, stat_y,
            n_leaves, max_len, nbox, nbox == 1 ? "" : "es",
            total_d2plus_leaves, n_leaves,
            n_leaves ? 100.0 * total_d2plus_leaves / n_leaves : 0.0,
            total_flat_leaves, n_leaves,
            n_leaves ? 100.0 * total_flat_leaves / n_leaves : 0.0,
            avg_ops_uw, avg_ops_w);

    fprintf(f, "</g>\n");
    *out_panel_h = h + 34;
    return w;
}

/* ---------- One distribution per call ---------- */

static double render_distribution(FILE *f, const char *name,
                                   const uint64_t *freq, double oy,
                                   double *out_height)
{
    pivco_table_t t_opt;
    if (pivco_build_table(bench_cfg(), freq, &t_opt) != PIVCO_OK) {
        fprintf(stderr, "build_table failed for %s\n", name);
        *out_height = 0;
        return 0;
    }
    pivco_table_t t_canon;
    build_canonical(t_opt.code_len, &t_canon);

    /* Distribution title. */
    int n_leaves = subtree_leaves(t_opt.tree, t_opt.tree_root);
    fprintf(f, "<g transform=\"translate(0, %.1f)\">\n", oy);
    fprintf(f, "  <text x=\"4\" y=\"14\" font-family=\"sans-serif\" "
            "font-size=\"13\" font-weight=\"bold\" fill=\"#000\">%s</text>\n",
            name);
    fprintf(f, "</g>\n");

    /* Extra vertical headroom above the canonical panel so the panel
     * title ("canonical"/"flat-aware") sits clear of the topmost
     * D=k flat-box labels. */
    double panel_oy = oy + 22 + 50;
    double w_top, w_bot;
    double h_top = 0, h_bot = 0;
    /* Stack the two panels vertically: canonical on top, flat-aware
     * below.  Aligning x makes leaf-position comparison straightforward. */
    double gap = 18.0;
    w_top = render_tree_panel(f, &t_canon, freq, "canonical",
                              8, panel_oy, &h_top);
    w_bot = render_tree_panel(f, &t_opt, freq, "flat-aware (production)",
                              8, panel_oy + h_top + gap, &h_bot);

    double w_max = w_top > w_bot ? w_top : w_bot;
    *out_height = 22 + 14 + h_top + gap + h_bot + 18;
    (void)n_leaves;
    return 8 + w_max + 8;
}

/* ---------- Graphviz DOT output ---------- */

/* Emit DOT nodes + edges for the subtree rooted at `node`.  Internal
 * nodes are tiny grey points; leaves are slightly larger black points.
 * No labels.  Optional `cluster_id` tags every node so we can pick
 * which ones go in flat-D clusters. */
static void dot_emit_subtree(FILE *f, const pivco_tree_node_t *tree,
                              int16_t node, const char *prefix)
{
    if (tree[node].symbol >= 0) {
        fprintf(f, "    %s_%d [shape=point, width=0.06, color=\"#222\"];\n",
                prefix, node);
    } else {
        fprintf(f, "    %s_%d [shape=point, width=0.03, color=\"#aaa\"];\n",
                prefix, node);
        fprintf(f, "    %s_%d -> %s_%d;\n", prefix, node, prefix, tree[node].left);
        fprintf(f, "    %s_%d -> %s_%d;\n", prefix, node, prefix, tree[node].right);
        dot_emit_subtree(f, tree, tree[node].left,  prefix);
        dot_emit_subtree(f, tree, tree[node].right, prefix);
    }
}

/* List every node (internal + leaf) inside the subtree rooted at `node`
 * into `ids[]`.  Used to populate flat-D cluster bodies. */
static void dot_collect_nodes(const pivco_tree_node_t *tree, int16_t node,
                               int16_t *ids, int *n)
{
    ids[(*n)++] = node;
    if (tree[node].symbol < 0) {
        dot_collect_nodes(tree, tree[node].left,  ids, n);
        dot_collect_nodes(tree, tree[node].right, ids, n);
    }
}

/* Walk the tree, emitting a DOT cluster for every maximal flat-D>=1
 * subtree.  Each cluster has a colored fill keyed on D, matching the
 * SVG palette. */
static void dot_emit_flat_clusters(FILE *f, const pivco_tree_node_t *tree,
                                    int16_t node, const char *prefix,
                                    int *cluster_seq)
{
    if (tree[node].symbol >= 0) return;
    int D = flat_D_at(tree, node);
    if (D >= 1) {
        const char *fill;
        if      (D == 1) fill = "#a8c3e0";
        else if (D >= 6) fill = "#1f7a3a";
        else if (D == 5) fill = "#3aa65b";
        else if (D == 4) fill = "#62c282";
        else if (D == 3) fill = "#9ad9b1";
        else             fill = "#cdebd6";
        int cid = (*cluster_seq)++;
        fprintf(f, "  subgraph cluster_%s_%d {\n", prefix, cid);
        fprintf(f, "    label=\"D=%d (%d)\";\n", D, 1 << D);
        fprintf(f, "    fontsize=8;\n");
        fprintf(f, "    style=\"filled,rounded\";\n");
        fprintf(f, "    fillcolor=\"%s\";\n", fill);
        fprintf(f, "    color=\"%s\";\n", fill);
        fprintf(f, "    margin=4;\n");
        int16_t ids[PIVCO_MAX_SYMBOLS * 2];
        int n_ids = 0;
        dot_collect_nodes(tree, node, ids, &n_ids);
        for (int i = 0; i < n_ids; i++) {
            fprintf(f, "    %s_%d;\n", prefix, ids[i]);
        }
        fprintf(f, "  }\n");
        return;
    }
    dot_emit_flat_clusters(f, tree, tree[node].left,  prefix, cluster_seq);
    dot_emit_flat_clusters(f, tree, tree[node].right, prefix, cluster_seq);
}

/* Render one named tree as a top-level DOT subgraph cluster (for the
 * "canonical vs flat-aware" side-by-side layout). */
static void dot_emit_tree_cluster(FILE *f, const pivco_table_t *t,
                                   const char *cluster_name,
                                   const char *title)
{
    fprintf(f, "subgraph cluster_%s {\n", cluster_name);
    fprintf(f, "  label=\"%s\";\n", title);
    fprintf(f, "  fontsize=11;\n");
    fprintf(f, "  style=rounded;\n");
    fprintf(f, "  color=\"#888\";\n");
    fprintf(f, "  margin=10;\n");

    /* Flat-D cluster subgraphs (must be inside the tree's cluster). */
    int seq = 0;
    dot_emit_flat_clusters(f, t->tree, t->tree_root, cluster_name, &seq);

    /* All nodes + edges (those in flat clusters are listed there too;
     * dot tolerates this — they end up positioned inside the cluster). */
    dot_emit_subtree(f, t->tree, t->tree_root, cluster_name);

    fprintf(f, "}\n");
}

static void render_distribution_dot(FILE *f, const char *name,
                                     const uint64_t *freq)
{
    pivco_table_t t_opt;
    if (pivco_build_table(bench_cfg(), freq, &t_opt) != PIVCO_OK) return;
    pivco_table_t t_canon;
    build_canonical(t_opt.code_len, &t_canon);

    fprintf(f, "digraph %s {\n", name);
    fprintf(f, "  rankdir=TB;\n");
    fprintf(f, "  bgcolor=\"#fafafa\";\n");
    fprintf(f, "  ranksep=0.18;\n");
    fprintf(f, "  nodesep=0.06;\n");
    fprintf(f, "  splines=line;\n");
    /* pack=true + packmode="array_t1" stacks subgraphs vertically
     * (1-column array, top-down); without this dot would lay them
     * side-by-side at the same rank. */
    fprintf(f, "  pack=true;\n");
    fprintf(f, "  packmode=\"array_t1\";\n");
    fprintf(f, "  edge [arrowhead=none, color=\"#888\", penwidth=0.5];\n");
    fprintf(f, "  node [label=\"\"];\n");
    fprintf(f, "  labelloc=\"t\";\n");
    fprintf(f, "  label=\"%s\";\n", name);
    fprintf(f, "  fontsize=14;\n");

    dot_emit_tree_cluster(f, &t_canon, "canon", "canonical");
    dot_emit_tree_cluster(f, &t_opt,   "opt",   "flat-aware (production)");

    fprintf(f, "}\n");
}

/* Compute the (width, height) a distribution's panel block will occupy,
 * without emitting any SVG.  Mirrors render_distribution's layout math
 * (canonical panel stacked vertically over flat-aware panel). */
static void measure_distribution(const uint64_t *freq, double *out_w,
                                  double *out_h)
{
    pivco_table_t t_opt;
    if (pivco_build_table(bench_cfg(), freq, &t_opt) != PIVCO_OK) {
        *out_w = 0; *out_h = 0; return;
    }
    int n_leaves = subtree_leaves(t_opt.tree, t_opt.tree_root);
    int max_len  = t_opt.max_len;
    double w_panel = n_leaves * LEAF_W;
    double h_panel = (double)max_len * LEVEL_H + LEAF_R * 2 + LEVEL_H * 0.5;
    double inter_panel_gap = 18.0;
    *out_w = 8 + w_panel + 8;
    *out_h = 22 + 50 + h_panel + 34 + inter_panel_gap + h_panel + 34 + 18;
}

int main(int argc, char **argv)
{
    bench_init();
    int n_dist = bench_num_distributions();
    int do_all = 0;
    int do_dot = 0;
    const char *which = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--all")) do_all = 1;
        else if (!strcmp(argv[i], "--dot")) do_dot = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: pivco_tree_viz [--dot] [distribution|--all] > out.{svg,dot}\n"
                    "  --dot      emit Graphviz DOT (pipe through `dot -Tsvg`)\n"
                    "  default    emit hand-rolled SVG with width-balanced layout\n"
                    "  available distributions:\n");
            for (int d = 0; d < n_dist; d++)
                fprintf(stderr, "    %s\n", bench_dist_name(d));
            return 0;
        } else {
            which = argv[i];
        }
    }

    if (!do_all && !which) which = "english";

    /* DOT mode: emit one digraph per distribution to stdout, separated by
     * blank lines (run through `gvpack` or `dot` per-graph as desired). */
    if (do_dot) {
        FILE *f = stdout;
        if (do_all) {
            for (int d = 0; d < n_dist; d++) {
                render_distribution_dot(f, bench_dist_name(d),
                                          bench_dist_freq(d));
                fprintf(f, "\n");
            }
        } else {
            for (int d = 0; d < n_dist; d++) {
                if (!strcmp(bench_dist_name(d), which)) {
                    render_distribution_dot(f, which, bench_dist_freq(d));
                    break;
                }
            }
        }
        return 0;
    }

    /* Pass 1: compute total canvas dimensions. */
    double max_w = 0;
    double total_h = 8;
    if (do_all) {
        for (int d = 0; d < n_dist; d++) {
            double w = 0, h = 0;
            measure_distribution(bench_dist_freq(d), &w, &h);
            if (w > max_w) max_w = w;
            total_h += h + 12;
        }
    } else {
        for (int d = 0; d < n_dist; d++) {
            if (!strcmp(bench_dist_name(d), which)) {
                double w = 0, h = 0;
                measure_distribution(bench_dist_freq(d), &w, &h);
                if (w > max_w) max_w = w;
                total_h += h;
                break;
            }
        }
    }

    /* Pass 2: emit SVG with computed dimensions. */
    FILE *f = stdout;
    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" "
            "viewBox=\"0 0 %.0f %.0f\" width=\"%.0f\" height=\"%.0f\">\n",
            max_w, total_h, max_w, total_h);
    fprintf(f, "<style>\n"
            "  text { dominant-baseline: middle; }\n"
            "</style>\n");
    fprintf(f, "<rect width=\"100%%\" height=\"100%%\" fill=\"#fafafa\"/>\n");

    double y = 8;
    if (do_all) {
        for (int d = 0; d < n_dist; d++) {
            const char *name = bench_dist_name(d);
            const uint64_t *freq = bench_dist_freq(d);
            double h = 0;
            render_distribution(f, name, freq, y, &h);
            y += h + 12;
        }
    } else {
        for (int d = 0; d < n_dist; d++) {
            if (!strcmp(bench_dist_name(d), which)) {
                double h = 0;
                render_distribution(f, which, bench_dist_freq(d), y, &h);
                y += h;
                break;
            }
        }
    }

    fprintf(f, "</svg>\n");
    return 0;
}
