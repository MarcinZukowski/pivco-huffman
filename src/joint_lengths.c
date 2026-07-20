/* ---------- Joint code-length / flat-shape optimization ----------
 *
 * pivco_huffman_build_table derives code lengths that minimize
 * compressed bits.  This pass additionally bends them -- at an
 * explicitly priced, guard-bounded cost in bits -- so the per-length
 * class counts land on round binary numbers and the OPTIMIZED chunk
 * decomposition in build_table_finish yields fewer, larger flat
 * subtrees and fewer merge passes.  Encoder side only: the wire
 * carries plain code lengths, so ANY decoder reads the output and
 * both sides rebuild identical tables.
 *
 * Chunk model: choosing lengths IS choosing at most one chunk per
 * (level L <= PIVCO_MAX_CODE_LEN, flat depth b <= min(8, L)) -- a
 * chunk holds 2^b symbols at length L inside a depth-b flat, so each
 * of its symbols' occurrences costs L bits and L - b merge passes.
 * Objective:
 *     J = sum_s n_s * (L_s + lambda*(L_s - b_s + kappa[b_s]))
 *         + lambda * gamma * blocks * records
 * subject to chunk-root Kraft equality.  For a fixed chunk multiset
 * the optimal symbol assignment deals freq-sorted symbols into
 * cost-sorted chunks (rearrangement inequality), which turns the
 * solve into a DP over (symbols placed, open slots); lambda = 0
 * degenerates to the Huffman baseline, so the result can only improve
 * in-model, and a kind-aware time model guards against out-of-model
 * regressions.  When the slot DP's validity condition fails (lambda >
 * 1/7 under zero kappa), the baseline is kept -- the same contract as
 * a guard reject.
 *
 * The effort modes (pivco_effort_t) pick the solve tier: BALANCED
 * runs a coarse grouped solve, FASTER_DECOMPRESS the auto tier,
 * FASTEST_DECOMPRESS the exact DP.  Grouping by g = 2^G solves the
 * identical problem G levels shallower (a group of g freq-sorted
 * symbols at real level L is a depth-G flat) at 4^G fewer states;
 * near-optimal solutions are dense enough that g = 2 loses ~0.13% of
 * J on average, g = 4 ~0.25% (measured on LZ-literal data), and the
 * guard still rejects any bad case.
 */

#include "pivco_huffman.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The guard's class-count bins index `length & 15`. */
_Static_assert(PIVCO_MAX_CODE_LEN <= 15,
               "joint pass assumes code lengths fit 4 bits");

/* Flat-depth cap: a depth-b flat holds 2^b symbols and the per-table
 * flat_code_to_sym pool is PIVCO_MAX_SYMBOLS entries, so b <= 8. */
#define JL_MAX_FLAT 8

/* Upper bound on the chunks of one Kraft-complete decomposition (per
 * length, one chunk per set bit of the class count).  The proven max
 * at max_len 11 is 33; 64 covers any PIVCO_MAX_CODE_LEN up to 15. */
#define JL_MAX_CHUNKS 64

/* Max compact DP row width: j <= 128 at sigma = 256, padded to x4. */
#define JL_WMAX 132

/* ---------- effort knob (process-global, same pattern as the FSE
 * toggle).  Read once per pivco_huffman_build_table call. ---------- */

static pivco_effort_t g_effort = PIVCO_EFFORT_BALANCED;

void pivco_huffman_set_effort(pivco_effort_t effort)
{
    g_effort = effort;
}

pivco_effort_t pivco_huffman_get_effort(void)
{
    return g_effort;
}

/* ---------- model knobs ----------
 *
 * Internal for now; the effort modes only vary `gran`.  gamma and
 * kappa are decode-cost model constants in merge element-pass units:
 * gamma prices one schedule record per PIVCO_BLOCK_SIZE-symbol block
 * (dispatch + wire header), kappa[b] prices one symbol of a depth-b
 * flat kernel (all-zero models the kernels free), mu_cst prices a
 * lone-leaf merge relative to a full partition.  The defaults were
 * tuned on windowed LZ-literal workloads (Apple M-class);
 * re-measuring them against this decoder's kernels is future tuning,
 * not correctness -- the adoption guard prices both sides with the
 * same model. */
typedef struct {
    double lambda;      /* bits one merge pass is worth */
    int    gran;        /* solve tier: 0 auto (exact DP to 64 symbols,
                           then grouped), 1 exact DP, 2/4/8 fixed
                           grouping, -1 coarse auto (one grouping step
                           chunkier: most of auto's decode win at a
                           fraction of its solve cost) */
    double guard_bits;  /* adopt only if bits <= guard_bits * baseline */
    double guard_time;  /* ... and time <= guard_time * baseline */
    double gamma;       /* decode cost per schedule record per block */
    double kappa[JL_MAX_FLAT + 1];  /* flat-kernel cost/symbol at depth b */
    double mu_cst;      /* lone-leaf merge cost relative to a full merge */
} joint_params_t;

static const joint_params_t joint_defaults = {
    0.1,            /* lambda */
    0,              /* gran (set per effort mode) */
    1.015,          /* guard_bits */
    0.90,           /* guard_time */
    170.0,          /* gamma */
    {0},            /* kappa */
    1.0,            /* mu_cst */
};

/* ---------- kind-aware decode-time model (the adoption guard) ----------
 *
 * The per-occurrence model above prices every merge alike, but the
 * decoder's merges differ by KIND: a merge with a lone-leaf child uses
 * the cheap cst kernels, a merge of two internal streams pays the full
 * partition.  Tree arrangement is deterministic from the chunk
 * multiset (build_table_finish depth-sorts chunk roots and assigns
 * canonical prefixes), so we simulate the skeleton exactly and price
 * each node by kind.  Used on BOTH sides of the guard's comparison;
 * the DP keeps its separable search cost (the guard is where
 * mispricing must not survive). */

/* One realized chunk: depth = tree depth of the chunk root, bit = flat
 * depth b (2^bit symbols), weight = total occurrence count -- the same
 * (depth, bit) naming as build_table_finish's chunk_t. */
typedef struct { uint8_t depth, bit; double weight; } jl_chunk_t;

/* Subtree at depth d spanning chunks ch[*i..): consumes them, returns
 * the subtree's decode-time units and its weight; *kind reports what
 * the parent sees (0 = lone leaf, 1 = internal).
 *
 * Iterative, explicit frame stack: phase 0 frames are waiting on their
 * left child, phase 1 on their right. */
static double sim_subtree_time(const jl_chunk_t *ch, int n, int *i, int d,
                               const joint_params_t *jp, const double *kap,
                               int *recs, double *weight_out, int *kind)
{
    struct {
        double tl, wl;
        int kl;
        uint8_t d, phase;
    } stk[PIVCO_MAX_CODE_LEN + 2];
    int sp = 0;
    double rt, rw;
    int rkind;

enter:
    if (d > PIVCO_MAX_CODE_LEN) {   /* non-tiling multiset: cut the walk;
                                     * the caller's i != n check reports
                                     * failure.  Unreachable from the
                                     * in-file callers (their multisets
                                     * are Kraft-exact by construction)
                                     * -- pure stack-safety. */
        rt = 0.0; rw = 0; rkind = 1;
        goto unwind;
    }
    if (*i < n && ch[*i].depth == d) {
        const jl_chunk_t *c = &ch[(*i)++];
        rw = c->weight;
        if (c->bit == 0) { rkind = 0; rt = 0.0; goto unwind; }
        rkind = 1;
        (*recs)++;                                 /* pair/flat record */
        rt = c->weight * kap[c->bit];
        goto unwind;
    }
    stk[sp].d = (uint8_t)d;
    stk[sp].phase = 0;
    sp++;
    (*recs)++;                                     /* merge record */
    d++;
    goto enter;

unwind:
    if (sp == 0) {
        *weight_out = rw;
        *kind = rkind;
        return rt;
    }
    if (stk[sp - 1].phase == 0) {                  /* left child done */
        stk[sp - 1].tl = rt;
        stk[sp - 1].wl = rw;
        stk[sp - 1].kl = rkind;
        stk[sp - 1].phase = 1;
        d = stk[sp - 1].d + 1;
        goto enter;                                /* right child */
    }
    {                                              /* right child done */
        const double wl = stk[sp - 1].wl, wr = rw;
        const double tl = stk[sp - 1].tl, tr = rt;
        const int kl = stk[sp - 1].kl, kr = rkind;
        const double w = wl + wr;
        double t;
        if (kl == 0 || kr == 0)
            t = w * jp->mu_cst;               /* one lone leaf: cst merge */
        else
            t = w;                            /* full partition */
        rt = t + tl + tr;
        rw = w;
        rkind = 1;
        sp--;
        goto unwind;
    }
}

/* Kind-aware decode time for a chunk list (any order; sorted here into
 * the order build_table_finish realizes: depth asc, bit asc.  The
 * builder stable-sorts its L-ascending generation by depth only, and
 * equal-depth chunks from lower classes have smaller bit, so
 * depth-then-bit ascending IS that order -- and since a class emits
 * each bit at most once, (depth, bit) is unique and the sort is
 * total). */
static double chunk_list_time(jl_chunk_t *ch, int n,
                              const joint_params_t *jp, const double *kap,
                              double total_weight)
{
    for (int i = 1; i < n; i++) {
        jl_chunk_t c = ch[i];
        int j = i - 1;
        while (j >= 0 && (ch[j].depth > c.depth ||
                          (ch[j].depth == c.depth && ch[j].bit > c.bit))) {
            ch[j + 1] = ch[j];
            j--;
        }
        ch[j + 1] = c;
    }
    int i = 0, kind, recs = 0;
    double w;
    double t = sim_subtree_time(ch, n, &i, 0, jp, kap, &recs, &w, &kind);
    if (i != n) return -1.0;    /* malformed multiset (cannot happen) */
    if (jp->gamma > 0) {        /* per-record fixed cost x blocks */
        double blocks = ceil(total_weight / (double)PIVCO_BLOCK_SIZE);
        if (blocks < 1) blocks = 1;
        t += jp->gamma * (double)recs * blocks;
    }
    return t;
}

/* ---------- slot-ledger DP (exact for lambda <= 1/7) ----------
 *
 * A state is (k symbols placed, s open slots at the current level);
 * Kraft EQUALITY forces s <= sigma - k at every level.  Levels are
 * processed ascending, chunk types within a level in cost order; that
 * equals GLOBAL chunk-cost order -- the sorted-matching exactness
 * requirement -- iff dp_take_order's spread bound holds (kappa = 0
 * recovers the classic lambda <= 1/7).  Three structural facts keep
 * the walk small and L1-resident:
 *
 * DIAGONALS.  A take (k, s) -> (k + 2^b, s - 2^b) preserves t = k + s,
 * so within a level the DP decomposes into independent diagonals.
 * Stored diagonal-major, all take sweeps of a level run over one short
 * row; the plane is traversed once per level (the doubling).
 *
 * PARITY.  Level-entry states have even s (they come from the doubling
 * s' = 2s) and takes with b >= 1 preserve s-parity, so the live
 * lattice is k == t (mod 2): compact index j = (k - (t&1))/2 halves
 * each row.  b = 0 -- the only parity flip, always last in the level's
 * cost order -- is folded into the doubling (an odd-s cell's unique
 * source is its even-lattice predecessor plus one lone leaf) and
 * reconstructed from s-parity at backtrack.  The deepest level never
 * takes b = 0: entry s is even and the terminal needs takes summing to
 * s exactly.
 *
 * CAPACITY BAND.  A state at level L can place at most s * 2^h more
 * symbols (h = levels below), so sigma - k <= (t - k) << h is
 * necessary -- and met by every completing trajectory, making the
 * prune exact.  Feasibility is preserved cell-to-cell by takes and by
 * the doubling, so pruned -- hence stale -- cells are never read.
 *
 * Terminal: (k = sigma, s = 0) after the deepest level.  Per-level u16
 * pick rows (bits 1..8; bit 0 is implicit in parity) are archived per
 * diagonal for backtrack. */

/* Largest compact index j on diagonal t whose k = 2j + (t&1) can still
 * feed sigma - k leaves through (t - k) slots h levels above the
 * bottom; -1 if the whole row is infeasible. */
static int dp_row_cap(int t, int h, int sigma)
{
    const int p = t & 1;
    int kcap;
    if (h == 0) {
        kcap = t;                     /* t == sigma: all k feasible */
    } else {
        const int num = (t << h) - sigma;
        if (num < 0) return -1;
        kcap = num / ((1 << h) - 1);
        if (kcap > t) kcap = t;
    }
    if (kcap < p) return -1;
    return (kcap - p) >> 1;
}

/* Within-level sweep/deal order under kernel costs.  cost(L, b) =
 * L(1+lam) + g(b) with g(b) = lam*(kap[b] - b): the within-level cost
 * order is L-independent, so one sorted order serves every level.
 * Exactness of the slot DP needs (a) cross-level monotonicity:
 * spread(g) <= 1 + lam (kappa = 0 recovers lam <= 1/7), and (b) b = 0
 * dearest within the level (the parity fold runs it last).  Fills
 * border[0..*nb) with b = 1..bcap by ascending g; returns 1 iff both
 * hold (on 0 the caller keeps the baseline). */
static int dp_take_order(double lam, const double *kap, int bcap,
                         int border[JL_MAX_FLAT], int *nb)
{
    double g[JL_MAX_FLAT + 1];
    double gmin = 0, gmax = 0;
    for (int b = 0; b <= bcap; b++) {
        g[b] = lam * (kap[b] - (double)b);
        if (b == 0 || g[b] < gmin) gmin = g[b];
        if (b == 0 || g[b] > gmax) gmax = g[b];
    }
    if (gmax - gmin > (1.0 + lam) * (1.0 - 1e-9)) return 0;
    int n = 0;
    for (int b = 1; b <= bcap; b++) {
        if (g[b] > g[0] + 1e-12) return 0;   /* b0 must stay dearest */
        int i = n++;
        while (i > 0 && (g[border[i - 1]] > g[b]
                         || (g[border[i - 1]] == g[b] && border[i - 1] < b))) {
            border[i] = border[i - 1];
            i--;
        }
        border[i] = b;                       /* ties: larger b first */
    }
    *nb = n;
    return 1;
}

/* lmax/bcap parameterize the level range and flat-depth cap so the
 * same solver runs the exact problem (PIVCO_MAX_CODE_LEN, 8) and the
 * 2^G-grouped coarse problem (PIVCO_MAX_CODE_LEN - G, 8 - G): a group
 * of 2^G sorted symbols at real level L is a depth-G flat, so the
 * coarse problem is this problem shifted by G with an identical cost
 * form.  tc0/tc1: per-take J constants (lambda * gamma * blocks *
 * records added) for b = 0 and b >= 1 takes.  On success fills
 * out_BL[L] with the level's takes as a b-bitmask and returns the
 * optimal J; returns -1.0 on an order-condition failure or OOM. */
static double solve_slot_dp(const double *P, int sigma, double lam,
                            int lmax, int bcap, const double *kap,
                            double tc0, double tc1,
                            uint16_t out_BL[PIVCO_MAX_CODE_LEN + 1])
{
    int border[JL_MAX_FLAT], nb;
    if (!dp_take_order(lam, kap, bcap, border, &nb))
        return -1.0;
    const int W = (((sigma >> 1) + 2) + 3) & ~3;   /* row width, x4 pad */
    const size_t plane = (size_t)(sigma + 1) * (size_t)W;
    /* One f32 cost plane + lmax u16 backtrack planes (~880 KB for the
     * exact solve at sigma = 256; the grouped tiers use a fraction). */
    uint8_t *buf = (uint8_t *)malloc(plane * (4 + 2 * (size_t)lmax));
    if (!buf) return -1.0;
    float *cost = (float *)buf;
    uint16_t *arch = (uint16_t *)(cost + plane);
    float dPt[2][JL_MAX_FLAT + 1][JL_WMAX];   /* [t&1][b][j]: P[k+2^b]-P[k] */
    float dP0[PIVCO_MAX_SYMBOLS + 1];         /* P[k] - P[k-1] */

    for (int p = 0; p < 2; p++)
        for (int b = 1; b <= bcap; b++) {
            const int cnk = 1 << b;
            for (int j = 0; j < W; j++) {
                const int k = 2 * j + p;
                dPt[p][b][j] = k + cnk <= sigma
                             ? (float)(P[k + cnk] - P[k]) : 0.0f;
            }
        }
    dP0[0] = 0.0f;
    for (int k = 1; k <= sigma; k++) dP0[k] = (float)(P[k] - P[k - 1]);

    /* Per-level diagonal band: t <= min(2^L, sigma) states exist, and
     * the capacity band needs t >= ceil(sigma / 2^h). */
    int tlo[PIVCO_MAX_CODE_LEN + 1], thi[PIVCO_MAX_CODE_LEN + 1];
    for (int L = 1; L <= lmax; L++) {
        const int h = lmax - L;
        thi[L] = (1 << L) > sigma ? sigma : (1 << L);
        tlo[L] = (sigma + (1 << h) - 1) >> h;
        if (tlo[L] < 1) tlo[L] = 1;
    }

    /* Lazy init: the doubling that produces a level writes each row up
     * to its cap, so only the level-1 band rows need priming. */
    for (int t = tlo[1]; t <= thi[1]; t++)
        for (int j = 0; j < W; j++) cost[(size_t)t * W + j] = INFINITY;
    cost[2 * W + 0] = 0.0f;      /* level-1 entry: k = 0, s = 2, t = 2 */

    for (int L = 1; L <= lmax; L++) {
        const int h = lmax - L;
        const int bmax = L < bcap ? L : bcap;
        uint16_t *archL = arch + (size_t)(L - 1) * plane;
        for (int t = tlo[L]; t <= thi[L]; t++) {
            const int p = t & 1;
            const int jcap = dp_row_cap(t, h, sigma);
            if (jcap < 0) continue;
            float *row = cost + (size_t)t * W;
            uint16_t *prow = archL + (size_t)t * W;   /* picks, archived
                                                       * in place */
            memset(prow, 0, (size_t)(jcap + 1) * sizeof(uint16_t));
            for (int oi = 0; oi < nb; oi++) {
                const int b = border[oi];
                if (b > bmax) continue;
                const int jstep = 1 << (b - 1);       /* = 2^b slots / 2 */
                const int jhi = jcap - jstep;         /* dest j <= jcap  */
                if (jhi < 0) continue;
                const float a = (float)((double)L
                                        + lam * ((double)(L - b) + kap[b]));
                const float tc = (float)tc1;
                const float *dpb = dPt[p][b];
                /* 0/1 in-place: dest j + jstep > src j, so iterate j
                 * descending -- a written dest is never re-read as a
                 * source for the same chunk type. */
                for (int j = jhi; j >= 0; j--) {
                    const float v = row[j];
                    if (!(v < INFINITY)) continue;
                    const float cand = v + a * dpb[j] + tc;
                    if (cand < row[j + jstep]) {
                        row[j + jstep] = cand;
                        prow[j + jstep] = (uint16_t)(prow[j] | (1u << b));
                    }
                }
            }
        }
        if (L == lmax) break;
        /* Doubling s' = 2s with the b = 0 take folded in.  Dest cell
         * (t', k) has the unique source (t = (t'+k)/2, k): even-lattice
         * there if k == t (mod 2), else the odd-s product of a lone
         * leaf taken at level L from (t, k-1).  In place, t' and j'
         * descending: sources live on rows <= t', and the single
         * same-row read (t = t', only at k = t') happens before its
         * cell is overwritten.
         *
         * Branchless: on dest row t' the source diagonal is t = t0 + j'
         * (t0 = (t'+p')/2), so the level-L band check hoists to a
         * j'-range, and the source parity d = (t^k)&1 alternates with
         * j' -- two constant-stride subloops with the unified source
         * index (k - d - (t&1))/2.  The subloop containing the top cell
         * runs first (it holds the only same-row read). */
        const float a0 = (float)((double)L * (1.0 + lam) + lam * kap[0]);
        const float tcz = (float)tc0;
        for (int tp = thi[L + 1]; tp >= tlo[L + 1]; tp--) {
            const int pp = tp & 1;
            const int jcap2 = dp_row_cap(tp, h - 1, sigma);
            if (jcap2 < 0) continue;
            float *nrow = cost + (size_t)tp * W;
            const int t0 = (tp + pp) >> 1;
            int jlo = tlo[L] - t0; if (jlo < 0) jlo = 0;
            int jhi2 = thi[L] - t0; if (jhi2 > jcap2) jhi2 = jcap2;
            for (int jp2 = jcap2; jp2 > jhi2; jp2--) nrow[jp2] = INFINITY;
            for (int jp2 = jlo - 1; jp2 >= 0; jp2--) nrow[jp2] = INFINITY;
            for (int half = 0; half < 2; half++) {
                int jp2 = jhi2 - half;
                if (jp2 < jlo) continue;
                const int k1 = 2 * jp2 + pp;
                const int t1 = t0 + jp2;
                const int d = (t1 ^ k1) & 1;
                /* Signed source index into cost[]: at the odd branch's
                 * degenerate k = 0 cell the offset (k1 - d - (t1 & 1))
                 * would go negative, and casting it to size_t would
                 * wrap the pointer backwards (UB); keeping the whole
                 * index signed and indexing cost[]/dP0[] avoids forming
                 * any out-of-array pointer (a negative sentinel index
                 * is just an integer). */
                ptrdiff_t si = (ptrdiff_t)t1 * W + ((k1 - d - (t1 & 1)) >> 1);
                if (d == 0) {
                    for (; jp2 >= jlo; jp2 -= 2, si -= 2 * W + 2)
                        nrow[jp2] = cost[si];
                } else {
                    /* k = 0 has no lone-leaf predecessor: if this
                     * chain reaches cell (jp2 = 0, k = 0), stop above
                     * it and mark it unreachable. */
                    int floor2 = jlo, patch0 = 0;
                    if (pp == 0 && (jp2 & 1) == 0 && jlo == 0) {
                        floor2 = 2;
                        patch0 = 1;
                    }
                    ptrdiff_t di = k1;
                    for (; jp2 >= floor2; jp2 -= 2, si -= 2 * W + 2, di -= 4)
                        nrow[jp2] = cost[si] + a0 * dP0[di] + tcz;
                    if (patch0)
                        nrow[0] = INFINITY;
                }
            }
        }
    }

    double J = cost[(size_t)sigma * W + (size_t)((sigma - (sigma & 1)) >> 1)];
    if (J < INFINITY) {
        /* Backtrack: invert each level's transition; odd end-of-level
         * s means the folded b0 was taken there -- recover its bits
         * from the even-lattice predecessor and set bit 0. */
        int k = sigma, s = 0;
        for (int L = lmax; L >= 1; L--) {
            const int t = k + s;
            const int p = t & 1;
            const uint16_t *arow = arch + (size_t)(L - 1) * plane
                                        + (size_t)t * W;
            uint16_t BL;
            if ((k ^ t) & 1)
                BL = (uint16_t)(arow[(k - 1 - p) >> 1] | 1u);
            else
                BL = arow[(k - p) >> 1];
            out_BL[L] = BL;
            int cL = 0;
            for (int b = 0; b <= JL_MAX_FLAT; b++)
                if (BL & (1 << b)) cL += 1 << b;
            k -= cL;
            s += cL;                  /* slots at level L entry (even) */
            if (L > 1) s >>= 1;       /* pre-doubling slots left       */
        }
    } else {
        J = -1.0;
    }
    free(buf);
    return J;
}

/* Realized chunk lists (depth, bit, weight) for BOTH lens images --
 * the incoming baseline lb and the deal's candidate lc -- priced as
 * the tables build_table_finish will build: within a class the builder
 * takes symbols in ascending symbol order and splits the count
 * largest-set-bit first, so chunk membership -- and with it each
 * chunk's weight -- is a function of the lengths alone, NOT of the
 * deal that chose them.  Pricing the guard on realized weights (both
 * sides) keeps it honest: the DP's sorted matching of heavy symbols to
 * cheap chunks is a search relaxation the canonical rebuild does not
 * reproduce.
 *
 * cnt* are per-class symbol counts, supplied by the caller; bins 0 and
 * PIVCO_MAX_CODE_LEN+1..15 are trash (absent / garbage lengths --
 * internal lengths never exceed the cap).  One fused ascending
 * 256-symbol sweep deals every class's chunk cursor on both sides
 * simultaneously -- ascending symbol order IS the builder's membership
 * order -- with absent symbols draining into a zero-weight dummy chunk
 * via the &15 trash bins, branchlessly (their weight contribution is
 * 0).  Frequencies narrow to u32 as in the caller's leaf collection;
 * u64 accumulators keep the sums exact with integer adds. */
static void realized_chunk_weights(const uint8_t *lb, const uint8_t *lc,
                                   const int cntb[16], const int cntc[16],
                                   const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                   jl_chunk_t chb[JL_MAX_CHUNKS], int *nb_out,
                                   jl_chunk_t chc[JL_MAX_CHUNKS], int *nc_out)
{
    int curb[16], curc[16], leftb[16], leftc[16], endb[16], endc[16];
    int nb = 0, nc = 0;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        curb[L] = nb;
        curc[L] = nc;
        for (int b = JL_MAX_FLAT; b >= 0; b--) {
            if (cntb[L] & (1 << b)) {
                chb[nb].depth = (uint8_t)(b ? L - b : L);
                chb[nb].bit = (uint8_t)b;
                nb++;
            }
            if (cntc[L] & (1 << b)) {
                chc[nc].depth = (uint8_t)(b ? L - b : L);
                chc[nc].bit = (uint8_t)b;
                nc++;
            }
        }
        endb[L]  = nb;
        endc[L]  = nc;
        leftb[L] = curb[L] < nb ? 1 << chb[curb[L]].bit : 1;
        leftc[L] = curc[L] < nc ? 1 << chc[curc[L]].bit : 1;
    }
    for (int t = 0; t < 16; t++)
        if (t == 0 || t > PIVCO_MAX_CODE_LEN) {
            curb[t] = nb; endb[t] = nb; leftb[t] = 0x7fffffff;
            curc[t] = nc; endc[t] = nc; leftc[t] = 0x7fffffff;
        }
    uint64_t wb[JL_MAX_CHUNKS + 1] = {0}, wc[JL_MAX_CHUNKS + 1] = {0};
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        const uint32_t f = (uint32_t)freq[s];
        const int Lb = lb[s] & 15, Lc = lc[s] & 15;
        wb[curb[Lb]] += f;
        if (--leftb[Lb] == 0 && ++curb[Lb] < endb[Lb])
            leftb[Lb] = 1 << chb[curb[Lb]].bit;
        wc[curc[Lc]] += f;
        if (--leftc[Lc] == 0 && ++curc[Lc] < endc[Lc])
            leftc[Lc] = 1 << chc[curc[Lc]].bit;
    }
    for (int i = 0; i < nb; i++) chb[i].weight = (double)wb[i];
    for (int i = 0; i < nc; i++) chc[i].weight = (double)wc[i];
    *nb_out = nb;
    *nc_out = nc;
}

/* ---------- driver ---------- */

typedef struct { uint32_t freq; uint16_t sym; } jl_leaf_t;

/* Stable ascending (freq, sym) sort.  Leaves arrive in symbol order
 * (the stable seed), so a stable freq sort IS the (freq, sym) order.
 * Small alphabets insertion-sort; larger ones take an LSD radix over
 * only the frequency bytes that VARY across the set (vary = OR ^ AND
 * of all freqs, a free by-product of the caller's scan) -- a constant
 * byte is an identity pass, so it is skipped outright. */
static void sort_leaves(jl_leaf_t *leaf, int n, uint32_t vary)
{
    int i, j;
    if (n <= 40) {
        for (i = 1; i < n; i++) {
            jl_leaf_t cur = leaf[i];
            for (j = i - 1; j >= 0 && leaf[j].freq > cur.freq; j--)
                leaf[j + 1] = leaf[j];
            leaf[j + 1] = cur;
        }
        return;
    }
    int shift[4], npass = 0;
    for (int b = 0; b < 32; b += 8)
        if ((vary >> b) & 0xFF) shift[npass++] = b;
    if (npass == 0) return;                    /* all frequencies equal */
    /* u8 bins cannot go wrong at n <= 256: a bin could only reach 256
     * if every leaf shared that byte, but such a plane does not vary
     * and is skipped, so varying-plane bins are <= 255.  A prefix that
     * wraps to 0 is only stored for an empty bin (never indexed), and
     * the final in-scatter increment that wraps is never read again. */
    uint8_t cnt[4][256];
    memset(cnt, 0, (size_t)npass * sizeof(cnt[0]));
    for (i = 0; i < n; i++)                    /* all planes in one pass */
        for (int p = 0; p < npass; p++)
            cnt[p][(leaf[i].freq >> shift[p]) & 0xFF]++;
    jl_leaf_t tmp[PIVCO_MAX_SYMBOLS], *src = leaf, *dst = tmp;
    for (int p = 0; p < npass; p++) {
        unsigned sum = 0;
        for (int k = 0; k < 256; k++) {
            unsigned c = cnt[p][k];
            cnt[p][k] = (uint8_t)sum;
            sum += c;
        }
        for (i = 0; i < n; i++)
            dst[cnt[p][(src[i].freq >> shift[p]) & 0xFF]++] = src[i];
        jl_leaf_t *t = src; src = dst; dst = t;
    }
    if (src != leaf) memcpy(leaf, src, (size_t)n * sizeof(*leaf));
}

/* Core over the ascending-sorted leaf array (reversed in place here;
 * ghost-padding may append).  Overwrites lengths[] on adoption; any
 * reject leaves them untouched. */
static int joint_core(jl_leaf_t *sf, int sigma,
                      const uint64_t freq[PIVCO_MAX_SYMBOLS],
                      uint8_t lengths[PIVCO_MAX_SYMBOLS],
                      const joint_params_t *jp)
{
    const double lam = jp->lambda;
    const double *kap = jp->kappa;
    if (sigma < 2) return -1;

    for (int i = 0; i < sigma / 2; i++) {  /* ascending -> descending */
        jl_leaf_t tmp = sf[i];
        sf[i] = sf[sigma - 1 - i];
        sf[sigma - 1 - i] = tmp;
    }
    double P[PIVCO_MAX_SYMBOLS + 1];
    P[0] = 0.0;
    for (int i = 0; i < sigma; i++) P[i + 1] = P[i] + (double)sf[i].freq;

    /* Baseline class counts (the guard itself is priced after the
     * deal, one fused pass covering both sides; internal lengths are
     * <= PIVCO_MAX_CODE_LEN so the &15 bins are exact, with 0
     * collecting absent symbols). */
    int cntb[16] = {0};
    for (int i = 0; i < sigma; i++)
        cntb[lengths[sf[i].sym] & 15]++;

    /* Per-take fixed-cost constants: lambda * gamma * blocks, one
     * record for b = 0 takes (the skeleton merge above the leaf), two
     * for deeper chunks (the flat record + its stitch merge). */
    double blocks = ceil(P[sigma] / (double)PIVCO_BLOCK_SIZE);
    if (blocks < 1) blocks = 1;
    const double tc0 = lam * jp->gamma * blocks;
    const double tc1 = 2.0 * tc0;

    /* Tier resolve.  Granularity g = 2^G groups the freq-sorted
     * symbols by g and solves the identical problem G levels shallower
     * (see the header comment); sigma is ghost-padded to a multiple of
     * g with zero-frequency unused byte values -- real leaves the
     * encoder never emits; there are always enough since
     * sigma % g != 0 implies sigma < 256. */
    int gran = jp->gran;
    if (gran != -1 && gran != 1 && gran != 2 && gran != 4 && gran != 8)
        gran = 0;
    if (gran == 0)        /* auto: keep the solve cheap at every sigma */
        gran = sigma <= 64 ? 1 : sigma <= 128 ? 2 : 4;
    else if (gran == -1)  /* coarse auto: one granularity step chunkier */
        gran = sigma <= 64 ? 2 : sigma <= 128 ? 4 : 8;
    int obuf[JL_MAX_FLAT], on;
    if (gran > 1 && (sigma < 8 * gran
                     || !dp_take_order(lam,
                                       kap + (gran == 8 ? 3 : gran == 4 ? 2 : 1),
                                       JL_MAX_FLAT - (gran == 8 ? 3 : gran == 4 ? 2 : 1),
                                       obuf, &on)))
        gran = 1;
    const int glog = gran == 8 ? 3 : gran == 4 ? 2 : gran == 2 ? 1 : 0;
    int sigma_pad = sigma;
    if (glog) {
        const int pad = (gran - (sigma % gran)) % gran;
        int added = 0;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS && added < pad; s++)
            if (!freq[s]) {
                sf[sigma_pad].freq = 0;
                sf[sigma_pad].sym = (uint16_t)s;
                P[sigma_pad + 1] = P[sigma];
                sigma_pad++;
                added++;
            }
        if (added < pad) return -1;      /* unreachable: pad <= 256-sigma */
    }

    uint16_t BL[PIVCO_MAX_CODE_LEN + 1] = {0};
    if (glog) {
        double Pg[PIVCO_MAX_SYMBOLS / 2 + 2];
        const int sp = sigma_pad / gran;
        for (int i = 0; i <= sp; i++) Pg[i] = P[i * gran];
        uint16_t BLc[PIVCO_MAX_CODE_LEN + 1] = {0};
        /* kap + glog: local b' prices the real depth b' + glog; a
         * grouped b' = 0 take is a real 2^glog flat, hence tc1 twice */
        if (solve_slot_dp(Pg, sp, lam, PIVCO_MAX_CODE_LEN - glog,
                          JL_MAX_FLAT - glog, kap + glog, tc1, tc1, BLc) < 0)
            return -1;
        for (int L = 1; L <= PIVCO_MAX_CODE_LEN - glog; L++)
            BL[L + glog] = (uint16_t)(BLc[L] << glog);
    } else if (solve_slot_dp(P, sigma, lam, PIVCO_MAX_CODE_LEN, JL_MAX_FLAT,
                             kap, tc0, tc1, BL) < 0) {
        return -1;      /* order condition failed (lambda > 1/7) or OOM */
    }

    /* Collect the chosen chunks in GLOBAL per-occurrence cost order --
     * under kappa the plain "L ascending, b descending" deal is no
     * longer the cost order, and the sorted matching the solver
     * assumes must be the assignment we actually realize. */
    struct { double cost; uint8_t L, b; uint16_t size; } chunks[JL_MAX_CHUNKS];
    int nchunks = 0;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++)
        for (int b = 0; b <= JL_MAX_FLAT; b++)
            if (BL[L] & (1 << b)) {
                double c = (double)L + lam * ((double)(L - b) + kap[b]);
                int i = nchunks++;
                while (i > 0 && (chunks[i - 1].cost > c
                                 || (chunks[i - 1].cost == c
                                     && (chunks[i - 1].L > L
                                         || (chunks[i - 1].L == L
                                             && chunks[i - 1].b < b))))) {
                    chunks[i] = chunks[i - 1];
                    i--;
                }
                chunks[i].cost = c;
                chunks[i].L    = (uint8_t)L;
                chunks[i].b    = (uint8_t)b;
                chunks[i].size = (uint16_t)(1 << b);
            }

    /* Deal freq-sorted symbols to the chunks in that same order, into
     * a CANDIDATE lens image (the caller's lengths hold the baseline
     * until the guard passes).  Ghosts (sorted last) take the dearest
     * chunks: unused byte values receive real codes the encoder never
     * emits.  dp_bits is exact off the deal -- bits depend only on
     * per-symbol length, which the rebuild preserves. */
    uint8_t cand[PIVCO_MAX_SYMBOLS];
    double dp_bits = 0, dp_time;
    memcpy(cand, lengths, PIVCO_MAX_SYMBOLS);
    {
        int cur = 0;
        for (int i = 0; i < nchunks; i++) {
            dp_bits += (P[cur + chunks[i].size] - P[cur]) * chunks[i].L;
            for (int j = 0; j < chunks[i].size; j++)
                cand[sf[cur++].sym] = chunks[i].L;
        }
        if (cur != sigma_pad) return -1;
    }
    /* Apply the adoption guard on the tables the decoder will actually
     * build: the deal's heavy-to-cheap matching is not realizable (the
     * rebuild redistributes a class's symbols over its chunks in
     * symbol order), so both sides price the realized weights.  Ghost
     * chunks carry zero weight, so real symbols are scored exactly. */
    double base_bits = 0, base_time;
    {
        jl_chunk_t chb[JL_MAX_CHUNKS], chc[JL_MAX_CHUNKS];
        int cntc[16] = {0}, nb, nc;
        for (int i = 0; i < nchunks; i++)
            cntc[chunks[i].L] += chunks[i].size;
        realized_chunk_weights(lengths, cand, cntb, cntc, freq,
                               chb, &nb, chc, &nc);
        for (int i = 0; i < nb; i++)
            base_bits += chb[i].weight * (double)(chb[i].depth + chb[i].bit);
        base_time = chunk_list_time(chb, nb, jp, kap, P[sigma]);
        dp_time   = chunk_list_time(chc, nc, jp, kap, P[sigma]);
        if (base_time < 0 || dp_time < 0) return -1;
    }
    if (!(dp_time <= jp->guard_time * base_time
          && dp_bits <= jp->guard_bits * base_bits))
        return -1;
    memcpy(lengths, cand, PIVCO_MAX_SYMBOLS);
    return 0;
}

int pivco_joint_optimize_lengths(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                 uint8_t lengths[PIVCO_MAX_SYMBOLS])
{
    if (!freq || !lengths) return -1;
    const pivco_effort_t effort = g_effort;
    if (effort == PIVCO_EFFORT_SIMPLEST_COMPRESS) return -1;
    /* The chunk model prices the OPTIMIZED decomposition; under the
     * other (ablation) tree modes the plain Huffman lengths are kept. */
    if (pivco_huffman_get_tree_mode() != PIVCO_TREE_MODE_OPTIMIZED) return -1;

    joint_params_t jp = joint_defaults;
    jp.gran = effort == PIVCO_EFFORT_FASTER_DECOMPRESS  ? 0
            : effort == PIVCO_EFFORT_FASTEST_DECOMPRESS ? 1
            : -1;   /* BALANCED -- and FASTEST_COMPRESS reaching a bare
                     * build_table, where no input size is available to
                     * resolve it (the pivcohuf file codec resolves it
                     * by size before building) */

    /* Frequencies narrow to u32 (and the model's sums wrap mod 2^32):
     * a histogram totalling >= 4 GiB may shape differently -- still
     * valid, still Kraft-exact -- lengths.  Correctness-only: every
     * index below is bounded structurally, never by frequency values. */
    jl_leaf_t leaf[PIVCO_MAX_SYMBOLS];
    uint32_t orv = 0, andv = ~(uint32_t)0;
    int n = 0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++)
        if (freq[i]) {
            uint32_t f = (uint32_t)freq[i];
            leaf[n].freq = f;
            leaf[n].sym  = (uint16_t)i;
            n++;
            orv |= f;
            andv &= f;
        }
    if (n < 2) return -1;
    sort_leaves(leaf, n, orv ^ andv);
    return joint_core(leaf, n, freq, lengths, &jp);
}
