#!/usr/bin/env python3
# Pareto sweep: cluster 256 symbols into K=4 buckets trading order-1 ratio gain
# against demux-chain yield (cross-bucket transition structure).
#
#   objective(asg) = chain_proxy(asg) + w * savings_frac(asg)
#
# chain_proxy: expected heads4 chain length (distinct-prefix cap 4, +1 terminal)
#              under a first-order Markov model of the class sequence.
# savings_frac: 1 - H1/H0 (Shannon), the order-1 context gain.
# w = 0    -> pure yield-max clustering
# w = inf  -> pure ratio (reference: the k-means clustering from k4yield)
# Also computes the clustering-INDEPENDENT yield ceiling: a chain must break
# when a raw SYMBOL recurs (same symbol => same class, any clustering).
import numpy as np, sys
from itertools import permutations
from k4yield import hufflen, o1_cost, yield_heads4, yield_win4, cluster_ctx, SAMPLE

K = 4

def yield_ub4(vl):
    N = len(vl); p = 0; lk = 0; em = 0
    while p + 6 < N:
        seen = set(); j = p
        while len(seen) < K:
            s = vl[j - 1] if j > 0 else -1
            if s in seen: break
            seen.add(s); j += 1
        j += 1; lk += 1; em += j - p; p = j
    return em / lk

# enumerate distinct-prefix paths once: lists of class tuples of length 2..4
PATHS = {L: [t for t in permutations(range(K), L)] for L in (2, 3, 4)}

def chain_proxy(T):
    tot = T.sum()
    pi = T.sum(1) / tot
    P = T / np.maximum(T.sum(1, keepdims=True), 1e-12)
    E = 1.0
    for L in (2, 3, 4):
        pr = 0.0
        for t in PATHS[L]:
            q = pi[t[0]]
            for i in range(L - 1): q *= P[t[i]][t[i + 1]]
            pr += q
        E += pr
    return E + 1.0                      # terminal emit

def xent(rows):                          # sum n * H(row) in bits
    tot = 0.0
    for r in rows:
        n = r.sum()
        if n <= 0: continue
        p = r[r > 0] / n
        tot += -n * (p * np.log2(p)).sum()
    return tot

def objective(M, asg, w, H0):
    A = np.zeros((256, K)); A[np.arange(256), asg] = 1
    T = A.T @ M @ A
    F = A.T @ M                          # follower dist per context class
    sav = 1.0 - xent(F) / H0
    return chain_proxy(T) + w * sav, sav

def climb(M, asg0, w, H0, passes=8):
    asg = asg0.copy()
    best, _ = objective(M, asg, w, H0)
    act = np.where((M.sum(1) + M.sum(0)) > 0)[0]
    for _ in range(passes):
        moved = 0
        for s in act:
            cur = asg[s]; bc = cur; bv = best
            for y in range(K):
                if y == cur: continue
                asg[s] = y
                o, _ = objective(M, asg, w, H0)
                if o > bv: bv, bc = o, y
            asg[s] = bc
            if bc != cur: best = bv; moved += 1
        if not moved: break
    return asg, best

def evaluate(v, asg, tag, f, o0):
    c = asg[v].astype(np.uint8)
    r = np.empty(len(v), np.uint8); r[0] = 0; r[1:] = c[:-1]
    rl = r.tolist()
    h4 = yield_heads4(rl); w3 = yield_win4(rl, 3)
    s4 = 100 * (1 - o1_cost(v, c, K) / o0)
    cross = float((c[1:] != c[:-1]).mean())
    p1 = np.bincount(c, minlength=K).max() / len(v)
    print(f"{f:9}{tag:7} {p1:5.2f} {cross:6.3f} | {h4:7.2f} {w3:7.2f} | {s4:6.2f}")
    return h4, s4

def main():
    files = sys.argv[1:] or ["dickens", "x-ray", "webster", "samba", "xml"]
    hdr = f"{'file':9}{'clust':7} {'p1':5} {'cross':6} | {'heads4':7} {'win4d3':7} | {'o1K4%':6}"
    print(hdr); print("-" * len(hdr))
    for f in files:
        v = np.fromfile(f"/tmp/phd_{f}/lit", dtype=np.uint8)[:SAMPLE]
        vl = v.tolist()
        idx = v[:-1].astype(np.int64) * 256 + v[1:].astype(np.int64)
        M = np.bincount(idx, minlength=65536).reshape(256, 256).astype(np.float64)
        H0 = xent([M.sum(0)])
        o0 = hufflen(np.bincount(v, minlength=256).tolist())
        ratio_asg = cluster_ctx(v, K)
        evaluate(v, ratio_asg, "ratio", f, o0)
        for w, tag in ((100, "w100"), ((30), "w30"), (0, "yield")):
            best_asg, best_o = None, -1e18
            for seed in (ratio_asg, np.arange(256) % K):
                a, o = climb(M, np.asarray(seed, np.int64), w, H0)
                if o > best_o: best_o, best_asg = o, a
            evaluate(v, best_asg, tag, f, o0)
        ub = yield_ub4(vl)
        print(f"{f:9}{'UB':7} {'':5} {'':6} | {ub:7.2f} {'':7} |   (any clustering)")
        print()

main()
