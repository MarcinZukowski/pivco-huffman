#!/usr/bin/env python3
# Yield simulation for K=4 (top-2-bit buckets) demux kernel proposals vs the
# built K=2 (6,6) window kernel.  Metric: bytes emitted per DEPENDENT table
# lookup -- the serial currency all these kernels spend.
#
# Kernels simulated (all walks operate on residence classes r[j] = class(v[j-1])):
#  heads4   : user's sketch.  Visible per lookup: class of the FIRST unconsumed
#             element (head) of each of the 4 buckets.  Chain = consume while
#             residence bucket not yet consumed this chain (distinct-prefix),
#             then emit ONE terminal element (position known: bucket = class of
#             last emitted, cursor known; its class is not needed until the next
#             lookup, which sees it in the shifted H regs).
#  win4(d)  : asymmetric window variant: d elements of lookahead in the chain's
#             STARTING bucket, 1 head for each other bucket.  Index = 2d+6+2 bits.
#  k2_66    : the built kernel: 6 lookahead each side, K=2 top-bit classes.
#             Reported without terminal (matches built code, calibrates cost)
#             and with terminal (+1/iter backport idea).
import numpy as np, heapq, sys, os

SAMPLE = 2_000_000

def hufflen(freq):
    nz = [f for f in freq if f > 0]
    if not nz: return 0
    if len(nz) == 1: return nz[0]          # 1 bit/sym convention
    heapq.heapify(nz); cost = 0
    while len(nz) > 1:
        a = heapq.heappop(nz); b = heapq.heappop(nz)
        cost += a + b; heapq.heappush(nz, a + b)
    return cost

def o1_cost(v, cls_prev, K):
    tot = 0
    for c in range(K):
        sel = v[1:][cls_prev[:-1] == c]
        if len(sel): tot += hufflen(np.bincount(sel, minlength=256).tolist())
    return tot

def yield_heads4(r):
    N = len(r); p = 0; lk = 0; em = 0
    while p + 6 < N:
        used = 0; j = p
        while True:
            b = 1 << r[j]
            if used & b: break
            used |= b; j += 1
        j += 1                              # terminal (position-determined)
        lk += 1; em += j - p; p = j
    return em / lk

def yield_win4(r, d):
    N = len(r); p = 0; lk = 0; em = 0
    while p + d + 6 < N:
        c0 = r[p]; cnt = [0, 0, 0, 0]; j = p
        while True:
            b = r[j]
            if cnt[b] >= (d if b == c0 else 1): break
            cnt[b] += 1; j += 1
        j += 1                              # terminal
        lk += 1; em += j - p; p = j
    return em / lk

def yield_k2(r, d, term):
    # term=0: built kernel -- next iteration restarts AT the stopping element.
    # term=1: +terminal backport -- emit the position-determined stopper too.
    N = len(r); p = 0; lk = 0; em = 0
    while p + 2 * d + 2 < N:
        cnt = [0, 0]; j = p
        while True:
            b = r[j]
            if cnt[b] >= d: break
            cnt[b] += 1; j += 1
        j += term
        lk += 1; em += j - p; p = j
    return em / lk

def cluster_ctx(v, K, iters=16):
    # k-means-style context clustering: partition prev-symbols into K classes
    # minimizing sum of per-context follower entropies (the remap objective).
    idx = v[:-1].astype(np.int64) * 256 + v[1:].astype(np.int64)
    M = np.bincount(idx, minlength=65536).reshape(256, 256).astype(np.float64)
    rowsum = M.sum(1); tot = rowsum.sum()
    order = np.argsort(-rowsum)
    asg = np.zeros(256, np.int64); c = 0; acc = 0.0
    for s in order:                      # init: cumulative-frequency quantiles
        asg[s] = c; acc += rowsum[s]
        if acc > (c + 1) * tot / K and c < K - 1: c += 1
    for _ in range(iters):
        C = np.zeros((K, 256))
        for k in range(K):
            sel = M[asg == k]
            if len(sel): C[k] = sel.sum(0)
        C += 0.5
        logp = np.log2(C / C.sum(1, keepdims=True))
        cost = -M @ logp.T
        newasg = cost.argmin(1)
        newasg[rowsum == 0] = asg[rowsum == 0]
        if (newasg == asg).all(): break
        asg = newasg
    return asg

def row(v, c2, c4, tag, f):
    r4 = np.empty(len(v), np.uint8); r4[0] = 0; r4[1:] = c4[:-1]
    r2 = np.empty(len(v), np.uint8); r2[0] = 0; r2[1:] = c2[:-1]
    r4l = r4.tolist(); r2l = r2.tolist()
    h4 = yield_heads4(r4l)
    w2 = yield_win4(r4l, 2); w3 = yield_win4(r4l, 3); w4 = yield_win4(r4l, 4)
    k2 = yield_k2(r2l, 6, 0); k2t = yield_k2(r2l, 6, 1)
    o0 = hufflen(np.bincount(v, minlength=256).tolist())
    s2 = 100 * (1 - o1_cost(v, c2, 2) / o0)
    s4 = 100 * (1 - o1_cost(v, c4, 4) / o0)
    p12 = np.bincount(c2, minlength=2).max() / len(v)
    p14 = np.bincount(c4, minlength=4).max() / len(v)
    print(f"{f:9}{tag:5} {p12:6.3f} {p14:6.3f} | {h4:7.2f} {w2:7.2f} {w3:7.2f} {w4:7.2f} {k2:6.2f} {k2t:6.2f} | {s2:6.2f} {s4:6.2f}")

def main():
    files = sys.argv[1:] or ["dickens", "mozilla", "x-ray", "webster", "samba", "xml"]
    hdr = f"{'file':9}{'cls':5} {'p1K2':6} {'p1K4':6} | {'heads4':7} {'win4d2':7} {'win4d3':7} {'win4d4':7} {'k2_66':6} {'+term':6} | {'o1K2%':6} {'o1K4%':6}"
    print(hdr); print("-" * len(hdr))
    for f in files:
        v = np.fromfile(f"/tmp/phd_{f}/lit", dtype=np.uint8)[:SAMPLE]
        row(v, (v >> 7).astype(np.uint8), (v >> 6).astype(np.uint8), "top", f)
        m2 = cluster_ctx(v, 2); m4 = cluster_ctx(v, 4)
        row(v, m2[v].astype(np.uint8), m4[v].astype(np.uint8), "opt", f)

if __name__ == "__main__":
    main()
