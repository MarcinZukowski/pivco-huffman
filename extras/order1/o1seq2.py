#!/usr/bin/env python3
# Joint (bucketed) contexts for sequence codes: raw-pair Shannon ceilings +
# k-means context bucketing to K tables with real per-bucket Huffman.
import numpy as np, heapq

def hufbits_hist(f):
    nz = [int(x) for x in f if x > 0]
    n = sum(nz)
    if len(nz) <= 1: return float(n)          # 1 bit/sym floor
    heapq.heapify(nz); cost = 0
    while len(nz) > 1:
        a = heapq.heappop(nz); b = heapq.heappop(nz)
        cost += a + b; heapq.heappush(nz, a + b)
    return float(cost)

def joint(ctx, tgt, Ac, At):
    return np.bincount(ctx.astype(np.int64) * At + tgt.astype(np.int64),
                       minlength=Ac * At).reshape(Ac, At).astype(np.float64)

def shannon_rows(M):
    n = M.sum(); h = 0.0
    for r in M:
        m = r.sum()
        if m: p = r[r > 0] / m; h += m * -(p * np.log2(p)).sum()
    return h / n

def huf_rows(M):
    n = M.sum()
    return sum(hufbits_hist(r) for r in M if r.sum()) / n

def cluster_rows(M, K, iters=25):
    mass = M.sum(1); act = np.where(mass > 0)[0]
    order = act[np.argsort(-mass[act])]
    asg = np.zeros(len(M), np.int64); tot = mass.sum(); acc = 0.0; c = 0
    for r in order:
        asg[r] = c; acc += mass[r]
        if acc > (c + 1) * tot / K and c < K - 1: c += 1
    for _ in range(iters):
        C = np.zeros((K, M.shape[1]))
        for k in range(K):
            sel = M[(asg == k)]
            if len(sel): C[k] = sel.sum(0)
        C += 0.5
        logp = np.log2(C / C.sum(1, keepdims=True))
        cost = -(M @ logp.T)
        new = cost.argmin(1); new[mass == 0] = asg[mass == 0]
        if (new == asg).all(): break
        asg = new
    Mk = np.zeros((K, M.shape[1]))
    for k in range(K):
        sel = M[(asg == k)]
        if len(sel): Mk[k] = sel.sum(0)
    return Mk

def prev(x):
    p = np.empty(len(x), x.dtype); p[0] = 0; p[1:] = x[:-1]; return p

for f in ["dickens", "webster", "xml", "samba", "x-ray", "mozilla"]:
    d = f"/tmp/phd_{f}"
    ll = np.fromfile(f"{d}/ll", np.uint8); ml = np.fromfile(f"{d}/ml", np.uint8)
    of = np.fromfile(f"{d}/of", np.uint8)
    Al, Am, Ao = int(ll.max()) + 1, int(ml.max()) + 1, int(of.max()) + 1
    print(f"{f}: nseq={len(ll)}")
    cases = [
        ("of|ml",          of, ml, Ao, Am),
        ("of|(ll,ml)",     of, ll.astype(np.int64) * Am + ml, Ao, Al * Am),
        ("of|(ml,pof)",    of, ml.astype(np.int64) * Ao + prev(of), Ao, Am * Ao),
        ("ml|ll",          ml, ll, Am, Al),
        ("ml|(ll,pml)",    ml, ll.astype(np.int64) * Am + prev(ml), Am, Al * Am),
        ("ll|pll",         ll, prev(ll), Al, Al),
        ("ll|(pll,pof)",   ll, prev(ll).astype(np.int64) * Ao + prev(of), Al, Al * Ao),
    ]
    for name, tgt, ctx, At, Ac in cases:
        M = joint(ctx, tgt, Ac, At)
        h0 = shannon_rows(M.sum(0, keepdims=True)); u0 = huf_rows(M.sum(0, keepdims=True))
        hs = shannon_rows(M)
        out = f"  {name:13} raw({Ac:5}ctx) sh {100*(1-hs/h0):5.1f}%"
        for K in (16, 32):
            Mk = cluster_rows(M, K)
            out += f"  K{K}: sh {100*(1-shannon_rows(Mk)/h0):5.1f}% huf {100*(1-huf_rows(Mk)/u0):5.1f}%"
        print(out)
