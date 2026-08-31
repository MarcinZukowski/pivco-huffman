#!/usr/bin/env python3
# How to STORE the pair-context map for ml|(ll,prev-ml):
#   freeform  k-means map over the (ll x pml) grid  (~1KB raw, entropy-coded less)
#   product   bucket = f(ll) x g(pml), two tiny per-dim maps (~27B)  [co-clustering]
#   pooled    one freeform map built on ALL files, shipped in the spec (0B)
# Gains = real per-bucket Huffman vs order-0 Huffman of the target stream.
import numpy as np

def huf_rows(M):
    import heapq
    n = M.sum(); tot = 0
    for r in M:
        m = int(r.sum())
        if not m: continue
        nz = [int(x) for x in r if x > 0]
        if len(nz) <= 1: tot += m; continue
        heapq.heapify(nz); c = 0
        while len(nz) > 1:
            a = heapq.heappop(nz); b = heapq.heappop(nz); c += a + b
            heapq.heappush(nz, a + b)
        tot += c
    return tot / n

def assign(R, asg, G, iters=12):
    for _ in range(iters):
        C = np.zeros((G, R.shape[1]))
        for k in range(G):
            sel = R[asg == k]
            if len(sel): C[k] = sel.sum(0)
        C += 0.5
        logp = np.log2(C / C.sum(1, keepdims=True))
        cost = -(R @ logp.T)
        new = cost.argmin(1); mass = R.sum(1); new[mass == 0] = asg[mass == 0]
        if (new == asg).all(): break
        asg = new
    return asg

def gain(M2, bucket_of_pair, At, base):
    K = bucket_of_pair.max() + 1
    Mk = np.zeros((K, At))
    for p in range(len(bucket_of_pair)):
        Mk[bucket_of_pair[p]] += M2[p]
    return 100 * (1 - huf_rows(Mk) / base)

def mapbytes_freeform(b, K):
    f = np.bincount(b, minlength=K).astype(float); f = f[f > 0]; p = f / f.sum()
    return int(-(f * np.log2(p)).sum() / 8) + 2   # entropy-coded grid + K

files = ["mozilla", "xml", "samba", "x-ray", "dickens"]
data = {}
A1 = A2 = At = 0
for f in files:
    ml = np.fromfile(f"/tmp/phd_{f}/ml", np.uint8)
    ll = np.fromfile(f"/tmp/phd_{f}/ll", np.uint8)
    pml = np.empty(len(ml), np.uint8); pml[0] = 0; pml[1:] = ml[:-1]
    data[f] = (ml, ll, pml)
    A1 = max(A1, int(ll.max()) + 1); A2 = max(A2, int(ml.max()) + 1)
At = A2
K, G1, G2 = 32, 4, 8

def joint3(f):
    ml, ll, pml = data[f]
    idx = (ll.astype(np.int64) * A2 + pml) * At + ml
    return np.bincount(idx, minlength=A1 * A2 * At).reshape(A1 * A2, At).astype(np.float64)

# pooled map from all files
Mpool = sum(joint3(f) for f in files)
mass = Mpool.sum(1)
order = np.argsort(-mass); asg0 = np.zeros(A1 * A2, np.int64)
acc = 0; c = 0
for r in order:
    asg0[r] = c; acc += mass[r]
    if acc > (c + 1) * mass.sum() / K and c < K - 1: c += 1
pooled = assign(Mpool, asg0.copy(), K)

print(f"{'file':9} {'o0 b/sym':9} {'free%':6} {'mapB':5} {'prod%':6} {'mapB':5} {'pooled%':7}")
for f in files:
    M2 = joint3(f)
    base = huf_rows(M2.sum(0, keepdims=True)[None, 0].reshape(1, -1))
    base = huf_rows(M2.sum(0).reshape(1, -1))
    # freeform
    m0 = M2.sum(1); o = np.argsort(-m0); a0 = np.zeros(A1 * A2, np.int64)
    acc = 0; c = 0
    for r in o:
        a0[r] = c; acc += m0[r]
        if acc > (c + 1) * m0.sum() / K and c < K - 1: c += 1
    free = assign(M2, a0.copy(), K)
    gf = gain(M2, free, At, base); bf = mapbytes_freeform(free, K)
    # product co-clustering
    M3 = M2.reshape(A1, A2, At)
    fA = np.arange(A1) % G1; gB = np.arange(A2) % G2
    for _ in range(6):
        R = np.zeros((A1, G2 * At))
        for j in range(G2): R[:, j * At:(j + 1) * At] = M3[:, gB == j, :].sum(1)
        fA = assign(R, fA, G1)
        Cm = np.zeros((A2, G1 * At))
        for i in range(G1): Cm[:, i * At:(i + 1) * At] = M3[fA == i, :, :].sum(0)
        gB = assign(Cm, gB, G2)
    prod = (fA[:, None] * G2 + gB[None, :]).reshape(-1)
    gp = gain(M2, prod, At, base)
    bp = (A1 * 2 + A2 * 3 + 7) // 8
    gpool = gain(M2, pooled, At, base)
    print(f"{f:9} {base:9.3f} {gf:6.2f} {bf:5d} {gp:6.2f} {bp:5d} {gpool:7.2f}")
