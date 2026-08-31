#!/usr/bin/env python3
# Capacity-CONSTRAINED context clustering, done right: the 64/128-codes-per-
# class cap lives INSIDE the optimization (alternating: class distributions
# <-> capacitated assignment), not as a post-hoc eviction.  Raw top-bit split
# is a feasible point and a seed, so the result is >= raw by construction.
#   K=2: assignment step exact (order by cost delta, take 128).
#   K=4: swap descent on the fixed cost matrix (swaps preserve caps).
# Best iterate chosen by TRUE Huffman-accounted order-1 cost.
import numpy as np, os
from k4yield import cluster_ctx, o1_cost, hufflen, SAMPLE

def bigram(v):
    idx = v[:-1].astype(np.int64) * 256 + v[1:].astype(np.int64)
    return np.bincount(idx, minlength=65536).reshape(256, 256).astype(np.float64)

def costmat(M, asg, K):
    C = np.zeros((K, 256))
    for k in range(K):
        sel = M[asg == k]
        if len(sel): C[k] = sel.sum(0)
    C += 0.5
    logp = np.log2(C / C.sum(1, keepdims=True))
    return -(M @ logp.T)                       # 256 x K

def assign_k2(c):
    order = np.argsort(c[:, 0] - c[:, 1])
    asg = np.ones(256, np.int64); asg[order[:128]] = 0
    return asg

def assign_swaps(c, asg, K):
    asg = asg.copy()
    for _ in range(2000):
        best = -1e-9; bs = None
        for a in range(K):
            Sa = np.where(asg == a)[0]
            for b in range(a + 1, K):
                Sb = np.where(asg == b)[0]
                if not len(Sa) or not len(Sb): continue
                D = (c[Sa, b] - c[Sa, a])[:, None] + (c[Sb, a] - c[Sb, b])[None, :]
                i, j = np.unravel_index(np.argmin(D), D.shape)
                if D[i, j] < best: best = D[i, j]; bs = (Sa[i], Sb[j], a, b)
        if bs is None: break
        s, t, a, b = bs; asg[s] = b; asg[t] = a
    return asg

def project(asg, K, cap, freq):
    # make an infeasible seed feasible: order by class, trim overflow (by freq)
    asg = asg.copy()
    for c in range(K):
        while int((asg == c).sum()) > cap:
            idxs = np.where(asg == c)[0]
            s = idxs[np.argmin(freq[idxs])]
            room = [k for k in range(K) if (asg == k).sum() < cap]
            asg[s] = room[0]
    return asg

def constrained(v, K, cap, o0):
    M = bigram(v); freq = np.bincount(v, minlength=256)
    raw = (np.arange(256) >> (9 - K.bit_length())).astype(np.int64)
    km = project(cluster_ctx(v, K), K, cap, freq)
    best_asg, best_o1 = None, None
    for seed in (raw, km):
        asg = seed.copy()
        for _ in range(12):
            c = costmat(M, asg, K)
            new = assign_k2(c) if K == 2 else assign_swaps(c, asg, K)
            o1 = o1_cost(v, new[v].astype(np.uint8), K)
            if best_o1 is None or o1 < best_o1: best_o1, best_asg = o1, new.copy()
            if (new == asg).all(): break
            asg = new
    return best_asg, 100 * (1 - best_o1 / o0)

def main():
    os.makedirs("/tmp/o1maps", exist_ok=True)

    for f in ["dickens", "webster", "xml", "samba", "x-ray", "mozilla"]:
        v = np.fromfile(f"/tmp/phd_{f}/lit", np.uint8)[:SAMPLE]
        o0 = hufflen(np.bincount(v, minlength=256).tolist())
        for K, cap, ext in ((4, 64, "map4"), (2, 128, "map2")):
            shift = 6 if K == 4 else 7
            raw_gain = 100 * (1 - o1_cost(v, (v >> shift).astype(np.uint8), K) / o0)
            ideal = 100 * (1 - o1_cost(v, cluster_ctx(v, K)[v].astype(np.uint8), K) / o0)
            asg, gain = constrained(v, K, cap, o0)
            # build bijective remap: class c -> codes [c*cap, (c+1)*cap)
            remap = np.zeros(256, np.uint8)
            for c in range(K):
                syms = np.where(asg == c)[0]
                assert len(syms) == cap
                for j, s in enumerate(syms): remap[s] = c * cap + j
            remap.tofile(f"/tmp/o1maps/{f}.{ext}")
            print(f"{f:9} K={K}  raw {raw_gain:5.2f}  constr {gain:5.2f}  ideal {ideal:5.2f}")

if __name__ == "__main__":
    main()
