#!/usr/bin/env python3
import numpy as np
from k4yield import cluster_ctx, o1_cost, hufflen

def sh(f):
    f = f[f > 0].astype(float); n = f.sum()
    return float(-(f * np.log2(f / n)).sum())

def o1_sh(v, ctx, K):
    tot = 0.0
    for c in range(K):
        s = v[1:][ctx[:-1] == c]
        if len(s): tot += sh(np.bincount(s, minlength=256))
    return tot

for f in ["mozilla", "xml", "samba", "x-ray", "dickens"]:
    for s in ["ll", "ml", "of"]:
        v = np.fromfile(f"/tmp/phd_{f}/{s}", np.uint8)
        A = int(v.max()) + 1
        o0h = hufflen(np.bincount(v, minlength=256).tolist())
        o0s = sh(np.bincount(v, minlength=256))
        m = cluster_ctx(v, 2)
        c2 = m[v].astype(np.uint8)
        k2h = 100 * (1 - o1_cost(v, c2, 2) / o0h)
        k2s = 100 * (1 - o1_sh(v, c2, 2) / o0s)
        fh = 100 * (1 - o1_cost(v, v, A) / o0h)
        fs = 100 * (1 - o1_sh(v, v, A) / o0s)
        print(f"{f:8} {s}: b/sym {o0s/len(v):5.3f}  K2 huf {k2h:4.1f}% fse {k2s:4.1f}%   full huf {fh:4.1f}% fse {fs:4.1f}%")
