#!/usr/bin/env python3
# Order-1 gain curve over K (ideal clustering) + constrained (top-bit remap,
# 256/K codes per class) for K=8/16, + full order-1 ceiling (K=256).
import numpy as np, os
from k4yield import cluster_ctx, o1_cost, hufflen, SAMPLE
from mapgen2 import constrained

os.makedirs("/tmp/o1maps", exist_ok=True)
KS = (2, 4, 8, 16, 32, 64)
print(f"{'file':9} " + " ".join(f"K{k:<4}" for k in KS) + "  full   | K8c   K16c")
for f in ["dickens", "webster", "xml", "samba", "x-ray", "mozilla"]:
    v = np.fromfile(f"/tmp/phd_{f}/lit", np.uint8)[:SAMPLE]
    o0 = hufflen(np.bincount(v, minlength=256).tolist())
    ideals = []
    for K in KS:
        asg = cluster_ctx(v, K)
        ideals.append(100 * (1 - o1_cost(v, asg[v].astype(np.uint8), K) / o0))
    full = 100 * (1 - o1_cost(v, v, 256) / o0)
    cons = []
    for K, ext in ((8, "map8"), (16, "map16")):
        cap = 256 // K
        asg, g = constrained(v, K, cap, o0)
        cons.append(g)
        remap = np.zeros(256, np.uint8)
        for c in range(K):
            syms = np.where(asg == c)[0]
            assert len(syms) == cap
            for j, s in enumerate(syms): remap[s] = c * cap + j
        remap.tofile(f"/tmp/o1maps/{f}.{ext}")
    print(f"{f:9} " + " ".join(f"{g:5.2f}" for g in ideals) +
          f" {full:5.2f}  | {cons[0]:5.2f} {cons[1]:5.2f}")
