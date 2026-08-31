#!/usr/bin/env python3
# K=2 self-contexts on ll/ml/of code streams vs full-alphabet contexts.
import numpy as np
from k4yield import cluster_ctx, o1_cost, hufflen

for f in ["mozilla", "xml", "samba", "x-ray", "dickens"]:
    row = f"{f:9}"
    for s in ["ll", "ml", "of"]:
        v = np.fromfile(f"/tmp/phd_{f}/{s}", np.uint8)
        o0 = hufflen(np.bincount(v, minlength=256).tolist())
        # full-alphabet lag-1 context (what o6-with-self-route would code)
        full = 100 * (1 - o1_cost(v, v, int(v.max()) + 1) / o0)
        # K=2 clustered contexts (the literal-style approach)
        m = cluster_ctx(v, 2)
        k2 = 100 * (1 - o1_cost(v, m[v].astype(np.uint8), 2) / o0)
        row += f"  {s}: K2 {k2:4.1f}% / full {full:4.1f}%"
    print(row)
