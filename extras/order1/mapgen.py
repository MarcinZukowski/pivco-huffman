#!/usr/bin/env python3
# Build bijective remap tables so class = top bit(s) of the remapped byte,
# from the ratio-optimal context clustering.  Constraint of the top-bit
# design: exactly 64 codes per class (K=4) / 128 (K=2); inactive symbols pad
# for free, overflowing actives are evicted lowest-frequency-first.
import numpy as np, os
from k4yield import cluster_ctx, o1_cost, hufflen, SAMPLE

os.makedirs("/tmp/o1maps", exist_ok=True)
for f in ["dickens", "webster", "xml", "samba", "x-ray", "mozilla"]:
    v = np.fromfile(f"/tmp/phd_{f}/lit", np.uint8)[:SAMPLE]
    freq = np.bincount(v, minlength=256)
    active = freq > 0
    o0 = hufflen(freq.tolist())
    for K, cap, ext in ((4, 64, "map4"), (2, 128, "map2")):
        asg = cluster_ctx(v, K).copy()
        for c in range(K):
            while int((active & (asg == c)).sum()) > cap:
                idxs = np.where(active & (asg == c))[0]
                s = idxs[np.argmin(freq[idxs])]
                counts = [int((active & (asg == k)).sum()) if k != c else 10**9
                          for k in range(K)]
                asg[s] = int(np.argmin(counts))
        inact = [s for s in range(256) if not active[s]]
        remap = np.zeros(256, np.uint8)
        for c in range(K):
            syms = [s for s in range(256) if asg[s] == c and active[s]]
            fill = cap - len(syms)
            grp = syms + inact[:fill]; inact = inact[fill:]
            assert len(grp) == cap
            for j, s in enumerate(grp): remap[s] = c * cap + j
        assert not inact
        shift = 6 if K == 4 else 7
        cls = (remap[v] >> shift).astype(np.uint8)
        gain = 100 * (1 - o1_cost(v, cls, K) / o0)
        p1 = np.bincount(cls, minlength=K).max() / len(v)
        remap.tofile(f"/tmp/o1maps/{f}.{ext}")
        print(f"{f:9} K={K}  p1={p1:5.3f}  o1 gain {gain:5.2f}%")
