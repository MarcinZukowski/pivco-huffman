#!/usr/bin/env python3
# Literals: Shannon (=FSE-achievable) vs Huffman, order-0 and K=2-context.
import numpy as np, heapq

def hufbits(f):
    nz = [int(x) for x in f if x > 0]
    n = sum(nz)
    if len(nz) <= 1: return float(n)
    heapq.heapify(nz); c = 0
    while len(nz) > 1:
        a = heapq.heappop(nz); b = heapq.heappop(nz); c += a + b
        heapq.heappush(nz, a + b)
    return float(c)

def shbits(f):
    f = f[f > 0].astype(float); n = f.sum()
    return float(-(f * np.log2(f / n)).sum())

for name in ["dickens", "webster", "xml", "samba", "x-ray", "mozilla"]:
    v0 = np.fromfile(f"/tmp/phd_{name}/lit", np.uint8)
    m = np.fromfile(f"/tmp/o1maps/{name}.map2", np.uint8)
    v = m[v0]                                    # remapped, class = top bit
    n = len(v)
    h = np.bincount(v, minlength=256)
    o0h = hufbits(h) / n; o0s = shbits(h) / n
    ctx = np.empty(n, np.uint8); ctx[0] = 0; ctx[1:] = v[:-1] >> 7
    c1h = c1s = 0.0
    for c in (0, 1):
        hh = np.bincount(v[ctx == c], minlength=256)
        c1h += hufbits(hh); c1s += shbits(hh)
    c1h /= n; c1s /= n
    print(f"{name:9} o0: huf {o0h:5.3f} sh {o0s:5.3f} (fse gain {o0h-o0s:5.3f})"
          f"   K2o1: huf {c1h:5.3f} sh {c1s:5.3f}"
          f"   [o1 gain {o0h-c1h:5.3f} | +fse {c1h-c1s:5.3f} | both {o0h-c1s:5.3f}]")
