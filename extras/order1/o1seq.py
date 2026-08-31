#!/usr/bin/env python3
# Order-1 structure of zstd's LL/ML/OF code streams (phaz dumps).
# Codes are already classes -> context = full prev code, no clustering.
# Shannon (model-level) + Huffman (realizable) per conditioning.
import numpy as np, heapq, os

def H0(x, A):
    f = np.bincount(x, minlength=A).astype(np.float64); f = f[f > 0]
    p = f / f.sum()
    return -(p * np.log2(p)).sum()

def Hcond(t, c, At, Ac):
    M = np.bincount(c.astype(np.int64) * At + t.astype(np.int64),
                    minlength=Ac * At).reshape(Ac, At).astype(np.float64)
    n = M.sum(); h = 0.0
    for r in M:
        m = r.sum()
        if m == 0: continue
        p = r[r > 0] / m
        h += m * -(p * np.log2(p)).sum()
    return h / n

def hufbits(x, A):
    f = [int(v) for v in np.bincount(x, minlength=A) if v > 0]
    if len(f) <= 1: return float(len(f))
    heapq.heapify(f); cost = 0; n = sum(f)
    while len(f) > 1:
        a = heapq.heappop(f); b = heapq.heappop(f); cost += a + b
        heapq.heappush(f, a + b)
    return cost / n

def hufcond(t, c, At, Ac):
    tot = 0; n = len(t)
    for cc in range(Ac):
        sel = t[c == cc]
        if len(sel): tot += hufbits(sel, At) * len(sel)
    return tot / n

for f in ["dickens", "webster", "xml", "samba", "x-ray", "mozilla"]:
    d = f"/tmp/phd_{f}"
    ll = np.fromfile(f"{d}/ll", np.uint8); ml = np.fromfile(f"{d}/ml", np.uint8)
    of = np.fromfile(f"{d}/of", np.uint8)
    xb = os.path.getsize(f"{d}/xb"); lit = os.path.getsize(f"{d}/lit")
    A = {"ll": int(ll.max()) + 1, "ml": int(ml.max()) + 1, "of": int(of.max()) + 1}
    n = len(ll)
    print(f"{f}: nseq={n}  alph ll/ml/of={A['ll']}/{A['ml']}/{A['of']}  "
          f"xb={xb}B lit={lit}B  codes={3*n}B raw")
    streams = {"ll": ll, "ml": ml, "of": of}
    for tn, t in streams.items():
        h0 = H0(t, A[tn]); hu0 = hufbits(t, A[tn])
        rows = []
        for cn, c in streams.items():
            if cn == tn:
                ctx = np.empty(n, np.uint8); ctx[0] = 0; ctx[1:] = t[:-1]
                lbl = f"{tn}|prev {tn}"
            else:
                ctx = c; lbl = f"{tn}|{cn} (same seq)"
            hs = Hcond(t, ctx, A[tn], A[cn]); hh = hufcond(t, ctx, A[tn], A[cn])
            rows.append((lbl, 100 * (1 - hs / h0), 100 * (1 - hh / hu0)))
        # prev-seq offset as context for everything
        pof = np.empty(n, np.uint8); pof[0] = 0; pof[1:] = of[:-1]
        if tn != "of":
            hs = Hcond(t, pof, A[tn], A["of"]); hh = hufcond(t, pof, A[tn], A["of"])
            rows.append((f"{tn}|prev of", 100 * (1 - hs / h0), 100 * (1 - hh / hu0)))
        print(f"  {tn}: H0={h0:5.3f}b huf={hu0:5.3f}b   " +
              "  ".join(f"{l}:{gs:5.1f}%/{gh:5.1f}%" for l, gs, gh in rows))
