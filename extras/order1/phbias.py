#!/usr/bin/env python3
# Traffic-weighted distribution of PH-tree node biases: what fraction of
# bitmap BITS sits at p_major >= threshold?  (Plain Huffman as PH proxy.)
import numpy as np, heapq

def tree_bias(freq):
    items = [(int(f), ()) for f in freq if f > 0]
    if len(items) < 2: return None
    h = [(f, i, None) for i, (f, _) in enumerate(items)]
    heap = [(f, i) for i, (f, _) in enumerate(items)]
    heapq.heapify(heap)
    nodes = {i: (f, None, None) for i, (f, _) in enumerate(items)}
    nid = len(items)
    internal = []
    while len(heap) > 1:
        fa, a = heapq.heappop(heap); fb, b = heapq.heappop(heap)
        internal.append((fa + fb, max(fa, fb) / (fa + fb)))
        nodes[nid] = (fa + fb, a, b)
        heapq.heappush(heap, (fa + fb, nid)); nid += 1
    tot = sum(t for t, _ in internal)         # total bitmap bits
    out = {}
    for thr in (0.625, 0.75, 0.85, 0.95):
        out[thr] = 100 * sum(t for t, b in internal if b >= thr) / tot
    mean = sum(t * b for t, b in internal) / tot
    return out, mean

for name, path in [
    ("dickens-lit", "/tmp/phd_dickens/lit"), ("webster-lit", "/tmp/phd_webster/lit"),
    ("xml-lit", "/tmp/phd_xml/lit"), ("samba-lit", "/tmp/phd_samba/lit"),
    ("x-ray-lit", "/tmp/phd_x-ray/lit"), ("mozilla-lit", "/tmp/phd_mozilla/lit"),
    ("x-ray-ml", "/tmp/phd_x-ray/ml"), ("x-ray-ll", "/tmp/phd_x-ray/ll"),
    ("dickens-ll", "/tmp/phd_dickens/ll"), ("mozilla-ml", "/tmp/phd_mozilla/ml"),
    ("dickens-ml", "/tmp/phd_dickens/ml"), ("xml-of", "/tmp/phd_xml/of")]:
    v = np.fromfile(path, np.uint8)
    r = tree_bias(np.bincount(v, minlength=256))
    if r is None: continue
    out, mean = r
    print(f"{name:12} mean bias {mean:5.3f}   bits at p>=.625: {out[0.625]:5.1f}%"
          f"  >=.75: {out[0.75]:5.1f}%  >=.85: {out[0.85]:5.1f}%  >=.95: {out[0.95]:5.1f}%")
