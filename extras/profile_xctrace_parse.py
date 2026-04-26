#!/usr/bin/env python3
"""Parse xctrace Time Profiler XML export, aggregate samples by source line.

Filters to decode-loop samples (backtrace contains decode_node_neon
or pivco_huffman_decode_neon) and aggregates by leaf-frame function
and (function, file, line).

Usage:  profile_xctrace_parse.py [xml_path]
   default: /tmp/time_profile.xml
"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter

xml_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/time_profile.xml'
tree = ET.parse(xml_path)
root = tree.getroot()

# path id -> path string
path_by_id = {}
for path in root.iter('path'):
    pid = path.get('id')
    if pid:
        path_by_id[pid] = path.text or ''

# frame id -> (name, line, file)
frame_info = {}
for frame in root.iter('frame'):
    fid = frame.get('id')
    if fid is None:
        continue
    name = frame.get('name', '')
    line = None
    fpath = ''
    src = frame.find('source')
    if src is not None:
        line = src.get('line')
        p = src.find('path')
        if p is not None:
            pid = p.get('id') or p.get('ref')
            if pid:
                fpath = path_by_id.get(pid, '')
            else:
                fpath = p.text or ''
    frame_info[fid] = (name, line, fpath)

# backtrace id -> list of frame ids (innermost first)
bt_frames = {}
for bt in root.iter('backtrace'):
    bid = bt.get('id')
    if bid is None:
        continue
    fs = []
    for frame in bt.findall('frame'):
        fid = frame.get('id') or frame.get('ref')
        if fid:
            fs.append(fid)
    bt_frames[bid] = fs


def is_decode_loop(frames):
    """True if backtrace passes through the decode entry points
    (so this sample is from the decode loop, not the encode setup
    the profile harness runs first)."""
    for f in frames:
        n, _, _ = frame_info.get(f, ('', '', ''))
        if n in ('decode_node_neon', 'pivco_huffman_decode_neon',
                 'pivco_huffman_decode'):
            return True
    return False


total_samples = 0
decode_total = 0
hits_func = Counter()
hits_line = Counter()  # (function, file_basename, line)

for row in root.iter('row'):
    bt_el = row.find('backtrace')
    if bt_el is None:
        continue
    bt_id = bt_el.get('id') or bt_el.get('ref')
    frames = bt_frames.get(bt_id, [])
    total_samples += 1
    if not is_decode_loop(frames):
        continue
    decode_total += 1
    if frames:
        leaf = frames[0]
        n, line, fpath = frame_info.get(leaf, ('?', None, ''))
        fbase = fpath.split('/')[-1]
        hits_func[n] += 1
        hits_line[(n, fbase, line)] += 1

print(f"Total samples (all phases):  {total_samples}")
print(f"Decode-loop samples:         {decode_total}")
print(f"  (filter: backtrace contains decode_node_neon or pivco_huffman_decode_neon)\n")

print("By function (leaf frame, % of decode-loop):")
for n, c in hits_func.most_common(20):
    pct = 100 * c / decode_total if decode_total else 0
    print(f"  {c:5d}  ({pct:5.1f}%)  {n}")

print("\nBy (function, file, line) — leaf frame:")
for (n, f, l), c in hits_line.most_common(30):
    pct = 100 * c / decode_total if decode_total else 0
    print(f"  {c:5d}  ({pct:5.1f}%)  {n} @ {f}:{l}")
