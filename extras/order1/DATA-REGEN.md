# Regenerating every uncommitted dataset

Everything the order-1 program measured is either committed here, sitting
in `data/` (gitignored), or regenerable with the commands below.  Two
caveats first:

- **The capture fix changes the parse.**  main commit 668aa37 ("phaz:
  capture the stock parse") makes phaz capture the stock zstd parse.
  Artifacts produced BEFORE it (`o1bundle.tgz`, the lvlstats census,
  anything labeled "old capture") are bit-reproducible only with that
  commit reverted.  The tarballs in `data/` are the archives of record
  for those.
- **Workflow-era probe tools are not archived.**  The 2026-08-24
  exploration's one-off probes (wire_audit.c, phprobe.c, e8e9.c, the
  4-model context scripts) did not survive the
  scratchpad; ledger sections citing them stand on their recorded
  outputs only.  Everything in `probes/` here IS archived.
- **The 2026-09-08 tools are not archived here yet.**  They live in
  `notes/2026-08-15-o1demux-k4/` (untracked); build lines below.  Their
  numbers (ledger §1.3, §4.3, §8.6, §9.1, §16.1–2, §23.1) stand on the
  archived result files there (`xfield_realfse_results.txt`,
  `k8_crossfield_results.txt`) and on `sessionratio` re-run 2026-09-09.
- **Scratchpads die.**  /tmp (and this session's scratchpad) gets purged
  by macOS; it already ate one worktree mid-session.  Nothing below
  depends on scratchpad state.

## Sources

| thing | where |
|---|---|
| silesia corpus | `curl -sLO http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip` (12 files, ~68 MB) |
| Nick's datasets | committed on branch `pr30-dynfse` (= e1bdce0) under `extras/datasets/` |
| prototype worktree | `git worktree add <dir> e1bdce0 && cd <dir> && patch -p1 < extras/order1/ph-ctx-session.patch` then `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`; for phaz e2e also apply main's capture fix: `git show 668aa37 \| patch -p1` and build phaz with `ZSTD_SRC=<mainrepo>/ext/zstd` |
| phaz | `extras/phaz/tools/build.sh` (patches a copy of ext/zstd, builds `./phaz`) |

## Prototype env knobs (all in ph-ctx-session.patch)

| knob | effect |
|---|---|
| `PIVCO_LVL_STATS=1` | per-tree-level winner/savings census to stderr on compress |
| `PIVCO_WIRE_STATS=1` | wire-category byte census (kr/marker/bitmap/pad/fse/framing), reconciles to container size |
| `PIVCO_FSE_RATIO/THRESH/MINB` | override the static-FSE commit gates (defaults 0.95 / 0.625 / 32) |
| `PIVCO_BLKTAB=K` | per-block table ring, K = 1..8 (flags bit 2 on the wire; decode self-describing) |
| `PIVCO_NODYN=1` | disable id 51 (dynamic nibble) |
| `PIVCO_CTX=1`, `PIVCO_CTX_DICT=N`, `PIVCO_CTX_FREE=1`, `PIVCO_NOFUSE=1` | ctx coder on / dictionary cap / no dictionary (pre-cap coder) / quad fusion off |
| `PIVCO_CTX_NOCACHE`, `PIVCO_CTX_DP`, `PIVCO_CTX_L=N` | no table reuse (streaming bound) / DP probability expansion / tableLog override (16-param modes use N−1) |
| `PIVCO_CTX_K1`, `PIVCO_CTX_K2`, `PIVCO_CTX_MAXD=D` | run id-52 as order-1 / drop the k=4 candidate / ctx attempts only at depth ≤ D |
| `PIVCO_CTX_W4`, `PIVCO_CTX_X4` | nibble-wide symbols (L can drop to 7) / 4-segment interleaved streams per record |
| `PIVCO_NIB2` | id-51 position-split experiment (1 = forced split, 2 = best-of with selector byte) |

On main since 3bfef16 (2026-09-09) the nibble table is production:
`pivco_cfg_t.fse_nibble_enabled` (opt-in; the CLI's `-a --ans-nibble` sets
it; `pivcohuf_compress_ex(.., 1)` is the static schedule only, so `nibbench`
passes a cfg, while phaz's `compress_ex` calls deliberately stay on the
static schedule until the nibble / k1 trade-offs are settled), `PIVCO_FSE_NIB_TABLELOG` = 7 with the
`#ifndef` still honouring `-DPIVCO_FSE_NIB_TABLELOG=$L` for sweeps, and a
decoder cap `PIVCO_FSE_NIB_TABLELOG_MAX` = 10.

## data/ (gitignored)

- **`o1bundle.tgz`** — level-3 stream dumps (`phd_<f>/{lit,ll,ml,of}`) +
  `o1maps/` for dickens/mozilla/samba/webster/x-ray/xml.  Regen (OLD
  capture — see caveat): `./phaz dump <silesia/f> /tmp/phd_<f> -l 3`
  per file, then `python3 mapgen2.py` (writes /tmp/o1maps).  Recipe also
  in README.md §"regenerate dumps".
- **`lvlstats-2026-08-24.tgz`** — the per-level census (ledger §1.1/§2.1):
  24 inputs × 2 arms.  On the prototype worktree, per input:
  `PIVCO_LVL_STATS=1 [PIVCO_CTX=1 PIVCO_NOFUSE=1] ./build/pivcohuf c -a -f IN /dev/null 2> OUT.err`
  Inputs: the 11 `extras/datasets/*` files + the 6 o1bundle `lit` streams
  + mozilla/x-ray `ll,ml,of`.  (Ran on OLD-capture L3 streams.)

## Level-19 dumps and frame anatomy (ledger §2.2, waterfall, capture fix)

```sh
./phaz dump  <silesia/f> OUTDIR -l 19     # lit/ll/ml/of/xb + meta.txt + blocks
./phaz stats <silesia/f> -l 19            # phaz vs stock zstd sizes + decode timing
zstd -19 -q --no-check -f <f> -o f.zst    # CLI frame; probes/zc19.c = library one-shot frame
```
Fixed-capture dumps need the capture fix applied; the pre-fix variants
need it reversed.

## probes/ (single-file C tools; all compile standalone)

| tool | what | build |
|---|---|---|
| `ztale.c` | zstd frame anatomy: literals vs sequences sections, table-mode histogram per block | `cc -O2 -o ztale ztale.c` |
| `hufadapt.c` | table-adaptation arms: per-128K fresh tables (Z) vs one global CTable (G), HUF for lit / FSE for codes | `cc -O2 -o hufadapt hufadapt.c -I<wt>/ext/fse/lib <wt>/build/libpivco_huffman.a <wt>/build/libhuf0.a -lm` |
| `zc19.c` | library one-shot `ZSTD_compress(level)` writer (vs CLI param drift) | `cc -O2 -o zc19 zc19.c -I extras/phaz/build/zstd/lib extras/phaz/build/zstd/lib/libzstd.a` |
| `tbuild.c` | `pivco_build_table_from_code_lens` microbench (per-new-table decode cost) | `cc -O2 -o tbuild tbuild.c -I<wt>/include <wt>/build/libpivco_huffman.a -lm` |
| `nibbench.c` | in-process PHA compress+decompress bench with slot-51 commit stats — the §1.2 tableLog sweep driver | `cc -O2 -falign-functions=64 -o nibbench nibbench.c -I<wt>/include -I<wt>/src -I<wt>/ext/fse/lib <wt-lib>` per L-variant lib |
| `dtbuild.c` | FSE_buildDTable cost ladder by tableLog (~1.8 ns/slot on M4) | `cc -O2 -o dtbuild dtbuild.c -I<wt>/ext/fse/lib <wt>/build/libpivco_huffman.a` |
| `bench_huf_density.c` | the wscan probe harness: W-sweep, bit-context, quad fusion, 3-level (oct) fusion, dyn51-on-quad — source of ledger §2's probe ladder and §3.1/§3.2 | `cc -O2 -o bench_hd bench_huf_density.c -I<wt>/include -I<wt>/src -I<wt>/ext/fse/lib <wt>/build/libpivco_huffman.a <wt>/build/libhuf0.a -lm`; run `./bench_hd --wscan FILE` |

(`<wt>` = the built prototype worktree.)

## 2026-09-08 tools (`notes/2026-08-15-o1demux-k4/`; main tree, not the prototype)

All need `/tmp/phd_<f>/{lit,ll,ml,of}` + `/tmp/o1maps/<f>.map{2,4}` from
`data/o1bundle.tgz` and a Release `build/`.  `<top>` = repo root.

| tool | what | build / run |
|---|---|---|
| `sessionratio.c` | real-FSE ratio of order0 / K2serial / K4serial / k2g2 / k2g4 / k4g16, whole-stream tables, decode-verified (§9.1) | `cc -O3 -mcpu=native -I ext/fse/lib sessionratio.c build/libpivco_huffman.a -o sessionratio && ./sessionratio [files]` |
| `sessionratio_blk.c` | same with per-block tables, env `BLK` (default 16384, 0 = whole stream) (§16.1) | as above; `BLK=16384 ./sessionratio_blk` |
| `sessionratio_rep.c` | + NCount repeat-mode (`REPEAT=1`) and the per-block mode argmax columns (§16.1) | as above; `REPEAT=1 ./sessionratio_rep` |
| `sessionspeed.c` | end-to-end decode (FSE `usingDTable` + production merges + NEON gather), byte-exact, ns/B (§8.6) | `cc -O3 -mcpu=native -I include -I src -I ext/fse/lib sessionspeed.c build/libpivco_huffman.a -o sessionspeed` |
| `g4merge.c` | pure reconstruction kernels: g2 merge+OR, g4 3-merge + pext, scalar (§8.6) | `cc -O3 -mcpu=native -I include -I src g4merge.c build/libpivco_huffman.a -o g4merge` |
| `planefse.c`, `planefse4.c` | class-plane decode, single- and 4-way-interleaved FSE | `cc -O3 -mcpu=native -I ext/fse/lib planefse4.c build/libpivco_huffman.a -o planefse4` |
| `bmdump.c` | dump id 51's real regions per 16K block via the public table API (§1.3) | `cc -O3 -I include bmdump.c build/libpivco_huffman.a -o bmdump && ./bmdump IN OUT.dump` |
| `coderbench.c` | nib / pair / byte / pos2 / repeat / tableLog-7 / straight-to-bytes coders on those regions (§1.3); env `TL256` for the pair table | `cc -O3 -mcpu=native -I include -I src -I ext/fse/lib coderbench.c build/libpivco_huffman.a -o coderbench && ./coderbench *.dump` |
| `fsesum.c` | real FSE byte cost of every `*.bin` in a directory (charging rules in the header) | `cc -O3 -I ext/fse/lib fsesum.c build/libpivco_huffman.a -o /tmp/fsesum` |
| `xfield_demux.py` | cross-field class-context demux of ll/ml/of into `/tmp/xf/`, then prices with `/tmp/fsesum` (§16.2) | `python3 xfield_demux.py` (writes `xfield_realfse_results.txt`) |
| `k8_crossfield.py` | Shannon accounting for kKgK² at K = 2/4/8 + cross-field (§9.1, §16.2 Shannon column) | `python3 k8_crossfield.py > k8_crossfield_results.txt` |
| `extras/bench/bench_merge4way.c` | one-go vs 3-binary 4-way merge (§4.3) — committed | CMake target `pivco_bench_merge4way` |

Sweeps behind the 2026-09-08/09 numbers:

- **ctx vs k4g16** (§23.1): worktree of e1bdce0 + `ph-ctx-session.patch`,
  `PIVCO_CTX=1 ./build/pivcohuf c -a` per lit stream (dict-48 default) vs
  `sessionratio` on the same streams.
- **tableLog re-check on the landing tree** (§1.2 “shipped”): two builds
  `-DCMAKE_C_FLAGS="-DPIVCO_FSE_NIB_TABLELOG=$L -falign-functions=64"`
  (L = 7, 10), `probes/nibbench.c` per stream; cross-cap decode = encode
  with the L=10 build's `pivcohuf c -a`, decode with the L=7 build, `cmp`.
- **Fleet byte-identity for the landing**: `pivcohuf c [-a] -b 32768` over
  the 18 o1bundle streams on c8i / c8a / c8g + M4, md5 compared (ph and
  pha), decode + `cmp` on every host.

## Sweeps behind ledger/chat numbers (exact arms)

- **Wire waterfall** (per stream, prototype pivcohuf):
  baseline `PIVCO_WIRE_STATS=1 PIVCO_NODYN=1 ... c -a`; gates-off arm adds
  `PIVCO_FSE_RATIO=1.0 PIVCO_FSE_THRESH=0.51 PIVCO_FSE_MINB=8`; zstd-style
  arms via `hufadapt lit|code`.
- **Table ring** (ledger-pending; chat 2026-08-24): global vs
  `PIVCO_BLKTAB=1` vs `PIVCO_BLKTAB=8`, `PIVCO_NODYN=1`, on the
  fixed-capture L19 streams; roundtrip via `pivcohuf d` + `cmp`.
  Inter-table delta analysis parses the K=8 containers directly
  (introductions carry their 128-B length nibbles in the wire).
- **ctx config grid** (ledger §2.2): dANS arm plain `-a`; ctx arms add
  `PIVCO_CTX=1` × {`PIVCO_NOFUSE`} × {`PIVCO_CTX_FREE`}.
- **Nibble tableLog sweep** (ledger §1.2): worktree of `pr30-dynfse`,
  four builds with `-DCMAKE_C_FLAGS="-DPIVCO_FSE_NIB_TABLELOG=$L
  -falign-functions=64"` (L = 6,7,8,10); `nibbench` per input over
  Nick's datasets + silesia raw + L19 streams.
- **3-level fusion / dyn51-on-quad** (ledger §3.1/§3.2):
  `bench_hd --wscan` over datasets + o1bundle streams; oct3/quad-dyn51
  lines in the output.
