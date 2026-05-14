# Real-world dataset samples for byte-frequency distributions

Source files used to derive the 11 real-world byte-frequency
distributions registered in `bench/bench_distributions.c` (also tabled
in the [main README](../../README.md#real-world-byte-distributions)).
Kept here for reproducibility — the embedded freq tables in
`bench/dist_real_freqs.h` were generated from these exact file
contents via the `pivco_file_to_dist` tool.

## Files

| File              | Bytes     | Distinct | Entropy (b/B) | Description                                                     |
|-------------------|----------:|---------:|--------------:|-----------------------------------------------------------------|
| `cat-wiki.html`   | 1,008,695 |      197 |        5.476  | English Wikipedia article "Cat", as served HTML                  |
| `pride.txt`       |   738,046 |       96 |        4.530  | Project Gutenberg plain-text *Pride and Prejudice*               |
| `cat-image.jpg`   |   279,603 |      256 |        7.887  | Wikimedia Commons photo "Cat03.jpg" (1600×1598 JPEG)             |
| `json_api.json`   |   526,902 |       98 |        5.198  | GitHub API commits feed JSON                                     |
| `source_c.c`      |   368,435 |       97 |        4.956  | zstd `lib/compress/zstd_compress.c` (single source file)         |
| `log_apache.log`  |   500,000 |       85 |        5.497  | Apache HTTP access log sample (Logstash example)                 |
| `dna_fasta.fa`    |   500,000 |       38 |        2.080  | NCBI *E. coli* K-12 genome FASTA (truncated)                     |
| `csv_numeric.csv` |   500,000 |       54 |        3.301  | OWID CO2 dataset CSV (truncated)                                 |
| `gzip_random.gz`  |   180,726 |      256 |        7.998  | `gzip(cat-wiki.html)` — exercises the near-uniform-256 path      |
| `chinese_text.txt`|   500,000 |      151 |        5.814  | Project Gutenberg 紅樓夢 (Chinese, UTF-8, truncated)              |
| `calgary_pic`     |   513,216 |      159 |        1.210  | Calgary Corpus 1bpp CCITT test page (1728×2376, mostly-white scanned text) — real proba80-like dataset |

`min/max code len` per distribution is in the
[main README](../../README.md#real-world-byte-distributions).

## Provenance / regenerate

Original sources for the 3 oldest entries:

```sh
curl -sL -A "Mozilla/5.0" -o extras/datasets/cat-wiki.html \
    https://en.wikipedia.org/wiki/Cat
curl -sL -o extras/datasets/pride.txt \
    https://www.gutenberg.org/files/1342/1342-0.txt
curl -sL -A "Mozilla/5.0" -o extras/datasets/cat-image.jpg \
    https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
```

`gzip_random.gz` is a deterministic derivative — `gzip -k` on
`cat-wiki.html` (or any file) produces a near-uniform 256-byte
distribution suitable for the corner case:

```sh
gzip -k -9 -c extras/datasets/cat-wiki.html > extras/datasets/gzip_random.gz
```

The remaining 6 files (`json_api`, `source_c`, `log_apache`,
`dna_fasta`, `csv_numeric`, `chinese_text`) were sampled from public
sources and **truncated to ~500 KB** before checking in to keep the
repo small while preserving each distribution's tail.  Original
sources:

| File              | Notes / source hint                                                                       |
|-------------------|-------------------------------------------------------------------------------------------|
| `json_api.json`   | `curl https://api.github.com/repos/<repo>/commits` (any active repo, paginated)           |
| `source_c.c`      | `cp $ZSTD_REPO/lib/compress/zstd_compress.c .` (zstd v1.5+)                               |
| `log_apache.log`  | Logstash sample dataset (`apache_logs.txt`) — head -c 500000                              |
| `dna_fasta.fa`    | NCBI *E. coli* K-12 MG1655 (assembly U00096.3) FASTA — head -c 500000                     |
| `csv_numeric.csv` | OWID CO2 dataset (`https://github.com/owid/co2-data/raw/master/owid-co2-data.csv`)        |
| `chinese_text.txt`| Project Gutenberg 紅樓夢 (Hong Lou Meng) plain text — head -c 500000                       |
| `calgary_pic`     | Calgary Compression Corpus, file `pic` — 1bpp CCITT scanned page used in compression research since 1989.  Fetched from `https://corpus.canterbury.ac.nz/resources/calgary.tar.gz` and extracted as-is. |

The committed files in this directory are authoritative for
benchmark reproducibility; re-fetching from the original sources
will likely produce slightly different distributions (newer Cat
article revisions, evolving zstd source, etc.).

## Regenerating the embedded freq tables

```sh
cmake --build build --target pivco_file_to_dist
./build/pivco_file_to_dist -n html_wiki    extras/datasets/cat-wiki.html
./build/pivco_file_to_dist -n prose_pride  extras/datasets/pride.txt
./build/pivco_file_to_dist -n image_jpeg   extras/datasets/cat-image.jpg
./build/pivco_file_to_dist -n json_api     extras/datasets/json_api.json
./build/pivco_file_to_dist -n source_c     extras/datasets/source_c.c
./build/pivco_file_to_dist -n log_apache   extras/datasets/log_apache.log
./build/pivco_file_to_dist -n dna_fasta    extras/datasets/dna_fasta.fa
./build/pivco_file_to_dist -n csv_numeric  extras/datasets/csv_numeric.csv
./build/pivco_file_to_dist -n gzip_random  extras/datasets/gzip_random.gz
./build/pivco_file_to_dist -n chinese_text extras/datasets/chinese_text.txt
./build/pivco_file_to_dist -n calgary_pic  extras/datasets/calgary_pic
```

Paste each resulting C array into `bench/dist_real_freqs.h`.

## Why these picks

Coverage axes the 10-dataset set is designed to span:

- **alphabet width**: 38 (DNA, 4 nucleotide letters + line/header chars)
  → 256 (`image_jpeg`, `gzip_random`).
- **entropy**: 2.08 b/B (DNA, near-2-bit) → 8.00 (gzip-compressed —
  cannot be meaningfully Huffman'd).
- **tree shape**: narrow (DNA, CSV) → deep with long tail (HTML,
  JSON, prose, source code — `max_len = 15`).
- **structure**: narrow alphabet (DNA, CSV), natural-language text
  (prose, chinese), markup (html, json), code (C), logs, compressed
  (jpg / gz).

Per-dataset rationale:

- **`html_wiki`** — real-world structured markup with a heavy tail of
  rare characters (URL escapes, Unicode citation glyphs, etc.).
  Mid-entropy (~5.5 b/B): tag/attr boilerplate is highly skewed but
  inline text adds a wide moderate-frequency band.
- **`prose_pride`** — Project Gutenberg literary prose (UTF-8 plain
  text with mixed case, digits, em-dashes, smart quotes).  Lower
  entropy (~4.5 b/B) but only ~96 distinct bytes — the deeper Huffman
  tree shape is a more honest stress test than the tiny `english`
  synthetic distribution (30 symbols, lowercase only).
- **`image_jpeg`** — a real JPEG photo.  Near-uniform byte
  distribution (entropy 7.89/8) — Huffman compression cannot help
  much, but the tree shape (256 leaves at depth 6–10) exercises a
  *broad-shallow* tree shape rather than a deep one.
- **`json_api`** — hierarchical JSON with quoted strings + ASCII
  punctuation.  Heavy on `"`, `,`, space, lowercase letters; small
  alphabet (98) but `max_len = 15`.
- **`source_c`** — C source with whitespace, identifiers, operators,
  comments — characteristic mid-entropy code workload.
- **`log_apache`** — repetitive line structure, bursty timestamps and
  IPs.  Stress test for the partition path on highly skewed but
  wide-tailed data.
- **`dna_fasta`** — ACGT-dominated, very low entropy (~2 b/B), but
  carries header lines and newlines so it isn't a clean 4-symbol
  alphabet.  Tests narrow alphabet + tail.
- **`csv_numeric`** — mostly digits, decimal points, commas, newlines.
  ~54 distinct bytes, low entropy (~3.3 b/B), but has long tail
  (`max_len = 15`).
- **`gzip_random`** — gzip output is statistically near-uniform;
  cannot benefit from Huffman compression but the resulting tree is
  pathologically broad.  Exercises the full-tree flat fast path.
- **`chinese_text`** — UTF-8 multi-byte sequences shift the byte
  distribution toward high-bit values.  ~151 distinct bytes,
  entropy ~5.8 b/B.  Most challenging real-world test for tree-walk
  decoders due to deep + wide tree.
