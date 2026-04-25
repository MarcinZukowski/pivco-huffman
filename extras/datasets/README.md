# Real-world dataset samples for byte-frequency distributions

Source files used to derive the `html_wiki`, `prose_pride`, and
`image_jpeg` distributions in `bench/bench_distributions.c`.  Kept here
for reproducibility — the freq tables were generated from these exact
file contents via the `pivco_file_to_dist` tool.

## Files

| File | Bytes | Distinct | Entropy | Description |
|---|--:|--:|--:|---|
| `cat-wiki.html`  | 1,008,695 | 197 | 5.476 b/B | English Wikipedia article "Cat", as served HTML |
| `pride.txt`      |   738,046 |  96 | 4.530 b/B | Project Gutenberg plain-text *Pride and Prejudice* |
| `cat-image.jpg`  |   279,603 | 256 | 7.887 b/B | Wikimedia Commons photo "Cat03.jpg" (1600×1598 JPEG) |

## Provenance / regenerate

```sh
curl -sL -A "Mozilla/5.0" -o extras/datasets/cat-wiki.html \
    https://en.wikipedia.org/wiki/Cat
curl -sL -o extras/datasets/pride.txt \
    https://www.gutenberg.org/files/1342/1342-0.txt
curl -sL -A "Mozilla/5.0" -o extras/datasets/cat-image.jpg \
    https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
```

Then to regenerate the embedded freq tables:

```sh
cmake --build build-release --target pivco_file_to_dist
./build-release/pivco_file_to_dist -n html_wiki   extras/datasets/cat-wiki.html
./build-release/pivco_file_to_dist -n prose_pride extras/datasets/pride.txt
./build-release/pivco_file_to_dist -n image_jpeg  extras/datasets/cat-image.jpg
```

Paste the resulting C arrays into `bench/bench_distributions.c`.

## Why these picks

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
  much, but the tree shape (very deep, 256 leaves at depth 8 or 9)
  exercises the *worst-case* path length for tree-walk decoders.
