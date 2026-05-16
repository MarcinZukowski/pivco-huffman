# OodleUE Build.cmake Patches

OodleUE's CMake build (smake-driven) doesn't compile the `.a64.S`
or `.nas` ASM kernels that Oodle's shipping Huffman decoder needs
to hit ryg's quoted ~1.3-1.5 cyc/sym.  Without these, you get the
portable-C fallback (~2-3x slower).

This dir snapshots the patches we applied to a *local clone* of
OodleUE (`ext/oodle/build/...`) to wire the ASM kernels in.  The
patches are kept here as a reference — pivco-huffman does **not**
ship Oodle source (UE EULA, see oodle.md), only this build-glue.

## Apply the patches

```sh
# Assumes ext/oodle is a symlink to your OodleUE clone (see
# extras/bench_oodle_wrapper.h for the symlink-setup recipe).
cp build-CMakeLists.txt    ext/oodle/build/CMakeLists.txt
cp data-CMakeLists.txt     ext/oodle/build/data/CMakeLists.txt
cp data-Build.cmake        ext/oodle/build/data/Build.cmake
cp normalize_and_patch.sh  ext/oodle/build/normalize_and_patch.sh
chmod +x                   ext/oodle/build/normalize_and_patch.sh
rm -rf ext/oodle/build-out  # force re-extract + re-patch
cmake -S ext/oodle/build -B ext/oodle/build-out -DCMAKE_BUILD_TYPE=Release
cmake --build ext/oodle/build-out -j 2   # -j 2 to avoid OOM on small EC2 instances
```

## What each patch does

### `build-CMakeLists.txt`

1. Disables the `oodle_network` FetchContent (older CMake fails on
   absolute-path URLs and we don't need NetworkCompression for
   the Huffman bench).
2. Replaces `git apply` patching with the `normalize_and_patch.sh`
   wrapper — Linux unzip preserves CRLF line endings stored in
   Oodle's source ZIP and `git apply` is intolerant of mixed
   line endings even with `--ignore-space-change`.

### `data-CMakeLists.txt`

Enables `ASM` (ARM64) or `ASM_NASM` (x86_64) at the parent scope
*before* the static/shared lib subdirs run.  Also registers the
`.nas` extension with `ASM_NASM` since CMake auto-detects only
`.asm`/`.nasm` by default.

### `data-Build.cmake`

The core patch:

- **ARM64**: adds `newlz_huff{3,6}_wide.a64.S`, `enchuff3c.a64.S`,
  `histo.a64.S` to the static lib.  Adds `-D__RADMACARM64__` on
  Apple targets so the asmlib's symbol-mangle picks the Mach-O
  underscore-prefix path.  Adds `-DNEWLZ_ARM64_HUFF_ASM=1` to the
  `.cpp` compile so `newlz_arrays_huff.cpp` dispatches into the
  ASM kernels.
- **x86_64**: adds 6 huff3/huff6 NASM kernels (generic/BMI2/Zen2),
  3 encode/histo NASMs.  Adds `-DNEWLZ_X64GENERIC_HUFF_ASM=1` +
  `-DOODLE_HISTO_X64GENERIC_ASM=1`.
- Replaces `s_set_arch(AVX2)` (which leaks `-march=x86-64-v3` to
  ASM_NASM — NASM doesn't understand `-march`) with a generator-
  expression-gated version that only applies `-march` to C/CXX.
- Adds `.nas` files via `target_sources()` + explicit
  `LANGUAGE ASM_NASM` per-file because smake's `s_add_file_force`
  doesn't survive CMake's source-language auto-detection on `.nas`.

### `normalize_and_patch.sh`

Strips CRLF→LF in the extracted source tree before applying
`data.patch`.  Needed because Linux `unzip` preserves the CRLF
line endings stored in Oodle's source ZIP (macOS `unzip` strips
them automatically).

## Per-host nasm install

x86 hosts need `nasm` installed:

```sh
sudo yum install -y nasm   # Amazon Linux 2023
sudo apt-get install -y nasm  # Debian/Ubuntu
brew install nasm          # macOS (only if you build NASM kernels there)
```

## Variant selection

We pick `newlz_huff{3,6}_wide.a64.S` (Apple M1-scheduled) on
ARM64 — works well on both M4 and Graviton 4 (Neoverse V2).
Alternatives: `_cortex_a57.a64.S` (older mobile), `_cortex_a78.a64.S`
(newer big cores).  Switch by editing the `s_add_file_force` lines
in `data-Build.cmake`.

x86 NASM dispatches at runtime via `rrCPUx86_feature_present` —
no variant choice needed at build time.
