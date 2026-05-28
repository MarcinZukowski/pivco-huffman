# OodleUE Build.cmake Patches

OodleUE's CMake build (smake-driven) doesn't compile the `.a64.S`
or `.nas` ASM kernels that Oodle's shipping Huffman decoder needs
to hit ryg's quoted ~1.3-1.5 cyc/sym.  Without these, you get the
portable-C fallback (~2-3x slower).

This dir snapshots the build-glue we apply to a *local clone* of
OodleUE (`ext/oodle/build/...`) to wire the ASM kernels in.  Every
file here is pivco-authored — pivco-huffman ships **no** Oodle
source and copies **nothing** out of the OodleUE tree (UE EULA,
see oodle.md).  The four files below are the only ones the recipe
installs; a clean OodleUE clone supplies all the rest (Oodle
source, the `data.patch`, smake helpers, etc).

The recipe deliberately builds **only** `liboodle-data-static.a`:

- the `test` target (RAD's `ozip` example, `test/main.cpp`) is
  disabled — it's Oodle source we don't ship or need;
- the `shared` lib is skipped — the bench links the static lib;
- `data-CMakeLists.txt` folds the static-lib declaration inline,
  so the recipe needs neither the upstream `data/static` nor
  `data/shared` smake stubs.

## Apply the patches

```sh
# Assumes ext/oodle is a symlink to your OodleUE clone (see
# extras/bench/bench_oodle_wrapper.h for the symlink-setup recipe).
cp build-CMakeLists.txt    ext/oodle/build/CMakeLists.txt
cp data-CMakeLists.txt     ext/oodle/build/data/CMakeLists.txt
cp data-Build.cmake        ext/oodle/build/data/Build.cmake
cp normalize_and_patch.sh  ext/oodle/build/normalize_and_patch.sh
cp pha_export.inc          ext/oodle/build/pha_export.inc
cp inject_export_shim.sh   ext/oodle/build/inject_export_shim.sh
chmod +x                   ext/oodle/build/normalize_and_patch.sh ext/oodle/build/inject_export_shim.sh
rm -rf ext/oodle/build-out  # force re-extract + re-patch
cmake -S ext/oodle/build -B ext/oodle/build-out -DCMAKE_BUILD_TYPE=Release
cmake --build ext/oodle/build-out -j 2   # -j 2 to avoid OOM on small EC2 instances
```

The clone's own `data.patch` (a 1-line Windows-ARM guard against
RAD source) is applied automatically if present, and harmlessly
skipped if absent — it isn't needed on Mac / Linux / Graviton.

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

Enables `ASM` (ARM64) or `ASM_NASM` (x86_64) and registers the
`.nas` extension with `ASM_NASM` (CMake auto-detects only
`.asm`/`.nasm` by default).  Then declares the static lib inline
(`PROJ_NAME/TYPE/DEF` + `include(Build.cmake)`) instead of
descending into upstream's `data/static` + `data/shared` stub
subdirs — so the recipe depends on no upstream project files.

### `data-Build.cmake`

The core patch:

- **ARM64**: adds `newlz_huff{3,6}_wide.a64.S`, `enchuff3c.a64.S`,
  `histo.a64.S` and `newlz_tans_wide.a64.S` to the static lib.
  Adds `-D__RADMACARM64__` on Apple targets so the asmlib's
  symbol-mangle picks the Mach-O underscore-prefix path.  Adds
  `-DNEWLZ_ARM64_HUFF_ASM=1` and `-DNEWLZ_ARM64_TANS_ASM=1` to the
  `.cpp` compile so `newlz_arrays_{huff,tans}.cpp` dispatch into
  the ASM kernels.
- **x86_64**: adds 6 huff3/huff6 NASM kernels (generic/BMI2/Zen2),
  3 encode/histo NASMs, and 3 tANS NASMs (generic/BMI2/BMI2-RPL).
  Adds `-DNEWLZ_X64GENERIC_HUFF_ASM=1`,
  `-DOODLE_HISTO_X64GENERIC_ASM=1` and
  `-DNEWLZ_X64GENERIC_TANS_ASM=1`.

The tANS kernel uses the same `_wide` (Apple M1-scheduled) ARM64
variant as huff, and the generic/BMI2/RaptorLake x86 trio that
`newlz_arrays_tans.cpp` dispatches between by runtime CPU feature.
Wired in 2026-05-22; before this the lib used the portable-C tANS
fallback (~15-30% slower than the ASM kernel on M4).
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

### `pha_export.inc` + `inject_export_shim.sh`

Prebuilt-table export shim.  Oodle's `newlz_get_array_tans` fuses
header-parse + decode-table-build + decode into one call, so it
can't be timed apples-to-apples against ph's *static* tables (no
per-call build).  `pha_export.inc` (pivco-authored) adds two
`extern "C"` wrappers — `pha_tans_build_decoder` (build once) and
`pha_tans_decode_prebuilt` (decode many) — that split the fused
path.  The build half is pure public API (`newlz_tans.h`); the
decode half mirrors the per-call state setup + kernel dispatch
(it must, since Oodle exports no decode-with-prebuilt-table entry).

`inject_export_shim.sh` (run from `data/CMakeLists.txt` via
`execute_process` at configure time) copies the `.inc` next to
`newlz_arrays_tans.cpp` and splices an `#include "pha_export.inc"`
immediately before the file's final `OODLE_NS_END` — placing it
inside `namespace oo2`, same translation unit as `KrakenTansState`
and the static `tansx2_*` kernels.  It is idempotent.  This keeps
the repo free of any Oodle source: we ship only our own `.inc` +
injector; the `#include` is spliced into the extracted clone and
never committed.  (A unified diff would have embedded Oodle context
lines — the script avoids that.)

Serialization of a built decoder needs **no** wrapper: the
`newlz_tans_Decoder` block holds one base-relative self-pointer, so
store the `Decoder_Size()` bytes, and on restore `memcpy` + call
`newlz_tans_Decoder_Init()` again to re-derive the pointer (the
table data is left untouched).  Verified: a block memcpy'd to a new
address + re-`Init` decodes identically to the original.

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

## Shipped lib cross-check (`OODLE_LIB_VARIANT`)

OodleUE also ships RAD's own prebuilt platform libs under
`Sdks/2.9.16/lib/{Mac,Linux,LinuxArm64,...}`.  These export the
internal `newlz_{get,put}_array_{huff,tans}` entry points (verified
with `nm`), so the bench can link them directly:

```sh
cmake -B build -DOODLE_LIB_VARIANT=shipped   # RAD's prebuilt lib
cmake -B build -DOODLE_LIB_VARIANT=source    # our build-out (default)
```

On M4 the from-source ASM build matches the shipped Mac lib within
run-to-run noise for both huff and tANS — confirming the ASM wiring
above reproduces RAD's production decode kernels.  Caveat: the
Linux x86 SDK ships only a `_dbg.a` static (release is `.so.9`), so
the `shipped` x86 static path points at the debug archive.
