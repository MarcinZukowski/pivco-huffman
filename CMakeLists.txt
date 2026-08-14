cmake_minimum_required(VERSION 3.16)

# Auto-select optimal compiler per platform unless explicitly overridden
# via CC=... or -DCMAKE_C_COMPILER=...  Recommendations distilled from
# results/compiler-sweep-20260507/SUMMARY.md.
#
#   - macOS arm64:  apple-clang (default — already best for our NEON path)
#   - Linux x86 + AVX-512 VBMI2 (Xeon): gcc-14  (best partition codegen)
#   - Linux x86 sans AVX-512 (Zen 3):   clang-20 (huge wins on flat unpack)
#   - Linux aarch64 (Graviton 4):       clang-20 (flat_M5/M6 +123/+136%)
if(NOT DEFINED ENV{CC} AND NOT CMAKE_C_COMPILER)
    # CMAKE_HOST_SYSTEM_PROCESSOR isn't set until project() — use uname.
    execute_process(COMMAND uname -s OUTPUT_VARIABLE _AUTO_OS   OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE _AUTO_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_AUTO_OS STREQUAL "Linux")
        if(_AUTO_ARCH STREQUAL "x86_64")
            execute_process(COMMAND grep -c avx512_vbmi2 /proc/cpuinfo
                            OUTPUT_VARIABLE _AUTO_AVX512
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            ERROR_QUIET)
            if(_AUTO_AVX512 AND NOT _AUTO_AVX512 STREQUAL "0")
                find_program(_AUTO_CC  NAMES gcc14-gcc gcc-14)
                find_program(_AUTO_CXX NAMES gcc14-c++ g++-14)
                set(_AUTO_LABEL "gcc-14 (x86 + AVX-512)")
            else()
                find_program(_AUTO_CC  NAMES clang-20)
                find_program(_AUTO_CXX NAMES clang++-20)
                set(_AUTO_LABEL "clang-20 (x86 sans AVX-512)")
            endif()
        elseif(_AUTO_ARCH STREQUAL "aarch64")
            find_program(_AUTO_CC  NAMES clang-20)
            find_program(_AUTO_CXX NAMES clang++-20)
            set(_AUTO_LABEL "clang-20 (aarch64 Linux)")
        endif()
        if(_AUTO_CC)
            set(CMAKE_C_COMPILER   "${_AUTO_CC}"  CACHE FILEPATH "" FORCE)
            if(_AUTO_CXX)
                set(CMAKE_CXX_COMPILER "${_AUTO_CXX}" CACHE FILEPATH "" FORCE)
            endif()
            message(STATUS "Auto-selected compiler: ${_AUTO_LABEL} (${_AUTO_CC})")
        endif()
    endif()
endif()

project(pivco_huffman C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
add_compile_options(-Wall -Wextra -O3)

# Built-in per-primitive profiling (PROF_TIC / PROF_TOC).  Disabled by
# default — adds RDTSC/CNTVCT pairs around hot primitives, which is
# observable on x86 (see IDEAS.md "Profiling overhead is HUGE on x86").
# Pass `-DPIVCO_PROF=ON` to cmake to enable.
option(PIVCO_PROF "Enable per-primitive PROF_TIC/PROF_TOC instrumentation" OFF)
if(PIVCO_PROF)
    add_compile_definitions(PIVCO_PROF=1)
    message(STATUS "PIVCO_PROF=1 — per-primitive prof timing enabled")
endif()

# Architecture detection.  Tricky on macOS when cmake itself runs under
# Rosetta — child processes inherit the x86_64 arch and uname/arch lie.
# sysctl is a syscall that bypasses Rosetta translation, so on Apple
# Silicon hw.optional.arm64=1 even when cmake is x86_64.
if(CMAKE_OSX_ARCHITECTURES)
    set(TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
elseif(APPLE)
    execute_process(COMMAND sysctl -n hw.optional.arm64
                    OUTPUT_VARIABLE _SYSCTL_ARM64
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_SYSCTL_ARM64 STREQUAL "1")
        set(TARGET_ARCH "arm64")
    else()
        set(TARGET_ARCH "x86_64")
    endif()
elseif(CMAKE_HOST_SYSTEM_PROCESSOR)
    set(TARGET_ARCH "${CMAKE_HOST_SYSTEM_PROCESSOR}")
else()
    execute_process(COMMAND uname -m OUTPUT_VARIABLE TARGET_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Core source files (backend-agnostic).  The per-backend codec object
# libraries (defined after the SIMD-detection block below) compile
# src/pivco_huffman_codec.c once per backend tier with the matching
# PIVCO_BACKEND_* define and get linked into the pivco_huffman library.
set(PIVCO_SOURCES
    src/huffman_table.c
    src/joint_lengths.c
    src/pivco_huffman.c
    src/pivco_instrumentation.c
    src/pivco_prof.c
    src/pivcohuf_file.c
    src/traditional_huffman.c
)

# FSE entropy coder (Yann Collet, vendored in ext/fse).  Used for
# per-internal-node bitmap compression on skewed partitions -- see
# docs/FSE-V0.md.  Sources are plain C with no SIMD deps; build-on/off via
# the PIVCO_FSE option (default ON; OFF only for ratio-no-fse compare).
option(PIVCO_FSE "Build the FSE entropy-coder integration" ON)
if(PIVCO_FSE)
    add_compile_definitions(PIVCO_HAS_FSE=1)
    list(APPEND PIVCO_SOURCES
        src/pivco_fse.c
        ext/fse/lib/fse_compress.c
        ext/fse/lib/fse_decompress.c
        ext/fse/lib/entropy_common.c
        ext/fse/lib/hist.c
        ext/fse/lib/debug.c
    )
    include_directories(ext/fse/lib)
endif()

# SIMD support
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    message(STATUS "ARM64 detected — enabling NEON")
    list(APPEND PIVCO_SOURCES
        src/pivco_huffman_neon_tables.c)
    add_compile_definitions(PIVCO_HAS_NEON=1)
    # Note: an SVE backend existed (pivco_huffman_sve.c, removed 2026-08-14
    # -- see git history) but was disabled and stale (pre-unified-codec).
    # Empirically confirmed loss vs NEON at
    # BOTH 128-bit SVE (Graviton 4 / Neoverse V2) and 256-bit SVE
    # (Graviton 3 / Neoverse V1).  Partition microbench on c7g
    # (extras/sve_probe/partition_probe.c) measured SVE-256 at 0.26x
    # NEON-128 — i.e. 4x slower, not faster.  Structural reasons:
    #   (1) svcompact only operates on 32/64-bit elements; widening
    #       uint16 codes costs more than the compress saves;
    #   (2) SVE has no efficient "8-bit mask -> 8-lane predicate" path
    #       (5+ ops vs NEON's single compress_tab[mask] load);
    #   (3) NEON vqtbl1q_u8 on V1 is 4/cycle, hard to beat.
    # AVX-512 VBMI2's vpcompressb is byte-granular and has no current
    # ARM analogue (SVE2's bdep_u8 is bit-deposit, not byte-compress).
    # See extras/sve_probe/ and IDEAS.md "SVE / SVE2 partition".
elseif(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64")
    # Three-tier x86 SIMD dispatch:
    #   AVX-512 VBMI2     (Ice Lake / Sapphire Rapids / Zen 4)
    #   AVX2 + BMI2       (Haswell / Zen 1+)         <-- new
    #   SSE4.1 + POPCNT   (Nehalem floor)
    #
    # Each tier is a strict superset of the next; AVX-512 always also
    # gets PIVCO_HAS_AVX2 + PIVCO_HAS_SSE4 so the AVX2 fast paths
    # compile in on those hosts too.
    include(CheckCCompilerFlag)
    check_c_compiler_flag("-mavx512vbmi2" COMPILER_HAS_AVX512_VBMI2)
    check_c_compiler_flag("-mavx2"        COMPILER_HAS_AVX2)
    if(COMPILER_HAS_AVX512_VBMI2)
        execute_process(COMMAND grep -c avx512_vbmi2 /proc/cpuinfo
                        OUTPUT_VARIABLE CPU_HAS_AVX512 OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
    endif()
    option(PIVCO_DISABLE_AVX512 "Force the AVX2 tier even on AVX-512 VBMI2 hosts (A/B testing)" OFF)
    if(PIVCO_DISABLE_AVX512)
        set(CPU_HAS_AVX512 "0")
    endif()
    if(COMPILER_HAS_AVX2)
        execute_process(COMMAND grep -c " avx2 " /proc/cpuinfo
                        OUTPUT_VARIABLE CPU_HAS_AVX2 OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
    endif()
    if(COMPILER_HAS_AVX512_VBMI2 AND CPU_HAS_AVX512 AND NOT CPU_HAS_AVX512 STREQUAL "0")
        message(STATUS "x86_64 detected — enabling AVX-512 VBMI2 + AVX2 + SSE4.1")
        list(APPEND PIVCO_SOURCES
            src/pivco_huffman_x86_tables.c)
        add_compile_definitions(PIVCO_HAS_AVX512=1 PIVCO_HAS_AVX2=1 PIVCO_HAS_SSE4=1)
        add_compile_options(-mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vbmi2
                            -mavx512vpopcntdq
                            -mavx2 -mbmi2 -msse4.1 -mpopcnt)
        # Every VBMI2 CPU (Icelake+, Zen4+) also has GFNI + BITALG; the
        # AVX-512 histogram primitive needs them.
        check_c_compiler_flag("-mgfni" COMPILER_HAS_GFNI)
        check_c_compiler_flag("-mavx512bitalg" COMPILER_HAS_BITALG)
        if(COMPILER_HAS_GFNI AND COMPILER_HAS_BITALG)
            add_compile_options(-mgfni -mavx512bitalg)
        endif()
    elseif(COMPILER_HAS_AVX2 AND CPU_HAS_AVX2 AND NOT CPU_HAS_AVX2 STREQUAL "0")
        message(STATUS "x86_64 detected — enabling AVX2 + BMI2 + SSE4.1")
        list(APPEND PIVCO_SOURCES
            src/pivco_huffman_x86_tables.c)
        add_compile_definitions(PIVCO_HAS_AVX2=1 PIVCO_HAS_SSE4=1)
        add_compile_options(-mavx2 -mbmi2 -msse4.1 -mpopcnt)
    else()
        message(STATUS "x86_64 detected — enabling SSE4.1 (+ POPCNT)")
        list(APPEND PIVCO_SOURCES
            src/pivco_huffman_x86_tables.c)
        add_compile_definitions(PIVCO_HAS_SSE4=1)
        add_compile_options(-msse4.1 -mpopcnt)
    endif()
else()
    message(STATUS "Architecture: ${TARGET_ARCH} — SIMD disabled")
endif()

# Per-backend codec object libraries.  Each compiles the SAME
# src/pivco_huffman_codec.c with a different PIVCO_BACKEND_* define;
# the codec includes pivco_huffman_primitives.h which routes to the
# matching primitives_<backend>.h.  Defined AFTER the SIMD-detection
# add_compile_options() calls so each object inherits -m<arch> flags
# (otherwise PIVCO_BLOCK_SIZE auto-detection would fall to the 4096
# scalar default on every non-arm64 host).
#
# Scalar is always built.  NEON is built on aarch64 hosts (Phase 3
# cutover landed 2026-05-14).  x86 is built on x86_64 hosts with
# SSE4.1 or better (Phase 4.3 cutover landed 2026-05-14, replacing
# the legacy pivco_huffman_x86.c + pivco_huffman_bu_x86.c).  AVX-512
# still builds from its legacy .c file until Phase 5.
add_library(pivco_codec_scalar OBJECT src/pivco_huffman_codec.c)
target_compile_definitions(pivco_codec_scalar PRIVATE PIVCO_BACKEND_SCALAR=1)
target_include_directories(pivco_codec_scalar PRIVATE include)

if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    add_library(pivco_codec_neon OBJECT src/pivco_huffman_codec.c)
    target_compile_definitions(pivco_codec_neon PRIVATE PIVCO_BACKEND_NEON=1)
    target_include_directories(pivco_codec_neon PRIVATE include)
endif()

if(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64")
    # pivco_codec_x86 covers SSE4.1 + AVX2 tiers (primitives_x86.h gates
    # AVX2 fast paths via PIVCO_HAS_AVX2).  AVX-512 builds add codec_avx512
    # below; the runtime dispatcher picks one or the other.  We don't fork
    # codec_x86 itself into per-ISA flavors -- one OBJECT lib serves the
    # SSE / AVX2 spectrum.
    add_library(pivco_codec_x86 OBJECT src/pivco_huffman_codec.c)
    target_compile_definitions(pivco_codec_x86 PRIVATE PIVCO_BACKEND_X86=1)
    target_include_directories(pivco_codec_x86 PRIVATE include)
endif()

if(COMPILER_HAS_AVX512_VBMI2 AND CPU_HAS_AVX512 AND NOT CPU_HAS_AVX512 STREQUAL "0")
    add_library(pivco_codec_avx512 OBJECT src/pivco_huffman_codec.c)
    target_compile_definitions(pivco_codec_avx512 PRIVATE PIVCO_BACKEND_AVX512=1)
    target_include_directories(pivco_codec_avx512 PRIVATE include)
endif()

# huff0/FSE library (for baseline comparison)
set(HUF0_SOURCES
    ext/fse/lib/entropy_common.c
    ext/fse/lib/huf_compress.c
    ext/fse/lib/huf_decompress.c
    ext/fse/lib/hist.c
    ext/fse/lib/fse_compress.c
    ext/fse/lib/fse_decompress.c
    ext/fse/lib/debug.c
)
add_library(huf0 ${HUF0_SOURCES})
target_include_directories(huf0 PUBLIC ext/fse/lib)
target_compile_options(huf0 PRIVATE -w)  # suppress warnings in external code

# Library — backend-agnostic sources plus the per-backend codec objects.
# The scalar codec is always linked.  NEON is linked on aarch64 hosts;
# x86 (SSE / AVX2) is linked on x86_64 hosts; AVX-512 is additionally
# linked on AVX-512 VBMI2 hosts (the dispatcher picks at runtime).
set(PIVCO_CODEC_OBJECTS $<TARGET_OBJECTS:pivco_codec_scalar>)
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    list(APPEND PIVCO_CODEC_OBJECTS $<TARGET_OBJECTS:pivco_codec_neon>)
elseif(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64")
    list(APPEND PIVCO_CODEC_OBJECTS $<TARGET_OBJECTS:pivco_codec_x86>)
    if(COMPILER_HAS_AVX512_VBMI2 AND CPU_HAS_AVX512 AND NOT CPU_HAS_AVX512 STREQUAL "0")
        list(APPEND PIVCO_CODEC_OBJECTS $<TARGET_OBJECTS:pivco_codec_avx512>)
    endif()
endif()
add_library(pivco_huffman
    ${PIVCO_SOURCES}
    ${PIVCO_CODEC_OBJECTS}
)
target_include_directories(pivco_huffman PUBLIC include)
if(UNIX AND NOT APPLE)
    # joint_lengths.c uses log2(); glibc keeps it in libm.  PUBLIC so every
    # executable and CMake consumer inherits it.  Raw-object consumers
    # (libpivco_huffman_local.o, phaz_local.o) carry the undefined symbol
    # and must add -lm at their final link -- TurboBench already does.
    target_link_libraries(pivco_huffman PUBLIC m)
endif()

# Relocatable, drop-in object with the vendored FiniteStateEntropy symbols
# localized (FSE_*/HUF_*/HIST_*/g_debuglevel, plus XXH/ERR/BIT/POOL/ZSTD
# defensively).  pivco's public API (pivco_*/pivcohuf_*/trad_*/init_*) stays
# global.  External consumers that also vendor FSE -- e.g. anything linking
# zstd (phaz, TurboBench) -- link this single .o instead of libpivco_huffman.a
# and the duplicate-symbol clash just goes away.  Always built (ALL).
set(PIVCO_LOCAL_OBJ ${CMAKE_BINARY_DIR}/libpivco_huffman_local.o)
if(APPLE)
    # drive the partial link through the compiler (injects -platform_version);
    # ld64's -unexported_symbol turns matched globals into locals
    add_custom_command(OUTPUT ${PIVCO_LOCAL_OBJ}
        COMMAND ${CMAKE_C_COMPILER} -nostdlib -Wl,-r -Wl,-all_load
            -Wl,-unexported_symbol,_FSE_* -Wl,-unexported_symbol,_HUF_*
            -Wl,-unexported_symbol,_HIST_* -Wl,-unexported_symbol,_XXH*
            -Wl,-unexported_symbol,_ERR_* -Wl,-unexported_symbol,_BIT_*
            -Wl,-unexported_symbol,_POOL* -Wl,-unexported_symbol,_ZSTD*
            -Wl,-unexported_symbol,_g_debuglevel
            $<TARGET_FILE:pivco_huffman> -o ${PIVCO_LOCAL_OBJ}
        DEPENDS pivco_huffman
        COMMENT "Localizing FSE/HUF symbols -> libpivco_huffman_local.o"
        VERBATIM)
else()
    # GNU binutils: partial-link the whole archive, then localize via objcopy
    add_custom_command(OUTPUT ${PIVCO_LOCAL_OBJ}
        COMMAND ld -r -o ${CMAKE_BINARY_DIR}/pivco_huffman_all.o
            --whole-archive $<TARGET_FILE:pivco_huffman> --no-whole-archive
        COMMAND objcopy -w
            --localize-symbol FSE_* --localize-symbol HUF_*
            --localize-symbol HIST_* --localize-symbol XXH*
            --localize-symbol ERR_* --localize-symbol BIT_*
            --localize-symbol POOL* --localize-symbol ZSTD*
            --localize-symbol g_debuglevel
            ${CMAKE_BINARY_DIR}/pivco_huffman_all.o ${PIVCO_LOCAL_OBJ}
        DEPENDS pivco_huffman
        COMMENT "Localizing FSE/HUF symbols -> libpivco_huffman_local.o"
        VERBATIM)
endif()
add_custom_target(pivco_huffman_local ALL DEPENDS ${PIVCO_LOCAL_OBJ})

# CLI tool: pivcohuf compress/decompress files using our format.
add_executable(pivcohuf tools/pivcohuf.c)
target_link_libraries(pivcohuf pivco_huffman)

# Minimal library-usage demo (examples/try.c): compress a file with ph + pha,
# verify roundtrip, print ratio + encode/decode MB/s.
add_executable(pivco_try examples/try.c)
target_link_libraries(pivco_try pivco_huffman)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    add_executable(pivco_huffman_tests
        test/test_main.c
        test/test_roundtrip.c
        test/test_edge_cases.c
    )
    target_link_libraries(pivco_huffman_tests pivco_huffman)
    # tests reach one internal header (pivco_fse.h) for the direct FSE
    # length-sweep roundtrip
    target_include_directories(pivco_huffman_tests PRIVATE src)
endif()

# Benchmarks
option(BUILD_BENCH "Build benchmarks" ON)
if(BUILD_BENCH)
    add_subdirectory(bench)
endif()

# HTML figure generation.  Reads `results/sweep_*.txt` and emits
# `figures/*.html`.  Manual run: `python3 extras/figures/build.py`.
# Part of the sweep workflow: after running pivco_huffman_bench(_encode)
# on each host and saving the outputs under results/, do
#     cmake --build build --target figures
# to regenerate the per-figure HTML pages from the latest data.
find_program(PYTHON3_EXECUTABLE python3)
if(PYTHON3_EXECUTABLE)
    add_custom_target(figures
        COMMAND ${PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/extras/figures/build.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating HTML figures into figures/ from results/sweep_*.txt"
        VERBATIM
    )
    # Paper-figure SVG capture pipeline.  Driven by paper/figures/fig.py
    # which reads paper/figures/figures.json (the figure registry,
    # tools + named figures + defaults) and invokes each figure's
    # tool.cli to render an .svg.  The tree_viz tool's cli uses
    # Playwright + headless Chromium; install once:
    #     pip install playwright
    #     playwright install chromium
    # fig.py prints a clear error if Playwright is missing.
    add_custom_target(figures-svg
        COMMAND ${PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/paper/figures/fig.py
                svg --all
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Rendering paper figures (paper/figures/figures.json) → paper/figures/*.svg"
        VERBATIM
    )
endif()
