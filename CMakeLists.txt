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

# Core source files (backend-agnostic).  The codec and the per-backend
# primitive-table object libraries (defined after the SIMD-detection
# block below) get linked into the pivco_huffman library.
set(PIVCO_SOURCES
    src/huffman_table.c
    src/joint_lengths.c
    src/pivco_cpu.c
    src/pivco_huffman.c
    src/pivco_instrumentation.c
    src/pivco_prof.c
    src/pivcohuf_file.c
    src/shake_lengths.c
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
        src/pivco_k1.c
        src/pivco_k2.c
        src/pivco_dhuf0.c
        src/pivco_nph.c
        ext/fse/lib/fse_compress.c
        ext/fse/lib/fse_decompress.c
        ext/fse/lib/huf_compress.c
        ext/fse/lib/huf_decompress.c
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
    # GFNI + BITALG are part of the AVX-512 floor (every VBMI2 CPU has them).
    check_c_compiler_flag("-mgfni" COMPILER_HAS_GFNI)
    check_c_compiler_flag("-mavx512bitalg" COMPILER_HAS_BITALG)
    if(NOT COMPILER_HAS_GFNI OR NOT COMPILER_HAS_BITALG)
        set(COMPILER_HAS_AVX512_VBMI2 OFF)
    endif()
    check_c_compiler_flag("-mavx2"        COMPILER_HAS_AVX2)
    # The build host's AVX-512 support, only for the extras benches that
    # are built for the host.
    if(COMPILER_HAS_AVX512_VBMI2)
        execute_process(COMMAND grep -c avx512_vbmi2 /proc/cpuinfo
                        OUTPUT_VARIABLE CPU_HAS_AVX512 OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
    endif()
    set(PIVCO_AVX2_FLAGS -mavx2 -mbmi2 -msse4.1 -mpopcnt)
    set(PIVCO_AVX512_FLAGS -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vbmi2
                           -mavx512vpopcntdq -mgfni -mavx512bitalg ${PIVCO_AVX2_FLAGS})
    # The SSE4.1 floor for everything; AVX2 and AVX-512 flags only on
    # their own codec and primitive-table objects.  The CPU is checked at
    # runtime (pivco_huffman.c), not here.
    message(STATUS "x86_64 runtime dispatch — SSE4.1 floor + every tier the compiler supports")
    list(APPEND PIVCO_SOURCES
        src/pivco_huffman_x86_tables.c)
    add_compile_definitions(PIVCO_HAS_SSE4=1)
    add_compile_options(-msse4.1 -mpopcnt)
    if(COMPILER_HAS_AVX2)
        set(PIVCO_TIER_AVX2 ON)
        add_compile_definitions(PIVCO_BUILT_AVX2=1)
    endif()
    if(COMPILER_HAS_AVX512_VBMI2)
        set(PIVCO_TIER_AVX512 ON)
        add_compile_definitions(PIVCO_HAS_AVX512=1)
    endif()
else()
    message(STATUS "Architecture: ${TARGET_ARCH} — SIMD disabled")
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

# The codec and the primitive tables.  The codec (pivco_huffman_codec.c)
# is compiled once, at the build's floor, and calls every primitive
# through the table picked at runtime (src/prims/tables.h).  Each backend
# compiles pivco_prims_table.c once with its own flags and exports its
# tables (the AVX-512 compile carries two).
set(PIVCO_CODEC_OBJECTS)
add_library(pivco_codec_prims OBJECT src/pivco_huffman_codec.c)
target_compile_definitions(pivco_codec_prims PRIVATE
    PIVCO_BACKEND_SUFFIX=prims PIVCO_PRIMS_INDIRECT=1)
target_include_directories(pivco_codec_prims PRIVATE include)
list(APPEND PIVCO_CODEC_OBJECTS $<TARGET_OBJECTS:pivco_codec_prims>)
# name|backend|flags (base, avx2, avx512)|extra defines (comma-separated)
set(PIVCO_TABLE_COMPILES "scalar|SCALAR|base|")
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    list(APPEND PIVCO_TABLE_COMPILES "neon|NEON|base|")
elseif(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64")
    list(APPEND PIVCO_TABLE_COMPILES "x86|X86|base|")
    if(PIVCO_TIER_AVX2)
        list(APPEND PIVCO_TABLE_COMPILES "x86_avx2|X86|avx2|PIVCO_HAS_AVX2=1")
        list(APPEND PIVCO_TABLE_COMPILES "x86_avx2_intel|X86|avx2|PIVCO_HAS_AVX2=1,PIVCO_X86_INTEL=1")
    endif()
    if(PIVCO_TIER_AVX512)
        list(APPEND PIVCO_TABLE_COMPILES "avx512|AVX512|avx512|PIVCO_HAS_AVX2=1")
    endif()
endif()
foreach(v ${PIVCO_TABLE_COMPILES})
    string(REPLACE "|" ";" vl "${v}")
    list(GET vl 0 vname)
    list(GET vl 1 vback)
    list(GET vl 2 vtier)
    list(GET vl 3 vdefs)
    string(REPLACE "," ";" vdefs "${vdefs}")
    set(vflags)
    if(vtier STREQUAL "avx2")
        set(vflags ${PIVCO_AVX2_FLAGS})
    elseif(vtier STREQUAL "avx512")
        set(vflags ${PIVCO_AVX512_FLAGS})
    endif()
    add_library(pivco_prims_${vname} OBJECT src/pivco_prims_table.c)
    target_compile_definitions(pivco_prims_${vname} PRIVATE
        PIVCO_BACKEND_${vback}=1 PIVCO_BACKEND_SUFFIX=${vname} ${vdefs})
    target_compile_options(pivco_prims_${vname} PRIVATE ${vflags})
    target_include_directories(pivco_prims_${vname} PRIVATE include)
    list(APPEND PIVCO_CODEC_OBJECTS $<TARGET_OBJECTS:pivco_prims_${vname}>)
endforeach()
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
