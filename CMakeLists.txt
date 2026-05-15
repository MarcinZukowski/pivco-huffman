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

# lemire/counters — hardware counter access (cycles, IPC, branch/cache miss).
# Used by the *_cnt benchmarks under extras/.  Pulled via CPM so a fresh
# checkout does not need a manual git clone.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)
set(COUNTERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(COUNTERS_INSTALL OFF CACHE BOOL "" FORCE)
CPMAddPackage(
    NAME counters
    GITHUB_REPOSITORY lemire/counters
    VERSION 4.1.0
)

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
    src/pivco_huffman.c
    src/pivco_instrumentation.c
    src/pivco_prof.c
    src/pivcohuf_file.c
    src/traditional_huffman.c
)

# FSE entropy coder (Yann Collet, vendored in ext/fse).  Used for
# per-internal-node bitmap compression on skewed partitions -- see
# FSE-V0.md.  Sources are plain C with no SIMD deps; build-on/off via
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
    # Note: SVE backend exists (pivco_huffman_sve.c) but svcompact at
    # 128-bit SVE (Graviton 4) is slower than NEON TBL because it only
    # handles 4 × uint32 per instruction vs TBL's 8 × uint16. SVE would
    # help at 256-bit+ vector lengths (e.g. Fujitsu A64FX).
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

# Brotli Huffman primitives (decoder + encoder).  Vendored under
# ext/brotli/ via `git clone --depth 1 https://github.com/google/brotli.git`.
# We only build the minimal Huffman subset; the rest of brotli (LZ77,
# bit stream framing) is unused.
set(BROTLI_HUFFMAN_SOURCES
    ext/brotli/c/dec/huffman.c
    ext/brotli/c/dec/bit_reader.c
    ext/brotli/c/enc/entropy_encode.c
    ext/brotli/c/common/platform.c
    ext/brotli/c/common/constants.c
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ext/brotli/c/dec/huffman.c)
    add_library(brotli_huffman ${BROTLI_HUFFMAN_SOURCES})
    target_include_directories(brotli_huffman PUBLIC
        ext/brotli/c/include
        ext/brotli/c/common
        ext/brotli/c/dec
        ext/brotli/c/enc)
    target_compile_options(brotli_huffman PRIVATE -w)
    set(HAVE_BROTLI_HUFFMAN ON)
else()
    message(STATUS "Brotli sources not found at ext/brotli; bench_brotli "
                   "will be skipped.  Clone with: "
                   "git clone --depth 1 https://github.com/google/brotli.git ext/brotli")
    set(HAVE_BROTLI_HUFFMAN OFF)
endif()

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

# CLI tool: pivcohuf compress/decompress files using our format.
add_executable(pivcohuf tools/pivcohuf.c)
target_link_libraries(pivcohuf pivco_huffman)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    add_executable(pivco_huffman_tests
        test/test_main.c
        test/test_roundtrip.c
        test/test_edge_cases.c
    )
    target_link_libraries(pivco_huffman_tests pivco_huffman)
endif()

# Benchmarks
option(BUILD_BENCH "Build benchmarks" ON)
if(BUILD_BENCH)
    add_executable(pivco_huffman_bench
        bench/bench_main.c
        bench/bench_distributions.c
        bench/bench_rans.cpp
    )
    target_link_libraries(pivco_huffman_bench pivco_huffman huf0 m)
    target_include_directories(pivco_huffman_bench PRIVATE ext/fse/lib ext/ryg_rans)

    add_executable(pivco_huffman_bench_encode
        bench/bench_encode.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_huffman_bench_encode pivco_huffman huf0 m)
    target_include_directories(pivco_huffman_bench_encode PRIVATE ext/fse/lib)

    # Brotli vs pivco vs huf0_x1 comparison bench.  Standalone harness:
    # encode + decode timing across the MAIN distribution set.  Skipped
    # if ext/brotli/ is not present.  See IDEAS.md 2026-05-12 for the
    # negative-result verdict; this binary kept under extras/ so the
    # comparison stays reproducible if libjxl/brotli ever change.
    if(HAVE_BROTLI_HUFFMAN)
        add_executable(pivco_bench_brotli
            extras/bench_brotli.c
            bench/bench_distributions.c
        )
        target_link_libraries(pivco_bench_brotli
            pivco_huffman huf0 brotli_huffman m)
        target_include_directories(pivco_bench_brotli PRIVATE
            ext/fse/lib
            ext/brotli/c/include
            ext/brotli/c/common
            ext/brotli/c/dec
            ext/brotli/c/enc)
    endif()

    if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
        add_executable(pivco_encode_split_micro
            extras/bench_encode_split.c
        )
        target_link_libraries(pivco_encode_split_micro pivco_huffman m)
        target_include_directories(pivco_encode_split_micro PRIVATE src)

        add_executable(pivco_enc_init_micro
            extras/bench_enc_init.c
        )
        target_link_libraries(pivco_enc_init_micro pivco_huffman m)
    endif()

    # Partition u8 vs u16 microbench: works on any tier (each ISA is
    # #ifdef-gated inside the source).
    add_executable(pivco_partition_u8_vs_u16
        extras/bench_partition_u8_vs_u16.c
    )
    target_link_libraries(pivco_partition_u8_vs_u16 pivco_huffman m)
    target_include_directories(pivco_partition_u8_vs_u16 PRIVATE src)

    # K_right wire-format storage analyzer (still useful after the
    # K_right format landed -- analyzes per-dist storage cost).
    add_executable(pivco_bench_kr_storage
        extras/bench_kr_storage.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_bench_kr_storage pivco_huffman huf0 m)
    target_include_directories(pivco_bench_kr_storage PRIVATE ext/fse/lib)

    add_executable(pivco_huffman_profile_english
        bench/bench_profile_english.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_huffman_profile_english pivco_huffman m)

    add_executable(pivco_one_leaf_stats
        bench/bench_one_leaf_stats.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_one_leaf_stats pivco_huffman m)

    add_executable(pivco_multi_stage_stats
        bench/bench_multi_stage_stats.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_multi_stage_stats pivco_huffman m)

    add_executable(pivco_flat_subtree_stats
        extras/bench_flat_subtree_stats.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_flat_subtree_stats pivco_huffman m)

    # Offline TANS-routing ratio bound: real files (per-block) or
    # MAIN bench distributions (per-histogram).
    add_executable(pivco_tans_bound
        extras/bench_tans_bound.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_tans_bound pivco_huffman m)

    # 4-way codec comparison: ph / phe / huf0 / fse on the same data.
    if(PIVCO_FSE)
        add_executable(pivco_bench_4way
            extras/bench_4way.c
            bench/bench_distributions.c
        )
        target_link_libraries(pivco_bench_4way pivco_huffman huf0 m)
        target_include_directories(pivco_bench_4way PRIVATE ext/fse/lib)

        # Per-FSE-table-id usage histogram dumper.
        add_executable(pivco_bench_fse_table_use
            extras/bench_fse_table_use.c
        )
        target_link_libraries(pivco_bench_fse_table_use pivco_huffman m)
    endif()

    add_executable(pivco_dist_stats
        extras/bench_dist_stats.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_dist_stats pivco_huffman m)

    add_executable(pivco_partition_skew
        extras/bench_partition_skew.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_partition_skew pivco_huffman m)

    add_executable(pivco_dense_runs
        extras/bench_dense_runs.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_dense_runs pivco_huffman m)

    add_executable(pivco_mask_pair_degeneracy
        extras/bench_mask_pair_degeneracy.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_mask_pair_degeneracy pivco_huffman m)

    add_executable(pivco_partition_micro
        extras/bench_partition_micro.c
    )
    target_link_libraries(pivco_partition_micro pivco_huffman m)
    target_include_directories(pivco_partition_micro PRIVATE src)

    # Counter-instrumented partition microbench (NEON only — kernels are
    # NEON intrinsics).  Reports cycles/instructions/IPC/cache-miss/branch-
    # miss per call.  Needs sudo on macOS or perf_event_paranoid=0 on Linux.
    if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
        add_executable(pivco_partition_micro_cnt
            extras/bench_partition_micro_cnt.cpp
        )
        target_link_libraries(pivco_partition_micro_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_partition_micro_cnt PRIVATE src)

        add_executable(pivco_lemire_compare
            extras/bench_lemire_compare.cpp
        )
        target_link_libraries(pivco_lemire_compare
            pivco_huffman counters::counters m)
        target_include_directories(pivco_lemire_compare PRIVATE src)

        add_executable(pivco_scatter_micro_cnt
            extras/bench_scatter_micro_cnt.cpp
        )
        target_link_libraries(pivco_scatter_micro_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_scatter_micro_cnt PRIVATE src)

        add_executable(pivco_fusion_micro_cnt
            extras/bench_fusion_micro_cnt.cpp
        )
        target_link_libraries(pivco_fusion_micro_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_fusion_micro_cnt PRIVATE src)

        add_executable(pivco_scatter_split_cnt
            extras/bench_scatter_split_cnt.cpp
        )
        target_link_libraries(pivco_scatter_split_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_scatter_split_cnt PRIVATE src)

        add_executable(pivco_p8_dual_cnt
            extras/bench_p8_dual_cnt.cpp
        )
        target_link_libraries(pivco_p8_dual_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_p8_dual_cnt PRIVATE src)

        add_executable(pivco_fusion_v2_cnt
            extras/bench_fusion_v2_cnt.cpp
        )
        target_link_libraries(pivco_fusion_v2_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_fusion_v2_cnt PRIVATE src)

        add_executable(pivco_fusion_v3_cnt
            extras/bench_fusion_v3_cnt.cpp
        )
        target_link_libraries(pivco_fusion_v3_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_fusion_v3_cnt PRIVATE src)

        add_executable(pivco_fusion_v4_cnt
            extras/bench_fusion_v4_cnt.cpp
        )
        target_link_libraries(pivco_fusion_v4_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_fusion_v4_cnt PRIVATE src)

        add_executable(pivco_fusion_v5_real_kernel_cnt
            extras/bench_fusion_v5_real_kernel_cnt.cpp
        )
        target_link_libraries(pivco_fusion_v5_real_kernel_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_fusion_v5_real_kernel_cnt PRIVATE src)

        add_executable(pivco_primitives_neon_cnt
            extras/bench_primitives_neon_cnt.cpp
        )
        target_link_libraries(pivco_primitives_neon_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_primitives_neon_cnt PRIVATE src)

        add_executable(pivco_tree_merge_neon_cnt
            extras/bench_tree_merge_neon_cnt.cpp
        )
        target_link_libraries(pivco_tree_merge_neon_cnt
            counters::counters m)
        target_include_directories(pivco_tree_merge_neon_cnt PRIVATE src)
    endif()

    # x86 microbenches (SSE4.1 + AVX-512 variants).  Built whenever the
    # corresponding kernels are compiled.
    if(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64")
        add_executable(pivco_micro_sse_cnt
            extras/bench_micro_sse_cnt.cpp
        )
        target_link_libraries(pivco_micro_sse_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_micro_sse_cnt PRIVATE src)

        if(COMPILER_HAS_AVX512_VBMI2 AND CPU_HAS_AVX512
                                    AND NOT CPU_HAS_AVX512 STREQUAL "0")
            add_executable(pivco_micro_avx512_cnt
                extras/bench_micro_avx512_cnt.cpp
            )
            target_link_libraries(pivco_micro_avx512_cnt
                pivco_huffman counters::counters m)
            target_include_directories(pivco_micro_avx512_cnt PRIVATE src)

            add_executable(pivco_fusion_v3_avx512_cnt
                extras/bench_fusion_v3_avx512_cnt.cpp
            )
            target_link_libraries(pivco_fusion_v3_avx512_cnt
                pivco_huffman counters::counters m)
            target_include_directories(pivco_fusion_v3_avx512_cnt PRIVATE src)

            add_executable(pivco_fusion_v4_avx512_cnt
                extras/bench_fusion_v4_avx512_cnt.cpp
            )
            target_link_libraries(pivco_fusion_v4_avx512_cnt
                pivco_huffman counters::counters m)
            target_include_directories(pivco_fusion_v4_avx512_cnt PRIVATE src)
        endif()

        add_executable(pivco_fusion_v3_sse_cnt
            extras/bench_fusion_v3_sse_cnt.cpp
        )
        target_link_libraries(pivco_fusion_v3_sse_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_fusion_v3_sse_cnt PRIVATE src)

        add_executable(pivco_fusion_v4_sse_cnt
            extras/bench_fusion_v4_sse_cnt.cpp
        )
        target_link_libraries(pivco_fusion_v4_sse_cnt
            pivco_huffman counters::counters m)
        target_include_directories(pivco_fusion_v4_sse_cnt PRIVATE src)

        add_executable(pivco_primitives_x86_cnt
            extras/bench_primitives_x86_cnt.cpp
        )
        target_compile_options(pivco_primitives_x86_cnt PRIVATE -mbmi2 -mavx2)
        target_link_libraries(pivco_primitives_x86_cnt
            counters::counters m)
        target_include_directories(pivco_primitives_x86_cnt PRIVATE src)

        add_executable(pivco_tree_merge_x86_cnt
            extras/bench_tree_merge_x86_cnt.cpp
        )
        target_compile_options(pivco_tree_merge_x86_cnt PRIVATE
            -msse4.1 -mavx2 -mbmi2)
        if(COMPILER_HAS_AVX512_VBMI2 AND CPU_HAS_AVX512
                                    AND NOT CPU_HAS_AVX512 STREQUAL "0")
            target_compile_options(pivco_tree_merge_x86_cnt PRIVATE
                -mavx512f -mavx512bw -mavx512vbmi -mavx512vbmi2)
        endif()
        target_link_libraries(pivco_tree_merge_x86_cnt
            counters::counters m)
        target_include_directories(pivco_tree_merge_x86_cnt PRIVATE src)
    endif()

    if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
        add_executable(pivco_bench_coalesce extras/bench_coalesce.c)
    endif()
    # bench_bu_decoder{,_x86} retired with the TD decoder (2026-05-14) --
    # they existed to A/B BU against TD; BU has been the production
    # decoder for weeks.

    add_executable(pivco_bench_multicore
        extras/bench_multicore.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_bench_multicore pivco_huffman huf0 pthread m)
    target_include_directories(pivco_bench_multicore PRIVATE ext/fse/lib)

    add_executable(pivco_flat_optimal
        extras/bench_flat_optimal.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_flat_optimal pivco_huffman m)

    add_executable(pivco_tree_viz
        extras/bench_tree_viz.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_tree_viz pivco_huffman m)

    add_executable(pivco_file_to_dist
        extras/file_to_dist.c
    )
    target_link_libraries(pivco_file_to_dist m)
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
endif()
