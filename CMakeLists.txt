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

# Core source files
set(PIVCO_SOURCES
    src/huffman_table.c
    src/pivco_huffman.c
    src/pivco_huffman_scalar.c
    src/pivco_instrumentation.c
    src/pivco_prof.c
    src/traditional_huffman.c
)

# SIMD support
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    message(STATUS "ARM64 detected — enabling NEON")
    list(APPEND PIVCO_SOURCES src/pivco_huffman_neon.c src/pivco_huffman_neon_prefix.c src/pivco_huffman_bu_neon.c)
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
        list(APPEND PIVCO_SOURCES src/pivco_huffman_avx512.c src/pivco_huffman_x86.c src/pivco_huffman_bu_x86.c)
        add_compile_definitions(PIVCO_HAS_AVX512=1 PIVCO_HAS_AVX2=1 PIVCO_HAS_SSE4=1)
        add_compile_options(-mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vbmi2
                            -mavx512vpopcntdq
                            -mavx2 -mbmi2 -msse4.1 -mpopcnt)
    elseif(COMPILER_HAS_AVX2 AND CPU_HAS_AVX2 AND NOT CPU_HAS_AVX2 STREQUAL "0")
        message(STATUS "x86_64 detected — enabling AVX2 + BMI2 + SSE4.1")
        list(APPEND PIVCO_SOURCES src/pivco_huffman_x86.c src/pivco_huffman_bu_x86.c)
        add_compile_definitions(PIVCO_HAS_AVX2=1 PIVCO_HAS_SSE4=1)
        add_compile_options(-mavx2 -mbmi2 -msse4.1 -mpopcnt)
    else()
        message(STATUS "x86_64 detected — enabling SSE4.1 (+ POPCNT)")
        list(APPEND PIVCO_SOURCES src/pivco_huffman_x86.c src/pivco_huffman_bu_x86.c)
        add_compile_definitions(PIVCO_HAS_SSE4=1)
        add_compile_options(-msse4.1 -mpopcnt)
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

# Library
add_library(pivco_huffman ${PIVCO_SOURCES})
target_include_directories(pivco_huffman PUBLIC include)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    add_executable(pivco_huffman_tests
        test/test_main.c
        test/test_roundtrip.c
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

    # Prefix profiler is NEON-only (includes <arm_neon.h> directly).
    if(TARGET_ARCH STREQUAL "aarch64")
        add_executable(pivco_prefix_profile
            bench/bench_prefix_profile.c
            bench/bench_distributions.c
        )
        target_link_libraries(pivco_prefix_profile pivco_huffman m)
    endif()

    add_executable(pivco_flat_subtree_stats
        extras/bench_flat_subtree_stats.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_flat_subtree_stats pivco_huffman m)

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

        add_executable(pivco_bench_bu_decoder
            extras/bench_bu_decoder.c
            bench/bench_distributions.c
        )
        target_link_libraries(pivco_bench_bu_decoder pivco_huffman m)
    endif()

    if(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64")
        add_executable(pivco_bench_bu_decoder_x86
            extras/bench_bu_decoder_x86.c
            bench/bench_distributions.c
        )
        target_link_libraries(pivco_bench_bu_decoder_x86 pivco_huffman m)
    endif()

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
