cmake_minimum_required(VERSION 3.16)
project(pivco_huffman C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
add_compile_options(-Wall -Wextra -O3)

# Architecture detection
if(CMAKE_OSX_ARCHITECTURES)
    set(TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
else()
    execute_process(COMMAND uname -m OUTPUT_VARIABLE TARGET_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Core source files
set(PIVCO_SOURCES
    src/huffman_table.c
    src/pivco_huffman.c
    src/pivco_huffman_scalar.c
    src/traditional_huffman.c
)

# SIMD support
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    message(STATUS "ARM64 detected — enabling NEON")
    list(APPEND PIVCO_SOURCES src/pivco_huffman_neon.c)
    add_compile_definitions(PIVCO_HAS_NEON=1)
    # Note: SVE backend exists (pivco_huffman_sve.c) but svcompact at
    # 128-bit SVE (Graviton 4) is slower than NEON TBL because it only
    # handles 4 × uint32 per instruction vs TBL's 8 × uint16. SVE would
    # help at 256-bit+ vector lengths (e.g. Fujitsu A64FX).
elseif(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64")
    # Check for AVX-512 VBMI2 (vpcompressw) — compiler support AND CPU support
    include(CheckCCompilerFlag)
    check_c_compiler_flag("-mavx512vbmi2" COMPILER_HAS_AVX512_VBMI2)
    if(COMPILER_HAS_AVX512_VBMI2)
        # Check if the CPU actually has avx512vbmi2
        execute_process(COMMAND grep -c avx512_vbmi2 /proc/cpuinfo
                        OUTPUT_VARIABLE CPU_HAS_AVX512 OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
    endif()
    if(COMPILER_HAS_AVX512_VBMI2 AND CPU_HAS_AVX512 AND NOT CPU_HAS_AVX512 STREQUAL "0")
        message(STATUS "x86_64 detected — enabling AVX-512 VBMI2 + SSE4.1")
        list(APPEND PIVCO_SOURCES src/pivco_huffman_avx512.c src/pivco_huffman_x86.c)
        add_compile_definitions(PIVCO_HAS_AVX512=1 PIVCO_HAS_SSE4=1)
        add_compile_options(-mavx512f -mavx512bw -mavx512vl -mavx512vbmi2 -msse4.1 -mpopcnt)
    else()
        message(STATUS "x86_64 detected — enabling SSE4.1")
        list(APPEND PIVCO_SOURCES src/pivco_huffman_x86.c)
        add_compile_definitions(PIVCO_HAS_SSE4=1)
        add_compile_options(-msse4.1)
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
    target_link_libraries(pivco_huffman_bench pivco_huffman huf0)
    target_include_directories(pivco_huffman_bench PRIVATE ext/fse/lib ext/ryg_rans)
endif()
