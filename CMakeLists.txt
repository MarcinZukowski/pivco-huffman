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

# NEON support for ARM64
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    message(STATUS "ARM64 detected — enabling NEON")
    list(APPEND PIVCO_SOURCES src/pivco_huffman_neon.c)
    add_compile_definitions(PIVCO_HAS_NEON=1)
else()
    message(STATUS "Architecture: ${TARGET_ARCH} — NEON disabled")
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
