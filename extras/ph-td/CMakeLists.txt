# Standalone build for the historical top-down (TD) NEON decoder.
#
# Extracted at SHA 31cbf75 (2026-05-14, just before TD entry points
# were retired from the public API).  See README.md for context.
#
# Build:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/ph_td_test
#
# Self-contained — does not depend on the parent project.

cmake_minimum_required(VERSION 3.16)
project(ph_td C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(FATAL_ERROR "ph-td slice is NEON-only — requires aarch64.")
endif()

# Release-tier flags; the upstream project ships -O3 -mcpu=native style,
# we keep it modest here to favour reproducibility over peak speed.
add_compile_options(-Wall -Wextra -O3)
add_compile_definitions(PIVCO_HAS_NEON=1)

set(PH_TD_SOURCES
    src/pivco_huffman.c
    src/pivco_huffman_neon.c
    src/pivco_huffman_neon_tables.c
    src/pivco_huffman_naive.c
    src/huffman_table.c
)

add_library(ph_td STATIC ${PH_TD_SOURCES})
target_include_directories(ph_td PUBLIC include PRIVATE src)

add_executable(ph_td_test test/ph_td_test.c)
target_link_libraries(ph_td_test PRIVATE ph_td)

# ----- ph-td-naive vs huf0 bench (uses upstream ext/fse) -----
# Pulls huf0 from the parent project's ext/fse tree.  Only built when
# that tree exists (it does in the standard repo layout).
set(FSE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../ext/fse")
message(STATUS "ph-td: looking for FSE at ${FSE_DIR}/lib/huf.h")
if(EXISTS "${FSE_DIR}/lib/huf.h")
    message(STATUS "ph-td: FSE found -- building ph_td_bench_naive")
    add_library(ph_td_huf0 STATIC
        "${FSE_DIR}/lib/entropy_common.c"
        "${FSE_DIR}/lib/huf_compress.c"
        "${FSE_DIR}/lib/huf_decompress.c"
        "${FSE_DIR}/lib/hist.c"
        "${FSE_DIR}/lib/fse_compress.c"
        "${FSE_DIR}/lib/fse_decompress.c"
        "${FSE_DIR}/lib/debug.c"
    )
    target_include_directories(ph_td_huf0 PUBLIC "${FSE_DIR}/lib")
    target_compile_options(ph_td_huf0 PRIVATE -O3 -w)

    add_executable(ph_td_bench_naive test/ph_td_bench_naive.c)
    target_link_libraries(ph_td_bench_naive PRIVATE ph_td ph_td_huf0)
    target_include_directories(ph_td_bench_naive PRIVATE "${FSE_DIR}/lib")
else()
    message(STATUS "ph-td-naive bench skipped (ext/fse not found at ${FSE_DIR})")
endif()
