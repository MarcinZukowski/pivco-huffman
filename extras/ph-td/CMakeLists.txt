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
    src/huffman_table.c
)

add_library(ph_td STATIC ${PH_TD_SOURCES})
target_include_directories(ph_td PUBLIC include PRIVATE src)

add_executable(ph_td_test test/ph_td_test.c)
target_link_libraries(ph_td_test PRIVATE ph_td)
