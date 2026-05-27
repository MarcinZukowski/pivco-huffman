# Standalone build for the top-down (TD) decoder slice.
#
# Originally extracted at SHA 31cbf75; as of the 2026-05-26 consolidation the
# slice no longer forks the primitive/table/flat headers — it pulls them from
# the parent repo's src/ + include/ (so TD tracks the current primitives).
# Only the TD-unique sources (TD decode walk, naive build, dispatcher) live
# here.  This build therefore DEPENDS on the parent tree (../../src, ../../
# include, ../../ext/fse).
#
# Build:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/ph_td_test

cmake_minimum_required(VERSION 3.16)
project(ph_td C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# NEON sources (the historical heavy-lifting decoder) only build on
# aarch64.  The naive + scalar-opt code is pure C and works anywhere,
# so we still build a usable bench on x86 (just without the NEON
# decode column).
set(PH_TD_HAVE_NEON OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(PH_TD_HAVE_NEON ON)
endif()

# AVX-512 VBMI2 sources (the resurrected legacy TD AVX-512 decoder)
# only build on x86_64 with the matching compiler flag support.  We
# probe by compiling a one-liner that uses the VBMI2 intrinsic.
set(PH_TD_HAVE_AVX512 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512vbmi -mavx512vbmi2")
    check_c_source_compiles("
        #include <immintrin.h>
        int main(void) {
            __m512i v = _mm512_setzero_si512();
            __mmask64 m = _mm512_test_epi8_mask(v, v);
            (void)m; return 0;
        }
    " PH_TD_AVX512_VBMI2_OK)
    set(CMAKE_REQUIRED_FLAGS "")
    if(PH_TD_AVX512_VBMI2_OK)
        set(PH_TD_HAVE_AVX512 ON)
    endif()
endif()

# Release-tier flags; the upstream project ships -O3 -mcpu=native style,
# we keep it modest here to favour reproducibility over peak speed.
add_compile_options(-Wall -Wextra -O3 -g -fno-omit-frame-pointer)
if(PH_TD_HAVE_NEON)
    add_compile_definitions(PIVCO_HAS_NEON=1)
endif()
if(PH_TD_HAVE_AVX512)
    add_compile_definitions(PIVCO_HAS_AVX512=1)
endif()

# Force 8192-symbol blocks on x86 too so this bench's PIVCO_BLOCK_SIZE
# matches the main repo's canonical numbers (which were measured at
# 8192 with -mavx512f defined).  The naive/scalar-opt code is purely
# scalar and doesn't care about AVX-512; we just need the constant.
if(NOT PH_TD_HAVE_NEON)
    add_compile_definitions(PIVCO_BLOCK_SIZE=8192)
endif()

set(PH_TD_SOURCES
    src/pivco_huffman.c
    src/pivco_huffman_naive.c
    src/huffman_table.c
    src/pivco_prof.c
)
add_compile_definitions(PIVCO_PROF=1)
if(PH_TD_HAVE_NEON)
    list(APPEND PH_TD_SOURCES
        src/pivco_huffman_neon.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/pivco_huffman_neon_tables.c)  # shared
endif()
if(PH_TD_HAVE_AVX512)
    list(APPEND PH_TD_SOURCES
        src/pivco_huffman_avx512.c)
endif()

add_library(ph_td STATIC ${PH_TD_SOURCES})
# Fork-unique headers (TD API, prof) from include/+src/ first; the shared
# primitive/table/flat headers fall through to the parent repo.
target_include_directories(ph_td PUBLIC include
    PRIVATE src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Per-source flags for the AVX-512 TU (everyone else stays plain -O3).
if(PH_TD_HAVE_AVX512)
    set_source_files_properties(src/pivco_huffman_avx512.c PROPERTIES
        COMPILE_FLAGS
        "-mavx512f -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vpopcntdq -mavx512vl")
endif()

if(PH_TD_HAVE_NEON)
    add_executable(ph_td_test test/ph_td_test.c)
    target_link_libraries(ph_td_test PRIVATE ph_td)
endif()

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
    target_link_libraries(ph_td_bench_naive PRIVATE ph_td ph_td_huf0 m)
    target_include_directories(ph_td_bench_naive PRIVATE "${FSE_DIR}/lib")
else()
    message(STATUS "ph-td-naive bench skipped (ext/fse not found at ${FSE_DIR})")
endif()
