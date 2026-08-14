# bench/CMakeLists.txt -- the core benchmark harnesses.
#
# Pulled in via `add_subdirectory(bench)` from the top CMakeLists (inside
# BUILD_BENCH).  Deliberately small: only the three headline benches and the
# two things pivco_fair_bench directly depends on -- the ph_td support library
# and Oodle detection.  Everything else extras-flavoured (counters, LZ4 / ZSTD /
# EWAH / Metal detection, dump / fuzz / metal tools, research micro-benches)
# lives one level down in extras/bench/, added at the end so it inherits the
# Oodle vars set here and the global ph_td target.

# These benches used to live in the top CMakeLists, so their binaries landed in
# build/ (where docs + sweep scripts expect ./build/pivco_huffman_bench etc.).
# Keep them there rather than build/bench/ now that they're a subdirectory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# bench_distributions.c reads the real-world dataset files verbatim.  Bake in
# the absolute path so every target that compiles it finds extras/datasets/.
set_source_files_properties(${CMAKE_SOURCE_DIR}/bench/bench_distributions.c PROPERTIES
    COMPILE_DEFINITIONS "PIVCO_DATASET_DIR=\"${CMAKE_SOURCE_DIR}/extras/datasets\"")

add_executable(pivco_huffman_bench
    ${CMAKE_SOURCE_DIR}/bench/bench_main.c
    ${CMAKE_SOURCE_DIR}/bench/bench_distributions.c
)
target_link_libraries(pivco_huffman_bench pivco_huffman huf0 m)
target_include_directories(pivco_huffman_bench PRIVATE ${CMAKE_SOURCE_DIR}/ext/fse/lib ${CMAKE_SOURCE_DIR}/bench)

# Independent (namespaced) top-down TD library, retired from the main
# lib but resurrected as its own lib so the fair-bench can show the
# TD 2x2 grid (tree naive/opt x prims scalar/simd) next to BU.  Its
# exported symbols are renamed to phtd_* via -include phtd_names.h, so
# it links alongside the main pivco_huffman (BU) lib.
set(PHTD_DIR ${CMAKE_SOURCE_DIR}/extras/ph-td)
set(PHTD_SOURCES
    ${PHTD_DIR}/src/pivco_huffman.c
    ${PHTD_DIR}/src/pivco_huffman_naive.c
    ${PHTD_DIR}/src/huffman_table.c
    ${PHTD_DIR}/src/phtd_shim.c)
set(PHTD_DEFS "")
set(PHTD_SIMD_DEF "")          # define passed to the bench so it knows the SIMD ISA
set(PHTD_AVX512_SRC "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    list(APPEND PHTD_SOURCES
        ${PHTD_DIR}/src/pivco_huffman_neon.c
        ${CMAKE_SOURCE_DIR}/src/pivco_huffman_neon_tables.c)   # shared (was forked copy)
    set(PHTD_DEFS PIVCO_HAS_NEON=1)
    set(PHTD_SIMD_DEF PIVCO_HAS_NEON=1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    # Probe for AVX-512 VBMI2 (the ISA the resurrected TD AVX-512
    # decoder needs).  If present, build it for the simd grid rows.
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512vbmi -mavx512vbmi2")
    check_c_source_compiles("#include <immintrin.h>\nint main(void){__m512i v=_mm512_setzero_si512();return (int)_mm512_test_epi8_mask(v,v);}" PHTD_AVX512_OK)
    set(CMAKE_REQUIRED_FLAGS "")
    if(PHTD_AVX512_OK)
        list(APPEND PHTD_SOURCES ${PHTD_DIR}/src/pivco_huffman_avx512.c)
        set(PHTD_AVX512_SRC ${PHTD_DIR}/src/pivco_huffman_avx512.c)
        set(PHTD_DEFS PIVCO_HAS_AVX512=1 PIVCO_BLOCK_SIZE=8192)
        set(PHTD_SIMD_DEF PIVCO_HAS_AVX512=1)
    else()
        set(PHTD_DEFS PIVCO_BLOCK_SIZE=8192)   # scalar-only TD rows
    endif()
else()
    set(PHTD_DEFS PIVCO_BLOCK_SIZE=8192)
endif()
add_library(ph_td STATIC ${PHTD_SOURCES})
# Fork-unique headers (TD API struct, prof) resolve from ph-td first; the
# shared primitive/table/flat headers (deleted from the fork) fall through
# to the main src/ + include/ so TD uses the CURRENT primitives.
target_include_directories(ph_td PUBLIC ${PHTD_DIR}/include
    PRIVATE ${PHTD_DIR}/src ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(ph_td PRIVATE ${PHTD_DEFS})
# ph-td was designed without the per-node FSE path (raw bitmaps); the
# main project's global -DPIVCO_HAS_FSE must be undone for this lib so
# neon.c doesn't try to include pivco_fse.h (which ph-td doesn't vendor).
target_compile_options(ph_td PRIVATE -O3 -w -UPIVCO_HAS_FSE
    -include ${PHTD_DIR}/include/phtd_names.h)
if(PHTD_AVX512_SRC)
    set_source_files_properties(${PHTD_AVX512_SRC} PROPERTIES COMPILE_OPTIONS
        "-mavx512f;-mavx512bw;-mavx512vbmi;-mavx512vbmi2;-mavx512vpopcntdq;-mavx512vl")
endif()

add_executable(pivco_fair_bench
    ${CMAKE_SOURCE_DIR}/bench/bench_fair.c
    ${CMAKE_SOURCE_DIR}/bench/bench_distributions.c
)
target_link_libraries(pivco_fair_bench pivco_huffman huf0 m ph_td)
target_include_directories(pivco_fair_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/ext/fse/lib
    ${CMAKE_SOURCE_DIR}/extras ${CMAKE_SOURCE_DIR}/extras/bench ${PHTD_DIR}/include)
if(PHTD_SIMD_DEF)
    target_compile_definitions(pivco_fair_bench PRIVATE ${PHTD_SIMD_DEF})
endif()

# Isolated decode-primitive microbench (flat unpack / scatter / merge on
# synthetic input).  Self-contained: only needs the backend flat headers
# in src/.  aarch64 baseline includes NEON; x86 SIMD variants will need
# per-arch flags when added.
add_executable(pivco_bench_prim ${CMAKE_SOURCE_DIR}/bench/bench_prim.c)
target_include_directories(pivco_bench_prim PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${PHTD_DIR}/include)
target_link_libraries(pivco_bench_prim pivco_huffman m)
if(UNIX AND NOT APPLE)
    # prim_variants keeps byte-exact asm replicas (c3roll asof pairs) that
    # address their tables absolutely -- R_X86_64_32S, illegal in a PIE
    # link.  The encoding IS what those variants preserve, so link this
    # research binary non-PIE instead of rewriting them RIP-relative.
    target_link_options(pivco_bench_prim PRIVATE -no-pie)
endif()

# Optional Oodle reference comparison, consumed here by pivco_fair_bench and --
# via the inherited OODLE_* vars -- by pivco_fse_xy_micro under extras/bench/.
# Auto-enables iff the user has symlinked ext/oodle -> a built OodleUE clone.
# See extras/bench/bench_oodle_wrapper.h for setup instructions.
if(PIVCO_FSE)
    set(OODLE_ROOT "${CMAKE_SOURCE_DIR}/ext/oodle/Engine/Source/Runtime/OodleDataCompression/Sdks/2.9.16")
    # OODLE_LIB_VARIANT=source links our from-source build-out lib
    # (so the build patches' ASM wiring is what's measured);
    # =shipped links RAD's prebuilt platform lib (their production
    # ASM build) for a head-to-head against our from-source numbers.
    set(OODLE_LIB_VARIANT "source" CACHE STRING "Oodle lib to link: source | shipped")
    if(OODLE_LIB_VARIANT STREQUAL "shipped")
        if(APPLE)
            set(OODLE_STATIC_LIB "${OODLE_ROOT}/lib/Mac/liboo2coremac64.a")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(OODLE_STATIC_LIB "${OODLE_ROOT}/lib/LinuxArm64/liboo2corelinuxarm64.a")
        else()
            set(OODLE_STATIC_LIB "${OODLE_ROOT}/lib/Linux/liboo2corelinux64.a")
        endif()
    else()
        set(OODLE_STATIC_LIB "${CMAKE_SOURCE_DIR}/ext/oodle/build-out/ar/liboodle-data-static.a")
    endif()
    if(EXISTS "${OODLE_ROOT}/src/oodle2/core/newlz_arrays_huff.h"
       AND EXISTS "${OODLE_STATIC_LIB}")
        set(OODLE_DETECTED ON)
        message(STATUS "Oodle detected at ext/oodle - enabling pivco_fse_xy_micro Oodle reference columns")
    else()
        set(OODLE_DETECTED OFF)
        message(STATUS "Oodle not detected at ext/oodle - pivco_fse_xy_micro will skip Oodle reference columns")
    endif()

    # fair-bench gets the Oodle opaque-only reference columns when
    # Oodle is detected.
    if(OODLE_DETECTED)
        target_sources(pivco_fair_bench PRIVATE ${CMAKE_SOURCE_DIR}/extras/bench/bench_oodle_wrapper.cpp)
        target_link_libraries(pivco_fair_bench "${OODLE_STATIC_LIB}")
        target_include_directories(pivco_fair_bench PRIVATE
            "${OODLE_ROOT}/src/oodle2/core"
            "${OODLE_ROOT}/src/oodle2/core/public"
            "${OODLE_ROOT}/src/oodle2/base"
            "${OODLE_ROOT}/include")
        target_compile_definitions(pivco_fair_bench PRIVATE PIVCO_HAS_OODLE=1)
        # set_source_files_properties has DIRECTORY scope, so the same
        # properties set in extras/bench/CMakeLists.txt do not apply
        # here; set them again for fair_bench's copy of the same file.
        set_source_files_properties(${CMAKE_SOURCE_DIR}/extras/bench/bench_oodle_wrapper.cpp
            PROPERTIES COMPILE_DEFINITIONS OODLE_IMPORT_LIB)
        set_source_files_properties(${CMAKE_SOURCE_DIR}/extras/bench/bench_oodle_wrapper.cpp
            PROPERTIES COMPILE_FLAGS "-std=c++20 -w")
    endif()
endif()

# Everything else -- standalone extras tools (dump / file_to_dist / fuzz /
# metal) and the research micro-benches -- lives under extras/bench/.  It is a
# child scope, so it inherits OODLE_* + the ph_td target, and does its own
# counters / LZ4 / ZSTD / EWAH detection.  Default its output to
# build/extras/bench/ (the tools that scripts reference re-pin themselves to
# build/).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/extras/bench)
add_subdirectory(${CMAKE_SOURCE_DIR}/extras/bench ${CMAKE_BINARY_DIR}/extras/bench)
