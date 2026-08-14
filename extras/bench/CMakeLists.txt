# extras/bench/CMakeLists.txt -- research / micro / one-off bench targets.
#
# Invoked via `add_subdirectory(extras/bench)` from bench/CMakeLists.txt (inside
# BUILD_BENCH).  From the parent it inherits the core libraries (pivco_huffman,
# huf0), the ph_td support lib, and the Oodle detection vars (OODLE_DETECTED,
# OODLE_STATIC_LIB, OODLE_ROOT) -- Oodle lives in the parent because
# pivco_fair_bench needs it too.  Everything else this directory needs
# (lemire/counters, system LZ4/ZSTD, EWAH, Metal) is detected right here, since
# only these targets consume it.
#
# Sources are local (CMAKE_CURRENT_SOURCE_DIR is extras/bench/), so each
# target lists its file as just `bench_X.c`.  Cross-tree sources use
# ${CMAKE_SOURCE_DIR} explicitly.

set(BENCH_DIST ${CMAKE_SOURCE_DIR}/bench/bench_distributions.c)

# ---- Dependency detection (consumed only by targets in this directory) -------

# lemire/counters — hardware counter access (cycles, IPC, branch/cache miss) for
# the *_cnt microbenches.  Via CPM so a fresh checkout needs no manual clone;
# fetched only when BUILD_BENCH is on (this dir is only added then).
include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)
set(COUNTERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(COUNTERS_INSTALL OFF CACHE BOOL "" FORCE)
CPMAddPackage(NAME counters GITHUB_REPOSITORY lemire/counters VERSION 4.1.0)
# Mark the counters headers SYSTEM so its template-instantiation warning noise
# doesn't drown out warnings in our own code.
if(TARGET counters)
    get_target_property(_counters_incs counters INTERFACE_INCLUDE_DIRECTORIES)
    if(_counters_incs)
        set_target_properties(counters PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
        target_include_directories(counters SYSTEM INTERFACE ${_counters_incs})
    endif()
endif()

# System LZ4 + zstd (homebrew) for the lz4-split / lz4-ph / fuzz benches.
find_library(LZ4_LIB lz4 PATHS /opt/homebrew/lib /usr/local/lib)
find_path(LZ4_INC lz4.h PATHS /opt/homebrew/include /usr/local/include)
find_library(ZSTD_LIB zstd PATHS /opt/homebrew/lib /usr/local/lib)
find_path(ZSTD_INC zstd.h PATHS /opt/homebrew/include /usr/local/include)

# EWAH (header-only) for pivco_bench_golomb's EWAH column.
if(EXISTS ${CMAKE_SOURCE_DIR}/ext/ewah/include/ewah/ewah.h)
    set(PIVCO_HAS_EWAH 1)
    message(STATUS "EWAH detected at ext/ewah — pivco_bench_golomb will include EWAH column")
else()
    set(PIVCO_HAS_EWAH 0)
    message(STATUS "EWAH not detected at ext/ewah — pivco_bench_golomb will skip EWAH column")
endif()

# ---- Standalone extras tools (moved out of the top CMakeLists).  Pinned to
#      build/ so documented paths like ./build/pivco_dump_distributions and
#      ./build/pivco_file_to_dist keep working. --------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Dump distribution histograms as a JS-loadable data file for figures/tree_viz.html.
include_directories(${CMAKE_SOURCE_DIR}/bench ${CMAKE_SOURCE_DIR}/src)

add_executable(pivco_dump_distributions
    ${CMAKE_SOURCE_DIR}/extras/dump_distributions.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_dump_distributions pivco_huffman m)

# Dump the production C-built decode tree in the figure's "download tree text"
# format so the two can be diff'd to confirm the figure matches the real codec.
add_executable(pivco_dump_tree_text
    ${CMAKE_SOURCE_DIR}/extras/dump_tree_text.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_dump_tree_text pivco_huffman m)

add_executable(pivco_file_to_dist ${CMAKE_SOURCE_DIR}/extras/file_to_dist.c)
target_link_libraries(pivco_file_to_dist m)

# Metal compute primitive microbench (Apple only): GPU throughput of the ph
# partition (encoder) + tree_merge (BU decoder) primitives vs the CPU NEON path.
# The .metal shader is loaded at runtime from a path relative to CWD (root).
if(APPLE)
    find_library(METAL_LIB Metal)
    find_library(FOUNDATION_LIB Foundation)
    if(METAL_LIB AND FOUNDATION_LIB)
        message(STATUS "Metal framework detected — enabling pivco_bench_metal_partition")
        add_executable(pivco_bench_metal_partition
            ${CMAKE_SOURCE_DIR}/extras/gpu/bench_metal_partition.mm)
        target_link_libraries(pivco_bench_metal_partition "${METAL_LIB}" "${FOUNDATION_LIB}")
        set_source_files_properties(${CMAKE_SOURCE_DIR}/extras/gpu/bench_metal_partition.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    else()
        message(STATUS "Metal framework not found — pivco_bench_metal_partition skipped")
    endif()
endif()

# Research micro-benches below stay under build/extras/bench/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/extras/bench)

# -------- Encoder-side standalone benches --------

add_executable(pivco_fse_tune
    bench_fse_tune.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_fse_tune pivco_huffman huf0 m)
target_include_directories(pivco_fse_tune PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ext/fse/lib ${CMAKE_SOURCE_DIR}/extras
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pivco_huffman_bench_select
    bench_select.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_huffman_bench_select pivco_huffman m)
target_include_directories(pivco_huffman_bench_select PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(pivco_huffman_bench_encode
    bench_encode.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_huffman_bench_encode pivco_huffman huf0 m)
target_include_directories(pivco_huffman_bench_encode PRIVATE
    ${CMAKE_SOURCE_DIR}/ext/fse/lib)

# Tree-shape ablation: OPTIMIZED / NAIVE / FUSED / CANONICAL_FLAT, with
# huf0 (stock) and oo-huff baselines measured in the same session for
# apples-to-apples decode-bandwidth comparison.  FSE disabled in all runs;
# this isolates tree-shape vs entropy-coding effects.
add_executable(pivco_bench_tree_modes
    bench_tree_modes.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_bench_tree_modes pivco_huffman huf0 m)
target_include_directories(pivco_bench_tree_modes PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ext/fse/lib
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pivco_bench_tree_table
    bench_tree_table.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_bench_tree_table pivco_huffman m)
target_include_directories(pivco_bench_tree_table PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR})
if(OODLE_DETECTED)
    target_sources(pivco_bench_tree_modes PRIVATE
        bench_oodle_wrapper.cpp)
    target_link_libraries(pivco_bench_tree_modes ${OODLE_STATIC_LIB})
    target_include_directories(pivco_bench_tree_modes PRIVATE
        "${OODLE_ROOT}/src/oodle2/core"
        "${OODLE_ROOT}/src/oodle2/core/public"
        "${OODLE_ROOT}/src/oodle2/base"
        "${OODLE_ROOT}/include")
    target_compile_definitions(pivco_bench_tree_modes PRIVATE PIVCO_HAS_OODLE=1)
endif()

# NEON-only encoder micros.
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    add_executable(pivco_encode_split_micro bench_encode_split.c)
    target_link_libraries(pivco_encode_split_micro pivco_huffman m)
    target_include_directories(pivco_encode_split_micro PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

    add_executable(pivco_enc_init_micro bench_enc_init.c)
    target_link_libraries(pivco_enc_init_micro pivco_huffman m)
endif()
# NEON merge_vec_vec + encode-partition variant microbenches retired —
# kernels mined into bench/prim_variants/ (graveyard).

# Partition u8 vs u16 microbench: works on any tier (each ISA is
# #ifdef-gated inside the source).
add_executable(pivco_partition_u8_vs_u16 bench_partition_u8_vs_u16.c)
target_link_libraries(pivco_partition_u8_vs_u16 pivco_huffman m)
target_include_directories(pivco_partition_u8_vs_u16 PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

# K_right wire-format storage analyzer (still useful after the
# K_right format landed -- analyzes per-dist storage cost).
add_executable(pivco_bench_kr_storage
    bench_kr_storage.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_bench_kr_storage pivco_huffman huf0 m)
target_include_directories(pivco_bench_kr_storage PRIVATE
    ${CMAKE_SOURCE_DIR}/ext/fse/lib)

# -------- Stats / analysis benches --------

add_executable(pivco_huffman_profile_english
    bench_profile_english.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_huffman_profile_english pivco_huffman m)

add_executable(pivco_node_bitmap_stats
    bench_node_bitmap_stats.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_node_bitmap_stats pivco_huffman m)

add_executable(pivco_multi_stage_stats
    bench_multi_stage_stats.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_multi_stage_stats pivco_huffman m)

add_executable(pivco_flat_subtree_stats
    bench_flat_subtree_stats.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_flat_subtree_stats pivco_huffman m)

# Per-dataset entropy + FSE-per-internal-node breakdown.  Validates
# theories about how much of the compression benefit comes from
# which nodes / which entropy-coding strategy.
if(PIVCO_FSE)
    add_executable(pivco_entropy_breakdown
        bench_entropy_breakdown.c
        ${BENCH_DIST}
    )
    target_link_libraries(pivco_entropy_breakdown pivco_huffman m)
    target_include_directories(pivco_entropy_breakdown PRIVATE
        ${CMAKE_SOURCE_DIR}/ext/fse/lib)
endif()

# Offline TANS-routing ratio bound: real files (per-block) or
# MAIN bench distributions (per-histogram).
add_executable(pivco_tans_bound
    bench_tans_bound.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_tans_bound pivco_huffman m)

# 4-way codec comparison: ph / phe / huf0 / fse on the same data.
if(PIVCO_FSE)
    add_executable(pivco_bench_4way
        bench_4way.c
        ${BENCH_DIST}
    )
    target_link_libraries(pivco_bench_4way pivco_huffman huf0 m)
    target_include_directories(pivco_bench_4way PRIVATE
        ${CMAKE_SOURCE_DIR}/ext/fse/lib)

    # Per-FSE-table-id usage histogram dumper.
    add_executable(pivco_bench_fse_table_use bench_fse_table_use.c)
    target_link_libraries(pivco_bench_fse_table_use pivco_huffman m)
endif()

add_executable(pivco_dist_stats
    bench_dist_stats.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_dist_stats pivco_huffman m)

# Self-contained cross-arch 4-way root-merge microbench (AVX-512 VBMI2 runs all
# variants; NEON runs vvvv only; selects via #if).  Needs -march=native.
add_executable(pivco_merge4way bench_merge4way.c)
target_compile_options(pivco_merge4way PRIVATE -march=native)
target_link_libraries(pivco_merge4way m)

add_executable(pivco_partition_skew
    bench_partition_skew.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_partition_skew pivco_huffman m)

add_executable(pivco_dense_runs
    bench_dense_runs.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_dense_runs pivco_huffman m)

add_executable(pivco_mask_pair_degeneracy
    bench_mask_pair_degeneracy.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_mask_pair_degeneracy pivco_huffman m)

# -------- Partition microbench (any arch; ISA gated inside) --------

add_executable(pivco_partition_micro bench_partition_micro.c)
target_link_libraries(pivco_partition_micro pivco_huffman m)
target_include_directories(pivco_partition_micro PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

# -------- Counter-instrumented NEON microbenches --------
# Reports cycles / instructions / IPC / cache-miss / branch-miss per
# call.  Needs sudo on macOS or perf_event_paranoid=0 on Linux.
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    set(_cnt_targets
        pivco_partition_micro_cnt:bench_partition_micro_cnt.cpp
        pivco_lemire_compare:bench_lemire_compare.cpp
        pivco_scatter_micro_cnt:bench_scatter_micro_cnt.cpp
        pivco_fusion_micro_cnt:bench_fusion_micro_cnt.cpp
        pivco_scatter_split_cnt:bench_scatter_split_cnt.cpp
        pivco_p8_dual_cnt:bench_p8_dual_cnt.cpp
        pivco_fusion_v2_cnt:bench_fusion_v2_cnt.cpp
        pivco_fusion_v3_cnt:bench_fusion_v3_cnt.cpp
        pivco_fusion_v4_cnt:bench_fusion_v4_cnt.cpp
        pivco_fusion_v5_real_kernel_cnt:bench_fusion_v5_real_kernel_cnt.cpp
        pivco_primitives_neon_cnt:bench_primitives_neon_cnt.cpp
    )
    foreach(_pair IN LISTS _cnt_targets)
        string(REPLACE ":" ";" _t "${_pair}")
        list(GET _t 0 _name)
        list(GET _t 1 _src)
        add_executable(${_name} ${_src})
        target_link_libraries(${_name} pivco_huffman counters::counters m)
        target_include_directories(${_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    endforeach()

    # tree_merge_neon_cnt links against counters/m only (no pivco_huffman).
    add_executable(pivco_tree_merge_neon_cnt bench_tree_merge_neon_cnt.cpp)
    target_link_libraries(pivco_tree_merge_neon_cnt counters::counters m)
    target_include_directories(pivco_tree_merge_neon_cnt PRIVATE
        ${CMAKE_SOURCE_DIR}/src)
endif()

# -------- Counter-instrumented x86 microbenches --------
if(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64")
    # x86 merge_vec_vec + encode-partition COM microbenches retired —
    # kernels mined into bench/prim_variants/ (graveyard).

    add_executable(pivco_micro_sse_cnt bench_micro_sse_cnt.cpp)
    target_link_libraries(pivco_micro_sse_cnt pivco_huffman counters::counters m)
    target_include_directories(pivco_micro_sse_cnt PRIVATE ${CMAKE_SOURCE_DIR}/src)

    if(COMPILER_HAS_AVX512_VBMI2 AND CPU_HAS_AVX512
                                AND NOT CPU_HAS_AVX512 STREQUAL "0")
        foreach(_pair
                pivco_micro_avx512_cnt:bench_micro_avx512_cnt.cpp
                pivco_fusion_v3_avx512_cnt:bench_fusion_v3_avx512_cnt.cpp
                pivco_fusion_v4_avx512_cnt:bench_fusion_v4_avx512_cnt.cpp)
            string(REPLACE ":" ";" _t "${_pair}")
            list(GET _t 0 _name)
            list(GET _t 1 _src)
            add_executable(${_name} ${_src})
            target_link_libraries(${_name} pivco_huffman counters::counters m)
            target_include_directories(${_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
        endforeach()

        # Bit-sliced (transposed) packing layout microbench --
        # classic LSB bit-pack vs bit-sliced layout for D in {3,5,7}.
        # Standalone; AVX-512 VBMI2 only.
        add_executable(pivco_bench_bitslice bench_bitslice.c)
        target_compile_options(pivco_bench_bitslice PRIVATE
            -mavx512f -mavx512bw -mavx512vbmi -mavx512vbmi2)
        target_include_directories(pivco_bench_bitslice PRIVATE
            ${CMAKE_SOURCE_DIR}/src)
    endif()

    foreach(_pair
            pivco_fusion_v3_sse_cnt:bench_fusion_v3_sse_cnt.cpp
            pivco_fusion_v4_sse_cnt:bench_fusion_v4_sse_cnt.cpp)
        string(REPLACE ":" ";" _t "${_pair}")
        list(GET _t 0 _name)
        list(GET _t 1 _src)
        add_executable(${_name} ${_src})
        target_link_libraries(${_name} pivco_huffman counters::counters m)
        target_include_directories(${_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    endforeach()

    add_executable(pivco_primitives_x86_cnt bench_primitives_x86_cnt.cpp)
    target_compile_options(pivco_primitives_x86_cnt PRIVATE -mbmi2 -mavx2)
    target_link_libraries(pivco_primitives_x86_cnt counters::counters m)
    target_include_directories(pivco_primitives_x86_cnt PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

    add_executable(pivco_tree_merge_x86_cnt bench_tree_merge_x86_cnt.cpp)
    target_compile_options(pivco_tree_merge_x86_cnt PRIVATE
        -msse4.1 -mavx2 -mbmi2)
    if(COMPILER_HAS_AVX512_VBMI2 AND CPU_HAS_AVX512
                                AND NOT CPU_HAS_AVX512 STREQUAL "0")
        target_compile_options(pivco_tree_merge_x86_cnt PRIVATE
            -mavx512f -mavx512bw -mavx512vbmi -mavx512vbmi2)
    endif()
    target_link_libraries(pivco_tree_merge_x86_cnt counters::counters m)
    target_include_directories(pivco_tree_merge_x86_cnt PRIVATE
        ${CMAKE_SOURCE_DIR}/src)
endif()

# Store-coalescing experiment (NEON only; see docs/COALESCE.md).
if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    add_executable(pivco_bench_coalesce bench_coalesce.c)
endif()
# bench_bu_decoder{,_x86} retired with the TD decoder (2026-05-14) --
# they existed to A/B BU against TD; BU has been the production
# decoder for weeks.

# -------- Standalone analysis / viz / multicore --------

add_executable(pivco_bench_multicore
    bench_multicore.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_bench_multicore pivco_huffman huf0 pthread m)
target_include_directories(pivco_bench_multicore PRIVATE
    ${CMAKE_SOURCE_DIR}/ext/fse/lib)

add_executable(pivco_flat_optimal
    bench_flat_optimal.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_flat_optimal pivco_huffman m)

add_executable(pivco_tree_viz
    bench_tree_viz.c
    ${BENCH_DIST}
)
target_link_libraries(pivco_tree_viz pivco_huffman m)

# -------- PIVCO_FSE-gated heavy benches (Oodle / LZ4 / EWAH) --------
if(PIVCO_FSE)
    # FSE cursor-count × unroll microbench (research; see IDEAS.md
    # "FSE cursor-count + unroll experiment").  Hand-rolled FSE
    # encoder + decoders varying x and y; doesn't touch the codec
    # wire format or runtime API.
    if(OODLE_DETECTED)
        add_executable(pivco_fse_xy_micro
            bench_fse_xy_micro.c
            bench_oodle_wrapper.cpp
        )
        target_link_libraries(pivco_fse_xy_micro pivco_huffman huf0 m
            "${OODLE_STATIC_LIB}")
        target_include_directories(pivco_fse_xy_micro PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/ext/fse/lib
            "${OODLE_ROOT}/src/oodle2/core"
            "${OODLE_ROOT}/src/oodle2/core/public"
            "${OODLE_ROOT}/src/oodle2/base"
            "${OODLE_ROOT}/include")
        target_compile_definitions(pivco_fse_xy_micro PRIVATE PIVCO_HAS_OODLE=1)
        # Oodle headers require OODLE_IMPORT_LIB when consuming
        # the built static lib (matches OODLE_BUILDING_LIB used
        # to build it).
        set_source_files_properties(bench_oodle_wrapper.cpp
            PROPERTIES COMPILE_DEFINITIONS OODLE_IMPORT_LIB)
        set_source_files_properties(bench_oodle_wrapper.cpp
            PROPERTIES COMPILE_FLAGS "-std=c++20 -w")
    else()
        add_executable(pivco_fse_xy_micro bench_fse_xy_micro.c)
        target_link_libraries(pivco_fse_xy_micro pivco_huffman huf0 m)
        target_include_directories(pivco_fse_xy_micro PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/ext/fse/lib)
    endif()

    # LZ4 output stream splitter + per-stream entropy / decode bench.
    if(LZ4_LIB AND LZ4_INC)
        add_executable(pivco_bench_lz4_split bench_lz4_split.c)
        target_link_libraries(pivco_bench_lz4_split pivco_huffman huf0 m
            "${LZ4_LIB}")
        target_include_directories(pivco_bench_lz4_split PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/ext/fse/lib
            "${LZ4_INC}")

        # LZ4 + ph stacked-codec prototype bench.
        if(ZSTD_LIB AND ZSTD_INC AND EXISTS ${CMAKE_SOURCE_DIR}/ext/lz4/lib/lz4hc.c)
            add_executable(pivco_bench_lz4_ph
                bench_lz4_ph.c
                ${CMAKE_SOURCE_DIR}/extras/lz4hc_split.c
                ${CMAKE_SOURCE_DIR}/extras/lz4_split_decode.c
                ${CMAKE_SOURCE_DIR}/ext/lz4/lib/lz4.c
            )
            target_link_libraries(pivco_bench_lz4_ph pivco_huffman huf0 m
                "${LZ4_LIB}" "${ZSTD_LIB}")
            target_include_directories(pivco_bench_lz4_ph PRIVATE
                ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/ext/fse/lib ${CMAKE_SOURCE_DIR}/ext/lz4/lib
                ${CMAKE_SOURCE_DIR}/extras
                "${LZ4_INC}" "${ZSTD_INC}")
            # Suppress -Wall noise from the upstream LZ4 source.
            set_source_files_properties(
                ${CMAKE_SOURCE_DIR}/extras/lz4hc_split.c
                ${CMAKE_SOURCE_DIR}/ext/lz4/lib/lz4.c
                PROPERTIES COMPILE_FLAGS "-w")

            # Correctness/fuzz harness for the same 4-stream split encoder+decoder.
            add_executable(pivco_fuzz_lz4_split
                ${CMAKE_SOURCE_DIR}/extras/fuzz_lz4_split.c
                ${CMAKE_SOURCE_DIR}/extras/lz4hc_split.c
                ${CMAKE_SOURCE_DIR}/extras/lz4_split_decode.c
                ${CMAKE_SOURCE_DIR}/ext/lz4/lib/lz4.c
            )
            target_link_libraries(pivco_fuzz_lz4_split m "${LZ4_LIB}")
            target_include_directories(pivco_fuzz_lz4_split PRIVATE
                ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/ext/lz4/lib ${CMAKE_SOURCE_DIR}/extras
                "${LZ4_INC}" "${ZSTD_INC}")
        endif()
    endif()

    # Rice (Golomb power-of-2 M) vs fast-FSE vs Lemire's EWAH on
    # skewed bitmaps.  EWAH support pulled in via a C++ wrapper around
    # ext/ewah (header-only library); if ext/ewah isn't cloned the
    # EWAH columns just stay empty in the output.
    if(PIVCO_HAS_EWAH)
        add_executable(pivco_bench_golomb
            bench_golomb.c
            bench_ewah_wrapper.cpp
        )
        target_compile_definitions(pivco_bench_golomb PRIVATE PIVCO_HAS_EWAH=1)
        target_include_directories(pivco_bench_golomb PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/ext/fse/lib
            ${CMAKE_SOURCE_DIR}/ext/ewah/include)
        set_source_files_properties(bench_ewah_wrapper.cpp
            PROPERTIES COMPILE_FLAGS "-std=c++17 -w")
    else()
        add_executable(pivco_bench_golomb bench_golomb.c)
        target_include_directories(pivco_bench_golomb PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/ext/fse/lib)
    endif()
    target_link_libraries(pivco_bench_golomb pivco_huffman m)
endif()

# -------- bench_unpack_dN / bench_fl_unpack / bench_partition_root_iota --------
# These are referenced in IDEAS.md / docs as historical microbenches but
# were not wired into the top CMake; they're left as orphan sources for
# manual compilation.  (Add add_executable() blocks here if they get
# resurrected.)
