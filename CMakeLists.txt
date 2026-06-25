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
# Mark the counters headers as SYSTEM so its -Wunused-variable noise from
# template instantiations doesn't drown out warnings in our own code.
if(TARGET counters)
    get_target_property(_counters_incs counters INTERFACE_INCLUDE_DIRECTORIES)
    if(_counters_incs)
        set_target_properties(counters PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
        target_include_directories(counters SYSTEM INTERFACE ${_counters_incs})
    endif()
endif()

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
# docs/FSE-V0.md.  Sources are plain C with no SIMD deps; build-on/off via
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
    # Note: SVE backend exists (pivco_huffman_sve.c) but is disabled and
    # stale (pre-unified-codec).  Empirically confirmed loss vs NEON at
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
endif()

# Benchmarks
option(BUILD_BENCH "Build benchmarks" ON)
if(BUILD_BENCH)
    # bench_distributions.c reads the real-world dataset files verbatim
    # from this directory.  Apply the compile-time path to every TU that
    # includes the file (set on the file so it propagates to every target
    # that uses bench/bench_distributions.c -- ~10 of them).
    set_source_files_properties(bench/bench_distributions.c PROPERTIES
        COMPILE_DEFINITIONS "PIVCO_DATASET_DIR=\"${CMAKE_SOURCE_DIR}/extras/datasets\"")

    add_executable(pivco_huffman_bench
        bench/bench_main.c
        bench/bench_distributions.c
        extras/bench/bench_rans.cpp
    )
    target_link_libraries(pivco_huffman_bench pivco_huffman huf0 m)
    target_include_directories(pivco_huffman_bench PRIVATE ext/fse/lib ext/ryg_rans)

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
        bench/bench_fair.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_fair_bench pivco_huffman huf0 m ph_td)
    target_include_directories(pivco_fair_bench PRIVATE include src ext/fse/lib extras extras/bench ${PHTD_DIR}/include)
    if(PHTD_SIMD_DEF)
        target_compile_definitions(pivco_fair_bench PRIVATE ${PHTD_SIMD_DEF})
    endif()

    # Isolated decode-primitive microbench (flat unpack / scatter / merge on
    # synthetic input).  Self-contained: only needs the backend flat headers
    # in src/.  aarch64 baseline includes NEON; x86 SIMD variants will need
    # per-arch flags when added.
    add_executable(pivco_bench_prim bench/bench_prim.c)
    target_include_directories(pivco_bench_prim PRIVATE include src extras/ph-td/include)
    target_link_libraries(pivco_bench_prim pivco_huffman m)

    # Dump distribution histograms as a JS-loadable data file for the
    # figures/tree_viz.html visualization.
    add_executable(pivco_dump_distributions
        extras/dump_distributions.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_dump_distributions pivco_huffman m)

    # Dump the production C-built decode tree in the same text format as the
    # "download tree text" button in figures/tree_viz.html, so the two can be
    # diff'd to confirm the figure reflects the real codec's tree.
    add_executable(pivco_dump_tree_text
        extras/dump_tree_text.c
        bench/bench_distributions.c
    )
    target_link_libraries(pivco_dump_tree_text pivco_huffman m)

    add_executable(pivco_file_to_dist
        extras/file_to_dist.c
    )
    target_link_libraries(pivco_file_to_dist m)

    # Detection vars consumed by both this scope (for fair_bench Oodle
    # additions, pivco_fuzz_lz4_split, pivco_bench_metal_partition) and the
    # extras/bench/ subdirectory (which has the heavier targets that depend
    # on Oodle / LZ4 / ZSTD / EWAH).
    if(PIVCO_FSE)
        # Optional Oodle reference comparison.  Auto-enables iff the
        # user has symlinked ext/oodle -> a built OodleUE clone.
        # See extras/bench/bench_oodle_wrapper.h for setup instructions.
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
            target_sources(pivco_fair_bench PRIVATE extras/bench/bench_oodle_wrapper.cpp)
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
            set_source_files_properties(extras/bench/bench_oodle_wrapper.cpp
                PROPERTIES COMPILE_DEFINITIONS OODLE_IMPORT_LIB)
            set_source_files_properties(extras/bench/bench_oodle_wrapper.cpp
                PROPERTIES COMPILE_FLAGS "-std=c++20 -w")
        endif()

        # LZ4 + zstd detection (used by pivco_fuzz_lz4_split here and by
        # pivco_bench_lz4_{split,ph} in extras/bench/).  Links against
        # system LZ4 (homebrew); vendoring at ext/lz4 is a possible future
        # step if we want this on CI hosts.
        find_library(LZ4_LIB lz4 PATHS /opt/homebrew/lib /usr/local/lib)
        find_path(LZ4_INC lz4.h PATHS /opt/homebrew/include /usr/local/include)
        find_library(ZSTD_LIB zstd PATHS /opt/homebrew/lib /usr/local/lib)
        find_path(ZSTD_INC zstd.h PATHS /opt/homebrew/include /usr/local/include)
        if(LZ4_LIB AND LZ4_INC AND ZSTD_LIB AND ZSTD_INC
                AND EXISTS ${CMAKE_SOURCE_DIR}/ext/lz4/lib/lz4hc.c)
            # Correctness harness for the 4-stream encoder + decoder used
            # by pivco_bench_lz4_ph (defined under extras/bench/).
            add_executable(pivco_fuzz_lz4_split
                extras/fuzz_lz4_split.c
                extras/lz4hc_split.c
                extras/lz4_split_decode.c
                ext/lz4/lib/lz4.c
            )
            target_link_libraries(pivco_fuzz_lz4_split m "${LZ4_LIB}")
            target_include_directories(pivco_fuzz_lz4_split PRIVATE
                src include ext/lz4/lib extras
                "${LZ4_INC}" "${ZSTD_INC}")
            # Suppress -Wall noise from the upstream LZ4 source (~150
            # warnings about implicit conversions etc.).  Same flags apply
            # when pivco_bench_lz4_ph (under extras/bench/) reuses these
            # sources.
            set_source_files_properties(extras/lz4hc_split.c ext/lz4/lib/lz4.c
                PROPERTIES COMPILE_FLAGS "-w")
        endif()

        # Metal compute primitive microbench (Apple only).  Measures GPU
        # throughput of the ph tree-walk primitives — partition (encoder
        # side) and tree_merge (BU decoder side) — vs the CPU NEON path.
        # All GPU sources live under extras/gpu/ ; the .metal shader is
        # loaded at runtime from a path relative to CWD (project root).
        if(APPLE)
            find_library(METAL_LIB Metal)
            find_library(FOUNDATION_LIB Foundation)
            if(METAL_LIB AND FOUNDATION_LIB)
                message(STATUS "Metal framework detected — enabling pivco_bench_metal_partition")
                add_executable(pivco_bench_metal_partition
                    extras/gpu/bench_metal_partition.mm)
                target_link_libraries(pivco_bench_metal_partition
                    "${METAL_LIB}" "${FOUNDATION_LIB}")
                set_source_files_properties(extras/gpu/bench_metal_partition.mm
                    PROPERTIES COMPILE_FLAGS "-fobjc-arc")
            else()
                message(STATUS "Metal framework not found — pivco_bench_metal_partition skipped")
            endif()
        endif()

        # EWAH detection (consumed by pivco_bench_golomb under extras/bench/).
        if(EXISTS ${CMAKE_SOURCE_DIR}/ext/ewah/include/ewah/ewah.h)
            set(PIVCO_HAS_EWAH 1)
            message(STATUS "EWAH detected at ext/ewah — pivco_bench_golomb will include EWAH column")
        else()
            set(PIVCO_HAS_EWAH 0)
            message(STATUS "EWAH not detected at ext/ewah — pivco_bench_golomb will skip EWAH column")
        endif()
    endif()

    # All other bench targets (research / micro / one-off comparisons)
    # live in their own CMakeLists.txt under extras/bench/.
    add_subdirectory(extras/bench)
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
