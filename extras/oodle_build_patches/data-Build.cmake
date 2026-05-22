s_project(oodle::data::${PROJ_NAME} ${PROJ_TYPE})
s_add_dir(${oodle_data_SOURCE_DIR}/base)
s_add_dir(${oodle_data_SOURCE_DIR}/core)

# Manually wire in the ARM64 / x86 ASM kernels.  OodleUE's smake
# s_add_dir uses aux_source_directory which silently drops .S
# and .nas files.  Patched 2026-05-15 by pivco-huffman.
#
# ARM64: pick the "wide" (Apple M1-scheduled) variant.  Good fit
# for both Apple silicon and Neoverse V2 (Graviton 4).
# x86-64: include generic + BMI2 + Zen2 variants - the .cpp
# does runtime CPU feature detection and dispatches dynamically.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    enable_language(ASM)
    s_add_file_force(${oodle_data_SOURCE_DIR}/core/newlz_huff3_wide.a64.S)
    s_add_file_force(${oodle_data_SOURCE_DIR}/core/newlz_huff6_wide.a64.S)
    s_add_file_force(${oodle_data_SOURCE_DIR}/core/enchuff3c.a64.S)
    s_add_file_force(${oodle_data_SOURCE_DIR}/core/histo.a64.S)
    # tANS decode kernel (wide / M1-scheduled, same rationale as huff).
    s_add_file_force(${oodle_data_SOURCE_DIR}/core/newlz_tans_wide.a64.S)
    if(APPLE)
        # asmlib_arm_a64.inc needs __RADMACARM64__ to pick the Mach-O
        # symbol-mangle path (prepend underscore).  Not set by
        # rrplatform.h - OodleUE's internal build sets it elsewhere.
        set_source_files_properties(
            ${oodle_data_SOURCE_DIR}/core/newlz_huff3_wide.a64.S
            ${oodle_data_SOURCE_DIR}/core/newlz_huff6_wide.a64.S
            ${oodle_data_SOURCE_DIR}/core/enchuff3c.a64.S
            ${oodle_data_SOURCE_DIR}/core/histo.a64.S
            ${oodle_data_SOURCE_DIR}/core/newlz_tans_wide.a64.S
            PROPERTIES COMPILE_OPTIONS "-D__RADMACARM64__"
        )
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    enable_language(ASM_NASM)
    set(PIVCO_NASM_KERNELS
        ${oodle_data_SOURCE_DIR}/core/newlz_huffx64_generic.nas
        ${oodle_data_SOURCE_DIR}/core/newlz_huffx64_bmi2.nas
        ${oodle_data_SOURCE_DIR}/core/newlz_huff_x64_zen2.nas
        ${oodle_data_SOURCE_DIR}/core/newlz_huff6_x64_generic.nas
        ${oodle_data_SOURCE_DIR}/core/newlz_huff6_x64_bmi2.nas
        ${oodle_data_SOURCE_DIR}/core/newlz_huff6_x64_zen2.nas
        ${oodle_data_SOURCE_DIR}/core/enchuff3_x64_generic.nas
        ${oodle_data_SOURCE_DIR}/core/enchuff3_x64_bmi2.nas
        ${oodle_data_SOURCE_DIR}/core/histo_x64_generic.nas
        # tANS decode kernels: generic + BMI2 + BMI2-RaptorLake.
        # newlz_arrays_tans.cpp dispatches dynamically by CPU feature.
        ${oodle_data_SOURCE_DIR}/core/newlz_tans_x64_generic.nas
        ${oodle_data_SOURCE_DIR}/core/newlz_tans_x64_bmi2.nas
        ${oodle_data_SOURCE_DIR}/core/newlz_tans_x64_bmi2_rpl.nas
    )
endif()

s_end_sources()

s_include_directories(PRIVATE ${oodle_data_SOURCE_DIR}/core)
s_include_directories(PRIVATE ${oodle_data_SOURCE_DIR}/core/public)
s_include_directories(INTERFACE ${CMAKE_SOURCE_DIR}/../Engine/Source/Runtime/OodleDataCompression/Sdks/${PROJECT_VERSION}/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i386|i686|x86)$")
    # Original: s_set_arch(AVX2) - replaced with gated version
    # because s_set_arch leaks -march to ASM_NASM which NASM
    # doesn't understand.  Patched 2026-05-15.
    s_compile_options(PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-march=x86-64-v3>)
else()
    if(NOT WIN32)
        # Enable ARM64 dotprod
        s_compile_options(PRIVATE -march=armv8.3-a+dotprod)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Fix clang-cl missing intrin.h inclusion
        s_compile_options(PRIVATE /FIintrin.h)
    endif()
endif()
s_set_cxx_standard(20)
s_compile_definitions(PRIVATE ${PROJ_DEF} OODLE_BUILDING_DATA)

# Tell newlz_arrays_huff.cpp to dispatch into the ASM kernels we
# just added.  Without these defines the .cpp keeps using the
# portable C fallback even if the .S / .nas objects are in the
# lib.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    s_compile_definitions(PRIVATE NEWLZ_ARM64_HUFF_ASM NEWLZ_ARM64_TANS_ASM)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    s_compile_definitions(PRIVATE NEWLZ_X64GENERIC_HUFF_ASM OODLE_HISTO_X64GENERIC_ASM NEWLZ_X64GENERIC_TANS_ASM)
    # CMake doesn't auto-detect .nas as ASM_NASM (only .asm/.nasm),
    # so explicitly set LANGUAGE for each file, then add via
    # target_sources.  smake's s_add_file_force would also work but
    # we already tried that and CMake dropped them silently.
    # Also tell NASM about the output format (defaults to ELF64
    # which is what we want on Linux, but no harm being explicit)
    # and override the inherited -march flag (from s_set_arch
    # AVX2) which NASM doesn't understand.
    set_source_files_properties(${PIVCO_NASM_KERNELS}
        PROPERTIES
            LANGUAGE ASM_NASM
            COMPILE_OPTIONS ""
            COMPILE_FLAGS "-f elf64")
    target_sources(${S_CURRENT_PROJECT_SANITIZED_NAME} PRIVATE
        ${PIVCO_NASM_KERNELS})
endif()
