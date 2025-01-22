#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.16)

project(KleidiAI
    VERSION 0.5.0
    LANGUAGES C CXX ASM
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(KLEIDIAI_BUILD_TESTS       "Build unit tests."             ON)
option(KLEIDIAI_BUILD_BENCHMARK   "Build the benchmark tool."     OFF)
option(KLEIDIAI_ENABLE_CLANG_TIDY "Build with Clang-Tidy checks." OFF)

if(KLEIDIAI_ENABLE_CLANG_TIDY)
    set(CMAKE_C_CLANG_TIDY "clang-tidy")
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
endif()

if(MSVC)
    set(KLEIDIAI_WARNING_FLAGS_C
        "/Wall"
    )
else()
    set(KLEIDIAI_WARNING_FLAGS_C
        "-Wall"
        "-Wdisabled-optimization"
        "-Werror"
        "-Wextra"
        "-Wformat-security"
        "-Wformat=2"
        "-Winit-self"
        "-Wno-ignored-attributes"
        "-Wno-misleading-indentation"
        "-Wno-overlength-strings"
        "-Wstrict-overflow=2"
        "-Wswitch-default"
    )

    set(KLEIDIAI_WARNING_FLAGS_CXX
        "-Wctor-dtor-privacy"
        "-Weffc++"
        "-Woverloaded-virtual"
        "-Wsign-promo"
    )
endif()

set(KLEIDIAI_WARNING_FLAGS
    ${KLEIDIAI_WARNING_FLAGS_C}
    $<$<COMPILE_LANGUAGE:CXX>:${KLEIDIAI_WARNING_FLAGS_CXX}>
)

set(KLEIDIAI_MIN_CLANG_VERSION 11)
set(KLEIDIAI_MIN_GNU_VERSION 11)

if(CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_C_COMPILER_VERSION VERSION_LESS ${KLEIDIAI_MIN_CLANG_VERSION})
    message(WARNING "KleidiAI: Using non-supported Clang version. Expected ${KLEIDIAI_MIN_CLANG_VERSION} or newer, received ${CMAKE_C_COMPILER_VERSION}.")
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS ${KLEIDIAI_MIN_GNU_VERSION})
    message(WARNING "KleidiAI: Using non-supported GCC version. Expected ${KLEIDIAI_MIN_GNU_VERSION} or newer, received ${CMAKE_C_COMPILER_VERSION}.")
endif()

set(KLEIDIAI_FILES_SCALAR
    kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
    kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.c
    kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c
    kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c
    kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c
)

set(KLEIDIAI_FILES_NEON_FP16
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
    kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
)

set(KLEIDIAI_FILES_NEON_BF16
    kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p1x4_f32_neon.c
    kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.c
    kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.c
    kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.c
    kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.c
)

set(KLEIDIAI_FILES_NEON_FP16_BF16
    kai/ukernels/matmul/pack/kai_lhs_pack_bf16p8x4_f16_neon.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p12x4biasf16_f16_neon.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p12x4biasf32_f16_neon.c
    kai/ukernels/matmul/matmul_clamp_f16_bf16p_bf16p/kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla.c
)

set(KLEIDIAI_FILES_NEON
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.c
    kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c
    kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c
    kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.c
    kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.c

)

set(KLEIDIAI_FILES_NEON_DOTPROD
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c
)

set(KLEIDIAI_FILES_NEON_I8MM
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c
)

set(KLEIDIAI_FILES_SME
    kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.c
    kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.c
    kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.c
    kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.c
    kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.c
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.c
)

set(KLEIDIAI_FILES_SME2
    kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.c
    kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.c
    kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.c
    kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.c
    kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.c
    kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.c
)

add_library(kleidiai)
target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_SCALAR})

# Selectively enable architecture features.
#
# Note: MSVC compiler does not support inline assembly for Arm® processors hence it cannot be used to compile the
# optimized KleidiAI micro-kernels. It can however still be used to compile scalar routines.
# https://learn.microsoft.com/en-us/cpp/assembler/inline/inline-assembler?view=msvc-170
#
if((CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_NAME STREQUAL "iOS") AND NOT MSVC)
    target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON})
    target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_FP16})
    target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_BF16})
    target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_FP16_BF16})
    target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_DOTPROD})
    target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_I8MM})
    target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_SME})
    target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_SME2})

    set_source_files_properties(${KLEIDIAI_FILES_SCALAR}         PROPERTIES COMPILE_OPTIONS -march=armv8-a${KLEIDIAI_INTERNAL_EXTRA_ARCH})
    set_source_files_properties(${KLEIDIAI_FILES_NEON}           PROPERTIES COMPILE_OPTIONS -march=armv8-a${KLEIDIAI_INTERNAL_EXTRA_ARCH})
    set_source_files_properties(${KLEIDIAI_FILES_NEON_FP16}      PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+fp16${KLEIDIAI_INTERNAL_EXTRA_ARCH})
    set_source_files_properties(${KLEIDIAI_FILES_NEON_BF16}      PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+bf16${KLEIDIAI_INTERNAL_EXTRA_ARCH})
    set_source_files_properties(${KLEIDIAI_FILES_NEON_FP16_BF16} PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+bf16+fp16${KLEIDIAI_INTERNAL_EXTRA_ARCH})
    set_source_files_properties(${KLEIDIAI_FILES_NEON_DOTPROD}   PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+dotprod${KLEIDIAI_INTERNAL_EXTRA_ARCH})
    set_source_files_properties(${KLEIDIAI_FILES_NEON_I8MM}      PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+i8mm${KLEIDIAI_INTERNAL_EXTRA_ARCH})

    # Use -fno-tree-vectorize option to disable compiler based vectorization
    set_source_files_properties(${KLEIDIAI_FILES_SME}          PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
    set_source_files_properties(${KLEIDIAI_FILES_SME2}         PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
endif()

target_include_directories(kleidiai
    PUBLIC  .
)

target_compile_options(kleidiai
    PRIVATE "-Wpedantic"
    PRIVATE ${KLEIDIAI_WARNING_FLAGS}
)

if(KLEIDIAI_BUILD_TESTS)
    include(FetchGTest)
    enable_testing()
    include(GoogleTest)

    add_library(kleidiai_test_framework
        test/common/data_type.cpp
        test/common/data_format.cpp
        test/common/printer.cpp
        test/common/int4.cpp
        test/common/compare.cpp
        test/common/matmul_test_common.cpp
        test/common/matrix_portion.cpp
        test/common/rect.cpp
        test/common/round.cpp
        test/common/bfloat16.cpp
        test/common/float16.cpp
        test/common/cpu_info.cpp
        test/common/sme.cpp

        test/reference/binary_elementwise.cpp
        test/reference/matmul.cpp
        test/reference/matmul_pack.cpp
        test/reference/fill.cpp
        test/reference/pack.cpp
        test/reference/pad.cpp
        test/reference/clamp.cpp
        test/reference/quantize.cpp
        test/reference/reduce.cpp
        test/reference/transpose.cpp
        test/reference/cast.cpp
        test/reference/reorder.cpp
    )

    target_compile_options(kleidiai_test_framework
        PUBLIC ${KLEIDIAI_WARNING_FLAGS}
        PUBLIC -march=armv8.2-a+fp16+bf16${KLEIDIAI_INTERNAL_EXTRA_ARCH}
    )

    set_source_files_properties(test/common/sme.cpp PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve${KLEIDIAI_INTERNAL_EXTRA_ARCH}")

    target_link_libraries(kleidiai_test_framework
        PUBLIC  kleidiai
    )

    add_executable(kleidiai_test
        test/tests/matmul_test.cpp
        test/tests/matmul_clamp_f32_f32_f32p_test.cpp
        test/tests/matmul_clamp_f32_qai8dxp_qsi4cxp_test.cpp
        test/tests/matmul_clamp_f32_qai8dxp_qsi8cxp_test.cpp
        test/tests/matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp
        test/tests/matmul_clamp_f32_qai8dxp_qsi4c32p_test.cpp
        test/tests/matmul_clamp_f16_bf16p_bf16p_test.cpp
        test/tests/matmul_clamp_f32_bf16p_bf16p_test.cpp
        test/tests/matmul_clamp_qai8_qai8p_qsi8cxp_test.cpp
    )

    target_link_libraries(kleidiai_test
        PRIVATE kleidiai_test_framework
        PRIVATE GTest::gtest_main
    )


    # Cross-compiling is a common use case which creates a conflict if DISCOVERY_MODE is set to POST_BUILD (by default)
    # since the host platform does not match the target. Setting the mode to PRE_TEST avoids this conflict. This feature
    # was added in CMake 3.18
    if(CMAKE_CROSSCOMPILING)
        cmake_minimum_required(VERSION 3.18)
    endif()
    gtest_discover_tests(kleidiai_test DISCOVERY_MODE PRE_TEST)
endif()

if(KLEIDIAI_BUILD_BENCHMARK)
    # https://github.com/google/benchmark/issues/351
    if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM"))
        set(CMAKE_CROSSCOMPILING TRUE CACHE INTERNAL "")
        set(HAVE_POSIX_REGEX FALSE CACHE INTERNAL "")
    endif()

    set(BENCHMARK_DOWNLOAD_DEPENDENCIES "OFF")
    set(BENCHMARK_ENABLE_TESTING "OFF")
    set(BENCHMARK_ENABLE_GTEST_TESTS "OFF")

    include(FetchGBench)

    add_executable(kleidiai_benchmark
        benchmark/main.cpp
        benchmark/matmul/matmul_f32_f32p_f32p.cpp
        benchmark/matmul/matmul_f32.cpp
        test/common/cpu_info.cpp)

    target_link_libraries(
        kleidiai_benchmark
        kleidiai
        benchmark::benchmark)

    set(KLEIDIAI_BENCHMARK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(kleidiai_benchmark
        PRIVATE KLEIDIAI_BENCHMARK_INCLUDE_DIR)
endif()
