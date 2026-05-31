#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(matmul_clamp_f16_f16_f16p)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(KAI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(KAI_BUILD ${KAI_PATH}/build)
include(FetchContent)

# We treat KleidiAI as an external dependency here.
fetchcontent_declare(
    kleidiai
    SOURCE_DIR ${KAI_PATH}
)

# Make external project available.
fetchcontent_makeavailable(kleidiai)

include_directories(
    ${kleidiai_SOURCE_DIR}/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/)

set(KAI_SOURCES
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c)

set(KAI_HEADERS
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h)

# Files requires to build the executable
add_executable(
    matmul_clamp_f16_f16_f16p matmul_clamp_f16_f16_f16p.cpp
    ${KAI_SOURCES}
    ${KAI_HEADERS}
    )

# Compile args needed for binary.
target_compile_options(matmul_clamp_f16_f16_f16p PRIVATE -march=armv8.2-a+fp16)

target_compile_definitions(matmul_clamp_f16_f16_f16p
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
