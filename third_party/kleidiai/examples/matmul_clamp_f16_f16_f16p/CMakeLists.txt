#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(KLEIDIAI_PATH ../../)
set(MATMUL_PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack/)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_PATH})

# Files requires to build the executable
add_executable(matmul_clamp_f16_f16_f16p
    matmul_clamp_f16_f16_f16p.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
)

target_compile_options(matmul_clamp_f16_f16_f16p
    PRIVATE -march=armv8.2-a+fp16
)

target_compile_definitions(matmul_clamp_f16_f16_f16p
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
