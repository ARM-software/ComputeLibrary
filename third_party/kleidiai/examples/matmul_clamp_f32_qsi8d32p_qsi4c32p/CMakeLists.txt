#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(kai_example_matmul_clamp_f32_qsi8d32p_qsi4c32p)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(KLEIDIAI_PATH ../../)
set(MATMUL_PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack/)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_PATH})

# Files requires to build the executable
add_executable(matmul_clamp_f32_qsi8d32p_qsi4c32p
    matmul_clamp_f32_qsi8d32p_qsi4c32p.cpp
    ${MATMUL_PACK_PATH}/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c
    ${MATMUL_PACK_PATH}/kai_lhs_quant_pack_qsi8d32p_f32.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c)

target_compile_options(matmul_clamp_f32_qsi8d32p_qsi4c32p
    PRIVATE -march=armv8.2-a+dotprod+i8mm
)
