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
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_PATH})

# Files requires to build the executable
add_executable(matmul_clamp_f32_qai8dxp_qsi4cxp
    matmul_clamp_f32_qai8dxp_qsi4cxp.cpp
    ${MATMUL_PACK_PATH}/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.c
    ${MATMUL_PACK_PATH}/kai_lhs_quant_pack_qai8dxp_f32.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.c
    )

# Compile with DotProd and I8MM features enabled
target_compile_options(matmul_clamp_f32_qai8dxp_qsi4cxp PRIVATE -march=armv8.2-a+dotprod+i8mm)
