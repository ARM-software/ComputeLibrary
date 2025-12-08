#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(conv2d_imatmul_clamp_f16_f16_f16p_sme2)

enable_language(ASM)

set(CMAKE_CXX_STANDARD 17)
set(KAI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(KAI_BUILD ${KAI_PATH}/build)

include_directories(${KAI_PATH})

set(KAI_SOURCES
    ${KAI_PATH}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa_asm.S
    ${KAI_PATH}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.c
    ${KAI_PATH}/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme_asm.S
    ${KAI_PATH}/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.c
    ${KAI_PATH}/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme_asm.S
    ${KAI_PATH}/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.c
    ${KAI_PATH}/kai/kai_common_sme_asm.S
)

# Files requires to build the executable
add_executable(conv2d_imatmul_clamp_f16_f16_f16p_sme2
    conv2d_imatmul_clamp_f16_f16_f16p.cpp
    ${KAI_SOURCES}
)

target_compile_options(conv2d_imatmul_clamp_f16_f16_f16p_sme2
    PRIVATE "-march=armv8.2-a+sve+sve2;-fno-tree-vectorize"
)

target_compile_definitions(conv2d_imatmul_clamp_f16_f16_f16p_sme2
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
