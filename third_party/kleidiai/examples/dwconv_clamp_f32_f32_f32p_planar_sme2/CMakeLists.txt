#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)
enable_language(ASM)

project(dwconv_clamp_f32_f32_f32p_planar_sme2)

set(CMAKE_CXX_STANDARD 17)
set(KAI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(KAI_BUILD ${KAI_PATH}/build)

include_directories(${KAI_PATH})

# Files requires to build the executable
add_executable(
    dwconv_clamp_f32_f32_f32p_planar_sme2 dwconv_clamp_f32_f32_f32p_planar_sme2.cpp
    "${KAI_PATH}/kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla_asm.S"
    "${KAI_PATH}/kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.c"
    "${KAI_PATH}/kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.c"
    "${KAI_PATH}/kai/kai_common_sme_asm.S"
    )

target_compile_options(dwconv_clamp_f32_f32_f32p_planar_sme2
    PRIVATE "-march=armv8.2-a+sve+sve2;-fno-tree-vectorize"
)

target_compile_definitions(dwconv_clamp_f32_f32_f32p_planar_sme2
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
