//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Compute args
static const size_t kai_m_step = 16;
static const size_t kai_n_step = 4;
// Packing args
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
static const size_t kai_kr = 8;
static const size_t kai_sr = 2;
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 2;
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric), and reduction sum (if LHS is
// asymmetric))
static const size_t kai_recip_num_bytes_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 2;
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_bl = 32;

inline static size_t kai_num_bytes_per_block_lhs(size_t bl) {
    return (bl * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs;
}

inline static size_t kai_num_bytes_per_block_rhs(size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    size_t num_bytes_per_block_rhs = (bl / kai_recip_num_bytes_qvalue_rhs) + kai_num_bytes_multiplier_rhs;
    return num_bytes_per_block_rhs;
}

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % kai_bl) == 0);

    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t bl) {
    return kai_mr * kai_num_blocks_per_row(k, bl) * kai_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % kai_bl) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block_rhs(bl);

    size_t rhs_packed_stride = kai_nr * (num_bytes_per_block * num_blocks_per_row);

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(
    size_t m_idx, size_t k, size_t bl) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(
    size_t m,                         //
    size_t n,                         //
    size_t k,                         //
    size_t bl,                        //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row,            //
    size_t dst_stride_col,            //
    float scalar_min,                 //
    float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));

    KAI_UNUSED(scalar_min);
    KAI_UNUSED(scalar_max);

    if (m == 0) {
        return;
    }

    size_t num_blocks = kai_num_blocks_per_row(k, bl);

    __asm__ __volatile__(
        "mov x13, %x[m]\n"
        "mov x12, #0x88\n"
        "cmp x13, #0x10\n"
        "mul x12, %x[num_blocks], x12\n"
        "blt 14f\n"
        "1:"  // Row loop
        "mov x11, %x[rhs_packed]\n"
        "mov x10, %x[n]\n"
        "add x9, %x[dst], %x[dst_stride_row], LSL #4\n"
        "2:"  // Column loop
        "mov x27, %x[lhs_packed]\n"
        "movi v31.16b, #0x0\n"
        "movi v30.16b, #0x0\n"
        "mov x23, %x[num_blocks]\n"
        "movi v29.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "movi v27.16b, #0x0\n"
        "movi v26.16b, #0x0\n"
        "add x22, x27, x12\n"
        "add x21, x22, x12\n"
        "movi v25.16b, #0x0\n"
        "movi v24.16b, #0x0\n"
        "add x20, x21, x12\n"
        "movi v23.16b, #0x0\n"
        "movi v22.16b, #0x0\n"
        "movi v21.16b, #0x0\n"
        "movi v20.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "movi v18.16b, #0x0\n"
        "movi v17.16b, #0x0\n"
        "movi v16.16b, #0x0\n"
        "3:"  // Block loop
        "ldr d15, [x11, #0x0]\n"
        "ldr d1, [x27, #0x0]\n"
        "add x11, x11, #0x8\n"
        "add x27, x27, #0x8\n"
        "ldr q0, [x11, #0x0]\n"
        "ldr q6, [x27, #0x0]\n"
        "movi v12.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "ldr q11, [x11, #0x10]\n"
        "ldr q5, [x27, #0x10]\n"
        "movi v10.4s, #0x0\n"
        "movi v13.4s, #0x0\n"
        "ldr q7, [x11, #0x20]\n"
        "ldr q3, [x27, #0x20]\n"
        "movi v9.16b, #0xf0\n"
        "fcvtl v14.4s, v15.4h\n"
        "ldr q15, [x11, #0x30]\n"
        "ldr q4, [x27, #0x30]\n"
        "shl v2.16b, v0.16b, #0x4\n"
        "fcvtl v1.4s, v1.4h\n"
        "and v0.16b, v0.16b, v9.16b\n"
        "add x11, x11, #0x40\n"
        ".inst 0x4f86e04c  // sdot v12.4s, v2.16b, v6.4b[0]\n"
        ".inst 0x4fa6e048  // sdot v8.4s, v2.16b, v6.4b[1]\n"
        ".inst 0x4f86e84a  // sdot v10.4s, v2.16b, v6.4b[2]\n"
        ".inst 0x4fa6e84d  // sdot v13.4s, v2.16b, v6.4b[3]\n"
        "shl v6.16b, v11.16b, #0x4\n"
        "and v11.16b, v11.16b, v9.16b\n"
        ".inst 0x4f85e0cc  // sdot v12.4s, v6.16b, v5.4b[0]\n"
        ".inst 0x4fa5e0c8  // sdot v8.4s, v6.16b, v5.4b[1]\n"
        ".inst 0x4f85e8ca  // sdot v10.4s, v6.16b, v5.4b[2]\n"
        ".inst 0x4fa5e8cd  // sdot v13.4s, v6.16b, v5.4b[3]\n"
        "shl v5.16b, v7.16b, #0x4\n"
        "and v7.16b, v7.16b, v9.16b\n"
        ".inst 0x4f83e0ac  // sdot v12.4s, v5.16b, v3.4b[0]\n"
        ".inst 0x4fa3e0a8  // sdot v8.4s, v5.16b, v3.4b[1]\n"
        ".inst 0x4f83e8aa  // sdot v10.4s, v5.16b, v3.4b[2]\n"
        ".inst 0x4fa3e8ad  // sdot v13.4s, v5.16b, v3.4b[3]\n"
        "shl v3.16b, v15.16b, #0x4\n"
        "and v15.16b, v15.16b, v9.16b\n"
        "ldr q9, [x27, #0x40]\n"
        ".inst 0x4f84e06c  // sdot v12.4s, v3.16b, v4.4b[0]\n"
        ".inst 0x4fa4e068  // sdot v8.4s, v3.16b, v4.4b[1]\n"
        ".inst 0x4f84e86a  // sdot v10.4s, v3.16b, v4.4b[2]\n"
        ".inst 0x4fa4e86d  // sdot v13.4s, v3.16b, v4.4b[3]\n"
        "ldr q4, [x27, #0x50]\n"
        ".inst 0x4f89e00c  // sdot v12.4s, v0.16b, v9.4b[0]\n"
        ".inst 0x4fa9e008  // sdot v8.4s, v0.16b, v9.4b[1]\n"
        ".inst 0x4f89e80a  // sdot v10.4s, v0.16b, v9.4b[2]\n"
        ".inst 0x4fa9e80d  // sdot v13.4s, v0.16b, v9.4b[3]\n"
        "ldr q9, [x27, #0x60]\n"
        ".inst 0x4f84e16c  // sdot v12.4s, v11.16b, v4.4b[0]\n"
        ".inst 0x4fa4e168  // sdot v8.4s, v11.16b, v4.4b[1]\n"
        ".inst 0x4f84e96a  // sdot v10.4s, v11.16b, v4.4b[2]\n"
        ".inst 0x4fa4e96d  // sdot v13.4s, v11.16b, v4.4b[3]\n"
        "ldr q4, [x27, #0x70]\n"
        "add x27, x27, #0x80\n"
        ".inst 0x4f89e0ec  // sdot v12.4s, v7.16b, v9.4b[0]\n"
        ".inst 0x4fa9e0e8  // sdot v8.4s, v7.16b, v9.4b[1]\n"
        ".inst 0x4f89e8ea  // sdot v10.4s, v7.16b, v9.4b[2]\n"
        ".inst 0x4fa9e8ed  // sdot v13.4s, v7.16b, v9.4b[3]\n"
        "fmul v9.4s, v14.4s, v1.s[0]\n"
        ".inst 0x4f84e1ec  // sdot v12.4s, v15.16b, v4.4b[0]\n"
        ".inst 0x4fa4e1e8  // sdot v8.4s, v15.16b, v4.4b[1]\n"
        ".inst 0x4f84e9ea  // sdot v10.4s, v15.16b, v4.4b[2]\n"
        ".inst 0x4fa4e9ed  // sdot v13.4s, v15.16b, v4.4b[3]\n"
        "fmul v4.4s, v14.4s, v1.s[1]\n"
        "scvtf v12.4s, v12.4s, #0x4\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v13.4s, v13.4s, #0x4\n"
        "fmla v31.4s, v12.4s, v9.4s\n"
        "fmul v12.4s, v14.4s, v1.s[2]\n"
        "fmul v1.4s, v14.4s, v1.s[3]\n"
        "fmla v30.4s, v8.4s, v4.4s\n"
        "fmla v29.4s, v10.4s, v12.4s\n"
        "fmla v28.4s, v13.4s, v1.4s\n"
        "ldr d13, [x22, #0x0]\n"
        "add x22, x22, #0x8\n"
        "movi v10.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "ldr q12, [x22, #0x0]\n"
        "ldr q1, [x22, #0x10]\n"
        "movi v9.4s, #0x0\n"
        "movi v4.4s, #0x0\n"
        "fcvtl v13.4s, v13.4h\n"
        ".inst 0x4f8ce04a  // sdot v10.4s, v2.16b, v12.4b[0]\n"
        ".inst 0x4face048  // sdot v8.4s, v2.16b, v12.4b[1]\n"
        ".inst 0x4f8ce849  // sdot v9.4s, v2.16b, v12.4b[2]\n"
        ".inst 0x4face844  // sdot v4.4s, v2.16b, v12.4b[3]\n"
        "ldr q12, [x22, #0x20]\n"
        ".inst 0x4f81e0ca  // sdot v10.4s, v6.16b, v1.4b[0]\n"
        ".inst 0x4fa1e0c8  // sdot v8.4s, v6.16b, v1.4b[1]\n"
        ".inst 0x4f81e8c9  // sdot v9.4s, v6.16b, v1.4b[2]\n"
        ".inst 0x4fa1e8c4  // sdot v4.4s, v6.16b, v1.4b[3]\n"
        "ldr q1, [x22, #0x30]\n"
        ".inst 0x4f8ce0aa  // sdot v10.4s, v5.16b, v12.4b[0]\n"
        ".inst 0x4face0a8  // sdot v8.4s, v5.16b, v12.4b[1]\n"
        ".inst 0x4f8ce8a9  // sdot v9.4s, v5.16b, v12.4b[2]\n"
        ".inst 0x4face8a4  // sdot v4.4s, v5.16b, v12.4b[3]\n"
        "ldr q12, [x22, #0x40]\n"
        ".inst 0x4f81e06a  // sdot v10.4s, v3.16b, v1.4b[0]\n"
        ".inst 0x4fa1e068  // sdot v8.4s, v3.16b, v1.4b[1]\n"
        ".inst 0x4f81e869  // sdot v9.4s, v3.16b, v1.4b[2]\n"
        ".inst 0x4fa1e864  // sdot v4.4s, v3.16b, v1.4b[3]\n"
        "ldr q1, [x22, #0x50]\n"
        ".inst 0x4f8ce00a  // sdot v10.4s, v0.16b, v12.4b[0]\n"
        ".inst 0x4face008  // sdot v8.4s, v0.16b, v12.4b[1]\n"
        ".inst 0x4f8ce809  // sdot v9.4s, v0.16b, v12.4b[2]\n"
        ".inst 0x4face804  // sdot v4.4s, v0.16b, v12.4b[3]\n"
        "ldr q12, [x22, #0x60]\n"
        ".inst 0x4f81e16a  // sdot v10.4s, v11.16b, v1.4b[0]\n"
        ".inst 0x4fa1e168  // sdot v8.4s, v11.16b, v1.4b[1]\n"
        ".inst 0x4f81e969  // sdot v9.4s, v11.16b, v1.4b[2]\n"
        ".inst 0x4fa1e964  // sdot v4.4s, v11.16b, v1.4b[3]\n"
        "ldr q1, [x22, #0x70]\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4f8ce0ea  // sdot v10.4s, v7.16b, v12.4b[0]\n"
        ".inst 0x4face0e8  // sdot v8.4s, v7.16b, v12.4b[1]\n"
        ".inst 0x4f8ce8e9  // sdot v9.4s, v7.16b, v12.4b[2]\n"
        ".inst 0x4face8e4  // sdot v4.4s, v7.16b, v12.4b[3]\n"
        "fmul v12.4s, v14.4s, v13.s[0]\n"
        ".inst 0x4f81e1ea  // sdot v10.4s, v15.16b, v1.4b[0]\n"
        ".inst 0x4fa1e1e8  // sdot v8.4s, v15.16b, v1.4b[1]\n"
        ".inst 0x4f81e9e9  // sdot v9.4s, v15.16b, v1.4b[2]\n"
        ".inst 0x4fa1e9e4  // sdot v4.4s, v15.16b, v1.4b[3]\n"
        "fmul v1.4s, v14.4s, v13.s[1]\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "fmla v27.4s, v10.4s, v12.4s\n"
        "fmul v10.4s, v14.4s, v13.s[2]\n"
        "fmul v13.4s, v14.4s, v13.s[3]\n"
        "fmla v26.4s, v8.4s, v1.4s\n"
        "fmla v25.4s, v9.4s, v10.4s\n"
        "fmla v24.4s, v4.4s, v13.4s\n"
        "ldr d12, [x21, #0x0]\n"
        "add x21, x21, #0x8\n"
        "movi v9.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "ldr q1, [x21, #0x0]\n"
        "ldr q4, [x21, #0x10]\n"
        "movi v10.4s, #0x0\n"
        "movi v13.4s, #0x0\n"
        "fcvtl v12.4s, v12.4h\n"
        ".inst 0x4f81e049  // sdot v9.4s, v2.16b, v1.4b[0]\n"
        ".inst 0x4fa1e048  // sdot v8.4s, v2.16b, v1.4b[1]\n"
        ".inst 0x4f81e84a  // sdot v10.4s, v2.16b, v1.4b[2]\n"
        ".inst 0x4fa1e84d  // sdot v13.4s, v2.16b, v1.4b[3]\n"
        "ldr q1, [x21, #0x20]\n"
        ".inst 0x4f84e0c9  // sdot v9.4s, v6.16b, v4.4b[0]\n"
        ".inst 0x4fa4e0c8  // sdot v8.4s, v6.16b, v4.4b[1]\n"
        ".inst 0x4f84e8ca  // sdot v10.4s, v6.16b, v4.4b[2]\n"
        ".inst 0x4fa4e8cd  // sdot v13.4s, v6.16b, v4.4b[3]\n"
        "ldr q4, [x21, #0x30]\n"
        ".inst 0x4f81e0a9  // sdot v9.4s, v5.16b, v1.4b[0]\n"
        ".inst 0x4fa1e0a8  // sdot v8.4s, v5.16b, v1.4b[1]\n"
        ".inst 0x4f81e8aa  // sdot v10.4s, v5.16b, v1.4b[2]\n"
        ".inst 0x4fa1e8ad  // sdot v13.4s, v5.16b, v1.4b[3]\n"
        "ldr q1, [x21, #0x40]\n"
        ".inst 0x4f84e069  // sdot v9.4s, v3.16b, v4.4b[0]\n"
        ".inst 0x4fa4e068  // sdot v8.4s, v3.16b, v4.4b[1]\n"
        ".inst 0x4f84e86a  // sdot v10.4s, v3.16b, v4.4b[2]\n"
        ".inst 0x4fa4e86d  // sdot v13.4s, v3.16b, v4.4b[3]\n"
        "ldr q4, [x21, #0x50]\n"
        ".inst 0x4f81e009  // sdot v9.4s, v0.16b, v1.4b[0]\n"
        ".inst 0x4fa1e008  // sdot v8.4s, v0.16b, v1.4b[1]\n"
        ".inst 0x4f81e80a  // sdot v10.4s, v0.16b, v1.4b[2]\n"
        ".inst 0x4fa1e80d  // sdot v13.4s, v0.16b, v1.4b[3]\n"
        "ldr q1, [x21, #0x60]\n"
        ".inst 0x4f84e169  // sdot v9.4s, v11.16b, v4.4b[0]\n"
        ".inst 0x4fa4e168  // sdot v8.4s, v11.16b, v4.4b[1]\n"
        ".inst 0x4f84e96a  // sdot v10.4s, v11.16b, v4.4b[2]\n"
        ".inst 0x4fa4e96d  // sdot v13.4s, v11.16b, v4.4b[3]\n"
        "ldr q4, [x21, #0x70]\n"
        "add x21, x21, #0x80\n"
        ".inst 0x4f81e0e9  // sdot v9.4s, v7.16b, v1.4b[0]\n"
        ".inst 0x4fa1e0e8  // sdot v8.4s, v7.16b, v1.4b[1]\n"
        ".inst 0x4f81e8ea  // sdot v10.4s, v7.16b, v1.4b[2]\n"
        ".inst 0x4fa1e8ed  // sdot v13.4s, v7.16b, v1.4b[3]\n"
        "fmul v1.4s, v14.4s, v12.s[0]\n"
        ".inst 0x4f84e1e9  // sdot v9.4s, v15.16b, v4.4b[0]\n"
        ".inst 0x4fa4e1e8  // sdot v8.4s, v15.16b, v4.4b[1]\n"
        ".inst 0x4f84e9ea  // sdot v10.4s, v15.16b, v4.4b[2]\n"
        ".inst 0x4fa4e9ed  // sdot v13.4s, v15.16b, v4.4b[3]\n"
        "fmul v4.4s, v14.4s, v12.s[1]\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v13.4s, v13.4s, #0x4\n"
        "fmla v23.4s, v9.4s, v1.4s\n"
        "fmul v1.4s, v14.4s, v12.s[2]\n"
        "fmul v9.4s, v14.4s, v12.s[3]\n"
        "fmla v22.4s, v8.4s, v4.4s\n"
        "fmla v21.4s, v10.4s, v1.4s\n"
        "fmla v20.4s, v13.4s, v9.4s\n"
        "ldr d13, [x20, #0x0]\n"
        "add x20, x20, #0x8\n"
        "movi v12.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "ldr q4, [x20, #0x0]\n"
        "ldr q1, [x20, #0x10]\n"
        "movi v10.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "fcvtl v13.4s, v13.4h\n"
        ".inst 0x4f84e04c  // sdot v12.4s, v2.16b, v4.4b[0]\n"
        ".inst 0x4fa4e048  // sdot v8.4s, v2.16b, v4.4b[1]\n"
        ".inst 0x4f84e84a  // sdot v10.4s, v2.16b, v4.4b[2]\n"
        ".inst 0x4fa4e849  // sdot v9.4s, v2.16b, v4.4b[3]\n"
        "ldr q2, [x20, #0x20]\n"
        "ldr q4, [x20, #0x30]\n"
        ".inst 0x4f81e0cc  // sdot v12.4s, v6.16b, v1.4b[0]\n"
        ".inst 0x4fa1e0c8  // sdot v8.4s, v6.16b, v1.4b[1]\n"
        ".inst 0x4f81e8ca  // sdot v10.4s, v6.16b, v1.4b[2]\n"
        ".inst 0x4fa1e8c9  // sdot v9.4s, v6.16b, v1.4b[3]\n"
        "ldr q1, [x20, #0x40]\n"
        "ldr q6, [x20, #0x50]\n"
        ".inst 0x4f82e0ac  // sdot v12.4s, v5.16b, v2.4b[0]\n"
        ".inst 0x4fa2e0a8  // sdot v8.4s, v5.16b, v2.4b[1]\n"
        ".inst 0x4f82e8aa  // sdot v10.4s, v5.16b, v2.4b[2]\n"
        ".inst 0x4fa2e8a9  // sdot v9.4s, v5.16b, v2.4b[3]\n"
        "ldr q2, [x20, #0x60]\n"
        "ldr q5, [x20, #0x70]\n"
        "add x20, x20, #0x80\n"
        ".inst 0x4f84e06c  // sdot v12.4s, v3.16b, v4.4b[0]\n"
        ".inst 0x4fa4e068  // sdot v8.4s, v3.16b, v4.4b[1]\n"
        ".inst 0x4f84e86a  // sdot v10.4s, v3.16b, v4.4b[2]\n"
        ".inst 0x4fa4e869  // sdot v9.4s, v3.16b, v4.4b[3]\n"
        "fmul v3.4s, v14.4s, v13.s[0]\n"
        "fmul v4.4s, v14.4s, v13.s[1]\n"
        ".inst 0x4f81e00c  // sdot v12.4s, v0.16b, v1.4b[0]\n"
        ".inst 0x4fa1e008  // sdot v8.4s, v0.16b, v1.4b[1]\n"
        ".inst 0x4f81e80a  // sdot v10.4s, v0.16b, v1.4b[2]\n"
        ".inst 0x4fa1e809  // sdot v9.4s, v0.16b, v1.4b[3]\n"
        "fmul v1.4s, v14.4s, v13.s[2]\n"
        "fmul v0.4s, v14.4s, v13.s[3]\n"
        ".inst 0x4f86e16c  // sdot v12.4s, v11.16b, v6.4b[0]\n"
        ".inst 0x4fa6e168  // sdot v8.4s, v11.16b, v6.4b[1]\n"
        ".inst 0x4f86e96a  // sdot v10.4s, v11.16b, v6.4b[2]\n"
        ".inst 0x4fa6e969  // sdot v9.4s, v11.16b, v6.4b[3]\n"
        ".inst 0x4f82e0ec  // sdot v12.4s, v7.16b, v2.4b[0]\n"
        ".inst 0x4fa2e0e8  // sdot v8.4s, v7.16b, v2.4b[1]\n"
        ".inst 0x4f82e8ea  // sdot v10.4s, v7.16b, v2.4b[2]\n"
        ".inst 0x4fa2e8e9  // sdot v9.4s, v7.16b, v2.4b[3]\n"
        ".inst 0x4f85e1ec  // sdot v12.4s, v15.16b, v5.4b[0]\n"
        ".inst 0x4fa5e1e8  // sdot v8.4s, v15.16b, v5.4b[1]\n"
        ".inst 0x4f85e9ea  // sdot v10.4s, v15.16b, v5.4b[2]\n"
        ".inst 0x4fa5e9e9  // sdot v9.4s, v15.16b, v5.4b[3]\n"
        "scvtf v12.4s, v12.4s, #0x4\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "fmla v19.4s, v12.4s, v3.4s\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "fmla v18.4s, v8.4s, v4.4s\n"
        "fmla v17.4s, v10.4s, v1.4s\n"
        "fmla v16.4s, v9.4s, v0.4s\n"
        "subs x23, x23, #0x1\n"
        "bgt 3b\n"
        "cmp x10, #0x4\n"
        "blt 8f\n"
        "mov x20, %x[dst]\n"
        "str q31, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q29, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q28, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q27, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q26, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q24, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q21, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q20, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q17, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q16, [x20, #0x0]\n"
        "b 13f\n"
        "8:"  // Partial output
        "mov x28, %x[dst]\n"
        "add x26, x28, %x[dst_stride_row], LSL #2\n"
        "add x25, x26, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row]\n"
        "add x23, x25, %x[dst_stride_row]\n"
        "add x22, x28, %x[dst_stride_row], LSL #1\n"
        "add x21, x28, %x[dst_stride_row]\n"
        "add x20, x22, %x[dst_stride_row]\n"
        "add x27, x23, %x[dst_stride_row]\n"
        "tbz x10, #1, 9f\n"
        "st1 { v24.d }[0], [x23], #0x8\n"
        "st1 { v25.d }[0], [x25], #0x8\n"
        "st1 { v26.d }[0], [x24], #0x8\n"
        "st1 { v27.d }[0], [x26], #0x8\n"
        "st1 { v28.d }[0], [x20], #0x8\n"
        "st1 { v29.d }[0], [x22], #0x8\n"
        "st1 { v30.d }[0], [x21], #0x8\n"
        "st1 { v31.d }[0], [x28], #0x8\n"
        "tbz x10, #0, 10f\n"
        "st1 { v24.s }[2], [x23]\n"
        "st1 { v25.s }[2], [x25]\n"
        "st1 { v26.s }[2], [x24]\n"
        "st1 { v27.s }[2], [x26]\n"
        "st1 { v28.s }[2], [x20]\n"
        "st1 { v29.s }[2], [x22]\n"
        "st1 { v30.s }[2], [x21]\n"
        "st1 { v31.s }[2], [x28]\n"
        "b 10f\n"
        "9:"  // Output block 0: partial_1_0
        "st1 { v24.s }[0], [x23]\n"
        "st1 { v25.s }[0], [x25]\n"
        "st1 { v26.s }[0], [x24]\n"
        "st1 { v27.s }[0], [x26]\n"
        "st1 { v28.s }[0], [x20]\n"
        "st1 { v29.s }[0], [x22]\n"
        "st1 { v30.s }[0], [x21]\n"
        "st1 { v31.s }[0], [x28]\n"
        "10:"  // Output block 0: Done
        "add x26, x27, %x[dst_stride_row], LSL #2\n"
        "add x25, x27, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row], LSL #1\n"
        "add x23, x27, %x[dst_stride_row]\n"
        "add x22, x25, %x[dst_stride_row]\n"
        "add x21, x26, %x[dst_stride_row]\n"
        "add x20, x24, %x[dst_stride_row]\n"
        "tbz x10, #1, 11f\n"
        "st1 { v16.d }[0], [x20], #0x8\n"
        "st1 { v17.d }[0], [x24], #0x8\n"
        "st1 { v18.d }[0], [x21], #0x8\n"
        "st1 { v19.d }[0], [x26], #0x8\n"
        "st1 { v20.d }[0], [x22], #0x8\n"
        "st1 { v21.d }[0], [x25], #0x8\n"
        "st1 { v22.d }[0], [x23], #0x8\n"
        "st1 { v23.d }[0], [x27], #0x8\n"
        "tbz x10, #0, 12f\n"
        "st1 { v16.s }[2], [x20]\n"
        "st1 { v17.s }[2], [x24]\n"
        "st1 { v18.s }[2], [x21]\n"
        "st1 { v19.s }[2], [x26]\n"
        "st1 { v20.s }[2], [x22]\n"
        "st1 { v21.s }[2], [x25]\n"
        "st1 { v22.s }[2], [x23]\n"
        "st1 { v23.s }[2], [x27]\n"
        "b 12f\n"
        "11:"  // Output block 1: partial_1_0
        "st1 { v16.s }[0], [x20]\n"
        "st1 { v17.s }[0], [x24]\n"
        "st1 { v18.s }[0], [x21]\n"
        "st1 { v19.s }[0], [x26]\n"
        "st1 { v20.s }[0], [x22]\n"
        "st1 { v21.s }[0], [x25]\n"
        "st1 { v22.s }[0], [x23]\n"
        "st1 { v23.s }[0], [x27]\n"
        "12:"  // Output block 1: Done
        "13:"  // Output stage exit
        "subs x10, x10, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "mov x20, #0x4\n"
        "sub x13, x13, #0x10\n"
        "cmp x13, #0x10\n"
        "mov %x[dst], x9\n"
        "madd %x[lhs_packed], x20, x12, %x[lhs_packed]\n"
        "bge 1b\n"
        "14:"  // Row loop skip
        "cbz x13, 23f\n"
        "15:"  // Row tail: Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "16:"  // Row tail: Column loop
        "movi v31.16b, #0x0\n"
        "movi v30.16b, #0x0\n"
        "mov x27, %x[lhs_packed]\n"
        "mov x20, %x[num_blocks]\n"
        "movi v29.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "17:"  // Row tail: Block loop
        "ldr d16, [x26, #0x0]\n"
        "ldr d11, [x27, #0x0]\n"
        "add x26, x26, #0x8\n"
        "add x27, x27, #0x8\n"
        "ldr q10, [x26, #0x0]\n"
        "ldr q18, [x27, #0x0]\n"
        "movi v9.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "ldr q7, [x26, #0x10]\n"
        "ldr q6, [x27, #0x10]\n"
        "movi v5.4s, #0x0\n"
        "movi v4.4s, #0x0\n"
        "ldr q3, [x26, #0x20]\n"
        "ldr q2, [x27, #0x20]\n"
        "movi v17.16b, #0xf0\n"
        "fcvtl v1.4s, v16.4h\n"
        "ldr q0, [x26, #0x30]\n"
        "ldr q27, [x27, #0x30]\n"
        "shl v16.16b, v10.16b, #0x4\n"
        "fcvtl v11.4s, v11.4h\n"
        "ldr q26, [x27, #0x40]\n"
        "ldr q25, [x27, #0x50]\n"
        "shl v24.16b, v7.16b, #0x4\n"
        "and v10.16b, v10.16b, v17.16b\n"
        "ldr q23, [x27, #0x60]\n"
        "ldr q22, [x27, #0x70]\n"
        "shl v21.16b, v3.16b, #0x4\n"
        "and v7.16b, v7.16b, v17.16b\n"
        ".inst 0x4f92e209  // sdot v9.4s, v16.16b, v18.4b[0]\n"
        ".inst 0x4fb2e208  // sdot v8.4s, v16.16b, v18.4b[1]\n"
        "shl v20.16b, v0.16b, #0x4\n"
        "add x26, x26, #0x40\n"
        ".inst 0x4f92ea05  // sdot v5.4s, v16.16b, v18.4b[2]\n"
        ".inst 0x4fb2ea04  // sdot v4.4s, v16.16b, v18.4b[3]\n"
        "and v3.16b, v3.16b, v17.16b\n"
        "add x27, x27, #0x80\n"
        "and v0.16b, v0.16b, v17.16b\n"
        "fmul v19.4s, v1.4s, v11.s[0]\n"
        "fmul v18.4s, v1.4s, v11.s[1]\n"
        "fmul v17.4s, v1.4s, v11.s[2]\n"
        ".inst 0x4f86e309  // sdot v9.4s, v24.16b, v6.4b[0]\n"
        ".inst 0x4fa6e308  // sdot v8.4s, v24.16b, v6.4b[1]\n"
        "fmul v16.4s, v1.4s, v11.s[3]\n"
        ".inst 0x4f86eb05  // sdot v5.4s, v24.16b, v6.4b[2]\n"
        ".inst 0x4fa6eb04  // sdot v4.4s, v24.16b, v6.4b[3]\n"
        ".inst 0x4f82e2a9  // sdot v9.4s, v21.16b, v2.4b[0]\n"
        ".inst 0x4fa2e2a8  // sdot v8.4s, v21.16b, v2.4b[1]\n"
        ".inst 0x4f82eaa5  // sdot v5.4s, v21.16b, v2.4b[2]\n"
        ".inst 0x4fa2eaa4  // sdot v4.4s, v21.16b, v2.4b[3]\n"
        ".inst 0x4f9be289  // sdot v9.4s, v20.16b, v27.4b[0]\n"
        ".inst 0x4fbbe288  // sdot v8.4s, v20.16b, v27.4b[1]\n"
        ".inst 0x4f9bea85  // sdot v5.4s, v20.16b, v27.4b[2]\n"
        ".inst 0x4fbbea84  // sdot v4.4s, v20.16b, v27.4b[3]\n"
        ".inst 0x4f9ae149  // sdot v9.4s, v10.16b, v26.4b[0]\n"
        ".inst 0x4fbae148  // sdot v8.4s, v10.16b, v26.4b[1]\n"
        ".inst 0x4f9ae945  // sdot v5.4s, v10.16b, v26.4b[2]\n"
        ".inst 0x4fbae944  // sdot v4.4s, v10.16b, v26.4b[3]\n"
        ".inst 0x4f99e0e9  // sdot v9.4s, v7.16b, v25.4b[0]\n"
        ".inst 0x4fb9e0e8  // sdot v8.4s, v7.16b, v25.4b[1]\n"
        ".inst 0x4f99e8e5  // sdot v5.4s, v7.16b, v25.4b[2]\n"
        ".inst 0x4fb9e8e4  // sdot v4.4s, v7.16b, v25.4b[3]\n"
        ".inst 0x4f97e069  // sdot v9.4s, v3.16b, v23.4b[0]\n"
        ".inst 0x4fb7e068  // sdot v8.4s, v3.16b, v23.4b[1]\n"
        ".inst 0x4f97e865  // sdot v5.4s, v3.16b, v23.4b[2]\n"
        ".inst 0x4fb7e864  // sdot v4.4s, v3.16b, v23.4b[3]\n"
        ".inst 0x4f96e009  // sdot v9.4s, v0.16b, v22.4b[0]\n"
        ".inst 0x4fb6e008  // sdot v8.4s, v0.16b, v22.4b[1]\n"
        ".inst 0x4f96e805  // sdot v5.4s, v0.16b, v22.4b[2]\n"
        ".inst 0x4fb6e804  // sdot v4.4s, v0.16b, v22.4b[3]\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "scvtf v5.4s, v5.4s, #0x4\n"
        "fmla v31.4s, v9.4s, v19.4s\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "fmla v30.4s, v8.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v17.4s\n"
        "fmla v28.4s, v4.4s, v16.4s\n"
        "subs x20, x20, #0x1\n"
        "bgt 17b\n"
        "cmp x25, #0x4\n"
        "blt 19f\n"
        "mov x20, %x[dst]\n"
        "cmp x13, #0x1\n"
        "str q31, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 22f\n"
        "cmp x13, #0x2\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 22f\n"
        "cmp x13, #0x3\n"
        "str q29, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 22f\n"
        "str q28, [x20, #0x0]\n"
        "b 22f\n"
        "19:"  // Row tail: Partial output
        "mov x23, %x[dst]\n"
        "cmp x13, #0x1\n"
        "add x22, x23, %x[dst_stride_row]\n"
        "csel x22, x22, x23, GT\n"
        "cmp x13, #0x2\n"
        "add x21, x23, %x[dst_stride_row], LSL #1\n"
        "csel x21, x21, x22, GT\n"
        "cmp x13, #0x3\n"
        "add x20, x21, %x[dst_stride_row]\n"
        "csel x20, x20, x21, GT\n"
        "tbz x25, #1, 20f\n"
        "st1 { v28.d }[0], [x20], #0x8\n"
        "st1 { v29.d }[0], [x21], #0x8\n"
        "st1 { v30.d }[0], [x22], #0x8\n"
        "st1 { v31.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 21f\n"
        "st1 { v28.s }[2], [x20]\n"
        "st1 { v29.s }[2], [x21]\n"
        "st1 { v30.s }[2], [x22]\n"
        "st1 { v31.s }[2], [x23]\n"
        "b 21f\n"
        "20:"  // Row tail: Output block 0: partial_1_0
        "st1 { v28.s }[0], [x20]\n"
        "st1 { v29.s }[0], [x21]\n"
        "st1 { v30.s }[0], [x22]\n"
        "st1 { v31.s }[0], [x23]\n"
        "21:"  // Row tail: Output block 0: Done
        "22:"  // Row tail: Output stage exit
        "subs x25, x25, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 16b\n"
        "subs x13, x13, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x12\n"
        "mov %x[dst], x24\n"
        "bgt 15b\n"
        "23:"  // Row tail: Row loop skip
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n), [num_blocks] "r"(num_blocks),
          [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v2", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v3", "v30", "v31", "v4", "v5", "v6", "v7",
          "v8", "v9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9");
}

#endif  // Architectural features check.
