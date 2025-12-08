//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "i8mm extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 16;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
static const size_t kai_kr = 16;
static const size_t kai_sr = 2;
static const size_t kai_bl = 32;
static const size_t kai_num_bytes_multiplier = sizeof(uint16_t);

inline static size_t kai_num_bytes_per_block_lhs(void) {
    return kai_bl * sizeof(int8_t) + kai_num_bytes_multiplier;
}

inline static size_t kai_num_bytes_per_block_rhs(void) {
    return (kai_bl / 2) * sizeof(int8_t) + kai_num_bytes_multiplier;
}

inline static size_t kai_num_blocks_per_row(size_t k) {
    KAI_ASSUME((k % kai_bl) == 0);
    return k / kai_bl;
}

inline static size_t kai_lhs_packed_stride(size_t k) {
    return kai_mr * kai_num_blocks_per_row(k) * kai_num_bytes_per_block_lhs();
}

inline static size_t kai_rhs_packed_stride(size_t k) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % kai_bl) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k);
    const size_t num_bytes_per_block = kai_num_bytes_per_block_rhs();

    return kai_nr * (num_bytes_per_block * num_blocks_per_row);
}

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
    size_t m_idx, size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
    size_t m, size_t n, size_t k, size_t bl, const void* lhs_packed, const void* rhs_packed,
    float* dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME(k % kai_bl == 0);
    KAI_ASSUME(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t num_blocks = k / kai_bl;
    float clamp_vals[2] = {scalar_min, scalar_max};

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
        "ldr d0, [x11, #0x0]\n"
        "ldr d3, [x27, #0x0]\n"
        "add x11, x11, #0x8\n"
        "add x27, x27, #0x8\n"
        "ldr q12, [x11, #0x0]\n"
        "ldr q4, [x11, #0x10]\n"
        "movi v5.4s, #0x0\n"
        "movi v14.4s, #0x0\n"
        "ldr q9, [x27, #0x0]\n"
        "ldr q10, [x27, #0x10]\n"
        "movi v7.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "ldr q2, [x11, #0x20]\n"
        "ldr q11, [x11, #0x30]\n"
        "movi v1.16b, #0xf0\n"
        "fcvtl v6.4s, v0.4h\n"
        "ldr q15, [x27, #0x20]\n"
        "shl v13.16b, v12.16b, #0x4\n"
        "shl v0.16b, v4.16b, #0x4\n"
        "add x11, x11, #0x40\n"
        "and v12.16b, v12.16b, v1.16b\n"
        "and v4.16b, v4.16b, v1.16b\n"
        "fcvtl v3.4s, v3.4h\n"
        ".inst 0x4e8da525  // smmla v5.4s, v9.16b, v13.16b\n"
        ".inst 0x4e80a52e  // smmla v14.4s, v9.16b, v0.16b\n"
        ".inst 0x4e8da547  // smmla v7.4s, v10.16b, v13.16b\n"
        ".inst 0x4e80a548  // smmla v8.4s, v10.16b, v0.16b\n"
        "shl v10.16b, v2.16b, #0x4\n"
        "shl v9.16b, v11.16b, #0x4\n"
        "and v2.16b, v2.16b, v1.16b\n"
        "and v11.16b, v11.16b, v1.16b\n"
        "ldr q1, [x27, #0x30]\n"
        ".inst 0x4e8aa5e5  // smmla v5.4s, v15.16b, v10.16b\n"
        ".inst 0x4e89a5ee  // smmla v14.4s, v15.16b, v9.16b\n"
        "ldr q15, [x27, #0x40]\n"
        ".inst 0x4e8aa427  // smmla v7.4s, v1.16b, v10.16b\n"
        ".inst 0x4e89a428  // smmla v8.4s, v1.16b, v9.16b\n"
        "ldr q1, [x27, #0x50]\n"
        ".inst 0x4e8ca5e5  // smmla v5.4s, v15.16b, v12.16b\n"
        ".inst 0x4e84a5ee  // smmla v14.4s, v15.16b, v4.16b\n"
        "ldr q15, [x27, #0x60]\n"
        ".inst 0x4e8ca427  // smmla v7.4s, v1.16b, v12.16b\n"
        ".inst 0x4e84a428  // smmla v8.4s, v1.16b, v4.16b\n"
        "ldr q1, [x27, #0x70]\n"
        "add x27, x27, #0x80\n"
        ".inst 0x4e82a5e5  // smmla v5.4s, v15.16b, v2.16b\n"
        ".inst 0x4e8ba5ee  // smmla v14.4s, v15.16b, v11.16b\n"
        "fmul v15.4s, v6.4s, v3.s[0]\n"
        ".inst 0x4e82a427  // smmla v7.4s, v1.16b, v2.16b\n"
        ".inst 0x4e8ba428  // smmla v8.4s, v1.16b, v11.16b\n"
        "uzp1 v1.2d, v5.2d, v14.2d\n"
        "uzp2 v5.2d, v5.2d, v14.2d\n"
        "fmul v14.4s, v6.4s, v3.s[1]\n"
        "scvtf v1.4s, v1.4s, #0x4\n"
        "scvtf v5.4s, v5.4s, #0x4\n"
        "fmla v31.4s, v1.4s, v15.4s\n"
        "fmul v15.4s, v6.4s, v3.s[2]\n"
        "fmul v3.4s, v6.4s, v3.s[3]\n"
        "uzp1 v1.2d, v7.2d, v8.2d\n"
        "uzp2 v8.2d, v7.2d, v8.2d\n"
        "fmla v30.4s, v5.4s, v14.4s\n"
        "scvtf v1.4s, v1.4s, #0x4\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "fmla v29.4s, v1.4s, v15.4s\n"
        "fmla v28.4s, v8.4s, v3.4s\n"
        "ldr d5, [x22, #0x0]\n"
        "add x22, x22, #0x8\n"
        "movi v3.4s, #0x0\n"
        "movi v1.4s, #0x0\n"
        "ldr q15, [x22, #0x0]\n"
        "ldr q7, [x22, #0x10]\n"
        "movi v14.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "fcvtl v5.4s, v5.4h\n"
        ".inst 0x4e8da5e3  // smmla v3.4s, v15.16b, v13.16b\n"
        ".inst 0x4e80a5e1  // smmla v1.4s, v15.16b, v0.16b\n"
        "ldr q15, [x22, #0x20]\n"
        ".inst 0x4e8da4ee  // smmla v14.4s, v7.16b, v13.16b\n"
        ".inst 0x4e80a4e8  // smmla v8.4s, v7.16b, v0.16b\n"
        "ldr q7, [x22, #0x30]\n"
        ".inst 0x4e8aa5e3  // smmla v3.4s, v15.16b, v10.16b\n"
        ".inst 0x4e89a5e1  // smmla v1.4s, v15.16b, v9.16b\n"
        "ldr q15, [x22, #0x40]\n"
        ".inst 0x4e8aa4ee  // smmla v14.4s, v7.16b, v10.16b\n"
        ".inst 0x4e89a4e8  // smmla v8.4s, v7.16b, v9.16b\n"
        "ldr q7, [x22, #0x50]\n"
        ".inst 0x4e8ca5e3  // smmla v3.4s, v15.16b, v12.16b\n"
        ".inst 0x4e84a5e1  // smmla v1.4s, v15.16b, v4.16b\n"
        "ldr q15, [x22, #0x60]\n"
        ".inst 0x4e8ca4ee  // smmla v14.4s, v7.16b, v12.16b\n"
        ".inst 0x4e84a4e8  // smmla v8.4s, v7.16b, v4.16b\n"
        "ldr q7, [x22, #0x70]\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4e82a5e3  // smmla v3.4s, v15.16b, v2.16b\n"
        ".inst 0x4e8ba5e1  // smmla v1.4s, v15.16b, v11.16b\n"
        "fmul v15.4s, v6.4s, v5.s[0]\n"
        ".inst 0x4e82a4ee  // smmla v14.4s, v7.16b, v2.16b\n"
        ".inst 0x4e8ba4e8  // smmla v8.4s, v7.16b, v11.16b\n"
        "uzp1 v7.2d, v3.2d, v1.2d\n"
        "uzp2 v1.2d, v3.2d, v1.2d\n"
        "fmul v3.4s, v6.4s, v5.s[1]\n"
        "scvtf v7.4s, v7.4s, #0x4\n"
        "scvtf v1.4s, v1.4s, #0x4\n"
        "fmla v27.4s, v7.4s, v15.4s\n"
        "fmul v15.4s, v6.4s, v5.s[2]\n"
        "fmul v7.4s, v6.4s, v5.s[3]\n"
        "uzp1 v5.2d, v14.2d, v8.2d\n"
        "uzp2 v14.2d, v14.2d, v8.2d\n"
        "fmla v26.4s, v1.4s, v3.4s\n"
        "scvtf v5.4s, v5.4s, #0x4\n"
        "scvtf v14.4s, v14.4s, #0x4\n"
        "fmla v25.4s, v5.4s, v15.4s\n"
        "fmla v24.4s, v14.4s, v7.4s\n"
        "ldr d1, [x21, #0x0]\n"
        "add x21, x21, #0x8\n"
        "movi v8.4s, #0x0\n"
        "movi v5.4s, #0x0\n"
        "ldr q3, [x21, #0x0]\n"
        "ldr q7, [x21, #0x10]\n"
        "movi v14.4s, #0x0\n"
        "movi v15.4s, #0x0\n"
        "fcvtl v1.4s, v1.4h\n"
        ".inst 0x4e8da468  // smmla v8.4s, v3.16b, v13.16b\n"
        ".inst 0x4e80a465  // smmla v5.4s, v3.16b, v0.16b\n"
        "ldr q3, [x21, #0x20]\n"
        ".inst 0x4e8da4ee  // smmla v14.4s, v7.16b, v13.16b\n"
        ".inst 0x4e80a4ef  // smmla v15.4s, v7.16b, v0.16b\n"
        "ldr q7, [x21, #0x30]\n"
        ".inst 0x4e8aa468  // smmla v8.4s, v3.16b, v10.16b\n"
        ".inst 0x4e89a465  // smmla v5.4s, v3.16b, v9.16b\n"
        "ldr q3, [x21, #0x40]\n"
        ".inst 0x4e8aa4ee  // smmla v14.4s, v7.16b, v10.16b\n"
        ".inst 0x4e89a4ef  // smmla v15.4s, v7.16b, v9.16b\n"
        "ldr q7, [x21, #0x50]\n"
        ".inst 0x4e8ca468  // smmla v8.4s, v3.16b, v12.16b\n"
        ".inst 0x4e84a465  // smmla v5.4s, v3.16b, v4.16b\n"
        "ldr q3, [x21, #0x60]\n"
        ".inst 0x4e8ca4ee  // smmla v14.4s, v7.16b, v12.16b\n"
        ".inst 0x4e84a4ef  // smmla v15.4s, v7.16b, v4.16b\n"
        "ldr q7, [x21, #0x70]\n"
        "add x21, x21, #0x80\n"
        ".inst 0x4e82a468  // smmla v8.4s, v3.16b, v2.16b\n"
        ".inst 0x4e8ba465  // smmla v5.4s, v3.16b, v11.16b\n"
        "fmul v3.4s, v6.4s, v1.s[0]\n"
        ".inst 0x4e82a4ee  // smmla v14.4s, v7.16b, v2.16b\n"
        ".inst 0x4e8ba4ef  // smmla v15.4s, v7.16b, v11.16b\n"
        "uzp1 v7.2d, v8.2d, v5.2d\n"
        "uzp2 v8.2d, v8.2d, v5.2d\n"
        "fmul v5.4s, v6.4s, v1.s[1]\n"
        "scvtf v7.4s, v7.4s, #0x4\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "fmla v23.4s, v7.4s, v3.4s\n"
        "fmul v3.4s, v6.4s, v1.s[2]\n"
        "fmul v1.4s, v6.4s, v1.s[3]\n"
        "uzp1 v7.2d, v14.2d, v15.2d\n"
        "uzp2 v14.2d, v14.2d, v15.2d\n"
        "fmla v22.4s, v8.4s, v5.4s\n"
        "scvtf v7.4s, v7.4s, #0x4\n"
        "scvtf v14.4s, v14.4s, #0x4\n"
        "fmla v21.4s, v7.4s, v3.4s\n"
        "fmla v20.4s, v14.4s, v1.4s\n"
        "ldr d3, [x20, #0x0]\n"
        "add x20, x20, #0x8\n"
        "movi v15.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "ldr q5, [x20, #0x0]\n"
        "ldr q14, [x20, #0x10]\n"
        "movi v1.4s, #0x0\n"
        "movi v7.4s, #0x0\n"
        "fcvtl v3.4s, v3.4h\n"
        ".inst 0x4e8da4af  // smmla v15.4s, v5.16b, v13.16b\n"
        ".inst 0x4e80a4a8  // smmla v8.4s, v5.16b, v0.16b\n"
        "ldr q5, [x20, #0x20]\n"
        ".inst 0x4e8da5c1  // smmla v1.4s, v14.16b, v13.16b\n"
        "ldr q13, [x20, #0x30]\n"
        ".inst 0x4e80a5c7  // smmla v7.4s, v14.16b, v0.16b\n"
        "ldr q14, [x20, #0x40]\n"
        "ldr q0, [x20, #0x50]\n"
        ".inst 0x4e8aa4af  // smmla v15.4s, v5.16b, v10.16b\n"
        ".inst 0x4e89a4a8  // smmla v8.4s, v5.16b, v9.16b\n"
        "ldr q5, [x20, #0x60]\n"
        ".inst 0x4e8aa5a1  // smmla v1.4s, v13.16b, v10.16b\n"
        "ldr q10, [x20, #0x70]\n"
        "add x20, x20, #0x80\n"
        ".inst 0x4e89a5a7  // smmla v7.4s, v13.16b, v9.16b\n"
        "fmul v13.4s, v6.4s, v3.s[0]\n"
        "fmul v9.4s, v6.4s, v3.s[1]\n"
        ".inst 0x4e8ca5cf  // smmla v15.4s, v14.16b, v12.16b\n"
        ".inst 0x4e84a5c8  // smmla v8.4s, v14.16b, v4.16b\n"
        "fmul v14.4s, v6.4s, v3.s[2]\n"
        "fmul v6.4s, v6.4s, v3.s[3]\n"
        ".inst 0x4e8ca401  // smmla v1.4s, v0.16b, v12.16b\n"
        ".inst 0x4e84a407  // smmla v7.4s, v0.16b, v4.16b\n"
        ".inst 0x4e82a4af  // smmla v15.4s, v5.16b, v2.16b\n"
        ".inst 0x4e8ba4a8  // smmla v8.4s, v5.16b, v11.16b\n"
        ".inst 0x4e82a541  // smmla v1.4s, v10.16b, v2.16b\n"
        ".inst 0x4e8ba547  // smmla v7.4s, v10.16b, v11.16b\n"
        "uzp1 v4.2d, v15.2d, v8.2d\n"
        "uzp2 v2.2d, v15.2d, v8.2d\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "uzp1 v8.2d, v1.2d, v7.2d\n"
        "uzp2 v0.2d, v1.2d, v7.2d\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v19.4s, v4.4s, v13.4s\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "fmla v18.4s, v2.4s, v9.4s\n"
        "fmla v17.4s, v8.4s, v14.4s\n"
        "fmla v16.4s, v0.4s, v6.4s\n"
        "subs x23, x23, #0x1\n"
        "bgt 3b\n"
        "ld1r { v1.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x10, #0x4\n"
        "ld1r { v0.4s }, [x20]\n"
        "fmax v31.4s, v31.4s, v1.4s\n"
        "fmax v30.4s, v30.4s, v1.4s\n"
        "fmax v29.4s, v29.4s, v1.4s\n"
        "fmax v28.4s, v28.4s, v1.4s\n"
        "fmax v27.4s, v27.4s, v1.4s\n"
        "fmax v26.4s, v26.4s, v1.4s\n"
        "fmax v25.4s, v25.4s, v1.4s\n"
        "fmax v24.4s, v24.4s, v1.4s\n"
        "fmax v23.4s, v23.4s, v1.4s\n"
        "fmax v22.4s, v22.4s, v1.4s\n"
        "fmax v21.4s, v21.4s, v1.4s\n"
        "fmax v20.4s, v20.4s, v1.4s\n"
        "fmax v19.4s, v19.4s, v1.4s\n"
        "fmax v18.4s, v18.4s, v1.4s\n"
        "fmax v17.4s, v17.4s, v1.4s\n"
        "fmax v16.4s, v16.4s, v1.4s\n"
        "fmin v31.4s, v31.4s, v0.4s\n"
        "fmin v30.4s, v30.4s, v0.4s\n"
        "fmin v29.4s, v29.4s, v0.4s\n"
        "fmin v28.4s, v28.4s, v0.4s\n"
        "fmin v27.4s, v27.4s, v0.4s\n"
        "fmin v26.4s, v26.4s, v0.4s\n"
        "fmin v25.4s, v25.4s, v0.4s\n"
        "fmin v24.4s, v24.4s, v0.4s\n"
        "fmin v23.4s, v23.4s, v0.4s\n"
        "fmin v22.4s, v22.4s, v0.4s\n"
        "fmin v21.4s, v21.4s, v0.4s\n"
        "fmin v20.4s, v20.4s, v0.4s\n"
        "fmin v19.4s, v19.4s, v0.4s\n"
        "fmin v18.4s, v18.4s, v0.4s\n"
        "fmin v17.4s, v17.4s, v0.4s\n"
        "fmin v16.4s, v16.4s, v0.4s\n"
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
        "ldr d10, [x27, #0x0]\n"
        "add x26, x26, #0x8\n"
        "add x27, x27, #0x8\n"
        "ldr q9, [x26, #0x0]\n"
        "ldr q8, [x26, #0x10]\n"
        "movi v7.4s, #0x0\n"
        "movi v6.4s, #0x0\n"
        "ldr q5, [x27, #0x0]\n"
        "ldr q4, [x27, #0x10]\n"
        "movi v3.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        "ldr q1, [x26, #0x20]\n"
        "ldr q0, [x26, #0x30]\n"
        "movi v27.16b, #0xf0\n"
        "fcvtl v26.4s, v16.4h\n"
        "ldr q23, [x27, #0x20]\n"
        "ldr q22, [x27, #0x30]\n"
        "shl v21.16b, v9.16b, #0x4\n"
        "shl v20.16b, v8.16b, #0x4\n"
        "ldr q25, [x27, #0x40]\n"
        "ldr q24, [x27, #0x50]\n"
        "and v9.16b, v9.16b, v27.16b\n"
        "and v8.16b, v8.16b, v27.16b\n"
        "ldr q19, [x27, #0x60]\n"
        "ldr q18, [x27, #0x70]\n"
        "shl v17.16b, v1.16b, #0x4\n"
        "shl v16.16b, v0.16b, #0x4\n"
        ".inst 0x4e95a4a7  // smmla v7.4s, v5.16b, v21.16b\n"
        ".inst 0x4e94a4a6  // smmla v6.4s, v5.16b, v20.16b\n"
        "and v1.16b, v1.16b, v27.16b\n"
        "add x26, x26, #0x40\n"
        ".inst 0x4e95a483  // smmla v3.4s, v4.16b, v21.16b\n"
        ".inst 0x4e94a482  // smmla v2.4s, v4.16b, v20.16b\n"
        "and v0.16b, v0.16b, v27.16b\n"
        "add x27, x27, #0x80\n"
        "fcvtl v10.4s, v10.4h\n"
        ".inst 0x4e91a6e7  // smmla v7.4s, v23.16b, v17.16b\n"
        ".inst 0x4e90a6e6  // smmla v6.4s, v23.16b, v16.16b\n"
        ".inst 0x4e91a6c3  // smmla v3.4s, v22.16b, v17.16b\n"
        ".inst 0x4e90a6c2  // smmla v2.4s, v22.16b, v16.16b\n"
        "fmul v23.4s, v26.4s, v10.s[0]\n"
        "fmul v22.4s, v26.4s, v10.s[1]\n"
        "fmul v21.4s, v26.4s, v10.s[2]\n"
        "fmul v20.4s, v26.4s, v10.s[3]\n"
        ".inst 0x4e89a727  // smmla v7.4s, v25.16b, v9.16b\n"
        ".inst 0x4e88a726  // smmla v6.4s, v25.16b, v8.16b\n"
        ".inst 0x4e89a703  // smmla v3.4s, v24.16b, v9.16b\n"
        ".inst 0x4e88a702  // smmla v2.4s, v24.16b, v8.16b\n"
        ".inst 0x4e81a667  // smmla v7.4s, v19.16b, v1.16b\n"
        ".inst 0x4e80a666  // smmla v6.4s, v19.16b, v0.16b\n"
        ".inst 0x4e81a643  // smmla v3.4s, v18.16b, v1.16b\n"
        ".inst 0x4e80a642  // smmla v2.4s, v18.16b, v0.16b\n"
        "uzp1 v19.2d, v7.2d, v6.2d\n"
        "uzp2 v18.2d, v7.2d, v6.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp1 v17.2d, v3.2d, v2.2d\n"
        "uzp2 v16.2d, v3.2d, v2.2d\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v31.4s, v19.4s, v23.4s\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v30.4s, v18.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v20.4s\n"
        "subs x20, x20, #0x1\n"
        "bgt 17b\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x25, #0x4\n"
        "ld1r { v16.4s }, [x20]\n"
        "fmax v31.4s, v31.4s, v17.4s\n"
        "fmax v30.4s, v30.4s, v17.4s\n"
        "fmax v29.4s, v29.4s, v17.4s\n"
        "fmax v28.4s, v28.4s, v17.4s\n"
        "fmin v31.4s, v31.4s, v16.4s\n"
        "fmin v30.4s, v30.4s, v16.4s\n"
        "fmin v29.4s, v29.4s, v16.4s\n"
        "fmin v28.4s, v28.4s, v16.4s\n"
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
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v2", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v3", "v30", "v31", "v4", "v5", "v6", "v7",
          "v8", "v9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9");
}

#endif  // Architectural feature check
