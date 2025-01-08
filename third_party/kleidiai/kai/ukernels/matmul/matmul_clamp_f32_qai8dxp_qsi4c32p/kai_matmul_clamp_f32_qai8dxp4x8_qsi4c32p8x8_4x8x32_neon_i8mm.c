//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "I8mm extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 4;
static const size_t kai_n_step = 8;
static const size_t kai_mr = 4;
static const size_t kai_nr = 8;
static const size_t kai_kr = 16;
static const size_t kai_sr = 2;
static const size_t kai_bl_multiple_of = 32;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(uint16_t);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_k_roundedup(size_t k) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for alignment
    size_t kr_sr_roundedup4 = kai_roundup(kai_kr * kai_sr, 4);
    return kai_roundup(k, kr_sr_roundedup4);
}

inline static size_t kai_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return kai_mr * (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSERT((bl % kai_kr) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = (bl / 2) + kai_num_bytes_multiplier_rhs;

    return kai_nr * ((num_bytes_per_block * num_blocks_per_row) + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(
    size_t m, size_t n, size_t k, size_t bl, const void* lhs_packed, const void* rhs_packed,
    float* dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT((bl % kai_kr) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    size_t num_subblocks = bl / kai_bl_multiple_of;
    size_t num_blocks = kai_num_blocks_per_row(k, bl);

    float clamp_vals[2] = {scalar_min, scalar_max};

    __asm__ __volatile__(
        "mov x28, #0x80\n"
        "mov x21, #0x3d800000\n"
        "movi v17.16b, #0xf0\n"
        "mov x20, #0x20\n"
        "mov x27, %x[m]\n"
        "mul x28, %x[num_subblocks], x28\n"
        "dup v14.4s, w21\n"
        "madd x28, %x[num_blocks], x28, x20\n"
        "cbz x27, 12f\n"
        "1:"  // Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "2:"  // Column loop
        "movi v1.16b, #0x0\n"
        "movi v12.16b, #0x0\n"
        "mov x22, %x[lhs_packed]\n"
        "mov x21, %x[num_blocks]\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "movi v18.16b, #0x0\n"
        "movi v27.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "3:"  // Block loop
        "movi v21.4s, #0x0\n"
        "movi v30.4s, #0x0\n"
        "mov x20, %x[num_subblocks]\n"
        "movi v24.4s, #0x0\n"
        "movi v23.4s, #0x0\n"
        "movi v7.4s, #0x0\n"
        "movi v3.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "4:"  // Sub block loop
        "ldr q6, [x26, #0x0]\n"
        "ldr q0, [x26, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q10, [x26, #0x20]\n"
        "ldr q26, [x26, #0x30]\n"
        "ldr q22, [x22, #0x0]\n"
        "ldr q20, [x22, #0x10]\n"
        "ldr q31, [x26, #0x40]\n"
        "ldr q15, [x26, #0x50]\n"
        "shl v29.16b, v6.16b, #0x4\n"
        "shl v9.16b, v0.16b, #0x4\n"
        "ldr q25, [x26, #0x60]\n"
        "ldr q16, [x26, #0x70]\n"
        "shl v5.16b, v10.16b, #0x4\n"
        "shl v19.16b, v26.16b, #0x4\n"
        "and v6.16b, v6.16b, v17.16b\n"
        "and v0.16b, v0.16b, v17.16b\n"
        "add x26, x26, #0x80\n"
        ".inst 0x4e9da6d5  // smmla v21.4s, v22.16b, v29.16b\n"
        ".inst 0x4e89a6d8  // smmla v24.4s, v22.16b, v9.16b\n"
        ".inst 0x4e9da687  // smmla v7.4s, v20.16b, v29.16b\n"
        "ldr q29, [x22, #0x20]\n"
        "and v10.16b, v10.16b, v17.16b\n"
        ".inst 0x4e85a6de  // smmla v30.4s, v22.16b, v5.16b\n"
        ".inst 0x4e93a6d7  // smmla v23.4s, v22.16b, v19.16b\n"
        "ldr q22, [x22, #0x30]\n"
        "and v26.16b, v26.16b, v17.16b\n"
        ".inst 0x4e89a682  // smmla v2.4s, v20.16b, v9.16b\n"
        "ldr q9, [x22, #0x40]\n"
        ".inst 0x4e85a683  // smmla v3.4s, v20.16b, v5.16b\n"
        "ldr q5, [x22, #0x50]\n"
        ".inst 0x4e93a688  // smmla v8.4s, v20.16b, v19.16b\n"
        "ldr q19, [x22, #0x60]\n"
        "shl v20.16b, v31.16b, #0x4\n"
        "and v31.16b, v31.16b, v17.16b\n"
        ".inst 0x4e94a7b5  // smmla v21.4s, v29.16b, v20.16b\n"
        ".inst 0x4e94a6c7  // smmla v7.4s, v22.16b, v20.16b\n"
        "ldr q20, [x22, #0x70]\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4e86a535  // smmla v21.4s, v9.16b, v6.16b\n"
        ".inst 0x4e86a4a7  // smmla v7.4s, v5.16b, v6.16b\n"
        "shl v6.16b, v15.16b, #0x4\n"
        "and v15.16b, v15.16b, v17.16b\n"
        ".inst 0x4e86a7b8  // smmla v24.4s, v29.16b, v6.16b\n"
        ".inst 0x4e86a6c2  // smmla v2.4s, v22.16b, v6.16b\n"
        "shl v6.16b, v25.16b, #0x4\n"
        "and v25.16b, v25.16b, v17.16b\n"
        ".inst 0x4e9fa675  // smmla v21.4s, v19.16b, v31.16b\n"
        ".inst 0x4e9fa687  // smmla v7.4s, v20.16b, v31.16b\n"
        "shl v31.16b, v16.16b, #0x4\n"
        "and v16.16b, v16.16b, v17.16b\n"
        ".inst 0x4e86a7be  // smmla v30.4s, v29.16b, v6.16b\n"
        ".inst 0x4e86a6c3  // smmla v3.4s, v22.16b, v6.16b\n"
        ".inst 0x4e80a538  // smmla v24.4s, v9.16b, v0.16b\n"
        ".inst 0x4e80a4a2  // smmla v2.4s, v5.16b, v0.16b\n"
        ".inst 0x4e9fa7b7  // smmla v23.4s, v29.16b, v31.16b\n"
        ".inst 0x4e9fa6c8  // smmla v8.4s, v22.16b, v31.16b\n"
        ".inst 0x4e8aa53e  // smmla v30.4s, v9.16b, v10.16b\n"
        ".inst 0x4e8aa4a3  // smmla v3.4s, v5.16b, v10.16b\n"
        ".inst 0x4e8fa678  // smmla v24.4s, v19.16b, v15.16b\n"
        ".inst 0x4e8fa682  // smmla v2.4s, v20.16b, v15.16b\n"
        ".inst 0x4e9aa537  // smmla v23.4s, v9.16b, v26.16b\n"
        ".inst 0x4e9aa4a8  // smmla v8.4s, v5.16b, v26.16b\n"
        ".inst 0x4e99a67e  // smmla v30.4s, v19.16b, v25.16b\n"
        ".inst 0x4e99a683  // smmla v3.4s, v20.16b, v25.16b\n"
        ".inst 0x4e90a677  // smmla v23.4s, v19.16b, v16.16b\n"
        ".inst 0x4e90a688  // smmla v8.4s, v20.16b, v16.16b\n"
        "bgt 4b\n"
        "ldr q29, [x26, #0x0]\n"
        "uzp1 v26.2d, v21.2d, v24.2d\n"
        "uzp2 v25.2d, v21.2d, v24.2d\n"
        "add x26, x26, #0x10\n"
        "uzp1 v24.2d, v30.2d, v23.2d\n"
        "uzp2 v23.2d, v30.2d, v23.2d\n"
        "uzp1 v22.2d, v7.2d, v2.2d\n"
        "uzp2 v21.2d, v7.2d, v2.2d\n"
        "shll v20.4s, v29.4h, #0x10\n"
        "shll2 v19.4s, v29.8h, #0x10\n"
        "uzp1 v0.2d, v3.2d, v8.2d\n"
        "uzp2 v8.2d, v3.2d, v8.2d\n"
        "scvtf v26.4s, v26.4s\n"
        "scvtf v24.4s, v24.4s\n"
        "fmul v20.4s, v20.4s, v14.4s\n"
        "fmul v19.4s, v19.4s, v14.4s\n"
        "scvtf v25.4s, v25.4s\n"
        "scvtf v23.4s, v23.4s\n"
        "scvtf v22.4s, v22.4s\n"
        "scvtf v0.4s, v0.4s\n"
        "scvtf v21.4s, v21.4s\n"
        "scvtf v8.4s, v8.4s\n"
        "fmla v1.4s, v26.4s, v20.4s\n"
        "fmla v12.4s, v24.4s, v19.4s\n"
        "fmla v11.4s, v25.4s, v20.4s\n"
        "fmla v13.4s, v23.4s, v19.4s\n"
        "fmla v18.4s, v22.4s, v20.4s\n"
        "fmla v27.4s, v0.4s, v19.4s\n"
        "fmla v28.4s, v21.4s, v20.4s\n"
        "fmla v4.4s, v8.4s, v19.4s\n"
        "subs x21, x21, #0x1\n"
        "bgt 3b\n"
        "ld1 { v23.4s }, [x22]\n"
        "ldr q22, [x26, #0x0]\n"
        "add x22, x22, #0x10\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ldr q9, [x26, #0x10]\n"
        "ldr q20, [x22, #0x0]\n"
        "cmp x25, #0x8\n"
        "ldr q19, [x26, #0x20]\n"
        "ldr q21, [x26, #0x30]\n"
        "add x26, x26, #0x40\n"
        "ld1r { v10.4s }, [%x[clamp_vals]]\n"
        "ld1r { v30.4s }, [x20]\n"
        "scvtf v23.4s, v23.4s\n"
        "fmla v1.4s, v22.4s, v23.s[0]\n"
        "fmla v12.4s, v9.4s, v23.s[0]\n"
        "fmla v11.4s, v22.4s, v23.s[1]\n"
        "fmla v13.4s, v9.4s, v23.s[1]\n"
        "fmla v18.4s, v22.4s, v23.s[2]\n"
        "fmla v27.4s, v9.4s, v23.s[2]\n"
        "fmla v28.4s, v22.4s, v23.s[3]\n"
        "fmla v4.4s, v9.4s, v23.s[3]\n"
        "fmul v1.4s, v1.4s, v20.s[0]\n"
        "fmul v12.4s, v12.4s, v20.s[0]\n"
        "fmul v11.4s, v11.4s, v20.s[1]\n"
        "fmul v13.4s, v13.4s, v20.s[1]\n"
        "fmul v18.4s, v18.4s, v20.s[2]\n"
        "fmul v27.4s, v27.4s, v20.s[2]\n"
        "fmul v28.4s, v28.4s, v20.s[3]\n"
        "fmul v4.4s, v4.4s, v20.s[3]\n"
        "fadd v1.4s, v1.4s, v19.4s\n"
        "fadd v12.4s, v12.4s, v21.4s\n"
        "fadd v11.4s, v11.4s, v19.4s\n"
        "fadd v13.4s, v13.4s, v21.4s\n"
        "fadd v18.4s, v18.4s, v19.4s\n"
        "fadd v27.4s, v27.4s, v21.4s\n"
        "fadd v28.4s, v28.4s, v19.4s\n"
        "fadd v4.4s, v4.4s, v21.4s\n"
        "fmax v1.4s, v1.4s, v10.4s\n"
        "fmax v12.4s, v12.4s, v10.4s\n"
        "fmax v11.4s, v11.4s, v10.4s\n"
        "fmax v13.4s, v13.4s, v10.4s\n"
        "fmax v18.4s, v18.4s, v10.4s\n"
        "fmax v27.4s, v27.4s, v10.4s\n"
        "fmax v28.4s, v28.4s, v10.4s\n"
        "fmax v4.4s, v4.4s, v10.4s\n"
        "fmin v1.4s, v1.4s, v30.4s\n"
        "fmin v12.4s, v12.4s, v30.4s\n"
        "fmin v11.4s, v11.4s, v30.4s\n"
        "fmin v13.4s, v13.4s, v30.4s\n"
        "fmin v18.4s, v18.4s, v30.4s\n"
        "fmin v27.4s, v27.4s, v30.4s\n"
        "fmin v28.4s, v28.4s, v30.4s\n"
        "fmin v4.4s, v4.4s, v30.4s\n"
        "blt 6f\n"
        "mov x20, %x[dst]\n"
        "cmp x27, #0x1\n"
        "str q1, [x20, #0x0]\n"
        "str q12, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 11f\n"
        "cmp x27, #0x2\n"
        "str q11, [x20, #0x0]\n"
        "str q13, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 11f\n"
        "cmp x27, #0x3\n"
        "str q18, [x20, #0x0]\n"
        "str q27, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 11f\n"
        "str q28, [x20, #0x0]\n"
        "str q4, [x20, #0x10]\n"
        "b 11f\n"
        "6:"  // Partial output
        "mov x23, %x[dst]\n"
        "cmp x27, #0x1\n"
        "add x22, x23, %x[dst_stride_row]\n"
        "csel x22, x22, x23, GT\n"
        "cmp x27, #0x2\n"
        "add x21, x23, %x[dst_stride_row], LSL #1\n"
        "csel x21, x21, x22, GT\n"
        "cmp x27, #0x3\n"
        "add x20, x21, %x[dst_stride_row]\n"
        "csel x20, x20, x21, GT\n"
        "tbz x25, #2, 8f\n"
        "st1 { v28.4s }, [x20], #0x10\n"
        "st1 { v18.4s }, [x21], #0x10\n"
        "st1 { v11.4s }, [x22], #0x10\n"
        "st1 { v1.4s }, [x23], #0x10\n"
        "tbz x25, #1, 7f\n"
        "st1 { v4.d }[0], [x20], #0x8\n"
        "st1 { v27.d }[0], [x21], #0x8\n"
        "st1 { v13.d }[0], [x22], #0x8\n"
        "st1 { v12.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 10f\n"
        "st1 { v4.s }[2], [x20]\n"
        "st1 { v27.s }[2], [x21]\n"
        "st1 { v13.s }[2], [x22]\n"
        "st1 { v12.s }[2], [x23]\n"
        "b 10f\n"
        "7:"  // Output block 0: partial_1_4
        "tbz x25, #0, 10f\n"
        "st1 { v4.s }[0], [x20]\n"
        "st1 { v27.s }[0], [x21]\n"
        "st1 { v13.s }[0], [x22]\n"
        "st1 { v12.s }[0], [x23]\n"
        "b 10f\n"
        "8:"  // Output block 0: partial_2_0
        "tbz x25, #1, 9f\n"
        "st1 { v28.d }[0], [x20], #0x8\n"
        "st1 { v18.d }[0], [x21], #0x8\n"
        "st1 { v11.d }[0], [x22], #0x8\n"
        "st1 { v1.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 10f\n"
        "st1 { v28.s }[2], [x20]\n"
        "st1 { v18.s }[2], [x21]\n"
        "st1 { v11.s }[2], [x22]\n"
        "st1 { v1.s }[2], [x23]\n"
        "b 10f\n"
        "9:"  // Output block 0: partial_1_0
        "st1 { v28.s }[0], [x20]\n"
        "st1 { v18.s }[0], [x21]\n"
        "st1 { v11.s }[0], [x22]\n"
        "st1 { v1.s }[0], [x23]\n"
        "10:"  // Output block 0: Done
        "11:"  // Output stage exit
        "subs x25, x25, #0x8\n"
        "add %x[dst], %x[dst], #0x20\n"
        "bgt 2b\n"
        "subs x27, x27, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x28\n"
        "mov %x[dst], x24\n"
        "bgt 1b\n"
        "12:"  // Row loop skip
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [num_subblocks] "r"(num_subblocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}
#endif  // Architectural feature check
