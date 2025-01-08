//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "I8mm extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 8;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
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

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
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
        "mov x12, #0x80\n"
        "mov x11, %x[m]\n"
        "movi v15.16b, #0xf0\n"
        "mov x21, #0x3d800000\n"
        "mov x20, #0x20\n"
        "mul x12, %x[num_subblocks], x12\n"
        "cmp x11, #0x8\n"
        "dup v24.4s, w21\n"
        "madd x12, %x[num_blocks], x12, x20\n"
        "blt 11f\n"
        "1:"  // Row loop
        "mov x10, %x[rhs_packed]\n"
        "mov x9, %x[n]\n"
        "add x28, %x[dst], %x[dst_stride_row], LSL #3\n"
        "2:"  // Column loop
        "mov x23, %x[lhs_packed]\n"
        "movi v12.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "mov x22, %x[num_blocks]\n"
        "movi v22.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v0.16b, #0x0\n"
        "movi v30.16b, #0x0\n"
        "movi v8.16b, #0x0\n"
        "add x21, x23, x12\n"
        "3:"  // Block loop
        "movi v6.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "mov x20, %x[num_subblocks]\n"
        "movi v4.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        "movi v31.4s, #0x0\n"
        "movi v3.4s, #0x0\n"
        "movi v7.4s, #0x0\n"
        "movi v23.4s, #0x0\n"
        "4:"  // Sub block loop
        "ldr q2, [x10, #0x0]\n"
        "ldr q20, [x10, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q25, [x23, #0x0]\n"
        "ldr q11, [x23, #0x10]\n"
        "ldr q9, [x21, #0x0]\n"
        "ldr q19, [x21, #0x10]\n"
        "ldr q1, [x10, #0x20]\n"
        "ldr q29, [x10, #0x30]\n"
        "shl v27.16b, v2.16b, #0x4\n"
        "shl v21.16b, v20.16b, #0x4\n"
        "ldr q17, [x23, #0x20]\n"
        "ldr q26, [x23, #0x30]\n"
        "and v2.16b, v2.16b, v15.16b\n"
        "and v20.16b, v20.16b, v15.16b\n"
        "ldr q28, [x21, #0x20]\n"
        "ldr q16, [x21, #0x30]\n"
        "add x10, x10, #0x40\n"
        ".inst 0x4e9ba726  // smmla v6.4s, v25.16b, v27.16b\n"
        ".inst 0x4e95a72a  // smmla v10.4s, v25.16b, v21.16b\n"
        "ldr q25, [x23, #0x40]\n"
        ".inst 0x4e9ba564  // smmla v4.4s, v11.16b, v27.16b\n"
        ".inst 0x4e95a572  // smmla v18.4s, v11.16b, v21.16b\n"
        "ldr q11, [x23, #0x50]\n"
        ".inst 0x4e9ba53f  // smmla v31.4s, v9.16b, v27.16b\n"
        ".inst 0x4e95a523  // smmla v3.4s, v9.16b, v21.16b\n"
        "ldr q9, [x21, #0x40]\n"
        ".inst 0x4e9ba667  // smmla v7.4s, v19.16b, v27.16b\n"
        "ldr q27, [x21, #0x50]\n"
        ".inst 0x4e95a677  // smmla v23.4s, v19.16b, v21.16b\n"
        "ldr q21, [x23, #0x60]\n"
        "shl v19.16b, v1.16b, #0x4\n"
        "and v1.16b, v1.16b, v15.16b\n"
        ".inst 0x4e93a626  // smmla v6.4s, v17.16b, v19.16b\n"
        ".inst 0x4e93a744  // smmla v4.4s, v26.16b, v19.16b\n"
        ".inst 0x4e93a79f  // smmla v31.4s, v28.16b, v19.16b\n"
        ".inst 0x4e93a607  // smmla v7.4s, v16.16b, v19.16b\n"
        "ldr q19, [x23, #0x70]\n"
        "add x23, x23, #0x80\n"
        ".inst 0x4e82a726  // smmla v6.4s, v25.16b, v2.16b\n"
        ".inst 0x4e82a564  // smmla v4.4s, v11.16b, v2.16b\n"
        ".inst 0x4e82a53f  // smmla v31.4s, v9.16b, v2.16b\n"
        ".inst 0x4e82a767  // smmla v7.4s, v27.16b, v2.16b\n"
        "shl v2.16b, v29.16b, #0x4\n"
        "and v29.16b, v29.16b, v15.16b\n"
        ".inst 0x4e82a62a  // smmla v10.4s, v17.16b, v2.16b\n"
        "ldr q17, [x21, #0x60]\n"
        ".inst 0x4e82a752  // smmla v18.4s, v26.16b, v2.16b\n"
        "ldr q26, [x21, #0x70]\n"
        "add x21, x21, #0x80\n"
        ".inst 0x4e82a783  // smmla v3.4s, v28.16b, v2.16b\n"
        ".inst 0x4e82a617  // smmla v23.4s, v16.16b, v2.16b\n"
        ".inst 0x4e81a6a6  // smmla v6.4s, v21.16b, v1.16b\n"
        ".inst 0x4e81a664  // smmla v4.4s, v19.16b, v1.16b\n"
        ".inst 0x4e81a63f  // smmla v31.4s, v17.16b, v1.16b\n"
        ".inst 0x4e94a72a  // smmla v10.4s, v25.16b, v20.16b\n"
        ".inst 0x4e94a572  // smmla v18.4s, v11.16b, v20.16b\n"
        ".inst 0x4e81a747  // smmla v7.4s, v26.16b, v1.16b\n"
        ".inst 0x4e94a523  // smmla v3.4s, v9.16b, v20.16b\n"
        ".inst 0x4e94a777  // smmla v23.4s, v27.16b, v20.16b\n"
        ".inst 0x4e9da6aa  // smmla v10.4s, v21.16b, v29.16b\n"
        ".inst 0x4e9da672  // smmla v18.4s, v19.16b, v29.16b\n"
        ".inst 0x4e9da623  // smmla v3.4s, v17.16b, v29.16b\n"
        ".inst 0x4e9da757  // smmla v23.4s, v26.16b, v29.16b\n"
        "bgt 4b\n"
        "ldr d20, [x10, #0x0]\n"
        "uzp1 v21.2d, v6.2d, v10.2d\n"
        "uzp2 v19.2d, v6.2d, v10.2d\n"
        "add x10, x10, #0x8\n"
        "uzp1 v17.2d, v4.2d, v18.2d\n"
        "uzp2 v16.2d, v4.2d, v18.2d\n"
        "shll v20.4s, v20.4h, #0x10\n"
        "scvtf v21.4s, v21.4s\n"
        "scvtf v19.4s, v19.4s\n"
        "scvtf v17.4s, v17.4s\n"
        "scvtf v16.4s, v16.4s\n"
        "fmul v20.4s, v20.4s, v24.4s\n"
        "fmla v12.4s, v21.4s, v20.4s\n"
        "fmla v13.4s, v19.4s, v20.4s\n"
        "fmla v22.4s, v17.4s, v20.4s\n"
        "fmla v14.4s, v16.4s, v20.4s\n"
        "uzp1 v19.2d, v31.2d, v3.2d\n"
        "uzp2 v18.2d, v31.2d, v3.2d\n"
        "uzp1 v17.2d, v7.2d, v23.2d\n"
        "uzp2 v16.2d, v7.2d, v23.2d\n"
        "scvtf v19.4s, v19.4s\n"
        "scvtf v18.4s, v18.4s\n"
        "scvtf v17.4s, v17.4s\n"
        "scvtf v16.4s, v16.4s\n"
        "fmla v5.4s, v19.4s, v20.4s\n"
        "fmla v0.4s, v18.4s, v20.4s\n"
        "fmla v30.4s, v17.4s, v20.4s\n"
        "fmla v8.4s, v16.4s, v20.4s\n"
        "subs x22, x22, #0x1\n"
        "bgt 3b\n"
        "ld1 { v23.4s }, [x23]\n"
        "ld1 { v1.4s }, [x21]\n"
        "add x23, x23, #0x10\n"
        "add x21, x21, #0x10\n"
        "ldr q21, [x10, #0x0]\n"
        "ldr q20, [x23, #0x0]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x9, #0x4\n"
        "ldr q19, [x21, #0x0]\n"
        "ldr q18, [x10, #0x10]\n"
        "add x10, x10, #0x20\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "ld1r { v16.4s }, [x20]\n"
        "scvtf v23.4s, v23.4s\n"
        "scvtf v1.4s, v1.4s\n"
        "fmla v12.4s, v21.4s, v23.s[0]\n"
        "fmla v13.4s, v21.4s, v23.s[1]\n"
        "fmla v22.4s, v21.4s, v23.s[2]\n"
        "fmla v14.4s, v21.4s, v23.s[3]\n"
        "fmla v5.4s, v21.4s, v1.s[0]\n"
        "fmla v0.4s, v21.4s, v1.s[1]\n"
        "fmla v30.4s, v21.4s, v1.s[2]\n"
        "fmla v8.4s, v21.4s, v1.s[3]\n"
        "fmul v12.4s, v12.4s, v20.s[0]\n"
        "fmul v13.4s, v13.4s, v20.s[1]\n"
        "fmul v22.4s, v22.4s, v20.s[2]\n"
        "fmul v14.4s, v14.4s, v20.s[3]\n"
        "fmul v5.4s, v5.4s, v19.s[0]\n"
        "fmul v0.4s, v0.4s, v19.s[1]\n"
        "fadd v12.4s, v12.4s, v18.4s\n"
        "fmul v30.4s, v30.4s, v19.s[2]\n"
        "fmul v8.4s, v8.4s, v19.s[3]\n"
        "fadd v13.4s, v13.4s, v18.4s\n"
        "fadd v22.4s, v22.4s, v18.4s\n"
        "fadd v14.4s, v14.4s, v18.4s\n"
        "fadd v5.4s, v5.4s, v18.4s\n"
        "fadd v0.4s, v0.4s, v18.4s\n"
        "fadd v30.4s, v30.4s, v18.4s\n"
        "fadd v8.4s, v8.4s, v18.4s\n"
        "fmax v12.4s, v12.4s, v17.4s\n"
        "fmax v13.4s, v13.4s, v17.4s\n"
        "fmax v22.4s, v22.4s, v17.4s\n"
        "fmax v14.4s, v14.4s, v17.4s\n"
        "fmax v5.4s, v5.4s, v17.4s\n"
        "fmax v0.4s, v0.4s, v17.4s\n"
        "fmax v30.4s, v30.4s, v17.4s\n"
        "fmax v8.4s, v8.4s, v17.4s\n"
        "fmin v12.4s, v12.4s, v16.4s\n"
        "fmin v13.4s, v13.4s, v16.4s\n"
        "fmin v22.4s, v22.4s, v16.4s\n"
        "fmin v14.4s, v14.4s, v16.4s\n"
        "fmin v5.4s, v5.4s, v16.4s\n"
        "fmin v0.4s, v0.4s, v16.4s\n"
        "fmin v30.4s, v30.4s, v16.4s\n"
        "fmin v8.4s, v8.4s, v16.4s\n"
        "blt 7f\n"
        "mov x20, %x[dst]\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q14, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q0, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q8, [x20, #0x0]\n"
        "b 10f\n"
        "7:"  // Partial output
        "mov x27, %x[dst]\n"
        "add x26, x27, %x[dst_stride_row], LSL #2\n"
        "add x25, x26, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row]\n"
        "add x23, x25, %x[dst_stride_row]\n"
        "add x22, x27, %x[dst_stride_row], LSL #1\n"
        "add x21, x27, %x[dst_stride_row]\n"
        "add x20, x22, %x[dst_stride_row]\n"
        "tbz x9, #1, 8f\n"
        "st1 { v8.d }[0], [x23], #0x8\n"
        "st1 { v30.d }[0], [x25], #0x8\n"
        "st1 { v0.d }[0], [x24], #0x8\n"
        "st1 { v5.d }[0], [x26], #0x8\n"
        "st1 { v14.d }[0], [x20], #0x8\n"
        "st1 { v22.d }[0], [x22], #0x8\n"
        "st1 { v13.d }[0], [x21], #0x8\n"
        "st1 { v12.d }[0], [x27], #0x8\n"
        "tbz x9, #0, 9f\n"
        "st1 { v8.s }[2], [x23]\n"
        "st1 { v30.s }[2], [x25]\n"
        "st1 { v0.s }[2], [x24]\n"
        "st1 { v5.s }[2], [x26]\n"
        "st1 { v14.s }[2], [x20]\n"
        "st1 { v22.s }[2], [x22]\n"
        "st1 { v13.s }[2], [x21]\n"
        "st1 { v12.s }[2], [x27]\n"
        "b 9f\n"
        "8:"  // Output block 0: partial_1_0
        "st1 { v8.s }[0], [x23]\n"
        "st1 { v30.s }[0], [x25]\n"
        "st1 { v0.s }[0], [x24]\n"
        "st1 { v5.s }[0], [x26]\n"
        "st1 { v14.s }[0], [x20]\n"
        "st1 { v22.s }[0], [x22]\n"
        "st1 { v13.s }[0], [x21]\n"
        "st1 { v12.s }[0], [x27]\n"
        "9:"   // Output block 0: Done
        "10:"  // Output stage exit
        "subs x9, x9, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "mov x20, #0x2\n"
        "sub x11, x11, #0x8\n"
        "cmp x11, #0x8\n"
        "mov %x[dst], x28\n"
        "madd %x[lhs_packed], x20, x12, %x[lhs_packed]\n"
        "bge 1b\n"
        "11:"  // Row loop skip
        "cbz x11, 21f\n"
        "12:"  // Row tail: Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "13:"  // Row tail: Column loop
        "movi v12.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "mov x23, %x[lhs_packed]\n"
        "mov x21, %x[num_blocks]\n"
        "movi v22.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "14:"  // Row tail: Block loop
        "movi v6.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "mov x20, %x[num_subblocks]\n"
        "movi v4.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        "15:"  // Row tail: Sub block loop
        "ldr q0, [x26, #0x0]\n"
        "ldr q31, [x26, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q11, [x23, #0x0]\n"
        "ldr q30, [x23, #0x10]\n"
        "ldr q29, [x26, #0x20]\n"
        "ldr q28, [x26, #0x30]\n"
        "add x26, x26, #0x40\n"
        "ldr q27, [x23, #0x20]\n"
        "ldr q26, [x23, #0x30]\n"
        "shl v25.16b, v0.16b, #0x4\n"
        "shl v23.16b, v31.16b, #0x4\n"
        "ldr q1, [x23, #0x40]\n"
        "ldr q21, [x23, #0x50]\n"
        "and v0.16b, v0.16b, v15.16b\n"
        "and v31.16b, v31.16b, v15.16b\n"
        "ldr q20, [x23, #0x60]\n"
        "ldr q19, [x23, #0x70]\n"
        "shl v17.16b, v29.16b, #0x4\n"
        "shl v16.16b, v28.16b, #0x4\n"
        ".inst 0x4e99a566  // smmla v6.4s, v11.16b, v25.16b\n"
        ".inst 0x4e97a56a  // smmla v10.4s, v11.16b, v23.16b\n"
        "and v29.16b, v29.16b, v15.16b\n"
        "add x23, x23, #0x80\n"
        ".inst 0x4e99a7c4  // smmla v4.4s, v30.16b, v25.16b\n"
        ".inst 0x4e97a7d2  // smmla v18.4s, v30.16b, v23.16b\n"
        "and v28.16b, v28.16b, v15.16b\n"
        ".inst 0x4e91a766  // smmla v6.4s, v27.16b, v17.16b\n"
        ".inst 0x4e90a76a  // smmla v10.4s, v27.16b, v16.16b\n"
        ".inst 0x4e91a744  // smmla v4.4s, v26.16b, v17.16b\n"
        ".inst 0x4e90a752  // smmla v18.4s, v26.16b, v16.16b\n"
        ".inst 0x4e80a426  // smmla v6.4s, v1.16b, v0.16b\n"
        ".inst 0x4e9fa42a  // smmla v10.4s, v1.16b, v31.16b\n"
        ".inst 0x4e80a6a4  // smmla v4.4s, v21.16b, v0.16b\n"
        ".inst 0x4e9fa6b2  // smmla v18.4s, v21.16b, v31.16b\n"
        ".inst 0x4e9da686  // smmla v6.4s, v20.16b, v29.16b\n"
        ".inst 0x4e9ca68a  // smmla v10.4s, v20.16b, v28.16b\n"
        ".inst 0x4e9da664  // smmla v4.4s, v19.16b, v29.16b\n"
        ".inst 0x4e9ca672  // smmla v18.4s, v19.16b, v28.16b\n"
        "bgt 15b\n"
        "ldr d16, [x26, #0x0]\n"
        "uzp1 v21.2d, v6.2d, v10.2d\n"
        "uzp2 v20.2d, v6.2d, v10.2d\n"
        "add x26, x26, #0x8\n"
        "uzp1 v19.2d, v4.2d, v18.2d\n"
        "uzp2 v17.2d, v4.2d, v18.2d\n"
        "shll v16.4s, v16.4h, #0x10\n"
        "scvtf v21.4s, v21.4s\n"
        "scvtf v20.4s, v20.4s\n"
        "scvtf v19.4s, v19.4s\n"
        "scvtf v17.4s, v17.4s\n"
        "fmul v16.4s, v16.4s, v24.4s\n"
        "fmla v12.4s, v21.4s, v16.4s\n"
        "fmla v13.4s, v20.4s, v16.4s\n"
        "fmla v22.4s, v19.4s, v16.4s\n"
        "fmla v14.4s, v17.4s, v16.4s\n"
        "subs x21, x21, #0x1\n"
        "bgt 14b\n"
        "ld1 { v21.4s }, [x23]\n"
        "ldr q20, [x26, #0x0]\n"
        "add x23, x23, #0x10\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ldr q19, [x23, #0x0]\n"
        "ldr q18, [x26, #0x10]\n"
        "cmp x25, #0x4\n"
        "add x26, x26, #0x20\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "ld1r { v16.4s }, [x20]\n"
        "scvtf v21.4s, v21.4s\n"
        "fmla v12.4s, v20.4s, v21.s[0]\n"
        "fmla v13.4s, v20.4s, v21.s[1]\n"
        "fmla v22.4s, v20.4s, v21.s[2]\n"
        "fmla v14.4s, v20.4s, v21.s[3]\n"
        "fmul v12.4s, v12.4s, v19.s[0]\n"
        "fmul v13.4s, v13.4s, v19.s[1]\n"
        "fmul v22.4s, v22.4s, v19.s[2]\n"
        "fadd v12.4s, v12.4s, v18.4s\n"
        "fmul v14.4s, v14.4s, v19.s[3]\n"
        "fadd v13.4s, v13.4s, v18.4s\n"
        "fadd v22.4s, v22.4s, v18.4s\n"
        "fadd v14.4s, v14.4s, v18.4s\n"
        "fmax v12.4s, v12.4s, v17.4s\n"
        "fmax v13.4s, v13.4s, v17.4s\n"
        "fmax v22.4s, v22.4s, v17.4s\n"
        "fmax v14.4s, v14.4s, v17.4s\n"
        "fmin v12.4s, v12.4s, v16.4s\n"
        "fmin v13.4s, v13.4s, v16.4s\n"
        "fmin v22.4s, v22.4s, v16.4s\n"
        "fmin v14.4s, v14.4s, v16.4s\n"
        "blt 17f\n"
        "mov x20, %x[dst]\n"
        "cmp x11, #0x1\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 20f\n"
        "cmp x11, #0x2\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 20f\n"
        "cmp x11, #0x3\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 20f\n"
        "str q14, [x20, #0x0]\n"
        "b 20f\n"
        "17:"  // Row tail: Partial output
        "mov x23, %x[dst]\n"
        "cmp x11, #0x1\n"
        "add x22, x23, %x[dst_stride_row]\n"
        "csel x22, x22, x23, GT\n"
        "cmp x11, #0x2\n"
        "add x21, x23, %x[dst_stride_row], LSL #1\n"
        "csel x21, x21, x22, GT\n"
        "cmp x11, #0x3\n"
        "add x20, x21, %x[dst_stride_row]\n"
        "csel x20, x20, x21, GT\n"
        "tbz x25, #1, 18f\n"
        "st1 { v14.d }[0], [x20], #0x8\n"
        "st1 { v22.d }[0], [x21], #0x8\n"
        "st1 { v13.d }[0], [x22], #0x8\n"
        "st1 { v12.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 19f\n"
        "st1 { v14.s }[2], [x20]\n"
        "st1 { v22.s }[2], [x21]\n"
        "st1 { v13.s }[2], [x22]\n"
        "st1 { v12.s }[2], [x23]\n"
        "b 19f\n"
        "18:"  // Row tail: Output block 0: partial_1_0
        "st1 { v14.s }[0], [x20]\n"
        "st1 { v22.s }[0], [x21]\n"
        "st1 { v13.s }[0], [x22]\n"
        "st1 { v12.s }[0], [x23]\n"
        "19:"  // Row tail: Output block 0: Done
        "20:"  // Row tail: Output stage exit
        "subs x25, x25, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 13b\n"
        "subs x11, x11, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x12\n"
        "mov %x[dst], x24\n"
        "bgt 12b\n"
        "21:"  // Row tail: Row loop skip
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [num_subblocks] "r"(num_subblocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "x9", "x10", "x11", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}
#endif  // Architectural feature check
