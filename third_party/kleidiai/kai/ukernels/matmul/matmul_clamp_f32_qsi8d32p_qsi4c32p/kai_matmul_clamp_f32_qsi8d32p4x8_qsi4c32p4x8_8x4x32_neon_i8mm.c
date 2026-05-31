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
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"

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

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t m_idx, size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
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
        "mov x12, %x[m]\n"
        "mov x11, #0x88\n"
        "movi v13.16b, #0xf0\n"
        "cmp x12, #0x8\n"
        "mul x11, %x[num_blocks], x11\n"
        "blt 8f\n"
        "1:"  // Row loop
        "mov x10, %x[rhs_packed]\n"
        "mov x9, %x[n]\n"
        "add x28, %x[dst], %x[dst_stride_row], LSL #3\n"
        "2:"  // Column loop
        "mov x22, %x[lhs_packed]\n"
        "movi v1.16b, #0x0\n"
        "movi v22.16b, #0x0\n"
        "mov x21, %x[num_blocks]\n"
        "movi v14.16b, #0x0\n"
        "movi v12.16b, #0x0\n"
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "movi v3.16b, #0x0\n"
        "movi v2.16b, #0x0\n"
        "add x20, x22, x11\n"
        "3:"  // Block loop
        "ldr d11, [x10, #0x0]\n"
        "ldr d10, [x22, #0x0]\n"
        "add x10, x10, #0x8\n"
        "add x22, x22, #0x8\n"
        "ldr q25, [x10, #0x0]\n"
        "ldr q30, [x10, #0x10]\n"
        "movi v6.4s, #0x0\n"
        "movi v21.4s, #0x0\n"
        "ldr d24, [x20, #0x0]\n"
        "ldr q28, [x22, #0x0]\n"
        "add x20, x20, #0x8\n"
        "movi v9.4s, #0x0\n"
        "ldr q4, [x22, #0x10]\n"
        "ldr q23, [x20, #0x0]\n"
        "movi v0.4s, #0x0\n"
        "movi v31.4s, #0x0\n"
        "ldr q17, [x20, #0x10]\n"
        "ldr q18, [x10, #0x20]\n"
        "shl v20.16b, v25.16b, #0x4\n"
        "shl v29.16b, v30.16b, #0x4\n"
        "ldr q16, [x10, #0x30]\n"
        "ldr q26, [x22, #0x20]\n"
        "movi v7.4s, #0x0\n"
        "movi v27.4s, #0x0\n"
        "ldr q8, [x22, #0x30]\n"
        "ldr q5, [x20, #0x20]\n"
        "and v25.16b, v25.16b, v13.16b\n"
        "and v30.16b, v30.16b, v13.16b\n"
        ".inst 0x4e94a786  // smmla v6.4s, v28.16b, v20.16b\n"
        ".inst 0x4e9da795  // smmla v21.4s, v28.16b, v29.16b\n"
        "ldr q28, [x20, #0x30]\n"
        "fcvtl v11.4s, v11.4h\n"
        ".inst 0x4e94a489  // smmla v9.4s, v4.16b, v20.16b\n"
        ".inst 0x4e9da480  // smmla v0.4s, v4.16b, v29.16b\n"
        "ldr q4, [x22, #0x40]\n"
        "fcvtl v10.4s, v10.4h\n"
        ".inst 0x4e94a6ff  // smmla v31.4s, v23.16b, v20.16b\n"
        ".inst 0x4e9da6e7  // smmla v7.4s, v23.16b, v29.16b\n"
        "ldr q23, [x22, #0x50]\n"
        "fcvtl v24.4s, v24.4h\n"
        ".inst 0x4e94a63b  // smmla v27.4s, v17.16b, v20.16b\n"
        "movi v20.4s, #0x0\n"
        "subs x21, x21, #0x1\n"
        "add x10, x10, #0x40\n"
        ".inst 0x4e9da634  // smmla v20.4s, v17.16b, v29.16b\n"
        "ldr q17, [x20, #0x40]\n"
        "shl v29.16b, v18.16b, #0x4\n"
        "and v18.16b, v18.16b, v13.16b\n"
        ".inst 0x4e9da746  // smmla v6.4s, v26.16b, v29.16b\n"
        ".inst 0x4e9da509  // smmla v9.4s, v8.16b, v29.16b\n"
        ".inst 0x4e9da4bf  // smmla v31.4s, v5.16b, v29.16b\n"
        ".inst 0x4e9da79b  // smmla v27.4s, v28.16b, v29.16b\n"
        "ldr q29, [x20, #0x50]\n"
        ".inst 0x4e99a486  // smmla v6.4s, v4.16b, v25.16b\n"
        ".inst 0x4e99a6e9  // smmla v9.4s, v23.16b, v25.16b\n"
        ".inst 0x4e99a63f  // smmla v31.4s, v17.16b, v25.16b\n"
        ".inst 0x4e99a7bb  // smmla v27.4s, v29.16b, v25.16b\n"
        "shl v25.16b, v16.16b, #0x4\n"
        "and v16.16b, v16.16b, v13.16b\n"
        ".inst 0x4e99a755  // smmla v21.4s, v26.16b, v25.16b\n"
        "ldr q26, [x22, #0x60]\n"
        ".inst 0x4e99a500  // smmla v0.4s, v8.16b, v25.16b\n"
        "ldr q8, [x22, #0x70]\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4e99a4a7  // smmla v7.4s, v5.16b, v25.16b\n"
        "ldr q5, [x20, #0x60]\n"
        ".inst 0x4e99a794  // smmla v20.4s, v28.16b, v25.16b\n"
        "ldr q25, [x20, #0x70]\n"
        "fmul v28.4s, v11.4s, v10.s[0]\n"
        "add x20, x20, #0x80\n"
        ".inst 0x4e92a746  // smmla v6.4s, v26.16b, v18.16b\n"
        ".inst 0x4e9ea495  // smmla v21.4s, v4.16b, v30.16b\n"
        "fmul v4.4s, v11.4s, v10.s[1]\n"
        ".inst 0x4e9ea6e0  // smmla v0.4s, v23.16b, v30.16b\n"
        ".inst 0x4e92a509  // smmla v9.4s, v8.16b, v18.16b\n"
        "fmul v23.4s, v11.4s, v10.s[2]\n"
        ".inst 0x4e9ea627  // smmla v7.4s, v17.16b, v30.16b\n"
        ".inst 0x4e92a4bf  // smmla v31.4s, v5.16b, v18.16b\n"
        "fmul v17.4s, v11.4s, v10.s[3]\n"
        ".inst 0x4e9ea7b4  // smmla v20.4s, v29.16b, v30.16b\n"
        ".inst 0x4e92a73b  // smmla v27.4s, v25.16b, v18.16b\n"
        "fmul v30.4s, v11.4s, v24.s[0]\n"
        ".inst 0x4e90a755  // smmla v21.4s, v26.16b, v16.16b\n"
        "fmul v29.4s, v11.4s, v24.s[1]\n"
        ".inst 0x4e90a500  // smmla v0.4s, v8.16b, v16.16b\n"
        "fmul v18.4s, v11.4s, v24.s[2]\n"
        "fmul v10.4s, v11.4s, v24.s[3]\n"
        ".inst 0x4e90a4a7  // smmla v7.4s, v5.16b, v16.16b\n"
        ".inst 0x4e90a734  // smmla v20.4s, v25.16b, v16.16b\n"
        "uzp1 v26.2d, v6.2d, v21.2d\n"
        "uzp2 v6.2d, v6.2d, v21.2d\n"
        "uzp1 v24.2d, v9.2d, v0.2d\n"
        "uzp2 v16.2d, v9.2d, v0.2d\n"
        "uzp1 v8.2d, v31.2d, v7.2d\n"
        "uzp2 v11.2d, v31.2d, v7.2d\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "uzp1 v31.2d, v27.2d, v20.2d\n"
        "uzp2 v7.2d, v27.2d, v20.2d\n"
        "scvtf v6.4s, v6.4s, #0x4\n"
        "scvtf v24.4s, v24.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "scvtf v8.4s, v8.4s, #0x4\n"
        "fmla v1.4s, v26.4s, v28.4s\n"
        "scvtf v11.4s, v11.4s, #0x4\n"
        "scvtf v31.4s, v31.4s, #0x4\n"
        "scvtf v7.4s, v7.4s, #0x4\n"
        "fmla v22.4s, v6.4s, v4.4s\n"
        "fmla v14.4s, v24.4s, v23.4s\n"
        "fmla v12.4s, v16.4s, v17.4s\n"
        "fmla v15.4s, v8.4s, v30.4s\n"
        "fmla v19.4s, v11.4s, v29.4s\n"
        "fmla v3.4s, v31.4s, v18.4s\n"
        "fmla v2.4s, v7.4s, v10.4s\n"
        "bgt 3b\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x9, #0x4\n"
        "ld1r { v10.4s }, [x20]\n"
        "fmax v1.4s, v1.4s, v17.4s\n"
        "fmax v22.4s, v22.4s, v17.4s\n"
        "fmax v14.4s, v14.4s, v17.4s\n"
        "fmax v12.4s, v12.4s, v17.4s\n"
        "fmax v15.4s, v15.4s, v17.4s\n"
        "fmax v19.4s, v19.4s, v17.4s\n"
        "fmax v3.4s, v3.4s, v17.4s\n"
        "fmax v2.4s, v2.4s, v17.4s\n"
        "fmin v1.4s, v1.4s, v10.4s\n"
        "fmin v22.4s, v22.4s, v10.4s\n"
        "fmin v14.4s, v14.4s, v10.4s\n"
        "fmin v12.4s, v12.4s, v10.4s\n"
        "fmin v15.4s, v15.4s, v10.4s\n"
        "fmin v19.4s, v19.4s, v10.4s\n"
        "fmin v3.4s, v3.4s, v10.4s\n"
        "fmin v2.4s, v2.4s, v10.4s\n"
        "blt 4f\n"
        "mov x20, %x[dst]\n"
        "str q1, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q14, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q3, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q2, [x20, #0x0]\n"
        "b 7f\n"
        "4:"  // Partial output
        "mov x27, %x[dst]\n"
        "add x26, x27, %x[dst_stride_row], LSL #2\n"
        "add x25, x26, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row]\n"
        "add x23, x25, %x[dst_stride_row]\n"
        "add x22, x27, %x[dst_stride_row], LSL #1\n"
        "add x21, x27, %x[dst_stride_row]\n"
        "add x20, x22, %x[dst_stride_row]\n"
        "tbz x9, #1, 5f\n"
        "st1 { v2.d }[0], [x23], #0x8\n"
        "st1 { v3.d }[0], [x25], #0x8\n"
        "st1 { v19.d }[0], [x24], #0x8\n"
        "st1 { v15.d }[0], [x26], #0x8\n"
        "st1 { v12.d }[0], [x20], #0x8\n"
        "st1 { v14.d }[0], [x22], #0x8\n"
        "st1 { v22.d }[0], [x21], #0x8\n"
        "st1 { v1.d }[0], [x27], #0x8\n"
        "tbz x9, #0, 6f\n"
        "st1 { v2.s }[2], [x23]\n"
        "st1 { v3.s }[2], [x25]\n"
        "st1 { v19.s }[2], [x24]\n"
        "st1 { v15.s }[2], [x26]\n"
        "st1 { v12.s }[2], [x20]\n"
        "st1 { v14.s }[2], [x22]\n"
        "st1 { v22.s }[2], [x21]\n"
        "st1 { v1.s }[2], [x27]\n"
        "b 6f\n"
        "5:"  // Output block 0: partial_1_0
        "st1 { v2.s }[0], [x23]\n"
        "st1 { v3.s }[0], [x25]\n"
        "st1 { v19.s }[0], [x24]\n"
        "st1 { v15.s }[0], [x26]\n"
        "st1 { v12.s }[0], [x20]\n"
        "st1 { v14.s }[0], [x22]\n"
        "st1 { v22.s }[0], [x21]\n"
        "st1 { v1.s }[0], [x27]\n"
        "6:"  // Output block 0: Done
        "7:"  // Output stage exit
        "subs x9, x9, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "mov x20, #0x2\n"
        "sub x12, x12, #0x8\n"
        "cmp x12, #0x8\n"
        "mov %x[dst], x28\n"
        "madd %x[lhs_packed], x20, x11, %x[lhs_packed]\n"
        "bge 1b\n"
        "8:"  // Row loop skip
        "cbz x12, 16f\n"
        "9:"  // Row tail: Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "10:"  // Row tail: Column loop
        "movi v1.16b, #0x0\n"
        "movi v22.16b, #0x0\n"
        "mov x22, %x[lhs_packed]\n"
        "mov x20, %x[num_blocks]\n"
        "movi v14.16b, #0x0\n"
        "movi v12.16b, #0x0\n"
        "11:"  // Row tail: Block loop
        "ldr d16, [x26, #0x0]\n"
        "ldr d6, [x22, #0x0]\n"
        "add x26, x26, #0x8\n"
        "add x22, x22, #0x8\n"
        "ldr q5, [x26, #0x0]\n"
        "ldr q4, [x26, #0x10]\n"
        "movi v7.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        "ldr q23, [x22, #0x0]\n"
        "ldr q27, [x22, #0x10]\n"
        "movi v0.4s, #0x0\n"
        "movi v31.4s, #0x0\n"
        "ldr q30, [x26, #0x20]\n"
        "ldr q29, [x26, #0x30]\n"
        "fcvtl v28.4s, v16.4h\n"
        "fcvtl v6.4s, v6.4h\n"
        "ldr q8, [x22, #0x20]\n"
        "ldr q26, [x22, #0x30]\n"
        "shl v21.16b, v5.16b, #0x4\n"
        "shl v20.16b, v4.16b, #0x4\n"
        "ldr q25, [x22, #0x40]\n"
        "ldr q24, [x22, #0x50]\n"
        "and v5.16b, v5.16b, v13.16b\n"
        "and v4.16b, v4.16b, v13.16b\n"
        "ldr q19, [x22, #0x60]\n"
        "ldr q18, [x22, #0x70]\n"
        "shl v17.16b, v30.16b, #0x4\n"
        "shl v16.16b, v29.16b, #0x4\n"
        ".inst 0x4e95a6e7  // smmla v7.4s, v23.16b, v21.16b\n"
        ".inst 0x4e94a6e2  // smmla v2.4s, v23.16b, v20.16b\n"
        "and v30.16b, v30.16b, v13.16b\n"
        "subs x20, x20, #0x1\n"
        ".inst 0x4e95a760  // smmla v0.4s, v27.16b, v21.16b\n"
        ".inst 0x4e94a77f  // smmla v31.4s, v27.16b, v20.16b\n"
        "and v29.16b, v29.16b, v13.16b\n"
        "add x26, x26, #0x40\n"
        "fmul v23.4s, v28.4s, v6.s[0]\n"
        "fmul v10.4s, v28.4s, v6.s[1]\n"
        "add x22, x22, #0x80\n"
        "fmul v21.4s, v28.4s, v6.s[2]\n"
        "fmul v20.4s, v28.4s, v6.s[3]\n"
        ".inst 0x4e91a507  // smmla v7.4s, v8.16b, v17.16b\n"
        ".inst 0x4e90a502  // smmla v2.4s, v8.16b, v16.16b\n"
        ".inst 0x4e91a740  // smmla v0.4s, v26.16b, v17.16b\n"
        ".inst 0x4e90a75f  // smmla v31.4s, v26.16b, v16.16b\n"
        ".inst 0x4e85a727  // smmla v7.4s, v25.16b, v5.16b\n"
        ".inst 0x4e84a722  // smmla v2.4s, v25.16b, v4.16b\n"
        ".inst 0x4e85a700  // smmla v0.4s, v24.16b, v5.16b\n"
        ".inst 0x4e84a71f  // smmla v31.4s, v24.16b, v4.16b\n"
        ".inst 0x4e9ea667  // smmla v7.4s, v19.16b, v30.16b\n"
        ".inst 0x4e9da662  // smmla v2.4s, v19.16b, v29.16b\n"
        ".inst 0x4e9ea640  // smmla v0.4s, v18.16b, v30.16b\n"
        ".inst 0x4e9da65f  // smmla v31.4s, v18.16b, v29.16b\n"
        "uzp1 v19.2d, v7.2d, v2.2d\n"
        "uzp2 v18.2d, v7.2d, v2.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp1 v17.2d, v0.2d, v31.2d\n"
        "uzp2 v16.2d, v0.2d, v31.2d\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v1.4s, v19.4s, v23.4s\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v22.4s, v18.4s, v10.4s\n"
        "fmla v14.4s, v17.4s, v21.4s\n"
        "fmla v12.4s, v16.4s, v20.4s\n"
        "bgt 11b\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x25, #0x4\n"
        "ld1r { v16.4s }, [x20]\n"
        "fmax v1.4s, v1.4s, v17.4s\n"
        "fmax v22.4s, v22.4s, v17.4s\n"
        "fmax v14.4s, v14.4s, v17.4s\n"
        "fmax v12.4s, v12.4s, v17.4s\n"
        "fmin v1.4s, v1.4s, v16.4s\n"
        "fmin v22.4s, v22.4s, v16.4s\n"
        "fmin v14.4s, v14.4s, v16.4s\n"
        "fmin v12.4s, v12.4s, v16.4s\n"
        "blt 12f\n"
        "mov x20, %x[dst]\n"
        "cmp x12, #0x1\n"
        "str q1, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 15f\n"
        "cmp x12, #0x2\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 15f\n"
        "cmp x12, #0x3\n"
        "str q14, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 15f\n"
        "str q12, [x20, #0x0]\n"
        "b 15f\n"
        "12:"  // Row tail: Partial output
        "mov x23, %x[dst]\n"
        "cmp x12, #0x1\n"
        "add x22, x23, %x[dst_stride_row]\n"
        "csel x22, x22, x23, GT\n"
        "cmp x12, #0x2\n"
        "add x21, x23, %x[dst_stride_row], LSL #1\n"
        "csel x21, x21, x22, GT\n"
        "cmp x12, #0x3\n"
        "add x20, x21, %x[dst_stride_row]\n"
        "csel x20, x20, x21, GT\n"
        "tbz x25, #1, 13f\n"
        "st1 { v12.d }[0], [x20], #0x8\n"
        "st1 { v14.d }[0], [x21], #0x8\n"
        "st1 { v22.d }[0], [x22], #0x8\n"
        "st1 { v1.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 14f\n"
        "st1 { v12.s }[2], [x20]\n"
        "st1 { v14.s }[2], [x21]\n"
        "st1 { v22.s }[2], [x22]\n"
        "st1 { v1.s }[2], [x23]\n"
        "b 14f\n"
        "13:"  // Row tail: Output block 0: partial_1_0
        "st1 { v12.s }[0], [x20]\n"
        "st1 { v14.s }[0], [x21]\n"
        "st1 { v22.s }[0], [x22]\n"
        "st1 { v1.s }[0], [x23]\n"
        "14:"  // Row tail: Output block 0: Done
        "15:"  // Row tail: Output stage exit
        "subs x25, x25, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 10b\n"
        "subs x12, x12, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x11\n"
        "mov %x[dst], x24\n"
        "bgt 9b\n"
        "16:"  // Row tail: Row loop skip
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "x9", "x10", "x11", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}

#endif  // Architectural feature check
