//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 1;
static const size_t kai_n_step = 8;
static const size_t kai_mr = 1;
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

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(
    size_t m, size_t n, size_t k, size_t bl, const void* restrict lhs_packed, const void* restrict rhs_packed,
    float* restrict dst,  // NOLINT(readability-non-const-parameter)
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
        "mov x27, #0x20\n"
        "mov x21, #0x3d800000\n"
        "movi v8.16b, #0xf0\n"
        "mov x20, #0x8\n"
        "mov x26, %x[m]\n"
        "mul x27, %x[num_subblocks], x27\n"
        "dup v7.4s, w21\n"
        "madd x27, %x[num_blocks], x27, x20\n"
        "1:"  // Row loop
        "mov x25, %x[rhs_packed]\n"
        "mov x24, %x[n]\n"
        "add x23, %x[dst], %x[dst_stride_row]\n"
        "2:"  // Column loop
        "mov x22, %x[lhs_packed]\n"
        "movi v6.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "mov x21, %x[num_blocks]\n"
        "3:"  // Block loop
        "movi v4.4s, #0x0\n"
        "movi v3.4s, #0x0\n"
        "mov x20, %x[num_subblocks]\n"
        "movi v2.4s, #0x0\n"
        "movi v1.4s, #0x0\n"
        "4:"  // Sub block loop
        "ldr q0, [x25, #0x0]\n"
        "ldr q31, [x25, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q30, [x25, #0x20]\n"
        "ldr q29, [x25, #0x30]\n"
        "ld1r { v28.2d }, [x22], #0x8\n"
        "ldr q27, [x25, #0x40]\n"
        "ldr q26, [x25, #0x50]\n"
        "ldr q25, [x25, #0x60]\n"
        "shl v24.16b, v0.16b, #0x4\n"
        "shl v18.16b, v31.16b, #0x4\n"
        "ldr q23, [x25, #0x70]\n"
        "shl v17.16b, v30.16b, #0x4\n"
        "shl v16.16b, v29.16b, #0x4\n"
        "add x25, x25, #0x80\n"
        "ld1r { v22.2d }, [x22], #0x8\n"
        "shl v21.16b, v27.16b, #0x4\n"
        "and v0.16b, v0.16b, v8.16b\n"
        "ld1r { v20.2d }, [x22], #0x8\n"
        "ld1r { v19.2d }, [x22], #0x8\n"
        ".inst 0x4e9c9704  // sdot v4.4s, v24.16b, v28.16b\n"
        ".inst 0x4e9c9643  // sdot v3.4s, v18.16b, v28.16b\n"
        "shl v18.16b, v26.16b, #0x4\n"
        ".inst 0x4e9c9622  // sdot v2.4s, v17.16b, v28.16b\n"
        ".inst 0x4e9c9601  // sdot v1.4s, v16.16b, v28.16b\n"
        "shl v17.16b, v25.16b, #0x4\n"
        "shl v16.16b, v23.16b, #0x4\n"
        "and v31.16b, v31.16b, v8.16b\n"
        "and v30.16b, v30.16b, v8.16b\n"
        "and v29.16b, v29.16b, v8.16b\n"
        ".inst 0x4e9696a4  // sdot v4.4s, v21.16b, v22.16b\n"
        ".inst 0x4e969643  // sdot v3.4s, v18.16b, v22.16b\n"
        "and v27.16b, v27.16b, v8.16b\n"
        ".inst 0x4e969622  // sdot v2.4s, v17.16b, v22.16b\n"
        ".inst 0x4e969601  // sdot v1.4s, v16.16b, v22.16b\n"
        "and v26.16b, v26.16b, v8.16b\n"
        "and v25.16b, v25.16b, v8.16b\n"
        "and v23.16b, v23.16b, v8.16b\n"
        ".inst 0x4e949404  // sdot v4.4s, v0.16b, v20.16b\n"
        ".inst 0x4e9497e3  // sdot v3.4s, v31.16b, v20.16b\n"
        ".inst 0x4e9497c2  // sdot v2.4s, v30.16b, v20.16b\n"
        ".inst 0x4e9497a1  // sdot v1.4s, v29.16b, v20.16b\n"
        ".inst 0x4e939764  // sdot v4.4s, v27.16b, v19.16b\n"
        ".inst 0x4e939743  // sdot v3.4s, v26.16b, v19.16b\n"
        ".inst 0x4e939722  // sdot v2.4s, v25.16b, v19.16b\n"
        ".inst 0x4e9396e1  // sdot v1.4s, v23.16b, v19.16b\n"
        "bgt 4b\n"
        "ldr q16, [x25, #0x0]\n"
        "addp v4.4s, v4.4s, v3.4s\n"
        "addp v2.4s, v2.4s, v1.4s\n"
        "sub x21, x21, #0x1\n"
        "add x25, x25, #0x10\n"
        "shll v17.4s, v16.4h, #0x10\n"
        "shll2 v16.4s, v16.8h, #0x10\n"
        "scvtf v4.4s, v4.4s\n"
        "scvtf v2.4s, v2.4s\n"
        "fmul v17.4s, v17.4s, v7.4s\n"
        "fmul v16.4s, v16.4s, v7.4s\n"
        "fmla v6.4s, v4.4s, v17.4s\n"
        "fmla v5.4s, v2.4s, v16.4s\n"
        "cbnz x21, 3b\n"
        "ld1r { v23.4s }, [x22]\n"
        "ldr q22, [x25, #0x0]\n"
        "add x22, x22, #0x4\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ldr q21, [x25, #0x10]\n"
        "ld1r { v20.4s }, [x22]\n"
        "cmp x24, #0x8\n"
        "ldr q19, [x25, #0x20]\n"
        "ldr q18, [x25, #0x30]\n"
        "add x25, x25, #0x40\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "ld1r { v16.4s }, [x20]\n"
        "scvtf v23.4s, v23.4s\n"
        "fmla v6.4s, v22.4s, v23.s[0]\n"
        "fmla v5.4s, v21.4s, v23.s[0]\n"
        "fmul v6.4s, v6.4s, v20.4s\n"
        "fadd v6.4s, v6.4s, v19.4s\n"
        "fmul v5.4s, v5.4s, v20.4s\n"
        "fadd v5.4s, v5.4s, v18.4s\n"
        "fmax v6.4s, v6.4s, v17.4s\n"
        "fmax v5.4s, v5.4s, v17.4s\n"
        "fmin v6.4s, v6.4s, v16.4s\n"
        "fmin v5.4s, v5.4s, v16.4s\n"
        "blt 5f\n"
        "str q6, [%x[dst], #0x0]\n"
        "str q5, [%x[dst], #0x10]\n"
        "b 10f\n"
        "5:"  // Partial output
        "mov x20, %x[dst]\n"
        "tbz x24, #2, 7f\n"
        "st1 { v6.4s }, [x20], #0x10\n"
        "tbz x24, #1, 6f\n"
        "st1 { v5.d }[0], [x20], #0x8\n"
        "tbz x24, #0, 9f\n"
        "st1 { v5.s }[2], [x20]\n"
        "b 9f\n"
        "6:"  // Output block 0: partial_1_4
        "tbz x24, #0, 9f\n"
        "st1 { v5.s }[0], [x20]\n"
        "b 9f\n"
        "7:"  // Output block 0: partial_2_0
        "tbz x24, #1, 8f\n"
        "st1 { v6.d }[0], [x20], #0x8\n"
        "tbz x24, #0, 9f\n"
        "st1 { v6.s }[2], [x20]\n"
        "b 9f\n"
        "8:"  // Output block 0: partial_1_0
        "st1 { v6.s }[0], [x20]\n"
        "9:"   // Output block 0: Done
        "10:"  // Stores done
        "subs x24, x24, #0x8\n"
        "add %x[dst], %x[dst], #0x20\n"
        "bgt 2b\n"
        "subs x26, x26, #0x1\n"
        "add %x[lhs_packed], %x[lhs_packed], x27\n"
        "mov %x[dst], x23\n"
        "bgt 1b\n"
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [num_subblocks] "r"(num_subblocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23",
          "x24", "x25", "x26", "x27");
}
#endif  // Architectural feature check
