//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Compute args
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;
// Packing args
static const size_t kai_mr = 1;
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

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(
    size_t m_idx, size_t k, size_t bl) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod(
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
    KAI_ASSUME(m == 1);
    KAI_ASSUME(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }
    size_t num_blocks = kai_num_blocks_per_row(k, bl);
    KAI_UNUSED(scalar_min);
    KAI_UNUSED(scalar_max);

    __asm__ __volatile__(
        "mov x26, #0x22\n"
        "movi v30.16b, #0xf0\n"
        "mov x25, %x[m]\n"
        "mul x26, %x[num_blocks], x26\n"
        "1:"  // Row loop
        "mov x24, %x[rhs_packed]\n"
        "mov x23, %x[n]\n"
        "add x22, %x[dst], %x[dst_stride_row]\n"
        "2:"  // Column loop
        "mov x21, %x[lhs_packed]\n"
        "movi v29.16b, #0x0\n"
        "mov x20, %x[num_blocks]\n"
        "3:"  // Block loop
        "ldr d16, [x24, #0x0]\n"
        "ld1r { v28.8h }, [x21]\n"
        "add x24, x24, #0x8\n"
        "add x21, x21, #0x2\n"
        "ldr q27, [x24, #0x0]\n"
        "ldr q26, [x21, #0x0]\n"
        "movi v25.4s, #0x0\n"
        "sub x20, x20, #0x1\n"
        "ldr q24, [x24, #0x10]\n"
        "ldr q23, [x24, #0x20]\n"
        "ldr q22, [x24, #0x30]\n"
        "ldr q21, [x21, #0x10]\n"
        "fcvtl v28.4s, v28.4h\n"
        "fcvtl v20.4s, v16.4h\n"
        "shl v19.16b, v27.16b, #0x4\n"
        "and v27.16b, v27.16b, v30.16b\n"
        "add x24, x24, #0x40\n"
        "add x21, x21, #0x20\n"
        "shl v18.16b, v24.16b, #0x4\n"
        "shl v17.16b, v23.16b, #0x4\n"
        "shl v16.16b, v22.16b, #0x4\n"
        "and v24.16b, v24.16b, v30.16b\n"
        ".inst 0x4f9ae279  // sdot v25.4s, v19.16b, v26.4b[0]\n"
        "and v23.16b, v23.16b, v30.16b\n"
        "and v22.16b, v22.16b, v30.16b\n"
        "fmul v20.4s, v20.4s, v28.4s\n"
        ".inst 0x4fbae259  // sdot v25.4s, v18.16b, v26.4b[1]\n"
        ".inst 0x4f9aea39  // sdot v25.4s, v17.16b, v26.4b[2]\n"
        ".inst 0x4fbaea19  // sdot v25.4s, v16.16b, v26.4b[3]\n"
        ".inst 0x4f95e379  // sdot v25.4s, v27.16b, v21.4b[0]\n"
        ".inst 0x4fb5e319  // sdot v25.4s, v24.16b, v21.4b[1]\n"
        ".inst 0x4f95eaf9  // sdot v25.4s, v23.16b, v21.4b[2]\n"
        ".inst 0x4fb5ead9  // sdot v25.4s, v22.16b, v21.4b[3]\n"
        "scvtf v25.4s, v25.4s, #0x4\n"
        "fmla v29.4s, v25.4s, v20.4s\n"
        "cbnz x20, 3b\n"
        "cmp x23, #0x4\n"
        "blt 4f\n"
        "str q29, [%x[dst], #0x0]\n"
        "b 7f\n"
        "4:"  // Partial output
        "mov x20, %x[dst]\n"
        "tbz x23, #1, 5f\n"
        "st1 { v29.d }[0], [x20], #0x8\n"
        "tbz x23, #0, 6f\n"
        "st1 { v29.s }[2], [x20]\n"
        "b 6f\n"
        "5:"  // Output block 0: partial_1_0
        "st1 { v29.s }[0], [x20]\n"
        "6:"  // Output block 0: Done
        "7:"  // Stores done
        "subs x23, x23, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "subs x25, x25, #0x1\n"
        "add %x[lhs_packed], %x[lhs_packed], x26\n"
        "mov %x[dst], x22\n"
        "bgt 1b\n"
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n), [num_blocks] "r"(num_blocks),
          [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "x20", "x21", "x22", "x23", "x24", "x25", "x26");
}

#endif  // Architectural features check.
