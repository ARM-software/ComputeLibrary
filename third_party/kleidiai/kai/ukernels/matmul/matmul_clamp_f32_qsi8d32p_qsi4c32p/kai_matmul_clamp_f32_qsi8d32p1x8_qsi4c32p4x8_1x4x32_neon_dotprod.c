
//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 1;
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

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
    size_t m_idx, size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
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
        "mov x26, #0x22\n"
        "movi v1.16b, #0xf0\n"
        "mov x25, %x[m]\n"
        "mul x26, %x[num_blocks], x26\n"
        "1:"  // Row loop
        "mov x24, %x[rhs_packed]\n"
        "mov x23, %x[n]\n"
        "add x22, %x[dst], %x[dst_stride_row]\n"
        "2:"  // Column loop
        "mov x21, %x[lhs_packed]\n"
        "movi v0.16b, #0x0\n"
        "mov x20, %x[num_blocks]\n"
        "3:"  // Block loop
        "ldr d16, [x24, #0x0]\n"
        "ld1r { v31.8h }, [x21]\n"
        "add x24, x24, #0x8\n"
        "add x21, x21, #0x2\n"
        "ldr q30, [x24, #0x0]\n"
        "ldr q29, [x24, #0x10]\n"
        "movi v28.4s, #0x0\n"
        "movi v27.4s, #0x0\n"
        "ld1r { v26.2d }, [x21], #0x8\n"
        "ldr q25, [x24, #0x20]\n"
        "sub x20, x20, #0x1\n"
        "ldr q24, [x24, #0x30]\n"
        "fcvtl v31.4s, v31.4h\n"
        "fcvtl v23.4s, v16.4h\n"
        "add x24, x24, #0x40\n"
        "ld1r { v22.2d }, [x21], #0x8\n"
        "shl v21.16b, v30.16b, #0x4\n"
        "shl v20.16b, v29.16b, #0x4\n"
        "ld1r { v19.2d }, [x21], #0x8\n"
        "ld1r { v18.2d }, [x21], #0x8\n"
        "shl v17.16b, v25.16b, #0x4\n"
        "and v30.16b, v30.16b, v1.16b\n"
        "shl v16.16b, v24.16b, #0x4\n"
        "and v29.16b, v29.16b, v1.16b\n"
        ".inst 0x4e9a96bc  // sdot v28.4s, v21.16b, v26.16b\n"
        ".inst 0x4e9a969b  // sdot v27.4s, v20.16b, v26.16b\n"
        "and v25.16b, v25.16b, v1.16b\n"
        "and v24.16b, v24.16b, v1.16b\n"
        "fmul v23.4s, v23.4s, v31.4s\n"
        ".inst 0x4e96963c  // sdot v28.4s, v17.16b, v22.16b\n"
        ".inst 0x4e96961b  // sdot v27.4s, v16.16b, v22.16b\n"
        ".inst 0x4e9397dc  // sdot v28.4s, v30.16b, v19.16b\n"
        ".inst 0x4e9397bb  // sdot v27.4s, v29.16b, v19.16b\n"
        ".inst 0x4e92973c  // sdot v28.4s, v25.16b, v18.16b\n"
        ".inst 0x4e92971b  // sdot v27.4s, v24.16b, v18.16b\n"
        "addp v28.4s, v28.4s, v27.4s\n"
        "scvtf v28.4s, v28.4s, #0x4\n"
        "fmla v0.4s, v28.4s, v23.4s\n"
        "cbnz x20, 3b\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x23, #0x4\n"
        "ld1r { v16.4s }, [x20]\n"
        "fmax v0.4s, v0.4s, v17.4s\n"
        "fmin v0.4s, v0.4s, v16.4s\n"
        "blt 4f\n"
        "str q0, [%x[dst], #0x0]\n"
        "b 7f\n"
        "4:"  // Partial output
        "mov x20, %x[dst]\n"
        "tbz x23, #1, 5f\n"
        "st1 { v0.d }[0], [x20], #0x8\n"
        "tbz x23, #0, 6f\n"
        "st1 { v0.s }[2], [x20]\n"
        "b 6f\n"
        "5:"  // Output block 0: partial_1_0
        "st1 { v0.s }[0], [x20]\n"
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
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26");
}

#endif  // Architectural feature check
