//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "I8mm extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 4;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
static const size_t kai_kr = 16;
static const size_t kai_sr = 2;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias = sizeof(float);

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

inline static size_t kai_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return kai_nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed,
    float* dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t k_internal = kai_k_roundedup(k);

    size_t num_blocks = k_internal / 32;

    float clamp_vals[2] = {scalar_min, scalar_max};

    __asm__ __volatile__(
        "mov x28, #0x80\n"
        "mov x20, #0x20\n"
        "movi v4.16b, #0xf0\n"
        "mov x27, %x[m]\n"
        "madd x28, %x[num_blocks], x28, x20\n"
        "cbz x27, 9f\n"
        "1:"  // Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "2:"  // Column loop
        "mov x21, %x[lhs_packed]\n"
        "movi v3.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        "mov x20, %x[num_blocks]\n"
        "movi v1.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "3:"  // Sub block loop
        "ldr q31, [x26, #0x0]\n"
        "ldr q30, [x26, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q29, [x21, #0x0]\n"
        "ldr q28, [x21, #0x10]\n"
        "ldr q27, [x26, #0x20]\n"
        "ldr q26, [x26, #0x30]\n"
        "add x26, x26, #0x40\n"
        "ldr q25, [x21, #0x20]\n"
        "ldr q24, [x21, #0x30]\n"
        "shl v23.16b, v31.16b, #0x4\n"
        "shl v22.16b, v30.16b, #0x4\n"
        "ldr q21, [x21, #0x40]\n"
        "ldr q20, [x21, #0x50]\n"
        "and v31.16b, v31.16b, v4.16b\n"
        "and v30.16b, v30.16b, v4.16b\n"
        "ldr q19, [x21, #0x60]\n"
        "ldr q18, [x21, #0x70]\n"
        "shl v17.16b, v27.16b, #0x4\n"
        "shl v16.16b, v26.16b, #0x4\n"
        ".inst 0x4e97a7a3  // smmla v3.4s, v29.16b, v23.16b\n"
        ".inst 0x4e96a7a2  // smmla v2.4s, v29.16b, v22.16b\n"
        "and v27.16b, v27.16b, v4.16b\n"
        "add x21, x21, #0x80\n"
        ".inst 0x4e97a781  // smmla v1.4s, v28.16b, v23.16b\n"
        ".inst 0x4e96a780  // smmla v0.4s, v28.16b, v22.16b\n"
        "and v26.16b, v26.16b, v4.16b\n"
        ".inst 0x4e91a723  // smmla v3.4s, v25.16b, v17.16b\n"
        ".inst 0x4e90a722  // smmla v2.4s, v25.16b, v16.16b\n"
        ".inst 0x4e91a701  // smmla v1.4s, v24.16b, v17.16b\n"
        ".inst 0x4e90a700  // smmla v0.4s, v24.16b, v16.16b\n"
        ".inst 0x4e9fa6a3  // smmla v3.4s, v21.16b, v31.16b\n"
        ".inst 0x4e9ea6a2  // smmla v2.4s, v21.16b, v30.16b\n"
        ".inst 0x4e9fa681  // smmla v1.4s, v20.16b, v31.16b\n"
        ".inst 0x4e9ea680  // smmla v0.4s, v20.16b, v30.16b\n"
        ".inst 0x4e9ba663  // smmla v3.4s, v19.16b, v27.16b\n"
        ".inst 0x4e9aa662  // smmla v2.4s, v19.16b, v26.16b\n"
        ".inst 0x4e9ba641  // smmla v1.4s, v18.16b, v27.16b\n"
        ".inst 0x4e9aa640  // smmla v0.4s, v18.16b, v26.16b\n"
        "bgt 3b\n"
        "ldr q18, [x26, #0x0]\n"
        "ld1 { v17.4s }, [x21]\n"
        "uzp1 v24.2d, v3.2d, v2.2d\n"
        "uzp2 v23.2d, v3.2d, v2.2d\n"
        "ldr q22, [x26, #0x10]\n"
        "uzp1 v21.2d, v1.2d, v0.2d\n"
        "uzp2 v20.2d, v1.2d, v0.2d\n"
        "add x21, x21, #0x10\n"
        "ldr q16, [x21, #0x0]\n"
        "add x26, x26, #0x20\n"
        "mla v24.4s, v18.4s, v17.s[0]\n"
        "mla v23.4s, v18.4s, v17.s[1]\n"
        "mla v21.4s, v18.4s, v17.s[2]\n"
        "mla v20.4s, v18.4s, v17.s[3]\n"
        "fmul v19.4s, v22.4s, v16.s[0]\n"
        "fmul v18.4s, v22.4s, v16.s[1]\n"
        "fmul v17.4s, v22.4s, v16.s[2]\n"
        "fmul v16.4s, v22.4s, v16.s[3]\n"
        "scvtf v24.4s, v24.4s\n"
        "scvtf v23.4s, v23.4s\n"
        "scvtf v21.4s, v21.4s\n"
        "scvtf v20.4s, v20.4s\n"
        "fmul v3.4s, v24.4s, v19.4s\n"
        "fmul v2.4s, v23.4s, v18.4s\n"
        "fmul v1.4s, v21.4s, v17.4s\n"
        "fmul v0.4s, v20.4s, v16.4s\n"
        "ldr q18, [x26, #0x0]\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x25, #0x4\n"
        "ld1r { v16.4s }, [x20]\n"
        "add x26, x26, #0x10\n"
        "fadd v3.4s, v3.4s, v18.4s\n"
        "fadd v2.4s, v2.4s, v18.4s\n"
        "fadd v1.4s, v1.4s, v18.4s\n"
        "fadd v0.4s, v0.4s, v18.4s\n"
        "fmax v3.4s, v3.4s, v17.4s\n"
        "fmax v2.4s, v2.4s, v17.4s\n"
        "fmax v1.4s, v1.4s, v17.4s\n"
        "fmax v0.4s, v0.4s, v17.4s\n"
        "fmin v3.4s, v3.4s, v16.4s\n"
        "fmin v2.4s, v2.4s, v16.4s\n"
        "fmin v1.4s, v1.4s, v16.4s\n"
        "fmin v0.4s, v0.4s, v16.4s\n"
        "blt 5f\n"
        "mov x20, %x[dst]\n"
        "cmp x27, #0x1\n"
        "str q3, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 8f\n"
        "cmp x27, #0x2\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 8f\n"
        "cmp x27, #0x3\n"
        "str q1, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 8f\n"
        "str q0, [x20, #0x0]\n"
        "b 8f\n"
        "5:"  // Partial output
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
        "tbz x25, #1, 6f\n"
        "st1 { v0.d }[0], [x20], #0x8\n"
        "st1 { v1.d }[0], [x21], #0x8\n"
        "st1 { v2.d }[0], [x22], #0x8\n"
        "st1 { v3.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 7f\n"
        "st1 { v0.s }[2], [x20]\n"
        "st1 { v1.s }[2], [x21]\n"
        "st1 { v2.s }[2], [x22]\n"
        "st1 { v3.s }[2], [x23]\n"
        "b 7f\n"
        "6:"  // Output block 0: partial_1_0
        "st1 { v0.s }[0], [x20]\n"
        "st1 { v1.s }[0], [x21]\n"
        "st1 { v2.s }[0], [x22]\n"
        "st1 { v3.s }[0], [x23]\n"
        "7:"  // Output block 0: Done
        "8:"  // Output stage exit
        "subs x25, x25, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "subs x27, x27, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x28\n"
        "mov %x[dst], x24\n"
        "bgt 1b\n"
        "9:"  // Row loop skip
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
          "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
          "x28");
}
#endif  // Architectural feature check
