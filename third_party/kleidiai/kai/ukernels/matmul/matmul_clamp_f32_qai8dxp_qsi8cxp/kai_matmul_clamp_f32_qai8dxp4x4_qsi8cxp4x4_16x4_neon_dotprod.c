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

#include "kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Compute args
static const size_t kai_m_step = 16;
static const size_t kai_n_step = 4;
// Packing args
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 4;
static const size_t kai_num_bytes_zp_lhs = 4;
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric), and reduction sum (if LHS is
// asymmetric))
static const size_t kai_num_bytes_qvalue_rhs = 1;
static const size_t kai_num_bytes_multiplier_rhs = 4;
static const size_t kai_num_bytes_rsum_rhs = 4;
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_num_bytes_bias = 4;
static const size_t kai_k_multiple_of = 32;

inline static size_t kai_k_roundedup(size_t k) {
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);
    size_t lhs_packed_stride = kai_mr * ((k_internal * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs);
    // Since the LHS matrix is asymmetric with per-row quantization, we must include the
    // the number of bytes to hold the zero point value
    lhs_packed_stride += kai_mr * kai_num_bytes_zp_lhs;

    return lhs_packed_stride;
}

inline static size_t kai_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);
    size_t rhs_packed_stride = kai_nr * (k_internal * kai_num_bytes_qvalue_rhs);
    rhs_packed_stride += kai_nr * kai_num_bytes_multiplier_rhs;
    // Since the LHS matrix is quantized asymmetric with per-row quantization, we also include
    // the number of bytes for the reduction sum
    rhs_packed_stride += kai_nr * kai_num_bytes_rsum_rhs;
    // Since the bias is packed with the RHS matrix, the stride is adjusted with the number of bytes of the bias
    rhs_packed_stride += kai_nr * kai_num_bytes_bias;

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(size_t n_idx, size_t k) {
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(
    size_t m,                         //
    size_t n,                         //
    size_t k,                         //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row,            //
    size_t dst_stride_col,            //
    float scalar_min,                 //
    float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t k_internal = kai_k_roundedup(k);
    const size_t num_blocks = k_internal / kai_k_multiple_of;
    const float clamp_vals[2] = {scalar_min, scalar_max};

    __asm__ __volatile__(
        "mov x13, %x[m]\n"
        "mov x12, #0x80\n"
        "mov x20, #0x20\n"
        "cmp x13, #0x10\n"
        "madd x12, %x[num_blocks], x12, x20\n"
        "blt 14f\n"
        "1:"  // Row loop
        "mov x11, %x[rhs_packed]\n"
        "mov x10, %x[n]\n"
        "add x9, %x[dst], %x[dst_stride_row], LSL #4\n"
        "2:"  // Column loop
        "mov x27, %x[lhs_packed]\n"
        "movi v31.4s, #0x0\n"
        "movi v30.4s, #0x0\n"
        "mov x23, %x[num_blocks]\n"
        "movi v29.4s, #0x0\n"
        "movi v28.4s, #0x0\n"
        "movi v27.4s, #0x0\n"
        "movi v26.4s, #0x0\n"
        "add x22, x27, x12\n"
        "add x21, x22, x12\n"
        "add x20, x21, x12\n"
        "movi v25.4s, #0x0\n"
        "movi v24.4s, #0x0\n"
        "movi v23.4s, #0x0\n"
        "movi v22.4s, #0x0\n"
        "movi v21.4s, #0x0\n"
        "movi v20.4s, #0x0\n"
        "movi v19.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        "movi v17.4s, #0x0\n"
        "movi v16.4s, #0x0\n"
        "3:"  // Sub block loop
        "ldr q15, [x11, #0x0]\n"
        "ldr q7, [x27, #0x0]\n"
        "subs x23, x23, #0x1\n"
        "ldr q5, [x22, #0x0]\n"
        "ldr q6, [x21, #0x0]\n"
        "ldr q4, [x20, #0x0]\n"
        "ldr q14, [x11, #0x10]\n"
        "ldr q3, [x27, #0x10]\n"
        "ldr q2, [x22, #0x10]\n"
        ".inst 0x4f87e1ff  // sdot v31.4s, v15.16b, v7.4b[0]\n"
        ".inst 0x4fa7e1fe  // sdot v30.4s, v15.16b, v7.4b[1]\n"
        "ldr q1, [x21, #0x10]\n"
        "ldr q0, [x20, #0x10]\n"
        ".inst 0x4f87e9fd  // sdot v29.4s, v15.16b, v7.4b[2]\n"
        ".inst 0x4fa7e9fc  // sdot v28.4s, v15.16b, v7.4b[3]\n"
        "ldr q10, [x11, #0x20]\n"
        "ldr q13, [x27, #0x20]\n"
        ".inst 0x4f85e1fb  // sdot v27.4s, v15.16b, v5.4b[0]\n"
        ".inst 0x4fa5e1fa  // sdot v26.4s, v15.16b, v5.4b[1]\n"
        "ldr q12, [x22, #0x20]\n"
        "ldr q11, [x21, #0x20]\n"
        ".inst 0x4f85e9f9  // sdot v25.4s, v15.16b, v5.4b[2]\n"
        ".inst 0x4fa5e9f8  // sdot v24.4s, v15.16b, v5.4b[3]\n"
        "ldr q9, [x20, #0x20]\n"
        "ldr q5, [x11, #0x30]\n"
        ".inst 0x4f86e1f7  // sdot v23.4s, v15.16b, v6.4b[0]\n"
        ".inst 0x4fa6e1f6  // sdot v22.4s, v15.16b, v6.4b[1]\n"
        "ldr q8, [x27, #0x30]\n"
        "ldr q7, [x22, #0x30]\n"
        ".inst 0x4f86e9f5  // sdot v21.4s, v15.16b, v6.4b[2]\n"
        ".inst 0x4fa6e9f4  // sdot v20.4s, v15.16b, v6.4b[3]\n"
        "ldr q6, [x21, #0x30]\n"
        ".inst 0x4f84e1f3  // sdot v19.4s, v15.16b, v4.4b[0]\n"
        ".inst 0x4fa4e1f2  // sdot v18.4s, v15.16b, v4.4b[1]\n"
        ".inst 0x4f84e9f1  // sdot v17.4s, v15.16b, v4.4b[2]\n"
        ".inst 0x4fa4e9f0  // sdot v16.4s, v15.16b, v4.4b[3]\n"
        "ldr q4, [x20, #0x30]\n"
        "ldr q15, [x11, #0x40]\n"
        ".inst 0x4f83e1df  // sdot v31.4s, v14.16b, v3.4b[0]\n"
        ".inst 0x4fa3e1de  // sdot v30.4s, v14.16b, v3.4b[1]\n"
        ".inst 0x4f83e9dd  // sdot v29.4s, v14.16b, v3.4b[2]\n"
        ".inst 0x4fa3e9dc  // sdot v28.4s, v14.16b, v3.4b[3]\n"
        "ldr q3, [x27, #0x40]\n"
        ".inst 0x4f82e1db  // sdot v27.4s, v14.16b, v2.4b[0]\n"
        ".inst 0x4fa2e1da  // sdot v26.4s, v14.16b, v2.4b[1]\n"
        ".inst 0x4f82e9d9  // sdot v25.4s, v14.16b, v2.4b[2]\n"
        ".inst 0x4fa2e9d8  // sdot v24.4s, v14.16b, v2.4b[3]\n"
        "ldr q2, [x22, #0x40]\n"
        ".inst 0x4f81e1d7  // sdot v23.4s, v14.16b, v1.4b[0]\n"
        ".inst 0x4fa1e1d6  // sdot v22.4s, v14.16b, v1.4b[1]\n"
        ".inst 0x4f81e9d5  // sdot v21.4s, v14.16b, v1.4b[2]\n"
        ".inst 0x4fa1e9d4  // sdot v20.4s, v14.16b, v1.4b[3]\n"
        "ldr q1, [x21, #0x40]\n"
        ".inst 0x4f80e1d3  // sdot v19.4s, v14.16b, v0.4b[0]\n"
        ".inst 0x4fa0e1d2  // sdot v18.4s, v14.16b, v0.4b[1]\n"
        ".inst 0x4f80e9d1  // sdot v17.4s, v14.16b, v0.4b[2]\n"
        ".inst 0x4fa0e9d0  // sdot v16.4s, v14.16b, v0.4b[3]\n"
        "ldr q0, [x20, #0x40]\n"
        "ldr q14, [x11, #0x50]\n"
        ".inst 0x4f8de15f  // sdot v31.4s, v10.16b, v13.4b[0]\n"
        ".inst 0x4fade15e  // sdot v30.4s, v10.16b, v13.4b[1]\n"
        ".inst 0x4f8de95d  // sdot v29.4s, v10.16b, v13.4b[2]\n"
        ".inst 0x4fade95c  // sdot v28.4s, v10.16b, v13.4b[3]\n"
        "ldr q13, [x27, #0x50]\n"
        ".inst 0x4f8ce15b  // sdot v27.4s, v10.16b, v12.4b[0]\n"
        ".inst 0x4face15a  // sdot v26.4s, v10.16b, v12.4b[1]\n"
        ".inst 0x4f8ce959  // sdot v25.4s, v10.16b, v12.4b[2]\n"
        ".inst 0x4face958  // sdot v24.4s, v10.16b, v12.4b[3]\n"
        "ldr q12, [x22, #0x50]\n"
        ".inst 0x4f8be157  // sdot v23.4s, v10.16b, v11.4b[0]\n"
        ".inst 0x4fabe156  // sdot v22.4s, v10.16b, v11.4b[1]\n"
        ".inst 0x4f8be955  // sdot v21.4s, v10.16b, v11.4b[2]\n"
        ".inst 0x4fabe954  // sdot v20.4s, v10.16b, v11.4b[3]\n"
        "ldr q11, [x21, #0x50]\n"
        ".inst 0x4f89e153  // sdot v19.4s, v10.16b, v9.4b[0]\n"
        ".inst 0x4fa9e152  // sdot v18.4s, v10.16b, v9.4b[1]\n"
        ".inst 0x4f89e951  // sdot v17.4s, v10.16b, v9.4b[2]\n"
        ".inst 0x4fa9e950  // sdot v16.4s, v10.16b, v9.4b[3]\n"
        "ldr q10, [x20, #0x50]\n"
        "ldr q9, [x11, #0x60]\n"
        ".inst 0x4f88e0bf  // sdot v31.4s, v5.16b, v8.4b[0]\n"
        ".inst 0x4fa8e0be  // sdot v30.4s, v5.16b, v8.4b[1]\n"
        ".inst 0x4f88e8bd  // sdot v29.4s, v5.16b, v8.4b[2]\n"
        ".inst 0x4fa8e8bc  // sdot v28.4s, v5.16b, v8.4b[3]\n"
        "ldr q8, [x27, #0x60]\n"
        ".inst 0x4f87e0bb  // sdot v27.4s, v5.16b, v7.4b[0]\n"
        ".inst 0x4fa7e0ba  // sdot v26.4s, v5.16b, v7.4b[1]\n"
        ".inst 0x4f87e8b9  // sdot v25.4s, v5.16b, v7.4b[2]\n"
        ".inst 0x4fa7e8b8  // sdot v24.4s, v5.16b, v7.4b[3]\n"
        "ldr q7, [x22, #0x60]\n"
        ".inst 0x4f86e0b7  // sdot v23.4s, v5.16b, v6.4b[0]\n"
        ".inst 0x4fa6e0b6  // sdot v22.4s, v5.16b, v6.4b[1]\n"
        ".inst 0x4f86e8b5  // sdot v21.4s, v5.16b, v6.4b[2]\n"
        ".inst 0x4fa6e8b4  // sdot v20.4s, v5.16b, v6.4b[3]\n"
        "ldr q6, [x21, #0x60]\n"
        ".inst 0x4f84e0b3  // sdot v19.4s, v5.16b, v4.4b[0]\n"
        ".inst 0x4fa4e0b2  // sdot v18.4s, v5.16b, v4.4b[1]\n"
        ".inst 0x4f84e8b1  // sdot v17.4s, v5.16b, v4.4b[2]\n"
        ".inst 0x4fa4e8b0  // sdot v16.4s, v5.16b, v4.4b[3]\n"
        "ldr q5, [x20, #0x60]\n"
        "ldr q4, [x11, #0x70]\n"
        ".inst 0x4f83e1ff  // sdot v31.4s, v15.16b, v3.4b[0]\n"
        ".inst 0x4fa3e1fe  // sdot v30.4s, v15.16b, v3.4b[1]\n"
        "add x11, x11, #0x80\n"
        ".inst 0x4f83e9fd  // sdot v29.4s, v15.16b, v3.4b[2]\n"
        ".inst 0x4fa3e9fc  // sdot v28.4s, v15.16b, v3.4b[3]\n"
        "ldr q3, [x27, #0x70]\n"
        "add x27, x27, #0x80\n"
        ".inst 0x4f82e1fb  // sdot v27.4s, v15.16b, v2.4b[0]\n"
        ".inst 0x4fa2e1fa  // sdot v26.4s, v15.16b, v2.4b[1]\n"
        ".inst 0x4f82e9f9  // sdot v25.4s, v15.16b, v2.4b[2]\n"
        ".inst 0x4fa2e9f8  // sdot v24.4s, v15.16b, v2.4b[3]\n"
        "ldr q2, [x22, #0x70]\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4f81e1f7  // sdot v23.4s, v15.16b, v1.4b[0]\n"
        ".inst 0x4fa1e1f6  // sdot v22.4s, v15.16b, v1.4b[1]\n"
        ".inst 0x4f81e9f5  // sdot v21.4s, v15.16b, v1.4b[2]\n"
        ".inst 0x4fa1e9f4  // sdot v20.4s, v15.16b, v1.4b[3]\n"
        "ldr q1, [x21, #0x70]\n"
        "add x21, x21, #0x80\n"
        ".inst 0x4f80e1f3  // sdot v19.4s, v15.16b, v0.4b[0]\n"
        ".inst 0x4fa0e1f2  // sdot v18.4s, v15.16b, v0.4b[1]\n"
        ".inst 0x4f80e9f1  // sdot v17.4s, v15.16b, v0.4b[2]\n"
        ".inst 0x4fa0e9f0  // sdot v16.4s, v15.16b, v0.4b[3]\n"
        "ldr q0, [x20, #0x70]\n"
        "add x20, x20, #0x80\n"
        ".inst 0x4f8de1df  // sdot v31.4s, v14.16b, v13.4b[0]\n"
        ".inst 0x4fade1de  // sdot v30.4s, v14.16b, v13.4b[1]\n"
        ".inst 0x4f8de9dd  // sdot v29.4s, v14.16b, v13.4b[2]\n"
        ".inst 0x4fade9dc  // sdot v28.4s, v14.16b, v13.4b[3]\n"
        ".inst 0x4f8ce1db  // sdot v27.4s, v14.16b, v12.4b[0]\n"
        ".inst 0x4face1da  // sdot v26.4s, v14.16b, v12.4b[1]\n"
        ".inst 0x4f8ce9d9  // sdot v25.4s, v14.16b, v12.4b[2]\n"
        ".inst 0x4face9d8  // sdot v24.4s, v14.16b, v12.4b[3]\n"
        ".inst 0x4f8be1d7  // sdot v23.4s, v14.16b, v11.4b[0]\n"
        ".inst 0x4fabe1d6  // sdot v22.4s, v14.16b, v11.4b[1]\n"
        ".inst 0x4f8be9d5  // sdot v21.4s, v14.16b, v11.4b[2]\n"
        ".inst 0x4fabe9d4  // sdot v20.4s, v14.16b, v11.4b[3]\n"
        ".inst 0x4f8ae1d3  // sdot v19.4s, v14.16b, v10.4b[0]\n"
        ".inst 0x4faae1d2  // sdot v18.4s, v14.16b, v10.4b[1]\n"
        ".inst 0x4f8ae9d1  // sdot v17.4s, v14.16b, v10.4b[2]\n"
        ".inst 0x4faae9d0  // sdot v16.4s, v14.16b, v10.4b[3]\n"
        ".inst 0x4f88e13f  // sdot v31.4s, v9.16b, v8.4b[0]\n"
        ".inst 0x4fa8e13e  // sdot v30.4s, v9.16b, v8.4b[1]\n"
        ".inst 0x4f88e93d  // sdot v29.4s, v9.16b, v8.4b[2]\n"
        ".inst 0x4fa8e93c  // sdot v28.4s, v9.16b, v8.4b[3]\n"
        ".inst 0x4f87e13b  // sdot v27.4s, v9.16b, v7.4b[0]\n"
        ".inst 0x4fa7e13a  // sdot v26.4s, v9.16b, v7.4b[1]\n"
        ".inst 0x4f87e939  // sdot v25.4s, v9.16b, v7.4b[2]\n"
        ".inst 0x4fa7e938  // sdot v24.4s, v9.16b, v7.4b[3]\n"
        ".inst 0x4f86e137  // sdot v23.4s, v9.16b, v6.4b[0]\n"
        ".inst 0x4fa6e136  // sdot v22.4s, v9.16b, v6.4b[1]\n"
        ".inst 0x4f86e935  // sdot v21.4s, v9.16b, v6.4b[2]\n"
        ".inst 0x4fa6e934  // sdot v20.4s, v9.16b, v6.4b[3]\n"
        ".inst 0x4f85e133  // sdot v19.4s, v9.16b, v5.4b[0]\n"
        ".inst 0x4fa5e132  // sdot v18.4s, v9.16b, v5.4b[1]\n"
        ".inst 0x4f85e931  // sdot v17.4s, v9.16b, v5.4b[2]\n"
        ".inst 0x4fa5e930  // sdot v16.4s, v9.16b, v5.4b[3]\n"
        ".inst 0x4f83e09f  // sdot v31.4s, v4.16b, v3.4b[0]\n"
        ".inst 0x4fa3e09e  // sdot v30.4s, v4.16b, v3.4b[1]\n"
        ".inst 0x4f83e89d  // sdot v29.4s, v4.16b, v3.4b[2]\n"
        ".inst 0x4fa3e89c  // sdot v28.4s, v4.16b, v3.4b[3]\n"
        ".inst 0x4f82e09b  // sdot v27.4s, v4.16b, v2.4b[0]\n"
        ".inst 0x4fa2e09a  // sdot v26.4s, v4.16b, v2.4b[1]\n"
        ".inst 0x4f82e899  // sdot v25.4s, v4.16b, v2.4b[2]\n"
        ".inst 0x4fa2e898  // sdot v24.4s, v4.16b, v2.4b[3]\n"
        ".inst 0x4f81e097  // sdot v23.4s, v4.16b, v1.4b[0]\n"
        ".inst 0x4fa1e096  // sdot v22.4s, v4.16b, v1.4b[1]\n"
        ".inst 0x4f81e895  // sdot v21.4s, v4.16b, v1.4b[2]\n"
        ".inst 0x4fa1e894  // sdot v20.4s, v4.16b, v1.4b[3]\n"
        ".inst 0x4f80e093  // sdot v19.4s, v4.16b, v0.4b[0]\n"
        ".inst 0x4fa0e092  // sdot v18.4s, v4.16b, v0.4b[1]\n"
        ".inst 0x4f80e891  // sdot v17.4s, v4.16b, v0.4b[2]\n"
        ".inst 0x4fa0e890  // sdot v16.4s, v4.16b, v0.4b[3]\n"
        "bgt 3b\n"
        "ldr q5, [x11, #0x0]\n"
        "ld1 { v1.4s }, [x27]\n"
        "add x27, x27, #0x10\n"
        "ldr q4, [x11, #0x10]\n"
        "ldr q0, [x27, #0x0]\n"
        "add x11, x11, #0x20\n"
        "mla v31.4s, v5.4s, v1.s[0]\n"
        "mla v30.4s, v5.4s, v1.s[1]\n"
        "mla v29.4s, v5.4s, v1.s[2]\n"
        "mla v28.4s, v5.4s, v1.s[3]\n"
        "fmul v3.4s, v4.4s, v0.s[0]\n"
        "fmul v2.4s, v4.4s, v0.s[1]\n"
        "fmul v1.4s, v4.4s, v0.s[2]\n"
        "scvtf v31.4s, v31.4s\n"
        "fmul v0.4s, v4.4s, v0.s[3]\n"
        "scvtf v30.4s, v30.4s\n"
        "scvtf v29.4s, v29.4s\n"
        "scvtf v28.4s, v28.4s\n"
        "fmul v31.4s, v31.4s, v3.4s\n"
        "fmul v30.4s, v30.4s, v2.4s\n"
        "fmul v29.4s, v29.4s, v1.4s\n"
        "fmul v28.4s, v28.4s, v0.4s\n"
        "ld1 { v1.4s }, [x22]\n"
        "add x22, x22, #0x10\n"
        "ldr q0, [x22, #0x0]\n"
        "mla v27.4s, v5.4s, v1.s[0]\n"
        "mla v26.4s, v5.4s, v1.s[1]\n"
        "mla v25.4s, v5.4s, v1.s[2]\n"
        "mla v24.4s, v5.4s, v1.s[3]\n"
        "fmul v3.4s, v4.4s, v0.s[0]\n"
        "fmul v2.4s, v4.4s, v0.s[1]\n"
        "fmul v1.4s, v4.4s, v0.s[2]\n"
        "scvtf v27.4s, v27.4s\n"
        "fmul v0.4s, v4.4s, v0.s[3]\n"
        "scvtf v26.4s, v26.4s\n"
        "scvtf v25.4s, v25.4s\n"
        "scvtf v24.4s, v24.4s\n"
        "fmul v27.4s, v27.4s, v3.4s\n"
        "fmul v26.4s, v26.4s, v2.4s\n"
        "fmul v25.4s, v25.4s, v1.4s\n"
        "fmul v24.4s, v24.4s, v0.4s\n"
        "ld1 { v1.4s }, [x21]\n"
        "add x21, x21, #0x10\n"
        "ldr q0, [x21, #0x0]\n"
        "mla v23.4s, v5.4s, v1.s[0]\n"
        "mla v22.4s, v5.4s, v1.s[1]\n"
        "mla v21.4s, v5.4s, v1.s[2]\n"
        "mla v20.4s, v5.4s, v1.s[3]\n"
        "fmul v3.4s, v4.4s, v0.s[0]\n"
        "fmul v2.4s, v4.4s, v0.s[1]\n"
        "fmul v1.4s, v4.4s, v0.s[2]\n"
        "scvtf v23.4s, v23.4s\n"
        "fmul v0.4s, v4.4s, v0.s[3]\n"
        "scvtf v22.4s, v22.4s\n"
        "scvtf v21.4s, v21.4s\n"
        "scvtf v20.4s, v20.4s\n"
        "fmul v23.4s, v23.4s, v3.4s\n"
        "fmul v22.4s, v22.4s, v2.4s\n"
        "fmul v21.4s, v21.4s, v1.4s\n"
        "fmul v20.4s, v20.4s, v0.4s\n"
        "ld1 { v1.4s }, [x20]\n"
        "add x20, x20, #0x10\n"
        "ldr q0, [x20, #0x0]\n"
        "mla v19.4s, v5.4s, v1.s[0]\n"
        "mla v18.4s, v5.4s, v1.s[1]\n"
        "mla v17.4s, v5.4s, v1.s[2]\n"
        "mla v16.4s, v5.4s, v1.s[3]\n"
        "fmul v3.4s, v4.4s, v0.s[0]\n"
        "fmul v2.4s, v4.4s, v0.s[1]\n"
        "fmul v1.4s, v4.4s, v0.s[2]\n"
        "scvtf v19.4s, v19.4s\n"
        "fmul v0.4s, v4.4s, v0.s[3]\n"
        "scvtf v18.4s, v18.4s\n"
        "scvtf v17.4s, v17.4s\n"
        "scvtf v16.4s, v16.4s\n"
        "fmul v19.4s, v19.4s, v3.4s\n"
        "fmul v18.4s, v18.4s, v2.4s\n"
        "fmul v17.4s, v17.4s, v1.4s\n"
        "fmul v16.4s, v16.4s, v0.4s\n"
        "ldr q2, [x11, #0x0]\n"
        "ld1r { v1.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x10, #0x4\n"
        "ld1r { v0.4s }, [x20]\n"
        "add x11, x11, #0x10\n"
        "fadd v31.4s, v31.4s, v2.4s\n"
        "fadd v30.4s, v30.4s, v2.4s\n"
        "fadd v29.4s, v29.4s, v2.4s\n"
        "fadd v28.4s, v28.4s, v2.4s\n"
        "fadd v27.4s, v27.4s, v2.4s\n"
        "fadd v26.4s, v26.4s, v2.4s\n"
        "fadd v25.4s, v25.4s, v2.4s\n"
        "fadd v24.4s, v24.4s, v2.4s\n"
        "fadd v23.4s, v23.4s, v2.4s\n"
        "fadd v22.4s, v22.4s, v2.4s\n"
        "fadd v21.4s, v21.4s, v2.4s\n"
        "fadd v20.4s, v20.4s, v2.4s\n"
        "fadd v19.4s, v19.4s, v2.4s\n"
        "fadd v18.4s, v18.4s, v2.4s\n"
        "fadd v17.4s, v17.4s, v2.4s\n"
        "fadd v16.4s, v16.4s, v2.4s\n"
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
        "mov x27, %x[lhs_packed]\n"
        "movi v31.4s, #0x0\n"
        "movi v30.4s, #0x0\n"
        "mov x20, %x[num_blocks]\n"
        "movi v29.4s, #0x0\n"
        "movi v28.4s, #0x0\n"
        "17:"  // Row tail: Sub block loop
        "ldr q17, [x26, #0x0]\n"
        "ldr q16, [x27, #0x0]\n"
        "subs x20, x20, #0x1\n"
        "ldr q1, [x26, #0x10]\n"
        "ldr q0, [x27, #0x10]\n"
        "ldr q27, [x26, #0x20]\n"
        "ldr q26, [x27, #0x20]\n"
        "ldr q25, [x26, #0x30]\n"
        "ldr q24, [x27, #0x30]\n"
        ".inst 0x4f90e23f  // sdot v31.4s, v17.16b, v16.4b[0]\n"
        ".inst 0x4fb0e23e  // sdot v30.4s, v17.16b, v16.4b[1]\n"
        "ldr q23, [x26, #0x40]\n"
        "ldr q22, [x27, #0x40]\n"
        ".inst 0x4f90ea3d  // sdot v29.4s, v17.16b, v16.4b[2]\n"
        ".inst 0x4fb0ea3c  // sdot v28.4s, v17.16b, v16.4b[3]\n"
        "ldr q21, [x26, #0x50]\n"
        "ldr q20, [x27, #0x50]\n"
        "ldr q19, [x26, #0x60]\n"
        "ldr q18, [x27, #0x60]\n"
        "ldr q17, [x26, #0x70]\n"
        "ldr q16, [x27, #0x70]\n"
        ".inst 0x4f80e03f  // sdot v31.4s, v1.16b, v0.4b[0]\n"
        ".inst 0x4fa0e03e  // sdot v30.4s, v1.16b, v0.4b[1]\n"
        ".inst 0x4f80e83d  // sdot v29.4s, v1.16b, v0.4b[2]\n"
        ".inst 0x4fa0e83c  // sdot v28.4s, v1.16b, v0.4b[3]\n"
        "add x27, x27, #0x80\n"
        "add x26, x26, #0x80\n"
        ".inst 0x4f9ae37f  // sdot v31.4s, v27.16b, v26.4b[0]\n"
        ".inst 0x4fbae37e  // sdot v30.4s, v27.16b, v26.4b[1]\n"
        ".inst 0x4f9aeb7d  // sdot v29.4s, v27.16b, v26.4b[2]\n"
        ".inst 0x4fbaeb7c  // sdot v28.4s, v27.16b, v26.4b[3]\n"
        ".inst 0x4f98e33f  // sdot v31.4s, v25.16b, v24.4b[0]\n"
        ".inst 0x4fb8e33e  // sdot v30.4s, v25.16b, v24.4b[1]\n"
        ".inst 0x4f98eb3d  // sdot v29.4s, v25.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb3c  // sdot v28.4s, v25.16b, v24.4b[3]\n"
        ".inst 0x4f96e2ff  // sdot v31.4s, v23.16b, v22.4b[0]\n"
        ".inst 0x4fb6e2fe  // sdot v30.4s, v23.16b, v22.4b[1]\n"
        ".inst 0x4f96eafd  // sdot v29.4s, v23.16b, v22.4b[2]\n"
        ".inst 0x4fb6eafc  // sdot v28.4s, v23.16b, v22.4b[3]\n"
        ".inst 0x4f94e2bf  // sdot v31.4s, v21.16b, v20.4b[0]\n"
        ".inst 0x4fb4e2be  // sdot v30.4s, v21.16b, v20.4b[1]\n"
        ".inst 0x4f94eabd  // sdot v29.4s, v21.16b, v20.4b[2]\n"
        ".inst 0x4fb4eabc  // sdot v28.4s, v21.16b, v20.4b[3]\n"
        ".inst 0x4f92e27f  // sdot v31.4s, v19.16b, v18.4b[0]\n"
        ".inst 0x4fb2e27e  // sdot v30.4s, v19.16b, v18.4b[1]\n"
        ".inst 0x4f92ea7d  // sdot v29.4s, v19.16b, v18.4b[2]\n"
        ".inst 0x4fb2ea7c  // sdot v28.4s, v19.16b, v18.4b[3]\n"
        ".inst 0x4f90e23f  // sdot v31.4s, v17.16b, v16.4b[0]\n"
        ".inst 0x4fb0e23e  // sdot v30.4s, v17.16b, v16.4b[1]\n"
        ".inst 0x4f90ea3d  // sdot v29.4s, v17.16b, v16.4b[2]\n"
        ".inst 0x4fb0ea3c  // sdot v28.4s, v17.16b, v16.4b[3]\n"
        "bgt 17b\n"
        "ldr q18, [x26, #0x0]\n"
        "ld1 { v17.4s }, [x27]\n"
        "add x27, x27, #0x10\n"
        "ldr q20, [x26, #0x10]\n"
        "ldr q16, [x27, #0x0]\n"
        "add x26, x26, #0x20\n"
        "mla v31.4s, v18.4s, v17.s[0]\n"
        "mla v30.4s, v18.4s, v17.s[1]\n"
        "mla v29.4s, v18.4s, v17.s[2]\n"
        "mla v28.4s, v18.4s, v17.s[3]\n"
        "fmul v19.4s, v20.4s, v16.s[0]\n"
        "fmul v18.4s, v20.4s, v16.s[1]\n"
        "fmul v17.4s, v20.4s, v16.s[2]\n"
        "scvtf v31.4s, v31.4s\n"
        "fmul v16.4s, v20.4s, v16.s[3]\n"
        "scvtf v30.4s, v30.4s\n"
        "scvtf v29.4s, v29.4s\n"
        "scvtf v28.4s, v28.4s\n"
        "fmul v31.4s, v31.4s, v19.4s\n"
        "fmul v30.4s, v30.4s, v18.4s\n"
        "fmul v29.4s, v29.4s, v17.4s\n"
        "fmul v28.4s, v28.4s, v16.4s\n"
        "ldr q18, [x26, #0x0]\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x25, #0x4\n"
        "ld1r { v16.4s }, [x20]\n"
        "add x26, x26, #0x10\n"
        "fadd v31.4s, v31.4s, v18.4s\n"
        "fadd v30.4s, v30.4s, v18.4s\n"
        "fadd v29.4s, v29.4s, v18.4s\n"
        "fadd v28.4s, v28.4s, v18.4s\n"
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
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
          "x28");
}

#endif  // Architectural features check.
