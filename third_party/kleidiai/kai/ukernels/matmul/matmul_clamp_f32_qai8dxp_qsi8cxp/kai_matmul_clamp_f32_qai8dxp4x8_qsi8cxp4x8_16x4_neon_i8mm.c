//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) && !defined(__ARM_FEATURE_MATMUL_INT8)
#error "I8mm extension required to compile this micro-kernel"
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"

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

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(size_t n_idx, size_t k) {
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
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
        "ldr q2, [x11, #0x0]\n"
        "ldr q1, [x11, #0x10]\n"
        "subs x23, x23, #0x1\n"
        "ldr q5, [x27, #0x0]\n"
        "ldr q9, [x27, #0x10]\n"
        "ldr q8, [x22, #0x0]\n"
        "ldr q7, [x22, #0x10]\n"
        "ldr q4, [x21, #0x0]\n"
        "ldr q14, [x21, #0x10]\n"
        "ldr q3, [x20, #0x0]\n"
        "ldr q0, [x20, #0x10]\n"
        ".inst 0x4e82a4bf  // smmla v31.4s, v5.16b, v2.16b\n"
        ".inst 0x4e81a4be  // smmla v30.4s, v5.16b, v1.16b\n"
        "ldr q6, [x11, #0x20]\n"
        "ldr q5, [x11, #0x30]\n"
        ".inst 0x4e82a53d  // smmla v29.4s, v9.16b, v2.16b\n"
        ".inst 0x4e81a53c  // smmla v28.4s, v9.16b, v1.16b\n"
        "ldr q13, [x27, #0x20]\n"
        "ldr q12, [x27, #0x30]\n"
        ".inst 0x4e82a51b  // smmla v27.4s, v8.16b, v2.16b\n"
        ".inst 0x4e81a51a  // smmla v26.4s, v8.16b, v1.16b\n"
        "ldr q11, [x22, #0x20]\n"
        "ldr q10, [x22, #0x30]\n"
        ".inst 0x4e82a4f9  // smmla v25.4s, v7.16b, v2.16b\n"
        ".inst 0x4e81a4f8  // smmla v24.4s, v7.16b, v1.16b\n"
        "ldr q9, [x21, #0x20]\n"
        "ldr q8, [x21, #0x30]\n"
        ".inst 0x4e82a497  // smmla v23.4s, v4.16b, v2.16b\n"
        ".inst 0x4e81a496  // smmla v22.4s, v4.16b, v1.16b\n"
        "ldr q7, [x20, #0x20]\n"
        "ldr q4, [x20, #0x30]\n"
        ".inst 0x4e82a5d5  // smmla v21.4s, v14.16b, v2.16b\n"
        ".inst 0x4e81a5d4  // smmla v20.4s, v14.16b, v1.16b\n"
        "ldr q15, [x11, #0x40]\n"
        "ldr q14, [x11, #0x50]\n"
        ".inst 0x4e82a473  // smmla v19.4s, v3.16b, v2.16b\n"
        ".inst 0x4e81a472  // smmla v18.4s, v3.16b, v1.16b\n"
        "ldr q3, [x27, #0x40]\n"
        ".inst 0x4e82a411  // smmla v17.4s, v0.16b, v2.16b\n"
        "ldr q2, [x27, #0x50]\n"
        ".inst 0x4e81a410  // smmla v16.4s, v0.16b, v1.16b\n"
        "ldr q1, [x22, #0x40]\n"
        "ldr q0, [x22, #0x50]\n"
        ".inst 0x4e86a5bf  // smmla v31.4s, v13.16b, v6.16b\n"
        ".inst 0x4e85a5be  // smmla v30.4s, v13.16b, v5.16b\n"
        "ldr q13, [x21, #0x40]\n"
        ".inst 0x4e86a59d  // smmla v29.4s, v12.16b, v6.16b\n"
        ".inst 0x4e85a59c  // smmla v28.4s, v12.16b, v5.16b\n"
        "ldr q12, [x21, #0x50]\n"
        ".inst 0x4e86a57b  // smmla v27.4s, v11.16b, v6.16b\n"
        ".inst 0x4e85a57a  // smmla v26.4s, v11.16b, v5.16b\n"
        "ldr q11, [x20, #0x40]\n"
        ".inst 0x4e86a559  // smmla v25.4s, v10.16b, v6.16b\n"
        ".inst 0x4e85a558  // smmla v24.4s, v10.16b, v5.16b\n"
        "ldr q10, [x20, #0x50]\n"
        ".inst 0x4e86a537  // smmla v23.4s, v9.16b, v6.16b\n"
        ".inst 0x4e85a536  // smmla v22.4s, v9.16b, v5.16b\n"
        "ldr q9, [x11, #0x60]\n"
        ".inst 0x4e86a515  // smmla v21.4s, v8.16b, v6.16b\n"
        ".inst 0x4e85a514  // smmla v20.4s, v8.16b, v5.16b\n"
        "ldr q8, [x11, #0x70]\n"
        "add x11, x11, #0x80\n"
        ".inst 0x4e86a4f3  // smmla v19.4s, v7.16b, v6.16b\n"
        ".inst 0x4e85a4f2  // smmla v18.4s, v7.16b, v5.16b\n"
        "ldr q7, [x27, #0x60]\n"
        ".inst 0x4e86a491  // smmla v17.4s, v4.16b, v6.16b\n"
        "ldr q6, [x27, #0x70]\n"
        ".inst 0x4e85a490  // smmla v16.4s, v4.16b, v5.16b\n"
        "ldr q5, [x22, #0x60]\n"
        "ldr q4, [x22, #0x70]\n"
        ".inst 0x4e8fa47f  // smmla v31.4s, v3.16b, v15.16b\n"
        ".inst 0x4e8ea47e  // smmla v30.4s, v3.16b, v14.16b\n"
        "ldr q3, [x21, #0x60]\n"
        ".inst 0x4e8fa45d  // smmla v29.4s, v2.16b, v15.16b\n"
        ".inst 0x4e8ea45c  // smmla v28.4s, v2.16b, v14.16b\n"
        "ldr q2, [x21, #0x70]\n"
        "add x27, x27, #0x80\n"
        ".inst 0x4e8fa43b  // smmla v27.4s, v1.16b, v15.16b\n"
        ".inst 0x4e8ea43a  // smmla v26.4s, v1.16b, v14.16b\n"
        "ldr q1, [x20, #0x60]\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4e8fa419  // smmla v25.4s, v0.16b, v15.16b\n"
        ".inst 0x4e8ea418  // smmla v24.4s, v0.16b, v14.16b\n"
        "ldr q0, [x20, #0x70]\n"
        "add x21, x21, #0x80\n"
        ".inst 0x4e8fa5b7  // smmla v23.4s, v13.16b, v15.16b\n"
        ".inst 0x4e8ea5b6  // smmla v22.4s, v13.16b, v14.16b\n"
        "add x20, x20, #0x80\n"
        ".inst 0x4e8fa595  // smmla v21.4s, v12.16b, v15.16b\n"
        ".inst 0x4e8ea594  // smmla v20.4s, v12.16b, v14.16b\n"
        ".inst 0x4e8fa573  // smmla v19.4s, v11.16b, v15.16b\n"
        ".inst 0x4e8ea572  // smmla v18.4s, v11.16b, v14.16b\n"
        ".inst 0x4e8fa551  // smmla v17.4s, v10.16b, v15.16b\n"
        ".inst 0x4e8ea550  // smmla v16.4s, v10.16b, v14.16b\n"
        ".inst 0x4e89a4ff  // smmla v31.4s, v7.16b, v9.16b\n"
        ".inst 0x4e88a4fe  // smmla v30.4s, v7.16b, v8.16b\n"
        ".inst 0x4e89a4dd  // smmla v29.4s, v6.16b, v9.16b\n"
        ".inst 0x4e88a4dc  // smmla v28.4s, v6.16b, v8.16b\n"
        ".inst 0x4e89a4bb  // smmla v27.4s, v5.16b, v9.16b\n"
        ".inst 0x4e88a4ba  // smmla v26.4s, v5.16b, v8.16b\n"
        ".inst 0x4e89a499  // smmla v25.4s, v4.16b, v9.16b\n"
        ".inst 0x4e88a498  // smmla v24.4s, v4.16b, v8.16b\n"
        ".inst 0x4e89a477  // smmla v23.4s, v3.16b, v9.16b\n"
        ".inst 0x4e88a476  // smmla v22.4s, v3.16b, v8.16b\n"
        ".inst 0x4e89a455  // smmla v21.4s, v2.16b, v9.16b\n"
        ".inst 0x4e88a454  // smmla v20.4s, v2.16b, v8.16b\n"
        ".inst 0x4e89a433  // smmla v19.4s, v1.16b, v9.16b\n"
        ".inst 0x4e88a432  // smmla v18.4s, v1.16b, v8.16b\n"
        ".inst 0x4e89a411  // smmla v17.4s, v0.16b, v9.16b\n"
        ".inst 0x4e88a410  // smmla v16.4s, v0.16b, v8.16b\n"
        "bgt 3b\n"
        "ldr q7, [x11, #0x0]\n"
        "ld1 { v4.4s }, [x27]\n"
        "uzp1 v3.2d, v31.2d, v30.2d\n"
        "uzp2 v2.2d, v31.2d, v30.2d\n"
        "ldr q6, [x11, #0x10]\n"
        "uzp1 v1.2d, v29.2d, v28.2d\n"
        "uzp2 v0.2d, v29.2d, v28.2d\n"
        "add x27, x27, #0x10\n"
        "ldr q28, [x27, #0x0]\n"
        "add x11, x11, #0x20\n"
        "mla v3.4s, v7.4s, v4.s[0]\n"
        "mla v2.4s, v7.4s, v4.s[1]\n"
        "mla v1.4s, v7.4s, v4.s[2]\n"
        "mla v0.4s, v7.4s, v4.s[3]\n"
        "fmul v31.4s, v6.4s, v28.s[0]\n"
        "fmul v30.4s, v6.4s, v28.s[1]\n"
        "fmul v29.4s, v6.4s, v28.s[2]\n"
        "fmul v28.4s, v6.4s, v28.s[3]\n"
        "scvtf v3.4s, v3.4s\n"
        "scvtf v2.4s, v2.4s\n"
        "scvtf v1.4s, v1.4s\n"
        "scvtf v0.4s, v0.4s\n"
        "fmul v31.4s, v3.4s, v31.4s\n"
        "fmul v30.4s, v2.4s, v30.4s\n"
        "fmul v29.4s, v1.4s, v29.4s\n"
        "fmul v28.4s, v0.4s, v28.4s\n"
        "ld1 { v5.4s }, [x22]\n"
        "uzp1 v4.2d, v27.2d, v26.2d\n"
        "uzp2 v3.2d, v27.2d, v26.2d\n"
        "add x22, x22, #0x10\n"
        "ldr q2, [x22, #0x0]\n"
        "uzp1 v1.2d, v25.2d, v24.2d\n"
        "uzp2 v0.2d, v25.2d, v24.2d\n"
        "mla v4.4s, v7.4s, v5.s[0]\n"
        "mla v3.4s, v7.4s, v5.s[1]\n"
        "mla v1.4s, v7.4s, v5.s[2]\n"
        "mla v0.4s, v7.4s, v5.s[3]\n"
        "fmul v27.4s, v6.4s, v2.s[0]\n"
        "fmul v26.4s, v6.4s, v2.s[1]\n"
        "fmul v25.4s, v6.4s, v2.s[2]\n"
        "scvtf v4.4s, v4.4s\n"
        "fmul v24.4s, v6.4s, v2.s[3]\n"
        "scvtf v3.4s, v3.4s\n"
        "scvtf v1.4s, v1.4s\n"
        "scvtf v0.4s, v0.4s\n"
        "fmul v27.4s, v4.4s, v27.4s\n"
        "fmul v26.4s, v3.4s, v26.4s\n"
        "fmul v25.4s, v1.4s, v25.4s\n"
        "fmul v24.4s, v0.4s, v24.4s\n"
        "ld1 { v5.4s }, [x21]\n"
        "uzp1 v4.2d, v23.2d, v22.2d\n"
        "uzp2 v3.2d, v23.2d, v22.2d\n"
        "add x21, x21, #0x10\n"
        "ldr q2, [x21, #0x0]\n"
        "uzp1 v1.2d, v21.2d, v20.2d\n"
        "uzp2 v0.2d, v21.2d, v20.2d\n"
        "mla v4.4s, v7.4s, v5.s[0]\n"
        "mla v3.4s, v7.4s, v5.s[1]\n"
        "mla v1.4s, v7.4s, v5.s[2]\n"
        "mla v0.4s, v7.4s, v5.s[3]\n"
        "fmul v23.4s, v6.4s, v2.s[0]\n"
        "fmul v22.4s, v6.4s, v2.s[1]\n"
        "fmul v21.4s, v6.4s, v2.s[2]\n"
        "scvtf v4.4s, v4.4s\n"
        "fmul v20.4s, v6.4s, v2.s[3]\n"
        "scvtf v3.4s, v3.4s\n"
        "scvtf v1.4s, v1.4s\n"
        "scvtf v0.4s, v0.4s\n"
        "fmul v23.4s, v4.4s, v23.4s\n"
        "fmul v22.4s, v3.4s, v22.4s\n"
        "fmul v21.4s, v1.4s, v21.4s\n"
        "fmul v20.4s, v0.4s, v20.4s\n"
        "ld1 { v5.4s }, [x20]\n"
        "uzp1 v4.2d, v19.2d, v18.2d\n"
        "uzp2 v3.2d, v19.2d, v18.2d\n"
        "add x20, x20, #0x10\n"
        "ldr q2, [x20, #0x0]\n"
        "uzp1 v1.2d, v17.2d, v16.2d\n"
        "uzp2 v0.2d, v17.2d, v16.2d\n"
        "mla v4.4s, v7.4s, v5.s[0]\n"
        "mla v3.4s, v7.4s, v5.s[1]\n"
        "mla v1.4s, v7.4s, v5.s[2]\n"
        "mla v0.4s, v7.4s, v5.s[3]\n"
        "fmul v19.4s, v6.4s, v2.s[0]\n"
        "fmul v18.4s, v6.4s, v2.s[1]\n"
        "fmul v17.4s, v6.4s, v2.s[2]\n"
        "scvtf v4.4s, v4.4s\n"
        "fmul v16.4s, v6.4s, v2.s[3]\n"
        "scvtf v3.4s, v3.4s\n"
        "scvtf v1.4s, v1.4s\n"
        "scvtf v0.4s, v0.4s\n"
        "fmul v19.4s, v4.4s, v19.4s\n"
        "fmul v18.4s, v3.4s, v18.4s\n"
        "fmul v17.4s, v1.4s, v17.4s\n"
        "fmul v16.4s, v0.4s, v16.4s\n"
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
        "ldr q19, [x26, #0x0]\n"
        "ldr q18, [x26, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q17, [x27, #0x0]\n"
        "ldr q16, [x27, #0x10]\n"
        "ldr q27, [x26, #0x20]\n"
        "ldr q26, [x26, #0x30]\n"
        "ldr q25, [x27, #0x20]\n"
        "ldr q24, [x27, #0x30]\n"
        "ldr q23, [x26, #0x40]\n"
        "ldr q22, [x26, #0x50]\n"
        ".inst 0x4e93a63f  // smmla v31.4s, v17.16b, v19.16b\n"
        ".inst 0x4e92a63e  // smmla v30.4s, v17.16b, v18.16b\n"
        "ldr q21, [x27, #0x40]\n"
        "ldr q20, [x27, #0x50]\n"
        ".inst 0x4e93a61d  // smmla v29.4s, v16.16b, v19.16b\n"
        ".inst 0x4e92a61c  // smmla v28.4s, v16.16b, v18.16b\n"
        "ldr q19, [x26, #0x60]\n"
        "ldr q18, [x26, #0x70]\n"
        "add x26, x26, #0x80\n"
        "ldr q17, [x27, #0x60]\n"
        "ldr q16, [x27, #0x70]\n"
        "add x27, x27, #0x80\n"
        ".inst 0x4e9ba73f  // smmla v31.4s, v25.16b, v27.16b\n"
        ".inst 0x4e9aa73e  // smmla v30.4s, v25.16b, v26.16b\n"
        ".inst 0x4e9ba71d  // smmla v29.4s, v24.16b, v27.16b\n"
        ".inst 0x4e9aa71c  // smmla v28.4s, v24.16b, v26.16b\n"
        ".inst 0x4e97a6bf  // smmla v31.4s, v21.16b, v23.16b\n"
        ".inst 0x4e96a6be  // smmla v30.4s, v21.16b, v22.16b\n"
        ".inst 0x4e97a69d  // smmla v29.4s, v20.16b, v23.16b\n"
        ".inst 0x4e96a69c  // smmla v28.4s, v20.16b, v22.16b\n"
        ".inst 0x4e93a63f  // smmla v31.4s, v17.16b, v19.16b\n"
        ".inst 0x4e92a63e  // smmla v30.4s, v17.16b, v18.16b\n"
        ".inst 0x4e93a61d  // smmla v29.4s, v16.16b, v19.16b\n"
        ".inst 0x4e92a61c  // smmla v28.4s, v16.16b, v18.16b\n"
        "bgt 17b\n"
        "ldr q18, [x26, #0x0]\n"
        "ld1 { v17.4s }, [x27]\n"
        "uzp1 v24.2d, v31.2d, v30.2d\n"
        "uzp2 v23.2d, v31.2d, v30.2d\n"
        "ldr q22, [x26, #0x10]\n"
        "uzp1 v21.2d, v29.2d, v28.2d\n"
        "uzp2 v20.2d, v29.2d, v28.2d\n"
        "add x27, x27, #0x10\n"
        "ldr q16, [x27, #0x0]\n"
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
        "fmul v31.4s, v24.4s, v19.4s\n"
        "fmul v30.4s, v23.4s, v18.4s\n"
        "fmul v29.4s, v21.4s, v17.4s\n"
        "fmul v28.4s, v20.4s, v16.4s\n"
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
