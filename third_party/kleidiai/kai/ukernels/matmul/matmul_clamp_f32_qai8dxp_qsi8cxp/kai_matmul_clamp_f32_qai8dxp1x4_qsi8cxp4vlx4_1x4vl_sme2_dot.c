//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error "This file must be compiled for AArch64, FEAT_SVE2"
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"

#include <stddef.h>

#include "kai/kai_common.h"

typedef struct {
    float* dst;              // 0
    const void* lhs_packed;  // 0x8
    const void* rhs_packed;  // 0x10
    size_t dst_stride_row;   // 0x18
    size_t m;                // 0x20
    size_t n;                // 0x28
    size_t k;                // 0x30
    size_t k_internal;       // 0x38
    size_t lhs_stride;       // 0x40
    size_t rhs_stride;       // 0x48
    size_t rhs_row_bytes;    // 0x50
    size_t lhs_end;          // 0x58
    float clamp_min;         // 0x60
    float clamp_max;         // 0x64
} KernelArgs;

void kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(KernelArgs* args_ptr);

// Compute args
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;  // multiple of vector length
// Packing args
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;  // multiple of vector length
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
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_get_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);
    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);
    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot();
    size_t lhs_packed_stride = mr * ((k_internal * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs);
    // Since the LHS matrix is asymmetric with per-row quantization, we must include the
    // the number of bytes to hold the zero point value
    lhs_packed_stride += mr * kai_num_bytes_zp_lhs;

    return lhs_packed_stride;
}

inline static size_t kai_get_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);
    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot();

    size_t rhs_packed_stride = nr * (k_internal * kai_num_bytes_qvalue_rhs);
    rhs_packed_stride += nr * kai_num_bytes_multiplier_rhs;
    // Since the LHS matrix is quantized asymmetric with per-row quantization, we also include
    // the number of bytes for the reduction sum
    rhs_packed_stride += nr * kai_num_bytes_rsum_rhs;
    // Since the bias is packed with the RHS matrix, the stride is adjusted with the number of bytes of the bias
    rhs_packed_stride += nr * kai_num_bytes_bias;

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_n_step * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_nr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot()) == 0);

    return (m_idx / kai_mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(size_t n_idx, size_t k) {
    KAI_ASSUME((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot()) == 0);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot();
    return (n_idx / nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot()) == 0);
    KAI_ASSUME((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot()) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(
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
    const size_t lhs_stride = kai_get_lhs_packed_stride(k);
    const size_t rhs_stride = kai_get_rhs_packed_stride(k);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot();

    const size_t rhs_row_bytes = nr * k_internal;
    const size_t lhs_end_ptr = ((size_t)lhs_packed) + (m * lhs_stride);

    KernelArgs args;

    args.dst = dst;
    args.lhs_packed = lhs_packed;
    args.rhs_packed = rhs_packed;
    args.clamp_max = scalar_max;
    args.clamp_min = scalar_min;
    args.dst_stride_row = dst_stride_row;
    args.m = m;
    args.n = n;
    args.k = k;
    args.k_internal = k_internal;
    args.lhs_stride = lhs_stride;
    args.rhs_stride = rhs_stride;
    args.rhs_row_bytes = rhs_row_bytes;
    args.lhs_end = lhs_end_ptr;

    kai_commit_za();

    kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(&args);
}

#endif  // Architectural features check.
