//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_DOTPROD) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error "Dotprod extension and fp16 vector arithmetic required to compile this micro-kernel"
#else  // Architectural features check.

#include "kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    uint16_t* dst;
    const void* lhs_packed;
    const void* rhs_packed;
    const float* clamp_vals;
    size_t dst_stride_row;
    size_t m;
    size_t n;
    size_t num_blocks;
} KernelArgs;

void kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(KernelArgs* args_ptr);

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
static const size_t kai_num_bytes_dst_value = 2;
// Extra args
static const size_t kai_num_bytes_bias = 4;
static const size_t kai_k_multiple_of = 32;

inline static size_t kai_get_k_roundedup(size_t k) {
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_get_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_get_k_roundedup(k);
    size_t lhs_packed_stride = kai_mr * ((k_internal * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs);
    // Since the LHS matrix is asymmetric with per-row quantization, we must include the
    // the number of bytes to hold the zero point value
    lhs_packed_stride += kai_mr * kai_num_bytes_zp_lhs;

    return lhs_packed_stride;
}

inline static size_t kai_get_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_get_k_roundedup(k);
    size_t rhs_packed_stride = kai_nr * (k_internal * kai_num_bytes_qvalue_rhs);

    rhs_packed_stride += kai_nr * kai_num_bytes_multiplier_rhs;
    // Since the LHS matrix is quantized asymmetric with per-row quantization, we also include
    // the number of bytes for the reduction sum
    rhs_packed_stride += kai_nr * kai_num_bytes_rsum_rhs;
    // Since the bias is packed with the RHS matrix, the stride is adjusted with the number of bytes of the bias
    rhs_packed_stride += kai_nr * kai_num_bytes_bias;

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(size_t n_idx, size_t k) {
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(
    size_t m,                         //
    size_t n,                         //
    size_t k,                         //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    void* restrict dst,               // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row,            //
    size_t dst_stride_col,            //
    float scalar_min,                 //
    float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(uint16_t));

    if (m == 0) {
        return;
    }
    const size_t num_blocks = kai_get_k_roundedup(k) / kai_k_multiple_of;
    const float clamp_vals[2] = {scalar_min, scalar_max};

    KernelArgs args;

    args.dst = dst;
    args.lhs_packed = lhs_packed;
    args.rhs_packed = rhs_packed;
    args.clamp_vals = clamp_vals;
    args.dst_stride_row = dst_stride_row;
    args.m = m;
    args.n = n;
    args.num_blocks = num_blocks;

    kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(&args);
}

#endif  // Architectural features check.
