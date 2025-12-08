//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
typedef struct {
    float* dst;                // 0
    const void* lhs_packed;    // 0x8
    const void* rhs_packed;    // 0x10
    size_t dst_stride_row;     // 0x18
    size_t lhs_packed_stride;  // 0x20
    size_t rhs_packed_stride;  // 0x28
    size_t bias;               // 0x30
    size_t m;                  // 0x38
    size_t n;                  // 0x40
    size_t k;                  // 0x48
    size_t bl;                 // 0x50
    const int32_t* lut;        // 0x58
    float min;                 // 0x60
    float max;                 // 0x64
} KernelArgs;

void kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(KernelArgs* args_ptr);
// Compute args
static const size_t kai_m_step = 1;  // Multiple of vector length
static const size_t kai_n_step = 4;  // Multiple of vector length
// Packing args
static const size_t kai_mr = 1;  // Multiple of vector length
static const size_t kai_nr = 4;  // Multiple of vector length
static const size_t kai_kr = 8;
static const size_t kai_sr = 2;
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 4;
static const size_t kai_num_bytes_sum_lhs = 4;
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric), and reduction sum (if LHS is
// asymmetric))
static const size_t kai_num_bytes_recip_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 4;
static const size_t kai_num_bytes_offset_rhs = 4;

// DST format args
static const size_t kai_num_bytes_dst_value = 2;
// Extra args
static const size_t kai_num_bytes_bias = 4;
static const size_t kai_bl = 32;

// Look-up table used for int4->int8 convert
static const int32_t lut[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

inline static size_t kai_get_num_bytes_per_block_lhs(size_t bl) {
    return (bl * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs + kai_num_bytes_sum_lhs;
}

inline static size_t kai_get_num_bytes_per_block_rhs(size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    size_t num_bytes_per_block_rhs =
        (bl / kai_num_bytes_recip_qvalue_rhs) + kai_num_bytes_multiplier_rhs + kai_num_bytes_offset_rhs;
    return num_bytes_per_block_rhs;
}

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);

    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_get_lhs_packed_stride(size_t k, size_t bl) {
    const size_t mr = kai_get_mr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
    return mr * kai_get_num_blocks_per_row(k, bl) * kai_get_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_get_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME(bl % kai_bl == 0);
    KAI_ASSUME((k % kai_bl) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);
    const size_t nr = kai_get_nr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();

    size_t rhs_packed_stride = nr * (num_bytes_per_block * num_blocks_per_row);
    // Since the bias is packed with the RHS matrix, the stride is adjusted with the number of bytes of the bias
    rhs_packed_stride += nr * kai_num_bytes_bias;

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(void) {
    return kai_m_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(void) {
    return kai_n_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(
    size_t m_idx, size_t k, size_t bl) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
    const size_t mr = kai_get_mr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();

    KAI_ASSUME((m_idx % m_step) == 0);

    return (m_idx / mr) * kai_get_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME((k % bl) == 0);
    const size_t n_step = kai_get_n_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
    const size_t nr = kai_get_nr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();

    KAI_ASSUME((n_idx % n_step) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
    const size_t n_step = kai_get_n_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
    KAI_ASSUME((m_idx % m_step) == 0);
    KAI_ASSUME((n_idx % n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(
    size_t m,                         //
    size_t n,                         //
    size_t k,                         //
    size_t bl,                        //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    void* restrict dst,               // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row,            //
    size_t dst_stride_col,            //
    float scalar_min,                 //
    float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(uint16_t));
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_bl) == 0);

    if (m == 0) {
        return;
    }

    const size_t nr = kai_get_nr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();

    KernelArgs args;
    args.dst = dst;
    args.lhs_packed = lhs_packed;
    args.rhs_packed = rhs_packed;
    args.dst_stride_row = dst_stride_row;
    args.lhs_packed_stride = kai_get_lhs_packed_stride(k, bl);
    args.rhs_packed_stride = kai_get_rhs_packed_stride(k, bl);
    args.bias = args.rhs_packed_stride - nr * kai_num_bytes_bias;
    args.m = m;
    args.n = n;
    args.k = k;
    args.bl = bl;
    args.lut = lut;
    args.min = scalar_min;
    args.max = scalar_max;

    kai_commit_za();

    kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(&args);
}

#endif  // Architectural features check.
