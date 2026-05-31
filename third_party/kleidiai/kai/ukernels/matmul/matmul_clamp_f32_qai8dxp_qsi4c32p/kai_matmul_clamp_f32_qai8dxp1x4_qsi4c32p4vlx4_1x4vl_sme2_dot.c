//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check

#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

#define KAI_LUT_NENTRIES 64

// Lut to be indexed by i4 resulting in its value in i8 (i.e. -2 = 1110 -> 1111 1110).
static const int8_t lut[KAI_LUT_NENTRIES] = {
    // clang-format off
     0, 0, 0, 0,
     1, 0, 0, 0,
     2, 0, 0, 0,
     3, 0, 0, 0,
     4, 0, 0, 0,
     5, 0, 0, 0,
     6, 0, 0, 0,
     7, 0, 0, 0,
    -8, 0, 0, 0,
    -7, 0, 0, 0,
    -6, 0, 0, 0,
    -5, 0, 0, 0,
    -4, 0, 0, 0,
    -3, 0, 0, 0,
    -2, 0, 0, 0,
    -1, 0, 0, 0
    // clang-format on
};

typedef struct {
    float* dst;              // 0   ( 0x00 )
    size_t dst_stride_row;   // 8   ( 0x08 )
    const int8_t* lut;       // 16  ( 0x10 )
    size_t m;                // 24  ( 0x18 )
    size_t n;                // 32  ( 0x20 )
    size_t k;                // 40  ( 0x28 )
    const void* lhs_packed;  // 48  ( 0x30 )
    const void* rhs_packed;  // 56  ( 0x38 )
    float scalar_max;        // 64  ( 0x40 )
    float scalar_min;        // 68  ( 0x44 )
    size_t k_internal;       // 72  ( 0x48 )
    size_t lhs_stride;       // 80  ( 0x50 )
    size_t rhs_stride;       // 88  ( 0x58 )
    size_t nr;               // 96  ( 0x60 )
    size_t rhs_row_bytes;    // 104 ( 0x68 )
    size_t lhs_end_ptr;      // 112 ( 0x70 )
    size_t bl;               // 120 ( 0x78 )
} KernelArgs;

void kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(KernelArgs* args_ptr);

// Compute args
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;  // multiple of vector length
// Packing args
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;  // multiple of vector length
static const size_t kai_kr = 8;
static const size_t kai_sr = 2;
// LHS format args
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
// RHS format args
static const size_t kai_num_bytes_recip_qvalue_rhs = 2;               // int4: 2 values per byte
static const size_t kai_num_bytes_multiplier_rhs = sizeof(uint16_t);  // BF16 scale per column
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);          // rsum per column
static const size_t kai_num_bytes_bias_rhs = sizeof(float);           // bias per column
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_k_multiple_of = 32;
static const size_t kai_bl = 32;

static size_t kai_k_roundedup(const size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

static size_t kai_get_lhs_packed_stride(const size_t k) {
    const size_t k_internal = kai_k_roundedup(k);
    return kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot() *
        (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

static size_t kai_get_num_bytes_per_block_rhs(const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    const size_t num_bytes_per_block_rhs = (bl / kai_num_bytes_recip_qvalue_rhs) + kai_num_bytes_multiplier_rhs;
    return num_bytes_per_block_rhs;
}

static size_t kai_get_num_blocks_per_row(const size_t k, const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    return kai_roundup(k, bl) / bl;
}

static size_t kai_get_rhs_packed_stride(const size_t k, const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    const size_t k_internal = kai_k_roundedup(k);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot();
    const size_t num_blocks_per_row = kai_roundup(k_internal, bl) / bl;

    // bytes_per_block: int4 packed weights (bl/2 bytes) + per-block scale bytes
    const size_t bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);
    size_t rhs_packed_stride = nr * (num_blocks_per_row * bytes_per_block);
    rhs_packed_stride += nr * kai_num_bytes_sum_rhs;   // per-column rsum
    rhs_packed_stride += nr * kai_num_bytes_bias_rhs;  // per-column bias

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(void) {
    return kai_n_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(void) {
    // For gemv mr must be 1 to consecutively read the data
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(
    const size_t m_idx, const size_t k) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    return (m_idx / kai_mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(
    const size_t n_idx, const size_t k, const size_t bl) {
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot();
    KAI_ASSUME((n_idx % n_step) == 0);
    const size_t row = n_idx / n_step;
    return row * kai_get_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(
    const size_t m_idx, const size_t n_idx, const size_t dst_stride) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot();
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot();
    KAI_ASSERT((m_idx % m_step) == 0);
    KAI_ASSERT((n_idx % n_step) == 0);
    return (n_idx * kai_num_bytes_dst_value) + (m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(const size_t m, const size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(
    const size_t m,                   //
    const size_t n,                   //
    const size_t k,                   //
    const size_t bl,                  //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    const size_t dst_stride_row,      //
    const size_t dst_stride_col,      //
    const float scalar_min,           //
    const float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));
    KAI_ASSUME(n > 0);
    KAI_ASSUME(m == 1);
    KAI_ASSUME(k > 0);
    KAI_ASSUME((bl % kai_k_multiple_of) == 0);
    KAI_ASSUME((k % bl) == 0);

    KernelArgs args;

    args.dst = dst;
    args.lhs_packed = lhs_packed;
    args.rhs_packed = rhs_packed;
    args.dst_stride_row = dst_stride_row;
    args.m = m;
    args.n = n;
    args.k = k;
    args.bl = bl;
    args.k_internal = kai_k_roundedup(k);
    args.lhs_stride = kai_get_lhs_packed_stride(k);
    args.rhs_stride = kai_get_rhs_packed_stride(k, bl);
    args.nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot();
    const size_t bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);
    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    args.rhs_row_bytes = args.nr * num_blocks_per_row * bytes_per_block;
    args.lhs_end_ptr = ((uint64_t)lhs_packed) + (m * args.lhs_stride);
    args.scalar_max = scalar_max;
    args.scalar_min = scalar_min;
    args.lut = lut;

    kai_commit_za();

    kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot(&args);
}
#endif  // Architectural features check.
