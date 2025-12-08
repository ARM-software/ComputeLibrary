//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    uint16_t maxval;
    uint16_t minval;
    const void* A_ptr;
    const void* B_ptr;
    size_t N;
    size_t K;
    void* output_ptr;
    uint64_t flags;
} KernelArgs;

static const size_t kai_m_step = 1;
static const size_t kai_nr = 2;
static const size_t kai_n_step = 8;
static const size_t kai_kr = 2;
static const size_t kai_sr = 1;

void kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(KernelArgs* args_ptr);

uint16_t kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(float value);

size_t kai_get_m_step_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(void) {
    return kai_n_step * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_nr_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(void) {
    return kai_nr * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_kr_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(void) {
    return kai_sr;
}

size_t kai_get_lhs_offset_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx == 0);

    return m_idx * k;
}

static size_t kai_get_rhs_packed_stride_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(size_t k) {
    return kai_get_n_step_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla() *
        (kai_roundup(k, kai_kr) * sizeof(uint16_t) + sizeof(uint16_t));
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla();
    return block_idx * kai_get_rhs_packed_stride_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(k);
}

size_t kai_get_dst_offset_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME(m_idx == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla() == 0);

    return (m_idx * dst_stride) + (n_idx * sizeof(uint16_t));
}

size_t kai_get_dst_size_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(size_t m, size_t n) {
    return m * n * sizeof(uint16_t);
}

void kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(
    size_t m, size_t n, size_t k, const void* lhs, size_t lhs_stride, const void* rhs_packed, void* dst,
    size_t dst_stride_row, size_t dst_stride_col, float clamp_min, float clamp_max) {
    KAI_UNUSED(dst_stride_row);
    KAI_UNUSED(dst_stride_col);
    KAI_UNUSED(lhs_stride);
    KAI_ASSUME(m == 1);

    uint64_t flags = 2;

    KernelArgs args;

    args.minval = kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(clamp_min);
    args.maxval = kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(clamp_max);
    args.A_ptr = lhs;
    args.B_ptr = rhs_packed;
    args.N = n;
    args.K = k;
    args.output_ptr = dst;
    args.flags = flags;

    kai_commit_za();

    kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(&args);
}

#endif  // Architectural features check.
