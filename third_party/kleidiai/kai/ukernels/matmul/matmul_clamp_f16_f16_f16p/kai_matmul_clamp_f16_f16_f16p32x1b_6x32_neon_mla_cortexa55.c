//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (                                                                          \
    !defined(__aarch64__) && !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)) &&                         \
    !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_FP16.
#else  // Architectural features check.

#include "kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    uint16_t maxval;
    uint16_t minval;
    unsigned int num_strings;
    const unsigned int* string_lengths;
    size_t N;
    const void* B_ptr;
    size_t output_offset;
    size_t input_initial_col;
    size_t input_offset;
    void* output_ptr;
    const void* bias;
} KernelArgs;

void kai_kernel_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(
    const void* input_ptr, size_t m, KernelArgs* args_ptr, unsigned long flags);
uint16_t kai_f16_from_float_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(float value);

static const size_t kai_mr = 6;
static const size_t kai_nr = 32;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(void) {
    return kai_mr;
}

size_t kai_get_n_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(void) {
    return kai_nr;
}

size_t kai_get_nr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(void) {
    return kai_sr;
}

size_t kai_get_lhs_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(size_t m_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55() == 0);

    return m_idx * stride;
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55() == 0);

    return n_idx / kai_nr * (kai_nr * sizeof(uint16_t) + kai_nr * k * sizeof(uint16_t));
}

size_t kai_get_dst_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(
    size_t m_idx, size_t n_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55() == 0);

    return m_idx * stride + n_idx * sizeof(uint16_t);
}

size_t kai_get_dst_size_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(size_t m, size_t n) {
    return m * n * sizeof(uint16_t);
}

void kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(
    size_t m, size_t n, size_t k,                             //
    const void* lhs, size_t lhs_stride,                       //
    const void* rhs_packed,                                   //
    void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
    float clamp_min, float clamp_max) {
    KAI_UNUSED(dst_stride_col);

    KernelArgs ka;

    unsigned long flags = 0;

    unsigned int string_length = k;
    ka.num_strings = 1;
    ka.string_lengths = &string_length;
    ka.N = n;
    ka.B_ptr = rhs_packed;
    ka.bias = NULL;

    // Direct input.
    const void* input_ptr = lhs;
    ka.input_offset = lhs_stride / sizeof(uint16_t);
    ka.input_initial_col = 0;

    // Direct output.
    ka.output_ptr = dst;
    ka.output_offset = dst_stride_row / sizeof(uint16_t);

    // Clamping output.
    flags |= 0x2;
    ka.maxval = kai_f16_from_float_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(clamp_max);
    ka.minval = kai_f16_from_float_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(clamp_min);

    kai_kernel_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(input_ptr, m, &ka, flags);
}

#endif  // Architectural features check.
