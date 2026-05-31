//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) && !defined(_M_ARM64))
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    float maxval;
    float minval;
    unsigned int num_strings;
    const unsigned int* string_lengths;
    size_t N;
    const void* B_ptr;
    size_t output_offset;
    size_t input_initial_col;
    size_t input_offset;
    void* output_ptr;
    const void* bias;
} kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_impl_args_t;

extern void kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_impl(
    const void* input_ptr, size_t m, kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_impl_args_t* args_ptr,
    unsigned long flags);

static const size_t kai_mr = 6;
static const size_t kai_nr = 8;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(void) {
    return kai_mr;
}

size_t kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(void) {
    return kai_nr;
}

size_t kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(void) {
    return kai_sr;
}

size_t kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(size_t m_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);

    return m_idx * stride;
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_nr == 0);

    return n_idx / kai_nr * (kai_nr * sizeof(float) + kai_nr * k * sizeof(float));
}

size_t kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
    size_t m_idx, size_t n_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);
    KAI_ASSUME(n_idx % kai_nr == 0);

    return m_idx * stride + n_idx * sizeof(float);
}

size_t kai_get_dst_size_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
    size_t m, size_t n, size_t k,                             //
    const void* lhs, size_t lhs_stride,                       //
    const void* rhs_packed,                                   //
    void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
    float clamp_min, float clamp_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_impl_args_t ka;

    unsigned long flags = 0;

    unsigned int string_length = k;
    ka.num_strings = 1;
    ka.string_lengths = &string_length;
    ka.N = n;
    ka.B_ptr = rhs_packed;
    ka.bias = NULL;

    // Direct input.
    const void* input_ptr = lhs;
    ka.input_offset = lhs_stride / sizeof(float);
    ka.input_initial_col = 0;

    // Direct output.
    ka.output_ptr = dst;
    ka.output_offset = dst_stride_row / sizeof(float);

    // Clamping output.
    flags |= 0x2;
    ka.maxval = clamp_max;
    ka.minval = clamp_min;

    kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_impl(input_ptr, m, &ka, flags);
}

#endif  // Architectural features check.
