//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16.
#else  // Architectural features check.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernels variants of the same type share the same interfaces
// In this case, the micro-kernel type is: matmul_clamp_f16_f16_f16p

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_m_step_func_t)(void);
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_n_step_func_t)(void);
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_nr_func_t)(void);
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_kr_func_t)(void);
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_sr_func_t)(void);
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_lhs_offset_func_t)(size_t m_idx, size_t lhs_stride);
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_rhs_packed_offset_func_t)(size_t n_idx, size_t k);
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_dst_offset_func_t)(size_t m_idx, size_t n_idx, size_t dst_stride);
typedef size_t (*kai_matmul_clamp_f16_f16_f16p_get_dst_size_func_t)(size_t m, size_t n);

/// Micro-kernel core function ("run" method)
typedef void (*kai_matmul_clamp_f16_f16_f16p_run_matmul_func_t)(
    size_t m, size_t n, size_t k, const void* lhs, size_t lhs_stride, const void* rhs_packed, void* dst,
    size_t dst_stride_row, size_t dst_stride_col, __fp16 scalar_min, __fp16 scalar_max);

/// Micro-kernel interface
struct kai_matmul_clamp_f16_f16_f16p_ukernel {
    kai_matmul_clamp_f16_f16_f16p_get_m_step_func_t get_m_step;
    kai_matmul_clamp_f16_f16_f16p_get_n_step_func_t get_n_step;
    kai_matmul_clamp_f16_f16_f16p_get_nr_func_t get_nr;
    kai_matmul_clamp_f16_f16_f16p_get_kr_func_t get_kr;
    kai_matmul_clamp_f16_f16_f16p_get_sr_func_t get_sr;
    kai_matmul_clamp_f16_f16_f16p_get_lhs_offset_func_t get_lhs_packed_offset;
    kai_matmul_clamp_f16_f16_f16p_get_rhs_packed_offset_func_t get_rhs_packed_offset;
    kai_matmul_clamp_f16_f16_f16p_get_dst_offset_func_t get_dst_offset;
    kai_matmul_clamp_f16_f16_f16p_get_dst_size_func_t get_dst_size;
    kai_matmul_clamp_f16_f16_f16p_run_matmul_func_t run_matmul;
};

#ifdef __cplusplus
}
#endif

#endif  // Architectural features check.
