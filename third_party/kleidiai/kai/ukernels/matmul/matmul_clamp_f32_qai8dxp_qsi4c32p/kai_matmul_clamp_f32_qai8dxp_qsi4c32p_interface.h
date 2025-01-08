//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernels variants of the same type share the same interfaces
// In this case, the micro-kernel type is: matmul_clamp_f32_qai8dxp_qsi4c32p

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_m_step_func_t)(void);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_n_step_func_t)(void);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_mr_func_t)(void);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_nr_func_t)(void);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_kr_func_t)(void);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_sr_func_t)(void);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_lhs_packed_offset_func_t)(size_t m_idx, size_t k);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_rhs_packed_offset_func_t)(size_t n_idx, size_t k, size_t bl);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_dst_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t dst_stride);
typedef size_t (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_dst_size_func_t)(size_t m, size_t n);

/// Micro-kernel core function ("run" method)
typedef void (*kai_matmul_clamp_f32_qai8dxp_qsi4c32p_run_matmul_func_t)(
    size_t m, size_t n, size_t k, size_t bl, const void* lhs_p, const void* rhs_p, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, float scalar_min, float scalar_max);

/// Micro-kernel interface
struct kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel {
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_m_step_func_t get_m_step;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_n_step_func_t get_n_step;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_mr_func_t get_mr;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_nr_func_t get_nr;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_kr_func_t get_kr;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_sr_func_t get_sr;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_lhs_packed_offset_func_t get_lhs_packed_offset;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_rhs_packed_offset_func_t get_rhs_packed_offset;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_dst_offset_func_t get_dst_offset;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_get_dst_size_func_t get_dst_size;
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_run_matmul_func_t run_matmul;
};

#ifdef __cplusplus
}
#endif
