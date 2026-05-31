//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernels variants of the same type share the same interfaces
// In this case, the micro-kernel type is: dwconv_clamp_f32_f32_f32p_planar
// NOTE:
// - get_n_step is not provided as n-step is not relevant in planar kernels.
// - get_lhs_packed_offset is not provided as the lhs is not packed with planar kernels.
// - get_rhs_packed_offset is not provided as rhs offset is not relevant with planar kernels.

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*kai_dwconv_clamp_f32_f32_f32p_planar_get_m_step_func_t)(void);
typedef size_t (*kai_dwconv_clamp_f32_f32_f32p_planar_get_dst_offset_func_t)(size_t out_row_idx, size_t dst_stride_row);
typedef size_t (*kai_dwconv_clamp_f32_f32_f32p_planar_get_dst_size_func_t)(
    size_t out_height, size_t out_width, size_t num_channels);

/// Micro-kernel core function ("run" method)
typedef void (*kai_dwconv_clamp_f32_f32_f32p_planar_run_dwconv_func_t)(
    const void* inptr, const void* packed_rhs, void* outptr_start, size_t in_stride_row, size_t in_stride_col,
    size_t dst_stride_row, size_t dst_stride_col, size_t valid_input_rows, size_t valid_out_rows, size_t pad_left,
    size_t pad_top, float pad_value, float clamp_min, float clamp_max);

/// Micro-kernel interface
struct kai_dwconv_clamp_f32_f32_f32p_planar_ukernel {
    kai_dwconv_clamp_f32_f32_f32p_planar_get_m_step_func_t get_m_step;
    kai_dwconv_clamp_f32_f32_f32p_planar_get_dst_offset_func_t get_dst_offset;
    kai_dwconv_clamp_f32_f32_f32p_planar_get_dst_size_func_t get_dst_size;
    kai_dwconv_clamp_f32_f32_f32p_planar_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
