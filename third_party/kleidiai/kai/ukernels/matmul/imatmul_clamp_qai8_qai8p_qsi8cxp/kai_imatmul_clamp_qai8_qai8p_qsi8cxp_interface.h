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
// In this case, the micro-kernel type is: imatmul_clamp_qai8_qai8p_qsi8cxp

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_m_step_func_t)(void);
typedef size_t (*kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_n_step_func_t)(void);
typedef size_t (*kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_lhs_packed_offset_func_t)(
    size_t m_idx, size_t k_chunk_count, size_t k_chunk_length);
typedef size_t (*kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_rhs_packed_offset_func_t)(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length);
typedef size_t (*kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_dst_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t dst_stride_row);
typedef size_t (*kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_dst_size_func_t)(size_t m, size_t n);

/// Micro-kernel core function ("run" method)
typedef void (*kai_imatmul_clamp_qai8_qai8p_qsi8cxp_run_imatmul_func_t)(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length, const void* lhs_packed, const void* rhs_packed,
    void* dst, size_t dst_stride_row, const struct kai_matmul_requantize32_params* params);

/// Micro-kernel interface
struct kai_imatmul_clamp_qai8_qai8p_qsi8cxp_ukernel {
    kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_m_step_func_t get_m_step;
    kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_n_step_func_t get_n_step;
    kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_lhs_packed_offset_func_t get_lhs_packed_offset;
    kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_rhs_packed_offset_func_t get_rhs_packed_offset;
    kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_dst_offset_func_t get_dst_offset;
    kai_imatmul_clamp_qai8_qai8p_qsi8cxp_get_dst_size_func_t get_dst_size;
    kai_imatmul_clamp_qai8_qai8p_qsi8cxp_run_imatmul_func_t run_imatmul;
};

#ifdef __cplusplus
}
#endif
