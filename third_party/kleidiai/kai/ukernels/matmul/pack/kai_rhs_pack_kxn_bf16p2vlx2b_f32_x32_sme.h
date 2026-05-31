//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value.
size_t kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(void);

/// Gets the offset in bytes to the data element in the RHS matrix buffer.
///
/// @param[in] n_idx Column index. Must be divisible by `n_step`
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t n_idx);

/// Gets the offset in bytes to the data element in the bias buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the data element.
size_t kai_get_bias_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t n_idx);

/// Get the row stride in bytes to the packed RHS matrix
///
/// @param[in] k In the RHS matrix (not packed), K is the number of columns.
///
/// @return The stride in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_stride_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t k);

/// Gets the offset in bytes to the data element in the packed RHS buffer.
///
/// @param[in] n_idx Column index. Must be divisible by `n_step`
/// @param[in] k Number of rows.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t n_idx, size_t k);

/// Gets the size in bytes of the packed RHS buffer.
///
/// @param[in] n Number of columns.
/// @param[in] k Number of rows.
///
/// @return The size in bytes of the packed RHS buffer.
size_t kai_get_rhs_packed_size_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t n, size_t k);

/// Runs the RHS packing function for matrix multiplication.
///
/// The pointer of each buffers (RHS, bias and packed RHS) needs to be added with offset
/// calculated using the following functions:
///
///   * RHS: @ref kai_get_rhs_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.
///   * Bias: @ref kai_get_bias_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.
///   * Output: @ref kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.
///
/// @param[in] num_groups Number of groups. It must be 1.
/// @param[in] n Number of columns of the output matrix.
/// @param[in] k Common dimension between the LHS and RHS matrix.
/// @param[in] nr Block size in N dimension. It must be `get_n_step`
/// @param[in] kr Block size in K dimension. It must be 2.
/// @param[in] sr Number of kr splits. It must be 1.
/// @param[in] rhs_stride Row stride in bytes of the RHS matrix.
/// @param[in] rhs RHS matrix data buffer.
/// @param[in] bias Bias matrix data buffer.
/// @param[in] scale Scale data buffer. It must be NULL.
/// @param[out] rhs_packed Packed RHS matrix.
/// @param[in] extra_bytes Extra bytes to append to the end of each row of the packed RHS matrix. It must be 0.
/// @param[in] params Extra packing parameters. It must be NULL.
void kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
