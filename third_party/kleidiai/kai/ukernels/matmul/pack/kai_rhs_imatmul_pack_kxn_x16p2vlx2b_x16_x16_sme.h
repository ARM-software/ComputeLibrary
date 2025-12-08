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
size_t kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(void);

/// Gets the offset in bytes to the data element in the RHS matrix buffer.
///
/// @param[in] n_idx Column index. Must be divisible by `n_step`
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t n_idx);

/// Gets the offset in bytes to the data element in the bias buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the data element.
size_t kai_get_bias_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t n_idx);

/// Gets row stride in bytes of the packed RHS matrix.
///
/// @param[in] k_chunk_count Number of chunks.
/// @param[in] k_chunk_length Number of rows in each chunk.
///
/// @return Row stride in bytes.
size_t kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
    size_t k_chunk_count, size_t k_chunk_length);

/// Gets the offset in bytes to the data element in the packed RHS buffer.
///
/// @param[in] n_idx Column index. Must be divisible by `n_step`
/// @param[in] k_chunk_count Number of chunks.
/// @param[in] k_chunk_length Number of rows in each chunk.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length);

/// Gets the size in bytes of the packed RHS buffer.
///
/// @param[in] n Number of columns.
/// @param[in] k_chunk_count Number of chunks.
/// @param[in] k_chunk_length Number of rows in each chunk.
///
/// @return The size in bytes of the packed RHS buffer.
size_t kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length);

/// Runs the RHS packing function for matrix multiplication.
///
/// The pointer of each buffer (RHS, bias and packed RHS) needs to be added with offset
/// calculated using the following functions:
///
///   * RHS: @ref kai_get_rhs_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.
///   * Bias: @ref kai_get_bias_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.
///   * Output: @ref kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.
///
/// @param[in] n Number of columns of the output matrix.
/// @param[in] k_chunk_count Number of chunks.
/// @param[in] k_chunk_length Number of rows in each chunk.
/// @param[in] rhs_stride_row Row stride in bytes of the RHS matrix.
/// @param[in] rhs RHS matrix data buffer.
/// @param[in] bias Bias matrix data buffer.
/// @param[out] rhs_packed Packed RHS matrix.
void kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length, size_t rhs_stride_row, const void* rhs, const void* bias,
    void* rhs_packed);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
