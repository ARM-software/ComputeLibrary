//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
/// The starting row index must be divisible by `n_step`.
///
/// @return The n step value.
size_t kai_get_n_step_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(void);

/// Gets the offset in bytes to the data element in the RHS matrix buffer.
///
/// @param[in] n_idx Column index.
/// @param[in] rhs_offset Row stride in bytes of the RHS matrix.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx, size_t rhs_offset);

/// Gets the offset in bytes to the data element in the bias buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the data element.
size_t kai_get_bias_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx);

/// Gets the row stride in bytes of the packed RHS matrix.
///
/// @param[in] k The number of columns in the transposed RHS matrix.
///
/// @return The row stride in bytes.
size_t kai_get_rhs_packed_stride_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t k);

/// Gets the offset in bytes to the data element in the packed RHS buffer.
///
/// @param[in] n_idx Row index.
/// @param[in] k Number of columns.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx, size_t k);

/// Gets the size in bytes of the packed RHS buffer.
///
/// @param[in] n Number of rows.
/// @param[in] k Number of columns.
///
/// @return The size in bytes of the packed RHS buffer.
size_t kai_get_rhs_packed_size_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n, size_t k);

/// Runs the RHS packing function for matrix multiplication.
///
/// The pointer of each buffers (RHS, bias and packed RHS) needs to be added with offset
/// calculated using the following functions:
///
///   * RHS: @ref kai_get_rhs_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.
///   * Bias: @ref kai_get_bias_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.
///   * Output: @ref kai_get_rhs_packed_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.
///
/// @param[in] num_groups Number of groups. It must be 1.
/// @param[in] n Number of columns of the output matrix.
/// @param[in] k Common dimension between the LHS and RHS matrix.
/// @param[in] nr Block size in N dimension. It must be 2 * kai_get_sme_vector_length_u16().
/// @param[in] kr Block size in K dimension. It must be 2.
/// @param[in] sr Number of kr splits. It must be 1.
/// @param[in] rhs_stride Row stride in bytes of the RHS matrix.
/// @param[in] rhs RHS matrix data buffer.
/// @param[in] bias Bias matrix data buffer.
/// @param[in] scale Scale data buffer. It must be NULL.
/// @param[out] rhs_packed Packed RHS matrix.
/// @param[in] extra_bytes Extra bytes to append to the end of each row of the packed RHS matrix. It must be 0.
/// @param[in] params Extra packing parameters. It must be NULL.
void kai_run_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
