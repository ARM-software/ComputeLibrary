//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "kai/kai_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel dependencies
///
/// -# kai_lhs_pack_x8p2vlx4_x8_sme to pack the LHS matrix.
/// -# kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme to pack the RHS matrix.

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value.
size_t kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value.
size_t kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void);

/// Gets mr value.
///
/// This is the packing parameter which must be used to pack the LHS matrix.
///
/// @return The mr value.
size_t kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void);

/// Gets nr value.
///
/// This is the packing parameter which must be used to pack the RHS matrix.
///
/// @return The nr value.
size_t kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void);

/// Gets kr value.
///
/// This is the packing parameter which must be used to pack the LHS and RHS matrix.
///
/// @return The kr value.
size_t kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void);

/// Gets sr value.
///
/// This is the packing parameter which must be used to pack the LHS and RHS matrix.
///
/// @return The sr value.
size_t kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void);

/// Gets the offset in bytes to the data element in the packed LHS matrix buffer.
///
/// @param[in] m_idx Row index in the unpacked LHS matrix.
/// @param[in] k Number of columns in the unpacked LHS matrix.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t m_idx, size_t k);

/// Gets the offset in bytes to the data element in the packed RHS matrix buffer.
///
/// @param[in] n_idx Column index in the unpacked RHS matrix.
/// @param[in] k Number of rows in the unpacked RHS matrix.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t n_idx, size_t k);

/// Gets the offset in bytes to the data element in the destination matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] dst_stride Row stride in bytes.
///
/// @return The offset in bytes to the data element.
size_t kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride);

/// Gets the size in bytes of the destination matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the destination matrix buffer.
size_t kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t m, size_t n);

/// Runs the matrix multiplication microkernel followed by a clamp operation.
///
/// The pointer of each buffers (packed LHS, packed RHS and output) needs to be added with offset
/// calculated using the following functions:
///
///   * Packed LHS: @ref kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.
///   * Packed RHS: @ref kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.
///   * Output: @ref kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.
///
/// @param[in] m Number of output rows to be computed.
/// @param[in] n Number of output columns to be computed.
/// @param[in] k Common dimension of the LHS and RHS operands.
/// @param[in] packed_lhs Packed LHS matrix buffer.
/// @param[in] packed_rhs Packed RHS matrix buffer.
/// @param[out] dst Output matrix buffer.
/// @param[in] dst_stride_row Row stride in bytes of the output matrix.
/// @param[in] dst_stride_col Column stride in bytes of the output matrix.
/// @param[in] params Requantization and clamp parmaters.
void kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct kai_matmul_requantize32_params* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
