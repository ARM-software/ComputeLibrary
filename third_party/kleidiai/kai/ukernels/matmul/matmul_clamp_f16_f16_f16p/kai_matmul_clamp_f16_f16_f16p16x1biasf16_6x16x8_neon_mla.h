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

/// Micro-kernel dependencies
///
/// -# kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon to pack the RHS matrix

/// --------------------------------------------------

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value.
size_t kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value.
size_t kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(void);

/// Gets nr value.
///
/// This is the packing parameter which must be used to pack the RHS matrix.
///
/// @return The nr value.
size_t kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(void);

/// Gets kr value.
///
/// This is the packing parameter which must be used to pack the RHS matrix.
///
/// @return The kr value.
size_t kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(void);

/// Gets sr value.
///
/// This is the packing parameter which must be used to pack the RHS matrix.
///
/// @return The sr value.
size_t kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(void);

/// Gets the offset in bytes to the data element in the LHS matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] stride Row stride in bytes.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(size_t m_idx, size_t stride);

/// Gets the offset in bytes to the data element in the packed RHS matrix buffer.
///
/// @param[in] n_idx Row index.
/// @param[in] k Number of columns.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(size_t n_idx, size_t k);

/// Gets the offset in bytes to the data element in the destination matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] stride Row stride in bytes.
///
/// @return The offset in bytes to the data element.
size_t kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(
    size_t m_idx, size_t n_idx, size_t stride);

/// Gets the size in bytes of the destination matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the destination matrix buffer.
size_t kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(size_t m, size_t n);

/// Runs the matrix multiplication microkernel followed by a clamp operation.
///
/// The pointer of each buffers (LHS, packed RHS and output) needs to be added with offset
/// calculated using the following functions:
///
///   * LHS: @ref kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.
///   * Packed RHS: @ref kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.
///   * Output: @ref kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.
///
/// @param[in]  m Number of output rows to be computed.
/// @param[in]  n Number of output columns to be computed.
/// @param[in]  k Common dimension of the LHS and RHS operand.
/// @param[in]  lhs LHS matrix buffer.
/// @param[in]  lhs_stride Row stride in bytes of the LHS matrix.
/// @param[in]  rhs_packed Packed RHS buffer.
/// @param[out] dst Output matrix buffer.
/// @param[in]  dst_stride_row Stride in bytes between two rows of the DST matrix.
/// @param[in]  dst_stride_col Stride in bytes between two columns of the DST matrix. For now, it must be
/// sizeof(uint16_t)
/// @param[in]  clamp_min Minimum value to clamp the final result.
/// @param[in]  clamp_max Maximum value to clamp the final result.
///
/// @note Clamp minimum and maximum values are cast internally to the destination type before clamping the computed
/// values.
///
void kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(
    size_t m, size_t n, size_t k,                             //
    const void* lhs, size_t lhs_stride,                       //
    const void* rhs_packed,                                   //
    void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
    float clamp_min, float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
