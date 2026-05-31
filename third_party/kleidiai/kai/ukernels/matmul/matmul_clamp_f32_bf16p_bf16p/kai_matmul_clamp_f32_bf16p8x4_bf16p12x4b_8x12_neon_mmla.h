//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// --------------------------------------------------

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value.
size_t kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value.
size_t kai_get_n_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void);

/// Gets mr value.
///
/// This is the packing parameter which must be used to pack the LHS matrix.
///
/// @return The mr value.
size_t kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void);

/// Gets nr value.
///
/// This is the packing parameter which must be used to pack the RHS matrix.
///
/// @return The nr value.
size_t kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void);

/// Gets kr value.
///
/// This is the packing parameter which must be used to pack the LHS & RHS matrices.
///
/// @return The kr value.
size_t kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void);

/// Gets sr value.
///
/// This is the packing parameter which must be used to pack the RHS matrix.
///
/// @return The sr value.
size_t kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void);

/// Gets the offset in bytes to the data element in the packed LHS matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] k Number of columns.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(size_t m_idx, size_t k);

/// Gets the offset in bytes to the data element in the packed RHS matrix buffer.
///
/// @param[in] n_idx Row index.
/// @param[in] k Number of columns.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(size_t n_idx, size_t k);

/// Gets the offset in bytes to the data element in the destination matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] stride Row stride in bytes.
///
/// @return The offset in bytes to the data element.
size_t kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
    size_t m_idx, size_t n_idx, size_t stride);

/// Gets the size in bytes of the destination matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the destination matrix buffer.
size_t kai_get_dst_size_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(size_t m, size_t n);

/// Runs the matrix multiplication microkernel followed by a clamp operation.
///
/// The pointer of each buffers (packed LHS, packed RHS and output) needs to be added with offset
/// calculated using the following functions:
///
///   * Packed LHS: @ref kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.
///   * Packed RHS: @ref kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.
///   * Output: @ref kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.
///
/// @param[in]  m Number of output rows to be computed.
/// @param[in]  n Number of output columns to be computed.
/// @param[in]  k Common dimension of the LHS and RHS operand.
/// @param[in]  lhs_packed Packed LHS buffer.
/// @param[in]  rhs_packed Packed RHS buffer.
/// @param[out] dst Output matrix buffer.
/// @param[in]  dst_stride_row Stride in bytes between two rows of the DST matrix.
/// @param[in]  dst_stride_col Stride in bytes between two columns of the DST matrix. For now, it must be sizeof(float)
/// @param[in]  clamp_min Minimum value to clamp the final result.
/// @param[in]  clamp_max Maximum value to clamp the final result.
void kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
    size_t m, size_t n, size_t k,                             //
    const void* lhs_packed,                                   //
    const void* rhs_packed,                                   //
    void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
    float clamp_min, float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
