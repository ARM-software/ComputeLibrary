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

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @param[in] mr Number of rows to be interleaved.
///
/// @return The m step value.
size_t kai_get_m_step_lhs_pack_x8p2vlx4_x8_sme(size_t mr);

/// Gets the offset in bytes to the data element in the LHS buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] lhs_stride Row stride in bytes.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme(size_t m_idx, size_t lhs_stride);

/// Gets the offset in bytes to the data element in the packed LHS buffer.
///
/// @param[in] m_idx Row index in the unpacked LHS matrix.
/// @param[in] k Number of columns in the unpacked LHS matrix.
/// @param[in] mr Number of rows to be interleaved.
/// @param[in] kr Unused. Must be 1.
/// @param[in] sr Unused. Must be 1.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);

/// Gets the size in bytes of the packed LHS buffer.
///
/// @param[in] m Number of rows in the unpacked LHS matrix.
/// @param[in] k Number of columns in the unpacked LHS matrix.
/// @param[in] mr Number of rows to be interleaved.
/// @param[in] kr Unused. Must be 1.
/// @param[in] sr Unused. Must be 1.
///
/// @return The size in bytes of the packed LHS buffer.
size_t kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme(size_t m, size_t k, size_t mr, size_t kr, size_t sr);

/// Runs the LHS packing function for matrix multiplication.
///
/// The pointer of each buffers (LHS and packed LHS) needs to be added with offset
/// calculated using the following functions:
///
///   * LHS: @ref kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme.
///   * Packed LHS: @ref kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme.
///
/// @param[in] m Number of rows of the unpacked LHS matrix.
/// @param[in] k Common dimension between the LHS and RHS matrix.
/// @param[in] mr Block size in M dimension. It must be 2 * kai_get_sme_vector_length_u8().
/// @param[in] kr Block size in K dimension. It must be 4.
/// @param[in] sr Number of kr splits. It must be 1.
/// @param[in] m_idx_start Unused. Must be 0.
/// @param[in] lhs LHS matrix data buffer.
/// @param[in] lhs_stride Row stride in bytes of the LHS matrix.
/// @param[out] lhs_packed Packed RHS matrix.
void kai_run_lhs_pack_x8p2vlx4_x8_sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
