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

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value.
size_t kai_get_m_step_lhs_imatmul_pack_x16p2vlx2_x16p_sme(void);

/// Gets the offset in bytes to the data element in the packed LHS buffer.
///
/// @param[in] m_idx Row index in the unpacked LHS matrix.
/// @param[in] k_chunk_count Number of LHS column splits.
/// @param[in] k_chunk_length Length of a LHS column split.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_packed_offset_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
    size_t m_idx, size_t k_chunk_count, size_t k_chunk_length);

/// Gets the size in bytes of the packed LHS buffer.
///
/// @param[in] m Number of rows in the unpacked LHS matrix.
/// @param[in] k_chunk_count Number of LHS column splits.
/// @param[in] k_chunk_length Length of a LHS column split.
///
/// @return The size in bytes of the packed LHS buffer.
size_t kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
    size_t m, size_t k_chunk_count, size_t k_chunk_length);

/// Pack the LHS matrix for use with indirect matrix multiplication
///
/// @param[in] m Number of rows of the unpacked LHS matrix.
/// @param[in] k_chunk_count Number of LHS column splits.
/// @param[in] k_chunk_length Length of a LHS column split.
/// @param[in] lhs_ptrs Pointer to an array of input pointers consisting of
///            `m * k_chunk_count` pointers.
/// @param[in] lhs_ptr_offset Offset to add to each pointer of the @ref lhs_ptrs
///            array, excluding zero pointers.
/// @param[in] pad_ptr Pointer to chunk used for padding. @ref lhs_ptr_offset is
///            not applied to this pointer when used in @ref lhs_ptrs. This can
///            be NULL if there is no padding used @ref lhs_ptrs
/// @param[out] lhs_packed Packed LHS matrix.
void kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
    size_t m, size_t k_chunk_count, size_t k_chunk_length, const void* const* lhs_ptrs, size_t lhs_ptr_offset,
    const void* pad_ptr, void* lhs_packed);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
