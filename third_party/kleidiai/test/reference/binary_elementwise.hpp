//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"

namespace kai::test {

/// Elementwise addition.
///
/// Broadcasting is supported for any dimension and both LHS and RHS operands.
///
/// @param[in] lhs LHS data buffer.
/// @param[in] lhs_dt LHS data type.
/// @param[in] lhs_height LHS height.
/// @param[in] lhs_width LHS width.
/// @param[in] rhs RHS data buffer.
/// @param[in] rhs_dt RHS data type.
/// @param[in] rhs_height RHS height.
/// @param[in] rhs_width RHS width.
///
/// @return The result matrix.
Buffer add(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width);

/// Elementwise subtraction.
///
/// Broadcasting is supported for any dimension and both LHS and RHS operands.
///
/// @param[in] lhs LHS data buffer.
/// @param[in] lhs_dt LHS data type.
/// @param[in] lhs_height LHS height.
/// @param[in] lhs_width LHS width.
/// @param[in] rhs RHS data buffer.
/// @param[in] rhs_dt RHS data type.
/// @param[in] rhs_height RHS height.
/// @param[in] rhs_width RHS width.
///
/// @return The result matrix.
Buffer sub(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width);

/// Elementwise subtraction.
///
/// Broadcasting is supported for any dimension and both LHS and RHS operands.
///
/// @tparam T The data type.
///
/// @param[in] lhs The LHS data buffer.
/// @param[in] lhs_height The number of rows of the LHS matrix.
/// @param[in] lhs_width The number of columns of the LHS matrix.
/// @param[in] rhs The RHS data buffer.
/// @param[in] rhs_height The number of rows of the RHS matrix.
/// @param[in] rhs_width The number of columns of the LHS matrix.
///
/// @return The result matrix.
template <typename T>
Buffer sub(
    const void* lhs, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, size_t rhs_height, size_t rhs_width);

/// Elementwise multiplication.
///
/// Broadcasting is supported for any dimension and both LHS and RHS operands.
///
/// @param[in] lhs LHS data buffer.
/// @param[in] lhs_dt LHS data type.
/// @param[in] lhs_height LHS height.
/// @param[in] lhs_width LHS width.
/// @param[in] rhs RHS data buffer.
/// @param[in] rhs_dt RHS data type.
/// @param[in] rhs_height RHS height.
/// @param[in] rhs_width RHS width.
///
/// @return The result matrix.
Buffer mul(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width);

/// Elementwise multiplication.
///
/// Broadcasting is supported for any dimension and both LHS and RHS operands.
///
/// @tparam T The data type.
///
/// @param[in] lhs The LHS data buffer.
/// @param[in] lhs_height The number of rows of the LHS matrix.
/// @param[in] lhs_width The number of columns of the LHS matrix.
/// @param[in] rhs The RHS data buffer.
/// @param[in] rhs_height The number of rows of the RHS matrix.
/// @param[in] rhs_width The number of columns of the LHS matrix.
///
/// @return The result matrix.
template <typename T>
Buffer mul(
    const void* lhs, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, size_t rhs_height, size_t rhs_width);

/// Elementwise division.
///
/// Broadcasting is supported for any dimension and both LHS and RHS operands.
///
/// @param[in] lhs LHS data buffer.
/// @param[in] lhs_dt LHS data type.
/// @param[in] lhs_height LHS height.
/// @param[in] lhs_width LHS width.
/// @param[in] rhs RHS data buffer.
/// @param[in] rhs_dt RHS data type.
/// @param[in] rhs_height RHS height.
/// @param[in] rhs_width RHS width.
///
/// @return The result matrix.
Buffer div(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width);

}  // namespace kai::test
