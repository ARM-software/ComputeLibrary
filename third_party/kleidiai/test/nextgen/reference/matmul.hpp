//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"

namespace kai::test {

/// Matrix multiplication.
///
/// @param[in] shape_m The size of M dimension.
/// @param[in] shape_n The size of N dimension.
/// @param[in] shape_k The size of K dimension.
/// @param[in] lhs The LHS matrix.
/// @param[in] rhs The RHS matrix.
///
/// @return The output matrix.
using MatMulFn =
    Buffer (*)(size_t shape_m, size_t shape_n, size_t shape_k, Span<const std::byte> lhs, Span<const std::byte> rhs);

/// Creates a matrix multiplication function for the specified data type.
///
/// The LHS matrix is non-transposed and the RHS matrix is transposed.
///
/// @param[in] dtype The data type.
///
/// @return The function pointer.
[[nodiscard]] MatMulFn make_matmul_nt_t(DataType dtype);

}  // namespace kai::test
