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

/// Performs binary elementwise operator.
///
/// @param[in] lhs_height The height of the LHS data.
/// @param[in] lhs_width The width of the LHS data.
/// @param[in] lhs_data The LHS data buffer.
/// @param[in] rhs_height The height of the RHS data.
/// @param[in] rhs_width The width of the RHS data.
/// @param[in] rhs_data The RHS data buffer.
///
/// @return The result data.
using BinaryElementwiseFn = Buffer (*)(
    size_t lhs_height, size_t lhs_width, Span<const std::byte> lhs_data, size_t rhs_height, size_t rhs_width,
    Span<const std::byte> rhs_data);

/// Creates an add operator for the specified data type.
///
/// @param[in] dtype The data type.
///
/// @return The function pointer.
[[nodiscard]] BinaryElementwiseFn make_add_2d(DataType dtype);

}  // namespace kai::test
