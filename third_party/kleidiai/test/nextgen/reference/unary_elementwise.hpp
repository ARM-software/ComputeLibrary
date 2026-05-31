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

/// Performs unary elementwise operator.
///
/// @param[in] shape The size of multidimensional array.
/// @param[in] data The data buffer.
///
/// @return The result data.
using UnaryElementwiseFn = Buffer (*)(Span<const size_t> shape, Span<const std::byte> data);

/// Creates a negate operator for the specified data type.
///
/// @param[in] dtype The data type.
///
/// @return The function pointer.
[[nodiscard]] UnaryElementwiseFn make_negate(DataType dtype);

/// Creates an operator to change the signedness of the specified data type.
///
/// This operator will add the middle point to each value so that the output data
/// is always within range.
///
/// @param[in] dtype The data type.
///
/// @return The function pointer.
[[nodiscard]] UnaryElementwiseFn make_change_signedness(DataType dtype);

}  // namespace kai::test
