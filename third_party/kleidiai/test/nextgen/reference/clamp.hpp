//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <tuple>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"

namespace kai::test {

/// Clamp limits.
template <typename T>
struct ClampLimits {
    T min_value;
    T max_value;
};

/// Determines the clamp range and clamps the data.
///
/// @param[in] ratio The ratio between the output range and the input range.
/// @param[in] shape The size of multidimensional array.
/// @param[in] data The data buffer.
///
/// @return The clamp range and clamped data.
using DynamicClampFn =
    std::tuple<Buffer, Buffer> (*)(float ratio, Span<const size_t> shape, Span<const std::byte> data);

/// Creates a clamp function for the specified data type.
///
/// @param[in] dtype The data type.
///
/// @return The function pointer.
[[nodiscard]] DynamicClampFn make_dynamic_clamp(DataType dtype);

}  // namespace kai::test
