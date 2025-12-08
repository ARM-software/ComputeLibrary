//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <tuple>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"

namespace kai::test {

/// Finds the clamping parameters to limit the dynamic range.
///
/// @param[in] src The data buffer.
/// @param[in] len The number of values.
/// @param[in] ratio The ratio between the output dynamic range and the input dynamic range.
///
/// @return The minimum value and the maximum value.
template <typename T>
std::tuple<T, T> find_clamp_range(const void* src, size_t len, float ratio);

/// Finds the clamping parameters to limit the dynamic range.
///
/// @param[in] type Array element data type.
/// @param[in] src The data buffer.
/// @param[in] len The number of values.
/// @param[in] ratio The ratio between the output dynamic range and the input dynamic range.
///
/// @return The minimum value and the maximum value.
std::tuple<float, float> find_clamp_range(DataType type, const void* src, size_t len, float ratio);

/// Clamps the matrix.
///
/// @param[in] src Data buffer of the source matrix.
/// @param[in] len Number of values in the source matrix.
/// @param[in] min_value Lower bound of clamp.
/// @param[in] width Upper bound of clamp.
template <typename T>
Buffer clamp(const void* src, size_t len, T min_value, T max_value);

/// Clamps the matrix.
///
/// @param[in] type Array element data type.
/// @param[in] src Data buffer of the source matrix.
/// @param[in] len Number of values in the source matrix.
/// @param[in] min_value Lower bound of clamp.
/// @param[in] max_value Upper bound of clamp.
Buffer clamp(DataType type, const void* src, size_t len, float min_value, float max_value);
}  // namespace kai::test
