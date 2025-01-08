//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kai::test {

/// Clamps the matrix.
///
/// @param[in] src Data buffer of the source matrix.
/// @param[in] len Number of values in the source matrix.
/// @param[in] min_value Lower bound of clamp.
/// @param[in] width Upper bound of clamp.
template <typename T>
std::vector<uint8_t> clamp(const void* src, size_t len, T min_value, T max_value);

}  // namespace kai::test
