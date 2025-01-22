//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "test/common/int4.hpp"

namespace kai::test {

/// Highest finite value of type `T`.
template <typename T>
inline constexpr T numeric_highest = std::numeric_limits<T>::max();

/// Highest finite value of type `T`.
template <>
inline constexpr UInt4 numeric_highest<UInt4>{15};

/// Highest finite value of type `T`.
template <>
inline constexpr Int4 numeric_highest<Int4>{7};

/// Lowest finite value of type `T`.
template <typename T>
inline constexpr T numeric_lowest = std::numeric_limits<T>::lowest();

/// Lowest finite value of type `T`.
template <>
inline constexpr UInt4 numeric_lowest<UInt4>{0};

/// Lowest finite value of type `T`.
template <>
inline constexpr Int4 numeric_lowest<Int4>{-8};

}  // namespace kai::test
