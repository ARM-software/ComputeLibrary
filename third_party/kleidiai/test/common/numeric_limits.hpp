//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <type_traits>

#include "test/common/bfloat16.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"

namespace kai::test {

/// Highest finite value of type `T`.
template <typename T>
inline constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> numeric_highest = std::numeric_limits<T>::max();

/// Highest finite value of @ref UInt4.
template <>
inline constexpr UInt4 numeric_highest<UInt4>{15};

/// Highest finite value of @ref Int4.
template <>
inline constexpr Int4 numeric_highest<Int4>{7};

/// Highest finite value of @ref Float16.
template <>
inline constexpr Float16 numeric_highest<Float16> = Float16::from_binary(0x7bff);

/// Highest finite value of @ref BFloat16.
template <>
inline constexpr BFloat16 numeric_highest<BFloat16<>> = BFloat16<>::from_binary(0x7f7f);

/// Lowest finite value of type `T`.
template <typename T>
inline constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> numeric_lowest = std::numeric_limits<T>::lowest();

/// Lowest finite value of @ref UInt4.
template <>
inline constexpr UInt4 numeric_lowest<UInt4>{0};

/// Lowest finite value of @ref Int4.
template <>
inline constexpr Int4 numeric_lowest<Int4>{-8};

/// Lowest finite value of @ref Float16.
template <>
inline constexpr Float16 numeric_lowest<Float16> = Float16::from_binary(0xfbff);

/// Lowest finite value of @ref BFloat16.
template <>
inline constexpr BFloat16 numeric_lowest<BFloat16<>> = BFloat16<>::from_binary(0xff7f);

}  // namespace kai::test
