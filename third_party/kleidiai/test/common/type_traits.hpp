//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <type_traits>

#include "test/common/float16.hpp"

namespace kai::test {

class UInt4;
class Int4;
class BFloat16;

/// `true` if `T` is unsigned numeric type.
template <typename T>
inline constexpr bool is_unsigned = std::is_unsigned_v<T>;

/// `true` if `T` is unsigned numeric type.
template <>
inline constexpr bool is_unsigned<UInt4> = true;

/// `true` if `T` is unsigned numeric type.
template <>
inline constexpr bool is_unsigned<Int4> = false;

/// `true` if `T` is unsigned numeric type.
template <>
inline constexpr bool is_unsigned<BFloat16> = false;

/// `true` if `T` is signed numeric type.
template <typename T>
inline constexpr bool is_signed = std::is_signed_v<T>;

/// `true` if `T` is signed numeric type.
template <>
inline constexpr bool is_signed<UInt4> = false;

/// `true` if `T` is signed numeric type.
template <>
inline constexpr bool is_signed<Int4> = true;

/// `true` if `T` is signed numeric type.
template <>
inline constexpr bool is_signed<BFloat16> = true;

/// `true` if `T` is integral numeric type.
template <typename T>
inline constexpr bool is_integral = std::is_integral_v<T>;

/// `true` if `T` is integral numeric type.
template <>
inline constexpr bool is_integral<UInt4> = true;

/// `true` if `T` is integral numeric type.
template <>
inline constexpr bool is_integral<Int4> = true;

/// `true` if `T` is integral numeric type.
template <>
inline constexpr bool is_integral<BFloat16> = false;

/// `true` if `T` is floating-point type.
template <typename T>
inline constexpr bool is_floating_point = std::is_floating_point_v<T>;

/// `true` if `T` is floating-point type.
template <>
inline constexpr bool is_floating_point<Float16> = true;

/// `true` if `T` is floating-point type.
template <>
inline constexpr bool is_floating_point<BFloat16> = true;

/// `true` if `T` is integral or floating-point type.
template <typename T>
inline constexpr bool is_arithmetic = is_integral<T> || is_floating_point<T>;

/// Signed version of type `T`.
template <typename T>
struct make_signed {
    using type = std::make_signed_t<T>;
};

/// Signed version of type `T`.
template <>
struct make_signed<UInt4> {
    using type = Int4;
};

/// Signed version of type `T`.
template <>
struct make_signed<Int4> {
    using type = Int4;
};

/// Signed version of type `T`.
template <typename T>
using make_signed_t = typename make_signed<T>::type;

}  // namespace kai::test
