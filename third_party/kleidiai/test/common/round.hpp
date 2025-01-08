//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace kai::test {

/// Rounds the specified value to nearest with tie to even.
///
/// For example:
///
///   * 0.4 is rounded to 0.
///   * 0.5 is rounded to 0 (as 0 is the nearest even value).
///   * 0.6 is rounded to 1.
///   * 1.4 is rounded to 1.
///   * 1.5 is rounded to 2 (as 2 is the nearest even value).
///   * 1.6 is rounded to 2.
///
/// @param[in] value Value to be rounded.
///
/// @return The rounded value.
int32_t round_to_nearest_even_i32(float value);

/// Rounds the specified value to nearest with tie to even.
///
/// For example:
///
///   * 0.4 is rounded to 0.
///   * 0.5 is rounded to 0 (as 0 is the nearest even value).
///   * 0.6 is rounded to 1.
///   * 1.4 is rounded to 1.
///   * 1.5 is rounded to 2 (as 2 is the nearest even value).
///   * 1.6 is rounded to 2.
///
/// @param[in] value Value to be rounded.
///
/// @return The rounded value.
size_t round_to_nearest_even_usize(float value);

/// Rounds the specified value to nearest with tie to even.
///
/// For example:
///
///   * 0.4 is rounded to 0.
///   * 0.5 is rounded to 0 (as 0 is the nearest even value).
///   * 0.6 is rounded to 1.
///   * 1.4 is rounded to 1.
///   * 1.5 is rounded to 2 (as 2 is the nearest even value).
///   * 1.6 is rounded to 2.
///
/// @tparam T The target data type (must be integer).
///
/// @param[in] value Value to be rounded.
///
/// @return The rounded value.
template <typename T>
T round_to_nearest_even(float value);

/// Rounds up the input value to the multiple of the unit value.
///
/// @param[in] a Input value.
/// @param[in] b Unit value.
///
/// @return The rounded value.
size_t round_up_multiple(size_t a, size_t b);

/// Divides and rounds up.
///
/// @param[in] a The dividend.
/// @param[in] b The divisor.
///
/// @return The division of a to b rounding up.
size_t round_up_division(size_t a, size_t b);

/// Rounds down the input value to the multiple of the unit value.
///
/// @param[in] a Input value.
/// @param[in] b Unit value.
///
/// @return The rounded value.
size_t round_down_multiple(size_t a, size_t b);

}  // namespace kai::test
