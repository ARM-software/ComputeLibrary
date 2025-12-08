//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/round.hpp"

#include <cstddef>
#include <cstdint>

#include "kai/kai_common.h"

extern "C" {

/// Rounds the specified value to nearest with tie to even.
///
/// @param[in] value The value to be rounded.
///
/// @return The rounded value.
int32_t kai_test_round_to_nearest_even_i32_f32(float value);

/// Rounds the specified value to nearest with tie to even.
///
/// @param[in] value The value to be rounded.
///
/// @return The rounded value.
int64_t kai_test_round_to_nearest_even_i64_f32(float value);
}

namespace kai::test {

int32_t round_to_nearest_even_i32(float value) {
    return kai_test_round_to_nearest_even_i32_f32(value);
}

size_t round_to_nearest_even_usize(float value) {
    static_assert(sizeof(size_t) == sizeof(uint64_t));
    KAI_ASSUME_ALWAYS(value >= 0);
    return static_cast<size_t>(kai_test_round_to_nearest_even_i64_f32(value));
}

template <>
int32_t round_to_nearest_even(float value) {
    return round_to_nearest_even_i32(value);
}

template <>
size_t round_to_nearest_even(float value) {
    return round_to_nearest_even_usize(value);
}

size_t round_up_multiple(size_t a, size_t b) {
    KAI_ASSUME_ALWAYS(b != 0);
    return ((a + b - 1) / b) * b;
}

size_t round_up_division(size_t a, size_t b) {
    KAI_ASSUME_ALWAYS(b != 0);
    return (a + b - 1) / b;
}

size_t round_down_multiple(size_t a, size_t b) {
    KAI_ASSUME_ALWAYS(b != 0);
    return (a / b) * b;
}

}  // namespace kai::test
