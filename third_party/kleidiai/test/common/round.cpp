//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/round.hpp"

#include <cstddef>
#include <cstdint>

namespace kai::test {

int32_t round_to_nearest_even_i32(float value) {
    int32_t rounded = 0;
    __asm__ __volatile__("fcvtns %w[output], %s[input]" : [output] "=r"(rounded) : [input] "w"(value));
    return rounded;
}

size_t round_to_nearest_even_usize(float value) {
    static_assert(sizeof(size_t) == sizeof(uint64_t));

    uint64_t rounded = 0;
    __asm__ __volatile__("fcvtns %x[output], %s[input]" : [output] "=r"(rounded) : [input] "w"(value));
    return rounded;
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
    return ((a + b - 1) / b) * b;
}

size_t round_up_division(size_t a, size_t b) {
    return (a + b - 1) / b;
}

size_t round_down_multiple(size_t a, size_t b) {
    return (a / b) * b;
}

}  // namespace kai::test
