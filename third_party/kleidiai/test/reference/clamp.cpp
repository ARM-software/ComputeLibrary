//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/clamp.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "test/common/memory.hpp"
#include "test/common/round.hpp"

namespace kai::test {

template <typename T>
std::vector<uint8_t> clamp(const void* src, size_t len, T min_value, T max_value) {
    std::vector<uint8_t> dst(round_up_division(len * size_in_bits<T>, 8));

    for (size_t i = 0; i < len; ++i) {
        write_array<T>(dst.data(), i, std::clamp(read_array<T>(src, i), min_value, max_value));
    }

    return dst;
}

template std::vector<uint8_t> clamp(const void* src, size_t len, float min_value, float max_value);

}  // namespace kai::test
