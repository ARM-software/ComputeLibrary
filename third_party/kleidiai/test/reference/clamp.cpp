//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/clamp.hpp"

#include <algorithm>
#include <cstddef>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/float16.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"

namespace kai::test {

template <typename T>
std::tuple<T, T> find_clamp_range(const void* src, size_t len, float ratio) {
    KAI_ASSUME_ALWAYS(ratio > 0.0F);
    KAI_ASSUME_ALWAYS(ratio <= 1.0F);

    T min_value = numeric_highest<T>;
    T max_value = numeric_lowest<T>;

    for (size_t i = 0; i < len; ++i) {
        const T value = read_array<T>(src, i);

        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    min_value = std::max(min_value, numeric_lowest<T>);
    max_value = std::min(max_value, numeric_highest<T>);

    const T range = max_value - min_value;
    const T reduction = static_cast<T>(static_cast<float>(range) * (1.0F - ratio) / 2);

    const T clamp_min_value = min_value + reduction;
    const T clamp_max_value = max_value - reduction;

    return {clamp_min_value, clamp_max_value};
}

template std::tuple<float, float> find_clamp_range(const void* src, size_t len, float ratio);
template std::tuple<Float16, Float16> find_clamp_range(const void* src, size_t len, float ratio);

std::tuple<float, float> find_clamp_range(DataType type, const void* src, size_t len, float ratio) {
    auto max = std::numeric_limits<float>::min();
    auto min = std::numeric_limits<float>::max();

    for (size_t i = 0; i < len; i += 1) {
        const float value = read_array(type, src, i);
        max = std::max(value, max);
        min = std::min(value, min);
    }

    const float reduction = (max - min) * (1.0F - ratio) / 2.0F;
    return {min + reduction, max - reduction};
}

template <typename T>
Buffer clamp(const void* src, size_t len, T min_value, T max_value) {
    Buffer dst(round_up_division(len * size_in_bits<T>, 8));

    for (size_t i = 0; i < len; ++i) {
        write_array<T>(dst.data(), i, std::clamp(read_array<T>(src, i), min_value, max_value));
    }

    return dst;
}

template Buffer clamp(const void* src, size_t len, float min_value, float max_value);
template Buffer clamp(const void* src, size_t len, Float16 min_value, Float16 max_value);

Buffer clamp(DataType type, const void* src, size_t len, float min_value, float max_value) {
    Buffer dst(round_up_division(len * data_type_size_in_bits(type), 8));

    for (size_t i = 0; i < len; ++i) {
        write_array(type, dst.data(), i, std::clamp<float>(read_array(type, src, i), min_value, max_value));
    }

    return dst;
}

}  // namespace kai::test
