//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/fill.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

namespace {

template <typename T>
std::vector<uint8_t> fill_matrix_raw(size_t height, size_t width, std::function<T(size_t, size_t)> gen) {
    const auto size = height * width * size_in_bits<T> / 8;
    KAI_ASSUME(width * size_in_bits<T> % 8 == 0);

    std::vector<uint8_t> data;
    data.resize(size);
    auto ptr = reinterpret_cast<T*>(data.data());

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            write_array<T>(ptr, y * width + x, gen(y, x));
        }
    }

    return data;
}

template <typename T>
std::vector<uint8_t> fill_matrix_random_raw(size_t height, size_t width, uint64_t seed) {
    using TDist = std::conditional_t<
        std::is_floating_point_v<T>, std::uniform_real_distribution<float>, std::uniform_int_distribution<T>>;

    std::mt19937 rnd(seed);
    TDist dist;

    return fill_matrix_raw<T>(height, width, [&](size_t, size_t) { return dist(rnd); });
}

template <>
std::vector<uint8_t> fill_matrix_random_raw<Float16>(size_t height, size_t width, uint64_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist;

    return fill_matrix_raw<Float16>(height, width, [&](size_t, size_t) { return static_cast<Float16>(dist(rnd)); });
}

template <>
std::vector<uint8_t> fill_matrix_random_raw<BFloat16>(size_t height, size_t width, uint64_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist;

    return fill_matrix_raw<BFloat16>(height, width, [&](size_t, size_t) { return static_cast<BFloat16>(dist(rnd)); });
}

template <>
std::vector<uint8_t> fill_matrix_random_raw<Int4>(size_t height, size_t width, uint64_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int8_t> dist(-8, 7);

    return fill_matrix_raw<Int4>(height, width, [&](size_t, size_t) { return Int4(dist(rnd)); });
}

template <>
std::vector<uint8_t> fill_matrix_random_raw<UInt4>(size_t height, size_t width, uint64_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int8_t> dist(0, 15);

    return fill_matrix_raw<UInt4>(height, width, [&](size_t, size_t) { return UInt4(dist(rnd)); });
}

}  // namespace

std::vector<uint8_t> fill_matrix_random(size_t height, size_t width, const DataFormat& format, uint64_t seed) {
    switch (format.pack_format()) {
        case DataFormat::PackFormat::NONE:
            switch (format.data_type()) {
                case DataType::FP32:
                    return fill_matrix_random_raw<float>(height, width, seed);

                case DataType::FP16:
                    return fill_matrix_random_raw<Float16>(height, width, seed);

                case DataType::BF16:
                    return fill_matrix_random_raw<BFloat16>(height, width, seed);

                case DataType::QSU4:
                    return fill_matrix_random_raw<UInt4>(height, width, seed);

                case DataType::QSI4:
                    return fill_matrix_random_raw<Int4>(height, width, seed);

                default:
                    KAI_ERROR("Unsupported data type!");
            }

            break;

        default:
            KAI_ERROR("Unsupported data format!");
    }
}

template <typename Value>
std::vector<uint8_t> fill_random(size_t length, uint64_t seed) {
    return fill_matrix_random_raw<Value>(1, length, seed);
}

template std::vector<uint8_t> fill_random<float>(size_t length, uint64_t seed);

}  // namespace kai::test
