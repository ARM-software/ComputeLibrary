//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/fill.hpp"

#include <cstddef>
#include <functional>
#include <random>
#include <type_traits>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"

namespace kai::test {

namespace {

template <typename T>
Buffer fill_matrix_random_raw(size_t height, size_t width, uint32_t seed) {
    using TDist = std::conditional_t<
        std::is_floating_point_v<T>, std::uniform_real_distribution<float>, std::uniform_int_distribution<T>>;

    std::mt19937 rnd(seed);
    TDist dist;

    return fill_matrix_raw<T>(height, width, [&](size_t, size_t) { return dist(rnd); });
}

template <>
Buffer fill_matrix_random_raw<Float16>(size_t height, size_t width, uint32_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist;

    return fill_matrix_raw<Float16>(height, width, [&](size_t, size_t) { return static_cast<Float16>(dist(rnd)); });
}

template <>
Buffer fill_matrix_random_raw<BFloat16<>>(size_t height, size_t width, uint32_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist;

    return fill_matrix_raw<BFloat16<>>(
        height, width, [&](size_t, size_t) { return static_cast<BFloat16<>>(dist(rnd)); });
}

template <>
Buffer fill_matrix_random_raw<BFloat16<false>>(size_t height, size_t width, uint32_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist;

    return fill_matrix_raw<BFloat16<false>>(
        height, width, [&](size_t, size_t) { return static_cast<BFloat16<false>>(dist(rnd)); });
}

template <>
Buffer fill_matrix_random_raw<Int4>(size_t height, size_t width, uint32_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int16_t> dist(-8, 7);

    return fill_matrix_raw<Int4>(height, width, [&](size_t, size_t) { return Int4(static_cast<int8_t>(dist(rnd))); });
}

template <>
Buffer fill_matrix_random_raw<UInt4>(size_t height, size_t width, uint32_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int16_t> dist(0, 15);

    return fill_matrix_raw<UInt4>(height, width, [&](size_t, size_t) { return UInt4(static_cast<int8_t>(dist(rnd))); });
}

}  // namespace

Buffer fill_matrix_random(size_t height, size_t width, const DataFormat& format, uint32_t seed) {
    switch (format.pack_format()) {
        case DataFormat::PackFormat::NONE:
            switch (format.data_type()) {
                case DataType::FP32:
                    return fill_matrix_random_raw<float>(height, width, seed);

                case DataType::FP16:
                    return fill_matrix_random_raw<Float16>(height, width, seed);

                case DataType::BF16:
                    return fill_matrix_random_raw<BFloat16<>>(height, width, seed);

                case DataType::QSU4:
                    return fill_matrix_random_raw<UInt4>(height, width, seed);

                case DataType::QAI4:
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
Buffer fill_random(size_t length, uint32_t seed) {
    return fill_matrix_random_raw<Value>(1, length, seed);
}

template Buffer fill_random<float>(size_t length, uint32_t seed);
template Buffer fill_random<Float16>(size_t length, uint32_t seed);
template Buffer fill_matrix_raw<float>(size_t height, size_t width, std::function<float(size_t, size_t)> gen);
template Buffer fill_matrix_raw<Float16>(size_t height, size_t width, std::function<Float16(size_t, size_t)> gen);
template Buffer fill_random<BFloat16<false>>(size_t length, uint32_t seed);

}  // namespace kai::test
