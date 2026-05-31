//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/clamp.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <tuple>
#include <utility>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/reference/clamp.hpp"

namespace kai::test {

namespace {

template <typename T>
std::tuple<Buffer, Buffer> dynamic_clamp(float ratio, Span<const size_t> shape, Span<const std::byte> data) {
    KAI_TEST_ASSERT(ratio > 0.0F);
    KAI_TEST_ASSERT(ratio <= 1.0F);

    const size_t num_dims = shape.size();
    const size_t width = shape.at(num_dims - 1);
    const size_t height = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<>());

    // Finds the input range.
    T src_min = numeric_highest<T>;
    T src_max = numeric_lowest<T>;

    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            const T value = read_2d<T>(data, width, row, col);

            src_min = std::min(src_min, value);
            src_max = std::max(src_max, value);
        }
    }

    // Determines the output range.
    src_min = std::max(src_min, numeric_lowest<T>);
    src_max = std::min(src_max, numeric_highest<T>);

    const T range = src_max - src_min;
    const T reduction = static_cast<T>(static_cast<float>(range) * (1.0F - ratio) / 2);

    const T dst_min = src_min + reduction;
    const T dst_max = src_max - reduction;

    Buffer limits(sizeof(ClampLimits<T>), 0);
    *reinterpret_cast<ClampLimits<T>*>(limits.data()) = {dst_min, dst_max};

    // Clamps the data.
    Buffer dst(height * round_up_division(width * size_in_bits<T>, 8), 0);

    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            const T value = read_2d<T>(data, width, row, col);
            const T dst_value = std::clamp(value, dst_min, dst_max);
            write_2d<T>(dst.view(), width, row, col, dst_value);
        }
    }

    return {std::move(limits), std::move(dst)};
}

}  // namespace

DynamicClampFn make_dynamic_clamp(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return dynamic_clamp<float>;

        default:
            KAI_TEST_ERROR("Not implemented.");
    }
}

}  // namespace kai::test
