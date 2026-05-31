//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/binary_elementwise.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"

namespace kai::test {

namespace {

template <typename T>
[[nodiscard]] Buffer add(
    size_t lhs_height, size_t lhs_width, Span<const std::byte> lhs_data, size_t rhs_height, size_t rhs_width,
    Span<const std::byte> rhs_data) {
    const size_t dst_height = std::max(lhs_height, rhs_height);
    KAI_TEST_ASSERT(lhs_height == rhs_height || lhs_height == 1 || rhs_height == 1);

    const size_t dst_width = std::max(lhs_width, rhs_width);
    KAI_TEST_ASSERT(lhs_width == rhs_width || lhs_width == 1 || rhs_width == 1);

    const size_t lhs_row_size = round_up_division(lhs_width * size_in_bits<T>, 8);
    const size_t rhs_row_size = round_up_division(rhs_width * size_in_bits<T>, 8);
    const size_t dst_row_size = round_up_division(dst_width * size_in_bits<T>, 8);

    const size_t dst_size = dst_height * dst_row_size;

    Buffer dst(dst_size, 0);

    for (size_t row = 0; row < dst_height; ++row) {
        const Span<const std::byte> lhs_row_data = lhs_data.subspan((row % lhs_height) * lhs_row_size, lhs_row_size);
        const Span<const std::byte> rhs_row_data = rhs_data.subspan((row % rhs_height) * rhs_row_size, rhs_row_size);
        const Span<std::byte> dst_row_data = Span<std::byte>(dst).subspan(row * dst_row_size, dst_row_size);

        for (size_t col = 0; col < dst_width; ++col) {
            const T lhs_value = read_array<T>(lhs_row_data, col % lhs_width);
            const T rhs_value = read_array<T>(rhs_row_data, col % rhs_width);

            const T dst_value = lhs_value + rhs_value;

            write_array<T>(dst_row_data, col, dst_value);
        }
    }

    return dst;
}

}  // namespace

BinaryElementwiseFn make_add_2d(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return add<float>;

        case DataType::U4:
        case DataType::I4:
            return add<Int4>;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

}  // namespace kai::test
