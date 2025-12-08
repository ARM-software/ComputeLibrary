//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/unary_elementwise.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/common/type_traits.hpp"

namespace kai::test {

namespace {

template <typename Op>
[[nodiscard]] Buffer unary_elementwise(Span<const size_t> shape, Span<const std::byte> data) {
    using Type = typename Op::Type;

    const size_t width = shape.at(shape.size() - 1);
    const size_t row_size = round_up_division(width * size_in_bits<Type>, 8);
    const size_t num_rows = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<>());
    const size_t size = num_rows * row_size;

    Buffer output(size, 0);

    for (size_t row = 0; row < num_rows; ++row) {
        const Span<const std::byte> src_row_data = data.subspan(row * row_size, row_size);
        const Span<std::byte> dst_row_data = Span<std::byte>(output).subspan(row * row_size, row_size);

        for (size_t col = 0; col < width; ++col) {
            const Type src_value = read_array<Type>(src_row_data, col);
            const Type dst_value = Op::compute(src_value);
            write_array<Type>(dst_row_data, col, dst_value);
        }
    }

    return output;
}

template <typename T>
struct NegateOp {
    using Type = T;

    [[nodiscard]] static T compute(T value) {
        return -value;
    }
};

template <typename T>
struct ChangeSignednessOp {
    using Type = T;

    [[nodiscard]] static T compute(T value) {
        static_assert(is_integral<T>);
        static_assert(sizeof(T) < sizeof(uint64_t));

        constexpr T mid_point = static_cast<T>(static_cast<uint64_t>(1) << (size_in_bits<T> - 1));

        return value + mid_point;
    }
};

}  // namespace

UnaryElementwiseFn make_negate(DataType dtype) {
    switch (dtype) {
        case DataType::I32:
            return unary_elementwise<NegateOp<int32_t>>;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

UnaryElementwiseFn make_change_signedness(DataType dtype) {
    switch (dtype) {
        case DataType::U4:
        case DataType::I4:
            return unary_elementwise<ChangeSignednessOp<UInt4>>;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

}  // namespace kai::test
