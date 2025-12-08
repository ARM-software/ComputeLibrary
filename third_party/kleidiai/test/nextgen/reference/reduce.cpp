//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/reduce.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"

namespace kai::test {

namespace {

template <typename Op>
[[nodiscard]] Buffer reduce(size_t axis, Span<const size_t> shape, Span<const std::byte> data) {
    using Input = typename Op::InputType;
    using Output = typename Op::OutputType;

    KAI_TEST_ASSERT(shape.size() > axis);

    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only 2D data is supported.");
    KAI_TEST_ASSERT_MSG(axis == 0, "Only row reduction is supported.");

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    const size_t src_row_size = round_up_division(width * size_in_bits<Input>, 8);

    const size_t dst_size = round_up_division(height * size_in_bits<Output>, 8);
    Buffer dst(dst_size, 0);

    for (size_t row = 0; row < height; ++row) {
        const Span<const std::byte> src_row_data = data.subspan(row * src_row_size, src_row_size);

        Output acc = Op::init();

        for (size_t col = 0; col < width; ++col) {
            const Input value = read_array<Input>(src_row_data, col);
            acc = Op::reduce(acc, value);
        }

        write_array<Output>(dst, row, acc);
    }

    return dst;
}

template <typename Input, typename Output>
struct AddOp {
    using InputType = Input;
    using OutputType = Output;

    [[nodiscard]] static Output init() {
        return {};
    }

    [[nodiscard]] static Output reduce(Output acc, Input value) {
        return acc + static_cast<Output>(value);
    }
};

}  // namespace

ReduceFn make_reduce_add(DataType src_dtype, DataType dst_dtype) {
    const auto dtypes = std::make_tuple(src_dtype, dst_dtype);

    if (dtypes == std::make_tuple(DataType::U4, DataType::I32)) {
        return reduce<AddOp<UInt4, int32_t>>;
    }

    KAI_TEST_ERROR("Not implemented.");
}

}  // namespace kai::test
