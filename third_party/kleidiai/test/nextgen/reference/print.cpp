//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/print.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>

#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"

namespace kai::test {

namespace {

template <typename T>
void print_impl(std::ostream& os, Span<const size_t> shape, Span<const std::byte> data, size_t level = 0) {
    const std::string indent(level * 2, ' ');
    const size_t len = shape.at(0);

    if (shape.size() == 1) {
        os << indent << "[";

        for (size_t i = 0; i < len; ++i) {
            const T value = read_array<T>(data, i);
            os << displayable(value) << ", ";
        }

        os << "]";
    } else {
        const size_t row_size = round_up_division(shape.at(shape.size() - 1) * size_in_bits<T>, 8);
        const size_t num_rows = std::accumulate(shape.begin() + 1, shape.end() - 1, 1, std::multiplies<>());
        const size_t stride = num_rows * row_size;

        os << indent << "[\n";

        for (size_t i = 0; i < len; ++i) {
            print_impl<T>(os, shape.subspan(1), data.subspan(i * stride), level + 1);
            os << ",\n";
        }

        os << indent << "]";
    }
}

template <typename T>
void print_array(std::ostream& os, Span<const size_t> shape, Span<const std::byte> data, size_t level) {
    print_impl<T>(os, shape, data, level);
}

}  // namespace

PrintFn make_print_array(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return print_array<float>;

        case DataType::I32:
            return print_array<int32_t>;

        case DataType::I8:
            return print_array<int8_t>;

        case DataType::U4:
            return print_array<UInt4>;

        case DataType::I4:
            return print_array<Int4>;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

}  // namespace kai::test
