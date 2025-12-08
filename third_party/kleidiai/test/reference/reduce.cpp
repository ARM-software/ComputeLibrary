//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/reduce.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"

namespace kai::test {

namespace {

template <const ReductionOperator op, typename T>
T scalar_reduce(T curr_value, T new_value) {
    if constexpr (op == ReductionOperator::ADD) {
        return curr_value + new_value;
    }
}

template <const ReductionOperator op, typename Input, typename Output>
Buffer reduce_any_op_type(const void* src, size_t height, size_t width, size_t dimension) {
    switch (dimension) {
        case 0: {
            Buffer dst(height * size_in_bits<Output> / 8);
            KAI_ASSUME_ALWAYS(height * size_in_bits<Output> % 8 == 0);

            for (size_t y = 0; y < height; ++y) {
                Output acc = read_array<Input>(src, y * width);

                for (size_t x = 1; x < width; ++x) {
                    Output value = read_array<Input>(src, y * width + x);
                    acc = scalar_reduce<op, Output>(acc, value);
                }

                write_array<Output>(dst.data(), y, acc);
            }

            return dst;
        }

        case 1: {
            Buffer dst(width * size_in_bits<Output> / 8);
            KAI_ASSUME_ALWAYS(width * size_in_bits<Output> % 8 == 0);

            for (size_t x = 0; x < width; ++x) {
                Output acc = read_array<Input>(src, x);

                for (size_t y = 1; y < height; ++y) {
                    Output value = read_array<Input>(src, y * width + x);
                    acc = scalar_reduce<op, Output>(acc, value);
                }

                write_array<Output>(dst.data(), x, acc);
            }

            return dst;
        }

        default:
            KAI_ERROR("Only 2D data is supported!");
    }
}

template <const ReductionOperator op>
Buffer reduce_any_op(
    const void* src, const DataFormat& src_format, size_t height, size_t width, const DataFormat& dst_format,
    size_t dimension) {
    KAI_ASSUME_ALWAYS(src_format.is_raw());
    KAI_ASSUME_ALWAYS(dst_format.is_raw());
    KAI_ASSUME_ALWAYS(dimension < 2);
    KAI_ASSUME_ALWAYS(height > 0);
    KAI_ASSUME_ALWAYS(width > 0);

    const auto src_dt = src_format.data_type();
    const auto dst_dt = dst_format.data_type();

    switch (src_dt) {
        case DataType::QSU4:
            switch (dst_dt) {
                case DataType::I32:
                    return reduce_any_op_type<op, UInt4, int32_t>(src, height, width, dimension);
                    break;

                default:
                    KAI_ERROR("Unsupported data type!");
            }

        default:
            KAI_ERROR("Unsupported data type!");
    }
}

}  // namespace

Buffer reduce_add(
    const void* src, const DataFormat& src_format, size_t height, size_t width, const DataFormat& dst_format,
    size_t dimension) {
    return reduce_any_op<ReductionOperator::ADD>(src, src_format, height, width, dst_format, dimension);
}

template <typename Value, typename Accumulator>
Buffer reduce_add_x(const void* src, size_t height, size_t width) {
    Buffer dst(round_up_division(height * size_in_bits<Accumulator>, 8));

    for (size_t y = 0; y < height; ++y) {
        Accumulator acc = 0;

        for (size_t x = 0; x < width; ++x) {
            acc += static_cast<Accumulator>(read_array<Value>(src, y * width + x));
        }

        write_array<Accumulator>(dst.data(), y, acc);
    }

    return dst;
}

template Buffer reduce_add_x<int8_t, int32_t>(const void* src, size_t height, size_t width);

template <typename T>
T reduce_min(const void* src, size_t len) {
    KAI_ASSUME_ALWAYS(len > 0);

    T min = read_array<T>(src, 0);

    for (size_t i = 1; i < len; ++i) {
        min = std::min(min, read_array<T>(src, i));
    }

    return min;
}

template float reduce_min(const void* src, size_t len);

template <typename T>
T reduce_max(const void* src, size_t len) {
    KAI_ASSUME_ALWAYS(len > 0);

    T max = read_array<T>(src, 0);

    for (size_t i = 1; i < len; ++i) {
        max = std::max(max, read_array<T>(src, i));
    }

    return max;
}

template float reduce_max(const void* src, size_t len);

}  // namespace kai::test
