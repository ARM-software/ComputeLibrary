//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/pad.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"

namespace kai::test {

template <typename T>
Buffer pad_row(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size, const uint8_t val) {
    Buffer output(dst_size, val);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto element = read_array<T>(data, (y * src_stride) + x);
            write_array<T>(output.data(), (y * dst_stride) + x, element);
        }
    }
    return output;
}
template Buffer pad_row<Int4>(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size, const uint8_t val);

template Buffer pad_row<UInt4>(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size, const uint8_t val);

template <typename T>
Buffer pad_matrix(
    const void* data, size_t height, size_t width, size_t pad_left, size_t pad_top, size_t pad_right, size_t pad_bottom,
    T pad_value) {
    const size_t dst_height = height + pad_top + pad_bottom;
    const size_t dst_width = width + pad_left + pad_right;
    const size_t dst_size = round_up_multiple(dst_height * dst_width * size_in_bits<T>, 8);

    Buffer dst(dst_size);

    for (size_t row = 0; row < dst_height; ++row) {
        for (size_t col = 0; col < dst_width; ++col) {
            const bool valid_row = row >= pad_top && row < pad_top + height;
            const bool valid_col = col >= pad_left && col < pad_left + width;
            if (valid_row && valid_col) {
                const T value = read_array<T>(data, (row - pad_top) * width + col - pad_left);
                write_array<T>(dst.data(), row * dst_width + col, value);
            } else {
                write_array<T>(dst.data(), row * dst_width + col, pad_value);
            }
        }
    }

    return dst;
}

template Buffer pad_matrix(
    const void* data, size_t height, size_t width, size_t pad_left, size_t pad_top, size_t pad_right, size_t pad_bottom,
    float pad_value);
template Buffer pad_matrix(
    const void* data, size_t height, size_t width, size_t pad_left, size_t pad_top, size_t pad_right, size_t pad_bottom,
    int32_t pad_value);

}  // namespace kai::test
