//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/pad.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

template <typename T>
std::vector<uint8_t> pad_row(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size, const uint8_t val) {
    std::vector<uint8_t> output(dst_size, val);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto element = read_array<T>(data, (y * src_stride) + x);
            write_array<T>(output.data(), (y * dst_stride) + x, element);
        }
    }
    return output;
}
template std::vector<uint8_t> pad_row<Int4>(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size, const uint8_t val);

template std::vector<uint8_t> pad_row<UInt4>(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size, const uint8_t val);
}  // namespace kai::test
