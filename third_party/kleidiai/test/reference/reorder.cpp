//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/reorder.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "test/common/buffer.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"

namespace kai::test {

template <typename T>
Buffer reorder_block(const void* src, size_t height, size_t width, size_t block_height, size_t block_width) {
    const auto num_dst_elements = round_up_multiple(height, block_height) * round_up_multiple(width, block_width);
    const auto dst_size = round_up_division(num_dst_elements * size_in_bits<T>, 8);

    Buffer dst(dst_size, 0);
    size_t dst_index = 0;

    for (size_t y_block = 0; y_block < height; y_block += block_height) {
        for (size_t x_block = 0; x_block < width; x_block += block_width) {
            for (size_t y_element = 0; y_element < block_height; ++y_element) {
                for (size_t x_element = 0; x_element < block_width; ++x_element) {
                    const auto y = y_block + y_element;
                    const auto x = x_block + x_element;

                    if (y < height && x < width) {
                        write_array<T>(dst.data(), dst_index, read_array<T>(src, y * width + x));
                    }

                    ++dst_index;
                }
            }
        }
    }

    return dst;
}

template Buffer reorder_block<int8_t>(
    const void* src, size_t height, size_t width, size_t block_height, size_t block_width);
template Buffer reorder_block<const void*>(
    const void* src, size_t height, size_t width, size_t block_height, size_t block_width);

}  // namespace kai::test
