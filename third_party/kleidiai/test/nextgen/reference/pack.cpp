//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/pack.hpp"

#include <cstddef>

#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
namespace kai::test {

namespace {

template <typename T>
size_t pack_block2d(
    size_t block_height, size_t block_width, size_t width_align, bool pad_right_same, size_t height, size_t width,
    Span<std::byte> packed_data, Span<const std::byte> data) {
    KAI_TEST_ASSERT(width_align % block_width == 0);

    const size_t num_block_rows = round_up_division(height, block_height);
    const size_t num_block_cols = round_up_multiple(width, width_align) / block_width;

    const size_t src_row_size = round_up_division(width * size_in_bits<T>, 8);

    size_t index = 0;

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        for (size_t block_col = 0; block_col < num_block_cols; ++block_col) {
            for (size_t elem_row = 0; elem_row < block_height; ++elem_row) {
                for (size_t elem_col = 0; elem_col < block_width; ++elem_col) {
                    const size_t row = block_row * block_height + elem_row;
                    size_t col = block_col * block_width + elem_col;

                    if (pad_right_same && col >= width) {
                        col = width - 1;
                    }

                    if (row < height && col < width) {
                        const Span<const std::byte> src_row_data = data.subspan(row * src_row_size, src_row_size);
                        const T value = read_array<T>(src_row_data, col);
                        write_array<T>(packed_data, index, value);
                    }

                    ++index;
                }
            }
        }
    }

    const size_t total_size =
        round_up_division(num_block_rows * num_block_cols * block_height * block_width * size_in_bits<T>, 8);
    return total_size;
}

}  // namespace

PackBlock2dFn make_pack_block2d(DataType dtype) {
    switch (dtype) {
        case DataType::I8:
            return pack_block2d<int8_t>;

        case DataType::I4:
            return pack_block2d<Int4>;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

}  // namespace kai::test
