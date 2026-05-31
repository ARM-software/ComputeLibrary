//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/format/block2d_row_format.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ostream>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/reference/compare.hpp"
#include "test/nextgen/reference/pack.hpp"
#include "test/nextgen/reference/print.hpp"

namespace kai::test {

size_t Block2dRowFormat::compute_offset(Span<const size_t> shape, Span<const size_t> indices) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(shape.size() == indices.size());

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t row = indices.at(0);
    const size_t col = indices.at(1);

    KAI_TEST_ASSERT(row < height);
    KAI_TEST_ASSERT(col < width);

    KAI_TEST_ASSERT(row % m_block_height == 0);
    KAI_TEST_ASSERT(col % m_block_width == 0);

    const bool has_per_row_component = !m_pre_dtypes.empty() || !m_post_dtypes.empty();
    if (has_per_row_component) {
        KAI_TEST_ASSERT(col == 0);
    }

    const size_t block_row = row / m_block_height;
    const size_t block_col = col / m_block_width;

    const size_t block_size = m_block_height * m_block_width * data_type_size_in_bits(m_dtype) / 8;
    const size_t num_blocks_per_row = round_up_multiple(width, m_width_align) / m_block_width;

    if (has_per_row_component) {
        size_t block_row_size = block_size * num_blocks_per_row;
        for (const DataType dtype : m_pre_dtypes) {
            block_row_size += m_block_height * data_type_size_in_bits(dtype) / 8;
        }
        for (const DataType dtype : m_post_dtypes) {
            block_row_size += m_block_height * data_type_size_in_bits(dtype) / 8;
        }

        return block_row * block_row_size;
    } else {
        return (block_row * num_blocks_per_row + block_col) * block_size;
    }
}

size_t Block2dRowFormat::compute_size(Span<const size_t> shape) const {
    KAI_TEST_ASSERT(shape.size() == 2);

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t padded_height = round_up_multiple(height, m_block_height);

    const size_t size = compute_offset({padded_height + m_block_height, width}, {padded_height, 0});
    return size;
}

Buffer Block2dRowFormat::generate_random([[maybe_unused]] Span<const size_t> shape, [[maybe_unused]] Rng& rng) const {
    KAI_TEST_ERROR("Not supported!");
}

Buffer Block2dRowFormat::pack(Span<const size_t> shape, Span<const Span<const std::byte>> buffers) const {
    KAI_TEST_ASSERT(shape.size() == 2);

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    const size_t num_block_rows = round_up_division(height, m_block_height);

    const size_t packed_size = compute_size(shape);
    Buffer packed_buffer(packed_size, 0);
    Span<std::byte> packed_data(packed_buffer);

    const PackBlock2dFn pack_data_fn = make_pack_block2d(m_dtype);

    const size_t num_pres = m_pre_dtypes.size();
    std::vector<Span<const std::byte>> pre_buffers;
    pre_buffers.reserve(num_pres);
    for (size_t i = 0; i < num_pres; ++i) {
        pre_buffers.emplace_back(buffers.at(i));
    }

    Span<const std::byte> data_buffer = buffers.at(num_pres);

    const size_t num_posts = m_post_dtypes.size();
    std::vector<Span<const std::byte>> post_buffers;
    post_buffers.reserve(num_posts);
    for (size_t i = 0; i < num_posts; ++i) {
        post_buffers.emplace_back(buffers.at(num_pres + 1 + i));
    }

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        const size_t remaining_height = std::min(m_block_height, height - block_row * m_block_height);

        for (size_t i = 0; i < num_pres; ++i) {
            const size_t copy_size = remaining_height * data_type_size_in_bits(m_pre_dtypes.at(i)) / 8;
            Span<const std::byte>& data = pre_buffers.at(i);

            std::copy_n(data.begin(), copy_size, packed_data.begin());

            data = data.subspan(copy_size);
            packed_data = packed_data.subspan(m_block_height * data_type_size_in_bits(m_pre_dtypes.at(i)) / 8);
        }

        {
            const size_t size = pack_data_fn(
                m_block_height, m_block_width, m_width_align, m_pad_right_same, remaining_height, width, packed_data,
                data_buffer);
            data_buffer =
                data_buffer.subspan(remaining_height * round_up_division(width * data_type_size_in_bits(m_dtype), 8));
            packed_data = packed_data.subspan(size);
        }

        for (size_t i = 0; i < num_posts; ++i) {
            const size_t copy_size = remaining_height * data_type_size_in_bits(m_post_dtypes.at(i)) / 8;
            Span<const std::byte>& data = post_buffers.at(i);

            std::copy_n(data.begin(), copy_size, packed_data.begin());

            data = data.subspan(copy_size);
            packed_data = packed_data.subspan(m_block_height * data_type_size_in_bits(m_post_dtypes.at(i)) / 8);
        }
    }

    KAI_TEST_ASSERT(data_buffer.empty());
    KAI_TEST_ASSERT(packed_data.empty());

    return packed_buffer;
}

bool Block2dRowFormat::compare(
    Span<const size_t> shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
    Span<const std::byte> imp_buffer, Span<const std::byte> ref_buffer, MismatchHandler& handler) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(shape.size() == tile_coords.size());
    KAI_TEST_ASSERT(shape.size() == tile_shape.size());

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t tile_row = tile_coords.at(0);
    const size_t tile_col = tile_coords.at(1);

    const size_t tile_height = tile_shape.at(0);
    size_t tile_width = tile_shape.at(1);

    KAI_TEST_ASSERT(tile_row % m_block_height == 0);
    KAI_TEST_ASSERT(tile_col % m_block_width == 0);
    KAI_TEST_ASSERT(tile_row + tile_height == height || (tile_row + tile_height) % m_block_height == 0);
    KAI_TEST_ASSERT(tile_col + tile_width == width || (tile_col + tile_width) % m_block_width == 0);

    if (m_pad_right_same) {
        // If the tile includes the last block column, extends the tile to cover the right padding blocks.
        // In SAME padding mode, these blocks contain data even though they are outside the tile of interests.
        // If we don't extend the tile, there will be mismatched because these data points are outside the tile
        // and the data is not 0.
        tile_width = round_up_multiple(tile_col + tile_width, m_width_align) - tile_col;
    }

    const size_t num_pre_rows = m_pre_dtypes.size();
    std::vector<CompareFn> pre_compares;
    pre_compares.reserve(num_pre_rows);
    for (const DataType dtype : m_pre_dtypes) {
        pre_compares.emplace_back(make_compare_plain_2d(dtype));
    }

    const CompareFn data_compare = make_compare_plain_2d(m_dtype);

    const size_t num_post_rows = m_post_dtypes.size();
    std::vector<CompareFn> post_compares;
    post_compares.reserve(num_post_rows);
    for (const DataType dtype : m_post_dtypes) {
        post_compares.emplace_back(make_compare_plain_2d(dtype));
    }

    const size_t num_block_rows = round_up_division(height, m_block_height);
    const size_t num_block_cols_padded = round_up_multiple(width, m_width_align) / m_block_width;
    const size_t block_size = round_up_division(m_block_height * m_block_width * data_type_size_in_bits(m_dtype), 8);

    const size_t tile_block_col = tile_col / m_block_width;
    const size_t tile_num_block_cols = round_up_division(tile_width, m_block_width);

    size_t num_checks = 0;

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        const bool block_row_in_tile =
            tile_row <= block_row * m_block_height && tile_row + tile_height > block_row * m_block_height;

        for (size_t i = 0; i < num_pre_rows; ++i) {
            num_checks += pre_compares.at(i)(
                {1, m_block_height}, {0, 0}, {1, block_row_in_tile ? m_block_height : 0}, imp_buffer, ref_buffer,
                [&](std::ostream& os, Span<const size_t> coords) {
                    os << "Mismatched at block row " << block_row << ", prefix per-row component " << i << ", element "
                       << coords.at(1);
                },
                handler);

            imp_buffer = imp_buffer.subspan(m_block_height * data_type_size_in_bits(m_pre_dtypes.at(i)) / 8);
            ref_buffer = ref_buffer.subspan(m_block_height * data_type_size_in_bits(m_pre_dtypes.at(i)) / 8);
        }

        {
            num_checks += data_compare(
                {num_block_cols_padded, m_block_height * m_block_width}, {tile_block_col, 0},
                {tile_num_block_cols, block_row_in_tile ? m_block_height * m_block_width : 0}, imp_buffer, ref_buffer,
                [&](std::ostream& os, Span<const size_t> coords) {
                    os << "Mismatched at block row " << block_row << ", blocked data, block column " << coords.at(0)
                       << ", element " << coords.at(1);
                },
                handler);

            imp_buffer = imp_buffer.subspan(num_block_cols_padded * block_size);
            ref_buffer = ref_buffer.subspan(num_block_cols_padded * block_size);
        }

        for (size_t i = 0; i < num_post_rows; ++i) {
            num_checks += post_compares.at(i)(
                {1, m_block_height}, {0, 0}, {1, block_row_in_tile ? m_block_height : 0}, imp_buffer, ref_buffer,
                [&](std::ostream& os, Span<const size_t> coords) {
                    os << "Mismatched at block row " << block_row << ", postfix per-row component " << i << ", element "
                       << coords.at(1);
                },
                handler);

            imp_buffer = imp_buffer.subspan(m_block_height * data_type_size_in_bits(m_post_dtypes.at(i)) / 8);
            ref_buffer = ref_buffer.subspan(m_block_height * data_type_size_in_bits(m_post_dtypes.at(i)) / 8);
        }
    }

    KAI_TEST_ASSERT(imp_buffer.empty());
    KAI_TEST_ASSERT(ref_buffer.empty());

    return handler.success(num_checks);
}

void Block2dRowFormat::print(std::ostream& os, Span<const size_t> shape, Span<const std::byte> data) const {
    if (shape.empty()) {
        os << "None";
    } else {
        KAI_TEST_ASSERT(shape.size() == 2);

        const size_t height = shape.at(0);
        const size_t width = shape.at(1);

        const PrintFn data_printer = make_print_array(m_dtype);

        std::vector<PrintFn> pre_row_printers;
        pre_row_printers.reserve(m_pre_dtypes.size());

        for (const DataType dtype : m_pre_dtypes) {
            pre_row_printers.emplace_back(make_print_array(dtype));
        }

        std::vector<PrintFn> post_row_printers;
        post_row_printers.reserve(m_post_dtypes.size());

        for (const DataType dtype : m_post_dtypes) {
            post_row_printers.emplace_back(make_print_array(dtype));
        }

        const bool has_per_row_component = !m_pre_dtypes.empty() || !m_post_dtypes.empty();

        const size_t num_block_rows = round_up_division(height, m_block_height);
        const size_t num_block_cols_padded = round_up_multiple(width, m_width_align) / m_block_width;
        const size_t block_size =
            round_up_division(m_block_height * m_block_width * data_type_size_in_bits(m_dtype), 8);

        os << "[\n";

        for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
            if (has_per_row_component) {
                os << "  {\n";

                for (size_t i = 0; i < m_pre_dtypes.size(); ++i) {
                    os << "    \"row_data_" << i << "\": ";
                    pre_row_printers.at(i)(os, std::array{m_block_height}, data, 0);
                    data =
                        data.subspan(round_up_division(m_block_height * data_type_size_in_bits(m_pre_dtypes.at(i)), 8));
                    os << ",\n";
                }

                os << "    \"data\": [\n";

                for (size_t i = 0; i < num_block_cols_padded; ++i) {
                    data_printer(os, std::array{m_block_height * m_block_width}, data, 3);
                    data = data.subspan(block_size);
                    os << ",\n";
                }

                os << "    ],\n";

                for (size_t i = 0; i < m_post_dtypes.size(); ++i) {
                    os << "    \"row_data_" << i + m_pre_dtypes.size() << "\": ";
                    post_row_printers.at(i)(os, std::array{m_block_height}, data, 0);
                    data = data.subspan(
                        round_up_division(m_block_height * data_type_size_in_bits(m_post_dtypes.at(i)), 8));
                    os << ",\n";
                }

                os << "  },\n";
            } else {
                for (size_t i = 0; i < num_block_cols_padded; ++i) {
                    data_printer(os, std::array{m_block_height * m_block_width}, data, 1);
                    data = data.subspan(block_size);
                    os << ",\n";
                }
            }
        }

        KAI_TEST_ASSERT(data.empty());

        os << "]";
    }
}

bool Block2dRowFormat::operator==(const Format& other) const {
    const auto* rhs = dynamic_cast<const Block2dRowFormat*>(&other);
    return rhs != nullptr && m_dtype == rhs->m_dtype;
}

}  // namespace kai::test
