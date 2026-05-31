//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/compare.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <ostream>

#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/span.hpp"
#include "test/common/type_traits.hpp"

namespace kai::test {

namespace {

/// Calculates the absolute and relative errors.
///
/// @param[in] imp Value under test.
/// @param[in] ref Reference value.
///
/// @return The absolute error and relative error.
template <typename T>
std::tuple<float, float> calculate_error(T imp, T ref) {
    const auto imp_f = static_cast<float>(imp);
    const auto ref_f = static_cast<float>(ref);

    const auto abs_error = std::abs(imp_f - ref_f);
    const auto rel_error = ref_f != 0 ? abs_error / std::abs(ref_f) : 0.0F;

    return {abs_error, rel_error};
}

template <typename T>
size_t compare_plain_2d(
    Span<const size_t> shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
    Span<const std::byte> imp_buffer, Span<const std::byte> ref_buffer,
    const std::function<void(std::ostream& os, Span<const size_t> coords)>& report_fn, MismatchHandler& handler) {
    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t start_row = tile_coords.at(0);
    const size_t start_col = tile_coords.at(1);

    const size_t tile_height = tile_shape.at(0);
    const size_t tile_width = tile_shape.at(1);

    const size_t end_row = start_row + tile_height;
    const size_t end_col = start_col + tile_width;

    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            const bool in_tile = row >= start_row && row < end_row && col >= start_col && col < end_col;
            const size_t index = row * width + col;

            const T imp_value = read_array<T>(imp_buffer, index);
            const T ref_value = in_tile ? read_array<T>(ref_buffer, index) : static_cast<T>(0);

            const auto [abs_err, rel_err] = calculate_error<T>(imp_value, ref_value);

            if (abs_err != 0 || rel_err != 0) {
                // If the mismatch happens outside the tile, it's an error straightaway
                // since these are expected to be 0 and the kernel is likely to write out-of-bound.
                // If the mismatch happens inside the tile, the mismatch handler makes the decision
                // based on the absolute error and relative error.

                if (!in_tile) {
                    handler.mark_as_failed();
                }

                const auto notifying = !in_tile || handler.handle_data(abs_err, rel_err);

                if (notifying) {
                    report_fn(std::cerr, std::array{row, col});
                    std::cerr << ": actual = " << displayable(imp_value) << ", expected = " << displayable(ref_value)
                              << "\n";
                }
            }
        }
    }

    return tile_height * tile_width;
}

}  // namespace

CompareFn make_compare_plain_2d(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return compare_plain_2d<float>;

        case DataType::I32:
            return compare_plain_2d<int32_t>;

        case DataType::I8:
            return compare_plain_2d<int8_t>;

        case DataType::I4:
            return compare_plain_2d<Int4>;

        default:
            KAI_TEST_ERROR("Not implemented.");
    }
}

}  // namespace kai::test
