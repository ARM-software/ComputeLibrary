//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/quantize.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/functions/round.hpp"

namespace kai::test {

namespace {

template <typename FpData, typename QData, typename QZp, RoundMode ZP_ROUND_MODE>
std::tuple<FpData, FpData, QZp> get_scale_zero_point_from_range(FpData min_value, FpData max_value) {
    const FpData q_min = numeric_lowest<QData>;
    const FpData q_max = numeric_highest<QData>;

    if (min_value > 0) {
        min_value = 0;
    }

    if (max_value < 0) {
        max_value = 0;
    }

    // The reason for computing the inverted scale first is to make it bit-perfect with quantized packing
    // micro-kernels. If those micro-kernels don't do it this way anymore, it makes more sense to calculate
    // the scale directly.
    const FpData inv_scale = max_value != min_value ? (q_max - q_min) / (max_value - min_value) : 1.0F;
    const FpData scale = 1.0F / inv_scale;

    const FpData scaled_min = min_value / scale;
    const FpData scaled_max = max_value / scale;

    const FpData zero_point_f = -(scaled_min + q_min) < scaled_max + q_max ? scaled_min - q_min : scaled_max - q_max;
    const QZp zero_point = -static_cast<QZp>(round<FpData, ZP_ROUND_MODE>(zero_point_f));

    return {scale, inv_scale, zero_point};
}

template <typename FpData, typename QData>
std::tuple<FpData, FpData> get_scale_from_max_abs(FpData max_abs) {
    const FpData scale = max_abs / static_cast<FpData>((1 << (size_in_bits<QData> - 1)) - 1);
    const FpData inv_scale = static_cast<FpData>(1) / scale;

    return {scale, inv_scale};
}

template <typename FpData, typename QData, RoundMode QDATA_ROUND_MODE>
QData quantize_symmetric(FpData value, FpData inv_scale) {
    int32_t quantized_value = round<FpData, QDATA_ROUND_MODE>(value * inv_scale);

    if (is_unsigned<QData>) {
        quantized_value += 1 << (size_in_bits<QData> - 1);
    }

    return static_cast<QData>(std::clamp<int32_t>(quantized_value, numeric_lowest<QData>, numeric_highest<QData>));
}

template <typename FpData, typename QData, typename QZp, RoundMode QDATA_ROUND_MODE>
[[nodiscard]] QData quantize_asymmetric(FpData value, FpData inv_scale, QZp zero_point) {
    const QZp quantized_value = static_cast<QZp>(round<FpData, QDATA_ROUND_MODE>(value * inv_scale)) + zero_point;
    return static_cast<QData>(std::clamp<QZp>(quantized_value, numeric_lowest<QData>, numeric_highest<QData>));
}

template <
    typename FpData, typename QData, typename QScale, typename QZp, RoundMode QDATA_ROUND_MODE,
    RoundMode QZP_ROUND_MODE>
[[nodiscard]] std::tuple<Buffer, Buffer, Buffer> dynamic_asymmetric_quantize_linear(
    size_t height, size_t width, size_t block_height, size_t block_width, Span<const std::byte> fp_data) {
    const size_t num_block_rows = round_up_division(height, block_height);
    const size_t num_block_cols = round_up_division(width, block_width);

    Buffer qdata(height * round_up_division(width * size_in_bits<QData>, 8), 0);
    Buffer qscale(num_block_rows * num_block_cols * size_in_bits<QScale> / 8, 0);
    static_assert(size_in_bits<QScale> % 8 == 0);
    Buffer qzp(num_block_rows * num_block_cols * size_in_bits<QZp> / 8);

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        for (size_t block_col = 0; block_col < num_block_cols; ++block_col) {
            const size_t block_idx = block_row * num_block_cols + block_col;
            const size_t start_row = block_row * block_height;
            const size_t start_col = block_col * block_width;
            const size_t size_row = std::min(block_height, height - start_row);
            const size_t size_col = std::min(block_width, width - start_col);

            // Finds the value range.
            FpData min_value = numeric_highest<FpData>;
            FpData max_value = numeric_lowest<FpData>;

            for (size_t row = 0; row < size_row; ++row) {
                for (size_t col = 0; col < size_col; ++col) {
                    const FpData value = read_2d<FpData>(fp_data, width, start_row + row, start_col + col);
                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                }
            }

            // Computes the quantization information.
            const auto [qscale_value, inv_qscale_value, qzp_value] =
                get_scale_zero_point_from_range<FpData, QData, QZp, QZP_ROUND_MODE>(min_value, max_value);

            write_array<QScale>(qscale, block_idx, qscale_value);
            write_array<QZp>(qzp, block_idx, qzp_value);

            // Quantizes the data.
            for (size_t row = 0; row < size_row; ++row) {
                for (size_t col = 0; col < size_col; ++col) {
                    const FpData value = read_2d<FpData>(fp_data, width, start_row + row, start_col + col);
                    const QData qvalue =
                        quantize_asymmetric<FpData, QData, QZp, QDATA_ROUND_MODE>(value, inv_qscale_value, qzp_value);
                    write_2d<QData>(qdata, width, start_row + row, start_col + col, qvalue);
                }
            }
        }
    }

    return {std::move(qdata), std::move(qscale), std::move(qzp)};
}

template <typename FpData, typename QData, typename QScale, RoundMode QDATA_ROUND_MODE>
[[nodiscard]] std::tuple<Buffer, Buffer, Buffer> dynamic_symmetric_quantize_linear(
    size_t height, size_t width, size_t block_height, size_t block_width, Span<const std::byte> fp_data) {
    const size_t num_block_rows = round_up_division(height, block_height);
    const size_t num_block_cols = round_up_division(width, block_width);

    Buffer qdata(height * round_up_division(width * size_in_bits<QData>, 8), 0);
    Buffer qscale(num_block_rows * num_block_cols * size_in_bits<QScale> / 8, 0);
    static_assert(size_in_bits<QScale> % 8 == 0);

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        for (size_t block_col = 0; block_col < num_block_cols; ++block_col) {
            const size_t block_idx = block_row * num_block_cols + block_col;
            const size_t start_row = block_row * block_height;
            const size_t start_col = block_col * block_width;
            const size_t size_row = std::min(block_height, height - start_row);
            const size_t size_col = std::min(block_width, width - start_col);

            // Finds the value range.
            FpData max_abs = numeric_lowest<FpData>;

            for (size_t row = 0; row < size_row; ++row) {
                for (size_t col = 0; col < size_col; ++col) {
                    const FpData value = read_2d<FpData>(fp_data, width, start_row + row, start_col + col);
                    max_abs = std::max(max_abs, std::abs(value));
                }
            }

            // Computes the quantization information.
            const auto [qscale_value, inv_qscale_value] = get_scale_from_max_abs<FpData, QData>(max_abs);
            write_array<QScale>(qscale, block_idx, qscale_value);

            // Quantizes the data.
            for (size_t row = 0; row < size_row; ++row) {
                for (size_t col = 0; col < size_col; ++col) {
                    const FpData value = read_2d<FpData>(fp_data, width, start_row + row, start_col + col);
                    const QData qvalue = quantize_symmetric<FpData, QData, QDATA_ROUND_MODE>(value, inv_qscale_value);
                    write_2d<QData>(qdata, width, start_row + row, start_col + col, qvalue);
                }
            }
        }
    }

    return {std::move(qdata), std::move(qscale), Buffer()};
}

}  // namespace

DynamicQuantizeLinearFn make_dynamic_asymmetric_quantize_linear(
    DataType fp_dtype, DataType qdata_dtype, DataType qscale_dtype, DataType qzp_dtype, RoundMode qdata_round_mode,
    RoundMode qzp_round_mode) {
    const auto params =
        std::make_tuple(fp_dtype, qdata_dtype, qscale_dtype, qzp_dtype, qdata_round_mode, qzp_round_mode);

    if (params ==
        std::make_tuple(
            DataType::FP32, DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT)) {
        return dynamic_asymmetric_quantize_linear<
            float, int8_t, float, int32_t, RoundMode::TIE_AWAY, RoundMode::CURRENT>;
    }

    KAI_TEST_ERROR("Not implemented.");
}

DynamicQuantizeLinearFn make_dynamic_symmetric_quantize_linear(
    DataType fp_dtype, DataType qdata_dtype, DataType qscale_dtype, RoundMode qdata_round_mode) {
    const auto params = std::make_tuple(fp_dtype, qdata_dtype, qscale_dtype, qdata_round_mode);

    if (params == std::make_tuple(DataType::FP32, DataType::U4, DataType::FP32, RoundMode::CURRENT)) {
        return dynamic_symmetric_quantize_linear<float, UInt4, float, RoundMode::CURRENT>;
    }

    KAI_TEST_ERROR("Not implemented.");
}

}  // namespace kai::test
