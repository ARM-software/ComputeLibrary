//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/dequantize.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/common/type_traits.hpp"

namespace kai::test {

namespace {

template <typename FpData, typename QData, typename QScale, typename QZp>
[[nodiscard]] Buffer dequantize_linear(
    size_t height, size_t width, size_t block_height, size_t block_width, Span<const std::byte> qdata,
    Span<const std::byte> qscale, Span<const std::byte> qzp) {
    Buffer fp_data(height * round_up_division(width * size_in_bits<FpData>, 8));

    const size_t quant_width = round_up_division(width, block_width);

    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            const QData qdata_value = read_2d<QData>(qdata, width, row, col);
            FpData fp_value = static_cast<FpData>(qdata_value);

            if constexpr (!std::is_same_v<QZp, void>) {
                const QZp qzp_value = read_2d<QZp>(qzp, quant_width, row / block_height, col / block_width);
                fp_value -= static_cast<FpData>(qzp_value);
            } else if constexpr (is_unsigned<QData>) {
                static_assert(size_in_bits<QData> <= 64);
                constexpr FpData zp_value = static_cast<FpData>(static_cast<uint64_t>(1) << (size_in_bits<QData> - 1));
                fp_value -= zp_value;
            }

            const QScale qscale_value = read_2d<QScale>(qscale, quant_width, row / block_height, col / block_width);
            fp_value *= static_cast<FpData>(qscale_value);

            write_2d<FpData>(fp_data.view(), width, row, col, fp_value);
        }
    }

    return fp_data;
}

}  // namespace

DequantizeLinearFn make_dequantize_linear(
    DataType fp_dtype, DataType qdata_dtype, DataType qscale_dtype, DataType qzp_dtype) {
    const auto dtypes = std::make_tuple(fp_dtype, qdata_dtype, qscale_dtype, qzp_dtype);

    if (dtypes == std::make_tuple(DataType::FP32, DataType::I8, DataType::FP32, DataType::I32)) {
        return dequantize_linear<float, int8_t, float, int32_t>;
    }

    if (dtypes == std::make_tuple(DataType::FP32, DataType::U4, DataType::FP32, DataType::UNKNOWN)) {
        return dequantize_linear<float, UInt4, float, void>;
    }

    KAI_TEST_ERROR("Not implemented.");
}

}  // namespace kai::test
