//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <tuple>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/functions/round.hpp"

namespace kai::test {

/// Dynamically quantizes the data using per-block linear quantization.
///
/// @param[in] height The height of data matrix.
/// @param[in] width The width of data matrix.
/// @param[in] block_height The height of quantization block.
/// @param[in] block_width The width of quantization block.
/// @param[in] fp_data The floating-point data.
/// @param[out] qdata The quantized data.
/// @param[out] qscale The quantization scale.
/// @param[out] qzp The quantization zero-point.
///
/// @return The quantized data, scale and zero-point.
using DynamicQuantizeLinearFn = std::tuple<Buffer, Buffer, Buffer> (*)(
    size_t height, size_t width, size_t block_height, size_t block_width, Span<const std::byte> fp_data);

/// Creates a dynamic quantization function using per-block linear asymmetric quantization.
///
/// @param[in] fp_dtype The data type of dequantized data.
/// @param[in] qdata_dtype The data type of quantized data.
/// @param[in] qscale_dtype The data type of quantization scale.
/// @param[in] qzp_dtype The data type of quantization zero-point.
/// @param[in] qdata_round_mode The rounding mode to calculate the quantized data.
/// @param[in] qzp_round_mode The rounding mode to calculate the quantization zero-point.
///
/// @return The function pointer.
[[nodiscard]] DynamicQuantizeLinearFn make_dynamic_asymmetric_quantize_linear(
    DataType fp_dtype, DataType qdata_dtype, DataType qscale_dtype, DataType qzp_dtype, RoundMode qdata_round_mode,
    RoundMode qzp_round_mode);

/// Creates a dynamic quantization function using per-block linear symmetric quantization.
///
/// @param[in] fp_dtype The data type of dequantized data.
/// @param[in] qdata_dtype The data type of quantized data.
/// @param[in] qscale_dtype The data type of quantization scale.
/// @param[in] qdata_round_mode The rounding mode to calculate the quantized data.
///
/// @return The function pointer.
[[nodiscard]] DynamicQuantizeLinearFn make_dynamic_symmetric_quantize_linear(
    DataType fp_dtype, DataType qdata_dtype, DataType qscale_dtype, RoundMode qdata_round_mode);

}  // namespace kai::test
