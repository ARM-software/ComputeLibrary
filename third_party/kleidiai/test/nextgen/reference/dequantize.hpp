//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"

namespace kai::test {

/// Dequantizes the data using per-block linear quantization.
///
/// @param[in] height The height of data matrix.
/// @param[in] width The width of data matrix.
/// @param[in] block_height The height of quantization block.
/// @param[in] block_width The width of quantization block.
/// @param[in] qdata The quantized data.
/// @param[in] qscale The quantization scale.
/// @param[in] qzp The quantization zero-point.
///
/// @return The dequantized data.
using DequantizeLinearFn = Buffer (*)(
    size_t height, size_t width, size_t block_height, size_t block_width, Span<const std::byte> qdata,
    Span<const std::byte> qscale, Span<const std::byte> qzp);

/// Creates a dequantization function using per-block linear quantization.
///
/// @param[in] fp_dtype The data type of dequantized data.
/// @param[in] qdata_dtype The data type of quantized data.
/// @param[in] qscale_dtype The data type of quantization scale.
/// @param[in] qzp_dtype The data type of quantization zero-point.
///
/// @return The function pointer.
[[nodiscard]] DequantizeLinearFn make_dequantize_linear(
    DataType fp_dtype, DataType qdata_dtype, DataType qscale_dtype, DataType qzp_dtype = DataType::UNKNOWN);

}  // namespace kai::test
