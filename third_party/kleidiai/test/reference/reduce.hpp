//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "test/common/buffer.hpp"

namespace kai::test {

class DataFormat;

/// Reduction operator.
enum class ReductionOperator : uint32_t {
    ADD,  ///< Addition.
};

/// Reduces the matrix value using addition.
///
/// @param[in] src Input data.
/// @param[in] src_format Input data format.
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
/// @param[in] dst_format Output data format.
/// @param[in] dimension Reduction dimension.
///
/// @return The reduced matrix.
Buffer reduce_add(
    const void* src, const DataFormat& src_format, size_t height, size_t width, const DataFormat& dst_format,
    size_t dimension);

/// Accumulates the matrix along the first dimension.
///
/// @tparam Value The data type of the matrix value.
/// @tparam Accumulator The data type of the accumulator.
///
/// @param[in] src The input data.
/// @param[in] height The number of rows of the input matrix.
/// @param[in] width The number of columns of the input matrix.
///
/// @return The vector containing the sum of each input matrix row.
template <typename Value, typename Accumulator>
Buffer reduce_add_x(const void* src, size_t height, size_t width);

/// Retrieve the minimum value in a provided matrix.
///
/// @tparam T Datatype of source matrix
///
/// @param[in] src The input data
/// @param[in] len The number of values within the source matrix.
///
/// @return The quantized data matrix, the quantization scale matrix and the quantization zero point matrix.
template <typename T>
T reduce_min(const void* src, size_t len);

/// Retrieve the maximum value in a provided matrix.
///
/// @tparam T Datatyoe of source matrix
///
/// @param[in] src The input data
/// @param[in] len The number of values within the source matrix.
///
/// @return The quantized data matrix, the quantization scale matrix and the quantization zero point matrix.
template <typename T>
T reduce_max(const void* src, size_t len);

}  // namespace kai::test
