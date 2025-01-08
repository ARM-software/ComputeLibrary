//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "test/common/data_type.hpp"

namespace kai::test {

/// Converts each element of the array to the specified data type.
///
/// @tparam DstType The data type to cast into.
/// @tparam SrcType The data type to cast from.
///
/// @param[in] src The source data.
/// @param[in] length The number of elements.
///
/// @return A new data buffer containing casted values.
template <typename DstType, typename SrcType>
std::vector<uint8_t> cast(const void* src, size_t length);

/// Converts each element of the source matrix to the new data type.
///
/// @param[in] src Source matrix data buffer.
/// @param[in] src_dt Data type of the source matrix.
/// @arapm[in] dst_dt Data type of the destination matrix.
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
///
/// @return The result matrix containing data in the destination data type.
std::vector<uint8_t> cast(const void* src, DataType src_dt, DataType dst_dt, size_t height, size_t width);

/// Converts each element of the source data from 4-bit signed symmetric quantized
/// to 4-bit unsigned symmetric quantized.
///
/// @param[in] src The source data.
/// @param[in] length The number of elements.
///
/// @return A new data buffer with converted values.
std::vector<uint8_t> cast_qsu4_qsi4(const void* src, size_t length);

}  // namespace kai::test
