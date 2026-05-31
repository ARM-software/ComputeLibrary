//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"

namespace kai::test {

/// Pads the rows in matrix.
/// Works for non-packed and packed using provided strides.
///
/// @param[in] data Data buffer.
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
/// @param[in] src_stride Stride of source buffer.
/// @param[in] dst_stride Stride for destination buffer.
/// @param[in] dst_size Size of destination buffer.
/// @param[in] pad_value Default value for the padding.
///
/// @return The padded matrix.
///
template <typename T>
Buffer pad_row(
    const void* data, size_t height, size_t width, size_t src_stride, size_t dst_stride, size_t dst_size,
    uint8_t val = 0);

/// Creates a padded matrix from an input matrix.
///
/// @param[in] data The input data buffer.
/// @param[in] height The number of input rows.
/// @param[in] width The number of input columns.
/// @param[in] pad_left The number of element padded to the left.
/// @param[in] pad_top The number of element padded to the top.
/// @param[in] pad_right The number of element padded to the right.
/// @param[in] pad_bottom The number of element padded to the bottom.
/// @param[in] pad_value The padding value.
///
/// @return The padded matrix.
template <typename T>
Buffer pad_matrix(
    const void* data, size_t height, size_t width, size_t pad_left, size_t pad_top, size_t pad_right, size_t pad_bottom,
    T pad_value);

}  // namespace kai::test
