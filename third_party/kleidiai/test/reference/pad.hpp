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
std::vector<uint8_t> pad_row(
    const void* data, size_t height, size_t width, size_t src_stride, size_t dst_stride, size_t dst_size,
    uint8_t val = 0);

}  // namespace kai::test
