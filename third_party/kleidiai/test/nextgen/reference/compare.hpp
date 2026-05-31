//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <ostream>

#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"

namespace kai::test {

/// Compares two data buffers.
///
/// The data inside the tile of interests of the two buffers are compared.
/// The data in the buffer under test that is outside the tile of intersts must be 0.
///
/// @param[in] shape The size of the multidimensional data.
/// @param[in] tile_coords The starting coordinate of the tile to be compared.
/// @param[in] tile_shape The size of the tile to be compared.
/// @param[in] imp_buffer The data buffer under test.
/// @param[in] ref_buffer The reference data buffer.
/// @param[in] report_fn The function to report the mismatch location.
/// @param[in] handler The mismatch handler.
///
/// @return The number of elements being checked.
using CompareFn = size_t (*)(
    Span<const size_t> shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
    Span<const std::byte> imp_buffer, Span<const std::byte> ref_buffer,
    const std::function<void(std::ostream& os, Span<const size_t> coords)>& report_fn, MismatchHandler& handler);

/// Gets the function to compare two plain 2D data buffers for the specified data type.
///
/// @param[in] dtype The data type.
///
/// @return The function pointer.
[[nodiscard]] CompareFn make_compare_plain_2d(DataType dtype);

}  // namespace kai::test
