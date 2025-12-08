//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>

#include "test/common/data_type.hpp"
#include "test/common/span.hpp"

namespace kai::test {

/// Prints a plain multidimensional array to the output stream.
///
/// @param[in] os The output stream to print to.
/// @param[in] shape The size of multidimensional array.
/// @param[in] data The data buffer.
/// @param[in] level The number of indentation levels.
using PrintFn = void (*)(std::ostream& os, Span<const size_t> shape, Span<const std::byte> data, size_t level);

/// Gets the pointer to the print function for the specified data type.
///
/// @param[in] dtype The data type.
///
/// @return The function pointer.
[[nodiscard]] PrintFn make_print_array(DataType dtype);

}  // namespace kai::test
