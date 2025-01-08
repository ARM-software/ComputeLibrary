//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <iosfwd>
#include <string_view>

namespace kai::test {

class DataFormat;

/// Prints the matrix data to the output stream.
///
/// @param[in] os Output stream to write the data to.
/// @param[in] name Matrix name.
/// @param[in] data Data buffer.
/// @param[in] format Data format.
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
void print_matrix(
    std::ostream& os, std::string_view name, const void* data, const DataFormat& format, size_t height, size_t width);

}  // namespace kai::test
