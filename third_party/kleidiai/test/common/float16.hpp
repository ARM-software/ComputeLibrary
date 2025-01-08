//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iosfwd>

namespace kai::test {

/// Half-precision floating-point.
using Float16 = __fp16;

/// Writes the value to the output stream.
///
/// @param[in] os Output stream to be written to.
/// @param[in] value Value to be written.
///
/// @return The output stream.
std::ostream& operator<<(std::ostream& os, Float16 value);

}  // namespace kai::test
