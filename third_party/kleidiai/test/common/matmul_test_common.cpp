//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_test_common.hpp"

#include <ostream>

namespace kai::test {
void PrintTo(const MatMulTestParams& param, std::ostream* os) {
    const auto& [method, shape, portion] = param;

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
    *os << "Method_" << method.name                                              //
        << "__M_" << shape.m << "__N_" << shape.n << "__K_" << shape.k           //
        << "__PortionStartRow_" << static_cast<int>(portion.start_row() * 1000)  //
        << "__PortionStartCol_" << static_cast<int>(portion.start_col() * 1000)  //
        << "__PortionHeight_" << static_cast<int>(portion.height() * 1000)       //
        << "__PortionWidth_" << static_cast<int>(portion.width() * 1000);
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
}
}  // namespace kai::test
