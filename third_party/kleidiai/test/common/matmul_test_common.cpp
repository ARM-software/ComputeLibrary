//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_test_common.hpp"

#include <sstream>

namespace kai::test {

std::ostream& operator<<(std::ostream& os, const MatMulShape& shape) {
    return os << "[m=" << shape.m << ", n=" << shape.n << ", k=" << shape.k << "]";
}

void PrintTo(const MatMulTestParams& param, std::ostream* os) {
    const auto& [method, shape, portion, bias_mode] = param;

    *os << method.name << "__";
    PrintTo(shape, os);
    *os << "__";
    PrintTo(portion, os);
    PrintTo(bias_mode, os);
}

void PrintTo(const MatMulShape& shape, std::ostream* os) {
    *os << "M_" << shape.m << "__N_" << shape.n << "__K_" << shape.k;
}

void PrintTo(const BiasMode& bias_mode, std::ostream* os) {
    // Preserve legacy test names
    if (bias_mode == BiasMode::INTERNAL) {
        *os << "__NullBias";
    }
}

void PrintTo(const MatrixPortion& portion, std::ostream* os) {
    *os << "Portion__R_" << static_cast<int>(portion.start_row() * 1000)  //
        << "__C_" << static_cast<int>(portion.start_col() * 1000)         //
        << "__H_" << static_cast<int>(portion.height() * 1000)            //
        << "__W_" << static_cast<int>(portion.width() * 1000);
}

std::string test_description(
    const std::string_view& name, const MatMulShape& shape, const MatrixPortion& portion, bool bias) {
    std::ostringstream os;

    os << name << "__";
    PrintTo(shape, &os);
    os << "__";
    PrintTo(portion, &os);
    if (bias) {
        os << "__Bias";
    }

    return os.str();
}

}  // namespace kai::test
