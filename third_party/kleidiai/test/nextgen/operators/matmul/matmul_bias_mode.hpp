//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

namespace kai::test {

/// Bias mode.
enum class MatMulBiasMode : uint8_t {
    NO_BIAS,  ///< No bias.
    PER_N,    ///< Column bias.
};

/// Gets the name of the bias mode.
[[nodiscard]] std::string matmul_bias_mode_name(MatMulBiasMode bias_mode);

}  // namespace kai::test
