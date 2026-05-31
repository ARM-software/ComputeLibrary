//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"

namespace kai::test {

/// Matrix multiplication operator configuration.
struct MatMulConfig {
    MatMulBiasMode bias_mode;
};

}  // namespace kai::test
