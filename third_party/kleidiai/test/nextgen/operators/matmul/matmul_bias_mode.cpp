//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"

#include <string>

#include "test/common/assert.hpp"

namespace kai::test {

std::string matmul_bias_mode_name(MatMulBiasMode bias_mode) {
    switch (bias_mode) {
        case MatMulBiasMode::NO_BIAS:
            return "no";

        case MatMulBiasMode::PER_N:
            return "col";

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

}  // namespace kai::test
