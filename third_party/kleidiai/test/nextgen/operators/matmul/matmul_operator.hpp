//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/quantization/quantizer.hpp"

namespace kai::test {

/// Matrix multiplication operator.
struct MatMulOperator {
    std::string_view name;

    bool (*is_cpu_supported)();
    bool (*is_shape_suitable)(size_t shape_m, size_t shape_n, size_t shape_k);

    std::vector<MatMulBiasMode> supported_bias_modes;

    std::optional<std::unique_ptr<Quantizer>> lhs_quant;
    std::optional<std::unique_ptr<Quantizer>> rhs_quant;
    std::optional<std::unique_ptr<Quantizer>> bias_quant;

    DataType acc_dtype;
    DataType dst_dtype;

    std::optional<std::unique_ptr<KernelWrapper>> pack_lhs;
    std::optional<std::unique_ptr<KernelWrapper>> pack_rhs;
    std::unique_ptr<KernelWrapper> matmul;
};

[[nodiscard]] Span<const MatMulOperator> get_available_matmul_operators();

}  // namespace kai::test
