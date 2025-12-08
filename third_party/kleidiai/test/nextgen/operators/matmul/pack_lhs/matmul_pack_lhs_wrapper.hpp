//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "test/nextgen/harness/kernel_wrapper.hpp"

namespace kai::test {

/// Creates a wrapper for kai_lhs_quant_pack_qai8dxp_f32 micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper> create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32();

/// Creates a wrapper for kai_lhs_quant_pack_qai8dxp_f32 micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper> create_matmul_lhs_quant_pack_qai8dxp1x4_f32();

}  // namespace kai::test
