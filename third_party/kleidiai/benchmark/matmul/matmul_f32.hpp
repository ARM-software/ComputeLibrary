//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>

namespace kai::bench::matmul_f32_qa8dxp_qs4cxp {

namespace dotprod {
void RegisterBenchmarks(size_t m, size_t n, size_t k);
}; /* namespace dotprod */

namespace i8mm {
void RegisterBenchmarks(size_t m, size_t n, size_t k);
}; /* namespace i8mm */

};  // namespace kai::bench::matmul_f32_qa8dxp_qs4cxp
