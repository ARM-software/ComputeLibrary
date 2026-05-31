//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KLEIDIAI_BENCHMARK_MATMUL_MATMUL_REGISTRY_HPP
#define KLEIDIAI_BENCHMARK_MATMUL_MATMUL_REGISTRY_HPP

#include <cstddef>

#include "test/common/matmul_test_common.hpp"

namespace kai::benchmark {
using test::MatMulShape;

/// Registers matrix multiplication micro-kernels for benchmarking.
///
/// @param shape Shape with M, N and K dimensions describing the matrix multiplication problem.
/// @param bl    Block size. Used for micro-kernels with dynamic blockwise quantization.
void RegisterMatMulBenchmarks(const MatMulShape& shape, size_t bl);

}  // namespace kai::benchmark

#endif  // KLEIDIAI_BENCHMARK_MATMUL_MATMUL_REGISTRY_HPP
