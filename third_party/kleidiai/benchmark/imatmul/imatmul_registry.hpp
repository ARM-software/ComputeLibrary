//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::benchmark {

/// Registers indirect matrix multiplication (imatmul) micro-kernels for benchmarking.
///
/// @param m    Number of rows in the LHS matrix.
/// @param n    Number of columns in the RHS matrix.
/// @param k_chunk_count   Number of K chunks.
/// @param k_chunk_length  Length of each K chunk.
void RegisteriMatMulBenchmarks(size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length);

}  // namespace kai::benchmark
