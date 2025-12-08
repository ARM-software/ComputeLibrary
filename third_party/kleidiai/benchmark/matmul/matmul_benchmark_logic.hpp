//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KLEIDIAI_BENCHMARK_MATMUL_MATMUL_BENCHMARK_LOGIC_HPP
#define KLEIDIAI_BENCHMARK_MATMUL_MATMUL_BENCHMARK_LOGIC_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "kai/kai_common.h"
#include "matmul_interface.hpp"
#include "matmul_runner.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

namespace kai::benchmark {
using Buffer = std::vector<uint8_t>;
using CpuRequirement = std::function<bool()>;

/// High level description of the matrix multiplication operation.
enum class MatMulOp : uint8_t {
    GEMM,
    GEMV,
};

/// Benchmarks a matrix multiplication micro-kernel.
///
/// @tparam MatMulInterface Interface of the matrix multiplication micro-kernel.
/// @param state            State for the benchmark to use.
/// @param matmul_interface Abstraction containing the micro-kernel to run.
/// @param dst_type         Output type of the micro-kernel. Required for the micro-kernel to make certain assumptions
///                         internally about the stride of the data.
/// @param matmul_op        Type of matrix multiplication operation.
/// @param is_cpu_supported Function that checks the CPU feature requirement to run this benchmark.
template <typename MatMulInterface>
void kai_benchmark_matmul(
    ::benchmark::State& state, const MatMulInterface matmul_interface, const DataType dst_type,
    const MatMulOp matmul_op, const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
    }

    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t bl = state.range(3);

    if (m > 1 && matmul_op == MatMulOp::GEMV) {
        state.SkipWithMessage("GEMV optimized for m=1 only");
    }

    if constexpr (
        std::is_same_v<MatMulInterface, MatMulBlockwiseDynamicQuantInterface> ||
        std::is_same_v<MatMulInterface, MatMulBlockwiseDynamicQuantGenericDstInterface>) {
        if (k % bl != 0) {
            state.SkipWithMessage("K must be a multiple of block size");
        }
    }

    // Create sufficiently large buffers
    size_t lhs_size = m * k * sizeof(uint64_t);
    size_t rhs_size = n * k * sizeof(uint64_t);
    size_t dst_size = m * n * sizeof(uint32_t);

    if (test::cpu_has_sme() || test::cpu_has_sme2()) {
        lhs_size *= kai_get_sme_vector_length_u32();
        rhs_size *= kai_get_sme_vector_length_u32();
        dst_size *= kai_get_sme_vector_length_u32();
    }

    const Buffer lhs(lhs_size);
    const Buffer rhs(rhs_size);
    Buffer dst(dst_size);

    MatMulRunner matmul_runner(matmul_interface, dst_type);
    matmul_runner.set_mnk(m, n, k);
    matmul_runner.set_bl(bl);

    for (auto _ : state) {
        matmul_runner.run(lhs.data(), rhs.data(), dst.data());
    }
}
}  // namespace kai::benchmark

#endif  // KLEIDIAI_BENCHMARK_MATMUL_MATMUL_BENCHMARK_LOGIC_HPP
