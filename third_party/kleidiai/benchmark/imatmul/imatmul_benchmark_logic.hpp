//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <test/common/cpu_info.hpp>
#include <vector>

#include "imatmul_interface.hpp"
#include "imatmul_runner.hpp"
#include "kai/kai_common.h"

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

/// Benchmarks an indirect matrix multiplication micro-kernel.
///
/// @tparam ImatmulInterface Interface of the indirect matrix multiplication micro-kernel.
/// @param state            State for the benchmark to use.
/// @param imatmul_interface Abstraction containing the micro-kernel to run.
/// @param dst_type         Output type of the micro-kernel.
/// @param is_cpu_supported Function that checks the CPU feature requirement to run this benchmark.
template <typename ImatmulInterface>
void kai_benchmark_imatmul(
    ::benchmark::State& state, const ImatmulInterface imatmul_interface, const DataType dst_type,
    const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
    }

    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k_chunk_count = state.range(2);
    const size_t k_chunk_length = state.range(3);
    const size_t k = k_chunk_count * k_chunk_length;

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

    ImatmulRunner imatmul_runner(imatmul_interface, dst_type);
    imatmul_runner.set_mnk_chunked(m, n, k_chunk_count, k_chunk_length);

    for (auto _ : state) {
        imatmul_runner.run(lhs.data(), rhs.data(), dst.data());
    }
}

}  // namespace kai::benchmark
