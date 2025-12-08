//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "imatmul_registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <test/common/cpu_info.hpp>
#include <test/common/data_type.hpp>

#include "imatmul_benchmark_logic.hpp"
#include "imatmul_interface.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

// Micro-kernels to register for benchmarking

// imatmul_clamp_f16_f16p_f16p
#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"

// imatmul_clamp_f32_f32p_f32p
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

// imatmul_clamp_qai8_qai8p_qsi8cxp
#include "kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"

namespace kai::benchmark {
using DataType = test::DataType;

// imatmul_clamp_f16_f16p_f16p
inline constexpr ImatmulBaseInterface kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa_interface{
    .run_imatmul = kai_run_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa,
};

inline constexpr ImatmulBaseInterface kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa_interface{
    .run_imatmul = kai_run_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
};

// imatmul_clamp_f16_f16_f16p
inline constexpr ImatmulBaseInterface kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa_interface{
    .run_imatmul = kai_run_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa,
};

inline constexpr ImatmulBaseInterface kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_interface{
    .run_imatmul = kai_run_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
};

// imatmul_clamp_qai8_qai8p_qsi8cxp
inline constexpr ImatmulStaticQuantInterface
    kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_interface{
        .run_imatmul = kai_run_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa,
    };

inline constexpr ImatmulStaticQuantInterface
    kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_interface{
        .run_imatmul = kai_run_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
    };

inline const std::array imatmul_benchmarks{
    // imatmul_clamp_f16_f16p_f16p
    RegisterBenchmark(
        "kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa", kai_benchmark_imatmul<ImatmulBaseInterface>,
        kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa_interface, DataType::FP16, test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa", kai_benchmark_imatmul<ImatmulBaseInterface>,
        kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa_interface, DataType::FP16, test::cpu_has_sme),

    // imatmul_clamp_f16_f16_f16p
    RegisterBenchmark(
        "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa", kai_benchmark_imatmul<ImatmulBaseInterface>,
        kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa_interface, DataType::FP32, test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa", kai_benchmark_imatmul<ImatmulBaseInterface>,
        kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_interface, DataType::FP32, test::cpu_has_sme),

    // imatmul_clamp_qai8_qai8p_qsi8cxp
    RegisterBenchmark(
        "kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa",
        kai_benchmark_imatmul<ImatmulStaticQuantInterface>,
        kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_interface, DataType::QAI8, test::cpu_has_sme),
    RegisterBenchmark(
        "kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa",
        kai_benchmark_imatmul<ImatmulStaticQuantInterface>,
        kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_interface, DataType::QAI8,
        test::cpu_has_sme2),

};

void RegisteriMatMulBenchmarks(size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length) {
    for (const auto& benchmark : imatmul_benchmarks) {
        benchmark
            ->Args(
                {static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k_chunk_count),
                 static_cast<int64_t>(k_chunk_length)})
            ->ArgNames({"m", "n", "k_chunk_count", "k_chunk_length"});
    }
}
}  // namespace kai::benchmark
