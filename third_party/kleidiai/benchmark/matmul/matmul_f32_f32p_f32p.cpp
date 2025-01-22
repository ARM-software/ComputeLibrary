//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/matmul/matmul_f32_f32p_f32p.hpp"

#include <benchmark/benchmark.h>

#include "benchmark/matmul/matmul_utils.hpp"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "test/common/cpu_info.hpp"

namespace kai::bench::matmul_f32_f32p_f32p {

const size_t seed_lhs = 4568;
const size_t seed_rhs = seed_lhs + 4;
const size_t seed_bias = seed_rhs + 4;

struct kai_matmul_ukernel_f32_f32p_f32p {
    kai_matmul_clamp_f32_f32p_f32p_ukernel ukernel;
    std::string name = {};
};

kai_matmul_ukernel_f32_f32p_f32p sme_variants[] = {
    {kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
     "matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa"},
};

struct kai_matmul_f32_f32p_f32p_sme {
    template <class... Args>
    void operator()(
        benchmark::State& state, kai_matmul_ukernel_f32_f32p_f32p variant, size_t m, size_t n, size_t k) const {
        const size_t lhs_size = m * k;
        const size_t rhs_size = n * k;
        const size_t bias_size = n;
        const size_t dst_size = m * n;

        float* lhs = new float[lhs_size];
        float* rhs = new float[rhs_size];
        float* bias = new float[bias_size];

        fill_uniform_random(m, k, lhs, seed_lhs);
        fill_uniform_random(k, n, rhs, seed_rhs);
        fill_uniform_random(1, n, bias, seed_bias);

        const size_t mr = variant.ukernel.get_mr();
        const size_t nr = variant.ukernel.get_nr();
        const size_t kr = variant.ukernel.get_kr();
        const size_t sr = variant.ukernel.get_sr();

        const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr);
        float* lhs_packed = new float[lhs_packed_size / sizeof(float)];

        const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k);

        float* rhs_packed = new float[rhs_packed_size / sizeof(float)];

        const size_t lhs_stride = k * sizeof(float);
        const size_t rhs_stride = n * sizeof(float);
        const size_t dst_stride_row = n * sizeof(float);
        const size_t dst_stride_col = sizeof(float);
        kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
            1, n, k, nr, kr, sr,  // Packing arguments
            rhs_stride, rhs, bias, NULL, rhs_packed, 0, NULL);

        kai_run_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr, 0, lhs, k * sizeof(float), lhs_packed);

        float* dst = new float[dst_size];
        for (auto _ : state) {
            // run matmul
            variant.ukernel.run_matmul(
                m, n, k,          // Dimensions
                lhs_packed,       // LHS
                rhs_packed,       // RHS packed
                dst,              // DST
                dst_stride_row,   // DST stride (row)
                dst_stride_col,   // DST stride (col)
                FLT_MIN, FLT_MAX  // Min and max for the clamp operation
            );
        }

        delete[] lhs;
        delete[] rhs;
        delete[] bias;
        delete[] rhs_packed;
        delete[] lhs_packed;
        delete[] dst;
    }
}; /* struct kai_matmul_f32_f32p_f32p_sme */

void RegisterBenchmarks(size_t m, size_t n, size_t k) {
    kai_matmul_f32_f32p_f32p_sme sme_kernel;
    if (kai::test::cpu_has_sme2()) {
        for (const auto& variant : sme_variants) {
            ::benchmark::RegisterBenchmark(variant.name, sme_kernel, variant, m, n, k)->Iterations(2000);
        }
    }
}
}  // namespace kai::bench::matmul_f32_f32p_f32p
