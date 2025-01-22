//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <benchmark/benchmark.h>

#include <climits>
#include <string>

#include "benchmark/matmul/matmul_utils.hpp"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "test/common/cpu_info.hpp"

namespace kai::bench::matmul_f32_qa8dxp_qs4cxp {

const size_t seed_lhs = 4568;
const size_t seed_rhs = seed_lhs + 4;
const size_t seed_bias = seed_rhs + 4;

struct kai_matmul_ukernel_f32_qa8dxp_qs4cxp {
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel ukernel;
    std::string name = {};
};

struct kai_matmul_f32_qai8_qsi4 {
    template <class... Args>
    void operator()(
        benchmark::State& state, kai_matmul_ukernel_f32_qa8dxp_qs4cxp variant, size_t m, size_t n, size_t k) const {
        const size_t lhs_native_size_f32 = m * k * sizeof(float);
        const size_t rhs_native_size_f32 = n * k * sizeof(float);
        const size_t rhs_native_size_qs4cx = n * (k / 2) * sizeof(uint8_t);
        const size_t rhs_scales_size_f32 = n * sizeof(float);

        // Allocate the memory
        uint8_t* lhs_native_mtx_f32 = new uint8_t[lhs_native_size_f32];
        uint8_t* rhs_native_mtx_f32 = new uint8_t[rhs_native_size_f32];
        uint8_t* rhs_native_mtx_qs4cx = new uint8_t[rhs_native_size_qs4cx];
        uint8_t* rhs_scales_f32 = new uint8_t[rhs_scales_size_f32];

        fill_uniform_random(m, k, (float*)lhs_native_mtx_f32, seed_lhs);
        fill_uniform_random(n, k, (float*)rhs_native_mtx_f32, seed_rhs);

        quant_qs4cx_f32(n, k, (const float*)rhs_native_mtx_f32, rhs_native_mtx_qs4cx, (float*)rhs_scales_f32);

        delete[] rhs_native_mtx_f32;

        // Get the packing parameters
        const size_t mr = variant.ukernel.get_mr();
        const size_t nr = variant.ukernel.get_nr();
        const size_t kr = variant.ukernel.get_kr();
        const size_t sr = variant.ukernel.get_sr();

        // Get the size in bytes for the packed matrices
        const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
        const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, nr, kr, sr);
        const size_t dst_size = variant.ukernel.get_dst_size(m, n);

        // Allocate the matrices
        uint8_t* lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
        uint8_t* rhs_packed_mtx_qs4cx = new uint8_t[rhs_packed_size];
        uint8_t* dst_act_mtx_f32 = new uint8_t[dst_size];

        // If the RHS matrix contains constant values, the packing can be performed
        // only once
        struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;

        // RHS packing
        kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
            1, n, k, nr, kr, sr,                     // Packing arguments
            (const uint8_t*)(rhs_native_mtx_qs4cx),  // RHS
            NULL,                                    // Bias
            (const float*)(rhs_scales_f32),          // Scale
            rhs_packed_mtx_qs4cx,                    // RHS packed
            0, &params);

        // LHS packing
        kai_run_lhs_quant_pack_qai8dxp_f32(
            m, k, mr, kr, sr, 0,               // Packing arguments
            (const float*)lhs_native_mtx_f32,  // LHS
            k * sizeof(float),                 // LHS stride
            lhs_packed_mtx_qa8dx);             // LHS packed

        const size_t dst_stride = n * sizeof(float);
        const size_t lhs_offset = variant.ukernel.get_lhs_packed_offset(0, k);
        const size_t rhs_offset = variant.ukernel.get_rhs_packed_offset(0, k);
        const size_t dst_offset = variant.ukernel.get_dst_offset(0, 0, dst_stride);

        const void* lhs_ptr = (const void*)((const char*)lhs_packed_mtx_qa8dx + lhs_offset);
        const void* rhs_ptr = (const void*)((const char*)rhs_packed_mtx_qs4cx + rhs_offset);
        float* dst_ptr = (float*)((uint8_t*)dst_act_mtx_f32 + dst_offset);

        // Matmul
        for (auto _ : state) {
            variant.ukernel.run_matmul(
                m, n, k,           // Dimensions
                lhs_ptr,           // LHS packed
                rhs_ptr,           // RHS packed
                dst_ptr,           // DST
                dst_stride,        // DST stride (row)
                sizeof(float),     // DST stride (col)
                -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
            );
        }

        delete[] lhs_packed_mtx_qa8dx;
        delete[] rhs_packed_mtx_qs4cx;
        delete[] dst_act_mtx_f32;

        delete[] lhs_native_mtx_f32;
        delete[] rhs_native_mtx_qs4cx;
        delete[] rhs_scales_f32;
    }
}; /* kai_matmul_f32_qai8_qsi4 */

namespace dotprod {
kai_matmul_ukernel_f32_qa8dxp_qs4cxp ukernel_variants[] = {
    {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     "matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     "matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     "matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod"},
};

void RegisterBenchmarks(size_t m, size_t n, size_t k) {
    kai::bench::matmul_f32_qa8dxp_qs4cxp::kai_matmul_f32_qai8_qsi4 matmul_f32;
    if (kai::test::cpu_has_dotprod()) {
        for (const auto& variant : dotprod::ukernel_variants) {
            ::benchmark::RegisterBenchmark(variant.name, matmul_f32, variant, m, n, k);
        }
    }
}

} /* namespace dotprod */

namespace i8mm {
kai_matmul_ukernel_f32_qa8dxp_qs4cxp ukernel_variants[] = {
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm"},
};

void RegisterBenchmarks(size_t m, size_t n, size_t k) {
    kai::bench::matmul_f32_qa8dxp_qs4cxp::kai_matmul_f32_qai8_qsi4 matmul_f32;
    if (kai::test::cpu_has_i8mm()) {
        for (const auto& variant : i8mm::ukernel_variants) {
            ::benchmark::RegisterBenchmark(variant.name, matmul_f32, variant, m, n, k);
        }
    }
}

};  /* namespace i8mm */
};  // namespace kai::bench::matmul_f32_qa8dxp_qs4cxp
