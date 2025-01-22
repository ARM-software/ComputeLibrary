//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "test/common/cpu_info.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pad.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

static const std::array<UkernelVariant<kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel>, 4>
    variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp = {{
        UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod, cpu_has_dotprod),
        UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod, cpu_has_dotprod),
        UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod, cpu_has_dotprod),
        UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm, cpu_has_i8mm),
    }};

class MatMulTest_f32_qai8dxp_qsi8cxp : public UkernelVariantTest {};

TEST_P(MatMulTest_f32_qai8dxp_qsi8cxp, EndToEnd_RHS_nxk_qsi8cx) {
    auto& [variant_index, matmul_shape] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP();
    }

    const uint64_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 8-bit symmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    const auto [ref_rhs_qsi8, ref_rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, float>(ref_rhs.data(), N, K, K);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, float, int32_t, int8_t, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), K, ref_rhs_qsi8.data(),
        ref_rhs_scales.data(), nullptr, K, ref_biases.data(), std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    std::vector<uint8_t> imp_packed_lhs(imp_packed_lhs_size);
    kai_run_lhs_quant_pack_qai8dxp_f32(
        M, K, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data()), K * sizeof(float), imp_packed_lhs.data());

    // Runs the RHS packing micro-kernel.
    //   * Generates the 8-bit signed symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);

    std::vector<uint8_t> imp_packed_rhs(imp_packed_rhs_size);
    const kai_rhs_pack_qsi8cx_params params{.lhs_zero_point = 1, .scale_multiplier = 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, N, K, nr, kr, sr, reinterpret_cast<const int8_t*>(ref_rhs_qsi8.data()),
        reinterpret_cast<const float*>(ref_biases.data()), reinterpret_cast<const float*>(ref_rhs_scales.data()),
        imp_packed_rhs.data(), 0, &params);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    std::vector<uint8_t> imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        M, N, K, imp_packed_lhs.data(), imp_packed_rhs.data(), reinterpret_cast<float*>(imp_dst.data()),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < M; ++y) {
        for (size_t x = 0; x < N; ++x) {
            const auto imp_value = read_array<float>(imp_dst.data(), y * N + x);
            const auto ref_value = read_array<float>(ref_dst.data(), y * N + x);
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

TEST_P(MatMulTest_f32_qai8dxp_qsi8cxp, EndToEnd_RHS_kxn_qsi8cx) {
    auto& [variant_index, matmul_shape] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP();
    }

    const uint64_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Transposed(nxk) RHS dimensions
    const size_t ref_rhs_qsi8_nxk_stride = K;

    // Non-Transposed(kxn) RHS dimensions
    const size_t ref_rhs_qsi8_kxn_stride = N;
    const size_t ref_rhs_qsi8_kxn_size_bytes = K * ref_rhs_qsi8_kxn_stride;

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 8-bit symmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    const auto [ref_rhs_qsi8_transposed, ref_rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, float>(ref_rhs.data(), N, K, K);

    const auto ref_rhs_qsi8 = transpose_with_padding<int8_t>(
        ref_rhs_qsi8_transposed.data(), N, K, ref_rhs_qsi8_nxk_stride, ref_rhs_qsi8_kxn_stride,
        ref_rhs_qsi8_kxn_size_bytes);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, float, int32_t, int8_t, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), K, ref_rhs_qsi8.data(),
        ref_rhs_scales.data(), nullptr, K, ref_biases.data(), std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    std::vector<uint8_t> imp_packed_lhs(imp_packed_lhs_size);
    kai_run_lhs_quant_pack_qai8dxp_f32(
        M, K, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data()), K * sizeof(float), imp_packed_lhs.data());

    // Runs the RHS packing micro-kernel.
    //   * Generates the 8-bit signed symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);

    std::vector<uint8_t> imp_packed_rhs(imp_packed_rhs_size);
    const kai_rhs_pack_qsi8cx_params params{.lhs_zero_point = 1, .scale_multiplier = 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, N, K, nr, kr, sr, reinterpret_cast<const int8_t*>(ref_rhs_qsi8.data()),
        reinterpret_cast<const float*>(ref_biases.data()), reinterpret_cast<const float*>(ref_rhs_scales.data()),
        imp_packed_rhs.data(), 0, &params);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    std::vector<uint8_t> imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        M, N, K, imp_packed_lhs.data(), imp_packed_rhs.data(), reinterpret_cast<float*>(imp_dst.data()),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < M; ++y) {
        for (size_t x = 0; x < N; ++x) {
            const auto imp_value = read_array<float>(imp_dst.data(), y * N + x);
            const auto ref_value = read_array<float>(ref_dst.data(), y * N + x);
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qai8dxp_qsi8cxp,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.size()),
        testing::Values(MatMulShape{17, 33, 67}, MatMulShape{19, 35, 63}, MatMulShape{1, 27, 31})));

}  // namespace kai::test
