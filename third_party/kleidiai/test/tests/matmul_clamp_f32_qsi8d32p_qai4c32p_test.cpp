//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p_qai4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s1s0_f32_f32_f32_neon.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {
// Interface for the LHS and RHS packed size and packing micro-kernels
using kai_get_lhs_packed_size_func_t = decltype(&kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon);
using kai_get_rhs_packed_size_func_t =
    decltype(&kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon);
using kai_get_lhs_packed_offset_func_t = decltype(&kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon);
using kai_get_rhs_packed_offset_func_t =
    decltype(&kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon);
using kai_get_lhs_offset_func_t = decltype(&kai_get_lhs_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon);
using kai_get_rhs_offset_func_t = decltype(&kai_get_rhs_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon);
using kai_run_lhs_pack_func_t = decltype(&kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon);
using kai_run_rhs_pack_func_t = decltype(&kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon);

// Micro-kernel interface
struct kai_qai4c32p_pack_functions {
    kai_get_rhs_packed_size_func_t packed_size;
    kai_get_rhs_packed_offset_func_t get_packed_offset;
    kai_get_rhs_offset_func_t get_offset;
    kai_run_rhs_pack_func_t run_pack;
};

struct kai_qsi8d32p_pack_functions {
    kai_get_lhs_packed_size_func_t packed_size;
    kai_get_lhs_packed_offset_func_t get_packed_offset;
    kai_get_lhs_offset_func_t get_offset;
    kai_run_lhs_pack_func_t run_pack;
};

static const std::array<
    UkernelMatmulPackVariant<
        kai_matmul_clamp_f32_qsi8d32p_qai4c32p_ukernel, kai_qsi8d32p_pack_functions, kai_qai4c32p_pack_functions>,
    8>
    variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p = {
        {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod, cpu_has_dotprod,
             lhs_quant_pack_qsi8d32pscalef32_f32_neon, rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon, true),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qsi8d32pscalef32_f32_neon,
             rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon, true),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod, cpu_has_dotprod,
             lhs_quant_pack_qsi8d32pscalef32_f32_neon, rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon, true),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod, cpu_has_dotprod,
             lhs_quant_pack_qsi8d32pscalef32_f32_neon, rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon, true),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot, cpu_has_sme2, lhs_quant_pack_qsi8d32pscalef32_f32_neon,
             rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s1s0_f32_f32_f32_neon, false),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa, cpu_has_sme2,
             lhs_quant_pack_qsi8d32pscalef32_f32_neon, rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s1s0_f32_f32_f32_neon,
             false),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot, cpu_has_sme2, lhs_quant_pack_qsi8d32pscalef32_f32_neon,
             rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, true),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa, cpu_has_sme2,
             lhs_quant_pack_qsi8d32pscalef32_f32_neon, rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon,
             true)}};

// Executes the LHS packing micro-kernel.
static inline std::tuple<Buffer, size_t> pack_lhs_qsi8d32p(
    const kai_qsi8d32p_pack_functions& pack_interface, size_t M, size_t K, size_t bl, size_t mr, size_t kr, size_t sr,
    const Buffer& lhs_values_qsi8, size_t stride, size_t rect_start_row, size_t rect_height) {
    const auto imp_packed_lhs_size = pack_interface.packed_size(M, K, bl, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size, 0);

    auto lhs_offset = pack_interface.get_offset(rect_start_row, stride);
    auto lhs_packed_offset = pack_interface.get_packed_offset(rect_start_row, K, bl, mr, kr, sr);

    abi_check(
        pack_interface.run_pack, rect_height, K, bl, mr, kr, sr, 0,
        reinterpret_cast<const float*>(lhs_values_qsi8.data() + lhs_offset), stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    return {std::move(imp_packed_lhs), lhs_packed_offset};
}

// Executes the RHS packing micro-kernel.
static inline std::tuple<Buffer, size_t> pack_rhs_qai4c32p(
    const kai_qai4c32p_pack_functions& pack_interface, size_t N, size_t K, size_t bl, size_t nr, size_t kr, size_t sr,
    const Buffer& rhs_values_qai4, const bool has_bias, const Buffer& biases, const Buffer& rhs_scales,
    const Buffer& rhs_zp, bool s0s1_input, size_t rect_start_row) {
    // Cast to unsigned int
    auto rhs_qau4s1s0 = cast_qsu4_qsi4(rhs_values_qai4.data(), N * K);

    const auto imp_packed_rhs_size = pack_interface.packed_size(N, K, nr, kr, bl);
    Buffer imp_packed_rhs(imp_packed_rhs_size);
    auto rhs_packed_offset = pack_interface.get_packed_offset(rect_start_row, K, nr, kr, bl);

    // Runs the RHS packing micro-kernel.
    kai_rhs_pack_nxk_qai4c32p_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;

    abi_check(
        pack_interface.run_pack, 1, N, K, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(s0s1_input ? convert_s0s1_s1s0(rhs_qau4s1s0).data() : rhs_qau4s1s0.data()),
        rhs_zp.data(), has_bias ? biases.data() : nullptr, rhs_scales.data(), imp_packed_rhs.data(), 0, &params);

    return {std::move(imp_packed_rhs), rhs_packed_offset};
}

class MatMulTest_f32_qsi8d32p_qai4c32p : public ::testing::TestWithParam<MatMulTestPortionedParamsWithBias_WithBL> {};

TEST_P(MatMulTest_f32_qsi8d32p_qai4c32p, LhsPackedWithSameBlockdepth) {
    // Verify LHS quant and pack int8 kernel behaves same for int4 and int8 matmul kernels,
    // when the block-depth is same for different values of kr, sr.

    const auto& [variant_index, matmul_shape, bl, portion, has_bias] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.at(variant_index);

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    if (K % bl != 0) {
        GTEST_SKIP() << "K must be a multiple of bl";
    }

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    auto lhs_stride = K * sizeof(float);

    auto [imp_packed_lhs, lhs_packed_offset] = pack_lhs_qsi8d32p(
        ukernel_variant.lhs_pack_interface, M, K, bl, mr, kr, sr, ref_lhs, lhs_stride, lhs_start_row, rect.height());

    const size_t kr_qsi8 = kr / sr;
    const size_t sr_qsi8 = 1;

    auto [imp_packed_lhs_qsi8, lhs_qsi8_packed_offset] = pack_lhs_qsi8d32p(
        ukernel_variant.lhs_pack_interface, M, K, bl, mr, kr_qsi8, sr_qsi8, ref_lhs, lhs_stride, lhs_start_row,
        rect.height());

    ASSERT_EQ(lhs_qsi8_packed_offset, lhs_packed_offset);

    auto* imp_packed_lhs_ptr = reinterpret_cast<const uint8_t*>(imp_packed_lhs.data());
    auto* imp_packed_lhs_qsi8_ptr = reinterpret_cast<const uint8_t*>(imp_packed_lhs_qsi8.data());
    for (size_t i = 0; i < ukernel_variant.lhs_pack_interface.packed_size(M, K, bl, mr, kr, sr); i++) {
        ASSERT_EQ(imp_packed_lhs_ptr[i], imp_packed_lhs_qsi8_ptr[i]);
    }
}

TEST_P(MatMulTest_f32_qsi8d32p_qai4c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape, bl, portion, has_bias] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.at(variant_index);

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    if (K % bl != 0) {
        GTEST_SKIP() << "K must be a multiple of bl";
    }

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    if (mr == 1 && M > 1) {
        GTEST_SKIP() << "Kernel does not support M != 1";
    }

    auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    Buffer ref_biases;

    if (has_bias) {
        ref_biases = fill_random<float>(N, seed + 2);
    }
    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit symmetric quantization.
    //   * Quantizes the RHS matrix using 4-bit asymmetric quantization.
    //   * Performs GEMM.
    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = bl;
    lhs_qinfo.dst_type = DataType::QSI8;
    lhs_qinfo.scale_type = DataType::FP32;
    const auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref_lhs.data(), DataType::FP32, M, K, lhs_qinfo);

    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = bl;
    rhs_qinfo.dst_type = DataType::QAI4;
    rhs_qinfo.scale_type = DataType::FP32;
    rhs_qinfo.zero_point_type = DataType::I32;
    const auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    const auto ref_dst_no_clamp =
        matmul_nt_t_quantized<int8_t, float, int32_t, Int4, float, int32_t, float, float, int32_t, float>(
            M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), nullptr, 1, bl, ref_rhs_quant.data(),
            rhs_qoutputs.scales.data(), rhs_qoutputs.zero_points.data(), 1, bl, has_bias ? ref_biases.data() : nullptr,
            nullptr, nullptr, 1);

    // Clamps the reference output.
    const auto clamp_ratio = 0.8F;
    const auto [clamp_min, clamp_max] = find_clamp_range<float>(ref_dst_no_clamp.data(), M * N, clamp_ratio);
    const auto ref_dst = clamp<float>(ref_dst_no_clamp.data(), M * N, clamp_min, clamp_max);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    auto [imp_packed_lhs, lhs_packed_offset] = pack_lhs_qsi8d32p(
        ukernel_variant.lhs_pack_interface, M, K, bl, mr, kr, sr, ref_lhs, K * sizeof(float), lhs_start_row,
        rect.height());
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    // Prepare the offsets as the RHS packing micro-kernel expects the scaled zero-points in float.
    const size_t num_blocks_per_row = round_up_division(K, bl);
    const size_t ref_zp_size = N * num_blocks_per_row;
    const size_t ref_zp_size_in_bytes = ref_zp_size * sizeof(float);
    Buffer ref_rhs_zp_f32(ref_zp_size_in_bytes);
    for (size_t i = 0; i < ref_zp_size; ++i) {
        reinterpret_cast<float*>(ref_rhs_zp_f32.data())[i] =
            -reinterpret_cast<const int32_t*>(rhs_qoutputs.zero_points.data())[i] *
            reinterpret_cast<const float*>(rhs_qoutputs.scales.data())[i];
    }

    const auto rhs_start_row = rect.start_col();
    auto [imp_packed_rhs, rhs_packed_offset] = pack_rhs_qai4c32p(
        ukernel_variant.rhs_pack_interface, N, K, bl, nr, kr, sr, ref_rhs_quant, has_bias, ref_biases,
        rhs_qoutputs.scales, ref_rhs_zp_f32, ukernel_variant.rhs_s0s1_input, rhs_start_row);

    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    const auto dst_stride_row = N * sizeof(float);
    const auto dst_stride_col = sizeof(float);
    const auto dst_offset =
        ukernel_variant.ukernel.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.ukernel.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    abi_check(
        ukernel_variant.ukernel.interface.run_matmul, rect.height(), rect.width(), K, bl,
        imp_packed_lhs.data() + lhs_matmul_offset, imp_packed_rhs.data() + rhs_matmul_offset,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col, clamp_min, clamp_max);

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::FP32);
    const auto success = compare(imp_dst.data(), ref_dst.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);
}
INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qsi8d32p_qai4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.size()),
        testing::Values(
            MatMulShape{1, 64, 32},    //
            MatMulShape{1, 63, 32},    //
            MatMulShape{1, 65, 32},    //
            MatMulShape{1, 64, 64},    //
            MatMulShape{1, 64, 128},   //
            MatMulShape{1, 128, 32},   //
            MatMulShape{1, 128, 128},  //
            MatMulShape{1, 2, 32},     //
            MatMulShape{1, 3, 32},     //
            MatMulShape{1, 4, 32},     //
            MatMulShape{1, 5, 32},     //
            MatMulShape{3, 3, 32},     //
            MatMulShape{4, 4, 32},     //
            MatMulShape{5, 5, 32},     //
            MatMulShape{32, 128, 32},  //
            MatMulShape{15, 64, 64},   //
            MatMulShape{17, 64, 64},   //
            MatMulShape{16, 63, 64},   //
            MatMulShape{16, 64, 64},   //
            MatMulShape{16, 65, 64},   //
            MatMulShape{32, 64, 64},   //
            MatMulShape{16, 32, 64},   //
            MatMulShape{8, 32, 64},    //
            MatMulShape{15, 32, 32},   //
            MatMulShape{77, 99, 64}),
        testing::Values(32, 64),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),         // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),      // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),      // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8),     // Somewhere Middle
            MatrixPortion(0.75, 0.75, 1, 1),   // Bottom-right corner.
            MatrixPortion(0.75, 0, 1, 1),      // Partial rows
            MatrixPortion(0.4, 0.5, 0.6, 0.8)  // Somewhere Middle
            ),
        testing::Bool()),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.at(variant_idx).ukernel.name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto bl = std::get<2>(info.param);
        const auto portion = std::get<3>(info.param);
        const auto has_bias = std::get<4>(info.param);

        std::ostringstream sstream;
        sstream << name << "__";
        PrintTo(shape, &sstream);
        sstream << "__BL_" << bl << "_";
        if (has_bias) {
            sstream << "_withBias_";
        } else {
            sstream << "_noBias_";
        }
        if (variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.at(variant_idx).rhs_s0s1_input) {
            sstream << "_RHS_s0s1__";
        } else {
            sstream << "_RHS_s1s0__";
        }
        PrintTo(portion, &sstream);

        return sstream.str();
    });

}  // namespace kai::test
