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
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp_qsi4cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cache.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_format.hpp"
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
#include "test/reference/pad.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

// Using BFloat truncate implementation (BFloat16<false>) to match existing packing/inference

namespace kai::test {

using Bf16Qai8Qsi4CacheDataId = std::tuple<  //
    MatMulShape,                             //
    DataFormat,                              // lhs format
    DataFormat,                              // rhs format
    DataFormat,                              // bias format
    float                                    // clamp_ratio
    >;

struct Bf16Qai8Qsi4CacheData {
    Buffer ref_dst_nt_t;
    Buffer ref_dst_nt_nt;
    Buffer ref_rhs_qsi4_nt_t;
    Buffer ref_rhs_qsi4_nt_nt;
    Buffer ref_rhs_scales;
    Buffer ref_lhs_bf16;
    Buffer ref_biases_buf;
    Range<float> clamp_nt_nt;
    Range<float> clamp_nt_t;
};

template <>
Bf16Qai8Qsi4CacheData ReferenceGenerator<Bf16Qai8Qsi4CacheDataId, Bf16Qai8Qsi4CacheData>::generate_reference(
    const Bf16Qai8Qsi4CacheDataId& data_id) {
    auto [shape, lhs_format, rhs_format, bias_format, clamp_ratio] = data_id;

    size_t M = shape.m;
    size_t N = shape.n;
    size_t K = shape.k;

    static size_t seed = 1;

    bool has_bias = bias_format.data_type() != DataType::UNKNOWN;
    Buffer lhs = fill_matrix_random(shape.m, shape.k, lhs_format, seed++);
    Buffer ref_rhs = fill_matrix_random(shape.n, shape.k, rhs_format, seed++);
    Buffer bias = has_bias ? fill_matrix_random(1, shape.n, bias_format, seed++) : Buffer();

    Bf16Qai8Qsi4CacheData out;
    // For reference implementation, Casting BF16 input to FP32 type and FP32 output back to BFP16 because the matmul
    // implementation works with FP32 accumulation and casts the result to BFP16
    const auto ref_lhs = cast<float, BFloat16<false>>(
        lhs.data(),  //
        lhs.size() * 8 / size_in_bits<BFloat16<false>>);

    // Transposed(nxk) RHS dimensions
    const size_t ref_rhs_qsi4_nxk_stride = K;

    // Non-Transposed(kxn) RHS dimensions
    const size_t ref_rhs_qsi4_kxn_stride = round_up_multiple(N, 2);
    const size_t ref_rhs_qsi4_kxn_size_bytes = round_up_division(K * ref_rhs_qsi4_kxn_stride, 2);

    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = K;
    lhs_qinfo.dst_type = DataType::QAI8;
    lhs_qinfo.scale_type = DataType::FP32;
    lhs_qinfo.zero_point_type = DataType::I32;
    const auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref_lhs.data(), DataType::FP32, M, K, lhs_qinfo);

    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = K;
    rhs_qinfo.dst_type = DataType::QSI4;
    rhs_qinfo.scale_type = DataType::FP32;
    auto [ref_rhs_quant_t, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    auto ref_rhs_qsi4 = transpose_with_padding<Int4>(
        ref_rhs_quant_t.data(), N, K, ref_rhs_qsi4_nxk_stride, ref_rhs_qsi4_kxn_stride, ref_rhs_qsi4_kxn_size_bytes);

    const auto ref_dst_nt_nt = matmul_clamp_nt_nt<int8_t, float, int32_t, Int4, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), K,
        ref_rhs_qsi4.data(), rhs_qoutputs.scales.data(), nullptr, K, has_bias ? bias.data() : nullptr,
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    const auto [clamp_min_nt_nt, clamp_max_nt_nt] = find_clamp_range<float>(ref_dst_nt_nt.data(), M * N, clamp_ratio);
    out.ref_rhs_qsi4_nt_nt = std::move(ref_rhs_qsi4);

    const auto ref_dst_float_nt_nt = clamp<float>(ref_dst_nt_nt.data(), M * N, clamp_min_nt_nt, clamp_max_nt_nt);

    auto ref_dst_nt_nt_bf16 =
        cast<BFloat16<false>, float>(ref_dst_float_nt_nt.data(), ref_dst_float_nt_nt.size() * 8 / size_in_bits<float>);
    out.ref_dst_nt_nt = std::move(ref_dst_nt_nt_bf16);

    out.clamp_nt_nt = {clamp_min_nt_nt, clamp_max_nt_nt};

    const auto ref_dst_nt_t =
        matmul_nt_t_quantized<int8_t, float, int32_t, Int4, float, int32_t, float, float, int32_t, float>(
            M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), 1, K,
            ref_rhs_quant_t.data(), rhs_qoutputs.scales.data(), nullptr, 1, K, has_bias ? bias.data() : nullptr,
            nullptr, nullptr, 1);

    const auto [clamp_min_nt_t, clamp_max_nt_t] = find_clamp_range<float>(ref_dst_nt_t.data(), M * N, clamp_ratio);
    out.ref_rhs_qsi4_nt_t = std::move(ref_rhs_quant_t);
    const auto ref_dst_nt_t_float = clamp<float>(ref_dst_nt_t.data(), M * N, clamp_min_nt_t, clamp_max_nt_t);

    auto ref_dst_nt_t_bf16 =
        cast<BFloat16<false>, float>(ref_dst_nt_t_float.data(), ref_dst_nt_t_float.size() * 8 / size_in_bits<float>);
    out.ref_dst_nt_t = std::move(ref_dst_nt_t_bf16);
    out.clamp_nt_t = {clamp_min_nt_t, clamp_max_nt_t};
    out.ref_lhs_bf16 = std::move(lhs);
    out.ref_biases_buf = std::move(bias);
    out.ref_rhs_scales = std::move(rhs_qoutputs.scales);
    return out;
}

static const std::array<UkernelVariant<kai_matmul_clamp_bf16_qai8dxp_qsi4cxp_ukernel>, 2>
    variants_kai_matmul_clamp_bf16_qai8dxp_qsi4cxp = {{
        {UKERNEL_MATMUL_VARIANT(clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod),
         "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod", cpu_has_dotprod_and_bf16},
        {UKERNEL_MATMUL_VARIANT(clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm),
         "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm", cpu_has_i8mm_and_bf16},
    }};

class MatMulTest_bf16_qai8dxp_qsi4cxp : public ::testing::TestWithParam<MatMulTestPortionedParamsWithBias> {};

TEST_P(MatMulTest_bf16_qai8dxp_qsi4cxp, EndToEnd_RHS_NxK) {
    const auto& [variant_index, matmul_shape, portion, has_bias] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_bf16_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    const auto lhs_format = DataFormat(DataType::BF16);
    const auto rhs_format = DataFormat(DataType::FP32);
    const auto bias_format = has_bias ? DataFormat(DataType::FP32) : DataFormat(DataType::UNKNOWN);

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    float clamp_ratio = 0.8F;

    const Bf16Qai8Qsi4CacheDataId testdata_id = {matmul_shape, lhs_format, rhs_format, bias_format, clamp_ratio};
    const Bf16Qai8Qsi4CacheData& testdata = getV<Bf16Qai8Qsi4CacheDataId, Bf16Qai8Qsi4CacheData>(testdata_id);

    const auto& ref_lhs_bf16 = testdata.ref_lhs_bf16;
    const auto& ref_rhs_qsi4 = testdata.ref_rhs_qsi4_nt_t;
    const auto& ref_biases_buf = testdata.ref_biases_buf;
    const auto& ref_rhs_scales = testdata.ref_rhs_scales;
    const auto& ref_dst = testdata.ref_dst_nt_t;
    auto [clamp_min, clamp_max] = testdata.clamp_nt_t;
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(M, K, mr, kr, sr);
    Buffer imp_packed_lhs_buf = Buffer(imp_packed_lhs_size);

    auto lhs_stride = K * sizeof(uint16_t);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
        rect.height() /* m */, K, mr, kr, sr, 0, ref_lhs_bf16.data() + lhs_offset, lhs_stride,
        reinterpret_cast<uint8_t*>(imp_packed_lhs_buf.data()) + lhs_packed_offset);

    const auto ref_rhs_qsi4_padded = pad_row<Int4>(
        ref_rhs_qsi4.data(), N, K, K, round_up_multiple(K, 2), round_up_division(N * round_up_multiple(K, 2), 2));

    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(N, K, nr, kr, sr);
    Buffer imp_packed_rhs_buf = Buffer(imp_packed_rhs_size);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);
    // Runs the RHS packing micro-kernel.
    kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 0;

    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
        1, N, K, nr, kr, sr, reinterpret_cast<const uint8_t*>(ref_rhs_qsi4_padded.data()),
        has_bias ? reinterpret_cast<const float*>(ref_biases_buf.data()) : nullptr,
        reinterpret_cast<const float*>(ref_rhs_scales.data()), reinterpret_cast<uint8_t*>(imp_packed_rhs_buf.data()), 0,
        &params);

    const auto dst_stride_row = N * sizeof(uint16_t);
    const auto dst_stride_col = sizeof(uint16_t);
    const auto dst_offset =
        ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst_buf = Buffer(imp_dst_size);

    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, reinterpret_cast<const uint8_t*>(imp_packed_lhs_buf.data()) + lhs_matmul_offset,
        reinterpret_cast<const uint8_t*>(imp_packed_rhs_buf.data()) + rhs_matmul_offset,
        reinterpret_cast<uint8_t*>(imp_dst_buf.data()) + dst_offset, dst_stride_row, dst_stride_col, clamp_min,
        clamp_max);

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::BF16);
    const auto success =
        compare(reinterpret_cast<const uint8_t*>(imp_dst_buf.data()), ref_dst.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);
}

TEST_P(MatMulTest_bf16_qai8dxp_qsi4cxp, EndToEnd_RHS_KxN) {
    const auto& [variant_index, matmul_shape, portion, has_bias] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_bf16_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    const auto lhs_format = DataFormat(DataType::BF16);
    const auto rhs_format = DataFormat(DataType::FP32);
    const auto bias_format = has_bias ? DataFormat(DataType::FP32) : DataFormat(DataType::UNKNOWN);

    // Generates input data.
    float clamp_ratio = 0.8F;

    const Bf16Qai8Qsi4CacheDataId testdata_id = {matmul_shape, lhs_format, rhs_format, bias_format, clamp_ratio};
    const Bf16Qai8Qsi4CacheData& testdata = getV<Bf16Qai8Qsi4CacheDataId, Bf16Qai8Qsi4CacheData>(testdata_id);

    const auto& ref_lhs_bf16 = testdata.ref_lhs_bf16;
    const auto& ref_rhs_qsi4 = testdata.ref_rhs_qsi4_nt_nt;
    const auto& ref_biases_buf = testdata.ref_biases_buf;
    const auto& ref_rhs_scales = testdata.ref_rhs_scales;
    const auto& ref_dst = testdata.ref_dst_nt_nt;
    auto [clamp_min, clamp_max] = testdata.clamp_nt_nt;
    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    size_t lhs_stride = K * sizeof(uint16_t);

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(M, K, mr, kr, sr);
    Buffer imp_packed_lhs_buf = Buffer(imp_packed_lhs_size);
    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
        rect.height() /* m */, K, mr, kr, sr, 0 /* m_idx_start*/, ref_lhs_bf16.data() + lhs_offset, lhs_stride,
        reinterpret_cast<uint8_t*>(imp_packed_lhs_buf.data()) + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    //   * Generates the 4-bit unsigned symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto ref_rhs_qsi4_padded = pad_row<Int4>(
        ref_rhs_qsi4.data(), K, N, N, round_up_multiple(N, 2), round_up_division(K * round_up_multiple(N, 2), 2));
    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(N, K, nr, kr, sr);

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    Buffer imp_packed_rhs_buf = Buffer(imp_packed_rhs_size);
    kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 0;
    kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
        1, N, K, nr, kr, sr, reinterpret_cast<const uint8_t*>(ref_rhs_qsi4_padded.data()),
        has_bias ? reinterpret_cast<const float*>(ref_biases_buf.data()) : nullptr,
        reinterpret_cast<const float*>(ref_rhs_scales.data()), imp_packed_rhs_buf.data(), 0, &params);

    const auto dst_stride = N * sizeof(uint16_t);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(uint16_t);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst_buf = Buffer(imp_dst_size);

    const auto dst_stride_row = N * sizeof(uint16_t);
    const auto dst_stride_col = sizeof(uint16_t);

    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, reinterpret_cast<const uint8_t*>(imp_packed_lhs_buf.data()) + lhs_matmul_offset,
        reinterpret_cast<const uint8_t*>(imp_packed_rhs_buf.data()) + rhs_matmul_offset,
        reinterpret_cast<uint8_t*>(imp_dst_buf.data()) + dst_offset, dst_stride_row, dst_stride_col, clamp_min,
        clamp_max);

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::BF16);
    const auto success =
        compare(reinterpret_cast<const uint8_t*>(imp_dst_buf.data()), ref_dst.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_bf16_qai8dxp_qsi4cxp,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_bf16_qai8dxp_qsi4cxp.size()),
        testing::Values(
            MatMulShape{1, 2, 32},    //
            MatMulShape{1, 3, 32},    //
            MatMulShape{1, 4, 32},    //
            MatMulShape{1, 5, 32},    //
            MatMulShape{3, 3, 32},    //
            MatMulShape{4, 4, 32},    //
            MatMulShape{5, 5, 32},    //
            MatMulShape{32, 64, 64},  //
            MatMulShape{16, 32, 64},  //
            MatMulShape{8, 32, 64},   //
            MatMulShape{15, 32, 32},  //
            MatMulShape{77, 99, 64},  //
            MatMulShape{77, 99, 66},  //
            MatMulShape{77, 99, 31}),
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
    [](const auto& info) -> std::string {
        const auto variant_idx = std::get<0>(info.param);
        const auto& name = variants_kai_matmul_clamp_bf16_qai8dxp_qsi4cxp[variant_idx].name;
        return test_description(
            name, std::get<MatMulShape>(info.param), std::get<2>(info.param), std::get<3>(info.param));
    });

}  // namespace kai::test
