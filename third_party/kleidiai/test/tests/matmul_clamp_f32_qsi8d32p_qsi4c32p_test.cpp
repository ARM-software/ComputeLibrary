//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/float16.hpp"
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
using kai_get_lhs_packed_size_func_t = decltype(&kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32);
using kai_get_rhs_packed_size_func_t = decltype(&kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0);
using kai_get_lhs_packed_offset_func_t = decltype(&kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32);
using kai_get_rhs_packed_offset_func_t =
    decltype(&kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0);
using kai_get_lhs_offset_func_t = decltype(&kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32);
using kai_get_rhs_offset_func_t = decltype(&kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0);
using kai_run_lhs_pack_func_t = decltype(&kai_run_lhs_quant_pack_qsi8d32p_f32);
using kai_run_rhs_pack_func_t = decltype(&kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0);

// Micro-kernel interface
struct kai_qsi8d32p_pack_functions {
    kai_get_lhs_packed_size_func_t packed_size;
    kai_get_lhs_packed_offset_func_t get_packed_offset;
    kai_get_lhs_offset_func_t get_offset;
    kai_run_lhs_pack_func_t run_pack;
};
struct kai_qsi4c32p_pack_functions {
    kai_get_rhs_packed_size_func_t packed_size;
    kai_get_rhs_packed_offset_func_t get_packed_offset;
    kai_get_rhs_offset_func_t get_offset;
    kai_run_rhs_pack_func_t run_pack;
};

struct UKernelVariants {
    UkernelMatmulPackVariant<
        kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel, kai_qsi8d32p_pack_functions, kai_qsi4c32p_pack_functions>
        variant;
    bool clamp_support;
};

// clang-format off
static const int num_non_clamping_kernels = 4;
static const std::array<UKernelVariants, 11>
    variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p = {
        {
         // NOTE: The following kernels do not support clamping despite their names.
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), false},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), false},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa, cpu_has_sme2, lhs_quant_pack_qsi8d32p_f32_neon,
             rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false), false},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot, cpu_has_sme2, lhs_quant_pack_qsi8d32p_f32_neon,
             rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false), false},

         // The kernels below this point will run clamping tests
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), true},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), true},
         {UKERNEL_MATMUL_PACK_VARIANT(
             4x8sb_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm, clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
             cpu_has_i8mm, lhs_quant_pack_qsi8d32p4x8sb_f32_neon, rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), true},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), true},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod, (cpu_check<cpu_has_sve_vl256, cpu_has_dotprod>), lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), true},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod, (cpu_check<cpu_has_sve_vl256, cpu_has_dotprod>), lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), true},
         {UKERNEL_MATMUL_PACK_VARIANT(
              clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm, (cpu_check<cpu_has_sve_vl256, cpu_has_i8mm>), lhs_quant_pack_qsi8d32p_f32,
              rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false), true}}};
// clang-format on

class MatMulTest_f32_qsi8d32p_qsi4c32p
    : public ::testing::TestWithParam<std::tuple<size_t, MatMulShape, MatrixPortion, float>> {};

static std::string test_description(
    const std::string_view& name, const MatMulShape& shape, const MatrixPortion& portion, float clamp_ratio,
    bool bias) {
    std::ostringstream os;

    os << name << "__";
    PrintTo(shape, &os);
    os << "__";
    PrintTo(portion, &os);
    os << "__clamp_ratio_" << static_cast<int>(clamp_ratio * 100);
    if (bias) {
        os << "__Bias";
    }

    return os.str();
}

// Ensure non-clamping tests are marked correctly.
TEST(KernelClampingCheck, SanityCheck) {
    for (size_t i = 0; i < variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.size(); i++) {
        ASSERT_EQ(variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(i).clamp_support, !(i < num_non_clamping_kernels));
    }
}

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, Offset_RHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_rate] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index).variant;

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t bl = 32;
    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();

    auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    auto m_step = ukernel_variant.ukernel.interface.get_m_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, Offset_LHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_rate] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index).variant;

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t bl = 32;
    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    auto n_step = ukernel_variant.ukernel.interface.get_n_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape, portion, clamp_rate] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index).variant;

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    // NOTE: Workaround - some kernels despite being called matmul_clamp do not support clamping.
    const bool clamp_support = variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index).clamp_support;
    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;
    const size_t bl = 32;

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    // Skip tests on clamping when kernel does not support it.
    KAI_ASSERT_ALWAYS_IF(clamp_rate != 0.0F, clamp_support);

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

    // Runs the reference implementation.
    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = bl;
    lhs_qinfo.dst_type = DataType::QSI8;
    lhs_qinfo.scale_type = DataType::FP16;
    const auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref_lhs.data(), DataType::FP32, M, K, lhs_qinfo);

    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = bl;
    rhs_qinfo.dst_type = DataType::QSI4;
    rhs_qinfo.scale_type = DataType::FP16;
    const auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, Float16, int32_t, Int4, Float16, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), nullptr, bl, ref_rhs_quant.data(),
        rhs_qoutputs.scales.data(), nullptr, bl, nullptr, std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());

    // Clamp reference output
    const auto [min, max] = find_clamp_range(DataType::FP32, ref_dst.data(), M * N, 1.0F - clamp_rate);
    const auto out_clamped = clamp(DataType::FP32, ref_dst.data(), M * N, min, max);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size = ukernel_variant.lhs_pack_interface.packed_size(M, K, bl, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_stride = K * sizeof(float);
    auto lhs_offset = ukernel_variant.lhs_pack_interface.get_offset(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    abi_check(
        ukernel_variant.lhs_pack_interface.run_pack, rect.height() /* m */, K, bl, mr, kr, sr, 0,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_quant.data(), N * K);
    const auto ref_rhs_qsu4_scale_f16 =
        pack_data_scales_interleave_block<UInt4, Float16>(ref_rhs_qsu4.data(), rhs_qoutputs.scales.data(), N, K, bl);

    const auto imp_packed_rhs_size = ukernel_variant.rhs_pack_interface.packed_size(N, K, nr, kr, bl);
    Buffer imp_packed_rhs(imp_packed_rhs_size);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    const kai_rhs_pack_qs4cxs1s0_param params{.lhs_zero_point = 1, .rhs_zero_point = 8};
    abi_check(
        ukernel_variant.rhs_pack_interface.run_pack, 1, N, K, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(ref_rhs_qsu4_scale_f16.data()), nullptr, imp_packed_rhs.data(), 0, &params);

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
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col, min, max);

    DefaultMismatchHandler handler(0, 0.0001, 0, 0.0001);
    const auto success = compare(imp_dst.data(), out_clamped.data(), DataType::FP32, M, N, rect, handler);

    ASSERT_TRUE(success);
}

// Test all kernels without clamping
INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.size()),
        testing::Values(
            MatMulShape{1, 2, 32},    //
            MatMulShape{1, 40, 32},   //
            MatMulShape{1, 33, 32},   //
            MatMulShape{32, 64, 64},  //
            MatMulShape{16, 32, 64},  //
            MatMulShape{8, 32, 64},   //
            MatMulShape{15, 32, 32},  //
            MatMulShape{77, 99, 64}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),     // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),  // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),  // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8)  // Somewhere Middle
            ),
        testing::ValuesIn(std::initializer_list<float>{0.0F})),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_idx).variant.ukernel.name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);
        const auto clamp_rate = std::get<3>(info.param);

        return test_description(name, shape, portion, clamp_rate, true);
    });

// Test supported matmul kernels with clamping support.
INSTANTIATE_TEST_SUITE_P(
    MatMulClamped, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(num_non_clamping_kernels, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.size()),
        testing::Values(
            MatMulShape{1, 2, 32},    //
            MatMulShape{1, 33, 32},   //
            MatMulShape{17, 32, 64},  //
            MatMulShape{32, 64, 64},  //
            MatMulShape{77, 99, 64}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),     // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),  // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),  // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8)  // Somewhere Middle
            ),
        testing::ValuesIn(std::initializer_list<float>{0.1F, 0.5F})),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_idx).variant.ukernel.name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);
        const auto clamp_rate = std::get<3>(info.param);

        return test_description(name, shape, portion, clamp_rate, true);
    });

}  // namespace kai::test
