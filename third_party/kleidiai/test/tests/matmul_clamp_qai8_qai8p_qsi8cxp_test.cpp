//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"
#include "test/common/cpu_info.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/rect.hpp"
#include "test/common/sme.hpp"
#include "test/reference/binary_elementwise.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/matmul_pack.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/reduce.hpp"
#include "test/reference/reorder.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

namespace {

struct GemmVariant {
    size_t acc_height;
    size_t acc_width;
    size_t acc_fanin;

    bool (*fn_is_supported)();

    size_t (*fn_pack_lhs_get_m_step)(size_t mr);
    size_t (*fn_pack_lhs_get_lhs_offset)(size_t m_idx, size_t lhs_stride);
    size_t (*fn_pack_lhs_get_packed_lhs_offset)(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);
    size_t (*fn_pack_lhs_get_packed_lhs_size)(size_t m, size_t k, size_t mr, size_t kr, size_t sr);
    void (*fn_pack_lhs_run)(
        size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
        void* lhs_packed);

    size_t (*fn_pack_rhs_get_n_step)();
    size_t (*fn_pack_rhs_get_rhs_offset)(size_t n_idx);
    size_t (*fn_pack_rhs_get_bias_offset)(size_t n_idx);
    size_t (*fn_pack_rhs_get_scale_offset)(size_t n_idx);
    size_t (*fn_pack_rhs_get_packed_rhs_offset)(size_t n_idx, size_t k);
    size_t (*fn_pack_rhs_get_packed_rhs_size)(size_t n, size_t k);
    void (*fn_pack_rhs_run)(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
        const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
        const struct kai_rhs_pack_qsi8cx_params* params);

    size_t (*fn_main_get_m_step)();
    size_t (*fn_main_get_n_step)();
    size_t (*fn_main_get_mr)();
    size_t (*fn_main_get_nr)();
    size_t (*fn_main_get_kr)();
    size_t (*fn_main_get_sr)();
    size_t (*fn_main_get_packed_lhs_offset)(size_t m_idx, size_t k);
    size_t (*fn_main_get_packed_rhs_offset)(size_t n_idx, size_t k);
    size_t (*fn_main_get_dst_offset)(size_t m_idx, size_t n_idx, size_t dst_stride);
    size_t (*fn_main_get_dst_size)(size_t m, size_t n);
    void (*fn_main_run)(
        size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
        size_t dst_stride_col, const kai_matmul_requantize32_params* params);
};

struct GemmShape {
    size_t m;
    size_t n;
    size_t k;
};

const std::array gemm_variants = {
    GemmVariant{
        .acc_height = 2 * get_sme_vector_length<int32_t>(),
        .acc_width = 2 * get_sme_vector_length<int32_t>(),
        .acc_fanin = sizeof(int32_t) / sizeof(int8_t),

        .fn_is_supported = cpu_has_sme2,

        .fn_pack_lhs_get_m_step = kai_get_m_step_lhs_pack_x8p2vlx4_x8_sme,
        .fn_pack_lhs_get_lhs_offset = kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme,
        .fn_pack_lhs_get_packed_lhs_offset = kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme,
        .fn_pack_lhs_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme,
        .fn_pack_lhs_run = kai_run_lhs_pack_x8p2vlx4_x8_sme,

        .fn_pack_rhs_get_n_step = kai_get_n_step_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme,
        .fn_pack_rhs_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme,
        .fn_pack_rhs_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme,
        .fn_pack_rhs_get_scale_offset = kai_get_scale_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme,
        .fn_pack_rhs_get_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme,
        .fn_pack_rhs_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme,
        .fn_pack_rhs_run = kai_run_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme,

        .fn_main_get_m_step = kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_n_step = kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_mr = kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_nr = kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_kr = kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_sr = kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_packed_lhs_offset =
            kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_packed_rhs_offset =
            kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_dst_offset = kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_get_dst_size = kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
        .fn_main_run = kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
    },
};

constexpr float output_clamp_rate = 0.1F;  // Clamping 10% the range of the output.

const std::array gemm_shapes = {
    GemmShape{1, 1, 1},  //
    GemmShape{
        2 * get_sme_vector_length<int32_t>(), 2 * get_sme_vector_length<int32_t>(),
        sizeof(int32_t) / sizeof(int8_t)},  //
    GemmShape{20, 30, 40},                  //
    GemmShape{1, 49, 21},                   //
    GemmShape{23, 1, 43},                   //
    GemmShape{32, 14, 1},                   //
    GemmShape{123, 85, 45},                 //
    GemmShape{130, 130, 6},
};

const std::array output_portions = {
    MatrixPortion(0, 0, 1, 1),        // Full matrix.
    MatrixPortion(0, 0, 0.25, 0.25),  // Top-left corner.
    MatrixPortion(0.75, 0.75, 1, 1),  // Bottom-right corner.
};

void run_test(const GemmShape& shape, const GemmVariant& variant, const MatrixPortion& output_portion) {
    const uint64_t seed = 0;

    if (!variant.fn_is_supported()) {
        GTEST_SKIP();
    }

    // ============================================================
    // Test the packing and scheduling parameters
    // ============================================================

    const auto imp_mr = variant.fn_main_get_mr();
    const auto imp_nr = variant.fn_main_get_nr();
    const auto imp_kr = variant.fn_main_get_kr();
    const auto imp_sr = variant.fn_main_get_sr();

    ASSERT_EQ(imp_mr, variant.acc_height);
    ASSERT_EQ(imp_nr, variant.acc_width);
    ASSERT_EQ(imp_kr, variant.acc_fanin);
    ASSERT_EQ(imp_sr, 1);

    const auto imp_m_step = variant.fn_main_get_m_step();
    const auto imp_n_step = variant.fn_main_get_n_step();

    ASSERT_EQ(imp_m_step, variant.acc_height);
    ASSERT_EQ(imp_n_step, variant.acc_width);

    // ============================================================
    // Calculates the output area under test
    // ============================================================

    const auto output_area = output_portion.compute_portion(shape.m, shape.n, variant.acc_height, variant.acc_width);

    // ============================================================
    // Generates input and reference output data
    // ============================================================

    // Generates the input data in floating-point.
    const auto lhs_f32 = fill_random<float>(shape.m * shape.k, seed + 0);
    const auto rhs_f32 = fill_random<float>(shape.k * shape.n, seed + 1);
    const auto bias_f32 = fill_random<float>(shape.n, seed + 2);

    // Quantizes the input data.
    //   * LHS: 8-bit asymmetric per-matrix quantization.
    //   * RHS: 8-bit symmetric per-channel quantization.
    //   * Bias: 32-bit symmetric per-channel quantization.
    const auto [lhs_qai8, lhs_qai8_scales, lhs_qai8_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(
            lhs_f32.data(), 1, shape.m * shape.k, shape.m * shape.k);
    const auto lhs_scale = read_array<float>(lhs_qai8_scales.data(), 0);
    const auto lhs_zero_point = read_array<int32_t>(lhs_qai8_zero_points.data(), 0);

    const auto rhs_f32_t = transpose<float>(rhs_f32.data(), shape.k, shape.n);
    const auto [rhs_qsi8_t, rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, float>(rhs_f32_t.data(), shape.n, shape.k, shape.k);
    const auto rhs_qsi8 = transpose<int8_t>(rhs_qsi8_t.data(), shape.n, shape.k);

    const auto bias_scale = mul<float>(&lhs_scale, 1, 1, rhs_scales.data(), 1, shape.n);
    const auto bias_qsi32 =
        quantize_symmetric_per_block<float, int32_t, float>(bias_f32.data(), bias_scale.data(), shape.n, 1, 1);

    // Runs the reference implementation of matmul to produce floating-point result.
    const auto ref_dst_f32 =
        matmul_nt_t_quantized<int8_t, float, int32_t, int8_t, float, int32_t, int32_t, float, int32_t, float>(
            shape.m, shape.n, shape.k, lhs_qai8.data(), &lhs_scale, &lhs_zero_point, shape.m, shape.k,
            rhs_qsi8_t.data(), rhs_scales.data(), nullptr, 1, shape.k, bias_qsi32.data(), bias_scale.data(), nullptr,
            1);

    // Computes the output quantization information and clamping limits.
    //
    // To get a realistic value for the output quantization information and clamping limits
    // and avoid uncontrolled saturation problem, these information will be calculated
    // based on the reference floating-point output.
    //
    // The clamping limits will be slightly narrower than the actual range of the output
    // so that a portion of the output will be clampped.
    const auto [dst_scales, dst_zero_points] =
        compute_asymmetric_per_block_quantization_info<float, int8_t, float, int32_t>(
            ref_dst_f32.data(), 1, shape.m * shape.n, shape.m * shape.n);
    const auto dst_scale = read_array<float>(dst_scales.data(), 0);
    const auto dst_zero_point = read_array<int32_t>(dst_zero_points.data(), 0);

    const auto ref_dst_f32_min = reduce_min<float>(ref_dst_f32.data(), shape.m * shape.n);
    const auto ref_dst_f32_max = reduce_max<float>(ref_dst_f32.data(), shape.m * shape.n);
    const auto ref_dst_f32_range = ref_dst_f32_max - ref_dst_f32_min;

    const auto ref_dst_f32_clamp_min = ref_dst_f32_min + ref_dst_f32_range * output_clamp_rate / 2;
    const auto ref_dst_f32_clamp_max = ref_dst_f32_max - ref_dst_f32_range * output_clamp_rate / 2;
    const auto dst_qai8_clamp_min =
        quantize_asymmetric<float, int8_t, int32_t>(ref_dst_f32_clamp_min, dst_scale, dst_zero_point);
    const auto dst_qai8_clamp_max =
        quantize_asymmetric<float, int8_t, int32_t>(ref_dst_f32_clamp_max, dst_scale, dst_zero_point);

    // Clamps and quantizes the reference output matrix.
    const auto ref_dst_f32_clamped =
        clamp<float>(ref_dst_f32.data(), shape.m * shape.n, ref_dst_f32_clamp_min, ref_dst_f32_clamp_max);
    const auto ref_dst_qsi8_clamped = quantize_asymmetric_per_block<float, int8_t, float, int32_t>(
        ref_dst_f32_clamped.data(), &dst_scale, &dst_zero_point, 1, shape.m * shape.n, shape.m * shape.n);

    // Runs the reference implementation of the packing functions.
    //
    // The reference packing functions cannot be executed earlier
    // because we need the reference floating-point output first to have
    // the quantization information.
    const auto ref_packed_lhs =
        reorder_block<int8_t>(lhs_qai8.data(), shape.m, shape.k, variant.acc_height, variant.acc_fanin);

    const auto ref_packed_rhs = matmul_pack_rhs_nxk_static_quantized<int8_t, float, int32_t>(
        rhs_qsi8_t.data(), rhs_scales.data(), lhs_scale, dst_scale, bias_qsi32.data(), lhs_zero_point, shape.n, shape.k,
        variant.acc_width, variant.acc_fanin);

    // ============================================================
    // Runs the optimized implementation and checks for correctness
    // ============================================================

    // Runs the optimized implementation of LHS packing.
    const auto imp_packed_lhs_size =
        variant.fn_pack_lhs_get_packed_lhs_size(shape.m, shape.k, variant.acc_height, variant.acc_fanin, 1);
    ASSERT_EQ(imp_packed_lhs_size, ref_packed_lhs.size());
    std::vector<uint8_t> imp_packed_lhs(imp_packed_lhs_size);

    {
        const auto imp_lhs_offset =
            variant.fn_pack_lhs_get_lhs_offset(output_area.start_row(), shape.k * sizeof(int8_t));
        const auto imp_packed_lhs_offset =
            variant.fn_pack_lhs_get_packed_lhs_offset(output_area.start_row(), shape.k, imp_mr, imp_kr, imp_sr);

        variant.fn_pack_lhs_run(
            output_area.height(), shape.k, imp_mr, imp_kr, imp_sr, 0, lhs_qai8.data() + imp_lhs_offset,
            shape.k * sizeof(int8_t), imp_packed_lhs.data() + imp_packed_lhs_offset);

        const auto imp_packed_lhs_end_offset = output_area.end_row() < shape.m
            ? variant.fn_pack_lhs_get_packed_lhs_offset(output_area.end_row(), shape.k, imp_mr, imp_kr, imp_sr)
            : imp_packed_lhs_size;

        for (size_t i = 0; i < ref_packed_lhs.size(); ++i) {
            if (i >= imp_packed_lhs_offset && i < imp_packed_lhs_end_offset) {
                ASSERT_EQ(imp_packed_lhs[i], ref_packed_lhs[i]);
            } else {
                ASSERT_EQ(imp_packed_lhs[i], 0);
            }
        }
    }

    // Runs the optimized implementation of RHS packing.
    const auto imp_packed_rhs_size = variant.fn_pack_rhs_get_packed_rhs_size(shape.n, shape.k);
    ASSERT_EQ(imp_packed_rhs_size, ref_packed_rhs.size());
    std::vector<uint8_t> imp_packed_rhs(imp_packed_rhs_size);

    {
        const auto imp_rhs_offset = variant.fn_pack_rhs_get_rhs_offset(output_area.start_col());
        const auto imp_bias_offset = variant.fn_pack_rhs_get_bias_offset(output_area.start_col());
        const auto imp_scale_offset = variant.fn_pack_rhs_get_scale_offset(output_area.start_col());
        const auto imp_packed_rhs_offset = variant.fn_pack_rhs_get_packed_rhs_offset(output_area.start_col(), shape.k);

        const kai_rhs_pack_qsi8cx_params imp_pack_rhs_params{
            .lhs_zero_point = lhs_zero_point,
            .scale_multiplier = lhs_scale / dst_scale,
        };

        variant.fn_pack_rhs_run(
            1, output_area.width(), shape.k, imp_nr, imp_kr, imp_sr, shape.n * sizeof(int8_t),
            rhs_qsi8.data() + imp_rhs_offset, bias_qsi32.data() + imp_bias_offset, rhs_scales.data() + imp_scale_offset,
            imp_packed_rhs.data() + imp_packed_rhs_offset, 0, &imp_pack_rhs_params);

        const auto imp_packed_rhs_end_offset = output_area.end_col() < shape.n
            ? variant.fn_pack_rhs_get_packed_rhs_offset(output_area.end_col(), shape.k)
            : imp_packed_rhs_size;

        for (size_t i = 0; i < ref_packed_rhs.size(); ++i) {
            if (i >= imp_packed_rhs_offset && i < imp_packed_rhs_end_offset) {
                ASSERT_EQ(imp_packed_rhs[i], ref_packed_rhs[i]);
            } else {
                ASSERT_EQ(imp_packed_rhs[i], 0);
            }
        }
    }

    // Runs the optimized implementation of GEMM kernel.
    const auto imp_dst_size = variant.fn_main_get_dst_size(shape.m, shape.n);
    ASSERT_EQ(imp_dst_size, ref_dst_qsi8_clamped.size());

    std::vector<uint8_t> imp_dst(imp_dst_size);

    {
        const auto imp_packed_lhs_offset = variant.fn_main_get_packed_lhs_offset(output_area.start_row(), shape.k);
        const auto imp_packed_rhs_offset = variant.fn_main_get_packed_rhs_offset(output_area.start_col(), shape.k);
        const auto imp_dst_offset =
            variant.fn_main_get_dst_offset(output_area.start_row(), output_area.start_col(), shape.n * sizeof(int8_t));
        ASSERT_EQ(imp_dst_offset, output_area.start_row() * shape.n + output_area.start_col());

        const kai_matmul_requantize32_params imp_main_params{
            .min_value = dst_qai8_clamp_min,
            .max_value = dst_qai8_clamp_max,
            .output_zero_point = dst_zero_point,
        };

        variant.fn_main_run(
            output_area.height(), output_area.width(), shape.k, imp_packed_lhs.data() + imp_packed_lhs_offset,
            imp_packed_rhs.data() + imp_packed_rhs_offset, imp_dst.data() + imp_dst_offset, shape.n * sizeof(int8_t),
            sizeof(int8_t), &imp_main_params);

        for (size_t y = 0; y < shape.m; ++y) {
            for (size_t x = 0; x < shape.n; ++x) {
                const auto i = y * shape.n + x;
                const auto in_area = y >= output_area.start_row() && y < output_area.end_row() &&
                    x >= output_area.start_col() && x < output_area.end_col();

                const int32_t imp_value = read_array<int8_t>(imp_dst.data(), i);
                const int32_t ref_value = in_area ? read_array<int8_t>(ref_dst_qsi8_clamped.data(), i) : 0;
                const auto error = std::abs(imp_value - ref_value);
                const auto threshold = in_area ? 1 : 0;

                if (error > threshold) {
                    ASSERT_EQ(imp_value, ref_value);
                }
            }
        }
    }
}

using ThisTest = testing::TestWithParam<std::tuple<GemmVariant, GemmShape, MatrixPortion>>;

TEST_P(ThisTest, EndToEnd) {
    const auto& [variant, shape, output_portion] = GetParam();

    run_test(shape, variant, output_portion);
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    matmul_clamp_qai8_qai8p_qsi8cxp, ThisTest,
    testing::Combine(
        testing::ValuesIn(gemm_variants), testing::ValuesIn(gemm_shapes), testing::ValuesIn(output_portions)));

}  // namespace kai::test
