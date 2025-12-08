//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string_view>
#include <tuple>
#include <utility>

#include "kai/kai_common.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/printer.hpp"
#include "test/common/sme.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"

// matmul_clamp_f32_bf16p_bf16p
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p8x4_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p1x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p12x4biasf32_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"

// SME files here.
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"

namespace kai::test {

/// List of supported matrix multiplication methods.
namespace {

static const std::array<MatMulMethod, 5>& get_gemm_methods() {
    static std::array<MatMulMethod, 5> gemm_methods{};
    gemm_methods[0].name = "matmul_nt_nt_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa";
    gemm_methods[0].m0 = 2 * get_sme_vector_length<float>();
    gemm_methods[0].n0 = 2 * get_sme_vector_length<float>();
    gemm_methods[0].k0 = 2;
    gemm_methods[0].dst_format = DataFormat(DataType::FP32);
    gemm_methods[0].lhs_format = DataFormat(DataType::FP32);
    gemm_methods[0].packed_lhs_format = DataFormat(
        DataType::BF16, 2 * get_sme_vector_length<float>(), 2, DataFormat::PackFormat::NONE, DataType::FP32,
        DataType::UNKNOWN, 2 * get_sme_vector_length<float>(), 2);
    gemm_methods[0].rhs_format = DataFormat(DataType::FP32);
    gemm_methods[0].packed_rhs_format = DataFormat(
        DataType::BF16, 2 * get_sme_vector_length<float>(), 2, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32,
        DataType::UNKNOWN, 2 * get_sme_vector_length<float>(), 2);
    gemm_methods[0].bias_format = DataFormat(DataType::FP32);
    gemm_methods[0].fn_is_supported = cpu_has_sme2;
    gemm_methods[0].fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme;
    gemm_methods[0].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme;
    gemm_methods[0].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme;
    gemm_methods[0].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_pack_lhs = kai_run_lhs_pack_bf16p2vlx2_f32_sme;
    gemm_methods[0].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme;
    gemm_methods[0].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme;
    gemm_methods[0].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_pack_rhs = kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme;
    gemm_methods[0].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme;
    gemm_methods[0].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;
    gemm_methods[0].fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa;

    gemm_methods[1].name = "matmul_nt_nt_f32_bf16p_bf16p_8x12_neon_mla";
    gemm_methods[1].m0 = 8;
    gemm_methods[1].n0 = 12;
    gemm_methods[1].k0 = 4;
    gemm_methods[1].dst_format = DataFormat(DataType::FP32);
    gemm_methods[1].lhs_format = DataFormat(DataType::FP32);
    gemm_methods[1].packed_lhs_format =
        DataFormat(DataType::BF16, 8, 4, DataFormat::PackFormat::NONE, DataType::FP32, DataType::UNKNOWN, 8, 4);
    gemm_methods[1].rhs_format = DataFormat(DataType::FP32);
    gemm_methods[1].packed_rhs_format = DataFormat(
        DataType::BF16, 12, 4, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 12, 4);
    gemm_methods[1].bias_format = DataFormat(DataType::FP32);
    gemm_methods[1].fn_is_supported = cpu_has_bf16;
    gemm_methods[1].fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[1].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p8x4_f32_neon;
    gemm_methods[1].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon;
    gemm_methods[1].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_pack_lhs = kai_run_lhs_quant_pack_bf16p8x4_f32_neon;
    gemm_methods[1].fn_get_rhs_offset = kai_get_rhs_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[1].fn_get_packed_rhs_size_generic_block_size =
        kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[1].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_pack_rhs = kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[1].fn_get_bias_offset = kai_get_bias_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[1].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[1].fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;

    gemm_methods[2].name = "matmul_nt_nt_f32_bf16p_bf16p_8x12_neon_mla_f16_inputs_f32_bias_and_output";
    gemm_methods[2].m0 = 8;
    gemm_methods[2].n0 = 12;
    gemm_methods[2].k0 = 4;
    gemm_methods[2].dst_format = DataFormat(DataType::FP32);
    gemm_methods[2].lhs_format = DataFormat(DataType::FP16);
    gemm_methods[2].packed_lhs_format =
        DataFormat(DataType::BF16, 8, 4, DataFormat::PackFormat::NONE, DataType::FP16, DataType::UNKNOWN, 8, 4);
    gemm_methods[2].rhs_format = DataFormat(DataType::FP16);
    gemm_methods[2].packed_rhs_format = DataFormat(
        DataType::BF16, 12, 4, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 12, 4);
    gemm_methods[2].bias_format = DataFormat(DataType::FP32);
    gemm_methods[2].fn_is_supported = cpu_has_bf16;
    gemm_methods[2].fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[2].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_bf16p8x4_f16_neon;
    gemm_methods[2].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon;
    gemm_methods[2].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_pack_lhs = kai_run_lhs_pack_bf16p8x4_f16_neon;
    gemm_methods[2].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[2].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[2].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_pack_rhs = kai_run_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[2].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[2].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[2].fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;

    gemm_methods[3].name = "matmul_nt_nt_f32_bf16p_bf16p_8x12_neon_mla_f16_inputs_f32_bias_and_output_opt_bias";
    gemm_methods[3].m0 = 8;
    gemm_methods[3].n0 = 12;
    gemm_methods[3].k0 = 4;
    gemm_methods[3].dst_format = DataFormat(DataType::FP32);
    gemm_methods[3].lhs_format = DataFormat(DataType::FP16);
    gemm_methods[3].packed_lhs_format =
        DataFormat(DataType::BF16, 8, 4, DataFormat::PackFormat::NONE, DataType::FP16, DataType::UNKNOWN, 8, 4);
    gemm_methods[3].rhs_format = DataFormat(DataType::FP16);
    gemm_methods[3].packed_rhs_format = DataFormat(
        DataType::BF16, 12, 4, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 12, 4);
    gemm_methods[3].bias_format = DataFormat(DataType::UNKNOWN);
    gemm_methods[3].fn_is_supported = cpu_has_bf16;
    gemm_methods[3].fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[3].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_bf16p8x4_f16_neon;
    gemm_methods[3].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon;
    gemm_methods[3].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_pack_lhs = kai_run_lhs_pack_bf16p8x4_f16_neon;
    gemm_methods[3].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[3].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[3].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_pack_rhs = kai_run_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[3].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_bf16p12x4biasf32_f16_neon;
    gemm_methods[3].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[3].fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;

    gemm_methods[4].name = "matmul_nt_nt_f32_bf16p_bf16p_8x12_neon_mla_opt_bias";
    gemm_methods[4].m0 = 8;
    gemm_methods[4].n0 = 12;
    gemm_methods[4].k0 = 4;
    gemm_methods[4].dst_format = DataFormat(DataType::FP32);
    gemm_methods[4].lhs_format = DataFormat(DataType::FP32);
    gemm_methods[4].packed_lhs_format =
        DataFormat(DataType::BF16, 8, 4, DataFormat::PackFormat::NONE, DataType::FP32, DataType::UNKNOWN, 8, 4);
    gemm_methods[4].rhs_format = DataFormat(DataType::FP32);
    gemm_methods[4].packed_rhs_format = DataFormat(
        DataType::BF16, 12, 4, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 12, 4);
    gemm_methods[4].bias_format = DataFormat(DataType::UNKNOWN);
    gemm_methods[4].fn_is_supported = cpu_has_bf16;
    gemm_methods[4].fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[4].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p8x4_f32_neon;
    gemm_methods[4].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon;
    gemm_methods[4].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_pack_lhs = kai_run_lhs_quant_pack_bf16p8x4_f32_neon;
    gemm_methods[4].fn_get_rhs_offset = kai_get_rhs_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[4].fn_get_packed_rhs_size_generic_block_size =
        kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[4].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_pack_rhs = kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[4].fn_get_bias_offset = kai_get_bias_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemm_methods[4].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;
    gemm_methods[4].fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla;

    return gemm_methods;
}

static const std::array<MatMulMethod, 2>& get_gemv_methods() {
    static std::array<MatMulMethod, 2> gemv_methods{};
    gemv_methods[0].name = "matmul_nt_nt_f32_bf16p_bf16p_1x36_neon_dot";
    gemv_methods[0].m0 = 1;
    gemv_methods[0].n0 = 12;
    gemv_methods[0].k0 = 4;
    gemv_methods[0].dst_format = DataFormat(DataType::FP32);
    gemv_methods[0].lhs_format = DataFormat(DataType::FP32);
    gemv_methods[0].packed_lhs_format =
        DataFormat(DataType::BF16, 1, 4, DataFormat::PackFormat::NONE, DataType::FP32, DataType::UNKNOWN, 1, 4);
    gemv_methods[0].rhs_format = DataFormat(DataType::FP32);
    gemv_methods[0].packed_rhs_format = DataFormat(
        DataType::BF16, 12, 4, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 12, 4);
    gemv_methods[0].bias_format = DataFormat(DataType::FP32);
    gemv_methods[0].fn_is_supported = cpu_has_bf16;
    gemv_methods[0].fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[0].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p1x4_f32_neon;
    gemv_methods[0].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p1x4_f32_neon;
    gemv_methods[0].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_pack_lhs = kai_run_lhs_quant_pack_bf16p1x4_f32_neon;
    gemv_methods[0].fn_get_rhs_offset = kai_get_rhs_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[0].fn_get_packed_rhs_size_generic_block_size =
        kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[0].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_pack_rhs = kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[0].fn_get_bias_offset = kai_get_bias_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[0].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[0].fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;

    gemv_methods[1].name = "matmul_nt_nt_f32_bf16p_bf16p_1x36_neon_dot_opt_bias";
    gemv_methods[1].m0 = 1;
    gemv_methods[1].n0 = 12;
    gemv_methods[1].k0 = 4;
    gemv_methods[1].dst_format = DataFormat(DataType::FP32);
    gemv_methods[1].lhs_format = DataFormat(DataType::FP32);
    gemv_methods[1].packed_lhs_format =
        DataFormat(DataType::BF16, 1, 4, DataFormat::PackFormat::NONE, DataType::FP32, DataType::UNKNOWN, 1, 4);
    gemv_methods[1].rhs_format = DataFormat(DataType::FP32);
    gemv_methods[1].packed_rhs_format = DataFormat(
        DataType::BF16, 12, 4, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 12, 4);
    gemv_methods[1].bias_format = DataFormat(DataType::UNKNOWN);
    gemv_methods[1].fn_is_supported = cpu_has_bf16;
    gemv_methods[1].fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[1].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p1x4_f32_neon;
    gemv_methods[1].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p1x4_f32_neon;
    gemv_methods[1].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_pack_lhs = kai_run_lhs_quant_pack_bf16p1x4_f32_neon;
    gemv_methods[1].fn_get_rhs_offset = kai_get_rhs_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[1].fn_get_packed_rhs_size_generic_block_size =
        kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[1].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_pack_rhs = kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[1].fn_get_bias_offset = kai_get_bias_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon;
    gemv_methods[1].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;
    gemv_methods[1].fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot;

    return gemv_methods;
}

}  // namespace

/// Matrix multiplication test fixture.
class MatMulTestBf16 : public testing::TestWithParam<MatMulTestParams> {
private:
    /// Unique ID: m, n, k
    using TestDataId = std::tuple<size_t, size_t, size_t, std::string_view>;

protected:
    /// Cached test data that is shared between multiple test case.
    struct TestData {
        Buffer lhs{};             ///< LHS operand.
        Buffer ref_packed_lhs{};  ///< Reference packed LHS.
        Buffer rhs{};             ///< RHS operand.
        Buffer rhs_scales{};      ///< RHS per-row quantization scales.
        Buffer bias{};            ///< Bias.
        Buffer ref_packed_rhs{};  ///< Reference packed RHS.
        Buffer ref_dst{};         ///< Reference output.
    };

    /// Gets the test data for the current test case.
    static const TestData& test_data() {
        const auto& [method, info, portion, bias_mode] = GetParam();
        const TestDataId data_id{info.m, info.n, info.k, method.name};

        // If the test data is already available, returns it.
        const auto data_it = _data.find(data_id);

        if (data_it != _data.end()) {
            return data_it->second;
        }

        // Generates the test data.
        const auto has_lhs_pack = method.packed_lhs_format.data_type() != DataType::UNKNOWN;
        const auto has_rhs_pack = method.packed_rhs_format.data_type() != DataType::UNKNOWN;
        const auto has_bias = method.bias_format.data_type() != DataType::UNKNOWN;

        const auto lhs_h = info.m;
        const auto lhs_w = info.k;
        auto lhs = fill_matrix_random(lhs_h, lhs_w, method.lhs_format, 0);
        Buffer ref_packed_lhs;

        if (has_lhs_pack) {
            ref_packed_lhs =
                pack(method.packed_lhs_format, lhs.data(), nullptr, nullptr, method.lhs_format, lhs_h, lhs_w);
        }

        const auto rhs_h = info.k;
        const auto rhs_w = info.n;
        auto rhs = fill_matrix_random(rhs_h, rhs_w, method.rhs_format, 1);

        Buffer rhs_scales;
        if (data_type_is_quantized(method.rhs_format.data_type()) &&
            method.rhs_format.pack_format() == DataFormat::PackFormat::NONE) {
            rhs_scales = fill_matrix_random(rhs_h, 1, DataFormat(DataType::FP32), 2);
        }

        const auto bias_h = 1;
        const auto bias_w = info.n;
        Buffer bias;

        if (has_bias) {
            bias = fill_matrix_random(bias_h, bias_w, method.bias_format, 3);
        }

        constexpr size_t nr = 12;
        constexpr size_t kr = 4;

        size_t packed_rhs_size = 0;

        if (method.fn_get_packed_rhs_size) {
            packed_rhs_size = method.fn_get_packed_rhs_size(rhs_w, rhs_h);
        } else if (method.fn_get_packed_rhs_size_generic_block_size) {
            packed_rhs_size = method.fn_get_packed_rhs_size_generic_block_size(rhs_w, rhs_h, nr, kr);
        } else {
            KAI_ERROR("No function to calculate Packed Rhs Matrix Size");
        }

        Buffer packed_rhs(packed_rhs_size);

        if (has_rhs_pack) {
            const auto ref_rhs_row_stride = method.rhs_format.default_row_stride(rhs_w);
            method.pack_rhs(
                info.n, info.k, rhs.data(), ref_rhs_row_stride, has_bias ? bias.data() : nullptr, nullptr,
                packed_rhs.data());
        }

        KAI_ASSUME_ALWAYS(method.lhs_format.is_raw());
        KAI_ASSUME_ALWAYS(method.rhs_format.is_raw());
        KAI_ASSUME_ALWAYS(method.dst_format.is_raw());

        Buffer tmp_lhs;
        Buffer tmp_rhs;
        const void* p_lhs_buff = lhs.data();
        const void* p_rhs_buff = rhs.data();

        if (method.lhs_format.data_type() == DataType::FP32 || method.lhs_format.data_type() == DataType::FP16) {
            tmp_lhs = cast(p_lhs_buff, method.lhs_format.data_type(), DataType::BF16, lhs_h, lhs_w);
            p_lhs_buff = tmp_lhs.data();
        }
        if (method.rhs_format.data_type() == DataType::FP32 || method.rhs_format.data_type() == DataType::FP16) {
            tmp_rhs = cast(p_rhs_buff, method.rhs_format.data_type(), DataType::BF16, rhs_h, rhs_w);
            p_rhs_buff = tmp_rhs.data();
        }

        auto ref_dst =
            matmul_nt_nt_quantized<BFloat16<>, float, float, BFloat16<>, float, float, float, float, float, float>(
                info.m, info.n, info.k, p_lhs_buff, nullptr, nullptr, 1, info.k, p_rhs_buff, nullptr, nullptr, 1,
                info.k, bias.data(), nullptr, nullptr, info.k);

        auto& data = _data[data_id] = {};
        data.lhs = std::move(lhs);
        data.ref_packed_lhs = std::move(ref_packed_lhs);
        data.rhs = std::move(rhs);
        data.rhs_scales = std::move(rhs_scales);
        data.bias = std::move(bias);
        data.ref_packed_rhs = std::move(packed_rhs);
        data.ref_dst = std::move(ref_dst);

        return data;
    }

private:
    // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
    static std::map<TestDataId, TestData> _data;
    // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::map<MatMulTestBf16::TestDataId, MatMulTestBf16::TestData> MatMulTestBf16::_data;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/// Tests the output.
TEST_P(MatMulTestBf16, Output) {
    const auto& [method, info, portion, bias_mode] = GetParam();

    if (method.fn_is_supported && !method.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    if (!method.has_main_kernel()) {
        GTEST_SKIP() << "No main kernel available";
    }

    const auto& data = test_data();
    const auto m_step = method.fn_get_main_m_step();
    ASSERT_TRUE(m_step % method.m0 == 0);

    const auto n_step = method.fn_get_main_n_step();
    ASSERT_TRUE(n_step % method.n0 == 0);

    const auto rect = portion.compute_portion(info.m, info.n, m_step, n_step);

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const size_t lhs_w = info.k;
    const size_t rhs_w = rect.width();
    const size_t bias_w = info.n;
    const size_t dst_w = info.n;
    const bool has_bias = (data.bias.size() > 0);

    const auto lhs_start_row = rect.start_row();
    const auto lhs_stride = method.lhs_format.default_row_stride(lhs_w);

    const size_t lhs_packed_size = method.fn_get_packed_lhs_size(info.m, info.k, method.m0, method.k0, 1 /* sr */);
    Buffer lhs_data(lhs_packed_size);

    uintptr_t lhs_offset = method.fn_get_lhs_offset(lhs_start_row, lhs_stride);
    uintptr_t lhs_packed_offset = method.fn_get_packed_lhs_offset(lhs_start_row, info.k);

    KAI_UNUSED(lhs_offset);
    abi_check(
        method.fn_pack_lhs, rect.height(), info.k, method.m0, method.k0, 1 /* sr */, 0 /* m_idx_start */,
        data.lhs.data() + lhs_offset, lhs_stride, lhs_data.data() + lhs_packed_offset);

    const auto rhs_stride = method.rhs_format.default_row_stride(info.n);

    size_t rhs_packed_size = 0;

    if (method.fn_get_packed_rhs_size_generic_block_size) {
        rhs_packed_size = method.fn_get_packed_rhs_size_generic_block_size(info.n, info.k, method.n0, method.k0);
    } else if (method.fn_get_packed_rhs_size) {
        rhs_packed_size = method.fn_get_packed_rhs_size(info.n, info.k);
    }

    Buffer rhs_data(rhs_packed_size);

    const auto packed_rhs_start_row = rect.start_col();
    const auto packed_rhs_start_col = 0;

    uintptr_t rhs_offset = method.fn_get_rhs_offset(rect.start_col());
    uintptr_t rhs_packed_offset = method.fn_get_main_packed_rhs_offset(packed_rhs_start_row, info.k);
    const auto ref_rhs_packed_offset =
        method.packed_rhs_format.default_offset_in_bytes(packed_rhs_start_row, packed_rhs_start_col, info.k);

    ASSERT_EQ(rhs_packed_offset, ref_rhs_packed_offset);

    uintptr_t bias_offset = sizeof(float) * rect.start_col();

    abi_check(
        method.fn_pack_rhs,
        1,  // num_groups
        rhs_w, info.k, method.n0, method.k0,
        1,  // sr
        rhs_stride, data.rhs.data() + rhs_offset, has_bias ? data.bias.data() + bias_offset : nullptr,
        nullptr,  // Scale
        rhs_data.data() + rhs_packed_offset, 0, nullptr);

    if (has_bias) {
        const auto ref_bias_offset = method.bias_format.default_offset_in_bytes(0, rect.start_col(), bias_w);
        ASSERT_EQ(ref_bias_offset, bias_offset);
    }

    const auto dst_stride = method.dst_format.default_row_stride(dst_w);
    const auto dst_offset = method.fn_get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = method.dst_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), dst_w);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    const auto dst_size = method.fn_get_dst_size(info.m, info.n);
    const auto ref_dst_size = method.dst_format.default_size_in_bytes(info.m, info.n);
    ASSERT_EQ(dst_size, ref_dst_size);

    Buffer dst(dst_size);
    abi_check(
        &MatMulMethod::main_kernel, method, rect.height(), rect.width(), info.k, lhs_data.data() + lhs_packed_offset,
        rhs_data.data() + rhs_packed_offset, nullptr, dst.data() + dst_offset, lhs_stride, rhs_stride, dst_stride,
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());

    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    const auto success = compare(dst.data(), data.ref_dst.data(), method.dst_format, info.m, info.n, rect, handler);

    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMulGemm, MatMulTestBf16,
    testing::Combine(
        testing::ValuesIn(get_gemm_methods()),
        testing::Values(
            MatMulShape{1, 1, 1},        // Smallest Possible Shape
            MatMulShape{3, 7, 3},        // Smaller than block size
            MatMulShape{12, 8, 4},       // Same block size
            MatMulShape{1, 1, 1023},     // Long K
            MatMulShape{1013, 1, 5},     // Long M
            MatMulShape{2, 1013, 6},     // Long N
            MatMulShape{13, 33, 23},     //
            MatMulShape{93, 57, 89},     //
            MatMulShape{256, 256, 256},  // Nice shapes
            MatMulShape{257, 113, 373}   // Prime numbers
            ),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),         // Full matrix.
            MatrixPortion(0, 0, 0.25, 0.25),   // Top-left corner.
            MatrixPortion(0.75, 0.75, 1, 1),   // Bottom-right corner.
            MatrixPortion(0.75, 0, 1, 1),      // Partial rows
            MatrixPortion(0.4, 0.5, 0.6, 0.8)  // Somewhere Middle
            ),
        testing::Values(BiasMode::PROVIDED)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulGemv, MatMulTestBf16,
    testing::Combine(
        testing::ValuesIn(get_gemv_methods()),
        testing::Values(
            MatMulShape{1, 1, 1},        // Smallest Possible Shape
            MatMulShape{1, 1, 1023},     // Long K
            MatMulShape{1, 1023, 1},     // Long N
            MatMulShape{1, 1013, 1023},  // Large Rhs
            MatMulShape{1, 37, 23},      //
            MatMulShape{1, 57, 89},      //
            MatMulShape{1, 36, 89},      //
            MatMulShape{1, 98, 23},      //
            MatMulShape{1, 64, 1024},    // Nice shapes - Long Rhs Rect
            MatMulShape{1, 1024, 64},    // Nice shapes - Wide Rhs Rect
            MatMulShape{1, 256, 256},    // Nice shapes - Square
            MatMulShape{1, 113, 373}     // Prime numbers
            ),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),     // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),  // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),  // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8)  // Somewhere Middle
            ),
        testing::Values(BiasMode::PROVIDED)),
    testing::PrintToStringParamName());
}  // namespace kai::test
