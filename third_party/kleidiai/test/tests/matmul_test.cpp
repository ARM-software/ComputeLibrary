//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/matmul.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
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
#include "test/common/sme.hpp"
#include "test/common/sve.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/generators.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/transpose.hpp"

// matmul_clamp_f16_f16_f16p
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"

// matmul_clamp_f16_f16p_f16p
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.h"

// matmul_clamp_f32_f32_f32p
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p16x1b_x32_x32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve.h"

// matmul_clamp_f32_f32p_f32p
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"

namespace kai::test {

static const auto& get_matmul_methods() {
    // List of supported matrix multiplication methods.
    static std::array<MatMulMethod, 7> matmul_methods{};

    matmul_methods[0].name = "matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla";
    matmul_methods[0].m0 = 6;
    matmul_methods[0].n0 = 16;
    matmul_methods[0].dst_format = DataFormat(DataType::FP16);
    matmul_methods[0].lhs_format = DataFormat(DataType::FP16);
    matmul_methods[0].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    matmul_methods[0].rhs_format = DataFormat(DataType::FP16);
    matmul_methods[0].packed_rhs_format = DataFormat(
        DataType::FP16, 16, 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP16, DataType::UNKNOWN, 16, 1);
    matmul_methods[0].bias_format = DataFormat(DataType::FP16);
    matmul_methods[0].fn_generate_lhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 0);
    matmul_methods[0].fn_generate_rhs = NormalRandomGenerator<Float16>(-1.0, 1.0, 1);
    matmul_methods[0].fn_generate_bias = UniformRandomGenerator<Float16>(-1.0, 1.0, 0);
    matmul_methods[0].fn_is_supported = cpu_has_fp16;
    matmul_methods[0].fn_get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon;
    matmul_methods[0].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon;
    matmul_methods[0].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon;
    matmul_methods[0].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon;
    matmul_methods[0].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_pack_rhs = kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon;
    matmul_methods[0].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon;
    matmul_methods[0].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;
    matmul_methods[0].fn_matmul_f16_f16_f16p = kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla;

    matmul_methods[1].name = "matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa";
    matmul_methods[1].m0 = 2 * get_sme_vector_length<float>();
    matmul_methods[1].n0 = 2 * get_sme_vector_length<float>();
    matmul_methods[1].dst_format = DataFormat(DataType::FP16);
    matmul_methods[1].lhs_format = DataFormat(DataType::FP16);
    matmul_methods[1].packed_lhs_format = DataFormat(DataType::FP16, 2 * get_sme_vector_length<float>(), 2);
    matmul_methods[1].rhs_format = DataFormat(DataType::FP16);
    matmul_methods[1].packed_rhs_format = DataFormat(
        DataType::FP16,                          // Output type
        2 * get_sme_vector_length<float>(), 2,   // Block size
        DataFormat::PackFormat::BIAS_PER_ROW,    // Data layout
        DataType::FP16,                          // Bias format
        DataType::UNKNOWN,                       // Scaling type
        2 * get_sme_vector_length<float>(), 2);  // Sub-block
    matmul_methods[1].bias_format = DataFormat(DataType::FP16);
    matmul_methods[1].fn_generate_lhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 0);
    matmul_methods[1].fn_generate_rhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 1);
    matmul_methods[1].fn_generate_bias = SequentialGenerator<Float16>(-1.0, 1.0);
    matmul_methods[1].fn_is_supported = cpu_has_sme2;
    matmul_methods[1].fn_get_mr = kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_get_nr = kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_get_kr = kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_get_sr = kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme;
    matmul_methods[1].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme;
    matmul_methods[1].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_pack_lhs = kai_run_lhs_pack_x16p2vlx2_x16_sme;
    matmul_methods[1].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_get_pack_rhs_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_pack_rhs = kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_pack_rhs_nxk_get_n_step = kai_get_n_step_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_pack_rhs_nxk_get_rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_pack_rhs_nxk_get_bias_offset = kai_get_bias_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_pack_rhs_nxk_get_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_pack_rhs_nxk_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_pack_rhs_nxk = kai_run_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[1].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    matmul_methods[1].fn_matmul_f16_f16p_f16p = kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;

    matmul_methods[2].name = "matmul_nt_nt_fp32_fp32_fp32_6x8_neon_mla";
    matmul_methods[2].m0 = 6;
    matmul_methods[2].n0 = 8;
    matmul_methods[2].dst_format = DataFormat(DataType::FP32);
    matmul_methods[2].lhs_format = DataFormat(DataType::FP32);
    matmul_methods[2].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    matmul_methods[2].rhs_format = DataFormat(DataType::FP32);
    matmul_methods[2].packed_rhs_format =
        DataFormat(DataType::FP32, 8, 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 8, 1);
    matmul_methods[2].bias_format = DataFormat(DataType::FP32);
    matmul_methods[2].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    matmul_methods[2].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    matmul_methods[2].fn_generate_bias = UniformRandomGenerator<float>(-1.0, 1.0, 3);
    matmul_methods[2].fn_is_supported = cpu_has_advsimd;
    matmul_methods[2].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon;
    matmul_methods[2].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon;
    matmul_methods[2].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon;
    matmul_methods[2].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon;
    matmul_methods[2].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_pack_rhs = kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon;
    matmul_methods[2].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon;
    matmul_methods[2].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;
    matmul_methods[2].fn_matmul_f32_f32_f32p = kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla;

    matmul_methods[3].name = "matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa";
    matmul_methods[3].m0 = 2 * get_sme_vector_length<float>();
    matmul_methods[3].n0 = 2 * get_sme_vector_length<float>();
    matmul_methods[3].dst_format = DataFormat(DataType::FP32);
    matmul_methods[3].lhs_format = DataFormat(DataType::FP32);
    matmul_methods[3].packed_lhs_format = DataFormat(DataType::FP32, 2 * get_sme_vector_length<float>(), 1);
    matmul_methods[3].rhs_format = DataFormat(DataType::FP32);
    matmul_methods[3].packed_rhs_format = DataFormat(
        DataType::FP32, 2 * get_sme_vector_length<float>(), 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32,
        DataType::UNKNOWN, 2 * get_sme_vector_length<float>(), 1);
    matmul_methods[3].bias_format = DataFormat(DataType::FP32);
    matmul_methods[3].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    matmul_methods[3].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    matmul_methods[3].fn_generate_bias = UniformRandomGenerator<float>(-1.0, 1.0, 3);
    matmul_methods[3].fn_is_supported = cpu_has_sme2;
    matmul_methods[3].fn_get_mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme;
    matmul_methods[3].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme;
    matmul_methods[3].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_pack_lhs = kai_run_lhs_pack_f32p2vlx1_f32_sme;
    matmul_methods[3].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_pack_rhs = kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_pack_rhs_nxk_get_n_step = kai_get_n_step_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_pack_rhs_nxk_get_rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_pack_rhs_nxk_get_bias_offset = kai_get_bias_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_pack_rhs_nxk_get_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_pack_rhs_nxk_get_packed_rhs_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_pack_rhs_nxk = kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[3].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
    matmul_methods[3].fn_matmul_f32_f32p_f32p = kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;

    matmul_methods[4].name = "matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa";
    matmul_methods[4].m0 = 2 * get_sme_vector_length<float>();
    matmul_methods[4].n0 = 2 * get_sme_vector_length<float>();
    matmul_methods[4].dst_format = DataFormat(DataType::FP32);
    matmul_methods[4].lhs_format = DataFormat(DataType::FP32);
    matmul_methods[4].packed_lhs_format = DataFormat(DataType::FP32, 2 * get_sme_vector_length<float>(), 1);
    matmul_methods[4].rhs_format = DataFormat(DataType::FP32);
    matmul_methods[4].packed_rhs_format = DataFormat(
        DataType::FP32, 2 * get_sme_vector_length<float>(), 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32,
        DataType::UNKNOWN, 2 * get_sme_vector_length<float>(), 1);
    matmul_methods[4].bias_format = DataFormat(DataType::FP32);
    matmul_methods[4].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    matmul_methods[4].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    matmul_methods[4].fn_generate_bias = UniformRandomGenerator<float>(-1.0, 1.0, 3);
    matmul_methods[4].fn_is_supported = cpu_has_sme;
    matmul_methods[4].fn_get_mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme;
    matmul_methods[4].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme;
    matmul_methods[4].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_pack_lhs = kai_run_lhs_pack_f32p2vlx1_f32_sme;
    matmul_methods[4].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_pack_rhs = kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_pack_rhs_nxk_get_n_step = kai_get_n_step_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_pack_rhs_nxk_get_rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_pack_rhs_nxk_get_bias_offset = kai_get_bias_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_pack_rhs_nxk_get_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_pack_rhs_nxk_get_packed_rhs_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_pack_rhs_nxk = kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    matmul_methods[4].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    matmul_methods[4].fn_matmul_f32_f32p_f32p = kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;

    matmul_methods[5].name = "matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa";
    matmul_methods[5].m0 = 2 * get_sme_vector_length<float>();
    matmul_methods[5].n0 = 2 * get_sme_vector_length<float>();
    matmul_methods[5].dst_format = DataFormat(DataType::FP16);
    matmul_methods[5].lhs_format = DataFormat(DataType::FP16);
    matmul_methods[5].packed_lhs_format = DataFormat(DataType::FP16, 2 * get_sme_vector_length<float>(), 2);
    matmul_methods[5].rhs_format = DataFormat(DataType::FP16);
    matmul_methods[5].packed_rhs_format = DataFormat(
        DataType::FP16,                          // Output type
        2 * get_sme_vector_length<float>(), 2,   // Block size
        DataFormat::PackFormat::BIAS_PER_ROW,    // Data layout
        DataType::FP16,                          // Bias format
        DataType::UNKNOWN,                       // Scaling type
        2 * get_sme_vector_length<float>(), 2);  // Sub-block
    matmul_methods[5].bias_format = DataFormat(DataType::FP16);
    matmul_methods[5].fn_generate_lhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 0);
    matmul_methods[5].fn_generate_rhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 1);
    matmul_methods[5].fn_generate_bias = UniformRandomGenerator<Float16>(-1.0, 1.0, 3);
    matmul_methods[5].fn_is_supported = cpu_has_sme;
    matmul_methods[5].fn_get_mr = kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_get_nr = kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_get_kr = kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_get_sr = kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme;
    matmul_methods[5].fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme;
    matmul_methods[5].fn_get_packed_lhs_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_pack_lhs = kai_run_lhs_pack_x16p2vlx2_x16_sme;
    matmul_methods[5].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_get_pack_rhs_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_pack_rhs = kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_pack_rhs_nxk_get_n_step = kai_get_n_step_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_pack_rhs_nxk_get_rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_pack_rhs_nxk_get_bias_offset = kai_get_bias_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_pack_rhs_nxk_get_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_pack_rhs_nxk_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_pack_rhs_nxk = kai_run_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    matmul_methods[5].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    matmul_methods[5].fn_matmul_f16_f16p_f16p = kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;

    matmul_methods[6].name = "matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla";
    matmul_methods[6].m0 = 1;
    matmul_methods[6].n0 = 4 * get_sve_vector_length<float>();
    matmul_methods[6].dst_format = DataFormat(DataType::FP32);
    matmul_methods[6].lhs_format = DataFormat(DataType::FP32);
    matmul_methods[6].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    matmul_methods[6].rhs_format = DataFormat(DataType::FP32);
    matmul_methods[6].packed_rhs_format = DataFormat(
        DataType::FP32, 4 * get_sve_vector_length<float>(), 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32,
        DataType::UNKNOWN, 4 * get_sve_vector_length<float>(), 1);
    matmul_methods[6].bias_format = DataFormat(DataType::FP32);
    matmul_methods[6].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    matmul_methods[6].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    matmul_methods[6].fn_generate_bias = UniformRandomGenerator<float>(-1.0, 1.0, 3);
    matmul_methods[6].fn_is_supported = cpu_has_sve;
    matmul_methods[6].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve;
    matmul_methods[6].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve;
    matmul_methods[6].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve;
    matmul_methods[6].fn_get_pack_rhs_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve;
    matmul_methods[6].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_pack_rhs = kai_run_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve;
    matmul_methods[6].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve;
    matmul_methods[6].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;
    matmul_methods[6].fn_matmul_f32_f32_f32p = kai_run_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla;

    return matmul_methods;
}

static const auto& get_vecmul_methods() {
    // List of supported vector by matrix multiplication methods
    static std::array<MatMulMethod, 5> vecmul_methods{};

    vecmul_methods[0].name = "matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot";
    vecmul_methods[0].m0 = 1;
    vecmul_methods[0].n0 = 16 * get_sme_vector_length<float>();
    vecmul_methods[0].dst_format = DataFormat(DataType::FP16);
    vecmul_methods[0].lhs_format = DataFormat(DataType::FP16);
    vecmul_methods[0].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    vecmul_methods[0].rhs_format = DataFormat(DataType::FP16);
    vecmul_methods[0].packed_rhs_format = DataFormat(
        DataType::FP16,                          // Output type
        2 * get_sme_vector_length<float>(), 2,   // Block size
        DataFormat::PackFormat::BIAS_PER_ROW,    // Data layout
        DataType::FP16,                          // Bias format
        DataType::UNKNOWN,                       // Scaling type
        2 * get_sme_vector_length<float>(), 2);  // Sub-block
    vecmul_methods[0].bias_format = DataFormat(DataType::FP16);
    vecmul_methods[0].fn_generate_lhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 0);
    vecmul_methods[0].fn_generate_rhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 1);
    vecmul_methods[0].fn_generate_bias = UniformRandomGenerator<Float16>(-1.0, 1.0, 3);
    vecmul_methods[0].fn_is_supported = cpu_has_sme2;
    vecmul_methods[0].fn_get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
    vecmul_methods[0].fn_get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
    vecmul_methods[0].fn_get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
    vecmul_methods[0].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
    vecmul_methods[0].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[0].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
    vecmul_methods[0].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme;
    vecmul_methods[0].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[0].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[0].fn_get_pack_rhs_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[0].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
    vecmul_methods[0].fn_pack_rhs = kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[0].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[0].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
    vecmul_methods[0].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
    vecmul_methods[0].fn_matmul_f16_f16_f16p = kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;

    vecmul_methods[1].name = "matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla";
    vecmul_methods[1].m0 = 1;
    vecmul_methods[1].n0 = 8 * get_sme_vector_length<float>();
    vecmul_methods[1].dst_format = DataFormat(DataType::FP16);
    vecmul_methods[1].lhs_format = DataFormat(DataType::FP16);
    vecmul_methods[1].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    vecmul_methods[1].rhs_format = DataFormat(DataType::FP16);
    vecmul_methods[1].packed_rhs_format = DataFormat(
        DataType::FP16,                          // Output type
        2 * get_sme_vector_length<float>(), 2,   // Block size
        DataFormat::PackFormat::BIAS_PER_ROW,    // Data layout
        DataType::FP16,                          // Bias format
        DataType::UNKNOWN,                       // Scaling type
        2 * get_sme_vector_length<float>(), 2);  // Sub-block
    vecmul_methods[1].bias_format = DataFormat(DataType::FP16);
    vecmul_methods[1].fn_generate_lhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 0);
    vecmul_methods[1].fn_generate_rhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 1);
    vecmul_methods[1].fn_generate_bias = UniformRandomGenerator<Float16>(-1.0, 1.0, 3);
    vecmul_methods[1].fn_is_supported = cpu_has_sme;
    vecmul_methods[1].fn_get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;
    vecmul_methods[1].fn_get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;
    vecmul_methods[1].fn_get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;
    vecmul_methods[1].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;
    vecmul_methods[1].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[1].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;
    vecmul_methods[1].fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme;
    vecmul_methods[1].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[1].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[1].fn_get_pack_rhs_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[1].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;
    vecmul_methods[1].fn_pack_rhs = kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[1].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme;
    vecmul_methods[1].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;
    vecmul_methods[1].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;
    vecmul_methods[1].fn_matmul_f16_f16_f16p = kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla;

    vecmul_methods[2].name = "matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla";
    vecmul_methods[2].m0 = 1;
    vecmul_methods[2].n0 = 8 * get_sme_vector_length<float>();
    vecmul_methods[2].dst_format = DataFormat(DataType::FP32);
    vecmul_methods[2].lhs_format = DataFormat(DataType::FP32);
    vecmul_methods[2].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    vecmul_methods[2].rhs_format = DataFormat(DataType::FP32);
    vecmul_methods[2].packed_rhs_format = DataFormat(
        DataType::FP32,                          // Output type
        2 * get_sme_vector_length<float>(), 1,   // Block size
        DataFormat::PackFormat::BIAS_PER_ROW,    // Data layout
        DataType::FP32,                          // Bias format
        DataType::UNKNOWN,                       // Scaling type
        2 * get_sme_vector_length<float>(), 1);  // Sub-block
    vecmul_methods[2].bias_format = DataFormat(DataType::FP32);
    vecmul_methods[2].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    vecmul_methods[2].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    vecmul_methods[2].fn_generate_bias = UniformRandomGenerator<float>(-1.0, 1.0, 3);
    vecmul_methods[2].fn_is_supported = cpu_has_sme;
    vecmul_methods[2].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[2].fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[2].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[2].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[2].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_pack_rhs = kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[2].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[2].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;
    vecmul_methods[2].fn_matmul_f32_f32_f32p = kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla;

    vecmul_methods[3].name = "matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla";
    vecmul_methods[3].m0 = 1;
    vecmul_methods[3].n0 = 16 * get_sme_vector_length<float>();
    vecmul_methods[3].dst_format = DataFormat(DataType::FP32);
    vecmul_methods[3].lhs_format = DataFormat(DataType::FP32);
    vecmul_methods[3].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    vecmul_methods[3].rhs_format = DataFormat(DataType::FP32);
    vecmul_methods[3].packed_rhs_format = DataFormat(
        DataType::FP32,                          // Output type
        2 * get_sme_vector_length<float>(), 1,   // Block size
        DataFormat::PackFormat::BIAS_PER_ROW,    // Data layout
        DataType::FP32,                          // Bias format
        DataType::UNKNOWN,                       // Scaling type
        2 * get_sme_vector_length<float>(), 1);  // Sub-block
    vecmul_methods[3].bias_format = DataFormat(DataType::FP32);
    vecmul_methods[3].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    vecmul_methods[3].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    vecmul_methods[3].fn_generate_bias = UniformRandomGenerator<float>(-1.0, 1.0, 3);
    vecmul_methods[3].fn_is_supported = cpu_has_sme2;
    vecmul_methods[3].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[3].fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[3].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[3].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[3].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_pack_rhs = kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[3].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme;
    vecmul_methods[3].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
    vecmul_methods[3].fn_matmul_f32_f32_f32p = kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;

    vecmul_methods[4].name = "matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla";
    vecmul_methods[4].m0 = 1;
    vecmul_methods[4].n0 = 16 * get_sme_vector_length<float>();
    vecmul_methods[4].dst_format = DataFormat(DataType::FP32);
    vecmul_methods[4].lhs_format = DataFormat(DataType::FP32);
    vecmul_methods[4].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    vecmul_methods[4].rhs_format = DataFormat(DataType::FP32);
    vecmul_methods[4].packed_rhs_format = DataFormat(
        DataType::FP32,                           // Output type
        16 * get_sme_vector_length<float>(), 1,   // Block size
        DataFormat::PackFormat::BIAS_PER_ROW,     // Data layout
        DataType::FP32,                           // Bias format
        DataType::UNKNOWN,                        // Scaling type
        16 * get_sme_vector_length<float>(), 1);  // Sub-block
    vecmul_methods[4].bias_format = DataFormat(DataType::FP32);
    vecmul_methods[4].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    vecmul_methods[4].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    vecmul_methods[4].fn_generate_bias = UniformRandomGenerator<float>(-1.0, 1.0, 3);
    vecmul_methods[4].fn_is_supported = cpu_has_sme2;
    vecmul_methods[4].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme;
    vecmul_methods[4].fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme;
    vecmul_methods[4].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme;
    vecmul_methods[4].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme;
    vecmul_methods[4].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_pack_rhs = kai_run_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme;
    vecmul_methods[4].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme;
    vecmul_methods[4].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;
    vecmul_methods[4].fn_matmul_f32_f32_f32p = kai_run_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla;

    return vecmul_methods;
}

static const auto& get_nullbias_matmul_methods() {
    // List of supported vector by matrix multiplication methods
    static std::array<MatMulMethod, 4> nullbias_matmul_methods{};

    nullbias_matmul_methods[0].name = "matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla";
    nullbias_matmul_methods[0].m0 = 6;
    nullbias_matmul_methods[0].n0 = 16;
    nullbias_matmul_methods[0].dst_format = DataFormat(DataType::FP32);
    nullbias_matmul_methods[0].lhs_format = DataFormat(DataType::FP32);
    nullbias_matmul_methods[0].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    nullbias_matmul_methods[0].rhs_format = DataFormat(DataType::FP32);
    nullbias_matmul_methods[0].packed_rhs_format = DataFormat(
        DataType::FP32, 16, 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 16, 1);
    nullbias_matmul_methods[0].bias_format = DataFormat(DataType::FP32);
    nullbias_matmul_methods[0].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    nullbias_matmul_methods[0].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    nullbias_matmul_methods[0].fn_generate_bias = ConstantGenerator<float>(0.0);
    nullbias_matmul_methods[0].fn_is_supported = cpu_has_advsimd;
    nullbias_matmul_methods[0].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[0].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[0].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[0].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[0].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_pack_rhs = kai_run_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[0].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[0].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;
    nullbias_matmul_methods[0].fn_matmul_f32_f32_f32p = kai_run_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla;

    nullbias_matmul_methods[1].name = "matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55";
    nullbias_matmul_methods[1].m0 = 6;
    nullbias_matmul_methods[1].n0 = 16;
    nullbias_matmul_methods[1].dst_format = DataFormat(DataType::FP32);
    nullbias_matmul_methods[1].lhs_format = DataFormat(DataType::FP32);
    nullbias_matmul_methods[1].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    nullbias_matmul_methods[1].rhs_format = DataFormat(DataType::FP32);
    nullbias_matmul_methods[1].packed_rhs_format = DataFormat(
        DataType::FP32, 16, 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 16, 1);
    nullbias_matmul_methods[1].bias_format = DataFormat(DataType::FP32);
    nullbias_matmul_methods[1].fn_generate_lhs = UniformRandomGenerator<float>(-1.0, 1.0, 0);
    nullbias_matmul_methods[1].fn_generate_rhs = UniformRandomGenerator<float>(-1.0, 1.0, 1);
    nullbias_matmul_methods[1].fn_generate_bias = ConstantGenerator<float>(0.0);
    nullbias_matmul_methods[1].fn_is_supported = cpu_has_advsimd;
    nullbias_matmul_methods[1].fn_get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_get_main_m_step =
        kai_get_m_step_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[1].fn_get_main_n_step =
        kai_get_n_step_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_get_lhs_offset =
        kai_get_lhs_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[1].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[1].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[1].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_pack_rhs = kai_run_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[1].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x32p16x1b_x32_x32_neon;
    nullbias_matmul_methods[1].fn_get_dst_offset =
        kai_get_dst_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_get_dst_size =
        kai_get_dst_size_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;
    nullbias_matmul_methods[1].fn_matmul_f32_f32_f32p = kai_run_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55;

    nullbias_matmul_methods[2].name = "matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla";
    nullbias_matmul_methods[2].m0 = 6;
    nullbias_matmul_methods[2].n0 = 32;
    nullbias_matmul_methods[2].dst_format = DataFormat(DataType::FP16);
    nullbias_matmul_methods[2].lhs_format = DataFormat(DataType::FP16);
    nullbias_matmul_methods[2].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    nullbias_matmul_methods[2].rhs_format = DataFormat(DataType::FP16);
    nullbias_matmul_methods[2].packed_rhs_format = DataFormat(
        DataType::FP16, 32, 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP16, DataType::UNKNOWN, 32, 1);
    nullbias_matmul_methods[2].bias_format = DataFormat(DataType::FP16);
    nullbias_matmul_methods[2].fn_generate_lhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 0);
    nullbias_matmul_methods[2].fn_generate_rhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 1);
    nullbias_matmul_methods[2].fn_generate_bias = ConstantGenerator<Float16>(0.0);
    nullbias_matmul_methods[2].fn_is_supported = cpu_has_fp16;
    nullbias_matmul_methods[2].fn_get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_get_main_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[2].fn_get_main_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[2].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[2].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[2].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_pack_rhs = kai_run_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[2].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[2].fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;
    nullbias_matmul_methods[2].fn_matmul_f16_f16_f16p = kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla;

    nullbias_matmul_methods[3].name = "matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55";
    nullbias_matmul_methods[3].m0 = 6;
    nullbias_matmul_methods[3].n0 = 32;
    nullbias_matmul_methods[3].dst_format = DataFormat(DataType::FP16);
    nullbias_matmul_methods[3].lhs_format = DataFormat(DataType::FP16);
    nullbias_matmul_methods[3].packed_lhs_format = DataFormat(DataType::UNKNOWN);
    nullbias_matmul_methods[3].rhs_format = DataFormat(DataType::FP16);
    nullbias_matmul_methods[3].packed_rhs_format = DataFormat(
        DataType::FP16, 32, 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP16, DataType::UNKNOWN, 32, 1);
    nullbias_matmul_methods[3].bias_format = DataFormat(DataType::FP16);
    nullbias_matmul_methods[3].fn_generate_lhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 0);
    nullbias_matmul_methods[3].fn_generate_rhs = UniformRandomGenerator<Float16>(-1.0, 1.0, 1);
    nullbias_matmul_methods[3].fn_generate_bias = ConstantGenerator<Float16>(0.0);
    nullbias_matmul_methods[3].fn_is_supported = cpu_has_fp16;
    nullbias_matmul_methods[3].fn_get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_get_main_m_step =
        kai_get_m_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[3].fn_get_main_n_step =
        kai_get_n_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_get_lhs_offset =
        kai_get_lhs_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[3].fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[3].fn_get_pack_rhs_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[3].fn_get_main_packed_rhs_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_pack_rhs = kai_run_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[3].fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon;
    nullbias_matmul_methods[3].fn_get_dst_offset =
        kai_get_dst_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_get_dst_size =
        kai_get_dst_size_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;
    nullbias_matmul_methods[3].fn_matmul_f16_f16_f16p = kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55;

    return nullbias_matmul_methods;
}

/// Matrix multiplication test fixture.
class MatMulTest : public testing::TestWithParam<MatMulTestParams> {
private:
    /// Unique ID: m, n, k, method_id.
    using TestDataId = std::tuple<size_t, size_t, size_t, std::string_view, BiasMode>;

protected:
    /// Cached test data that is shared between multiple test case.
    struct TestData {
        Buffer lhs{};             ///< LHS operand.
        Buffer ref_packed_lhs{};  ///< Reference packed LHS.
        Buffer rhs{};             ///< RHS operand.
        Buffer rhs_scales{};      ///< RHS per-row quantization scales.
        Buffer bias{};            ///< Bias.
        Buffer rhs_t{};           ///< Transposed RHS matrix.
        Buffer ref_packed_rhs{};  ///< Reference packed RHS.
        Buffer ref_dst{};         ///< Reference output.
        float clamp_min{};        ///< Minimum output value.
        float clamp_max{};        ///< Maximum output value.
    };

    /// Gets the test data for the current test case.
    static const TestData& test_data() {
        const auto& [method, info, portion, bias_mode] = GetParam();
        const TestDataId data_id{info.m, info.n, info.k, method.name, bias_mode};

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
        auto lhs = method.fn_generate_lhs(lhs_h, lhs_w);
        Buffer ref_packed_lhs;

        if (has_lhs_pack) {
            ref_packed_lhs =
                pack(method.packed_lhs_format, lhs.data(), nullptr, nullptr, method.lhs_format, lhs_h, lhs_w);
        }

        const auto rhs_h = info.k;
        const auto rhs_w = info.n;
        auto rhs = method.fn_generate_rhs(rhs_h, rhs_w);

        KAI_ASSUME_ALWAYS(method.rhs_format.is_raw());
        auto rhs_t = transpose(rhs.data(), method.rhs_format.data_type(), rhs_h, rhs_w);

        Buffer rhs_scales;
        if (data_type_is_quantized(method.rhs_format.data_type()) &&
            method.rhs_format.pack_format() == DataFormat::PackFormat::NONE) {
            const auto rhs_scales_gen = UniformRandomGenerator<float>(0.0, 1.0, 2);
            rhs_scales = rhs_scales_gen(rhs_h, 1);
        }

        const auto bias_h = 1;
        const auto bias_w = info.n;
        Buffer bias;

        if (has_bias) {
            bias = method.fn_generate_bias(bias_h, bias_w);
        }

        Buffer packed_rhs;
        if (has_rhs_pack) {
            packed_rhs = matmul_pack_rhs(
                rhs.data(), rhs_scales.data(), bias.data(), method.rhs_format, method.packed_rhs_format, info.n, info.k,
                true);
        }

        KAI_ASSUME_ALWAYS(method.lhs_format.is_raw());
        KAI_ASSUME_ALWAYS(method.rhs_format.is_raw());
        KAI_ASSUME_ALWAYS(method.dst_format.is_raw());
        auto ref_dst = matmul(
            lhs.data(), nullptr, nullptr, method.lhs_format.data_type(),            //
            rhs.data(), rhs_scales.data(), nullptr, method.rhs_format.data_type(),  //
            bias.data(), nullptr, nullptr, method.bias_format.data_type(),          //
            method.dst_format.data_type(),                                          //
            info.m, info.n, info.k, false, false);

        static constexpr float clamp_ratio = 0.8F;
        const auto [clamp_min, clamp_max] =
            find_clamp_range(method.dst_format.data_type(), ref_dst.data(), info.m * info.n, clamp_ratio);
        ref_dst = clamp(method.dst_format.data_type(), ref_dst.data(), info.m * info.n, clamp_min, clamp_max);

        auto& data = _data[data_id] = {};
        data.lhs = std::move(lhs);
        data.ref_packed_lhs = std::move(ref_packed_lhs);
        data.rhs = std::move(rhs);
        data.rhs_scales = std::move(rhs_scales);
        data.bias = std::move(bias);
        data.rhs_t = std::move(rhs_t);
        data.ref_packed_rhs = std::move(packed_rhs);
        data.ref_dst = std::move(ref_dst);
        data.clamp_min = clamp_min;
        data.clamp_max = clamp_max;

        return data;
    }

private:
    // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
    static std::map<TestDataId, TestData> _data;
    // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::map<MatMulTest::TestDataId, MatMulTest::TestData> MatMulTest::_data;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/// Tests the LHS packing micro-kernel.
TEST_P(MatMulTest, PackedLhs) {
    const auto& [method, info, portion, bias_mode] = GetParam();

    if (method.fn_is_supported && !method.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    if (!method.is_pack_lhs_needed()) {
        GTEST_SKIP() << "Test not valid w/o LHS pack";
    }

    const auto& data = test_data();
    const auto lhs_h = info.m;
    const auto lhs_w = info.k;

    const auto rect = portion.compute_portion(
        lhs_h, lhs_w, method.packed_lhs_format.scheduler_block_height(lhs_h),
        lhs_w);  // LHS packing micro-kernel API doesn't support scheduling over K dimension.

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto mr = method.fn_get_mr();
    const auto kr = method.fn_get_kr();
    const auto sr = method.fn_get_sr();
    const auto ref_lhs_row_stride = method.lhs_format.default_row_stride(lhs_w);

    const auto packed_lhs_size = method.fn_get_packed_lhs_size(info.m, info.k, mr, kr, sr);
    const auto ref_packed_lhs_size = method.packed_lhs_format.default_size_in_bytes(lhs_h, lhs_w);
    ASSERT_EQ(packed_lhs_size, ref_packed_lhs_size);

    const auto lhs_offset = method.fn_get_lhs_offset(rect.start_row(), ref_lhs_row_stride);
    const auto ref_lhs_offset = method.lhs_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), lhs_w);
    ASSERT_EQ(lhs_offset, ref_lhs_offset);

    const auto packed_lhs_offset = method.fn_get_packed_lhs_offset(rect.start_row(), info.k);
    const auto ref_packed_lhs_offset = method.packed_lhs_format.default_offset_in_bytes(rect.start_row(), 0, lhs_w);
    ASSERT_EQ(packed_lhs_offset, ref_packed_lhs_offset);

    Buffer packed_lhs(packed_lhs_size, 0);
    abi_check(
        method.fn_pack_lhs, rect.height(), rect.width(), mr, kr, sr, 0, data.lhs.data() + lhs_offset,
        ref_lhs_row_stride, packed_lhs.data() + packed_lhs_offset);

    DefaultMismatchHandler handler(0, 0.0001, 0, 0.001);
    const auto success =
        compare(packed_lhs.data(), data.ref_packed_lhs.data(), method.packed_lhs_format, lhs_h, lhs_w, rect, handler);
    ASSERT_TRUE(success);
}

/// Tests the RHS packing micro-kernel.
TEST_P(MatMulTest, PackedRhs) {
    const auto& [method, info, portion, bias_mode] = GetParam();

    if (method.fn_is_supported && !method.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    if (!method.is_pack_rhs_needed()) {
        GTEST_SKIP() << "Test not valid w/o RHS pack";
    }

    const auto& data = test_data();
    const auto rhs_full_width = info.n;
    const auto rhs_full_height = info.k;

    const auto block_height = method.packed_rhs_format.scheduler_block_height(rhs_full_width);
    const auto block_width = method.packed_rhs_format.scheduler_block_width(rhs_full_height);

    const auto null_bias_mode = bias_mode == BiasMode::INTERNAL;

    const Rect rect = portion.compute_portion(rhs_full_width, rhs_full_height, block_height, block_width);

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto rhs_start_row = rect.start_row();
    const auto rhs_start_col = rect.start_col();
    const auto width = rect.width();
    const auto height = rect.height();
    const auto rhs_row_stride = method.rhs_format.default_row_stride(rhs_full_width);

    /** Ensure that all relevant parameters are sane **/
    const auto n_step = method.fn_get_pack_rhs_n_step();
    const auto ref_n_step = block_height;
    ASSERT_EQ(n_step, ref_n_step);

    const auto rhs_offset = method.fn_get_rhs_offset(rhs_start_row);
    const auto ref_rhs_offset =
        method.rhs_format.default_offset_in_bytes(rhs_start_col, rhs_start_row, rhs_full_height);
    ASSERT_EQ(rhs_offset, ref_rhs_offset);

    const auto packed_rhs_size = method.fn_get_packed_rhs_size(rhs_full_width, rhs_full_height);
    const auto ref_packed_rhs_size = method.packed_rhs_format.default_size_in_bytes(rhs_full_width, rhs_full_height);
    ASSERT_EQ(packed_rhs_size, ref_packed_rhs_size);

    const auto packed_rhs_offset = method.fn_get_pack_rhs_packed_rhs_offset(rhs_start_row, rhs_full_height);
    const auto ref_packed_rhs_offset =
        method.packed_rhs_format.default_offset_in_bytes(rhs_start_row, rhs_start_col, rhs_full_height);
    ASSERT_EQ(packed_rhs_offset, ref_packed_rhs_offset);

    const auto scale_type = method.packed_rhs_format.scale_data_type();
    const auto ref_rhs_scales_offset = rhs_start_row * data_type_size_in_bits(scale_type) / 8;

    const auto bias_offset = method.fn_get_bias_offset(rhs_start_row);
    const auto ref_bias_offset =
        !null_bias_mode ? method.bias_format.default_offset_in_bytes(0, rhs_start_row, rhs_full_height) : bias_offset;
    ASSERT_EQ(bias_offset, ref_bias_offset);

    /** Perform RHS packing, and compare with reference result **/
    Buffer packed_rhs(packed_rhs_size, 0);
    abi_check(
        &MatMulMethod::pack_rhs, method, height, width, data.rhs.data() + rhs_offset, rhs_row_stride,
        !null_bias_mode ? data.bias.data() + bias_offset : nullptr,
        data.rhs_scales.data() != nullptr ? data.rhs_scales.data() + ref_rhs_scales_offset : nullptr,
        packed_rhs.data() + packed_rhs_offset);

    const bool exact = method.packed_rhs_format.pack_format() != DataFormat::PackFormat::QUANTIZE_PER_ROW;
    DefaultMismatchHandler handler(0, exact ? 0 : 0.0001, 0, exact ? 0 : 0.001);
    const auto success = compare(
        packed_rhs.data(), data.ref_packed_rhs.data(), method.packed_rhs_format, rhs_full_width, rhs_full_height, rect,
        handler);
    ASSERT_TRUE(success);
}

/// Tests the transposed RHS packing micro-kernel.
TEST_P(MatMulTest, PackedTransposedRhs) {
    const auto& [method, info, portion, bias_mode] = GetParam();

    if (method.fn_is_supported && !method.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    if (!method.is_pack_rhs_nxk_needed()) {
        GTEST_SKIP() << "Test not valid w/o pre-processing of transposed RHS matrix";
    }

    const auto& data = test_data();
    const auto n_step = method.fn_pack_rhs_nxk_get_n_step();
    const auto ref_n_step = method.packed_rhs_format.scheduler_block_height(info.n);
    ASSERT_EQ(n_step, ref_n_step);

    const auto rect = portion.compute_portion(
        info.n, info.k, method.packed_rhs_format.scheduler_block_height(info.n),
        method.packed_rhs_format.scheduler_block_width(info.k));

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto ref_rhs_row_stride = method.rhs_format.default_row_stride(info.k);

    const auto rhs_offset = method.fn_pack_rhs_nxk_get_rhs_offset(rect.start_row(), ref_rhs_row_stride);
    const auto ref_rhs_offset = method.rhs_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), info.k);
    ASSERT_EQ(rhs_offset, ref_rhs_offset);

    const auto packed_rhs_size = method.fn_pack_rhs_nxk_get_packed_rhs_size(info.n, info.k);
    const auto ref_packed_rhs_size = method.packed_rhs_format.default_size_in_bytes(info.n, info.k);
    ASSERT_EQ(packed_rhs_size, ref_packed_rhs_size);

    const auto packed_rhs_offset = method.fn_pack_rhs_nxk_get_packed_rhs_offset(rect.start_row(), info.k);
    const auto ref_packed_rhs_offset =
        method.packed_rhs_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), info.k);
    ASSERT_EQ(packed_rhs_offset, ref_packed_rhs_offset);

    const auto ref_rhs_scales_offset =
        rect.start_row() * data_type_size_in_bits(method.packed_rhs_format.scale_data_type()) / 8;

    const auto bias_offset = method.fn_get_bias_offset(rect.start_row());
    const auto ref_bias_offset = method.bias_format.default_offset_in_bytes(0, rect.start_row(), info.n);
    ASSERT_EQ(bias_offset, ref_bias_offset);

    Buffer packed_rhs(packed_rhs_size, 0);

    abi_check(
        &MatMulMethod::pack_rhs_nxk, method, rect.height(), rect.width(), data.rhs_t.data() + rhs_offset,
        ref_rhs_row_stride, data.bias.data() + bias_offset,
        data.rhs_scales.data() != nullptr ? data.rhs_scales.data() + ref_rhs_scales_offset : nullptr,
        packed_rhs.data() + packed_rhs_offset);

    const auto exact = method.packed_rhs_format.pack_format() != DataFormat::PackFormat::QUANTIZE_PER_ROW;
    DefaultMismatchHandler handler(0, exact ? 0 : 0.0001, 0, exact ? 0 : 0.001);
    const auto success =
        compare(packed_rhs.data(), data.ref_packed_rhs.data(), method.packed_rhs_format, info.n, info.k, rect, handler);
    ASSERT_TRUE(success);
}

/// Tests the output.
TEST_P(MatMulTest, Output) {
    const auto& [method, info, portion, bias_mode] = GetParam();

    if (method.fn_is_supported && !method.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    if (!method.has_main_kernel()) {
        GTEST_SKIP() << "No main kernel available";
    }

    const auto& data = test_data();
    const auto m_step = method.fn_get_main_m_step();
    ASSERT_EQ(m_step, method.m0);

    const auto n_step = method.fn_get_main_n_step();
    ASSERT_EQ(n_step, method.n0);

    const auto rect = portion.compute_portion(info.m, info.n, method.m0, method.n0);

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_w = info.k;
    const auto rhs_w = info.n;
    const auto bias_w = info.n;
    const auto dst_w = info.n;

    const auto lhs_start_row = rect.start_row();
    const auto lhs_start_col = 0;
    const auto lhs_stride = method.lhs_format.default_row_stride(lhs_w);

    const std::byte* lhs_data = nullptr;
    uintptr_t lhs_offset = 0;

    if (method.is_pack_lhs_needed()) {
        lhs_data = data.ref_packed_lhs.data();

        const auto ref_packed_lhs_offset =
            method.packed_lhs_format.default_offset_in_bytes(lhs_start_row, lhs_start_col, info.k);
        lhs_offset = method.fn_get_packed_lhs_offset(lhs_start_row, info.k);
        ASSERT_EQ(lhs_offset, ref_packed_lhs_offset);
    } else {
        lhs_data = data.lhs.data();

        lhs_offset = method.fn_get_lhs_offset(lhs_start_row, lhs_stride);
        const auto ref_lhs_offset = method.lhs_format.default_offset_in_bytes(lhs_start_row, lhs_start_col, lhs_w);
        ASSERT_EQ(lhs_offset, ref_lhs_offset);
    }

    const auto rhs_stride = method.rhs_format.default_row_stride(rhs_w);

    const std::byte* rhs_data = nullptr;
    uintptr_t rhs_offset = 0;

    if (method.is_pack_rhs_needed()) {
        const auto packed_rhs_start_row = rect.start_col();
        const auto packed_rhs_start_col = 0;

        rhs_data = data.ref_packed_rhs.data();

        rhs_offset = method.fn_get_main_packed_rhs_offset(packed_rhs_start_row, info.k);
        const auto ref_rhs_offset =
            method.packed_rhs_format.default_offset_in_bytes(packed_rhs_start_row, packed_rhs_start_col, info.k);
        ASSERT_EQ(rhs_offset, ref_rhs_offset);
    } else {
        const auto rhs_start_row = 0;
        const auto rhs_start_col = rect.start_col();

        rhs_data = data.rhs.data();
        rhs_offset = method.rhs_format.default_offset_in_bytes(rhs_start_row, rhs_start_col, rhs_w);
    }

    const auto* bias_data = data.bias.data();
    const auto bias_offset = method.bias_format.default_offset_in_bytes(0, rect.start_row(), bias_w);

    const auto dst_stride = method.dst_format.default_row_stride(dst_w);
    const auto dst_offset = method.fn_get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = method.dst_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), dst_w);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    const auto dst_size = method.fn_get_dst_size(info.m, info.n);
    const auto ref_dst_size = method.dst_format.default_size_in_bytes(info.m, info.n);
    ASSERT_EQ(dst_size, ref_dst_size);

    Buffer dst(dst_size, 0);

    abi_check(
        &MatMulMethod::main_kernel, method, rect.height(), rect.width(), info.k, lhs_data + lhs_offset,
        rhs_data + rhs_offset, bias_data + bias_offset, dst.data() + dst_offset, lhs_stride, rhs_stride, dst_stride,
        data.clamp_min, data.clamp_max);

    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    const auto success = compare(dst.data(), data.ref_dst.data(), method.dst_format, info.m, info.n, rect, handler);
    ASSERT_TRUE(success);
}

const std::vector<MatrixPortion> MatrixPortions = {
    {0, 0, 1, 1},
    {0, 0, 0.25, 0.25},
    {0.75, 0.75, 1, 1},
};
const std::vector<MatMulShape> MatMulShapes = {
    {1, 16, 16},   //
    {20, 1, 20},   //
    {6, 16, 32},   //
    {12, 32, 17},  //
    {13, 33, 23},  //
    {87, 93, 56},  //
};

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest,
    testing::Combine(
        testing::ValuesIn(get_matmul_methods()),  //
        testing::ValuesIn(MatMulShapes),          //
        testing::ValuesIn(MatrixPortions),        //
        testing::Values(BiasMode::PROVIDED)       //
        ),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    NullBiasMatMul, MatMulTest,
    testing::Combine(
        testing::ValuesIn(get_nullbias_matmul_methods()),        //
        testing::ValuesIn(MatMulShapes),                         //
        testing::ValuesIn(MatrixPortions),                       //
        testing::Values(BiasMode::INTERNAL, BiasMode::PROVIDED)  //
        ),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    VecMul, MatMulTest,
    testing::Combine(
        testing::ValuesIn(get_vecmul_methods()),
        testing::Values(
            MatMulShape{1, 16, 16},    //
            MatMulShape{1, 1, 20},     //
            MatMulShape{1, 16, 32},    //
            MatMulShape{1, 32, 17},    //
            MatMulShape{1, 33, 23},    //
            MatMulShape{1, 1500, 20},  //
            MatMulShape{1, 93, 56},    //
            MatMulShape{1, 1, 1},      //
            MatMulShape{1, 16, 1},     //
            MatMulShape{1, 32, 64},    //
            MatMulShape{1, 7, 74},     //
            MatMulShape{1, 800, 64},   //
            MatMulShape{1, 512, 130}   //
            ),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),      // Full row.
            MatrixPortion(0, 0, 1, 0.5),    // First half
            MatrixPortion(0, .4, 1, 0.3),   // mid row-section.
            MatrixPortion(0, 0.75, 1, .25)  // right row section
            ),
        testing::Values(BiasMode::PROVIDED)),
    testing::PrintToStringParamName());

}  // namespace kai::test
