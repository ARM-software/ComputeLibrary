//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul_operator.hpp"

#include <array>
#include <memory>
#include <optional>

#include "test/common/cpu_info.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/functions/round.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper.hpp"
#include "test/nextgen/quantization/asymm_linear_quantizer.hpp"
#include "test/nextgen/quantization/symm_linear_quantizer.hpp"

namespace kai::test {

Span<const MatMulOperator> get_available_matmul_operators() {
    static std::array<MatMulOperator, 2> operators;

    // matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa
    operators[0].name = "matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa";

    operators[0].is_cpu_supported = []() { return cpu_has_sme2(); };
    operators[0].is_shape_suitable = [](size_t, size_t, size_t) { return true; };

    operators[0].supported_bias_modes = {MatMulBiasMode::NO_BIAS, MatMulBiasMode::PER_N};

    operators[0].lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    operators[0].rhs_quant =
        std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    operators[0].bias_quant = std::nullopt;

    operators[0].acc_dtype = DataType::FP32;
    operators[0].dst_dtype = DataType::FP32;

    operators[0].pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32();
    operators[0].pack_rhs = create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon();
    operators[0].matmul = create_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();

    // matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot
    operators[1].name = "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot";

    operators[1].is_cpu_supported = []() { return cpu_has_sme2(); };
    operators[1].is_shape_suitable = [](size_t, size_t, size_t) { return true; };

    operators[1].supported_bias_modes = {MatMulBiasMode::NO_BIAS, MatMulBiasMode::PER_N};

    operators[1].lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    operators[1].rhs_quant =
        std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    operators[1].bias_quant = std::nullopt;

    operators[1].acc_dtype = DataType::FP32;
    operators[1].dst_dtype = DataType::FP32;

    operators[1].pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1x4_f32();
    operators[1].pack_rhs = create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon();
    operators[1].matmul = create_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();

    return operators;
}

}  // namespace kai::test
