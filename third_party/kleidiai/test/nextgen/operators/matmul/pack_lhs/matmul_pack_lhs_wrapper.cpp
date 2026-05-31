//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper.hpp"

#include <array>
#include <memory>
#include <string>
#include <string_view>

#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "test/common/data_type.hpp"
#include "test/common/sme.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_dq_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_interface.hpp"

namespace kai::test {

namespace {

std::unique_ptr<KernelWrapper> create_matmul_lhs_quant_pack_qai8dxp_f32(
    std::string_view block_name, size_t block_height, size_t block_width) {
    return std::make_unique<MatMulPackLhsDqWrapper>(
        "matmul_lhs_quant_pack_qai8dxp" + std::string(block_name) + "_f32",
        MatMulPackLhsDqInterface{
            kai_get_m_step_lhs_quant_pack_qai8dxp_f32,
            kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
            kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
            kai_run_lhs_quant_pack_qai8dxp_f32,
        },
        make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            block_height, block_width, 32, true, DataType::I8, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32}));
}

}  // namespace

std::unique_ptr<KernelWrapper> create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32() {
    return create_matmul_lhs_quant_pack_qai8dxp_f32("1vlx4", 1 * get_sme_vector_length<float>(), 4);
}

std::unique_ptr<KernelWrapper> create_matmul_lhs_quant_pack_qai8dxp1x4_f32() {
    return create_matmul_lhs_quant_pack_qai8dxp_f32("1x4", 1, 4);
}

}  // namespace kai::test
