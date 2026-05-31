//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper.hpp"

#include <array>
#include <memory>

#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.h"
#include "test/common/data_type.hpp"
#include "test/common/sme.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_interface.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_quant_wrapper.hpp"

namespace kai::test {

std::unique_ptr<KernelWrapper> create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon() {
    return std::make_unique<MatMulPackRhsQuantWrapper>(
        "matmul_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon",
        MatMulPackRhsQuantInterface{
            kai_get_n_step_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_get_rhs_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
        },
        make_poly<PlainFormat>(DataType::U4), make_poly<PlainFormat>(DataType::FP32),
        make_poly<PlainFormat>(DataType::FP32), make_poly<PlainFormat>(DataType::I32),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_length<float>(), 4, 32, false, DataType::I4, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32, DataType::FP32}));
}

}  // namespace kai::test
