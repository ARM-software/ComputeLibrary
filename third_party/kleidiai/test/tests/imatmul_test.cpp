//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p_f16p_interface.h"
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/sme.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/reorder.hpp"

namespace kai::test {

// Ensure static linkage for all functionality local to this test file
namespace {

/// Convenience wrapper for K-chunk handling
struct KChunk {
    size_t count;
    size_t length;
};

/// Interface for indirect matmul LHS packing micro-kernel
struct LhsPackIndirectKernel {
    std::function<size_t()> get_m_step;
    std::function<size_t(size_t m_idx, size_t k_chunk_count, size_t k_chunk_length)> get_lhs_packed_offset;
    std::function<size_t(size_t m, size_t k_chunk_count, size_t k_chunk_length)> get_lhs_packed_size;
    std::function<void(
        size_t m, size_t k_chunk_count, size_t k_chunk_length, const void* const* lhs_ptrs, size_t lhs_ptr_offset,
        const void* zero, void* lhs_packed)>
        pack;
};

/// Interface for indirect matmul RHS packing micro-kernel
struct RhsPackIndirectKernel {
    std::function<size_t()> get_n_step;
    std::function<size_t(size_t n_idx)> get_rhs_offset;
    std::function<size_t(size_t n_idx)> get_bias_offset;
    std::function<size_t(size_t n_idx, size_t k_chunk_count, size_t k_chunk_length)> get_rhs_packed_offset;
    std::function<size_t(size_t n, size_t k_chunk_count, size_t k_chunk_length)> get_rhs_packed_size;
    std::function<void(
        size_t n, size_t k_chunk_count, size_t k_chunk_length, size_t rhs_row_stride, const void* rhs, const void* bias,
        void* rhs_packed)>
        pack;
};

/// Interface for indirect matmul kernel
struct MatMulIndirectKernel {
    std::function<size_t(void)> get_m_step;
    std::function<size_t(void)> get_n_step;
    std::function<size_t(void)> get_mr;
    std::function<size_t(void)> get_nr;
    std::function<size_t(void)> get_kr;
    std::function<size_t(size_t m_idx, size_t k_chunk_count, size_t k_chunk_length)> get_lhs_packed_offset;
    std::function<size_t(size_t n_idx, size_t k_chunk_count, size_t k_chunk_length)> get_rhs_packed_offset;
    std::function<size_t(size_t m_idx, size_t n_idx, size_t dst_stride_row)> get_dst_offset;
    std::function<size_t(size_t m, size_t n)> get_dst_size;
    std::function<void(
        size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length, const void* lhs_packed, const void* rhs_packed,
        void* dst, size_t dst_stride_row, float clamp_min, float clamp_max)>
        imatmul;
};

/// Description of a Indirect Matmul kernel set
struct IndirectMatMul {
    std::string_view name;
    std::function<bool(void)> is_supported;

    MatMulShape pack_shape;
    struct Format {
        DataFormat lhs;
        DataFormat rhs;
        DataFormat bias;
        DataFormat out;

        struct Hash {
            size_t operator()(const Format& format) const {
                return                                        //
                    (DataFormat::Hash{}(format.lhs) << 0) ^   //
                    (DataFormat::Hash{}(format.rhs) << 1) ^   //
                    (DataFormat::Hash{}(format.bias) << 2) ^  //
                    (DataFormat::Hash{}(format.out) << 3);
            }
        };

    private:
        friend bool operator==(const Format& lhs, const Format& rhs) {
            return                       //
                lhs.lhs == rhs.lhs &&    //
                lhs.rhs == rhs.rhs &&    //
                lhs.bias == rhs.bias &&  //
                lhs.out == rhs.out;
        }
    } format;

    LhsPackIndirectKernel lhs;
    RhsPackIndirectKernel rhs;
    MatMulIndirectKernel imatmul;
};

/// Test parameter bundle type
using IndirectMatMulTestParams = std::tuple<IndirectMatMul, MatMulShape, size_t, MatrixPortion, float>;

/// Test type
using IndirectMatMulTest = testing::TestWithParam<IndirectMatMulTestParams>;

/// Use interface for matmul kernel
const kai_imatmul_clamp_f16_f16p_f16p_ukernel& get_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa() {
    static kai_imatmul_clamp_f16_f16p_f16p_ukernel ukernel;
    ukernel.get_m_step = kai_get_m_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    ukernel.get_n_step = kai_get_n_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    ukernel.get_lhs_packed_offset = kai_get_lhs_packed_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    ukernel.get_rhs_packed_offset = kai_get_rhs_packed_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    ukernel.get_dst_offset = kai_get_dst_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    ukernel.get_dst_size = kai_get_dst_size_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    ukernel.run_imatmul = kai_run_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
    return ukernel;
}

const kai_imatmul_clamp_f16_f16p_f16p_ukernel& get_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa() {
    static kai_imatmul_clamp_f16_f16p_f16p_ukernel ukernel;
    ukernel.get_m_step = kai_get_m_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    ukernel.get_n_step = kai_get_n_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    ukernel.get_lhs_packed_offset = kai_get_lhs_packed_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    ukernel.get_rhs_packed_offset = kai_get_rhs_packed_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    ukernel.get_dst_offset = kai_get_dst_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    ukernel.get_dst_size = kai_get_dst_size_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    ukernel.run_imatmul = kai_run_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa;
    return ukernel;
}

/// Use interface for matmul kernel
const kai_imatmul_clamp_f32_f32p_f32p_ukernel& get_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa() {
    static kai_imatmul_clamp_f32_f32p_f32p_ukernel ukernel;
    ukernel.get_m_step = kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa;
    ukernel.get_n_step = kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa;
    ukernel.get_lhs_packed_offset = kai_get_lhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa;
    ukernel.get_rhs_packed_offset = kai_get_rhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa;
    ukernel.get_dst_offset = kai_get_dst_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa;
    ukernel.get_dst_size = kai_get_dst_size_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa;
    ukernel.run_imatmul = kai_run_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa;
    return ukernel;
}

const kai_imatmul_clamp_f32_f32p_f32p_ukernel& get_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() {
    static kai_imatmul_clamp_f32_f32p_f32p_ukernel ukernel;
    ukernel.get_m_step = kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    ukernel.get_n_step = kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    ukernel.get_lhs_packed_offset = kai_get_lhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    ukernel.get_rhs_packed_offset = kai_get_rhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    ukernel.get_dst_offset = kai_get_dst_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    ukernel.get_dst_size = kai_get_dst_size_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    ukernel.run_imatmul = kai_run_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa;
    return ukernel;
}

/// Retreive the test list
const auto& get_indirect_matmul_methods() {
    static std::array<IndirectMatMul, 4> indirect_matmul_methods{};

    // F16 IMATMUL SME2 ///////////////////////////////////////////////////////
    indirect_matmul_methods[0].name = "imatmul_f16_f16p_f16p_2vlx2vl_sme2_mopa";
    indirect_matmul_methods[0].is_supported = cpu_has_sme2;
    indirect_matmul_methods[0].pack_shape.m = 2 * get_sme_vector_length<int32_t>();
    indirect_matmul_methods[0].pack_shape.n = 2 * get_sme_vector_length<int32_t>();
    indirect_matmul_methods[0].pack_shape.k = sizeof(int32_t);
    indirect_matmul_methods[0].format.lhs = DataFormat(DataType::FP16);
    indirect_matmul_methods[0].format.rhs = DataFormat(DataType::FP16);
    indirect_matmul_methods[0].format.bias = DataFormat(DataType::FP16);
    indirect_matmul_methods[0].format.out = DataFormat(DataType::FP16);

    // LHS
    indirect_matmul_methods[0].lhs.get_m_step = kai_get_m_step_lhs_imatmul_pack_x16p2vlx2_x16p_sme;
    indirect_matmul_methods[0].lhs.get_lhs_packed_offset =
        kai_get_lhs_packed_offset_lhs_imatmul_pack_x16p2vlx2_x16p_sme;
    indirect_matmul_methods[0].lhs.get_lhs_packed_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme;
    indirect_matmul_methods[0].lhs.pack = kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme;

    // RHS
    indirect_matmul_methods[0].rhs.get_n_step = kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[0].rhs.get_rhs_offset = kai_get_rhs_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[0].rhs.get_bias_offset = kai_get_bias_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[0].rhs.get_rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[0].rhs.get_rhs_packed_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[0].rhs.pack = kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;

    // IMATMUL
    const kai_imatmul_clamp_f16_f16p_f16p_ukernel& ukernel_f16_sme2 =
        get_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
    indirect_matmul_methods[0].imatmul.get_m_step = ukernel_f16_sme2.get_m_step;
    indirect_matmul_methods[0].imatmul.get_n_step = ukernel_f16_sme2.get_n_step;
    indirect_matmul_methods[0].imatmul.get_lhs_packed_offset = ukernel_f16_sme2.get_lhs_packed_offset;
    indirect_matmul_methods[0].imatmul.get_rhs_packed_offset = ukernel_f16_sme2.get_rhs_packed_offset;
    indirect_matmul_methods[0].imatmul.get_dst_offset = ukernel_f16_sme2.get_dst_offset;
    indirect_matmul_methods[0].imatmul.get_dst_size = ukernel_f16_sme2.get_dst_size;
    indirect_matmul_methods[0].imatmul.imatmul = ukernel_f16_sme2.run_imatmul;

    // F32 IMATMUL SME2 ///////////////////////////////////////////////////////
    indirect_matmul_methods[1].name = "imatmul_f32_f32p_f32p_2vlx2vl_sme2_mopa";
    indirect_matmul_methods[1].is_supported = cpu_has_sme2;
    indirect_matmul_methods[1].pack_shape.m = 2 * get_sme_vector_length<int32_t>();
    indirect_matmul_methods[1].pack_shape.n = 2 * get_sme_vector_length<int32_t>();
    indirect_matmul_methods[1].pack_shape.k = sizeof(int32_t);
    indirect_matmul_methods[1].format.lhs = DataFormat(DataType::FP32);
    indirect_matmul_methods[1].format.rhs = DataFormat(DataType::FP32);
    indirect_matmul_methods[1].format.bias = DataFormat(DataType::FP32);
    indirect_matmul_methods[1].format.out = DataFormat(DataType::FP32);

    // LHS
    indirect_matmul_methods[1].lhs.get_m_step = kai_get_m_step_lhs_imatmul_pack_x32p2vlx1_x32p_sme;
    indirect_matmul_methods[1].lhs.get_lhs_packed_offset =
        kai_get_lhs_packed_offset_lhs_imatmul_pack_x32p2vlx1_x32p_sme;
    indirect_matmul_methods[1].lhs.get_lhs_packed_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme;
    indirect_matmul_methods[1].lhs.pack = kai_run_lhs_imatmul_pack_x32p2vlx1_x32p_sme;

    // RHS
    indirect_matmul_methods[1].rhs.get_n_step = kai_get_n_step_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[1].rhs.get_rhs_offset = kai_get_rhs_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[1].rhs.get_bias_offset = kai_get_bias_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[1].rhs.get_rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[1].rhs.get_rhs_packed_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[1].rhs.pack = kai_run_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;

    // IMATMUL
    const kai_imatmul_clamp_f32_f32p_f32p_ukernel& ukernel_f32_sme2 =
        get_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa();
    indirect_matmul_methods[1].imatmul.get_m_step = ukernel_f32_sme2.get_m_step;
    indirect_matmul_methods[1].imatmul.get_n_step = ukernel_f32_sme2.get_n_step;
    indirect_matmul_methods[1].imatmul.get_lhs_packed_offset = ukernel_f32_sme2.get_lhs_packed_offset;
    indirect_matmul_methods[1].imatmul.get_rhs_packed_offset = ukernel_f32_sme2.get_rhs_packed_offset;
    indirect_matmul_methods[1].imatmul.get_dst_offset = ukernel_f32_sme2.get_dst_offset;
    indirect_matmul_methods[1].imatmul.get_dst_size = ukernel_f32_sme2.get_dst_size;
    indirect_matmul_methods[1].imatmul.imatmul = ukernel_f32_sme2.run_imatmul;

    // F16 IMATMUL SME ////////////////////////////////////////////////////////
    indirect_matmul_methods[2].name = "imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa";
    indirect_matmul_methods[2].is_supported = cpu_has_sme;
    indirect_matmul_methods[2].pack_shape.m = 2 * get_sme_vector_length<int32_t>();
    indirect_matmul_methods[2].pack_shape.n = 2 * get_sme_vector_length<int32_t>();
    indirect_matmul_methods[2].pack_shape.k = sizeof(int32_t);
    indirect_matmul_methods[2].format.lhs = DataFormat(DataType::FP16);
    indirect_matmul_methods[2].format.rhs = DataFormat(DataType::FP16);
    indirect_matmul_methods[2].format.bias = DataFormat(DataType::FP16);
    indirect_matmul_methods[2].format.out = DataFormat(DataType::FP16);

    // LHS
    indirect_matmul_methods[2].lhs.get_m_step = kai_get_m_step_lhs_imatmul_pack_x16p2vlx2_x16p_sme;
    indirect_matmul_methods[2].lhs.get_lhs_packed_offset =
        kai_get_lhs_packed_offset_lhs_imatmul_pack_x16p2vlx2_x16p_sme;
    indirect_matmul_methods[2].lhs.get_lhs_packed_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme;
    indirect_matmul_methods[2].lhs.pack = kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme;

    // RHS
    indirect_matmul_methods[2].rhs.get_n_step = kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[2].rhs.get_rhs_offset = kai_get_rhs_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[2].rhs.get_bias_offset = kai_get_bias_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[2].rhs.get_rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[2].rhs.get_rhs_packed_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;
    indirect_matmul_methods[2].rhs.pack = kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme;

    // IMATMUL
    const kai_imatmul_clamp_f16_f16p_f16p_ukernel& ukernel_f16_sme =
        get_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa();
    indirect_matmul_methods[2].imatmul.get_m_step = ukernel_f16_sme.get_m_step;
    indirect_matmul_methods[2].imatmul.get_n_step = ukernel_f16_sme.get_n_step;
    indirect_matmul_methods[2].imatmul.get_lhs_packed_offset = ukernel_f16_sme.get_lhs_packed_offset;
    indirect_matmul_methods[2].imatmul.get_rhs_packed_offset = ukernel_f16_sme.get_rhs_packed_offset;
    indirect_matmul_methods[2].imatmul.get_dst_offset = ukernel_f16_sme.get_dst_offset;
    indirect_matmul_methods[2].imatmul.get_dst_size = ukernel_f16_sme.get_dst_size;
    indirect_matmul_methods[2].imatmul.imatmul = ukernel_f16_sme.run_imatmul;

    // F32 IMATMUL SME ////////////////////////////////////////////////////////
    indirect_matmul_methods[3].name = "imatmul_f32_f32p_f32p_2vlx2vl_sme_mopa";
    indirect_matmul_methods[3].is_supported = cpu_has_sme;
    indirect_matmul_methods[3].pack_shape.m = 2 * get_sme_vector_length<int32_t>();
    indirect_matmul_methods[3].pack_shape.n = 2 * get_sme_vector_length<int32_t>();
    indirect_matmul_methods[3].pack_shape.k = sizeof(int32_t);
    indirect_matmul_methods[3].format.lhs = DataFormat(DataType::FP32);
    indirect_matmul_methods[3].format.rhs = DataFormat(DataType::FP32);
    indirect_matmul_methods[3].format.bias = DataFormat(DataType::FP32);
    indirect_matmul_methods[3].format.out = DataFormat(DataType::FP32);

    // LHS
    indirect_matmul_methods[3].lhs.get_m_step = kai_get_m_step_lhs_imatmul_pack_x32p2vlx1_x32p_sme;
    indirect_matmul_methods[3].lhs.get_lhs_packed_offset =
        kai_get_lhs_packed_offset_lhs_imatmul_pack_x32p2vlx1_x32p_sme;
    indirect_matmul_methods[3].lhs.get_lhs_packed_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme;
    indirect_matmul_methods[3].lhs.pack = kai_run_lhs_imatmul_pack_x32p2vlx1_x32p_sme;

    // RHS
    indirect_matmul_methods[3].rhs.get_n_step = kai_get_n_step_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[3].rhs.get_rhs_offset = kai_get_rhs_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[3].rhs.get_bias_offset = kai_get_bias_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[3].rhs.get_rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[3].rhs.get_rhs_packed_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;
    indirect_matmul_methods[3].rhs.pack = kai_run_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme;

    // IMATMUL
    const kai_imatmul_clamp_f32_f32p_f32p_ukernel& ukernel_f32_sme =
        get_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    indirect_matmul_methods[3].imatmul.get_m_step = ukernel_f32_sme.get_m_step;
    indirect_matmul_methods[3].imatmul.get_n_step = ukernel_f32_sme.get_n_step;
    indirect_matmul_methods[3].imatmul.get_lhs_packed_offset = ukernel_f32_sme.get_lhs_packed_offset;
    indirect_matmul_methods[3].imatmul.get_rhs_packed_offset = ukernel_f32_sme.get_rhs_packed_offset;
    indirect_matmul_methods[3].imatmul.get_dst_offset = ukernel_f32_sme.get_dst_offset;
    indirect_matmul_methods[3].imatmul.get_dst_size = ukernel_f32_sme.get_dst_size;
    indirect_matmul_methods[3].imatmul.imatmul = ukernel_f32_sme.run_imatmul;

    return indirect_matmul_methods;
}

/// Test reference identification
struct TestDataId {
    MatMulShape shape;
    MatMulShape pack_shape;
    IndirectMatMul::Format format;
    size_t k_chunk_length;
    float clamp_rate;

    struct Hash {
        size_t operator()(const TestDataId& test_id) const {
            return                                                       //
                (MatMulShape::Hash{}(test_id.shape) << 0) ^              //
                (MatMulShape::Hash{}(test_id.pack_shape) << 1) ^         //
                (IndirectMatMul::Format::Hash{}(test_id.format) << 2) ^  //
                (std::hash<size_t>{}(test_id.k_chunk_length) << 3) ^     //
                (std::hash<float>{}(test_id.clamp_rate) << 4);           //
        }
    };

private:
    friend bool operator==(const TestDataId& lhs, const TestDataId& rhs) {
        return                                           //
            lhs.shape == rhs.shape &&                    //
            lhs.pack_shape == rhs.pack_shape &&          //
            lhs.format == rhs.format &&                  //
            lhs.k_chunk_length == rhs.k_chunk_length &&  //
            lhs.clamp_rate == rhs.clamp_rate;
    }
};

/// Test reference data
struct TestData {
    Buffer lhs;                    ///< LHS input matrix
    Buffer rhs;                    ///< RHS input matrix
    Buffer bias;                   ///< Bias vector
    Buffer out;                    ///< Reference imatmul result
    Buffer indirection;            ///< LHS indirection buffer
    uintptr_t indirection_offset;  ///< LHS indirection buffer offset
    Buffer padding;                ///< Padding buffer
    Range<float> clamp_range;      ///< Clamp range
};

/// Reference data generator
///
/// Uses test id to generate reference data, and caches it.
struct ReferenceGenerator {
    /// Retrieve reference data for the provided test identification
    static const TestData& get_test_reference(const TestDataId& test_id) {
        static std::unordered_map<TestDataId, TestData, TestDataId::Hash> m_data;
        if (const auto itr = m_data.find(test_id); itr != end(m_data)) {
            return itr->second;
        }

        return m_data[test_id] = generate_reference(test_id);
    }

private:
    /// Return incremented seed value
    static size_t get_seed() {
        static size_t seed = 0;
        return seed++;
    }

    /// Generate reference data. Not intended to be called
    /// directly, as this would bypass caching mechanism.
    static TestData generate_reference(const TestDataId& test_id) {
        const auto& [chunked_shape, pack_shape, format, k_chunk_length, clamp_rate] = test_id;

        // The LHS matrix will be split into several chunks in the K dimension
        const size_t k_chunk_count = chunked_shape.k;
        MatMulShape shape = {chunked_shape.m, chunked_shape.n, k_chunk_count * k_chunk_length};

        // Generate random input data
        Buffer lhs = fill_matrix_random(shape.m, shape.k, format.lhs, get_seed());
        Buffer rhs = fill_matrix_random(shape.k, shape.n, format.rhs, get_seed());
        Buffer bias = fill_matrix_random(1, shape.n, format.bias, get_seed());

        // Data types used
        const DataType lhs_dt = format.lhs.data_type();
        const DataType rhs_dt = format.rhs.data_type();
        const DataType out_dt = format.out.data_type();
        const DataType bias_dt = format.bias.data_type();

        // Create a padding chunk
        const size_t k_chunk_size = round_up_division(k_chunk_length * data_type_size_in_bits(lhs_dt), 8);
        const size_t row_size = k_chunk_count * k_chunk_size;
        Buffer lhs_padding(k_chunk_size);
        for (size_t i = 0; i < k_chunk_length; i += 1) {
            static constexpr double padding_value = 0;
            write_array(lhs_dt, lhs_padding.data(), i, padding_value);
        }

        // Set up indirection buffer
        const uintptr_t indirection_offset = reinterpret_cast<uintptr_t>(lhs.data());
        std::vector<const void*> indirection(chunked_shape.m * chunked_shape.k);
        for (size_t i_m = 0; i_m < chunked_shape.m; i_m += 1) {
            for (size_t i_k = 0; i_k < chunked_shape.k; i_k += 1) {
                const size_t idx = i_m * chunked_shape.k + i_k;
                // Test padding pointers using first LHS row for shapes where M > 1
                if (chunked_shape.m > 1 && i_m == 0) {
                    indirection.at(idx) = lhs_padding.data();
                } else {
                    uintptr_t offset = i_m * row_size + i_k * k_chunk_size;
                    indirection.at(idx) = reinterpret_cast<const void*>(offset);
                }
            }
        }

        // Pack indirection buffer
        Buffer indirection_packed = reorder_block<const void*>(
            reinterpret_cast<const void* const*>(indirection.data()), chunked_shape.m, chunked_shape.k, pack_shape.m,
            1);

        Buffer out = indirect_matmul(                                                              //
            indirection.data(), indirection_offset, lhs_padding.data(), nullptr, nullptr, lhs_dt,  // LHS
            rhs.data(), nullptr, nullptr, rhs_dt,                                                  // RHS
            bias.data(), nullptr, nullptr, bias_dt,                                                // Bias
            out_dt,                                                                                // Out
            chunked_shape.m, chunked_shape.n, chunked_shape.k, k_chunk_length);

        // Calculate clamping range based on full range of values, and then clamp values
        const auto [min, max] = find_clamp_range(out_dt, out.data(), shape.m * shape.n, 1.0F - clamp_rate);
        Buffer out_clamped = clamp(out_dt, out.data(), shape.m * shape.n, min, max);

        // Populate reference data
        TestData test_reference;
        test_reference.lhs = std::move(lhs);
        test_reference.rhs = std::move(rhs);
        test_reference.bias = std::move(bias);
        test_reference.padding = std::move(lhs_padding);
        test_reference.out = std::move(out_clamped);
        test_reference.indirection_offset = indirection_offset;
        test_reference.indirection = std::move(indirection_packed);
        test_reference.clamp_range = {min, max};

        return test_reference;
    };
};

/// Perform LHS packing for indirect matmul
Buffer pack_lhs(
    const LhsPackIndirectKernel& kernel, const Rect& portion, const TestData& reference, size_t m,
    const KChunk& k_chunk) {
    const void* const* indirection_pointer = reinterpret_cast<const void* const*>(reference.indirection.data());

    // Calculate size, and allocate buffer
    const size_t dst_size = kernel.get_lhs_packed_size(m, k_chunk.count, k_chunk.length);
    Buffer dst(dst_size);

    // Calculate portion offsets
    const size_t input_offset = portion.start_row() * k_chunk.count;
    const size_t dst_offset = kernel.get_lhs_packed_offset(portion.start_row(), k_chunk.count, k_chunk.length);

    // Perform packing
    abi_check(
        kernel.pack,                                      // Kernel
        portion.height(), k_chunk.count, k_chunk.length,  // Dimensions
        indirection_pointer + input_offset,               // Indirection input
        reference.indirection_offset,                     // Chunk offset
        reference.padding.data(),                         // Padding pointer
        dst.data() + dst_offset);
    return dst;
}

/// Perform RHS packign for indirect matmul
Buffer pack_rhs(
    const RhsPackIndirectKernel& kernel, const Rect& portion, const TestData& reference, size_t n,
    const KChunk& k_chunk, DataType type) {
    // Calculate size, and allocate buffer
    const size_t row_stride = round_up_division(n * data_type_size_in_bits(type), 8);
    const size_t dst_size = kernel.get_rhs_packed_size(n, k_chunk.count, k_chunk.length);
    Buffer dst(dst_size);

    // Calculate offsets
    const size_t rhs_offset = kernel.get_rhs_offset(portion.start_col());
    const size_t bias_offset = kernel.get_bias_offset(portion.start_col());
    const size_t dst_offset = kernel.get_rhs_packed_offset(portion.start_col(), k_chunk.count, k_chunk.length);

    // Perform actual packing
    abi_check(
        kernel.pack,                                                 // Kernel
        portion.width(), k_chunk.count, k_chunk.length, row_stride,  // Dimensions
        reference.rhs.data() + rhs_offset,                           // RHS input
        reference.bias.data() + bias_offset,                         // Bias
        dst.data() + dst_offset);                                    // Output
    return dst;
}

/// Perform imatmul
///
/// Note, this should not be aware of reference result, as to make it clear that
/// any produced result is strictly from the code under test
Buffer imatmul(
    const MatMulIndirectKernel& kernel, const Rect& portion, const MatMulShape& shape, const KChunk& k_chunk,
    const Buffer& lhs_packed, const Buffer& rhs_packed, Range<float> clamp_range, DataType type) {
    // Calculate size, and allocate buffer
    const size_t dst_size = kernel.get_dst_size(shape.m, shape.n);
    const size_t row_stride = round_up_division(shape.n * data_type_size_in_bits(type), 8);
    Buffer dst(dst_size);

    // Calculate portion offsets
    const size_t lhs_offset = kernel.get_lhs_packed_offset(portion.start_row(), k_chunk.count, k_chunk.length);
    const size_t rhs_offset = kernel.get_rhs_packed_offset(portion.start_col(), k_chunk.count, k_chunk.length);
    const size_t dst_offset = kernel.get_dst_offset(portion.start_row(), portion.start_col(), row_stride);

    // Call matmul kernel
    abi_check(
        kernel.imatmul,                                                    // Kernel
        portion.height(), portion.width(), k_chunk.count, k_chunk.length,  // Dimensions
        lhs_packed.data() + lhs_offset,                                    // LHS
        rhs_packed.data() + rhs_offset,                                    // RHS
        dst.data() + dst_offset,                                           // DST
        row_stride, clamp_range.min, clamp_range.max);

    return dst;
}

}  // namespace

/// End-to-end test for indirection matmul kernels
TEST_P(IndirectMatMulTest, Output) {
    const auto& [method, shape, k_chunk_length, output_portion, clamp_rate] = GetParam();
    if (not method.is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const KChunk k_chunk{shape.k, k_chunk_length};

    // Retrieve reference data
    const TestDataId test_id{shape, method.pack_shape, method.format, k_chunk_length, clamp_rate};
    const TestData& test_data = ReferenceGenerator::get_test_reference(test_id);
    const Rect portion = output_portion.compute_portion(shape.m, shape.n, method.pack_shape.m, method.pack_shape.n);

    if (portion.height() == 0 || portion.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << portion.width() << "," << portion.height() << ")";
    }

    // Call packing micro-kernels, and then imatmul kernel
    Buffer lhs_packed = pack_lhs(method.lhs, portion, test_data, shape.m, k_chunk);
    Buffer rhs_packed = pack_rhs(method.rhs, portion, test_data, shape.n, k_chunk, method.format.rhs.data_type());
    Buffer out = imatmul(
        method.imatmul, portion, shape, k_chunk, lhs_packed, rhs_packed, test_data.clamp_range,
        method.format.out.data_type());

    // Compare the actual result with the reference result
    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    const auto success =
        compare(out.data(), test_data.out.data(), method.format.out.data_type(), shape.m, shape.n, portion, handler);
    ASSERT_TRUE(success);
}

/// Name generator for test case
[[maybe_unused]] static void PrintTo(const IndirectMatMulTestParams& param, std::ostream* os) {
    const auto& [method, shape, k_chunk_length, portion, clamp_rate] = param;
    *os << method.name << "__";
    PrintTo(shape, os);
    *os << "__K_chunk_length_" << k_chunk_length;
    *os << "__clamp_rate_" << static_cast<int>(clamp_rate * 100) << "__";
    PrintTo(portion, os);
}

static auto get_indirect_matmul_shapes() {
    static const std::array indirect_matmul_shapes{
        // clang-format off
        MatMulShape{  1,   1,   1},
        MatMulShape{  1,  17,   4},
        MatMulShape{  1,  19,  24},
        MatMulShape{  1,  32,   4},
        MatMulShape{  1,  32,  32},
        MatMulShape{  1,  33,   7},
        MatMulShape{  1,  49,  21},
        MatMulShape{  1,  64,   4},
        MatMulShape{  1,  65,   4},
        MatMulShape{  3,   6,   6},
        MatMulShape{  3,  28,  25},
        MatMulShape{  4,  16,   4},
        MatMulShape{  4,  16,  27},
        MatMulShape{  6,  18,  31},
        MatMulShape{  6,  28,   1},
        MatMulShape{  6,  29,  24},
        MatMulShape{  8,  16,  16},
        MatMulShape{ 16,  16,   4},
        MatMulShape{ 16,  16,  16},
        MatMulShape{ 20,  30,  40},
        MatMulShape{ 23,   1,  43},
        MatMulShape{ 32,  14,   1},
        MatMulShape{ 32,  16,  27},
        MatMulShape{ 32,  32,   3},
        MatMulShape{ 32,  32,   4},
        MatMulShape{ 33,  29,  24},
        MatMulShape{ 64,  64,   3},
        MatMulShape{ 64,  64,   4},
        MatMulShape{ 96,  96,   3},
        MatMulShape{ 96,  97,   3},
        MatMulShape{ 97,  96,   3},
        MatMulShape{123,  85,  45},
        MatMulShape{128, 128,   3},
        MatMulShape{130, 130,   6},
        // clang-format on
    };

    return indirect_matmul_shapes;
}

static auto get_indirect_matmul_portions() {
    static const std::array<MatrixPortion, 6> indirect_matmul_portions{
        //       (Start row , start col , height , width)
        MatrixPortion(0, 0, 1, 1),          // Full matrix.
        MatrixPortion(0, 0, 1, 0.5),        // Left half
        MatrixPortion(0, 0, 0.5, 1),        // Upper half
        MatrixPortion(0, 0.5, 1, 0.5),      // Right half
        MatrixPortion(0.5, 0, 0.5, 1),      // Bottom half
        MatrixPortion(0.4, 0.4, 0.3, 0.3),  // Center ninth
    };

    return indirect_matmul_portions;
}

// Test suite focused on small K chunk
INSTANTIATE_TEST_SUITE_P(
    ShapesSmallKC, IndirectMatMulTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_matmul_methods()),                         //
        testing::ValuesIn(get_indirect_matmul_shapes()),                          //
        testing::ValuesIn(std::initializer_list<size_t>{1, 2, 3, 4, 8, 11, 16}),  //
        testing::ValuesIn(get_indirect_matmul_portions()),                        //
        testing::Values(0.5F)),                                                   //
    testing::PrintToStringParamName());

// Test suite focused on K chunk 31
INSTANTIATE_TEST_SUITE_P(
    ShapesKC31, IndirectMatMulTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_matmul_methods()),   //
        testing::ValuesIn(get_indirect_matmul_shapes()),    //
        testing::Values(static_cast<size_t>(31)),           //
        testing::ValuesIn(get_indirect_matmul_portions()),  //
        testing::Values(0.5F)),                             //
    testing::PrintToStringParamName());

// Test suite focused on K chunk 32
INSTANTIATE_TEST_SUITE_P(
    ShapesKC32, IndirectMatMulTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_matmul_methods()),   //
        testing::ValuesIn(get_indirect_matmul_shapes()),    //
        testing::Values(static_cast<size_t>(32)),           //
        testing::ValuesIn(get_indirect_matmul_portions()),  //
        testing::Values(0.5F)),                             //
    testing::PrintToStringParamName());

// Test suite focused on K chunk 64
INSTANTIATE_TEST_SUITE_P(
    ShapesKC64, IndirectMatMulTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_matmul_methods()),   //
        testing::ValuesIn(get_indirect_matmul_shapes()),    //
        testing::Values(static_cast<size_t>(64)),           //
        testing::ValuesIn(get_indirect_matmul_portions()),  //
        testing::Values(0.5F)),                             //
    testing::PrintToStringParamName());

// Test suite focused on K chunk 65, other parametes are limited
INSTANTIATE_TEST_SUITE_P(
    ShapesKC65, IndirectMatMulTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_matmul_methods()),   //
        testing::ValuesIn(get_indirect_matmul_shapes()),    //
        testing::Values(static_cast<size_t>(65)),           //
        testing::ValuesIn(get_indirect_matmul_portions()),  //
        testing::Values(0.5F)),                             //
    testing::PrintToStringParamName());

// Test suite focused on clamping values, other parametes are limited
INSTANTIATE_TEST_SUITE_P(
    Clamp, IndirectMatMulTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_matmul_methods()),  //
        testing::ValuesIn(get_indirect_matmul_shapes()),   //
        testing::Values(static_cast<size_t>(3)),           //
        testing::Values(MatrixPortion(0, 0, 1, 1)),        //
        testing::Values(0.0F, 0.1F, 0.5F)),                //
    testing::PrintToStringParamName());

}  // namespace kai::test
