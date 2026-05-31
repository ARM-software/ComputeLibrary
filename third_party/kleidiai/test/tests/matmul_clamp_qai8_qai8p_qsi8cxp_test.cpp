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
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/matmul_test_common.hpp"
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

// Ensure static linkage for all functionality local to this test file
namespace {

struct KChunk {
    size_t count;
    size_t length;
};

struct LhsPackKernel {
    std::function<size_t(size_t mr)> get_m_step;
    std::function<size_t(size_t m_idx, size_t lhs_stride)> get_lhs_offset;
    std::function<size_t(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr)> get_packed_lhs_offset;
    std::function<size_t(size_t m, size_t k, size_t mr, size_t kr, size_t sr)> get_packed_lhs_size;
    std::function<void(
        size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
        void* lhs_packed)>
        pack;
};

struct LhsPackIndirectKernel {
    std::function<size_t()> get_m_step;
    std::function<size_t(size_t m_idx, size_t k_chunk_count, size_t k_chunk_length)> get_packed_lhs_offset;
    std::function<size_t(size_t m, size_t k_chunk_count, size_t k_chunk_length)> get_packed_lhs_size;
    std::function<void(
        size_t m, size_t k_chunk_count, size_t k_chunk_length, const void* const* lhs_ptrs, size_t lhs_ptr_offset,
        const void* zero, void* packed_lhs)>
        pack;
};

struct RhsPackKernel {
    std::function<size_t()> get_n_step;
    std::function<size_t(size_t n_idx)> get_rhs_offset;
    std::function<size_t(size_t n_idx)> get_bias_offset;
    std::function<size_t(size_t n_idx)> get_scale_offset;
    std::function<size_t(size_t n_idx, size_t k)> get_packed_rhs_offset;
    std::function<size_t(size_t n, size_t k)> get_packed_rhs_size;
    std::function<void(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
        const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
        const struct kai_rhs_pack_qsi8cx_params* params)>
        pack;
};

struct RhsPackIndirectKernel {
    std::function<size_t()> get_n_step;
    std::function<size_t(size_t n_idx)> get_rhs_offset;
    std::function<size_t(size_t n_idx)> get_bias_offset;
    std::function<size_t(size_t n_idx)> get_scale_offset;
    std::function<size_t(size_t n_idx, size_t k_chunk_count, size_t k_chunk_length)> get_packed_rhs_offset;
    std::function<size_t(size_t n, size_t k_chunk_count, size_t k_chunk_length)> get_packed_rhs_size;
    std::function<void(
        size_t n, size_t k_chunk_count, size_t k_chunk_length, size_t rhs_stride, const void* rhs, const void* bias,
        const void* scale, void* rhs_packed, const kai_rhs_pack_qsi8cx_params* params)>
        pack;
};

struct MatMulKernel {
    std::function<size_t(void)> get_m_step;
    std::function<size_t(void)> get_n_step;
    std::function<size_t(void)> get_mr;
    std::function<size_t(void)> get_nr;
    std::function<size_t(void)> get_kr;
    std::function<size_t(void)> get_sr;
    std::function<size_t(size_t m_idx, size_t k)> get_packed_lhs_offset;
    std::function<size_t(size_t n_idx, size_t k)> get_packed_rhs_offset;
    std::function<size_t(size_t m_idx, size_t n_idx, size_t dst_stride)> get_dst_offset;
    std::function<size_t(size_t m, size_t n)> get_dst_size;
    std::function<void(
        size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
        size_t dst_stride_col, const kai_matmul_requantize32_params* params)>
        matmul;
};

struct MatMulIndirectKernel {
    std::function<size_t(void)> get_m_step;
    std::function<size_t(void)> get_n_step;
    std::function<size_t(size_t m_idx, size_t k_chunk_count, size_t k_chunk_length)> get_lhs_packed_offset;
    std::function<size_t(size_t n_idx, size_t k_chunk_count, size_t k_chunk_length)> get_rhs_packed_offset;
    std::function<size_t(size_t m_idx, size_t n_idx, size_t dst_stride_row)> get_dst_offset;
    std::function<size_t(size_t m, size_t n)> get_dst_size;
    std::function<void(
        size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_lenght, const void* lhs_packed, const void* rhs_packed,
        void* dst, size_t dst_stride_row, const kai_matmul_requantize32_params* params)>
        imatmul;
};

/// Make sure that interface matches for qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa
const kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_ukernel&
get_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_interface() {
    static kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_ukernel ukernel;

    ukernel.get_m_step = kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_n_step = kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_mr = kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_nr = kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_kr = kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_sr = kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_lhs_packed_offset =
        kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_rhs_packed_offset =
        kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_dst_offset = kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_dst_size = kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.run_matmul = kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;

    return ukernel;
}

/// Make sure that interface matches for qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme_mopa
const kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_ukernel&
get_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_interface() {
    static kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_ukernel ukernel;

    ukernel.get_m_step = kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_n_step = kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_mr = kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_nr = kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_kr = kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_sr = kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_lhs_packed_offset =
        kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_rhs_packed_offset =
        kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_dst_offset = kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_dst_size = kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.run_matmul = kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;

    return ukernel;
}

/// Make sure that interface matches for qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot
const kai_matmul_clamp_qai8_qai8p_qsi8cxp_ukernel&
get_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot_interface() {
    static kai_matmul_clamp_qai8_qai8p_qsi8cxp_ukernel ukernel;

    ukernel.get_m_step = kai_get_m_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.get_n_step = kai_get_n_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.get_nr = kai_get_nr_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.get_kr = kai_get_kr_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.get_sr = kai_get_sr_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.get_lhs_offset = kai_get_lhs_offset_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.get_dst_offset = kai_get_dst_offset_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.get_dst_size = kai_get_dst_size_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;
    ukernel.run_matmul = kai_run_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot;

    return ukernel;
};

/// Make sure that interface matches qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa
const kai_imatmul_clamp_qai8_qai8p_qsi8cxp_ukernel&
get_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_interface() {
    static kai_imatmul_clamp_qai8_qai8p_qsi8cxp_ukernel ukernel;

    ukernel.get_m_step = kai_get_m_step_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_n_step = kai_get_n_step_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_lhs_packed_offset =
        kai_get_lhs_packed_offset_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_rhs_packed_offset =
        kai_get_rhs_packed_offset_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_dst_offset = kai_get_dst_offset_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.get_dst_size = kai_get_dst_size_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
    ukernel.run_imatmul = kai_run_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;

    return ukernel;
};

/// Make sure that interface matches qai8_qai8p2vlx4_qsi8cxps2vlx4b_2vlx2vl_sme_mopa
const kai_imatmul_clamp_qai8_qai8p_qsi8cxp_ukernel&
get_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_interface() {
    static kai_imatmul_clamp_qai8_qai8p_qsi8cxp_ukernel ukernel;

    ukernel.get_m_step = kai_get_m_step_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_n_step = kai_get_n_step_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_lhs_packed_offset =
        kai_get_lhs_packed_offset_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_rhs_packed_offset =
        kai_get_rhs_packed_offset_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_dst_offset = kai_get_dst_offset_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.get_dst_size = kai_get_dst_size_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;
    ukernel.run_imatmul = kai_run_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa;

    return ukernel;
};

const RhsPackKernel& get_rhs_pack() {
    static RhsPackKernel ukernel;

    ukernel.get_n_step = kai_get_n_step_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    ukernel.get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    ukernel.get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    ukernel.get_scale_offset = kai_get_scale_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    ukernel.get_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    ukernel.get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    ukernel.pack = kai_run_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;

    return ukernel;
}

const LhsPackKernel& get_lhs_pack() {
    static LhsPackKernel ukernel;

    ukernel.get_m_step = kai_get_m_step_lhs_pack_x8p2vlx4_x8_sme;
    ukernel.get_lhs_offset = kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme;
    ukernel.get_packed_lhs_offset = kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme;
    ukernel.get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme;
    ukernel.pack = kai_run_lhs_pack_x8p2vlx4_x8_sme;

    return ukernel;
}

struct MatMulVariant {
    std::string_view name;  ///< Test identification
    MatMulShape acc_pack;   ///< Accumulator shape for packing (mr/nr/kr)
    MatMulShape acc_step;   ///< Accumulator shape for matmul (stepping)

    std::function<bool(void)> is_supported;  ///< HW support check

    std::optional<LhsPackKernel> lhs_pack;  ///< LHS packing micro-kernel interface
    RhsPackKernel rhs_pack;                 ///< RHS packing micro-kernel interface
    MatMulKernel matmul;                    ///< Matmul kernel interface
};

struct IndirectMatMulVariant {
    std::string_view name;  ///< Test identification
    MatMulShape acc_pack;   ///< Accumulator shape for packing (mr/nr/kr)
    MatMulShape acc_step;   ///< Accumulator shape for matmul (stepping)

    std::function<bool(void)> is_supported;  ///< HW support check

    LhsPackIndirectKernel lhs_pack;  ///< LHS packing micro-kernel interface
    RhsPackIndirectKernel rhs_pack;  ///< RHS packing micro-kernel interface
    MatMulIndirectKernel matmul;     ///< Matmul kernel interface
};

const auto& get_gemm_variants() {
    static std::array<MatMulVariant, 2> variants;
    static const kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_ukernel& ukernel_sme2 =
        get_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_interface();
    static const kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_ukernel& ukernel_sme =
        get_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_interface();

    variants[0].name = "matmul_qai8_qai8p_qsi8cxp_sme";
    variants[0].acc_pack.m = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_pack.n = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_pack.k = sizeof(int32_t) / sizeof(int8_t);
    variants[0].acc_step.m = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_step.n = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_step.k = sizeof(int32_t) / sizeof(int8_t);
    variants[0].is_supported = cpu_has_sme;
    variants[0].lhs_pack = get_lhs_pack();
    variants[0].rhs_pack = get_rhs_pack();
    variants[0].matmul.get_m_step = ukernel_sme.get_m_step;
    variants[0].matmul.get_n_step = ukernel_sme.get_n_step;
    variants[0].matmul.get_mr = ukernel_sme.get_mr;
    variants[0].matmul.get_nr = ukernel_sme.get_nr;
    variants[0].matmul.get_kr = ukernel_sme.get_kr;
    variants[0].matmul.get_sr = ukernel_sme.get_sr;
    variants[0].matmul.get_packed_lhs_offset = ukernel_sme.get_lhs_packed_offset;
    variants[0].matmul.get_packed_rhs_offset = ukernel_sme.get_rhs_packed_offset;
    variants[0].matmul.get_dst_offset = ukernel_sme.get_dst_offset;
    variants[0].matmul.get_dst_size = ukernel_sme.get_dst_size;
    variants[0].matmul.matmul = ukernel_sme.run_matmul;

    variants[1].name = "matmul_qai8_qai8p_qsi8cxp_sme2";
    variants[1].acc_pack.m = 2 * get_sme_vector_length<int32_t>();
    variants[1].acc_pack.n = 2 * get_sme_vector_length<int32_t>();
    variants[1].acc_pack.k = sizeof(int32_t) / sizeof(int8_t);
    variants[1].acc_step.m = 2 * get_sme_vector_length<int32_t>();
    variants[1].acc_step.n = 2 * get_sme_vector_length<int32_t>();
    variants[1].acc_step.k = sizeof(int32_t) / sizeof(int8_t);
    variants[1].is_supported = cpu_has_sme2;
    variants[1].lhs_pack = get_lhs_pack();
    variants[1].rhs_pack = get_rhs_pack();
    variants[1].matmul.get_m_step = ukernel_sme2.get_m_step;
    variants[1].matmul.get_n_step = ukernel_sme2.get_n_step;
    variants[1].matmul.get_mr = ukernel_sme2.get_mr;
    variants[1].matmul.get_nr = ukernel_sme2.get_nr;
    variants[1].matmul.get_kr = ukernel_sme2.get_kr;
    variants[1].matmul.get_sr = ukernel_sme2.get_sr;
    variants[1].matmul.get_packed_lhs_offset = ukernel_sme2.get_lhs_packed_offset;
    variants[1].matmul.get_packed_rhs_offset = ukernel_sme2.get_rhs_packed_offset;
    variants[1].matmul.get_dst_offset = ukernel_sme2.get_dst_offset;
    variants[1].matmul.get_dst_size = ukernel_sme2.get_dst_size;
    variants[1].matmul.matmul = ukernel_sme2.run_matmul;

    return variants;
}

const auto& get_indirect_gemm_variants() {
    static std::array<IndirectMatMulVariant, 2> variants;
    static const kai_imatmul_clamp_qai8_qai8p_qsi8cxp_ukernel& ukernel_sme =
        get_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_interface();
    static const kai_imatmul_clamp_qai8_qai8p_qsi8cxp_ukernel& ukernel_sme2 =
        get_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_interface();

    variants[0].name = "imatmul_qai8_qai8p_qsi8cxp_sme";
    variants[0].acc_pack.m = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_pack.n = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_pack.k = sizeof(int32_t) / sizeof(int8_t);
    variants[0].acc_step.m = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_step.n = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_step.k = sizeof(int32_t) / sizeof(int8_t);
    variants[0].is_supported = cpu_has_sme;
    variants[0].lhs_pack.get_m_step = kai_get_m_step_lhs_imatmul_pack_x8p2vlx4_x8p_sme;
    variants[0].lhs_pack.get_packed_lhs_offset = kai_get_lhs_packed_offset_lhs_imatmul_pack_x8p2vlx4_x8p_sme;
    variants[0].lhs_pack.get_packed_lhs_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme;
    variants[0].lhs_pack.pack = kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme;
    variants[0].rhs_pack.get_n_step = kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[0].rhs_pack.get_rhs_offset = kai_get_rhs_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[0].rhs_pack.get_bias_offset = kai_get_bias_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[0].rhs_pack.get_scale_offset = kai_get_scale_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[0].rhs_pack.get_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[0].rhs_pack.get_packed_rhs_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[0].rhs_pack.pack = kai_run_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[0].matmul.get_m_step = ukernel_sme.get_m_step;
    variants[0].matmul.get_n_step = ukernel_sme.get_n_step;
    variants[0].matmul.get_lhs_packed_offset = ukernel_sme.get_lhs_packed_offset;
    variants[0].matmul.get_rhs_packed_offset = ukernel_sme.get_rhs_packed_offset;
    variants[0].matmul.get_dst_offset = ukernel_sme.get_dst_offset;
    variants[0].matmul.get_dst_size = ukernel_sme.get_dst_size;
    variants[0].matmul.imatmul = ukernel_sme.run_imatmul;

    variants[1].name = "imatmul_qai8_qai8p_qsi8cxp_sme2";
    variants[1].acc_pack.m = 2 * get_sme_vector_length<int32_t>();
    variants[1].acc_pack.n = 2 * get_sme_vector_length<int32_t>();
    variants[1].acc_pack.k = sizeof(int32_t) / sizeof(int8_t);
    variants[1].acc_step.m = 2 * get_sme_vector_length<int32_t>();
    variants[1].acc_step.n = 2 * get_sme_vector_length<int32_t>();
    variants[1].acc_step.k = sizeof(int32_t) / sizeof(int8_t);
    variants[1].is_supported = cpu_has_sme2;
    variants[1].lhs_pack.get_m_step = kai_get_m_step_lhs_imatmul_pack_x8p2vlx4_x8p_sme;
    variants[1].lhs_pack.get_packed_lhs_offset = kai_get_lhs_packed_offset_lhs_imatmul_pack_x8p2vlx4_x8p_sme;
    variants[1].lhs_pack.get_packed_lhs_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme;
    variants[1].lhs_pack.pack = kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme;
    variants[1].rhs_pack.get_n_step = kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[1].rhs_pack.get_rhs_offset = kai_get_rhs_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[1].rhs_pack.get_bias_offset = kai_get_bias_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[1].rhs_pack.get_scale_offset = kai_get_scale_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[1].rhs_pack.get_packed_rhs_offset =
        kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[1].rhs_pack.get_packed_rhs_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[1].rhs_pack.pack = kai_run_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme;
    variants[1].matmul.get_m_step = ukernel_sme2.get_m_step;
    variants[1].matmul.get_n_step = ukernel_sme2.get_n_step;
    variants[1].matmul.get_lhs_packed_offset = ukernel_sme2.get_lhs_packed_offset;
    variants[1].matmul.get_rhs_packed_offset = ukernel_sme2.get_rhs_packed_offset;
    variants[1].matmul.get_dst_offset = ukernel_sme2.get_dst_offset;
    variants[1].matmul.get_dst_size = ukernel_sme2.get_dst_size;
    variants[1].matmul.imatmul = ukernel_sme2.run_imatmul;

    return variants;
}

const auto& get_gemv_variants() {
    static std::array<MatMulVariant, 1> variants;
    static const kai_matmul_clamp_qai8_qai8p_qsi8cxp_ukernel& ukernel =
        get_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot_interface();

    variants[0].name = "matmul_qai8_qai8_qsi8cxp";
    variants[0].acc_pack.m = 1;
    variants[0].acc_pack.n = 2 * get_sme_vector_length<int32_t>();
    variants[0].acc_pack.k = sizeof(int32_t) / sizeof(int8_t);
    variants[0].acc_step.m = 1;
    variants[0].acc_step.n = 16 * get_sme_vector_length<int32_t>();
    variants[0].acc_step.k = sizeof(int32_t) / sizeof(int8_t);
    variants[0].is_supported = cpu_has_sme2;
    variants[0].lhs_pack = std::nullopt;
    variants[0].rhs_pack = get_rhs_pack();
    variants[0].matmul.get_m_step = ukernel.get_m_step;
    variants[0].matmul.get_n_step = ukernel.get_n_step;
    variants[0].matmul.get_mr = []() -> size_t { return 1; };
    variants[0].matmul.get_nr = ukernel.get_nr;
    variants[0].matmul.get_kr = ukernel.get_kr;
    variants[0].matmul.get_sr = ukernel.get_sr;
    variants[0].matmul.get_packed_lhs_offset = nullptr;
    variants[0].matmul.get_packed_rhs_offset = ukernel.get_rhs_packed_offset;
    variants[0].matmul.get_dst_offset = ukernel.get_dst_offset;
    variants[0].matmul.get_dst_size = ukernel.get_dst_size;
    variants[0].matmul.matmul = ukernel.run_matmul;

    return variants;
}

constexpr uint32_t seed = 0;  ///< Random seed used for tests

/// Quantization parameters
struct Quant {
    float scale;
    int32_t zero_point;
};

/// Reference test data
struct TestReference {
    Range<int8_t> clamp;

    Quant qa_lhs;
    Quant qa_dst;

    Buffer lhs_qai8;
    Buffer lhs_qai8_scales;
    Buffer lhs_qai8_zero_points;
    Buffer lhs_qai8_indirect;
    Buffer lhs_qai8_indirect_packed;
    Buffer lhs_qai8_indirect_padding;
    size_t lhs_qai8_indirect_offset;

    Buffer rhs_qsi8;
    Buffer rhs_scales;

    Buffer bias_qsi32;

    Buffer dst_qsi8_clamped;

    Buffer packed_lhs;
    Buffer packed_rhs;
};

constexpr int8_t padding_value = 0;

// Functionality for hashing generated test data.
// This is particularly useful for portion testing
// which reuses the exact same data for all portions
struct TestDataId {
    MatMulShape shape;
    MatMulShape shape_pack;
    size_t chunk_len;
    bool pad_testing;
    float clamp_ratio;

    struct Hash {
        size_t operator()(const TestDataId& id) const {
            return                                           //
                (MatMulShape::Hash{}(id.shape) << 0) ^       //
                (MatMulShape::Hash{}(id.shape_pack) << 1) ^  //
                (std::hash<size_t>{}(id.chunk_len) << 2) ^   //
                (std::hash<bool>{}(id.pad_testing) << 3) ^   //
                (std::hash<float>{}(id.clamp_ratio) << 4);
        }
    };

private:
    friend bool operator==(const TestDataId& lhs, const TestDataId& rhs) {
        return                                     //
            lhs.shape == rhs.shape &&              //
            lhs.shape_pack == rhs.shape_pack &&    //
            lhs.chunk_len == rhs.chunk_len &&      //
            lhs.pad_testing == rhs.pad_testing &&  //
            lhs.clamp_ratio == rhs.clamp_ratio;
    }
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<TestDataId, TestReference, TestDataId::Hash> g_data;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/// Generate test reference data
const TestReference& get_test_reference(const TestDataId& test_data_id) {
    // ============================================================
    // Generates input and reference output data
    // ============================================================

    // Attempt to find test data in cache
    const auto data_it = g_data.find(test_data_id);
    if (data_it != g_data.end()) {
        return data_it->second;
    }

    const auto& [shape, pack_shape, k_chunk_len, pad_testing, clamp_ratio] = test_data_id;

    // Generates the input data in floating-point.
    Buffer lhs_f32 = fill_random<float>(shape.m * shape.k, seed);
    const Buffer rhs_f32 = fill_random<float>(shape.k * shape.n, seed);
    const Buffer bias_f32 = fill_random<float>(shape.n, seed);

    // Quantizes the input data.
    //   * LHS: 8-bit asymmetric per-matrix quantization.
    //   * RHS: 8-bit symmetric per-channel quantization.
    //   * Bias: 32-bit symmetric per-channel quantization.

    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = shape.m * shape.k;
    lhs_qinfo.dst_type = DataType::QAI8;
    lhs_qinfo.scale_type = DataType::FP32;
    lhs_qinfo.zero_point_type = DataType::I32;
    auto [lhs_ref_quant, lhs_qoutputs] =
        quantize_dynamic(lhs_f32.data(), DataType::FP32, 1, shape.m * shape.k, lhs_qinfo);
    const auto lhs_scale = read_array<float>(lhs_qoutputs.scales.data(), 0);
    const auto lhs_zero_point = read_array<int32_t>(lhs_qoutputs.zero_points.data(), 0);

    const size_t k_chunk_count = shape.k / k_chunk_len;
    assert(k_chunk_count * k_chunk_len == shape.k);

    // Setup an indirection buffer, where each "row" contains `k_chunk_count`
    // pointers to chunks of length `k_chunk_len` in the input_buffer
    Buffer lhs_qai8_indirect(shape.m * k_chunk_count * sizeof(void*));
    Buffer lhs_padding(k_chunk_len, padding_value);
    auto* lhs_qai8_indirect_ptr = reinterpret_cast<uint8_t**>(lhs_qai8_indirect.data());
    for (size_t m_i = 0; m_i < shape.m; ++m_i) {
        for (size_t k_chunk_idx = 0; k_chunk_idx < k_chunk_count; ++k_chunk_idx) {
            const size_t idx = m_i * k_chunk_count + k_chunk_idx;
            if (pad_testing and m_i == 0) {
                // Push padding pointers for first row
                lhs_qai8_indirect_ptr[idx] = reinterpret_cast<uint8_t*>(lhs_padding.data());
            } else {
                uintptr_t offset = m_i * shape.k + k_chunk_idx * k_chunk_len;
                lhs_qai8_indirect_ptr[idx] = reinterpret_cast<uint8_t*>(offset);
            }
        }
    }
    const auto indirection_base = reinterpret_cast<uintptr_t>(lhs_ref_quant.data());

    // Reorder indirection pointers to layout the packing micro-kernel expects
    Buffer lhs_qai8_indirect_packed = reorder_block<const void*>(
        reinterpret_cast<const void*>(lhs_qai8_indirect.data()), shape.m, k_chunk_count, pack_shape.m, 1);

    // Transpose, then quantize symmetrically, then transpose back. This will give one
    // quantization value for each column
    const auto rhs_f32_t = transpose<float>(rhs_f32.data(), shape.k, shape.n);

    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = shape.k;
    rhs_qinfo.dst_type = DataType::QSI8;
    rhs_qinfo.scale_type = DataType::FP32;
    auto [rhs_ref_quant_t, rhs_qoutputs] =
        quantize_dynamic(rhs_f32_t.data(), DataType::FP32, shape.n, shape.k, rhs_qinfo);
    auto rhs_qsi8 = transpose<int8_t>(rhs_ref_quant_t.data(), shape.n, shape.k);

    // Multiply all bias values with the LHS scale
    const auto bias_scales = mul<float>(&lhs_scale, 1, 1, rhs_qoutputs.scales.data(), 1, shape.n);
    // Calculate quantized bias values, by treating bias as column, and
    // scale using RHS scales. This will scale each bias value indiviually
    auto bias_qsi32 =
        quantize_symmetric_per_block<float, int32_t, float>(bias_f32.data(), bias_scales.data(), shape.n, 1, 1);

    // Runs the reference implementation of matmul to produce floating-point result.
    const void* const* lhs_iptr = reinterpret_cast<const void* const*>(lhs_qai8_indirect.data());
    const auto ref_dst_f32 =
        indirect_matmul_nt_t_quantized<int8_t, float, int32_t, int8_t, float, int32_t, int32_t, float, int32_t, float>(
            shape.m, shape.n, k_chunk_count, k_chunk_len,                 // matmul shape
            lhs_iptr, indirection_base, lhs_padding.data(),               // LHS indirection, offset and padding
            &lhs_scale, &lhs_zero_point,                                  // LHS, scaling factor and zero point
            shape.m, shape.k,                                             // LHS quantization window shape
            rhs_ref_quant_t.data(), rhs_qoutputs.scales.data(), nullptr,  // RHS scaling factors
            1, shape.k,                                                   // RHS quantization window shape
            bias_qsi32.data(), bias_scales.data(), nullptr,               // Bias, scaling and zero points
            1                                                             // Bias quantization window shape
        );

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

    const auto ref_dst_f32_clamp_min = ref_dst_f32_min + ref_dst_f32_range * clamp_ratio / 2;
    const auto ref_dst_f32_clamp_max = ref_dst_f32_max - ref_dst_f32_range * clamp_ratio / 2;
    const auto dst_qai8_clamp_min =
        quantize_asymmetric<float, int8_t, int32_t>(ref_dst_f32_clamp_min, dst_scale, dst_zero_point);
    const auto dst_qai8_clamp_max =
        quantize_asymmetric<float, int8_t, int32_t>(ref_dst_f32_clamp_max, dst_scale, dst_zero_point);

    // Clamps and quantizes the reference output matrix.
    const auto ref_dst_f32_clamped =
        clamp<float>(ref_dst_f32.data(), shape.m * shape.n, ref_dst_f32_clamp_min, ref_dst_f32_clamp_max);
    auto ref_dst_qsi8_clamped = quantize_asymmetric_per_block<float, int8_t, float, int32_t>(
        ref_dst_f32_clamped.data(), &dst_scale, &dst_zero_point,  // values, scales, zero point
        1, shape.m * shape.n,                                     // data shape
        shape.m * shape.n                                         // quantization window width
    );

    // Runs the reference implementation of the packing micro-kernels.
    //
    // The reference packing micro-kernels cannot be executed earlier
    // because we need the reference floating-point output first to have
    // the quantization information.
    auto packed_lhs = reorder_block<int8_t>(lhs_ref_quant.data(), shape.m, shape.k, pack_shape.m, pack_shape.k);
    auto packed_rhs = matmul_pack_rhs_nxk_static_quantized<int8_t, float, int32_t>(
        rhs_ref_quant_t.data(), rhs_qoutputs.scales.data(), lhs_scale, dst_scale, bias_qsi32.data(), lhs_zero_point,
        shape.n, shape.k, pack_shape.n, pack_shape.k);

    TestReference& reference = g_data[test_data_id];
    reference.clamp.min = dst_qai8_clamp_min;
    reference.clamp.max = dst_qai8_clamp_max;
    reference.qa_lhs.scale = lhs_scale;
    reference.qa_lhs.zero_point = lhs_zero_point;
    reference.qa_dst.scale = dst_scale;
    reference.qa_dst.zero_point = dst_zero_point;
    reference.lhs_qai8 = std::move(lhs_ref_quant);
    reference.lhs_qai8_scales = std::move(lhs_qoutputs.scales);
    reference.lhs_qai8_zero_points = std::move(lhs_qoutputs.zero_points);
    reference.lhs_qai8_indirect = std::move(lhs_qai8_indirect);
    reference.lhs_qai8_indirect_packed = std::move(lhs_qai8_indirect_packed);
    reference.lhs_qai8_indirect_padding = std::move(lhs_padding);
    reference.lhs_qai8_indirect_offset = indirection_base;
    reference.rhs_qsi8 = std::move(rhs_qsi8);
    reference.rhs_scales = std::move(rhs_qoutputs.scales);
    reference.bias_qsi32 = std::move(bias_qsi32);
    reference.dst_qsi8_clamped = std::move(ref_dst_qsi8_clamped);
    reference.packed_lhs = std::move(packed_lhs);
    reference.packed_rhs = std::move(packed_rhs);

    return reference;
}

/// Test LHS packing
void test_lhs_pack(
    const MatMulShape& shape, const MatMulVariant& variant, const Rect& output_area, const TestReference& reference) {
    KAI_ASSUME_ALWAYS(variant.lhs_pack.has_value());

    const auto imp_packed_lhs_size =
        variant.lhs_pack->get_packed_lhs_size(shape.m, shape.k, variant.acc_pack.m, variant.acc_pack.k, 1);
    ASSERT_EQ(imp_packed_lhs_size, reference.packed_lhs.size());

    Buffer imp_packed_lhs(imp_packed_lhs_size, 0);
    const auto imp_lhs_offset = variant.lhs_pack->get_lhs_offset(output_area.start_row(), shape.k * sizeof(int8_t));
    const auto imp_packed_lhs_offset = variant.lhs_pack->get_packed_lhs_offset(
        output_area.start_row(), shape.k, variant.acc_pack.m, variant.acc_pack.k, 1);

    abi_check(
        variant.lhs_pack->pack, output_area.height(), shape.k, variant.acc_pack.m, variant.acc_pack.k, 1, 0,
        reference.lhs_qai8.data() + imp_lhs_offset, shape.k * sizeof(int8_t),
        imp_packed_lhs.data() + imp_packed_lhs_offset);

    const auto imp_packed_lhs_end_offset = output_area.end_row() < shape.m
        ? variant.lhs_pack->get_packed_lhs_offset(
              output_area.end_row(), shape.k, variant.acc_pack.m, variant.acc_pack.k, 1)
        : imp_packed_lhs_size;

    const auto* imp_packed_lhs_ptr = reinterpret_cast<const uint8_t*>(imp_packed_lhs.data());
    const auto* ref_packed_lhs_ptr = reinterpret_cast<const uint8_t*>(reference.packed_lhs.data());

    for (size_t i = 0; i < reference.packed_lhs.size(); ++i) {
        if (i >= imp_packed_lhs_offset && i < imp_packed_lhs_end_offset) {
            ASSERT_EQ(imp_packed_lhs_ptr[i], ref_packed_lhs_ptr[i]);
        } else {
            ASSERT_EQ(imp_packed_lhs_ptr[i], 0);
        }
    }
}

/// Test RHS packing
void test_rhs_pack(
    const MatMulShape& shape, const MatMulVariant& variant, const Rect& output_area, const TestReference& reference) {
    const auto imp_packed_rhs_size = variant.rhs_pack.get_packed_rhs_size(shape.n, shape.k);
    ASSERT_EQ(imp_packed_rhs_size, reference.packed_rhs.size());
    Buffer imp_packed_rhs(imp_packed_rhs_size, 0);

    const auto imp_rhs_offset = variant.rhs_pack.get_rhs_offset(output_area.start_col());
    const auto imp_bias_offset = variant.rhs_pack.get_bias_offset(output_area.start_col());
    const auto imp_scale_offset = variant.rhs_pack.get_scale_offset(output_area.start_col());
    const auto imp_packed_rhs_offset = variant.rhs_pack.get_packed_rhs_offset(output_area.start_col(), shape.k);

    kai_rhs_pack_qsi8cx_params imp_pack_rhs_params{};
    imp_pack_rhs_params.lhs_zero_point = reference.qa_lhs.zero_point;
    imp_pack_rhs_params.scale_multiplier = reference.qa_lhs.scale / reference.qa_dst.scale;

    abi_check(
        variant.rhs_pack.pack, 1, output_area.width(), shape.k, variant.acc_pack.n, variant.acc_pack.k, 1,
        shape.n * sizeof(int8_t), reference.rhs_qsi8.data() + imp_rhs_offset,
        reference.bias_qsi32.data() + imp_bias_offset, reference.rhs_scales.data() + imp_scale_offset,
        imp_packed_rhs.data() + imp_packed_rhs_offset, 0, &imp_pack_rhs_params);

    const auto imp_packed_rhs_end_offset = output_area.end_col() < shape.n
        ? variant.rhs_pack.get_packed_rhs_offset(output_area.end_col(), shape.k)
        : imp_packed_rhs_size;

    size_t mismatches = 0;
    const auto* imp_packed_rhs_ptr = reinterpret_cast<const uint8_t*>(imp_packed_rhs.data());
    const auto* ref_packed_rhs_ptr = reinterpret_cast<const uint8_t*>(reference.packed_rhs.data());

    for (size_t i = 0; i < reference.packed_rhs.size(); ++i) {
        if (i >= imp_packed_rhs_offset && i < imp_packed_rhs_end_offset) {
            if (imp_packed_rhs_ptr[i] != ref_packed_rhs_ptr[i]) {
                mismatches += 1;
            }
        } else {
            if (imp_packed_rhs_ptr[i] != 0) {
                mismatches += 1;
            }
        }
    }
    ASSERT_EQ(mismatches, 0) << "There are an unexpected amount of mismatches in RHS packing";
}

void compare_matmul_result(
    const MatMulShape& shape, const Rect& output_area, const Buffer& actual, const Buffer& reference) {
    size_t mismatches = 0;
    bool printed_row = false;
    std::ostringstream sstream;
    for (size_t m_i = 0; m_i < shape.m; ++m_i) {
        for (size_t n_i = 0; n_i < shape.n; ++n_i) {
            const auto i = m_i * shape.n + n_i;
            const auto in_area = m_i >= output_area.start_row() && m_i < output_area.end_row() &&
                n_i >= output_area.start_col() && n_i < output_area.end_col();

            const auto imp_value = read_array<int8_t>(actual.data(), i);
            const auto ref_value = in_area ? read_array<int8_t>(reference.data(), i) : 0;
            const auto error = std::abs(imp_value - ref_value);
            const auto threshold = in_area ? 1 : 0;
            const bool mismatch = error > threshold;
            if (mismatch) {
                if (not printed_row) {
                    sstream << " row=" << m_i << ", columns: ";
                    printed_row = true;
                }
                sstream << n_i << ", ";
            }
            mismatches += static_cast<size_t>(mismatch);
        }
        if (printed_row) {
            sstream << "\n";
        }
        printed_row = false;
    }
    ASSERT_EQ(mismatches, 0) << "Mismatches between reference result and actual result:\n" << sstream.str();
}

/// Test MatMul of GEMM/GEMV like kernel
void test_matmul(
    const MatMulShape& shape, const MatMulVariant& variant, const Rect& output_area, const TestReference& reference) {
    const auto imp_dst_size = variant.matmul.get_dst_size(shape.m, shape.n);
    ASSERT_EQ(imp_dst_size, reference.dst_qsi8_clamped.size());

    Buffer imp_dst(imp_dst_size, 0);
    const auto [imp_lhs_offset, lhs_data] = [&]() -> std::tuple<size_t, const Buffer&> {
        if (variant.lhs_pack.has_value()) {
            return {variant.matmul.get_packed_lhs_offset(output_area.start_row(), shape.k), reference.packed_lhs};
        }
        return {output_area.start_row() * shape.k, reference.lhs_qai8};
    }();
    const size_t imp_packed_rhs_offset = variant.matmul.get_packed_rhs_offset(output_area.start_col(), shape.k);
    const size_t imp_dst_offset =
        variant.matmul.get_dst_offset(output_area.start_row(), output_area.start_col(), shape.n * sizeof(int8_t));
    ASSERT_EQ(imp_dst_offset, output_area.start_row() * shape.n + output_area.start_col());

    kai_matmul_requantize32_params imp_main_params{};
    imp_main_params.min_value = reference.clamp.min;
    imp_main_params.max_value = reference.clamp.max;
    imp_main_params.output_zero_point = reference.qa_dst.zero_point;

    abi_check(
        variant.matmul.matmul, output_area.height(), output_area.width(), shape.k, lhs_data.data() + imp_lhs_offset,
        reference.packed_rhs.data() + imp_packed_rhs_offset, imp_dst.data() + imp_dst_offset, shape.n * sizeof(int8_t),
        sizeof(int8_t), &imp_main_params);

    compare_matmul_result(shape, output_area, imp_dst, reference.dst_qsi8_clamped);
}

}  // namespace

using MatMulQuantizedTest = testing::TestWithParam<std::tuple<MatMulVariant, MatMulShape, MatrixPortion, float>>;
using IndirectMatMulQuantizedTestParams = std::tuple<IndirectMatMulVariant, MatMulShape, size_t, MatrixPortion, float>;
using IndirectMatMulQuantizedTest = testing::TestWithParam<IndirectMatMulQuantizedTestParams>;

static std::string test_description(
    const MatMulVariant& variant,  //
    const MatMulShape& shape,      //
    const MatrixPortion& portion, float clamp_ratio) {
    std::ostringstream sstream;

    sstream << test_description(variant.name, shape, portion, true)  //
            << "__clamp_ratio_" << static_cast<int>(clamp_ratio * 100);

    return sstream.str();
};

[[maybe_unused]] static void PrintTo(const IndirectMatMulQuantizedTestParams& param, std::ostream* os) {
    const auto& [variant, shape, k_chunk_length, portion, clamp_rate] = param;

    *os << variant.name << "__";
    PrintTo(shape, os);
    *os << "__K_chunk_length_" << k_chunk_length;
    *os << "__clamp_rate_" << static_cast<int>(clamp_rate * 100) << "__";
    PrintTo(portion, os);
};

TEST_P(MatMulQuantizedTest, EndToEnd) {
    const auto& [variant, shape, output_portion, clamp_ratio] = GetParam();

    if (!variant.is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    TestDataId test_data_id{shape, variant.acc_pack, shape.k, false, clamp_ratio};
    const TestReference& reference = get_test_reference(test_data_id);

    // Check scheduling parameters
    const auto imp_mr = variant.matmul.get_mr();
    const auto imp_nr = variant.matmul.get_nr();
    const auto imp_kr = variant.matmul.get_kr();
    const auto imp_sr = variant.matmul.get_sr();

    ASSERT_EQ(imp_mr, variant.acc_pack.m);
    ASSERT_EQ(imp_nr, variant.acc_pack.n);
    ASSERT_EQ(imp_kr, variant.acc_pack.k);
    ASSERT_EQ(imp_sr, 1);

    // Check that stepping is a multiple of accumulation
    const auto imp_m_step = variant.matmul.get_m_step();
    const auto imp_n_step = variant.matmul.get_n_step();
    ASSERT_EQ(imp_m_step, variant.acc_step.m);
    ASSERT_EQ(imp_n_step, variant.acc_step.n);

    // Test kernels. Note that packing and actual stepping might not be the same
    const auto pack_portion = output_portion.compute_portion(shape.m, shape.n, variant.acc_pack.m, variant.acc_pack.n);
    const auto matmul_portion =
        output_portion.compute_portion(shape.m, shape.n, variant.acc_step.m, variant.acc_step.n);
    if (variant.lhs_pack.has_value()) {
        test_lhs_pack(shape, variant, pack_portion, reference);
    }
    test_rhs_pack(shape, variant, pack_portion, reference);
    test_matmul(shape, variant, matmul_portion, reference);
}

namespace imatmul {

/// Perform LHS IMATMUL packing
static Buffer lhs_pack(
    const LhsPackIndirectKernel& variant, const Rect& portion, const TestReference& reference, size_t m,
    const KChunk& k_chunk) {
    const void* const* indirection_pointer =
        reinterpret_cast<const void* const*>(reference.lhs_qai8_indirect_packed.data());

    // Allocate buffer
    const size_t dst_size = variant.get_packed_lhs_size(m, k_chunk.count, k_chunk.length);
    Buffer packed(dst_size);

    // Calculate offsets
    const size_t input_offset = portion.start_row() * k_chunk.count;
    const size_t dst_offset = variant.get_packed_lhs_offset(portion.start_row(), k_chunk.count, k_chunk.length);

    abi_check(
        variant.pack,                                     // Kernel
        portion.height(), k_chunk.count, k_chunk.length,  // Dimensions
        indirection_pointer + input_offset,               // Indirection input
        reference.lhs_qai8_indirect_offset,               // chunk offset
        reference.lhs_qai8_indirect_padding.data(),       // padding pointer
        packed.data() + dst_offset);

    return packed;
}

/// Perform RHS IMATMUL packing
static Buffer rhs_pack(
    const RhsPackIndirectKernel& variant, const Rect& portion, const TestReference& reference, size_t n,
    const KChunk& k_chunk) {
    // Allocate output buffer
    const size_t dst_size = variant.get_packed_rhs_size(n, k_chunk.count, k_chunk.length);
    Buffer packed(dst_size);

    // Caluclate effective quantization parameters
    const kai_rhs_pack_qsi8cx_params quantization{
        reference.qa_lhs.zero_point,
        reference.qa_lhs.scale / reference.qa_dst.scale,
    };

    // Calculate offsets
    const size_t rhs_offset = variant.get_rhs_offset(portion.start_col());
    const size_t bias_offset = variant.get_bias_offset(portion.start_col());
    const size_t scale_offset = variant.get_scale_offset(portion.start_col());
    const size_t dst_offset = variant.get_packed_rhs_offset(portion.start_col(), k_chunk.count, k_chunk.length);

    // Pack
    abi_check(
        variant.pack,                                    // Kernel
        portion.width(), k_chunk.count, k_chunk.length,  // Dimensions
        n * sizeof(uint8_t),                             // Row stride
        reference.rhs_qsi8.data() + rhs_offset,          // RHS matrix
        reference.bias_qsi32.data() + bias_offset,       // Bias
        reference.rhs_scales.data() + scale_offset,      // Scales
        packed.data() + dst_offset,                      // Output
        &quantization);

    return packed;
}

/// Calculate the matmul result from IMATMUL kernels
static Buffer matmul(
    const MatMulIndirectKernel& variant, const Rect& portion, const TestReference& reference, const Buffer& packed_lhs,
    const Buffer& packed_rhs, const MatMulShape& shape, const KChunk& k_chunk) {
    // Calculate portion offsets.
    size_t dst_offset = variant.get_dst_offset(portion.start_row(), portion.start_col(), shape.n);
    size_t lhs_offset = variant.get_lhs_packed_offset(portion.start_row(), k_chunk.count, k_chunk.length);
    size_t rhs_offset = variant.get_rhs_packed_offset(portion.start_col(), k_chunk.count, k_chunk.length);

    // Allocate output buffer
    const size_t dst_size = variant.get_dst_size(shape.m, shape.n);
    Buffer dst(dst_size, 0);

    // Calculate geffective uantization parameters
    kai_matmul_requantize32_params requantization{};
    requantization.min_value = reference.clamp.min;
    requantization.max_value = reference.clamp.max;
    requantization.output_zero_point = reference.qa_dst.zero_point;

    // Call matmul kernel
    abi_check(
        variant.imatmul,                                                   // Kernel
        portion.height(), portion.width(), k_chunk.count, k_chunk.length,  // Dimensions
        packed_lhs.data() + lhs_offset,                                    // LHS
        packed_rhs.data() + rhs_offset,                                    // RHS
        dst.data() + dst_offset,                                           // DST
        shape.n * sizeof(uint8_t), &requantization);

    return dst;
}
}  // namespace imatmul

TEST_P(IndirectMatMulQuantizedTest, EndToEnd) {
    /* This is a bit special, as shape.k must be k_chunk_len * k_chunk_count
     * so instead of inventing a new special kind of shape, simply multiply
     * with `k_chunk_len` here */
    const auto& [variant, shape_k_chunk, k_chunk_len, output_portion, clamp_ratio] = GetParam();
    const KChunk k_chunk{shape_k_chunk.k, k_chunk_len};
    MatMulShape shape{shape_k_chunk.m, shape_k_chunk.n, k_chunk.count * k_chunk.length};

    if (!variant.is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    // Toggle padding testst when LHS has more than one row
    TestDataId test_data_id{shape, variant.acc_pack, k_chunk.length, shape.m > 1, clamp_ratio};
    const TestReference& reference = get_test_reference(test_data_id);
    const Rect portion = output_portion.compute_portion(shape.m, shape.n, variant.acc_step.m, variant.acc_step.n);

    Buffer packed_lhs = imatmul::lhs_pack(variant.lhs_pack, portion, reference, shape.m, k_chunk);
    Buffer packed_rhs = imatmul::rhs_pack(variant.rhs_pack, portion, reference, shape.n, k_chunk);
    Buffer impl_result = imatmul::matmul(variant.matmul, portion, reference, packed_lhs, packed_rhs, shape, k_chunk);
    compare_matmul_result(shape, portion, impl_result, reference.dst_qsi8_clamped);
}

static constexpr std::array shapes{
    // clang-format off
    MatMulShape{  1,    1,  1},
    MatMulShape{  1,   16,  4},
    MatMulShape{  1,   16, 16},
    MatMulShape{  1,   17,  4},
    MatMulShape{  1,   19, 24},
    MatMulShape{  1,   32,  4},
    MatMulShape{  1,   32, 32},
    MatMulShape{  1,   33,200},
    MatMulShape{  1,   49, 21},
    MatMulShape{  1,   64,  4},
    MatMulShape{  1,   65,  4},
    MatMulShape{  1,  300, 10},
    MatMulShape{  1,  512,  4},
    MatMulShape{  1, 1523, 10},
    MatMulShape{  2,  195, 50},
    MatMulShape{  3,    6,  6},
    MatMulShape{  3,   28, 25},
    MatMulShape{  3,  184,177},
    MatMulShape{  4,   16, 27},
    MatMulShape{  5,  136, 23},
    MatMulShape{  6,   18, 31},
    MatMulShape{  6,   28,  1},
    MatMulShape{  6,   29, 24},
    MatMulShape{ 16,   16,  4},
    MatMulShape{ 20,   30, 40},
    MatMulShape{ 23,    1, 43},
    MatMulShape{ 32,   14,  1},
    MatMulShape{ 32,   16, 27},
    MatMulShape{ 32,   32,  3},
    MatMulShape{ 32,   32,  4},
    MatMulShape{ 33,   29, 24},
    MatMulShape{ 64,   64,  3},
    MatMulShape{ 64,   64,  4},
    MatMulShape{ 96,   96,  3},
    MatMulShape{123,   85, 45},
    MatMulShape{128,  128,  3},
    MatMulShape{130,  130,  6},
    // clang-format on
};

static constexpr std::array portions{
    // clang-format off
    //       (Start row , start col , height , width)
    MatrixPortion(   0  , 0         , 1      , 1)     , // Full matrix.
    MatrixPortion(   0  , 0         , 1      , 0.5)   , // Left half
    MatrixPortion(   0  , 0         , 0.5    , 1)     , // Upper half
    MatrixPortion(   0  , 0.5       , 1      , 0.5)   , // Right half
    MatrixPortion( 0.5  , 0         , 0.5    , 1)     , // Bottom half
    MatrixPortion( 0.4  , 0.4       , 0.3    , 0.3)   , // Center ninth
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(
    matmul_clamp_qai8_qai8p_qsi8cxp, MatMulQuantizedTest,
    testing::Combine(
        testing::ValuesIn(get_gemm_variants()),  //
        testing::ValuesIn(shapes),               //
        testing::ValuesIn({
            // clang-format off
            MatrixPortion(   0,    0,    1,    1), // Full matrix.
            MatrixPortion(   0,    0, 0.25, 0.25), // Top-left corner.
            MatrixPortion(0.75, 0.75,    1,    1), // Bottom-right corner.
            // clang-format on
        }),
        testing::ValuesIn(std::initializer_list<float>{0.0F, 0.1F, 0.5F})),
    [](const auto& info) -> std::string {
        return test_description(
            std::get<MatMulVariant>(info.param),  //
            std::get<MatMulShape>(info.param),    //
            std::get<MatrixPortion>(info.param),  //
            std::get<float>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    matmul_clamp_qai8_qai8_qsi8cxp, MatMulQuantizedTest,
    testing::Combine(
        testing::ValuesIn(get_gemv_variants()),
        testing::ValuesIn({
            // clang-format off
            MatMulShape{  1,    1,  1},
            MatMulShape{  1,   16,  4},
            MatMulShape{  1,   16, 16},
            MatMulShape{  1,   17,  4},
            MatMulShape{  1,   19, 24},
            MatMulShape{  1,   32,  4},
            MatMulShape{  1,   32, 32},
            MatMulShape{  1,   33,200},
            MatMulShape{  1,   49, 21},
            MatMulShape{  1,   64,  4},
            MatMulShape{  1,   65,  4},
            MatMulShape{  1,  300, 10},
            MatMulShape{  1,  512,  4},
            MatMulShape{  1, 1523, 10},
            // clang-format on
        }),
        testing::ValuesIn({
            // clang-format off
            MatrixPortion(0,   0, 1,  1), // Full matrix.
            MatrixPortion(0,  .5, 1, .5), // Right half
            MatrixPortion(0,   0, 1, .5), // Left half
            MatrixPortion(0, .25, 1, .5)  // Middle half
            // clang-format on
        }),
        // Clamp range
        testing::ValuesIn(std::initializer_list<float>{0.0F, 0.1F, 0.5F})),
    [](const auto& info) -> std::string {
        return test_description(
            std::get<MatMulVariant>(info.param),  //
            std::get<MatMulShape>(info.param),    //
            std::get<MatrixPortion>(info.param),  //
            std::get<float>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    ShapesSmallKC, IndirectMatMulQuantizedTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_gemm_variants()),  //
        testing::ValuesIn(shapes),                        //
        // k_chunk_len
        testing::ValuesIn(std::initializer_list<size_t>{1, 2, 3, 4, 8, 11}),  //
        testing::ValuesIn(portions),                                          //
        // Clamp range
        testing::Values(0.1F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    ShapesKC32, IndirectMatMulQuantizedTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_gemm_variants()),  //
        testing::ValuesIn(shapes),                        //
        // k_chunk_len
        testing::ValuesIn(std::initializer_list<size_t>{32}),  //
        testing::ValuesIn(portions),                           //
        // Clamp range
        testing::Values(0.1F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    Clamp, IndirectMatMulQuantizedTest,
    testing::Combine(
        testing::ValuesIn(get_indirect_gemm_variants()),  //
        testing::ValuesIn(shapes),                        //
        // k_chunk_len
        testing::ValuesIn(std::initializer_list<size_t>{1}),  //
        testing::Values(MatrixPortion(0, 0, 1, 1)),           //
        // Clamp range
        testing::ValuesIn(std::initializer_list<float>{0.0F, 0.1F, 0.5F})),  //
    testing::PrintToStringParamName());

}  // namespace kai::test
