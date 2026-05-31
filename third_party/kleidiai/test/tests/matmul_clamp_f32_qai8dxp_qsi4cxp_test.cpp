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
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pad.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {
/// Matrix multiplication test information.

enum class RhsPackType { NxK, KxN };

using ukernel_rhs_pack_function = std::function<decltype(kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0)>;
using ukernel_get_rhs_packed_size = std::function<decltype(kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0)>;
using ukernel_get_rhs_packed_offset = std::function<decltype(kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0)>;
using ukernel_get_rhs_offset = std::function<decltype(kai_get_rhs_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon)>;

template <typename T>
struct UkernelVariantCustom : public UkernelVariant<T> {
    ukernel_rhs_pack_function run_rhs_pack;
    ukernel_get_rhs_packed_size get_rhs_packed_size;
    ukernel_get_rhs_packed_offset get_rhs_packed_offset;
    ukernel_get_rhs_offset get_rhs_offset;
    RhsPackType rhs_pack_type;

    UkernelVariantCustom() = delete;

    UkernelVariantCustom(
        T interface, std::string_view name, const std::function<bool(void)>& fn_is_supported,
        ukernel_rhs_pack_function run_rhs_pack, ukernel_get_rhs_packed_size get_rhs_packed_size,
        ukernel_get_rhs_packed_offset get_rhs_packed_offset, ukernel_get_rhs_offset get_rhs_offset,
        const RhsPackType pack_type) :
        UkernelVariant<T>(interface, name, fn_is_supported),
        run_rhs_pack(std::move(run_rhs_pack)),
        get_rhs_packed_size(std::move(get_rhs_packed_size)),
        get_rhs_packed_offset(std::move(get_rhs_packed_offset)),
        get_rhs_offset(std::move(get_rhs_offset)),
        rhs_pack_type(pack_type) {
    }
};

static const std::array<UkernelVariantCustom<kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel>, 20>
    variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp = {
        {{UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa),
          "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa__RHS_NxK__", cpu_has_sme2,
          kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
          kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
          kai_get_rhs_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon, RhsPackType::NxK},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot),
          "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot__RHS_NxK__", cpu_has_sme2,
          kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
          kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
          kai_get_rhs_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon, RhsPackType::NxK},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod__RHS_NxK__", cpu_has_dotprod,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod__RHS_KxN__", cpu_has_dotprod,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod__RHS_NxK__", cpu_has_dotprod,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod__RHS_KxN__", cpu_has_dotprod,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod__RHS_NxK__", cpu_has_dotprod,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod__RHS_KxN__", cpu_has_dotprod,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod__RHS_NxK__", cpu_has_dotprod,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod__RHS_KxN__", cpu_has_dotprod,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod__RHS_NxK__", cpu_has_dotprod,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod__RHS_KxN__", cpu_has_dotprod,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm__RHS_NxK__", cpu_has_i8mm,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm__RHS_KxN__", cpu_has_i8mm,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm__RHS_NxK__", cpu_has_i8mm,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm__RHS_KxN__", cpu_has_i8mm,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm__RHS_NxK__", cpu_has_i8mm,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm__RHS_KxN__", cpu_has_i8mm,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN},

         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm__RHS_NxK__", cpu_has_i8mm,
          kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
          RhsPackType::NxK},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm__RHS_KxN__", cpu_has_i8mm,
          kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0, kai_get_rhs_offset_rhs_pack_kxn_qsi4cxp_qs4cxs1s0,
          RhsPackType::KxN}}

};

class MatMulTest_f32_qai8dxp_qsi4cxp : public ::testing::TestWithParam<MatMulTestPortionedParams> {};

TEST_P(MatMulTest_f32_qai8dxp_qsi4cxp, Offset_RHS) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    auto m_step = ukernel_variant.interface.get_m_step();
    auto n_step = ukernel_variant.interface.get_n_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.get_rhs_packed_offset(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qai8dxp_qsi4cxp, Offset_LHS) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    auto m_step = ukernel_variant.interface.get_m_step();
    auto n_step = ukernel_variant.interface.get_n_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto mr = ukernel_variant.interface.get_mr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    const auto lhs_start_row = rect.start_row();
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qai8dxp_qsi4cxp, EndToEnd_RHS_nxk_qsi4cx) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }
    if (ukernel_variant.rhs_pack_type == RhsPackType::KxN) {
        GTEST_SKIP() << "Wrong type. This test for NxK";
    }

    const uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_biases = fill_random<float>(N, seed + 2);

    std::uniform_real_distribution<float> dist(-10.0, 1.0);
    std::mt19937 rnd(seed + 1);
    const auto ref_rhs = fill_matrix_raw<float>(1, N * K, [&dist, &rnd](size_t, size_t) { return dist(rnd); });

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 4-bit symmetric quantization.
    //   * Performs GEMM.
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
    const auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, float, int32_t, Int4, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), K,
        ref_rhs_quant.data(), rhs_qoutputs.scales.data(), nullptr, K, ref_biases.data(),
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    size_t lhs_stride = K * sizeof(float);

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height() /* m */, K, mr, kr, sr, 0 /* m_idx_start*/,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    //   * Generates the 4-bit unsigned symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto ref_rhs_qsi4_padded = pad_row<Int4>(
        ref_rhs_quant.data(), N, K, K, round_up_multiple(K, 2), round_up_division(N * round_up_multiple(K, 2), 2));

    const auto imp_packed_rhs_size = ukernel_variant.get_rhs_packed_size(N, K, nr, kr, sr);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.get_rhs_packed_offset(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    auto rhs_offset = ukernel_variant.get_rhs_offset(rhs_start_row, round_up_division(K, 2));
    size_t bias_offset = rhs_start_row * sizeof(float);
    size_t scale_offset = rhs_start_row * sizeof(float);

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 0;

    abi_check(
        ukernel_variant.run_rhs_pack, 1, rect.width() /* n */, K, nr, kr, sr,
        reinterpret_cast<const uint8_t*>(ref_rhs_qsi4_padded.data() + rhs_offset),
        reinterpret_cast<const float*>(ref_biases.data() + bias_offset),
        reinterpret_cast<const float*>(rhs_qoutputs.scales.data() + scale_offset),
        imp_packed_rhs.data() + rhs_packed_offset, 0, &params);

    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    abi_check(
        ukernel_variant.interface.run_matmul, rect.height(), rect.width(), K, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    for (size_t y = 0; y < rect.height(); ++y) {
        for (size_t x = 0; x < rect.width(); ++x) {
            const auto imp_value =
                read_array<float>(imp_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto ref_value =
                read_array<float>(ref_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : imp_value;

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

TEST_P(MatMulTest_f32_qai8dxp_qsi4cxp, EndToEnd_RHS_nxk_qsu4cx) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }
    if (ukernel_variant.rhs_pack_type == RhsPackType::KxN) {
        GTEST_SKIP() << "Wrong type. This test for NxK";
    }

    const uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);

    std::uniform_real_distribution<float> dist(-10.0, 1.0);
    std::mt19937 rnd(seed + 1);
    const auto ref_rhs = fill_matrix_raw<float>(1, N * K, [&dist, &rnd](size_t, size_t) { return dist(rnd); });

    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 4-bit symmetric quantization.
    //   * Performs GEMM.
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
    const auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, float, int32_t, Int4, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), K,
        ref_rhs_quant.data(), rhs_qoutputs.scales.data(), nullptr, K, ref_biases.data(),
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    size_t lhs_stride = K * sizeof(float);

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height() /* m */, K, mr, kr, sr, 0 /* m_idx_start*/,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_quant.data(), N * K);
    // Runs the RHS packing micro-kernel.
    //   * Generates the 4-bit unsigned symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto ref_rhs_qsu4_padded = pad_row<UInt4>(
        ref_rhs_qsu4.data(), N, K, K, round_up_multiple(K, 2), round_up_division(N * round_up_multiple(K, 2), 2));

    const auto imp_packed_rhs_size = ukernel_variant.get_rhs_packed_size(N, K, nr, kr, sr);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.get_rhs_packed_offset(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    auto rhs_offset = ukernel_variant.get_rhs_offset(rhs_start_row, round_up_division(K, 2));
    size_t bias_offset = rhs_start_row * sizeof(float);
    size_t scale_offset = rhs_start_row * sizeof(float);

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    abi_check(
        ukernel_variant.run_rhs_pack, 1, rect.width() /* n */, K, nr, kr, sr,
        reinterpret_cast<const uint8_t*>(ref_rhs_qsu4_padded.data() + rhs_offset),
        reinterpret_cast<const float*>(ref_biases.data() + bias_offset),
        reinterpret_cast<const float*>(rhs_qoutputs.scales.data() + scale_offset),
        imp_packed_rhs.data() + rhs_packed_offset, 0, &params);

    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);
    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    abi_check(
        ukernel_variant.interface.run_matmul, rect.height(), rect.width(), K, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    for (size_t y = 0; y < rect.height(); ++y) {
        for (size_t x = 0; x < rect.width(); ++x) {
            const auto imp_value =
                read_array<float>(imp_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto ref_value =
                read_array<float>(ref_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : imp_value;

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

TEST_P(MatMulTest_f32_qai8dxp_qsi4cxp, EndToEnd_RHS_kxn_qsi4cx) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }
    if (ukernel_variant.rhs_pack_type == RhsPackType::NxK) {
        GTEST_SKIP() << "Wrong type. This test for KxN";
    }

    const uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);

    std::uniform_real_distribution<float> dist(-10.0, 1.0);
    std::mt19937 rnd(seed + 1);
    const auto ref_rhs = fill_matrix_raw<float>(1, N * K, [&dist, &rnd](size_t, size_t) { return dist(rnd); });

    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Transposed(nxk) RHS dimensions
    const size_t ref_rhs_qsi4_nxk_stride = K;

    // Non-Transposed(kxn) RHS dimensions
    const size_t ref_rhs_qsi4_kxn_stride = round_up_multiple(N, 2);
    const size_t ref_rhs_qsi4_kxn_size_bytes = round_up_division(K * ref_rhs_qsi4_kxn_stride, 2);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 4-bit symmetric quantization.
    //   * Performs GEMM.
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
    const auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    const auto ref_rhs_qsi4 = transpose_with_padding<Int4>(
        ref_rhs_quant.data(), N, K, ref_rhs_qsi4_nxk_stride, ref_rhs_qsi4_kxn_stride, ref_rhs_qsi4_kxn_size_bytes);

    const auto ref_dst = matmul_clamp_nt_nt<int8_t, float, int32_t, Int4, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), K,
        ref_rhs_qsi4.data(), rhs_qoutputs.scales.data(), nullptr, K, ref_biases.data(),
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    size_t lhs_stride = K * sizeof(float);

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);
    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height() /* m */, K, mr, kr, sr, 0 /* m_idx_start*/,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    //   * Generates the 4-bit unsigned symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto ref_rhs_qsi4_padded = pad_row<Int4>(
        ref_rhs_qsi4.data(), K, N, N, round_up_multiple(N, 2), round_up_division(K * round_up_multiple(N, 2), 2));
    const auto imp_packed_rhs_size = ukernel_variant.get_rhs_packed_size(N, K, nr, kr, sr);

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.get_rhs_packed_offset(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 0;
    abi_check(
        ukernel_variant.run_rhs_pack, 1, N, K, nr, kr, sr, reinterpret_cast<const uint8_t*>(ref_rhs_qsi4_padded.data()),
        reinterpret_cast<const float*>(ref_biases.data()), reinterpret_cast<const float*>(rhs_qoutputs.scales.data()),
        imp_packed_rhs.data(), 0, &params);

    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    abi_check(
        ukernel_variant.interface.run_matmul, rect.height(), rect.width(), K, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    for (size_t y = 0; y < rect.height(); ++y) {
        for (size_t x = 0; x < rect.width(); ++x) {
            const auto imp_value =
                read_array<float>(imp_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto ref_value =
                read_array<float>(ref_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : imp_value;

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

TEST_P(MatMulTest_f32_qai8dxp_qsi4cxp, EndToEnd_RHS_kxn_qsu4cx) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }
    if (ukernel_variant.rhs_pack_type == RhsPackType::NxK) {
        GTEST_SKIP() << "Wrong type. This test for KxN";
    }

    const uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);

    std::uniform_real_distribution<float> dist(-10.0, 1.0);
    std::mt19937 rnd(seed + 1);
    const auto ref_rhs = fill_matrix_raw<float>(1, N * K, [&dist, &rnd](size_t, size_t) { return dist(rnd); });

    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Transposed(nxk) RHS dimensions
    const size_t ref_rhs_qsi4_nxk_stride = K;

    // Non-Transposed(kxn) RHS dimensions
    const size_t ref_rhs_qsi4_kxn_stride = round_up_multiple(N, 2);
    const size_t ref_rhs_qsi4_kxn_size = K * ref_rhs_qsi4_kxn_stride;
    const size_t ref_rhs_qsi4_kxn_size_bytes = round_up_division(ref_rhs_qsi4_kxn_size, 2);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 4-bit symmetric quantization.
    //   * Performs GEMM.
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
    const auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    const auto ref_rhs_qsi4 = transpose_with_padding<Int4>(
        ref_rhs_quant.data(), N, K, ref_rhs_qsi4_nxk_stride, ref_rhs_qsi4_kxn_stride, ref_rhs_qsi4_kxn_size_bytes);

    const auto ref_dst = matmul_clamp_nt_nt<int8_t, float, int32_t, Int4, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), K,
        ref_rhs_qsi4.data(), rhs_qoutputs.scales.data(), nullptr, K, ref_biases.data(),
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    size_t lhs_stride = K * sizeof(float);

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height() /* m */, K, mr, kr, sr, 0 /* m_idx_start*/,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    //   * Generates the 4-bit unsigned symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_qsi4.data(), ref_rhs_qsi4_kxn_size);
    const auto ref_rhs_qsu4_padded = pad_row<UInt4>(
        ref_rhs_qsu4.data(), K, N, N, round_up_multiple(N, 2), round_up_division(K * round_up_multiple(N, 2), 2));
    const auto imp_packed_rhs_size = ukernel_variant.get_rhs_packed_size(N, K, nr, kr, sr);

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    abi_check(
        ukernel_variant.run_rhs_pack, 1, N, K, nr, kr, sr, reinterpret_cast<const uint8_t*>(ref_rhs_qsu4_padded.data()),
        reinterpret_cast<const float*>(ref_biases.data()), reinterpret_cast<const float*>(rhs_qoutputs.scales.data()),
        imp_packed_rhs.data(), 0, &params);

    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    abi_check(
        ukernel_variant.interface.run_matmul, rect.height(), rect.width(), K, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::FP32);
    const auto success = compare(imp_dst.data(), ref_dst.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qai8dxp_qsi4cxp,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp.size()),
        testing::Values(
            MatMulShape{16, 32, 64},   //
            MatMulShape{16, 32, 36},   //
            MatMulShape{15, 35, 65},   //
            MatMulShape{8, 32, 64},    //
            MatMulShape{15, 31, 45},   //
            MatMulShape{1, 35, 65},    //
            MatMulShape{1, 128, 32},   //
            MatMulShape{64, 128, 32},  //
            MatMulShape{1, 225, 55},   //
            MatMulShape{125, 200, 56}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),     // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),  // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),  // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8)  // Somewhere Middle
            )),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qai8dxp_qsi4cxp.at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);

        return test_description(name, shape, portion, true);
    });

}  // namespace kai::test
