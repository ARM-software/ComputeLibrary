//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.h"
#include "test/common/abi_checker.hpp"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cache.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_type.hpp"
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
#include "test/reference/pad.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

namespace {

// LHS QAI8DXP
using kai_get_lhs_packed_size_func_t = decltype(&kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32);
using kai_get_lhs_packed_offset_func_t = decltype(&kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32);
using kai_get_lhs_offset_func_t = decltype(&kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32);
using kai_run_lhs_pack_func_t = decltype(&kai_run_lhs_quant_pack_qai8dxp_f32);

// LHS QAI8DXP pack interface
struct kai_qai8dxp_pack_functions {
    kai_get_lhs_packed_size_func_t packed_size;
    kai_get_lhs_packed_offset_func_t get_packed_offset;
    kai_get_lhs_offset_func_t get_offset;
    kai_run_lhs_pack_func_t run_pack;
};

// RHS QSI4C32P (nxk, BF16 block scales; sums float, bias float)
using kai_get_rhs_packed_size_func_t = decltype(&kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0);
using kai_get_rhs_packed_offset_func_t = decltype(&kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0);
using kai_get_rhs_offset_func_t = decltype(&kai_get_rhs_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0);
using kai_run_rhs_pack_func_t = decltype(&kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0);

// RHS QSI4C32P pack interface
struct kai_qsi4c32p_pack_functions {
    kai_get_rhs_packed_size_func_t packed_size;
    kai_get_rhs_packed_offset_func_t get_packed_offset;
    kai_get_rhs_offset_func_t get_offset;
    kai_run_rhs_pack_func_t run_pack;
};

const auto& get_f32_gemm_variants() noexcept {
    using Variant = UkernelMatmulPackVariant<
        kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel, kai_qai8dxp_pack_functions, kai_qsi4c32p_pack_functions>;

    static const std::array<Variant, 12> variants = {{
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
            /*rhs_s0s1_input=*/false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32p_qsu4c32s1s0, false),
        // SME2 MOPA
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa, cpu_has_sme2, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon, false),
    }};

    return variants;
}

const auto& get_f32_gemv_variants() noexcept {
    using Variant = UkernelMatmulPackVariant<
        kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel, kai_qai8dxp_pack_functions, kai_qsi4c32p_pack_functions>;

    static const std::array<Variant, 1> variants = {{
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot, cpu_has_sme2, lhs_quant_pack_qai8dxp_f32,
            rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon, false),
    }};

    return variants;
}

const auto& get_bf16_gemm_variants() noexcept {
    using Variant = UkernelVariant<kai_matmul_clamp_bf16_qai8dxp_qsi4c32p_ukernel>;
    static const std::array<Variant, 2> variants = {
        Variant{
            UKERNEL_MATMUL_VARIANT(clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod),
            "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod", cpu_has_dotprod_and_bf16},
        Variant{
            UKERNEL_MATMUL_VARIANT(clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm),
            "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm", cpu_has_i8mm_and_bf16},
    };
    return variants;
}

// NEON/i8mm only (exclude SME2)
const auto& get_f32_neon_gemm_variants_only() {
    static std::vector<UkernelMatmulPackVariant<
        kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel, kai_qai8dxp_pack_functions, kai_qsi4c32p_pack_functions>>
        filtered;
    if (filtered.empty()) {
        const auto& all = get_f32_gemm_variants();
        for (const auto& v : all) {
            const char* n = v.ukernel.name.data();
            if (n == nullptr || std::strstr(n, "sme2") == nullptr) {
                filtered.push_back(v);
            }
        }
    }
    return filtered;
}

enum class RhsPackType : std::uint8_t { NxK = 0, KxN = 1 };

std::tuple<Buffer, size_t> pack_lhs_qai8dxp(
    const kai_qai8dxp_pack_functions& pack_interface, const size_t M, const size_t K, const size_t mr, const size_t kr,
    const size_t sr, const Buffer& lhs_values_f32, const size_t lhs_stride_bytes, const size_t rect_start_row,
    const size_t rect_height) {
    const auto lhs_packed_size = pack_interface.packed_size(M, K, mr, kr, sr);
    Buffer lhs_packed(lhs_packed_size, 0);

    const auto lhs_offset = pack_interface.get_offset(rect_start_row, lhs_stride_bytes);
    const auto lhs_packed_offset = pack_interface.get_packed_offset(rect_start_row, K, mr, kr, sr);

    abi_check(
        pack_interface.run_pack, rect_height, K, mr, kr, sr, 0,
        reinterpret_cast<const float*>(lhs_values_f32.data() + lhs_offset), lhs_stride_bytes,
        lhs_packed.data() + lhs_packed_offset);

    return {std::move(lhs_packed), lhs_packed_offset};
}

// Executes the scalar RHS packing micro-kernel.
std::tuple<Buffer, size_t> pack_rhs_qsi4c32pscalebf16(
    // clang-format off
    const size_t N,
    const size_t K,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    const size_t bl,
    const Buffer& rhs_values_qsi4,
    const Buffer& biases,
    const size_t bias_offset,
    const Buffer& rhs_scales,
    const RhsPackType pack_type,
    const size_t rect_start_row,
    const size_t rect_width,
    const bool use_ps1s0) {
    // clang-format on
    const size_t width = pack_type == RhsPackType::KxN ? N : K;
    const size_t height = pack_type == RhsPackType::KxN ? K : N;
    constexpr kai_datatype scale_dt = kai_dt_bf16;

    const size_t rhs_stride = round_up_multiple(width, 2);
    const size_t rhs_stride_bytes = round_up_division(width, 2);
    const size_t scales_stride_bytes = round_up_division(K, bl) * kai_get_datatype_size_in_bytes(scale_dt);

    KAI_ASSUME_ALWAYS(rhs_values_qsi4.size() == round_up_division(height * rhs_stride, 2));

    const auto rhs_values_qsu4 = cast_qsu4_qsi4(rhs_values_qsi4.data(), rhs_values_qsi4.size() * 2);
    const size_t dst_bytes_total = round_up_division(height * rhs_stride, 2);
    const size_t dst_bytes_total_safe = dst_bytes_total + rhs_stride_bytes + 8;
    const auto rhs_qsu4 =
        pad_row<UInt4>(rhs_values_qsu4.data(), height, width, width, rhs_stride_bytes * 2, dst_bytes_total_safe);

    const size_t scale_offset = rect_start_row * scales_stride_bytes;
    size_t rhs_offset = 0;
    size_t rhs_packed_offset = 0;
    size_t imp_packed_rhs_size = 0;

    if (pack_type == RhsPackType::KxN) {
        if (use_ps1s0) {
            rhs_offset =
                kai_get_rhs_offset_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(rect_start_row, rhs_stride_bytes);
            rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
                rect_start_row, K, nr, kr, sr, bl, scale_dt);
            imp_packed_rhs_size =
                kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(N, K, nr, kr, sr, bl, scale_dt);
        } else {
            rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(rect_start_row, rhs_stride_bytes);
            rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
                rect_start_row, K, nr, kr, sr, bl, scale_dt);
            imp_packed_rhs_size =
                kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, bl, scale_dt);
        }
    } else {
        rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(rect_start_row, rhs_stride_bytes);
        rhs_packed_offset =
            kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(rect_start_row, K, nr, kr, sr, bl, scale_dt);
        imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, bl, scale_dt);
    }

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    if (pack_type == RhsPackType::KxN) {
        kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params{};
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        params.scale_dt = scale_dt;

        if (use_ps1s0) {
            // clang-format off
            abi_check(
                kai_run_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon,
                1,                              // num_groups
                rect_width,                     // n
                K,                              // k
                nr, kr, sr, bl,                 // packing args
                reinterpret_cast<const uint8_t*>(rhs_qsu4.data() + rhs_offset),
                rhs_stride_bytes,
                reinterpret_cast<const float*>(biases.data() + bias_offset),
                reinterpret_cast<const void*>(rhs_scales.data() + scale_offset),
                scales_stride_bytes,
                static_cast<void*>(imp_packed_rhs.data() + rhs_packed_offset),
                0,
                &params);
            // clang-format on
        } else {
            kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params_kxn{};
            params_kxn.lhs_zero_point = 1;
            params_kxn.rhs_zero_point = 8;
            params_kxn.scale_dt = scale_dt;

            // clang-format off
            abi_check(
                kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0,
                1,
                rect_width,
                K,
                nr, kr, sr, bl,
                reinterpret_cast<const uint8_t*>(rhs_qsu4.data() + rhs_offset),
                rhs_stride_bytes,
                reinterpret_cast<const float*>(biases.data() + bias_offset),
                reinterpret_cast<const void*>(rhs_scales.data() + scale_offset),
                scales_stride_bytes,
                static_cast<void*>(imp_packed_rhs.data() + rhs_packed_offset),
                0,
                &params_kxn);
            // clang-format on
        }
    } else {
        kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{};
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        params.scale_dt = scale_dt;

        abi_check(
            // clang-format off
            kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
            1,
            rect_width,
            K,
            nr, kr, sr, bl,
            reinterpret_cast<const uint8_t*>(rhs_qsu4.data() + rhs_offset),
            rhs_stride_bytes,
            reinterpret_cast<const float*>(biases.data() + bias_offset),
            reinterpret_cast<const void*>(rhs_scales.data() + scale_offset),
            scales_stride_bytes,
            static_cast<void*>(imp_packed_rhs.data() + rhs_packed_offset),
            0,
            &params);
        // clang-format on
    }

    return {std::move(imp_packed_rhs), rhs_packed_offset};
}

/// Executes RHS NxK packing helper
std::tuple<Buffer, size_t> pack_rhs_qsi4c32p_nxk(
    const kai_qsi4c32p_pack_functions& pack_iface, const size_t N, const size_t K, const size_t nr, const size_t kr,
    const size_t sr, const size_t bl, const Buffer& rhs_values_qsi4, const float* bias, const Buffer& rhs_scales,
    const size_t rect_start_row, const size_t rect_width, const bool rhs_s0s1_input) {
    // Convert signed int4 -> unsigned int4, preserving any row padding in the source buffer.
    const auto rhs_qsu4s1s0 = cast_qsu4_qsi4(rhs_values_qsi4.data(), rhs_values_qsi4.size() * 2);

    const auto rhs_packed_size = pack_iface.packed_size(N, K, nr, kr, sr, bl, kai_dt_bf16);
    Buffer rhs_packed(rhs_packed_size);
    const auto rhs_packed_offset = pack_iface.get_packed_offset(rect_start_row, K, nr, kr, sr, bl, kai_dt_bf16);

    const size_t rhs_stride_bytes = round_up_division(K, 2);  // bytes per row
    const size_t scales_stride_bytes = round_up_division(K, bl) * kai_get_datatype_size_in_bytes(kai_dt_bf16);
    const size_t scale_offset = rect_start_row * scales_stride_bytes;
    const size_t rhs_offset = pack_iface.get_offset(rect_start_row, rhs_stride_bytes);

    kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = kai_dt_bf16;

    // Apply optional s0s1 -> s1s0 nibble swap.
    const Buffer* rhs_qsu4_ptr = &rhs_qsu4s1s0;
    Buffer rhs_qsu4_converted;
    if (rhs_s0s1_input) {
        rhs_qsu4_converted = convert_s0s1_s1s0(rhs_qsu4s1s0);
        rhs_qsu4_ptr = &rhs_qsu4_converted;
    }

    abi_check(
        pack_iface.run_pack, 1, rect_width, K, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(rhs_qsu4_ptr->data() + rhs_offset), rhs_stride_bytes, bias,
        rhs_scales.data() + scale_offset, scales_stride_bytes, rhs_packed.data() + rhs_packed_offset, 0, &params);

    return {std::move(rhs_packed), rhs_packed_offset};
}

// Executes F32-only RHS KxN packing helper (wrapper around BF16-scaled helper for clarity)
std::tuple<Buffer, size_t> pack_rhs_qsi4c32p_kxn(
    const size_t N, const size_t K, const size_t nr, const size_t kr, const size_t sr, const size_t bl,
    const Buffer& rhs_values_qsi4, const Buffer& biases, const size_t bias_offset, const Buffer& rhs_scales,
    const size_t rect_start_row, const size_t rect_width, const bool use_ps1s0) {
    return pack_rhs_qsi4c32pscalebf16(
        N, K, nr, kr, sr, bl, rhs_values_qsi4, biases, bias_offset, rhs_scales, RhsPackType::KxN, rect_start_row,
        rect_width, use_ps1s0);
}

/// Executes the vectorized RHS packing micro-kernels for block length of 4 bytes or 8 bytes
std::tuple<Buffer, size_t> pack_rhs_qsi4c32pscalebf16_neon(
    const size_t N, const size_t K, const size_t nr, const size_t kr, const size_t sr, const size_t bl,
    const Buffer& rhs_values_qsi4, const Buffer& biases, const size_t bias_offset, const Buffer& rhs_scales,
    const RhsPackType pack_type, const size_t rect_start_row, const size_t rect_width) {
    KAI_ASSUME_ALWAYS(kr / sr == 8 || kr / sr == 4);
    const size_t width = pack_type == RhsPackType::KxN ? N : K;
    const size_t height = pack_type == RhsPackType::KxN ? K : N;
    constexpr kai_datatype scale_dt = kai_dt_bf16;

    const size_t rhs_stride = round_up_multiple(width, 2);
    const size_t rhs_stride_bytes = round_up_division(width, 2);
    const size_t scales_stride_bytes = round_up_division(K, bl) * kai_get_datatype_size_in_bytes(scale_dt);

    KAI_ASSUME_ALWAYS(rhs_values_qsi4.size() == round_up_division(height * rhs_stride, 2));

    const auto rhs_values_qsu4 = cast_qsu4_qsi4(rhs_values_qsi4.data(), rhs_values_qsi4.size() * 2);
    const size_t dst_bytes_total = round_up_division(height * rhs_stride, 2);
    const size_t dst_bytes_total_safe = dst_bytes_total + rhs_stride_bytes + 8;
    const auto rhs_qsu4 =
        pad_row<UInt4>(rhs_values_qsu4.data(), height, width, width, rhs_stride_bytes * 2, dst_bytes_total_safe);

    const size_t scale_offset = rect_start_row * scales_stride_bytes;

    size_t imp_packed_rhs_size_neon = 0;
    size_t rhs_packed_offset_neon = 0;
    size_t rhs_offset_neon = 0;

    if (kr / sr == 8) {
        imp_packed_rhs_size_neon =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon(N, K, nr, kr, sr, bl, scale_dt);
        rhs_packed_offset_neon = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon(
            rect_start_row, K, nr, kr, sr, bl, scale_dt);
        rhs_offset_neon =
            kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon(rect_start_row, rhs_stride_bytes);
    } else {
        imp_packed_rhs_size_neon =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(N, K, nr, kr, sr, bl, scale_dt);
        rhs_packed_offset_neon = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
            rect_start_row, K, nr, kr, sr, bl, scale_dt);
        rhs_offset_neon =
            kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(rect_start_row, rhs_stride_bytes);
    }

    kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = scale_dt;

    Buffer imp_packed_rhs_neon(imp_packed_rhs_size_neon);
    if (kr / sr == 8) {
        kai_run_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon(
            1, rect_width /* n */, K, nr, kr, sr, bl,
            reinterpret_cast<const uint8_t*>(rhs_qsu4.data() + rhs_offset_neon), rhs_stride_bytes,
            reinterpret_cast<const float*>(biases.data() + bias_offset), rhs_scales.data() + scale_offset,
            scales_stride_bytes, imp_packed_rhs_neon.data() + rhs_packed_offset_neon, 0, &params);
    } else {
        kai_run_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
            1, rect_width /* n */, K, nr, kr, sr, bl,
            reinterpret_cast<const uint8_t*>(rhs_qsu4.data() + rhs_offset_neon), rhs_stride_bytes,
            reinterpret_cast<const float*>(biases.data() + bias_offset), rhs_scales.data() + scale_offset,
            scales_stride_bytes, imp_packed_rhs_neon.data() + rhs_packed_offset_neon, 0, &params);
    }
    return {std::move(imp_packed_rhs_neon), rhs_packed_offset_neon};
}

std::string test_description(
    const std::string& name, const RhsPackType rhs_pack_type, const MatMulShape& shape, const size_t bl,
    const MatrixPortion& portion) {
    // Remove redundant prefix to make output easier to read
    std::string clean_name = name;
    const std::string prefix = "kai_matmul_clamp_";
    if (clean_name.rfind(prefix, 0) == 0) {  // starts with prefix
        clean_name.erase(0, prefix.length());
    }

    std::ostringstream sstream;
    sstream << test_description(clean_name, shape, portion, /*bias=*/false) << "__BL_" << bl << "__"
            << ((rhs_pack_type == RhsPackType::NxK) ? "NxK" : "KxN");

    return sstream.str();
}

// Adds clamp_ratio suffix.
std::string test_description(
    const std::string& name, const RhsPackType rhs_pack_type, const MatMulShape& shape, const size_t bl,
    const MatrixPortion& portion, const float clamp_ratio) {
    std::ostringstream sstream;

    sstream << test_description(name, rhs_pack_type, shape, bl, portion)  //
            << "__clamp_ratio_" << static_cast<int>(clamp_ratio * 100);

    return sstream.str();
}

constexpr uint32_t seed = 0;  ///< Random seed used for tests

struct TestData {
    size_t M{}, N{}, K{}, bl{};

    Rect rect{0, 0, 0, 0};

    Buffer lhs;
    Buffer rhs;
    Buffer bias;

    Buffer rhs_quant;
    Buffer rhs_scales;

    Buffer lhs_packed;
    size_t lhs_packed_offset{};

    Buffer ref_dst_clamped;
    Range<float> clamp;
};

using BF16QMatMulRefKey = std::tuple<
    MatMulShape,                     // shape
    size_t,                          // bl
    size_t,                          // mr
    size_t,                          // nr
    size_t,                          // kr
    size_t,                          // sr
    size_t, size_t, size_t, size_t,  // rect.start_row, rect.start_col, rect.height, rect.width
    RhsPackType                      // rhs_pack_type
    >;

struct BF16TestData {
    size_t M{}, N{}, K{}, bl{};
    Rect rect{0, 0, 0, 0};

    Buffer lhs_bf16;    // Original BF16 LHS (kept for completeness)
    Buffer bias;        // Biases (FP32)
    Buffer rhs_quant;   // QSI4 quantized RHS (possibly transposed to match pack type)
    Buffer rhs_scales;  // BF16 per-block scales

    Buffer lhs_packed;           // Packed LHS buffer (BF16 dynamic quant + pack)
    size_t lhs_packed_offset{};  // Offset for rect.start_row

    Range<float> clamp{};  // Clamp range used for matmul
    Buffer ref_dst_bf16;   // Reference DST in BF16 (clamped)
};

}  // anonymous namespace

using QMatmulClampF32ParamT = std::tuple<size_t, bool, MatMulShape, size_t, MatrixPortion, RhsPackType, float>;

class QMatMulClampF32Test : public ::testing::TestWithParam<QMatmulClampF32ParamT> {
    struct TestParams {
        const UkernelMatmulPackVariant<
            kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel, kai_qai8dxp_pack_functions, kai_qsi4c32p_pack_functions>*
            variant;
        size_t variant_index;
        MatMulShape matmul_shape;
        size_t bl;
        MatrixPortion portion;
        RhsPackType rhs_pack_type;
        Rect rect;
        float clamp_ratio;
        bool is_sme2;

        TestParams() :
            variant(nullptr),
            variant_index(0),
            matmul_shape{0, 0, 0},
            bl(32),
            portion(0, 0, 1, 1),
            rhs_pack_type(RhsPackType::NxK),
            rect(0, 0, 0, 0),
            clamp_ratio(0.8F),
            is_sme2(false) {
        }

        TestParams(
            const UkernelMatmulPackVariant<
                kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel, kai_qai8dxp_pack_functions, kai_qsi4c32p_pack_functions>*
                variant,
            const size_t v_idx, const MatMulShape& shape, const size_t bl, const MatrixPortion p, const RhsPackType r,
            const Rect& rect, const float clamp_ratio) :
            variant(variant),
            variant_index(v_idx),
            matmul_shape(shape),
            bl(bl),
            portion(p),
            rhs_pack_type(r),
            rect(rect),
            clamp_ratio(clamp_ratio),
            is_sme2(false) {
        }
    };

    TestParams params;

protected:
    static const TestData& test_data();
    void SetupCommonForParam() {
        TestWithParam::SetUp();
        if (std::get<1>(GetParam())) {  // is_gemm
            SetupCommon(get_f32_gemm_variants());
        } else {
            SetupCommon(get_f32_gemv_variants());
        }
    }

    [[nodiscard]] const TestParams& GetParams() const {
        return params;
    }
    TestParams& GetParams() {
        return params;
    }

    void SetUp() override {
        // Gate CPU features before computing kernel interface params (which may touch unsupported instructions).
        const auto& param = GetParam();
        const size_t variant_index = std::get<0>(param);
        const bool is_gemm = std::get<1>(param);
        const auto& variant =
            is_gemm ? get_f32_gemm_variants().at(variant_index) : get_f32_gemv_variants().at(variant_index);

        if (variant.ukernel.fn_is_supported && !variant.ukernel.fn_is_supported()) {
            GTEST_SKIP() << "Unsupported CPU feature";
            return;
        }

        // Safe to compute aligned params/rect now.
        SetupCommonForParam();
        const auto& p = GetParams();

        // GEMV vs GEMM constraints (after params are set)
        if (!is_gemm) {
            if (p.matmul_shape.m != 1) {
                GTEST_SKIP() << "GEMV requires M=1";
                return;
            }
            if (p.rect.height() != 1 || p.rect.start_row() != 0) {
                GTEST_SKIP() << "GEMV portion invalid, rect height != 1 or start_row != 0";
                return;
            }
        }
    }

    template <size_t ArrN>
    void SetupCommon(
        const std::array<
            UkernelMatmulPackVariant<
                kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel, kai_qai8dxp_pack_functions, kai_qsi4c32p_pack_functions>,
            ArrN>& variants) {
        const auto& [variant_index, is_gemm, shape, bl, portion, rhs_dir, clamp_ratio] = GetParam();
        const auto& variant = variants.at(variant_index);

        params.variant = &variant;
        params.variant_index = variant_index;

        // Compute aligned portion rect once
        const size_t m_step = variant.ukernel.interface.get_m_step();
        const size_t n_step = variant.ukernel.interface.get_n_step();
        const Rect rect = portion.compute_portion(shape.m, shape.n, m_step, n_step);

        params.matmul_shape = shape;
        params.bl = bl;
        params.portion = portion;
        params.rhs_pack_type = rhs_dir;
        params.rect = rect;
        params.clamp_ratio = clamp_ratio;
        params.is_sme2 =
            (variant.ukernel.name.data() != nullptr && std::strstr(variant.ukernel.name.data(), "sme2") != nullptr);
    }
};

using F32QMatMulRefKey = std::tuple<
    MatMulShape,  // shape
    size_t,       // bl
    size_t,       // mr
    size_t,       // kr
    size_t,       // sr
    size_t,       // rect_start_row
    size_t,       // rect_start_col
    size_t,       // rect_height
    size_t,       // rect_width
    RhsPackType,  // rhs_pack_type
    int,          // clamp_pct
    const void*   // lhs_pack_key
    >;

template <>
TestData ReferenceGenerator<F32QMatMulRefKey, TestData>::generate_reference(const F32QMatMulRefKey& test_id) {
    TestData ref{};

    const auto& [shape, bl, mr, kr, sr, rect_start_row, rect_start_col, rect_height, rect_width, rhs_pack_type, clamp_pct, lhs_pack_key] =
        test_id;
    KAI_UNUSED(lhs_pack_key);
    const float clamp_ratio = static_cast<float>(clamp_pct) / 100.0F;
    const Rect rect(rect_start_row, rect_start_col, rect_height, rect_width);

    ref.M = shape.m;
    ref.N = shape.n;
    ref.K = shape.k;
    ref.bl = bl;
    ref.rect = rect;

    ref.lhs = fill_random<float>(ref.M * ref.K, seed + 0);
    ref.rhs = fill_random<float>(ref.N * ref.K, seed + 1);
    ref.bias = fill_random<float>(ref.N, seed + 2);

    // Dynamic LHS quantization (reference only).
    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = ref.K;
    lhs_qinfo.dst_type = DataType::QAI8;
    lhs_qinfo.scale_type = DataType::FP32;
    lhs_qinfo.zero_point_type = DataType::I32;
    auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref.lhs.data(), DataType::FP32, ref.M, ref.K, lhs_qinfo);

    // Dynamic RHS quantization to QSI4 with BF16 block scales.
    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = bl;
    rhs_qinfo.dst_type = DataType::QSI4;
    rhs_qinfo.scale_type = DataType::BF16;
    auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref.rhs.data(), DataType::FP32, ref.N, ref.K, rhs_qinfo);

    ref.rhs_quant = std::move(ref_rhs_quant);
    ref.rhs_scales = std::move(rhs_qoutputs.scales);

    const bool transposed = (rhs_pack_type == RhsPackType::NxK);
    const size_t width = transposed ? ref.K : ref.N;
    const size_t height = transposed ? ref.N : ref.K;

    const size_t qsi4_stride = round_up_multiple(width, 2);
    const size_t qsi4_size_bytes = round_up_division(height * qsi4_stride, 2);

    if (!transposed) {
        ref.rhs_quant =
            transpose_with_padding<Int4>(ref.rhs_quant.data(), ref.N, ref.K, ref.K, qsi4_stride, qsi4_size_bytes);
    }

    Buffer ref_dst_noclamp;
    if (transposed) {
        ref_dst_noclamp =
            matmul_nt_t_quantized<int8_t, float, int32_t, Int4, BFloat16<false>, int32_t, float, float, int32_t, float>(
                ref.M, ref.N, ref.K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(),
                1, ref.K, ref.rhs_quant.data(), ref.rhs_scales.data(), nullptr, 1, bl, ref.bias.data(), nullptr,
                nullptr, 1);
    } else {
        ref_dst_noclamp = matmul_nt_nt_quantized<
            int8_t, float, int32_t, Int4, BFloat16<false>, int32_t, float, float, int32_t, float>(
            ref.M, ref.N, ref.K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), 1,
            ref.K, ref.rhs_quant.data(), ref.rhs_scales.data(), nullptr, 1, bl, ref.bias.data(), nullptr, nullptr, 1);
    }

    const float retain = (clamp_ratio < 1.0F) ? (1.0F - clamp_ratio) : 1.0e-6F;
    const auto [cmin, cmax] = find_clamp_range<float>(ref_dst_noclamp.data(), ref.M * ref.N, retain);
    ref.clamp = {cmin, cmax};
    ref.ref_dst_clamped = clamp<float>(ref_dst_noclamp.data(), ref.M * ref.N, cmin, cmax);

    // Pack LHS once for this key.
    const size_t lhs_stride_bytes = ref.K * sizeof(float);
    constexpr kai_qai8dxp_pack_functions lhs_iface{
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
        kai_run_lhs_quant_pack_qai8dxp_f32,
    };

    auto [lhs_packed, lhs_packed_offset] = pack_lhs_qai8dxp(
        lhs_iface, ref.M, ref.K, mr, kr, sr, ref.lhs, lhs_stride_bytes, rect.start_row(), rect.height());

    ref.lhs_packed = std::move(lhs_packed);
    ref.lhs_packed_offset = lhs_packed_offset;

    return ref;
}

[[maybe_unused]] static void PrintTo(const QMatmulClampF32ParamT& param, std::ostream* os) {
    const auto& [variant_idx, is_gemm, shape, bl, portion, rhs_pack_type, clamp_ratio] = param;
    const auto name = std::string(
        (is_gemm ? get_f32_gemm_variants().at(variant_idx).ukernel.name
                 : get_f32_gemv_variants().at(variant_idx).ukernel.name));
    *os << test_description(name, rhs_pack_type, shape, bl, portion, clamp_ratio);
}

const TestData& QMatMulClampF32Test::test_data() {
    const auto& param = GetParam();
    const size_t variant_index = std::get<0>(param);
    const bool is_gemm = std::get<1>(param);
    const MatMulShape& shape = std::get<2>(param);
    const size_t bl = std::get<3>(param);
    const MatrixPortion& portion = std::get<4>(param);
    const RhsPackType rhs_pack_type = std::get<5>(param);
    const float clamp_ratio = std::get<6>(param);

    const auto& variant =
        is_gemm ? get_f32_gemm_variants().at(variant_index) : get_f32_gemv_variants().at(variant_index);
    const auto& iface = variant.ukernel.interface;

    const size_t mr = iface.get_mr();
    const size_t kr = iface.get_kr();
    const size_t sr = iface.get_sr();
    const size_t m_step = iface.get_m_step();
    const size_t n_step = iface.get_n_step();
    const Rect rect = portion.compute_portion(shape.m, shape.n, m_step, n_step);

    const int clamp_pct = static_cast<int>(clamp_ratio * 100 + 0.5F);

    const F32QMatMulRefKey key{
        shape,
        bl,
        mr,
        kr,
        sr,
        rect.start_row(),
        rect.start_col(),
        rect.height(),
        rect.width(),
        rhs_pack_type,
        clamp_pct,
        reinterpret_cast<const void*>(variant.lhs_pack_interface.run_pack)};

    return getV<F32QMatMulRefKey, TestData>(key);
}

using MatMulTestParams_withBL_withRHSPackType = std::tuple<size_t, MatMulShape, size_t, MatrixPortion, RhsPackType>;

[[maybe_unused]] static void PrintTo(const MatMulTestParams_withBL_withRHSPackType& param, std::ostream* os) {
    const size_t variant_idx = std::get<0>(param);
    const MatMulShape shape = std::get<1>(param);
    const size_t bl = std::get<2>(param);
    const MatrixPortion portion = std::get<3>(param);
    const RhsPackType rhs_pack_type = std::get<4>(param);
    const std::string name{get_bf16_gemm_variants().at(variant_idx).name};
    *os << test_description(name, rhs_pack_type, shape, bl, portion);
}

template <>
BF16TestData ReferenceGenerator<BF16QMatMulRefKey, BF16TestData>::generate_reference(const BF16QMatMulRefKey& test_id) {
    BF16TestData ref{};

    const MatMulShape shape = std::get<0>(test_id);
    const size_t bl = std::get<1>(test_id);
    const size_t mr = std::get<2>(test_id);
    const size_t nr = std::get<3>(test_id);
    KAI_UNUSED(nr);
    const size_t kr = std::get<4>(test_id);
    const size_t sr = std::get<5>(test_id);
    const size_t rect_start_row = std::get<6>(test_id);
    const size_t rect_start_col = std::get<7>(test_id);
    const size_t rect_height = std::get<8>(test_id);
    const size_t rect_width = std::get<9>(test_id);
    const RhsPackType rhs_pack_type = std::get<10>(test_id);

    ref.M = shape.m;
    ref.N = shape.n;
    ref.K = shape.k;
    ref.bl = bl;
    ref.rect = Rect(rect_start_row, rect_start_col, rect_height, rect_width);

    // Inputs
    ref.lhs_bf16 = fill_random<BFloat16<false>>(ref.M * ref.K, seed + 0);
    Buffer const ref_rhs = fill_random<float>(ref.N * ref.K, seed + 1);
    ref.bias = fill_random<float>(ref.N, seed + 2);

    // Cast BF16 LHS to FP32 for reference quantization
    const Buffer ref_lhs =
        cast<float, BFloat16<false>>(ref.lhs_bf16.data(), ref.lhs_bf16.size() * 8 / size_in_bits<BFloat16<false>>);

    // Reference quantizations for LHS and RHS
    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = ref.K;
    lhs_qinfo.dst_type = DataType::QAI8;
    lhs_qinfo.scale_type = DataType::FP32;
    lhs_qinfo.zero_point_type = DataType::I32;
    auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref_lhs.data(), DataType::FP32, ref.M, ref.K, lhs_qinfo);

    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = bl;
    rhs_qinfo.dst_type = DataType::QSI4;
    rhs_qinfo.scale_type = DataType::BF16;
    auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, ref.N, ref.K, rhs_qinfo);

    // Prepare RHS layout per pack type
    const bool transposed = (rhs_pack_type == RhsPackType::NxK);
    const size_t width = transposed ? ref.K : ref.N;
    const size_t height = transposed ? ref.N : ref.K;

    const size_t qsi4_stride = round_up_multiple(width, 2);
    const size_t qsi4_size_bytes = round_up_division(height * qsi4_stride, 2);

    ref.rhs_quant = std::move(ref_rhs_quant);
    if (!transposed) {
        ref.rhs_quant =
            transpose_with_padding<Int4>(ref.rhs_quant.data(), ref.N, ref.K, ref.K, qsi4_stride, qsi4_size_bytes);
    }
    ref.rhs_scales = std::move(rhs_qoutputs.scales);

    // Compute reference destination (float), clamp, and cast to BF16
    Buffer ref_dst_noclamp;
    if (transposed) {
        ref_dst_noclamp =
            matmul_nt_t_quantized<int8_t, float, int32_t, Int4, BFloat16<false>, int32_t, float, float, int32_t, float>(
                ref.M, ref.N, ref.K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(),
                1, ref.K, ref.rhs_quant.data(), ref.rhs_scales.data(), nullptr, 1, bl, ref.bias.data(), nullptr,
                nullptr, 1);
    } else {
        ref_dst_noclamp = matmul_nt_nt_quantized<
            int8_t, float, int32_t, Int4, BFloat16<false>, int32_t, float, float, int32_t, float>(
            ref.M, ref.N, ref.K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), 1,
            ref.K, ref.rhs_quant.data(), ref.rhs_scales.data(), nullptr, 1, bl, ref.bias.data(), nullptr, nullptr, 1);
    }

    constexpr auto clamp_ratio = 0.8F;
    const auto [clamp_min, clamp_max] = find_clamp_range<float>(ref_dst_noclamp.data(), ref.M * ref.N, clamp_ratio);
    ref.clamp = {clamp_min, clamp_max};
    const Buffer ref_dst_float = clamp<float>(ref_dst_noclamp.data(), ref.M * ref.N, clamp_min, clamp_max);
    ref.ref_dst_bf16 =
        cast<BFloat16<false>, float>(ref_dst_float.data(), ref_dst_float.size() * 8 / size_in_bits<float>);

    // Pack LHS once (BF16 packer)
    const size_t imp_packed_lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(ref.M, ref.K, mr, kr, sr);
    ref.lhs_packed = Buffer(imp_packed_lhs_size);

    const size_t lhs_stride = ref.K * sizeof(uint16_t);
    const size_t lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon(rect_start_row, lhs_stride);
    ref.lhs_packed_offset =
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(rect_start_row, ref.K, mr, kr, sr);

    kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
        rect_height, ref.K, mr, kr, sr, 0, ref.lhs_bf16.data() + lhs_offset, lhs_stride,
        reinterpret_cast<uint8_t*>(ref.lhs_packed.data()) + ref.lhs_packed_offset);

    return ref;
}

/// Verifies RHS packed offsets (KxN vs NxK) match each other and the matmul interface at n_step.
TEST_P(QMatMulClampF32Test, OffsetRHS) {
    const auto& p = GetParams();
    const auto fn_supported = p.variant->ukernel.fn_is_supported;
    if (fn_supported && !fn_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
        return;
    }

    const auto& ukernel = p.variant->ukernel;
    const size_t K = p.matmul_shape.k;
    const size_t bl = p.bl;
    const auto nr = ukernel.interface.get_nr();
    const auto kr = ukernel.interface.get_kr();
    const auto sr = ukernel.interface.get_sr();
    const auto n_step = ukernel.interface.get_n_step();

    const auto rhs_packed_offset_kxn =
        kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(n_step, K, nr, kr, sr, bl, kai_dt_bf16);
    const auto rhs_packed_offset_kxn_ps1s0 = kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
        n_step, K, nr, kr, sr, bl, kai_dt_bf16);
    const auto rhs_packed_offset_nxk =
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(n_step, K, nr, kr, sr, bl, kai_dt_bf16);
    const auto rhs_packed_offset_nxk_ps1s0_nrx4 =
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
            n_step, K, nr, kr, sr, bl, kai_dt_bf16);

    ASSERT_EQ(rhs_packed_offset_kxn, rhs_packed_offset_kxn_ps1s0);
    ASSERT_EQ(rhs_packed_offset_kxn_ps1s0, rhs_packed_offset_nxk);
    ASSERT_EQ(rhs_packed_offset_nxk, rhs_packed_offset_nxk_ps1s0_nrx4);

    const auto rhs_matmul_offset = ukernel.interface.get_rhs_packed_offset(n_step, K, bl);
    ASSERT_EQ(rhs_packed_offset_kxn, rhs_matmul_offset);
}

/// Verifies LHS packed offset matches the matmul interface at m_step.
TEST_P(QMatMulClampF32Test, OffsetLHS) {
    const auto& p = GetParams();
    const auto fn_supported = p.variant->ukernel.fn_is_supported;
    if (fn_supported && !fn_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
        return;
    }

    const auto& ukernel = p.variant->ukernel;
    const size_t K = p.matmul_shape.k;
    const auto mr = ukernel.interface.get_mr();
    const auto kr = ukernel.interface.get_kr();
    const auto sr = ukernel.interface.get_sr();
    const auto m_step = ukernel.interface.get_m_step();

    const auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(m_step, K, mr, kr, sr);
    const auto lhs_matmul_offset = ukernel.interface.get_lhs_packed_offset(m_step, K);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

/// Verifies the kernel’s get_dst_offset computes row/col addressing correctly at tile-aligned starts:
TEST_P(QMatMulClampF32Test, OffsetDst) {
    const auto& p = GetParams();
    const auto fn_supported = p.variant->ukernel.fn_is_supported;
    if (fn_supported && !fn_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
        return;
    }

    const auto& ukernel = p.variant->ukernel;
    const size_t M = p.matmul_shape.m;
    const size_t N = p.matmul_shape.n;

    const auto dst_stride_row = N * sizeof(float);
    constexpr auto dst_stride_col = sizeof(float);

    const auto m_step = ukernel.interface.get_m_step();
    const auto n_step = ukernel.interface.get_n_step();

    ASSERT_TRUE(m_step % ukernel.interface.get_mr() == 0);
    ASSERT_TRUE(n_step % ukernel.interface.get_nr() == 0);

    const size_t m_idx = (M > m_step) ? m_step : 0;
    const size_t n_idx = (N > n_step) ? n_step : 0;

    const auto off00 = ukernel.interface.get_dst_offset(0, 0, dst_stride_row);
    ASSERT_EQ(off00, 0U);

    const auto off10 = ukernel.interface.get_dst_offset(m_idx, 0, dst_stride_row);
    ASSERT_EQ(off10, m_idx * dst_stride_row);

    const auto off01 = ukernel.interface.get_dst_offset(0, n_idx, dst_stride_row);
    ASSERT_EQ(off01, n_idx * dst_stride_col);

    const auto off11 = ukernel.interface.get_dst_offset(m_idx, n_idx, dst_stride_row);
    ASSERT_EQ(off11, m_idx * dst_stride_row + n_idx * dst_stride_col);
}

/// Sanity-checks kernel interface parameters (mr/nr/kr/sr and step alignment).
TEST_P(QMatMulClampF32Test, KernelInvariants) {
    const auto& p = GetParams();
    const auto fn_supported = p.variant->ukernel.fn_is_supported;
    if (fn_supported && !fn_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
        return;
    }

    const auto& ukernel = p.variant->ukernel;
    const auto mr = ukernel.interface.get_mr();
    const auto nr = ukernel.interface.get_nr();
    const auto kr = ukernel.interface.get_kr();
    const auto sr = ukernel.interface.get_sr();
    const auto m_step = ukernel.interface.get_m_step();
    const auto n_step = ukernel.interface.get_n_step();

    ASSERT_GT(mr, 0U);
    ASSERT_GT(nr, 0U);
    ASSERT_GT(kr, 0U);
    ASSERT_GT(sr, 0U);

    ASSERT_EQ(m_step % mr, 0U);
    ASSERT_EQ(n_step % nr, 0U);
    ASSERT_EQ(kr % sr, 0U);
}

/// Verifies RHS row stride using difference of offsets equals the layout formula.
TEST_P(QMatMulClampF32Test, RhsStrideByDifference) {
    const auto& p = GetParams();
    const auto fn_supported = p.variant->ukernel.fn_is_supported;
    if (fn_supported && !fn_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
        return;
    }

    const auto& ukernel = p.variant->ukernel;
    const size_t K = p.matmul_shape.k;
    const size_t bl = p.bl;
    const auto nr = ukernel.interface.get_nr();
    const auto n_step = ukernel.interface.get_n_step();

    // Stride by difference using kernel offsets at 0 and n_step.
    const size_t off0 = ukernel.interface.get_rhs_packed_offset(0, K, bl);
    const size_t off1 = ukernel.interface.get_rhs_packed_offset(n_step, K, bl);
    const size_t stride_by_diff = off1 - off0;

    // Expected stride formula for qsi4c32p with BF16 scales:
    // nr * ( num_blocks * (bl/2 + 2) + 4 /*rsum*/ + 4 /*bias*/ )
    const size_t k_internal = round_up_multiple(K, 32);
    const size_t num_blocks = round_up_division(k_internal, bl);
    const size_t bytes_per_block = (bl / 2) + 2;  // int4 values + BF16 scale
    const size_t expected_stride = nr * (num_blocks * bytes_per_block) + nr * 4 + nr * 4;

    ASSERT_EQ(stride_by_diff, expected_stride);
}

/// Validation of the packed group slice against a reconstructed reference.
TEST_P(QMatMulClampF32Test, LhsPackBufferMatchesReference) {
    const auto& p = GetParams();
    const auto fn_supported = p.variant->ukernel.fn_is_supported;
    if (fn_supported && !fn_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
        return;
    }
    const auto& uk = p.variant->ukernel;

    const size_t M = p.matmul_shape.m;
    const size_t K = p.matmul_shape.k;
    const size_t mr = uk.interface.get_mr();
    const size_t kr = uk.interface.get_kr();
    const size_t sr = uk.interface.get_sr();

    const size_t k_block_len = kr / sr;
    const size_t k_internal = ((K + 31) / 32) * 32;

    const size_t i8_region_bytes = mr * k_internal;
    const size_t neg_zero_point_region_bytes = mr * sizeof(int32_t);
    const size_t recip_scale_region_bytes = mr * sizeof(float);
    const size_t group_stride = i8_region_bytes + neg_zero_point_region_bytes + recip_scale_region_bytes;

    constexpr size_t rect_start_row = 0;
    constexpr size_t rect_height = 1;

    const auto ref_lhs = fill_random<float>(M * K, seed);

    const size_t lhs_stride = K * sizeof(float);
    std::tuple<Buffer, size_t> pack_pair = pack_lhs_qai8dxp(
        p.variant->lhs_pack_interface, M, K, mr, kr, sr, ref_lhs, lhs_stride, rect_start_row, rect_height);

    Buffer const lhs_packed = std::move(std::get<0>(pack_pair));
    const size_t lhs_packed_off = std::get<1>(pack_pair);

    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = K;
    lhs_qinfo.dst_type = DataType::QAI8;
    lhs_qinfo.scale_type = DataType::FP32;
    lhs_qinfo.zero_point_type = DataType::I32;
    auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref_lhs.data(), DataType::FP32, M, K, lhs_qinfo);

    Buffer const expected(group_stride, 0);
    std::byte* expected_bytes = expected.data();

    // Build reference layout into `expected`
    constexpr size_t lane_row_idx = rect_start_row;
    const size_t lane = lane_row_idx % mr;
    const size_t ref_row_base = lane_row_idx * K;
    const auto pad_val = read_array<int8_t>(ref_lhs_quant.data(), ref_row_base + (K - 1));

    size_t ref_idx = 0;
    const size_t num_blocks_internal = k_internal / k_block_len;

    for (size_t b = 0; b < num_blocks_internal; ++b) {
        const size_t block_base = b * mr * k_block_len;
        const size_t lane_offset = block_base + lane * k_block_len;

        for (size_t i = 0; i < k_block_len; ++i) {
            const size_t dst_index = lane_offset + i;
            const bool in_range = ref_idx < K;

            const int8_t val = in_range ? read_array<int8_t>(ref_lhs_quant.data(), ref_row_base + ref_idx) : pad_val;

            write_array<int8_t>(expected_bytes, dst_index, val);

            if (in_range) {
                ++ref_idx;
            }
        }
    }

    // Header (per-lane): neg_zero_point, recip_scale
    const size_t neg_zero_point_elem_base = i8_region_bytes / sizeof(int32_t);
    const size_t recip_scale_elem_base = (i8_region_bytes + neg_zero_point_region_bytes) / sizeof(float);

    write_array<int32_t>(
        expected_bytes, neg_zero_point_elem_base + lane,
        -read_array<int32_t>(lhs_qoutputs.zero_points.data(), lane_row_idx));

    write_array<float>(
        expected_bytes, recip_scale_elem_base + lane, read_array<float>(lhs_qoutputs.scales.data(), lane_row_idx));

    // Validate packed buffer vs reference
    KAI_ASSUME_ALWAYS(lhs_packed_off + group_stride <= lhs_packed.size());

    // Int8 region: allow ±1 LSB
    for (size_t i = 0; i < i8_region_bytes; ++i) {
        const auto g = read_array<int8_t>(lhs_packed.data(), lhs_packed_off + i);
        const auto e = read_array<int8_t>(expected.data(), i);
        const int dq = static_cast<int>(g) - static_cast<int>(e);
        EXPECT_LE(std::abs(dq), 1) << "int8 mismatch at byte " << i << " (got=" << static_cast<int>(g)
                                   << ", exp=" << static_cast<int>(e) << ", dq=" << dq << ")";
    }

    // Region offsets (in bytes)
    const size_t neg_zero_point_offset = i8_region_bytes;
    const size_t recip_scale_offset = neg_zero_point_offset + neg_zero_point_region_bytes;

    // neg_zero_point (exact)
    for (size_t hdr_lane = 0; hdr_lane < mr; ++hdr_lane) {
        const auto gzp = read_array<int32_t>(
            lhs_packed.data(), lhs_packed_off / sizeof(int32_t) + (neg_zero_point_offset / sizeof(int32_t)) + hdr_lane);
        const auto ezp = read_array<int32_t>(expected.data(), (neg_zero_point_offset / sizeof(int32_t)) + hdr_lane);
        EXPECT_EQ(gzp, ezp) << "neg_zp mismatch at lane " << hdr_lane;
    }

    // recip_scale (near-equal)
    for (size_t hdr_lane = 0; hdr_lane < mr; ++hdr_lane) {
        const auto gsc = read_array<float>(
            lhs_packed.data(), lhs_packed_off / sizeof(float) + (recip_scale_offset / sizeof(float)) + hdr_lane);
        const auto esc = read_array<float>(expected.data(), (recip_scale_offset / sizeof(float)) + hdr_lane);
        EXPECT_NEAR(gsc, esc, 1e-5F) << "recip_scale mismatch at lane " << hdr_lane;
    }
}

TEST_P(QMatMulClampF32Test, EndToEnd) {
    const auto& p = GetParams();
    const auto fn_supported = p.variant->ukernel.fn_is_supported;
    if (fn_supported && !fn_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
        return;
    }
    const auto& ukernel = p.variant->ukernel;

    const size_t bl = p.bl;
    const RhsPackType rhs_pack_type = p.rhs_pack_type;

    KAI_ASSUME_ALWAYS(bl % 32 == 0);

    const auto nr = ukernel.interface.get_nr();
    const auto kr = ukernel.interface.get_kr();
    const auto sr = ukernel.interface.get_sr();

    const auto n_step = ukernel.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = p.rect;
    ASSERT_GT(rect.height(), 0U);
    ASSERT_GT(rect.width(), 0U);

    const auto& data = test_data();

    const auto rhs_start_col = rect.start_col();
    const size_t bias_offset_bytes = rhs_start_col * sizeof(float);

    Buffer imp_packed_rhs;
    size_t rhs_packed_offset = 0;
    if (rhs_pack_type == RhsPackType::NxK) {
        const float* bias_ptr = reinterpret_cast<const float*>(data.bias.data()) + rhs_start_col;
        std::tie(imp_packed_rhs, rhs_packed_offset) = pack_rhs_qsi4c32p_nxk(
            p.variant->rhs_pack_interface, data.N, data.K, nr, kr, sr, bl, data.rhs_quant, bias_ptr, data.rhs_scales,
            rhs_start_col, rect.width(), p.variant->rhs_s0s1_input);
    } else {
        if ((rhs_start_col % 2) != 0) {
            GTEST_SKIP() << "KxN RHS pack requires even N-start index";
            return;
        }
        std::tie(imp_packed_rhs, rhs_packed_offset) = pack_rhs_qsi4c32p_kxn(
            data.N, data.K, nr, kr, sr, bl, data.rhs_quant, data.bias, bias_offset_bytes, data.rhs_scales,
            rhs_start_col, rect.width(), p.is_sme2);
    }

    ASSERT_EQ(rhs_packed_offset, ukernel.interface.get_rhs_packed_offset(rhs_start_col, data.K, bl));

    // Destination buffer and offsets
    const auto dst_stride_row = data.N * sizeof(float);
    constexpr auto dst_stride_col = sizeof(float);
    const auto dst_offset = ukernel.interface.get_dst_offset(rect.start_row(), rhs_start_col, dst_stride_row);
    const auto imp_dst_size = ukernel.interface.get_dst_size(data.M, data.N);
    ASSERT_EQ(imp_dst_size, data.ref_dst_clamped.size());
    Buffer const imp_dst(imp_dst_size);

    // Run matmul
    abi_check(
        ukernel.interface.run_matmul, rect.height(), rect.width(), data.K, bl,
        data.lhs_packed.data() + data.lhs_packed_offset, imp_packed_rhs.data() + rhs_packed_offset,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col, data.clamp.min,
        data.clamp.max);

    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    const auto dst_format = DataFormat(DataType::FP32);
    const auto success =
        compare(imp_dst.data(), data.ref_dst_clamped.data(), dst_format, data.M, data.N, rect, handler);
    ASSERT_TRUE(success);
}

/// RHS vectorised packer format is s16s0 this is not relevant for sme2 kernels
class NeonRhsPackF32Test : public QMatMulClampF32Test {};

TEST_P(NeonRhsPackF32Test, EndToEndNeonRhsPack) {
    const auto& p = GetParams();
    const auto fn_supported = p.variant->ukernel.fn_is_supported;
    if (fn_supported && !fn_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
        return;
    }
    const auto& ukernel = p.variant->ukernel;

    const size_t mr = ukernel.interface.get_mr();
    const size_t nr = ukernel.interface.get_nr();
    const size_t kr = ukernel.interface.get_kr();
    const size_t sr = ukernel.interface.get_sr();
    ASSERT_EQ(ukernel.interface.get_m_step() % mr, 0U);
    ASSERT_EQ(ukernel.interface.get_n_step() % nr, 0U);

    if (p.rhs_pack_type != RhsPackType::NxK || (kr / sr != 8 && kr / sr != 4)) {
        GTEST_SKIP() << "RHS packers not applicable";
    }
    ASSERT_GT(p.rect.height(), 0U);
    ASSERT_GT(p.rect.width(), 0U);

    const auto& data = test_data();

    // LHS pack
    const size_t lhs_stride_bytes = data.K * sizeof(float);
    auto [imp_packed_lhs, lhs_packed_offset] = pack_lhs_qai8dxp(
        p.variant->lhs_pack_interface, data.M, data.K, mr, kr, sr, data.lhs, lhs_stride_bytes, p.rect.start_row(),
        p.rect.height());
    ASSERT_EQ(lhs_packed_offset, ukernel.interface.get_lhs_packed_offset(p.rect.start_row(), data.K));

    // RHS pack
    const size_t rhs_start_row = p.rect.start_col();
    const size_t bias_offset = rhs_start_row * sizeof(float);
    const auto [imp_packed_rhs_neon, rhs_packed_offset_neon] = pack_rhs_qsi4c32pscalebf16_neon(
        data.N, data.K, nr, kr, sr, p.bl, data.rhs_quant, data.bias, bias_offset, data.rhs_scales, p.rhs_pack_type,
        rhs_start_row, p.rect.width());

    ASSERT_EQ(rhs_packed_offset_neon, ukernel.interface.get_rhs_packed_offset(rhs_start_row, data.K, p.bl));

    const auto dst_stride_row = data.N * sizeof(float);
    Buffer const imp_dst(ukernel.interface.get_dst_size(data.M, data.N));
    const auto dst_offset = ukernel.interface.get_dst_offset(p.rect.start_row(), rhs_start_row, dst_stride_row);

    // Run matmul
    abi_check(
        ukernel.interface.run_matmul, p.rect.height(), p.rect.width(), data.K, p.bl,
        imp_packed_lhs.data() + lhs_packed_offset, imp_packed_rhs_neon.data() + rhs_packed_offset_neon,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, sizeof(float), data.clamp.min,
        data.clamp.max);

    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    const DataFormat dst_format(DataType::FP32);
    ASSERT_TRUE(compare(imp_dst.data(), data.ref_dst_clamped.data(), dst_format, data.M, data.N, p.rect, handler));
}

class QMatMulClampBF16Test : public ::testing::TestWithParam<MatMulTestParams_withBL_withRHSPackType> {};
TEST_P(QMatMulClampBF16Test, EndToEnd) {
    const auto& [variant_index, matmul_shape, bl, portion, rhs_pack_type] = GetParam();
    const auto& ukernel_variant = get_bf16_gemm_variants().at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    const auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    const auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    ASSERT_GT(rect.height(), 0U);
    ASSERT_GT(rect.width(), 0U);

    // Cached reference and inputs
    const BF16QMatMulRefKey key{
        matmul_shape,  bl,           mr,           nr, kr, sr, rect.start_row(), rect.start_col(),
        rect.height(), rect.width(), rhs_pack_type};
    const BF16TestData& data = getV<BF16QMatMulRefKey, BF16TestData>(key);

    // Verify LHS offsets match interface
    const auto lhs_start_row = rect.start_row();
    const auto lhs_packed_offset =
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(lhs_start_row, K, mr, kr, sr);
    const auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    // RHS: pack using cached quant/scales/bias
    const size_t rhs_start_row = rect.start_col();
    const size_t bias_offset = rhs_start_row * sizeof(float);
    if (rhs_pack_type == RhsPackType::KxN && (rhs_start_row % 2) != 0) {
        GTEST_SKIP() << "KxN RHS pack requires even N-start index";
        return;
    }

    auto [imp_packed_rhs, rhs_packed_offset] = pack_rhs_qsi4c32pscalebf16(
        N, K, nr, kr, sr, bl, data.rhs_quant, data.bias, bias_offset, data.rhs_scales, rhs_pack_type, rhs_start_row,
        rect.width(), false);

    const auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    // Destination
    const auto dst_stride_row = N * sizeof(uint16_t);
    constexpr auto dst_stride_col = sizeof(uint16_t);
    const auto dst_offset =
        ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, data.ref_dst_bf16.size());
    Buffer imp_dst(imp_dst_size);

    // Run matmul
    abi_check(
        ukernel_variant.interface.run_matmul, rect.height(), rect.width(), K, bl,
        reinterpret_cast<const uint8_t*>(data.lhs_packed.data()) + lhs_matmul_offset,
        reinterpret_cast<const uint8_t*>(imp_packed_rhs.data()) + rhs_matmul_offset, imp_dst.data() + dst_offset,
        dst_stride_row, dst_stride_col, data.clamp.min, data.clamp.max);

    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    auto dst_format = DataFormat(DataType::BF16);
    const auto success = compare(imp_dst.data(), data.ref_dst_bf16.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);

    // Test vectorized packing micro-kernels, if packing parameters allow
    if (rhs_pack_type == RhsPackType::NxK && (kr / sr == 8 || kr / sr == 4)) {
        const auto [imp_packed_rhs_neon, rhs_packed_offset_neon] = pack_rhs_qsi4c32pscalebf16_neon(
            N, K, nr, kr, sr, bl, data.rhs_quant, data.bias, bias_offset, data.rhs_scales, rhs_pack_type, rhs_start_row,
            rect.width());
        ASSERT_EQ(rhs_packed_offset_neon, rhs_packed_offset);
    }
}

// clang-format off

/// Portion categories (GEMM/GEMV)
static constexpr std::array gemm_portions{
    MatrixPortion(0,    0,    1, 1),      // Full matrix
    MatrixPortion(0.4,  0.5,  0.6, 0.8),  // Middle block
};
static constexpr std::array gemv_portions{
    MatrixPortion(0, 0,   1,   1),    // Full width
    MatrixPortion(0, 0.5, 1,   0.5),  // Right half
};

/// Shape categories (GEMM/GEMV)

/// Small/Odd edge coverage (odd m/n, varied K)
static constexpr std::array gemm_shapes_small_odd{
    MatMulShape{ 17,  25,  64},
    MatMulShape{ 31,  31,  64},
    MatMulShape{ 21,  53, 256},
    MatMulShape{ 35,  27, 320},
};

/// Aligned squares (cache-friendly, power-of-two-ish)
static constexpr std::array gemm_shapes_aligned{
    MatMulShape{ 32,  32, 128},
    MatMulShape{ 64,  64, 128},
    MatMulShape{128, 128, 256},
    MatMulShape{192, 192, 384},
};

/// Rectangular (skinny/wide), varied K
static constexpr std::array gemm_shapes_rect{
    MatMulShape{ 64, 128, 256},  // wide N
    MatMulShape{128,  64, 256},  // tall M
    MatMulShape{ 96, 192, 384},
    MatMulShape{160,  96, 320},
};

/// Larger/stress (within reason for CI)
static constexpr std::array gemm_shapes_large{
    MatMulShape{128, 160, 320},
    MatMulShape{160, 128, 320},
    MatMulShape{224, 160, 320},
    MatMulShape{160, 224, 320},
};

/// GEMV shape categories (F32)
/// M = 1, RHS NxK only in instantiation

/// Small/medium N, diverse K (aligned/odd N)
static constexpr std::array gemv_shapes_small{
    MatMulShape{  1,   16,  64},
    MatMulShape{  1,   31,  64},
    MatMulShape{  1,  128, 256},
    MatMulShape{  1,  256, 256},
    MatMulShape{  1,  320, 320},
};

/// Larger N bands (bandwidth/cache stress)
static constexpr std::array gemv_shapes_large{
    MatMulShape{  1,  512, 256},
    MatMulShape{  1,  640, 320},
    MatMulShape{  1,  768, 384},
    MatMulShape{  1, 1024, 256},
    MatMulShape{  1,  896, 384},
};

static constexpr std::array bf16_shapes {
    MatMulShape{ 32,  32,  64},   // small aligned
    MatMulShape{ 48,  64,  64},   // rectangular (tall K-block reuse)
    MatMulShape{ 64,  64, 128},   // aligned square
    MatMulShape{ 96,  96, 192},   // larger aligned
    MatMulShape{128,  64, 256},   // rectangular (tall M)
    MatMulShape{ 17,  25,  64},   // odd sizes (edge behavior)
    MatMulShape{ 33,  29, 192},   // odd sizes with larger K
    MatMulShape{128, 160, 320},   // larger rectangular
};

/// Dedicated clamp sweep ratios
static constexpr std::array<float, 3> clamp_ratios_sweep{
    0.0F,  // no clamp
    0.5F,  // clamp away 50%
    0.9F,  // clamp away 90%
};

INSTANTIATE_TEST_SUITE_P(
    MatMulGemm_SmallOdd, QMatMulClampF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_gemm_variants().size()),
        testing::Values(true),
        testing::ValuesIn(gemm_shapes_small_odd),
        testing::Values(32),
        testing::ValuesIn(gemm_portions),
        testing::Values(RhsPackType::NxK, RhsPackType::KxN),
        testing::Values(0.5F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulGemm_Aligned, QMatMulClampF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_gemm_variants().size()),
        testing::Values(true),
        testing::ValuesIn(gemm_shapes_aligned),
        testing::Values(32),
        testing::ValuesIn(gemm_portions),
        testing::Values(RhsPackType::NxK, RhsPackType::KxN),
        testing::Values(0.5F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulGemm_Rect, QMatMulClampF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_gemm_variants().size()),
        testing::Values(true),
        testing::ValuesIn(gemm_shapes_rect),
        testing::Values(32, 64),
        testing::ValuesIn(gemm_portions),
        testing::Values(RhsPackType::NxK, RhsPackType::KxN),
        testing::Values(0.5F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulGemm_Large, QMatMulClampF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_gemm_variants().size()),
        testing::Values(true),
        testing::ValuesIn(gemm_shapes_large),
        testing::Values(32),
        testing::ValuesIn(gemm_portions),
        testing::Values(RhsPackType::NxK, RhsPackType::KxN),
        testing::Values(0.5F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulGemv_Small, QMatMulClampF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_gemv_variants().size()),
        testing::Values(false),
        testing::ValuesIn(gemv_shapes_small),
        testing::Values(32),
        testing::ValuesIn(gemv_portions),
        testing::Values(RhsPackType::NxK),
        testing::Values(0.5F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulGemv_Large, QMatMulClampF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_gemv_variants().size()),
        testing::Values(false),
        testing::ValuesIn(gemv_shapes_large),
        testing::Values(32),
        testing::ValuesIn(gemv_portions),
        testing::Values(RhsPackType::NxK),
        testing::Values(0.5F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulNeonRhsPackGemm_SmallOdd, NeonRhsPackF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_neon_gemm_variants_only().size()),
        testing::Values(true),
        testing::ValuesIn(gemm_shapes_small_odd),
        testing::Values(32),
        testing::ValuesIn(gemm_portions),
        testing::Values(RhsPackType::NxK),
        testing::Values(0.5F)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulNeonRhsPackGemm_Aligned, NeonRhsPackF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_neon_gemm_variants_only().size()),
        testing::Values(true),
        testing::ValuesIn(gemm_shapes_aligned),
        testing::Values(32),
        testing::ValuesIn(gemm_portions),
        testing::Values(RhsPackType::NxK),
        testing::Values(0.5F)),
    testing::PrintToStringParamName());

static constexpr std::array clamp_sweep_shapes{
    MatMulShape{ 64, 64, 128 },
    MatMulShape{ 64, 128, 256 },
};

INSTANTIATE_TEST_SUITE_P(
    MatMulGemm_ClampSweep, QMatMulClampF32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_f32_gemm_variants().size()),
        testing::Values(true),
        testing::ValuesIn(clamp_sweep_shapes),
        testing::Values(32),
        testing::Values(MatrixPortion(0, 0, 1, 1)),
        testing::Values(RhsPackType::NxK, RhsPackType::KxN),
        testing::ValuesIn(clamp_ratios_sweep)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulBF16_SingleSet, QMatMulClampBF16Test,
    testing::Combine(
        testing::Range<size_t>(0, get_bf16_gemm_variants().size()),
        testing::ValuesIn(bf16_shapes),
        testing::Values(32),
        testing::Values(MatrixPortion(0, 0, 1, 1)),
        testing::Values(RhsPackType::NxK, RhsPackType::KxN)),
    testing::PrintToStringParamName());

}  // namespace kai::test
