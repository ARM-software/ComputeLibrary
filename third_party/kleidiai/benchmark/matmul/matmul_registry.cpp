//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <test/common/cpu_info.hpp>
#include <test/common/data_type.hpp>

#include "matmul_benchmark_logic.hpp"
#include "matmul_interface.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

// Micro-kernels to register for benchmarking

// matmul_clamp_f16_bf16p_bf16p
#include "kai/ukernels/matmul/matmul_clamp_f16_bf16p_bf16p/kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"

// matmul_clamp_f16_f16_f16p
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55.h"

// matmul_clamp_f16_f16p_f16p
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"

// matmul_clamp_f32_bf16p_bf16p
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"

// matmul_clamp_f32_f32_f32p
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"

// matmul_clamp_f32_f32p_f32p
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"

// matmul_clamp_f32_qai8dxp_qsi4c32p
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

// matmul_clamp_f32_qai8dxp_qsi4cxp
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

// matmul_clamp_f32_qai8dxp_qsi8cxp
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"

// matmul_clamp_f32_qsi8d32p_qsi4c32p
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"

// matmul_clamp_fp32_bf16p_bf16p
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"

// matmul_clamp_qai8_qai8_qsi8cxp
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot.h"

// matmul_clamp_qai8_qai8p_qsi8cxp
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"

// matmul_clamp_f16_qai8dxp_qsi4cxp
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm.h"

// matmul_clamp_f16_qai8dxp_qsi8cxp
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"

// matmul_clamp_f16_qsi8d32p_qai4c32p
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"

// matmul_clamp_f32_qsi8d32p_qai4c32p
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"

// matmul_clamp_bf16_qai8dxp_qsi4c32p
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm.h"

// matmul_clamp_bf16_qai8dxp_qsi4cxp
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm.h"

namespace kai::benchmark {
using DataType = test::DataType;

// matmul_clamp_f16_bf16p_bf16p
inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla_interface{
    .run_matmul = kai_run_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
};

// matmul_clamp_f16_f16_f16p
inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot_interface{
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55_interface{
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55,
};

// matmul_clamp_f16_f16p_f16p
inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
};

// matmul_clamp_f32_bf16p_bf16p
inline constexpr MatMulBaseInterface kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot_interface{
    .run_matmul = kai_run_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla_interface{
    .run_matmul = kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
};

// matmul_clamp_f32_f32_f32p
inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
};

inline constexpr MatMulStridedLhsInterface kai_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla,
};

// matmul_clamp_f32_f32p_f32p
inline constexpr MatMulBaseInterface kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
};

// matmul_clamp_f32_qai8dxp_qsi4c32p
inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod,
    };
inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
    };

// matmul_clamp_f32_qai8dxp_qsi4cxp
inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
};

// matmul_clamp_f32_qai8dxp_qsi8cxp
inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
};

inline constexpr MatMulFloatInterface kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_interface{
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
};

// matmul_clamp_f32_qsi8d32p_qsi4c32p
inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
    };
inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
    };
inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
    };

// matmul_clamp_fp32_bf16p_bf16p
inline constexpr MatMulBaseInterface kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
};

// matmul_clamp_qai8_qai8_qsi8cxp
inline constexpr MatMulStaticQuantInterface kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot_interface{
    .run_matmul = kai_run_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot,
};

// matmul_clamp_qai8_qai8p_qsi8cxp
inline constexpr MatMulStaticQuantInterface kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa,
};

inline constexpr MatMulStaticQuantInterface kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_interface{
    .run_matmul = kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa,
};

// matmul_clamp_bf16_qai8dxp_qsi4c32p
inline constexpr MatMulBlockwiseDynamicQuantGenericDstInterface
    kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantGenericDstInterface
    kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm,
    };

// matmul_clamp_bf16_qai8dxp_qsi4cxp
inline constexpr MatMulBaseInterface kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm_interface{
    .run_matmul = kai_run_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
};

// matmul_clamp_f16_qai8dxp_qsi4cxp
inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm_interface{
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
};

// matmul_clamp_f16_qai8dxp_qsi8cxp
inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod_interface{
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
};

inline constexpr MatMulBaseInterface kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_interface{
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
};

// matmul_clamp_f16_qsi8d32p_qai4c32p
inline constexpr MatMulBlockwiseDynamicQuantGenericDstInterface
    kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa_interface{
        .run_matmul = kai_run_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa,
    };

inline constexpr MatMulBlockwiseDynamicQuantGenericDstInterface
    kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot_interface{
        .run_matmul = kai_run_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot,
    };

inline constexpr MatMulBlockwiseDynamicQuantGenericDstInterface
    kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantGenericDstInterface
    kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantGenericDstInterface
    kai_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantGenericDstInterface
    kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
    };

// matmul_clamp_f32_qsi8d32p_qai4c32p
inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
    };

inline constexpr MatMulBlockwiseDynamicQuantInterface
    kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_interface{
        .run_matmul = kai_run_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
    };

inline const std::array matmul_benchmarks{
    // matmul_clamp_f16_bf16p_bf16p
    RegisterBenchmark(
        "kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_bf16),

    // matmul_clamp_f16_f16_f16p
    RegisterBenchmark(
        "kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla_interface, DataType::FP16, MatMulOp::GEMV, test::cpu_has_sme),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot_interface, DataType::FP16, MatMulOp::GEMV,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_interface, DataType::FP16, MatMulOp::GEMM, test::cpu_has_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_fp16),

    // matmul_clamp_f16_f16p_f16p
    RegisterBenchmark(
        "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_sme),

    // matmul_clamp_f32_bf16p_bf16p
    RegisterBenchmark(
        "kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),

    // matmul_clamp_f32_f32_f32p
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla_interface, DataType::FP32, MatMulOp::GEMV, test::cpu_has_sme),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_advsimd),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_advsimd),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_advsimd),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla", kai_benchmark_matmul<MatMulStridedLhsInterface>,
        kai_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla_interface, DataType::FP32, MatMulOp::GEMM, test::cpu_has_sve),

    // matmul_clamp_f32_f32p_f32p
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme),

    // matmul_clamp_f32_qai8dxp_qsi4c32p
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme2),

    // matmul_clamp_f32_qai8dxp_qsi4cxp
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),

    // matmul_clamp_f32_qai8dxp_qsi8cxp
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_sme),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot", kai_benchmark_matmul<MatMulFloatInterface>,
        kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_sme2),

    // matmul_clamp_f32_qsi8d32p_qsi4c32p
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        (test::cpu_check<test::cpu_has_sve_vl256, test::cpu_has_dotprod>)),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        (test::cpu_check<test::cpu_has_sve_vl256, test::cpu_has_dotprod>)),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        (test::cpu_check<test::cpu_has_sve_vl256, test::cpu_has_i8mm>)),

    // matmul_clamp_fp32_bf16p_bf16p
    RegisterBenchmark(
        "kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme2),

    // matmul_clamp_qai8_qai8_qsi8cxp
    RegisterBenchmark(
        "kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot", kai_benchmark_matmul<MatMulStaticQuantInterface>,
        kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot_interface, DataType::QAI8, MatMulOp::GEMV,
        test::cpu_has_sme2),

    // matmul_clamp_qai8_qai8p_qsi8cxp
    RegisterBenchmark(
        "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa",
        kai_benchmark_matmul<MatMulStaticQuantInterface>,
        kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_interface, DataType::QAI8, MatMulOp::GEMM,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa",
        kai_benchmark_matmul<MatMulStaticQuantInterface>,
        kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_interface, DataType::QAI8, MatMulOp::GEMM,
        test::cpu_has_sme),

    // matmul_clamp_bf16_qai8dxp_qsi4c32p
    RegisterBenchmark(
        "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantGenericDstInterface>,
        kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod_interface, DataType::BF16, MatMulOp::GEMV,
        test::cpu_has_dotprod_and_bf16),
    RegisterBenchmark(
        "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantGenericDstInterface>,
        kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm_interface, DataType::BF16, MatMulOp::GEMM,
        test::cpu_has_i8mm_and_bf16),

    // matmul_clamp_bf16_qai8dxp_qsi4cxp
    RegisterBenchmark(
        "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod_interface, DataType::BF16, MatMulOp::GEMV,
        test::cpu_has_dotprod_and_bf16),
    RegisterBenchmark(
        "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm_interface, DataType::BF16, MatMulOp::GEMM,
        test::cpu_has_i8mm_and_bf16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMV,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMV,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_i8mm_and_fp16),

    // matmul_clamp_f16_qai8dxp_qsi8cxp
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMV,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMV,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm", kai_benchmark_matmul<MatMulBaseInterface>,
        kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_i8mm_and_fp16),

    // matmul_clamp_f16_qsi8d32p_qai4c32p
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantGenericDstInterface>,
        kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantGenericDstInterface>,
        kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot_interface, DataType::FP16, MatMulOp::GEMV,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantGenericDstInterface>,
        kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMV,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantGenericDstInterface>,
        kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMV,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantGenericDstInterface>,
        kai_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_dotprod_and_fp16),
    RegisterBenchmark(
        "kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantGenericDstInterface>,
        kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_interface, DataType::FP16, MatMulOp::GEMM,
        test::cpu_has_i8mm_and_fp16),

    // matmul_clamp_f32_qsi8d32p_qai4c32p
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_sme2),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMV,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_dotprod),
    RegisterBenchmark(
        "kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm",
        kai_benchmark_matmul<MatMulBlockwiseDynamicQuantInterface>,
        kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_interface, DataType::FP32, MatMulOp::GEMM,
        test::cpu_has_i8mm),

};

void RegisterMatMulBenchmarks(const MatMulShape& shape, const size_t bl) {
    for (const auto& benchmark : matmul_benchmarks) {
        benchmark
            ->Args(
                {static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n), static_cast<int64_t>(shape.k),
                 static_cast<int64_t>(bl)})
            ->ArgNames({"m", "n", "k", "bl"});
    }
}
}  // namespace kai::benchmark
