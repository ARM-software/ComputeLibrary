//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai::test {

enum {
    MATMUL_SLOT_CONFIG,       ///< Matrix multiplication operator configuration.
    MATMUL_SLOT_PACK_ARGS,    ///< Packing arguments.
    MATMUL_SLOT_MATMUL_ARGS,  ///< Matrix multiplication micro-kernel parameters.

    MATMUL_SLOT_LHS_RAW,         ///< LHS data in F32.
    MATMUL_SLOT_LHS_DATA,        ///< LHS data after conversion.
    MATMUL_SLOT_LHS_QDATA,       ///< LHS data after quantization.
    MATMUL_SLOT_LHS_QSCALE,      ///< LHS quantization scale.
    MATMUL_SLOT_LHS_QZP,         ///< LHS quantization zero-point.
    MATMUL_SLOT_LHS_QZP_NEG,     ///< Negative LHS quantization zero-point.
    MATMUL_SLOT_REF_LHS_PACKED,  ///< Reference packed LHS.
    MATMUL_SLOT_IMP_LHS_PACKED,  ///< Packed LHS from micro-kernel.

    MATMUL_SLOT_RHS_RAW,               ///< RHS data in F32.
    MATMUL_SLOT_RHS_T_RAW,             ///< Transposed RHS data in F32.
    MATMUL_SLOT_RHS_T_DATA,            ///< Transposed RHS data after conversion.
    MATMUL_SLOT_RHS_T_QDATA,           ///< Transposed RHS data after quantization.
    MATMUL_SLOT_RHS_T_QDATA_SIGN,      ///< Transposed RHS data after quantization with opposite signedness.
    MATMUL_SLOT_RHS_T_QDATA_SIGN_SUM,  ///< Row sum of transposed RHS after quantization with opposite signedness.
    MATMUL_SLOT_RHS_T_QSCALE,          ///< Transposed RHS quantization scale.
    MATMUL_SLOT_RHS_T_QZP,             ///< Transposed RHS quantization zero-point.
    MATMUL_SLOT_REF_RHS_PACKED,        ///< Reference packed RHS.
    MATMUL_SLOT_IMP_RHS_PACKED,        ///< Packed RHS from micro-kernel.

    MATMUL_SLOT_BIAS_RAW,
    MATMUL_SLOT_BIAS_DATA,
    MATMUL_SLOT_BIAS_SCALE,
    MATMUL_SLOT_BIAS_ZP,
    MATMUL_SLOT_BIAS_PACKED,

    MATMUL_SLOT_DST_RAW,
    MATMUL_SLOT_REF_DST_DATA,
    MATMUL_SLOT_DST_SCALE,
    MATMUL_SLOT_DST_ZP,

    MATMUL_SLOT_IMP_DST_DATA,

    NUM_MATMUL_SLOTS,
};

}  // namespace kai::test
