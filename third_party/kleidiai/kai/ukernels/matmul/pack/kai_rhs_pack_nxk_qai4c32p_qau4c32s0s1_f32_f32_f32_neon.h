//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>

#include "kai/kai_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef kai_rhs_pack_nxk_qai4c32p_params
#define kai_rhs_pack_nxk_qai4c32p_params kai_rhs_pack_qs4cxs1s0_param
#endif

/// Gets the offset in bytes for the RHS matrix (not packed), which holds
/// the int4 values in a N x K matrix, where N is number of rows and K is the number of columns.
/// Two int4 K values are stored in one byte. These values are stored in blocks
///
/// @param[in] n_idx      Row index in the RHS matrix (not packed).
/// @param[in] rhs_stride The number of bytes in in each row of the RHS matrix (not packed)
///
/// @return the offset in bytes to the RHS matrix (not packed)
size_t kai_get_rhs_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
    size_t n_idx,        //
    size_t rhs_stride);  //

/// Gets the offset in bytes for the packed RHS matrix.
///
/// @param[in] n_idx    Row index in the RHS matrix (not packed).
/// @param[in] k        The common dimension between the LHS and RHS matrix (K)
/// @param[in] nr       The number of columns written by the matmul micro-kernel
/// @param[in] kr       The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] bl       The block length, which defines the number of K values stored in a single block. It must be a
/// multiple of 32.
///
/// @return the offset in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
    size_t n_idx,  //
    size_t k,      //
    size_t nr,     //
    size_t kr,     //
    size_t bl      //
);

/// Gets the size in bytes for the quantized and packed RHS matrix.
///
/// @param[in] n  The number of rows in the RHS matrix (not packed)
/// @param[in] k  The number of columns in the RHS matrix (not packed).
/// @param[in] nr The number of columns written by the matmul micro-kernel
/// @param[in] kr The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] bl The block length, which defines the number of K values stored in a single block. It must be a multiple
/// of 32.
///
/// @return the packed RHS matrix size in bytes
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
    size_t n,   //
    size_t k,   //
    size_t nr,  //
    size_t kr,  //
    size_t bl   //
);

/// Run the micro-kernel to pack the RHS matrix.
///
/// @note   The int4 values are stored in a N x K matrix, where N is number of rows and K is the number of columns.
///         Two int4 values are stored in one byte.
///
/// @param[in]  num_groups  The number of groups. It must be 1.
/// @param[in]  n           The number of columns of the output matrix (N).
/// @param[in]  k           The common dimension between the LHS and RHS matrix (K).
/// @param[in]  nr          The number of N rows to interleave on the same output row.
/// @param[in]  kr          The number of K values loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in]  sr          The number of kr splits. It can be 1 (no splits) up to kr.
///                         However, kr must be multiple of sr.
/// @param[in]  bl          The block length, which defines the number of
///                         K values stored in a single block. It must be a multiple of 32.
/// @param[in]  rhs         The RHS matrix containing the 4-bit values.
///                         Size in bytes is expected to be greater than or equal to n * k * (sizeof(uint8_t) / 2).
/// @param[in]  zero        The zero point.
/// @param[in]  bias        The biases.
/// @param[in]  scale       The scale for each output channel.
/// @param[out] rhs_packed  The packed RHS matrix.
/// @param[in]  extra_bytes Extra bytes to append to the end of each row of the packed RHS matrix.
/// @param[in]  params      Parameters for the micro-kernel.
void kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
    size_t num_groups,   //
    size_t n,            //
    size_t k,            //
    size_t nr,           //
    size_t kr,           //
    size_t sr,           //
    size_t bl,           //
    const uint8_t* rhs,  //
    const void* zero,    //
    const void* bias,    //
    const void* scale,   //
    void* rhs_packed,    //
    size_t extra_bytes,  //
    const struct kai_rhs_pack_nxk_qai4c32p_params* params);
#ifdef __cplusplus
}
#endif
