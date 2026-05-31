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

/// Get the n step value.
/// The micro-kernel can process any N values. However, the starting N index to
/// be processed must be a multiple of n step.
///
/// @param[in] nr The number of columns written by the matmul micro-kernel
///
/// @return the n step value
size_t kai_get_n_step_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(size_t nr);

/// Gets the offset in bytes for the RHS matrix (not packed), which holds
/// the int4 values in a N x K matrix, where N is number of rows and K is the number of columns.
///
/// Two int4 values are stored in one byte.
///        The lower order part of the byte (low) holds the first nibble (K-index + 0).
///        The higher order of the byte holds the second nibble (K-index + 1).
///
/// @param[in] n_idx      Row index in the RHS matrix (not packed).
/// @param[in] rhs_stride The number of bytes in in each row of the RHS matrix (not packed)
///
/// @return the offset in bytes to the RHS matrix (not packed)
size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
    size_t n_idx,        //
    size_t rhs_stride);  //

/// Get the row stride in bytes to the packed RHS matrix
///
/// @param[in] k        The number of columns in the RHS matrix (not packed).
/// @param[in] nr       The number of columns written by the matmul micro-kernel.
/// @param[in] kr       The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr       The number of kr splits. It can be 1 (no splits) up to kr.
/// @param[in] bl       The block length, which defines the number of K values stored in a single block. It must be a
/// multiple of 32.
/// @param[in] scale_dt Block scale data type
///
/// @return the stride in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
    size_t k,                     //
    size_t nr,                    //
    size_t kr,                    //
    size_t sr,                    //
    size_t bl,                    //
    enum kai_datatype scale_dt);  //

/// Gets the offset in bytes for the packed RHS matrix.
///
/// @param[in] n_idx    Row index in the RHS matrix (not packed).
/// @param[in] k        The number of columns in the RHS matrix (not packed).
/// @param[in] nr       The number of columns written by the matmul micro-kernel
/// @param[in] kr       The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr       The number of kr splits. It can be 1 (no splits) up to kr.
/// @param[in] bl       The block length, which defines the number of K values stored in a single block. It must be a
/// multiple of 32.
/// @param[in] scale_dt Block scale data type
///
/// @return the offset in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
    size_t n_idx,                 //
    size_t k,                     //
    size_t nr,                    //
    size_t kr,                    //
    size_t sr,                    //
    size_t bl,                    //
    enum kai_datatype scale_dt);  //

/// Gets the size in bytes for the quantized and packed RHS matrix.
///
/// @param[in] n  The number of rows in the RHS matrix (not packed)
/// @param[in] k  The number of columns in the RHS matrix (not packed).
/// @param[in] nr The number of columns written by the matmul micro-kernel
/// @param[in] kr The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr The number of kr splits. It can be 1 (no splits) up to kr.
/// @param[in] bl The block length, which defines the number of K values stored in a single block. It must be a multiple
/// of 32.
/// @param[in] scale_dt Block scale data type
///
/// @return the packed RHS matrix size in bytes
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
    size_t n,                     //
    size_t k,                     //
    size_t nr,                    //
    size_t kr,                    //
    size_t sr,                    //
    size_t bl,                    //
    enum kai_datatype scale_dt);  //

/// Runs the RHS packing micro-kernel.
///
/// The int4 values are stored in a N x K matrix, where N is number of rows and K is the number of columns.
/// Two int4 values are stored in one byte. The lower order part of the byte (low) holds
/// the first nibble (K-index + 0). The higher order of the byte holds the second nibble (K-index + 1).
///
/// @param[in]  num_groups   The number of groups. It must be 1.
/// @param[in]  n            The number of rows in the RHS matrix (not packed).
/// @param[in]  k            The number of columns in the RHS matrix (not packed).
/// @param[in]  nr           The number of columns written by the matmul micro-kernel. It must be a multiple of 4.
/// @param[in]  kr           The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in]  sr           The number of kr splits. It can be 1 (no splits) up to kr.
///                          However, kr must be multiple of sr.
/// @param[in]  bl           The block length, which defines the number of
///                          K values stored in a single block. It must be a multiple of 32.
/// @param[in]  rhs          The RHS matrix containing the 4-bit values.
///                          Size in bytes is expected to be greater than or equal to n * k * (sizeof(uint8_t) / 2).
/// @param[in]  rhs_stride   The number of bytes per row in bytes of the RHS matrix
/// @param[in]  bias         The biases.
/// @param[in]  scale        The per-block quantization scales.
///                          The scale data type must be provided with the params object.
///                          Supported scale data types are FP32, FP16 and BF16.
/// @param[in]  scale_stride The number of bytes per row in bytes of the scale matrix
/// @param[out] rhs_packed   The packed RHS matrix.
/// @param[in]  extra_bytes  Extra bytes to append to the end of each row of the packed RHS matrix.
/// @param[in]  params       Parameters for the micro-kernel.
void kai_run_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
    size_t num_groups,                                                   //
    size_t n,                                                            //
    size_t k,                                                            //
    size_t nr,                                                           //
    size_t kr,                                                           //
    size_t sr,                                                           //
    size_t bl,                                                           //
    const uint8_t* rhs,                                                  //
    size_t rhs_stride,                                                   //
    const float* bias,                                                   //
    const void* scale,                                                   //
    size_t scale_stride,                                                 //
    void* rhs_packed,                                                    //
    size_t extra_bytes,                                                  //
    const struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params* params);  //

#ifdef __cplusplus
}
#endif
