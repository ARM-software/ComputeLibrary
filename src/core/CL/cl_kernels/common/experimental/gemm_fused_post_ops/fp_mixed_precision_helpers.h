/*
 * Copyright (c) 2021-2022 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "common/experimental/gemm_fused_post_ops/fp_elementwise_op_helpers.h"
#include "gemm_helpers.h"
#include "load_store_utility.h"

/** (EXPERIMENTAL_POST_OPS) Convenience macros for automatically handling mixed precision (fp16 and fp32) operations
 * -DMIXED_PRECISION toggles mixed precision mode
 */

/** Mixed-Precision-Aware Activation Block
 * @name MIXED_PRECISION_ACTIVATION_BLOCK
 * params N ... B_VAL: same as those in @ref ACTIVATION_BLOCK
 *
 * @param[in] DATA_TYPE_ACCUMULATR Higher-precision accumulator data type in case of mixed-precision op
 * @{
 */
#if defined(MIXED_PRECISION)
#define MIXED_PRECISION_ACTIVATION_BLOCK(N, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL, DATA_TYPE_ACCUMULATOR) \
    ACTIVATION_BLOCK(N, ACTIVATION_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, BASENAME, A_VAL, B_VAL);
#else // defined(MIXED_PRECISION)
#define MIXED_PRECISION_ACTIVATION_BLOCK(N, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL, DATA_TYPE_ACCUMULATOR) \
    ACTIVATION_BLOCK(N, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL);
#endif    // defined(MIXED_PRECISION)
/** @} */ // end of group MIXED_PRECISION_ACTIVATION_BLOCK

/** Mixed-Precision-Aware Elementwise Op Block
 * Performs OPERAND1 = OP(OPERAND1, OPERAND2)
 * @name MIXED_PRECISION_ELTWISE_OP_BLOCK
 *
 * @param[in] OP                   The elementwise post op
 * @param[in] M0                   The number of consecutive rows
 * @param[in] N0                   The number of consecutive columns
 * @param[in] OPERAND1             The basename of the first and result operand variables
 * @param[in] OPERAND2             The basename of the second operand variables
 * @param[in] DATA_TYPE_ACCUMULATR Higher-precision accumulator data type in case of mixed-precision op
 * @param[in] CONVERTED_OPERAND2   The basename of the second operand variables converted to higher-precision in case of mixed-precision op
 * @{
 */
#if defined(MIXED_PRECISION)
#define MIXED_PRECISION_ELTWISE_OP_BLOCK(OP, M0, N0, OPERAND1, OPERAND2, DATA_TYPE_ACCUMULATOR, CONVERTED_OPERAND2) \
    CONVERT_BLOCK(M0, N0, DATA_TYPE_ACCUMULATOR, OPERAND2, CONVERTED_OPERAND2);                                     \
    ELTWISE_OP_BLOCK(OP, M0, OPERAND1, CONVERTED_OPERAND2);
#else // defined(MIXED_PRECISION)
#define MIXED_PRECISION_ELTWISE_OP_BLOCK(OP, M0, N0, OPERAND1, OPERAND2, DATA_TYPE_ACCUMULATOR, CONVERTED_OPERAND2) \
    ELTWISE_OP_BLOCK(OP, M0, OPERAND1, OPERAND2);
#endif    // defined(MIXED_PRECISION)
/** @} */ // end of group MIXED_PRECISION_ELTWISE_OP_BLOCK

/** Mixed-Precision-Aware Elementwise Op Broadcast Block
 * Performs OPERAND1 = OP(OPERAND1, OPERAND2)
 * @name MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST
 * @note Only support:
 *      case 1 broadcast in Y dimension : Operand1 [YxX] + Operand2 [1xX]; this means @p N0 > 1
 *      case 2 broadcast in both Y and X dimensions : Operand1 [YxX] + Operand2 [1x1] (scalar) ; this means @p N0 == 1
 *      Does NOT support broad cast in X dimension: Operand1 [YxX] + Operand2 [Yx1]; this means @p M0 should never == 1
 *
 * @param[in] OP                   The elementwise post op
 * @param[in] M0                   The number of consecutive rows, > 1
 * @param[in] N0                   The number of consecutive columns, >= 1
 * @param[in] OPERAND1             The basename of the first and result operand variables
 * @param[in] OPERAND2             The basename of the second operand variables
 * @param[in] DATA_TYPE_ACCUMULATR Higher-precision accumulator data type in case of mixed-precision op
 * @param[in] CONVERTED_OPERAND2   The basename of the second operand variables converted to higher-precision in case of mixed-precision op
 * @{
 */
#if defined(MIXED_PRECISION)
#define MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST(OP, M0, N0, OPERAND1, OPERAND2, DATA_TYPE_ACCUMULATOR, CONVERTED_OPERAND2) \
    CONVERT_BLOCK(1, N0, DATA_TYPE_ACCUMULATOR, OPERAND2, CONVERTED_OPERAND2);                                                \
    ELTWISE_OP_BLOCK_BROADCAST(OP, M0, OPERAND1, CONVERTED_OPERAND2##0);
#else // defined(MIXED_PRECISION)
#define MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST(OP, M0, N0, OPERAND1, OPERAND2, DATA_TYPE_ACCUMULATOR, CONVERTED_OPERAND2) \
    ELTWISE_OP_BLOCK_BROADCAST(OP, M0, OPERAND1, OPERAND2##0);
#endif    // defined(MIXED_PRECISION)
/** @} */ // end of group MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST

/** Mixed-Precision-Aware Boundary-Aware Store Block
 * @name MIXED_PRECISION_STORE_BLOCK_BOUNDARY_AWARE
 * params M0 ... PARTIAL_COND_X, same as those in STORE_BLOCK_BOUNDARY_AWARE
 *
 * @param[in] BASENAME_LP The name of the low precision variables, converted from BASENAME, in case of mixed-precision op
 * @{
 */
#if defined(MIXED_PRECISION)
#define MIXED_PRECISION_STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X, BASENAME_LP) \
    CONVERT_BLOCK(M0, N0, DATA_TYPE, BASENAME, BASENAME_LP);                                                                                                                       \
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME_LP, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X);
#else // defined(MIXED_PRECISION)
#define MIXED_PRECISION_STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X, BASENAME_LP) \
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X);
#endif    // defined(MIXED_PRECISION)
/** @} */ // end of group MIXED_PRECISION_STORE_BLOCK_BOUNDARY_AWARE