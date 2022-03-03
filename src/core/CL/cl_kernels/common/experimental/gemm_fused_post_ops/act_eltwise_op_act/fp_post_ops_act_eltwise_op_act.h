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
#include "common/experimental/gemm_fused_post_ops/fp_mixed_precision_helpers.h"

/** (EXPERIMENTAL_POST_OPS) Post Op expansions for the post op sequence:
 * act (optional): POST_OP1_ACTIVATION_OPTIONAL
 * eltwise_op   : POST_OP2_ELTWISE_OP
 * act (optional): POST_OP3_ACTIVATION_OPTIONAL
 */

/** Post Op 1: Activation Block (Optional)
 * @name POST_OP1_ACTIVATION_OPTIONAL
 * Toggled by -DP1_ACTIVATION_TYPE
 * params: same as those in @ref MIXED_PRECISION_ACTIVATION_BLOCK
 * @{
 */
#if defined(P1_ACTIVATION_TYPE) && defined(P1_ACTIVATION_A_VAL) && defined(P1_ACTIVATION_B_VAL)
#define POST_OP1_ACTIVATION_OPTIONAL(N, DATA_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, BASENAME) \
    MIXED_PRECISION_ACTIVATION_BLOCK(N, P1_ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, P1_ACTIVATION_A_VAL, P1_ACTIVATION_B_VAL, DATA_TYPE_ACCUMULATOR);
#else                                                                                         // defined(P1_ACTIVATION_TYPE) && defined(P1_ACTIVATION_A_VAL) && defined(P1_ACTIVATION_B_VAL)
#define POST_OP1_ACTIVATION_OPTIONAL(N, DATA_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, BASENAME) // noop
#endif                                                                                        // defined(P1_ACTIVATION_TYPE) && defined(P1_ACTIVATION_A_VAL) && defined(P1_ACTIVATION_B_VAL)
/** @} */                                                                                     // end of group POST_OP1_ACTIVATION_OPTIONAL

/** Post Op 2: Eltwise Op Block
 * Handles both broadcasting and non-broadcasting cases
 * @name POST_OP2_ELTWISE_OP
 *
 * @param[in] P2_ELTWISE_ARG1_HEIGHT Height (number of rows) of the @ref ELTWISE_OPERAND_NAME tensor
 * @param[in] P2_ELTWISE_ARG1_WIDTH  Width (number of columns) of the @ref ELTWISE_OPERAND_NAME tensor
 * @param[in] OP                     The elementwise post op
 * @param[in] M0                     The number of consecutive rows
 * @param[in] N0                     The number of consecutive columns
 * @param[in] BASENAME               The basename of the result variables
 * @param[in] ELTWISE_OPERAND_NAME   The basename of the other operand variables
 * @param[in] ELTWISE_OPERAND_ROW    The starting row of the other operand variables. Required as different boundary handling strategies are used by different kernels
 *                                   E.g. reshaped_only_rhs and native kernels shifts rows (by using COMPUTE_M0_START_ROW) to handle boundary rows,
 *                                   whereas reshaped kernels do not shift rows
 * @param[in] DATA_TYPE              Data type of the result variables
 * @param[in] DATA_TYPE_ACCUMULATR   Higher-precision accumulator data type in case of mixed-precision op
 * @param[in] ZERO                   Zero vector for z offset
 * @param[in] PARTIAL_LOAD_M0        The partial size in y, for partial blocks. Supported: [0, @p M0)
 * @param[in] PARTIAL_LOAD_N0        The partial size in x, for partial blocks. Supported: [0, @p N0)
 * @param[in] PARTIAL_COND_Y         Condition on the y axis to perform the partial load Y. True to use PARTIAL_LOAD_M0 rather than M0.
 * @param[in] PARTIAL_COND_X         Condition on the x axis to perform the partial load X. True to use PARTIAL_LOAD_N0 rather than N0.
 * @{
 */
#if defined(P2_ELTWISE_ARG1_HEIGHT) && defined(P2_ELTWISE_ARG1_WIDTH)
#if P2_ELTWISE_ARG1_HEIGHT == 1
#if P2_ELTWISE_ARG1_WIDTH == 1 // Case 1: Broadcasting in both X and Y; op2 arg tile shape[YxX] == [1x1]
#define POST_OP2_ELTWISE_OP(OP, M0, N0, BASENAME, ELTWISE_OPERAND_NAME, ELTWISE_OPERAND_ROW, DATA_TYPE, DATA_TYPE_ACCUMULATOR, ZERO, PARTIAL_LOAD_M0, PARTIAL_LOAD_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    __global uchar *ELTWISE_OPERAND_NAME##_addr = ELTWISE_OPERAND_NAME##_ptr + ELTWISE_OPERAND_NAME##_offset_first_element_in_bytes + get_global_id(2) * ELTWISE_OPERAND_NAME##_stride_z;              \
    VEC_DATA_TYPE(DATA_TYPE, 1)                                                                                                                                                                        \
    ELTWISE_OPERAND_NAME##0 = VLOAD(1)(0, (__global DATA_TYPE *)ELTWISE_OPERAND_NAME##_addr);                                                                                                          \
    MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST(OP, M0, 1, BASENAME, ELTWISE_OPERAND_NAME, DATA_TYPE_ACCUMULATOR, ELTWISE_OPERAND_NAME##_hp);
#else // P2_ELTWISE_ARG1_WIDTH == 1; Case 2: Broadcasting in only Y; op2 arg tile shape[YxX] == [1xN0]
#define POST_OP2_ELTWISE_OP(OP, M0, N0, BASENAME, ELTWISE_OPERAND_NAME, ELTWISE_OPERAND_ROW, DATA_TYPE, DATA_TYPE_ACCUMULATOR, ZERO, PARTIAL_LOAD_M0, PARTIAL_LOAD_N0, PARTIAL_COND_Y, PARTIAL_COND_X)                                        \
    __global uchar *ELTWISE_OPERAND_NAME##_addr = ELTWISE_OPERAND_NAME##_ptr + ELTWISE_OPERAND_NAME##_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + get_global_id(2) * ELTWISE_OPERAND_NAME##_stride_z; \
    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, ELTWISE_OPERAND_NAME, ELTWISE_OPERAND_NAME##_addr, 0, ELTWISE_OPERAND_NAME##_stride_y, ZERO, 1, PARTIAL_LOAD_N0, false, PARTIAL_COND_X);                                                      \
    MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST(OP, M0, N0, BASENAME, ELTWISE_OPERAND_NAME, DATA_TYPE_ACCUMULATOR, ELTWISE_OPERAND_NAME##_hp);
#endif // P2_ELTWISE_ARG1_WIDTH == 1
#else  // P2_ELTWISE_ARG1_HEIGHT == 1; Case 3: No broadcasting; op2 arg tile shape[YxX] == [M0xN0]
#define POST_OP2_ELTWISE_OP(OP, M0, N0, BASENAME, ELTWISE_OPERAND_NAME, ELTWISE_OPERAND_ROW, DATA_TYPE, DATA_TYPE_ACCUMULATOR, ZERO, PARTIAL_LOAD_M0, PARTIAL_LOAD_N0, PARTIAL_COND_Y, PARTIAL_COND_X)                                                                                                  \
    __global uchar *ELTWISE_OPERAND_NAME##_addr = ELTWISE_OPERAND_NAME##_ptr + ELTWISE_OPERAND_NAME##_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (ELTWISE_OPERAND_ROW * ELTWISE_OPERAND_NAME##_stride_y) + get_global_id(2) * ELTWISE_OPERAND_NAME##_stride_z; \
    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, ELTWISE_OPERAND_NAME, ELTWISE_OPERAND_NAME##_addr, 0, ELTWISE_OPERAND_NAME##_stride_y, ZERO, PARTIAL_LOAD_M0, PARTIAL_LOAD_N0, PARTIAL_COND_Y, PARTIAL_COND_X);                                                                                        \
    MIXED_PRECISION_ELTWISE_OP_BLOCK(OP, M0, N0, BASENAME, ELTWISE_OPERAND_NAME, DATA_TYPE_ACCUMULATOR, ELTWISE_OPERAND_NAME##_hp);
#endif    // P2_ELTWISE_ARG1_HEIGHT == 1
#endif    // defined(P2_ELTWISE_ARG1_HEIGHT) && defined(P2_ELTWISE_ARG1_WIDTH)
/** @} */ // end of group POST_OP2_ELTWISE_OP
/** Post Op 3: Activation Block (Optional)
 * @name POST_OP3_ACTIVATION_OPTIONAL
 * Toggled by -DP3_ACTIVATION_TYPE
 * params: same as those in @ref MIXED_PRECISION_ACTIVATION_BLOCK
 * @{
 */
#if defined(P3_ACTIVATION_TYPE) && defined(P3_ACTIVATION_A_VAL) && defined(P3_ACTIVATION_B_VAL)
#define POST_OP3_ACTIVATION_OPTIONAL(N, DATA_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, BASENAME) \
    MIXED_PRECISION_ACTIVATION_BLOCK(N, P3_ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, P3_ACTIVATION_A_VAL, P3_ACTIVATION_B_VAL, DATA_TYPE_ACCUMULATOR);
#else                                                                                         // defined(P3_ACTIVATION_TYPE) && defined(P3_ACTIVATION_A_VAL) && defined(P3_ACTIVATION_B_VAL)
#define POST_OP3_ACTIVATION_OPTIONAL(N, DATA_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, BASENAME) // noop
#endif                                                                                        // defined(P3_ACTIVATION_TYPE) && defined(P3_ACTIVATION_A_VAL) && defined(P3_ACTIVATION_B_VAL)
/** @} */                                                                                     // end of group POST_OP3_ACTIVATION_OPTIONAL
