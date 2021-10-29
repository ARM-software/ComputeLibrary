/*
 * Copyright (c) 2021 Arm Limited.
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
#include "helpers.h"

/** (EXPERIMENTAL_POST_OPS) Macros for (binary) elementwise operations */

/** List of (binary) elementwise operators, accounting for the argument position of argument X
 * @note X_Pos denotes the position of argument X. e.g. X_POS_0 means X is in the first place whereas X_POS_1 means X is in the second place
 * @name elementwise_post_ops
 * @{
 */
#if defined(N0) && !defined(VEC_SIZE)
#define VEC_SIZE N0
#endif // defined(N0) && !defined(VEC_SIZE)

#if defined(VEC_SIZE) && defined(DATA_TYPE)

#define ADD_X_POS_0(x, y) (x) + (y)
#define SUB_X_POS_0(x, y) (x) - (y)
#define MAX_X_POS_0(x, y) max(x, y)
#define MIN_X_POS_0(x, y) min(x, y)
#define SQUARED_DIFF_X_POS_0(x, y) (x - y) * (x - y)
#define POWER_X_POS_0(x, y) pow(x, y)
#if VEC_SIZE == 1
#define PRELU_X_POS_0(x, y) (x > 0 ? x : x * y)
#else // VEC_SIZE == 1

#if defined(MIXED_PRECISION)
#define PRELU_X_POS_0(x, y) (select(y * x, x, CONVERT((x > (DATA_TYPE_ACCUMULATOR)0), SELECT_VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, VEC_SIZE))))
#else // MIXED_PRECISION
#define PRELU_X_POS_0(x, y) (select(y * x, x, CONVERT((x > (DATA_TYPE)0), SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))))
#endif // MIXED_PRECISION

#endif // VEC_SIZE == 1
#define DIV_X_POS_0(x, y) (x / y)
#define AND_X_POS_0(x, y) (CONVERT((x && y), VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)) & ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))1))
#define OR_X_POS_0(x, y) (CONVERT((x || y), VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)) & ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))1))

#define ADD_X_POS_1(x, y) ADD_X_POS_0(x, y)
#define SUB_X_POS_1(x, y) (y) - (x)
#define MAX_X_POS_1(x, y) MAX_X_POS_0(x, y)
#define MIN_X_POS_1(x, y) MIN_X_POS_0(x, y)
#define SQUARED_DIFF_X_POS_1(x, y) SQUARED_DIFF_X_POS_0(x, y)
#define POWER_X_POS_1(x, y) pow(y, x)
#if VEC_SIZE == 1
#define PRELU_X_POS_1(x, y) (y > 0 ? y : y * x)
#else // VEC_SIZE == 1

#if defined(MIXED_PRECISION)
#define PRELU_X_POS_1(x, y) (select(x * y, y, CONVERT((y > (DATA_TYPE_ACCUMULATOR)0), SELECT_VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, VEC_SIZE))))
#else // MIXED_PRECISION
#define PRELU_X_POS_1(x, y) (select(x * y, y, CONVERT((y > (DATA_TYPE)0), SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))))
#endif // MIXED_PRECISION

#endif // VEC_SIZE == 1
#define DIV_X_POS_1(x, y) (y / x)
#define AND_X_POS_1(x, y) AND_X_POS_0(x, y)
#define OR_X_POS_1(x, y) OR_X_POS_0(x, y)

// By default use the order of the arguments as they are passed in, ie. _X_POS_0
#define ADD(x, y) ADD_X_POS_0(x, y)
#define SUB(x, y) SUB_X_POS_0(x, y)
#define MAX(x, y) MAX_X_POS_0(x, y)
#define MIN(x, y) MIN_X_POS_0(x, y)
#define SQUARED_DIFF(x, y) SQUARED_DIFF_X_POS_0(x, y)
#define POWER(x, y) POWER_X_POS_0(x, y)
#define PRELU(x, y) PRELU_X_POS_0(x, y)
#define DIV(x, y) DIV_X_POS_0(x, y)
#define AND(x, y) AND_X_POS_0(x, y)
#define OR(x, y) OR_X_POS_0(x, y)

#endif    // defined(VEC_SIZE) && defined(DATA_TYPE)
/** @} */ // end of group elementwise_post_ops

/** Performs OPERAND1 = OP(OPERAND1, OPERAND2)
 * @name ELTWISE_OP_ROW_n
 *
 * @param[in]      OP       The elementwise post op
 * @param[in, out] OPERAND1 The basename of the destination and operand 1 variables
 * @param[in]      OPERAND2 The basename of the operand 2 variables
 * @{
 */
#define ELTWISE_OP_ROW_1(OP, OPERAND1, OPERAND2) \
    OPERAND1##0 = OP(OPERAND1##0, OPERAND2##0);

#define ELTWISE_OP_ROW_2(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_1(OP, OPERAND1, OPERAND2)     \
    OPERAND1##1 = OP(OPERAND1##1, OPERAND2##1);

#define ELTWISE_OP_ROW_3(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_2(OP, OPERAND1, OPERAND2)     \
    OPERAND1##2 = OP(OPERAND1##2, OPERAND2##2);

#define ELTWISE_OP_ROW_4(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_3(OP, OPERAND1, OPERAND2)     \
    OPERAND1##3 = OP(OPERAND1##3, OPERAND2##3);

#define ELTWISE_OP_ROW_5(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_4(OP, OPERAND1, OPERAND2)     \
    OPERAND1##4 = OP(OPERAND1##4, OPERAND2##4);

#define ELTWISE_OP_ROW_6(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_5(OP, OPERAND1, OPERAND2)     \
    OPERAND1##5 = OP(OPERAND1##5, OPERAND2##5);

#define ELTWISE_OP_ROW_7(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_6(OP, OPERAND1, OPERAND2)     \
    OPERAND1##6 = OP(OPERAND1##6, OPERAND2##6);

#define ELTWISE_OP_ROW_8(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_7(OP, OPERAND1, OPERAND2)     \
    OPERAND1##7 = OP(OPERAND1##7, OPERAND2##7);

#define ELTWISE_OP_ROW_9(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_8(OP, OPERAND1, OPERAND2)     \
    OPERAND1##8 = OP(OPERAND1##8, OPERAND2##8);

#define ELTWISE_OP_ROW_10(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_9(OP, OPERAND1, OPERAND2)      \
    OPERAND1##9 = OP(OPERAND1##9, OPERAND2##9);

#define ELTWISE_OP_ROW_11(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_10(OP, OPERAND1, OPERAND2)     \
    OPERAND1##A = OP(OPERAND1##A, OPERAND2##A);

#define ELTWISE_OP_ROW_12(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_11(OP, OPERAND1, OPERAND2)     \
    OPERAND1##B = OP(OPERAND1##B, OPERAND2##B);

#define ELTWISE_OP_ROW_13(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_12(OP, OPERAND1, OPERAND2)     \
    OPERAND1##C = OP(OPERAND1##C, OPERAND2##C);

#define ELTWISE_OP_ROW_14(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_13(OP, OPERAND1, OPERAND2)     \
    OPERAND1##D = OP(OPERAND1##D, OPERAND2##D);

#define ELTWISE_OP_ROW_15(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_14(OP, OPERAND1, OPERAND2)     \
    OPERAND1##E = OP(OPERAND1##E, OPERAND2##E);

#define ELTWISE_OP_ROW_16(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_15(OP, OPERAND1, OPERAND2)     \
    OPERAND1##F = OP(OPERAND1##F, OPERAND2##F);

/** @} */ // end of group ELTWISE_OP_ROW_n

/** Performs OPERAND1 = OP(OPERAND1, OPERAND2)
 * @name ELTWISE_OP_BLOCK
 *
 * Supported cases are N=1,2,3,...,16
 *
 * @param[in] OP       The elementwise post op
 * @param[in] N        The number of vectors in the block
 * @param[in] OPERAND1 The basename of the destination and operand 1 variables
 * @param[in] OPERAND2 The basename of the operand 2 variables
 * @{
 */
#define ELTWISE_OP_BLOCK_STR(OP, N, OPERAND1, OPERAND2) ELTWISE_OP_ROW_##N(OP, OPERAND1, OPERAND2)
#define ELTWISE_OP_BLOCK(OP, N, OPERAND1, OPERAND2) ELTWISE_OP_BLOCK_STR(OP, N, OPERAND1, OPERAND2)
/** @} */ // end of group ELTWISE_OP_BLOCK

/** Performs OPERAND1 = OP(OPERAND1, OPERAND2) with broadcasting
 * @name ELTWISE_OP_ROW_BROADCAST_n
 *
 * @param[in]      OP       The elementwise post op
 * @param[in, out] OPERAND1 The basename of the destination and operand 1 variables
 * @param[in]      OPERAND2 The basename of the broadcast operand 2 variables
 * @{
 */
#define ELTWISE_OP_ROW_BROADCAST_1(OP, OPERAND1, OPERAND2) \
    OPERAND1##0 = OP(OPERAND1##0, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_2(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_1(OP, OPERAND1, OPERAND2)     \
    OPERAND1##1 = OP(OPERAND1##1, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_3(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_2(OP, OPERAND1, OPERAND2)     \
    OPERAND1##2 = OP(OPERAND1##2, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_4(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_3(OP, OPERAND1, OPERAND2)     \
    OPERAND1##3 = OP(OPERAND1##3, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_5(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_4(OP, OPERAND1, OPERAND2)     \
    OPERAND1##4 = OP(OPERAND1##4, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_6(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_5(OP, OPERAND1, OPERAND2)     \
    OPERAND1##5 = OP(OPERAND1##5, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_7(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_6(OP, OPERAND1, OPERAND2)     \
    OPERAND1##6 = OP(OPERAND1##6, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_8(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_7(OP, OPERAND1, OPERAND2)     \
    OPERAND1##7 = OP(OPERAND1##7, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_9(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_8(OP, OPERAND1, OPERAND2)     \
    OPERAND1##8 = OP(OPERAND1##8, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_10(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_9(OP, OPERAND1, OPERAND2)      \
    OPERAND1##9 = OP(OPERAND1##9, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_11(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_10(OP, OPERAND1, OPERAND2)     \
    OPERAND1##A = OP(OPERAND1##A, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_12(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_11(OP, OPERAND1, OPERAND2)     \
    OPERAND1##B = OP(OPERAND1##B, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_13(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_12(OP, OPERAND1, OPERAND2)     \
    OPERAND1##C = OP(OPERAND1##C, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_14(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_13(OP, OPERAND1, OPERAND2)     \
    OPERAND1##D = OP(OPERAND1##D, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_15(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_14(OP, OPERAND1, OPERAND2)     \
    OPERAND1##E = OP(OPERAND1##E, OPERAND2);

#define ELTWISE_OP_ROW_BROADCAST_16(OP, OPERAND1, OPERAND2) \
    ELTWISE_OP_ROW_BROADCAST_15(OP, OPERAND1, OPERAND2)     \
    OPERAND1##F = OP(OPERAND1##F, OPERAND2);

/** @} */ // end of group ELTWISE_OP_ROW_BROADCAST_n

/** Performs OPERAND1 = OP(OPERAND1, OPERAND2) with broadcasting
 * @name ELTWISE_OP_BLOCK_BROADCAST
 * @note Only support:
 *      case 1 broadcast in Y dimension : Operand1 [YxX] + Operand2 [1xX];
 *      case 2 broadcast in both Y and X dimensions : Operand1 [YxX] + Operand2 [1x1] (scalar);
 *      Does NOT support broad cast in X dimension: Operand1 [YxX] + Operand2 [Yx1];
 *
 * Supported cases are N=1,2,3,...,16
 *
 * @param[in] OP       The elementwise post op
 * @param[in] N        The number of vectors in the block
 * @param[in] OPERAND1 The basename of the destination and operand 1 variables
 * @param[in] OPERAND2 The basename of the operand 2 variables
 * @{
 */
#define ELTWISE_OP_BLOCK_BROADCAST_STR(OP, N, OPERAND1, OPERAND2) ELTWISE_OP_ROW_BROADCAST_##N(OP, OPERAND1, OPERAND2)
#define ELTWISE_OP_BLOCK_BROADCAST(OP, N, OPERAND1, OPERAND2) ELTWISE_OP_BLOCK_BROADCAST_STR(OP, N, OPERAND1, OPERAND2)
/** @} */ // end of group ELTWISE_OP_BLOCK_BROADCAST