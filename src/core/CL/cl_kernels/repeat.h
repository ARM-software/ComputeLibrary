/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_REPEAT_H
#define ARM_COMPUTE_REPEAT_H

/** Macros that help in loop unrolling */
//Repeat macros with 3 param, excluding the implicit ID param
#define REPEAT_3_1(P_X, P_A, P_B, P_C) P_X##_DEF(0, P_A, P_B, P_C)
#define REPEAT_3_2(P_X, P_A, P_B, P_C) \
    P_X##_DEF(1, P_A, P_B, P_C);       \
    REPEAT_3_1(P_X, P_A, P_B, P_C)
#define REPEAT_3_3(P_X, P_A, P_B, P_C) \
    P_X##_DEF(2, P_A, P_B, P_C);       \
    REPEAT_3_2(P_X, P_A, P_B, P_C)
#define REPEAT_3_4(P_X, P_A, P_B, P_C) \
    P_X##_DEF(3, P_A, P_B, P_C);       \
    REPEAT_3_3(P_X, P_A, P_B, P_C)
#define REPEAT_3_5(P_X, P_A, P_B, P_C) \
    P_X##_DEF(4, P_A, P_B, P_C);       \
    REPEAT_3_4(P_X, P_A, P_B, P_C)
#define REPEAT_3_6(P_X, P_A, P_B, P_C) \
    P_X##_DEF(5, P_A, P_B, P_C);       \
    REPEAT_3_5(P_X, P_A, P_B, P_C)
#define REPEAT_3_7(P_X, P_A, P_B, P_C) \
    P_X##_DEF(6, P_A, P_B, P_C);       \
    REPEAT_3_6(P_X, P_A, P_B, P_C)
#define REPEAT_3_8(P_X, P_A, P_B, P_C) \
    P_X##_DEF(7, P_A, P_B, P_C);       \
    REPEAT_3_7(P_X, P_A, P_B, P_C)
#define REPEAT_3_9(P_X, P_A, P_B, P_C) \
    P_X##_DEF(8, P_A, P_B, P_C);       \
    REPEAT_3_8(P_X, P_A, P_B, P_C)
#define REPEAT_3_10(P_X, P_A, P_B, P_C) \
    P_X##_DEF(9, P_A, P_B, P_C);        \
    REPEAT_3_9(P_X, P_A, P_B, P_C)
#define REPEAT_3_11(P_X, P_A, P_B, P_C) \
    P_X##_DEF(A, P_A, P_B, P_C);        \
    REPEAT_3_10(P_X, P_A, P_B, P_C)
#define REPEAT_3_12(P_X, P_A, P_B, P_C) \
    P_X##_DEF(B, P_A, P_B, P_C);        \
    REPEAT_3_11(P_X, P_A, P_B, P_C)
#define REPEAT_3_13(P_X, P_A, P_B, P_C) \
    P_X##_DEF(C, P_A, P_B, P_C);        \
    REPEAT_3_12(P_X, P_A, P_B, P_C)
#define REPEAT_3_14(P_X, P_A, P_B, P_C) \
    P_X##_DEF(D, P_A, P_B, P_C);        \
    REPEAT_3_13(P_X, P_A, P_B, P_C)
#define REPEAT_3_15(P_X, P_A, P_B, P_C) \
    P_X##_DEF(E, P_A, P_B, P_C);        \
    REPEAT_3_14(P_X, P_A, P_B, P_C)
#define REPEAT_3_16(P_X, P_A, P_B, P_C) \
    P_X##_DEF(F, P_A, P_B, P_C);        \
    REPEAT_3_15(P_X, P_A, P_B, P_C)

#define REPEAT_DEF_3_N(P_NUM, P_OP, P_A, P_B, P_C) REPEAT_3_##P_NUM(P_OP, P_A, P_B, P_C) //One level of indirection to ensure order of expansion does not affect preprocessing P_NUM
#define REPEAT_3_N(P_NUM, P_OP, P_A, P_B, P_C) REPEAT_DEF_3_N(P_NUM, P_OP, P_A, P_B, P_C)

//Macro for initializing N variables. generates N statements that defines VAR##N = RHS_ACCESSOR_DEF(...)
#define VAR_INIT_TO_CONST_DEF(ID, TYPE, VAR, VAL) TYPE VAR##ID = VAL
#define REPEAT_VAR_INIT_TO_CONST(N, TYPE, VAR, VAL) REPEAT_3_N(N, VAR_INIT_TO_CONST, TYPE, VAR, VAL)

#endif // ARM_COMPUTE_REPEAT_H
