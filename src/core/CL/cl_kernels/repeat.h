/*
 * Copyright (c) 2019-2020 Arm Limited.
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

#include "helpers.h"

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

// Repeat macros with 4 param, excluding the implicit ID param
#define REPEAT_4_1(P_X, P_A, P_B, P_C, P_D) P_X##_DEF(0, P_A, P_B, P_C, P_D)
#define REPEAT_4_2(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(1, P_A, P_B, P_C, P_D);       \
    REPEAT_4_1(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_3(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(2, P_A, P_B, P_C, P_D);       \
    REPEAT_4_2(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_4(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(3, P_A, P_B, P_C, P_D);       \
    REPEAT_4_3(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_5(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(4, P_A, P_B, P_C, P_D);       \
    REPEAT_4_4(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_6(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(5, P_A, P_B, P_C, P_D);       \
    REPEAT_4_5(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_7(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(6, P_A, P_B, P_C, P_D);       \
    REPEAT_4_6(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_8(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(7, P_A, P_B, P_C, P_D);       \
    REPEAT_4_7(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_9(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(8, P_A, P_B, P_C, P_D);       \
    REPEAT_4_8(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_10(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(9, P_A, P_B, P_C, P_D);        \
    REPEAT_4_9(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_11(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(A, P_A, P_B, P_C, P_D);        \
    REPEAT_4_10(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_12(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(B, P_A, P_B, P_C, P_D);        \
    REPEAT_4_11(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_13(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(C, P_A, P_B, P_C, P_D);        \
    REPEAT_4_12(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_14(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(D, P_A, P_B, P_C, P_D);        \
    REPEAT_4_13(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_15(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(E, P_A, P_B, P_C, P_D);        \
    REPEAT_4_14(P_X, P_A, P_B, P_C, P_D)
#define REPEAT_4_16(P_X, P_A, P_B, P_C, P_D) \
    P_X##_DEF(F, P_A, P_B, P_C, P_D);        \
    REPEAT_4_15(P_X, P_A, P_B, P_C, P_D)

#define REPEAT_DEF_4_N(P_NUM, P_OP, P_A, P_B, P_C, P_D) REPEAT_4_##P_NUM(P_OP, P_A, P_B, P_C, P_D) //One level of indirection to ensure order of expansion does not affect preprocessing P_NUM
#define REPEAT_4_N(P_NUM, P_OP, P_A, P_B, P_C, P_D) REPEAT_DEF_4_N(P_NUM, P_OP, P_A, P_B, P_C, P_D)

// Macro for initializing N variables. Generates N statements that defines VAR##N = RHS_ACCESSOR_DEF(...)
#define VAR_INIT_TO_CONST_DEF(ID, TYPE, VAR, VAL) TYPE VAR##ID = VAL
#define REPEAT_VAR_INIT_TO_CONST(N, TYPE, VAR, VAL) REPEAT_3_N(N, VAR_INIT_TO_CONST, TYPE, VAR, VAL)

// Macro for initializing N variables by converting the data type. Generates N statements that defines VAR##N = RHS_ACCESSOR_DEF(...)
#define VAR_INIT_CONVERT_DEF(ID, TYPE_OUT, VAR_IN, VAR_OUT) TYPE_OUT VAR_OUT##ID = CONVERT(VAR_IN##ID, TYPE_OUT)
#define REPEAT_VAR_INIT_CONVERT(N, TYPE_OUT, VAR_IN, VAR_OUT) REPEAT_3_N(N, VAR_INIT_CONVERT, TYPE_OUT, VAR_IN, VAR_OUT)

// Macro for initializing N variables by converting the data type with saturation. Generates N statements that defines VAR##N = RHS_ACCESSOR_DEF(...)
#define VAR_INIT_CONVERT_SAT_DEF(ID, TYPE_OUT, VAR_IN, VAR_OUT) TYPE_OUT VAR_OUT##ID = CONVERT_SAT(VAR_IN##ID, TYPE_OUT)
#define REPEAT_VAR_INIT_CONVERT_SAT(N, TYPE_OUT, VAR_IN, VAR_OUT) REPEAT_3_N(N, VAR_INIT_CONVERT_SAT, TYPE_OUT, VAR_IN, VAR_OUT)

// Macro for adding a constant to N variables. Generates N statements that defines VAR##N =RHS_ACCESSOR_DEF(...)
#define ADD_CONST_TO_VAR_DEF(ID, TYPE, VAR, VAL) VAR##ID += (TYPE)VAL
#define REPEAT_ADD_CONST_TO_VAR(N, TYPE, VAR, VAL) REPEAT_3_N(N, ADD_CONST_TO_VAR, TYPE, VAR, VAL)

// Macro for multiplying N variables (VAR_B) by a constant (VAL) and adding to other N variables (VAR_A). Generates N statements that defines VAR_A##N =RHS_ACCESSOR_DEF(...)
#define MLA_VAR_WITH_CONST_VEC_DEF(ID, VAR_A, VAR_B, VAL) VAR_A##ID += VAR_B##ID * VAL
#define REPEAT_MLA_VAR_WITH_CONST_VEC(N, VAR_A, VAR_B, VAL) REPEAT_3_N(N, MLA_VAR_WITH_CONST_VEC, VAR_A, VAR_B, VAL)

// Macro for adding a vector to N-variables. Generates N statements that defines VAR##N =RHS_ACCESSOR_DEF(...)
#define ADD_VECTOR_TO_VAR_DEF(ID, TYPE, VAR, VEC) VAR##ID += VEC
#define REPEAT_ADD_VECTOR_TO_VAR(N, VAR, VEC) REPEAT_3_N(N, ADD_VECTOR_TO_VAR, "", VAR, VEC)

// Macro for adding a two N-variables. Generates N statements that defines VAR##N =RHS_ACCESSOR_DEF(...)
#define ADD_TWO_VARS_DEF(ID, TYPE, VAR_A, VAR_B) VAR_A##ID += VAR_B##ID
#define REPEAT_ADD_TWO_VARS(N, VAR_A, VAR_B) REPEAT_3_N(N, ADD_TWO_VARS, "", VAR_A, VAR_B)

// Macro for performing Max between a constant and N variables. Generates N statements that defines VAR##N =RHS_ACCESSOR_DEF(...)
#define MAX_CONST_VAR_DEF(ID, TYPE, VAR, VAL) VAR##ID = max(VAR##ID, (TYPE)VAL)
#define REPEAT_MAX_CONST_VAR(N, TYPE, VAR, VAL) REPEAT_3_N(N, MAX_CONST_VAR, TYPE, VAR, VAL)

// Macro for performing Min between a constant and N variables. Generates N statements that defines VAR##N =RHS_ACCESSOR_DEF(...)
#define MIN_CONST_VAR_DEF(ID, TYPE, VAR, VAL) VAR##ID = min(VAR##ID, (TYPE)VAL)
#define REPEAT_MIN_CONST_VAR(N, TYPE, VAR, VAL) REPEAT_3_N(N, MIN_CONST_VAR, TYPE, VAR, VAL)

// Macro for performing ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE to N variables. Generates N statements that defines VAR##N =RHS_ACCESSOR_DEF(...)
#define ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE_DEF(ID, SIZE, VAR, RES_MUL, RES_SHIFT) VAR##ID = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(VAR##ID, RES_MUL, RES_SHIFT, SIZE)
#define REPEAT_ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(N, SIZE, VAR, RES_MUL, RES_SHIFT) REPEAT_4_N(N, ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE, SIZE, VAR, RES_MUL, RES_SHIFT)

// Macro for performing ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE to N variables. Generates N statements that defines VAR##N =RHS_ACCESSOR_DEF(...)
#define ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE_DEF(ID, SIZE, VAR, RES_MUL, RES_SHIFT) VAR##ID = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(VAR##ID, RES_MUL, RES_SHIFT, SIZE)
#define REPEAT_ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(N, SIZE, VAR, RES_MUL, RES_SHIFT) REPEAT_4_N(N, ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE, SIZE, VAR, RES_MUL, RES_SHIFT)

// Macro for performing per-channel ASYMM_MULT_BY_QUANT_MULTIPLIER to N variables.
#define ASYMM_MULT_BY_QUANT_MULTIPLIER_PER_CHANNEL_DEF(ID, SIZE, VAR, RES_MUL, RES_SHIFT)                     \
    ({                                                                                                        \
        VEC_DATA_TYPE(int, N0)                                                                                \
        VAR##ID_shift_lt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(VAR##ID, RES_MUL, RES_SHIFT, N0); \
        VEC_DATA_TYPE(int, N0)                                                                                \
        VAR##ID_shift_gt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(VAR##ID, RES_MUL, RES_SHIFT, N0);    \
        VAR##ID           = select(VAR##ID_shift_lt0, VAR##ID_shift_gt0, RES_SHIFT >= 0);                     \
    })
#define REPEAT_ASYMM_MULT_BY_QUANT_MULTIPLIER_PER_CHANNEL(N, SIZE, VAR, RES_MUL, RES_SHIFT) REPEAT_4_N(N, ASYMM_MULT_BY_QUANT_MULTIPLIER_PER_CHANNEL, SIZE, VAR, RES_MUL, RES_SHIFT)

#endif // ARM_COMPUTE_REPEAT_H
