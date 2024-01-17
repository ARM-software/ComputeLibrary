/*
* Copyright (c) 2023 Arm Limited.
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

#ifndef CKW_INCLUDE_CKW_TYPES_OPERATORS_H
#define CKW_INCLUDE_CKW_TYPES_OPERATORS_H

#include <cstdint>

namespace ckw
{

/** Unary operators and functions. */
enum class UnaryOp : int32_t
{
    LogicalNot = 0x0000, // !
    BitwiseNot = 0x0001, // ~

    Exp   = 0x0010,
    Tanh  = 0x0011,
    Sqrt  = 0x0012,
    Erf   = 0x0013,
    Fabs  = 0x0014,
    Log   = 0x0015,
    Round = 0x0016,
    Floor = 0x0017,
};

/** Assignment operators. */
enum class AssignmentOp : int32_t
{
    Increment = 0x0000, // +=
    Decrement = 0x0001, // -=
};

/** Binary operators. */
enum class BinaryOp : int32_t
{
    // Elementwise
    Add = 0x0000, // +
    Sub = 0x0001, // -
    Mul = 0x0002, // *
    Div = 0x0003, // /
    Mod = 0x0004, // %

    // Relational
    Equal        = 0x1000, // ==
    Less         = 0x1001, // <
    LessEqual    = 0x1002, // <=
    Greater      = 0x1003, // >
    GreaterEqual = 0x1004, // >=

    // Algebra
    MatMul_Nt_Nt = 0x2000, // X
    MatMul_Nt_T  = 0x2001, // X
    MatMul_T_Nt  = 0x2002, // X
    MatMul_T_T   = 0x2003, // X
    Dot          = 0x2004, // .

    // Logical
    LogicalAnd = 0x3000, // &&
    LogicalOr  = 0x3001, // ||

    // Bitwise
    BitwiseXOR = 0x4000, // ^

    // Functions
    Min = 0x8000,
    Max = 0x8001,
};

/** Ternary operators. */
enum class TernaryOp : int32_t
{
    Select = 0x0000,
    Clamp  = 0x0001,
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TYPES_OPERATORS_H
